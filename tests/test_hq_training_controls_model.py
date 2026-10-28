"""Behavioral training-control guards; real circuit inputs retain gradients."""
import copy
import io
from pathlib import Path

import pytest
import torch
from torch import nn

from utilsTorch import hq_direction as legacy
from utilsTorch import hq_training_controls as controls


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_imports_bind_to_this_worktree():
    root = Path(__file__).resolve().parents[1]
    assert Path(controls.__file__).resolve() == root / 'utilsTorch/hq_training_controls.py'
    assert Path(legacy.__file__).resolve() == root / 'utilsTorch/hq_direction.py'


@pytest.mark.parametrize('control', ('quantum', 'mlp'))
def test_legacy_state_and_training_and_eval_outputs_are_bitwise_unchanged(control):
    old = legacy.HQDirectionUNet(kind=control, seed=77, base_channels=1, image_size=64)
    new = controls.build_model(control, seed=77, base_channels=1, image_size=64)
    assert type(new) is legacy.HQDirectionUNet
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
    image = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        for training in (True, False):
            old.train(training)
            new.train(training)
            assert torch.equal(new(image), old(image))
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())


def test_all_controls_share_noncore_initial_weights_bn_and_preserve_caller_rng():
    before = torch.get_rng_state().clone()
    models = [controls.build_model(control, seed=-17) for control in controls.CONTROLS]
    assert torch.equal(before, torch.get_rng_state())
    assert len({controls.shared_state_sha256(model) for model in models}) == 1
    reference = snapshot(models[0])
    assert any(name.endswith('running_mean') for name in reference)
    assert any(name.endswith('num_batches_tracked') for name in reference)
    for model in models:
        common = {name: value for name, value in model.state_dict().items()
                  if not name.startswith('bottleneck.core.')}
        assert common.keys() == {name for name in reference if not name.startswith('bottleneck.core.')}
        assert all(torch.equal(value, reference[name]) for name, value in common.items())
    for index in (1, 2):
        assert torch.equal(models[index].bottleneck.core.weights, models[0].bottleneck.core.weights)
    models[0].stem[2].num_batches_tracked.add_(1)
    assert controls.shared_state_sha256(models[0]) != controls.shared_state_sha256(models[1])


@pytest.mark.parametrize('control,expected', (
    ('quantum', (344480, 344480, 8, 8, 270539, 270539)),
    ('frozen_quantum', (344480, 344472, 8, 0, 270539, 270531)),
    ('zero_bottleneck', (344480, 73941, 8, 0, 270539, 0)),
    ('compact_affine', (346040, 346040, 1568, 1568, 272099, 272099)),
    ('mlp', (349688, 349688, 5216, 5216, 275747, 275747)),
))
def test_parameter_counts_distinguish_stored_trainable_and_bottleneck_parameters(control, expected):
    counts = controls.parameter_counts(controls.build_model(control, seed=42))
    assert counts == dict(zip(('total', 'trainable', 'core_total', 'core_trainable',
                              'bottleneck_total', 'bottleneck_trainable'), expected))


def test_frozen_quantum_preserves_nonzero_input_and_projection_gradients_but_not_weight_updates():
    model = controls.build_model('frozen_quantum', seed=42, base_channels=1, image_size=64)
    block = model.bottleneck
    before = snapshot(model)
    assert all(not value.requires_grad for value in block.core.parameters())
    assert all(value.requires_grad for name, value in model.named_parameters()
               if not name.startswith('bottleneck.core.'))
    latent = torch.randn(1, 8, 8, 8, generator=torch.Generator().manual_seed(31), requires_grad=True)
    optimizer = torch.optim.Adam([value for value in model.parameters() if value.requires_grad], lr=.003)
    output = block(latent)
    coefficients = torch.linspace(-.7, .9, output.numel()).reshape_as(output)
    (output * coefficients).mean().backward()
    for values in ([latent], list(block.pre_projection.parameters()), list(block.post_projection.parameters())):
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in values)
        assert sum(float(value.grad.abs().sum()) for value in values) > 0
    assert all(value.grad is None for value in block.core.parameters())
    optimizer.step()
    assert torch.equal(block.core.weights, before['bottleneck.core.weights'])
    assert not torch.equal(block.pre_projection.weight, before['bottleneck.pre_projection.weight'])


def test_zero_training_control_cannot_be_reenabled_and_never_executes_bottleneck_layers(monkeypatch):
    model = controls.build_model('zero_bottleneck', seed=42, base_channels=1, image_size=64)
    block = model.bottleneck
    latent = torch.randn(2, 8, 8, 8)
    with torch.no_grad():
        block.post_projection.bias.fill_(19)
    def forbidden(*args, **kwargs):
        raise AssertionError('A permanently zero bottleneck executed a parameterized layer')
    for module in (block.pre_projection, block.core, block.post_projection):
        monkeypatch.setattr(module, 'forward', forbidden)
    before_rng = torch.get_rng_state().clone()
    for training in (True, False):
        model.train(training)
        for mode in ('on', 'zero_core', 'random_core', 'zero_bottleneck'):
            block.set_mode(mode, seed=123)
            assert block.mode == 'zero_bottleneck'
            assert torch.count_nonzero(block(latent)) == 0
        block.mode = 'on'  # Even stale mutable diagnostic metadata cannot bypass it.
        assert torch.count_nonzero(block.forward(latent)) == 0
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all(not value.requires_grad for value in block.parameters())
    with pytest.raises(ValueError, match='mode'):
        block.set_mode('invalid')
    with pytest.raises(ValueError, match='seed'):
        block.set_mode('on', seed=True)


def test_zero_bottleneck_still_trains_via_ordinary_unet_skips():
    model = controls.build_model('zero_bottleneck', seed=42, base_channels=1, image_size=64)
    image = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(9), requires_grad=True)
    before = snapshot(model)
    optimizer = torch.optim.Adam([value for value in model.parameters() if value.requires_grad], lr=.003)
    nn.functional.cross_entropy(model(image), torch.arange(2*64*64).reshape(2, 64, 64) % 2).backward()
    assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    assert model.stem[0].weight.grad is not None and model.stem[0].weight.grad.abs().sum() > 0
    assert all(value.grad is None for value in model.bottleneck.parameters())
    optimizer.step()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items()
               if name.startswith('bottleneck.'))
    assert not torch.equal(before['stem.0.weight'], model.stem[0].weight)


def test_compact_core_is_full_rank_affine_with_tanh_and_isolated_repeatable_diagnostic_rng():
    model = controls.build_model('compact_affine', seed=42, base_channels=1, image_size=64)
    block = model.bottleneck
    assert len(block.core) == 3 and isinstance(block.core[0], nn.Flatten) and isinstance(block.core[2], nn.Tanh)
    assert isinstance(block.core[1], nn.Linear)
    assert block.core[1].weight.shape == (32, 48) and block.core[1].bias.shape == (32,)
    assert int(torch.linalg.matrix_rank(block.core[1].weight)) == 32
    angles = torch.randn(2, 16, 3)
    assert torch.equal(block.core(angles), nn.functional.linear(angles.flatten(1), block.core[1].weight, block.core[1].bias).tanh())
    latent = torch.randn(1, 8, 8, 8)
    before = snapshot(model)
    before_rng = torch.get_rng_state().clone()
    with torch.no_grad():
        block.set_mode('random_core', seed=17)
        first = block(latent)
        block.set_mode('random_core', seed=17)
        assert torch.equal(first, block(latent))
        block.set_mode('random_core', seed=18)
        assert not torch.equal(first, block(latent))
        block.set_mode('on')
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize('control', controls.CONTROLS)
def test_strict_checkpoint_config_restores_bn_weights_modes_and_requires_grad(control):
    model = controls.build_model(control, seed=77, base_channels=1, image_size=64)
    model.train()
    # Full real U-Net BN updates with no circuit work needed for this test.
    model.bottleneck.set_mode('zero_bottleneck')
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        model(image)
    model.bottleneck.set_mode('on')
    trainability = {name: value.requires_grad for name, value in model.named_parameters()}
    buffer = io.BytesIO()
    torch.save(dict(config=controls.model_config(model), state_dict=model.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location='cpu', weights_only=True)
    restored = controls.build_from_config(checkpoint['config'])
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    assert controls.model_config(restored) == controls.model_config(model)
    assert {name: value.requires_grad for name, value in restored.named_parameters()} == trainability
    assert restored.bottleneck.mode == ('zero_bottleneck' if control == 'zero_bottleneck' else 'on')
    assert int(restored.stem[2].num_batches_tracked) == 1
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    assert controls.shared_state_sha256(restored) == controls.shared_state_sha256(model)
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.equal(model(image[:1]), restored(image[:1]))
    invalid = copy.copy(checkpoint['state_dict'])
    invalid.pop('stem.2.running_mean')
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(invalid, strict=True)
    invalid = copy.copy(checkpoint['state_dict'])
    invalid['bottleneck.post_projection.weight'] = invalid['bottleneck.post_projection.weight'][:1]
    with pytest.raises(RuntimeError, match='size mismatch'):
        restored.load_state_dict(invalid, strict=True)


def test_strict_factory_config_rejects_unknown_missing_legacy_and_invalid_values():
    model = controls.build_model('compact_affine', seed=-17, base_channels=1, image_size=64)
    config = controls.model_config(model)
    assert set(config) == {'control', 'seed', 'base_channels', 'image_size'}
    invalid_configs = (dict(config, extra=True), {key: value for key, value in config.items() if key != 'control'},
                       legacy.model_config(model), dict(config, control='separable'), dict(config, control=True),
                       dict(config, seed=True), dict(config, base_channels=0), dict(config, image_size=32),
                       dict(config, image_size=127), ['compact_affine'])
    for invalid in invalid_configs:
        with pytest.raises(ValueError):
            controls.build_from_config(invalid)
    with pytest.raises(ValueError, match='factory'):
        controls.model_config(legacy.HQDirectionUNet(kind='mlp', base_channels=1, image_size=64))
    restored = controls.build_from_config(config)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    changed = controls.build_model('compact_affine', seed=-16, base_channels=1, image_size=64)
    assert controls.shared_state_sha256(model) != controls.shared_state_sha256(changed)
