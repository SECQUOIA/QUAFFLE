"""Geometry, independent circuit parity and diagnostic contracts for HQ pilot."""
import copy
import io

import pytest
import torch
from torch import nn

from utilsTorch.hq_direction import (
    HQDirectionUNet, build_from_config, model_config, shared_state_sha256,
)
from utilsTorch.unetTorch import (
    HAS_PENNYLANE, pennylane_hqunet_state_qnode, torch_hqunet_expvals,
)


def _snapshot(model):
    return {name: value.clone() for name, value in model.state_dict().items()}


def test_common_initialization_includes_all_batchnorm_buffers_and_preserves_rng():
    before = torch.get_rng_state().clone()
    quantum = HQDirectionUNet(kind='quantum', seed=42)
    classical = HQDirectionUNet(kind='mlp', seed=42)
    assert torch.equal(torch.get_rng_state(), before)
    assert shared_state_sha256(quantum) == shared_state_sha256(classical)
    common = {name: value for name, value in quantum.state_dict().items()
              if not name.startswith('bottleneck.core.')}
    assert any(name.endswith('num_batches_tracked') for name in common)
    assert any(name.endswith('running_var') for name in common)
    assert all(torch.equal(value, classical.state_dict()[name]) for name, value in common.items())
    assert sum(p.numel() for p in quantum.bottleneck.core_parameters()) == 8
    assert sum(p.numel() for p in classical.bottleneck.core_parameters()) == 5216
    original = shared_state_sha256(quantum)
    with torch.no_grad():
        quantum.bottleneck.core.weights.add_(1)
    assert shared_state_sha256(quantum) == original
    quantum.stem[2].num_batches_tracked.add_(1)
    assert shared_state_sha256(quantum) != original


def test_full_geometry_replaces_deepest_doubleconv_and_preserves_unet_skips():
    model = HQDirectionUNet(kind='mlp', base_channels=2).eval()
    shapes = {}
    handles = [getattr(model, name).register_forward_hook(
        lambda module, args, output, name=name: shapes.__setitem__(name, tuple(output.shape)))
        for name in ('stem', 'down1', 'down2', 'down3', 'bottleneck', 'up1', 'up2', 'up3', 'up4')]
    try:
        with torch.no_grad():
            output = model(torch.randn(1, 3, 128, 128))
    finally:
        for handle in handles:
            handle.remove()
    assert output.shape == (1, 2, 128, 128)
    assert shapes == dict(stem=(1, 2, 128, 128), down1=(1, 4, 64, 64),
                         down2=(1, 8, 32, 32), down3=(1, 16, 16, 16),
                         bottleneck=(1, 32, 8, 8), up1=(1, 16, 16, 16),
                         up2=(1, 8, 32, 32), up3=(1, 4, 64, 64), up4=(1, 2, 128, 128))
    assert model.bottleneck.post_projection.in_features == 32
    assert model.bottleneck.post_projection.out_features == 32 * 8 * 8
    assert sum(isinstance(module, nn.ConvTranspose2d) for module in model.modules()) == 4
    assert not any(isinstance(module, (nn.GroupNorm, nn.Dropout)) for module in model.modules())
    assert model.stem[0].groups == 3


def test_encoding_uses_pool_then_projection_and_tanh_radians_without_pi():
    model = HQDirectionUNet(kind='mlp', base_channels=1, image_size=64)
    block = model.bottleneck
    with torch.no_grad():
        block.pre_projection.weight.zero_()
        block.pre_projection.bias.copy_(torch.tensor([-2.0, 0.25, 2.0]))
    latent = torch.randn(2, 8, 8, 8)
    before = _snapshot(model)
    angles = block.encode_angles(latent)
    expected = torch.tensor([-2.0, 0.25, 2.0]).tanh().expand(2, 16, 3)
    assert torch.equal(angles, expected)
    diagnostics = block.angle_diagnostics(latent)
    assert diagnostics['n_angles'] == 96
    assert diagnostics['max_radians'] < 1
    assert diagnostics['min_radians'] > -1
    assert diagnostics['saturation_fraction'] == pytest.approx(2 / 3)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert not any('input_scales' in name for name, _ in model.named_parameters())


@pytest.mark.parametrize('kind', ('quantum', 'mlp'))
def test_modes_separate_core_zeroing_bias_and_full_bottleneck_zero_without_mutation(kind):
    model = HQDirectionUNet(kind=kind, base_channels=1, image_size=64).eval()
    block = model.bottleneck
    with torch.no_grad():
        block.post_projection.bias.copy_(torch.linspace(-0.3, 0.4, block.post_projection.out_features))
    original = _snapshot(model)
    rng = torch.get_rng_state().clone()
    latent = torch.randn(1, 8, 8, 8)
    rng_after_input = torch.get_rng_state().clone()
    with torch.no_grad():
        block.set_mode('zero_core')
        output = block(latent)
        assert torch.equal(output.flatten(), block.post_projection.bias)
        block.set_mode('zero_bottleneck')
        assert torch.count_nonzero(block(latent)) == 0
        assert torch.count_nonzero(block(latent + 5)) == 0
        block.set_mode('random_core', seed=17)
        first = block(latent)
        block.set_mode('random_core', seed=17)
        assert torch.equal(first, block(latent))
        block.set_mode('random_core', seed=18)
        assert not torch.equal(first, block(latent))
        block.set_mode('on')
    assert torch.equal(torch.get_rng_state(), rng_after_input)
    assert not torch.equal(rng, rng_after_input)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())
    assert tuple(model.state_dict()) == tuple(original)
    with pytest.raises(ValueError, match='mode'):
        block.set_mode('zero')


def test_no_bottleneck_identity_or_reconstruction_bypass():
    block = HQDirectionUNet(kind='mlp', base_channels=1, image_size=64).bottleneck
    with torch.no_grad():
        block.post_projection.weight.zero_()
        block.post_projection.bias.zero_()
        first = block(torch.randn(2, 8, 8, 8))
        second = block(torch.randn(2, 8, 8, 8) + 100)
    assert torch.count_nonzero(first) == torch.count_nonzero(second) == 0


@pytest.mark.parametrize('kind', ('quantum', 'mlp'))
def test_input_pre_core_post_and_encoder_receive_finite_nonzero_gradients(kind):
    model = HQDirectionUNet(kind=kind, base_channels=1, image_size=64)
    image = torch.randn(1, 3, 64, 64, requires_grad=True)
    target = torch.arange(64 * 64).reshape(1, 64, 64) % 2
    loss = nn.functional.cross_entropy(model(image), target)
    loss.backward()
    groups = (('input', [image]), ('pre', model.bottleneck.pre_projection.parameters()),
              ('core', model.bottleneck.core_parameters()),
              ('post', model.bottleneck.post_projection.parameters()),
              ('encoder', model.stem.parameters()))
    for name, values in groups:
        gradients = [value.grad for value in values]
        assert gradients and all(gradient is not None and torch.isfinite(gradient).all()
                                 for gradient in gradients), name
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0, name


@pytest.mark.parametrize('kind', ('quantum', 'mlp'))
def test_config_and_strict_checkpoint_roundtrip_including_trained_batchnorm(kind):
    model = HQDirectionUNet(kind=kind, base_channels=1, image_size=64, seed=-17)
    # Exercise real BN updates without paying another circuit forward.
    model.bottleneck.set_mode('zero_bottleneck')
    with torch.no_grad():
        model(torch.randn(2, 3, 64, 64))
    model.bottleneck.set_mode('on')
    buffer = io.BytesIO()
    torch.save(dict(config=model_config(model), state_dict=model.state_dict()), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location='cpu', weights_only=True)
    restored = build_from_config(checkpoint['config'])
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    assert model_config(restored) == model_config(model)
    assert shared_state_sha256(restored) == shared_state_sha256(model)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    assert restored.bottleneck.mode == 'on'
    assert int(restored.stem[2].num_batches_tracked) == 1
    model.eval()
    restored.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(image), restored(image))
    incomplete = copy.copy(checkpoint['state_dict'])
    incomplete.pop('stem.2.running_mean')
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(incomplete, strict=True)


def test_config_validation_and_seed_isolation():
    config = model_config(HQDirectionUNet(kind='mlp', base_channels=1))
    for bad in (dict(config, surprise=True), {key: value for key, value in config.items() if key != 'seed'}):
        with pytest.raises(ValueError, match='exactly'):
            build_from_config(bad)
    for patch in (dict(kind='other'), dict(seed=True), dict(base_channels=0),
                  dict(image_size=32), dict(image_size=127), dict(in_channels=True)):
        with pytest.raises(ValueError):
            HQDirectionUNet(**patch)
    first = HQDirectionUNet(kind='mlp', seed=42, base_channels=1)
    repeat = build_from_config(model_config(first))
    different = HQDirectionUNet(kind='mlp', seed=43, base_channels=1)
    assert all(torch.equal(value, repeat.state_dict()[name]) for name, value in first.state_dict().items())
    assert shared_state_sha256(first) != shared_state_sha256(different)
    with pytest.raises(ValueError, match='Expected'):
        first(torch.randn(1, 3, 64, 64))


@pytest.mark.skipif(not HAS_PENNYLANE, reason='Independent PennyLane reference unavailable')
def test_16qubit_engine_outputs_and_gradients_match_independent_pennylane_reference():
    generator = torch.Generator().manual_seed(71)
    angles = (torch.rand(1, 16, 3, dtype=torch.float64, generator=generator) - 0.5).requires_grad_()
    weights = (torch.rand(1, 2, 4, dtype=torch.float64, generator=generator) - 0.5).requires_grad_()
    coefficients = torch.linspace(-0.7, 0.9, 32, dtype=torch.float64)
    actual = torch_hqunet_expvals(angles, weights, grid_size=4)
    actual_grads = torch.autograd.grad((actual * coefficients).sum(), (angles, weights))
    reference_angles, reference_weights = angles.detach().requires_grad_(), weights.detach().requires_grad_()
    state = pennylane_hqunet_state_qnode(n_layers=1, grid_size=4)(reference_angles, reference_weights)
    # Derive observables independently rather than calling the Torch readout helper.
    indices = torch.arange(1 << 16)
    probabilities = state.abs().square()
    z = [((1 - 2 * ((indices >> (15 - q)) & 1)) * probabilities).sum(-1) for q in range(16)]
    x = [(state.conj() * state[:, indices ^ (1 << (15 - q))]).sum(-1).real for q in range(16)]
    expected = torch.stack(z + x, dim=1)
    expected_grads = torch.autograd.grad((expected * coefficients).sum(), (reference_angles, reference_weights))
    assert torch.allclose(actual, expected, atol=2e-12, rtol=2e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0
        assert torch.allclose(actual_grad, expected_grad, atol=3e-11, rtol=3e-11)
