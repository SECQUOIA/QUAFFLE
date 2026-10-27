"""Independent readout identities and local/core/backbone contracts."""
import copy
import argparse

import pytest
import torch
from torch.nn import functional as F

from scripts.run_parameter_budget_comparison import base_config
from tools import training_common as tc
from tools.checkpoint_evaluation import load_checkpoint_model
from utilsTorch.local_spatial import (
    LOCAL_EDGES, LocalSpatialBottleneckBlock, _state_circuit,
    build_local_spatial_block, local_spatial_options, state_readouts,
)
from utilsTorch.utilsTorch import QVUNetSegmentation, ClassicalAblationUNetSegmentation


def config(kind='quantum', readout='xz_zz', control='trainable'):
    return dict(base_config(8, 1), seed=42, image_size=32, dataset='combined',
        local_spatial_kernel=kind, local_spatial_readout=readout,
        local_spatial_channels=4, q_layers=1, q_training_control=control,
        batch_size=1, learning_rate=0.001, num_epochs=1)


def test_readouts_agree_with_independent_pennylane_observables_and_gradients():
    qml = pytest.importorskip('pennylane')
    generator = torch.Generator().manual_seed(123)
    tokens = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(1, 4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    scales = torch.ones(1, 4, dtype=torch.float64, requires_grad=True)
    state = _state_circuit(1, True)(tokens, weights, scales)
    actual = state_readouts(state)
    dev = qml.device('default.qubit', wires=4)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def reference(x, w, s):
        for wire in range(4):
            qml.RY(torch.pi * torch.tanh(s[0, wire] * x[:, wire]), wires=wire)
            qml.Rot(*w[0, wire], wires=wire)
        for first, second in ((0, 1), (2, 3), (0, 2), (1, 3)):
            qml.CNOT(wires=[first, second])
        return ([qml.expval(qml.PauliZ(wire)) for wire in range(4)]
                + [qml.expval(qml.PauliX(wire)) for wire in range(4)]
                + [qml.expval(qml.PauliZ(first) @ qml.PauliZ(second)) for first, second in LOCAL_EDGES])

    expected = torch.stack(reference(tokens, weights, scales), dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    actual_grad = torch.autograd.grad(actual.square().sum(), (tokens, weights, scales))
    expected_grad = torch.autograd.grad(expected.square().sum(), (tokens, weights, scales))
    for value, reference_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(value, reference_value, atol=1e-9, rtol=1e-9)


def test_separable_z_has_no_cross_position_derivatives_and_zz_is_product():
    block = LocalSpatialBottleneckBlock(4, q_training_control='no_entanglement', seed=9)
    tokens = torch.tensor([[0.2, -0.4, 0.6, 0.1]], requires_grad=True)
    outputs = block._evaluate_core(tokens)
    jacobian = torch.autograd.functional.jacobian(block._evaluate_core, tokens)[0, :4, 0]
    assert jacobian[~torch.eye(4, dtype=torch.bool)].abs().max() < 1e-7
    assert jacobian.diag().abs().min() > 1e-3
    for edge_index, (first, second) in enumerate(LOCAL_EDGES):
        torch.testing.assert_close(outputs[:, 8 + edge_index], outputs[:, first] * outputs[:, second])
    entangled = LocalSpatialBottleneckBlock(4, seed=9)
    coupled = torch.autograd.functional.jacobian(entangled._evaluate_core, tokens)[0, :4, 0]
    assert coupled[~torch.eye(4, dtype=torch.bool)].abs().max() > 1e-3


@pytest.mark.parametrize('kind', ['quantum', 'mlp', 'conv', 'sincos'])
def test_only_actual_2x2_neighboring_locations_can_affect_one_output(kind):
    torch.manual_seed(21)
    block = LocalSpatialBottleneckBlock(3, kind=kind, feature_channels=2, seed=7)
    image = torch.randn(1, 3, 5, 6, requires_grad=True)
    block(image)[0, :, 1, 2].sum().backward()
    spatial_gradient = image.grad.abs().sum(dim=(0, 1))
    support = torch.zeros(5, 6, dtype=torch.bool)
    support[1:3, 2:4] = True
    assert spatial_gradient[~support].max() == 0
    assert torch.all(spatial_gradient[support] > 0)


def test_conv_control_equals_a_standard_shared_2x2_convolution():
    torch.manual_seed(8)
    block = LocalSpatialBottleneckBlock(3, kind='conv', feature_channels=2)
    image = torch.randn(2, 3, 4, 5)
    projected = block.pre_q_conv(image)
    linear = block.core[0]
    per_channel = projected.reshape(4, 1, 4, 5)
    filtered = F.silu(F.conv2d(F.pad(per_channel, (0, 1, 0, 1), mode='replicate'),
        linear.weight.reshape(12, 1, 2, 2), linear.bias))
    expected = block.post_q_proj(filtered.reshape(2, 24, 4, 5))
    torch.testing.assert_close(block(image), expected)


def test_z_padding_keeps_projection_shape_initialization_and_z_values():
    torch.manual_seed(44)
    state = torch.random.get_rng_state()
    rich = LocalSpatialBottleneckBlock(4, seed=2)
    torch.random.set_rng_state(state)
    z_only = LocalSpatialBottleneckBlock(4, readout='z', seed=2)
    assert rich.state_dict().keys() == z_only.state_dict().keys()
    assert all(torch.equal(value, z_only.state_dict()[key]) for key, value in rich.state_dict().items())
    tokens = torch.randn(3, 4)
    z, full = z_only._evaluate_core(tokens), rich._evaluate_core(tokens)
    torch.testing.assert_close(z[:, :4], full[:, :4])
    assert z[:, 4:].count_nonzero() == 0


def test_core_seed_offset_accepts_the_full_torch_seed_range():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((1 << 64) - 1)
        first = LocalSpatialBottleneckBlock(1, feature_channels=1)
        torch.manual_seed((1 << 64) - 1)
        second = LocalSpatialBottleneckBlock(1, feature_channels=1)
        assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())


def test_core_initialization_does_not_seed_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: calls.append(seed))
    LocalSpatialBottleneckBlock(2, feature_channels=1, seed=7)
    assert calls == []


def test_all_controls_share_identical_backbone_and_projection_initial_values():
    configurations = [config(), config(readout='z'), config(control='frozen'), config(control='no_entanglement')]
    configurations += [config(kind) for kind in ('mlp', 'conv', 'sincos')]
    reference = None
    for cfg in configurations:
        tc.seed_everything(42)
        cls = QVUNetSegmentation if cfg['local_spatial_kernel'] == 'quantum' else ClassicalAblationUNetSegmentation
        model = tc.build_model(cls, cfg)
        shared = {name: value for name, value in model.state_dict().items()
                  if '.quantum_params.' not in name and '.input_scales.' not in name and '.core.' not in name}
        if reference is None:
            reference = shared
        assert shared.keys() == reference.keys()
        assert all(torch.equal(reference[key], value) for key, value in shared.items())
        target, actual = tc.bottleneck_param_counts(cfg, 'quantum' if cls is QVUNetSegmentation else 'ablation')
        block = model.model.vertex.q_block
        assert actual == sum(parameter.numel() for parameter in block.parameters())
        assert target == block.bottleneck_target_params


def test_chunked_checkpoint_forward_and_backward_match_unchunked():
    block = LocalSpatialBottleneckBlock(2, feature_channels=1, seed=3)
    clone = copy.deepcopy(block)
    clone._MAX_TOKENS = 3
    first = torch.randn(1, 2, 3, 3, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    a, b = block(first), clone(second)
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(first.grad, second.grad, atol=1e-6, rtol=1e-5)
    for parameter, other in zip(block.parameters(), clone.parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('control', ['trainable', 'frozen', 'no_entanglement'])
def test_checkpoint_replay_preserves_local_architecture_and_frozen_flags(tmp_path, control):
    cfg = config(control=control)
    model = tc.build_model(QVUNetSegmentation, cfg)
    images = torch.randn(1, 3, 32, 32)
    model(images).square().mean().backward()
    block = model.model.vertex.q_block
    assert block.pre_q_conv.weight.grad.abs().max() > 0
    assert model.model.down.init_conv.weight.grad.abs().max() > 0
    for parameter in (*block.quantum_params, *block.input_scales):
        assert parameter.requires_grad == (control != 'frozen')
        assert (parameter.grad is None) == (control == 'frozen')
    path = tmp_path / 'checkpoint.pt'
    torch.save(dict(best_state_dict=model.state_dict(), settings=cfg, model_type='quantum'), path)
    restored, loaded, kind = load_checkpoint_model(path, data_root=tmp_path)
    assert kind == 'quantum' and loaded['local_spatial_readout'] == 'xz_zz'
    assert restored.model.vertex.q_block.q_training_control == control
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(images), model(images))


@pytest.mark.parametrize('kind', ['quantum', 'mlp', 'conv', 'sincos'])
def test_knockout_is_repeatable_input_dependent_and_leaves_weights_untouched(kind):
    block = LocalSpatialBottleneckBlock(2, kind=kind, feature_channels=1, seed=4)
    images = torch.randn(1, 2, 3, 3)
    initial = {key: value.clone() for key, value in block.state_dict().items()}
    on = block(images)
    block.set_quantum_mode('zero')
    assert block(images).count_nonzero() == 0
    block.set_quantum_mode('random', knockout_seed=9)
    first = block(images)
    block.set_quantum_mode('random', knockout_seed=9)
    torch.testing.assert_close(first, block(images))
    assert not torch.equal(first, block(images + 0.3))
    assert not torch.equal(first, on)
    block.set_quantum_mode('on')
    torch.testing.assert_close(block(images), on)
    assert all(torch.equal(value, initial[key]) for key, value in block.state_dict().items())


def test_metadata_fingerprints_tags_and_invalid_combinations():
    cfg = dict(config(), fed_rounds=1, local_epochs=1, num_clients=2)
    for fingerprint in (tc.centralized_fingerprint, tc.run_fingerprint):
        initial = fingerprint(cfg, 'quantum')
        assert initial['local_spatial_readout'] == 'xz_zz'
        for change in ({'local_spatial_readout': 'z'}, {'local_spatial_channels': 2},
                       {'q_training_control': 'frozen'}):
            assert fingerprint(dict(cfg, **change), 'quantum') != initial
    assert tc.variant_tag(config()) != tc.variant_tag(config(readout='z'))
    assert tc.variant_tag(config()) != tc.variant_tag(config(control='frozen'))
    original = dict(config(), local_spatial_kernel=None)
    assert tc.variant_tag(original) != tc.variant_tag(dict(original, q_training_control='frozen'))
    assert tc.variant_tag(original) != tc.variant_tag(dict(original, q_training_control='no_entanglement'))
    for invalid in ({'local_spatial_channels': True}, {'local_spatial_channels': 1.5},
                    {'local_spatial_readout': 'nonsense'}, {'local_spatial_kernel': 'nonsense'}):
        with pytest.raises(ValueError):
            local_spatial_options(dict(cfg, **invalid))
    for change in ({'q_backend': 'torch'}, {'q_spatial_mode': 'patch4x4'}, {'n_qubits': 8}, {'q_ansatz': 'hqunet'}):
        with pytest.raises(ValueError):
            build_local_spatial_block(dict(cfg, **change), 'quantum')
    with pytest.raises(ValueError, match='model_type'):
        build_local_spatial_block(config('mlp'), 'quantum')


def test_angle_diagnostics_reports_saturation_without_changing_gradients():
    block = LocalSpatialBottleneckBlock(1, feature_channels=1, seed=5)
    with torch.no_grad():
        block.pre_q_conv.weight.fill_(1)
        block.pre_q_conv.bias.zero_()
    low = block.angle_diagnostics(torch.zeros(1, 1, 2, 2))
    high = block.angle_diagnostics(torch.full((1, 1, 2, 2), 100.0))
    assert low['n_angles'] == 16
    assert low['saturation_fraction'] == 0
    assert high['saturation_fraction'] == 1
    assert all(parameter.grad is None for parameter in block.parameters())


@pytest.mark.parametrize('kind', ['mlp', 'conv'])
def test_unbounded_classical_cores_have_no_input_angle_diagnostic(kind):
    from scripts.run_real_spatial_pilot import output_probe
    block = LocalSpatialBottleneckBlock(1, kind=kind, feature_channels=1)
    images = torch.randn(1, 1, 2, 2)
    assert block.angle_diagnostics(images) is None
    assert 'input_angles' not in output_probe(block, [block], images)[0]


@pytest.mark.parametrize('kind', ['mlp', 'conv', 'sincos'])
def test_classical_checkpoint_replay_preserves_local_core(tmp_path, kind):
    cfg = config(kind)
    model = tc.build_model(ClassicalAblationUNetSegmentation, cfg).eval()
    path = tmp_path / 'checkpoint.pt'
    torch.save(dict(best_state_dict=model.state_dict(), settings=cfg, model_type='ablation'), path)
    restored, loaded, model_type = load_checkpoint_model(path, data_root=tmp_path)
    assert model_type == 'ablation' and loaded['local_spatial_kernel'] == kind
    assert restored.model.vertex.q_block.kind == kind
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(restored(images), model(images))


def test_checkpoint_rejects_missing_local_control_semantics(tmp_path):
    cfg = config(control='frozen')
    model = tc.build_model(QVUNetSegmentation, cfg)
    cfg.pop('q_training_control')
    path = tmp_path / 'checkpoint.pt'
    torch.save(dict(best_state_dict=model.state_dict(), settings=cfg, model_type='quantum'), path)
    with pytest.raises(ValueError, match='q_training_control'):
        load_checkpoint_model(path, data_root=tmp_path)


@pytest.mark.parametrize('federated', [False, True])
def test_common_cli_preserves_local_options_and_control(federated):
    parser = argparse.ArgumentParser()
    tc.add_common_args(parser, quantum=True, federated=federated)
    args = parser.parse_args(['--local-spatial-kernel', 'quantum', '--local-spatial-readout', 'z',
        '--local-spatial-channels', '2', '--q-training-control', 'frozen', '--q-layers', '1'])
    cfg = tc.apply_common_args(config(), args)
    assert cfg['local_spatial_kernel'] == 'quantum'
    assert cfg['local_spatial_readout'] == 'z'
    assert cfg['local_spatial_channels'] == 2
    assert cfg['q_training_control'] == 'frozen'
    assert cfg['q_layers'] == 1
    block = build_local_spatial_block(cfg, 'quantum')
    assert not block.quantum_params[0].requires_grad
