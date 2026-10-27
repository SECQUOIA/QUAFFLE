"""Training controls must preserve initialization, gradients and replay semantics."""
import pytest
import torch
import json
from pathlib import Path
import subprocess
import sys

from scripts.run_parameter_budget_comparison import base_config
from tools import training_common as tc
from tools.checkpoint_evaluation import load_checkpoint_model
from utilsTorch.unetTorch import QuantumBlock
from utilsTorch.utilsTorch import QVUNetSegmentation


def block(control):
    return QuantumBlock(quantum_channels=4, seed=19, q_training_control=control)


def test_no_entanglement_has_no_cross_input_derivatives_and_does_not_change_cached_trainable_circuit():
    x = torch.tensor([[0.2, -0.4, 0.7, 0.1]], requires_grad=True)
    independent = block('no_entanglement')
    trained = block('trainable')
    assert all(torch.equal(a, b) for a, b in zip(independent.parameters(), trained.parameters()))
    def jacobian(model):
        return torch.autograd.functional.jacobian(model._eval_tokens_pennylane_flat, x)[0, :, 0, :]
    separate = jacobian(independent)
    entangled = jacobian(trained)
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    assert separate[off_diagonal].abs().max() < 1e-7
    assert separate.diag().abs().max() > 1e-3
    assert entangled[off_diagonal].abs().max() > 1e-3
    independent.set_q_backend('pennylane_pergroup')
    tokens = x[:, :, None, None]
    torch.testing.assert_close(independent(tokens).flatten(1), independent._eval_tokens_pennylane_flat(x))


def test_frozen_angles_and_scales_preserve_initial_values_and_input_gradient():
    frozen, trained = block('frozen'), block('trainable')
    assert all(torch.equal(a, b) for a, b in zip(frozen.parameters(), trained.parameters()))
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    x = torch.tensor([[0.2, -0.4, 0.7, 0.1]], requires_grad=True)
    frozen._eval_tokens_pennylane_flat(x).sum().backward()
    assert x.grad is not None and x.grad.abs().max() > 1e-3
    assert all(parameter.grad is None for parameter in frozen.parameters())


@pytest.mark.parametrize('control', ['frozen', 'no_entanglement'])
def test_selected_checkpoint_replays_control_and_preserves_backbone_gradients(tmp_path, control):
    cfg = dict(base_config(8, 1), dataset='combined', image_size=32, seed=42,
               q_training_control=control, batch_size=1, learning_rate=0.001, num_epochs=1)
    model = tc.build_model(QVUNetSegmentation, cfg)
    images = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    model(images).square().mean().backward()
    assert model.model.down.init_conv.weight.grad.abs().max() > 0
    core = model.model.vertex.q_block
    assert all(p.grad is None for p in core.parameters()) if control == 'frozen' else any(
        p.grad is not None and p.grad.abs().max() > 0 for p in core.parameters())
    path = tmp_path / 'checkpoint.pt'
    torch.save(dict(best_state_dict=model.state_dict(), settings=cfg, model_type='quantum'), path)
    restored, loaded, kind = load_checkpoint_model(path, data_root=tmp_path)
    assert kind == 'quantum' and loaded['q_training_control'] == control
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(images), model(images))
    assert all(p.requires_grad == (control != 'frozen') for p in restored.model.vertex.q_block.parameters())


def test_controls_are_bound_into_centralized_and_federated_fingerprints():
    cfg = dict(base_config(8, 1), seed=42, image_size=32, batch_size=1, learning_rate=0.001,
               num_epochs=1, fed_rounds=1, local_epochs=1, num_clients=2)
    for fingerprint in (tc.centralized_fingerprint, tc.run_fingerprint):
        original = fingerprint(cfg, 'quantum')
        assert original['q_training_control'] == 'trainable'
        for control in ('frozen', 'no_entanglement'):
            assert fingerprint(dict(cfg, q_training_control=control), 'quantum') != original
        legacy = {k: v for k, v in original.items() if k != 'q_training_control'}
        assert tc._normalize_saved_fingerprint(legacy)['q_training_control'] == 'trainable'


def test_unsupported_control_backends_and_ansatz_are_rejected():
    with pytest.raises(ValueError, match='PennyLane'):
        QuantumBlock(q_backend='torch', q_training_control='no_entanglement')
    with pytest.raises(ValueError, match='reupload'):
        QuantumBlock(q_ansatz='legacy', q_training_control='frozen')
    with pytest.raises(ValueError, match='q_training_control'):
        block('unknown')
    with pytest.raises(ValueError, match='PennyLane'):
        block('no_entanglement').set_q_backend('torch')


def test_control_plan_reads_only_validation_and_inherits_the_original_budget(tmp_path):
    from scripts.run_quantum_training_controls import prepare
    settings = dict(seeds=[42, 77, 123], learning_rates=[0.001, 0.003], epochs=60,
                    eval_every=2, fit_cap=128, batch_size=16, image_size=64, device='cuda')
    specs = []
    for mode, gain in [('global', 0.01), ('patch4x4', 0.009)]:
        for kind in ('mlp', 'quantum'):
            name = kind + '_' + mode
            specs.append(dict(id=name, config=dict(base_config(8, 1), q_spatial_mode=mode)))
            for seed in settings['seeds']:
                rates = settings['learning_rates'] if seed == 42 else [0.001]
                for rate in rates:
                    row = dict(id=name, seed=seed, learning_rate=rate,
                               val={'iou': 0.5 + (gain if kind == 'quantum' else 0)},
                               diagnostics={'movement': {'circuit_angles': {'l2': 0.1}}})
                    path = (tmp_path / 'tuning' / name / str(rate) / 'seed42' if seed == 42
                            else tmp_path / 'replication' / name / ('seed' + str(seed)))
                    path.mkdir(parents=True, exist_ok=True)
                    (path / 'metrics.json').write_text(json.dumps(row))
    (tmp_path / 'plan.json').write_text(json.dumps(dict(settings=settings, specs=specs,
        source_revision='fixture', input_identity={'raw': 'fixture'})))
    (tmp_path / 'summary.json').write_text('Invalid JSON: test outcomes must not be read')
    inherited, controls, context = prepare(tmp_path)
    assert inherited == settings
    assert {spec['config']['q_training_control'] for spec in controls} == {'frozen', 'no_entanglement'}
    assert all(spec['config']['q_spatial_mode'] == 'global' for spec in controls)
    assert context['validation_control_decision']['selected_mode'] == 'global'
    for path in tmp_path.glob('replication/quantum_*/seed*/metrics.json'):
        row = json.loads(path.read_text())
        row['val']['iou'] = 0.4
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Neither quantum mode'):
        prepare(tmp_path)


def test_bounded_control_training_freezes_only_the_circuit_and_records_all_candidates(tmp_path):
    from tools.make_mini_flood_dataset import generate_mini_flood_dataset
    data, run = tmp_path / 'data', tmp_path / 'run'
    generate_mini_flood_dataset(str(data), n_train=8, n_test=3, seed=42, seeds=[42, 77, 123])
    code = '''
import sys
from scripts import run_real_spatial_pilot as pilot
from scripts.run_parameter_budget_comparison import base_config
specs = [dict(id='quantum_' + control + '_global', model_type='quantum',
              config=dict(base_config(8, 1), dataset='combined', q_training_control=control))
         for control in ('frozen', 'no_entanglement')]
pilot.main(sys.argv[1:], spec_factory=lambda: specs,
           extra_source_files=('scripts/run_quantum_training_controls.py', 'scripts/check_spatial_control_trigger.py'),
           experiment_context={'kind': 'control_api_smoke'})
'''
    args = ['--data-root', str(data), '--work-dir', str(run), '--device', 'cpu',
            '--seeds', '42', '77', '123', '--learning-rates', '0.001', '0.003',
            '--epochs', '1', '--eval-every', '1', '--fit-cap', '8', '--batch-size', '2', '--image-size', '32']
    with (tmp_path / 'smoke.log').open('w') as log:
        subprocess.run([sys.executable, '-c', code, *args], check=True, stdout=log, stderr=subprocess.STDOUT)
    summary = json.loads((run / 'summary.json').read_text())
    assert len(summary['records']) == 6
    assert summary['plan']['experiment_context'] == {'kind': 'control_api_smoke'}
    assert 'scripts/run_quantum_training_controls.py' in summary['plan']['source_hashes']
    candidates = json.loads((run / 'tuning_candidates.json').read_text())
    assert len(candidates) == 4 and all('test' not in row for row in candidates)
    assert len(list(run.glob('replication/*/seed*/metrics.json'))) == 4
    assert all(len(row['test_names']) == 6 for row in summary['records'])
    frozen = [row for row in summary['records'] if row['settings']['q_training_control'] == 'frozen']
    assert all(row['trainable_params'] == row['total_params'] - 512 for row in frozen)
    assert all(row['diagnostics']['movement']['circuit_angles']['l2'] == 0 for row in frozen)
    assert all(row['diagnostics']['movement']['input_scales']['l2'] == 0 for row in frozen)
    assert all(row['diagnostics']['movement']['backbone']['l2'] > 0 for row in frozen)
    for row in candidates:
        other = next(item for item in candidates if item['id'] != row['id'] and item['learning_rate'] == row['learning_rate'])
        assert row['initial_parameter_sha256'] == other['initial_parameter_sha256']
