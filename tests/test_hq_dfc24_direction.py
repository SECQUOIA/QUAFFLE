"""Data pairing, checkpoint buffers and the complete-grid test boundary."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_hq_dfc24_direction as study


@pytest.fixture
def tiny_api(monkeypatch):
    """Real Torch/BN/optimizer mechanics without invoking the16-qubit simulator."""
    import torch
    from torch import nn

    class Bottleneck(nn.Module):
        def __init__(self, kind):
            super().__init__()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(100 if kind == 'quantum' else 200)
                self.core = nn.Conv2d(3, 3, 1)
            self.mode = 'on'

        def core_parameters(self):
            return self.core.parameters()

        def angle_diagnostics(self, values):
            return dict(n_angles=values.numel(), min_radians=float(values.min()),
                        max_radians=float(values.max()), mean_abs_radians=float(values.abs().mean()),
                        saturation_fraction=float((values.abs() >= .95).float().mean()))

        def set_mode(self, mode, seed=0):
            assert mode in ('on', *study.MODES)
            self.mode = mode

        def forward(self, values):
            if self.mode in ('zero_core', 'zero_bottleneck'):
                return torch.zeros_like(values)
            output = self.core(values)
            return -output if self.mode == 'random_core' else output

    class Model(nn.Module):
        def __init__(self, kind, seed, base_channels, image_size, in_channels=3, out_channels=2):
            super().__init__()
            self.config = dict(kind=kind, seed=seed, base_channels=base_channels,
                               image_size=image_size, in_channels=in_channels, out_channels=out_channels)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.stem = nn.Conv2d(3, 3, 1)
                self.bn = nn.BatchNorm2d(3)
                self.bottleneck = Bottleneck(kind)
                self.head = nn.Conv2d(3, 2, 1)

        def forward(self, values):
            return self.head(torch.relu(self.bottleneck(self.bn(self.stem(values)))))

    monkeypatch.setattr(study, 'build_model', lambda kind, seed, args: Model(kind, seed, args.base_channels, args.image_size))
    monkeypatch.setattr(study, 'model_configuration', lambda model: dict(model.config))
    monkeypatch.setattr(study, 'model_from_config', lambda cfg: Model(**cfg))
    monkeypatch.setattr(study, 'shared_hash', lambda model: study.state_hash({name: value for name, value in model.state_dict().items()
        if not name.startswith('bottleneck.core.')}))
    return Model


def args_for(tmp_path):
    return SimpleNamespace(phase='run', data_root=tmp_path / 'data', work_dir=tmp_path / 'run',
        device='cpu', batch_size=2, epochs=1, fit_cap=4, image_size=128, base_channels=8,
        eval_every=1, augment=True, resume=False)


def metrics(n_samples, quality=0):
    samples = (71, 134) if n_samples == 205 else (35, 195)
    result = dict(loss=.5, auc=.7)
    totals = dict(tp=0, fp=0, fn=0, tn=0)
    for sensor, count in zip(('optical', 'sar'), samples):
        values = dict(tp=count * (512 + quality), fp=count * 128, fn=count * 256)
        values['tn'] = count * 128 * 128 - sum(values.values())
        for key, value in values.items():
            result[sensor + '_subset_' + key] = value
            totals[key] += value
        result[sensor + '_subset_n_samples'] = count
        result[sensor + '_subset_iou'] = values['tp'] / (values['tp'] + values['fp'] + values['fn'])
    tp, fp, fn, tn = (totals[key] for key in ('tp', 'fp', 'fn', 'tn'))
    result.update(iou=tp / (tp + fp + fn), dice=2 * tp / (2 * tp + fp + fn),
                  accuracy=(tp + tn) / sum(totals.values()))
    return result


@pytest.fixture
def fake_campaign(tmp_path, monkeypatch, tiny_api):
    import torch
    class Data:
        names = ['tile-' + str(index) for index in range(1533)]
        modalities = ['optical' if index % 5 == 0 else 'sar' for index in range(1533)]

        def __len__(self):
            return 1533

        def __getitem__(self, index):
            grid = torch.arange(128 * 128).float().reshape(128, 128) / (128 * 128)
            return torch.stack((grid, grid.T, grid.flip(0))) + index * .001, (grid > .5).float().unsqueeze(0)

    def data_for(args, seed):
        dataset = Data()
        fit, validation = list(range(args.fit_cap)), list(range(1303, 1533))
        return dict(dataset=dataset, fit=fit, validation=validation, all_fit=list(range(1303)),
            train=torch.utils.data.Subset(dataset, fit), val=torch.utils.data.Subset(dataset, validation),
            normalization=dict(fit_only=True, seed=seed, stats_sha256='fixture-' + str(seed)))

    source = tmp_path / 'source'
    (source / 'docs').mkdir(parents=True)
    (source / 'docs/HQ_DFC24_DIRECTION_PROTOCOL.md').write_text('Synthetic protocol fixture')
    monkeypatch.setattr(study, 'ROOT', source)
    monkeypatch.setattr(study, 'source_hashes', lambda: {'producer.py': 'same-executable'})
    monkeypatch.setattr(study, 'runtime', lambda device: dict(device=str(device), python='fixture', threads=2))
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *args, **kwargs: 'original-producer\n')
    monkeypatch.setattr(study.pilot, 'load_training_data', data_for)
    monkeypatch.setattr(study.pilot, 'verify_inputs', lambda root: {'raw': 'same-inputs'})
    monkeypatch.setattr(study, 'evaluate', lambda model, loader, device:
        metrics(len(loader.dataset), int(model.bn.num_batches_tracked)))
    return data_for


def test_schedule_is_stateless_and_joint_d4_preserves_image_mask_alignment():
    import random
    import torch
    from tools import training_common as tc
    indices = list(range(17, 49))
    first = study.epoch_schedule(indices, 42, 1, True)
    random.seed(999)
    torch.manual_seed(987)
    _ = [random.random() for _ in range(100)]
    assert study.epoch_schedule(indices, 42, 1, True) == first
    assert study.epoch_schedule(indices, 42, 2, True) != first
    assert set(first[1].values()) == set(range(8))
    image = torch.arange(16).reshape(1, 4, 4)
    dataset = [(image, image.clone()) for _ in indices]
    augmented = study.EpochDataset(dataset, indices, 42, 1, True)
    for index in augmented.order:
        actual_image, actual_mask = augmented[index]
        expected, _ = tc.apply_dihedral_transform(image, image, augmented.codes[index])
        assert torch.equal(actual_image, expected) and torch.equal(actual_mask, expected)
    assert augmented.observed_digest() == augmented.expected_digest
    assert set(study.epoch_schedule(indices, 42, 1, False)[1].values()) == {0}


@pytest.mark.parametrize('field', ['source_hashes', 'input_identity', 'runtime', 'settings'])
def test_resume_rejects_changed_execution_or_data_but_preserves_original_revision(tmp_path, field):
    path = tmp_path / 'run/plan.json'
    plan = dict(source_revision='first', source_hashes={'script': 'same'}, input_identity={'data': 'same'},
                runtime={'torch': 'same'}, settings={'augment': True})
    study.frozen_plan(path, plan, False)
    assert study.frozen_plan(path, dict(plan, source_revision='reporting-only-child'), True) == plan
    changed = dict(plan, **{field: {'changed': True}})
    with pytest.raises(ValueError, match=field):
        study.frozen_plan(path, changed, True)
    assert json.loads(path.read_text()) == plan


def test_eight_fit_boundary_strict_bn_checkpoints_and_resume(tmp_path, monkeypatch, fake_campaign):
    import torch
    from tools import training_common as tc
    args = args_for(tmp_path)
    completed, test_calls = [], []
    original_fit = study.fit_one
    def fit(*values, **kwargs):
        row = original_fit(*values, **kwargs)
        completed.append((row['id'], row['seed'], row['learning_rate']))
        return row
    def evaluated(model, loader, device):
        model.eval()
        if len(loader.dataset) == 205:
            assert len(completed) == len(set(completed)) == 8
            assert (args.work_dir / 'checkpoint_audit.json').is_file()
            test_calls.append(model.bottleneck.mode)
        return metrics(len(loader.dataset), int(model.bn.num_batches_tracked))
    monkeypatch.setattr(study, 'fit_one', fit)
    monkeypatch.setattr(study, 'evaluate', evaluated)
    monkeypatch.setattr(tc, 'resolve_data_dirs', lambda *args: (None, None, None, 'test-images', 'test-masks'))
    class TestData:
        def __init__(self, cfg):
            self.norm_provenance = cfg['_resolved_norm_provenance']
            self.names = ['test-' + str(i) for i in range(205)]
        def __len__(self):
            return 205
    monkeypatch.setattr(tc, 'load_dataset', lambda images, masks, cfg: TestData(cfg))
    result = study.run(args, torch.device('cpu'))
    assert len(completed) == 8 and len(test_calls) == 24
    assert set(test_calls) == {'on', *study.MODES}
    assert result['selection'] == {'hq_quantum': .001, 'hq_mlp': .001}
    assert len(result['records']) == 6
    audits = json.loads((args.work_dir / 'checkpoint_audit.json').read_text())
    assert len(audits) == 8 and all(row['finite_weights_and_bn_buffers'] for row in audits)
    fits = json.loads((args.work_dir / 'all_fits.json').read_text())
    assert len(fits) == 8 and all('test' not in row['record'] for row in fits)
    paths = [args.work_dir / item['directory'] / 'metrics.json' for item in fits]
    before = {str(path): path.stat().st_mtime_ns for path in paths}
    assert all(row['best_epoch'] == 1 for row in result['records'])
    checkpoint = args.work_dir / result['records'][0]['checkpoint']
    model, _ = study.restore_checkpoint(checkpoint, result['records'][0], 'cpu')
    assert int(model.bn.num_batches_tracked) == 2
    assert torch.isfinite(model.bn.running_var).all()
    # A reporting-only revision must not restamp the original producer or refit.
    completed.clear()
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *args, **kwargs: 'reporting-child\n')
    args.resume = True
    resumed = study.run(args, torch.device('cpu'))
    assert resumed['plan']['source_revision'] == 'original-producer'
    assert {str(path): path.stat().st_mtime_ns for path in paths} == before


def test_epoch_zero_can_win_and_bn_buffers_are_strictly_audited(tmp_path, monkeypatch, fake_campaign):
    import torch
    args = args_for(tmp_path)
    monkeypatch.setattr(study, 'evaluate', lambda model, loader, device: metrics(len(loader.dataset)))
    row = study.fit_one('quantum', 42, .001, args, fake_campaign(args, 42), tmp_path / 'fit', torch.device('cpu'))
    assert row['best_epoch'] == 0
    assert row['selected_state_sha256'] == row['initial_state_sha256']
    path = tmp_path / 'fit/checkpoint.pt'
    checkpoint = torch.load(path, weights_only=False)
    assert int(checkpoint['best_state_dict']['bn.num_batches_tracked']) == 0
    checkpoint['best_state_dict']['bn.running_var'][0] = float('nan')
    torch.save(checkpoint, path)
    with pytest.raises(FloatingPointError, match='BatchNorm'):
        study.restore_checkpoint(path, row)


def test_failed_fit_is_retained_and_cannot_be_silently_overwritten(tmp_path, monkeypatch, fake_campaign):
    import torch
    from utilsTorch import utilsTorch as ut
    args = args_for(tmp_path)
    monkeypatch.setattr(ut, 'segmentation_loss', lambda *args, **kwargs: torch.tensor(float('nan')))
    directory = tmp_path / 'failed'
    with pytest.raises(ValueError, match='Nonfinite training loss'):
        study.fit_one('quantum', 42, .001, args, fake_campaign(args, 42), directory, torch.device('cpu'))
    failure = (directory / 'failure.json').read_bytes()
    history = (directory / 'history.json').read_bytes()
    with pytest.raises(ValueError, match='Incomplete attempt preserved'):
        study.fit_one('quantum', 42, .001, args, fake_campaign(args, 42), directory, torch.device('cpu'))
    assert (directory / 'failure.json').read_bytes() == failure
    assert (directory / 'history.json').read_bytes() == history


def test_cpu_profile_uses_real_gradients_core_movement_and_bn_roundtrip(tmp_path, monkeypatch, fake_campaign):
    import torch
    args = args_for(tmp_path)
    args.phase = 'profile'
    rows = study.profile(args, torch.device('cpu'))
    assert {row['kind'] for row in rows} == set(study.KINDS)
    assert len({row['shared_initial_state_sha256'] for row in rows}) == 1
    assert all(row['gradients']['core'] > 0 and row['movement']['core']['l2'] > 0 for row in rows)
    assert all(row['strict_state_and_bn_roundtrip'] for row in rows)


def test_analysis_rejects_missing_duplicate_or_unpaired_rows():
    rows = []
    for kind in study.KINDS:
        for seed in study.SEEDS:
            value = .5 + (.01 if kind == 'quantum' else 0)
            rows.append(dict(id='hq_' + kind, seed=seed, total_params=4, core_params=1,
                learning_rate=.001, fit_names=['fit'], val_names=['val'], test_names=['test'],
                normalization={'same': True}, initial_shared_state_sha256='same', history=[],
                val=dict(iou=value, optical_subset_iou=value, sar_subset_iou=value),
                test=dict(iou=value, optical_subset_iou=value, sar_subset_iou=value)))
    result = study.analysis(rows)
    assert all(row['mean'] == pytest.approx(1) and row['wins'] == 3 for row in result['paired_iou'])
    with pytest.raises(ValueError, match='Duplicate'):
        study.analysis(rows + [rows[0]])
    with pytest.raises(ValueError, match='three paired seeds'):
        study.analysis(rows[:-1])
    rows[-1]['normalization'] = {'different': True}
    with pytest.raises(ValueError, match='Unpaired'):
        study.analysis(rows)


@pytest.mark.parametrize('fail', [False, True])
def test_knockout_boundary_uses_actual_hq_bottleneck_api_and_restores_on(tmp_path, monkeypatch, fail):
    from utilsTorch.hq_direction import HQDirectionUNet
    model = HQDirectionUNet(kind='mlp', seed=42, base_channels=8, image_size=128,
                            in_channels=3, out_channels=2)
    calls = []
    actual_setter = model.bottleneck.set_mode
    def setter(mode, seed=0):
        calls.append((mode, seed))
        return actual_setter(mode, seed=seed)
    def evaluated(*args):
        if fail:
            raise RuntimeError('synthetic evaluation failure')
        return metrics(205)
    monkeypatch.setattr(model.bottleneck, 'set_mode', setter)
    monkeypatch.setattr(study, 'evaluate', evaluated)
    if fail:
        with pytest.raises(RuntimeError, match='synthetic evaluation failure'):
            study.knockout_evaluation(model, None, 'cpu', 77)
        assert calls == [('zero_core', 77), ('on', 77)]
    else:
        result = study.knockout_evaluation(model, None, 'cpu', 77)
        assert set(result) == set(study.MODES)
        assert calls == [(mode, 77) for mode in (*study.MODES, 'on')]
