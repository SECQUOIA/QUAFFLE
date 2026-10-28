"""Synthetic fit mechanics and the fixed-rate20-fit/15-replay boundary."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import collect_hq_dfc24_direction as legacy_collector
from scripts import run_hq_training_controls as study


def arguments(root, *, tiny=False):
    return SimpleNamespace(phase='run', data_root=root / 'data', work_dir=root / 'run',
        legacy_root=root / 'legacy', profile=root / 'profile.json', device='cpu',
        batch_size=2 if tiny else 8, epochs=1 if tiny else 60,
        fit_cap=2 if tiny else 128, image_size=128, base_channels=8,
        eval_every=1 if tiny else 2, augment=True, resume=False)


class SyntheticTiles(torch.utils.data.Dataset):
    names = ['synthetic-' + str(index) for index in range(6)]
    modalities = ['optical', 'sar', 'optical', 'sar', 'optical', 'sar']

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        axis = torch.linspace(-1, 1, 128)
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        image = torch.stack((x + .03*index, y - .02*index, x*y + .01*index))
        mask = (x + y > .05*(index-2)).float().unsqueeze(0)
        return image, mask


def synthetic_data(seed=42):
    dataset = SyntheticTiles()
    fit, validation, all_fit = [0, 1], [2, 3], [0, 1, 4, 5]
    return dict(dataset=dataset, fit=fit, validation=validation, all_fit=all_fit,
        train=torch.utils.data.Subset(dataset, fit), val=torch.utils.data.Subset(dataset, validation),
        normalization=dict(fit_only=True, seed=seed, stats_sha256='synthetic-no-data-loading'))


@pytest.fixture(scope='module')
def actual_fits(tmp_path_factory):
    """Three actual128px fits; no real tiles, GPU or fake model/evaluation."""
    root = tmp_path_factory.mktemp('hq-controls-real-mechanics')
    args, data = arguments(root, tiny=True), synthetic_data()
    result = {}
    for kind in study.NEW_KINDS:
        directory = root / kind
        row = study.fit_one(kind, 42, .003, args, data, directory, torch.device('cpu'))
        result[kind] = dict(row=row, directory=directory, args=args, data=data)
    return result


def test_imported_runner_and_factory_are_the_requested_worktree():
    from utilsTorch import hq_training_controls
    root = Path(__file__).resolve().parents[1]
    assert Path(study.__file__).resolve() == root / 'scripts/run_hq_training_controls.py'
    assert Path(hq_training_controls.__file__).resolve() == root / 'utilsTorch/hq_training_controls.py'


@pytest.mark.parametrize('kind', study.NEW_KINDS)
def test_real_one_epoch_fits_preserve_frozen_state_and_restore_bn_trainability(actual_fits, kind):
    fixture = actual_fits[kind]
    row, directory = fixture['row'], fixture['directory']
    assert [point['epoch'] for point in row['history']] == [0, 1]
    assert len(row['history'][1]['gradient_norms']) == 2
    assert row['history'][1]['gradient_norms']['common']['mean'] > 0
    assert row['diagnostics']['gradient_norms']['common']['steps'] == 1
    frozen = row['diagnostics']['frozen_parameters']
    assert frozen['unchanged'] and frozen['initial_sha256'] == frozen['selected_sha256']
    assert json.loads((directory / 'metrics.json').read_text()) == row
    assert json.loads((directory / 'history.json').read_text()) == row['history']
    assert row['diagnostics']['angle_probe_scope'] == (
        'inactive_counterfactual' if kind == 'zero_bottleneck' else 'active_fixed_probe')
    restored, audit = study.restore_checkpoint(directory / 'checkpoint.pt', row)
    assert audit['strict_restore'] and audit['finite_weights_and_bn_buffers']
    assert audit['checkpoint_sha256'] == study.digest(directory / 'checkpoint.pt')
    assert study.trainability(restored) == row['trainability']
    assert study.counts(restored) == row['parameter_counts']
    assert int(restored.stem[2].num_batches_tracked) == row['best_epoch']
    assert study.state_hash(restored.state_dict()) == row['selected_state_sha256']
    assert all(torch.isfinite(value).all() for value in restored.state_dict().values())
    if kind in ('frozen_quantum', 'zero_bottleneck'):
        assert row['history'][1]['gradient_norms']['core']['max'] == 0
        assert row['diagnostics']['movement']['core']['l2'] == 0
        assert all(not value.requires_grad for value in restored.bottleneck.core_parameters())
    else:
        assert row['history'][1]['gradient_norms']['core']['min'] > 0
        assert row['parameter_counts']['core_total'] == 1568
    if kind == 'zero_bottleneck':
        assert restored.bottleneck.mode == 'zero_bottleneck'
        assert all(not value.requires_grad for value in restored.bottleneck.parameters())
        restored.train()
        restored.bottleneck.set_mode('on')
        assert torch.count_nonzero(restored.bottleneck(torch.randn(1, 64, 16, 16))) == 0


def test_frozen_checkpoint_retains_real_circuit_input_gradients(actual_fits):
    fixture = actual_fits['frozen_quantum']
    model, _ = study.restore_checkpoint(fixture['directory'] / 'checkpoint.pt', fixture['row'])
    latent = torch.randn(1, 64, 16, 16, requires_grad=True)
    output = model.bottleneck(latent)
    coefficients = torch.linspace(-.7, .9, output.numel()).reshape_as(output)
    (output*coefficients).mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
    assert model.bottleneck.pre_projection.weight.grad is not None
    assert model.bottleneck.pre_projection.weight.grad.abs().sum() > 0
    assert model.bottleneck.core.weights.grad is None


@pytest.mark.parametrize('field', ('kind', 'seed', 'best_epoch', 'settings', 'model_config', 'trainability', 'parameter_counts'))
def test_checkpoint_metadata_corruption_is_rejected(actual_fits, tmp_path, field):
    fixture = actual_fits['frozen_quantum']
    checkpoint = torch.load(fixture['directory'] / 'checkpoint.pt', weights_only=False)
    original = checkpoint[field]
    checkpoint[field] = dict(original, corrupt=True) if isinstance(original, dict) else str(original) + '-corrupt'
    path = tmp_path / 'corrupt.pt'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='metadata differs: ' + field):
        study.restore_checkpoint(path, fixture['row'])


@pytest.mark.parametrize('field', ('trainability', 'parameter_counts'))
def test_consistently_forged_semantics_still_fail_factory_validation(actual_fits, tmp_path, field):
    fixture = actual_fits['frozen_quantum']
    row = copy.deepcopy(fixture['row'])
    checkpoint = torch.load(fixture['directory'] / 'checkpoint.pt', weights_only=False)
    if field == 'trainability':
        row[field]['bottleneck.core.weights'] = True
    else:
        row[field]['core_trainable'] = 8
    checkpoint[field] = copy.deepcopy(row[field])
    path = tmp_path / 'forged.pt'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='semantics/counts differ'):
        study.restore_checkpoint(path, row)


@pytest.mark.parametrize('kind', study.NEW_KINDS)
@pytest.mark.parametrize('corruption', ('nonfinite_bn', 'missing_bn', 'changed_finite_state'))
def test_checkpoint_strict_bn_and_selected_state_integrity(actual_fits, tmp_path, kind, corruption):
    fixture = actual_fits[kind]
    checkpoint = torch.load(fixture['directory'] / 'checkpoint.pt', weights_only=False)
    if corruption == 'nonfinite_bn':
        checkpoint['best_state_dict']['stem.2.running_var'][0] = float('nan')
        expected_error, message = FloatingPointError, 'BatchNorm'
    elif corruption == 'missing_bn':
        checkpoint['best_state_dict'].pop('stem.2.running_mean')
        expected_error, message = RuntimeError, 'Missing key'
    else:
        checkpoint['best_state_dict']['stem.2.running_mean'][0] += 1
        expected_error, message = ValueError, 'Selected state differs'
    path = tmp_path / 'corrupt.pt'
    torch.save(checkpoint, path)
    with pytest.raises(expected_error, match=message):
        study.restore_checkpoint(path, fixture['row'])


@pytest.mark.parametrize('kind', ('frozen_quantum', 'zero_bottleneck'))
def test_forged_selected_hash_cannot_hide_changed_frozen_weights(actual_fits, tmp_path, kind):
    fixture = actual_fits[kind]
    row = copy.deepcopy(fixture['row'])
    checkpoint = torch.load(fixture['directory'] / 'checkpoint.pt', weights_only=False)
    checkpoint['best_state_dict']['bottleneck.core.weights'].add_(.1)
    row['selected_state_sha256'] = study.state_hash(checkpoint['best_state_dict'])
    path = tmp_path / 'forged-frozen.pt'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='Restored frozen parameters differ'):
        study.restore_checkpoint(path, row)


def test_completed_fit_resume_checks_config_and_never_retrains(actual_fits, monkeypatch):
    fixture = actual_fits['compact_affine']
    directory, args, data = fixture['directory'], fixture['args'], fixture['data']
    before = {path.name: (study.digest(path), path.stat().st_mtime_ns) for path in directory.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed fitting work must not run again')
    monkeypatch.setattr(study, 'build_model', forbidden)
    row = study.fit_one('compact_affine', 42, .003, args, data, directory, torch.device('cpu'))
    assert row == fixture['row']
    for field, value in (('epochs', 2), ('augment', False), ('base_channels', 4)):
        changed = copy.copy(args)
        setattr(changed, field, value)
        with pytest.raises(ValueError, match='config differs|settings differ'):
            study.fit_one('compact_affine', 42, .003, changed, data, directory, torch.device('cpu'))
    assert {path.name: (study.digest(path), path.stat().st_mtime_ns) for path in directory.iterdir()} == before


def test_failed_fit_attempt_and_original_history_are_preserved(tmp_path, monkeypatch):
    from utilsTorch import utilsTorch as utils
    args, data = arguments(tmp_path, tiny=True), synthetic_data()
    original_loss = utils.segmentation_loss
    def nonfinite_training_loss(*args, **kwargs):
        if torch.is_grad_enabled():
            return torch.tensor(float('nan'))
        return original_loss(*args, **kwargs)
    monkeypatch.setattr(utils, 'segmentation_loss', nonfinite_training_loss)
    directory = tmp_path / 'failed'
    with pytest.raises(ValueError, match='Nonfinite training loss'):
        study.fit_one('compact_affine', 42, .003, args, data, directory, torch.device('cpu'))
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert set(before) == {'history.json', 'failure.json'}
    assert json.loads(before['failure.json'])['completed_epochs'] == 0
    assert len(json.loads(before['history.json'])) == 1
    with pytest.raises(ValueError, match='Incomplete attempt preserved'):
        study.fit_one('compact_affine', 42, .003, args, data, directory, torch.device('cpu'))
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def metric_fixture(n_samples, quality=0):
    sensors = (71, 134) if n_samples == 205 else (30, n_samples-30)
    result, sums = dict(loss=.5, auc=.7), dict(tp=0, fp=0, fn=0, tn=0)
    for sensor, count in zip(('optical', 'sar'), sensors):
        counts = dict(tp=count*(512+quality), fp=count*128, fn=count*256)
        counts['tn'] = count*128*128 - sum(counts.values())
        for key, value in counts.items():
            result[sensor + '_subset_' + key] = value
            sums[key] += value
        result[sensor + '_subset_n_samples'] = count
        result[sensor + '_subset_iou'] = counts['tp'] / (counts['tp']+counts['fp']+counts['fn'])
        result[sensor + '_subset_dice'] = 2*counts['tp'] / (2*counts['tp']+counts['fp']+counts['fn'])
        result[sensor + '_subset_accuracy'] = (counts['tp']+counts['tn']) / sum(counts.values())
        result[sensor + '_subset_auc'] = .7
    tp, fp, fn, tn = (sums[key] for key in ('tp', 'fp', 'fn', 'tn'))
    result.update(iou=tp/(tp+fp+fn), dice=2*tp/(2*tp+fp+fn), accuracy=(tp+tn)/sum(sums.values()))
    return result


def cohort_row(kind, seed, rate):
    # Auxiliary validation is deliberately better; it must never replace.003.
    quality = 1000 if rate == .001 else 1
    totals = {'quantum': (344480, 8), 'frozen_quantum': (344480, 8),
              'zero_bottleneck': (344480, 8), 'compact_affine': (346040, 1568), 'mlp': (349688, 5216)}
    state = 'circuit' if kind in ('quantum', 'frozen_quantum', 'zero_bottleneck') else kind
    normalization = dict(fit_only=True, seed=seed, stats_sha256='fixture-' + str(seed))
    return dict(id='hq_' + kind, kind=kind, seed=seed, learning_rate=rate, best_epoch=0,
        initial_state_sha256=state + str(seed), initial_shared_state_sha256='shared' + str(seed),
        selected_state_sha256='selected-' + kind + str(seed), fit_names=['fit-' + str(i) for i in range(128)],
        val_names=['val-' + str(i) for i in range(230)], normalization=normalization,
        settings=dict(_resolved_norm_provenance=normalization), val=metric_fixture(230, quality),
        history=[dict(epoch=0, train_schedule=None), dict(epoch=1, train_schedule='paired-' + str(seed))],
        total_params=totals[kind][0], core_params=totals[kind][1])


def complete_rows():
    return [cohort_row(kind, seed, rate) for kind, seed, rate in sorted(study.expected_grid())]


@pytest.mark.parametrize('corruption,match', (
    ('missing', 'fit grid'), ('duplicate', 'fit grid'), ('common_state', 'initial_shared_state'),
    ('circuit_state', 'different states'), ('fit_names', 'fit_names'),
    ('normalization', 'normalization'), ('schedule', 'augmentation'),
))
def test_paired_grid_rejects_missing_duplicate_or_unpaired_inputs(corruption, match):
    rows = complete_rows()
    study.paired_audit(rows)
    if corruption == 'missing':
        rows.pop()
    elif corruption == 'duplicate':
        rows[-1] = copy.deepcopy(rows[0])
    else:
        row = next(row for row in rows if row['kind'] == 'frozen_quantum' and row['seed'] == 42 and row['learning_rate'] == .003)
        if corruption == 'common_state':
            row['initial_shared_state_sha256'] = 'changed'
        elif corruption == 'circuit_state':
            row['initial_state_sha256'] = 'changed'
        elif corruption == 'fit_names':
            row['fit_names'][0] = 'different'
        elif corruption == 'normalization':
            row['normalization']['stats_sha256'] = 'different'
        else:
            row['history'][1]['train_schedule'] = 'different'
    with pytest.raises(ValueError, match=match):
        study.paired_audit(rows)


@pytest.mark.parametrize('origin', ('fresh', 'reused'))
@pytest.mark.parametrize('escape', ('../outside', '/absolute/outside', 'symlink'))
def test_entry_paths_cannot_escape_their_actual_evidence_root(tmp_path, origin, escape):
    work, legacy = tmp_path / 'run', tmp_path / 'legacy'
    work.mkdir()
    legacy.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    root = legacy if origin == 'reused' else work
    if escape == 'symlink':
        (root / 'alias').symlink_to(outside, target_is_directory=True)
        escape = 'alias'
    with pytest.raises(ValueError, match='escapes'):
        study.entry_directory(work, legacy, dict(origin=origin, directory=escape))


def test_entry_path_rejects_unknown_origin(tmp_path):
    with pytest.raises(ValueError, match='Unknown fit evidence origin'):
        study.entry_directory(tmp_path, tmp_path, dict(origin='other', directory='fit'))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Mock only expensive fit/restore/inference for orchestration coverage."""
    from tools import training_common as tc
    args = arguments(tmp_path)
    source = tmp_path / 'source'
    (source / 'docs').mkdir(parents=True)
    (source / study.PROTOCOL).write_text('Synthetic orchestration protocol')
    monkeypatch.setattr(study, 'ROOT', source)
    monkeypatch.setattr(study, 'source_hashes', lambda: {'new_runner.py': 'new-source'})
    runtime = dict(python='fixture', device='cpu', threads=2)
    monkeypatch.setattr(study.hq, 'runtime', lambda device: runtime)
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *args, **kwargs: 'new-producer\n')
    inputs = dict(manifest_sha256='same-inputs')
    monkeypatch.setattr(study.pilot, 'verify_inputs', lambda root: inputs)
    class GlobalData:
        names = ['global-' + str(i) for i in range(1533)]
        modalities = ['sar']*1533
    def training_data(args, seed):
        return dict(dataset=GlobalData(), all_fit=list(range(1303)), validation=list(range(1303, 1533)),
                    fit=list(range(128)), normalization=dict(fit_only=True, seed=seed))
    monkeypatch.setattr(study.pilot, 'load_training_data', training_data)
    args.legacy_root.mkdir()
    old_plan = dict(source_revision='historical-producer', source_hashes={'old_runner.py': 'old-source'},
                    input_identity=inputs, runtime=copy.deepcopy(runtime))
    study.pilot.write_json(args.legacy_root / 'plan.json', old_plan)
    events, old_fits = [], []
    def write_fit(directory, row):
        directory.mkdir(parents=True, exist_ok=True)
        study.pilot.write_json(directory / 'metrics.json', row)
        study.pilot.write_json(directory / 'history.json', row['history'])
        (directory / 'checkpoint.pt').write_bytes(b'not-a-real-checkpoint-or-inference-fixture')
    for kind in ('quantum', 'mlp'):
        for seed, rate in ((42, .001), (42, .003), (77, .003), (123, .003)):
            row = cohort_row(kind, seed, rate)
            relative = f'{kind}/{rate}/seed{seed}'
            write_fit(args.legacy_root / relative, row)
            old_fits.append(dict(directory=relative, record=row))
    historical = dict(plan=old_plan, fits=old_fits, raw_files={'plan.json': study.digest(args.legacy_root/'plan.json')},
                      summary=dict(selection={'hq_quantum': .003, 'hq_mlp': .003}))
    def legacy_audit(work_dir, data_root):
        assert work_dir == args.legacy_root and data_root == args.data_root
        events.append(('legacy_audit',))
        return historical
    monkeypatch.setattr(legacy_collector, 'audit_campaign', legacy_audit)
    profile = dict(status='passed', source_hashes=study.source_hashes(), protocol_sha256=study.digest(source / study.PROTOCOL),
        runtime=runtime, cases=[dict(kind=kind, batch_size=8, strict_restore=True, frozen_unchanged=True) for kind in study.NEW_KINDS])
    study.pilot.write_json(args.profile, profile)
    def fit(kind, seed, rate, args, data, directory, device):
        events.append(('fit', kind, seed, rate))
        row = cohort_row(kind, seed, rate)
        write_fit(directory, row)
        return row
    monkeypatch.setattr(study, 'fit_one', fit)
    def restore(path, row, device='cpu'):
        assert len([event for event in events if event[0] == 'fit']) % 12 == 0
        events.append(('restore', *study.identity(row)))
        model = SimpleNamespace(kind=row['kind'], seed=row['seed'])
        return model, dict(checkpoint_sha256=study.digest(path), strict_restore=True,
            finite_weights_and_bn_buffers=True, selected_state_sha256=row['selected_state_sha256'])
    monkeypatch.setattr(study, 'restore_checkpoint', restore)
    class TestData:
        def __init__(self, cfg):
            self.names = ['test-' + str(index) for index in range(205)]
            self.norm_provenance = cfg['_resolved_norm_provenance']
        def __len__(self):
            return 205
    monkeypatch.setattr(tc, 'resolve_data_dirs', lambda *args: (None, None, None, 'test-images', 'test-masks'))
    monkeypatch.setattr(tc, 'load_dataset', lambda images, masks, cfg: TestData(cfg))
    def evaluate(model, loader, device):
        assert len(loader.dataset) == 205
        boundary = json.loads((args.work_dir/'fit_boundary.json').read_text())
        assert boundary['completed'] and boundary['fits'] == 20 and boundary['fresh'] == 12 and boundary['reused'] == 8
        assert len(json.loads((args.work_dir/'all_fits.json').read_text())) == 20
        assert len(json.loads((args.work_dir/'checkpoint_audit.json').read_text())) == 20
        assert boundary['all_fits_sha256'] == study.digest(args.work_dir/'all_fits.json')
        assert boundary['checkpoint_audit_sha256'] == study.digest(args.work_dir/'checkpoint_audit.json')
        events.append(('test', model.kind, model.seed))
        return metric_fixture(205, 7)
    monkeypatch.setattr(study, 'evaluate', evaluate)
    return SimpleNamespace(args=args, events=events, historical=historical, inputs=inputs, profile=profile)


def test_all20_fits_and_audits_precede15_new_fixed003_tests_even_when_auxiliary_is_better(campaign):
    result = study.run(campaign.args, torch.device('cpu'))
    events = campaign.events
    first_test = next(i for i, event in enumerate(events) if event[0] == 'test')
    assert len([event for event in events[:first_test] if event[0] == 'fit']) == 12
    # Twenty audit restores precede the additional restore for the first test.
    assert len([event for event in events[:first_test] if event[0] == 'restore']) == 21
    tests = [event[1:] for event in events if event[0] == 'test']
    assert len(tests) == len(set(tests)) == 15
    assert set(tests) == {(kind, seed) for kind in study.KINDS for seed in study.SEEDS}
    assert len([pair for pair in tests if pair[0] in ('quantum', 'mlp')]) == 6
    assert all(row['learning_rate'] == .003 and row['test'] == metric_fixture(205, 7) for row in result['records'])
    assert result['analysis']['primary_rate'] == .003
    entries = json.loads((campaign.args.work_dir/'all_fits.json').read_text())
    assert all('test' not in entry['record'] for entry in entries)
    assert sum(entry['origin'] == 'reused' for entry in entries) == 8
    assert sum(entry['origin'] == 'fresh' for entry in entries) == 12
    for kind in study.KINDS:
        auxiliary = next(entry['record'] for entry in entries if study.identity(entry['record']) == (kind, 42, .001))
        primary = next(row for row in result['records'] if study.identity(row) == (kind, 42, .003))
        assert auxiliary['val']['iou'] > primary['val']['iou']
    assert result['plan']['legacy_source_hashes'] == {'old_runner.py': 'old-source'}
    assert result['plan']['legacy_plan_sha256'] == study.digest(campaign.args.legacy_root/'plan.json')
    assert result['plan']['source_revision'] == 'new-producer'


def test_completed_campaign_resume_reuses_bound_replays_without_inference_or_overwrite(campaign, monkeypatch):
    original = study.run(campaign.args, torch.device('cpu'))
    paths = list((campaign.args.work_dir/'replays').rglob('*.json'))
    assert len(paths) == 15
    before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    tests_before = len([event for event in campaign.events if event[0] == 'test'])
    def forbidden(*args):
        raise AssertionError('Completed saved test replays must not evaluate again')
    monkeypatch.setattr(study, 'evaluate', forbidden)
    monkeypatch.setattr(study.subprocess, 'check_output', lambda *args, **kwargs: 'reporting-only-child\n')
    campaign.args.resume = True
    resumed = study.run(campaign.args, torch.device('cpu'))
    assert resumed == original
    assert resumed['plan']['source_revision'] == 'new-producer'
    assert len([event for event in campaign.events if event[0] == 'test']) == tests_before == 15
    assert {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before


def test_tampered_completed_replay_binding_is_rejected_and_preserved(campaign, monkeypatch):
    study.run(campaign.args, torch.device('cpu'))
    path = campaign.args.work_dir/'replays/hq_quantum/seed42.json'
    replay = json.loads(path.read_text())
    assert set(replay) == {'binding', 'record'}
    replay['binding'] = {'tampered': True}
    study.pilot.write_json(path, replay)
    before = path.read_bytes(), path.stat().st_mtime_ns
    def forbidden(*args):
        raise AssertionError('A malformed replay must not trigger inference')
    monkeypatch.setattr(study, 'evaluate', forbidden)
    campaign.args.resume = True
    with pytest.raises(ValueError, match='[Rr]eplay|[Bb]inding'):
        study.run(campaign.args, torch.device('cpu'))
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_failed_inference_marker_blocks_silent_retry_and_preserves_failure(campaign, monkeypatch):
    def fail(*args):
        raise RuntimeError('Synthetic inference failure')
    monkeypatch.setattr(study, 'evaluate', fail)
    with pytest.raises(RuntimeError, match='Synthetic inference failure'):
        study.run(campaign.args, torch.device('cpu'))
    failures = list((campaign.args.work_dir/'replay_failures').rglob('*.json'))
    assert len(failures) == 1
    before = {str(path): path.read_bytes() for path in failures}
    campaign.args.resume = True
    def forbidden(*args):
        raise AssertionError('An archived failure must require an explicit new campaign')
    monkeypatch.setattr(study, 'evaluate', forbidden)
    with pytest.raises(ValueError, match='[Ff]ail|[Rr]eplay'):
        study.run(campaign.args, torch.device('cpu'))
    assert {str(path): path.read_bytes() for path in failures} == before


@pytest.mark.parametrize('change,match', (('source', 'profile'), ('runtime', '[Rr]untime'),
                                       ('inputs', 'input bytes differ'), ('settings', 'settings')))
def test_campaign_resume_refuses_changed_provenance_without_touching_completed_outputs(campaign, monkeypatch, change, match):
    study.run(campaign.args, torch.device('cpu'))
    before = {str(path.relative_to(campaign.args.work_dir)): study.digest(path)
              for path in campaign.args.work_dir.rglob('*') if path.is_file()}
    campaign.args.resume = True
    if change == 'source':
        monkeypatch.setattr(study, 'source_hashes', lambda: {'new_runner.py': 'different'})
    elif change == 'runtime':
        monkeypatch.setattr(study.hq, 'runtime', lambda device: dict(python='changed'))
    elif change == 'inputs':
        monkeypatch.setattr(study.pilot, 'verify_inputs', lambda root: {'manifest_sha256': 'different'})
    else:
        campaign.args.epochs = 62
    with pytest.raises(ValueError, match=match):
        study.run(campaign.args, torch.device('cpu'))
    after = {str(path.relative_to(campaign.args.work_dir)): study.digest(path)
             for path in campaign.args.work_dir.rglob('*') if path.is_file()}
    assert after == before


def test_legacy_audit_failure_blocks_all_new_work(campaign, monkeypatch):
    def reject(*args):
        raise ValueError('Historical checkpoint provenance differs')
    monkeypatch.setattr(legacy_collector, 'audit_campaign', reject)
    with pytest.raises(ValueError, match='Historical checkpoint provenance'):
        study.run(campaign.args, torch.device('cpu'))
    assert not campaign.args.work_dir.exists()
    assert campaign.events == []


def test_original_baseline_runtime_must_match_even_with_current_valid_profile(campaign):
    campaign.historical['plan']['runtime']['python'] = 'a-different-producer'
    with pytest.raises(ValueError, match='[Rr]untime'):
        study.run(campaign.args, torch.device('cpu'))
    assert not campaign.args.work_dir.exists()
    assert not any(event[0] == 'fit' for event in campaign.events)


def test_producer_source_inventory_binds_invoked_historical_collector():
    assert 'scripts/collect_hq_dfc24_direction.py' in study.SOURCE_FILES
    assert len(study.SOURCE_FILES) == len(set(study.SOURCE_FILES))


def test_historical_source_verifier_rejects_changed_or_missing_bound_inputs():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root/'benchmarks/hq_dfc24_direction_20261028/plan.json').read_text())
    legacy_collector.verify_source(plan)
    changed = copy.deepcopy(plan)
    changed['source_hashes']['utilsTorch/hq_direction.py'] = '0'*64
    with pytest.raises(ValueError, match='Producer source differs'):
        legacy_collector.verify_source(changed)
    changed = copy.deepcopy(plan)
    changed['source_hashes'].pop('flood_data.py')
    with pytest.raises(ValueError, match='source inventory differs'):
        legacy_collector.verify_source(changed)


@pytest.mark.parametrize('kind', ('quantum', 'mlp'))
def test_actual_legacy_checkpoint_restore_preserves_original_config_and_adds_control_metadata(tmp_path, kind):
    args = arguments(tmp_path, tiny=True)
    model = study.hq.build_model(kind, 42, args)
    initial = study.state_hash(model.state_dict())
    row = dict(kind=kind, seed=42, best_epoch=0, settings={'historical': True},
        model_config=study.hq.model_configuration(model), initial_state_sha256=initial,
        initial_shared_state_sha256=study.shared_hash(model), selected_state_sha256=initial,
        total_params=sum(value.numel() for value in model.parameters()),
        core_params=sum(value.numel() for value in model.bottleneck.core_parameters()))
    path = tmp_path/'legacy.pt'
    torch.save(dict(format='hq-direction-v1', kind=kind, seed=42, best_epoch=0,
        settings=row['settings'], model_config=row['model_config'], best_state_dict=model.state_dict()), path)
    restored, audit = study.restore_checkpoint(path, row)
    assert restored.training_control == kind
    assert study.counts(restored)['total'] == row['total_params']
    assert study.state_hash(restored.state_dict()) == initial
    assert audit['checkpoint_sha256'] == study.digest(path)
    assert study.hq.model_configuration(restored) == row['model_config']
