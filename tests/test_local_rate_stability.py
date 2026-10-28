"""The fixed-rate study must retain complete, paired and auditable cells."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_local_rate_stability as study


def fixture_records():
    specs = [dict(id=name, model_type='quantum', config={}) for name in study.QUANTUM]
    records = []
    for spec in specs:
        for seed, gain in zip(study.SEEDS, [0.01, -0.02, 0.04]):
            for rate in study.RATES:
                iou = 0.5 + (gain if spec['id'] == study.QUANTUM[0] else 0)
                scores = dict(iou=iou, optical_subset_iou=iou, sar_subset_iou=iou)
                records.append(dict(id=spec['id'], seed=seed, learning_rate=rate,
                    fit_names=['fit-' + str(seed)], val_names=['val-' + str(seed)],
                    test_names=['test'], normalization={'seed': seed},
                    initial_parameter_sha256='initial-' + str(seed), val=dict(scores), test=dict(scores),
                    best_epoch=60, diagnostics={'gradient_norms': {'angles': {'max': 3}}},
                    history=[dict(epoch=0, train_loss=None, val=dict(scores, iou=iou - 0.1)),
                             dict(epoch=60, train_loss=0.3, val=scores)]))
    return specs, records


def test_fixed_rate_contrasts_compute_paired_sample_statistics():
    specs, records = fixture_records()
    result = study.summarize(records, specs)
    assert len(result['paired_iou']) == 2
    for contrast in result['paired_iou']:
        assert contrast['mean'] == pytest.approx(0.01)
        assert contrast['std'] == pytest.approx(0.03)
        assert contrast['wins'] == 2
        assert contrast['n'] == 3
    # A consulted test score cannot influence validation analysis.
    records[0]['test']['iou'] = 0.99
    assert study.summarize(records, specs)['paired_iou'] == result['paired_iou']


@pytest.mark.parametrize('change, message', [
    ('missing', 'Incomplete'), ('duplicate', 'Duplicate'),
    ('extra', 'unexpected'), ('cohort', 'Unpaired'), ('initial', 'initialization'),
    ('test_names', 'Unpaired'),
])
def test_factorial_rejects_incomplete_or_unpaired_comparisons(change, message):
    specs, records = fixture_records()
    if change == 'missing':
        records.pop()
    elif change == 'duplicate':
        records.append(copy.deepcopy(records[0]))
    elif change == 'extra':
        records[0]['learning_rate'] = 0.009
    elif change == 'cohort':
        records[0]['fit_names'] = ['different']
    elif change == 'initial':
        records[0]['initial_parameter_sha256'] = 'different'
    elif change == 'test_names':
        records[0]['test_names'] = ['different']
    with pytest.raises(ValueError, match=message):
        study.audit_factorial(records, specs, test=True)


def test_resume_keeps_original_producer_and_rejects_execution_drift(tmp_path):
    path = tmp_path / 'run/plan.json'
    candidate = dict(source_revision='original', source_hashes={'runner': 'same'},
                     python='runtime', historical_fits=[{'checkpoint_sha256': 'original'}])
    study.frozen_plan(path, candidate, False)
    child_revision = dict(candidate, source_revision='child-with-identical-executable-bytes')
    assert study.frozen_plan(path, child_revision, True) == candidate
    assert json.loads(path.read_text()) == candidate
    for key, value in [('source_hashes', {'runner': 'changed'}), ('python', 'changed'),
                       ('historical_fits', [{'checkpoint_sha256': 'changed'}])]:
        with pytest.raises(ValueError, match=key):
            study.frozen_plan(path, dict(candidate, **{key: value}), True)
    with pytest.raises(ValueError, match='new/empty'):
        study.frozen_plan(path, candidate, False)


def record_fixture():
    spec = dict(id='quantum_global', model_type='quantum', config={'q_spatial_mode': 'global'})
    args = SimpleNamespace(data_root=Path('/data'), image_size=64, batch_size=16, epochs=2, eval_every=2, fit_cap=128)
    data = dict(fit=[0], validation=[1], all_fit=[0, 2],
                dataset=SimpleNamespace(names=['fit', 'val', 'unused']), normalization={'fit_only': True})
    row = dict(id=spec['id'], model_type='quantum', seed=42, learning_rate=0.001,
        settings=study.expected_settings(spec, 42, 0.001, args, data), fit_indices=[0], val_indices=[1],
        fit_names=['fit'], val_names=['val'], normalization=data['normalization'], n_global_fit=2,
        best_epoch=2, val={'iou': 0.6},
        history=[dict(epoch=0, train_loss=None, val={'iou': 0.3}),
                 dict(epoch=1, train_loss=0.4), dict(epoch=2, train_loss=0.3, val={'iou': 0.6})])
    return spec, args, data, row


def test_record_reuse_accepts_legacy_default_and_storage_root_only():
    spec, args, data, row = record_fixture()
    row['settings'].pop('q_training_control')
    row['settings']['data_root'] = '/historical/storage'
    study.audit_record(row, spec, 42, 0.001, args, data)


@pytest.mark.parametrize('change', ['extra', 'missing'])
def test_reuse_requires_the_exact_validation_evaluation_schedule(change):
    spec, args, data, row = record_fixture()
    if change == 'extra':
        row['history'][1]['val'] = {'iou': 0.4}
    else:
        del row['history'][0]['val']
    with pytest.raises(ValueError, match='evaluation schedule'):
        study.audit_record(row, spec, 42, 0.001, args, data)


@pytest.mark.parametrize('change, message', [
    ('control', 'configuration'), ('cohort', 'cohort'), ('normalization', 'cohort'),
    ('budget', 'configuration'), ('history', 'complete'), ('selection', 'selection'),
])
def test_record_reuse_rejects_changed_scientific_inputs(change, message):
    spec, args, data, row = record_fixture()
    if change == 'control':
        row['settings']['q_training_control'] = 'no_entanglement'
    elif change == 'cohort':
        row['fit_names'] = ['different']
    elif change == 'normalization':
        row['normalization'] = {'fit_only': False}
    elif change == 'budget':
        row['settings']['num_epochs'] = 120
    elif change == 'history':
        row['history'].pop()
    elif change == 'selection':
        row['best_epoch'] = 0
    with pytest.raises(ValueError, match=message):
        study.audit_record(row, spec, 42, 0.001, args, data)


@pytest.mark.parametrize('field', ['loss', 'validation', 'gradient'])
def test_record_audit_rejects_nonfinite_recorded_evidence(field):
    spec, args, data, row = record_fixture()
    if field == 'loss':
        row['history'][1]['train_loss'] = float('nan')
    elif field == 'validation':
        row['val']['iou'] = float('inf')
    else:
        row['diagnostics'] = {'gradient_norms': {'angles': {'max': float('inf')}}}
    with pytest.raises(ValueError, match='Nonfinite'):
        study.audit_record(row, spec, 42, 0.001, args, data)


def test_discovery_binds_historical_producer_and_forbids_test_record_reuse(tmp_path):
    spec, _, _, row = record_fixture()
    plan = dict(source_revision='historical', source_hashes={'producer': 'historical-bytes'},
        input_identity={'raw': 'same'}, **{key: None for key in study.RUNTIME_KEYS})
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    directory = tmp_path / 'fits/cell'
    directory.mkdir(parents=True)
    (directory / 'metrics.json').write_text(json.dumps(row))
    (directory / 'checkpoint.pt').write_bytes(b'bound-checkpoint')
    result = study.discover_reuse([tmp_path], [spec])
    item = next(iter(result.values()))
    assert item['producer_source_revision'] == 'historical'
    assert item['checkpoint_sha256'] == study.digest(directory / 'checkpoint.pt')
    assert item['producer_plan_sha256'] == study.digest(tmp_path / 'plan.json')
    row['test'] = {'iou': 1}
    (directory / 'metrics.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='validation-only'):
        study.discover_reuse([tmp_path], [spec])


@pytest.mark.parametrize('change, message', [
    (None, None), ('metadata', 'metadata'), ('weights', 'nonfinite'),
    ('initial', 'initialization'), ('replay', 'replay'),
])
def test_checkpoint_audit_checks_actual_finite_state_init_and_validation(tmp_path, monkeypatch, change, message):
    import torch
    from tools import checkpoint_evaluation as ce
    from tools import training_common as tc
    from utilsTorch import utilsTorch as ut
    spec, args, data, row = record_fixture()
    template = torch.nn.Linear(1, 1)
    row['initial_parameter_sha256'] = study.initial_hash(template)
    row['total_params'] = 2
    counts = {sensor + '_subset_' + count: 3 for sensor in ('optical', 'sar')
              for count in ('tp', 'fp', 'fn', 'tn', 'n_samples')}
    row['val'] = dict(counts, iou=0.6)
    saved = dict(best_epoch=row['best_epoch'], settings=row['settings'], model_type='quantum',
                 best_state_dict=copy.deepcopy(template.state_dict()))
    replay = dict(row['val'])
    if change == 'metadata':
        saved['best_epoch'] = 0
    elif change == 'weights':
        saved['best_state_dict']['weight'].fill_(float('nan'))
    elif change == 'initial':
        row['initial_parameter_sha256'] = 'wrong'
    elif change == 'replay':
        replay['optical_subset_tp'] += 1
    torch.save(saved, tmp_path / 'checkpoint.pt')
    monkeypatch.setattr(tc, 'seed_everything', lambda seed: None)
    monkeypatch.setattr(tc, 'build_model', lambda cls, cfg: copy.deepcopy(template))
    monkeypatch.setattr(tc, 'make_loader', lambda *args: None)
    monkeypatch.setattr(ce, 'load_checkpoint_model', lambda *args, **kwargs: (copy.deepcopy(template), {}, 'quantum'))
    monkeypatch.setattr(ut, 'evaluate_model', lambda *args: replay)
    if change:
        with pytest.raises(ValueError, match=message):
            study.audit_checkpoint(tmp_path, row, spec, args, dict(data, val=None), 'cpu', replay_validation=True)
    else:
        result = study.audit_checkpoint(tmp_path, row, spec, args, dict(data, val=None), 'cpu', replay_validation=True)
        assert result['validation_counts_equal'] is True


def test_long_training_analysis_reports_prefix60_and_gradient_extremes():
    specs, records = fixture_records()
    for row in records:
        row['history'].append(dict(epoch=120, train_loss=0.2, val=dict(row['val'], iou=row['val']['iou'] + 0.02)))
        row['best_epoch'] = 120
        row['val'] = row['history'][-1]['val']
    result = study.summarize(records, specs)
    assert all(row['later_validation_gain'] == pytest.approx(0.02) for row in result['trajectories'])
    assert all(row['best_at_or_before60_epoch'] == 60 for row in result['trajectories'])
    assert all(row['gradient_norm_extremes']['angles']['max'] == 3 for row in result['trajectories'])


@pytest.mark.parametrize('validation_only', [False, True])
def test_cli_finishes_all18_validation_fits_before_test_access(tmp_path, monkeypatch, validation_only):
    import torch
    from tools import checkpoint_evaluation as ce
    from tools import training_common as tc
    from utilsTorch import utilsTorch as ut
    specs = [dict(id=name, model_type='ablation' if name == 'mlp_patch4x4' else 'quantum', config={})
             for name in (*study.QUANTUM, 'mlp_patch4x4')]
    fitted, tested = [], []
    monkeypatch.setattr(study, 'specs_for', lambda phase: specs)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda n: None)
    monkeypatch.setattr(study.pilot, 'verify_inputs', lambda root: {'raw': 'same'})

    def training_data(args, seed):
        return dict(fit=[0], validation=[1], all_fit=[0, 2],
                    dataset=SimpleNamespace(names=['fit', 'val', 'unused']),
                    normalization={'seed': seed})

    def fit_one(spec, seed, rate, args, data, directory, device):
        fitted.append((spec['id'], seed, rate))
        scores = lambda epoch: dict(iou=0.5 + epoch / 1000,
            optical_subset_iou=0.4 + epoch / 1000, sar_subset_iou=0.6 + epoch / 1000)
        history = [dict(epoch=epoch, train_loss=0.3 if epoch else None,
                        **({'val': scores(epoch)} if epoch % 2 == 0 else {}))
                   for epoch in range(121)]
        row = dict(id=spec['id'], model_type=spec['model_type'], seed=seed, learning_rate=rate,
            settings=study.expected_settings(spec, seed, rate, args, data), fit_indices=[0], val_indices=[1],
            fit_names=['fit'], val_names=['val'], normalization=data['normalization'], n_global_fit=2,
            initial_parameter_sha256='initial-' + str(seed), best_epoch=120, val=scores(120),
            diagnostics={'gradient_norms': {}}, history=history)
        directory.mkdir(parents=True)
        (directory / 'checkpoint.pt').write_bytes(b'fake-model')
        (directory / 'metrics.json').write_text(json.dumps(row))
        return row

    def load_model(path, **kwargs):
        assert len(fitted) == 18
        assert len(set(fitted)) == 18
        assert (tmp_path / 'runs/validation_fits.json').is_file()
        tested.append(path)
        row = json.loads((path.parent / 'metrics.json').read_text())
        return object(), {'_resolved_norm_provenance': row['normalization']}, row['model_type']

    monkeypatch.setattr(study.pilot, 'load_training_data', training_data)
    monkeypatch.setattr(study.pilot, 'fit_one', fit_one)
    monkeypatch.setattr(study, 'audit_checkpoint', lambda *args, **kwargs: {'strict_restore': True})
    monkeypatch.setattr(ce, 'load_checkpoint_model', load_model)
    monkeypatch.setattr(tc, 'resolve_data_dirs', lambda *args: (None, None, None, 'images', 'masks'))
    monkeypatch.setattr(tc, 'load_dataset', lambda images, masks, cfg:
                        SimpleNamespace(norm_provenance=cfg['_resolved_norm_provenance'], names=['test']))
    monkeypatch.setattr(tc, 'make_loader', lambda *args: None)
    monkeypatch.setattr(tc, 'knockout_eval', lambda *args, **kwargs: {'zero': {'iou': 0.5}})
    monkeypatch.setattr(ut, 'evaluate_model', lambda *args: dict(iou=0.6, optical_subset_iou=0.5, sar_subset_iou=0.7))
    argv = ['--phase', 'long120', '--device', 'cpu', '--data-root', str(tmp_path / 'data'),
            '--work-dir', str(tmp_path / 'runs')]
    if validation_only:
        argv.append('--validation-only')
    result = study.main(argv)
    assert len(fitted) == 18
    assert len(tested) == (0 if validation_only else 18)
    assert ('test_analysis' in result) is not validation_only
    assert all(('test' in row) is not validation_only for row in result['records'])
    # Validation-only retained fits are never mutated by descriptive test replay.
    assert all('test' not in row for row in json.loads((tmp_path / 'runs/validation_fits.json').read_text()))
