"""Retained factorial outputs must agree with raw fits, audits and producers."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import collect_local_rate_stability as collect
from scripts import run_local_rate_stability as study


def write(path, value):
    study.pilot.write_json(path, value)


def scores(epoch, seed, model, *, test=False):
    result, total = {}, dict(tp=0, fp=0, fn=0, tn=0)
    offset = study.SEEDS.index(seed) + (2 if model == study.QUANTUM[0] else 0)
    for sensor, n, start in [('optical', 71 if test else 50, 80), ('sar', 134 if test else 180, 120)]:
        counts = dict(tp=start + epoch // 4 + offset, fp=70, fn=100, tn=500, n_samples=n)
        result.update({sensor + '_subset_' + key: value for key, value in counts.items()})
        result[sensor + '_subset_iou'] = counts['tp'] / (counts['tp'] + counts['fp'] + counts['fn'])
        for key in total:
            total[key] += counts[key]
    tp, fp, fn, tn = (total[key] for key in ('tp', 'fp', 'fn', 'tn'))
    result.update(iou=tp / (tp + fp + fn), dice=2 * tp / (2 * tp + fp + fn),
                  accuracy=(tp + tn) / (tp + fp + fn + tn), auc=0.8, loss=1 - epoch / 1000)
    return result


def build_runs(tmp_path, monkeypatch):
    """Synthetic metadata/checkpoint bytes exercise auditing, not model accuracy."""
    input_identity = {'raw_inventory_sha256': 'synthetic-fixture'}
    monkeypatch.setattr(study.pilot, 'verify_inputs', lambda root: input_identity)
    runtime = dict(python='fixture-runtime', packages={'torch': 'fixture'}, device='cpu',
                   gpu=None, gpu_allocator_fraction=None, threads=2)
    producers = {}
    for name in ('original', 'controls'):
        root = tmp_path / ('historical-' + name)
        plan = dict(source_revision='fixture-' + name, source_hashes={'historical.py': 'old-bytes'},
                    input_identity=input_identity, **runtime)
        write(root / 'plan.json', plan)
        producers[name] = root, plan
    paths = []
    for phase, epochs in [('matched60', 60), ('long120', 120)]:
        run = tmp_path / phase
        run.mkdir()
        specs = study.specs_for(phase)
        records, validations, audits, historical, reuses = [], [], [], [], []
        for spec in specs:
            for seed in study.SEEDS:
                for rate in study.RATES:
                    key = spec['id'], seed, rate
                    norm = dict(fit_only=True, seed=seed, stats_sha256='stats-' + str(seed))
                    fit_names = [f'fit-{seed}-{index}' for index in range(128)]
                    val_names = [f'val-{seed}-{index}' for index in range(230)]
                    names = fit_names + val_names + ['unused'] * (1533 - 358)
                    data = dict(fit=list(range(128)), validation=list(range(128, 358)),
                                all_fit=range(1303), dataset=SimpleNamespace(names=names), normalization=norm)
                    args = SimpleNamespace(data_root=tmp_path / 'data', epochs=epochs, **collect.FIXED_SETTINGS)
                    settings = study.expected_settings(spec, seed, rate, args, data)
                    if phase == 'matched60' and key in collect.HISTORICAL_CELLS and spec['id'] == study.QUANTUM[0]:
                        settings.pop('q_training_control')  # Historical implicit default.
                    history = [dict(epoch=epoch, train_loss=(1 - epoch / 1000) if epoch else None,
                        **({'val': scores(epoch, seed, spec['id'])} if epoch % 2 == 0 else {}))
                               for epoch in range(epochs + 1)]
                    base = dict(id=spec['id'], model_type=spec['model_type'], seed=seed, learning_rate=rate,
                        settings=settings, fit_indices=list(range(128)), val_indices=list(range(128, 358)),
                        fit_names=fit_names, val_names=val_names, n_global_fit=1303, normalization=norm,
                        initial_parameter_sha256=('quantum' if spec['id'] in study.QUANTUM else 'mlp') + str(seed),
                        total_params=500522 if spec['id'] == 'mlp_patch4x4' else 500394,
                        best_epoch=epochs, val=history[-1]['val'], history=history,
                        diagnostics={'gradient_norms': {'backbone': {'max': 2, 'min': 0.01}}}, elapsed_s=8.0)
                    is_historical = phase == 'matched60' and key in collect.HISTORICAL_CELLS
                    if is_historical:
                        name = 'original' if spec['id'] == study.QUANTUM[0] else 'controls'
                        producer_root, producer = producers[name]
                        directory = producer_root / spec['id'] / str(rate) / ('seed' + str(seed))
                        revision = producer['source_revision']
                    else:
                        directory = run / 'fits' / spec['id'] / str(rate) / ('seed' + str(seed))
                        revision = 'fixture-new-producer'
                    write(directory / 'metrics.json', base)
                    write(directory / 'history.json', history)
                    checkpoint = directory / 'checkpoint.pt'
                    checkpoint.write_bytes(('synthetic-checkpoint-' + str(key) + phase).encode())
                    audit = dict(id=spec['id'], seed=seed, learning_rate=rate,
                        origin='historical' if is_historical else 'new', producer_source_revision=revision,
                        metrics_sha256=study.digest(directory / 'metrics.json'), checkpoint_sha256=study.digest(checkpoint),
                        **{name: True for name in collect.AUDIT_FLAGS})
                    audits.append(audit)
                    if is_historical:
                        reuses.append(audit)
                        historical.append(dict(id=spec['id'], seed=seed, learning_rate=rate,
                            directory=str(directory), metrics_sha256=audit['metrics_sha256'],
                            checkpoint_sha256=audit['checkpoint_sha256'], producer_source_revision=revision,
                            producer_source_hashes=producer['source_hashes'], producer_runtime=runtime,
                            producer_input_identity=input_identity,
                            producer_plan_sha256=study.digest(producer_root / 'plan.json')))
                    validations.append(base)
                    row = copy.deepcopy(base)
                    row.update(test=scores(epochs, seed, spec['id'], test=True),
                               test_names=['test-' + str(index) for index in range(205)],
                               knockout={'zero': {'iou': 0.4}}, checkpoint_sha256=audit['checkpoint_sha256'])
                    records.append(row)
                    write(run / 'test_replay' / spec['id'] / str(rate) / ('seed' + str(seed) + '.json'), row)
        records.sort(key=study.cell_key)
        validations.sort(key=study.cell_key)
        plan = dict(phase=phase, specs=json.loads(json.dumps(specs)), seeds=list(study.SEEDS),
            learning_rates=list(study.RATES), epochs=epochs, validation_only=False, historical_fits=historical,
            source_revision='fixture-new-producer', source_hashes={name: study.digest(collect.ROOT / name) for name in study.SOURCE_FILES},
            input_identity=input_identity, **runtime, **collect.FIXED_SETTINGS)
        summary = dict(plan=plan, records=records, checkpoint_audit=audits,
            validation_analysis=study.summarize(records, specs), test_analysis=study.summarize(records, specs, test=True))
        write(run / 'plan.json', plan)
        write(run / 'summary.json', summary)
        write(run / 'validation_fits.json', validations)
        write(run / 'checkpoint_audit.json', audits)
        if phase == 'matched60':
            write(run / 'reuse_audit.json', reuses)
        (run / 'report.md').write_text(study.format_report(summary))
        paths.append(run)
    return paths


def test_complete_collection_regenerates_every_artifact_byte_for_byte(tmp_path, monkeypatch):
    matched, longer = build_runs(tmp_path, monkeypatch)
    first, second = tmp_path / 'retained1', tmp_path / 'retained2'
    collect.collect(matched, longer, first)
    collect.collect(matched, longer, second)
    manifest = collect.read(first / 'artifact_sha256.json')
    assert manifest == collect.read(second / 'artifact_sha256.json')
    assert set(manifest) == {str(path.relative_to(first)) for path in first.rglob('*') if path.is_file()
                             and path.name != 'artifact_sha256.json'}
    for name, expected in manifest.items():
        assert study.digest(first / name) == expected
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert len(list(csv.DictReader((first / 'per_cell.csv').open()))) == 30
    audit = collect.read(first / 'audit.json')
    assert audit['historical_fits'] == 8 and audit['new_fits'] == 22
    assert audit['strict_checkpoint_hashes'] == 30
    assert all(row['exact_prefix_history_equal'] for row in audit['prefix60_checks'])
    assert len(list((first / 'historical_producers').glob('*.json'))) == 2
    assert (first / 'validation_curves_120ep.png').stat().st_size > 1000
    assert (first / 'training_loss_curves_120ep.png').stat().st_size > 1000
    assert (first / 'training_loss_curves_120ep.pdf').read_bytes().startswith(b'%PDF')
    assert (first / 'paired_entanglement_60_120.pdf').read_bytes().startswith(b'%PDF')
    stability = collect.read(first / 'stability.json')
    assert len(stability) == len(list(csv.DictReader((first / 'stability.csv').open()))) == 30
    assert all(row['max_positive_adjacent_loss_jump'] == 0 and row['max_positive_adjacent_loss_jump_epoch'] is None
               for row in stability)
    assert all(row['initial_train_loss_epoch'] == row['max_train_loss_epoch'] == 1
               and row['max_train_loss_after_epoch10_epoch'] == 11 for row in stability)
    assert 'not a diagnostic threshold' in (first / 'STABILITY.md').read_text()
    with pytest.raises(ValueError, match='new or empty'):
        collect.collect(matched, longer, first)


@pytest.mark.parametrize('corruption, message', [
    ('missing_cell', 'Incomplete'), ('duplicate_cell', 'Duplicate'), ('analysis', 'recomputed'),
    ('source', 'Executable source'), ('source_inventory', 'inventory'), ('config', 'configuration'),
    ('nonfinite', 'Nonfinite'), ('checkpoint', 'Checkpoint bytes'), ('history', 'learning curve'),
    ('audit', 'unverified'), ('historical_plan', 'producer plan bytes'),
    ('historical_metric', 'Historical artifact'), ('test_replay', 'test replay'),
    ('missing_output', 'No such file'),
])
def test_collection_rejects_stale_missing_or_corrupt_evidence(tmp_path, monkeypatch, corruption, message):
    matched, longer = build_runs(tmp_path, monkeypatch)
    summary = collect.read(longer / 'summary.json')
    row = summary['records'][0]
    directory = longer / 'fits' / row['id'] / str(row['learning_rate']) / ('seed' + str(row['seed']))
    if corruption == 'missing_cell':
        summary['records'].pop()
    elif corruption == 'duplicate_cell':
        summary['records'].append(copy.deepcopy(row))
    elif corruption == 'analysis':
        summary['validation_analysis']['rows'][0]['mean'] += 0.1
    elif corruption in ('source', 'source_inventory'):
        plan = collect.read(longer / 'plan.json')
        if corruption == 'source':
            plan['source_hashes'][study.SOURCE_FILES[0]] = 'changed'
        else:
            plan['source_hashes'].pop(study.SOURCE_FILES[0])
        summary['plan'] = plan
        write(longer / 'plan.json', plan)
    elif corruption == 'config':
        row['settings']['q_layers'] = 7
        validations = [{key: value for key, value in item.items() if key not in collect.TEST_FIELDS}
                       for item in summary['records']]
        write(longer / 'validation_fits.json', validations)
    elif corruption == 'nonfinite':
        row['history'][1]['train_loss'] = float('nan')
        (longer / 'summary.json').write_text(json.dumps(summary))
    elif corruption == 'checkpoint':
        (directory / 'checkpoint.pt').write_bytes(b'changed')
    elif corruption == 'history':
        (directory / 'history.json').write_text('[]')
    elif corruption == 'audit':
        summary['checkpoint_audit'][0]['validation_counts_equal'] = False
        write(longer / 'checkpoint_audit.json', summary['checkpoint_audit'])
    elif corruption.startswith('historical_'):
        item = collect.read(matched / 'plan.json')['historical_fits'][0]
        old_directory = Path(item['directory'])
        if corruption == 'historical_plan':
            path = collect.historical_plan(old_directory)
            path.write_text(path.read_text() + '\n')
        else:
            path = old_directory / 'metrics.json'
            path.write_text(path.read_text() + '\n')
    elif corruption == 'test_replay':
        replay = longer / 'test_replay' / row['id'] / str(row['learning_rate']) / ('seed' + str(row['seed']) + '.json')
        replay.write_text('{}')
    elif corruption == 'missing_output':
        (longer / 'validation_fits.json').unlink()
    if corruption != 'nonfinite':
        write(longer / 'summary.json', summary)
    out = tmp_path / 'out'
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        collect.collect(matched, longer, out)
    assert not out.exists()


@pytest.mark.parametrize('change, message', [('input_identity', 'input'), ('python', 'runtime')])
def test_cross_phase_audit_rejects_input_or_runtime_drift(tmp_path, monkeypatch, change, message):
    matched, longer = build_runs(tmp_path, monkeypatch)
    early, late = collect.audit_run(matched, 'matched60'), collect.audit_run(longer, 'long120')
    late['plan'][change] = 'different'
    with pytest.raises(ValueError, match=message):
        collect.audit_pair(early, late)


def test_prefix_audit_exposes_every_changed_count_and_loss_without_tolerance(tmp_path, monkeypatch):
    matched, longer = build_runs(tmp_path, monkeypatch)
    early, late = collect.audit_run(matched, 'matched60'), collect.audit_run(longer, 'long120')
    row = next(row for row in late['summary']['records'] if row['id'] == study.QUANTUM[0])
    row['history'][1]['train_loss'] += 1e-14
    row['history'][2]['val']['optical_subset_tp'] += 1
    result = collect.audit_pair(early, late)
    changed = next(check for check in result if not check['exact_prefix_history_equal'])
    assert changed['training_losses_exactly_equal'] is False
    assert changed['validation_classification_counts_exactly_equal'] is False
    assert changed['changed_validation_classification_epochs'] == [2]
    assert changed['training_loss_differences'][0]['difference'] != 0
    assert changed['validation_metric_differences'][0]['differences']['optical_subset_tp'] == 1
    report = collect.combined_report(early, late, collect.duration_comparison(early, late), result)
    assert 'must not be attributed solely to extra epochs' in report


def test_stability_summaries_keep_spike_epoch_ties_and_warmup_definition():
    losses = [5, 4, 3, 2, 2, 3, 2, 1, 2, 1, 2, 2]
    record = dict(id=study.QUANTUM[0], seed=42, learning_rate=0.001, best_epoch=57,
                  history=[dict(epoch=0, train_loss=None)] +
                          [dict(epoch=index, train_loss=loss) for index, loss in enumerate(losses, start=1)])
    matched = dict(plan={'epochs': 60, 'eval_every': 2}, summary={'records': [record]})
    later_record = copy.deepcopy(record)
    later_record['best_epoch'] = 118
    longer = dict(plan={'epochs': 120, 'eval_every': 2}, summary={'records': [later_record]})
    rows = collect.stability_summaries(matched, longer)
    first, second = rows
    assert first['initial_train_loss'] == first['max_train_loss'] == 5
    assert first['initial_train_loss_epoch'] == first['max_train_loss_epoch'] == 1
    assert first['max_train_loss_after_epoch10'] == 2
    assert first['max_train_loss_after_epoch10_epoch'] == 11
    assert first['max_positive_adjacent_loss_jump'] == 1
    assert first['max_positive_adjacent_loss_jump_epoch'] == 6
    assert first['selected_checkpoint_late'] is False
    assert second['selected_checkpoint_late'] is True
    report = collect.stability_report(rows)
    assert 'An initially high loss is not automatically instability' in report
    assert 'strictly after epoch 10' in report and 'not a diagnostic threshold' in report


def test_training_loss_figure_uses_every_epoch_seed_and_independent_linear_scales(tmp_path, monkeypatch):
    _, longer = build_runs(tmp_path, monkeypatch)
    summary = collect.read(longer / 'summary.json')
    inspected = []

    def inspect_figure(fig, out, stem, title):
        inspected.append(stem)
        assert stem == 'training_loss_curves_120ep'
        assert len(fig.axes) == 6
        for axis in fig.axes:
            assert axis.get_yscale() == 'linear'
            assert len(axis.get_shared_y_axes().get_siblings(axis)) == 1
            assert len(axis.lines) == 4  # Three complete seeds and the epoch60 boundary.
            for line in axis.lines[:3]:
                assert list(line.get_xdata()) == list(range(1, 121))
                assert len(line.get_ydata()) == 120
            assert list(axis.lines[-1].get_xdata()) == [60, 60]
        assert 'independent panel scales' in fig._suptitle.get_text()

    monkeypatch.setattr(collect, 'save_figure', inspect_figure)
    collect.plot_training_curves(summary, tmp_path)
    assert inspected == ['training_loss_curves_120ep']
