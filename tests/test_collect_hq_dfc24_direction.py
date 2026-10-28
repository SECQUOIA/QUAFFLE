"""Synthetic audit fixtures are plumbing tests, never scientific fit evidence."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import collect_hq_dfc24_direction as collect
from scripts import run_hq_dfc24_direction as study
from utilsTorch.hq_direction import HQDirectionUNet, model_config


def metrics(n_samples, quality=0, *, test=False):
    result = dict(loss=.7, auc=.6)
    sizes = (71, 134) if test else (35, 195)
    assert sum(sizes) == n_samples
    pooled = dict(tp=0, fp=0, fn=0, tn=0)
    for sensor, samples in zip(('optical', 'sar'), sizes):
        counts = dict(tp=samples * (2000 + quality), fp=samples * 400, fn=samples * 800)
        counts['tn'] = samples * 128 * 128 - sum(counts.values())
        for key, value in counts.items():
            result[sensor + '_subset_' + key] = value
            pooled[key] += value
        tp, fp, fn, tn = (counts[key] for key in ('tp', 'fp', 'fn', 'tn'))
        result.update({sensor + '_subset_iou': tp/(tp+fp+fn), sensor + '_subset_dice': 2*tp/(2*tp+fp+fn),
            sensor + '_subset_accuracy': (tp+tn)/sum(counts.values()), sensor + '_subset_auc': .6,
            sensor + '_subset_n_samples': samples})
    tp, fp, fn, tn = (pooled[key] for key in ('tp', 'fp', 'fn', 'tn'))
    return dict(result, iou=tp/(tp+fp+fn), dice=2*tp/(2*tp+fp+fn), accuracy=(tp+tn)/sum(pooled.values()))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source, raw, data_root = (tmp_path / name for name in ('source', 'raw', 'data'))
    for name in (*study.SOURCE_FILES, collect.PROTOCOL, 'scripts/collect_hq_dfc24_direction.py'):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic inert source fixture: ' + name)
    names = [f'train{i:04d}' for i in range(1533)]
    manifest = dict(conflict_policy='drop', train=names, test=[f'test{i:04d}' for i in range(205)])
    collect.write(source / 'splits/flood_combined.json', manifest)
    collect.write(data_root / 'splits/flood_combined.json', manifest)
    inputs = dict(manifest_sha256=collect.digest(source / 'splits/flood_combined.json'), verified_files=3476,
        stats_files_sha256={}, raw_inventory_sha256='synthetic-input-identity')
    monkeypatch.setattr(collect, 'ROOT', source)
    monkeypatch.setattr(study.pilot, 'verify_inputs', lambda path: copy.deepcopy(inputs))
    settings = dict(device='cuda', batch_size=8, epochs=60, fit_cap=128, image_size=128,
                    base_channels=8, eval_every=2, augment=True)
    plan = dict(settings=settings, seeds=list(study.SEEDS), learning_rates=list(study.RATES),
        source_hashes={name: collect.digest(source / name) for name in study.SOURCE_FILES},
        source_revision='synthetic-not-an-experiment', input_identity=inputs,
        protocol_sha256=collect.digest(source / collect.PROTOCOL),
        test_policy='All eight fits finish before six selected checkpoint tests; test already exposed/descriptive',
        augmentation_policy='Stateless D4 by(seed,epoch,original index), independent seeded sample order; train only',
        normalization_policy='All 1303 global fit images, validation excluded; optimization cap separate',
        runtime=dict(**collect.current_runtime(), device='cuda:0', gpu='Synthetic fixture',
                     gpu_allocator_fraction=.12, threads=2, interop_threads=2))
    def cohort(seed):
        fit = list(range(128))
        val = list(range(1303, 1533))
        return dict(fit=fit, validation=val, all_fit=list(range(1303)),
            dataset=SimpleNamespace(names=names, modalities=['optical' if i % 5 == 0 else 'sar' for i in range(1533)]),
            normalization=dict(fit_only=True, seed=seed, fit_names_sha256='synthetic-fit-' + str(seed)))
    monkeypatch.setattr(study.pilot, 'load_training_data', lambda args, seed: cohort(seed))
    args = argparse.Namespace(**settings, data_root=data_root)
    gradients = dict(min=.01, max=.03, mean=.02, steps=16, nonzero_steps=16)
    probes = dict(n_angles=384, min_radians=-.8, max_radians=.7, mean_abs_radians=.3, saturation_fraction=0.)
    entries, paths = [], {}
    def one(kind, seed, rate):
        data = cohort(seed)
        model = HQDirectionUNet(kind=kind, seed=seed)
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        config = model_config(model)
        quality = (40 if kind == 'quantum' else 20) + (10 if rate == (.003 if kind == 'quantum' else .001) else 0) + seed % 3
        val = metrics(230, quality)
        history = []
        for epoch in range(61):
            schedule = None if epoch == 0 else dict(sha256=study.epoch_schedule(data['fit'], seed, epoch, True)[2], n_samples=128)
            point = dict(epoch=epoch, train_loss=None if epoch == 0 else .7, train_schedule=schedule,
                         gradient_norms={} if epoch == 0 else dict(core=gradients, common=gradients))
            if epoch % 2 == 0:
                point['val'] = val if epoch >= 2 else metrics(230, quality-1)
            history.append(point)
        state_hash = study.state_hash(model.state_dict())
        row = dict(id='hq_' + kind, kind=kind, seed=seed, learning_rate=rate, model_config=config,
            settings=study.fit_settings(kind, seed, rate, args, data),
            total_params=sum(p.numel() for p in model.parameters()),
            trainable_params=sum(p.numel() for p in model.parameters()),
            core_params=sum(p.numel() for p in model.bottleneck.core_parameters()),
            initial_state_sha256=state_hash, selected_state_sha256=state_hash,
            initial_shared_state_sha256=study.shared_hash(model), best_epoch=2, val=val,
            history=history, elapsed_s=1., fit_indices=data['fit'], val_indices=data['validation'],
            fit_names=names[:128], val_names=names[1303:], fit_sensor_counts=dict(optical=26, sar=102),
            normalization=data['normalization'], n_global_fit=1303,
            diagnostics=dict(initial_angle_probes=[probes], selected_angle_probes=[probes],
                movement=study.movement(model, initial), probe_names=names[:8],
                gradient_norms={group: dict(gradients, steps=960, nonzero_steps=960) for group in ('core', 'common')}))
        directory = collect.expected_directory(row)
        path = raw / directory
        collect.write(path / 'metrics.json', row)
        collect.write(path / 'history.json', history)
        torch.save(dict(format='hq-direction-v1', kind=kind, seed=seed, best_epoch=2,
            settings=row['settings'], model_config=config, best_state_dict=model.state_dict()), path / 'checkpoint.pt')
        entries.append(dict(directory=directory, record=row))
        paths[collect.identity(row)] = path
        return row
    tuning = [one(kind, 42, rate) for kind in study.KINDS for rate in study.RATES]
    chosen = study.pilot.choose_rates(tuning)
    selected = [copy.deepcopy(chosen['hq_' + kind]) for kind in study.KINDS]
    for kind in study.KINDS:
        selected += [one(kind, seed, chosen['hq_' + kind]['learning_rate']) for seed in (77, 123)]
    selected = copy.deepcopy(selected)
    for row in selected:
        row.update(test=metrics(205, 20 if row['kind'] == 'quantum' else 0, test=True), test_names=manifest['test'],
            knockout={mode: metrics(205, -index, test=True) for index, mode in enumerate(study.MODES)})
        row['checkpoint'] = str((paths[collect.identity(row)] / 'checkpoint.pt').relative_to(raw))
        collect.write(paths[collect.identity(row)] / 'selected_test.json', row)
    summary = dict(plan=plan, selection={name: row['learning_rate'] for name, row in chosen.items()},
                   records=selected, analysis=study.analysis(selected))
    audits = []
    for entry in entries:
        row = entry['record']
        restored, audit = study.restore_checkpoint(paths[collect.identity(row)] / 'checkpoint.pt', row)
        del restored
        audits.append(dict(id=row['id'], seed=row['seed'], learning_rate=row['learning_rate'], **audit))
    for filename, value in (('plan.json', plan), ('summary.json', summary), ('all_fits.json', entries),
        ('tuning_candidates.json', tuning), ('checkpoint_audit.json', audits),
        ('validation_selection.json', {name: dict(learning_rate=row['learning_rate'], best_epoch=2, val_iou=row['val']['iou']) for name, row in chosen.items()})):
        collect.write(raw / filename, value)
    return raw, data_root, source


def test_complete_grid_actual_checkpoint_restoration_and_sensor_pairing(campaign):
    raw, data, _ = campaign
    result = collect.audit_campaign(raw, data)
    assert len(result['fits']) == len(result['checkpoint_audit']) == 8
    assert len(result['summary']['records']) == 6
    assert result['summary']['selection'] == dict(hq_quantum=.003, hq_mlp=.001)
    contrasts = collect.comparison(result)
    assert len(contrasts['paired']) == len(contrasts['knockouts']) == 6
    assert all(row['wins'] == 3 and not row['same_learning_rate'] for row in contrasts['paired'])
    rows = collect.tables(result, contrasts)
    assert {key: len(values) for key, values in rows.items()} == dict(summary=2, per_seed=6, tuning=4, paired=6, knockout=6, diagnostics=16)
    assert sum(not row['selected_rate'] for row in rows['tuning']) == 2


@pytest.mark.parametrize('damage', ('source', 'runtime', 'budget', 'missing_fit', 'history', 'test_counts',
                                   'selection', 'replay', 'checkpoint', 'movement', 'extra_fit'))
def test_corrupt_or_incomplete_campaign_is_rejected(campaign, damage):
    raw, data, source = campaign
    summary = collect.read(raw / 'summary.json')
    selected = summary['records'][0]
    directory = raw / Path(selected['checkpoint']).parent
    if damage == 'source':
        (source / 'utilsTorch/hq_direction.py').write_text('stale')
    elif damage in ('runtime', 'budget'):
        plan = collect.read(raw / 'plan.json')
        if damage == 'runtime':
            plan['runtime']['packages']['torch'] = 'different'
        else:
            plan['settings']['epochs'] = 62
        summary['plan'] = plan
        collect.write(raw / 'plan.json', plan)
        collect.write(raw / 'summary.json', summary)
    elif damage == 'missing_fit':
        (directory / 'history.json').unlink()
    elif damage == 'history':
        history = collect.read(directory / 'history.json')
        history.pop()
        collect.write(directory / 'history.json', history)
    elif damage == 'test_counts':
        selected['test']['optical_subset_iou'] += .01
        collect.write(directory / 'selected_test.json', selected)
        collect.write(raw / 'summary.json', summary)
    elif damage == 'selection':
        summary['selection']['hq_quantum'] = .001
        collect.write(raw / 'summary.json', summary)
    elif damage == 'replay':
        replay = collect.read(directory / 'selected_test.json')
        replay['test_names'].reverse()
        collect.write(directory / 'selected_test.json', replay)
    elif damage == 'checkpoint':
        (directory / 'checkpoint.pt').write_bytes(b'bad checkpoint')
    elif damage == 'movement':
        fits = collect.read(raw / 'all_fits.json')
        row = fits[0]['record']
        row['diagnostics']['movement']['core']['l2'] = 7.
        collect.write(raw / fits[0]['directory'] / 'metrics.json', row)
        collect.write(raw / 'all_fits.json', fits)
    else:
        collect.write(raw / 'unplanned/metrics.json', {})
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError, EOFError, KeyError, IndexError)):
        collect.audit_campaign(raw, data)


def test_exports_cover_every_file_preserve_raw_bytes_and_are_regenerable(campaign, tmp_path):
    raw, data, _ = campaign
    result = collect.audit_campaign(raw, data)
    first, second = tmp_path / 'out1', tmp_path / 'out2'
    collect.export(result, raw, data, first)
    collect.export(result, raw, data, second)
    hashes = collect.read(first / 'artifact_sha256.json')
    assert hashes == collect.read(second / 'artifact_sha256.json')
    assert set(hashes) == {str(path.relative_to(first)) for path in first.rglob('*') if path.is_file()} - {'artifact_sha256.json'}
    assert all(collect.digest(first / name) == value for name, value in hashes.items())
    assert (first / 'summary.json').read_bytes() == (raw / 'summary.json').read_bytes()
    assert len(list((first / 'plots').glob('*.png'))) == len(list((first / 'plots').glob('*.pdf'))) == 4
    assert not list(first.rglob('*.pt'))
    assert len(list((first / 'raw_fits').rglob('history.json'))) == 8
    assert len(list((first / 'raw_fits').rglob('selected_test.json'))) == 6
    provenance = collect.read(first / 'provenance.json')
    assert provenance['independent_inference_replays'] == 0
    with pytest.raises(ValueError, match='empty'):
        collect.export(result, raw, data, first)


def test_output_must_not_overlap_inputs(campaign):
    raw, data, _ = campaign
    with pytest.raises(ValueError):
        collect.main(['--work-dir', str(raw), '--data-root', str(data), '--out-dir', str(raw / 'new-output')])
