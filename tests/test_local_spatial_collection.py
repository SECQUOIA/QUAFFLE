"""Audit gates use complete synthetic records; checkpoint tests use real models."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import collect_local_spatial_study as collect
from scripts import run_local_spatial_study as study
from scripts import run_real_spatial_pilot as pilot


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    collect.write(path, value)


def metrics(n_samples, quality=0, test=False):
    counts = (71, 134) if test else (35, 195)
    assert sum(counts) == n_samples
    result = dict(loss=0.5, auc=0.7)
    pooled = dict(tp=0, fp=0, fn=0, tn=0)
    for sensor, samples in zip(('optical', 'sar'), counts):
        values = dict(tp=samples * (512 + quality), fp=samples * 128, fn=samples * 256)
        values['tn'] = samples * 4096 - sum(values.values())
        for name, value in values.items():
            result[sensor + '_subset_' + name] = value
            pooled[name] += value
        tp, fp, fn, tn = (values[key] for key in ('tp', 'fp', 'fn', 'tn'))
        result.update({sensor + '_subset_n_samples': samples, sensor + '_subset_iou': tp / (tp + fp + fn),
            sensor + '_subset_dice': 2 * tp / (2 * tp + fp + fn), sensor + '_subset_accuracy': (tp + tn) / (samples * 4096),
            sensor + '_subset_auc': 0.7})
    tp, fp, fn, tn = (pooled[key] for key in ('tp', 'fp', 'fn', 'tn'))
    result.update(iou=tp / (tp + fp + fn), dice=2 * tp / (2 * tp + fp + fn), accuracy=(tp + tn) / (n_samples * 4096))
    return result


def fake_checkpoint(path, row):
    return dict(sha256=collect.digest(path), strict_restore=True, finite_weights=True,
        initial_parameter_sha256=row['initial_parameter_sha256'],
        shared_initial_parameter_sha256='shared-seed-' + str(row['seed']),
        core_params=row['total_params'] - 503278, total_params=row['total_params'], trainable_params=row['trainable_params'])


def build_phase(directory, specs, epochs, source_hashes, manifest, context):
    settings = dict(seeds=[42, 77, 123], learning_rates=[0.001, 0.003], epochs=epochs,
        eval_every=2, fit_cap=128, batch_size=16, image_size=64, device='cuda', specs=None)
    plan = dict(settings=settings, specs=collect.canonical(specs), source_hashes=source_hashes,
        source_revision='synthetic-producing-revision', **collect.current_runtime(),
        input_identity=dict(manifest_sha256=collect.digest(manifest), verified_files=3476,
            stats_files_sha256={'splits/norm_stats.json': source_hashes['splits/norm_stats.json']}, raw_inventory_sha256='raw-input'),
        normalization_policy='All global fit images; validation excluded; optimization capped separately',
        test_policy='No trained-model test evaluation until all validation tuning and replication fits finish',
        device='cuda', gpu='NVIDIA RTX A1000', gpu_allocator_fraction=0.12, threads=2, experiment_context=context)
    save(directory / 'plan.json', plan)
    names = collect.read(manifest)
    fit_indices, val_indices = list(range(128)), list(range(1303, 1533))
    norm = dict(fit_only=True, n_fit_train_tiles=1303, n_excluded_val_tiles=230,
        fit_names_sha256=hashlib.sha256('\n'.join(names['train'][:1303]).encode()).hexdigest()[:16])
    core_counts = dict(quantum=16, mlp=284, conv=60, sincos=160)
    quality_by = dict(local_quantum_z=20, local_quantum_xz_zz=50, local_quantum_frozen=10,
                      local_quantum_no_entanglement=30, local_mlp=40, local_conv=45, local_sincos=25)
    trials, paths, records = [], {}, []
    def one(spec, seed, rate):
        cfg = spec['config']
        quality = quality_by[spec['id']] + (3 if rate == .003 else 0) + seed % 3
        val = metrics(230, quality)
        row = dict(id=spec['id'], model_type=spec['model_type'], seed=seed, learning_rate=rate,
            total_params=503278 + core_counts[cfg['local_spatial_kernel']],
            trainable_params=503278 if cfg['q_training_control'] == 'frozen' else 503278 + core_counts[cfg['local_spatial_kernel']],
            fit_indices=fit_indices, val_indices=val_indices, fit_names=names['train'][:128], val_names=names['train'][1303:],
            normalization=norm, fit_sensor_counts={'optical': 20, 'sar': 108}, n_global_fit=1303,
            initial_parameter_sha256=(cfg['local_spatial_kernel'] + '-seed-' + str(seed)),
            best_epoch=2, val=val, elapsed_s=1.0, diagnostics={})
        row['settings'] = collect.expected_fit_settings(spec, dict(row, settings={'data_root': '/immutable/synthetic-data'}), settings)
        row['history'] = [dict(epoch=epoch, train_loss=None if epoch == 0 else 0.7,
            **({'val': val if epoch == 2 else metrics(230, quality - 2)} if epoch % 2 == 0 else {})) for epoch in range(epochs + 1)]
        probe = dict(mean=0.1, std=0.2, between_sample_std=0.1, spatial_std=0.1)
        if cfg['local_spatial_kernel'] in ('quantum', 'sincos'):
            probe['input_angles'] = dict(n_angles=8192, saturation_fraction=0.1, mean_abs_tanh=0.4, max_abs_tanh=0.99)
        row['diagnostics'] = dict(initial_bottleneck_output=[copy.deepcopy(probe), copy.deepcopy(probe)],
            selected_bottleneck_output=[copy.deepcopy(probe), copy.deepcopy(probe)], movement={})
        path = (directory / 'tuning' / spec['id'] / str(rate) / 'seed42' if seed == 42
                else directory / 'replication' / spec['id'] / ('seed' + str(seed)))
        save(path / 'metrics.json', row)
        save(path / 'history.json', row['history'])
        (path / 'checkpoint.pt').write_bytes(('synthetic-checkpoint-' + str(collect.fit_identity(row))).encode())
        paths[spec['id'], seed, rate] = path
        return row
    for spec in specs:
        trials += [one(spec, 42, rate) for rate in collect.RATES]
    chosen = pilot.choose_rates(trials)
    for spec in specs:
        selected = chosen[spec['id']]
        records.append(copy.deepcopy(selected))
        records += [one(spec, seed, selected['learning_rate']) for seed in (77, 123)]
    for row in records:
        row['test'] = metrics(205, quality_by[row['id']], test=True)
        row['test_names'] = names['test']
        row['knockout'] = {mode: metrics(205, quality_by[row['id']] - index - 1, test=True)
                           for index, mode in enumerate(('zero', 'random'))}
        path = paths[collect.fit_identity(row)]
        row['checkpoint'] = str((path / 'checkpoint.pt').relative_to(directory))
        save(path / 'selected_test.json', row)
    summary = dict(plan=plan, selection={name: row['learning_rate'] for name, row in chosen.items()},
                   records=records, analysis=pilot.aggregate(records))
    save(directory / 'summary.json', summary)
    save(directory / 'tuning_candidates.json', trials)
    save(directory / 'validation_selection.json', {name: dict(learning_rate=row['learning_rate'],
        val_iou=row['val']['iou'], best_epoch=row['best_epoch']) for name, row in chosen.items()})
    save(directory / 'paired_spatial_analysis.json', study.paired_analysis(records))
    save(directory / 'fixed_rate_readout_analysis.json', study.fixed_rate_readout_contrasts(trials, require_both=epochs == 60))
    return summary


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    for name in collect.SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic bound source: ' + name)
    manifest = source / 'splits/flood_combined.json'
    # Some basenames deliberately occur in both directory-scoped splits.
    save(manifest, dict(conflict_policy='drop', train=['tile' + str(i) for i in range(1533)],
                        test=['tile' + str(i) for i in range(205)]))
    hashes = {name: collect.digest(source / name) for name in collect.SOURCE_FILES}
    monkeypatch.setattr(collect, 'ROOT', source)
    monkeypatch.setattr(collect, 'audit_checkpoint', fake_checkpoint)
    short, long = tmp_path / 'short', tmp_path / 'long'
    summary = build_phase(short, study.study_specs(), 60, hashes, manifest,
                         dict(phase='pilot', protocol='local-spatial-seven-spec-v1'))
    build_phase(long, study.finalist_specs(summary), 120, hashes, manifest,
        dict(phase='extend', protocol='local-spatial-two-finalists-v1',
            baseline_summary_sha256=collect.digest(short / 'summary.json'),
            selection_policy='Mean paired validation IoU; Q tie Z; classical tie smaller core then MLP/conv/sincos'))
    return short, long, source


def test_complete_study_retains_all36_fits_and_losing_rates(campaign):
    short, long, _ = campaign
    result = collect.audit_study(short, long)
    assert [len(result[phase]['fits']) for phase in ('pilot60', 'extend120')] == [28, 8]
    assert [len(result[phase]['tuning']) for phase in ('pilot60', 'extend120')] == [14, 4]
    assert [len(result[phase]['summary']['records']) for phase in ('pilot60', 'extend120')] == [21, 6]
    assert result['finalists'] == ['local_quantum_xz_zz', 'local_conv']
    assert len(result['pilot60']['fixed_rate_readout']) == 2
    assert result['extend120']['fixed_rate_readout'] == []
    rows, seeds, tuning = collect.summary_tables(result)
    assert len(rows) == 9 and len(seeds) == 27 and len(tuning) == 18
    assert sum(not row['selected_rate'] for row in tuning) == 9
    prefixes = result['duration_evidence']['prefixes']
    assert len(prefixes) == 8 and all(row['comparable'] and row['exact_history_match'] for row in prefixes)


@pytest.mark.parametrize('damage', ['source', 'missing_fit', 'history', 'selection', 'replay', 'checkpoint_path',
                                  'test_cohort', 'normalization', 'sensor_counts', 'paired_analysis', 'fixed_rate', 'context', 'runtime'])
def test_corrupted_study_is_rejected(campaign, damage):
    short, long, source = campaign
    summary = collect.read(short / 'summary.json')
    row = summary['records'][0]
    selected_dir = short / row['checkpoint']
    if damage == 'source':
        (source / 'utilsTorch/local_spatial.py').write_text('stale source')
    elif damage == 'missing_fit':
        (selected_dir.parent / 'metrics.json').unlink()
    elif damage == 'history':
        history = collect.read(selected_dir.parent / 'history.json')
        history.pop()
        save(selected_dir.parent / 'history.json', history)
    elif damage == 'selection':
        summary['selection'][row['id']] = .001
        save(short / 'summary.json', summary)
    elif damage == 'replay':
        saved = collect.read(selected_dir.parent / 'selected_test.json')
        saved['test']['iou'] += .1
        save(selected_dir.parent / 'selected_test.json', saved)
    elif damage == 'checkpoint_path':
        row['checkpoint'] = summary['records'][1]['checkpoint']
        save(short / 'summary.json', summary)
    elif damage == 'test_cohort':
        row['test_names'] = list(reversed(row['test_names']))
        save(selected_dir.parent / 'selected_test.json', row)
        save(short / 'summary.json', summary)
    elif damage == 'normalization':
        record = collect.read(selected_dir.parent / 'metrics.json')
        record['normalization']['fit_names_sha256'] = 'incorrect'
        record['settings']['_resolved_norm_provenance'] = record['normalization']
        save(selected_dir.parent / 'metrics.json', record)
    elif damage == 'sensor_counts':
        row['test']['sar_subset_tp'] += 1
        save(selected_dir.parent / 'selected_test.json', row)
        save(short / 'summary.json', summary)
    elif damage == 'paired_analysis':
        save(short / 'paired_spatial_analysis.json', [])
    elif damage == 'fixed_rate':
        save(short / 'fixed_rate_readout_analysis.json', [])
    elif damage == 'context':
        plan = collect.read(long / 'plan.json')
        plan['experiment_context']['baseline_summary_sha256'] = 'another baseline'
        saved = collect.read(long / 'summary.json'); saved['plan'] = plan
        save(long / 'plan.json', plan); save(long / 'summary.json', saved)
    else:
        plan = collect.read(long / 'plan.json'); plan['packages']['pennylane'] = 'another version'
        saved = collect.read(long / 'summary.json'); saved['plan'] = plan
        save(long / 'plan.json', plan); save(long / 'summary.json', saved)
    with pytest.raises(ValueError):
        collect.audit_study(short, long)


def test_collection_writes_tables_plots_and_hashes(campaign, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda value: None)
    short, long, _ = campaign
    output = tmp_path / 'artifacts'
    collect.main(['--pilot-dir', str(short), '--extension-dir', str(long), '--out-dir', str(output)])
    manifest = collect.read(output / 'artifact_sha256.json')
    assert all(collect.digest(output / name) == value for name, value in manifest.items())
    assert len(list(csv.DictReader((output / 'per_seed.csv').open()))) == 27
    assert len(list(csv.DictReader((output / 'tuning.csv').open()))) == 18
    assert len(list(csv.DictReader((output / 'summary.csv').open()))) == 9
    for name in ('paired_quantum_classical', 'validation_curves', 'input_angle_saturation'):
        assert (output / 'plots' / (name + '.png')).read_bytes().startswith(b'\x89PNG')
        assert (output / 'plots' / (name + '.pdf')).read_bytes().startswith(b'%PDF')
    text = (output / 'README.md').read_text()
    assert 'n=1' in text and 'not independent replication' in text


@pytest.mark.parametrize('damage', [None, 'nonfinite', 'alias', 'metadata', 'movement'])
def test_real_checkpoint_audit_rejects_corrupt_weights_metadata_and_aliases(tmp_path, damage):
    import torch
    from tools import training_common as tc
    from utilsTorch.utilsTorch import QVUNetSegmentation
    cfg = dict(study.study_specs()[0]['config'], seed=42)
    tc.seed_everything(42)
    model = tc.build_model(QVUNetSegmentation, cfg)
    row = dict(settings=cfg, model_type='quantum', best_epoch=0, seed=42,
        initial_parameter_sha256=collect.initial_parameter_digest(model),
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad), diagnostics={'movement': {}})
    groups, _ = pilot.parameter_groups(model)
    for name, parameters in groups.items():
        if parameters:
            squared = sum(float(parameter.detach().square().sum()) for _, parameter in parameters)
            row['diagnostics']['movement'][name] = dict(l2=0., relative_l2=0. if squared else None)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = dict(settings=copy.deepcopy(cfg), model_type='quantum', best_epoch=0, best_state_dict=state)
    if damage == 'nonfinite':
        next(iter(state.values())).flatten()[0] = float('nan')
    elif damage == 'alias':
        keys = [key for key in state if key.endswith('quantum_params.0')]
        assert len(keys) >= 2
        state[keys[0]].flatten()[0] += .1
    elif damage == 'metadata':
        checkpoint['settings']['local_spatial_readout'] = 'xz_zz'
    elif damage == 'movement':
        row['diagnostics']['movement']['circuit_angles']['l2'] = 1.0
    path = tmp_path / 'checkpoint.pt'; torch.save(checkpoint, path)
    if damage:
        with pytest.raises(ValueError):
            collect.audit_checkpoint(path, row)
    else:
        details = collect.audit_checkpoint(path, row)
        assert details['strict_restore'] and details['core_params'] == 16


def test_z_readout_phase_angles_are_unobservable_but_rich_readout_exposes_them():
    import torch
    from utilsTorch.local_spatial import LocalSpatialBottleneckBlock
    tokens = torch.tensor([[.2, -.4, .6, .1], [-.3, .2, .5, -.6]])
    gradients = {}
    for readout in ('z', 'xz_zz'):
        block = LocalSpatialBottleneckBlock(1, readout=readout, feature_channels=1, seed=17)
        weights = torch.arange(1., 13.).reshape(1, 12)
        (block._evaluate_core(tokens) * weights).sum().backward()
        gradients[readout] = block.quantum_params[0].grad
    assert gradients['z'][..., 2].abs().max() < 1e-8
    assert gradients['xz_zz'][..., 2].abs().max() > .01


def test_gradient_record_distinguishes_frozen_absence_from_small_derivatives():
    import torch
    from scripts.audit_local_spatial_angles import gradient_record
    parameter = torch.nn.Parameter(torch.tensor([.1, .2]))
    initial = parameter.detach().clone()
    parameter.grad = torch.tensor([0., 1e-12])
    record = gradient_record(parameter, initial)
    assert record['gradient_nonzero_exact'] == [False, True]
    assert record['gradient_above_threshold'] == [False, False]
    parameter.requires_grad_(False); parameter.grad = None
    record = gradient_record(parameter, initial)
    assert record['gradient'] is None and record['movement_l2'] == 0


@pytest.mark.parametrize('control', ['trainable', 'frozen', 'no_entanglement'])
def test_fixed_optimization_probe_retains_every_angle_and_frozen_semantics(control):
    import torch
    from tools import training_common as tc
    from utilsTorch.utilsTorch import QVUNetSegmentation
    from scripts.audit_local_spatial_angles import probe_model
    cfg = dict(study.study_specs()[1]['config'], seed=42, q_training_control=control)
    tc.seed_everything(42)
    model = tc.build_model(QVUNetSegmentation, cfg)
    block = model.model.vertex.q_block
    initial = {name: value.detach().clone() for name, value in block.named_parameters()
               if name in ('quantum_params.0', 'input_scales.0')}
    if control != 'frozen':
        with torch.no_grad():
            block.quantum_params[0][0, 0, 0] += .1
    images = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    masks = torch.arange(2 * 32 * 32).reshape(2, 1, 32, 32) % 2
    result = probe_model(model, images, masks, initial)
    assert len(result['input_angle_saturation']) == 2
    diagnostics = result['parameter_diagnostics']
    assert len(diagnostics['quantum_params.0']['movement'][0]) == 4
    assert len(diagnostics['quantum_params.0']['movement'][0][0]) == 3
    assert (diagnostics['quantum_params.0']['gradient'] is None) == (control == 'frozen')
    assert diagnostics['quantum_params.0']['movement_l2'] == pytest.approx(0 if control == 'frozen' else .1)


@pytest.mark.parametrize('damage', [None, 'probe', 'checkpoint', 'gradient_flags', 'gradient_mask'])
def test_optional_individual_angle_audit_is_bound_to_models_probes_and_values(campaign, tmp_path, damage):
    import torch
    from scripts.audit_local_spatial_angles import gradient_record
    short, long, source = campaign
    result = collect.audit_study(short, long)
    diagnostic_source = source / 'scripts/audit_local_spatial_angles.py'
    diagnostic_source.write_text('Synthetic diagnostic source identity')
    audit = dict(summary_sha256={phase: result[phase]['summary_sha256'] for phase in ('pilot60', 'extend120')},
        diagnostic_source_sha256=collect.digest(diagnostic_source), collector_source_sha256=collect.digest(collect.__file__),
        runtime=dict(**collect.current_runtime(), device='cpu', threads=2),
        source_hashes=result['pilot60']['summary']['plan']['source_hashes'],
        input_identity=result['pilot60']['summary']['plan']['input_identity'], records=[])
    for phase in ('pilot60', 'extend120'):
        for selected in result[phase]['summary']['records']:
            if selected['model_type'] != 'quantum':
                continue
            frozen = selected['settings']['q_training_control'] == 'frozen'
            diagnostics = {}
            for name, shape in (('quantum_params.0', (1, 4, 3)), ('input_scales.0', (1, 4))):
                parameter = torch.nn.Parameter(torch.zeros(shape), requires_grad=not frozen)
                if not frozen:
                    parameter.grad = torch.ones(shape) * .001
                diagnostics[name] = gradient_record(parameter, parameter.detach().clone())
            audit['records'].append(dict(phase=phase, id=selected['id'], seed=selected['seed'],
                probe_names=selected['fit_names'][:8], initial_parameter_sha256=selected['initial_parameter_sha256'],
                checkpoint_sha256=result[phase]['checkpoints'][selected['checkpoint']]['sha256'],
                probe_tensor_sha256='f' * 64, parameter_diagnostics=diagnostics))
    if damage == 'probe':
        audit['records'][0]['probe_names'] = ['validation-image']
    elif damage == 'checkpoint':
        audit['records'][0]['checkpoint_sha256'] = 'different bytes'
    elif damage == 'gradient_flags':
        audit['records'][0]['parameter_diagnostics']['quantum_params.0']['requires_grad'] = False
    elif damage == 'gradient_mask':
        audit['records'][0]['parameter_diagnostics']['quantum_params.0']['gradient_above_threshold'][0][0][0] = False
    path = tmp_path / 'individual_angles.json'; save(path, audit)
    if damage:
        with pytest.raises(ValueError):
            collect.validate_angle_audit(path, result)
    else:
        assert len(collect.validate_angle_audit(path, result)['records']) == 15


def test_duration_inventory_does_not_invent_missing_replication_rate_prefixes(campaign):
    short, long, _ = campaign
    result = collect.audit_study(short, long)
    extension = copy.deepcopy(result['extend120'])
    for name in extension['summary']['selection']:
        extension['summary']['selection'][name] = .001
    for row in extension['summary']['records']:
        row['learning_rate'] = .001
        row['settings']['learning_rate'] = .001
    for item in extension['fits']:
        row = item['record']
        if row['seed'] != 42:
            row['learning_rate'] = .001
            row['settings']['learning_rate'] = .001
    evidence = collect.duration_evidence(result['pilot60'], extension)
    assert sum(row['comparable'] for row in evidence['prefixes']) == 4
    assert {row['seed'] for row in evidence['prefixes'] if row['comparable']} == {42}
    assert all(not row['same_learning_rate'] and 'not an epoch-only contrast' in row['qualification']
               for row in evidence['selected_rate_contrasts'])


def test_duration_prefix_differences_are_measured_without_claiming_equality(campaign):
    short, long, _ = campaign
    result = collect.audit_study(short, long)
    result['extend120']['fits'][0]['record']['history'][1]['train_loss'] += .5
    evidence = collect.duration_evidence(result['pilot60'], result['extend120'])
    changed = next(row for row in evidence['prefixes'] if not row['exact_history_match'])
    assert changed['max_abs_train_loss_delta'] == pytest.approx(.5)
    assert changed['max_abs_validation_iou_delta'] == 0
