#!/usr/bin/env python3
"""Strictly audit and retain the completed 60/120-epoch local spatial study.

This collector never loads image data or reevaluates a model. Checkpoints are
strictly restored on the CPU to bind saved metadata, finite weights, parameter
counts, initialization and movement to every retained fit, including losing
learning-rate candidates. Run the command in a bounded, GPU-hidden CPU scope.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_local_spatial_study as study
from scripts import run_real_spatial_pilot as pilot

SEEDS = (42, 77, 123)
RATES = (0.001, 0.003)
SOURCE_FILES = {
    'scripts/run_real_spatial_pilot.py', 'scripts/run_parameter_budget_comparison.py',
    'utilsTorch/unetTorch.py', 'utilsTorch/utilsTorch.py', 'tools/training_common.py',
    'tools/checkpoint_evaluation.py', 'flood_data.py', 'splits/norm_stats.json',
    *study.EXTRA_SOURCES,
}
RUNTIME_KEYS = ('python', 'packages', 'device', 'gpu', 'gpu_allocator_fraction', 'threads')
COHORT_KEYS = ('fit_indices', 'val_indices', 'fit_names', 'val_names', 'normalization',
               'fit_sensor_counts', 'n_global_fit')
METRIC_KEYS = ('iou', 'dice', 'accuracy', 'auc')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def safe_relative(root, name):
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Path escapes its run/source directory: ' + str(name))
    return path


def verify_source(plan):
    require(set(plan['source_hashes']) == SOURCE_FILES, 'Producing source inventory differs from the planned study')
    for name, expected in plan['source_hashes'].items():
        path = safe_relative(ROOT, name)
        require(path.is_file() and digest(path) == expected, 'Executable source differs from producing plan: ' + name)


def verify_manifest(plan):
    path = ROOT / 'splits/flood_combined.json'
    require(digest(path) == plan['input_identity']['manifest_sha256'], 'Committed manifest differs from the producing input identity')
    manifest = read(path)
    require(manifest['conflict_policy'] == 'drop' and len(manifest['train']) == 1533 and len(manifest['test']) == 205,
            'Manifest is not the planned conflict-free DFC24 cohort')
    return manifest


def current_runtime():
    return dict(python=platform.python_version(), packages={name: importlib.metadata.version(name)
        for name in ('torch', 'pennylane', 'numpy', 'rasterio')})


def verify_runtime(plan):
    actual = current_runtime()
    require(all(actual[key] == plan[key] for key in ('python', 'packages')),
            'Collector/diagnostic Python or package versions differ from the producing plan')
    return actual


def check_metrics(metrics, n_samples, label, *, test=False):
    for name in METRIC_KEYS:
        require(isinstance(metrics.get(name), (int, float)) and math.isfinite(metrics[name])
                and 0 <= metrics[name] <= 1, label + ': invalid ' + name)
    require(math.isfinite(metrics.get('loss', float('nan'))) and metrics['loss'] >= 0, label + ': invalid loss')
    totals = {key: 0 for key in ('tp', 'fp', 'fn', 'tn', 'n_samples')}
    for sensor in ('optical', 'sar'):
        prefix = sensor + '_subset_'
        counts = {key: metrics.get(prefix + key) for key in totals}
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in counts.values()), label + ': invalid sensor counts')
        for key, value in counts.items():
            totals[key] += value
        tp, fp, fn, tn = (counts[key] for key in ('tp', 'fp', 'fn', 'tn'))
        denominators = dict(iou=tp + fp + fn, dice=2 * tp + fp + fn, accuracy=tp + fp + fn + tn)
        numerators = dict(iou=tp, dice=2 * tp, accuracy=tp + tn)
        for key in METRIC_KEYS:
            value = metrics.get(prefix + key)
            require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
                    label + ': invalid sensor ' + key)
            if key != 'auc':
                expected = numerators[key] / denominators[key] if denominators[key] else 0.0
                require(math.isclose(value, expected, abs_tol=1e-12), label + ': sensor confusion mismatch ' + key)
    require(totals['n_samples'] == n_samples, label + ': sensor sample count mismatch')
    require(sum(totals[key] for key in ('tp', 'fp', 'fn', 'tn')) == n_samples * 64 * 64,
            label + ': pixel coverage differs from the fixed image budget')
    tp, fp, fn, tn = (totals[key] for key in ('tp', 'fp', 'fn', 'tn'))
    for key, numerator, denominator in (('iou', tp, tp + fp + fn), ('dice', 2 * tp, 2 * tp + fp + fn),
                                       ('accuracy', tp + tn, tp + fp + fn + tn)):
        expected = numerator / denominator if denominator else 0.0
        require(math.isclose(metrics[key], expected, abs_tol=1e-12), label + ': pooled sensor recombination mismatch ' + key)
    if test:
        require(metrics['optical_subset_n_samples'] == 71 and metrics['sar_subset_n_samples'] == 134,
                label + ': test sensor coverage differs from 71 Optical/134 SAR')


def fit_identity(row):
    return row['id'], row['seed'], row['learning_rate']


def expected_fit_settings(spec, row, settings):
    return canonical(dict(spec['config'], dataset='combined', data_root=row['settings']['data_root'],
        seed=row['seed'], image_size=settings['image_size'], max_samples=None,
        batch_size=settings['batch_size'], num_epochs=settings['epochs'],
        learning_rate=row['learning_rate'], optimization_fit_cap=settings['fit_cap'],
        _resolved_norm_provenance=row['normalization']))


def audit_fit_record(row, spec, settings, history_path):
    label = '/'.join(map(str, fit_identity(row)))
    require(row['model_type'] == spec['model_type'], label + ': model type mismatch')
    require(row['settings'] == expected_fit_settings(spec, row, settings), label + ': fit settings differ from frozen plan')
    require(len(row['fit_names']) == len(row['fit_indices']) == settings['fit_cap'], label + ': fit count mismatch')
    require(len(row['val_names']) == len(row['val_indices']) == 230 and row['n_global_fit'] == 1303,
            label + ': global fit/validation cohort mismatch')
    for key in ('fit_names', 'val_names', 'fit_indices', 'val_indices'):
        require(len(row[key]) == len(set(row[key])), label + ': duplicate ' + key)
    require(all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < 1533
                for key in ('fit_indices', 'val_indices') for index in row[key]), label + ': out-of-range sample index')
    require(not set(row['fit_names']) & set(row['val_names']) and
            not set(row['fit_indices']) & set(row['val_indices']), label + ': fit/validation overlap')
    require(row['normalization'].get('fit_only') is True and
            row['normalization'].get('n_fit_train_tiles') == 1303 and
            row['normalization'].get('n_excluded_val_tiles') == 230,
            label + ': normalization is not the global fit partition')
    require(sum(row['fit_sensor_counts'].values()) == settings['fit_cap'], label + ': fit sensor count mismatch')
    history = row['history']
    require(history == read(history_path), label + ': saved history differs from fit record')
    require([point['epoch'] for point in history] == list(range(settings['epochs'] + 1)), label + ': incomplete epoch history')
    expected_evals = set(range(0, settings['epochs'] + 1, settings['eval_every'])) | {settings['epochs']}
    require({point['epoch'] for point in history if 'val' in point} == expected_evals, label + ': validation schedule mismatch')
    for point in history:
        require((point['train_loss'] is None) == (point['epoch'] == 0), label + ': missing training loss')
        if point['epoch']:
            require(math.isfinite(point['train_loss']) and point['train_loss'] >= 0, label + ': invalid training loss')
        if 'val' in point:
            check_metrics(point['val'], 230, label + '/val/' + str(point['epoch']))
    best = max((point for point in history if 'val' in point), key=lambda point: point['val']['iou'])
    require(row['best_epoch'] == best['epoch'] and row['val'] == best['val'], label + ': selected validation checkpoint mismatch')
    require(math.isfinite(row['elapsed_s']) and row['elapsed_s'] > 0, label + ': invalid fit duration')
    for moment in ('initial_bottleneck_output', 'selected_bottleneck_output'):
        probes = row['diagnostics'][moment]
        require(len(probes) == 2, label + ': expected two uses of the shared bottleneck')
        for probe in probes:
            actual_angles = spec['config']['local_spatial_kernel'] in ('quantum', 'sincos')
            require(('input_angles' in probe) == actual_angles, label + ': angle diagnostics on a non-angle core or missing')
            if actual_angles:
                angles = probe['input_angles']
                require(angles['n_angles'] > 0 and 0 <= angles['saturation_fraction'] <= 1 and
                        0 <= angles['mean_abs_tanh'] <= angles['max_abs_tanh'] <= 1,
                        label + ': invalid input angle diagnostics')


def initial_parameter_digest(model):
    return hashlib.sha256(b''.join(name.encode() + parameter.detach().cpu().numpy().tobytes()
                                  for name, parameter in model.named_parameters())).hexdigest()


def shared_parameter_digest(model):
    from utilsTorch.local_spatial import LocalSpatialBottleneckBlock
    blocks = [block for block in model.modules() if isinstance(block, LocalSpatialBottleneckBlock)]
    require(len(blocks) == 1, 'Expected one shared local spatial core')
    core_ids = {id(parameter) for name, parameter in blocks[0].named_parameters()
                if not name.startswith(('pre_q_conv.', 'post_q_proj.'))}
    digest_value = hashlib.sha256(b''.join(name.encode() + parameter.detach().cpu().numpy().tobytes()
        for name, parameter in model.named_parameters() if id(parameter) not in core_ids)).hexdigest()
    return digest_value, sum(parameter.numel() for parameter in model.parameters() if id(parameter) in core_ids)


def audit_checkpoint(path, row):
    """Strict CPU reconstruction, including the producing initialization/movement."""
    import torch
    from tools import training_common as tc
    from utilsTorch.utilsTorch import QVUNetSegmentation, ClassicalAblationUNetSegmentation
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    require(checkpoint['settings'] == row['settings'] and checkpoint['model_type'] == row['model_type']
            and checkpoint['best_epoch'] == row['best_epoch'], 'Checkpoint metadata differ from fit record: ' + str(path))
    state = checkpoint['best_state_dict']
    require(isinstance(state, dict) and all(torch.is_tensor(value) and bool(torch.isfinite(value).all())
            for value in state.values()), 'Checkpoint contains invalid/nonfinite weights: ' + str(path))
    tc.seed_everything(row['seed'])
    cls = QVUNetSegmentation if row['model_type'] == 'quantum' else ClassicalAblationUNetSegmentation
    model = tc.build_model(cls, row['settings']).to('cpu')
    require(initial_parameter_digest(model) == row['initial_parameter_sha256'], 'Recorded initialization differs: ' + str(path))
    require(sum(p.numel() for p in model.parameters()) == row['total_params'] and
            sum(p.numel() for p in model.parameters() if p.requires_grad) == row['trainable_params'],
            'Parameter count or frozen flags differ: ' + str(path))
    shared_hash, core_count = shared_parameter_digest(model)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.load_state_dict(state)
    require(all(torch.equal(value, model.state_dict()[key]) for key, value in state.items()),
            'Checkpoint contains inconsistent aliases of shared parameters: ' + str(path))
    groups, _ = pilot.parameter_groups(model)
    for name, parameters in groups.items():
        if not parameters:
            continue
        squared = sum(float((p.detach() - initial[key]).square().sum()) for key, p in parameters)
        start_squared = sum(float(initial[key].square().sum()) for key, _ in parameters)
        movement = row['diagnostics']['movement'][name]
        require(math.isclose(movement['l2'], math.sqrt(squared), rel_tol=1e-9, abs_tol=1e-9),
                'Recorded parameter movement differs: ' + str(path) + '/' + name)
        expected_relative = math.sqrt(squared / start_squared) if start_squared else None
        require((movement['relative_l2'] is None and expected_relative is None) or
                (movement['relative_l2'] is not None and expected_relative is not None and
                 math.isclose(movement['relative_l2'], expected_relative, rel_tol=1e-9, abs_tol=1e-9)),
                'Recorded relative parameter movement differs: ' + str(path) + '/' + name)
    return dict(sha256=digest(path), strict_restore=True, finite_weights=True,
        initial_parameter_sha256=row['initial_parameter_sha256'], shared_initial_parameter_sha256=shared_hash,
        core_params=core_count, total_params=row['total_params'], trainable_params=row['trainable_params'])


def audit_phase(run_dir, epochs, expected_specs):
    run_dir = Path(run_dir)
    summary, plan = read(run_dir / 'summary.json'), read(run_dir / 'plan.json')
    require(summary['plan'] == plan, 'Summary plan differs from frozen plan')
    verify_source(plan)
    verify_runtime(plan)
    manifest = verify_manifest(plan)
    expected_budget = dict(seeds=list(SEEDS), learning_rates=list(RATES), epochs=epochs,
        eval_every=2, fit_cap=128, batch_size=16, image_size=64, device='cuda', specs=None)
    require(plan['settings'] == expected_budget, 'Study budget differs from the fixed protocol')
    require(plan['specs'] == canonical(expected_specs), 'Study model specifications differ from the protocol')
    require(plan['threads'] == 2 and plan['gpu_allocator_fraction'] == 0.12 and plan['device'] == 'cuda',
            'Recorded execution bounds differ from the fixed protocol')
    specs = {spec['id']: spec for spec in expected_specs}
    tuning_paths = sorted((run_dir / 'tuning').glob('*/*/seed42/metrics.json'))
    replication_paths = sorted((run_dir / 'replication').glob('*/seed*/metrics.json'))
    paths = tuning_paths + replication_paths
    require(len(paths) == len(specs) * 4 and len(tuning_paths) == len(specs) * 2,
            'Incomplete tuning/replication fit count')
    fits, by_identity, checkpoints, initial_shared = [], {}, {}, {}
    for path in paths:
        row = read(path)
        require('test' not in row and 'knockout' not in row, 'Fit record contains test evaluation before replay boundary')
        require(row['id'] in specs and fit_identity(row) not in by_identity, 'Unexpected or duplicate fit record')
        audit_fit_record(row, specs[row['id']], plan['settings'], path.parent / 'history.json')
        require(row['fit_names'] == [manifest['train'][index] for index in row['fit_indices']] and
                row['val_names'] == [manifest['train'][index] for index in row['val_indices']],
                'Fit/validation indices disagree with their bound manifest names')
        global_fit = [name for index, name in enumerate(manifest['train']) if index not in set(row['val_indices'])]
        require(hashlib.sha256('\n'.join(global_fit).encode()).hexdigest()[:16] == row['normalization']['fit_names_sha256'],
                'Normalization fit-name digest differs from the global fit cohort')
        identity = fit_identity(row)
        expected_directory = (Path('tuning') / row['id'] / str(row['learning_rate']) / 'seed42'
                              if row['seed'] == 42 else Path('replication') / row['id'] / ('seed' + str(row['seed'])))
        require(path.parent.relative_to(run_dir) == expected_directory, 'Fit path disagrees with model/seed/rate metadata')
        by_identity[identity] = row
        checkpoint = path.parent / 'checkpoint.pt'
        details = audit_checkpoint(checkpoint, row)
        relative = str(checkpoint.relative_to(run_dir))
        checkpoints[relative] = details
        shared = details['shared_initial_parameter_sha256']
        require(row['seed'] not in initial_shared or initial_shared[row['seed']] == shared,
                'Backbone/projection initialization differs between study configurations')
        initial_shared[row['seed']] = shared
        fits.append(dict(directory=str(path.parent.relative_to(run_dir)), record=row,
                         checkpoint=relative, checkpoint_audit=details))
    trials = [read(path) for path in tuning_paths]
    require({fit_identity(row) for row in trials} == {(name, 42, rate) for name in specs for rate in RATES},
            'Tuning grid differs from planned rates/models')
    saved_trials = read(run_dir / 'tuning_candidates.json')
    require(sorted(saved_trials, key=fit_identity) == sorted(trials, key=fit_identity), 'Saved tuning candidate table differs from fit records')
    chosen = pilot.choose_rates(trials)
    selection = {name: row['learning_rate'] for name, row in chosen.items()}
    require(summary['selection'] == selection, 'Summary rates are not validation-selected')
    require(read(run_dir / 'validation_selection.json') == {name: dict(learning_rate=row['learning_rate'],
            val_iou=row['val']['iou'], best_epoch=row['best_epoch']) for name, row in chosen.items()},
            'Saved validation selection differs from candidate scores')
    expected_fits = {(name, 42, rate) for name in specs for rate in RATES}
    expected_fits |= {(name, seed, selection[name]) for name in specs for seed in SEEDS[1:]}
    require(set(by_identity) == expected_fits, 'Fit grid differs from complete tuning/selected-rate replication')
    require(set(run_dir.rglob('metrics.json')) == set(paths) and
            {path.parent for path in run_dir.rglob('checkpoint.pt')} == {path.parent for path in paths} and
            {path.parent for path in run_dir.rglob('history.json')} == {path.parent for path in paths},
            'Unplanned or incomplete fit artifacts remain outside the declared grid')
    cohort_by_seed = {}
    init_by_model_seed = {}
    for row in by_identity.values():
        cohort = {key: row[key] for key in COHORT_KEYS}
        require(row['seed'] not in cohort_by_seed or cohort_by_seed[row['seed']] == cohort, 'Unpaired fit cohorts/normalization')
        cohort_by_seed[row['seed']] = cohort
        key = row['id'], row['seed']
        require(key not in init_by_model_seed or init_by_model_seed[key] == row['initial_parameter_sha256'],
                'Learning-rate candidates have different initialization')
        init_by_model_seed[key] = row['initial_parameter_sha256']
    records = summary['records']
    require(len(records) == len(specs) * 3 and {(row['id'], row['seed']) for row in records} ==
            {(name, seed) for name in specs for seed in SEEDS}, 'Incomplete selected model/seed coverage')
    for row in records:
        identity = fit_identity(row)
        require(row['learning_rate'] == selection[row['id']], 'Selected replication rate differs from validation decision')
        fit_only = {key: value for key, value in row.items() if key not in ('test', 'test_names', 'checkpoint', 'knockout')}
        require(fit_only == by_identity[identity], 'Selected summary differs from validation-only fit record')
        checkpoint = safe_relative(run_dir, row['checkpoint'])
        require(str(checkpoint.relative_to(run_dir.resolve())) in checkpoints, 'Selected checkpoint is outside the audited fit grid')
        expected_checkpoint = next(item['checkpoint'] for item in fits if fit_identity(item['record']) == identity)
        require(row['checkpoint'] == expected_checkpoint, 'Selected checkpoint path disagrees with its fit identity')
        require(read(checkpoint.parent / 'selected_test.json') == row, 'Saved selected replay differs from summary')
        # Basenames can repeat across Training/Testing directories. Raw identity
        # belongs to the bound deduplicated manifest, not basename disjointness.
        require(len(row['test_names']) == len(set(row['test_names'])) == 205, 'Test cohort is incomplete or duplicated')
        require(row['test_names'] == manifest['test'], 'Test names/order differ from the bound full test manifest')
        check_metrics(row['test'], 205, row['id'] + '/test', test=True)
        require(set(row['knockout']) == {'zero', 'random'}, 'Missing planned output knockouts')
        for mode in ('zero', 'random'):
            check_metrics(row['knockout'][mode], 205, row['id'] + '/' + mode, test=True)
    require(summary['analysis'] == pilot.aggregate(records), 'Saved aggregate differs from selected records')
    comparisons = study.paired_analysis(records)
    require(read(run_dir / 'paired_spatial_analysis.json') == comparisons, 'Saved paired spatial analysis differs from recomputation')
    fixed = study.fixed_rate_readout_contrasts(trials, require_both=epochs == 60)
    require(read(run_dir / 'fixed_rate_readout_analysis.json') == fixed, 'Saved fixed-rate readout analysis differs from recomputation')
    return dict(summary=summary, fits=sorted(fits, key=lambda item: fit_identity(item['record'])),
        tuning=sorted(trials, key=fit_identity), checkpoints=checkpoints, comparisons=comparisons,
        fixed_rate_readout=fixed, shared_initial=initial_shared,
        summary_sha256=digest(run_dir / 'summary.json'), plan_sha256=digest(run_dir / 'plan.json'))


def audit_study(pilot_dir, extension_dir):
    short = audit_phase(pilot_dir, 60, study.study_specs())
    require(short['summary']['plan']['experiment_context'] == dict(phase='pilot', protocol='local-spatial-seven-spec-v1'),
            'Pilot context differs from the fixed protocol')
    finalists = study.finalist_specs(short['summary'])
    long = audit_phase(extension_dir, 120, finalists)
    expected_context = dict(phase='extend', protocol='local-spatial-two-finalists-v1',
        baseline_summary_sha256=short['summary_sha256'],
        selection_policy='Mean paired validation IoU; Q tie Z; classical tie smaller core then MLP/conv/sincos')
    require(long['summary']['plan']['experiment_context'] == expected_context, 'Extension references another baseline or selection policy')
    first_plan, second_plan = short['summary']['plan'], long['summary']['plan']
    for key in ('input_identity', 'source_hashes', 'normalization_policy', 'test_policy', *RUNTIME_KEYS):
        require(first_plan[key] == second_plan[key], 'Duration phases differ in source/input/runtime: ' + key)
    short_by = {(row['id'], row['seed']): row for row in short['summary']['records']}
    for row in long['summary']['records']:
        old = short_by[row['id'], row['seed']]
        for key in (*COHORT_KEYS, 'test_names', 'initial_parameter_sha256', 'total_params', 'trainable_params'):
            require(old[key] == row[key], 'Duration phases differ beyond epoch budget/rate selection: ' + key)
    return dict(pilot60=short, extend120=long, finalists=[spec['id'] for spec in finalists],
        duration_evidence=duration_evidence(short, long),
        scope='Exploratory same-data duration/readout study; validation-led selection; descriptive test evidence, not untouched-holdout confirmation')


def stats(values):
    values = list(values)
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else None, n=len(values))


def duration_evidence(short, long):
    """Separate selected-rate duration deltas from actual same-rate prefixes."""
    short_fits = {fit_identity(item['record']): item['record'] for item in short['fits']}
    prefixes = []
    for item in long['fits']:
        row = item['record']
        previous = short_fits.get(fit_identity(row))
        entry = dict(id=row['id'], seed=row['seed'], learning_rate=row['learning_rate'], comparable=previous is not None)
        if previous is None:
            entry['reason'] = 'This model/seed/rate was not run in the 60-epoch grid; replication retained only its then-selected rate'
        else:
            old_cfg = {key: value for key, value in previous['settings'].items() if key != 'num_epochs'}
            new_cfg = {key: value for key, value in row['settings'].items() if key != 'num_epochs'}
            require(old_cfg == new_cfg and previous['initial_parameter_sha256'] == row['initial_parameter_sha256'] and
                    all(previous[key] == row[key] for key in COHORT_KEYS), 'Matched-rate duration prefix has different inputs/initialization')
            prefix = row['history'][:61]
            require(len(previous['history']) == len(prefix) == 61, 'Incomplete first-60-epoch prefix')
            entry.update(exact_history_match=prefix == previous['history'],
                max_abs_train_loss_delta=max(abs(before['train_loss'] - after['train_loss'])
                    for before, after in zip(previous['history'][1:], prefix[1:])),
                max_abs_validation_iou_delta=max(abs(before['val']['iou'] - after['val']['iou'])
                    for before, after in zip(previous['history'], prefix) if 'val' in before))
        prefixes.append(entry)
    old_records = {(row['id'], row['seed']): row for row in short['summary']['records']}
    contrasts = []
    for name in long['summary']['selection']:
        old_rate, new_rate = short['summary']['selection'][name], long['summary']['selection'][name]
        records = [row for row in long['summary']['records'] if row['id'] == name]
        for split in ('val', 'test'):
            differences = {str(row['seed']): 100 * (row[split]['iou'] - old_records[name, row['seed']][split]['iou']) for row in records}
            contrasts.append(dict(id=name, split=split, rate60=old_rate, rate120=new_rate,
                same_learning_rate=old_rate == new_rate, differences=differences, **stats(differences.values()),
                unit='IoU percentage points', qualification=('Same selected rate; fresh fits share seeds/data, not independent replication'
                    if old_rate == new_rate else 'Epoch budget and validation-selected learning rate both changed; not an epoch-only contrast')))
    return dict(prefixes=prefixes, selected_rate_contrasts=contrasts,
        prefix_policy='Only actual matching model/seed/rate fits supply first-60-epoch prefixes; unmatched replication rates are explicitly unmeasured')


def summary_tables(result):
    rows, per_seed, tuning_rows = [], [], []
    for phase in ('pilot60', 'extend120'):
        data = result[phase]
        groups = {}
        for record in data['summary']['records']:
            groups.setdefault(record['id'], []).append(record)
            per_seed.append(dict(phase=phase, id=record['id'], seed=record['seed'],
                learning_rate=record['learning_rate'], total_params=record['total_params'], trainable_params=record['trainable_params'],
                best_epoch=record['best_epoch'], val_iou=record['val']['iou'], test_iou=record['test']['iou'],
                optical_test_iou=record['test']['optical_subset_iou'], sar_test_iou=record['test']['sar_subset_iou'],
                on_minus_zero_pp=100 * (record['test']['iou'] - record['knockout']['zero']['iou']),
                on_minus_random_pp=100 * (record['test']['iou'] - record['knockout']['random']['iou'])))
        for name, records in groups.items():
            require(len({row['total_params'] for row in records}) == len({row['trainable_params'] for row in records}) == 1,
                    'Parameter count differs across seeds')
            selected_rate = data['summary']['selection'][name]
            matching = next(item for item in data['fits'] if item['record']['id'] == name)
            rows.append(dict(phase=phase, id=name, total_params=records[0]['total_params'],
                trainable_params=records[0]['trainable_params'], core_params=matching['checkpoint_audit']['core_params'],
                selected_rate=selected_rate, val_iou=stats(row['val']['iou'] for row in records),
                test_iou=stats(row['test']['iou'] for row in records),
                optical_test_iou=stats(row['test']['optical_subset_iou'] for row in records),
                sar_test_iou=stats(row['test']['sar_subset_iou'] for row in records),
                on_minus_zero_pp=stats(100 * (row['test']['iou'] - row['knockout']['zero']['iou']) for row in records),
                on_minus_random_pp=stats(100 * (row['test']['iou'] - row['knockout']['random']['iou']) for row in records),
                late_selected=sum(row['best_epoch'] >= data['summary']['plan']['settings']['epochs'] - 2 for row in records)))
        for row in data['tuning']:
            tuning_rows.append(dict(phase=phase, id=row['id'], seed=42, learning_rate=row['learning_rate'],
                selected_rate=row['learning_rate'] == data['summary']['selection'][row['id']],
                best_epoch=row['best_epoch'], val_iou=row['val']['iou'], total_params=row['total_params']))
    return rows, per_seed, tuning_rows


def report(result, rows):
    lines = ['# Local spatial quantum study', '', result['scope'] + '.', '',
        'Seven spatially matched configurations received two-rate tuning on seed 42 and selected-rate replication on seeds 77/123 (28 fits).',
        'The validation-selected quantum readout and strongest validation-selected classical control then received the same fresh 120-epoch protocol (8 fits).',
        'All 36 checkpoints, including losing-rate candidates, were strictly restored and checked for finite weights, exact saved settings, initial parameter identity and parameter movement.', '',
        'All runs use 64-pixel inputs, 128 optimization tiles, 230 validation tiles and 205 test tiles (71 Optical/134 SAR).',
        'Normalization uses all 1,303 global fit tiles while excluding validation; this is not pure few-image learning.',
        'Readout/gate comparisons at independently selected rates include tuning differences. The two common-rate readout rows use only one tuning seed and have no across-seed uncertainty estimate.', '',
        '| Phase | Model | Total / trainable / core parameters | LR | Validation IoU mean ± SD | Test IoU mean ± SD | Optical IoU | SAR IoU | Late selections |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['phase']} | {row['id']} | {row['total_params']:,} / {row['trainable_params']:,} / {row['core_params']} | {row['selected_rate']:g} | "
            f"{row['val_iou']['mean']:.4f} ± {row['val_iou']['std']:.4f} | {row['test_iou']['mean']:.4f} ± {row['test_iou']['std']:.4f} | "
            f"{row['optical_test_iou']['mean']:.4f} | {row['sar_test_iou']['mean']:.4f} | {row['late_selected']}/3 |")
    lines += ['', '| Phase | Paired contrast | Split | Mean ± sample SD (IoU percentage points) | Wins | Same selected rates |',
        '| --- | --- | --- | ---: | ---: | --- |']
    for phase in ('pilot60', 'extend120'):
        for row in result[phase]['comparisons']:
            lines.append(f"| {phase} | {row['left']} − {row['right']} | {row['split']} | {row['mean']:+.3f} ± {row['std']:.3f} | {row['wins']}/3 | {row['same_learning_rates']} |")
    lines += ['', '| Fixed rate | Seed | Rich Z/X/ZZ minus Z-only validation IoU (percentage points) |', '| ---: | ---: | ---: |']
    for row in result['pilot60']['fixed_rate_readout']:
        lines.append(f"| {row['learning_rate']:g} | 42 (n=1) | {row['difference_pp']:+.3f} |")
    lines += ['', 'Learning rates were retuned for the 120-epoch extension. A changed selected rate makes the following a joint duration/tuning contrast, not an epoch-only effect.', '',
        '| Model | Split | LR at 60 / 120 epochs | Same rate | 120 − 60 IoU points, mean ± sample SD |', '| --- | --- | ---: | --- | ---: |']
    for row in result['duration_evidence']['selected_rate_contrasts']:
        lines.append(f"| {row['id']} | {row['split']} | {row['rate60']:g} / {row['rate120']:g} | {row['same_learning_rate']} | {row['mean']:+.3f} ± {row['std']:.3f} |")
    prefixes = result['duration_evidence']['prefixes']
    comparable = [row for row in prefixes if row['comparable']]
    lines += ['', f"Of the eight long fits, {len(comparable)} have an actually run 60-epoch prefix at the same model/seed/rate; "
              f"{len(prefixes) - len(comparable)} have no comparable short replication rate. "
              f"Exact first-60-epoch histories match in {sum(row['exact_history_match'] for row in comparable)} of the {len(comparable)} comparable cases.",
              'The [prefix inventory](duration_evidence.json) retains every comparable/unmeasured case and the measured maximum loss/validation-IoU differences.']
    lines += ['', '| Phase | Model | On − zero IoU points, mean ± SD | On − random IoU points, mean ± SD |', '| --- | --- | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['phase']} | {row['id']} | {row['on_minus_zero_pp']['mean']:+.3f} ± {row['on_minus_zero_pp']['std']:.3f} | {row['on_minus_random_pp']['mean']:+.3f} ± {row['on_minus_random_pp']['std']:.3f} |")
    lines += ['', 'Zero knockout removes the entire projected block output. Random knockout changes core weights while retaining trained projections and input dependence.',
        'Z-only pads eight readout slots with zero; its equal stored projection count does not give those inactive columns capacity.',
        'Separable ZZ readouts retain classical product features. Larger classical core capacities are reported rather than artificially reduced.',
        'The 120-epoch fits reuse the same seeds/data and are a duration check, not independent replication. Test scores did not choose finalists or learning rates.',
        'Fit times are retained for provenance, not treated as controlled throughput measurements. These are local classical-simulator results; Alan owns Anvil execution.', '',
        'Plots: [paired outcomes](plots/paired_quantum_classical.png), [validation curves](plots/validation_curves.png), [input-angle saturation](plots/input_angle_saturation.png).', '']
    return '\n'.join(lines)


def plot_results(result, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = out_dir / 'plots'
    plot_dir.mkdir()
    colors = dict(zip(SEEDS, ('#0072B2', '#D55E00', '#009E73')))
    labels = dict(local_quantum_z='Q: Z', local_quantum_xz_zz='Q: Z/X/ZZ', local_quantum_frozen='Q: frozen',
                  local_quantum_no_entanglement='Q: separable', local_mlp='MLP', local_conv='Convolution', local_sincos='Sine/cosine')
    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(plot_dir / (name + '.' + suffix), dpi=150, bbox_inches='tight')
        plt.close(fig)
    quantum, classical = result['finalists']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for index, phase in enumerate(('pilot60', 'extend120')):
        selected = {(row['id'], row['seed']): row for row in result[phase]['summary']['records']}
        for column, split in enumerate(('val', 'test')):
            axis = axes[index, column]
            for seed in SEEDS:
                axis.plot([0, 1], [selected[name, seed][split]['iou'] for name in (classical, quantum)],
                          'o-', color=colors[seed], label='seed ' + str(seed))
            rates = result[phase]['summary']['selection']
            axis.set(xticks=[0, 1], xticklabels=[labels[classical] + '\nLR ' + str(rates[classical]), labels[quantum] + '\nLR ' + str(rates[quantum])],
                     ylabel='IoU', title=phase + ': ' + split + ' (selected rates; paired seeds)')
            axis.legend(fontsize=8)
    save(fig, 'paired_quantum_classical')
    ids = [spec['id'] for spec in study.study_specs()]
    fig, axes = plt.subplots(7, 3, figsize=(13, 18), squeeze=False)
    for index, name in enumerate(ids):
        for column, seed in enumerate(SEEDS):
            axis = axes[index, column]
            for phase, style in (('pilot60', '-'), ('extend120', '--')):
                records = [row for row in result[phase]['summary']['records'] if row['id'] == name and row['seed'] == seed]
                for row in records:
                    points = [point for point in row['history'] if 'val' in point]
                    axis.plot([point['epoch'] for point in points], [point['val']['iou'] for point in points], style,
                              label=phase + ', LR ' + str(row['learning_rate']))
                    axis.scatter([row['best_epoch']], [row['val']['iou']], s=15)
            axis.set(title=labels[name] + ', seed ' + str(seed), xlabel='Epoch', ylabel='Validation IoU')
            axis.legend(fontsize=6)
    save(fig, 'validation_curves')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    for column, phase in enumerate(('pilot60', 'extend120')):
        axis = axes[0, column]
        names = [name for name in ids if any(row['id'] == name and row['diagnostics']['initial_bottleneck_output'][0].get('input_angles')
                 for row in result[phase]['summary']['records'])]
        for index, name in enumerate(names):
            for seed in SEEDS:
                row = next(row for row in result[phase]['summary']['records'] if row['id'] == name and row['seed'] == seed)
                values = []
                for moment in ('initial_bottleneck_output', 'selected_bottleneck_output'):
                    probes = [probe['input_angles'] for probe in row['diagnostics'][moment]]
                    values.append(sum(probe['n_angles'] * probe['saturation_fraction'] for probe in probes) / sum(probe['n_angles'] for probe in probes))
                offset = (SEEDS.index(seed) - 1) * 0.08
                axis.plot([index - .16 + offset, index + .16 + offset], values, 'o-', color=colors[seed],
                          label='seed ' + str(seed) if index == 0 else None, markersize=3)
        axis.set(xticks=range(len(names)), xticklabels=[labels[name] for name in names],
                 ylabel='Fraction |tanh(scale × input)| ≥ 0.95', title=phase + ': initial → selected', ylim=(-.02, 1.02))
        axis.tick_params(axis='x', rotation=30)
        axis.legend(fontsize=8)
    save(fig, 'input_angle_saturation')


def validate_angle_audit(path, result):
    audit = read(path)
    require(audit['diagnostic_source_sha256'] == digest(ROOT / 'scripts/audit_local_spatial_angles.py') and
            audit['collector_source_sha256'] == digest(__file__), 'Individual-angle executable provenance differs')
    require(audit['summary_sha256'] == {phase: result[phase]['summary_sha256'] for phase in ('pilot60', 'extend120')},
            'Angle audit references another set of summaries')
    require(audit['source_hashes'] == result['pilot60']['summary']['plan']['source_hashes'] and
            audit['input_identity'] == result['pilot60']['summary']['plan']['input_identity'], 'Angle audit source/input identity differs')
    require(audit['runtime'] == dict(**current_runtime(), device='cpu', threads=2), 'Angle audit execution runtime differs')
    expected = {(phase, row['id'], row['seed']): row for phase in ('pilot60', 'extend120')
                for row in result[phase]['summary']['records'] if row['model_type'] == 'quantum'}
    require(len(audit['records']) == len(expected), 'Incomplete individual-angle audit')
    actual = {(row['phase'], row['id'], row['seed']): row for row in audit['records']}
    require(set(actual) == set(expected), 'Individual-angle audit has duplicate/missing model/seed records')
    probe_hashes = {}
    for key, row in actual.items():
        selected = expected[key]
        require(row['probe_names'] == selected['fit_names'][:8] and row['initial_parameter_sha256'] == selected['initial_parameter_sha256'],
                'Angle audit does not use the fixed optimization probe/initialization')
        details = result[key[0]]['checkpoints'][selected['checkpoint']]
        require(row['checkpoint_sha256'] == details['sha256'], 'Angle audit references different checkpoint bytes')
        probe_hash = row['probe_tensor_sha256']
        require(isinstance(probe_hash, str) and len(probe_hash) == 64 and
                all(character in '0123456789abcdef' for character in probe_hash), 'Invalid optimization-probe tensor hash')
        require(row['seed'] not in probe_hashes or probe_hashes[row['seed']] == probe_hash,
                'Optimization-probe tensor bytes differ across paired models/phases')
        probe_hashes[row['seed']] = probe_hash
        require(set(row['parameter_diagnostics']) == {'quantum_params.0', 'input_scales.0'}, 'Incomplete individual parameter diagnostics')
        for name, diagnostic in row['parameter_diagnostics'].items():
            expected_count = 12 if name == 'quantum_params.0' else 4
            def flatten(values):
                if isinstance(values, list):
                    return [leaf for value in values for leaf in flatten(value)]
                return [values]
            for moment in ('initial', 'selected', 'movement'):
                values = flatten(diagnostic[moment])
                require(len(values) == expected_count and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values),
                        'Invalid per-angle values: ' + moment)
            start, end, movement = (flatten(diagnostic[moment]) for moment in ('initial', 'selected', 'movement'))
            require(all(math.isclose(after - before, delta, abs_tol=1e-7, rel_tol=1e-6)
                        for before, after, delta in zip(start, end, movement)), 'Per-angle movement differs from initial/selected values')
            require(math.isclose(diagnostic['movement_l2'], math.sqrt(sum(value * value for value in movement)),
                                 abs_tol=1e-7, rel_tol=1e-6), 'Individual-angle movement norm differs')
            frozen = selected['settings']['q_training_control'] == 'frozen'
            require(diagnostic['requires_grad'] == (not frozen) and (diagnostic['gradient'] is None) == frozen,
                    'Individual-angle frozen/gradient flags differ from checkpoint semantics')
            if frozen:
                require(diagnostic['movement_l2'] == 0 and not any(flatten(diagnostic['movement'])), 'Frozen individual angle moved')
                require(diagnostic['gradient_nonzero_exact'] is None and diagnostic['gradient_above_threshold'] is None,
                        'Frozen gradient masks must remain absent')
            else:
                gradient = flatten(diagnostic['gradient'])
                require(len(gradient) == expected_count and all(math.isfinite(value) for value in gradient), 'Invalid per-angle gradient')
                require(flatten(diagnostic['gradient_nonzero_exact']) == [value != 0 for value in gradient] and
                        flatten(diagnostic['gradient_above_threshold']) == [abs(value) > diagnostic['threshold'] for value in gradient],
                        'Per-angle gradient masks differ from retained derivative values')
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--extension-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--angle-audit', type=Path)
    args = parser.parse_args(argv)
    require(not args.out_dir.exists() or not any(args.out_dir.iterdir()), 'Use a new or empty output directory')
    for run_dir in (args.pilot_dir, args.extension_dir):
        require(not args.out_dir.resolve().is_relative_to(run_dir.resolve()) and not run_dir.resolve().is_relative_to(args.out_dir.resolve()),
                'Output directory aliases or contains input run artifacts')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    result = audit_study(args.pilot_dir, args.extension_dir)
    rows, per_seed, tuning = summary_tables(result)
    optional_angles = validate_angle_audit(args.angle_audit, result) if args.angle_audit else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for phase in ('pilot60', 'extend120'):
        directory = args.out_dir / phase
        directory.mkdir()
        data = result[phase]
        for filename, value in (('summary.json', data['summary']), ('plan.json', data['summary']['plan']),
                ('all_fits.json', data['fits']), ('tuning_candidates.json', data['tuning']),
                ('paired_analysis.json', data['comparisons']), ('fixed_rate_readout.json', data['fixed_rate_readout']),
                ('checkpoint_audit.json', data['checkpoints'])):
            write(directory / filename, value)
    write(args.out_dir / 'comparison.json', dict(scope=result['scope'], finalists=result['finalists'], rows=rows,
        comparisons={phase: result[phase]['comparisons'] for phase in ('pilot60', 'extend120')},
        fixed_rate_readout=result['pilot60']['fixed_rate_readout'], duration_evidence=result['duration_evidence']))
    write(args.out_dir / 'duration_evidence.json', result['duration_evidence'])
    summary_csv = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for name, value in row.items():
            if isinstance(value, dict):
                flat.update({name + '_' + statistic: number for statistic, number in value.items()})
        summary_csv.append(flat)
    for filename, records in (('summary.csv', summary_csv), ('per_seed.csv', per_seed), ('tuning.csv', tuning)):
        with (args.out_dir / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)
    text = report(result, rows)
    if optional_angles is not None:
        text += ('\n[Individual angle/scaling diagnostics](individual_angle_audit.json) retain each coefficient, selected-minus-initial movement '
                 'and a Dice+CE derivative on the fixed first-eight optimization tiles. These are local probe derivatives, '
                 'not accumulated training gradients or proof of parameter usefulness. Frozen gradients remain absent.\n')
    (args.out_dir / 'README.md').write_text(text)
    if optional_angles is not None:
        write(args.out_dir / 'individual_angle_audit.json', optional_angles)
    plot_results(result, args.out_dir)
    provenance = dict(collector_source_sha256=digest(__file__),
        raw_runs=dict(pilot60=str(args.pilot_dir.resolve()), extend120=str(args.extension_dir.resolve())),
        producing_revisions={phase: result[phase]['summary']['plan']['source_revision'] for phase in ('pilot60', 'extend120')},
        source_hashes=result['pilot60']['summary']['plan']['source_hashes'], input_identity=result['pilot60']['summary']['plan']['input_identity'],
        runtime={key: result['pilot60']['summary']['plan'][key] for key in RUNTIME_KEYS},
        collector_runtime=dict(**current_runtime(), device='cpu', threads=2),
        input_summary_sha256={phase: result[phase]['summary_sha256'] for phase in ('pilot60', 'extend120')},
        input_plan_sha256={phase: result[phase]['plan_sha256'] for phase in ('pilot60', 'extend120')},
        checked_fit_count=36, checked_selected_replays=27,
        angle_audit_sha256=digest(args.angle_audit) if args.angle_audit else None,
        checks=['complete 28+8 fit grids', 'all checkpoint strict restoration and finite weights',
                'exact producing source and runtime/data identity', 'full curves and first strict validation maximum',
                'validation-only rates and finalists', 'paired initialization/cohorts/normalization',
                'confusion-count recombination', 'saved summary/replay/analysis equality'])
    write(args.out_dir / 'provenance.json', provenance)
    hashes = {str(path.relative_to(args.out_dir)): digest(path) for path in sorted(args.out_dir.rglob('*')) if path.is_file()}
    write(args.out_dir / 'artifact_sha256.json', hashes)
    print('Verified 36 fit checkpoints, 27 selected replays; artifacts:', args.out_dir)
    return result


if __name__ == '__main__':
    main()
