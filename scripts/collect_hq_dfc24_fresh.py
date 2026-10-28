#!/usr/bin/env python3
"""Audit a complete fresh HQ campaign without fitting or repeating inference.

Rehearsals validate software on validation images only. Scientific decision
screens are defined exclusively for a complete, full-mode 16-fit campaign.
"""
import argparse
from collections import Counter
from fractions import Fraction
import csv
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shlex
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_hq_dfc24_fresh as study
from scripts import run_hq_dfc24_direction as hq

require, digest, read = study.require, study.digest, study.read_json
CONTROLS, SEEDS, RATES = study.CONTROLS, study.SEEDS, study.RATES
METRICS = ('flood_iou', 'background_iou', 'binary_macro_iou', 'flood_dice', 'accuracy')
SCOPES = ('pooled', 'optical', 'sar')
COLLECTOR = 'scripts/collect_hq_dfc24_fresh.py'
LOADED_COLLECTOR_SHA256 = digest(__file__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inside(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Expected relative artifact path')
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Artifact path escapes its evidence root')
    return path


def inventory(root):
    root = Path(root)
    return {str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob('*')) if path.is_file()}


def identity(row):
    require(row['kind'] in CONTROLS and type(row['seed']) is int and row['seed'] in SEEDS and
            type(row['learning_rate']) is float and row['learning_rate'] in RATES,
            'Invalid typed fit identity')
    require(row['id'] == 'hq_' + row['kind'], 'Fit label/control differs')
    return row['kind'], row['seed'], row['learning_rate']


def current_runtime():
    import torch
    return dict(python=platform.python_version(), torch_cuda=torch.version.cuda,
        packages={name: importlib.metadata.version(name) for name in
            ('torch', 'pennylane', 'numpy', 'rasterio', 'scikit-learn', 'scipy', 'Pillow')})


def verify_source(plan):
    require(set(plan['source_hashes']) == set(study.SOURCE_FILES), 'Producer source inventory differs')
    require(COLLECTOR not in plan['source_hashes'], 'Collector incorrectly bound as a fitting dependency')
    for name, expected in plan['source_hashes'].items():
        require(digest(inside(ROOT, name)) == expected, 'Producer executable source differs: ' + name)
    require(digest(ROOT / study.PROTOCOL) == plan['protocol_sha256'], 'Producer protocol differs')


def verify_collection_source():
    require(digest(__file__) == LOADED_COLLECTOR_SHA256 and
            digest(ROOT/COLLECTOR) == LOADED_COLLECTOR_SHA256,
            'Collector source changed after import or differs from its retained snapshot')


def verify_runtime(plan):
    actual, producer = current_runtime(), plan['runtime']
    for key in ('python', 'packages', 'torch_cuda'):
        require(actual[key] == producer[key], 'Collection runtime compatibility differs: ' + key)
    require(producer['threads'] == producer['interop_threads'] == 2, 'Producer numerical thread bounds differ')
    require(producer['device'] in ('cpu', 'cuda:0'), 'Unsupported producing device')
    if producer['device'] == 'cuda:0':
        require(bool(producer['gpu']) and type(producer['gpu_allocator_fraction']) in (float, int) and
                not isinstance(producer['gpu_allocator_fraction'], bool) and
                0 < producer['gpu_allocator_fraction'] <= 1, 'Invalid producing GPU allocator identity')
    else:
        require(producer['gpu_allocator_fraction'] is None, 'CPU producer has a GPU allocator fraction')
    # GPU model and interpreter path may differ during CPU restoration. Exact
    # initial-state reconstruction and bound Python/package/build versions remain gates.
    return actual


def derived_metrics(counts):
    require(set(counts) == {'tp', 'fp', 'fn', 'tn'} and
            all(type(value) is int and value >= 0 for value in counts.values()),
            'Invalid integer confusion counts')
    tp, fp, fn, tn = (counts[key] for key in ('tp', 'fp', 'fn', 'tn'))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else 0.
    flood, background = ratio(tp, tp+fp+fn), ratio(tn, tn+fp+fn)
    return dict(flood_iou=flood, background_iou=background,
        binary_macro_iou=(flood+background)/2, flood_dice=ratio(2*tp, 2*tp+fp+fn),
        accuracy=ratio(tp+tn, tp+fp+fn+tn))


def check_metrics(metrics, n_samples, *, full_test=False):
    pooled = dict(tp=0, fp=0, fn=0, tn=0)
    results, samples = {}, 0
    for scope in ('optical', 'sar'):
        prefix = scope + '_subset_'
        n = metrics[prefix+'n_samples']
        require(type(n) is int and n >= 0, 'Invalid sensor sample count')
        samples += n
        counts = {key: metrics[prefix+key] for key in pooled}
        derived = derived_metrics(counts)
        require(sum(counts.values()) == n*128*128, 'Sensor pixel coverage differs')
        for key in pooled:
            pooled[key] += counts[key]
        for key, name in (('iou', 'flood_iou'), ('dice', 'flood_dice'), ('accuracy', 'accuracy')):
            require(math.isclose(metrics[prefix+key], derived[name], rel_tol=0, abs_tol=2e-15),
                    'Saved sensor/count metric mismatch: ' + prefix + key)
        require(0 <= metrics[prefix+'auc'] <= 1, 'Invalid saved sensor AUC')
        results[scope] = dict(counts=counts, n_samples=n, metrics=derived, saved_auc=metrics[prefix+'auc'])
    require(samples == n_samples, 'Evaluation sample coverage differs')
    derived = derived_metrics(pooled)
    for key, name in (('iou', 'flood_iou'), ('dice', 'flood_dice'), ('accuracy', 'accuracy')):
        require(math.isclose(metrics[key], derived[name], rel_tol=0, abs_tol=2e-15),
                'Saved pooled/count metric mismatch: ' + key)
    require(metrics['loss'] >= 0 and 0 <= metrics['auc'] <= 1, 'Invalid saved loss/AUC')
    if full_test:
        require((results['optical']['n_samples'], results['sar']['n_samples']) == (71, 134),
                'Full test replay must cover 71 Optical and 134 SAR tiles')
    results['pooled'] = dict(counts=pooled, n_samples=n_samples, metrics=derived, saved_auc=metrics['auc'])
    return results


def checkpoint_audit(path, row):
    from utilsTorch.hq_training_controls import build_model, parameter_counts
    model, audit = study.restore_checkpoint(path, row, device='cpu')
    initial = build_model(row['kind'], row['seed'], base_channels=8, image_size=128)
    counts = parameter_counts(initial)
    require(hq.state_hash(initial.state_dict()) == row['initial_state_sha256'] and
            hq.shared_hash(initial) == row['initial_shared_state_sha256'], 'Initial model/BN identity differs')
    expected_flags = {name: parameter.requires_grad for name, parameter in initial.named_parameters()}
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    require(flags == expected_flags, 'Restored trainability differs')
    require(parameter_counts(model) == counts and row['total_params'] == counts['total'] and
            row['trainable_params'] == counts['trainable'] and row['core_params'] == counts['core_total'],
            'Stored/trainable/core parameter counts differ')
    require(model.bottleneck.mode == ('zero_bottleneck' if row['kind'] == 'zero_bottleneck' else 'on'),
            'Restored control mode differs')
    require(hq.state_hash(model.state_dict()) == row['selected_state_sha256'], 'Selected weights/BN state differs')
    initial_parameters = {name: parameter.detach().clone() for name, parameter in initial.named_parameters()}
    require(hq.movement(model, initial_parameters) == row['diagnostics']['movement'],
            'Saved parameter movement differs from checkpoint')
    for name, parameter in model.named_parameters():
        if not expected_flags[name]:
            import torch
            require(torch.equal(parameter, initial_parameters[name]), 'Frozen parameter changed: ' + name)
    del model, initial
    return audit, counts


def check_history(row, args, data):
    history = row['history']
    require([point['epoch'] for point in history] == list(range(args.epochs+1)) and
            all(type(point['epoch']) is int for point in history), 'Incomplete/invalid epoch history')
    expected_epochs = set(range(0, args.epochs+1, args.eval_every)) | {args.epochs}
    require({point['epoch'] for point in history if 'val' in point} == expected_epochs,
            'Validation evaluation schedule differs')
    evaluations = []
    per_epoch = math.ceil(len(data['fit'])/args.batch_size)
    expected_sensors = Counter(data['dataset'].modalities[index] for index in data['validation'])
    for point in history:
        epoch = point['epoch']
        if epoch:
            require(point['train_loss'] is not None and point['train_loss'] >= 0, 'Missing/negative training loss')
            schedule = hq.epoch_schedule(data['fit'], row['seed'], epoch, True)[2]
            require(point['train_schedule'] == dict(sha256=schedule, n_samples=len(data['fit'])),
                    'Actual sample order/augmentation schedule differs')
            for group in ('core', 'common'):
                values = point['gradient_norms'][group]
                require(type(values['steps']) is int and values['steps'] == per_epoch and
                        type(values['nonzero_steps']) is int and 0 <= values['nonzero_steps'] <= per_epoch and
                        0 <= values['min'] <= values['mean'] <= values['max'] and
                        ((values['max'] == 0) == (values['nonzero_steps'] == 0)),
                        'Invalid per-epoch gradient coverage/range')
        else:
            require(point['train_loss'] is None and point['train_schedule'] is None,
                    'Epoch zero unexpectedly contains training')
        if 'val' in point:
            checked = check_metrics(point['val'], len(data['validation']))
            require(all(checked[sensor]['n_samples'] == expected_sensors[sensor] for sensor in ('optical', 'sar')),
                    'Validation sensor cohort coverage differs')
            evaluations.append(point)
    best = max(evaluations, key=lambda point: point['val']['iou'])
    require(row['best_epoch'] == best['epoch'] and row['val'] == best['val'],
            'Checkpoint is not the earliest strict best validation epoch')
    for group in ('core', 'common'):
        total = row['diagnostics']['gradient_norms'][group]
        values = [point['gradient_norms'][group] for point in history[1:]]
        require(total['steps'] == per_epoch*args.epochs and
                total['nonzero_steps'] == sum(value['nonzero_steps'] for value in values) and
                total['min'] == min(value['min'] for value in values) and
                total['max'] == max(value['max'] for value in values) and
                math.isclose(total['mean'], statistics.mean(value['mean'] for value in values),
                             rel_tol=1e-12, abs_tol=1e-15), 'Aggregate gradient summary differs')
    if row['kind'] in ('frozen_quantum', 'zero_bottleneck'):
        require(row['diagnostics']['gradient_norms']['core']['max'] == 0 and
                row['diagnostics']['movement']['core']['l2'] == 0,
                'Frozen/inactive core changed or recorded gradients')
    probes = min(8, args.batch_size, len(data['fit']))
    require(row['diagnostics']['probe_names'] == row['fit_names'][:probes], 'Fixed training probe names differ')
    for key in ('initial_angle_probes', 'selected_angle_probes'):
        values = row['diagnostics'][key]
        require(len(values) == 1 and values[0]['n_angles'] == probes*16*3 and
                -1 <= values[0]['min_radians'] <= values[0]['max_radians'] <= 1,
                'Fixed tanh-radian probe dimensions/range differ')


def select_rates(entries):
    tuning = [entry['record'] for entry in entries if entry['record']['seed'] == 42]
    require(len(tuning) == 8 and {identity(row) for row in tuning} == {
        (kind, 42, rate) for kind in CONTROLS for rate in RATES}, 'Incomplete seed42 tuning grid')
    selected = {}
    for kind in CONTROLS:
        candidates = sorted((row for row in tuning if row['kind'] == kind), key=lambda row: row['learning_rate'])
        best = max(candidates, key=lambda row: (row['val']['iou'], -row['learning_rate']))
        selected[kind] = dict(learning_rate=best['learning_rate'], selected_seed=42,
            candidates=[dict(learning_rate=row['learning_rate'], val_iou=row['val']['iou'],
                             best_epoch=row['best_epoch']) for row in candidates])
    return selected


def audit_campaign(work_dir, data_root):
    verify_collection_source()
    work_dir, data_root = Path(work_dir).resolve(), Path(data_root).resolve()
    initial_inventory = inventory(work_dir)
    plan, summary = read(work_dir/'plan.json'), read(work_dir/'summary.json')
    require(plan['format'] == 'hq-dfc24-fresh-v1' and summary['plan'] == plan, 'Fresh plan/summary format differs')
    require(isinstance(plan['source_revision'], str) and len(plan['source_revision']) == 40 and
            set(plan['source_revision']) <= set('0123456789abcdef'), 'Invalid producing revision identity')
    require(plan['mode'] in ('full', 'rehearsal'), 'Unknown study mode')
    mode = plan['mode']
    mode_values = dict(full=dict(epochs=60, fit_cap=None, validation_cap=None, eval_every=2),
                       rehearsal=dict(epochs=2, fit_cap=16, validation_cap=16, eval_every=1))[mode]
    settings = dict(mode=mode, device=plan['settings']['device'], batch_size=8,
        image_size=128, base_channels=8, augment=True, **mode_values)
    require(plan['settings'] == settings and plan['settings']['augment'] is True and
            plan['settings']['device'] in ('cpu', 'cuda'), 'Mode fitting settings differ')
    require(plan['controls'] == list(CONTROLS) and plan['seeds'] == list(SEEDS) and
            all(type(seed) is int for seed in plan['seeds']) and plan['rates'] == list(RATES),
            'Fixed control/seed/rate plan differs')
    require(isinstance(plan['data_root'], str) and Path(plan['data_root']).is_absolute(),
            'Original data root is not absolute')
    require(type(plan['budget_seconds']) in (int, float) and not isinstance(plan['budget_seconds'], bool)
            and math.isfinite(plan['budget_seconds']) and plan['budget_seconds'] > 0, 'Invalid run wall budget')
    split = 'test' if mode == 'full' else 'validation'
    require(plan['replay_split'] == split, 'Mode/replay split differs')
    require(all(isinstance(plan[key], str) and plan[key] for key in ('selection_policy', 'test_policy')),
            'Missing selection/evaluation policy')
    verify_source(plan)
    collection_runtime = verify_runtime(plan)
    require(plan['runtime']['device'].split(':')[0] == settings['device'], 'Producer device settings differ')
    inputs = hq.pilot.verify_inputs(data_root)
    require(inputs == plan['input_identity'] and inputs['verified_files'] == 3476, 'Actual input bytes differ')
    manifest = read(data_root/'splits/flood_combined.json')
    require(manifest['conflict_policy'] == 'drop' and len(manifest['train']) == 1533 and
            len(manifest['test']) == 205, 'Clean manifest dimensions/policy differ')
    profile_path = work_dir/'resource_profile.json'
    require(digest(profile_path) == plan['resource_profile_sha256'], 'Bound resource profile bytes differ')
    profile = read(profile_path)
    require(profile['format'] == 'hq-dfc24-fresh-profile-v1' and profile['status'] == 'passed',
            'Missing successful actual-mode profile')
    require(profile['profile_learning_rate'] == .001, 'Fixed profile learning rate differs')
    for name in ('mode', 'settings', 'source_hashes', 'protocol_sha256', 'input_identity', 'runtime'):
        require(profile[name] == plan[name], 'Profile/producer binding differs: ' + name)
    require(profile['cohort'] == plan['cohorts']['42'], 'Profile/production seed42 cohort differs')
    require(profile['data_loading_s'] >= 0, 'Invalid profile data-loading duration')
    require(len(profile['cases']) == 4 and {case['kind'] for case in profile['cases']} == set(CONTROLS),
            'Incomplete actual-control resource profile')
    for case in profile['cases']:
        require(case['batch_size'] == 8 and case['strict_restore'] is True and case['frozen_unchanged'] is True,
                'Profile checkpoint/frozen checks failed')
        require(len(case['gradient_norms']) == 2 and len(case['timings_s']['adam_steps']) == 2 and
                all(value >= 0 for value in case['timings_s']['adam_steps']),
                'Profile must retain two actual Adam-step observations')
        require(all(case['timings_s'][key] >= 0 for key in
                    ('validation', 'checkpoint_save', 'checkpoint_load_restore')) and
                type(case['host_peak_rss_bytes']) is int and case['host_peak_rss_bytes'] > 0,
                'Invalid profile stage timing/host-memory record')
        if plan['runtime']['device'] == 'cuda:0':
            require(all(type(case[key]) is int and case[key] >= 0 for key in
                        ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')),
                    'Missing CUDA peak allocator observations')
        else:
            require(case['peak_cuda_allocated_bytes'] is case['peak_cuda_reserved_bytes'] is None,
                    'CPU profile has unexpected CUDA observations')
        for observed in case['gradient_norms']:
            require(set(observed) == {'core', 'common'} and all(value >= 0 for value in observed.values()),
                    'Invalid profile gradient norms')
            if case['kind'] in ('frozen_quantum', 'zero_bottleneck'):
                require(observed['core'] == 0, 'Profile frozen core recorded a gradient')
    require(digest(work_dir/'budget.json') == plan['budget_sha256'], 'Bound deadline/budget bytes differ')
    budget = read(work_dir/'budget.json')
    require(budget['format'] == 'hq-dfc24-fresh-budget-v1' and
            budget['budget_seconds'] == plan['budget_seconds'] and budget['queue_downtime_counts'] is True and
            type(budget['started_unix']) in (int, float) and budget['started_unix'] > 0 and
            budget['deadline_unix'] == budget['started_unix']+budget['budget_seconds'],
            'Immutable wall-budget ledger differs')
    entries = read(work_dir/'all_fits.json')
    require(len(entries) == 16 and all(entry['origin'] == 'fresh' for entry in entries),
            'Expected sixteen entirely fresh fit records')
    require(read(work_dir/'fit_progress.json') == entries, 'Completed fit progress differs')
    selection = select_rates(entries)
    require(read(work_dir/'selection.json') == summary['selection'] == selection,
            'Saved rate selection differs from validation-only rule/tie break')
    expected = {(kind, 42, rate) for kind in CONTROLS for rate in RATES} | {
        (kind, seed, selection[kind]['learning_rate']) for kind in CONTROLS for seed in SEEDS[1:]}
    require({identity(entry['record']) for entry in entries} == expected,
            'Fresh grid differs from selected-rate follow-ups')
    produced = read(work_dir/'checkpoint_audit.json')
    require(len(produced) == 16 and len({(row['id'], row['seed'], row['learning_rate']) for row in produced}) == 16,
            'Incomplete/duplicate producer checkpoint audits')
    audits = {(row['id'], row['seed'], row['learning_rate']): row for row in produced}
    boundary = read(work_dir/'fit_boundary.json')
    require(boundary == dict(format='hq-dfc24-fresh-boundary-v1', completed=True, fits=16,
        selected=12, replay_split=split, plan_sha256=digest(work_dir/'plan.json'),
        all_fits_sha256=digest(work_dir/'all_fits.json'), selection_sha256=digest(work_dir/'selection.json'),
        checkpoint_audit_sha256=digest(work_dir/'checkpoint_audit.json')),
        'Completed fit/audit boundary before replay differs')
    rows, paths, counts_by_control, restored_audits, validation_sensors = {}, {}, {}, [], {}
    require(set(plan['cohorts']) == {str(seed) for seed in SEEDS}, 'Incomplete planned seed cohorts')
    for seed in SEEDS:
        actual_args = argparse.Namespace(**settings, data_root=data_root)
        data = study.load_data(actual_args, seed)
        require(study.cohort_record(data) == plan['cohorts'][str(seed)], 'Actual fit/evaluation/normalization cohort differs')
        cohort = plan['cohorts'][str(seed)]
        validation_sensors[seed] = Counter(data['dataset'].modalities[index] for index in data['validation'])
        require(len(cohort['global_fit_indices']) == len(cohort['global_fit_names']) == 1303 and
                len(cohort['global_validation_indices']) == len(cohort['global_validation_names']) == 230 and
                cohort['n_global_fit'] == 1303 and cohort['normalization']['fit_only'] is True and
                cohort['normalization']['n_fit_train_tiles'] == 1303 and
                cohort['normalization']['n_excluded_val_tiles'] == 230 and
                cohort['normalization']['max_samples'] is None,
                'Global normalization-fit/validation coverage differs')
        require(len(cohort['fit_indices']) == len(cohort['fit_names']) == (1303 if mode == 'full' else 16) and
                len(cohort['validation_indices']) == len(cohort['validation_names']) == (230 if mode == 'full' else 16),
                'Mode optimization/evaluation cohort coverage differs')
        for field in ('global_fit_indices', 'global_validation_indices', 'fit_indices', 'validation_indices'):
            require(all(type(index) is int for index in cohort[field]) and
                    len(cohort[field]) == len(set(cohort[field])), 'Invalid typed/duplicate cohort indices')
        require(not set(cohort['global_fit_indices']) & set(cohort['global_validation_indices']) and
                set(cohort['fit_indices']) <= set(cohort['global_fit_indices']) and
                set(cohort['validation_indices']) <= set(cohort['global_validation_indices']),
                'Optimization/normalization/validation cohorts overlap incorrectly')
        for entry in (entry for entry in entries if entry['record']['seed'] == seed):
            row = entry['record']; key = identity(row)
            require(key not in rows, 'Duplicate fit observation')
            require(not ({'test', 'test_names', 'replay', 'replay_names', 'replay_split', 'checkpoint'} & set(row)),
                    'Fit record crosses selected replay boundary')
            relative = str(Path('fits')/row['id']/str(row['learning_rate'])/('seed'+str(seed)))
            require(entry['directory'] == relative, 'Unexpected fit directory')
            directory = inside(work_dir, relative)
            for filename in ('metrics.json', 'history.json', 'checkpoint.pt'):
                require(digest(directory/filename) == entry[filename.split('.')[0]+'_sha256'],
                        'Raw fit artifact hash differs: ' + filename)
            require(read(directory/'metrics.json') == row and read(directory/'history.json') == row['history'],
                    'Raw fit/history differs from complete-grid record')
            original_args = argparse.Namespace(**settings, data_root=Path(plan['data_root']))
            study.audit_fit(row, row['kind'], seed, row['learning_rate'], original_args, data)
            check_history(row, original_args, data)
            require(row['settings']['data_root'] == plan['data_root'] and row['elapsed_s'] > 0 and
                    row['fit_sensor_counts'] == dict(Counter(data['dataset'].modalities[index] for index in data['fit'])),
                    'Historical fit root/duration/sensor counts differ')
            actual, counts = checkpoint_audit(directory/'checkpoint.pt', row)
            checked = dict(id=row['id'], seed=seed, learning_rate=row['learning_rate'], origin='fresh', **actual)
            require(audits.get((row['id'], seed, row['learning_rate'])) == checked,
                    'Producer audit differs from actual strict checkpoint restoration')
            require(row['kind'] not in counts_by_control or counts_by_control[row['kind']] == counts,
                    'Control parameter accounting changes across cells')
            counts_by_control[row['kind']] = counts
            rows[key], paths[key] = row, directory
            restored_audits.append(checked)
        del data
        gc.collect()
    study.paired_audit(list(rows.values()))
    for seed in SEEDS:
        cohort = [row for row in rows.values() if row['seed'] == seed]
        for field in ('initial_shared_state_sha256', 'fit_indices', 'val_indices', 'fit_names', 'val_names', 'normalization'):
            require(all(row[field] == cohort[0][field] for row in cohort), 'Unpaired initialization/data: ' + field)
        require(all([point['train_schedule'] for point in row['history']] ==
                    [point['train_schedule'] for point in cohort[0]['history']] for row in cohort),
                    'Unpaired observed sample/augmentation schedules')
    records = summary['records']
    require(len(records) == 12 and {identity(row) for row in records} == {
        (kind, seed, selection[kind]['learning_rate']) for kind in CONTROLS for seed in SEEDS},
        'Incomplete/duplicated selected replay grid')
    selected_paths, intent_paths, verified_metrics = set(), set(), {}
    for row in records:
        key = identity(row)
        require({name: value for name, value in row.items() if name not in
            ('replay', 'replay_names', 'replay_split', 'checkpoint')} == rows[key], 'Replay differs from audited fit')
        names = manifest['test'] if split == 'test' else plan['cohorts'][str(row['seed'])]['validation_names']
        require(row['replay_split'] == split and row['replay_names'] == names and len(names) == len(set(names)),
                'Selected replay split/names differ')
        entry = next(entry for entry in entries if identity(entry['record']) == key)
        require(row['checkpoint'] == dict(origin='fresh', path=str(Path(entry['directory'])/'checkpoint.pt')),
                'Selected checkpoint reference differs')
        path = work_dir/'replays'/row['id']/('seed'+str(row['seed'])+'.json')
        intent = work_dir/'replay_intents'/row['id']/('seed'+str(row['seed'])+'.json')
        payload = read(path)
        study.validate_replay(payload, entry, digest(work_dir/'plan.json'), names, split)
        binding = study.replay_binding(entry, digest(work_dir/'plan.json'), names, split)
        require(read(intent) == dict(binding=binding), 'Durable replay intent differs from completed payload')
        require(payload == dict(binding=binding,
                                record=row), 'Saved replay differs from bound summary')
        if mode == 'rehearsal':
            require(not ({'test', 'test_names', 'test_metrics'} & set(row)), 'Rehearsal contains test scores')
        verified_metrics[key] = dict(val=check_metrics(row['val'], len(plan['cohorts'][str(row['seed'])]['validation_names'])),
            replay=check_metrics(row['replay'], len(names), full_test=split == 'test'))
        if split == 'validation':
            require(all(verified_metrics[key]['replay'][sensor]['n_samples'] == validation_sensors[row['seed']][sensor]
                        for sensor in ('optical', 'sar')), 'Replay validation sensor cohort differs')
        selected_paths.add(path)
        intent_paths.add(intent)
    require(set((work_dir/'replays').rglob('*.json')) == selected_paths, 'Extra/missing selected replay artifact')
    require(set((work_dir/'replay_intents').rglob('*.json')) == intent_paths, 'Extra/missing durable replay intent')
    for filename in ('metrics.json', 'history.json', 'checkpoint.pt'):
        require(set(work_dir.rglob(filename)) == {path/filename for path in paths.values()},
                'Extra/missing fit artifact: ' + filename)
    expected_directories = set()
    for path in paths.values():
        while path != work_dir/'fits':
            expected_directories.add(path)
            path = path.parent
    require({path.resolve() for path in (work_dir/'fits').rglob('*') if path.is_dir()} == expected_directories,
            'Extra/missing fit directory, including incomplete empty attempts')
    require({path.resolve() for path in (work_dir/'fits').rglob('*') if path.is_file()} == {
        path/filename for path in paths.values() for filename in ('metrics.json', 'history.json', 'checkpoint.pt')},
        'Extra/missing file within fitting evidence')
    require(not any('failure' in str(path.relative_to(work_dir)).lower()
            for path in work_dir.rglob('*.json')), 'Failed attempts require separate explicit accounting')
    require(summary['analysis'] == study.analysis(records, split), 'Producer summary analysis differs')
    for case in profile['cases']:
        from utilsTorch.hq_training_controls import build_model
        model = build_model(case['kind'], 42, base_channels=8, image_size=128)
        flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        require(case['parameter_counts'] == counts_by_control[case['kind']] and
                case['trainability'] == flags and
                case['control_mode'] == ('zero_bottleneck' if case['kind'] == 'zero_bottleneck' else 'on'),
                'Resource profile count/control semantics differ')
        metrics = check_metrics(case['validation'], len(plan['cohorts']['42']['validation_names']))
        require(all(metrics[sensor]['n_samples'] == validation_sensors[42][sensor] for sensor in ('optical', 'sar')),
                'Profile validation sensor coverage differs')
        del model
    verify_source(plan)
    require(hq.pilot.verify_inputs(data_root) == inputs, 'Input bytes changed during collection')
    require(inventory(work_dir) == initial_inventory, 'Raw campaign bytes changed during collection')
    verify_collection_source()
    return dict(plan=plan, summary=summary, fits=entries, selection=selection,
        checkpoint_audit=sorted(restored_audits, key=lambda row: (row['id'], row['seed'], row['learning_rate'])),
        parameter_counts=counts_by_control, metrics=verified_metrics, raw_files=initial_inventory,
        collector_runtime=collection_runtime)


def stats(values):
    values = list(values)
    return dict(mean=statistics.mean(values), std=statistics.stdev(values), n=len(values))


def analysis(result):
    require(len(result['fits']) == len(result['checkpoint_audit']) == 16 and
            len(result['summary']['records']) == 12 and
            all(entry['origin'] == 'fresh' for entry in result['fits']),
            'Incomplete audit cannot receive analysis or a gate verdict')
    by_key = {(row['kind'], row['seed']): row for row in result['summary']['records']}
    paired = []
    for other in CONTROLS[1:]:
        for phase in ('val', 'replay'):
            for scope in SCOPES:
                for metric in METRICS:
                    values = {seed: 100*(result['metrics'][identity(by_key['quantum', seed])][phase][scope]['metrics'][metric]
                        - result['metrics'][identity(by_key[other, seed])][phase][scope]['metrics'][metric]) for seed in SEEDS}
                    followup = stats(values[seed] for seed in SEEDS[1:])
                    paired.append(dict(left='quantum', right=other, phase=phase, scope=scope, metric=metric,
                        **stats(values.values()), wins=sum(value > 0 for value in values.values()),
                        seed42_pp=values[42], seed77_pp=values[77], seed123_pp=values[123],
                        followup_mean_pp=followup['mean'], followup_std_pp=followup['std'], followup_n=2,
                        same_selected_rate=result['selection']['quantum']['learning_rate'] ==
                            result['selection'][other]['learning_rate']))
    if result['plan']['mode'] == 'rehearsal':
        gate = dict(status='not_applicable_rehearsal',
            reason='Software rehearsal uses validation replays only; no scientific decision screen is calculated.')
    else:
        # Decisions use exact audited counts; floating-point tables above stay unchanged.
        def exact_iou(kind, seed):
            counts = result['metrics'][identity(by_key[kind, seed])]['val']['pooled']['counts']
            tp, fp, fn = (counts[key] for key in ('tp', 'fp', 'fn'))
            return Fraction(tp, tp+fp+fn) if tp+fp+fn else Fraction(0)
        checks = {}
        for other in ('frozen_quantum', 'compact_affine'):
            gains = [exact_iou('quantum', seed)-exact_iou(other, seed) for seed in SEEDS]
            checks[other] = dict(mean_gain_at_least_one_point=sum(gains, Fraction(0))/len(SEEDS) >= Fraction(1,100),
                all_three_positive=all(value > 0 for value in gains))
        quantum = sum((exact_iou('quantum', seed) for seed in SEEDS), Fraction(0))/len(SEEDS)
        zero = sum((exact_iou('zero_bottleneck', seed) for seed in SEEDS), Fraction(0))/len(SEEDS)
        passed = all(all(value.values()) for value in checks.values()) and quantum >= zero
        gate = dict(status='pass' if passed else 'fail', comparisons=checks,
            quantum_mean_at_least_zero=quantum >= zero, threshold_iou_fraction=.01,
            threshold_percentage_points=1.,
            interpretation='Chosen resource-prioritization screen; not significance, equivalence or quantum superiority.')
    return dict(paired=paired, gate=gate, unit='percentage points',
        estimand='Equally tuned training procedures; different selected rates preclude trainability-only attribution.')


def tables(result, derived):
    records = result['summary']['records']
    observations, candidates, summaries, trainability = [], [], [], []
    for row in sorted(records, key=identity):
        for phase in ('val', 'replay'):
            for scope in SCOPES:
                values = result['metrics'][identity(row)][phase][scope]
                observations.append(dict(kind=row['kind'], seed=row['seed'], learning_rate=row['learning_rate'],
                    role='rate_selection_seed' if row['seed'] == 42 else 'followup_seed',
                    phase=phase, replay_split=result['plan']['replay_split'], scope=scope,
                    best_epoch=row['best_epoch'], n_samples=values['n_samples'],
                    **values['counts'], **values['metrics'], saved_auc=values['saved_auc']))
    for kind in CONTROLS:
        for phase in ('val', 'replay'):
            for scope in SCOPES:
                group = [row for row in observations if row['kind'] == kind and row['phase'] == phase and row['scope'] == scope]
                for metric in METRICS:
                    summaries.append(dict(kind=kind, phase=phase, scope=scope, metric=metric,
                        selected_rate=result['selection'][kind]['learning_rate'], **stats(row[metric] for row in group)))
    for entry in sorted(result['fits'], key=lambda entry: identity(entry['record'])):
        row = entry['record']
        metrics = check_metrics(row['val'], len(row['val_names']))
        for scope in SCOPES:
            values = metrics[scope]
            candidates.append(dict(kind=row['kind'], seed=row['seed'], learning_rate=row['learning_rate'],
                role='tuning_candidate' if row['seed'] == 42 else 'selected_followup',
                selected_rate=row['learning_rate'] == result['selection'][row['kind']]['learning_rate'],
                best_epoch=row['best_epoch'], scope=scope, n_samples=values['n_samples'],
                **values['counts'], **values['metrics'], saved_auc=values['saved_auc']))
        gradient = row['diagnostics']['gradient_norms']['core']
        trainability.append(dict(kind=row['kind'], seed=row['seed'], learning_rate=row['learning_rate'],
            **result['parameter_counts'][row['kind']],
            core_gradient_min=gradient['min'], core_gradient_mean=gradient['mean'], core_gradient_max=gradient['max'],
            core_gradient_steps=gradient['steps'], core_gradient_nonzero_steps=gradient['nonzero_steps'],
            core_movement_l2=row['diagnostics']['movement']['core']['l2'],
            angle_probe_scope=row['diagnostics'].get('angle_probe_scope', 'active_fixed_probe')))
    return dict(summary=summaries, candidates=candidates, per_seed=observations,
                paired=derived['paired'], trainability=trainability)


def report(result, derived, rows):
    plan, mode = result['plan'], result['plan']['mode']
    lines = ['# Fresh HQ campaign audit: ' + mode, '']
    if mode == 'rehearsal':
        lines += ['**Software verification only.** All 16 fresh fits use two epochs and 16 optimization/evaluation-validation tiles. The 12 selected replays use validation images only. No test dataset was evaluated and no scientific gate is calculated: `not_applicable_rehearsal`.', '']
    else:
        lines += ['Complete exploratory full-fit screen: 16 fresh fits and 12 selected descriptive test replays. The existing 205-tile test split is exposed and is not independent confirmation.', '',
            'Resource-prioritization screen: **' + derived['gate']['status'] + '**. This chosen screen is not a significance or equivalence test; a pass motivates independent evaluation and a failure does not prove circuit effects absent.', '']
    lines += ['Every arm receives the same two-rate opportunity (0.001 and 0.003) on seed42, then selected-rate follow-ups on seeds77/123. Exact rate ties choose the lower rate; checkpoints use the earliest strict validation maximum, including epoch zero. Seed42 participates in tuning and aggregation. The two follow-ups are displayed separately and are not independent-data confirmation. Different selected rates compare training procedures rather than circuit trainability alone.', '',
        'All 16 checkpoints were restored strictly on CPU, including BatchNorm buffers, frozen parameters, trainability and permanent-zero mode. Collection audits saved records and count-derived metrics; it neither trains nor repeats inference. CPU restoration can run on another GPU host when Python/package/Torch-build identities match and exact initial state reconstruction succeeds; original GPU identity remains bound to the producer/profile.', '',
        'Normalization uses all 1,303 global-fit images and excludes all 230 global validation images. The rehearsal subsets do not create a pure16-shot or untouched reserve. Models retain base8,128-pixel inputs,batch8,ordinary skips,full16-qubit computation,tanh-radian encoding without pi,32 X/Z outputs and full reconstruction. These are binary DFC24 controls, not a LandCover.ai replication or quantum-hardware/federated-accuracy result.', '',
        '| Control | Selected rate | Validation foreground IoU mean ± sample SD | ' +
        ('Validation replay' if mode == 'rehearsal' else 'Descriptive test') + ' foreground IoU mean ± sample SD |',
        '| --- | ---: | ---: | ---: |']
    for kind in CONTROLS:
        values = {phase: next(row for row in rows['summary'] if row['kind'] == kind and
            row['phase'] == phase and row['scope'] == 'pooled' and row['metric'] == 'flood_iou') for phase in ('val', 'replay')}
        lines.append(f"| {kind} | {result['selection'][kind]['learning_rate']} | {values['val']['mean']:.6f} ± {values['val']['std']:.6f} | {values['replay']['mean']:.6f} ± {values['replay']['std']:.6f} |")
    lines += ['', 'Metrics are fractions. Pooled scores use summed pixel confusion counts; sensor scores use their own counts. Empty IoU/Dice denominators yield zero. Binary macro IoU is the arithmetic mean of flood and background IoU. AUC is retained only from saved evaluations and cannot be derived from confusion counts.', '',
        'Paired differences are computed within seed before mean and sample SD, then expressed in percentage points. `paired.csv` retains seed42,77,123 differences and the two-follow-up mean/SD separately. Sample SD is not a confidence interval. Registered/trainable counts are not functional-activity counts: the zero arm retains parameters whose input is permanently absent. Zero-arm angle probes are inactive counterfactual encodings.', '',
        'See [all candidates](candidates.csv), [selected observations with counts](per_seed.csv), [paired contrasts](paired.csv), [trainability](trainability.csv), [gate and analysis](analysis.json) and [provenance](provenance.json). Original JSON bytes and inert source snapshots are retained; no weights or images are exported.', '']
    return '\n'.join(lines)


def export(result, work_dir, data_root, output_dir):
    verify_collection_source()
    work_dir, data_root, output_dir = (Path(value).resolve() for value in (work_dir, data_root, output_dir))
    require(not output_dir.exists() or not any(output_dir.iterdir()), 'Use a new or empty output directory')
    for source in (work_dir, data_root, ROOT):
        require(output_dir != source and not source.is_relative_to(output_dir), 'Output contains an input/source root')
    require(not output_dir.is_relative_to(work_dir) and not output_dir.is_relative_to(data_root),
            'Output aliases raw inputs')
    require(inventory(work_dir) == result['raw_files'], 'Raw inputs changed before export')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in result['raw_files']:
        if name.endswith('.json'):
            target = output_dir/'raw_campaign'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((work_dir/name).read_bytes())
    for name in ('plan.json', 'summary.json', 'selection.json', 'all_fits.json'):
        (output_dir/name).write_bytes((work_dir/name).read_bytes())
    derived = analysis(result)
    rows = tables(result, derived)
    write(output_dir/'analysis.json', derived)
    write(output_dir/'checkpoint_audit.json', result['checkpoint_audit'])
    write(output_dir/'parameter_counts.json', result['parameter_counts'])
    for name, values in rows.items():
        with (output_dir/(name+'.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(values)
    source_manifest = {}
    for name in sorted(set(result['plan']['source_hashes']) | {study.PROTOCOL, COLLECTOR}):
        target_name = 'sources/'+name+'.txt'
        target = output_dir/target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        source_manifest[name] = dict(sha256=digest(target), retained_as=target_name)
    write(output_dir/'source_manifest.json', source_manifest)
    (output_dir/'input_manifest.json').write_bytes((data_root/'splits/flood_combined.json').read_bytes())
    for name, expected in result['plan']['input_identity']['stats_files_sha256'].items():
        path = inside(data_root, name)
        require(digest(path) == expected, 'Normalization metadata changed')
        target = output_dir/'input_metadata'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write(output_dir/'provenance.json', dict(producer_revision=result['plan']['source_revision'],
        producer_source_hashes=result['plan']['source_hashes'], protocol_sha256=result['plan']['protocol_sha256'],
        collector_source_hashes={name: value['sha256'] for name, value in source_manifest.items()},
        collector_runtime=result['collector_runtime'], producer_runtime=result['plan']['runtime'],
        input_identity=result['plan']['input_identity'], original_data_root=result['plan']['data_root'],
        actual_data_root=str(data_root), raw_work_dir=str(work_dir), raw_artifact_sha256=result['raw_files'],
        audited_fresh_fits=16, audited_saved_replays=12, independent_inference_replays=0,
        replay_split=result['plan']['replay_split'], mode=result['plan']['mode'],
        collection_scope='Strict CPU restoration and saved-record/count audit, not independent inference or throughput'))
    (output_dir/'README.md').write_text(report(result, derived, rows))
    command = ' '.join(map(shlex.quote, [sys.executable, str(ROOT/COLLECTOR), '--work-dir', str(work_dir),
        '--data-root', str(data_root), '--output-dir', '/tmp/quaffle-hq-fresh-regenerated']))
    (output_dir/'REPRODUCTION.md').write_text('# Regenerate the audit\n\nUse the bound producer source and compatible Python/package/Torch build. The collector has its own source binding, separate from fitting dependencies. Original source bytes are retained under inert `sources/*.txt` paths.\n\n```bash\n'+command+'\n```\n\nUse an empty output directory and the prescribed OS memory/zero-swap/two-thread bounds. All raw checkpoints must remain available locally. A byte-identical dataset may be relocated without rewriting original producer records; cohort and normalization identities must still match. CPU collection does not require the original GPU host. Raw JSON bytes, source snapshots and derived CSV/JSON are exported; no trained weights or image data are copied. `artifact_sha256.json` covers every output except itself.\n')
    require(inventory(work_dir) == result['raw_files'], 'Raw inputs changed during export')
    verify_source(result['plan'])
    require(hq.pilot.verify_inputs(data_root) == result['plan']['input_identity'], 'Data bytes changed during export')
    verify_collection_source()
    write(output_dir/'artifact_sha256.json', inventory(output_dir))
    return derived


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), 'Use a new or empty output directory')
    for source in (args.work_dir, args.data_root):
        require(not args.output_dir.resolve().is_relative_to(source.resolve()) and
                not source.resolve().is_relative_to(args.output_dir.resolve()), 'Output aliases/contains raw inputs')
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    result = audit_campaign(args.work_dir, args.data_root)
    export(result, args.work_dir, args.data_root, args.output_dir)
    print('Audited16 fresh checkpoints and12 saved replays:', args.output_dir)
    return result


if __name__ == '__main__':
    main()
