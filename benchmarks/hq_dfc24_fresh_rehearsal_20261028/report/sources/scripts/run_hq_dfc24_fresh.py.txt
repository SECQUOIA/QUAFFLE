#!/usr/bin/env python3
"""Fresh four-arm HQ experiment, with a validation-only software rehearsal."""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_hq_dfc24_direction as hq
from scripts import run_hq_training_controls as study

CONTROLS = ('quantum', 'frozen_quantum', 'zero_bottleneck', 'compact_affine')
SEEDS, RATES = (42, 77, 123), (.001, .003)
PROFILE_LEARNING_RATE = .001
MODE_SPECS = dict(full=dict(epochs=60, fit_cap=None, validation_cap=None, eval_every=2),
                  rehearsal=dict(epochs=2, fit_cap=16, validation_cap=16, eval_every=1))
PROTOCOL = 'docs/HQ_DFC24_FRESH_PROTOCOL.md'
SOURCE_FILES = (*hq.SOURCE_FILES, 'scripts/run_hq_training_controls.py',
                'utilsTorch/hq_training_controls.py', 'scripts/run_hq_dfc24_fresh.py')
require, digest, canonical = hq.require, hq.digest, hq.canonical
pilot = hq.pilot
_CONTROL_FILES = {'.writer.lock', 'budget.json', 'plan.json', 'resource_profile.json',
    'fit_progress.json', 'selection.json', 'all_fits.json', 'checkpoint_audit.json',
    'fit_boundary.json', 'summary.json', 'fits', 'replays', 'replay_intents', 'replay_failures'}


def read_json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    result = json.loads(Path(path).read_text(), parse_constant=invalid, object_pairs_hook=unique)
    hq.finite_record(result)
    return result


def source_hashes():
    return {name: digest(ROOT/name) for name in SOURCE_FILES}


def runtime(device, gpu_memory_fraction):
    value = hq.runtime(device)
    if device.type == 'cuda':
        require(type(gpu_memory_fraction) in (int, float) and math.isfinite(gpu_memory_fraction)
                and 0 < gpu_memory_fraction <= 1, 'CUDA requires an explicit valid allocator fraction')
    value['gpu_allocator_fraction'] = gpu_memory_fraction if device.type == 'cuda' else None
    return value


def settings_for_mode(mode, device='cuda'):
    require(mode in MODE_SPECS and device in ('cpu', 'cuda'), 'Unknown execution mode/device')
    return dict(mode=mode, device=device, batch_size=8, image_size=128, base_channels=8,
                augment=True, **MODE_SPECS[mode])


def _apply_settings(args, device):
    require(args.device == device.type, 'Requested and actual device differ')
    require(type(args.budget_seconds) in (int, float) and math.isfinite(args.budget_seconds)
            and args.budget_seconds > 0, 'A positive finite wall budget is required')
    for name, value in settings_for_mode(args.mode, args.device).items():
        if hasattr(args, name):
            require(getattr(args, name) == value, 'Fixed mode setting differs: ' + name)
        setattr(args, name, value)
    args.data_root, args.work_dir = Path(args.data_root).resolve(), Path(args.work_dir).resolve()
    if getattr(args, 'profile', None) is not None:
        args.profile = Path(args.profile).resolve()
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 2:
        torch.set_num_interop_threads(2)
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable; no silent fallback')
        runtime(device, args.gpu_memory_fraction)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)


def load_data(args, seed):
    import torch
    data = pilot.load_training_data(args, seed)
    data['global_fit'] = list(data['all_fit'])
    data['global_validation'] = list(data['validation'])
    require(len(data['global_fit']) == 1303 and len(data['global_validation']) == 230,
            'Expected the full1303-fit/230-validation normalization partition')
    if args.validation_cap is not None:
        data['validation'] = pilot.selected_fit_indices(data['global_validation'], args.validation_cap, seed)
        data['val'] = torch.utils.data.Subset(data['dataset'], data['validation'])
    record = cohort_record(data)
    require(len(record['fit_indices']) == (args.fit_cap or 1303) and
            len(record['validation_indices']) == (args.validation_cap or 230), 'Mode cohort caps differ')
    norm = record['normalization']
    require(norm['fit_only'] is True and norm['n_fit_train_tiles'] == 1303
            and norm['n_excluded_val_tiles'] == 230 and norm['max_samples'] is None,
            'Normalization must exclude all230 validation tiles, independently of caps')
    return data


def cohort_record(data):
    record = dict(normalization=data['normalization'], n_global_fit=len(data['all_fit']))
    for prefix, indices in (('global_fit', data['global_fit']), ('global_validation', data['global_validation']),
                            ('fit', data['fit']), ('validation', data['validation'])):
        require(all(type(index) is int for index in indices) and len(set(indices)) == len(indices),
                'Cohort indices must be unique integers')
        record[prefix+'_indices'] = list(indices)
        record[prefix+'_names'] = [data['dataset'].names[index] for index in indices]
    require(set(record['fit_indices']) <= set(record['global_fit_indices']) and
            set(record['validation_indices']) <= set(record['global_validation_indices']) and
            not set(record['global_fit_indices']) & set(record['global_validation_indices']), 'Cohort partitions overlap')
    return canonical(record)


def _identity(row):
    require(row['kind'] in CONTROLS and type(row['seed']) is int and row['seed'] in SEEDS
            and type(row['learning_rate']) in (int, float) and row['learning_rate'] in RATES,
            'Invalid typed fit identity')
    require(row['id'] == 'hq_'+row['kind'], 'Fit ID differs from control')
    return row['kind'], row['seed'], row['learning_rate']


def check_metrics(value, n_samples, test=False):
    hq.validate_sensor_metrics(value, n_samples, test=test)
    pooled = dict(tp=0, fp=0, fn=0, tn=0)
    for prefix in ('optical_subset_', 'sar_subset_'):
        counts = {name: value[prefix+name] for name in pooled}
        for name, count in counts.items():
            pooled[name] += count
        tp, fp, fn, tn = (counts[name] for name in ('tp', 'fp', 'fn', 'tn'))
        expected = dict(iou=tp/(tp+fp+fn) if tp+fp+fn else 0.,
            dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            accuracy=(tp+tn)/sum(counts.values()) if sum(counts.values()) else 0.)
        require(sum(counts.values()) == value[prefix+'n_samples']*128*128, 'Sensor pixel coverage differs')
        for name, score in expected.items():
            require(math.isclose(value[prefix+name], score, rel_tol=0, abs_tol=1e-12), 'Sensor metric/count mismatch')
        require(0 <= value[prefix+'auc'] <= 1, 'Invalid saved sensor AUC')
    tp, fp, fn, tn = (pooled[name] for name in ('tp', 'fp', 'fn', 'tn'))
    require(math.isclose(value['dice'], 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., rel_tol=0, abs_tol=1e-12)
            and math.isclose(value['accuracy'], (tp+tn)/sum(pooled.values()), rel_tol=0, abs_tol=1e-12),
            'Pooled Dice/accuracy/count mismatch')
    require(value['loss'] >= 0 and 0 <= value['auc'] <= 1, 'Invalid saved loss/AUC')


def audit_fit(row, kind, seed, rate, args, data):
    require(_identity(row) == (kind, seed, rate), 'Cached cell identity differs')
    require(not {'test', 'test_names', 'replay', 'replay_names', 'replay_split', 'checkpoint'} & row.keys(),
            'Fit record crosses the replay boundary')
    (hq.audit_fit if kind == 'quantum' else study.audit_fit)(row, kind, seed, rate, args, data)
    steps = math.ceil(len(data['fit'])/args.batch_size)
    for point in row['history']:
        if point['epoch']:
            for group in ('core', 'common'):
                gradient = point['gradient_norms'][group]
                require(gradient['steps'] == steps and 0 <= gradient['nonzero_steps'] <= steps
                        and 0 <= gradient['min'] <= gradient['mean'] <= gradient['max'], 'Incomplete gradient coverage')
        if 'val' in point:
            check_metrics(point['val'], len(data['validation']))
    require(row['elapsed_s'] > 0 and row['diagnostics']['probe_names'] == row['fit_names'][:8],
            'Invalid fit duration/probe cohort')


def restore_checkpoint(path, row, device='cpu'):
    from utilsTorch import hq_training_controls as factory
    model, details = study.restore_checkpoint(path, row, device)
    # Native Q checkpoints use the legacy constructor; this is metadata only.
    model.training_control = row['kind']
    config = row['model_config']
    initial = factory.build_model(row['kind'], row['seed'], config['base_channels'], config['image_size'])
    require(hq.state_hash(initial.state_dict()) == row['initial_state_sha256'] and
            hq.shared_hash(initial) == row['initial_shared_state_sha256'], 'Independent initialization differs')
    flags, expected_flags = study.trainability(model), study.trainability(initial)
    actual_counts, expected_counts = factory.parameter_counts(model), factory.parameter_counts(initial)
    require(flags == expected_flags and actual_counts == expected_counts, 'Control trainability/counts differ')
    require((actual_counts['total'], actual_counts['trainable'], actual_counts['core_total']) ==
            (row['total_params'], row['trainable_params'], row['core_params']), 'Native fit parameter counts differ')
    mode = 'zero_bottleneck' if row['kind'] == 'zero_bottleneck' else 'on'
    require(model.bottleneck.mode == mode, 'Restored control mode differs')
    frozen = study.frozen_hash(model)
    require(frozen == study.frozen_hash(initial), 'Frozen parameter identity changed')
    hq.finite_tensors(model.state_dict().items(), 'restored parameters and BatchNorm buffers')
    details.update(training_control=row['kind'], control_mode=mode, parameter_counts=actual_counts,
        trainability_sha256=hashlib.sha256(json.dumps(flags, sort_keys=True).encode()).hexdigest(),
        frozen_parameters_sha256=frozen, frozen_unchanged=True)
    return model, details


def fit_cell(kind, seed, rate, args, data, directory, device):
    _identity(dict(id='hq_'+kind, kind=kind, seed=seed, learning_rate=rate))
    directory = Path(directory)
    require(not (directory/'failure.json').exists(), 'Failed fit evidence preserved: '+str(directory))
    if directory.exists():
        require(all((directory/name).is_file() for name in ('metrics.json', 'history.json', 'checkpoint.pt')),
                'Incomplete fit evidence preserved: '+str(directory))
        row = read_json(directory/'metrics.json')
        require(read_json(directory/'history.json') == row['history'], 'Cached raw history differs')
    else:
        try:
            # Q must retain its original hq-direction-v1 config/checkpoint format.
            fitter = hq.fit_one if kind == 'quantum' else study.fit_one
            row = fitter(kind, seed, rate, args, data, directory, device)
        except BaseException as error:
            directory.mkdir(parents=True, exist_ok=True)
            if not (directory/'failure.json').exists():
                pilot.write_json(directory/'failure.json', dict(error_type=type(error).__name__,
                    error=str(error), traceback=traceback.format_exc()))
            raise
    audit_fit(row, kind, seed, rate, args, data)
    restored, _ = restore_checkpoint(directory/'checkpoint.pt', row)
    del restored
    return row


def select_rates(entries):
    candidates = []
    for entry in entries:
        require(entry['origin'] == 'fresh', 'Historical reuse is forbidden')
        row = entry['record']; _identity(row)
        if row['seed'] == 42:
            candidates.append(row)
    require(len(candidates) == 8 and {_identity(row) for row in candidates} ==
            {(kind, 42, rate) for kind in CONTROLS for rate in RATES}, 'Incomplete/duplicated tuning grid')
    selection = {}
    for kind in CONTROLS:
        group = sorted((row for row in candidates if row['kind'] == kind), key=lambda row:row['learning_rate'])
        best = max(group, key=lambda row:(row['val']['iou'], -row['learning_rate']))
        selection[kind] = dict(learning_rate=best['learning_rate'], selected_seed=42,
            candidates=[dict(learning_rate=row['learning_rate'], val_iou=row['val']['iou'],
                             best_epoch=row['best_epoch']) for row in group])
    return selection


def expected_grid(selection):
    require(set(selection) == set(CONTROLS) and all(value['learning_rate'] in RATES and
            type(value['selected_seed']) is int and value['selected_seed'] == 42 for value in selection.values()),
            'Invalid four-arm selection')
    return {(kind, 42, rate) for kind in CONTROLS for rate in RATES} | {
        (kind, seed, selection[kind]['learning_rate']) for kind in CONTROLS for seed in SEEDS[1:]}


def paired_audit(rows):
    selection = select_rates([dict(origin='fresh', record=row) for row in rows])
    require(len(rows) == 16 and {_identity(row) for row in rows} == expected_grid(selection),
            'Incomplete, duplicate or unexpected16-fit grid')
    for seed in SEEDS:
        cohort = [row for row in rows if row['seed'] == seed]
        for field in ('initial_shared_state_sha256', 'fit_names', 'val_names', 'fit_indices', 'val_indices', 'normalization'):
            require(all(row[field] == cohort[0][field] for row in cohort), 'Unpaired '+field)
        require(all([point['train_schedule'] for point in row['history']] ==
                    [point['train_schedule'] for point in cohort[0]['history']] for row in cohort),
                    'Unpaired training order/augmentation')
        circuits = [row for row in cohort if row['kind'] != 'compact_affine']
        require(len({row['initial_state_sha256'] for row in circuits}) == 1, 'Circuit initial states differ')
        for kind in CONTROLS:
            require(len({row['initial_state_sha256'] for row in cohort if row['kind'] == kind}) == 1,
                    'Rate candidates have different initial states')


def replay_binding(entry, plan_sha256, names, split):
    require(split in ('validation', 'test') and entry['origin'] == 'fresh', 'Invalid fresh replay split/origin')
    return dict(plan_sha256=plan_sha256, origin='fresh',
        checkpoint_path=str(Path(entry['directory'])/'checkpoint.pt'),
        checkpoint_sha256=entry['checkpoint_sha256'], metrics_sha256=entry['metrics_sha256'],
        history_sha256=entry['history_sha256'], names_sha256=hashlib.sha256(
            json.dumps(list(names), separators=(',', ':')).encode()).hexdigest(), split=split)


def validate_replay(payload, entry, plan_sha256, names, split):
    require(set(payload) == {'binding', 'record'} and payload['binding'] == replay_binding(entry, plan_sha256, names, split),
            'Saved replay binding differs')
    row = payload['record']; _identity(row)
    require({key:value for key,value in row.items() if key not in ('replay', 'replay_names', 'replay_split', 'checkpoint')}
            == entry['record'], 'Saved replay fit differs')
    require(row['replay_names'] == list(names) and len(set(names)) == len(names) and row['replay_split'] == split
            and row['checkpoint'] == dict(origin='fresh', path=payload['binding']['checkpoint_path']),
            'Saved replay checkpoint/cohort/split differs')
    check_metrics(row['replay'], len(names), test=split == 'test')
    return row


def analysis(records, replay_split):
    require(len(records) == 12 and {(_identity(row)[0], row['seed']) for row in records} ==
            {(kind, seed) for kind in CONTROLS for seed in SEEDS}, 'Incomplete selected replay grid')
    metrics = ('iou', 'dice', 'accuracy', 'optical_subset_iou', 'sar_subset_iou')
    stats = lambda values:dict(mean=statistics.mean(values), std=statistics.stdev(values), n=3)
    by_key = {(row['kind'], row['seed']):row for row in records}
    rows = [dict(kind=kind, **{split:{metric:stats([by_key[kind, seed][split][metric] for seed in SEEDS])
            for metric in metrics} for split in ('val', 'replay')}) for kind in CONTROLS]
    pairs = []
    for other in CONTROLS[1:]:
        for split in ('val', 'replay'):
            for metric in metrics:
                differences = {str(seed):100*(by_key['quantum',seed][split][metric]-by_key[other,seed][split][metric]) for seed in SEEDS}
                pairs.append(dict(left='quantum', right=other, split=split, metric=metric,
                    differences_pp=differences, **stats(list(differences.values())),
                    wins=sum(value > 0 for value in differences.values()), unit='percentage points'))
    return dict(format='hq-fresh-analysis-v1', replay_split=replay_split, rows=rows, paired=pairs,
        estimand='Equal two-rate validation-tuned procedures; differing selected rates prevent trainability-only attribution.',
        gate=dict(status='not_applicable_rehearsal' if replay_split == 'validation' else 'not_evaluated_by_runner'))


def _write_once(path, value):
    if path.exists():
        require(read_json(path) == canonical(value), 'Retained artifact differs: '+path.name)
    else:
        pilot.write_json(path, value)


def _progress(path, entries):
    if path.exists():
        previous = read_json(path)
        if len(entries) <= len(previous):
            require(entries == previous[:len(entries)], 'Completed progress binding differs')
            return
        require(previous == entries[:len(previous)], 'Partial progress binding differs')
    pilot.write_json(path, entries)


@contextmanager
def _session(args, phase):
    """One live writer; immutable learning deadline includes downtime/queue time.

    Completed resume is read-only verification with a fresh wall allowance; the
    original learning budget/deadline and every raw file remain unchanged.
    """
    work = args.work_dir
    for source in (args.data_root, getattr(args, 'profile', None)):
        if source is not None:
            source = Path(source).resolve()
            require(work != source and not work.is_relative_to(source) and not source.is_relative_to(work),
                    'Output overlaps data/profile input')
    require(not work.exists() or work.is_dir(), 'Output is not a directory')
    if work.exists() and (phase == 'profile' or not args.resume):
        require(not [p for p in work.iterdir() if p.name != '.writer.lock'], 'Use a new/empty output or explicit resume')
    readonly = phase == 'run' and args.resume and (work/'summary.json').exists()
    if readonly:
        required = ('.writer.lock','budget.json','plan.json','resource_profile.json','fit_progress.json',
                    'selection.json','all_fits.json','checkpoint_audit.json','fit_boundary.json','summary.json')
        require(all((work/name).is_file() for name in required), 'Completed resume is missing required raw artifacts')
        for namespace in ('replays','replay_intents'):
            expected = {work/namespace/('hq_'+kind)/('seed'+str(seed)+'.json') for kind in CONTROLS for seed in SEEDS}
            require({p for p in (work/namespace).rglob('*') if p.is_file()} == expected,
                    'Completed resume is missing or has unexpected '+namespace)
    work.mkdir(parents=True, exist_ok=True)
    lock = (work/'.writer.lock').open('a+b')
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another live writer holds this output') from error
        budget_path = work/'budget.json'
        if budget_path.exists():
            budget = read_json(budget_path)
            require(args.resume and budget['format'] == 'hq-dfc24-fresh-budget-v1' and
                    budget['budget_seconds'] == args.budget_seconds and budget['queue_downtime_counts'] is True,
                    'Resume budget differs from the original declaration')
            require(budget['deadline_unix'] == budget['started_unix']+budget['budget_seconds'],
                    'Retained fixed deadline differs')
        else:
            require(not args.resume, 'Resume requires the retained budget')
            started = time.time()
            budget = dict(format='hq-dfc24-fresh-budget-v1', budget_seconds=args.budget_seconds,
                started_unix=started, deadline_unix=started+args.budget_seconds, queue_downtime_counts=True)
            pilot.write_json(budget_path, budget)
        remaining = args.budget_seconds if readonly else budget['deadline_unix']-time.time()
        require(remaining > 0, 'Original learning deadline expired; incomplete state preserved')
        previous_handler, previous_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
        def timeout(signum, frame):
            raise TimeoutError('Declared finite wall budget exhausted; attempt preserved')
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            yield readonly
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    finally:
        lock.close()


def _cell_directory(work, kind, seed, rate):
    return work/'fits'/('hq_'+kind)/str(rate)/('seed'+str(seed))


def _entry(work, directory, row):
    require(directory.resolve().is_relative_to(work.resolve()), 'Fit path escapes output')
    return dict(origin='fresh', directory=str(directory.relative_to(work)), record=row,
        **{name.split('.')[0]+'_sha256':digest(directory/name) for name in ('metrics.json','history.json','checkpoint.pt')})


def _existing_cells(args, data_by_seed):
    cache = {}
    require(all(p.name in _CONTROL_FILES for p in args.work_dir.iterdir()), 'Unexpected existing output artifact')
    root = args.work_dir/'fits'
    require(root.resolve().is_relative_to(args.work_dir), 'Fit root escapes output')
    for directory in sorted(root.glob('*/*/*')):
        require(directory.is_dir(), 'Unexpected fit path')
        require(directory.resolve().is_relative_to(args.work_dir), 'Fit directory escapes output')
        require(not (directory/'failure.json').exists(), 'Failed fit evidence preserved')
        require(all((directory/name).is_file() for name in ('metrics.json','history.json','checkpoint.pt')),
                'Incomplete fit evidence preserved')
        row = read_json(directory/'metrics.json'); kind, seed, rate = _identity(row)
        require(directory == _cell_directory(args.work_dir, kind, seed, rate) and
                directory.resolve().is_relative_to(root.resolve()), 'Unexpected/escaping fit directory')
        require(read_json(directory/'history.json') == row['history'], 'Retained raw history differs')
        audit_fit(row, kind, seed, rate, args, data_by_seed[seed])
        model, _ = restore_checkpoint(directory/'checkpoint.pt', row)
        del model
        cache[kind, seed, rate] = _entry(args.work_dir, directory, row)
    expected_files = {args.work_dir/entry['directory']/name for entry in cache.values()
                      for name in ('metrics.json','history.json','checkpoint.pt')}
    require({p for p in root.rglob('*') if p.is_file()} == expected_files, 'Unexpected fit file or incomplete directory')
    for directory in (p for p in root.rglob('*') if p.is_dir()):
        parts = directory.relative_to(root).parts
        require(directory.resolve().is_relative_to(args.work_dir) and len(parts) <= 3 and parts[0] in {'hq_'+kind for kind in CONTROLS}
                and (len(parts) < 2 or parts[1] in {str(rate) for rate in RATES})
                and (len(parts) < 3 or parts[2] in {'seed'+str(seed) for seed in SEEDS}), 'Unexpected fit namespace')
    if (args.work_dir/'fit_progress.json').exists():
        progress = read_json(args.work_dir/'fit_progress.json')
        require(len({_identity(e['record']) for e in progress}) == len(progress) and
                all(e['origin'] == 'fresh' and cache.get(_identity(e['record'])) == e for e in progress),
                'Retained progress inventory differs from cached cells')
    if (args.work_dir/'selection.json').exists():
        selection = select_rates(list(cache.values()))
        require(read_json(args.work_dir/'selection.json') == selection and set(cache) <= expected_grid(selection),
                'Retained selection or cached grid differs')
    require(not list((args.work_dir/'replay_failures').rglob('*.json')), 'Failed replay evidence preserved')
    for intent in (args.work_dir/'replay_intents').rglob('*.json'):
        replay = args.work_dir/'replays'/intent.relative_to(args.work_dir/'replay_intents')
        require(replay.is_file(), 'Unfinished replay intent preserved; inference cannot retry')
        require(read_json(intent) == dict(binding=read_json(replay)['binding']), 'Completed replay intent binding differs')
    if list((args.work_dir/'replays').rglob('*.json')):
        require(len(cache) == 16 and all((args.work_dir/name).is_file() for name in
                ('selection.json','all_fits.json','checkpoint_audit.json','fit_boundary.json')),
                'Replay evidence precedes the completed-fit boundary')
    return cache


def _validate_profile(profile, args, device, inputs, data):
    require(profile['format'] == 'hq-dfc24-fresh-profile-v1' and profile['status'] == 'passed', 'Missing completed profile')
    expected = dict(mode=args.mode, settings=settings_for_mode(args.mode,args.device),
        source_hashes=source_hashes(), protocol_sha256=digest(ROOT/PROTOCOL),
        runtime=runtime(device,args.gpu_memory_fraction), input_identity=inputs, cohort=cohort_record(data),
        profile_learning_rate=PROFILE_LEARNING_RATE)
    for key, value in expected.items():
        require(profile[key] == canonical(value), 'Profile binding differs: '+key)
    require(len(profile['cases']) == 4 and {case['kind'] for case in profile['cases']} == set(CONTROLS)
            and all(case['batch_size'] == 8 and case['strict_restore'] is True and case['frozen_unchanged'] is True
                    for case in profile['cases']), 'Incomplete four-arm actual-model profile')


def profile(args, device):
    import torch
    from tools import training_common as tc
    from utilsTorch import hq_training_controls as factory
    from utilsTorch.utilsTorch import segmentation_loss
    _apply_settings(args, device)
    require(not args.resume, 'Profiles never resume')
    with _session(args, 'profile'):
        try:
            hashes, inputs = source_hashes(), pilot.verify_inputs(args.data_root)
            started = time.perf_counter(); data = load_data(args,42); loading_s = time.perf_counter()-started
            result = dict(format='hq-dfc24-fresh-profile-v1', mode=args.mode,
                profile_learning_rate=PROFILE_LEARNING_RATE,
                settings=settings_for_mode(args.mode,args.device), source_revision=subprocess.check_output(
                    ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), source_hashes=hashes,
                protocol_sha256=digest(ROOT/PROTOCOL), input_identity=inputs,
                runtime=runtime(device,args.gpu_memory_fraction), cohort=cohort_record(data),
                data_loading_s=loading_s, cases=[],
                instrumentation_scope='Two actual batch8 Adam steps, mode-validation and strict checkpoint I/O; not a throughput or accuracy benchmark. Host RSS is process-lifetime high water; CUDA peaks span each case including restore.')
            for kind in CONTROLS:
                tc.seed_everything(42)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                model = factory.build_model(kind,42,args.base_channels,args.image_size).to(device).train()
                initial = {name:p.detach().cpu().clone() for name,p in model.named_parameters()}
                frozen = study.frozen_hash(model)
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=PROFILE_LEARNING_RATE)
                augmented = hq.EpochDataset(data['train'],data['fit'],42,1,True)
                loader = torch.utils.data.DataLoader(augmented,batch_size=8,sampler=augmented.order,num_workers=0,
                    generator=torch.Generator().manual_seed(hq.stable_seed(42,1,'profile-loader')))
                iterator = iter(loader); timings, gradients = [], []
                for _ in range(2):
                    began = time.perf_counter(); images,masks = next(iterator)
                    images,masks = images.to(device),masks.to(device)
                    optimizer.zero_grad(set_to_none=True); output = model(images)
                    require(tuple(output.shape) == (8,2,128,128), 'Profile geometry differs')
                    hq.finite_tensors([('output',output)],'profile output')
                    loss = segmentation_loss(output,masks,loss_mode='dice_ce')
                    require(bool(torch.isfinite(loss)), 'Nonfinite profile loss')
                    loss.backward(); gradients.append(study.gradient_norms(model)); optimizer.step()
                    hq.finite_tensors(model.state_dict().items(),'profile weights and BatchNorm')
                    if device.type == 'cuda':torch.cuda.synchronize(device)
                    timings.append(time.perf_counter()-began)
                moved = hq.movement(model,initial)
                require(study.frozen_hash(model) == frozen, 'Profile changed frozen parameters')
                if kind in ('quantum','compact_affine'):
                    require(all(g['core'] > 0 for g in gradients) and moved['core']['l2'] > 0, 'Profile core is detached/inactive')
                else:
                    require(all(g['core'] == 0 for g in gradients) and moved['core']['l2'] == 0, 'Frozen core moved')
                val_loader = tc.make_loader(data['val'],8,False,42)
                began = time.perf_counter(); validation = hq.evaluate(model,val_loader,device)
                check_metrics(validation,len(data['validation'])); validation_s = time.perf_counter()-began
                checkpoint = args.work_dir/'profile_checkpoints'/('hq_'+kind+'.pt')
                checkpoint.parent.mkdir(exist_ok=True)
                began = time.perf_counter()
                tc.atomic_torch_save(dict(format='hq-dfc24-profile-checkpoint-v1',config=factory.model_config(model),
                    flags=study.trainability(model),counts=factory.parameter_counts(model),
                    state=tc.clone_state_dict_cpu(model.state_dict())),str(checkpoint))
                save_s = time.perf_counter()-began; began = time.perf_counter()
                saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
                restored = factory.build_from_config(saved['config']).to(device)
                restored.load_state_dict(saved['state'],strict=True)
                require(study.trainability(restored) == saved['flags'] and factory.parameter_counts(restored) == saved['counts'],
                        'Profile strict trainability/count restore differs')
                require(restored.bottleneck.mode == ('zero_bottleneck' if kind == 'zero_bottleneck' else 'on')
                        and study.frozen_hash(restored) == frozen, 'Profile strict mode/frozen restore differs')
                hq.finite_tensors(restored.state_dict().items(),'profile restored state')
                model.eval(); restored.eval()
                with torch.no_grad():
                    require(torch.equal(model(images),restored(images)), 'Profile restored predictions differ')
                load_s = time.perf_counter()-began
                result['cases'].append(dict(kind=kind,batch_size=8,parameter_counts=saved['counts'],trainability=saved['flags'],
                    initial_shared_state_sha256=hq.state_hash({name:value for name,value in
                        factory.build_model(kind,42,args.base_channels,args.image_size).state_dict().items()
                        if not name.startswith('bottleneck.core.')}),
                    control_mode=restored.bottleneck.mode,gradient_norms=gradients,movement=moved,
                    frozen_unchanged=True,strict_restore=True,validation=validation,
                    timings_s=dict(adam_steps=timings,validation=validation_s,checkpoint_save=save_s,checkpoint_load_restore=load_s),
                    checkpoint_sha256=digest(checkpoint),host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None))
                del model,restored,optimizer,images,masks,output,loss,iterator,loader,augmented
                gc.collect()
                if device.type=='cuda':torch.cuda.empty_cache()
            require(len({case['initial_shared_state_sha256'] for case in result['cases']}) == 1, 'Profile common initialization differs')
            require(source_hashes() == hashes and pilot.verify_inputs(args.data_root) == inputs, 'Profile source/input changed')
            result['status'] = 'passed'
            _write_once(args.work_dir/'resource_profile.json',result)
            return result
        except BaseException as error:
            _write_once(args.work_dir/'profile_failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
            raise


def run(args, device):
    from tools import training_common as tc
    _apply_settings(args, device)
    require(args.profile is not None, 'Run requires an actual mode-matched profile')
    with _session(args,'run') as readonly:
        hashes, inputs = source_hashes(), pilot.verify_inputs(args.data_root)
        data = {seed:load_data(args,seed) for seed in SEEDS}
        resource_profile = read_json(args.profile)
        _validate_profile(resource_profile,args,device,inputs,data[42])
        candidate = canonical(dict(format='hq-dfc24-fresh-v1',mode=args.mode,controls=CONTROLS,seeds=SEEDS,rates=RATES,
            settings=settings_for_mode(args.mode,args.device),source_revision=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=hashes,
            protocol_sha256=digest(ROOT/PROTOCOL),input_identity=inputs,runtime=runtime(device,args.gpu_memory_fraction),
            resource_profile_sha256=digest(args.profile),data_root=str(args.data_root),
            cohorts={str(seed):cohort_record(data[seed]) for seed in SEEDS},
            replay_split='validation' if args.mode=='rehearsal' else 'test',budget_seconds=args.budget_seconds,
            budget_sha256=digest(args.work_dir/'budget.json'),
            selection_policy='Two rates on seed42; maximum validation foregroundIoU, exact tie lower rate; earliest strict best epoch including0.',
            test_policy='All16 fresh fits and strict audits before12 selected replays; rehearsal validation only, full exposed test descriptive.'))
        plan_path = args.work_dir/'plan.json'
        if plan_path.exists():
            require(args.resume,'Existing plan requires resume')
            plan = read_json(plan_path)
            require({k:v for k,v in plan.items() if k!='source_revision'} ==
                    {k:v for k,v in candidate.items() if k!='source_revision'}, 'Frozen source/input/runtime/profile/cohort/settings plan differs')
        else:
            require(not args.resume,'Resume requires the original plan')
            _write_once(plan_path,candidate); plan = candidate
        retained = args.work_dir/'resource_profile.json'
        if retained.exists():require(digest(retained)==plan['resource_profile_sha256'],'Retained profile differs')
        else:
            require(not readonly,'Completed resume is missing its retained profile')
            retained.write_bytes(args.profile.read_bytes())
        cache = _existing_cells(args,data)
        if readonly:
            require(len(cache)==16,'Completed resume has missing cells')
        entries = []
        def cell(kind,seed,rate):
            key = kind,seed,rate
            require(source_hashes() == hashes and runtime(device,args.gpu_memory_fraction)==plan['runtime'],
                    'Source/runtime changed before fitting')
            if key in cache:
                entry = cache[key]
            else:
                require(not readonly,'Completed resume cannot fit missing cells')
                directory = _cell_directory(args.work_dir,kind,seed,rate)
                require(directory.resolve().is_relative_to(args.work_dir),'Fit path escapes output')
                row = fit_cell(kind,seed,rate,args,data[seed],directory,device)
                entry = _entry(args.work_dir,directory,row); cache[key] = entry
            entries.append(entry); _progress(args.work_dir/'fit_progress.json',entries)
        for kind in CONTROLS:
            for rate in RATES:cell(kind,42,rate)
        selection = select_rates(entries)
        require(set(cache) <= expected_grid(selection),'Unexpected cached cell outside selected grid')
        _write_once(args.work_dir/'selection.json',selection)
        for kind in CONTROLS:
            for seed in SEEDS[1:]:cell(kind,seed,selection[kind]['learning_rate'])
        paired_audit([entry['record'] for entry in entries])
        require(source_hashes() == hashes and pilot.verify_inputs(args.data_root) == inputs,'Source/input changed before replay boundary')
        audits = []
        for entry in entries:
            model,audit = restore_checkpoint(args.work_dir/entry['directory']/'checkpoint.pt',entry['record'])
            require(audit['checkpoint_sha256']==entry['checkpoint_sha256'],'Checkpoint bytes changed')
            audits.append(dict(id=entry['record']['id'],seed=entry['record']['seed'],learning_rate=entry['record']['learning_rate'],
                               origin='fresh',**audit)); del model
        _write_once(args.work_dir/'all_fits.json',entries)
        _write_once(args.work_dir/'checkpoint_audit.json',audits)
        boundary = dict(format='hq-dfc24-fresh-boundary-v1',completed=True,fits=16,selected=12,replay_split=plan['replay_split'],
            plan_sha256=digest(plan_path),all_fits_sha256=digest(args.work_dir/'all_fits.json'),
            selection_sha256=digest(args.work_dir/'selection.json'),checkpoint_audit_sha256=digest(args.work_dir/'checkpoint_audit.json'))
        _write_once(args.work_dir/'fit_boundary.json',boundary)
        records = []
        replay_paths = set()
        for entry in entries:
            row = entry['record']
            if row['learning_rate'] != selection[row['kind']]['learning_rate']:continue
            replay = args.work_dir/'replays'/row['id']/('seed'+str(row['seed'])+'.json')
            replay_paths.add(replay)
            intent = args.work_dir/'replay_intents'/row['id']/('seed'+str(row['seed'])+'.json')
            failure = args.work_dir/'replay_failures'/row['id']/('seed'+str(row['seed'])+'.json')
            require(not failure.exists(),'Failed replay evidence preserved')
            if plan['replay_split']=='validation':
                dataset = data[row['seed']]['val']
                names = plan['cohorts'][str(row['seed'])]['validation_names']
            else:
                cfg = dict(row['settings'],augment=False)
                _,_,_,images,masks = tc.resolve_data_dirs('combined',str(args.data_root))
                dataset = tc.load_dataset(images,masks,cfg)
                require(dataset.norm_provenance == row['normalization'],'Test normalization differs')
                names = list(dataset.names)
                require(names == read_json(args.data_root/'splits/flood_combined.json')['test'] and len(names)==205,
                        'Full ordered test cohort differs')
            if replay.exists():
                require(intent.is_file() and read_json(intent) == dict(binding=read_json(replay)['binding']),
                        'Completed replay intent missing or differs')
                selected = validate_replay(read_json(replay),entry,digest(plan_path),names,plan['replay_split'])
            else:
                require(not readonly,'Completed resume cannot infer missing replays')
                require(not intent.exists(),'Unfinished replay intent preserved; inference cannot retry')
                binding = replay_binding(entry,digest(plan_path),names,plan['replay_split'])
                _write_once(intent,dict(binding=binding))
                try:
                    model,_ = restore_checkpoint(args.work_dir/entry['directory']/'checkpoint.pt',row,device)
                    scores = hq.evaluate(model,tc.make_loader(dataset,args.batch_size,False,row['seed']),device)
                    selected = dict(copy.deepcopy(row),replay=scores,replay_names=list(names),replay_split=plan['replay_split'],
                                    checkpoint=dict(origin='fresh',path=binding['checkpoint_path']))
                    payload = dict(binding=binding,record=selected)
                    validate_replay(payload,entry,digest(plan_path),names,plan['replay_split'])
                    _write_once(replay,payload); del model
                except BaseException as error:
                    _write_once(failure,dict(binding=binding,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
                    raise
            records.append(selected)
        require(set((args.work_dir/'replays').rglob('*.json')) == replay_paths,'Unexpected selected replay files')
        require({args.work_dir/'replays'/p.relative_to(args.work_dir/'replay_intents')
                 for p in (args.work_dir/'replay_intents').rglob('*.json')} == replay_paths,
                'Unexpected selected replay intents')
        require(source_hashes() == hashes and pilot.verify_inputs(args.data_root) == inputs
                and runtime(device,args.gpu_memory_fraction)==plan['runtime'],'Final source/input/runtime guard differs')
        summary = dict(plan=plan,selection=selection,records=records,analysis=analysis(records,plan['replay_split']))
        _write_once(args.work_dir/'summary.json',summary)
        print('FRESH HQ COMPLETE:',args.mode,'16 fresh fits,12 selected',plan['replay_split'],'replays',flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('profile','run'))
    parser.add_argument('--mode',choices=tuple(MODE_SPECS),required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--gpu-memory-fraction',type=float)
    parser.add_argument('--budget-seconds',type=float,required=True)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget_seconds) or args.budget_seconds<=0:
        parser.error('--budget-seconds must be positive and finite')
    if args.device=='cuda' and (args.gpu_memory_fraction is None or
            not math.isfinite(args.gpu_memory_fraction) or not 0<args.gpu_memory_fraction<=1):
        parser.error('CUDA requires an explicit --gpu-memory-fraction in(0,1]')
    if args.phase=='run' and args.profile is None:parser.error('run requires --profile')
    if args.phase=='profile' and args.resume:parser.error('profiles do not resume')
    import torch
    device = torch.device('cuda:0' if args.device=='cuda' else 'cpu')
    return profile(args,device) if args.phase=='profile' else run(args,device)


if __name__=='__main__':
    main()
