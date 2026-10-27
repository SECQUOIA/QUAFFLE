#!/usr/bin/env python3
"""Complete the paired rate factorial, then study longer training locally.

The two phases have separate output roots and complete every validation fit
before any test replay. Tests are descriptive: this dataset's test split has
already informed the research direction. Historical fits retain their original
producer identity and are reused only after a strict current-checkpoint audit.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_real_spatial_pilot as pilot

SEEDS = (42, 77, 123)
RATES = (0.001, 0.003)
QUANTUM = ('quantum_global', 'quantum_no_entanglement_global')
SOURCE_FILES = (
    'scripts/run_local_rate_stability.py', 'scripts/run_real_spatial_pilot.py',
    'scripts/run_parameter_budget_comparison.py', 'utilsTorch/unetTorch.py',
    'utilsTorch/utilsTorch.py', 'tools/training_common.py',
    'tools/checkpoint_evaluation.py', 'flood_data.py', 'splits/norm_stats.json',
)
RUNTIME_KEYS = ('python', 'packages', 'device', 'gpu', 'gpu_allocator_fraction', 'threads')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    """Normalize JSON tuples and the legacy implicit trainable default."""
    result = json.loads(json.dumps(value))
    if isinstance(result, dict):
        result.setdefault('q_training_control', 'trainable')
    return result


def specs_for(phase):
    from scripts.run_parameter_budget_comparison import base_config
    cfg = dict(base_config(8, 1), dataset='combined')
    specs = [dict(id=QUANTUM[0], model_type='quantum',
                  config=dict(cfg, q_training_control='trainable')),
             dict(id=QUANTUM[1], model_type='quantum',
                  config=dict(cfg, q_training_control='no_entanglement'))]
    if phase == 'long120':
        specs.append(dict(id='mlp_patch4x4', model_type='ablation',
                          config=dict(cfg, q_spatial_mode='patch4x4')))
    return specs


def cell_key(row):
    return row['id'], row['seed'], row['learning_rate']


def expected_cells(specs, seeds=SEEDS, rates=RATES):
    return set(itertools.product([spec['id'] for spec in specs], seeds, rates))


def audit_factorial(records, specs, seeds=SEEDS, rates=RATES, *, test=False):
    """Reject incomplete/duplicated cells and mismatched paired identities."""
    cells = {}
    for row in records:
        key = cell_key(row)
        if key in cells:
            raise ValueError('Duplicate factorial cell: ' + str(key))
        cells[key] = row
    if set(cells) != expected_cells(specs, seeds, rates):
        raise ValueError('Incomplete or unexpected factorial cells')
    for seed in seeds:
        paired = [row for row in records if row['seed'] == seed]
        for key in ('fit_names', 'val_names', 'normalization') + (('test_names',) if test else ()):
            if any(row[key] != paired[0][key] for row in paired):
                raise ValueError('Unpaired factorial identity: ' + key)
        quantum = [row for row in paired if row['id'] in QUANTUM]
        if len({row['initial_parameter_sha256'] for row in quantum}) != 1:
            raise ValueError('Quantum interventions/rates have different initialization')
        for spec in specs:
            group = [row for row in paired if row['id'] == spec['id']]
            if len({row['initial_parameter_sha256'] for row in group}) != 1:
                raise ValueError('Rate candidates have different initialization')
    return cells


def frozen_plan(path, candidate, resume):
    """Keep the original producer revision when executable bytes are unchanged."""
    path = Path(path)
    if resume:
        previous = json.loads(path.read_text())
        for key in candidate:
            if key != 'source_revision' and previous.get(key) != candidate[key]:
                raise ValueError('Resume inputs differ from frozen plan: ' + key)
        return previous
    if path.parent.exists() and any(path.parent.iterdir()):
        raise ValueError('Use a new/empty output directory or identical --resume')
    pilot.write_json(path, candidate)
    return candidate


def expected_settings(spec, seed, rate, args, data):
    return canonical(dict(spec['config'], dataset='combined', data_root=str(args.data_root),
        seed=seed, image_size=args.image_size, max_samples=None, batch_size=args.batch_size,
        num_epochs=args.epochs, learning_rate=rate, optimization_fit_cap=args.fit_cap,
        _resolved_norm_provenance=dict(data['normalization'])))


def require_finite(value, path='fit'):
    """JSON readers accept NaN/Infinity; reject them before reusing evidence."""
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite(child, path + '.' + key)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite(child, path + '[' + str(index) + ']')
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError('Nonfinite recorded value: ' + path)


def audit_record(row, spec, seed, rate, args, data):
    require_finite(row)
    if cell_key(row) != (spec['id'], seed, rate) or row['model_type'] != spec['model_type']:
        raise ValueError('Fit record model/seed/rate differs from planned cell')
    actual = canonical(row['settings'])
    expected = expected_settings(spec, seed, rate, args, data)
    # The root path is storage metadata; bytes and names are independently bound.
    actual.pop('data_root', None)
    expected.pop('data_root', None)
    if actual != expected:
        raise ValueError('Fit configuration differs from planned cell')
    expected_identity = dict(fit_indices=data['fit'], val_indices=data['validation'],
        fit_names=[data['dataset'].names[i] for i in data['fit']],
        val_names=[data['dataset'].names[i] for i in data['validation']],
        normalization=data['normalization'], n_global_fit=len(data['all_fit']))
    if any(row[key] != value for key, value in expected_identity.items()):
        raise ValueError('Fit cohort or normalization differs from current inputs')
    epochs = [entry['epoch'] for entry in row['history']]
    if epochs != list(range(args.epochs + 1)):
        raise ValueError('Fit does not contain the complete planned epoch history')
    evaluations = [entry for entry in row['history'] if 'val' in entry]
    expected_evaluations = [epoch for epoch in range(args.epochs + 1)
                            if epoch == 0 or epoch % args.eval_every == 0 or epoch == args.epochs]
    if [entry['epoch'] for entry in evaluations] != expected_evaluations:
        raise ValueError('Fit validation evaluation schedule differs from the planned budget')
    chosen = max(evaluations, key=lambda entry: (entry['val']['iou'], -entry['epoch']))
    if row['best_epoch'] != chosen['epoch'] or row['val'] != chosen['val']:
        raise ValueError('Fit checkpoint selection differs from validation history')


def discover_reuse(roots, specs):
    """Bind only planned validation records; never import historical test scores."""
    allowed = expected_cells(specs)
    result = {}
    for root in map(Path, roots):
        plan_path = root / 'plan.json'
        plan = json.loads(plan_path.read_text())
        for metrics in sorted(root.rglob('metrics.json')):
            row = json.loads(metrics.read_text())
            key = cell_key(row)
            if key not in allowed:
                continue
            if key in result:
                raise ValueError('Duplicate historical factorial cell: ' + str(key))
            if 'test' in row or 'test_names' in row:
                raise ValueError('Historical reuse must read validation-only records')
            checkpoint = metrics.parent / 'checkpoint.pt'
            result[key] = dict(record=row, directory=str(metrics.parent.resolve()),
                metrics_sha256=digest(metrics), checkpoint_sha256=digest(checkpoint),
                producer_source_revision=plan['source_revision'],
                producer_source_hashes=plan['source_hashes'],
                producer_plan_sha256=digest(plan_path), producer_runtime={k: plan[k] for k in RUNTIME_KEYS},
                producer_input_identity=plan['input_identity'])
    return result


def initial_hash(model):
    return hashlib.sha256(b''.join(name.encode() + parameter.detach().cpu().numpy().tobytes()
        for name, parameter in model.named_parameters())).hexdigest()


def audit_checkpoint(directory, row, spec, args, data, device, *, replay_validation):
    """Strictly restore finite weights and verify saved metadata/init/validation."""
    import torch
    from tools import training_common as tc
    from tools.checkpoint_evaluation import load_checkpoint_model
    from utilsTorch.utilsTorch import (ClassicalAblationUNetSegmentation,
                                      QVUNetSegmentation, evaluate_model)
    checkpoint = Path(directory) / 'checkpoint.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['best_epoch'] != row['best_epoch'] or canonical(saved['settings']) != canonical(row['settings']):
        raise ValueError('Checkpoint metadata differs from validation record')
    if saved['model_type'] != spec['model_type'] or not saved.get('best_state_dict'):
        raise ValueError('Checkpoint model type or validation-selected weights missing')
    if any(not bool(torch.isfinite(value).all()) for value in saved['best_state_dict'].values()):
        raise ValueError('Checkpoint contains nonfinite weights')
    tc.seed_everything(row['seed'])
    cls = QVUNetSegmentation if spec['model_type'] == 'quantum' else ClassicalAblationUNetSegmentation
    initial = tc.build_model(cls, row['settings'])
    if initial_hash(initial) != row['initial_parameter_sha256']:
        raise ValueError('Current model initialization differs from historical producer')
    if sum(p.numel() for p in initial.parameters()) != row['total_params']:
        raise ValueError('Checkpoint parameter count differs from record')
    del initial, saved
    model, _, _ = load_checkpoint_model(checkpoint, data_root=args.data_root, device=device)
    if replay_validation:
        replay = evaluate_model(model, tc.make_loader(data['val'], args.batch_size, False, row['seed']), device)
        count_keys = [sensor + '_subset_' + count for sensor in ('optical', 'sar')
                      for count in ('tp', 'fp', 'fn', 'tn', 'n_samples')]
        for key in count_keys + ['iou']:
            if replay[key] != row['val'][key]:
                raise ValueError('Current checkpoint validation replay differs: ' + key)
    del model
    return dict(strict_restore=True, finite_weights=True, initial_parameters_equal=True,
                validation_replayed=replay_validation,
                validation_counts_equal=True if replay_validation else None)


def summarize(records, specs, *, test=False):
    cells = audit_factorial(records, specs, test=test)
    metric = 'test' if test else 'val'
    rows = []
    for spec in specs:
        for rate in RATES:
            runs = [cells[spec['id'], seed, rate] for seed in SEEDS]
            values = [row[metric]['iou'] for row in runs]
            row = dict(id=spec['id'], learning_rate=rate, mean=statistics.mean(values),
                       std=statistics.stdev(values), n=len(values))
            for key in ('optical_subset_iou', 'sar_subset_iou'):
                sensor_values = [run[metric][key] for run in runs]
                row[key] = dict(mean=statistics.mean(sensor_values), std=statistics.stdev(sensor_values), n=len(sensor_values))
            rows.append(row)
    contrasts = []
    comparisons = [(QUANTUM[0], QUANTUM[1])]
    if any(spec['id'] == 'mlp_patch4x4' for spec in specs):
        comparisons += [(name, 'mlp_patch4x4') for name in QUANTUM]
    for left, right in comparisons:
        for rate in RATES:
            differences = {str(seed): cells[left, seed, rate][metric]['iou'] -
                           cells[right, seed, rate][metric]['iou'] for seed in SEEDS}
            values = list(differences.values())
            contrasts.append(dict(left=left, right=right, learning_rate=rate,
                differences=differences, mean=statistics.mean(values), std=statistics.stdev(values),
                wins=sum(value > 0 for value in values), n=len(values), unit='IoU'))
    trajectories = []
    for row in records:
        available = [point for point in row['history'] if 'val' in point]
        early = max((point for point in available if point['epoch'] <= 60),
                    key=lambda point: (point['val']['iou'], -point['epoch']))
        losses = [point['train_loss'] for point in row['history'] if point['train_loss'] is not None]
        trajectories.append(dict(id=row['id'], seed=row['seed'], learning_rate=row['learning_rate'],
            best_epoch=row['best_epoch'], best_val_iou=row['val']['iou'],
            best_at_or_before60_epoch=early['epoch'], best_at_or_before60_iou=early['val']['iou'],
            later_validation_gain=row['val']['iou'] - early['val']['iou'],
            max_train_loss=max(losses), median_train_loss=statistics.median(losses),
            max_over_median_train_loss=max(losses) / statistics.median(losses),
            gradient_norm_extremes=row['diagnostics']['gradient_norms'],
            late_selection=row['best_epoch'] >= row['history'][-1]['epoch'] - 2))
    return dict(metric=metric, rows=rows, paired_iou=contrasts, trajectories=trajectories,
        scope='Three paired seeds; fixed-rate descriptive comparisons, not a significance or equivalence test')


def format_report(summary):
    result = summary['validation_analysis']
    lines = ['# Local learning-rate and stability follow-up', '',
        f"Phase: {summary['plan']['phase']}; {len(summary['records'])} factorial cells.",
        'All validation fits completed before test replay. Test results are descriptive because the original test split was already consulted.',
        'Normalization uses all global fit images, validation excluded; optimization uses 128 images. This is not pure few-shot learning.', '',
        '| Model | Fixed LR | Validation IoU mean ± sample SD |', '| --- | ---: | ---: |']
    for row in result['rows']:
        lines.append(f"| {row['id']} | {row['learning_rate']:g} | {row['mean']:.4f} ± {row['std']:.4f} |")
    if 'test_analysis' in summary:
        lines += ['', '| Model | Fixed LR | Descriptive test IoU mean ± sample SD | Optical IoU | SAR IoU |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for row in summary['test_analysis']['rows']:
            lines.append(f"| {row['id']} | {row['learning_rate']:g} | {row['mean']:.4f} ± {row['std']:.4f} | "
                         f"{row['optical_subset_iou']['mean']:.4f} | {row['sar_subset_iou']['mean']:.4f} |")
    for label, analysis in [('Validation', result), ('Descriptive test', summary.get('test_analysis'))]:
        if analysis is None:
            continue
        lines += ['', f'| {label} paired contrast | Fixed LR | IoU percentage points, mean ± sample SD | Wins |',
                  '| --- | ---: | ---: | ---: |']
        for row in analysis['paired_iou']:
            lines.append(f"| {row['left']} − {row['right']} | {row['learning_rate']:g} | "
                         f"{100*row['mean']:+.3f} ± {100*row['std']:.3f} | {row['wins']}/{row['n']} |")
    lines += ['', 'Every cell, full learning curve, initialization identity and provenance are retained. Training times are not controlled performance benchmarks.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['matched60', 'long120'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path)
    parser.add_argument('--control-run', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args(argv)
    if args.phase == 'matched60' and (args.baseline_run is None or args.control_run is None):
        parser.error('matched60 requires both historical run roots')
    if args.phase == 'long120' and (args.baseline_run is not None or args.control_run is not None):
        parser.error('long120 trains every cell fresh; historical reuse is forbidden')
    import torch
    from tools import training_common as tc
    from tools.checkpoint_evaluation import load_checkpoint_model
    from utilsTorch.utilsTorch import evaluate_model
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.12)
    device = torch.device(args.device)
    args.epochs = 60 if args.phase == 'matched60' else 120
    args.eval_every, args.fit_cap, args.batch_size, args.image_size = 2, 128, 16, 64
    specs = specs_for(args.phase)
    inputs = pilot.verify_inputs(args.data_root)
    reused = discover_reuse([args.baseline_run, args.control_run], specs) if args.phase == 'matched60' else {}
    if args.phase == 'matched60' and len(reused) != 8:
        raise ValueError('The historical comparison must supply exactly eight planned fits')
    runtime = dict(python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['torch', 'pennylane', 'numpy', 'rasterio']},
        device=str(device), gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
        gpu_allocator_fraction=0.12 if args.device == 'cuda' else None, threads=2)
    for item in reused.values():
        if item['producer_runtime'] != runtime or item['producer_input_identity'] != inputs:
            raise ValueError('Historical input bytes or execution runtime differ')
    candidate = dict(phase=args.phase, specs=specs, seeds=list(SEEDS), learning_rates=list(RATES),
        epochs=args.epochs, eval_every=2, fit_cap=128, batch_size=16, image_size=64,
        validation_only=args.validation_only,
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes={path: digest(ROOT / path) for path in SOURCE_FILES}, input_identity=inputs,
        historical_fits=[dict(id=key[0], seed=key[1], learning_rate=key[2],
                             **{k: v for k, v in item.items() if k != 'record'}) for key, item in sorted(reused.items())],
        normalization_policy='All global fit images; validation excluded; optimization capped separately',
        test_policy='All factorial fits finish first. Previously exposed test split: descriptive only.', **runtime)
    candidate = json.loads(json.dumps(candidate))
    plan = frozen_plan(args.work_dir / 'plan.json', candidate, args.resume)
    data_by_seed = {seed: pilot.load_training_data(args, seed) for seed in SEEDS}
    records, directories, evidence = [], {}, []
    # Audit every reused cell before launching any new fit.
    for spec in specs:
        for seed, rate in itertools.product(SEEDS, RATES):
            key = spec['id'], seed, rate
            if key not in reused:
                continue
            item, data = reused[key], data_by_seed[seed]
            row = copy.deepcopy(item['record'])
            audit_record(row, spec, seed, rate, args, data)
            audit = audit_checkpoint(item['directory'], row, spec, args, data, device, replay_validation=True)
            records.append(row)
            directories[key] = Path(item['directory'])
            evidence.append(dict(id=key[0], seed=seed, learning_rate=rate, origin='historical',
                producer_source_revision=item['producer_source_revision'],
                checkpoint_sha256=item['checkpoint_sha256'], metrics_sha256=item['metrics_sha256'], **audit))
            pilot.write_json(args.work_dir / 'reuse_audit.json', evidence)
    for spec in specs:
        for seed, rate in itertools.product(SEEDS, RATES):
            key = spec['id'], seed, rate
            if key in directories:
                continue
            directory = args.work_dir / 'fits' / spec['id'] / str(rate) / ('seed' + str(seed))
            row = pilot.fit_one(spec, seed, rate, args, data_by_seed[seed], directory, device)
            audit_record(row, spec, seed, rate, args, data_by_seed[seed])
            audit = audit_checkpoint(directory, row, spec, args, data_by_seed[seed], device, replay_validation=True)
            records.append(row)
            directories[key] = directory
            evidence.append(dict(id=key[0], seed=seed, learning_rate=rate, origin='new',
                producer_source_revision=plan['source_revision'], checkpoint_sha256=digest(directory / 'checkpoint.pt'),
                metrics_sha256=digest(directory / 'metrics.json'), **audit))
    records.sort(key=cell_key)
    audit_factorial(records, specs)
    pilot.write_json(args.work_dir / 'validation_fits.json', records)
    pilot.write_json(args.work_dir / 'checkpoint_audit.json', evidence)
    validation_analysis = summarize(records, specs)
    summary = dict(plan=plan, validation_analysis=validation_analysis)
    # This is the only path that accesses trained-model test data.
    if not args.validation_only:
        for row in records:
            directory = directories[cell_key(row)]
            model, cfg, kind = load_checkpoint_model(directory / 'checkpoint.pt', data_root=args.data_root, device=device)
            _, _, _, images, masks = tc.resolve_data_dirs('combined', str(args.data_root))
            dataset = tc.load_dataset(images, masks, cfg)
            if dataset.norm_provenance != row['normalization']:
                raise ValueError('Test normalization differs from the training checkpoint')
            loader = tc.make_loader(dataset, args.batch_size, False, row['seed'])
            row['test'] = evaluate_model(model, loader, device)
            row['test_names'] = list(dataset.names)
            row['knockout'] = tc.knockout_eval(model, loader, device, evaluate_model, knockout_seed=row['seed']) if kind != 'classical' else None
            row['checkpoint_sha256'] = digest(directory / 'checkpoint.pt')
            path = args.work_dir / 'test_replay' / row['id'] / str(row['learning_rate']) / ('seed' + str(row['seed']) + '.json')
            pilot.write_json(path, row)
            del model
        summary['test_analysis'] = summarize(records, specs, test=True)
    summary['records'] = records
    summary['checkpoint_audit'] = evidence
    pilot.write_json(args.work_dir / 'summary.json', summary)
    (args.work_dir / 'report.md').write_text(format_report(summary))
    print('LOCAL FOLLOW-UP COMPLETE', str(args.work_dir / 'summary.json'), flush=True)
    return summary


if __name__ == '__main__':
    main()
