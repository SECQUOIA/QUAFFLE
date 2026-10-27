#!/usr/bin/env python3
"""Audit paired training controls against the retained original spatial pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.collect_real_spatial_results import audit_run


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values), n=len(values))


def compare(baseline, controls):
    context = controls['plan']['experiment_context']
    decision = context['validation_control_decision']
    mode = decision['selected_mode']
    if not decision['ready'] or mode not in ('global', 'patch4x4'):
        raise ValueError('Missing completed validation-only control decision')
    settings = baseline['plan']['settings']
    for key in ('seeds', 'learning_rates', 'epochs', 'eval_every', 'fit_cap', 'batch_size', 'image_size', 'device'):
        if controls['plan']['settings'][key] != settings[key]:
            raise ValueError('Control budget differs from the baseline: ' + key)
    if context['baseline_input_identity'] != baseline['plan']['input_identity']:
        raise ValueError('Control inputs differ from the baseline')
    if context['baseline_source_revision'] != baseline['plan']['source_revision']:
        raise ValueError('Control context references another baseline producer')
    for key in ('python', 'packages', 'device', 'gpu', 'gpu_allocator_fraction', 'threads'):
        if controls['plan'][key] != baseline['plan'][key]:
            raise ValueError('Recorded execution environment differs: ' + key)
    seeds = settings['seeds']
    if len(seeds) != 3 or set(seeds) != {42, 77, 123}:
        raise ValueError('Expected the three paired study seeds')
    groups = {}
    for row in baseline['records'] + controls['records']:
        group = groups.setdefault(row['id'], {})
        if row['seed'] in group:
            raise ValueError('Duplicate model/seed result')
        group[row['seed']] = row
    if any(set(group) != set(seeds) for group in groups.values()):
        raise ValueError('Incomplete paired seed coverage')
    quantum = 'quantum_' + mode
    frozen = 'quantum_frozen_' + mode
    noent = 'quantum_no_entanglement_' + mode
    if {row['id'] for row in controls['records']} != {frozen, noent}:
        raise ValueError('Unexpected control model set')
    checks = []
    for seed in seeds:
        reference = groups[quantum][seed]
        for group in groups.values():
            for key in ('fit_names', 'val_names', 'test_names', 'normalization'):
                if group[seed][key] != reference[key]:
                    raise ValueError('Unpaired comparison: ' + key)
        for name, control in ((frozen, 'frozen'), (noent, 'no_entanglement')):
            row = groups[name][seed]
            original_settings = {key: value for key, value in reference['settings'].items()
                                 if key not in ('learning_rate', 'q_training_control')}
            control_settings = {key: value for key, value in row['settings'].items()
                                if key not in ('learning_rate', 'q_training_control')}
            if original_settings != control_settings:
                raise ValueError('Control configuration differs beyond the intended intervention and selected rate')
            if (row['initial_parameter_sha256'] != reference['initial_parameter_sha256']
                    or row['total_params'] != reference['total_params']):
                raise ValueError('Control initialization or total count differs from the original quantum model')
            if row['settings']['q_training_control'] != control:
                raise ValueError('Control settings disagree with the model label')
        row = groups[frozen][seed]
        movement = row['diagnostics']['movement']
        if any(movement[key]['l2'] != 0 for key in ('circuit_angles', 'input_scales')):
            raise ValueError('Frozen circuit parameters changed')
        cfg = row['settings']
        core = (cfg['quantum_channels'] // cfg['n_qubits']) * cfg['n_qubits'] * cfg['q_layers'] * 4
        core *= 1 if cfg['q_share_blocks'] else 2
        if row['trainable_params'] != row['total_params'] - core:
            raise ValueError('Frozen trainable parameter count is incorrect')
        if groups[noent][seed]['trainable_params'] != reference['total_params']:
            raise ValueError('No-entanglement control unexpectedly freezes parameters')
        checks.append(dict(seed=seed, same_initial_parameters=True, same_cohorts_and_normalization=True,
                           frozen_core_unchanged=True, frozen_backbone_selected_moved=movement['backbone']['l2'] > 0))
    rows = []
    for name, group in groups.items():
        total = {row['total_params'] for row in group.values()}
        trainable = {row.get('trainable_params', row['total_params']) for row in group.values()}
        if len(total) != 1 or len(trainable) != 1:
            raise ValueError('Parameter counts vary across seeds')
        rates = {row['learning_rate'] for row in group.values()}
        if len(rates) != 1:
            raise ValueError('Selected rate varies across seeds')
        rows.append(dict(id=name, total_params=total.pop(), trainable_params=trainable.pop(),
            learning_rate=rates.pop(),
            **{key: stats([group[seed]['test'][key] for seed in seeds])
               for key in ('iou', 'optical_subset_iou', 'sar_subset_iou')}))
    contrasts = []
    for left, right in ((quantum, frozen), (quantum, noent), (noent, frozen), (frozen, 'mlp_' + mode)):
        differences = {str(seed): groups[left][seed]['test']['iou'] - groups[right][seed]['test']['iou'] for seed in seeds}
        contrasts.append(dict(left=left, right=right, differences=differences,
            wins=sum(value > 0 for value in differences.values()), **stats(list(differences.values()))))
    return dict(selected_mode=mode, rows=rows, paired_iou=contrasts, checks=checks,
        validation_decision=decision,
        scope='Same dataset, three paired training/split seeds; descriptive differences, not significance or equivalence claims')


def report(comparison):
    lines = ['# Quantum training-control comparison', '', comparison['scope'], '',
        '| Model | Total parameters | Trainable parameters | Selected LR | Test IoU mean ± sample SD | Optical IoU | SAR IoU |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in comparison['rows']:
        lines.append(f"| {row['id']} | {row['total_params']:,} | {row['trainable_params']:,} | {row['learning_rate']:g} | "
                     f"{row['iou']['mean']:.4f} ± {row['iou']['std']:.4f} | "
                     f"{row['optical_subset_iou']['mean']:.4f} | {row['sar_subset_iou']['mean']:.4f} |")
    lines += ['', '| Paired contrast | Mean ± sample SD, IoU percentage points | Wins |', '| --- | ---: | ---: |']
    for row in comparison['paired_iou']:
        lines.append(f"| {row['left']} − {row['right']} | {100*row['mean']:+.3f} ± {100*row['std']:.3f} | {row['wins']}/3 |")
    lines += ['', 'Control choice used validation-only records. Every control received the original rate grid and budget.',
        'Each model chose its learning rate independently from that grid; the comparisons include those selected-rate differences.',
        'Frozen circuits retain their initial angles and input scales while the classical model trains. No-entanglement controls remove CNOT gates and train all parameters.',
        'Initial parameters, cohorts, normalization and parameter counts were checked against the original quantum model for every seed.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--control-run', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    for name, expected in json.loads((args.baseline_dir / 'artifact_sha256.json').read_text()).items():
        if hashlib.sha256((args.baseline_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Retained baseline artifact changed: ' + name)
    controls, _, _, _ = audit_run(args.control_run)
    if controls['plan']['experiment_context']['baseline_plan_sha256'] != hashlib.sha256(
            (args.baseline_dir / 'plan.json').read_bytes()).hexdigest():
        raise ValueError('Controls reference a different baseline plan')
    baseline = json.loads((args.baseline_dir / 'summary.json').read_text())
    result = compare(baseline, controls)
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['input_summary_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
        ('baseline', args.baseline_dir / 'summary.json'), ('controls', args.control_run / 'summary.json'))}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.out_dir / name for name in ('comparison.json', 'COMPARISON.md')]
    if any(path.exists() for path in paths):
        parser.error('Refusing to overwrite an existing comparison')
    paths[0].write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    paths[1].write_text(report(result))
    print(report(result))
    return result


if __name__ == '__main__':
    main()
