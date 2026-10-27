#!/usr/bin/env python3
"""Audit a completed spatial pilot and retain its small, reviewable artifacts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_real_spatial_pilot import aggregate, choose_rates


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(run):
    summary = read(run / 'summary.json')
    plan = read(run / 'plan.json')
    assert summary['plan'] == plan
    for name, expected in plan['source_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Executable source differs from the producing run: ' + name)
    settings = plan['settings']
    specs = {spec['id'] for spec in plan['specs']}
    trials = []
    for path in sorted((run / 'tuning').glob('*/*/seed42/metrics.json')):
        row = read(path)
        assert 'test' not in row
        trials.append(row)
    assert len(trials) == len(specs) * len(settings['learning_rates'])
    assert {(row['id'], row['learning_rate']) for row in trials} == {
        (name, rate) for name in specs for rate in settings['learning_rates']}
    chosen = choose_rates(trials)
    assert summary['selection'] == {name: row['learning_rate'] for name, row in chosen.items()}
    records = summary['records']
    assert len(records) == len(specs) * len(settings['seeds'])
    if {(row['id'], row['seed']) for row in records} != {
            (name, seed) for name in specs for seed in settings['seeds']}:
        raise ValueError('Selected results do not cover the planned model/seed pairs')
    checkpoints = {}
    flat = []
    for row in records:
        assert row['learning_rate'] == chosen[row['id']]['learning_rate']
        evaluated = [point for point in row['history'] if 'val' in point]
        best = max(evaluated, key=lambda point: point['val']['iou'])
        assert row['best_epoch'] == best['epoch'] and row['val'] == best['val']
        assert len(row['history']) == settings['epochs'] + 1
        assert set(row['fit_names']).isdisjoint(row['val_names'])
        assert len(row['fit_names']) <= settings['fit_cap']
        checkpoint = run / row['checkpoint']
        fit_record = {key: value for key, value in row.items()
                      if key not in ('test', 'test_names', 'checkpoint', 'knockout')}
        if read(checkpoint.parent / 'metrics.json') != fit_record:
            raise ValueError('Selected summary differs from its saved validation-only fit record')
        if read(checkpoint.parent / 'selected_test.json') != row:
            raise ValueError('Selected summary differs from its saved test replay')
        checkpoints[row['checkpoint']] = digest(checkpoint)
        test = row['test']
        totals = {name: sum(test[sensor + '_subset_' + name] for sensor in ('optical', 'sar'))
                  for name in ('tp', 'fp', 'fn', 'tn')}
        denominator = totals['tp'] + totals['fp'] + totals['fn']
        recombined = totals['tp'] / denominator if denominator else 0
        assert math.isclose(test['iou'], recombined, abs_tol=1e-12)
        assert sum(test[sensor + '_subset_n_samples'] for sensor in ('optical', 'sar')) == len(row['test_names'])
        assert all(0 <= test[key] <= 1 for key in ('iou', 'dice', 'accuracy', 'auc', 'optical_subset_iou', 'sar_subset_iou'))
        flat.append(dict(id=row['id'], seed=row['seed'], learning_rate=row['learning_rate'],
            total_params=row['total_params'], best_epoch=row['best_epoch'], n_fit=len(row['fit_names']),
            n_val=len(row['val_names']), n_test=len(row['test_names']), val_iou=row['val']['iou'],
            test_iou=test['iou'], optical_test_iou=test['optical_subset_iou'], sar_test_iou=test['sar_subset_iou']))
    assert summary['analysis'] == aggregate(records)
    fit_paths = list((run / 'tuning').glob('*/*/seed42/metrics.json')) + list((run / 'replication').glob('*/seed*/metrics.json'))
    assert len(fit_paths) == len(specs) * (len(settings['learning_rates']) + len(settings['seeds']) - 1)
    # Preserve all trials, not only validation-selected winners.
    fits = [dict(directory=str(path.parent.relative_to(run)), record=read(path)) for path in sorted(fit_paths)]
    assert all('test' not in item['record'] for item in fits)
    expected_fits = {(name, 42, rate) for name in specs for rate in settings['learning_rates']}
    expected_fits.update((name, seed, chosen[name]['learning_rate'])
                         for name in specs for seed in settings['seeds'][1:])
    if {(item['record']['id'], item['record']['seed'], item['record']['learning_rate'])
            for item in fits} != expected_fits:
        raise ValueError('Fit records do not cover the planned tuning and replication grid')
    return summary, fits, flat, checkpoints


def plot_histories(records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(row['id'] for row in records))
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 2.8 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        for row in [item for item in records if item['id'] == name]:
            history = row['history']
            train = [point for point in history if point['train_loss'] is not None]
            val = [point for point in history if 'val' in point]
            label = 'seed ' + str(row['seed'])
            axes[i, 0].plot([point['epoch'] for point in train], [point['train_loss'] for point in train], label=label)
            axes[i, 1].plot([point['epoch'] for point in val], [point['val']['iou'] for point in val], label=label)
        axes[i, 0].set(title=name, xlabel='Epoch', ylabel='Optimization loss')
        axes[i, 1].set(xlabel='Epoch', ylabel='Validation IoU', ylim=(0, 1))
        axes[i, 1].legend()
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out / ('learning_curves.' + extension), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--screen-dir', type=Path, help='Retain the earlier validation-only screen and its stopping reason')
    args = parser.parse_args(argv)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Use a new or empty output directory')
    summary, fits, flat, checkpoints = audit_run(args.run_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ('plan.json', 'summary.json', 'tuning_candidates.json', 'validation_selection.json'):
        shutil.copy2(args.run_dir / name, args.out_dir / name)
    (args.out_dir / 'all_fits.json').write_text(json.dumps(fits, indent=2, allow_nan=False) + '\n')
    (args.out_dir / 'checkpoint_sha256.json').write_text(json.dumps(checkpoints, indent=2) + '\n')
    with (args.out_dir / 'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    shutil.copy2(args.run_dir / 'report.md', args.out_dir / 'README.md')
    if args.screen_dir:
        assert not list(args.screen_dir.rglob('selected_test.json'))
        screen = dict(disposition=read(args.screen_dir / 'screen_disposition.json'),
                      plan=read(args.screen_dir / 'plan.json'),
                      fits=[read(path) for path in sorted(args.screen_dir.rglob('metrics.json'))],
                      partial_histories={str(path.relative_to(args.screen_dir)): read(path)
                                         for path in sorted(args.screen_dir.rglob('history.json'))})
        (args.out_dir / 'validation_screen_30ep.json').write_text(json.dumps(screen, indent=2, allow_nan=False) + '\n')
    plot_histories(summary['records'], args.out_dir)
    audit = dict(producing_source_revision=summary['plan']['source_revision'],
                 input_summary_sha256=digest(args.run_dir / 'summary.json'),
                 collector_source_sha256=digest(Path(__file__)),
                 completed_fit_records=len(fits), selected_checkpoint_hashes=len(checkpoints),
                 checks=['source hashes', 'complete tuning grid', 'validation-only rate selection',
                         'best validation checkpoints', 'paired cohorts and normalization',
                         'sensor-confusion recombination', 'paired aggregation'])
    (args.out_dir / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    files = {str(path.relative_to(args.out_dir)): digest(path) for path in args.out_dir.iterdir() if path.is_file()}
    (args.out_dir / 'artifact_sha256.json').write_text(json.dumps(files, indent=2) + '\n')
    print('Verified retained fits:', len(fits), 'selected checkpoints:', len(checkpoints))


if __name__ == '__main__':
    main()
