#!/usr/bin/env python3
"""Apply the fixed validation-only rule for additional quantum controls."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_real_spatial_pilot import choose_rates


def decide(records, seeds=(42, 77, 123)):
    groups = {}
    for row in records:
        if 'test' in row:
            raise ValueError('The control decision accepts validation-only fit records')
        group = groups.setdefault(row['id'], {})
        if row['seed'] in group:
            raise ValueError('Duplicate selected model/seed')
        group[row['seed']] = row
    modes = {}
    for mode in ('global', 'patch4x4'):
        names = ['quantum_' + mode, 'mlp_' + mode]
        if any(set(groups.get(name, {})) != set(seeds) for name in names):
            modes[mode] = dict(ready=False)
            continue
        quantum, mlp = (groups[name] for name in names)
        differences = {str(seed): quantum[seed]['val']['iou'] - mlp[seed]['val']['iou'] for seed in seeds}
        mean = statistics.mean(differences.values())
        wins = sum(value > 0 for value in differences.values())
        moved = all(quantum[seed]['diagnostics']['movement']['circuit_angles']['l2'] > 0 for seed in seeds)
        modes[mode] = dict(ready=True, differences=differences, mean=mean, wins=wins,
                           all_circuits_moved=moved, qualifies=mean >= 0.005 and wins >= 2 and moved)
    ready = all(value['ready'] for value in modes.values())
    candidates = [mode for mode, value in modes.items() if ready and value['qualifies']]
    selected = min(candidates, key=lambda mode: (-modes[mode]['mean'], mode)) if candidates else None
    return dict(ready=ready, selected_mode=selected, modes=modes,
                rule='Mean paired validation gain >=0.005; wins in >=2/3 seeds; nonzero selected circuit movement in every seed; prioritize largest mean gain')


def inspect_run(run):
    plan = json.loads((run / 'plan.json').read_text())
    settings = plan['settings']
    trials = [json.loads(path.read_text()) for path in sorted((run / 'tuning').glob('*/*/seed42/metrics.json'))]
    expected = len(plan['specs']) * len(settings['learning_rates'])
    if len(trials) != expected:
        return dict(ready=False, stage='tuning', completed_candidates=len(trials), expected_candidates=expected)
    chosen = choose_rates(trials)
    records = list(chosen.values())
    for name in chosen:
        for seed in settings['seeds'][1:]:
            path = run / 'replication' / name / ('seed' + str(seed)) / 'metrics.json'
            if path.is_file():
                row = json.loads(path.read_text())
                if row['learning_rate'] != chosen[name]['learning_rate']:
                    raise ValueError('Replication rate differs from validation selection')
                records.append(row)
    return decide(records, settings['seeds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(inspect_run(args.run_dir), indent=2, allow_nan=False) + '\n')
