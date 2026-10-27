#!/usr/bin/env python3
"""Run the fixed additional controls only after the paired validation trigger."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_real_spatial_pilot as pilot
from scripts.check_spatial_control_trigger import inspect_run


def prepare(baseline_run):
    """Read validation-only records and inherit the original comparison budget."""
    baseline_run = Path(baseline_run)
    plan_path = baseline_run / 'plan.json'
    plan = json.loads(plan_path.read_text())
    decision = inspect_run(baseline_run)
    if not decision['ready']:
        raise ValueError('All paired validation fits must finish before selecting controls')
    mode = decision['selected_mode']
    if mode is None:
        raise ValueError('Neither quantum mode meets the fixed additional-control criterion')
    original = next(spec for spec in plan['specs'] if spec['id'] == 'quantum_' + mode)
    specs = [dict(id='quantum_' + control + '_' + mode, model_type='quantum',
                  config=dict(original['config'], q_training_control=control))
             for control in ('frozen', 'no_entanglement')]
    context = dict(baseline_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        baseline_source_revision=plan['source_revision'],
        baseline_input_identity=plan['input_identity'], validation_control_decision=decision,
        policy='Same rate grid, three seeds, fit cap, normalization, epochs and evaluation schedule; no test input to selection')
    return plan['settings'], specs, context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    try:
        settings, specs, context = prepare(args.baseline_run)
    except ValueError as error:
        parser.error(str(error))
    forwarded = ['--data-root', str(args.data_root), '--work-dir', str(args.work_dir)]
    for name in ('device', 'epochs', 'eval_every', 'fit_cap', 'batch_size', 'image_size'):
        forwarded += ['--' + name.replace('_', '-'), str(settings[name])]
    for name in ('seeds', 'learning_rates'):
        forwarded += ['--' + name.replace('_', '-'), *map(str, settings[name])]
    if args.resume:
        forwarded.append('--resume')
    return pilot.main(forwarded, spec_factory=lambda: specs,
        extra_source_files=('scripts/run_quantum_training_controls.py', 'scripts/check_spatial_control_trigger.py'),
        experiment_context=context, expected_input_identity=context['baseline_input_identity'])


if __name__ == '__main__':
    main()
