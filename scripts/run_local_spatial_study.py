#!/usr/bin/env python3
"""Validation-led local spatial/readout pilot, followed by a planned duration check."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_real_spatial_pilot as pilot
from scripts.run_parameter_budget_comparison import base_config

EXTRA_SOURCES = ('scripts/run_local_spatial_study.py', 'utilsTorch/local_spatial.py')
QUANTUM_IDS = ('local_quantum_z', 'local_quantum_xz_zz')
CLASSICAL_IDS = ('local_mlp', 'local_conv', 'local_sincos')


def study_specs():
    common = dict(base_config(8, 1), dataset='combined', q_layers=1,
                  local_spatial_channels=4, local_spatial_readout='xz_zz')
    specs = [dict(id='local_quantum_' + readout, model_type='quantum',
                  config=dict(common, local_spatial_kernel='quantum',
                              local_spatial_readout=readout, q_training_control='trainable'))
             for readout in ('z', 'xz_zz')]
    specs += [dict(id='local_quantum_' + control, model_type='quantum',
                   config=dict(common, local_spatial_kernel='quantum', q_training_control=control))
              for control in ('frozen', 'no_entanglement')]
    specs += [dict(id='local_' + kind, model_type='ablation',
                   config=dict(common, local_spatial_kernel=kind, q_training_control='trainable'))
              for kind in ('mlp', 'conv', 'sincos')]
    return specs


def finalist_specs(summary):
    """Use paired validation only; test and knockout scores never enter selection."""
    groups = {}
    for row in summary['records']:
        group = groups.setdefault(row['id'], {})
        if row['seed'] in group:
            raise ValueError('Duplicate model/seed in finalist input')
        group[row['seed']] = row
    expected = {spec['id'] for spec in study_specs()}
    if set(groups) != expected or any(set(group) != {42, 77, 123} for group in groups.values()):
        raise ValueError('Finalists require all seven configurations and three paired seeds')
    def score(name):
        return statistics.mean(row['val']['iou'] for row in groups[name].values())
    quantum = max(QUANTUM_IDS, key=lambda name: (score(name), name == 'local_quantum_z'))
    classical = max(CLASSICAL_IDS, key=lambda name: (score(name),
        -groups[name][42]['total_params'], -CLASSICAL_IDS.index(name)))
    specs = {spec['id']: spec for spec in study_specs()}
    return [specs[quantum], specs[classical]]


def paired_analysis(records):
    """Compute seed differences before aggregating, preserving tuning qualifications."""
    by_id = {}
    for row in records:
        group = by_id.setdefault(row['id'], {})
        if row['seed'] in group:
            raise ValueError('Duplicate model/seed result')
        group[row['seed']] = row
    if not by_id or any(set(group) != {42, 77, 123} for group in by_id.values()):
        raise ValueError('Analysis requires the three planned paired seeds')
    for seed in (42, 77, 123):
        cohort = [group[seed] for group in by_id.values()]
        for key in ('fit_names', 'val_names', 'test_names', 'normalization'):
            if any(row[key] != cohort[0][key] for row in cohort):
                raise ValueError('Unpaired ' + key)
    pairs = [('local_quantum_xz_zz', 'local_quantum_z'),
             ('local_quantum_xz_zz', 'local_quantum_frozen'),
             ('local_quantum_xz_zz', 'local_quantum_no_entanglement')]
    pairs += [(quantum, classical) for quantum in QUANTUM_IDS for classical in CLASSICAL_IDS]
    if len(by_id) == 2:
        quantum = next(name for name in by_id if name in QUANTUM_IDS)
        classical = next(name for name in by_id if name in CLASSICAL_IDS)
        pairs = [(quantum, classical)]
    comparisons = []
    for left, right in pairs:
        if left not in by_id or right not in by_id:
            continue
        for split in ('val', 'test'):
            differences = {str(seed): 100 * (by_id[left][seed][split]['iou'] - by_id[right][seed][split]['iou'])
                           for seed in (42, 77, 123)}
            comparisons.append(dict(left=left, right=right, split=split, unit='IoU percentage points',
                differences=differences, mean=statistics.mean(differences.values()),
                std=statistics.stdev(differences.values()), wins=sum(value > 0 for value in differences.values()), n=3,
                same_learning_rates=all(by_id[left][seed]['learning_rate'] == by_id[right][seed]['learning_rate']
                                        for seed in (42, 77, 123))))
    return comparisons


def summarize(summary, work_dir):
    comparisons = paired_analysis(summary['records'])
    pilot.write_json(work_dir / 'paired_spatial_analysis.json', comparisons)
    lines = ['# Local spatial quantum comparison', '',
        'All comparisons use validation-selected rates; unequal selected rates remain a confound for gate/readout attribution.',
        'The test set was exposed during earlier research. These are exploratory descriptive results, not an untouched-holdout confirmation.', '',
        '| Contrast | Split | Mean ± sample SD (IoU percentage points) | Wins | Same rates |',
        '| --- | --- | ---: | ---: | --- |']
    for row in comparisons:
        lines.append(f"| {row['left']} − {row['right']} | {row['split']} | {row['mean']:+.3f} ± {row['std']:.3f} | {row['wins']}/3 | {row['same_learning_rates']} |")
    lines += ['', 'All per-seed outcomes, learning histories, input-angle diagnostics, parameter movement and output knockouts are retained in summary.json.',
              'Normalization uses the entire global fit partition while optimization uses 128 tiles per seed; this is not a pure few-image experiment.', '']
    (work_dir / 'spatial_report.md').write_text('\n'.join(lines))


def profile(device, output):
    """Bounded random-tensor resource and initialization audit, not accuracy evidence."""
    import torch
    from tools import training_common as tc
    from utilsTorch.local_spatial import LocalSpatialBottleneckBlock
    from utilsTorch.utilsTorch import QVUNetSegmentation, ClassicalAblationUNetSegmentation, segmentation_loss
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.12)
    rows = []
    expected_shared = None
    for spec in study_specs():
        tc.seed_everything(42)
        cls = QVUNetSegmentation if spec['model_type'] == 'quantum' else ClassicalAblationUNetSegmentation
        model = tc.build_model(cls, dict(spec['config'], seed=42)).to(device)
        blocks = [module for module in model.modules() if isinstance(module, LocalSpatialBottleneckBlock)]
        if len(blocks) != 1:
            raise ValueError('The study requires one core shared across both bottleneck residual blocks')
        core_ids = {id(p) for block in blocks for name, p in block.named_parameters()
                    if not name.startswith(('pre_q_conv.', 'post_q_proj.'))}
        shared = hashlib.sha256(b''.join(name.encode() + p.detach().cpu().numpy().tobytes()
            for name, p in model.named_parameters() if id(p) not in core_ids)).hexdigest()
        if expected_shared is not None and shared != expected_shared:
            raise ValueError('Backbone and projection initialization differs between controls')
        expected_shared = shared
        images = torch.randn(16, 3, 64, 64, device=device)
        masks = torch.randint(0, 2, (16, 64, 64), device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad()
            loss = segmentation_loss(model(images), masks, loss_mode='dice_ce')
            loss.backward()
            optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        rows.append(dict(id=spec['id'], total_params=sum(p.numel() for p in model.parameters()),
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            core_params=sum(p.numel() for p in model.parameters() if id(p) in core_ids),
            shared_initial_parameter_sha256=shared, two_step_elapsed_s=time.perf_counter() - started,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if device == 'cuda' else None,
            finite_loss=bool(torch.isfinite(loss))))
        del model, optimizer, images, masks, loss
        if device == 'cuda':
            torch.cuda.empty_cache()
    pilot.write_json(output, dict(device=device, batch_size=16, image_size=64, threads=2,
        gpu_allocator_fraction=0.12 if device == 'cuda' else None,
        policy='Two instrumented random-tensor steps; not a controlled throughput benchmark', records=rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('profile', 'pilot', 'extend'))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.phase == 'profile':
        if args.work_dir.exists() and any(args.work_dir.iterdir()):
            parser.error('Resource profiling requires a new/empty work directory')
        profile(args.device, args.work_dir / 'resource_profile.json')
        return
    if args.data_root is None:
        parser.error('--data-root is required for training')
    specs, context, expected_input = study_specs(), dict(phase='pilot', protocol='local-spatial-seven-spec-v1'), None
    if args.phase == 'extend':
        if args.baseline_run is None:
            parser.error('The duration extension requires --baseline-run')
        baseline = json.loads((args.baseline_run / 'summary.json').read_text())
        specs = finalist_specs(baseline)
        for path, digest in baseline['plan']['source_hashes'].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                parser.error('The producing source changed before the duration extension: ' + path)
        expected_input = baseline['plan']['input_identity']
        context = dict(phase='extend', protocol='local-spatial-two-finalists-v1',
            baseline_summary_sha256=hashlib.sha256((args.baseline_run / 'summary.json').read_bytes()).hexdigest(),
            selection_policy='Mean paired validation IoU; Q tie Z; classical tie smaller core then MLP/conv/sincos')
    forwarded = ['--data-root', str(args.data_root), '--work-dir', str(args.work_dir),
                 '--epochs', '60' if args.phase == 'pilot' else '120', '--device', args.device]
    if args.resume:
        forwarded.append('--resume')
    summary = pilot.main(forwarded, spec_factory=lambda: specs, extra_source_files=EXTRA_SOURCES,
        experiment_context=context, expected_input_identity=expected_input)
    summarize(summary, args.work_dir)
    return summary


if __name__ == '__main__':
    main()
