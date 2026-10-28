#!/usr/bin/env python3
"""Audit the complete HQ/MLP direction screen and export a reproducible report.

Run on CPU in a bounded memory scope. This command reads the real training data
to reconstruct splits/normalization, strictly restores all eight checkpoints,
and checks saved test replays. It does not train or repeat test inference.
"""
import argparse
from collections import Counter
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
from scripts import run_hq_dfc24_direction as study

require, digest = study.require, study.digest
PROTOCOL = 'docs/HQ_DFC24_DIRECTION_PROTOCOL.md'
SEEDS, RATES = study.SEEDS, study.RATES
IDS = ('hq_quantum', 'hq_mlp')
SCORES = ('iou', 'optical_subset_iou', 'sar_subset_iou')
COUNTS = {'quantum': (344480, 8), 'mlp': (349688, 5216)}
SAVED_TOP = ('plan.json', 'summary.json', 'all_fits.json', 'tuning_candidates.json',
             'validation_selection.json', 'checkpoint_audit.json')


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    study.finite_record(value)
    return value


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inside(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Expected a relative artifact path')
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Artifact path escapes its root: ' + name)
    return path


def current_runtime():
    import torch
    return dict(python=platform.python_version(), executable=str(Path(sys.executable).resolve()),
        torch_cuda=torch.version.cuda,
        packages={name: importlib.metadata.version(name) for name in
                  ('torch', 'pennylane', 'numpy', 'rasterio', 'scikit-learn', 'scipy', 'Pillow')})


def verify_source(plan):
    require(set(plan['source_hashes']) == set(study.SOURCE_FILES), 'Producer source inventory differs')
    for name, expected in plan['source_hashes'].items():
        require(digest(inside(ROOT, name)) == expected, 'Producer source differs: ' + name)
    require(digest(ROOT / PROTOCOL) == plan['protocol_sha256'], 'Frozen protocol differs')
    # HEAD may legitimately move after fitting; executable bytes are the gate.


def verify_runtime(plan):
    actual, recorded = current_runtime(), plan['runtime']
    require(actual['python'] == recorded['python'] and actual['packages'] == recorded['packages']
            and actual['torch_cuda'] == recorded['torch_cuda']
            and actual['executable'] == str(Path(recorded['executable']).resolve()),
            'Collector interpreter/packages differ from producing runtime')
    require(recorded['threads'] == recorded['interop_threads'] == 2, 'Producer thread bounds differ')
    require(recorded['device'] in ('cpu', 'cuda:0'), 'Unexpected producing device')
    if recorded['device'] == 'cuda:0':
        require(recorded['gpu_allocator_fraction'] == 0.12 and bool(recorded['gpu']), 'GPU bounds differ')
    return actual


def check_metrics(metrics, n_samples, *, test=False):
    study.validate_sensor_metrics(metrics, n_samples, test=test)
    require(metrics['loss'] >= 0, 'Invalid evaluation loss')
    pooled = dict(tp=0, fp=0, fn=0, tn=0)
    for prefix in ('', 'optical_subset_', 'sar_subset_'):
        if prefix:
            values = {key: metrics[prefix + key] for key in pooled}
            require(sum(values.values()) == metrics[prefix + 'n_samples'] * 128 * 128,
                    'Sensor pixel coverage differs')
            for key, value in values.items():
                pooled[key] += value
        else:
            continue
        tp, fp, fn, tn = (values[key] for key in ('tp', 'fp', 'fn', 'tn'))
        expected = dict(iou=tp / (tp + fp + fn) if tp + fp + fn else 0.,
            dice=2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else 0.,
            accuracy=(tp + tn) / sum(values.values()) if sum(values.values()) else 0.)
        for key in ('iou', 'dice', 'accuracy', 'auc'):
            require(0 <= metrics[prefix + key] <= 1, 'Invalid sensor ' + key)
            if key != 'auc':
                require(metrics[prefix + key] == expected[key], 'Sensor count/metric mismatch: ' + prefix + key)
    tp, fp, fn, tn = (pooled[key] for key in ('tp', 'fp', 'fn', 'tn'))
    for key, numerator, denominator in (('iou', tp, tp+fp+fn), ('dice', 2*tp, 2*tp+fp+fn),
                                        ('accuracy', tp+tn, tp+fp+fn+tn)):
        expected = numerator / denominator if denominator else 0.
        # Pooled sklearn accuracy/Dice and explicit count arithmetic can differ
        # by one rounding operation; sensor IoU itself is exact count division.
        require(math.isclose(metrics[key], expected, rel_tol=0, abs_tol=2e-15),
                'Pooled count/metric mismatch: ' + key)
    require(0 <= metrics['auc'] <= 1, 'Invalid pooled AUC')


def identity(row):
    return row['id'], row['seed'], row['learning_rate']


def expected_directory(row):
    return str(Path('tuning') / row['id'] / str(row['learning_rate']) / 'seed42'
               if row['seed'] == 42 else Path('replication') / row['id'] / ('seed' + str(row['seed'])))


def validate_diagnostics(row, args):
    diagnostics = row['diagnostics']
    require(diagnostics['probe_names'] == row['fit_names'][:8], 'Optimization probe names differ')
    for moment in ('initial_angle_probes', 'selected_angle_probes'):
        require(len(diagnostics[moment]) == 1, 'Expected one bottleneck angle probe')
        angles = diagnostics[moment][0]
        require(angles['n_angles'] == 8 * 16 * 3 and
                -1 <= angles['min_radians'] <= angles['max_radians'] <= 1 and
                0 <= angles['mean_abs_radians'] <= 1 and 0 <= angles['saturation_fraction'] <= 1,
                'Invalid bounded-radian angle diagnostic')
    require(set(diagnostics['movement']) == set(diagnostics['gradient_norms']) == {'core', 'common'},
            'Incomplete core/common diagnostics')
    steps = math.ceil(args.fit_cap / args.batch_size)
    for group in ('core', 'common'):
        move = diagnostics['movement'][group]
        require(move['l2'] >= 0 and (move['relative_l2'] is None or move['relative_l2'] >= 0), 'Invalid movement')
        per_epoch = [point['gradient_norms'][group] for point in row['history'][1:]]
        for entry in per_epoch + [diagnostics['gradient_norms'][group]]:
            require(0 <= entry['min'] <= entry['mean'] <= entry['max'] and
                    isinstance(entry['steps'], int) and isinstance(entry['nonzero_steps'], int) and
                    0 <= entry['nonzero_steps'] <= entry['steps'], 'Invalid gradient summary')
        require(all(entry['steps'] == steps for entry in per_epoch), 'Per-epoch gradient coverage differs')
        overall = diagnostics['gradient_norms'][group]
        require(overall['steps'] == steps * args.epochs and
                overall['nonzero_steps'] == sum(entry['nonzero_steps'] for entry in per_epoch) and
                overall['min'] == min(entry['min'] for entry in per_epoch) and
                overall['max'] == max(entry['max'] for entry in per_epoch), 'Aggregate gradient coverage differs')


def audit_checkpoint(path, row):
    model, audit = study.restore_checkpoint(path, row, device='cpu')
    initial_model = study.model_from_config(row['model_config'])
    initial = {name: value.detach().clone() for name, value in initial_model.named_parameters()}
    require(study.movement(model, initial) == row['diagnostics']['movement'], 'Checkpoint parameter movement differs')
    total, core = COUNTS[row['kind']]
    require(row['total_params'] == row['trainable_params'] == total and row['core_params'] == core,
            'Reported model/core parameter counts differ')
    require(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == total,
            'Restored trainable parameter count differs')
    del model, initial_model
    return audit


def audit_campaign(work_dir, data_root):
    work_dir, data_root = Path(work_dir).resolve(), Path(data_root).resolve()
    plan, summary = read(work_dir / 'plan.json'), read(work_dir / 'summary.json')
    require(summary['plan'] == plan, 'Summary/frozen plan differ')
    verify_source(plan)
    runtime = verify_runtime(plan)
    require(plan['seeds'] == list(SEEDS) and plan['learning_rates'] == list(RATES), 'Planned seed/rate grid differs')
    expected = dict(device=plan['settings']['device'], batch_size=8, epochs=60, fit_cap=128,
                    image_size=128, base_channels=8, eval_every=2, augment=True)
    require(plan['settings'] == expected and plan['settings']['augment'] is True, 'Frozen protocol budget differs')
    require(all(type(plan['settings'][key]) is int for key in
                ('batch_size', 'epochs', 'fit_cap', 'image_size', 'base_channels', 'eval_every')),
            'Frozen integer budgets have invalid types')
    for key, value in dict(
        test_policy='All eight fits finish before six selected checkpoint tests; test already exposed/descriptive',
        augmentation_policy='Stateless D4 by(seed,epoch,original index), independent seeded sample order; train only',
        normalization_policy='All 1303 global fit images, validation excluded; optimization cap separate').items():
        require(plan[key] == value, 'Frozen study policy differs: ' + key)
    require(plan['settings']['device'] == plan['runtime']['device'].split(':')[0], 'Recorded device differs')
    inputs = study.pilot.verify_inputs(data_root)
    require(inputs == plan['input_identity'] and inputs['verified_files'] == 3476, 'Actual input bytes differ')
    manifest = read(data_root / 'splits/flood_combined.json')
    require(digest(ROOT / 'splits/flood_combined.json') == inputs['manifest_sha256'] and
            manifest['conflict_policy'] == 'drop' and len(manifest['train']) == 1533 and len(manifest['test']) == 205,
            'Committed conflict-free manifest differs')
    entries = read(work_dir / 'all_fits.json')
    require(len(entries) == 8, 'Expected all eight fit records')
    produced_audits = read(work_dir / 'checkpoint_audit.json')
    require(len(produced_audits) == 8 and len({identity(row) for row in produced_audits}) == 8,
            'Incomplete/duplicate producer checkpoint audit')
    produced_by_key = {identity(row): row for row in produced_audits}
    rows, paths, fresh_audits = {}, {}, []
    historical_roots = {entry['record']['settings']['data_root'] for entry in entries}
    require(len(historical_roots) == 1 and Path(next(iter(historical_roots))).is_absolute(), 'Historical data roots differ')
    raw_files = {name: digest(work_dir / name) for name in SAVED_TOP}
    for seed in SEEDS:
        load_args = argparse.Namespace(**plan['settings'], data_root=data_root)
        data = study.pilot.load_training_data(load_args, seed)
        require(len(data['all_fit']) == 1303 and len(data['validation']) == 230, 'Actual global split differs')
        audit_args = argparse.Namespace(**plan['settings'], data_root=Path(next(iter(historical_roots))))
        cohort = [entry for entry in entries if entry['record']['seed'] == seed]
        require(len(cohort) == (4 if seed == 42 else 2), 'Incomplete by-seed fit coverage')
        for entry in cohort:
            row, relative = entry['record'], entry['directory']
            key = identity(row)
            require(row['kind'] in study.KINDS and key not in rows and relative == expected_directory(row), 'Duplicate/unplanned fit identity/path')
            directory = inside(work_dir, relative)
            require(not ({'test', 'test_names', 'knockout', 'checkpoint'} & set(row)), 'Fit record crosses test boundary')
            require(read(directory / 'metrics.json') == row and read(directory / 'history.json') == row['history'],
                    'Raw metrics/history differ from all-fits record')
            study.audit_fit(row, row['kind'], seed, row['learning_rate'], audit_args, data)
            require(row['learning_rate'] in RATES and row['elapsed_s'] > 0, 'Invalid rate/fit duration')
            require(row['fit_sensor_counts'] == dict(Counter(data['dataset'].modalities[i] for i in data['fit'])), 'Optimization sensor counts differ')
            for point in row['history']:
                if 'val' in point:
                    check_metrics(point['val'], 230)
            validate_diagnostics(row, audit_args)
            checkpoint = directory / 'checkpoint.pt'
            require(key in produced_by_key and digest(checkpoint) == produced_by_key[key]['checkpoint_sha256'],
                    'Checkpoint bytes differ from producing audit')
            details = audit_checkpoint(checkpoint, row)
            fresh_audits.append(dict(id=row['id'], seed=seed, learning_rate=row['learning_rate'], **details))
            rows[key], paths[key] = row, directory
            for filename in ('metrics.json', 'history.json', 'checkpoint.pt'):
                raw_files[str((directory / filename).relative_to(work_dir))] = digest(directory / filename)
        require(len({entry['record']['initial_shared_state_sha256'] for entry in cohort}) == 1,
                'Common initial parameters/BN buffers are not paired')
        for kind in study.KINDS:
            group = [entry['record'] for entry in cohort if entry['record']['kind'] == kind]
            require(len({row['initial_state_sha256'] for row in group}) == 1, 'Rate candidates have different initialization')
        schedules = [[point['train_schedule'] for point in entry['record']['history']] for entry in cohort]
        require(all(value == schedules[0] for value in schedules), 'Paired data order/augmentation differs')
        del data
        gc.collect()
    tuning = [rows[key] for key in sorted(rows) if key[1] == 42]
    require({identity(row) for row in tuning} == {(name, 42, rate) for name in IDS for rate in RATES}, 'Incomplete tuning grid')
    require(sorted(read(work_dir / 'tuning_candidates.json'), key=identity) == tuning, 'Saved tuning candidates differ')
    chosen = study.pilot.choose_rates(tuning)
    selection = {name: row['learning_rate'] for name, row in chosen.items()}
    require(summary['selection'] == selection, 'Selected rates are not validation-only winners')
    require(read(work_dir / 'validation_selection.json') == {name: dict(learning_rate=row['learning_rate'],
            best_epoch=row['best_epoch'], val_iou=row['val']['iou']) for name, row in chosen.items()}, 'Saved validation decision differs')
    expected_keys = {(name, 42, rate) for name in IDS for rate in RATES} | {
        (name, seed, selection[name]) for name in IDS for seed in SEEDS[1:]}
    require(set(rows) == expected_keys, 'Replication grid differs from selected rates')
    require(sorted(produced_audits, key=identity) == sorted(fresh_audits, key=identity),
            'Producer checkpoint digests/audits differ from actual bytes')
    records = summary['records']
    require(len(records) == 6 and {(row['id'], row['seed']) for row in records} == {
        (name, seed) for name in IDS for seed in SEEDS}, 'Incomplete six selected results')
    selected_paths = set()
    for row in records:
        key = identity(row)
        require(row['learning_rate'] == selection[row['id']] and key in rows, 'Selected replay has an unselected rate')
        require({key: value for key, value in row.items() if key not in ('test', 'test_names', 'knockout', 'checkpoint')} == rows[key],
                'Selected result differs from audited fit')
        directory = paths[key]
        require(row['checkpoint'] == str((directory / 'checkpoint.pt').relative_to(work_dir)), 'Selected checkpoint path differs')
        replay = directory / 'selected_test.json'
        require(read(replay) == row, 'Saved selected replay differs from summary')
        require(row['test_names'] == manifest['test'] and len(set(row['test_names'])) == 205, 'Test names/order differ')
        check_metrics(row['test'], 205, test=True)
        require(set(row['knockout']) == set(study.MODES), 'Incomplete diagnostic modes')
        for mode in study.MODES:
            check_metrics(row['knockout'][mode], 205, test=True)
        selected_paths.add(replay)
        raw_files[str(replay.relative_to(work_dir))] = digest(replay)
    for filename in ('metrics.json', 'history.json', 'checkpoint.pt'):
        require(set(work_dir.rglob(filename)) == {directory / filename for directory in paths.values()}, 'Extra/missing fit artifact: ' + filename)
    require(set(work_dir.rglob('selected_test.json')) == selected_paths, 'Extra/missing selected replay')
    analysis = study.analysis(records)
    require(summary['analysis'] == analysis, 'Saved paired/summary analysis differs')
    verify_source(plan)
    require(study.pilot.verify_inputs(data_root) == inputs, 'Input bytes changed during collection')
    require(all(digest(work_dir / name) == value for name, value in raw_files.items()), 'Raw artifacts changed during collection')
    return dict(plan=plan, summary=summary, fits=entries, tuning=tuning, checkpoint_audit=sorted(fresh_audits, key=identity),
                analysis=analysis, collector_runtime=runtime, raw_files=raw_files)


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values), n=len(values))


def comparison(result):
    records = result['summary']['records']
    by_key = {(row['id'], row['seed']): row for row in records}
    pairs, knockouts = [], []
    for split in ('val', 'test'):
        for metric in SCORES:
            values = {str(seed): 100 * (by_key['hq_quantum', seed][split][metric] -
                                      by_key['hq_mlp', seed][split][metric]) for seed in SEEDS}
            pairs.append(dict(split=split, metric=metric, **stats(list(values.values())),
                wins=sum(value > 0 for value in values.values()), differences_pp=values,
                same_learning_rate=len(set(result['summary']['selection'].values())) == 1))
    for name in IDS:
        for mode in study.MODES:
            values = {str(seed): 100 * (by_key[name, seed]['test']['iou'] - by_key[name, seed]['knockout'][mode]['iou']) for seed in SEEDS}
            knockouts.append(dict(id=name, mode=mode, **stats(list(values.values())),
                wins=sum(value > 0 for value in values.values()), differences_pp=values))
    return dict(paired=pairs, knockouts=knockouts, unit='IoU percentage points',
                scope='DFC24 architecture-transfer screen; exposed test descriptive; validation primary')


def tables(result, contrasts):
    selected = result['summary']['records']
    summary, seeds, tuning, diagnostics = [], [], [], []
    for name in IDS:
        group = [row for row in selected if row['id'] == name]
        record = dict(id=name, learning_rate=result['summary']['selection'][name],
                      total_params=group[0]['total_params'], core_params=group[0]['core_params'])
        for split in ('val', 'test'):
            for metric in SCORES:
                record.update({f'{split}_{metric}_{key}': value for key, value in stats([row[split][metric] for row in group]).items()})
        summary.append(record)
    for row in sorted(selected, key=identity):
        flat = {key: row[key] for key in ('id', 'seed', 'learning_rate', 'best_epoch', 'total_params', 'core_params')}
        flat.update({f'{split}_{key}': value for split in ('val', 'test') for key, value in row[split].items()})
        seeds.append(flat)
    for row in result['tuning']:
        tuning.append(dict(id=row['id'], seed=42, learning_rate=row['learning_rate'], best_epoch=row['best_epoch'],
            selected_rate=row['learning_rate'] == result['summary']['selection'][row['id']],
            **{'val_' + key: value for key, value in row['val'].items()}))
    for entry in result['fits']:
        row = entry['record']
        for moment in ('initial', 'selected'):
            diagnostics.append(dict(id=row['id'], seed=row['seed'], learning_rate=row['learning_rate'], moment=moment,
                selected_rate=row['learning_rate'] == result['summary']['selection'][row['id']],
                **row['diagnostics'][moment + '_angle_probes'][0],
                core_movement_l2=row['diagnostics']['movement']['core']['l2'],
                common_movement_l2=row['diagnostics']['movement']['common']['l2'],
                core_gradient_mean=row['diagnostics']['gradient_norms']['core']['mean'],
                core_gradient_steps=row['diagnostics']['gradient_norms']['core']['steps'],
                core_gradient_nonzero_steps=row['diagnostics']['gradient_norms']['core']['nonzero_steps']))
    paired = [{**{key: value for key, value in row.items() if key != 'differences_pp'},
               **{'seed' + seed + '_pp': value for seed, value in row['differences_pp'].items()}}
              for row in contrasts['paired']]
    knockout = [{**{key: value for key, value in row.items() if key != 'differences_pp'},
                 **{'seed' + seed + '_pp': value for seed, value in row['differences_pp'].items()}}
                for row in contrasts['knockouts']]
    return dict(summary=summary, per_seed=seeds, tuning=tuning, paired=paired, knockout=knockout, diagnostics=diagnostics)


def report(result, contrasts, rows):
    lines = ['# HQ-style direction screen on DFC24', '',
        'Architecture-transfer pilot, not a reproduction of the five-class LandCover.ai paper. Validation is primary; the previously exposed test split is descriptive.', '',
        'All eight planned fits and checkpoints were audited, including both losing learning rates. Six selected saved test replays were checked for exact record/identity consistency and count-derived metrics; collection does not independently repeat inference.', '',
        '128-pixel inputs; 128 optimization tiles per seed; 1,303 fit images used for normalization, excluding 230 validation images; all 205 test images (71 Optical/134 SAR). Three paired seeds, 60 epochs, batch size 8, Adam, Dice+CE and deterministic training-only D4 augmentation.', '',
        'The MLP deliberately has more core parameters (5,216 versus 8); total parameters are 349,688 versus 344,480. Interfaces and initial non-core state, including BatchNorm buffers, are identical per seed. This is not exact parameter-count matching.', '',
        '| Model | Selected LR | Validation IoU mean ± sample SD | Descriptive test IoU mean ± sample SD | Optical test | SAR test |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows['summary']:
        lines.append(f"| {row['id']} | {row['learning_rate']} | {row['val_iou_mean']:.4f} ± {row['val_iou_std']:.4f} | {row['test_iou_mean']:.4f} ± {row['test_iou_std']:.4f} | {row['test_optical_subset_iou_mean']:.4f} | {row['test_sar_subset_iou_mean']:.4f} |")
    lines += ['', 'Differences subtract paired seeds before aggregation. Units below are IoU percentage points; ± is sample SD, not a standard error or significance test.', '',
        '| Quantum − MLP | Split | Mean ± sample SD | Positive seeds | Same selected LR |', '| --- | --- | ---: | ---: | --- |']
    for row in contrasts['paired']:
        lines.append(f"| {row['metric']} | {row['split']} | {row['mean']:+.3f} ± {row['std']:.3f} | {row['wins']}/3 | {row['same_learning_rate']} |")
    lines += ['', '| Model | On − diagnostic test IoU | Mean ± sample SD (points) |', '| --- | --- | ---: |']
    for row in contrasts['knockouts']:
        lines.append(f"| {row['id']} | {row['mode']} | {row['mean']:+.3f} ± {row['std']:.3f} |")
    lines += ['', '`zero_core` retains reconstruction bias; `zero_bottleneck` also removes that bias. `random_core` replaces only core coefficients. Drops measure dependence on an information path, not quantum superiority. Ordinary U-Net skips remain.', '',
        'Angles are tanh-bounded radians without pi scaling. Each angle probe contains 384 values from the first eight unaugmented optimization images for its seed, measured at initialization and the selected checkpoint. Gradient/movement summaries establish observed optimization activity, not parameter importance or an accuracy mechanism. Curves retain all eight fits and every epoch training loss; incomplete convergence remains a limitation.', '',
        'Three seeds and one classical control establish neither significance, equivalence, quantum advantage nor a general impossibility result. This local centralized screen makes no Anvil, hardware or throughput claim.', '',
        'Figures: [all learning curves](plots/learning_curves.png), [paired pooled/sensor outcomes](plots/paired_iou.png), [knockouts](plots/knockouts.png), [angle/core diagnostics](plots/diagnostics.png). Each has a PDF export. [Reproduction and provenance](REPRODUCTION.md).', '']
    return '\n'.join(lines)


def plot_results(result, contrasts, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = Path(out_dir) / 'plots'
    plot_dir.mkdir()
    def save(fig, name):
        if name == 'knockouts':
            fig.tight_layout(rect=(0, .14, 1, 1))
        else:
            fig.tight_layout()
        fig.savefig(plot_dir / (name + '.png'), dpi=150, bbox_inches='tight')
        fig.savefig(plot_dir / (name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
    colors = {42: 'tab:blue', 77: 'tab:orange', 123: 'tab:green'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, name in enumerate(IDS):
        for entry in result['fits']:
            row = entry['record']
            if row['id'] != name:
                continue
            selected = row['learning_rate'] == result['summary']['selection'][name]
            style = '-' if selected else '--'
            label = f"seed{row['seed']} LR{row['learning_rate']}" + ('' if selected else ' (unselected)')
            for col, field in ((0, 'train_loss'), (1, 'val')):
                points = [point for point in row['history'] if point.get(field) is not None]
                axes[i, col].plot([point['epoch'] for point in points],
                    [point[field]['iou'] if field == 'val' else point[field] for point in points],
                    style, color=colors[row['seed']], alpha=1 if selected else .6, label=label)
                axes[i, col].set(xlabel='Epoch', ylabel='Training Dice+CE' if col == 0 else 'Validation foreground IoU', title=name)
                axes[i, col].legend(fontsize=8)
    save(fig, 'learning_curves')
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    records = {(row['id'], row['seed']): row for row in result['summary']['records']}
    for i, split in enumerate(('val', 'test')):
        for j, metric in enumerate(SCORES):
            ax = axes[i, j]
            for seed in SEEDS:
                ax.plot([0, 1], [records[name, seed][split][metric] for name in IDS], 'o-', color=colors[seed], label=f'seed{seed}')
            delta = next(row for row in contrasts['paired'] if row['split'] == split and row['metric'] == metric)
            ax.set(xticks=[0, 1], xticklabels=['Quantum', 'MLP'], ylabel='IoU',
                   title=f"{'Validation' if split == 'val' else 'Descriptive test'}: {metric}\nQ−MLP {delta['mean']:+.2f} ± {delta['std']:.2f} pp")
            ax.legend(fontsize=8)
    save(fig, 'paired_iou')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name in zip(axes, IDS):
        for i, mode in enumerate(study.MODES):
            row = next(row for row in contrasts['knockouts'] if row['id'] == name and row['mode'] == mode)
            for offset, seed in zip((-.12, 0, .12), SEEDS):
                ax.scatter(i + offset, row['differences_pp'][str(seed)], color=colors[seed],
                           label=f'seed{seed}' if i == 0 else None)
            ax.errorbar(i, row['mean'], yerr=row['std'], color='black', fmt='_', capsize=4)
        ax.axhline(0, color='grey', linewidth=.8)
        ax.set(xticks=range(3), xticklabels=study.MODES, ylabel='On − diagnostic IoU (points)', title=name + ' — descriptive test')
    handles, seed_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, seed_labels, loc='lower center', bbox_to_anchor=(.5, .01), ncol=3)
    save(fig, 'knockouts')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fits = sorted(result['fits'], key=lambda entry: identity(entry['record']))
    labels = [f"{'Q' if entry['record']['kind'] == 'quantum' else 'MLP'} s{entry['record']['seed']}\nLR{entry['record']['learning_rate']}" for entry in fits]
    for index, entry in enumerate(fits):
        row = entry['record']
        for ax, key, scale in ((axes[0, 0], 'saturation_fraction', 100), (axes[0, 1], 'mean_abs_radians', 1)):
            values = [row['diagnostics'][moment + '_angle_probes'][0][key] * scale for moment in ('initial', 'selected')]
            ax.plot([index-.12, index+.12], values, 'o-', color=colors[row['seed']])
        axes[1, 0].scatter(index, row['diagnostics']['movement']['core']['l2'], color=colors[row['seed']])
        axes[1, 1].scatter(index, row['diagnostics']['gradient_norms']['core']['mean'], color=colors[row['seed']])
    for ax, title, ylabel in ((axes[0, 0], 'Eight fixed training images\nInitial → selected (left → right)', 'Angles with |tanh| ≥ .95 (%)'),
                              (axes[0, 1], 'Eight fixed training images\nInitial → selected (left → right)', 'Mean absolute angle (radians)'),
                              (axes[1, 0], 'Selected parameter movement', 'Core L2 movement'),
                              (axes[1, 1], 'All training batches', 'Mean core gradient norm')):
        ax.set(xticks=range(len(fits)), xticklabels=labels, title=title, ylabel=ylabel)
        ax.tick_params(axis='x', labelsize=7)
    save(fig, 'diagnostics')


def export(result, work_dir, data_root, out_dir):
    work_dir, data_root, out_dir = Path(work_dir).resolve(), Path(data_root).resolve(), Path(out_dir).resolve()
    require(not out_dir.exists() or not any(out_dir.iterdir()), 'Use a new or empty output directory')
    for source in (work_dir, data_root, ROOT):
        require(out_dir != source and not source.is_relative_to(out_dir), 'Output contains an input/source root')
    require(not out_dir.is_relative_to(work_dir) and not out_dir.is_relative_to(data_root), 'Output aliases raw inputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SAVED_TOP:
        target = 'producer_checkpoint_audit.json' if name == 'checkpoint_audit.json' else name
        (out_dir / target).write_bytes((work_dir / name).read_bytes())
    for name in result['raw_files']:
        if name not in SAVED_TOP and name.endswith('.json'):
            path = out_dir / 'raw_fits' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((work_dir / name).read_bytes())
    contrasts = comparison(result)
    rows = tables(result, contrasts)
    write(out_dir / 'checkpoint_audit.json', result['checkpoint_audit'])
    write(out_dir / 'comparison.json', contrasts)
    for name, values in rows.items():
        with (out_dir / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(values)
    source_manifest = {}
    for name in (*study.SOURCE_FILES, PROTOCOL, 'scripts/collect_hq_dfc24_direction.py'):
        archived = 'sources/' + name + '.txt'
        path = out_dir / archived
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        source_manifest[name] = dict(sha256=digest(path), retained_as=archived)
    write(out_dir / 'source_manifest.json', source_manifest)
    (out_dir / 'input_manifest.json').write_bytes((data_root / 'splits/flood_combined.json').read_bytes())
    for name, expected in result['plan']['input_identity']['stats_files_sha256'].items():
        source = inside(data_root, name)
        require(digest(source) == expected, 'Input normalization metadata differs: ' + name)
        target = out_dir / 'input_metadata' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    provenance = dict(producing_revision=result['plan']['source_revision'], source_hashes=result['plan']['source_hashes'],
        protocol_sha256=result['plan']['protocol_sha256'], input_identity=result['plan']['input_identity'],
        producer_runtime=result['plan']['runtime'], collector_runtime=result['collector_runtime'],
        matplotlib_version=importlib.metadata.version('matplotlib'), collector_source_sha256=digest(__file__),
        raw_work_dir=str(work_dir), data_root=str(data_root), raw_artifact_sha256=result['raw_files'],
        audited_fits=8, audited_selected_saved_replays=6, independent_inference_replays=0,
        checks=['actual data bytes before/after; reconstructed fit/validation cohorts and normalization',
                'all checkpoint bytes, strict state/BN restoration, initialization/counts/movement',
                'validation-only rates, first strict best epoch, complete histories/data schedules',
                'paired common state, cohorts, data order and D4 augmentation',
                'saved replay equality, test names and confusion-derived pooled/sensor scores'])
    write(out_dir / 'provenance.json', provenance)
    (out_dir / 'README.md').write_text(report(result, contrasts, rows))
    command = ('systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= '
        'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '
        'MPLCONFIGDIR=/tmp/quaffle-hq-collector-mpl PYTHONPATH=' + shlex.quote(str(ROOT) + ':/tmp/quaffle-wave4a-deps') + ' ' +
        shlex.quote(sys.executable) + ' ' + shlex.quote(str(ROOT / 'scripts/collect_hq_dfc24_direction.py')) +
        ' --work-dir ' + shlex.quote(str(work_dir)) + ' --data-root ' + shlex.quote(str(data_root)) +
        ' --out-dir /tmp/quaffle-hq-dfc24-direction-regenerated')
    (out_dir / 'REPRODUCTION.md').write_text('# Regenerate this report\n\nRun from the source checkout whose producer hashes match `plan.json` and whose collector hash matches `provenance.json`. A later Git revision is allowed when every bound executable byte is unchanged. The exact original source/protocol and collector are retained as inert files under `sources/`.\n\n```bash\n' + command + '\n```\n\nUse a new/empty output directory. No weights or images are copied into this report. All eight checkpoint identities and persistent raw locations are in `provenance.json`; weights and data must remain available locally. Data may be relocated only when the complete bound input identity matches; original saved settings retain their historical data-root paths.\n\nThe collector reloads training cohorts/fit-only normalization and restores checkpoints on CPU. Saved test predictions are not independently rerun: record equality and count/identity consistency are the audited layer. `artifact_sha256.json` covers every generator output except itself. PDF creation/modification dates are suppressed for reproducibility. Resource profiles, external test logs and subsequent interpretation added by the publication owner are separate evidence.\n')
    plot_results(result, contrasts, out_dir)
    for name, expected in result['raw_files'].items():
        require(digest(work_dir / name) == expected, 'Raw artifacts changed during export')
    verify_source(result['plan'])
    require(study.pilot.verify_inputs(data_root) == result['plan']['input_identity'], 'Inputs changed during export')
    hashes = {str(path.relative_to(out_dir)): digest(path) for path in sorted(out_dir.rglob('*')) if path.is_file()}
    write(out_dir / 'artifact_sha256.json', hashes)
    return contrasts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out_dir.exists() or not any(args.out_dir.iterdir()), 'Use a new or empty output directory')
    for source in (args.work_dir, args.data_root):
        require(not args.out_dir.resolve().is_relative_to(source.resolve()) and not source.resolve().is_relative_to(args.out_dir.resolve()),
                'Output aliases or contains raw inputs')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    result = audit_campaign(args.work_dir, args.data_root)
    export(result, args.work_dir, args.data_root, args.out_dir)
    print('Audited8 checkpoints and6 saved selected replays; report:', args.out_dir)
    return result


if __name__ == '__main__':
    main()
