#!/usr/bin/env python3
"""Audit and retain both completed local rate/duration factorials."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_local_rate_stability as study

TEST_FIELDS = {'test', 'test_names', 'knockout', 'checkpoint_sha256'}
AUDIT_FLAGS = ('strict_restore', 'finite_weights', 'initial_parameters_equal',
               'validation_replayed', 'validation_counts_equal')
FIXED_SETTINGS = dict(eval_every=2, fit_cap=128, batch_size=16, image_size=64)
HISTORICAL_CELLS = {
    ('quantum_global', 42, 0.001), ('quantum_global', 42, 0.003),
    ('quantum_global', 77, 0.001), ('quantum_global', 123, 0.001),
    ('quantum_no_entanglement_global', 42, 0.001),
    ('quantum_no_entanglement_global', 42, 0.003),
    ('quantum_no_entanglement_global', 77, 0.003),
    ('quantum_no_entanglement_global', 123, 0.003),
}


def read(path):
    value = json.loads(Path(path).read_text())
    study.require_finite(value)
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keyed(rows, label):
    result = {}
    for row in rows:
        key = study.cell_key(row)
        require(key not in result, 'Duplicate ' + label + ' cell: ' + str(key))
        result[key] = row
    return result


def source_audit(plan):
    require(set(plan['source_hashes']) == set(study.SOURCE_FILES),
            'Producer source inventory is incomplete or unexpected')
    for name, expected in plan['source_hashes'].items():
        require(study.digest(ROOT / name) == expected,
                'Executable source differs from producing run: ' + name)


def validate_metrics(metrics, names, label):
    for key in ('iou', 'dice', 'accuracy', 'auc', 'optical_subset_iou', 'sar_subset_iou'):
        require(0 <= metrics[key] <= 1, 'Invalid ' + label + ' metric: ' + key)
    totals = {}
    for count in ('tp', 'fp', 'fn', 'tn', 'n_samples'):
        values = [metrics[sensor + '_subset_' + count] for sensor in ('optical', 'sar')]
        require(all(isinstance(value, int) and value >= 0 for value in values),
                'Invalid sensor confusion counts: ' + label)
        totals[count] = sum(values)
    require(totals['n_samples'] == len(names), 'Sensor sample counts differ: ' + label)
    union = totals['tp'] + totals['fp'] + totals['fn']
    require(math.isclose(metrics['iou'], totals['tp'] / union if union else 0, abs_tol=1e-12),
            'Sensor counts do not recombine to pooled IoU: ' + label)
    for sensor in ('optical', 'sar'):
        tp, fp, fn = [metrics[sensor + '_subset_' + count] for count in ('tp', 'fp', 'fn')]
        require(math.isclose(metrics[sensor + '_subset_iou'], tp / (tp + fp + fn) if tp + fp + fn else 0,
                             abs_tol=1e-12), 'Sensor IoU differs from confusion counts: ' + label)


def record_audit(row, spec, plan):
    epochs = plan['epochs']
    require(len(row['fit_names']) == plan['fit_cap'], 'Optimization cohort differs from fixed cap')
    require(len(row['fit_names']) == len(set(row['fit_names'])), 'Duplicate optimization image names')
    require(len(row['val_names']) == len(set(row['val_names'])), 'Duplicate validation image names')
    require(set(row['fit_names']).isdisjoint(row['val_names']), 'Optimization and validation overlap')
    require(row['normalization']['fit_only'] is True, 'Normalization is not fit-only')
    require(row['settings']['_resolved_norm_provenance'] == row['normalization'],
            'Checkpoint settings and record normalization differ')
    indices = row['fit_indices'] + row['val_indices']
    n_training = row['n_global_fit'] + len(row['val_names'])
    require(all(isinstance(index, int) and 0 <= index < n_training for index in indices),
            'Invalid training-partition index')
    require(len(set(row['fit_indices'])) == len(row['fit_names'])
            and len(set(row['val_indices'])) == len(row['val_names'])
            and set(row['fit_indices']).isdisjoint(row['val_indices']),
            'Training-partition index identities differ')
    names = [None] * n_training
    for subset in ('fit', 'val'):
        for index, name in zip(row[subset + '_indices'], row[subset + '_names']):
            names[index] = name
    args = SimpleNamespace(data_root=Path(row['settings']['data_root']), epochs=epochs,
                           **FIXED_SETTINGS)
    data = dict(fit=row['fit_indices'], validation=row['val_indices'],
                all_fit=range(row['n_global_fit']), dataset=SimpleNamespace(names=names),
                normalization=row['normalization'])
    study.audit_record(row, spec, row['seed'], row['learning_rate'], args, data)
    expected_evaluations = set(range(0, epochs + 1, plan['eval_every'])) | {epochs}
    require({point['epoch'] for point in row['history'] if 'val' in point} == expected_evaluations,
            'Validation evaluation schedule differs from the plan')
    for point in row['history']:
        if 'val' in point:
            validate_metrics(point['val'], row['val_names'], 'validation')
    validate_metrics(row['test'], row['test_names'], 'test')
    require(len(row['test_names']) == len(set(row['test_names'])) == 205,
            'Test replay does not contain all 205 unique DFC24 tiles')


def historical_plan(directory):
    for ancestor in (directory, *directory.parents):
        candidate = ancestor / 'plan.json'
        if candidate.is_file():
            return candidate
    raise ValueError('Historical producer plan is missing: ' + str(directory))


def audit_run(run, phase):
    run = Path(run)
    plan, summary = read(run / 'plan.json'), read(run / 'summary.json')
    require(plan['phase'] == phase and summary['plan'] == plan, 'Summary phase or frozen plan differs')
    source_audit(plan)
    require(plan['seeds'] == list(study.SEEDS) and plan['learning_rates'] == list(study.RATES),
            'Paired seed/rate grid differs')
    require(all(plan[key] == value for key, value in FIXED_SETTINGS.items()), 'Fixed study budget differs')
    require(plan['epochs'] == (60 if phase == 'matched60' else 120), 'Training duration differs')
    require(plan['validation_only'] is False, 'A completed test replay is required for collection')
    specs = study.specs_for(phase)
    require(plan['specs'] == json.loads(json.dumps(specs)), 'Model configurations differ from protocol')
    records = summary['records']
    cells = study.audit_factorial(records, specs, test=True)
    expected_count = 12 if phase == 'matched60' else 18
    require(len(records) == expected_count, 'Unexpected completed cell count')
    validations = read(run / 'validation_fits.json')
    require(validations == [{key: value for key, value in row.items() if key not in TEST_FIELDS} for row in records],
            'Summary differs from validation-only fit records')
    evidence = read(run / 'checkpoint_audit.json')
    require(summary['checkpoint_audit'] == evidence, 'Summary checkpoint audit differs')
    audits = keyed(evidence, 'checkpoint audit')
    historical = keyed(plan['historical_fits'], 'historical manifest')
    require(set(audits) == set(cells), 'Checkpoint audits do not cover all cells')
    require(set(historical) == (HISTORICAL_CELLS if phase == 'matched60' else set()),
            'Historical reuse cell set differs from the planned eight fits')
    if phase == 'matched60':
        require(read(run / 'reuse_audit.json') == [row for row in evidence if row['origin'] == 'historical'],
                'Historical replay audit differs from checkpoint audit')
    checkpoint_files, producers, fit_records, flat = {}, {}, [], []
    expected_new_metrics, expected_test_replays = set(), set()
    for row in records:
        key, audit = study.cell_key(row), audits[study.cell_key(row)]
        require(all(audit[flag] is True for flag in AUDIT_FLAGS), 'Checkpoint audit has an unverified guard')
        spec = next(spec for spec in specs if spec['id'] == row['id'])
        record_audit(row, spec, plan)
        if key in historical:
            item = historical[key]
            directory = Path(item['directory'])
            producer_path = historical_plan(directory)
            producer = read(producer_path)
            require(study.digest(producer_path) == item['producer_plan_sha256'], 'Historical producer plan bytes changed')
            require(producer['source_revision'] == item['producer_source_revision'] == audit['producer_source_revision'],
                    'Historical producer revision differs')
            require(producer['source_hashes'] == item['producer_source_hashes'], 'Historical source manifest differs')
            require({name: producer[name] for name in study.RUNTIME_KEYS} == item['producer_runtime']
                    == {name: plan[name] for name in study.RUNTIME_KEYS}, 'Historical runtime differs')
            require(producer['input_identity'] == item['producer_input_identity'] == plan['input_identity'],
                    'Historical input identity differs')
            require(audit['origin'] == 'historical', 'Historical checkpoint mislabeled as new')
            for filename, field in [('metrics.json', 'metrics_sha256'), ('checkpoint.pt', 'checkpoint_sha256')]:
                require(study.digest(directory / filename) == item[field] == audit[field],
                        'Historical artifact bytes changed: ' + filename)
            producers[item['producer_plan_sha256']] = producer_path
        else:
            directory = run / 'fits' / row['id'] / str(row['learning_rate']) / ('seed' + str(row['seed']))
            expected_new_metrics.add(directory / 'metrics.json')
            require(audit['origin'] == 'new' and audit['producer_source_revision'] == plan['source_revision'],
                    'New checkpoint producer identity differs')
        base = {name: value for name, value in row.items() if name not in TEST_FIELDS}
        require(read(directory / 'metrics.json') == base, 'Raw validation fit differs from retained summary')
        require(read(directory / 'history.json') == row['history'], 'Raw learning curve differs from fit record')
        require(study.digest(directory / 'metrics.json') == audit['metrics_sha256'], 'Validation metrics bytes changed')
        checkpoint = directory / 'checkpoint.pt'
        require(study.digest(checkpoint) == audit['checkpoint_sha256'] == row['checkpoint_sha256'],
                'Checkpoint bytes differ from training/test audits')
        replay = run / 'test_replay' / row['id'] / str(row['learning_rate']) / ('seed' + str(row['seed']) + '.json')
        require(read(replay) == row, 'Saved test replay differs from summary')
        expected_test_replays.add(replay)
        checkpoint_files[str(checkpoint)] = audit['checkpoint_sha256']
        fit_records.append(dict(phase=phase, origin=audit['origin'], directory=str(directory),
            producer_source_revision=audit['producer_source_revision'], metrics_sha256=audit['metrics_sha256'],
            checkpoint_sha256=audit['checkpoint_sha256'], record=base))
        flat.append(dict(phase=phase, id=row['id'], seed=row['seed'], learning_rate=row['learning_rate'],
            epochs=plan['epochs'], best_epoch=row['best_epoch'], total_params=row['total_params'],
            trainable_params=row.get('trainable_params', row['total_params']),
            n_fit=len(row['fit_names']), n_val=len(row['val_names']), n_test=len(row['test_names']),
            val_iou=row['val']['iou'], optical_val_iou=row['val']['optical_subset_iou'],
            sar_val_iou=row['val']['sar_subset_iou'], test_iou=row['test']['iou'],
            optical_test_iou=row['test']['optical_subset_iou'], sar_test_iou=row['test']['sar_subset_iou'],
            elapsed_s=row['elapsed_s'], origin=audit['origin'],
            producer_source_revision=audit['producer_source_revision'],
            checkpoint_sha256=audit['checkpoint_sha256'], metrics_sha256=audit['metrics_sha256']))
    require(set((run / 'fits').rglob('metrics.json')) == expected_new_metrics, 'Unexpected or missing new fit record inventory')
    require(set((run / 'test_replay').rglob('*.json')) == expected_test_replays, 'Unexpected or missing test replay inventory')
    recomputed = copy.deepcopy(summary)
    recomputed['validation_analysis'] = study.summarize(records, specs)
    recomputed['test_analysis'] = study.summarize(records, specs, test=True)
    require(summary == recomputed, 'Saved analyses differ from recomputed paired analysis')
    report = study.format_report(recomputed)
    require((run / 'report.md').read_text() == report, 'Saved report differs from recomputed result')
    return dict(run=run, plan=plan, summary=recomputed, report=report,
                flat=flat, fits=fit_records, checkpoints=checkpoint_files, producers=producers)


def audit_pair(matched, longer):
    left, right = matched['plan'], longer['plan']
    for key in (*study.RUNTIME_KEYS, 'input_identity', 'source_hashes'):
        require(left[key] == right[key], 'Cross-phase input/source/runtime differs: ' + key)
    early = keyed(matched['summary']['records'], 'matched records')
    late = keyed(longer['summary']['records'], 'long records')
    prefix_checks = []
    for key, row in early.items():
        other = late[key]
        for field in ('fit_names', 'val_names', 'test_names', 'fit_indices', 'val_indices',
                      'normalization', 'initial_parameter_sha256', 'total_params'):
            require(row[field] == other[field], 'Cross-phase paired identity differs: ' + field)
        def config(record):
            return {name: value for name, value in study.canonical(record['settings']).items()
                    if name not in ('num_epochs', 'data_root')}
        require(config(row) == config(other), 'Cross-phase model differs beyond epoch budget')
        prefix = other['history'][:61]
        count_keys = [sensor + '_subset_' + count for sensor in ('optical', 'sar')
                      for count in ('tp', 'fp', 'fn', 'tn', 'n_samples')]
        validation_differences, loss_differences, changed_classifications = [], [], []
        for early_point, late_point in zip(row['history'], prefix):
            require(early_point['epoch'] == late_point['epoch'], 'Cross-phase prefix epoch order differs')
            if early_point['train_loss'] != late_point['train_loss']:
                loss_differences.append(dict(epoch=early_point['epoch'],
                    matched60=early_point['train_loss'], long120=late_point['train_loss'],
                    difference=late_point['train_loss'] - early_point['train_loss']))
            if 'val' in early_point:
                differences = {name: late_point['val'][name] - value
                               for name, value in early_point['val'].items() if late_point['val'][name] != value}
                if differences:
                    validation_differences.append(dict(epoch=early_point['epoch'], differences=differences))
                if any(early_point['val'][name] != late_point['val'][name] for name in count_keys):
                    changed_classifications.append(early_point['epoch'])
        prefix_checks.append(dict(id=key[0], seed=key[1], learning_rate=key[2],
            epoch_order_equal=True,
            exact_prefix_history_equal=prefix == row['history'],
            training_losses_exactly_equal=not loss_differences,
            validation_metrics_exactly_equal=not validation_differences,
            validation_classification_counts_exactly_equal=not changed_classifications,
            changed_validation_classification_epochs=changed_classifications,
            training_loss_differences=loss_differences,
            validation_metric_differences=validation_differences,
            max_prefix_validation_iou_difference=max(abs(a['val']['iou'] - b['val']['iou'])
                for a, b in zip(prefix, row['history']) if 'val' in a),
            max_prefix_train_loss_difference=max(abs(a['train_loss'] - b['train_loss'])
                for a, b in zip(prefix[1:], row['history'][1:]))))
    data_root = Path(longer['summary']['records'][0]['settings']['data_root'])
    require(study.pilot.verify_inputs(data_root) == right['input_identity'],
            'Current raw input bytes differ from frozen plans')
    return prefix_checks


def duration_comparison(matched, longer):
    early = keyed(matched['summary']['records'], 'matched records')
    late = keyed(longer['summary']['records'], 'long records')
    results = []
    for metric, name, rate in itertools.product(('val', 'test'), study.QUANTUM, study.RATES):
        differences = {str(seed): late[name, seed, rate][metric]['iou'] - early[name, seed, rate][metric]['iou']
                       for seed in study.SEEDS}
        values = list(differences.values())
        results.append(dict(metric=metric, id=name, learning_rate=rate, differences=differences,
            mean=statistics.mean(values), std=statistics.stdev(values), wins=sum(value > 0 for value in values),
            n=len(values), unit='IoU'))
    return results


def stability_summaries(matched, longer):
    """Describe loss trajectories without classifying normal warmup as a defect."""
    rows = []
    for phase, artifact in (('matched60', matched), ('long120', longer)):
        for record in artifact['summary']['records']:
            losses = [point for point in record['history'] if point['train_loss'] is not None]
            initial = losses[0]
            maximum = max(losses, key=lambda point: (point['train_loss'], -point['epoch']))
            after10 = max((point for point in losses if point['epoch'] > 10),
                          key=lambda point: (point['train_loss'], -point['epoch']))
            jumps = [(later['train_loss'] - earlier['train_loss'], later['epoch'])
                     for earlier, later in zip(losses, losses[1:])
                     if later['epoch'] == earlier['epoch'] + 1]
            positive = [(jump, epoch) for jump, epoch in jumps if jump > 0]
            largest = max(positive, key=lambda item: (item[0], -item[1])) if positive else (0.0, None)
            rows.append(dict(phase=phase, id=record['id'], seed=record['seed'],
                learning_rate=record['learning_rate'], epochs=artifact['plan']['epochs'],
                initial_train_loss=initial['train_loss'], initial_train_loss_epoch=initial['epoch'],
                max_train_loss=maximum['train_loss'], max_train_loss_epoch=maximum['epoch'],
                max_train_loss_after_epoch10=after10['train_loss'],
                max_train_loss_after_epoch10_epoch=after10['epoch'],
                max_positive_adjacent_loss_jump=largest[0], max_positive_adjacent_loss_jump_epoch=largest[1],
                selected_checkpoint_epoch=record['best_epoch'],
                selected_checkpoint_late=record['best_epoch'] >= artifact['plan']['epochs'] - artifact['plan']['eval_every']))
    return rows


def stability_report(rows):
    lines = ['# Descriptive training stability summaries', '',
        'Every loss is the recorded per-epoch optimization Dice plus cross-entropy loss. '
        'An initially high loss is not automatically instability. The maximum strictly after epoch 10 is a specified descriptive summary, '
        'not a diagnostic threshold or a criterion for selecting models.',
        'Adjacent-epoch jumps use later minus earlier loss; zero with no epoch means no positive increase occurred. '
        'Ties select the earliest epoch. A late checkpoint was selected in the final evaluation window (last two epochs).', '',
        '| Budget | Model | Seed | Fixed LR | Initial loss | Maximum loss (epoch) | Maximum after epoch 10 (epoch) | Largest positive adjacent jump (epoch) | Selected epoch; late |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        jump_epoch = row['max_positive_adjacent_loss_jump_epoch']
        jump_label = 'none' if jump_epoch is None else str(jump_epoch)
        lines.append(f"| {row['epochs']} | {row['id']} | {row['seed']} | {row['learning_rate']:g} | "
                     f"{row['initial_train_loss']:.4f} | {row['max_train_loss']:.4f} ({row['max_train_loss_epoch']}) | "
                     f"{row['max_train_loss_after_epoch10']:.4f} ({row['max_train_loss_after_epoch10_epoch']}) | "
                     f"{row['max_positive_adjacent_loss_jump']:.4f} ({jump_label}) | "
                     f"{row['selected_checkpoint_epoch']}; {'yes' if row['selected_checkpoint_late'] else 'no'} |")
    lines += ['', 'These summaries describe observed trajectories; they do not establish significance, an optimization mechanism or a causal quantum benefit.', '']
    return '\n'.join(lines)


def save_figure(fig, out, stem, title):
    timestamp = datetime(2026, 10, 27, tzinfo=timezone.utc)
    fig.savefig(out / (stem + '.png'), dpi=150, bbox_inches='tight', metadata={'Software': 'QUAFFLE local rate study'})
    fig.savefig(out / (stem + '.pdf'), bbox_inches='tight', metadata=dict(
        Title=title, Author='QUAFFLE', Creator='QUAFFLE local rate study',
        CreationDate=timestamp, ModDate=timestamp))


def plot_curves(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = (*study.QUANTUM, 'mlp_patch4x4')
    labels = ('Entangled global circuit', 'Separable global circuit', 'Patchwise classical MLP')
    colors = ('#0072B2', '#D55E00', '#009E73')
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True)
    for index, (name, label) in enumerate(zip(names, labels)):
        for column, rate in enumerate(study.RATES):
            axis = axes[index, column]
            for seed, color in zip(study.SEEDS, colors):
                row = next(row for row in summary['records'] if study.cell_key(row) == (name, seed, rate))
                points = [point for point in row['history'] if 'val' in point]
                axis.plot([point['epoch'] for point in points], [point['val']['iou'] for point in points],
                          color=color, label='Seed ' + str(seed), linewidth=1.4)
            axis.axvline(60, color='0.4', linestyle='--', linewidth=1)
            axis.set(title=label + f'\nLR {rate:g}', xlim=(0, 120), ylim=(0, 1))
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel('Validation IoU')
            if index == 2:
                axis.set_xlabel('Epoch')
    axes[0, 0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Full 120-epoch validation trajectories; dashed line marks epoch 60', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out, 'validation_curves_120ep', 'Full 120-epoch validation trajectories')
    plt.close(fig)


def plot_training_curves(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = (*study.QUANTUM, 'mlp_patch4x4')
    labels = ('Entangled global circuit', 'Separable global circuit', 'Patchwise classical MLP')
    colors = ('#0072B2', '#D55E00', '#009E73')
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=False)
    for index, (name, label) in enumerate(zip(names, labels)):
        for column, rate in enumerate(study.RATES):
            axis = axes[index, column]
            for seed, color in zip(study.SEEDS, colors):
                row = next(row for row in summary['records'] if study.cell_key(row) == (name, seed, rate))
                points = [point for point in row['history'] if point['train_loss'] is not None]
                axis.plot([point['epoch'] for point in points], [point['train_loss'] for point in points],
                          color=color, label='Seed ' + str(seed), linewidth=1.4)
            axis.axvline(60, color='0.4', linestyle='--', linewidth=1)
            axis.set(title=label + f'\nLR {rate:g}', xlim=(0, 120), yscale='linear')
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel('Training loss (Dice + cross-entropy)')
            if index == 2:
                axis.set_xlabel('Epoch')
    axes[0, 0].legend(loc='best', fontsize=9)
    fig.suptitle('Full 120-epoch training loss; linear axes with independent panel scales', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out, 'training_loss_curves_120ep', 'Full 120-epoch training loss; independent linear panel scales')
    plt.close(fig)


def plot_entanglement(matched, longer, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#0072B2', '#D55E00', '#009E73')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for axis, metric, label in zip(axes, ('val', 'test'), ('Validation', 'Previously exposed test; descriptive')):
        groups = []
        for rate in study.RATES:
            for duration, artifact in ((60, matched), (120, longer)):
                cells = keyed(artifact['summary']['records'], 'plot records')
                groups.append([100 * (cells[study.QUANTUM[0], seed, rate][metric]['iou'] -
                                    cells[study.QUANTUM[1], seed, rate][metric]['iou']) for seed in study.SEEDS])
        for seed_index, (seed, color) in enumerate(zip(study.SEEDS, colors)):
            offset = (seed_index - 1) * 0.065
            for start in (0, 2):
                axis.plot([start + offset, start + 1 + offset],
                    [groups[start][seed_index], groups[start + 1][seed_index]],
                    marker='o', color=color, alpha=0.85, linewidth=1,
                    label='Seed ' + str(seed) if start == 0 else None)
        axis.scatter(range(4), [statistics.mean(group) for group in groups], marker='D',
                     facecolors='none', edgecolors='black', linewidths=1.4,
                     s=70, label='Paired mean', zorder=4)
        axis.axhline(0, color='0.35', linewidth=1)
        axis.set(title=label, xticks=range(4),
                 xticklabels=[f'{rate:g}\n{epochs} epochs' for rate in study.RATES for epochs in (60, 120)],
                 xlabel='Fixed learning rate and training budget')
        axis.grid(axis='y', alpha=0.2)
    axes[0].set_ylabel('Entangled minus separable IoU (percentage points)')
    axes[0].legend(fontsize=8, loc='best')
    fig.tight_layout()
    save_figure(fig, out, 'paired_entanglement_60_120', 'Paired fixed-rate entanglement contrasts')
    plt.close(fig)


def combined_report(matched, longer, duration, prefix_checks):
    lines = ['# Local rate and duration study', '',
        'The study completes 12 matched60 cells (eight audited historical fits and four new fits) and 18 fresh 120-epoch cells.',
        'Validation is the development metric. The previously inspected test split is descriptive, not an untouched confirmation set.',
        'All models use 64-pixel images, 128 optimization tiles, global-fit normalization excluding validation, and three paired seeds.',
        'The patchwise MLP is the strongest observed original classical control; its bottleneck spatial operation differs from the globally pooled circuits. Encoder and attention context remain unchanged.', '',
        '[60-epoch fixed-rate tables](matched60/README.md) · [120-epoch fixed-rate tables](long120/README.md)', '',
        '| Model | Fixed LR | Metric | 120 minus 60 IoU percentage points, mean ± sample SD | Wins |',
        '| --- | ---: | --- | ---: | ---: |']
    for row in duration:
        label = 'Validation' if row['metric'] == 'val' else 'Descriptive test'
        lines.append(f"| {row['id']} | {row['learning_rate']:g} | {label} | "
                     f"{100*row['mean']:+.3f} ± {100*row['std']:.3f} | {row['wins']}/{row['n']} |")
    exact = sum(row['exact_prefix_history_equal'] for row in prefix_checks)
    classifications = sum(row['validation_classification_counts_exactly_equal'] for row in prefix_checks)
    lines += ['', f'The first 60-epoch history is exactly identical in {exact}/{len(prefix_checks)} matching quantum cells; '
              f'validation classification counts are identical in {classifications}/{len(prefix_checks)}.',
              '[The prefix audit](prefix60_comparison.json) lists every training-loss and validation-metric difference without numerical suppression.']
    if exact != len(prefix_checks):
        lines += ['The independently rerun prefixes differ. Duration contrasts therefore include that observed trajectory variation; '
                  'they must not be attributed solely to extra epochs.']
    lines += ['', '![Full validation trajectories](validation_curves_120ep.png)', '',
        '![Full training-loss trajectories](training_loss_curves_120ep.png)', '',
        '[Descriptive loss and checkpoint summaries](STABILITY.md) retain every cell. '
        'The training-loss panels use independent linear y scales. An initially high loss is not automatically instability; '
        'the maximum after epoch 10 is a specified summary, not a diagnostic threshold.', '',
        '![Paired entanglement contrasts](paired_entanglement_60_120.png)', '',
        'Points show every paired seed; black diamonds show paired means. Connecting lines link the same seed/rate across training budgets.',
        'The full 120-epoch best-validation score includes the first 60 epochs by construction; that within-trajectory increase alone is not independent replication.',
        'All raw histories, parameter/gradient diagnostics, per-sensor metrics, checkpoint hashes and producer identities are retained. No significance, equivalence, computational advantage, Anvil or physical-hardware claim follows from this pilot.', '']
    return '\n'.join(lines)


def collect(matched_run, long_run, out_dir):
    out = Path(out_dir)
    require(not out.exists() or not any(out.iterdir()), 'Use a new or empty output directory')
    matched, longer = audit_run(matched_run, 'matched60'), audit_run(long_run, 'long120')
    prefix_checks = audit_pair(matched, longer)
    duration = duration_comparison(matched, longer)
    stability = stability_summaries(matched, longer)
    out.mkdir(parents=True, exist_ok=True)
    for phase, artifact in (('matched60', matched), ('long120', longer)):
        target = out / phase
        target.mkdir()
        for name in ('plan.json', 'validation_fits.json', 'checkpoint_audit.json'):
            shutil.copyfile(artifact['run'] / name, target / name)
        if phase == 'matched60':
            shutil.copyfile(artifact['run'] / 'reuse_audit.json', target / 'reuse_audit.json')
        study.pilot.write_json(target / 'summary.json', artifact['summary'])
        study.pilot.write_json(target / 'all_fits.json', artifact['fits'])
        (target / 'README.md').write_text(artifact['report'])
    producers = dict(matched['producers'], **longer['producers'])
    if producers:
        (out / 'historical_producers').mkdir()
        for name, path in sorted(producers.items()):
            shutil.copyfile(path, out / 'historical_producers' / (name + '.json'))
    flat = matched['flat'] + longer['flat']
    with (out / 'per_cell.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(flat)
    study.pilot.write_json(out / 'duration_comparison.json', duration)
    study.pilot.write_json(out / 'stability.json', stability)
    with (out / 'stability.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stability[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(stability)
    (out / 'STABILITY.md').write_text(stability_report(stability))
    study.pilot.write_json(out / 'prefix60_comparison.json', prefix_checks)
    study.pilot.write_json(out / 'checkpoint_sha256.json', dict(matched['checkpoints'], **longer['checkpoints']))
    (out / 'README.md').write_text(combined_report(matched, longer, duration, prefix_checks))
    plot_curves(longer['summary'], out)
    plot_training_curves(longer['summary'], out)
    plot_entanglement(matched, longer, out)
    study.pilot.write_json(out / 'audit.json', dict(
        collector_source_sha256=study.digest(Path(__file__)),
        matplotlib=importlib.metadata.version('matplotlib'),
        input_summary_sha256={phase: study.digest(artifact['run'] / 'summary.json')
                              for phase, artifact in (('matched60', matched), ('long120', longer))},
        completed_cells=dict(matched60=12, long120=18), historical_fits=8, new_fits=22,
        strict_checkpoint_hashes=len(matched['checkpoints']) + len(longer['checkpoints']),
        prefix60_checks=prefix_checks,
        checks=['exact executable inventories/hashes', 'runtime/input identity', 'current raw input hashes',
                'complete factorials', 'finite records', 'saved configuration and cohort identities',
                'checkpoint/replay audit and artifact hashes', 'historical producer manifests',
                'sensor confusion recombination', 'recomputed paired/sample statistics',
                'cross-phase initialization/cohorts/configuration'],
        figure_reproducibility='Fixed PDF creation/modification metadata and PNG software metadata; compare all artifact bytes on regeneration'))
    manifest = {str(path.relative_to(out)): study.digest(path)
                for path in sorted(out.rglob('*')) if path.is_file()}
    study.pilot.write_json(out / 'artifact_sha256.json', manifest)
    print('VERIFIED LOCAL RATE STUDY: 12 matched cells, 18 long cells, 8 historical fits, 22 new fits', flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched-run', type=Path, required=True)
    parser.add_argument('--long-run', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    return collect(args.matched_run, args.long_run, args.out_dir)


if __name__ == '__main__':
    main()
