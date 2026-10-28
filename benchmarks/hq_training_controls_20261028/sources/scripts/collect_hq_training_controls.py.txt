#!/usr/bin/env python3
"""Audit the fixed-rate HQ training controls and export a deterministic report.

This collector restores checkpoints and checks saved evaluation records. It does
not train models or independently repeat inference. Original legacy observations
remain historical evidence; reused fits are never counted as fresh replications.
"""
import argparse
from collections import Counter
import csv
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import shlex
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_hq_dfc24_direction as legacy
from scripts import run_hq_training_controls as study

require, digest, read, write, inside = (legacy.require, legacy.digest, legacy.read,
                                      legacy.write, legacy.inside)
SCORES = ('iou', 'optical_subset_iou', 'sar_subset_iou')
LABELS = dict(quantum='Learned quantum', frozen_quantum='Frozen quantum',
              zero_bottleneck='Zero bottleneck', compact_affine='Compact affine', mlp='MLP')
PRIMARY_RATE = 0.003


def identity(row):
    return row['kind'], row['seed'], row['learning_rate']


def verify_source(plan):
    require(set(plan['source_hashes']) == set(study.SOURCE_FILES), 'Producer source inventory differs')
    for name, expected in plan['source_hashes'].items():
        require(digest(inside(ROOT, name)) == expected, 'Producer source differs: ' + name)
    require(digest(ROOT / study.PROTOCOL) == plan['protocol_sha256'], 'Frozen protocol differs')
    require(plan['legacy_source_hashes'] == {name: digest(inside(ROOT, name))
        for name in legacy.study.SOURCE_FILES}, 'Legacy executable source differs')


def audit_checkpoint(path, row):
    model, audit = study.restore_checkpoint(path, row, device='cpu')
    factory = legacy.study.model_from_config if row['kind'] in legacy.study.KINDS else study.model_from_config
    initial_model = factory(row['model_config'])
    initial = {name: value.detach().clone() for name, value in initial_model.named_parameters()}
    require(study.movement(model, initial) == row['diagnostics']['movement'],
            'Checkpoint parameter movement differs')
    expected_flags = {name: parameter.requires_grad for name, parameter in initial_model.named_parameters()}
    actual_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    require(actual_flags == expected_flags, 'Restored parameter trainability differs')
    require(model.bottleneck.mode == ('zero_bottleneck' if row['kind'] == 'zero_bottleneck' else 'on'),
            'Restored training-control mode differs')
    require(sum(p.numel() for p in model.parameters()) == row['total_params'] and
            sum(p.numel() for p in model.parameters() if p.requires_grad) == row['trainable_params'] and
            sum(p.numel() for p in model.bottleneck.core_parameters()) == row['core_params'],
            'Restored model/core/trainable parameter counts differ')
    if row['kind'] in ('frozen_quantum', 'zero_bottleneck'):
        require(all(not p.requires_grad for p in model.bottleneck.core_parameters()),
                'Inactive/frozen core unexpectedly trainable')
        require(row['diagnostics']['movement']['core']['l2'] == 0 and
                row['diagnostics']['gradient_norms']['core']['max'] == 0,
                'Inactive/frozen core changed or recorded a gradient')
    else:
        require(all(p.requires_grad for p in model.bottleneck.core_parameters()),
                'Active core unexpectedly frozen')
    del model, initial_model
    return audit


def verify_gradient_diagnostics(row, args):
    # A frozen or disabled core still records all optimizer-batch observations;
    # zero gradient norms here denote inactivity rather than missing coverage.
    legacy.validate_diagnostics(row, args)
    if row['kind'] in study.NEW_KINDS:
        require(row['diagnostics']['angle_probe_scope'] == (
            'inactive_counterfactual' if row['kind'] == 'zero_bottleneck' else 'active_fixed_probe'),
            'Angle probe scope differs from training-control activity')
    expected_steps = math.ceil(args.fit_cap / args.batch_size) * args.epochs
    for group in ('core', 'common'):
        recorded = row['diagnostics']['gradient_norms'][group]
        epoch_values = [point['gradient_norms'][group] for point in row['history'][1:]]
        for point in [recorded, *epoch_values]:
            require((point['max'] == 0) == (point['nonzero_steps'] == 0) and
                    (point['min'] == 0 or point['nonzero_steps'] == point['steps']),
                    'Gradient zero/nonzero counts contradict their range')
        require(recorded['steps'] == expected_steps, 'Incomplete optimizer batch coverage')
        mean = statistics.mean(point['mean'] for point in epoch_values)
        require(math.isclose(recorded['mean'], mean, rel_tol=1e-12, abs_tol=1e-15),
                'Aggregate gradient mean differs from epoch summaries')


def file_inventory(root):
    root = Path(root)
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*')) if path.is_file()}


def audit_campaign(work_dir, data_root):
    work_dir, data_root = Path(work_dir).resolve(), Path(data_root).resolve()
    plan, summary = read(work_dir / 'plan.json'), read(work_dir / 'summary.json')
    require(summary['plan'] == plan, 'Summary/frozen plan differ')
    verify_source(plan)
    runtime = legacy.verify_runtime(plan)
    settings = dict(device=plan['settings']['device'], batch_size=8, epochs=60, fit_cap=128,
                    image_size=128, base_channels=8, eval_every=2, augment=True)
    require(plan['settings'] == settings and plan['settings']['augment'] is True,
            'Frozen training-control budget differs')
    require(all(type(plan['settings'][name]) is int for name in
        ('batch_size', 'epochs', 'fit_cap', 'image_size', 'base_channels', 'eval_every')),
        'Frozen integer budgets have invalid types')
    require(plan['controls'] == list(study.KINDS) and plan['seeds'] == list(study.SEEDS) and
            plan['learning_rates'] == list(study.RATES) and plan['primary_rate'] == PRIMARY_RATE,
            'Frozen primary/auxiliary grid differs')
    for name, expected in dict(
        primary_policy='Fixed 0.003 for all controls/seeds; no rate reselection',
        test_policy='All 20 fits audited before 15 new primary replays; exposed test descriptive',
        auxiliary_policy='Rate0.001 on seed42 for all five controls; sensitivity only').items():
        require(plan[name] == expected, 'Frozen study policy differs: ' + name)
    require(plan['settings']['device'] == plan['runtime']['device'].split(':')[0],
            'Recorded producing device differs')
    legacy_root = Path(plan['legacy_root']).resolve()
    require(legacy_root != work_dir and not legacy_root.is_relative_to(work_dir) and
            not work_dir.is_relative_to(legacy_root), 'Fresh and legacy evidence roots overlap')
    starting_raw_files = dict(fresh=file_inventory(work_dir), legacy=file_inventory(legacy_root))
    require(digest(legacy_root / 'plan.json') == plan['legacy_plan_sha256'], 'Legacy plan bytes differ')
    old = legacy.audit_campaign(legacy_root, data_root)
    require(old['plan']['source_hashes'] == plan['legacy_source_hashes'] and
            old['raw_files'] == plan['legacy_raw_files'], 'Legacy source/raw artifact identities differ')
    require(plan['runtime'] == old['plan']['runtime'], 'Fresh and reused producing runtimes differ')
    require(plan['input_identity'] == old['plan']['input_identity'], 'Fresh/reused input identities differ')
    inputs = study.pilot.verify_inputs(data_root)
    require(inputs == plan['input_identity'] and inputs['verified_files'] == 3476, 'Actual input bytes differ')
    manifest = read(data_root / 'splits/flood_combined.json')
    require(manifest['conflict_policy'] == 'drop' and len(manifest['train']) == 1533 and
            len(manifest['test']) == 205, 'Committed conflict-free manifest differs')
    entries = read(work_dir / 'all_fits.json')
    require(len(entries) == 20 and Counter(entry['origin'] for entry in entries) ==
            dict(fresh=12, reused=8), 'Expected twelve fresh and eight reused fit records')
    require(read(work_dir / 'fit_progress.json') == entries, 'Completed progress ledger differs')
    produced = read(work_dir / 'checkpoint_audit.json')
    require(len(produced) == 20 and len({(row['id'], row['seed'], row['learning_rate'])
        for row in produced}) == 20, 'Incomplete/duplicate producer checkpoint audit')
    produced_by_key = {(row['id'], row['seed'], row['learning_rate']): row for row in produced}
    boundary = read(work_dir / 'fit_boundary.json')
    require(boundary == dict(completed=True, fits=20, fresh=12, reused=8, primary=15,
        all_fits_sha256=digest(work_dir / 'all_fits.json'),
        checkpoint_audit_sha256=digest(work_dir / 'checkpoint_audit.json')),
        'Invalid completed-fit boundary before primary test replays')
    profile = work_dir / 'resource_profile.json'
    require(profile.is_file(), 'Missing bound resource profile')
    require(digest(profile) == plan['resource_profile_sha256'], 'Bound resource profile differs')
    probe = read(profile)
    require(probe['status'] == 'passed' and probe['source_hashes'] == plan['source_hashes'] and
            probe['protocol_sha256'] == plan['protocol_sha256'] and probe['runtime'] == plan['runtime'],
            'Resource profile source/runtime differs')
    require(len(probe['cases']) == len(study.NEW_KINDS) and
            {case['kind'] for case in probe['cases']} == set(study.NEW_KINDS) and
            all(case['batch_size'] == 8 and case['strict_restore'] and case['frozen_unchanged']
                for case in probe['cases']), 'Incomplete actual-control resource profile')
    rows, paths, fresh_audits = {}, {}, []
    old_by_key = {identity(entry['record']): entry for entry in old['fits']}
    fresh_roots = set()
    for seed in study.SEEDS:
        load_args = argparse.Namespace(**settings, data_root=data_root)
        data = study.pilot.load_training_data(load_args, seed)
        require(len(data['fit']) == 128 and len(data['all_fit']) == 1303 and
                len(data['validation']) == 230, 'Actual optimization/global-fit/validation split differs')
        for entry in (entry for entry in entries if entry['record']['seed'] == seed):
            row = entry['record']
            key = identity(row)
            require(key not in rows and row['id'] == 'hq_' + row['kind'] and key in study.expected_grid(),
                    'Duplicate/unplanned fit identity')
            require(not ({'test', 'test_names', 'knockout', 'checkpoint'} & set(row)),
                    'Fit record crosses primary test boundary')
            directory = study.entry_directory(work_dir, legacy_root, entry)
            if entry['origin'] == 'reused':
                require(key in old_by_key and row == old_by_key[key]['record'] and
                        entry['directory'] == old_by_key[key]['directory'], 'Reused fit differs from original record')
            else:
                require(row['kind'] in study.NEW_KINDS, 'A baseline was incorrectly labeled a fresh fit')
                require(entry['directory'] == str(Path('fits') / row['id'] / str(row['learning_rate']) /
                        ('seed' + str(seed))), 'Unexpected fresh fit directory')
                fresh_roots.add(row['settings']['data_root'])
            for filename in ('metrics.json', 'history.json', 'checkpoint.pt'):
                require(digest(directory / filename) == entry[filename.split('.')[0] + '_sha256'],
                        'Raw fit artifact digest differs: ' + filename)
            require(read(directory / 'metrics.json') == row and read(directory / 'history.json') == row['history'],
                    'Raw fit/history bytes differ from complete-grid records')
            audit_args = argparse.Namespace(**settings, data_root=Path(row['settings']['data_root']))
            study.audit_fit(row, row['kind'], seed, row['learning_rate'], audit_args, data)
            require(row['elapsed_s'] > 0 and row['fit_sensor_counts'] ==
                    dict(Counter(data['dataset'].modalities[index] for index in data['fit'])),
                    'Fit duration or optimization sensor counts differ')
            for point in row['history']:
                if 'val' in point:
                    legacy.check_metrics(point['val'], 230)
            verify_gradient_diagnostics(row, audit_args)
            details = audit_checkpoint(directory / 'checkpoint.pt', row)
            checked = dict(id=row['id'], seed=seed, learning_rate=row['learning_rate'],
                           origin=entry['origin'], **details)
            require(produced_by_key.get((row['id'], seed, row['learning_rate'])) == checked,
                    'Producer checkpoint audit differs from restored bytes')
            fresh_audits.append(checked)
            rows[key], paths[key] = row, directory
        del data
        gc.collect()
    require(len(fresh_roots) == 1 and Path(next(iter(fresh_roots))).is_absolute(),
            'Fresh producing data roots differ')
    study.paired_audit(list(rows.values()))
    records = summary['records']
    require(len(records) == 15 and {identity(row) for row in records} == {
        (kind, seed, PRIMARY_RATE) for kind in study.KINDS for seed in study.SEEDS},
        'Incomplete fifteen matched primary results')
    selected_paths = set()
    for row in records:
        key = identity(row)
        require({key: value for key, value in row.items() if key not in ('test', 'test_names', 'checkpoint')} == rows[key],
                'Primary replay differs from audited fit')
        entry = next(entry for entry in entries if identity(entry['record']) == key)
        require(row['checkpoint'] == dict(origin=entry['origin'],
                path=str(Path(entry['directory']) / 'checkpoint.pt')), 'Primary checkpoint location differs')
        replay = work_dir / 'replays' / row['id'] / ('seed' + str(row['seed']) + '.json')
        payload = read(replay)
        study.validate_replay(payload, entry, digest(work_dir / 'plan.json'), manifest['test'])
        require(payload == dict(binding=study.replay_binding(entry, digest(work_dir / 'plan.json'), manifest['test']),
                                record=row), 'Saved primary replay differs from summary/binding')
        require(row['test_names'] == manifest['test'] and len(set(row['test_names'])) == 205,
                'Primary test names/order differ')
        legacy.check_metrics(row['test'], 205, test=True)
        selected_paths.add(replay)
    require(set((work_dir / 'replays').rglob('*.json')) == selected_paths, 'Extra/missing primary replay')
    for filename in ('metrics.json', 'history.json', 'checkpoint.pt'):
        expected = {paths[identity(entry['record'])] / filename for entry in entries if entry['origin'] == 'fresh'}
        require(set(work_dir.rglob(filename)) == expected, 'Extra/missing fresh fit artifact: ' + filename)
    require(not list(work_dir.rglob('failure.json')), 'Failed fitting attempt needs separate explicit accounting')
    require(not list((work_dir / 'replay_failures').rglob('*.json')),
            'Failed replay attempt needs separate explicit accounting')
    require(summary['analysis'] == study.analysis(records), 'Saved primary summary/paired analysis differs')
    raw_files = dict(fresh=file_inventory(work_dir), legacy=file_inventory(legacy_root))
    require(raw_files == starting_raw_files, 'Raw artifacts changed during collection')
    require(raw_files['legacy'] == plan['legacy_raw_files'], 'Original legacy inventory changed')
    verify_source(plan)
    require(study.pilot.verify_inputs(data_root) == inputs, 'Input bytes changed during collection')
    parameter_counts = {}
    for kind in study.KINDS:
        model = study.build_model(kind, study.SEEDS[0], argparse.Namespace(**settings))
        parameter_counts[kind] = study.counts(model)
        reference = rows[kind, study.SEEDS[0], PRIMARY_RATE]
        require(parameter_counts[kind]['total'] == reference['total_params'] and
                parameter_counts[kind]['trainable'] == reference['trainable_params'] and
                parameter_counts[kind]['core_total'] == reference['core_params'],
                'Detailed parameter accounting differs from audited fit')
        del model
    return dict(plan=plan, summary=summary, fits=entries, checkpoint_audit=sorted(fresh_audits,
        key=lambda row: (row['id'], row['seed'], row['learning_rate'])), collector_runtime=runtime,
        raw_files=raw_files, parameter_counts=parameter_counts)


def stats(values):
    values = list(values)
    require(len(values) >= 2, 'At least two paired observations are required')
    return dict(mean=statistics.mean(values), std=statistics.stdev(values), n=len(values))


def comparison(result):
    records = result['summary']['records']
    by_key = {(row['kind'], row['seed']): row for row in records}
    pairs = []
    # Quantum-vs-each-control is declared before looking at any test outcome.
    for right in study.KINDS:
        if right == 'quantum':
            continue
        for split in ('val', 'test'):
            for metric in SCORES:
                values = {str(seed): 100 * (by_key['quantum', seed][split][metric] -
                    by_key[right, seed][split][metric]) for seed in study.SEEDS}
                pairs.append(dict(left='quantum', right=right, split=split, metric=metric,
                    **stats(values.values()), wins=sum(value > 0 for value in values.values()),
                    differences_pp=values, common_learning_rate=PRIMARY_RATE))
    return dict(paired=pairs, unit='IoU percentage points',
        scope='Fixed-rate paired DFC24 training controls; validation primary; exposed test descriptive',
        primary_learning_rate=PRIMARY_RATE)


def tables(result, contrasts):
    selected = result['summary']['records']
    summaries, per_seed, auxiliary, trainability = [], [], [], []
    for control in study.KINDS:
        group = [row for row in selected if row['kind'] == control]
        row = dict(kind=control, learning_rate=PRIMARY_RATE,
            total_params=group[0]['total_params'], trainable_params=group[0]['trainable_params'],
            core_params=group[0]['core_params'], **result['parameter_counts'][control])
        for split in ('val', 'test'):
            for metric in SCORES:
                row.update({f'{split}_{metric}_{key}': value
                    for key, value in stats(value[split][metric] for value in group).items()})
        summaries.append(row)
    for row in sorted(selected, key=identity):
        flat = {key: row[key] for key in ('kind', 'seed', 'learning_rate', 'best_epoch',
                                         'total_params', 'trainable_params', 'core_params')}
        flat.update({f'{split}_{key}': value for split in ('val', 'test')
                     for key, value in row[split].items()})
        per_seed.append(flat)
    for entry in sorted(result['fits'], key=lambda entry: identity(entry['record'])):
        row = entry['record']
        auxiliary.append(dict(control=row['kind'], seed=row['seed'], learning_rate=row['learning_rate'],
            role='matched_primary' if row['learning_rate'] == PRIMARY_RATE else 'auxiliary_rate',
            origin=entry['origin'], best_epoch=row['best_epoch'],
            **{'val_' + key: value for key, value in row['val'].items()}))
        diagnostics = row['diagnostics']
        for group in ('core', 'common'):
            gradient = diagnostics['gradient_norms'][group]
            trainability.append(dict(control=row['kind'], seed=row['seed'],
                learning_rate=row['learning_rate'], origin=entry['origin'], group=group,
                total_params=row['total_params'], trainable_params=row['trainable_params'],
                core_params=row['core_params'],
                angle_probe_scope=diagnostics.get('angle_probe_scope', 'active_fixed_probe'),
                movement_l2=diagnostics['movement'][group]['l2'],
                relative_movement_l2=diagnostics['movement'][group]['relative_l2'],
                gradient_min=gradient['min'], gradient_mean=gradient['mean'], gradient_max=gradient['max'],
                gradient_steps=gradient['steps'], gradient_nonzero_steps=gradient['nonzero_steps']))
    paired = [{**{key: value for key, value in row.items() if key != 'differences_pp'},
        **{'seed' + seed + '_pp': value for seed, value in row['differences_pp'].items()}}
        for row in contrasts['paired']]
    return dict(summary=summaries, per_seed=per_seed, paired=paired,
                tuning=auxiliary, trainability=trainability)


def report(result, contrasts, rows):
    lines = ['# HQ training controls on DFC24', '',
        'Validation is primary. The previously exposed test split remains descriptive; no fresh holdout or independent confirmation is claimed.', '',
        'Five controls share the same depthwise U-Net, interfaces and initial non-core state, including BatchNorm buffers. The primary contrast fixes Adam learning rate 0.003 for every control and seed. Seed-42 observations at 0.001 are retained as auxiliary rate checks, without selecting a rate or changing the primary contrast. A common fixed rate isolates this training setting; it is not an independently optimized ranking of every control.', '',
        'Twenty fit records comprise twelve fresh fits and eight byte-audited fits reused from the preceding HQ campaign. They are not twenty independent seeds. All twenty completed fits precede fifteen new saved descriptive test replays. Six older replays remain separately labeled historical evidence. The collector restores every checkpoint and checks saved replay identities and confusion-derived scores; it does not independently repeat inference.', '',
        'Three paired seeds; 128-pixel images; base width 8; actual batch 8; 60 epochs and 960 optimizer batches per fit. Each seed optimizes on 128 tiles. Normalization uses 1,303 global-fit images excluding 230 validation images, so this is not pure 128-shot learning. All 205 test tiles comprise 71 Optical and 134 SAR.', '',
        '| Control | Parameters / trainable | Core total / trainable | Bottleneck total / trainable | Validation IoU mean ± sample SD | Descriptive test IoU mean ± sample SD |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows['summary']:
        lines.append(f"| {LABELS[row['kind']]} | {row['total']:,} / {row['trainable']:,} | {row['core_total']:,} / {row['core_trainable']:,} | {row['bottleneck_total']:,} / {row['bottleneck_trainable']:,} | {row['val_iou_mean']:.4f} ± {row['val_iou_std']:.4f} | {row['test_iou_mean']:.4f} ± {row['test_iou_std']:.4f} |")
    lines += ['', 'Differences are computed within each paired seed before aggregation. Units are IoU percentage points; ± denotes sample SD, not a standard error or significance test.', '',
        '| Learned quantum minus control | Split | Mean ± sample SD (points) | Positive seeds |',
        '| --- | --- | ---: | ---: |']
    for row in contrasts['paired']:
        if row['metric'] == 'iou':
            lines.append(f"| {LABELS[row['right']]} | {row['split']} | {row['mean']:+.3f} ± {row['std']:.3f} | {row['wins']}/3 |")
    lines += ['',
        'The frozen quantum control retains differentiable input dependence while its circuit coefficients stay at initialization. The zero-bottleneck control removes the deepest reconstructed map throughout training; it differs from knocking out a trained model afterward. Compact affine is a cheap classical transformation control, and the larger MLP remains a separate strong control. Ordinary encoder-to-decoder skips remain available.', '',
        'Raw zero-bottleneck angle probes describe counterfactual encodings on an inactive path; no circuit uses those angles during that control\'s training or inference. Stored and trainable parameter counts also differ from the number of parameters that actively affect predictions.', '',
        'Training-time contrasts can constrain proposed mechanisms, but three paired seeds do not establish quantum advantage, equivalence, significance or a general impossibility result. Seed changes vary the initialization, optimization subset and validation cohort together. Shared initial state and data schedules are paired across arms.', '',
        'This is a noiseless local classical simulation and a binary DFC24 adaptation, not a reproduction of the motivating LandCover.ai result, a hardware experiment, an Anvil accuracy result or a throughput benchmark. Alan owns Anvil execution.', '',
        'Tables: [summary](summary.csv), [individual seeds](per_seed.csv), [paired contrasts](paired.csv), [all primary/auxiliary fits](tuning.csv), [trainability diagnostics](trainability.csv). Curves retain every fit, including auxiliary rates. [Provenance and regeneration](REPRODUCTION.md).', '']
    return '\n'.join(lines)


def plot_results(result, contrasts, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = Path(out_dir) / 'plots'
    plot_dir.mkdir()
    colors = {42: 'tab:blue', 77: 'tab:orange', 123: 'tab:green'}
    def save(fig, name, *, legend=False):
        bottom = (.12 if name == 'contrasts' else .05) if legend else 0
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.savefig(plot_dir / (name + '.png'), dpi=150, bbox_inches='tight')
        fig.savefig(plot_dir / (name + '.pdf'), bbox_inches='tight',
                    metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
    fig, axes = plt.subplots(len(study.KINDS), 2, figsize=(11, 13))
    for index, control in enumerate(study.KINDS):
        for entry in result['fits']:
            row = entry['record']
            if row['kind'] != control:
                continue
            primary = row['learning_rate'] == PRIMARY_RATE
            label = f"s{row['seed']} LR{row['learning_rate']}" + ('' if primary else ' auxiliary')
            for column, field in enumerate(('train_loss', 'val')):
                points = [point for point in row['history'] if point.get(field) is not None]
                axes[index, column].plot([point['epoch'] for point in points],
                    [point[field]['iou'] if field == 'val' else point[field] for point in points],
                    '-' if primary else '--', color=colors[row['seed']], alpha=1 if primary else .55,
                    label=label)
                axes[index, column].set(title=LABELS[control], xlabel='Epoch',
                    ylabel='Training Dice+CE' if field == 'train_loss' else 'Validation IoU')
                axes[index, column].legend(fontsize=7)
    save(fig, 'learning_curves')
    by_key = {(row['kind'], row['seed']): row for row in result['summary']['records']}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for i, split in enumerate(('val', 'test')):
        for j, metric in enumerate(SCORES):
            ax = axes[i, j]
            for seed in study.SEEDS:
                ax.plot(range(len(study.KINDS)),
                    [by_key[control, seed][split][metric] for control in study.KINDS],
                    'o-', color=colors[seed], label=f'seed {seed}')
            ax.set(xticks=range(len(study.KINDS)),
                xticklabels=[LABELS[control].replace(' ', '\n') for control in study.KINDS],
                ylabel='Foreground IoU', title=('Validation' if split == 'val' else 'Descriptive test')
                    + ': ' + metric.replace('_subset_', ' '))
            ax.tick_params(axis='x', labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    save(fig, 'paired_iou', legend=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    controls = [name for name in study.KINDS if name != 'quantum']
    for ax, split in zip(axes, ('val', 'test')):
        for index, control in enumerate(controls):
            row = next(row for row in contrasts['paired']
                if row['right'] == control and row['split'] == split and row['metric'] == 'iou')
            for offset, seed in zip((-.12, 0, .12), study.SEEDS):
                ax.scatter(index+offset, row['differences_pp'][str(seed)], color=colors[seed],
                           label=f'seed {seed}' if index == 0 else None)
            ax.errorbar(index, row['mean'], yerr=row['std'], color='black', fmt='_', capsize=4)
        ax.axhline(0, color='grey', linewidth=.8)
        ax.set(xticks=range(len(controls)), xticklabels=[LABELS[name].replace(' ', '\n') for name in controls],
               ylabel='Learned quantum − control IoU (points)',
               title='Validation' if split == 'val' else 'Descriptive test')
        ax.tick_params(axis='x', labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    save(fig, 'contrasts', legend=True)


def export(result, work_dir, data_root, out_dir):
    work_dir, data_root, out_dir = (Path(value).resolve() for value in (work_dir, data_root, out_dir))
    legacy_root = Path(result['plan']['legacy_root']).resolve()
    require(not out_dir.exists() or not any(out_dir.iterdir()), 'Use a new or empty output directory')
    for source in (work_dir, legacy_root, data_root, ROOT):
        require(out_dir != source and not source.is_relative_to(out_dir), 'Output contains an input/source root')
    for source in (work_dir, legacy_root, data_root):
        require(not out_dir.is_relative_to(source), 'Output aliases raw inputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep every producer JSON byte. In particular, old test replays stay under
    # the explicitly historical subtree instead of becoming new observations.
    roots = {'fresh': work_dir, 'legacy': legacy_root}
    for origin, inventory in result['raw_files'].items():
        for name, expected in inventory.items():
            path = inside(roots[origin], name)
            require(digest(path) == expected, 'Raw artifact changed before export: ' + name)
            if not name.endswith('.json'):
                continue
            target = out_dir / ('historical_legacy' if origin == 'legacy' else 'raw_campaign') / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    for name in ('plan.json', 'summary.json', 'all_fits.json'):
        (out_dir / name).write_bytes((work_dir / name).read_bytes())
    contrasts = comparison(result)
    rows = tables(result, contrasts)
    write(out_dir / 'checkpoint_audit.json', result['checkpoint_audit'])
    write(out_dir / 'parameter_counts.json', result['parameter_counts'])
    write(out_dir / 'comparison.json', contrasts)
    for name, values in rows.items():
        with (out_dir / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(values)
    source_manifest = {}
    sources = sorted(set(study.SOURCE_FILES) | set(legacy.study.SOURCE_FILES) |
        {study.PROTOCOL, legacy.PROTOCOL, 'scripts/collect_hq_training_controls.py',
         'scripts/collect_hq_dfc24_direction.py'})
    for name in sources:
        archived = 'sources/' + name + '.txt'
        target = out_dir / archived
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        source_manifest[name] = dict(sha256=digest(target), retained_as=archived)
    write(out_dir / 'source_manifest.json', source_manifest)
    (out_dir / 'input_manifest.json').write_bytes((data_root / 'splits/flood_combined.json').read_bytes())
    for name, expected in result['plan']['input_identity']['stats_files_sha256'].items():
        source = inside(data_root, name)
        require(digest(source) == expected, 'Input normalization metadata differs: ' + name)
        target = out_dir / 'input_metadata' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    provenance = dict(producing_revision=result['plan']['source_revision'],
        source_hashes=result['plan']['source_hashes'], protocol_sha256=result['plan']['protocol_sha256'],
        legacy_source_hashes=result['plan']['legacy_source_hashes'],
        legacy_plan_sha256=result['plan']['legacy_plan_sha256'],
        input_identity=result['plan']['input_identity'], producer_runtime=result['plan']['runtime'],
        collector_runtime=result['collector_runtime'],
        matplotlib_version=importlib.metadata.version('matplotlib'), collector_source_sha256=digest(__file__),
        raw_work_dir=str(work_dir), legacy_root=str(legacy_root), data_root=str(data_root),
        raw_artifact_sha256=result['raw_files'], audited_fits=20, fresh_fits=12, reused_fits=8,
        parameter_counts=result['parameter_counts'],
        audited_selected_saved_replays=15, retained_historical_replays=6, independent_inference_replays=0,
        checks=['full original legacy campaign audit plus exact legacy input/source/runtime/record binding',
            'all twenty checkpoint bytes and strict state/BN/trainability/control-mode restoration',
            'actual input bytes before/after; reconstructed cohorts, fit-only normalization and paired D4 schedules',
            'fixed-rate primary grid and retained auxiliary rates; first strict best epoch and complete histories',
            'new saved replay identities, complete test names and pooled/sensor confusion-derived metrics'])
    write(out_dir / 'provenance.json', provenance)
    (out_dir / 'README.md').write_text(report(result, contrasts, rows))
    command = ('systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= '
        'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '
        'MPLCONFIGDIR=/tmp/quaffle-hq-controls-collector-mpl PYTHONPATH=' +
        shlex.quote(str(ROOT) + ':/tmp/quaffle-wave4a-deps') + ' ' + shlex.quote(sys.executable) + ' ' +
        shlex.quote(str(ROOT / 'scripts/collect_hq_training_controls.py')) +
        ' --work-dir ' + shlex.quote(str(work_dir)) + ' --data-root ' + shlex.quote(str(data_root)) +
        ' --out-dir /tmp/quaffle-hq-training-controls-regenerated')
    (out_dir / 'REPRODUCTION.md').write_text('# Regenerate this report\n\n'
        'Use the source checkout matching every producer hash in `plan.json` and the collector hash in '
        '`provenance.json`. Both producer source families, protocols and collectors are archived byte-for-byte '
        'under inert `sources/*.txt` paths. A later Git revision is acceptable only when those executable '
        'bytes still match.\n\n```bash\n' + command + '\n```\n\n'
        'Use an empty output directory. The retained original legacy root is required as bound in the plan; '
        'all twelve fresh and eight reused checkpoint files must remain available locally. Image data may '
        'be relocated only if the complete input identity matches. No trained weights or images are included '
        'in this report. Original fit settings retain their historical data-root paths.\n\n'
        'The collector audits all twenty checkpoints and fifteen saved primary replays without independently '
        'repeating test inference. Six older replays are preserved under `historical_legacy` and are not fresh '
        'replications. `artifact_sha256.json` covers every generator output except itself. Both PNG and PDF '
        'figures are generated; PDF creation/modification dates are omitted for deterministic bytes. '
        'Subsequent interpretations, actual reviewer reports and external execution logs are separate evidence.\n')
    plot_results(result, contrasts, out_dir)
    for origin, inventory in result['raw_files'].items():
        require(file_inventory(roots[origin]) == inventory, 'Raw artifacts changed during export')
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
        require(not args.out_dir.resolve().is_relative_to(source.resolve()) and
                not source.resolve().is_relative_to(args.out_dir.resolve()), 'Output aliases or contains raw inputs')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    result = audit_campaign(args.work_dir, args.data_root)
    export(result, args.work_dir, args.data_root, args.out_dir)
    print('Audited20 checkpoints and15 saved primary replays; report:', args.out_dir)
    return result


if __name__ == '__main__':
    main()
