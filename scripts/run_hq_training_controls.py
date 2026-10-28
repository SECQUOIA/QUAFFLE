#!/usr/bin/env python3
"""Fixed-rate training controls for the existing local 16-qubit DFC24 screen."""
import argparse
from collections import Counter
import copy
import gc
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_hq_dfc24_direction as hq

KINDS = ('quantum', 'frozen_quantum', 'zero_bottleneck', 'compact_affine', 'mlp')
NEW_KINDS = ('frozen_quantum', 'zero_bottleneck', 'compact_affine')
SEEDS, RATES = hq.SEEDS, hq.RATES
PRIMARY_RATE = 0.003
PROTOCOL = 'docs/HQ_TRAINING_CONTROLS_PROTOCOL.md'
SOURCE_FILES = (*hq.SOURCE_FILES, 'scripts/run_hq_training_controls.py',
                'utilsTorch/hq_training_controls.py')
require, digest, canonical = hq.require, hq.digest, hq.canonical
pilot = hq.pilot
state_hash, shared_hash, finite_tensors = hq.state_hash, hq.shared_hash, hq.finite_tensors
movement, parameter_groups, evaluate = hq.movement, hq.parameter_groups, hq.evaluate


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def build_model(kind, seed, args):
    from utilsTorch.hq_training_controls import build_model as factory
    return factory(kind, seed, base_channels=args.base_channels, image_size=args.image_size)


def model_configuration(model):
    from utilsTorch.hq_training_controls import model_config
    return model_config(model)


def model_from_config(config):
    from utilsTorch.hq_training_controls import build_from_config
    return build_from_config(config)


def counts(model):
    from utilsTorch.hq_training_controls import parameter_counts
    return parameter_counts(model)


def trainability(model):
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def frozen_hash(model):
    return state_hash({name: parameter for name, parameter in model.named_parameters()
                       if not parameter.requires_grad})


def gradient_norms(model):
    result = {}
    for group, parameters in parameter_groups(model).items():
        active = [(name, p) for name, p in parameters if p.requires_grad]
        require(all(p.grad is None for _, p in parameters if not p.requires_grad),
                'A frozen parameter acquired a gradient')
        if group == 'core':
            require(all(p.grad is not None for _, p in active), 'A trainable core tensor has no gradient')
        gradients = [(name, p.grad) for name, p in active if p.grad is not None]
        finite_tensors(gradients, 'gradient')
        result[group] = math.sqrt(sum(float(g.detach().double().square().sum()) for _, g in gradients))
        require(math.isfinite(result[group]), 'Nonfinite gradient norm')
    return result


def fit_settings(kind, seed, rate, args, data):
    return hq.fit_settings(kind, seed, rate, args, data)


def audit_fit(row, kind, seed, rate, args, data):
    if kind in hq.KINDS:
        return hq.audit_fit(row, kind, seed, rate, args, data)
    require(kind in NEW_KINDS, 'Unknown training control')
    hq.finite_record(row)
    require((row['id'], row['kind'], row['seed'], row['learning_rate']) ==
            ('hq_' + kind, kind, seed, rate), 'Cached training-control identity differs')
    require(row['model_config'] == dict(control=kind, seed=seed,
            base_channels=args.base_channels, image_size=args.image_size), 'Cached model config differs')
    require(row['settings'] == canonical(fit_settings(kind, seed, rate, args, data)), 'Cached fit settings differ')
    for key, expected in dict(fit_indices=data['fit'], val_indices=data['validation'],
            fit_names=[data['dataset'].names[i] for i in data['fit']],
            val_names=[data['dataset'].names[i] for i in data['validation']],
            normalization=data['normalization'], n_global_fit=len(data['all_fit'])).items():
        require(row[key] == expected, 'Cached cohort/normalization differs: ' + key)
    require([p['epoch'] for p in row['history']] == list(range(args.epochs + 1)), 'Incomplete epoch history')
    require({p['epoch'] for p in row['history'] if 'val' in p} ==
            set(range(0, args.epochs + 1, args.eval_every)) | {args.epochs}, 'Invalid validation schedule')
    steps = math.ceil(len(data['fit']) / args.batch_size)
    for point in row['history']:
        require((point['train_loss'] is None) == (point['epoch'] == 0), 'Invalid training loss history')
        if point['epoch']:
            require(point['train_loss'] >= 0, 'Negative training loss')
            _, _, expected = hq.epoch_schedule(data['fit'], seed, point['epoch'], args.augment)
            require(point['train_schedule'] == dict(sha256=expected, n_samples=len(data['fit'])),
                    'Actual augmentation/data order differs')
            for group in ('core', 'common'):
                record = point['gradient_norms'][group]
                require(record['steps'] == steps and 0 <= record['nonzero_steps'] <= steps
                        and 0 <= record['min'] <= record['mean'] <= record['max'], 'Invalid gradient coverage')
            if kind in ('frozen_quantum', 'zero_bottleneck'):
                require(point['gradient_norms']['core']['max'] == 0, 'Frozen core has nonzero gradient')
        if 'val' in point:
            hq.validate_sensor_metrics(point['val'], len(data['validation']))
    best = max((p for p in row['history'] if 'val' in p), key=lambda p: p['val']['iou'])
    require(row['best_epoch'] == best['epoch'] and row['val'] == best['val'], 'Invalid checkpoint selection')
    frozen = row['diagnostics']['frozen_parameters']
    require(frozen['initial_sha256'] == frozen['selected_sha256'] and frozen['unchanged'] is True,
            'Frozen parameters changed')


def restore_checkpoint(path, row, device='cpu'):
    import torch
    if row['kind'] in hq.KINDS:
        model, audit = hq.restore_checkpoint(path, row, device)
        model.training_control = row['kind']
        return model, audit
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    require(checkpoint['format'] == 'hq-training-controls-v1', 'Unknown checkpoint format')
    for key in ('kind', 'seed', 'best_epoch', 'settings', 'model_config', 'trainability', 'parameter_counts'):
        require(checkpoint[key] == row[key], 'Checkpoint metadata differs: ' + key)
    model = model_from_config(row['model_config'])
    require(state_hash(model.state_dict()) == row['initial_state_sha256']
            and shared_hash(model) == row['initial_shared_state_sha256'], 'Initialization identity differs')
    require(trainability(model) == row['trainability'] and counts(model) == row['parameter_counts'],
            'Training-control semantics/counts differ')
    initial_frozen = frozen_hash(model)
    model.load_state_dict(checkpoint['best_state_dict'], strict=True)
    finite_tensors(model.state_dict().items(), 'restored weights and BatchNorm')
    require(state_hash(model.state_dict()) == row['selected_state_sha256'], 'Selected state differs')
    require(frozen_hash(model) == initial_frozen == row['diagnostics']['frozen_parameters']['initial_sha256'],
            'Restored frozen parameters differ')
    return model.to(device).eval(), dict(checkpoint_sha256=digest(path), strict_restore=True,
        finite_weights_and_bn_buffers=True, selected_state_sha256=row['selected_state_sha256'])


def fit_one(kind, seed, rate, args, data, directory, device):
    import torch
    from tools import training_common as tc
    from utilsTorch.utilsTorch import segmentation_loss
    metrics, checkpoint = directory / 'metrics.json', directory / 'checkpoint.pt'
    if directory.exists():
        require(metrics.is_file() and checkpoint.is_file() and (directory / 'history.json').is_file(),
                'Incomplete attempt preserved at ' + str(directory))
        row = json.loads(metrics.read_text())
        audit_fit(row, kind, seed, rate, args, data)
        require(json.loads((directory / 'history.json').read_text()) == row['history'], 'Raw history differs')
        restored, _ = restore_checkpoint(checkpoint, row)
        del restored
        return row
    directory.mkdir(parents=True)
    tc.seed_everything(seed)
    model = build_model(kind, seed, args).to(device)
    initial_state, common_initial = state_hash(model.state_dict()), shared_hash(model)
    initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    initial_frozen = frozen_hash(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=rate)
    val_loader = tc.make_loader(data['val'], args.batch_size, False, seed)
    probe_images = torch.stack([data['train'][i][0] for i in range(min(8, args.batch_size))]).to(device)
    probes = hq.angle_probe(model, probe_images)
    history, best_score, best_state, best_epoch, best_val = [], -float('inf'), None, None, None
    all_norms = dict(core=[], common=[])
    started = time.perf_counter()
    try:
        for epoch in range(args.epochs + 1):
            loss_total, epoch_norms, schedule = 0.0, dict(core=[], common=[]), None
            if epoch:
                model.train()
                augmented = hq.EpochDataset(data['train'], data['fit'], seed, epoch, args.augment)
                generator = torch.Generator().manual_seed(hq.stable_seed(seed, epoch, 'loader'))
                loader = torch.utils.data.DataLoader(augmented, batch_size=args.batch_size,
                    sampler=augmented.order, num_workers=0, generator=generator)
                for images, masks in loader:
                    images, masks = images.to(device), masks.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = segmentation_loss(model(images), masks, loss_mode='dice_ce')
                    require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                    loss.backward()
                    for group, norm in gradient_norms(model).items():
                        epoch_norms[group].append(norm)
                        all_norms[group].append(norm)
                    optimizer.step()
                    finite_tensors(model.state_dict().items(), 'post-step state including BatchNorm')
                    loss_total += float(loss.detach()) * len(images)
                require(augmented.observed_digest() == augmented.expected_digest, 'Observed data order differs')
                require(frozen_hash(model) == initial_frozen, 'A frozen parameter changed during training')
                schedule = dict(sha256=augmented.observed_digest(), n_samples=len(data['fit']))
            point = dict(epoch=epoch, train_loss=loss_total / len(data['train']) if epoch else None,
                train_schedule=schedule, gradient_norms={g: dict(min=min(v), max=max(v), mean=statistics.mean(v),
                    steps=len(v), nonzero_steps=sum(x > 0 for x in v)) for g, v in epoch_norms.items() if v})
            if epoch == 0 or epoch % args.eval_every == 0 or epoch == args.epochs:
                point['val'] = evaluate(model, val_loader, device)
                hq.validate_sensor_metrics(point['val'], len(data['validation']))
                if tc.is_better_val_score(point['val']['iou'], best_score, 'iou'):
                    best_score, best_epoch, best_val = point['val']['iou'], epoch, point['val']
                    best_state = tc.clone_state_dict_cpu(model.state_dict())
            history.append(point)
            pilot.write_json(directory / 'history.json', history)
            print(json.dumps(dict(id='hq_' + kind, seed=seed, learning_rate=rate, **point)), flush=True)
        model.load_state_dict(best_state, strict=True)
        diagnostics = dict(initial_angle_probes=probes, selected_angle_probes=hq.angle_probe(model, probe_images),
            movement=movement(model, initial), probe_names=[data['dataset'].names[data['fit'][i]] for i in range(len(probe_images))],
            gradient_norms={g: dict(min=min(v), max=max(v), mean=statistics.mean(v), steps=len(v),
                nonzero_steps=sum(x > 0 for x in v)) for g, v in all_norms.items()},
            frozen_parameters=dict(initial_sha256=initial_frozen, selected_sha256=frozen_hash(model),
                                   unchanged=frozen_hash(model) == initial_frozen))
        row = dict(id='hq_' + kind, kind=kind, seed=seed, learning_rate=rate,
            model_config=model_configuration(model), settings=canonical(fit_settings(kind, seed, rate, args, data)),
            parameter_counts=counts(model), trainability=trainability(model),
            total_params=sum(p.numel() for p in model.parameters()),
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            core_params=sum(p.numel() for p in model.bottleneck.core_parameters()),
            initial_state_sha256=initial_state, initial_shared_state_sha256=common_initial,
            selected_state_sha256=state_hash(model.state_dict()), best_epoch=best_epoch, val=best_val,
            history=history, diagnostics=diagnostics, elapsed_s=time.perf_counter() - started,
            fit_indices=data['fit'], val_indices=data['validation'],
            fit_names=[data['dataset'].names[i] for i in data['fit']],
            val_names=[data['dataset'].names[i] for i in data['validation']],
            normalization=data['normalization'], n_global_fit=len(data['all_fit']),
            fit_sensor_counts=dict(Counter(data['dataset'].modalities[i] for i in data['fit'])))
        audit_fit(row, kind, seed, rate, args, data)
        pilot.write_json(metrics, row)
        torch.save(dict(format='hq-training-controls-v1', **{key: row[key] for key in
            ('kind', 'seed', 'best_epoch', 'settings', 'model_config', 'trainability', 'parameter_counts')},
            best_state_dict=best_state), checkpoint)
        return row
    except BaseException as error:
        pilot.write_json(directory / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), completed_epochs=len(history) - 1))
        raise


def expected_grid():
    return {(kind, seed, PRIMARY_RATE) for kind in KINDS for seed in SEEDS} | {
        (kind, 42, 0.001) for kind in KINDS}


def identity(row):
    return row['kind'], row['seed'], row['learning_rate']


def entry_directory(work_dir, legacy_root, entry):
    root = legacy_root if entry['origin'] == 'reused' else work_dir
    path = (Path(root) / entry['directory']).resolve()
    require(not Path(entry['directory']).is_absolute() and path.is_relative_to(Path(root).resolve()),
            'Fit directory escapes its evidence root')
    return path


def paired_audit(rows):
    require(len(rows) == 20 and {identity(row) for row in rows} == expected_grid(), 'Incomplete or duplicated fit grid')
    for seed in SEEDS:
        cohort = [row for row in rows if row['seed'] == seed]
        for key in ('initial_shared_state_sha256', 'fit_names', 'val_names', 'normalization'):
            require(all(row[key] == cohort[0][key] for row in cohort), 'Unpaired fit: ' + key)
        require(all([p['train_schedule'] for p in row['history']] ==
                    [p['train_schedule'] for p in cohort[0]['history']] for row in cohort), 'Unpaired data/augmentation')
        circuits = [r for r in cohort if r['kind'] in ('quantum', 'frozen_quantum', 'zero_bottleneck')]
        require(len({r['initial_state_sha256'] for r in circuits}) == 1, 'Circuit controls start from different states')


def analysis(records):
    require(len(records) == 15 and {identity(row) for row in records} ==
            {(kind, seed, PRIMARY_RATE) for kind in KINDS for seed in SEEDS}, 'Invalid primary replay grid')
    groups = {kind: {r['seed']: r for r in records if r['kind'] == kind} for kind in KINDS}
    for seed in SEEDS:
        cohort = [groups[kind][seed] for kind in KINDS]
        for key in ('fit_names', 'val_names', 'test_names', 'normalization', 'initial_shared_state_sha256'):
            require(all(row[key] == cohort[0][key] for row in cohort), 'Unpaired primary result: ' + key)
    scores = ('iou', 'optical_subset_iou', 'sar_subset_iou')
    stats = lambda values: dict(mean=statistics.mean(values), std=statistics.stdev(values), n=len(values))
    rows = [dict(id='hq_' + kind, kind=kind, total_params=groups[kind][42]['total_params'],
        core_params=groups[kind][42]['core_params'], **{split: {key: stats([groups[kind][s][split][key] for s in SEEDS])
        for key in scores} for split in ('val', 'test')}) for kind in KINDS]
    paired = []
    for other in KINDS[1:]:
        for split in ('val', 'test'):
            for score in scores:
                differences = {str(s): 100 * (groups['quantum'][s][split][score] - groups[other][s][split][score]) for s in SEEDS}
                paired.append(dict(left='hq_quantum', right='hq_' + other, split=split, score=score,
                    differences=differences, **stats(list(differences.values())),
                    wins=sum(value > 0 for value in differences.values()), unit='IoU percentage points', same_learning_rate=True))
    return dict(rows=rows, paired_iou=paired, primary_rate=PRIMARY_RATE,
        test_scope='Previously exposed DFC24 test; descriptive only')


def profile(args, device):
    import torch
    from utilsTorch.utilsTorch import segmentation_loss
    require(not args.work_dir.exists() or not any(args.work_dir.iterdir()), 'Use an empty profile directory')
    result = dict(source_hashes=source_hashes(), protocol_sha256=digest(ROOT / PROTOCOL),
                  runtime=hq.runtime(device), cases=[])
    for kind in NEW_KINDS:
        model = build_model(kind, 42, args).to(device).train()
        initial_frozen = frozen_hash(model)
        initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        generator = torch.Generator().manual_seed(1234)
        images = torch.randn(args.batch_size, 3, 128, 128, generator=generator).to(device)
        masks = torch.randint(0, 2, (args.batch_size, 128, 128), generator=generator).to(device)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=PRIMARY_RATE)
        timings, norms = [], []
        for _ in range(2):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            require(tuple(output.shape) == (args.batch_size, 2, 128, 128), 'Profile geometry differs')
            loss = segmentation_loss(output, masks, loss_mode='dice_ce')
            require(bool(torch.isfinite(loss)), 'Nonfinite profile loss')
            loss.backward()
            norms.append(gradient_norms(model))
            optimizer.step()
            finite_tensors(model.state_dict().items(), 'profile state')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - started)
        require(frozen_hash(model) == initial_frozen, 'Profile changed frozen parameters')
        restored = model_from_config(model_configuration(model)).to(device)
        restored.load_state_dict(model.state_dict(), strict=True)
        require(trainability(restored) == trainability(model), 'Profile trainability did not restore')
        model.eval(); restored.eval()
        with torch.no_grad():
            require(torch.equal(model(images), restored(images)), 'Strict restored predictions differ')
        result['cases'].append(dict(kind=kind, batch_size=args.batch_size, parameter_counts=counts(model),
            timings_s=timings, gradient_norms=norms, movement=movement(model, initial), frozen_unchanged=True,
            strict_restore=True, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None))
        del model, restored, optimizer, images, masks, output, loss
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    require(source_hashes() == result['source_hashes'], 'Source changed during profile')
    result['status'] = 'passed'
    pilot.write_json(args.work_dir / 'resource_profile.json', result)
    return result


def run(args, device):
    import torch
    from tools import training_common as tc
    from scripts.collect_hq_dfc24_direction import audit_campaign
    old = audit_campaign(args.legacy_root, args.data_root)
    require(old['summary']['selection'] == {'hq_quantum': PRIMARY_RATE, 'hq_mlp': PRIMARY_RATE},
            'Historical baseline rates differ from this fixed-rate protocol')
    resource = json.loads(args.profile.read_text())
    require(resource['status'] == 'passed' and resource['source_hashes'] == source_hashes()
            and resource['protocol_sha256'] == digest(ROOT / PROTOCOL), 'Missing/stale actual-model profile')
    require({case['kind'] for case in resource['cases']} == set(NEW_KINDS)
            and all(case['batch_size'] == 8 and case['strict_restore'] and case['frozen_unchanged']
                    for case in resource['cases']), 'Incomplete resource profile')
    require(resource['runtime'] == hq.runtime(device), 'Resource-profile runtime differs')
    data_by_seed = {seed: pilot.load_training_data(args, seed) for seed in SEEDS}
    require(all(len(d['all_fit']) == 1303 and len(d['validation']) == 230 and len(d['fit']) == 128
                and d['normalization']['fit_only'] is True for d in data_by_seed.values()), 'Training cohorts differ')
    inputs = pilot.verify_inputs(args.data_root)
    require(inputs == old['plan']['input_identity'], 'New and reused input bytes differ')
    settings = {key: value for key, value in vars(args).items() if key in
        ('device', 'batch_size', 'epochs', 'fit_cap', 'image_size', 'base_channels', 'eval_every', 'augment')}
    candidate = canonical(dict(settings=settings, controls=KINDS, seeds=SEEDS, learning_rates=RATES,
        primary_rate=PRIMARY_RATE, source_hashes=source_hashes(),
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_identity=inputs, runtime=hq.runtime(device), protocol_sha256=digest(ROOT / PROTOCOL),
        resource_profile_sha256=digest(args.profile), legacy_root=str(args.legacy_root.resolve()),
        legacy_plan_sha256=digest(args.legacy_root / 'plan.json'), legacy_source_hashes=old['plan']['source_hashes'],
        legacy_raw_files=old['raw_files'], primary_policy='Fixed 0.003 for all controls/seeds; no rate reselection',
        test_policy='All 20 fits audited before 15 new primary replays; exposed test descriptive',
        auxiliary_policy='Rate0.001 on seed42 for all five controls; sensitivity only'))
    plan = hq.frozen_plan(args.work_dir / 'plan.json', candidate, args.resume)
    saved_profile = args.work_dir / 'resource_profile.json'
    if saved_profile.exists():
        require(digest(saved_profile) == plan['resource_profile_sha256'], 'Retained profile differs')
    else:
        saved_profile.write_bytes(args.profile.read_bytes())
    entries = []
    for entry in old['fits']:
        row, directory = entry['record'], args.legacy_root / entry['directory']
        entries.append(dict(origin='reused', directory=entry['directory'], record=row,
            **{name.replace('.json', '').replace('.pt', '') + '_sha256': digest(directory / name)
               for name in ('metrics.json', 'history.json', 'checkpoint.pt')}))
    for kind in NEW_KINDS:
        grid = [(42, rate) for rate in RATES] + [(seed, PRIMARY_RATE) for seed in SEEDS[1:]]
        for seed, rate in grid:
            require(source_hashes() == plan['source_hashes'], 'Source changed during fitting')
            directory = args.work_dir / 'fits' / ('hq_' + kind) / str(rate) / ('seed' + str(seed))
            row = fit_one(kind, seed, rate, args, data_by_seed[seed], directory, device)
            entries.append(dict(origin='fresh', directory=str(directory.relative_to(args.work_dir)), record=row,
                **{name.replace('.json', '').replace('.pt', '') + '_sha256': digest(directory / name)
                   for name in ('metrics.json', 'history.json', 'checkpoint.pt')}))
            pilot.write_json(args.work_dir / 'fit_progress.json', entries)
    paired_audit([entry['record'] for entry in entries])
    require(source_hashes() == plan['source_hashes'] and pilot.verify_inputs(args.data_root) == inputs,
            'Source/input bytes changed before primary replay')
    audits = []
    for entry in entries:
        directory = entry_directory(args.work_dir, args.legacy_root, entry)
        model, audit = restore_checkpoint(directory / 'checkpoint.pt', entry['record'])
        audits.append(dict(id=entry['record']['id'], seed=entry['record']['seed'],
            learning_rate=entry['record']['learning_rate'], origin=entry['origin'], **audit))
        del model
    pilot.write_json(args.work_dir / 'all_fits.json', entries)
    pilot.write_json(args.work_dir / 'checkpoint_audit.json', audits)
    pilot.write_json(args.work_dir / 'fit_boundary.json', dict(completed=True, fits=20, fresh=12, reused=8,
        primary=15, all_fits_sha256=digest(args.work_dir / 'all_fits.json'),
        checkpoint_audit_sha256=digest(args.work_dir / 'checkpoint_audit.json')))
    records = []
    for entry in entries:
        if entry['record']['learning_rate'] != PRIMARY_RATE:
            continue
        row = copy.deepcopy(entry['record'])
        directory = entry_directory(args.work_dir, args.legacy_root, entry)
        model, _ = restore_checkpoint(directory / 'checkpoint.pt', row, device)
        cfg = dict(row['settings'], augment=False)
        _, _, _, images, masks = tc.resolve_data_dirs('combined', str(args.data_root))
        dataset = tc.load_dataset(images, masks, cfg)
        require(dataset.norm_provenance == row['normalization'], 'Test normalization differs')
        loader = tc.make_loader(dataset, args.batch_size, False, row['seed'])
        row['test'] = evaluate(model, loader, device)
        row['test_names'] = list(dataset.names)
        hq.validate_sensor_metrics(row['test'], 205, test=True)
        row['checkpoint'] = dict(origin=entry['origin'], path=str(Path(entry['directory']) / 'checkpoint.pt'))
        replay = args.work_dir / 'replays' / row['id'] / ('seed' + str(row['seed']) + '.json')
        pilot.write_json(replay, row)
        records.append(row)
        del model
    require(source_hashes() == plan['source_hashes'] and pilot.verify_inputs(args.data_root) == inputs,
            'Source/inputs changed during replays')
    summary = dict(plan=plan, records=records, analysis=analysis(records))
    pilot.write_json(args.work_dir / 'summary.json', summary)
    print('HQ TRAINING CONTROLS COMPLETE', str(args.work_dir / 'summary.json'), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('profile', 'run'))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--legacy-root', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, choices=(8,), default=8)
    parser.add_argument('--epochs', type=int, choices=(60,), default=60)
    parser.add_argument('--fit-cap', type=int, choices=(128,), default=128)
    parser.add_argument('--image-size', type=int, choices=(128,), default=128)
    parser.add_argument('--base-channels', type=int, choices=(8,), default=8)
    parser.add_argument('--eval-every', type=int, choices=(2,), default=2)
    parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not args.augment:
        parser.error('The frozen control protocol requires train-only D4 augmentation')
    if args.phase == 'run' and any(getattr(args, field) is None for field in ('data_root', 'legacy_root', 'profile')):
        parser.error('run requires --data-root, --legacy-root and --profile')
    if args.phase == 'profile' and args.resume:
        parser.error('Profiles do not resume')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable; no silent CPU fallback')
        torch.cuda.set_per_process_memory_fraction(0.12, device)
    return profile(args, device) if args.phase == 'profile' else run(args, device)


if __name__ == '__main__':
    main()
