#!/usr/bin/env python3
"""A paired, bounded real-data pilot; validation chooses rates before test replay."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_inputs(data_root):
    """Bind the experiment to actual kept image/mask bytes, before any training."""
    manifest_path = data_root / 'splits/flood_combined.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('conflict_policy') != 'drop':
        raise ValueError('The pilot requires conflict-free drop-policy manifests')
    inventory = []
    for split, records in manifest['hashes'].items():
        kept = set(manifest[split])
        directory = 'Training' if split == 'train' else 'Testing'
        for record in records:
            if record['name'] not in kept:
                continue
            for kind, field in [('images', 'image'), ('labels', 'mask')]:
                relative = Path('flood_combined') / directory / kind / record[field + '_file']
                path = data_root / relative
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                if digest != record[field + '_md5']:
                    raise ValueError('Raw file differs from the split manifest: ' + str(relative))
                inventory.append(str(relative) + ':' + digest)
    return dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                verified_files=len(inventory),
                stats_files_sha256={str(path.relative_to(data_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (data_root / 'splits/norm_stats.json', data_root / 'norm_stats.json') if path.is_file()},
                raw_inventory_sha256=hashlib.sha256('\n'.join(sorted(inventory)).encode()).hexdigest())


def plan_specs():
    from scripts.run_parameter_budget_comparison import base_config, make_plan
    width = make_plan()['selected_width']
    specs = [dict(id=f'unet_width{width}', model_type='classical', config=base_config(width, 1)),
             dict(id='unet_width8', model_type='classical', config=base_config(8, 1))]
    for kind, label in [('ablation', 'mlp'), ('quantum', 'quantum')]:
        for mode in ('global', 'patch4x4'):
            specs.append(dict(id=label + '_' + mode, model_type=kind,
                              config=dict(base_config(8, 1), q_spatial_mode=mode)))
    for spec in specs:
        spec['config']['dataset'] = 'combined'
    return specs


def choose_rates(records):
    """Only validation IoU participates; equal scores prefer the lower rate."""
    chosen = {}
    for record in records:
        previous = chosen.get(record['id'])
        if previous is None or (record['val']['iou'], -record['learning_rate']) > (
                previous['val']['iou'], -previous['learning_rate']):
            chosen[record['id']] = record
    return chosen


def selected_fit_indices(train_indices, cap, seed):
    import numpy as np
    from tools import training_common as tc
    values = list(map(int, train_indices))
    if cap and cap < len(values):
        rng = np.random.default_rng(tc.derive_seed(seed, 913))
        values = sorted(map(int, rng.choice(values, size=cap, replace=False)))
    return values


def parameter_groups(model):
    from utilsTorch.unetTorch import QuantumBlock, ClassicalBottleneckBlock
    blocks = [module for module in model.modules() if isinstance(module, (QuantumBlock, ClassicalBottleneckBlock))]
    bottleneck = {id(p) for module in blocks for p in module.parameters()}
    groups = dict(backbone=[], bottleneck=[], circuit_angles=[], input_scales=[])
    for name, parameter in model.named_parameters():
        groups['bottleneck' if id(parameter) in bottleneck else 'backbone'].append((name, parameter))
        if id(parameter) in bottleneck and '.quantum_params.' in name:
            groups['circuit_angles'].append((name, parameter))
        if id(parameter) in bottleneck and '.input_scales.' in name:
            groups['input_scales'].append((name, parameter))
    return groups, blocks


def output_probe(model, blocks, images):
    import torch
    rows = []
    def capture(module, inputs, output):
        output = output.detach().float()
        rows.append(dict(mean=float(output.mean()), std=float(output.std(unbiased=False)),
                         between_sample_std=float(output.std(dim=0, unbiased=False).mean()),
                         spatial_std=float(output.flatten(2).std(dim=2, unbiased=False).mean())))
    handles = [block.register_forward_hook(capture) for block in blocks]
    try:
        model.eval()
        with torch.no_grad():
            model(images)
    finally:
        for handle in handles:
            handle.remove()
    return rows


def load_training_data(args, seed):
    import torch
    import flood_data
    from tools import training_common as tc
    cfg = dict(plan_specs()[0]['config'], data_root=str(args.data_root), dataset='combined',
               image_size=args.image_size, seed=seed, max_samples=None)
    _, train_images, train_labels, _, _ = tc.resolve_data_dirs('combined', str(args.data_root))
    dataset = tc.load_dataset(train_images, train_labels, cfg)
    all_fit, validation, _ = tc.holdout_validation(dataset, seed, cfg['val_fraction'])
    provenance = dataset.norm_provenance
    digest = flood_data._fit_names_digest([dataset.names[int(i)] for i in all_fit])
    if not provenance or provenance.get('fit_names_sha256') != digest or not provenance.get('fit_only'):
        raise ValueError('Normalization must exclude exactly the global validation partition')
    fit = selected_fit_indices(all_fit, args.fit_cap, seed)
    if set(fit) & set(validation):
        raise ValueError('Optimization and validation indices overlap')
    return dict(dataset=dataset, fit=fit, all_fit=list(map(int, all_fit)), validation=list(map(int, validation)),
                train=torch.utils.data.Subset(dataset, fit), val=torch.utils.data.Subset(dataset, validation),
                normalization=provenance)


def fit_one(spec, seed, rate, args, data, directory, device):
    import torch
    from tools import training_common as tc
    from utilsTorch.utilsTorch import UNetSegmentation, QVUNetSegmentation, ClassicalAblationUNetSegmentation, evaluate_model, segmentation_loss
    metrics_path = directory / 'metrics.json'
    checkpoint = directory / 'checkpoint.pt'
    if metrics_path.is_file() and checkpoint.is_file():
        return json.loads(metrics_path.read_text())
    if directory.exists():
        destination = directory.with_name(directory.name + '-incomplete-' + str(time.time_ns()))
        directory.rename(destination)
    directory.mkdir(parents=True)
    cfg = dict(spec['config'], dataset='combined', data_root=str(args.data_root), seed=seed,
        image_size=args.image_size, max_samples=None, batch_size=args.batch_size,
        num_epochs=args.epochs, learning_rate=rate, optimization_fit_cap=args.fit_cap,
        _resolved_norm_provenance=dict(data['normalization']))
    tc.seed_everything(seed)
    classes = dict(classical=UNetSegmentation, quantum=QVUNetSegmentation, ablation=ClassicalAblationUNetSegmentation)
    model = tc.build_model(classes[spec['model_type']], cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    groups, blocks = parameter_groups(model)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    initial_hash = hashlib.sha256(b''.join(name.encode() + value.numpy().tobytes() for name, value in initial.items())).hexdigest()
    gradient_values = {name: [] for name, parameters in groups.items() if parameters}
    train = tc.make_loader(data['train'], args.batch_size, True, seed)
    val = tc.make_loader(data['val'], args.batch_size, False, seed)
    probe_images = torch.stack([data['train'][i][0] for i in range(min(8, len(data['train'])))]).to(device)
    initial_output = output_probe(model, blocks, probe_images)
    started = time.perf_counter()
    history, best_score, best_state, best_epoch, best_val = [], -float('inf'), None, None, None
    for epoch in range(args.epochs + 1):
        train_loss = None
        if epoch:
            model.train()
            total = 0.0
            for images, masks in train:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                loss = segmentation_loss(model(images), masks, loss_mode='dice_ce')
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                for name, parameters in groups.items():
                    squares = [p.grad.detach().float().square().sum() for _, p in parameters if p.grad is not None]
                    if squares:
                        norm = float(torch.stack(squares).sum().sqrt())
                        if not math.isfinite(norm):
                            raise FloatingPointError('Nonfinite gradient: ' + name)
                        gradient_values[name].append(norm)
                optimizer.step()
                total += float(loss.detach()) * len(images)
            train_loss = total / len(data['train'])
        row = dict(epoch=epoch, train_loss=train_loss)
        if epoch == 0 or epoch % args.eval_every == 0 or epoch == args.epochs:
            row['val'] = evaluate_model(model, val, device)
            if tc.is_better_val_score(row['val']['iou'], best_score, 'iou'):
                best_score, best_epoch, best_val = row['val']['iou'], epoch, row['val']
                best_state = tc.clone_state_dict_cpu(model.state_dict())
        history.append(row)
        write_json(directory / 'history.json', history)
        print(json.dumps(dict(id=spec['id'], seed=seed, learning_rate=rate, **row)), flush=True)
    model.load_state_dict(best_state)
    movement = {}
    for name, parameters in groups.items():
        if parameters:
            squared = sum(float((p.detach().cpu() - initial[key]).square().sum()) for key, p in parameters)
            initial_squared = sum(float(initial[key].square().sum()) for key, _ in parameters)
            movement[name] = dict(l2=math.sqrt(squared), relative_l2=math.sqrt(squared / initial_squared) if initial_squared else None)
    diagnostics = dict(initial_bottleneck_output=initial_output,
        selected_bottleneck_output=output_probe(model, blocks, probe_images), movement=movement,
        gradient_norms={name: dict(mean=statistics.mean(values), min=min(values), max=max(values),
                                  nonzero_steps=sum(value > 0 for value in values), steps=len(values))
                        for name, values in gradient_values.items() if values})
    record = dict(id=spec['id'], model_type=spec['model_type'], seed=seed, learning_rate=rate,
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad), settings=cfg,
        fit_indices=data['fit'], val_indices=data['validation'],
        fit_names=[data['dataset'].names[i] for i in data['fit']],
        val_names=[data['dataset'].names[i] for i in data['validation']],
        fit_sensor_counts=dict(Counter(data['dataset'].modalities[i] for i in data['fit'])),
        n_global_fit=len(data['all_fit']), normalization=data['normalization'],
        initial_parameter_sha256=initial_hash, best_epoch=best_epoch, val=best_val,
        history=history, diagnostics=diagnostics, elapsed_s=time.perf_counter() - started)
    tc.atomic_torch_save(dict(best_state_dict=best_state, settings=cfg, model_type=spec['model_type'],
                             best_epoch=best_epoch), str(checkpoint))
    write_json(metrics_path, record)
    return record


def aggregate(records):
    def stats(values):
        return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else 0.0, n=len(values))
    by_id = {}
    for record in records:
        group = by_id.setdefault(record['id'], {})
        if record['seed'] in group:
            raise ValueError('Duplicate selected model/seed result')
        group[record['seed']] = record
    seed_sets = [set(group) for group in by_id.values()]
    if not seed_sets or any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError('Selected models require identical seed coverage')
    for seed in seed_sets[0]:
        cohort = [group[seed] for group in by_id.values()]
        for key in ('fit_names', 'val_names', 'test_names', 'normalization'):
            if any(row[key] != cohort[0][key] for row in cohort):
                raise ValueError('Unpaired comparison: ' + key)
    rows = [dict(id=name, total_params=next(iter(group.values()))['total_params'],
                 **{metric: stats([row['test'][metric] for row in group.values()])
                    for metric in ('iou', 'optical_subset_iou', 'sar_subset_iou')})
            for name, group in by_id.items()]
    contrasts = []
    for left, right in [('quantum_patch4x4', 'quantum_global'), ('mlp_patch4x4', 'mlp_global'),
                        ('quantum_global', 'mlp_global'), ('quantum_patch4x4', 'mlp_patch4x4'),
                        ('quantum_patch4x4', 'unet_width7'), ('quantum_patch4x4', 'unet_width8')]:
        if left in by_id and right in by_id:
            differences = {str(seed): by_id[left][seed]['test']['iou'] - by_id[right][seed]['test']['iou'] for seed in sorted(seed_sets[0])}
            contrasts.append(dict(left=left, right=right, differences=differences, **stats(list(differences.values()))))
    return dict(rows=rows, paired_iou=contrasts)


def format_report(summary):
    settings = summary['plan']['settings']
    lines = ['# Real DFC24 spatial pilot', '',
        f"Paired seeds: {settings['seeds']}. {settings['image_size']}-pixel inputs; "
        f"at most {settings['fit_cap']} optimization tiles per seed; {settings['epochs']} epochs.",
        'Normalization uses all global fit tiles with validation excluded; every model shares the same selected optimization and validation indices.',
        'The first seed (42) selects one of the same learning-rate candidates per model using validation IoU. Replication uses the chosen rate. Test replay starts after every fit is complete.', '',
        '| Model | Parameters | Selected LR | Test IoU mean ± sample SD | Optical IoU | SAR IoU |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['analysis']['rows']:
        pooled, optical, sar = (row[key] for key in ['iou', 'optical_subset_iou', 'sar_subset_iou'])
        lines.append(f"| {row['id']} | {row['total_params']:,} | {summary['selection'][row['id']]} | "
                     f"{pooled['mean']:.4f} ± {pooled['std']:.4f} | {optical['mean']:.4f} | {sar['mean']:.4f} |")
    lines += ['', '| Paired contrast | IoU percentage points, mean ± sample SD |', '| --- | ---: |']
    for contrast in summary['analysis']['paired_iou']:
        lines.append(f"| {contrast['left']} − {contrast['right']} | {100 * contrast['mean']:+.3f} ± {100 * contrast['std']:.3f} |")
    lines += ['', 'Per-seed histories, validation selections, best epochs, gradient norms, parameter movement, output variation and zero/random knockout results are retained in the JSON records.',
        'This is a centralized pilot with capped optimization and full-fit normalization, not a pure few-image learning experiment or an Anvil/federated result. Seed variation, incomplete convergence and small tuning budgets limit conclusions; this does not establish quantum computational advantage.', '']
    late = [row for row in summary['records'] if row['best_epoch'] >= settings['epochs'] - settings['eval_every']]
    if late:
        lines += [f"Validation selected a checkpoint in the final evaluation window for {len(late)} of {len(summary['records'])} runs; those curves warrant longer training before a final ranking.", '']
    return '\n'.join(lines)


def main(argv=None, *, spec_factory=None, extra_source_files=(), experiment_context=None,
         expected_input_identity=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 77, 123])
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[0.001, 0.003])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval-every', type=int, default=2)
    parser.add_argument('--fit-cap', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--specs', nargs='+', help='Optional smoke subset of predefined model ids')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or args.seeds[0] != 42:
        parser.error('Use unique paired seeds with the predefined tuning seed 42 first')
    if min(args.epochs, args.eval_every, args.fit_cap, args.batch_size) < 1 or args.image_size < 32 or args.image_size % 8:
        parser.error('Use positive budgets and image size >=32 divisible by eight')
    if len(set(args.learning_rates)) != len(args.learning_rates) or any(rate <= 0 or not math.isfinite(rate) for rate in args.learning_rates):
        parser.error('Learning rates must be finite, positive and unique')
    import torch
    from tools import training_common as tc
    from tools.checkpoint_evaluation import load_checkpoint_model
    from utilsTorch.utilsTorch import evaluate_model
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.12)
    device = torch.device(args.device)
    specs = (spec_factory or plan_specs)()
    if args.specs:
        if not set(args.specs) <= {spec['id'] for spec in specs}:
            parser.error('Unknown model id')
        specs = [spec for spec in specs if spec['id'] in args.specs]
    source_files = ['scripts/run_real_spatial_pilot.py', 'scripts/run_parameter_budget_comparison.py',
                    'utilsTorch/unetTorch.py', 'utilsTorch/utilsTorch.py', 'tools/training_common.py',
                    'tools/checkpoint_evaluation.py', 'flood_data.py', 'splits/norm_stats.json']
    source_files += list(extra_source_files)
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_files}
    input_identity = verify_inputs(args.data_root)
    if expected_input_identity is not None and input_identity != expected_input_identity:
        parser.error('Input data differ from the frozen baseline comparison')
    settings = {key: value for key, value in vars(args).items() if key not in ('resume', 'work_dir', 'data_root')}
    plan = dict(settings=settings, specs=specs, source_hashes=hashes,
        experiment_context=experiment_context,
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in ['torch', 'pennylane', 'numpy', 'rasterio']},
        input_identity=input_identity,
        normalization_policy='All global fit images; validation excluded; optimization capped separately',
        test_policy='No trained-model test evaluation until all validation tuning and replication fits finish',
        device=str(device), gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
        gpu_allocator_fraction=0.12 if args.device == 'cuda' else None, threads=2)
    plan = json.loads(json.dumps(plan))
    plan_path = args.work_dir / 'plan.json'
    if args.resume:
        previous = json.loads(plan_path.read_text())
        for key in ('settings', 'specs', 'source_hashes', 'input_identity', 'packages', 'device', 'experiment_context'):
            if previous[key] != plan[key]:
                parser.error('Resume inputs differ from the frozen plan: ' + key)
    else:
        if args.work_dir.exists() and any(args.work_dir.iterdir()):
            parser.error('Use a new/empty work directory, or --resume with identical inputs')
        write_json(plan_path, plan)
    data_by_seed = {seed: load_training_data(args, seed) for seed in args.seeds}
    tuning = []
    paths = {}
    for spec in specs:
        for rate in args.learning_rates:
            path = args.work_dir / 'tuning' / spec['id'] / str(rate) / 'seed42'
            record = fit_one(spec, 42, rate, args, data_by_seed[42], path, device)
            tuning.append(record)
            paths[(spec['id'], 42, rate)] = path
    chosen = choose_rates(tuning)
    for model_id in chosen:
        if len({row['initial_parameter_sha256'] for row in tuning if row['id'] == model_id}) != 1:
            raise ValueError('Learning-rate candidates must share identical initial parameters')
    write_json(args.work_dir / 'tuning_candidates.json', tuning)
    write_json(args.work_dir / 'validation_selection.json', {name: dict(learning_rate=row['learning_rate'],
        val_iou=row['val']['iou'], best_epoch=row['best_epoch']) for name, row in chosen.items()})
    selected = []
    for spec in specs:
        rate = chosen[spec['id']]['learning_rate']
        selected.append(chosen[spec['id']])
        for seed in args.seeds[1:]:
            path = args.work_dir / 'replication' / spec['id'] / ('seed' + str(seed))
            selected.append(fit_one(spec, seed, rate, args, data_by_seed[seed], path, device))
            paths[(spec['id'], seed, rate)] = path
    # This boundary is deliberately after every fit and validation-rate decision.
    for seed in args.seeds:
        for record in [row for row in selected if row['seed'] == seed]:
            path = paths[(record['id'], seed, record['learning_rate'])]
            model, cfg, kind = load_checkpoint_model(path / 'checkpoint.pt', data_root=args.data_root, device=device)
            _, _, _, images, masks = tc.resolve_data_dirs('combined', str(args.data_root))
            dataset = tc.load_dataset(images, masks, cfg)
            if dataset.norm_provenance != record['normalization']:
                raise ValueError('Test normalization must match the selected training checkpoint')
            loader = tc.make_loader(dataset, args.batch_size, False, seed)
            record['test'] = evaluate_model(model, loader, device)
            record['test_names'] = list(dataset.names)
            record['checkpoint'] = str((path / 'checkpoint.pt').relative_to(args.work_dir))
            record['knockout'] = tc.knockout_eval(model, loader, device, evaluate_model, knockout_seed=seed) if kind != 'classical' else None
            write_json(path / 'selected_test.json', record)
    summary = dict(plan=plan, selection={name: row['learning_rate'] for name, row in chosen.items()},
                   records=selected, analysis=aggregate(selected))
    write_json(args.work_dir / 'summary.json', summary)
    (args.work_dir / 'report.md').write_text(format_report(summary))
    print('PILOT COMPLETE', str(args.work_dir / 'summary.json'), flush=True)
    return summary


if __name__ == '__main__':
    main()
