#!/usr/bin/env python3
"""Bounded HQ-style quantum/MLP direction screen on cleaned DFC24 data."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_real_spatial_pilot as pilot

SEEDS = (42, 77, 123)
RATES = (0.001, 0.003)
KINDS = ('quantum', 'mlp')
MODES = ('zero_core', 'random_core', 'zero_bottleneck')
SOURCE_FILES = (
    'scripts/run_hq_dfc24_direction.py', 'utilsTorch/hq_direction.py',
    'scripts/run_real_spatial_pilot.py', 'scripts/run_parameter_budget_comparison.py',
    'utilsTorch/__init__.py', 'utilsTorch/unetTorch.py', 'utilsTorch/utilsTorch.py', 'tools/training_common.py',
    'flood_data.py', 'splits/norm_stats.json',
)


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_record(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_record(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite_record(child)
    elif isinstance(value, (int, float)):
        require(math.isfinite(value), 'Nonfinite recorded number')


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def runtime(device):
    import torch
    return dict(python=platform.python_version(), executable=sys.executable,
        packages={name: importlib.metadata.version(name) for name in ('torch', 'pennylane', 'numpy', 'rasterio', 'scikit-learn', 'scipy', 'Pillow')},
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        torch_cuda=torch.version.cuda, gpu_allocator_fraction=0.12 if device.type == 'cuda' else None,
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:4], 'little')


def epoch_schedule(indices, seed, epoch, augment):
    """Independent data RNG; model construction/dropout cannot advance it."""
    order = list(range(len(indices)))
    random.Random(stable_seed(seed, epoch, 'order')).shuffle(order)
    codes = {index: stable_seed(seed, epoch, indices[index], 'D4') % 8 if augment else 0
             for index in order}
    seen = [(int(indices[index]), codes[index]) for index in order]
    payload = json.dumps(seen, separators=(',', ':')).encode()
    return order, codes, hashlib.sha256(payload).hexdigest()


class EpochDataset:
    """Apply a fixed D4 transform to each original index for this epoch only."""
    def __init__(self, dataset, indices, seed, epoch, augment):
        self.dataset = dataset
        self.indices = indices
        self.order, self.codes, self.expected_digest = epoch_schedule(indices, seed, epoch, augment)
        self.seen = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        from tools import training_common as tc
        images, masks = self.dataset[index]
        code = self.codes[index]
        self.seen.append((int(self.indices[index]), code))
        return tc.apply_dihedral_transform(images, masks, code)

    def observed_digest(self):
        return hashlib.sha256(json.dumps(self.seen, separators=(',', ':')).encode()).hexdigest()


def build_model(kind, seed, args):
    from utilsTorch.hq_direction import HQDirectionUNet
    return HQDirectionUNet(kind=kind, seed=seed, base_channels=args.base_channels,
                          image_size=args.image_size, in_channels=3, out_channels=2)


def model_configuration(model):
    from utilsTorch.hq_direction import model_config
    return model_config(model)


def model_from_config(config):
    from utilsTorch.hq_direction import build_from_config
    return build_from_config(config)


def shared_hash(model):
    from utilsTorch.hq_direction import shared_state_sha256
    return shared_state_sha256(model)


def state_hash(state):
    return hashlib.sha256(b''.join(name.encode() + str(value.dtype).encode() +
        str(tuple(value.shape)).encode() + value.detach().cpu().numpy().tobytes()
        for name, value in state.items())).hexdigest()


def finite_tensors(named, label):
    import torch
    groups = {}
    for name, value in named:
        groups.setdefault(value.device, []).append((name, torch.isfinite(value).all()))
    for values in groups.values():
        if not bool(torch.stack([flag for _, flag in values]).all()):
            bad = [name for name, flag in values if not bool(flag)]
            raise FloatingPointError('Nonfinite ' + label + ': ' + ', '.join(bad))


def parameter_groups(model):
    core = {id(parameter) for parameter in model.bottleneck.core_parameters()}
    return {group: [(name, parameter) for name, parameter in model.named_parameters()
                    if (id(parameter) in core) == (group == 'core')]
            for group in ('core', 'common')}


def gradient_norms(model):
    result = {}
    for group, parameters in parameter_groups(model).items():
        gradients = [(name, parameter.grad) for name, parameter in parameters if parameter.grad is not None]
        if group == 'core':
            require(len(gradients) == len(parameters) and bool(parameters), 'A trainable core tensor has no gradient')
        finite_tensors(gradients, 'gradient')
        result[group] = math.sqrt(sum(float(value.detach().double().square().sum()) for _, value in gradients))
        require(math.isfinite(result[group]), 'Nonfinite accumulated gradient norm')
    return result


def movement(model, initial):
    result = {}
    for group, parameters in parameter_groups(model).items():
        squared = sum(float((parameter.detach().cpu().double() - initial[name].double()).square().sum())
                      for name, parameter in parameters)
        baseline = sum(float(initial[name].double().square().sum()) for name, _ in parameters)
        result[group] = dict(l2=math.sqrt(squared), relative_l2=math.sqrt(squared / baseline) if baseline else None)
    return result


def angle_probe(model, images):
    import torch
    result = []
    before = state_hash(model.state_dict())
    was_training = model.training
    handle = model.bottleneck.register_forward_pre_hook(
        lambda module, inputs: result.append(module.angle_diagnostics(inputs[0].detach())))
    try:
        model.eval()
        with torch.no_grad():
            output = model(images)
        finite_tensors([('probe_output', output)], 'probe output')
    finally:
        handle.remove()
        model.train(was_training)
    require(state_hash(model.state_dict()) == before, 'An evaluation angle probe mutated model/BatchNorm state')
    finite_record(result)
    return result


def evaluate(model, loader, device):
    from utilsTorch.utilsTorch import evaluate_model
    result = evaluate_model(model, loader, device, loss_mode='dice_ce')
    finite_record(result)
    for key in ('iou', 'dice', 'accuracy', 'auc'):
        require(0 <= result[key] <= 1, 'Invalid evaluation metric: ' + key)
    return result


def validate_sensor_metrics(metrics, n_samples, *, test=False):
    finite_record(metrics)
    for sensor in ('optical', 'sar'):
        for key in ('tp', 'fp', 'fn', 'tn', 'n_samples'):
            value = metrics[sensor + '_subset_' + key]
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    'Invalid sensor confusion count')
    totals = {key: sum(metrics[sensor + '_subset_' + key] for sensor in ('optical', 'sar'))
              for key in ('tp', 'fp', 'fn', 'tn', 'n_samples')}
    require(totals['n_samples'] == n_samples, 'Evaluation sensor sample coverage differs')
    require(sum(totals[key] for key in ('tp', 'fp', 'fn', 'tn')) == n_samples * 128 * 128,
            'Evaluation pixel coverage differs from the fixed128-pixel images')
    union = totals['tp'] + totals['fp'] + totals['fn']
    require(math.isclose(metrics['iou'], totals['tp'] / union if union else 0, abs_tol=1e-12),
            'Sensor confusion counts do not recombine to pooled IoU')
    if test:
        require((metrics['optical_subset_n_samples'], metrics['sar_subset_n_samples']) == (71, 134),
                'Expected the full 71 Optical/134 SAR descriptive test cohort')


def knockout_evaluation(model, loader, device, seed):
    results = {}
    try:
        for mode in MODES:
            model.bottleneck.set_mode(mode, seed=seed)
            results[mode] = evaluate(model, loader, device)
            validate_sensor_metrics(results[mode], 205, test=True)
    finally:
        model.bottleneck.set_mode('on', seed=seed)
    return results


def fit_settings(kind, seed, rate, args, data):
    return dict(kind=kind, seed=seed, learning_rate=rate, epochs=args.epochs,
        batch_size=args.batch_size, eval_every=args.eval_every, fit_cap=args.fit_cap,
        image_size=args.image_size, base_channels=args.base_channels, augment=args.augment,
        loss='dice_ce', dataset='combined', data_root=str(args.data_root.resolve()),
        max_samples=None, fit_only_norm=True, val_fraction=0.15, legacy_data=False,
        combined_input_mode='shared3', _resolved_norm_provenance=data['normalization'])


def audit_fit(row, kind, seed, rate, args, data):
    finite_record(row)
    require((row['id'], row['kind'], row['seed'], row['learning_rate']) == ('hq_' + kind, kind, seed, rate), 'Cached fit identity differs')
    require(canonical(row['model_config']) == dict(kind=kind, seed=seed, base_channels=args.base_channels,
        image_size=args.image_size, in_channels=3, out_channels=2), 'Cached model architecture differs')
    require(canonical(row['settings']) == canonical(fit_settings(kind, seed, rate, args, data)), 'Cached fit settings differ')
    for key, value in dict(fit_indices=data['fit'], val_indices=data['validation'],
        fit_names=[data['dataset'].names[index] for index in data['fit']],
        val_names=[data['dataset'].names[index] for index in data['validation']],
        normalization=data['normalization'], n_global_fit=len(data['all_fit'])).items():
        require(row[key] == value, 'Cached fit cohort/normalization differs: ' + key)
    require([point['epoch'] for point in row['history']] == list(range(args.epochs + 1)), 'Incomplete fit history')
    require({point['epoch'] for point in row['history'] if 'val' in point} ==
            set(range(0, args.epochs + 1, args.eval_every)) | {args.epochs}, 'Cached validation schedule differs')
    for point in row['history']:
        require((point['train_loss'] is None) == (point['epoch'] == 0), 'Cached training loss is missing')
        if point['epoch']:
            require(point['train_loss'] >= 0, 'Cached training loss is negative')
            _, _, expected = epoch_schedule(data['fit'], seed, point['epoch'], args.augment)
            require(point['train_schedule'] == dict(sha256=expected, n_samples=len(data['fit'])),
                    'Cached actual data order/augmentation differs')
        if 'val' in point:
            validate_sensor_metrics(point['val'], len(data['validation']))
    best = max((point for point in row['history'] if 'val' in point), key=lambda point: point['val']['iou'])
    require(row['best_epoch'] == best['epoch'] and row['val'] == best['val'], 'Invalid validation checkpoint selection')


def restore_checkpoint(path, row, device='cpu'):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    require(checkpoint['format'] == 'hq-direction-v1' and checkpoint['kind'] == row['kind']
            and checkpoint['seed'] == row['seed'] and checkpoint['best_epoch'] == row['best_epoch'],
            'Checkpoint identity/selected epoch differs')
    require(canonical(checkpoint['settings']) == canonical(row['settings'])
            and canonical(checkpoint['model_config']) == canonical(row['model_config']), 'Checkpoint settings differ')
    finite_tensors(checkpoint['best_state_dict'].items(), 'checkpoint state including BatchNorm buffers')
    model = model_from_config(checkpoint['model_config'])
    require(state_hash(model.state_dict()) == row['initial_state_sha256']
            and shared_hash(model) == row['initial_shared_state_sha256'], 'Checkpoint initialization identity differs')
    require(sum(p.numel() for p in model.parameters()) == row['total_params'], 'Checkpoint parameter count differs')
    require(sum(p.numel() for p in model.bottleneck.core_parameters()) == row['core_params'], 'Checkpoint core count differs')
    model.load_state_dict(checkpoint['best_state_dict'], strict=True)
    require(state_hash(model.state_dict()) == row['selected_state_sha256'], 'Strict restored state differs')
    finite_tensors(model.state_dict().items(), 'restored state including BatchNorm buffers')
    return model.to(device).eval(), dict(checkpoint_sha256=digest(path), strict_restore=True,
        finite_weights_and_bn_buffers=True, selected_state_sha256=row['selected_state_sha256'])


def fit_one(kind, seed, rate, args, data, directory, device):
    import torch
    from tools import training_common as tc
    from utilsTorch.utilsTorch import segmentation_loss
    metrics, checkpoint_path = directory / 'metrics.json', directory / 'checkpoint.pt'
    if directory.exists():
        require(metrics.is_file() and checkpoint_path.is_file() and (directory / 'history.json').is_file(),
                'Incomplete attempt preserved at ' + str(directory) + '; use a new run or archive it explicitly')
        row = json.loads(metrics.read_text())
        audit_fit(row, kind, seed, rate, args, data)
        require(row['history'] == json.loads((directory / 'history.json').read_text()), 'Cached history file differs')
        restored, _ = restore_checkpoint(checkpoint_path, row, device='cpu')
        del restored
        return row
    directory.mkdir(parents=True)
    tc.seed_everything(seed)
    model = build_model(kind, seed, args).to(device)
    settings = fit_settings(kind, seed, rate, args, data)
    initial_state = state_hash(model.state_dict())
    common_initial = shared_hash(model)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    val_loader = tc.make_loader(data['val'], args.batch_size, False, seed)
    n_probe = min(8, args.batch_size, len(data['train']))
    probe_images = torch.stack([data['train'][index][0] for index in range(n_probe)]).to(device)
    probes = angle_probe(model, probe_images)
    history, best_score, best_state, best_epoch, best_val = [], -float('inf'), None, None, None
    all_norms = dict(core=[], common=[])
    started = time.perf_counter()
    try:
        for epoch in range(args.epochs + 1):
            loss_total, epoch_norms, schedule = 0.0, dict(core=[], common=[]), None
            if epoch:
                model.train()
                augmented = EpochDataset(data['train'], data['fit'], seed, epoch, args.augment)
                generator = torch.Generator().manual_seed(stable_seed(seed, epoch, 'loader'))
                loader = torch.utils.data.DataLoader(augmented, batch_size=args.batch_size,
                    sampler=augmented.order, num_workers=0, generator=generator)
                for images, masks in loader:
                    require(tuple(images.shape[-2:]) == (args.image_size, args.image_size), 'Training image geometry differs')
                    images, masks = images.to(device), masks.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = segmentation_loss(model(images), masks, loss_mode='dice_ce')
                    require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                    loss.backward()
                    for name, norm in gradient_norms(model).items():
                        epoch_norms[name].append(norm)
                        all_norms[name].append(norm)
                    optimizer.step()
                    finite_tensors(model.state_dict().items(), 'post-step state including BatchNorm buffers')
                    loss_total += float(loss.detach()) * len(images)
                require(augmented.observed_digest() == augmented.expected_digest, 'Actual data order/augmentation differs from schedule')
                schedule = dict(sha256=augmented.observed_digest(), n_samples=len(augmented.seen))
            point = dict(epoch=epoch, train_loss=loss_total / len(data['train']) if epoch else None,
                train_schedule=schedule, gradient_norms={name: dict(min=min(values), max=max(values),
                    mean=statistics.mean(values), steps=len(values), nonzero_steps=sum(value > 0 for value in values))
                    for name, values in epoch_norms.items() if values})
            if epoch == 0 or epoch % args.eval_every == 0 or epoch == args.epochs:
                point['val'] = evaluate(model, val_loader, device)
                validate_sensor_metrics(point['val'], len(data['validation']))
                if tc.is_better_val_score(point['val']['iou'], best_score, 'iou'):
                    best_score, best_epoch, best_val = point['val']['iou'], epoch, point['val']
                    best_state = tc.clone_state_dict_cpu(model.state_dict())
            history.append(point)
            pilot.write_json(directory / 'history.json', history)
            print(json.dumps(dict(id='hq_' + kind, seed=seed, learning_rate=rate, **point)), flush=True)
        model.load_state_dict(best_state, strict=True)
        finite_tensors(model.state_dict().items(), 'selected state including BatchNorm buffers')
        diagnostics = dict(initial_angle_probes=probes, selected_angle_probes=angle_probe(model, probe_images),
            movement=movement(model, initial), probe_names=[data['dataset'].names[data['fit'][i]] for i in range(n_probe)],
            gradient_norms={name: dict(min=min(values), max=max(values), mean=statistics.mean(values),
                nonzero_steps=sum(value > 0 for value in values), steps=len(values)) for name, values in all_norms.items() if values})
        row = dict(id='hq_' + kind, kind=kind, seed=seed, learning_rate=rate,
            model_config=model_configuration(model), settings=settings,
            total_params=sum(p.numel() for p in model.parameters()),
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            core_params=sum(p.numel() for p in model.bottleneck.core_parameters()),
            initial_state_sha256=initial_state, initial_shared_state_sha256=common_initial,
            selected_state_sha256=state_hash(model.state_dict()), best_epoch=best_epoch, val=best_val,
            history=history, diagnostics=diagnostics, elapsed_s=time.perf_counter() - started,
            fit_indices=data['fit'], val_indices=data['validation'],
            fit_names=[data['dataset'].names[i] for i in data['fit']],
            val_names=[data['dataset'].names[i] for i in data['validation']],
            fit_sensor_counts=dict(Counter(data['dataset'].modalities[i] for i in data['fit'])),
            normalization=data['normalization'], n_global_fit=len(data['all_fit']))
        finite_record(row)
        tc.atomic_torch_save(dict(format='hq-direction-v1', kind=kind, seed=seed,
            settings=settings, model_config=row['model_config'], best_epoch=best_epoch,
            best_state_dict=best_state), str(checkpoint_path))
        pilot.write_json(metrics, row)
        return row
    except BaseException as error:
        pilot.write_json(directory / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), completed_epochs=len(history) - 1))
        raise


def frozen_plan(path, candidate, resume):
    if resume:
        previous = json.loads(path.read_text())
        require(set(previous) == set(candidate), 'Resume plan schema differs')
        for key in candidate:
            if key != 'source_revision':
                require(previous[key] == candidate[key], 'Resume inputs differ: ' + key)
        return previous
    require(not path.parent.exists() or not any(path.parent.iterdir()), 'Use a new/empty work directory')
    pilot.write_json(path, candidate)
    return candidate


def analysis(records):
    groups = {}
    for row in records:
        group = groups.setdefault(row['id'], {})
        require(row['seed'] not in group, 'Duplicate selected model/seed')
        group[row['seed']] = row
    require(set(groups) == {'hq_quantum', 'hq_mlp'} and all(set(group) == set(SEEDS) for group in groups.values()),
            'Selected results require both arms and all three paired seeds')
    rows, paired = [], []
    for name, group in groups.items():
        rows.append(dict(id=name, total_params=group[42]['total_params'], core_params=group[42]['core_params'],
            **{split: {key: dict(mean=statistics.mean([group[seed][split][key] for seed in SEEDS]),
                std=statistics.stdev([group[seed][split][key] for seed in SEEDS]), n=3)
                for key in ('iou', 'optical_subset_iou', 'sar_subset_iou')} for split in ('val', 'test')}))
    for seed in SEEDS:
        left, right = groups['hq_quantum'][seed], groups['hq_mlp'][seed]
        for key in ('fit_names', 'val_names', 'test_names', 'normalization', 'initial_shared_state_sha256'):
            require(left[key] == right[key], 'Unpaired selected comparison: ' + key)
        require([point['train_schedule'] for point in left['history']] == [point['train_schedule'] for point in right['history']],
                'Selected arms have different data/augmentation order')
    for split in ('val', 'test'):
        differences = {str(seed): 100 * (groups['hq_quantum'][seed][split]['iou'] -
                                        groups['hq_mlp'][seed][split]['iou']) for seed in SEEDS}
        paired.append(dict(left='hq_quantum', right='hq_mlp', split=split, differences=differences,
            mean=statistics.mean(differences.values()), std=statistics.stdev(differences.values()),
            wins=sum(value > 0 for value in differences.values()), n=3, unit='IoU percentage points',
            same_learning_rate=all(groups['hq_quantum'][seed]['learning_rate'] == groups['hq_mlp'][seed]['learning_rate'] for seed in SEEDS)))
    return dict(rows=rows, paired_iou=paired, test_scope='Previously exposed test set; descriptive only')


def profile(args, device):
    import torch
    from tools import training_common as tc
    from utilsTorch.utilsTorch import segmentation_loss
    require(not args.work_dir.exists() or not any(args.work_dir.iterdir()), 'Use a new/empty profile directory')
    plan = dict(phase='profile', source_hashes=source_hashes(), settings=vars(args).copy(), runtime=runtime(device))
    plan['settings'] = {key: str(value) if isinstance(value, Path) else value for key, value in plan['settings'].items()}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    pilot.write_json(args.work_dir / 'profile_plan.json', plan)
    records, expected = [], None
    for kind in KINDS:
        started = time.perf_counter()
        try:
            tc.seed_everything(42)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = build_model(kind, 42, args).to(device).train()
            common = shared_hash(model)
            require(expected is None or common == expected, 'Profile shared initialization differs')
            expected = common
            initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            generator = torch.Generator().manual_seed(123)
            images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, generator=generator).to(device)
            masks = torch.randint(0, 2, (args.batch_size, args.image_size, args.image_size), generator=generator).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            require(tuple(output.shape) == (args.batch_size, 2, args.image_size, args.image_size), 'Profile output shape differs')
            finite_tensors([('output', output)], 'profile output')
            loss = segmentation_loss(output, masks, loss_mode='dice_ce')
            require(bool(torch.isfinite(loss)), 'Profile loss is nonfinite')
            loss.backward()
            gradients = gradient_norms(model)
            require(gradients['core'] > 0, 'Profile core gradient is zero')
            optimizer.step()
            finite_tensors(model.state_dict().items(), 'profile post-step state including BatchNorm')
            moved = movement(model, initial)
            require(moved['core']['l2'] > 0, 'Profile core did not move')
            restored = model_from_config(model_configuration(model))
            restored.load_state_dict({name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, strict=True)
            require(state_hash(restored.state_dict()) == state_hash(model.state_dict()), 'Profile strict state/BatchNorm roundtrip differs')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            records.append(dict(kind=kind, shared_initial_state_sha256=common,
                total_params=sum(p.numel() for p in model.parameters()), core_params=sum(p.numel() for p in model.bottleneck.core_parameters()),
                output_shape=list(output.shape), loss=float(loss.detach()), gradients=gradients, movement=moved,
                strict_state_and_bn_roundtrip=True,
                elapsed_s=time.perf_counter() - started,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None))
            del model, restored, optimizer, output, loss, images, masks, initial
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            pilot.write_json(args.work_dir / 'profile.json', dict(plan=plan, records=records,
                scope='Random-input instrumented full forward/backward/one Adam step; no data or accuracy or throughput claim'))
        except BaseException as error:
            pilot.write_json(args.work_dir / 'failure.json', dict(kind=kind, error_type=type(error).__name__,
                error=str(error), completed_records=records, traceback=traceback.format_exc()))
            raise
    return records


def run(args, device):
    import torch
    from tools import training_common as tc
    data_by_seed = {seed: pilot.load_training_data(args, seed) for seed in SEEDS}
    for data in data_by_seed.values():
        require(len(data['all_fit']) == 1303 and len(data['validation']) == 230,
                'Expected the global1303 fit/230 validation split')
        require(data['normalization'].get('fit_only') is True, 'Normalization must exclude validation')
    inputs = pilot.verify_inputs(args.data_root)
    settings = {key: value for key, value in vars(args).items() if key not in ('resume', 'work_dir', 'data_root', 'phase')}
    candidate = canonical(dict(settings=settings, seeds=SEEDS, learning_rates=RATES,
        source_hashes=source_hashes(), source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_identity=inputs, runtime=runtime(device),
        protocol_sha256=digest(ROOT / 'docs/HQ_DFC24_DIRECTION_PROTOCOL.md'),
        test_policy='All eight fits finish before six selected checkpoint tests; test already exposed/descriptive',
        augmentation_policy='Stateless D4 by(seed,epoch,original index), independent seeded sample order; train only',
        normalization_policy='All 1303 global fit images, validation excluded; optimization cap separate'))
    plan = frozen_plan(args.work_dir / 'plan.json', candidate, args.resume)
    fits, paths = [], {}
    for kind in KINDS:
        for rate in RATES:
            require(source_hashes() == plan['source_hashes'], 'Source changed during the run')
            path = args.work_dir / 'tuning' / ('hq_' + kind) / str(rate) / 'seed42'
            row = fit_one(kind, 42, rate, args, data_by_seed[42], path, device)
            fits.append(row); paths[(row['id'], 42, rate)] = path
    chosen = pilot.choose_rates(fits)
    pilot.write_json(args.work_dir / 'tuning_candidates.json', fits)
    pilot.write_json(args.work_dir / 'validation_selection.json', {name: dict(learning_rate=row['learning_rate'],
        best_epoch=row['best_epoch'], val_iou=row['val']['iou']) for name, row in chosen.items()})
    selected = [copy.deepcopy(chosen['hq_' + kind]) for kind in KINDS]
    for kind in KINDS:
        rate = chosen['hq_' + kind]['learning_rate']
        for seed in SEEDS[1:]:
            require(source_hashes() == plan['source_hashes'], 'Source changed during the run')
            path = args.work_dir / 'replication' / ('hq_' + kind) / ('seed' + str(seed))
            row = fit_one(kind, seed, rate, args, data_by_seed[seed], path, device)
            fits.append(row); selected.append(copy.deepcopy(row)); paths[(row['id'], seed, rate)] = path
    require(len(fits) == 8 and len(selected) == 6, 'Incomplete fitting grid')
    for seed in SEEDS:
        cohort = [row for row in fits if row['seed'] == seed]
        require(len({row['initial_shared_state_sha256'] for row in cohort}) == 1, 'Initial common weights/BatchNorm differ')
        for kind in KINDS:
            group = [row for row in cohort if row['kind'] == kind]
            require(len({row['initial_state_sha256'] for row in group}) == 1, 'Rate candidates have different initialization')
        require(all([point['train_schedule'] for point in row['history']] ==
                    [point['train_schedule'] for point in cohort[0]['history']] for row in cohort),
                'Data order or augmentation differs across paired fits')
    require(source_hashes() == plan['source_hashes'] and pilot.verify_inputs(args.data_root) == inputs,
            'Source or inputs changed before test replay')
    audits = []
    for row in fits:
        path = paths[(row['id'], row['seed'], row['learning_rate'])]
        restored, audit = restore_checkpoint(path / 'checkpoint.pt', row, device='cpu')
        audits.append(dict(id=row['id'], seed=row['seed'], learning_rate=row['learning_rate'], **audit))
        del restored
    pilot.write_json(args.work_dir / 'all_fits.json', [dict(directory=str(paths[(row['id'], row['seed'], row['learning_rate'])].relative_to(args.work_dir)), record=row) for row in fits])
    pilot.write_json(args.work_dir / 'checkpoint_audit.json', audits)
    # The only trained-model test access is below this completed-grid boundary.
    for row in selected:
        path = paths[(row['id'], row['seed'], row['learning_rate'])]
        model, _ = restore_checkpoint(path / 'checkpoint.pt', row, device=device)
        cfg = dict(row['settings'], augment=False)
        _, _, _, images, masks = tc.resolve_data_dirs('combined', str(args.data_root))
        dataset = tc.load_dataset(images, masks, cfg)
        require(dataset.norm_provenance == row['normalization'], 'Test normalization differs from selected fit')
        loader = tc.make_loader(dataset, args.batch_size, False, row['seed'])
        row['test'] = evaluate(model, loader, device)
        row['test_names'] = list(dataset.names)
        require(len(row['test_names']) == 205, 'Expected all 205 descriptive test tiles')
        validate_sensor_metrics(row['test'], 205, test=True)
        row['knockout'] = knockout_evaluation(model, loader, device, row['seed'])
        row['checkpoint'] = str((path / 'checkpoint.pt').relative_to(args.work_dir))
        pilot.write_json(path / 'selected_test.json', row)
        del model
    summary = dict(plan=plan, selection={name: row['learning_rate'] for name, row in chosen.items()},
                   records=selected, analysis=analysis(selected))
    require(source_hashes() == plan['source_hashes'] and pilot.verify_inputs(args.data_root) == inputs,
            'Source or inputs changed during test replay')
    pilot.write_json(args.work_dir / 'summary.json', summary)
    print('HQ DIRECTION STUDY COMPLETE', str(args.work_dir / 'summary.json'), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['profile', 'run'])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--fit-cap', type=int, default=128)
    parser.add_argument('--image-size', type=int, choices=[128], default=128)
    parser.add_argument('--base-channels', type=int, choices=[8], default=8)
    parser.add_argument('--eval-every', type=int, default=2)
    parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if min(args.batch_size, args.epochs, args.fit_cap, args.eval_every) < 1:
        parser.error('Training/profile budgets must be positive')
    if args.phase == 'run' and args.data_root is None:
        parser.error('run requires --data-root')
    if args.phase == 'profile' and args.resume:
        parser.error('A resource profile does not resume')
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
