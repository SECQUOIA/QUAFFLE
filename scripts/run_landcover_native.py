#!/usr/bin/env python3
"""Profile, freeze and run a bounded native LandCover.ai comparison.

The test split is never loaded. A common learning rate is selected using only
the compact classical arm at seed42, before quantum/frozen training. All other
training conditions, data schedules and checkpoint selection rules are paired.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTROLS = ('compact_affine', 'quantum', 'frozen_quantum', 'zero_bottleneck')
SOURCES = ('scripts/run_landcover_native.py', 'scripts/prepare_landcover_native.py',
           'scripts/collect_landcover_native.py', 'tools/landcover_data.py',
           'tools/landcover_metrics.py', 'utilsTorch/hq_landcover.py',
           'utilsTorch/landcover_circuit.py',
           'utilsTorch/hq_direction.py', 'utilsTorch/unetTorch.py',
           'utilsTorch/__init__.py', 'utilsTorch/utilsTorch.py',
           'docs/LANDCOVER_NATIVE_PROTOCOL.md')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sources():
    return {name: sha(ROOT/name) for name in SOURCES}


def state_sha(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        result.update(f'{name}:{value.dtype}:{tuple(value.shape)}\n'.encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def runtime(device):
    import torch
    return dict(python=sys.version, executable=sys.executable, device=str(device),
                packages={key: importlib.metadata.version(key)
                          for key in ('torch', 'pennylane', 'numpy', 'Pillow', 'rasterio')},
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                cuda=torch.version.cuda, threads=torch.get_num_threads(),
                gpu_fraction=.12 if device.type == 'cuda' else None)


def setup(device_name):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0' if device_name == 'cuda' else device_name)
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'Requested CUDA is unavailable')
        torch.cuda.set_per_process_memory_fraction(.12, device)
    return device


@contextmanager
def campaign_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.campaign.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def synchronize(device):
    if device.type == 'cuda':
        import torch
        torch.cuda.synchronize(device)


def checked_gradients(model):
    """Inspect actual backpropagation, including frozen-input semantics."""
    import torch
    core, common = [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            require(bool(torch.isfinite(parameter.grad).all()), 'Nonfinite gradient: '+name)
            (core if name.startswith('bottleneck.core.') else common).append(parameter.grad)
    return dict(core_l2=math.sqrt(sum(float(v.detach().double().square().sum()) for v in core)),
                common_l2=math.sqrt(sum(float(v.detach().double().square().sum()) for v in common)))


def profile(args):
    import torch
    from utilsTorch import hq_landcover as native
    from tools.landcover_metrics import segmentation_loss
    output = Path(args.output).resolve()
    require(not output.exists(), 'Profile output already exists')
    device = setup(args.device)
    identity = sources()
    result = dict(format='landcover-native-profile-v1', source=identity,
                  runtime=runtime(device), batch_size=args.batch_size,
                  base_channels=args.base_channels, warmup_steps=1, steps=2,
                  actual_optimizer_steps=3, arms={})
    started = time.monotonic()
    for control in CONTROLS:
        model = native.build_model(control, seed=42, base_channels=args.base_channels).to(device)
        before = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        image = torch.rand(args.batch_size, 3, 128, 128,
                           generator=torch.Generator().manual_seed(123)).to(device)*2-1
        mask = (torch.arange(args.batch_size*128*128).reshape(args.batch_size, 128, 128) % 5).to(device)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(3):
            require(time.monotonic()-started < args.budget_seconds, 'Profile time budget exhausted')
            synchronize(device)
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(image), mask)
            require(bool(torch.isfinite(loss)), 'Nonfinite profile loss')
            loss.backward()
            gradients = checked_gradients(model)
            optimizer.step()
            synchronize(device)
            durations.append(time.monotonic()-tick)
        core_change = sum(float((p.detach().cpu().double()-before[n].double()).square().sum())
                          for n, p in model.named_parameters() if n.startswith('bottleneck.core.'))
        require(gradients['common_l2'] > 0, 'No common gradient')
        require((core_change > 0 and gradients['core_l2'] > 0)
                if control in ('quantum', 'compact_affine') else core_change == 0,
                'Incorrect core trainability in profile')
        # A frozen circuit must still transmit a real upstream input derivative.
        projection_grad = model.bottleneck.pre_projection.weight.grad
        if control == 'frozen_quantum':
            require(projection_grad is not None and float(projection_grad.abs().sum()) > 0,
                    'Frozen circuit disconnected its inputs')
        model.eval()
        synchronize(device)
        tick = time.monotonic()
        with torch.no_grad():
            predictions = model(image)
        synchronize(device)
        eval_seconds = time.monotonic()-tick
        result['arms'][control] = dict(warmup_seconds=durations[0], step_seconds=durations[1:],
            eval_batch_seconds=eval_seconds, finite_output=bool(torch.isfinite(predictions).all()),
            gradients=gradients, core_change_l2=math.sqrt(core_change),
            initial_common_sha=native.shared_state_sha256(native.build_model(control, seed=42,
                base_channels=args.base_channels)), counts=native.parameter_counts(model),
            peak_cuda_allocated=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            peak_cuda_reserved=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)
        del model, optimizer, image, mask, predictions, loss, before, projection_grad
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    require(len({row['initial_common_sha'] for row in result['arms'].values()}) == 1,
            'Control initialization is not paired')
    require(all(row['finite_output'] for row in result['arms'].values()), 'Nonfinite profile outputs')
    require(identity == sources(), 'Source changed during profile')
    result.update(status='passed', elapsed_seconds=time.monotonic()-started)
    write_json(output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


def estimate_cost(profile_record, epochs, train_tiles, val_tiles):
    """Preliminary whole-study seconds; startup is paid once per fit.

    Include all 13 fits, their validation passes, 12 selected checkpoint replays,
    and 30 minutes reserved for file loading/writes/audits/collection. A separate
    2x acceptance margin and persisted deadline guard this short profile estimate.
    """
    require(profile_record['warmup_steps'] == 1 and profile_record['steps'] == 2
            and profile_record['actual_optimizer_steps'] == 3, 'Wrong profile warmup/measurement coverage')
    batch = profile_record['batch_size']
    train_steps, evaluation_batches = epochs*train_tiles//batch, val_tiles*16/batch
    counts = {control: 4 if control == 'compact_affine' else 3 for control in CONTROLS}
    parts = dict(startup_seconds=0., optimization_seconds=0., validation_seconds=0.,
                 selected_replay_seconds=0., io_and_collection_allowance_seconds=1800.)
    require(set(profile_record['arms']) == set(CONTROLS), 'Incomplete profile controls')
    for control, row in profile_record['arms'].items():
        durations = row['step_seconds']
        require(len(durations) == 2 and all(math.isfinite(v) and v > 0
            for v in [*durations, row['warmup_seconds'], row['eval_batch_seconds']]), 'Invalid profile durations')
        n = counts[control]
        parts['startup_seconds'] += n*row['warmup_seconds']
        parts['optimization_seconds'] += n*train_steps*max(durations)
        parts['validation_seconds'] += n*(1+epochs//5)*evaluation_batches*row['eval_batch_seconds']
        parts['selected_replay_seconds'] += 3*evaluation_batches*row['eval_batch_seconds']
    parts['total_seconds'] = math.fsum(parts.values())
    return parts


def make_plan(args):
    from tools.landcover_data import read_manifest, deterministic_subset
    profile_record = read_json(args.profile)
    require(profile_record['status'] == 'passed' and profile_record['source'] == sources(),
            'Profile is absent/stale/failed')
    require((args.epochs, args.train_tiles, args.val_tiles) == (60, 256, 64),
            'The frozen native protocol requires 60 epochs, 256 train tiles and 64 validation tiles')
    require(args.train_tiles % profile_record['batch_size'] == 0, 'Training cohort must fill batches')
    data_root, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    manifest = read_manifest(data_root)
    train_ids = deterministic_subset(manifest['splits']['train'], args.train_tiles, seed=742)
    val_ids = deterministic_subset(manifest['splits']['val'], args.val_tiles, seed=943)
    class_counts = [sum(manifest['tiles'][name]['class_pixel_counts'][i] for name in val_ids) for i in range(5)]
    train_class_counts = [sum(manifest['tiles'][name]['class_pixel_counts'][i] for name in train_ids) for i in range(5)]
    require(all(v > 0 for v in class_counts), 'Validation subset omits a true class')
    require(all(v > 0 for v in train_class_counts), 'Training subset omits a true class')
    batch = profile_record['batch_size']
    training_steps = args.epochs * args.train_tiles//batch
    cost = estimate_cost(profile_record, args.epochs, args.train_tiles, args.val_tiles)
    estimate = cost['total_seconds']
    require(2*estimate < args.budget_seconds, 'Profile predicts insufficient campaign time budget')
    record = dict(format='landcover-native-plan-v1', source=sources(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dataset_manifest_sha=sha(data_root/'manifest.json'), archive=manifest.get('archive'),
        profile_sha=sha(args.profile), profile=profile_record, seeds=[42, 77, 123],
        controls=list(CONTROLS), rate_candidates=[.001, .003],
        common_rate_selection='highest seed42 compact_affine validation mIoU; lower LR on exact ties',
        train_ids=train_ids, val_ids=val_ids, val_class_pixels=class_counts,
        train_class_pixels=train_class_counts,
        image_size=128, epochs=args.epochs, evaluation_every=5,
        batch_size=batch, base_channels=profile_record['base_channels'],
        training_steps_per_fit=training_steps, training_examples_per_fit=args.epochs*args.train_tiles,
        validation_examples_per_evaluation=16*args.val_tiles,
        optimizer='Adam(default betas/eps/weight_decay=0)',
        loss='0.5 unweighted CE + 0.5 macro soft Dice, all5 classes',
        checkpoint_selection='earliest strict highest validation mIoU, including epoch0',
        test_predictions=0, estimated_seconds=estimate, cost_estimate=cost, budget_seconds=args.budget_seconds,
        runtime=profile_record['runtime'])
    with campaign_lock(output):
        require(not (output/'plan.json').exists(), 'An immutable plan already exists')
        write_json(output/'plan.json', record)
    print(json.dumps(dict(status='planned', estimated_seconds=estimate,
                          output=str(output), plan_sha=sha(output/'plan.json'))), flush=True)


def evaluate(model, dataset, batch_size, device, deadline):
    import numpy as np
    import torch
    from tools.landcover_metrics import confusion_counts, metrics_from_counts, segmentation_loss
    before = state_sha(model.state_dict())
    was_training = model.training
    model.eval()
    counts, losses, examples = np.zeros((5, 5), dtype=np.int64), 0., 0
    try:
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
                require(time.time() < deadline, 'Persisted campaign deadline exceeded')
                image, mask = batch['image'].to(device), batch['mask'].to(device)
                logits = model(image)
                loss = segmentation_loss(logits, mask)
                require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)), 'Nonfinite evaluation')
                counts += confusion_counts(mask.cpu().numpy(), logits.argmax(1).cpu().numpy())
                losses += float(loss)*len(mask)
                examples += len(mask)
    finally:
        model.train(was_training)
    require(state_sha(model.state_dict()) == before, 'Evaluation mutated model/BN state')
    require(examples == len(dataset), 'Incomplete validation evaluation')
    result = metrics_from_counts(counts)
    result.update(loss=losses/examples, examples=examples)
    return result


def fit(control, seed, rate, plan, data_root, destination, deadline, device):
    import torch
    from tools.landcover_data import NativeLandCoverDataset
    from tools.landcover_metrics import segmentation_loss
    from utilsTorch import hq_landcover as native
    require(not destination.exists(), 'Fit destination exists; partial work is never silently retried')
    destination.mkdir()
    write_json(destination/'status.json', dict(status='started', control=control, seed=seed,
        rate=rate, source=plan['source'], plan_sha=plan['_plan_sha']))
    started = time.monotonic()
    model = native.build_model(control, seed=seed, base_channels=plan['base_channels']).to(device)
    initial_common = native.shared_state_sha256(model)
    initial_core = {n: p.detach().cpu().clone() for n,p in model.named_parameters()
                    if n.startswith('bottleneck.core.')}
    train_data = NativeLandCoverDataset(data_root, 'train', tile_ids=plan['train_ids'], seed=seed)
    val_data = NativeLandCoverDataset(data_root, 'val', tile_ids=plan['val_ids'], seed=seed)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=rate)
    history, training_history, best_score, best_epoch, optimizer_steps = [], [], -1., None, 0
    schedule_hashes = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(plan['epochs']+1):
        require(time.time() < deadline, 'Persisted campaign deadline exceeded')
        if epoch:
            model.train()
            train_data.set_epoch(epoch)
            order_seed = int.from_bytes(hashlib.sha256(f'{seed}:{epoch}:order'.encode()).digest()[:8], 'little')
            generator = torch.Generator().manual_seed(order_seed)
            loader = torch.utils.data.DataLoader(train_data, batch_size=plan['batch_size'],
                shuffle=True, generator=generator, num_workers=0, drop_last=False)
            schedule, total_loss, n = hashlib.sha256(), 0., 0
            for batch in loader:
                require(time.time() < deadline, 'Persisted campaign deadline exceeded')
                identity = list(zip(batch['tile_id'], batch['crop_x'].tolist(),
                    batch['crop_y'].tolist(), batch['d4'].tolist()))
                schedule.update(json.dumps(identity, separators=(',', ':')).encode())
                image, mask = batch['image'].to(device), batch['mask'].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = segmentation_loss(model(image), mask)
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                loss.backward()
                if optimizer_steps == 0:
                    first_gradients = checked_gradients(model)
                optimizer.step()
                total_loss += float(loss.detach())*len(mask)
                n += len(mask)
                optimizer_steps += 1
            require(n == len(train_data), 'Incomplete training epoch')
            schedule_hashes.append(schedule.hexdigest())
            train_loss = total_loss/n
        else:
            train_loss = None
        if epoch % plan['evaluation_every'] == 0:
            validation = evaluate(model, val_data, plan['batch_size'], device, deadline)
            row = dict(epoch=epoch, optimizer_steps=optimizer_steps, train_loss=train_loss,
                       validation=validation, elapsed_seconds=time.monotonic()-started)
            history.append(row)
            if validation['miou'] > best_score:
                best_score, best_epoch = validation['miou'], epoch
                checkpoint = dict(format='landcover-native-checkpoint-v1', model_config=native.model_config(model),
                    state_dict={n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
                    source=plan['source'], plan_sha=plan['_plan_sha'],
                    epoch=epoch, optimizer_steps=optimizer_steps, validation=validation)
                temporary = destination/'best.pt.tmp'
                torch.save(checkpoint, temporary)
                temporary.replace(destination/'best.pt')
            write_json(destination/'history.json', history)
            print(json.dumps(dict(control=control, seed=seed, rate=rate, **row)), flush=True)
        training_history.append(dict(epoch=epoch, optimizer_steps=optimizer_steps,
            train_loss=train_loss, elapsed_seconds=time.monotonic()-started))
        write_json(destination/'training_history.json', training_history)
    require(optimizer_steps == plan['training_steps_per_fit'], 'Wrong completed optimizer step count')
    core_change = math.sqrt(sum(float((p.detach().cpu().double()-initial_core[n].double()).square().sum())
                               for n,p in model.named_parameters() if n in initial_core))
    require((core_change > 0 and first_gradients['core_l2'] > 0)
            if control in ('quantum', 'compact_affine') else core_change == 0,
            'Incorrect trained core movement')
    result = dict(status='complete', control=control, seed=seed, rate=rate,
        model_config=native.model_config(model), counts=native.parameter_counts(model),
        initial_common_sha=initial_common, first_gradients=first_gradients,
        core_change_l2=core_change, optimizer_steps=optimizer_steps,
        epochs=plan['epochs'], best_epoch=best_epoch, best_miou=best_score,
        checkpoint_sha=sha(destination/'best.pt'), history_sha=sha(destination/'history.json'),
        history=history, training_history=training_history,
        training_history_sha=sha(destination/'training_history.json'), schedule_hashes=schedule_hashes,
        final_state_sha=state_sha(model.state_dict()), source=plan['source'], plan_sha=plan['_plan_sha'],
        elapsed_seconds=time.monotonic()-started,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
    write_json(destination/'result.json', result)
    write_json(destination/'status.json', dict(status='complete', result_sha=sha(destination/'result.json')))
    del model, optimizer, checkpoint, train_data, val_data
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def verify_completed_fit(destination, plan, expected_identity):
    status = read_json(destination/'status.json')
    require(status['status'] == 'complete', 'Partial fit is not silently retried')
    result = read_json(destination/'result.json')
    require((result['control'], result['seed'], result['rate']) == expected_identity,
            'Completed fit identity differs')
    require(sha(destination/'result.json') == status['result_sha']
        and result['source'] == plan['source'] and result['plan_sha'] == plan['_plan_sha']
        and sha(destination/'best.pt') == result['checkpoint_sha']
        and sha(destination/'history.json') == result['history_sha']
        and sha(destination/'training_history.json') == result['training_history_sha'], 'Completed fit changed')
    return result


def run(args):
    from tools.landcover_data import read_manifest
    output, data_root = Path(args.output).resolve(), Path(args.dataset).resolve()
    with campaign_lock(output):
        plan = read_json(output/'plan.json')
        require(plan['format'] == 'landcover-native-plan-v1' and plan['source'] == sources(), 'Stale plan source')
        require(plan['dataset_manifest_sha'] == sha(data_root/'manifest.json'), 'Dataset manifest changed')
        device = setup(args.device)
        require(runtime(device) == plan['runtime'], 'Profile/plan runtime changed')
        plan['_plan_sha'] = sha(output/'plan.json')
        manifest = read_manifest(data_root)
        # Verify every selected file before any fit; only official train/val.
        for name in plan['train_ids']+plan['val_ids']:
            for key in ('image', 'mask'):
                row = manifest['tiles'][name]
                require(sha(data_root/row[key]) == row[key+'_sha256'], 'Selected data changed')
        state_path = output/'status.json'
        if state_path.exists():
            state = read_json(state_path)
            require(state['plan_sha'] == plan['_plan_sha'], 'Persisted plan binding changed')
            if state['status'] == 'complete':
                require(sha(output/'all_fits.json') == state['all_fits_sha'], 'Completed results changed')
                rows = read_json(output/'all_fits.json')
                require(len(rows) == state['fits'] == 13, 'Completed fit grid differs')
                for row in rows:
                    directory = output/f"{row['control']}_seed{row['seed']}_lr{row['rate']:g}"
                    require(verify_completed_fit(directory, plan,
                        (row['control'], row['seed'], row['rate'])) == row, 'Completed fit record differs')
                print('Completed campaign verified; no fitting or inference repeated', flush=True)
                return
            require(state['status'] == 'running', 'Failed campaigns require explicit diagnosis; no automatic retry')
            deadline = state['deadline_unix']
        else:
            deadline = time.time()+plan['budget_seconds']
            write_json(state_path, dict(status='running', plan_sha=plan['_plan_sha'], deadline_unix=deadline))
        require(time.time() < deadline, 'Persisted campaign deadline exceeded')
        all_fits = []
        def obtain(control, seed, rate):
            destination = output/f'{control}_seed{seed}_lr{rate:g}'
            if destination.exists():
                result = verify_completed_fit(destination, plan, (control,seed,rate))
            else:
                result = fit(control, seed, rate, plan, data_root, destination, deadline, device)
            all_fits.append(result)
            write_json(output/'all_fits.json', all_fits)
            return result
        try:
            candidates = [obtain('compact_affine', 42, rate) for rate in plan['rate_candidates']]
            selected = max(candidates, key=lambda row: (row['best_miou'], -row['rate']))['rate']
            selection = dict(rate=selected, source='seed42 compact_affine validation only',
                candidates=[{key:row[key] for key in ('rate','best_miou','best_epoch','checkpoint_sha')}
                            for row in candidates], plan_sha=plan['_plan_sha'])
            selection_path = output/'rate_selection.json'
            if selection_path.exists():
                require(read_json(selection_path) == selection, 'Persisted common rate changed')
            else:
                write_json(selection_path, selection)
            for seed in plan['seeds']:
                for control in plan['controls']:
                    if control == 'compact_affine' and seed == 42:
                        continue  # Audited selected tuning fit is the primary fit.
                    obtain(control, seed, selected)
            for seed in plan['seeds']:
                rows = [row for row in all_fits if row['seed'] == seed and row['rate'] == selected]
                require(len(rows) == len(CONTROLS) and len({row['initial_common_sha'] for row in rows}) == 1,
                        'Unpaired primary initialization')
                require(all(row['schedule_hashes'] == rows[0]['schedule_hashes'] for row in rows),
                        'Unpaired training crop/augmentation/order schedules')
            require(plan['source'] == sources(), 'Source changed during campaign')
            write_json(state_path, dict(status='complete', plan_sha=plan['_plan_sha'],
                deadline_unix=deadline, all_fits_sha=sha(output/'all_fits.json'), fits=len(all_fits),
                test_predictions=0))
        except BaseException:
            write_json(state_path, dict(status='failed', plan_sha=plan['_plan_sha'],
                deadline_unix=deadline, error=traceback.format_exc()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('profile')
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=8, choices=(4, 8))
    p.add_argument('--base-channels', type=int, default=64, choices=(64,))
    p.add_argument('--budget-seconds', type=int, default=300)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p = commands.add_parser('plan')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--train-tiles', type=int, default=256, choices=(256,))
    p.add_argument('--val-tiles', type=int, default=64, choices=(64,))
    p.add_argument('--epochs', type=int, default=60, choices=(60,))
    p.add_argument('--budget-seconds', type=int, default=86400)
    p = commands.add_parser('run')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    args = parser.parse_args()
    {'profile': profile, 'plan': make_plan, 'run': run}[args.command](args)


if __name__ == '__main__':
    main()
