#!/usr/bin/env python3
"""Freeze validation predictions for fixed-rule, descriptive HQ figures.

The complete historical campaign is audited without changing it. Affine example
selection is saved before any other model is predicted. Rendering is separate.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import collect_hq_training_controls as historical

CONTROLS = ('quantum', 'frozen_quantum', 'zero_bottleneck', 'compact_affine')
SEEDS = (42, 77, 123)
RATE = .003
PROTOCOL = 'docs/HQ_PUBLICATION_FIGURES_PROTOCOL.md'
SELF = 'misc/collect_hq_validation_predictions.py'
SOURCE_FILES = tuple(sorted(set(historical.study.SOURCE_FILES) | {
    SELF, 'scripts/collect_hq_training_controls.py', historical.study.PROTOCOL,
    historical.legacy.PROTOCOL}))
RULE = dict(reference='compact_affine', seed=42, learning_rate=RATE,
    minimum_ground_truth_flood_fraction=.05, sensors=['optical', 'sar'],
    ordering=['affine_foreground_iou', 'validation_index'],
    validation_index='Ordinal position in the saved ordered230 validation subset',
    worst_rank=0, lower_median_rank='(n-1)//2', retain_duplicate_selections=True)
require, digest = historical.require, historical.digest


def source_hashes():
    return {name: digest(ROOT/name) for name in SOURCE_FILES}


LOADED_SOURCES = source_hashes()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: '+value)
    result = json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)
    historical.study.hq.finite_record(result)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = f'{value.dtype}:{tuple(value.shape)}\n'.encode()
    return hashlib.sha256(header+value.tobytes()).hexdigest()


def save_arrays(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return dict(sha256=digest(path), arrays={name:dict(dtype=str(value.dtype),shape=list(value.shape),
        sha256=array_hash(value)) for name,value in arrays.items()})


def counts(prediction, target):
    return dict(tp=int(np.count_nonzero((prediction == 1) & (target == 1))),
        fp=int(np.count_nonzero((prediction == 1) & (target == 0))),
        fn=int(np.count_nonzero((prediction == 0) & (target == 1))),
        tn=int(np.count_nonzero((prediction == 0) & (target == 0))))


def score(values):
    denominator = values['tp']+values['fp']+values['fn']
    return values['tp']/denominator if denominator else 0.


def metrics(predictions, targets, names, sensors):
    per_image, pooled = [], []
    for control, prediction in zip(CONTROLS, predictions):
        for index, (mask, target) in enumerate(zip(prediction, targets)):
            values = counts(mask,target)
            per_image.append(dict(control=control,validation_index=index,name=names[index],sensor=sensors[index],
                **values,foreground_iou=score(values),ground_truth_flood_fraction=float(np.mean(target))))
        for sensor in ('pooled','optical','sar'):
            indices = np.arange(len(targets)) if sensor == 'pooled' else np.flatnonzero(np.asarray(sensors) == sensor)
            values = counts(prediction[indices],targets[indices])
            pooled.append(dict(control=control,sensor=sensor,n_samples=len(indices),**values,foreground_iou=score(values)))
    disagreements = []
    for left in range(len(CONTROLS)):
        for right in range(left+1,len(CONTROLS)):
            for index in range(len(targets)):
                different = int(np.count_nonzero(predictions[left,index] != predictions[right,index]))
                disagreements.append(dict(left=CONTROLS[left],right=CONTROLS[right],validation_index=index,
                    name=names[index],sensor=sensors[index],different_pixels=different,
                    fraction=different/(128*128)))
    return dict(per_image=per_image,pooled=pooled,disagreements=disagreements)


def select_examples(affine, targets, names, sensors, global_indices):
    rows = []
    for sensor in ('optical','sar'):
        eligible = [index for index in range(len(targets)) if sensors[index] == sensor
                    and float(np.mean(targets[index])) >= .05]
        ordered = sorted(eligible,key=lambda index:(score(counts(affine[index],targets[index])),index))
        for role, rank in (('worst',0),('lower_median',(len(ordered)-1)//2)):
            row = dict(slot=len(rows),sensor=sensor,role=role,eligible_count=len(ordered))
            if ordered:
                index = ordered[rank]
                row.update(status='selected',validation_index=index,global_index=global_indices[index],name=names[index],
                    affine_foreground_iou=score(counts(affine[index],targets[index])),
                    ground_truth_flood_fraction=float(np.mean(targets[index])))
            else:
                row['status'] = 'empty_eligible_set'
            rows.append(row)
    return rows


def parity(control, prediction, targets, sensors, row):
    observed = {}
    for sensor in ('optical','sar'):
        indices = np.flatnonzero(np.asarray(sensors) == sensor)
        actual = counts(prediction[indices],targets[indices])
        expected = {name:row['val'][sensor+'_subset_'+name] for name in actual}
        require(actual == expected and len(indices) == row['val'][sensor+'_subset_n_samples'],
                'Original validation count parity differs: '+control+'/'+sensor)
        observed[sensor] = dict(n_samples=len(indices),**actual)
    pooled = {name:sum(observed[sensor][name] for sensor in ('optical','sar')) for name in ('tp','fp','fn','tn')}
    require(math.isclose(score(pooled),row['val']['iou'],rel_tol=0,abs_tol=2e-15),
            'Original pooled validation IoU differs: '+control)
    return dict(control=control,exact_sensor_counts=True,pooled=dict(n_samples=len(targets),**pooled),sensors=observed)


@contextmanager
def wall_budget(seconds):
    require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds > 0, 'Positive finite wall budget required')
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(*args):
        raise TimeoutError('Prediction cache wall budget exhausted; partial evidence retained')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_handler)
        signal.setitimer(signal.ITIMER_REAL,*previous_timer)


def infer(control, row, checkpoint, data, device, *, retain_inputs=False):
    import torch
    from tools import training_common as tc
    tc.seed_everything(42)
    model, audit = historical.study.restore_checkpoint(checkpoint,row,device=device)
    model.eval()
    before = historical.study.hq.state_hash(model.state_dict())
    loader = tc.make_loader(data['val'],8,False,42)
    schedule = list(loader.batch_sampler)
    require([index for batch in schedule for index in batch] == list(range(230)) and
            all(len(batch) == 8 for batch in schedule[:-1]) and len(schedule[-1]) == 6,
            'Validation loader order/batching differs from original batch8 evaluation')
    predictions, targets, inputs = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            require(tuple(images.shape[1:]) == (3,128,128) and tuple(masks.shape[1:]) == (1,128,128),
                    'Validation tensor geometry differs')
            require(bool(torch.isfinite(images).all()) and bool(((masks == 0) | (masks == 1)).all()),
                    'Nonfinite inputs or nonbinary validation target')
            output = model(images.to(device))
            require(tuple(output.shape) == (len(images),2,128,128) and bool(torch.isfinite(output).all()),
                    'Invalid inference logits')
            predictions.append(output.argmax(dim=1).cpu().numpy().astype(np.uint8))
            targets.append(masks[:,0].cpu().numpy().astype(np.uint8))
            if retain_inputs:
                inputs.append(images.cpu().numpy().astype(np.float32))
    require(historical.study.hq.state_hash(model.state_dict()) == before,
            'Evaluation changed checkpoint weights or BatchNorm buffers')
    del model
    gc.collect()
    return np.concatenate(predictions),np.concatenate(targets),np.concatenate(inputs) if retain_inputs else None,audit


def collect(work_dir, data_root, output_dir, *, device='cuda', gpu_memory_fraction=None, budget_seconds=600):
    import torch
    work_dir,data_root,output_dir = (Path(value).resolve() for value in (work_dir,data_root,output_dir))
    require(not output_dir.exists() or not any(output_dir.iterdir()), 'Use a new or empty output directory; no partial resume')
    initial_plan = read_json(work_dir/'plan.json')
    legacy_root = Path(initial_plan['legacy_root']).resolve()
    for source in (work_dir,legacy_root,data_root,ROOT):
        require(output_dir != source and not source.is_relative_to(output_dir), 'Output contains an input/source root')
    for source in (work_dir,legacy_root,data_root):
        require(not output_dir.is_relative_to(source), 'Output aliases original raw inputs')
    require(source_hashes() == LOADED_SOURCES, 'Prediction source changed after import')
    require(device in ('cpu','cuda'), 'Unsupported prediction device')
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 2:
        torch.set_num_interop_threads(2)
    target_device = torch.device('cuda:0' if device == 'cuda' else 'cpu')
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable; no silent fallback')
        require(type(gpu_memory_fraction) in (int,float) and math.isfinite(gpu_memory_fraction)
                and 0 < gpu_memory_fraction <= 1, 'CUDA needs an explicit valid allocator fraction')
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction,target_device)
    else:
        require(gpu_memory_fraction is None, 'CPU prediction must not specify GPU allocator memory')
    output_dir.mkdir(parents=True,exist_ok=True)
    with (output_dir/'.writer.lock').open('x') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with wall_budget(budget_seconds):
                audited = historical.audit_campaign(work_dir,data_root)
                require(len(audited['fits']) == 20 and
                        sum(entry['origin'] == 'fresh' for entry in audited['fits']) == 12,
                        'Expected complete20-fit historical source campaign (12fresh/8reused)')
                selected = [entry for entry in audited['fits'] if entry['record']['kind'] in CONTROLS
                            and entry['record']['learning_rate'] == RATE]
                require(len(selected) == 12 and {(entry['record']['kind'],entry['record']['seed']) for entry in selected} ==
                        {(control,seed) for control in CONTROLS for seed in SEEDS} and
                        sum(entry['origin'] == 'fresh' for entry in selected) == 9,
                        'Primary figure population differs from fixed12 histories (9fresh/3reused)')
                original_inventory = dict(campaign=historical.file_inventory(work_dir),legacy=historical.file_inventory(legacy_root))
                original_input = audited['plan']['input_identity']
                runtime = historical.study.hq.runtime(target_device)
                runtime['gpu_allocator_fraction'] = gpu_memory_fraction if device == 'cuda' else None
                settings = argparse.Namespace(**audited['plan']['settings'],data_root=data_root)
                data = historical.study.pilot.load_training_data(settings,42)
                names = [data['dataset'].names[index] for index in data['validation']]
                sensors = [data['dataset'].modalities[index] for index in data['validation']]
                require(len(names) == len(set(names)) == 230 and len(data['all_fit']) == 1303 and len(data['fit']) == 128,
                        'Historical optimization/normalization/validation dimensions differ')
                seed42 = {entry['record']['kind']:entry for entry in selected if entry['record']['seed'] == 42}
                checkpoints = {}
                for control,entry in seed42.items():
                    row = entry['record']
                    require(row['val_names'] == names and row['val_indices'] == data['validation'] and
                            row['normalization'] == data['normalization'], 'Saved validation cohort/normalization differs')
                    directory = historical.study.entry_directory(work_dir,legacy_root,entry)
                    checkpoints[control] = dict(origin=entry['origin'],directory=entry['directory'],
                        checkpoint_path=str(directory/'checkpoint.pt'),checkpoint_sha256=entry['checkpoint_sha256'],
                        metrics_sha256=entry['metrics_sha256'],history_sha256=entry['history_sha256'],
                        selected_state_sha256=row['selected_state_sha256'],best_epoch=row['best_epoch'])
                plan = dict(format='hq-validation-prediction-plan-v1',controls=list(CONTROLS),seeds=list(SEEDS),
                    primary_learning_rate=RATE,prediction_seed=42,inference_order=['compact_affine',*CONTROLS[:-1]],
                    selection_rule=RULE,source_hashes=LOADED_SOURCES,protocol_sha256=digest(ROOT/PROTOCOL),
                    parent_plan_sha256=digest(work_dir/'plan.json'),parent_source_revision=audited['plan']['source_revision'],
                    parent_source_hashes=audited['plan']['source_hashes'],parent_runtime=audited['plan']['runtime'],
                    inference_runtime=runtime,input_identity=original_input,raw_inventory=original_inventory,
                    normalization=data['normalization'],validation_names=names,validation_sensors=sensors,
                    validation_global_indices=data['validation'],checkpoints=checkpoints,
                    original_campaign=dict(fits=20,fresh=12,reused=8),figure_history_population=dict(fits=12,fresh=9,reused=3),
                    settings=dict(batch_size=8,image_size=128,base_channels=8,budget_seconds=budget_seconds,
                        sequential_validation=True,model_eval=True,finite_shots=False),
                    execution_scope='Validation-only prediction of existing fixed checkpoints; no training or test prediction')
                write_json(output_dir/'plan.json',plan)
                histories = [dict(control=entry['record']['kind'],seed=entry['record']['seed'],origin=entry['origin'],
                    learning_rate=RATE,best_epoch=entry['record']['best_epoch'],history=entry['record']['history'],
                    checkpoint_sha256=entry['checkpoint_sha256'],history_sha256=entry['history_sha256']) for entry in selected]
                write_json(output_dir/'histories.json',histories)
                affine = seed42['compact_affine']
                prediction,targets,all_inputs,audit = infer('compact_affine',affine['record'],
                    checkpoints['compact_affine']['checkpoint_path'],data,target_device,retain_inputs=True)
                parity_records = [parity('compact_affine',prediction,targets,sensors,affine['record'])]
                affine_archive = save_arrays(output_dir/'affine_predictions.npz',predictions=prediction,targets=targets)
                selection = dict(format='hq-validation-example-selection-v1',rule=RULE,
                    plan_sha256=digest(output_dir/'plan.json'),affine_archive=affine_archive,
                    affine_checkpoint_sha256=checkpoints['compact_affine']['checkpoint_sha256'],
                    examples=select_examples(prediction,targets,names,sensors,data['validation']))
                write_json(output_dir/'selection.json',selection)
                # This file is the durable phase boundary: other model outputs
                # cannot affect selection, and partial inference is never retried.
                selection_hash = digest(output_dir/'selection.json')
                display = np.zeros((4,3,128,128),dtype=np.float32)
                valid = np.zeros(4,dtype=np.bool_)
                for item in selection['examples']:
                    if item['status'] == 'selected':
                        display[item['slot']] = all_inputs[item['validation_index']]
                        valid[item['slot']] = True
                del all_inputs
                predictions = np.zeros((4,230,128,128),dtype=np.uint8)
                predictions[3] = prediction
                for control in CONTROLS[:-1]:
                    require(digest(output_dir/'selection.json') == selection_hash, 'Frozen example selection changed')
                    entry = seed42[control]
                    current,current_targets,_,audit = infer(control,entry['record'],checkpoints[control]['checkpoint_path'],data,target_device)
                    require(np.array_equal(current_targets,targets), 'Validation targets/order changed between models')
                    parity_records.append(parity(control,current,targets,sensors,entry['record']))
                    predictions[CONTROLS.index(control)] = current
                payload = save_arrays(output_dir/'validation_predictions.npz',predictions=predictions,targets=targets,
                                      display_inputs=display,display_valid=valid)
                write_json(output_dir/'metrics.json',metrics(predictions,targets,names,sensors))
                write_json(output_dir/'parity.json',dict(passed=True,records=parity_records,
                    equality='Exact integer sensor confusion counts and cohort sizes; pooled counts follow by summation',
                    loss_or_auc_recovery=False))
                require(source_hashes() == LOADED_SOURCES and digest(ROOT/PROTOCOL) == plan['protocol_sha256'],
                        'Prediction source/protocol changed during inference')
                require(historical.study.pilot.verify_inputs(data_root) == original_input, 'Input bytes changed during prediction')
                require(dict(campaign=historical.file_inventory(work_dir),legacy=historical.file_inventory(legacy_root)) == original_inventory,
                        'Original historical artifacts changed during prediction')
                require(digest(output_dir/'selection.json') == selection_hash, 'Selection bytes changed after inference')
                source_manifest = {}
                for name in (*SOURCE_FILES,PROTOCOL):
                    target = output_dir/'sources'/(name+'.txt');target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes((ROOT/name).read_bytes())
                    source_manifest[name] = dict(sha256=digest(target),retained_as=str(target.relative_to(output_dir)))
                write_json(output_dir/'source_manifest.json',source_manifest)
                cache = dict(format='hq-publication-validation-cache-v1',complete=True,controls=list(CONTROLS),
                    names=names,sensors=sensors,global_indices=data['validation'],payload=payload,
                    payload_file='validation_predictions.npz',plan_sha256=digest(output_dir/'plan.json'),
                    selection_sha256=selection_hash,histories_sha256=digest(output_dir/'histories.json'),
                    metrics_sha256=digest(output_dir/'metrics.json'),parity_sha256=digest(output_dir/'parity.json'),
                    source_manifest_sha256=digest(output_dir/'source_manifest.json'),
                    scope='Descriptive seed42 validation examples, selected only from affine predictions; not independent evidence')
                write_json(output_dir/'cache.json',cache)
                write_json(output_dir/'artifact_sha256.json',historical.file_inventory(output_dir))
                return cache
        except BaseException as error:
            if not (output_dir/'failure.json').exists():
                write_json(output_dir/'failure.json',dict(error_type=type(error).__name__,error=str(error),
                    traceback=traceback.format_exc(),partial_evidence_preserved=True))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--gpu-memory-fraction',type=float)
    parser.add_argument('--budget-seconds',type=float,required=True)
    args = parser.parse_args(argv)
    result = collect(args.work_dir,args.data_root,args.output_dir,device=args.device,
        gpu_memory_fraction=args.gpu_memory_fraction,budget_seconds=args.budget_seconds)
    print('Frozen four-model230-tile validation cache:',args.output_dir)
    return result


if __name__ == '__main__':
    main()
