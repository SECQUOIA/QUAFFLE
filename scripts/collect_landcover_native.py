#!/usr/bin/env python3
"""Replay native validation checkpoints once, then export without model imports.

The replay phase has a durable finite deadline and write-before-forward intents.
The export phase uses only immutable numeric records and produces descriptive
five-class comparisons; it never opens image tiles or imports Torch/models.
"""
import argparse
import ast
from contextlib import contextmanager
import csv
import fcntl
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
SELF = 'scripts/collect_landcover_native.py'
CONTROLS = ('compact_affine','quantum','frozen_quantum','zero_bottleneck')
SEEDS = (42,77,123)
CLASSES = ('Background','Building','Woodland','Water','Road')
LABELS = dict(compact_affine='Compact affine',quantum='Learned quantum',
              frozen_quantum='Frozen quantum',zero_bottleneck='Zero bottleneck')


def require(value,message):
    if not value: raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


LOADED_SELF = sha(ROOT/SELF)


def read(path):
    def unique(pairs):
        result = {}
        for key,value in pairs:
            require(key not in result,'Duplicate JSON key: '+key);result[key] = value
        return result
    def invalid(value):raise ValueError('Nonfinite JSON: '+value)
    result = json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=invalid)
    finite(result)
    return result


def finite(value):
    if isinstance(value,float):require(math.isfinite(value),'Nonfinite number')
    elif isinstance(value,dict):
        for entry in value.values():finite(entry)
    elif isinstance(value,list):
        for entry in value:finite(entry)


def write(path,value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def inventory(root):
    return {str(path.relative_to(root)):sha(path) for path in sorted(Path(root).rglob('*')) if path.is_file()}


def inside(root,name):
    path = (root/name).resolve()
    require(path != root and path.is_relative_to(root),'Unsafe artifact/source path')
    return path


def integer(value,name,minimum=0):
    require(type(value) is int and value >= minimum,'Invalid integer: '+name)
    return value


def identity(row):return row['control'],row['seed'],row['rate']


def fit_name(row):return f"{row['control']}_seed{row['seed']}_lr{row['rate']:g}"


def metric_counts(counts):
    """Independent pooled five-class arithmetic; rows truth, columns prediction."""
    require(isinstance(counts,list) and len(counts) == 5 and all(isinstance(row,list) and len(row) == 5 for row in counts),
            'Confusion matrix must be 5 by 5')
    require(all(type(value) is int and 0 <= value < 2**50 for row in counts for value in row),'Counts must be bounded nonnegative integers')
    matrix = np.asarray(counts,dtype=np.int64)
    require(matrix.max(initial=0) < 2**60 and int(matrix.sum()) > 0,'Invalid count scale/empty matrix')
    truth,prediction,diagonal = matrix.sum(1),matrix.sum(0),matrix.diagonal()
    union = truth+prediction-diagonal
    values = [int(tp)/int(n) if n else 0. for tp,n in zip(diagonal,union)]
    return dict(confusion=counts,pixels=int(matrix.sum()),truth_pixels=truth.tolist(),prediction_pixels=prediction.tolist(),
        class_iou=dict(zip(CLASSES,values)),absent_union=[CLASSES[i] for i in range(5) if union[i] == 0],
        miou=sum(values)/5,foreground_miou=sum(values[1:])/4,accuracy=int(diagonal.sum())/int(matrix.sum()))


def check_metrics(row,plan):
    expected = metric_counts(row['confusion'])
    for key,value in expected.items():
        if isinstance(value,float):
            require(type(row[key]) in (int,float) and math.isclose(row[key],value,rel_tol=0,abs_tol=2e-15),
                    'Confusion-derived metric differs: '+key)
        elif isinstance(value,dict):
            require(set(row[key]) == set(value) and all(math.isclose(row[key][k],v,rel_tol=0,abs_tol=2e-15)
                    for k,v in value.items()),'Class IoUs differ')
        else:require(row[key] == value,'Confusion-derived metric differs: '+key)
    require(row['truth_pixels'] == plan['val_class_pixels'],'Validation truth class counts differ')
    require(row['examples'] == plan['validation_examples_per_evaluation'] and
            row['pixels'] == row['examples']*128*128,'Validation crop/pixel coverage differs')
    require(type(row['loss']) in (int,float) and math.isfinite(row['loss']) and row['loss'] >= 0,'Invalid validation loss')
    return expected


def audit_campaign(campaign,dataset=None):
    """Audit the whole completed grid without Torch, model loading or inference."""
    root = Path(campaign).resolve();plan = read(root/'plan.json');plan_sha = sha(root/'plan.json')
    require(plan['format'] == 'landcover-native-plan-v1' and plan['test_predictions'] == 0,'Wrong native plan/test exposure')
    require(plan['controls'] == list(CONTROLS) and plan['seeds'] == list(SEEDS) and
            plan['rate_candidates'] == [.001,.003],'Frozen controls/seeds/rates differ')
    require(plan['image_size'] == 128 and plan['base_channels'] == 64 and plan['batch_size'] in (4,8) and
            plan['epochs'] == 60 and plan['evaluation_every'] == 5 and len(plan['train_ids']) == 256 and
            len(plan['val_ids']) == 64,'Full native protocol dimensions differ')
    require(len(set(plan['train_ids'])) == 256 and len(set(plan['val_ids'])) == 64 and
            not set(plan['train_ids']).intersection(plan['val_ids']),'Invalid selected cohorts')
    require(plan['training_steps_per_fit'] == 60*256//plan['batch_size'] and
            plan['training_examples_per_fit'] == 60*256 and plan['validation_examples_per_evaluation'] == 64*16,
            'Optimizer steps/examples differ')
    require(len(plan['val_class_pixels']) == 5 and all(type(v) is int and v > 0 for v in plan['val_class_pixels']) and
            sum(plan['val_class_pixels']) == 64*512*512,'Validation class/pixel coverage differs')
    source_tree = ast.parse((ROOT/'scripts/run_landcover_native.py').read_text())
    declared = next(ast.literal_eval(node.value) for node in source_tree.body
                    if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id == 'SOURCES'
                                                          for target in node.targets))
    require(set(plan['source']) == set(declared) and SELF in plan['source'] and
            all(sha(inside(ROOT,name)) == value for name,value in plan['source'].items()),
            'Bound executable source differs')
    profile = plan['profile']
    encoded_profile = (json.dumps(profile,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(hashlib.sha256(encoded_profile).hexdigest() == plan['profile_sha'],'Embedded profile byte binding differs')
    require(profile['format'] == 'landcover-native-profile-v1' and profile['status'] == 'passed' and
            profile['source'] == plan['source'] and profile['runtime'] == plan['runtime'] and
            profile['base_channels'] == plan['base_channels'] and profile['batch_size'] == plan['batch_size'] and
            profile['warmup_steps'] == 1 and profile['steps'] == 2 and profile['actual_optimizer_steps'] == 3 and
            set(profile['arms']) == set(CONTROLS),'Profile binding differs')
    status = read(root/'status.json')
    require(status['status'] == 'complete' and status['plan_sha'] == plan_sha and status['fits'] == 13 and
            status['test_predictions'] == 0 and status['all_fits_sha'] == sha(root/'all_fits.json'),'Incomplete/changed campaign')
    fits = read(root/'all_fits.json');selection = read(root/'rate_selection.json')
    require(len(fits) == 13 and len({identity(row) for row in fits}) == 13,'Incomplete or duplicate fit grid')
    candidates = [row for row in fits if row['control'] == 'compact_affine' and row['seed'] == 42]
    require(len(candidates) == 2 and [row['rate'] for row in candidates] == [.001,.003],'Affine tuning candidates differ')
    chosen = max(candidates,key=lambda row:(row['best_miou'],-row['rate']))['rate']
    require(selection == dict(rate=chosen,source='seed42 compact_affine validation only',
        candidates=[{key:row[key] for key in ('rate','best_miou','best_epoch','checkpoint_sha')} for row in candidates],
        plan_sha=plan_sha),'Common-rate selection/tie/binding differs')
    order = [('compact_affine',42,.001),('compact_affine',42,.003)] + [(control,seed,chosen)
        for seed in SEEDS for control in CONTROLS if (control,seed) != ('compact_affine',42)]
    require([identity(row) for row in fits] == order,'Actual fit order/grid differs')
    for row in fits:
        directory = root/fit_name(row)
        require(read(directory/'result.json') == row and read(directory/'status.json') ==
            dict(status='complete',result_sha=sha(directory/'result.json')),'Fit record/status binding differs')
        require(row['status'] == 'complete' and row['source'] == plan['source'] and row['plan_sha'] == plan_sha and
            row['checkpoint_sha'] == sha(directory/'best.pt') and row['history_sha'] == sha(directory/'history.json') and
            row['history'] == read(directory/'history.json'),'Raw fit/history/checkpoint binding differs')
        require(row['training_history_sha'] == sha(directory/'training_history.json') and
                row['training_history'] == read(directory/'training_history.json'),'Full training history binding differs')
        training = row['training_history']
        require([point['epoch'] for point in training] == list(range(61)),'Full training epoch coverage differs')
        for point in training:
            require(point['optimizer_steps'] == point['epoch']*256//plan['batch_size'],'Full training step coverage differs')
            require(point['train_loss'] is None if point['epoch'] == 0 else
                    type(point['train_loss']) in (int,float) and point['train_loss'] >= 0,'Full training loss differs')
        require(row['epochs'] == 60 and row['optimizer_steps'] == plan['training_steps_per_fit'] and
            len(row['schedule_hashes']) == 60 and all(isinstance(v,str) and len(v) == 64 for v in row['schedule_hashes']),
            'Incomplete optimizer/schedule coverage')
        history = row['history']
        require([point['epoch'] for point in history] == list(range(0,61,5)),'Validation epoch coverage differs')
        for point in history:
            require(point['optimizer_steps'] == point['epoch']*256//plan['batch_size'],'Recorded optimizer-step schedule differs')
            require(point['train_loss'] is None if point['epoch'] == 0 else
                    type(point['train_loss']) in (int,float) and point['train_loss'] >= 0,'Invalid recorded training loss')
            check_metrics(point['validation'],plan)
            require(point['train_loss'] == training[point['epoch']]['train_loss'],
                    'Training and validation histories disagree at a recorded epoch')
        best = max(history,key=lambda point:point['validation']['miou'])
        require(row['best_epoch'] == best['epoch'] and row['best_miou'] == best['validation']['miou'],
                'Checkpoint not earliest strict best validation mIoU')
        require(row['model_config'] == dict(format='hq-landcover-native-v1',control=row['control'],seed=row['seed'],
            base_channels=64,image_size=128,in_channels=3,out_channels=5,circuit_edge_order='brickwork'),'Native model config differs')
        counts = row['counts']
        require(counts == profile['arms'][row['control']]['counts'],'Fit/profile parameter counts differ')
        if row['control'] in ('quantum','compact_affine'):
            require(row['first_gradients']['core_l2'] > 0 and row['core_change_l2'] > 0,'Active core did not train')
        else:
            require(row['first_gradients']['core_l2'] == row['core_change_l2'] == 0,'Frozen/inactive core changed')
        require(row['first_gradients']['common_l2'] > 0,'No observed common training gradient')
    primary = [row for row in fits if row['rate'] == chosen]
    require(len(primary) == 12,'Primary grid incomplete')
    for seed in SEEDS:
        rows = [row for row in primary if row['seed'] == seed]
        require(len({row['initial_common_sha'] for row in rows}) == 1 and
                all(row['schedule_hashes'] == rows[0]['schedule_hashes'] for row in rows),'Initialization/training schedule not paired')
    if dataset is not None:
        data = Path(dataset).resolve()
        require(sha(data/'manifest.json') == plan['dataset_manifest_sha'],'Dataset manifest differs')
        manifest = read(data/'manifest.json')
        require(manifest['format'] == 'landcover-ai-native-v1' and manifest['class_ids'] == list(range(5)) and
                manifest['archive']['md5'] == '3268c89070e8734b4e91d531c0617e03','Native dataset version/classes differ')
        require(manifest['test_pixel_windows_read'] == manifest['test_tiles_written'] == 0,'Unexpected test exposure')
        require(set(plan['train_ids']) <= set(manifest['splits']['train']) and set(plan['val_ids']) <= set(manifest['splits']['val'])
                and not set(plan['train_ids']+plan['val_ids']).intersection(manifest['splits']['test']),'Selected official split differs')
        for name in plan['train_ids']+plan['val_ids']:
            item = manifest['tiles'][name]
            for key in ('image','mask'):
                require(sha(inside(data,item[key])) == item[key+'_sha256'],'Selected prepared data bytes differ')
        require([sum(manifest['tiles'][name]['class_pixel_counts'][i] for name in plan['val_ids']) for i in range(5)] ==
                plan['val_class_pixels'],'Selected validation class counts differ')
    return dict(root=root,plan=plan,plan_sha=plan_sha,fits=fits,primary=primary,selection=selection,
                inventory=inventory(root))


def strict_restore(path,row,audit,device='cpu'):
    """Reconstruct factory trainability then strictly restore finite weights/BN."""
    import torch
    from utilsTorch import hq_landcover as native
    from scripts import run_landcover_native as run
    require(sha(path) == row['checkpoint_sha'],'Checkpoint bytes changed before restore')
    payload = torch.load(path,map_location='cpu',weights_only=True)
    require(payload['format'] == 'landcover-native-checkpoint-v1' and payload['model_config'] == row['model_config'] and
            payload['source'] == audit['plan']['source'] and payload['plan_sha'] == audit['plan_sha'],
            'Native checkpoint configuration/source/plan differs')
    best = next(point for point in row['history'] if point['epoch'] == row['best_epoch'])
    require(payload['epoch'] == row['best_epoch'] and payload['optimizer_steps'] == best['optimizer_steps'] and
            payload['validation'] == best['validation'],'Checkpoint selected epoch/metrics differ')
    model = native.build_from_config(payload['model_config'])
    require(native.parameter_counts(model) == row['counts'] and native.shared_state_sha256(model) == row['initial_common_sha'],
            'Factory parameter counts/initial state differ')
    model.load_state_dict(payload['state_dict'],strict=True)
    require(all(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()),
            'Nonfinite restored weights/BN buffers')
    if row['best_epoch'] == 0:
        initial = native.build_from_config(payload['model_config'])
        require(run.state_sha(initial.state_dict()) == run.state_sha(model.state_dict()),'Epoch-zero checkpoint differs from initialization')
        del initial
    model.to(device)
    return model,dict(control=row['control'],seed=row['seed'],checkpoint_sha=row['checkpoint_sha'],
        selected_state_sha=run.state_sha(model.state_dict()),counts=native.parameter_counts(model),strict_restore=True,
        trainability={name:parameter.requires_grad for name,parameter in model.named_parameters()})


def audit_checkpoints(audit):
    result = []
    for row in audit['primary']:
        model,record = strict_restore(audit['root']/fit_name(row)/'best.pt',row,audit)
        result.append(record);del model
    return result


def replay_binding(row,audit):
    return dict(plan_sha=audit['plan_sha'],all_fits_sha=sha(audit['root']/'all_fits.json'),
        result_sha=sha(audit['root']/fit_name(row)/'result.json'),checkpoint_sha=row['checkpoint_sha'],
        history_sha=row['history_sha'],control=row['control'],seed=row['seed'],rate=row['rate'],
        dataset_manifest_sha=audit['plan']['dataset_manifest_sha'],val_ids=audit['plan']['val_ids'],
        examples=audit['plan']['validation_examples_per_evaluation'])


def validate_record(payload,row,audit):
    require(payload['binding'] == replay_binding(row,audit),'Replay binding differs')
    observed = payload['validation'];check_metrics(observed,audit['plan'])
    expected = next(point['validation'] for point in row['history'] if point['epoch'] == row['best_epoch'])
    require(observed['confusion'] == expected['confusion'],'Replay confusion differs from original selected checkpoint')
    return observed


@contextmanager
def deadline_guard(deadline):
    require(math.isfinite(deadline) and time.time() < deadline,'Persisted replay deadline exhausted')
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(*args):raise TimeoutError('Persisted replay deadline exhausted')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        signal.setitimer(signal.ITIMER_REAL,*previous_timer)


def validate_replay(replay_dir,audit):
    root = Path(replay_dir).resolve();cache = read(root/'complete.json')
    require(cache['format'] == 'landcover-native-validation-replays-v1' and cache['complete'] is True and
            cache['plan_sha'] == audit['plan_sha'] and cache['all_fits_sha'] == sha(audit['root']/'all_fits.json') and
            cache['test_predictions'] == 0,
            'Replay completion binding differs')
    require(not (root/'failure.json').exists(),'Failed replay evidence cannot be exported')
    actual = inventory(root);actual.pop('artifact_sha256.json',None)
    require(actual == read(root/'artifact_sha256.json'),'Replay bytes/inventory differ')
    plan = read(root/'plan.json')
    require(plan == cache['replay_plan'] and plan['format'] == 'landcover-native-replay-plan-v1' and
            plan['collector_sha'] == sha(ROOT/SELF) and plan['source'] == audit['plan']['source'] and
            plan['dataset_manifest_sha'] == audit['plan']['dataset_manifest_sha'] and plan['test_predictions'] == 0 and
            plan['plan_sha'] == audit['plan_sha'] and plan['downtime_counts'] is True and
            type(plan['budget_seconds']) in (int,float) and plan['budget_seconds'] > 0 and
            plan['deadline_unix'] == plan['started_unix']+plan['budget_seconds'],'Replay producer/source/deadline binding differs')
    checkpoints = read(root/'checkpoint_audit.json')
    require(len(checkpoints) == 12 and cache['checkpoint_audit_sha'] == sha(root/'checkpoint_audit.json'),
            'Pre-inference checkpoint audit incomplete')
    records = []
    runtime = None
    for row,checkpoint in zip(audit['primary'],checkpoints):
        name = fit_name(row);intent = read(root/'intents'/(name+'.json'));payload = read(root/'records'/(name+'.json'))
        require(intent == dict(binding=replay_binding(row,audit)),'Durable inference intent differs')
        require(checkpoint['control'] == row['control'] and checkpoint['seed'] == row['seed'] and
                checkpoint['checkpoint_sha'] == row['checkpoint_sha'] and checkpoint['counts'] == row['counts'] and
                checkpoint['strict_restore'] is True and payload['checkpoint'] == checkpoint,'Strict checkpoint audit binding differs')
        flags = checkpoint['trainability']
        core = [value for name,value in flags.items() if name.startswith('bottleneck.core.')]
        require(flags and core and all(type(value) is bool for value in flags.values()),'Checkpoint trainability inventory missing')
        require(all(core) if row['control'] in ('quantum','compact_affine') else not any(core),'Core trainability flags differ')
        if row['control'] == 'zero_bottleneck':
            require(not any(value for name,value in flags.items() if name.startswith('bottleneck.')),'Zero bottleneck reactivated')
        for key in ('python','executable','packages','threads','cuda'):
            require(payload['runtime'][key] == audit['plan']['runtime'][key],'Cached replay software runtime differs')
        if runtime is None:runtime = payload['runtime']
        require(payload['runtime'] == runtime,'Replay runtime changed between checkpoints')
        records.append(dict(control=row['control'],seed=row['seed'],rate=row['rate'],validation=validate_record(payload,row,audit)))
    require(cache['records'] == records and set(path.stem for path in (root/'records').glob('*.json')) ==
            {fit_name(row) for row in audit['primary']} and set(path.stem for path in (root/'intents').glob('*.json')) ==
            {fit_name(row) for row in audit['primary']},'Replay grid differs')
    return cache


def replay(campaign,dataset,output,device='cuda',budget_seconds=3600):
    entered = time.time()
    audit = audit_campaign(campaign,dataset);out = Path(output).resolve();data = Path(dataset).resolve()
    for source in (audit['root'],data,ROOT):
        require(out != source and not source.is_relative_to(out),'Replay output contains input/source')
    require(not out.is_relative_to(audit['root']) and not out.is_relative_to(data),'Replay output aliases input')
    require(type(budget_seconds) in (int,float) and math.isfinite(budget_seconds) and budget_seconds > 0,'Finite positive replay budget required')
    if (out/'complete.json').exists():
        cached = validate_replay(out,audit)
        require(cached['replay_plan']['device'] == device and cached['replay_plan']['budget_seconds'] == budget_seconds,
                'Completed replay invocation configuration differs')
        return cached
    require(not (out/'failure.json').exists(),'Failed replay attempt is sticky; no retry')
    require(sha(ROOT/SELF) == LOADED_SELF,'Collector source changed after import')
    out.mkdir(parents=True,exist_ok=True)
    lock_path = out/'.replay.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan_path = out/'plan.json'
            if plan_path.exists():
                replay_plan = read(plan_path)
                require(replay_plan['plan_sha'] == audit['plan_sha'] and replay_plan['collector_sha'] == LOADED_SELF and
                        replay_plan['budget_seconds'] == budget_seconds and replay_plan['device'] == device,
                        'Persisted replay configuration differs')
            else:
                require(set(inventory(out)) == {'.replay.lock'},'Unknown/partial replay output cannot be replaced')
                started = entered
                replay_plan = dict(format='landcover-native-replay-plan-v1',plan_sha=audit['plan_sha'],
                    collector_sha=LOADED_SELF,source=audit['plan']['source'],dataset_manifest_sha=audit['plan']['dataset_manifest_sha'],
                    device=device,budget_seconds=budget_seconds,started_unix=started,deadline_unix=started+budget_seconds,
                    downtime_counts=True,test_predictions=0)
                write(plan_path,replay_plan)
            require(replay_plan['deadline_unix'] == replay_plan['started_unix']+replay_plan['budget_seconds'] and
                    replay_plan['downtime_counts'] is True,'Persisted deadline differs')
            with deadline_guard(replay_plan['deadline_unix']):
                from scripts import run_landcover_native as run
                from tools.landcover_data import NativeLandCoverDataset
                target_device = run.setup(device)
                actual_runtime = run.runtime(target_device)
                expected_runtime = audit['plan']['runtime']
                for key in ('python','executable','packages','threads','cuda'):
                    require(actual_runtime[key] == expected_runtime[key],'Replay software runtime differs: '+key)
                cp_path = out/'checkpoint_audit.json'
                if cp_path.exists():
                    expected_audit = read(cp_path)
                    require(len(expected_audit) == 12,'Partial checkpoint audit')
                    # Restoring every selected checkpoint precedes every newly
                    # inferred record, including continuation of a complete cell.
                    require(audit_checkpoints(audit) == expected_audit,'Checkpoint audit changed')
                else:write(cp_path,audit_checkpoints(audit))
                for row in audit['primary']:
                    name = fit_name(row);binding = replay_binding(row,audit)
                    intent = out/'intents'/(name+'.json');record = out/'records'/(name+'.json')
                    if intent.exists():
                        require(record.exists(),'Unmatched replay intent blocks repeated inference')
                        require(read(intent) == dict(binding=binding),'Existing replay intent differs')
                        validate_record(read(record),row,audit);continue
                    require(not record.exists(),'Replay result has no original intent')
                    write(intent,dict(binding=binding))
                    model,restore = strict_restore(audit['root']/name/'best.pt',row,audit,device=target_device)
                    validation_data = NativeLandCoverDataset(data,'val',tile_ids=audit['plan']['val_ids'],seed=row['seed'])
                    observed = run.evaluate(model,validation_data,audit['plan']['batch_size'],target_device,replay_plan['deadline_unix'])
                    payload = dict(binding=binding,validation=observed,checkpoint=restore,runtime=actual_runtime)
                    validate_record(payload,row,audit);write(record,payload)
                    del model,validation_data
                require(audit_campaign(campaign,dataset)['inventory'] == audit['inventory'],'Campaign changed during replay')
                records = [dict(control=row['control'],seed=row['seed'],rate=row['rate'],
                    validation=read(out/'records'/(fit_name(row)+'.json'))['validation']) for row in audit['primary']]
                complete = dict(format='landcover-native-validation-replays-v1',complete=True,plan_sha=audit['plan_sha'],
                    all_fits_sha=sha(audit['root']/'all_fits.json'),checkpoint_audit_sha=sha(cp_path),
                    replay_plan=replay_plan,records=records,test_predictions=0)
                write(out/'complete.json',complete);write(out/'artifact_sha256.json',inventory(out))
                return validate_replay(out,audit)
        except BaseException as error:
            if not (out/'failure.json').exists():write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),sticky=True))
            raise


def learning_quality(primary):
    require(len(primary) == 12,'Training quality requires all twelve primary fits')
    no_gain = sum(row['best_miou'] <= row['history'][0]['validation']['miou'] for row in primary)
    late = sum(row['best_epoch'] >= 55 for row in primary)
    return dict(initial_to_best_nonpositive_count=no_gain,all_arms_no_learning=no_gain == 12,
        late_best_count=late,horizons_unresolved=late > 0,
        training_quality='no_observed_learning' if no_gain == 12 else 'learning_observed')


def summary(records,primary=None):
    by_key = {(row['control'],row['seed']):row['validation'] for row in records}
    summaries,paired = [],[]
    for control in CONTROLS:
        row = dict(control=control)
        for metric in ('miou','accuracy','foreground_miou'):
            values = [by_key[control,seed][metric] for seed in SEEDS]
            row[metric+'_mean'] = statistics.mean(values);row[metric+'_sd'] = statistics.stdev(values)
        summaries.append(row)
    for control in ('compact_affine','frozen_quantum','zero_bottleneck'):
        values = [100*(by_key['quantum',seed]['miou']-by_key[control,seed]['miou']) for seed in SEEDS]
        paired.append(dict(reference=control,metric='miou',seed42_pp=values[0],seed77_pp=values[1],seed123_pp=values[2],
            mean_pp=statistics.mean(values),sample_sd_pp=statistics.stdev(values),positive_seeds=sum(value > 0 for value in values)))
    by_reference = {row['reference']:row for row in paired}
    def exact_miou(row):
        matrix = row['confusion']
        return sum((Fraction(matrix[i][i],sum(matrix[i])+sum(matrix[j][i] for j in range(5))-matrix[i][i])
                    if sum(matrix[i])+sum(matrix[j][i] for j in range(5))-matrix[i][i] else Fraction(0)
                    for i in range(5)),Fraction(0))/5
    exact = {control:[exact_miou(by_key['quantum',seed])-exact_miou(by_key[control,seed]) for seed in SEEDS]
             for control in ('compact_affine','frozen_quantum','zero_bottleneck')}
    practical = all(sum(exact[control])/3 >= Fraction(1,100) and all(value > 0 for value in exact[control])
                    for control in ('compact_affine','frozen_quantum')) and sum(exact['zero_bottleneck']) >= 0
    for row in paired:
        row['calibration_seed42_pp'] = row['seed42_pp']
        row['followup_seeds77_123_mean_pp'] = statistics.mean((row['seed77_pp'],row['seed123_pp']))
        row['followup_seeds77_123_sample_sd_pp'] = statistics.stdev((row['seed77_pp'],row['seed123_pp']))
    result = dict(arms=summaries,paired=paired,practical_screen_passed=practical,
        screening_scope='Descriptive fixed practical screen, not significance, equivalence or quantum advantage')
    if primary is not None:result.update(learning_quality(primary))
    return result


def csv_file(path,rows,fields=None):
    fields = fields or list(rows[0])
    with path.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def export(campaign,replay_dir,output):
    require(sha(ROOT/SELF) == LOADED_SELF,'Collector/export source changed after import')
    audit = audit_campaign(campaign);cache = validate_replay(replay_dir,audit)
    out = Path(output).resolve();replay_root = Path(replay_dir).resolve()
    for source in (audit['root'],replay_root,ROOT):
        require(out != source and not source.is_relative_to(out),'Export output contains input/source')
    require(not out.is_relative_to(audit['root']) and not out.is_relative_to(replay_root),'Export aliases input')
    require(not out.exists() or not any(out.iterdir()),'Export output must be new or empty')
    before = inventory(replay_root);out.mkdir(parents=True,exist_ok=True)
    result = summary(cache['records'],audit['primary']);write(out/'summary.json',result)
    per_seed,class_rows,curves,training_curves,costs,learning = [],[],[],[],[],[]
    for row in audit['primary']:
        observed = next(record['validation'] for record in cache['records'] if
                        (record['control'],record['seed']) == (row['control'],row['seed']))
        best = next(point for point in row['history'] if point['epoch'] == row['best_epoch'])
        base = dict(control=row['control'],seed=row['seed'],rate=row['rate'])
        per_seed.append(dict(**base,best_epoch=row['best_epoch'],miou=observed['miou'],accuracy=observed['accuracy'],
            foreground_miou=observed['foreground_miou'],examples=observed['examples'],pixels=observed['pixels'],
            checkpoint_sha=row['checkpoint_sha'],optimizer_steps=row['optimizer_steps']))
        for i,name in enumerate(CLASSES):
            confusion = observed['confusion'];tp = confusion[i][i]
            class_rows.append(dict(**base,class_id=i,class_name=name,iou=observed['class_iou'][name],
                truth_pixels=observed['truth_pixels'][i],prediction_pixels=observed['prediction_pixels'][i],
                tp=tp,fp=observed['prediction_pixels'][i]-tp,fn=observed['truth_pixels'][i]-tp))
        for point in row['history']:
            curves.append(dict(**base,epoch=point['epoch'],optimizer_steps=point['optimizer_steps'],
                train_loss=point['train_loss'],validation_loss=point['validation']['loss'],
                miou=point['validation']['miou'],accuracy=point['validation']['accuracy'],
                foreground_miou=point['validation']['foreground_miou'],selected=point['epoch'] == row['best_epoch']))
        for point in row['training_history']:
            training_curves.append(dict(**base,epoch=point['epoch'],optimizer_steps=point['optimizer_steps'],train_loss=point['train_loss']))
        initial,last = row['history'][0]['validation']['miou'],row['history'][-1]['validation']['miou']
        learning.append(dict(**base,initial_miou=initial,best_miou=best['validation']['miou'],final_miou=last,
            initial_to_best_pp=100*(best['validation']['miou']-initial),best_epoch=row['best_epoch'],
            best_at_cap=row['best_epoch'] == audit['plan']['epochs'],best_at_or_after_55=row['best_epoch'] >= 55,
            last_interval_improvement_pp=100*(last-row['history'][-2]['validation']['miou'])))
        costs.append(dict(**base,elapsed_seconds=row['elapsed_seconds'],peak_cuda_allocated=row['peak_cuda_allocated'],**row['counts']))
    candidates = [dict(control=row['control'],seed=row['seed'],rate=row['rate'],best_miou=row['best_miou'],
        best_epoch=row['best_epoch'],selected_common_rate=row['rate'] == audit['selection']['rate'])
        for row in audit['fits'] if row['control'] == 'compact_affine' and row['seed'] == 42]
    for name,rows in (('per_seed',per_seed),('class_metrics',class_rows),('paired',result['paired']),
        ('arm_summary',result['arms']),('curves',curves),('training_curves',training_curves),
        ('fit_costs',costs),('learning',learning),('rate_candidates',candidates)):
        csv_file(out/(name+'.csv'),rows)
    profile_rows = [dict(control=control,warmup_seconds=arm['warmup_seconds'],
        step1_seconds=arm['step_seconds'][0],step2_seconds=arm['step_seconds'][1],
        eval_batch_seconds=arm['eval_batch_seconds'],peak_cuda_allocated=arm['peak_cuda_allocated'],
        peak_cuda_reserved=arm['peak_cuda_reserved'],**arm['counts']) for control,arm in audit['plan']['profile']['arms'].items()]
    csv_file(out/'profile_costs.csv',profile_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    colors = dict(zip(CONTROLS,('#CC79A7','#0072B2','#E69F00','#009E73')))
    fig,axes = plt.subplots(1,2,figsize=(7.2,3.7))
    for control in CONTROLS:
        rows = [row for row in audit['primary'] if row['control'] == control]
        epochs = [point['epoch'] for point in rows[0]['history']]
        for column,metric in enumerate(('train_loss','miou')):
            x = list(range(1,61)) if metric == 'train_loss' else epochs
            values = np.asarray([[point['train_loss'] for point in row['training_history'][1:]] if metric == 'train_loss'
                                 else [point['validation'][metric] for point in row['history']] for row in rows])
            mean,sd = values.mean(0),values.std(0,ddof=1)
            axes[column].plot(x,mean,color=colors[control],label=LABELS[control],linewidth=1.2)
            axes[column].fill_between(x,mean-sd,mean+sd,color=colors[control],alpha=.15)
    for column in range(2):
        axes[column].set(xlabel='Epoch',ylabel='Training loss (each epoch)' if column == 0 else 'Validation mIoU (all 5 classes)',
                         xlim=(-1,61));axes[column].grid(alpha=.2)
    handles,labels = axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7)
    fig.suptitle(f"Native LandCover.ai local pilot: common learning rate {audit['selection']['rate']:g}",fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.95))
    fig.savefig(out/'learning_curves.pdf',metadata={'CreationDate':None,'ModDate':None,'Creator':'Native LandCover collector'})
    fig.savefig(out/'learning_curves.png',dpi=180,metadata={'Software':'Native LandCover collector'});plt.close(fig)
    passed = result['practical_screen_passed']
    lines = ['# Native LandCover.ai local comparison','',
        'This centralized, noiseless simulator pilot uses 256 official training tiles, 64 validation tiles, '
        'three training seeds, 60 epochs, RGB inputs and five classes including background. All 1,024 '
        'validation crops are pooled. The common learning rate was chosen using only two affine seed 42 fits; '
        'the campaign contains 13 fits and 12 primary comparisons. No test images were evaluated.','',
        '| Control | Validation mIoU mean ± sample SD | Accuracy mean ± sample SD |',
        '|---|---:|---:|']
    for row in result['arms']:
        lines.append(f"| {LABELS[row['control']]} | {row['miou_mean']:.4f} ± {row['miou_sd']:.4f} | {row['accuracy_mean']:.4f} ± {row['accuracy_sd']:.4f} |")
    lines += ['','| Control | Total / trainable parameters | Core total / trainable | Profile step 1 / 2 (s) | Mean fit elapsed (s) |',
        '|---|---:|---:|---:|---:|']
    for control in CONTROLS:
        arm = audit['plan']['profile']['arms'][control];counts = arm['counts']
        elapsed = statistics.mean(row['elapsed_seconds'] for row in audit['primary'] if row['control'] == control)
        lines.append(f"| {LABELS[control]} | {counts['total']:,} / {counts['trainable']:,} | "
            f"{counts['core_total']:,} / {counts['core_trainable']:,} | "
            f"{arm['step_seconds'][0]:.3f} / {arm['step_seconds'][1]:.3f} | {elapsed:.1f} |")
    lines += ['','Quantum minus reference, paired percentage points (mean ± sample SD):']
    for row in result['paired']:
        lines.append(f"- {LABELS[row['reference']]}: {row['mean_pp']:+.3f} ± {row['sample_sd_pp']:.3f}; positive in {row['positive_seeds']}/3 seeds.")
    lines += ['','Seed 42 also calibrated the common rate through the affine candidates. '
        'Its contrast is separate from the mean and sample SD of follow-up seeds 77 and 123.','',
        '| Quantum minus reference | Calibration seed 42 (pp) | Follow-up seeds 77/123 mean ± sample SD (pp) |',
        '|---|---:|---:|']
    for row in result['paired']:
        lines.append(f"| {LABELS[row['reference']]} | {row['calibration_seed42_pp']:+.3f} | "
            f"{row['followup_seeds77_123_mean_pp']:+.3f} ± {row['followup_seeds77_123_sample_sd_pp']:.3f} |")
    lines += ['',f"The fixed practical screen {'passed' if passed else 'did not pass'}. It requires at least +1 percentage point "
        'mean quantum improvement over both affine and frozen controls, positive in every seed for each, '
        'and nonnegative mean improvement over zero. This is a descriptive screening rule, not a significance '
        'test, an equivalence result or quantum advantage.','',
        f"Selected checkpoints at epoch 60: {sum(row['best_at_cap'] for row in learning)}/12. "
        f"Selected checkpoints at or after epoch 55: {sum(row['best_at_or_after_55'] for row in learning)}/12. "
        f"Positive final-interval validation changes: {sum(row['last_interval_improvement_pp'] > 0 for row in learning)}/12. "
        'The learning table retains epoch-zero to best changes and final interval behavior; a completed budget '
        'does not establish convergence. Late best epochs leave convergence unresolved.','',
        f"Training quality: {result['training_quality']}. Fits with no epoch-zero-to-best gain: "
        f"{result['initial_to_best_nonpositive_count']}/12; all arms showed no learning: {result['all_arms_no_learning']}. "
        f"Late best epochs: {result['late_best_count']}/12; horizon unresolved: {result['horizons_unresolved']}. "
        'A failed screen is not a refutation of the paper. No observed learning across the arms is an inconclusive '
        'optimization result, and any late best epoch leaves the training horizon unresolved.','',
        'Curves show means and sample standard deviations over the same three training seeds at the common '
        'selected rate. Training loss retains every epoch; validation is recorded every five epochs. '
        'Validation was reused for rate and checkpoint selection. The cohort is not a scene-disjoint holdout. '
        'This independent implementation does not establish the paper’s full-test score, general quantum '
        'superiority, quantum hardware benefit or Anvil performance. No raw imagery or weights are exported.','',
        'IoU = TP/(TP+FP+FN), with zero for an empty union. Primary mIoU averages all five classes including '
        'background; foreground mIoU is secondary. Accuracy is total diagonal counts divided by total pixels. '
        'Reported trainable counts describe requires_grad flags, not functional influence. Profile costs use '
        'synthetic batches and are distinct from full fit elapsed costs.']
    lines += ['','| Control | Epoch-zero to best mIoU gain mean ± sample SD (pp) | Best epochs (42 / 77 / 123) |',
        '|---|---:|---|']
    for control in CONTROLS:
        rows = sorted((row for row in learning if row['control'] == control),key=lambda row:SEEDS.index(row['seed']))
        gains = [row['initial_to_best_pp'] for row in rows]
        lines.append(f"| {LABELS[control]} | {statistics.mean(gains):+.3f} ± {statistics.stdev(gains):.3f} | "
            + ' / '.join(str(row['best_epoch']) for row in rows)+' |')
    lines += ['',f"All twelve fits lacked an epoch-zero-to-best gain: {all(row['initial_to_best_pp'] <= 0 for row in learning)}. "
        'This quality flag describes the observed optimization and does not change the frozen screening rule.']
    (out/'INTERPRETATION.md').write_text('\n'.join(lines)+'\n')
    (out/'collector_source.py.txt').write_bytes((ROOT/SELF).read_bytes())
    write(out/'provenance.json',dict(format='landcover-native-export-v1',collector_sha=LOADED_SELF,
        plan_sha=audit['plan_sha'],all_fits_sha=sha(audit['root']/'all_fits.json'),replay_complete_sha=sha(replay_root/'complete.json'),
        replay_inventory_sha=sha(replay_root/'artifact_sha256.json'),source=audit['plan']['source'],
        numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,inference_performed=False,test_predictions=0))
    require(inventory(replay_root) == before and audit_campaign(campaign)['inventory'] == audit['inventory'],
            'Input artifacts changed during export')
    require(sha(ROOT/SELF) == LOADED_SELF,'Export source changed during rendering')
    write(out/'artifact_sha256.json',inventory(out))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__);commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('replay');p.add_argument('--campaign',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--budget-seconds',type=float,default=3600)
    p = commands.add_parser('export');p.add_argument('--campaign',required=True);p.add_argument('--replay',required=True)
    p.add_argument('--output',required=True)
    args = parser.parse_args(argv)
    if args.command == 'replay':replay(args.campaign,args.dataset,args.output,args.device,args.budget_seconds)
    else:export(args.campaign,args.replay,args.output)
    print('Native LandCover phase completed:',args.command,args.output)


if __name__ == '__main__':main()
