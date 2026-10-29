"""Synthetic software evidence; no LandCover fitting or image inference."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import collect_landcover_native as collect


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def confusion(error):
    truth = [3355444,3355443,3355443,3355443,3355443]
    matrix = [[0]*5 for _ in range(5)]
    for i,n in enumerate(truth):
        wrong = int(n*error);matrix[i][i] = n-wrong;matrix[i][(i+1)%5] = wrong
    return matrix


def expected_metrics(matrix):
    # An independent hand-written diagonal/union calculation for fixture data.
    tp = [matrix[i][i] for i in range(5)]
    truth = [sum(row) for row in matrix]
    predicted = [sum(matrix[i][j] for i in range(5)) for j in range(5)]
    values = [tp[i]/(truth[i]+predicted[i]-tp[i]) if truth[i]+predicted[i]-tp[i] else 0 for i in range(5)]
    return dict(confusion=matrix,pixels=sum(truth),truth_pixels=truth,prediction_pixels=predicted,
        class_iou=dict(zip(collect.CLASSES,values)),absent_union=[],miou=sum(values)/5,
        foreground_miou=sum(values[1:])/4,accuracy=sum(tp)/sum(truth),loss=.4,examples=1024)


def refresh_campaign(root):
    rows = collect.read(root/'all_fits.json')
    for row in rows:
        folder = root/collect.fit_name(row)
        put(folder/'result.json',row)
        put(folder/'status.json',dict(status='complete',result_sha=collect.sha(folder/'result.json')))
    status = collect.read(root/'status.json');status['all_fits_sha'] = collect.sha(root/'all_fits.json')
    put(root/'status.json',status)


@pytest.fixture
def campaign(tmp_path):
    from scripts import run_landcover_native as run
    import torch
    root,data = tmp_path/'campaign',tmp_path/'dataset';root.mkdir();data.mkdir()
    source = run.sources()
    runtime = run.runtime(torch.device('cpu'))
    counts = {}
    for control in collect.CONTROLS:
        active = control not in ('frozen_quantum','zero_bottleneck')
        core = 1568 if control == 'compact_affine' else 8
        counts[control] = dict(total=8154275+core,trainable=8154275+core if active else
            (8154275 if control == 'frozen_quantum' else 5988512),core_total=core,core_trainable=core if active else 0,
            bottleneck_total=2165763+core,bottleneck_trainable=2165763+core if active else
            (2165763 if control == 'frozen_quantum' else 0))
    profile = dict(format='landcover-native-profile-v1',status='passed',source=source,runtime=runtime,
        base_channels=64,batch_size=4,warmup_steps=1,steps=2,actual_optimizer_steps=3,
        arms={control:dict(counts=counts[control],warmup_seconds=.2,step_seconds=[.1,.1],
            eval_batch_seconds=.01,peak_cuda_allocated=0,peak_cuda_reserved=0) for control in collect.CONTROLS})
    profile_bytes = (json.dumps(profile,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    image,mask = data/'image.png',data/'mask.png';image.write_bytes(b'synthetic identity only');mask.write_bytes(b'no actual image inference')
    train_ids = [f'train{i}' for i in range(256)];val_ids = [f'val{i}' for i in range(64)]
    selected = train_ids+val_ids
    class_counts = [52429,52429,52429,52429,52428]
    rows = {name:dict(split='train' if name in train_ids else 'val',image='image.png',mask='mask.png',
        image_sha256=collect.sha(image),mask_sha256=collect.sha(mask),shape=[512,512],class_pixel_counts=class_counts)
        for name in selected}
    manifest = dict(format='landcover-ai-native-v1',class_ids=list(range(5)),test_pixel_windows_read=0,test_tiles_written=0,
        archive={'md5':'3268c89070e8734b4e91d531c0617e03'},splits=dict(train=train_ids,val=val_ids,test=['unopened_test']),tiles=rows)
    put(data/'manifest.json',manifest)
    truth = [64*value for value in class_counts]
    plan = dict(format='landcover-native-plan-v1',test_predictions=0,source=source,profile=profile,
        profile_sha=collect.hashlib.sha256(profile_bytes).hexdigest(),runtime=runtime,controls=list(collect.CONTROLS),seeds=list(collect.SEEDS),
        rate_candidates=[.001,.003],image_size=128,base_channels=64,batch_size=4,epochs=60,evaluation_every=5,
        train_ids=train_ids,val_ids=val_ids,val_class_pixels=truth,training_steps_per_fit=3840,
        training_examples_per_fit=15360,validation_examples_per_evaluation=1024,dataset_manifest_sha=collect.sha(data/'manifest.json'))
    put(root/'plan.json',plan);plan_sha = collect.sha(root/'plan.json')
    order = [('compact_affine',42,.001),('compact_affine',42,.003)]+[(control,seed,.003)
        for seed in collect.SEEDS for control in collect.CONTROLS if (control,seed) != ('compact_affine',42)]
    fits = []
    for control,seed,rate in order:
        end_error = dict(compact_affine=.15,quantum=.12,frozen_quantum=.14,zero_bottleneck=.16)[control]
        if rate == .001:end_error += .05
        end_error += {42:0,77:.005,123:-.005}[seed]
        history = []
        for epoch in range(0,61,5):
            # Constant truth marginals across every crop/evaluation and arm.
            matrix = [[0]*5 for _ in range(5)]
            for i,n in enumerate(truth):
                wrong = int(n*(.7+(end_error-.7)*epoch/60));matrix[i][i] = n-wrong;matrix[i][(i+1)%5] = wrong
            history.append(dict(epoch=epoch,optimizer_steps=epoch*64,train_loss=None if epoch == 0 else 1/(epoch+1),
                validation=expected_metrics(matrix),elapsed_seconds=epoch))
        training = [dict(epoch=epoch,optimizer_steps=epoch*64,train_loss=None if epoch == 0 else 1/(epoch+1),
                        elapsed_seconds=epoch) for epoch in range(61)]
        model_config = dict(format='hq-landcover-native-v1',control=control,seed=seed,base_channels=64,image_size=128,
                            in_channels=3,out_channels=5,circuit_edge_order='brickwork')
        row = dict(status='complete',control=control,seed=seed,rate=rate,source=source,plan_sha=plan_sha,
            model_config=model_config,history=history,training_history=training,epochs=60,optimizer_steps=3840,
            schedule_hashes=[f'{seed:064x}']*60,initial_common_sha=f'{seed:064x}',counts=counts[control],
            best_epoch=60,best_miou=history[-1]['validation']['miou'],
            first_gradients=dict(core_l2=.1 if control in ('quantum','compact_affine') else 0,common_l2=.1),
            core_change_l2=.1 if control in ('quantum','compact_affine') else 0,
            elapsed_seconds=60.,peak_cuda_allocated=0,final_state_sha='f'*64)
        folder = root/collect.fit_name(row);folder.mkdir()
        (folder/'best.pt').write_bytes(f'fixture checkpoint boundary: {control} {seed} {rate}'.encode())
        put(folder/'history.json',history);put(folder/'training_history.json',training)
        row.update(checkpoint_sha=collect.sha(folder/'best.pt'),history_sha=collect.sha(folder/'history.json'),
                   training_history_sha=collect.sha(folder/'training_history.json'))
        fits.append(row)
    put(root/'all_fits.json',fits)
    put(root/'rate_selection.json',dict(rate=.003,source='seed42 compact_affine validation only',
        candidates=[{key:row[key] for key in ('rate','best_miou','best_epoch','checkpoint_sha')} for row in fits[:2]],plan_sha=plan_sha))
    put(root/'status.json',dict(status='complete',plan_sha=plan_sha,fits=13,test_predictions=0,
                              all_fits_sha=collect.sha(root/'all_fits.json')))
    refresh_campaign(root)
    return root,data


@pytest.fixture
def replayed(campaign,tmp_path,monkeypatch):
    from scripts import run_landcover_native as run
    from tools import landcover_data
    import torch
    root,data = campaign;out = tmp_path/'replay';events=[]
    def receipt(row):
        return dict(control=row['control'],seed=row['seed'],checkpoint_sha=row['checkpoint_sha'],counts=row['counts'],
            selected_state_sha='a'*64,strict_restore=True,trainability={'head.weight':True,
            'bottleneck.pre_projection.weight':row['control'] != 'zero_bottleneck',
            'bottleneck.core.weights':row['control'] in ('quantum','compact_affine')})
    def checkpoints(audit):
        events.append(('all12_prechecked',len(audit['primary'])))
        return [receipt(row) for row in audit['primary']]
    def restore(path,row,audit,device='cpu'):
        assert events and events[0] == ('all12_prechecked',12)
        assert (out/'intents'/(collect.fit_name(row)+'.json')).is_file()
        return SimpleNamespace(row=row),receipt(row)
    def dataset(path,split,tile_ids,seed):
        assert split == 'val' and tile_ids == collect.read(root/'plan.json')['val_ids']
        return SimpleNamespace(seed=seed)
    def evaluate(model,*args):
        events.append(('evaluate',model.row['control'],model.row['seed']))
        # Loss roundoff can differ while exact confusion-derived metrics agree.
        result = copy.deepcopy(model.row['history'][-1]['validation']);result['loss'] += 1e-8
        return result
    monkeypatch.setattr(collect,'audit_checkpoints',checkpoints)
    monkeypatch.setattr(collect,'strict_restore',restore)
    monkeypatch.setattr(run,'setup',lambda device:torch.device(device))
    monkeypatch.setattr(run,'evaluate',evaluate)
    monkeypatch.setattr(landcover_data,'NativeLandCoverDataset',dataset)
    cache = collect.replay(root,data,out,device='cpu',budget_seconds=120)
    return root,data,out,events,cache


def test_audit_complete_grid_histories_and_counts(campaign):
    root,data = campaign
    audit = collect.audit_campaign(root,data)
    assert len(audit['fits']) == 13 and len(audit['primary']) == 12
    assert audit['selection']['rate'] == .003
    assert all(len(row['training_history']) == 61 and len(row['history']) == 13 for row in audit['fits'])


@pytest.mark.parametrize('mutation',[
    lambda rows:rows.pop(),
    lambda rows:rows[3].update(optimizer_steps=1920),
    lambda rows:rows[3].update(best_epoch=55),
    lambda rows:rows[3]['schedule_hashes'].__setitem__(0,'a'*64),
    lambda rows:rows[3]['model_config'].update(out_channels=2),
    lambda rows:rows[3]['first_gradients'].update(core_l2=.1),
])
def test_internally_rehashed_bad_grid_steps_selection_pairing_configs_rejected(campaign,mutation):
    root,data = campaign;rows = collect.read(root/'all_fits.json');mutation(rows)
    put(root/'all_fits.json',rows);refresh_campaign(root)
    with pytest.raises(ValueError):collect.audit_campaign(root,data)


def test_training_history_incomplete_is_not_rescued_by_validation_rows(campaign):
    root,data = campaign;rows = collect.read(root/'all_fits.json');row = rows[2]
    row['training_history'].pop(2)
    path = root/collect.fit_name(row)/'training_history.json';put(path,row['training_history'])
    row['training_history_sha'] = collect.sha(path)
    put(root/'all_fits.json',rows);refresh_campaign(root)
    with pytest.raises(ValueError,match='Full training epoch'):collect.audit_campaign(root,data)


def test_dataset_byte_corruption_stops_before_loading_any_model(campaign,monkeypatch):
    root,data = campaign;(data/'image.png').write_bytes(b'changed')
    monkeypatch.setattr(collect,'audit_checkpoints',lambda *a:pytest.fail('must not load checkpoints'))
    with pytest.raises(ValueError,match='data bytes'):collect.replay(root,data,data.parent/'replay',device='cpu')


def test_once_replays_precheck_all_checkpoints_intents_and_readonly_completion(replayed,monkeypatch):
    root,data,out,events,cache = replayed
    assert events[0] == ('all12_prechecked',12) and len(events) == 13
    assert len(cache['records']) == 12
    before = {str(path):path.stat().st_mtime_ns for path in out.rglob('*') if path.is_file()}
    bytes_before = collect.inventory(out)
    monkeypatch.setattr(collect,'audit_checkpoints',lambda *a:pytest.fail('completed replay must not restore again'))
    # A completed run remains readable after its deadline, with no new inference.
    monkeypatch.setattr(collect.time,'time',lambda:10**12)
    assert collect.replay(root,data,out,device='cpu',budget_seconds=120) == cache
    assert collect.inventory(out) == bytes_before
    assert before == {str(path):path.stat().st_mtime_ns for path in out.rglob('*') if path.is_file()}


def test_sticky_failure_retains_intent_and_blocks_repeated_forward(campaign,tmp_path,monkeypatch):
    from scripts import run_landcover_native as run
    from tools import landcover_data
    import torch
    root,data = campaign;out=tmp_path/'failed';calls=[]
    monkeypatch.setattr(run,'setup',lambda device:torch.device(device))
    monkeypatch.setattr(collect,'audit_checkpoints',lambda audit:[{}]*12)
    monkeypatch.setattr(collect,'strict_restore',lambda *a,**k:(None,{}))
    monkeypatch.setattr(landcover_data,'NativeLandCoverDataset',lambda *a,**k:None)
    def fail(*args):calls.append('forward');raise RuntimeError('synthetic evaluation failure')
    monkeypatch.setattr(run,'evaluate',fail)
    with pytest.raises(RuntimeError,match='synthetic'):collect.replay(root,data,out,device='cpu',budget_seconds=120)
    assert len(list((out/'intents').glob('*.json'))) == 1 and not (out/'complete.json').exists()
    before=collect.inventory(out)
    with pytest.raises(ValueError,match='sticky'):collect.replay(root,data,out,device='cpu',budget_seconds=120)
    assert calls == ['forward'] and collect.inventory(out) == before


def test_replay_changed_count_rejected_even_when_other_metrics_recomputed(replayed):
    root,data,out,events,cache = replayed;audit = collect.audit_campaign(root)
    row=audit['primary'][0];payload=collect.read(out/'records'/(collect.fit_name(row)+'.json'))
    payload['validation']=copy.deepcopy(payload['validation'])
    matrix=payload['validation']['confusion'];matrix[0][0]-=1;matrix[0][1]+=1
    payload['validation']=expected_metrics(matrix)
    with pytest.raises(ValueError,match='Replay confusion'):collect.validate_record(payload,row,audit)


def test_export_is_deterministic_and_does_not_import_models_or_infer(replayed,tmp_path,monkeypatch):
    root,data,out,events,cache=replayed
    monkeypatch.setattr(collect,'strict_restore',lambda *a,**k:pytest.fail('export must not restore'))
    before=collect.inventory(out)
    first,second=tmp_path/'export1',tmp_path/'export2'
    collect.export(root,out,first);collect.export(root,out,second)
    assert collect.inventory(first) == collect.inventory(second)
    assert collect.inventory(out) == before and len(events) == 13
    assert len((first/'training_curves.csv').read_text().splitlines()) == 12*61+1
    assert len((first/'curves.csv').read_text().splitlines()) == 12*13+1
    assert len((first/'class_metrics.csv').read_text().splitlines()) == 12*5+1
    assert (first/'learning_curves.pdf').stat().st_size > 1000
    text=(first/'INTERPRETATION.md').read_text()
    assert 'three training seeds' in text and 'Seed 42 also calibrated' in text and 'not a significance' in text
    with pytest.raises(ValueError,match='new or empty'):collect.export(root,out,first)
    # A fresh interpreter traps every Torch/model import during real export.
    script = """import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name=='torch' or name.startswith('torch.') or name.startswith('utilsTorch') or name=='scripts.run_landcover_native':
  raise RuntimeError('forbidden model import '+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from scripts import collect_landcover_native as c
c.export(sys.argv[1],sys.argv[2],sys.argv[3])
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable,'-c',script,str(root),str(out),str(tmp_path/'subprocess_export')],check=True,cwd=collect.ROOT)


def test_confusion_metric_conventions_and_exact_practical_boundary():
    matrix=[[80 if i==j else (5 if j==(i+1)%5 else 0) for j in range(5)] for i in range(5)]
    result=collect.metric_counts(matrix)
    assert result['miou'] == pytest.approx(80/90,rel=0,abs=1e-15) and result['accuracy'] == 80/85
    assert result['class_iou']['Background'] == 80/90
    with pytest.raises(ValueError):collect.metric_counts([[1.]*5]*5)
    def row(correct):
        matrix=[[0]*5 for _ in range(5)]
        for i in range(5):matrix[i][i]=correct;matrix[i][(i+1)%5]=100-correct
        return collect.metric_counts(matrix)
    # IoU(100)=1, IoU(99)=99/101; exact improvements exceed .01.
    records=[dict(control=control,seed=seed,validation=row(100 if control=='quantum' else 99))
             for control in collect.CONTROLS for seed in collect.SEEDS]
    assert collect.summary(records)['practical_screen_passed'] is True
    for item in records:
        if item['control']=='frozen_quantum':item['validation']=row(100)
    assert collect.summary(records)['practical_screen_passed'] is False


def test_practical_gate_uses_exact_counts_at_one_percentage_point_boundary():
    scale=10**13
    perfect=[[0]*5 for _ in range(5)];reference=[[0]*5 for _ in range(5)]
    for i in range(5):perfect[i][i]=79*scale if i < 2 else 100*scale
    reference=copy.deepcopy(perfect)
    reference[0][0]=reference[1][1]=78*scale
    reference[0][1]=reference[1][0]=scale
    def records(matrix):
        return [dict(control=control,seed=seed,validation=collect.metric_counts(perfect if control in
            ('quantum','zero_bottleneck') else matrix)) for control in collect.CONTROLS for seed in collect.SEEDS]
    assert collect.summary(records(reference))['practical_screen_passed'] is True
    # One corrected reference pixel makes the exact improvement smaller than
    # 1 pp, even though its rounded binary64 aggregate remains unchanged.
    changed=copy.deepcopy(reference);changed[0][0]+=1
    assert collect.metric_counts(changed)['miou'] == collect.metric_counts(reference)['miou']
    assert collect.summary(records(changed))['practical_screen_passed'] is False


@pytest.mark.parametrize('control',['quantum','frozen_quantum','zero_bottleneck','compact_affine'])
def test_real_native_checkpoint_strict_restore_and_trainability(tmp_path,control):
    import torch
    from utilsTorch import hq_landcover as native
    model=native.build_model(control,seed=42,base_channels=1,image_size=64)
    validation={'fixture':'checkpoint metadata only'}
    payload=dict(format='landcover-native-checkpoint-v1',model_config=native.model_config(model),state_dict=model.state_dict(),
        source={'fixture':'bytes'},plan_sha='a'*64,epoch=0,optimizer_steps=0,validation=validation)
    path=tmp_path/'best.pt';torch.save(payload,path)
    row=dict(control=control,seed=42,model_config=payload['model_config'],counts=native.parameter_counts(model),
        initial_common_sha=native.shared_state_sha256(model),checkpoint_sha=collect.sha(path),best_epoch=0,
        history=[dict(epoch=0,optimizer_steps=0,validation=validation)])
    audit=dict(plan={'source':payload['source']},plan_sha=payload['plan_sha'])
    restored,receipt=collect.strict_restore(path,row,audit)
    assert receipt['strict_restore'] and native.shared_state_sha256(restored)==native.shared_state_sha256(model)
    assert {name:p.requires_grad for name,p in restored.named_parameters()} == {name:p.requires_grad for name,p in model.named_parameters()}
    payload['state_dict']=dict(payload['state_dict']);payload['state_dict'].pop('head.bias');torch.save(payload,path)
    row['checkpoint_sha']=collect.sha(path)
    with pytest.raises(RuntimeError,match='Missing key'):collect.strict_restore(path,row,audit)


def test_machine_readable_underlearning_and_late_horizon_flags():
    rows=[dict(best_miou=.2,best_epoch=0,history=[{'validation':{'miou':.2}}]) for _ in range(12)]
    quality=collect.learning_quality(rows)
    assert quality == dict(initial_to_best_nonpositive_count=12,all_arms_no_learning=True,
        late_best_count=0,horizons_unresolved=False,training_quality='no_observed_learning')
    rows[4].update(best_miou=.3,best_epoch=55)
    assert collect.learning_quality(rows) == dict(initial_to_best_nonpositive_count=11,all_arms_no_learning=False,
        late_best_count=1,horizons_unresolved=True,training_quality='learning_observed')


def test_exact_affine_tie_requires_lower_common_rate(campaign):
    root,data=campaign;rows=collect.read(root/'all_fits.json')
    rows[0]['history']=copy.deepcopy(rows[1]['history']);rows[0]['best_miou']=rows[1]['best_miou']
    path=root/collect.fit_name(rows[0])/'history.json';put(path,rows[0]['history']);rows[0]['history_sha']=collect.sha(path)
    put(root/'all_fits.json',rows);refresh_campaign(root)
    with pytest.raises(ValueError,match='Common-rate selection'):collect.audit_campaign(root,data)


def test_orphan_intent_blocks_inference_without_overwrite(campaign,tmp_path,monkeypatch):
    from scripts import run_landcover_native as run
    import torch
    root,data=campaign;audit=collect.audit_campaign(root,data);out=tmp_path/'orphan';out.mkdir()
    started=time.time();plan=dict(format='landcover-native-replay-plan-v1',plan_sha=audit['plan_sha'],
        collector_sha=collect.LOADED_SELF,source=audit['plan']['source'],dataset_manifest_sha=audit['plan']['dataset_manifest_sha'],
        device='cpu',budget_seconds=120,started_unix=started,deadline_unix=started+120,downtime_counts=True,test_predictions=0)
    put(out/'plan.json',plan)
    first=audit['primary'][0];intent=out/'intents'/(collect.fit_name(first)+'.json')
    put(intent,dict(binding=collect.replay_binding(first,audit)));before=intent.read_bytes()
    monkeypatch.setattr(run,'setup',lambda device:torch.device(device))
    monkeypatch.setattr(collect,'audit_checkpoints',lambda audit:[{}]*12)
    monkeypatch.setattr(run,'evaluate',lambda *a:pytest.fail('must not repeat inference'))
    with pytest.raises(ValueError,match='Unmatched replay intent'):
        collect.replay(root,data,out,device='cpu',budget_seconds=120)
    assert intent.read_bytes()==before and collect.read(out/'failure.json')['sticky'] is True


def test_persisted_deadline_counts_downtime_before_restoration(campaign,tmp_path,monkeypatch):
    root,data=campaign;audit=collect.audit_campaign(root,data);out=tmp_path/'expired';out.mkdir()
    started=time.time()-1000
    put(out/'plan.json',dict(format='landcover-native-replay-plan-v1',plan_sha=audit['plan_sha'],
        collector_sha=collect.LOADED_SELF,source=audit['plan']['source'],dataset_manifest_sha=audit['plan']['dataset_manifest_sha'],
        device='cpu',budget_seconds=120,started_unix=started,deadline_unix=started+120,downtime_counts=True,test_predictions=0))
    before=(out/'plan.json').read_bytes()
    monkeypatch.setattr(collect,'audit_checkpoints',lambda *a:pytest.fail('expired budget must not restore'))
    with pytest.raises(ValueError,match='deadline exhausted'):
        collect.replay(root,data,out,device='cpu',budget_seconds=120)
    assert (out/'plan.json').read_bytes()==before
