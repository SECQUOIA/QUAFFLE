"""Native quantum formats, the16-fit boundary, bounded/resumable fresh runs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import signal
import time

import pytest
import torch

from scripts import run_hq_dfc24_fresh as fresh


def args_for(root,mode='rehearsal'):
    return SimpleNamespace(mode=mode,device='cpu',data_root=root/'data',work_dir=root/'run',
        profile=root/'profile.json',gpu_memory_fraction=None,budget_seconds=300.,resume=False)


class SyntheticTiles(torch.utils.data.Dataset):
    names=[str(i) for i in range(1533)]
    modalities=['optical' if i%2==0 else 'sar' for i in range(1533)]

    def __len__(self):return 1533

    def __getitem__(self,index):
        axis=torch.linspace(-1,1,128)
        y,x=torch.meshgrid(axis,axis,indexing='ij')
        return torch.stack((x,y,x*y))+index*.001,(x+y>.05).float().unsqueeze(0)


def make_data(args,seed,tiny=False):
    dataset=SyntheticTiles()
    global_fit,global_validation=list(range(1303)),list(range(1303,1533))
    fit=global_fit[:2 if tiny else (16 if args.mode=='rehearsal' else 1303)]
    validation=global_validation[:2 if tiny else (16 if args.mode=='rehearsal' else 230)]
    norm=dict(fit_only=True,seed=seed,n_fit_train_tiles=1303,n_excluded_val_tiles=230,max_samples=None)
    return dict(dataset=dataset,all_fit=global_fit,global_fit=global_fit,global_validation=global_validation,
        fit=fit,validation=validation,train=torch.utils.data.Subset(dataset,fit),
        val=torch.utils.data.Subset(dataset,validation),normalization=norm)


@pytest.fixture(scope='module')
def native_cells(tmp_path_factory):
    root=tmp_path_factory.mktemp('fresh-hq-native-formats')
    args=args_for(root)
    for key,value in dict(epochs=1,batch_size=2,fit_cap=2,validation_cap=2,image_size=128,
                          base_channels=1,eval_every=1,augment=True).items():setattr(args,key,value)
    data=make_data(args,42,tiny=True)
    result={}
    for kind in fresh.CONTROLS:
        directory=root/kind
        row=fresh.fit_cell(kind,42,.001,args,data,directory,torch.device('cpu'))
        result[kind]=dict(directory=directory,row=row,args=args,data=data)
    return result


@pytest.mark.parametrize('kind',fresh.CONTROLS)
def test_actual_native_formats_restore_flags_bn_and_frozen_weights(native_cells,kind):
    item=native_cells[kind]
    row=item['row'];checkpoint=torch.load(item['directory']/'checkpoint.pt',weights_only=False)
    assert checkpoint['format']==('hq-direction-v1' if kind=='quantum' else 'hq-training-controls-v1')
    assert ('kind' in row['model_config'])==(kind=='quantum')
    assert ('control' in row['model_config'])==(kind!='quantum')
    model,audit=fresh.restore_checkpoint(item['directory']/'checkpoint.pt',row)
    assert model.training_control==kind
    assert audit['strict_restore'] and audit['frozen_unchanged']
    assert audit['parameter_counts']['total']==row['total_params']
    assert int(model.stem[2].num_batches_tracked)==row['best_epoch']
    assert len(row['history'])==2 and row['history'][1]['gradient_norms']['common']['mean']>0
    if kind in ('frozen_quantum','zero_bottleneck'):
        assert all(not parameter.requires_grad for parameter in model.bottleneck.core_parameters())
        assert row['history'][1]['gradient_norms']['core']['max']==0
        assert row['diagnostics']['frozen_parameters']['unchanged']
    if kind=='zero_bottleneck':
        model.bottleneck.set_mode('on')
        assert torch.count_nonzero(model.bottleneck(torch.randn(1,8,16,16)))==0


def test_restored_frozen_circuit_still_passes_input_gradients(native_cells):
    item=native_cells['frozen_quantum']
    model,_=fresh.restore_checkpoint(item['directory']/'checkpoint.pt',item['row'])
    latent=torch.randn(1,8,16,16,requires_grad=True)
    output=model.bottleneck(latent)
    (output*torch.linspace(-.7,.9,output.numel()).reshape_as(output)).mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.abs().sum()>0
    assert model.bottleneck.core.weights.grad is None


def test_quantum_dispatch_never_enters_control_fitter(tmp_path,monkeypatch):
    calls=[];row={'native':'quantum'}
    def quantum(*args):calls.append('native-hq');return row
    def forbidden(*args):raise AssertionError('Fresh Q cannot use the control checkpoint format')
    monkeypatch.setattr(fresh.hq,'fit_one',quantum)
    monkeypatch.setattr(fresh.study,'fit_one',forbidden)
    monkeypatch.setattr(fresh,'audit_fit',lambda *args:None)
    monkeypatch.setattr(fresh,'restore_checkpoint',lambda *args:(object(),{}))
    assert fresh.fit_cell('quantum',42,.001,None,None,tmp_path/'cell','cpu')==row
    assert calls==['native-hq']


@pytest.mark.parametrize('kind',('quantum','frozen_quantum','zero_bottleneck','compact_affine'))
def test_checkpoint_corruption_rejects_nonfinite_bn(native_cells,tmp_path,kind):
    item=native_cells[kind]
    checkpoint=torch.load(item['directory']/'checkpoint.pt',weights_only=False)
    checkpoint['best_state_dict']['stem.2.running_var'][0]=float('nan')
    path=tmp_path/'bad.pt';torch.save(checkpoint,path)
    with pytest.raises((ValueError,FloatingPointError),match='BatchNorm|Nonfinite'):
        fresh.restore_checkpoint(path,item['row'])


def test_cached_complete_cell_never_calls_fitters_and_preserves_bytes(native_cells,monkeypatch):
    item=native_cells['quantum'];directory=item['directory']
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in directory.iterdir()}
    def forbidden(*args):raise AssertionError('Completed native cell must not refit')
    monkeypatch.setattr(fresh.hq,'fit_one',forbidden)
    monkeypatch.setattr(fresh.study,'fit_one',forbidden)
    assert fresh.fit_cell('quantum',42,.001,item['args'],item['data'],directory,'cpu')==item['row']
    assert {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in directory.iterdir()}==before


def test_failed_and_partial_attempts_are_preserved_without_retry(tmp_path,monkeypatch):
    def fail(*args):raise RuntimeError('synthetic fitting failure')
    monkeypatch.setattr(fresh.hq,'fit_one',fail)
    directory=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='synthetic fitting failure'):
        fresh.fit_cell('quantum',42,.001,None,None,directory,'cpu')
    before=(directory/'failure.json').read_bytes()
    with pytest.raises(ValueError,match='Failed fit evidence'):
        fresh.fit_cell('quantum',42,.001,None,None,directory,'cpu')
    assert (directory/'failure.json').read_bytes()==before
    partial=tmp_path/'partial';partial.mkdir();(partial/'history.json').write_text('[]')
    with pytest.raises(ValueError,match='Incomplete fit evidence'):
        fresh.fit_cell('quantum',42,.001,None,None,partial,'cpu')
    assert (partial/'history.json').read_text()=='[]'


def metrics(n,quality=0):
    optical=71 if n==205 else min(2,n)
    output=dict(loss=.5,auc=.7);pooled=dict(tp=0,fp=0,fn=0,tn=0)
    for sensor,count in (('optical',optical),('sar',n-optical)):
        tp,fp,fn=count*(512+quality),count*128,count*256
        tn=count*128*128-tp-fp-fn
        for key,value in zip(('tp','fp','fn','tn'),(tp,fp,fn,tn)):
            output[sensor+'_subset_'+key]=value;pooled[key]+=value
        output.update({sensor+'_subset_n_samples':count,sensor+'_subset_iou':tp/(tp+fp+fn) if count else 0.,
            sensor+'_subset_dice':2*tp/(2*tp+fp+fn) if count else 0.,
            sensor+'_subset_accuracy':(tp+tn)/(count*128*128) if count else 0.,sensor+'_subset_auc':.7 if count else 0.})
    tp,fp,fn,tn=(pooled[k] for k in ('tp','fp','fn','tn'))
    output.update(iou=tp/(tp+fp+fn),dice=2*tp/(2*tp+fp+fn),accuracy=(tp+tn)/(n*128*128))
    return output


def fake_row(kind,seed,rate,args,data):
    preferred=dict(quantum=.003,frozen_quantum=.001,zero_bottleneck=.003,compact_affine=.001)
    quality=5 if kind=='compact_affine' else (20 if rate==preferred[kind] else 0)
    val=metrics(len(data['validation']),quality)
    history=[dict(epoch=epoch,train_loss=None if epoch==0 else .5,train_schedule=None if epoch==0 else
        dict(sha256=fresh.hq.epoch_schedule(data['fit'],seed,epoch,True)[2],n_samples=len(data['fit'])),
        **({'val':val} if epoch%args.eval_every==0 or epoch==args.epochs else {})) for epoch in range(args.epochs+1)]
    return dict(id='hq_'+kind,kind=kind,seed=seed,learning_rate=rate,best_epoch=0,val=val,history=history,
        initial_shared_state_sha256='shared'+str(seed),initial_state_sha256=('affine' if kind=='compact_affine' else 'circuit')+str(seed),
        fit_indices=data['fit'],val_indices=data['validation'],fit_names=[data['dataset'].names[i] for i in data['fit']],
        val_names=[data['dataset'].names[i] for i in data['validation']],normalization=data['normalization'],
        settings=dict(_resolved_norm_provenance=data['normalization'],augment=True))


@pytest.fixture
def campaign(tmp_path,monkeypatch):
    from tools import training_common as tc
    args=args_for(tmp_path)
    source=tmp_path/'source';(source/'docs').mkdir(parents=True)
    (source/fresh.PROTOCOL).write_text('Synthetic frozen execution protocol')
    monkeypatch.setattr(fresh,'ROOT',source)
    monkeypatch.setattr(fresh,'source_hashes',lambda:{'producer.py':'same-bytes'})
    runtime=dict(device='cpu',threads=2,interop_threads=2,gpu_allocator_fraction=None)
    monkeypatch.setattr(fresh,'runtime',lambda *args:runtime)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda threads:None)
    monkeypatch.setattr(fresh.subprocess,'check_output',lambda *args,**kwargs:'original-producer\n')
    inputs={'manifest_sha256':'same-inputs'}
    monkeypatch.setattr(fresh.pilot,'verify_inputs',lambda root:inputs)
    monkeypatch.setattr(fresh,'load_data',lambda args,seed:make_data(args,seed))
    events=[]
    def fit(kind,seed,rate,args,data,directory,device):
        events.append(('fit',kind,seed,rate));row=fake_row(kind,seed,rate,args,data)
        directory.mkdir(parents=True)
        fresh.pilot.write_json(directory/'metrics.json',row);fresh.pilot.write_json(directory/'history.json',row['history'])
        (directory/'checkpoint.pt').write_bytes(json.dumps((kind,seed,rate)).encode())
        return row
    monkeypatch.setattr(fresh,'fit_cell',fit)
    def audit(row,kind,seed,rate,args,data):
        assert (row['kind'],row['seed'],row['learning_rate'])==(kind,seed,rate)
        assert row['fit_indices']==data['fit'] and row['val_indices']==data['validation']
    monkeypatch.setattr(fresh,'audit_fit',audit)
    def restore(path,row,device='cpu'):
        events.append(('restore',row['kind'],row['seed'],row['learning_rate']))
        return SimpleNamespace(row=row),dict(checkpoint_sha256=fresh.digest(path),strict_restore=True,
            finite_weights_and_bn_buffers=True,selected_state_sha256='fixture-state')
    monkeypatch.setattr(fresh,'restore_checkpoint',restore)
    def dirs(*args):
        assert (args_for_root.work_dir/'fit_boundary.json').is_file()
        events.append(('construct_test',))
        return None,None,None,'test-images','test-masks'
    class TestData:
        names=['test-'+str(i) for i in range(205)]
        def __init__(self,cfg):self.norm_provenance=cfg['_resolved_norm_provenance']
        def __len__(self):return 205
    monkeypatch.setattr(tc,'resolve_data_dirs',dirs)
    monkeypatch.setattr(tc,'load_dataset',lambda images,masks,cfg:TestData(cfg))
    args.data_root.mkdir();(args.data_root/'splits').mkdir()
    fresh.pilot.write_json(args.data_root/'splits/flood_combined.json',{'test':TestData.names})
    def evaluated(model,loader,device):
        boundary=fresh.read_json(args.work_dir/'fit_boundary.json')
        assert boundary['fits']==16 and boundary['selected']==12
        assert len(fresh.read_json(args.work_dir/'all_fits.json'))==16
        assert len(fresh.read_json(args.work_dir/'checkpoint_audit.json'))==16
        assert len([e for e in events if e[0]=='fit'])==16
        events.append(('replay',model.row['kind'],model.row['seed']))
        return model.row['val'] if args.mode=='rehearsal' else metrics(205)
    monkeypatch.setattr(fresh.hq,'evaluate',evaluated)
    args_for_root=args
    def write_profile():
        for key,value in fresh.settings_for_mode(args.mode,'cpu').items():setattr(args,key,value)
        profile=dict(format='hq-dfc24-fresh-profile-v1',status='passed',mode=args.mode,settings=fresh.settings_for_mode(args.mode,'cpu'),
            profile_learning_rate=.001,
            runtime=runtime,source_hashes=fresh.source_hashes(),protocol_sha256=fresh.digest(source/fresh.PROTOCOL),
            input_identity=inputs,cohort=fresh.cohort_record(make_data(args,42)),
            cases=[dict(kind=kind,batch_size=8,strict_restore=True,frozen_unchanged=True) for kind in fresh.CONTROLS])
        fresh.pilot.write_json(args.profile,profile)
    write_profile()
    return SimpleNamespace(args=args,events=events,write_profile=write_profile,inputs=inputs,runtime=runtime)


@pytest.mark.parametrize('mode',('rehearsal','full'))
def test_complete_fresh_grid_precedes_replays_and_rehearsal_never_loads_test(campaign,mode):
    args=campaign.args
    if mode!=args.mode:
        for key in fresh.settings_for_mode(args.mode,'cpu'):
            if key not in ('mode','device'):delattr(args,key)
        args.mode=mode;campaign.write_profile()
    summary=fresh.run(args,torch.device('cpu'))
    first=next(i for i,e in enumerate(campaign.events) if e[0]=='replay')
    assert len([e for e in campaign.events[:first] if e[0]=='fit'])==16
    assert len([e for e in campaign.events[:first] if e[0]=='restore'])==17
    assert len(summary['records'])==12
    assert summary['selection']['quantum']['learning_rate']==.003
    assert summary['selection']['frozen_quantum']['learning_rate']==.001
    assert summary['selection']['compact_affine']['learning_rate']==.001  # Exact tie lower rate.
    assert all('test' not in row and 'test_names' not in row for row in summary['records'])
    if mode=='rehearsal':
        assert not [e for e in campaign.events if e[0]=='construct_test']
        assert summary['analysis']['gate']=={'status':'not_applicable_rehearsal'}
        assert all(row['replay_split']=='validation' and len(row['replay_names'])==16 for row in summary['records'])
    else:
        assert len([e for e in campaign.events if e[0]=='construct_test'])==12
        assert all(row['replay_split']=='test' and len(row['replay_names'])==205 for row in summary['records'])
    cohorts=summary['plan']['cohorts']
    assert all(len(c['global_validation_names'])==230 and len(c['global_fit_names'])==1303 for c in cohorts.values())


def inventory(root):
    return {str(p.relative_to(root)):(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


def test_completed_resume_after_learning_deadline_is_readonly(campaign,monkeypatch):
    original=fresh.run(campaign.args,torch.device('cpu'));before=inventory(campaign.args.work_dir)
    campaign.args.resume=True
    def forbidden(*args):raise AssertionError('Completed resume must not fit/infer')
    monkeypatch.setattr(fresh,'fit_cell',forbidden);monkeypatch.setattr(fresh.hq,'evaluate',forbidden)
    original_time=time.time;monkeypatch.setattr(fresh.time,'time',lambda:original_time()+10000)
    monkeypatch.setattr(fresh.subprocess,'check_output',lambda *args,**kwargs:'reporting-child\n')
    assert fresh.run(campaign.args,torch.device('cpu'))==original
    assert inventory(campaign.args.work_dir)==before


@pytest.mark.parametrize('what',('source','runtime','inputs','profile','cohorts'))
def test_resume_binding_mismatch_prevents_any_new_fit(campaign,monkeypatch,what):
    fresh.run(campaign.args,torch.device('cpu'));before=inventory(campaign.args.work_dir);campaign.args.resume=True
    def forbidden(*args):raise AssertionError('Changed identity must stop before fitting')
    monkeypatch.setattr(fresh,'fit_cell',forbidden)
    if what=='source':monkeypatch.setattr(fresh,'source_hashes',lambda:{'producer.py':'changed'})
    elif what=='runtime':monkeypatch.setattr(fresh,'runtime',lambda *args:dict(changed=True))
    elif what=='inputs':monkeypatch.setattr(fresh.pilot,'verify_inputs',lambda root:{'changed':True})
    elif what=='profile':
        profile=fresh.read_json(campaign.args.profile);profile['settings']['epochs']=99;fresh.pilot.write_json(campaign.args.profile,profile)
    else:
        def changed_data(args,seed):
            value=make_data(args,seed);value['normalization']=dict(value['normalization'],changed=True);return value
        monkeypatch.setattr(fresh,'load_data',changed_data)
    with pytest.raises(ValueError,match='Profile|profile|Frozen'):
        fresh.run(campaign.args,torch.device('cpu'))
    assert inventory(campaign.args.work_dir)==before


def test_malformed_saved_replay_binding_is_preserved(campaign,monkeypatch):
    fresh.run(campaign.args,torch.device('cpu'));campaign.args.resume=True
    path=campaign.args.work_dir/'replays/hq_quantum/seed42.json'
    payload=fresh.read_json(path);payload['binding']['split']='test';fresh.pilot.write_json(path,payload)
    before=inventory(campaign.args.work_dir)
    monkeypatch.setattr(fresh.hq,'evaluate',lambda *args:pytest.fail('Must not retry malformed replay'))
    with pytest.raises(ValueError,match='binding differs'):
        fresh.run(campaign.args,torch.device('cpu'))
    assert inventory(campaign.args.work_dir)==before


def test_failed_inference_marker_blocks_silent_retry(campaign,monkeypatch):
    def fail(*args):raise RuntimeError('synthetic inference failure')
    monkeypatch.setattr(fresh.hq,'evaluate',fail)
    with pytest.raises(RuntimeError,match='synthetic inference failure'):
        fresh.run(campaign.args,torch.device('cpu'))
    before=inventory(campaign.args.work_dir);campaign.args.resume=True
    with pytest.raises(ValueError,match='Failed replay evidence'):
        fresh.run(campaign.args,torch.device('cpu'))
    assert inventory(campaign.args.work_dir)==before


def test_one_writer_lock_budget_and_signal_state_are_enforced(tmp_path):
    args=args_for(tmp_path);args.data_root.mkdir();args.work_dir=args.work_dir.resolve();args.data_root=args.data_root.resolve()
    with fresh._session(args,'run'):
        second=copy.copy(args);second.resume=True
        with pytest.raises(ValueError,match='live writer'):
            with fresh._session(second,'run'):pass
    original_handler=signal.getsignal(signal.SIGALRM)
    args=args_for(tmp_path/'timeout');args.budget_seconds=.01
    with pytest.raises(TimeoutError,match='wall budget'):
        with fresh._session(args,'profile'):time.sleep(.03)
    assert signal.getsignal(signal.SIGALRM)==original_handler
    assert signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
    before=inventory(args.work_dir);args.resume=True
    with pytest.raises(ValueError,match='deadline expired'):
        with fresh._session(args,'run'):pass
    assert inventory(args.work_dir)==before


@pytest.mark.parametrize('text',('{"key":1,"key":2}','{"x":NaN}','{"x":1e999}'))
def test_duplicate_or_nonfinite_json_is_rejected(tmp_path,text):
    path=tmp_path/'bad.json';path.write_text(text)
    with pytest.raises(ValueError):fresh.read_json(path)


@pytest.mark.parametrize('budget',('0','-1','nan','inf'))
def test_cli_requires_explicit_finite_budget_mode_and_cuda_fraction(tmp_path,budget):
    with pytest.raises(SystemExit):
        fresh.main(['profile','--mode','rehearsal','--device','cpu','--data-root',str(tmp_path),
            '--work-dir',str(tmp_path/'out'),'--budget-seconds',budget])
    with pytest.raises(SystemExit):
        fresh.main(['profile','--mode','rehearsal','--data-root',str(tmp_path),
            '--work-dir',str(tmp_path/'out'),'--budget-seconds','60'])


@pytest.mark.parametrize('bad_normalization',(False,True))
def test_load_data_keeps_full_validation_exclusion_when_rehearsal_caps_evaluation(tmp_path,monkeypatch,bad_normalization):
    args=args_for(tmp_path)
    for key,value in fresh.settings_for_mode('rehearsal','cpu').items():setattr(args,key,value)
    full_args=copy.copy(args);full_args.mode='full'
    data=make_data(full_args,42)
    data['fit']=list(range(16));data['train']=torch.utils.data.Subset(data['dataset'],data['fit'])
    if bad_normalization:data['normalization']['n_excluded_val_tiles']=16
    monkeypatch.setattr(fresh.pilot,'load_training_data',lambda args,seed:data)
    if bad_normalization:
        with pytest.raises(ValueError,match='all230 validation'):
            fresh.load_data(args,42)
    else:
        loaded=fresh.load_data(args,42);cohort=fresh.cohort_record(loaded)
        assert len(cohort['global_fit_indices'])==1303 and len(cohort['global_validation_indices'])==230
        assert len(cohort['fit_indices'])==len(cohort['validation_indices'])==16
        assert set(cohort['validation_indices'])<=set(cohort['global_validation_indices'])
        assert loaded['normalization']['n_excluded_val_tiles']==230


def minimal_candidates():
    return [dict(origin='fresh',record=dict(id='hq_'+kind,kind=kind,seed=42,learning_rate=rate,
        val={'iou':.5},best_epoch=0)) for kind in fresh.CONTROLS for rate in fresh.RATES]


@pytest.mark.parametrize('change',('missing','duplicate','float_seed','bool_seed','historical_origin'))
def test_tuning_grid_rejects_missing_duplicate_untyped_or_historical_cells(change):
    entries=minimal_candidates()
    assert all(value['learning_rate']==.001 for value in fresh.select_rates(entries).values())
    if change=='missing':entries.pop()
    elif change=='duplicate':entries[-1]=copy.deepcopy(entries[0])
    elif change=='float_seed':entries[0]['record']['seed']=42.
    elif change=='bool_seed':entries[0]['record']['seed']=True
    else:entries[0]['origin']='reused'
    with pytest.raises(ValueError):fresh.select_rates(entries)


@pytest.mark.parametrize('change',('missing','duplicate','shared_state','circuit_state','schedule'))
def test_paired_complete_grid_rejects_incomplete_or_changed_inputs(campaign,change):
    fresh.run(campaign.args,torch.device('cpu'))
    rows=[entry['record'] for entry in fresh.read_json(campaign.args.work_dir/'all_fits.json')]
    fresh.paired_audit(rows)
    if change=='missing':rows.pop()
    elif change=='duplicate':rows[-1]=copy.deepcopy(rows[-2])
    else:
        row=next(row for row in rows if row['kind']=='frozen_quantum' and row['seed']==42)
        if change=='shared_state':row['initial_shared_state_sha256']='different'
        elif change=='circuit_state':row['initial_state_sha256']='different'
        else:row['history'][1]['train_schedule']['sha256']='different'
    with pytest.raises(ValueError):fresh.paired_audit(rows)


def test_runtime_records_actual_operator_fraction_not_legacy_hardcoded_value(monkeypatch):
    monkeypatch.setattr(fresh.hq,'runtime',lambda device:dict(device='cuda:0',gpu_allocator_fraction=.12))
    assert fresh.runtime(SimpleNamespace(type='cuda'),.4)['gpu_allocator_fraction']==.4
    assert fresh.runtime(SimpleNamespace(type='cpu'),None)['gpu_allocator_fraction'] is None
    with pytest.raises(ValueError,match='explicit valid'):
        fresh.runtime(SimpleNamespace(type='cuda'),None)


@pytest.mark.parametrize('removed',('all_replays_and_selection','one_replay','selection','progress','boundary','lock','intent'))
def test_damaged_completed_resume_does_not_repair_or_rewrite_any_raw_state(campaign,removed):
    fresh.run(campaign.args,torch.device('cpu'));campaign.args.resume=True
    work=campaign.args.work_dir
    if removed=='all_replays_and_selection':
        for path in (work/'replays').rglob('*.json'):path.unlink()
        (work/'selection.json').unlink()
    else:
        path={'one_replay':work/'replays/hq_quantum/seed42.json','selection':work/'selection.json',
            'progress':work/'fit_progress.json','boundary':work/'fit_boundary.json','lock':work/'.writer.lock',
            'intent':work/'replay_intents/hq_quantum/seed42.json'}[removed]
        path.unlink()
    before=inventory(work);namespace={str(p.relative_to(work)) for p in work.rglob('*')}
    with pytest.raises(ValueError,match='Completed resume'):
        fresh.run(campaign.args,torch.device('cpu'))
    assert inventory(work)==before
    assert {str(p.relative_to(work)) for p in work.rglob('*')}==namespace


def test_unfinished_bound_replay_intent_prevents_measurement_retry(campaign,monkeypatch):
    fresh.run(campaign.args,torch.device('cpu'));campaign.args.resume=True
    work=campaign.args.work_dir
    # State equivalent to process death after intent creation and before a saved
    # replay/final summary. A signal handler could not have written failure.json.
    (work/'summary.json').unlink();(work/'replays/hq_quantum/seed42.json').unlink()
    before=inventory(work)
    def forbidden(*args):raise AssertionError('An interrupted measurement cannot retry')
    monkeypatch.setattr(fresh.hq,'evaluate',forbidden);monkeypatch.setattr(fresh,'fit_cell',forbidden)
    with pytest.raises(ValueError,match='Unfinished replay intent'):
        fresh.run(campaign.args,torch.device('cpu'))
    assert inventory(work)==before
