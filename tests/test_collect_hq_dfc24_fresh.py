"""Synthetic records/checkpoints verify audit contracts, never scientific fits."""
import argparse
import copy
from fractions import Fraction
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from scripts import collect_hq_dfc24_fresh as collect
from scripts import run_hq_training_controls as controls
from utilsTorch.hq_training_controls import build_model, parameter_counts

study, hq = collect.study, collect.hq


def metrics(n, optical, quality=0):
    result, pooled = dict(loss=.7, auc=.6), dict(tp=0, fp=0, fn=0, tn=0)
    for sensor, samples in (('optical', optical), ('sar', n-optical)):
        counts = dict(tp=samples*(2000+quality), fp=samples*400, fn=samples*800)
        counts['tn'] = samples*128*128-sum(counts.values())
        for name, value in counts.items():
            result[sensor+'_subset_'+name] = value; pooled[name] += value
        tp, fp, fn, tn = (counts[name] for name in ('tp', 'fp', 'fn', 'tn'))
        result.update({sensor+'_subset_n_samples': samples,
            sensor+'_subset_iou': tp/(tp+fp+fn) if tp+fp+fn else 0.,
            sensor+'_subset_dice': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            sensor+'_subset_accuracy': (tp+tn)/sum(counts.values()) if sum(counts.values()) else 0.,
            sensor+'_subset_auc': .6})
    tp, fp, fn, tn = (pooled[name] for name in ('tp', 'fp', 'fn', 'tn'))
    return dict(result, iou=tp/(tp+fp+fn), dice=2*tp/(2*tp+fp+fn), accuracy=(tp+tn)/sum(pooled.values()))


@pytest.fixture
def campaign_factory(tmp_path, monkeypatch):
    def make(mode='rehearsal'):
        base = tmp_path/mode
        source, raw, data_root = (base/name for name in ('source', 'raw', 'data'))
        for name in (*study.SOURCE_FILES, study.PROTOCOL, collect.COLLECTOR):
            path = source/name; path.parent.mkdir(parents=True, exist_ok=True)
            if name == collect.COLLECTOR:
                path.write_bytes(Path(collect.__file__).read_bytes())
            else:
                path.write_text('Synthetic inert source: '+name)
        names = [f'train{i:04d}' for i in range(1533)]
        manifest = dict(conflict_policy='drop', train=names, test=[f'test{i:04d}' for i in range(205)])
        collect.write(data_root/'splits/flood_combined.json', manifest)
        inputs = dict(manifest_sha256=collect.digest(data_root/'splits/flood_combined.json'), verified_files=3476,
            stats_files_sha256={}, raw_inventory_sha256='synthetic-input-identity')
        monkeypatch.setattr(collect, 'ROOT', source)
        monkeypatch.setattr(hq.pilot, 'verify_inputs', lambda root: copy.deepcopy(inputs))
        settings = study.settings_for_mode(mode, 'cuda')
        runtime = dict(**collect.current_runtime(), executable='/original/producing/venv/python', device='cuda:0',
            gpu='Original synthetic GPU', gpu_allocator_fraction=.5, threads=2, interop_threads=2)
        def cohort(seed):
            fit = list(range(1303)) if mode == 'full' else list(range(16))
            global_val = list(range(1303,1533)); val = global_val if mode == 'full' else global_val[:16]
            return dict(fit=fit, validation=val, global_fit=list(range(1303)), all_fit=list(range(1303)),
                global_validation=global_val, dataset=SimpleNamespace(names=names,
                    modalities=['optical' if i%5 == 0 else 'sar' for i in range(1533)]),
                normalization=dict(fit_only=True, n_fit_train_tiles=1303, n_excluded_val_tiles=230,
                    max_samples=None, seed=seed, fit_names_sha256='synthetic-global-fit-'+str(seed)))
        monkeypatch.setattr(study, 'load_data', lambda args, seed: cohort(seed))
        args = argparse.Namespace(**settings, data_root=data_root)
        entries, paths = [], {}
        def one(kind, seed, rate):
            data = cohort(seed); model = build_model(kind,seed)
            config = hq.model_configuration(model) if kind == 'quantum' else controls.model_configuration(model)
            initial = {name: value.detach().clone() for name, value in model.named_parameters()}
            # Quantum/zero select .003, frozen selects .001, compact has an exact
            # score tie. This exercises independent tuning and the lower-rate tie.
            preferred = .003 if kind in ('quantum','zero_bottleneck') else .001
            quality = 80-study.CONTROLS.index(kind)*8 + (12 if rate == preferred and kind != 'compact_affine' else 0)
            n = len(data['validation']); optical = sum(i%5 == 0 for i in data['validation'])
            val = metrics(n,optical,quality)
            steps = (len(data['fit'])+7)//8
            active = dict(min=.01,mean=.02,max=.03,steps=steps,nonzero_steps=steps)
            core = active if kind not in ('frozen_quantum','zero_bottleneck') else dict(
                min=0.,mean=0.,max=0.,steps=steps,nonzero_steps=0)
            history = []
            for epoch in range(args.epochs+1):
                point = dict(epoch=epoch,train_loss=None if not epoch else .7,
                    train_schedule=None if not epoch else dict(
                        sha256=hq.epoch_schedule(data['fit'],seed,epoch,True)[2],n_samples=len(data['fit'])),
                    gradient_norms={} if not epoch else dict(core=core,common=active))
                if epoch%args.eval_every == 0 or epoch == args.epochs:
                    point['val'] = val
                history.append(point)
            # Every epoch ties at the maximum: epoch0 is a legitimate selected
            # checkpoint even though all training-batch gradient records are nonzero.
            probe = dict(n_angles=384,min_radians=-.8,max_radians=.7,mean_abs_radians=.3,saturation_fraction=0.)
            diagnostics = dict(initial_angle_probes=[probe],selected_angle_probes=[probe],
                probe_names=names[:8],movement=hq.movement(model,initial),
                gradient_norms={group:dict(value,steps=value['steps']*args.epochs,
                    nonzero_steps=value['nonzero_steps']*args.epochs)
                    for group,value in dict(core=core,common=active).items()})
            row = dict(id='hq_'+kind,kind=kind,seed=seed,learning_rate=rate,model_config=config,
                settings=hq.fit_settings(kind,seed,rate,args,data),
                total_params=sum(p.numel() for p in model.parameters()),
                trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                core_params=sum(p.numel() for p in model.bottleneck.core_parameters()),
                initial_state_sha256=hq.state_hash(model.state_dict()),selected_state_sha256=hq.state_hash(model.state_dict()),
                initial_shared_state_sha256=hq.shared_hash(model),best_epoch=0,val=val,history=history,
                diagnostics=diagnostics,elapsed_s=1.,fit_indices=data['fit'],val_indices=data['validation'],
                fit_names=[names[i] for i in data['fit']],val_names=[names[i] for i in data['validation']],
                normalization=data['normalization'],n_global_fit=1303,
                fit_sensor_counts=dict(optical=sum(i%5 == 0 for i in data['fit']),sar=sum(i%5 != 0 for i in data['fit'])))
            if kind == 'quantum':
                checkpoint = dict(format='hq-direction-v1',kind=kind,seed=seed,best_epoch=0,
                    settings=row['settings'],model_config=config,best_state_dict=model.state_dict())
            else:
                row.update(trainability=controls.trainability(model),parameter_counts=parameter_counts(model))
                frozen = controls.frozen_hash(model)
                diagnostics['frozen_parameters'] = dict(initial_sha256=frozen,selected_sha256=frozen,unchanged=True)
                diagnostics['angle_probe_scope'] = 'inactive_counterfactual' if kind == 'zero_bottleneck' else 'active_fixed_probe'
                checkpoint = dict(format='hq-training-controls-v1',**{key:row[key] for key in
                    ('kind','seed','best_epoch','settings','model_config','trainability','parameter_counts')},
                    best_state_dict=model.state_dict())
            relative = str(Path('fits')/row['id']/str(rate)/('seed'+str(seed)));path=raw/relative
            collect.write(path/'metrics.json',row);collect.write(path/'history.json',history)
            torch.save(checkpoint,path/'checkpoint.pt')
            entry = dict(origin='fresh',directory=relative,record=row,**{name.split('.')[0]+'_sha256':
                collect.digest(path/name) for name in ('metrics.json','history.json','checkpoint.pt')})
            entries.append(entry);paths[collect.identity(row)]=path
        for kind in study.CONTROLS:
            for rate in study.RATES:one(kind,42,rate)
        selection = collect.select_rates(entries)
        for kind in study.CONTROLS:
            for seed in study.SEEDS[1:]:one(kind,seed,selection[kind]['learning_rate'])
        budget=dict(format='hq-dfc24-fresh-budget-v1',budget_seconds=1800,started_unix=1000.,
            deadline_unix=2800.,queue_downtime_counts=True)
        collect.write(raw/'budget.json',budget)
        plan = dict(format='hq-dfc24-fresh-v1',mode=mode,controls=list(study.CONTROLS),seeds=list(study.SEEDS),
            rates=list(study.RATES),settings=settings,source_revision='0'*40,
            source_hashes={name:collect.digest(source/name) for name in study.SOURCE_FILES},
            protocol_sha256=collect.digest(source/study.PROTOCOL),input_identity=inputs,runtime=runtime,
            data_root=str(data_root),cohorts={str(seed):study.cohort_record(cohort(seed)) for seed in study.SEEDS},
            replay_split='test' if mode == 'full' else 'validation',budget_seconds=1800,
            budget_sha256=collect.digest(raw/'budget.json'),
            selection_policy='Maximum val flood IoU, exact tie lower rate; earliest strict best epoch including0',
            test_policy='All16 fits/audits precede12selected replays; rehearsal validation only')
        cases=[]
        for kind in study.CONTROLS:
            model=build_model(kind,42)
            cases.append(dict(kind=kind,batch_size=8,parameter_counts=parameter_counts(model),trainability=controls.trainability(model),
                control_mode='zero_bottleneck' if kind == 'zero_bottleneck' else 'on',
                gradient_norms=[dict(core=0. if kind in ('frozen_quantum','zero_bottleneck') else .01,common=.1)]*2,
                timings_s=dict(adam_steps=[.1,.2],validation=.1,checkpoint_save=.1,checkpoint_load_restore=.1),
                host_peak_rss_bytes=123456,peak_cuda_allocated_bytes=10000,peak_cuda_reserved_bytes=20000,
                validation=metrics(len(cohort(42)['validation']),
                    sum(i%5==0 for i in cohort(42)['validation']),30),
                strict_restore=True,frozen_unchanged=True))
        profile=dict(format='hq-dfc24-fresh-profile-v1',mode=mode,settings=settings,source_revision='0'*40,
            source_hashes=plan['source_hashes'],protocol_sha256=plan['protocol_sha256'],input_identity=inputs,
            runtime=runtime,cohort=plan['cohorts']['42'],cases=cases,status='passed',data_loading_s=.1,profile_learning_rate=.001)
        collect.write(raw/'resource_profile.json',profile);plan['resource_profile_sha256']=collect.digest(raw/'resource_profile.json')
        collect.write(raw/'plan.json',plan)
        audits,records=[],[]
        for entry in entries:
            row=entry['record'];model,audit=study.restore_checkpoint(paths[collect.identity(row)]/'checkpoint.pt',row)
            audits.append(dict(id=row['id'],seed=row['seed'],learning_rate=row['learning_rate'],origin='fresh',**audit))
            if row['learning_rate'] != selection[row['kind']]['learning_rate']:continue
            row=copy.deepcopy(row);names_replay=manifest['test'] if mode=='full' else row['val_names']
            row.update(replay=metrics(205,71,50-study.CONTROLS.index(row['kind'])*7) if mode=='full' else copy.deepcopy(row['val']),
                replay_names=names_replay,replay_split=plan['replay_split'],
                checkpoint=dict(origin='fresh',path=str(Path(entry['directory'])/'checkpoint.pt')))
            binding=study.replay_binding(entry,collect.digest(raw/'plan.json'),names_replay,plan['replay_split'])
            collect.write(raw/'replay_intents'/row['id']/('seed'+str(row['seed'])+'.json'),dict(binding=binding))
            collect.write(raw/'replays'/row['id']/('seed'+str(row['seed'])+'.json'),dict(binding=binding,record=row))
            records.append(row)
        for filename,value in dict(all_fits=entries,fit_progress=entries,selection=selection,checkpoint_audit=audits,
            summary=dict(plan=plan,selection=selection,records=records,analysis=study.analysis(records,plan['replay_split'])),
            budget=budget).items():
            collect.write(raw/(filename+'.json'),value)
        collect.write(raw/'fit_boundary.json',dict(format='hq-dfc24-fresh-boundary-v1',completed=True,fits=16,
            selected=12,replay_split=plan['replay_split'],plan_sha256=collect.digest(raw/'plan.json'),
            all_fits_sha256=collect.digest(raw/'all_fits.json'),selection_sha256=collect.digest(raw/'selection.json'),
            checkpoint_audit_sha256=collect.digest(raw/'checkpoint_audit.json')))
        (raw/'.writer.lock').touch()
        return raw,data_root,source
    return make


def test_confusion_metrics_have_independent_hand_computable_values():
    values=collect.derived_metrics(dict(tp=3,fp=1,fn=2,tn=4))
    assert values == dict(flood_iou=.5,background_iou=4/7,binary_macro_iou=(.5+4/7)/2,flood_dice=2/3,accuracy=.7)
    assert set(collect.derived_metrics(dict(tp=0,fp=0,fn=0,tn=0)).values()) == {0.}
    with pytest.raises(ValueError):collect.derived_metrics(dict(tp=True,fp=1,fn=2,tn=4))


@pytest.mark.parametrize('mode',('rehearsal','full'))
def test_complete_native_checkpoint_grid_and_mode_specific_analysis(campaign_factory,mode):
    raw,data,_=campaign_factory(mode)
    result=collect.audit_campaign(raw,data)
    assert len(result['fits'])==len(result['checkpoint_audit'])==16
    assert len(result['summary']['records'])==12
    assert result['selection']['compact_affine']['learning_rate']==.001
    assert result['selection']['quantum']['learning_rate']==.003
    assert all(row['record']['best_epoch']==0 for row in result['fits'])
    derived=collect.analysis(result)
    assert len(derived['paired'])==90
    assert derived['gate']['status']==('not_applicable_rehearsal' if mode=='rehearsal' else 'fail')
    if mode=='rehearsal':assert set(derived['gate'])=={'status','reason'}
    rows=collect.tables(result,derived)
    assert {name:len(values) for name,values in rows.items()}==dict(summary=120,candidates=48,per_seed=72,paired=90,trainability=16)
    assert all(row['core_gradient_steps']==(4 if mode=='rehearsal' else 9780) for row in rows['trainability'])


@pytest.mark.parametrize('damage',('source','self_source','runtime','mode','missing_fit','checkpoint','history','selection',
    'boundary','replay_binding','replay_counts','replay_intent','missing_intent','orphan_intent',
    'typed_seed','extra_fit','empty_fit','failure','profile','budget'))
def test_corruption_and_partial_runs_fail_closed(campaign_factory,damage):
    raw,data,source=campaign_factory()
    entries=collect.read(raw/'all_fits.json');entry=entries[0];directory=raw/entry['directory']
    if damage=='source':(source/'scripts/run_hq_dfc24_fresh.py').write_text('stale')
    elif damage=='self_source':(source/collect.COLLECTOR).write_text('stale collector snapshot')
    elif damage in ('runtime','mode'):
        plan=collect.read(raw/'plan.json');summary=collect.read(raw/'summary.json')
        if damage=='runtime':plan['runtime']['packages']['torch']='different'
        else:plan['mode']='full'
        summary['plan']=plan;collect.write(raw/'plan.json',plan);collect.write(raw/'summary.json',summary)
    elif damage=='missing_fit':(directory/'history.json').unlink()
    elif damage=='checkpoint':(directory/'checkpoint.pt').write_bytes(b'bad checkpoint')
    elif damage=='history':
        value=collect.read(directory/'history.json');value.pop();collect.write(directory/'history.json',value)
    elif damage=='selection':
        value=collect.read(raw/'selection.json');value['quantum']['learning_rate']=.001;collect.write(raw/'selection.json',value)
    elif damage=='boundary':
        value=collect.read(raw/'fit_boundary.json');value['fits']=15;collect.write(raw/'fit_boundary.json',value)
    elif damage=='replay_intent':
        path=raw/'replay_intents/hq_quantum/seed42.json';value=collect.read(path)
        value['binding']['split']='test';collect.write(path,value)
    elif damage=='missing_intent':(raw/'replay_intents/hq_quantum/seed42.json').unlink()
    elif damage=='orphan_intent':collect.write(raw/'replay_intents/hq_quantum/seed999.json',dict(binding={}))
    elif damage.startswith('replay_'):
        path=raw/'replays/hq_quantum/seed42.json';value=collect.read(path)
        if damage=='replay_binding':value['binding']['split']='test'
        else:value['record']['replay']['sar_subset_iou']+=.1
        collect.write(path,value)
    elif damage=='typed_seed':
        entries[0]['record']['seed']=42.;collect.write(raw/'all_fits.json',entries);collect.write(raw/'fit_progress.json',entries)
    elif damage=='extra_fit':collect.write(raw/'fits/unexpected/metrics.json',{})
    elif damage=='empty_fit':(raw/'fits/hq_quantum/0.009/seed999').mkdir(parents=True)
    elif damage=='failure':collect.write(raw/'replay_failures/hq_quantum/seed42.json',dict(error='retained'))
    elif damage=='profile':(raw/'resource_profile.json').write_text('{}')
    else:
        value=collect.read(raw/'budget.json');value['deadline_unix']+=1;collect.write(raw/'budget.json',value)
    with pytest.raises((ValueError,FileNotFoundError,RuntimeError,KeyError,EOFError)):
        collect.audit_campaign(raw,data)


def test_collector_exports_original_bytes_and_is_deterministic_without_inference(campaign_factory,tmp_path,monkeypatch):
    raw,data,_=campaign_factory()
    def forbidden(*args,**kwargs):raise AssertionError('collector must not fit or infer')
    monkeypatch.setattr(study,'fit_cell',forbidden)
    monkeypatch.setattr(hq,'evaluate',forbidden)
    result=collect.audit_campaign(raw,data)
    first,second=tmp_path/'out1',tmp_path/'out2'
    collect.export(result,raw,data,first);collect.export(result,raw,data,second)
    hashes=collect.read(first/'artifact_sha256.json')
    assert hashes==collect.read(second/'artifact_sha256.json')
    assert set(hashes)=={str(path.relative_to(first)) for path in first.rglob('*') if path.is_file()}-{'artifact_sha256.json'}
    assert all(collect.digest(first/name)==value for name,value in hashes.items())
    assert (first/'raw_campaign/summary.json').read_bytes()==(raw/'summary.json').read_bytes()
    assert len(list((first/'raw_campaign').rglob('history.json')))==16
    assert len(list((first/'raw_campaign/replays').rglob('*.json')))==12
    assert not list(first.rglob('*.pt'))
    assert collect.read(first/'analysis.json')['gate']['status']=='not_applicable_rehearsal'
    with pytest.raises(ValueError,match='empty'):collect.export(result,raw,data,first)
    with pytest.raises(ValueError,match='aliases'):collect.export(result,raw,data,data/'new-output')


def test_duplicate_keys_nonfinite_json_and_source_aliases_are_rejected(tmp_path):
    path=tmp_path/'bad.json'
    for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e999}'):
        path.write_text(text)
        with pytest.raises(ValueError):collect.read(path)
    with pytest.raises(ValueError):collect.main(['--work-dir',str(tmp_path),'--data-root',str(tmp_path),
        '--output-dir',str(tmp_path/'output')])


@pytest.mark.parametrize('damage',('trainability','batchnorm','mode'))
def test_checkpoint_semantic_damage_is_rejected(campaign_factory,tmp_path,monkeypatch,damage):
    raw,_,_=campaign_factory()
    kind='zero_bottleneck' if damage=='mode' else 'frozen_quantum'
    entry=next(entry for entry in collect.read(raw/'all_fits.json') if entry['record']['kind']==kind)
    row=entry['record'];path=raw/entry['directory']/'checkpoint.pt'
    if damage=='mode':
        restore=study.restore_checkpoint
        def wrong(*args,**kwargs):
            model,audit=restore(*args,**kwargs);model.bottleneck.mode='on';return model,audit
        monkeypatch.setattr(study,'restore_checkpoint',wrong)
        with pytest.raises(ValueError,match='mode'):collect.checkpoint_audit(path,row)
        return
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if damage=='trainability':
        name=next(name for name in row['trainability'] if name.startswith('bottleneck.core.'))
        checkpoint['trainability'][name]=row['trainability'][name]=True
    else:
        name=next(name for name in checkpoint['best_state_dict'] if name.endswith('running_mean'))
        checkpoint['best_state_dict'][name].add_(.5)
    damaged=tmp_path/'damaged.pt';torch.save(checkpoint,damaged)
    with pytest.raises(ValueError):collect.checkpoint_audit(damaged,row)


def test_cpu_collection_accepts_byte_identical_relocated_data_without_rewriting_records(campaign_factory,tmp_path):
    raw,data,_=campaign_factory()
    before=collect.inventory(raw);destination=tmp_path/'relocated-data';shutil.copytree(data,destination)
    result=collect.audit_campaign(raw,destination)
    assert collect.inventory(raw)==before
    assert result['plan']['data_root']==str(data)
    assert all(entry['record']['settings']['data_root']==str(data) for entry in result['fits'])
    # The producer's synthetic GPU name and interpreter path are intentionally
    # different. Collection requires compatible builds and exact CPU states.
    assert result['plan']['runtime']['gpu']=='Original synthetic GPU'


def test_collection_detects_a_raw_byte_change_during_restore(campaign_factory,monkeypatch):
    raw,data,_=campaign_factory();restore=study.restore_checkpoint
    def mutate(*args,**kwargs):
        value=restore(*args,**kwargs);(raw/'.writer.lock').write_text('changed');return value
    monkeypatch.setattr(study,'restore_checkpoint',mutate)
    with pytest.raises(ValueError,match='changed during collection'):collect.audit_campaign(raw,data)


def test_full_gate_exact_count_boundaries_and_conjunction_do_not_apply_to_rehearsal(campaign_factory):
    raw,data,_=campaign_factory('full');result=collect.audit_campaign(raw,data)
    def set_score(row,score):
        # All arms have the same million positive pixels, entirely in SAR.
        # This realizes exact decimal IoUs without inconsistent saved fields.
        positive=1_000_000
        rational=Fraction(str(score))*positive
        assert rational.denominator==1
        tp=int(rational)
        saved=dict(loss=.7,auc=.6)
        totals=dict(tp=0,fp=0,fn=0,tn=0)
        for sensor in ('optical','sar'):
            n=row['val'][sensor+'_subset_n_samples']
            counts=dict(tp=tp if sensor=='sar' else 0,fp=0,
                fn=positive-tp if sensor=='sar' else 0,tn=n*128*128-positive if sensor=='sar' else n*128*128)
            assert all(value>=0 for value in counts.values())
            derived=collect.derived_metrics(counts)
            saved.update({sensor+'_subset_'+key:value for key,value in counts.items()})
            saved.update({sensor+'_subset_n_samples':n,sensor+'_subset_iou':derived['flood_iou'],
                sensor+'_subset_dice':derived['flood_dice'],sensor+'_subset_accuracy':derived['accuracy'],
                sensor+'_subset_auc':.6 if sensor=='sar' else 0.})
            for key,value in counts.items():totals[key]+=value
        derived=collect.derived_metrics(totals)
        saved.update(iou=derived['flood_iou'],dice=derived['flood_dice'],accuracy=derived['accuracy'])
        row['val']=saved
        result['metrics'][collect.identity(row)]['val']=collect.check_metrics(saved,230)
    def scores(quantum=.70,frozen=.68,affine=.675,zero=.69):
        targets=dict(quantum=quantum,frozen_quantum=frozen,compact_affine=affine,zero_bottleneck=zero)
        for row in result['summary']['records']:
            set_score(row,targets[row['kind']])
    scores()
    assert collect.analysis(result)['gate']['status']=='pass'
    scores(frozen=.695)  # +0.5pp is below the chosen +1pp screen.
    assert collect.analysis(result)['gate']['status']=='fail'
    scores(zero=.701)
    assert collect.analysis(result)['gate']['status']=='fail'
    scores()
    row=next(row for row in result['summary']['records'] if row['kind']=='frozen_quantum' and row['seed']==77)
    set_score(row,.70)
    gate=collect.analysis(result)['gate']
    assert gate['status']=='fail' and gate['comparisons']['frozen_quantum']['mean_gain_at_least_one_point']
    assert not gate['comparisons']['frozen_quantum']['all_three_positive']  # A tie fails despite mean>1pp.
    scores(quantum=.29,frozen=.28,affine=.28,zero=.28)
    boundary=collect.analysis(result)
    assert boundary['gate']['status']=='pass'  # Exactly29/100−28/100 =1/100, inclusive.
    reported=next(row for row in boundary['paired'] if row['right']=='frozen_quantum' and
        row['phase']=='val' and row['scope']=='pooled' and row['metric']=='flood_iou')
    assert reported['mean']<1.  # Preserve the ordinary floating-point reporting.
    scores(quantum=.289999,frozen=.28,affine=.28,zero=.28)
    below=collect.analysis(result)['gate']
    assert below['status']=='fail'
    assert not below['comparisons']['frozen_quantum']['mean_gain_at_least_one_point']
    assert below['comparisons']['frozen_quantum']['all_three_positive']
    scores(quantum=.29,frozen=.28,affine=.28,zero=.29)
    equal_zero=collect.analysis(result)['gate']
    assert equal_zero['status']=='pass' and equal_zero['quantum_mean_at_least_zero']
    result['plan']['mode']='rehearsal'
    assert collect.analysis(result)['gate']['status']=='not_applicable_rehearsal'
    result['fits'].pop()
    with pytest.raises(ValueError,match='Incomplete'):collect.analysis(result)
