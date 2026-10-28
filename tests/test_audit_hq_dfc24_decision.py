"""Count oracles, data-byte integrity and exposed-data audit boundaries."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import audit_hq_dfc24_decision as audit


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def md5(value):
    return hashlib.md5(value).hexdigest()


def saved_metrics(samples,quality=0):
    """Independent integer pixel construction, including both classes/sensors."""
    optical=71 if samples==205 else 28
    counts=[]
    result={'loss':.5,'auc':.7}
    for sensor,n in (('optical',optical),('sar',samples-optical)):
        tp,fp,fn=n*(128+quality),n*64,n*128
        tn=n*128*128-tp-fp-fn
        counts.append((tp,fp,fn,tn))
        result.update({sensor+'_subset_'+k:v for k,v in zip(('tp','fp','fn','tn'),(tp,fp,fn,tn))})
        result.update({sensor+'_subset_n_samples':n,sensor+'_subset_iou':tp/(tp+fp+fn),
            sensor+'_subset_dice':2*tp/(2*tp+fp+fn),sensor+'_subset_accuracy':(tp+tn)/(n*128*128),sensor+'_subset_auc':.7})
    tp,fp,fn,tn=(sum(values) for values in zip(*counts))
    result.update(iou=tp/(tp+fp+fn),dice=2*tp/(2*tp+fp+fn),accuracy=(tp+tn)/(samples*128*128))
    return result


def seal_packet(root):
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in ('artifact_sha256.json','supplemental_sha256.json')]
    base={str(p.relative_to(root)):audit.file_hash(p) for p in files if p.name!='note.md'}
    extra={'note.md':audit.file_hash(root/'note.md')}
    write_json(root/'artifact_sha256.json',base)
    write_json(root/'supplemental_sha256.json',extra)


@pytest.fixture(scope='module')
def parent_packet(tmp_path_factory):
    """Production-size identities with tiny bytes, not decoded images/models."""
    root=tmp_path_factory.mktemp('dfc24-decision-byte-fixture')
    data,evidence=root/'data',root/'packet'
    data.mkdir();evidence.mkdir()
    train=[str(i) for i in range(1533)]
    test=[str(i) for i in range(205)]  # Same basenames across splits are valid.
    records={'train':[],'test':[]};inventory=[]
    for split,names,directory in (('train',train,'Training'),('test',test,'Testing')):
        for name in names:
            row={'name':name,'image_pix':md5((split+':pixel:'+name).encode()),'mask_pix':md5((split+':maskpixel:'+name).encode())}
            for kind,field,extension in (('images','image','.tif'),('labels','mask','.png')):
                filename=name+extension;payload=(split+':'+field+':'+name).encode()
                relative=str(Path('flood_combined')/directory/kind/filename)
                path=data/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
                row[field+'_file']=filename;row[field+'_md5']=md5(payload)
                inventory.append(relative+':'+md5(payload))
            records[split].append(row)
    manifest=dict(conflict_policy='drop',train=train,test=test,hashes=records)
    write_json(data/'splits/flood_combined.json',manifest)
    write_json(data/'splits/norm_stats.json',{'synthetic':True})
    shutil.copyfile(data/'splits/flood_combined.json',evidence/'input_manifest.json')
    stats=evidence/'input_metadata/splits/norm_stats.json';stats.parent.mkdir(parents=True)
    shutil.copyfile(data/'splits/norm_stats.json',stats)
    inputs=dict(manifest_sha256=audit.file_hash(evidence/'input_manifest.json'),verified_files=3476,
        raw_inventory_sha256=hashlib.sha256('\n'.join(sorted(inventory)).encode()).hexdigest(),
        stats_files_sha256={'splits/norm_stats.json':audit.file_hash(stats)})
    names=set(audit.PRODUCER_SOURCES)|{audit.NEW_PROTOCOL,audit.OLD_PROTOCOL,'scripts/collect_hq_training_controls.py'}
    sources={}
    for name in sorted(names):
        relative='sources/'+name+'.txt';path=evidence/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(stats.read_bytes() if name=='splits/norm_stats.json' else ('# synthetic source '+name+'\n').encode())
        sources[name]=dict(retained_as=relative,sha256=audit.file_hash(path))
    write_json(evidence/'source_manifest.json',sources)
    producer_sources={name:sources[name]['sha256'] for name in audit.PRODUCER_SOURCES}
    legacy_sources={name:value for name,value in producer_sources.items() if name not in {
        'scripts/run_hq_training_controls.py','utilsTorch/hq_training_controls.py','scripts/collect_hq_dfc24_direction.py'}}
    runtime={'device':'cuda:0','packages':{'fixture':'no imports'}}
    old_plan=dict(source_revision='b'*40,source_hashes=legacy_sources,runtime=runtime,input_identity=inputs,
                  protocol_sha256=sources[audit.OLD_PROTOCOL]['sha256'])
    write_json(evidence/'historical_legacy/plan.json',old_plan)
    legacy_inventory={'plan.json':audit.file_hash(evidence/'historical_legacy/plan.json')}
    fresh_inventory={}
    entries=[]
    for kind in audit.CONTROLS:
        for seed,rate in ((42,.001),(42,.003),(77,.003),(123,.003)):
            quality=audit.CONTROLS.index(kind)+audit.SEEDS.index(seed)
            val=saved_metrics(230,quality)
            fit,val_names=train[230:358],train[:230]
            norm=dict(fit_only=True,seed=seed,max_samples=None,n_fit_train_tiles=1303,n_excluded_val_tiles=230,
                fit_names_sha256=hashlib.sha256('\n'.join(train[230:]).encode()).hexdigest()[:16])
            history=[dict(epoch=i,train_loss=None if i==0 else .5,train_schedule=None if i==0 else {'seed':seed,'epoch':i},
                **({'val':copy.deepcopy(val)} if i%2==0 else {})) for i in range(61)]
            row=dict(id='hq_'+kind,kind=kind,seed=seed,learning_rate=rate,best_epoch=0,val=val,history=history,
                fit_names=fit,val_names=val_names,fit_indices=list(range(230,358)),val_indices=list(range(230)),
                n_global_fit=1303,normalization=norm,settings=dict(max_samples=None,_resolved_norm_provenance=norm),
                initial_shared_state_sha256='shared-'+str(seed))
            origin='reused' if kind in ('quantum','mlp') else 'fresh'
            relative=f'fits/hq_{kind}/{rate}/seed{seed}'
            base=evidence/('historical_legacy' if origin=='reused' else 'raw_campaign')/relative
            write_json(base/'metrics.json',row);write_json(base/'history.json',history)
            entry=dict(origin=origin,directory=relative,record=row,
                metrics_sha256=audit.file_hash(base/'metrics.json'),history_sha256=audit.file_hash(base/'history.json'),
                checkpoint_sha256=hashlib.sha256(('synthetic-weight-identity-'+str((kind,seed,rate))).encode()).hexdigest())
            inventory=legacy_inventory if origin=='reused' else fresh_inventory
            for filename in ('metrics.json','history.json','checkpoint.pt'):
                inventory[str(Path(relative)/filename)]=entry[filename.split('.')[0]+'_sha256']
            entries.append(entry)
    old_entries=[dict(directory=e['directory'],record=e['record']) for e in entries if e['origin']=='reused']
    write_json(evidence/'historical_legacy/all_fits.json',old_entries)
    legacy_inventory['all_fits.json']=audit.file_hash(evidence/'historical_legacy/all_fits.json')
    profile=dict(status='passed',source_hashes=producer_sources,runtime=runtime,protocol_sha256=sources[audit.NEW_PROTOCOL]['sha256'])
    write_json(evidence/'raw_campaign/resource_profile.json',profile)
    fresh_inventory['resource_profile.json']=audit.file_hash(evidence/'raw_campaign/resource_profile.json')
    plan=dict(source_revision='a'*40,source_hashes=producer_sources,runtime=runtime,input_identity=inputs,
        protocol_sha256=sources[audit.NEW_PROTOCOL]['sha256'],legacy_plan_sha256=audit.file_hash(evidence/'historical_legacy/plan.json'),
        legacy_source_hashes=legacy_sources,legacy_raw_files=legacy_inventory,
        resource_profile_sha256=audit.file_hash(evidence/'raw_campaign/resource_profile.json'),
        controls=list(audit.CONTROLS),seeds=list(audit.SEEDS),learning_rates=[.001,.003],primary_rate=.003,
        test_policy='All 20 fits audited before 15 new primary replays; exposed test descriptive',
        settings=dict(device='cuda',batch_size=8,epochs=60,fit_cap=128,image_size=128,base_channels=8,eval_every=2,augment=True))
    write_json(evidence/'plan.json',plan)
    selected=[]
    for entry in entries:
        if entry['record']['learning_rate']!=.003:continue
        row=copy.deepcopy(entry['record']);row.update(test=saved_metrics(205,audit.CONTROLS.index(row['kind'])),test_names=test,
            checkpoint=dict(origin=entry['origin'],path=str(Path(entry['directory'])/'checkpoint.pt')))
        binding=dict(plan_sha256=audit.file_hash(evidence/'plan.json'),origin=entry['origin'],
            checkpoint_path=row['checkpoint']['path'],checkpoint_sha256=entry['checkpoint_sha256'],
            metrics_sha256=entry['metrics_sha256'],history_sha256=entry['history_sha256'],
            test_names_sha256=hashlib.sha256(json.dumps(test,separators=(',',':')).encode()).hexdigest())
        name='replays/'+row['id']+'/seed'+str(row['seed'])+'.json'
        write_json(evidence/'raw_campaign'/name,dict(binding=binding,record=row))
        fresh_inventory[name]=audit.file_hash(evidence/'raw_campaign'/name)
        selected.append(row)
    write_json(evidence/'all_fits.json',entries)
    write_json(evidence/'summary.json',dict(plan=plan,records=selected))
    for name in ('plan.json','all_fits.json','summary.json'):
        shutil.copyfile(evidence/name,evidence/'raw_campaign'/name)
        fresh_inventory[name]=audit.file_hash(evidence/name)
    provenance=dict(producing_revision=plan['source_revision'],producer_runtime=runtime,
        collector_source_sha256=sources['scripts/collect_hq_training_controls.py']['sha256'],
        raw_artifact_sha256=dict(fresh=fresh_inventory,legacy=legacy_inventory),
        **{key:plan[key] for key in ('source_hashes','protocol_sha256','legacy_source_hashes','legacy_plan_sha256','input_identity')})
    write_json(evidence/'provenance.json',provenance)
    (evidence/'note.md').write_text('Synthetic supplement, not empirical results.\n')
    seal_packet(evidence)
    return data,evidence


@pytest.fixture
def packet(parent_packet,tmp_path):
    data,original=parent_packet
    evidence=tmp_path/'evidence'
    shutil.copytree(original,evidence)
    return data,evidence


@pytest.fixture(autouse=True)
def fixed_audit_source(monkeypatch):
    monkeypatch.setattr(audit,'source_identity',lambda:dict(revision='c'*40,
        source_sha256=audit.file_hash(Path(audit.__file__)),python_version='fixture',dependencies='Python standard library only'))


def test_hand_computable_binary_counts_and_absent_class_convention():
    values=audit.derived_metrics(dict(tp=3,fp=1,fn=2,tn=4))
    assert values['flood_iou']==.5
    assert values['background_iou']==pytest.approx(4/7)
    assert values['binary_macro_iou']==pytest.approx(15/28)
    assert values['flood_dice']==pytest.approx(2/3)
    assert values['accuracy']==.7
    absent=audit.derived_metrics(dict(tp=0,fp=0,fn=0,tn=10))
    assert absent['flood_iou']==0 and absent['background_iou']==1 and absent['binary_macro_iou']==.5


@pytest.mark.parametrize('bad',(-1,True,1.0))
def test_confusion_counts_require_nonnegative_integers(bad):
    with pytest.raises(ValueError,match='integer confusion'):
        audit.derived_metrics(dict(tp=bad,fp=1,fn=2,tn=4))


@pytest.mark.parametrize('key',('iou','dice','accuracy','optical_subset_iou','sar_subset_accuracy'))
def test_saved_metrics_must_reconcile_with_confusion_counts(key):
    saved=saved_metrics(205)
    saved[key]+=.1
    with pytest.raises(ValueError,match='metric differs from counts'):
        audit.validate_saved_metrics(saved,205,test=True)


def test_sensor_pixel_totals_and_pooled_aggregation_are_checked():
    saved=saved_metrics(205)
    saved['optical_subset_n_samples']+=1
    with pytest.raises(ValueError,match='coverage'):
        audit.validate_saved_metrics(saved,205,test=True)
    saved=saved_metrics(205)
    saved['sar_subset_tn']+=1
    with pytest.raises(ValueError,match='coverage'):
        audit.validate_saved_metrics(saved,205,test=True)


def test_auc_range_is_checked_without_claiming_count_reconstruction():
    saved=saved_metrics(205)
    saved['auc']=.123  # Different ranking statistic with the same threshold counts.
    assert audit.validate_saved_metrics(saved,205,test=True)['pooled']['flood_iou']==saved['iou']
    saved['auc']=1.1
    with pytest.raises(ValueError,match='AUC'):
        audit.validate_saved_metrics(saved,205,test=True)


@pytest.mark.parametrize('corruption',('unbound_source','resigned_source','source_inventory'))
def test_source_hash_and_snapshot_bindings_are_independent(packet,corruption):
    _,evidence=packet
    name='utilsTorch/hq_direction.py'
    if corruption=='source_inventory':
        plan=audit.read_json(evidence/'plan.json');plan['source_hashes'].pop(name);write_json(evidence/'plan.json',plan)
        seal_packet(evidence)
    else:
        (evidence/'sources'/Path(name+'.txt')).write_text('tampered\n')
        if corruption=='resigned_source':seal_packet(evidence)
    with pytest.raises(ValueError,match='hash|source inventory|source binding'):
        audit.verify_packet(evidence)


def test_current_kept_raw_bytes_are_rehashed_not_trusted_from_manifest(parent_packet):
    data,evidence=parent_packet
    plan=audit.read_json(evidence/'plan.json')
    path=data/'flood_combined/Testing/images/0.tif';original=path.read_bytes()
    try:
        path.write_bytes(b'corrupt-current-data')
        with pytest.raises(ValueError,match='Raw data MD5 differs'):
            audit.verify_data(data,evidence,plan)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize('which',('fit_missing','fit_duplicate','selected_missing','selected_duplicate'))
def test_complete_exact_twenty_fit_and_fifteen_selected_grid(packet,which):
    _,evidence=packet;plan=audit.read_json(evidence/'plan.json');manifest=audit.read_json(evidence/'input_manifest.json')
    path=evidence/('all_fits.json' if which.startswith('fit') else 'summary.json')
    payload=audit.read_json(path);rows=payload if isinstance(payload,list) else payload['records']
    if which.endswith('missing'):rows.pop()
    else:rows[-1]=copy.deepcopy(rows[0])
    write_json(path,payload)
    with pytest.raises(ValueError,match='fit grid|selected grid'):
        audit.audit_records(evidence,manifest,plan)


def test_saved_replay_plan_binding_cannot_be_replaced(packet):
    _,evidence=packet
    path=evidence/'raw_campaign/replays/hq_quantum/seed42.json'
    payload=audit.read_json(path);payload['binding']['plan_sha256']='0'*64;write_json(path,payload)
    with pytest.raises(ValueError,match='Saved replay binding'):
        audit.audit_records(evidence,audit.read_json(evidence/'input_manifest.json'),audit.read_json(evidence/'plan.json'))


@pytest.mark.parametrize('which',('fit','selected'))
def test_seed_float_cannot_pass_integer_grid_identity_by_python_equality(packet,which):
    _,evidence=packet
    path=evidence/('all_fits.json' if which=='fit' else 'summary.json')
    payload=audit.read_json(path)
    row=payload[0]['record'] if which=='fit' else payload['records'][0]
    row['seed']=float(row['seed'])
    write_json(path,payload)
    with pytest.raises(ValueError,match='identity'):
        audit.audit_records(evidence,audit.read_json(evidence/'input_manifest.json'),audit.read_json(evidence/'plan.json'))


def test_historical_data_cannot_be_relabeled_as_fresh_holdout(packet):
    _,evidence=packet
    summary=audit.read_json(evidence/'summary.json');plan=summary['plan'];plan['test_policy']='fresh untouched confirmation'
    write_json(evidence/'summary.json',summary)
    with pytest.raises(ValueError,match='exposure policy'):
        audit.audit_records(evidence,audit.read_json(evidence/'input_manifest.json'),plan)


@pytest.mark.parametrize('target',('data','data_child','evidence','evidence_child','ancestor','symlink'))
def test_output_cannot_alias_overlap_or_contain_inputs(parent_packet,tmp_path,target):
    data,evidence=parent_packet
    choices=dict(data=data,data_child=data/'out',evidence=evidence,evidence_child=evidence/'out',ancestor=data.parent)
    if target=='symlink':
        path=tmp_path/'alias';path.symlink_to(data,target_is_directory=True)
    else:path=choices[target]
    with pytest.raises(ValueError,match='overlaps'):
        audit.generate(data,evidence,path)


def test_existing_output_and_input_bytes_are_never_overwritten(parent_packet,tmp_path):
    data,evidence=parent_packet;out=tmp_path/'out';out.mkdir();marker=out/'keep.txt';marker.write_text('keep')
    with pytest.raises(ValueError,match='new or empty'):
        audit.generate(data,evidence,out)
    assert marker.read_text()=='keep'


def test_generations_are_byte_identical_portable_and_exposure_honest(parent_packet,tmp_path):
    data,evidence=parent_packet
    first=audit.generate(data,evidence,tmp_path/'first')
    second=audit.generate(data,evidence,tmp_path/'second')
    assert first==second
    assert {p.name:p.read_bytes() for p in (tmp_path/'first').iterdir()}=={p.name:p.read_bytes() for p in (tmp_path/'second').iterdir()}
    assert first['verification']['historical_validation_metric_blocks']==620
    assert first['verification']['selected_saved_replays']==15
    assert first['verification']['new_fits']==first['verification']['new_inference_calls']==0
    assert first['exposure']['already_exposed_data'] is True
    assert first['exposure']['untouched_event_holdout_established'] is False
    assert all(seed['normalization_or_validation_union']==1533 for seed in first['exposure']['per_seed'])
    assert first['event_metadata_scope']['tiff_headers_inspected'] is False
    assert first['event_metadata_scope']['upstream_metadata_inspected'] is False
    assert len(first['primary_metrics'])==90 and len(first['summary'])==30 and len(first['paired'])==120
    pair=next(r for r in first['paired'] if r['right']=='mlp' and r['split']=='test' and r['scope']=='pooled' and r['metric']=='flood_iou')
    expected=100*(128/320-132/324)
    assert pair['mean']==pytest.approx(expected) and pair['sample_sd']==0 and pair['wins']==0
    for path in (tmp_path/'first').iterdir():
        assert str(data).encode() not in path.read_bytes() and str(evidence).encode() not in path.read_bytes()
        if path.suffix=='.csv':
            rows=list(csv.reader(path.open()))
            assert len(rows[0])==len(set(rows[0])) and all(len(row)==len(rows[0]) for row in rows)
