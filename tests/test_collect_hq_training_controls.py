"""Synthetic records exercise audit failures; they are not scientific results."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import collect_hq_training_controls as collect

legacy, study = collect.legacy, collect.study


def metrics(n_samples, quality=0, *, test=False):
    # Derive all metrics from hand-defined counts, independently of the exporter.
    result = dict(loss=.7, auc=.6)
    pooled = dict(tp=0, fp=0, fn=0, tn=0)
    for sensor, samples in zip(('optical', 'sar'), (71, 134) if test else (35, 195)):
        counts = dict(tp=samples*(2000+quality), fp=samples*400, fn=samples*800)
        counts['tn'] = samples*128*128 - sum(counts.values())
        for key, value in counts.items():
            result[sensor + '_subset_' + key] = value
            pooled[key] += value
        tp, fp, fn, tn = (counts[key] for key in ('tp', 'fp', 'fn', 'tn'))
        result.update({sensor+'_subset_iou': tp/(tp+fp+fn),
            sensor+'_subset_dice': 2*tp/(2*tp+fp+fn), sensor+'_subset_accuracy': (tp+tn)/sum(counts.values()),
            sensor+'_subset_auc': .6, sensor+'_subset_n_samples': samples})
    assert sum(result[key+'_subset_n_samples'] for key in ('optical', 'sar')) == n_samples
    tp, fp, fn, tn = (pooled[key] for key in ('tp', 'fp', 'fn', 'tn'))
    return dict(result, iou=tp/(tp+fp+fn), dice=2*tp/(2*tp+fp+fn), accuracy=(tp+tn)/sum(pooled.values()))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source, raw, old_root, data_root = (tmp_path/name for name in ('source', 'raw', 'legacy', 'data'))
    sources = set(study.SOURCE_FILES) | set(legacy.study.SOURCE_FILES) | {
        study.PROTOCOL, legacy.PROTOCOL, 'scripts/collect_hq_training_controls.py',
        'scripts/collect_hq_dfc24_direction.py'}
    for name in sources:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic inert source: ' + name)
    names = [f'train{i:04d}' for i in range(1533)]
    manifest = dict(conflict_policy='drop', train=names, test=[f'test{i:04d}' for i in range(205)])
    collect.write(source/'splits/flood_combined.json', manifest)
    collect.write(data_root/'splits/flood_combined.json', manifest)
    inputs = dict(manifest_sha256=collect.digest(source/'splits/flood_combined.json'),
        verified_files=3476, stats_files_sha256={}, raw_inventory_sha256='synthetic-input-identity')
    monkeypatch.setattr(collect, 'ROOT', source)
    monkeypatch.setattr(legacy, 'ROOT', source)
    monkeypatch.setattr(study.pilot, 'verify_inputs', lambda path: copy.deepcopy(inputs))
    settings = dict(device='cuda', batch_size=8, epochs=60, fit_cap=128,
                    image_size=128, base_channels=8, eval_every=2, augment=True)
    runtime = dict(**legacy.current_runtime(), device='cuda:0', gpu='Synthetic fixture',
                   gpu_allocator_fraction=.12, threads=2, interop_threads=2)
    def cohort(seed):
        return dict(fit=list(range(128)), validation=list(range(1303,1533)), all_fit=list(range(1303)),
            dataset=SimpleNamespace(names=names, modalities=['optical' if i%5 == 0 else 'sar' for i in range(1533)]),
            normalization=dict(fit_only=True, seed=seed, fit_names_sha256='synthetic-fit-'+str(seed)))
    monkeypatch.setattr(study.pilot, 'load_training_data', lambda args, seed: cohort(seed))
    args = argparse.Namespace(**settings, data_root=data_root)
    old_entries, entries, paths = [], [], {}
    def one(kind, seed, rate, root, *, reused=False):
        data = cohort(seed)
        model = (legacy.study.build_model if reused else study.build_model)(kind, seed, args)
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        config = (legacy.study.model_configuration if reused else study.model_configuration)(model)
        gradients = dict(min=.01, mean=.02, max=.03, steps=16, nonzero_steps=16)
        core_gradients = dict(gradients) if kind not in ('frozen_quantum','zero_bottleneck') else dict(
            min=0., mean=0., max=0., steps=16, nonzero_steps=0)
        quality = 50-study.KINDS.index(kind)*5 + (10 if rate == .003 else 0) + seed%3
        val = metrics(230, quality)
        history = []
        for epoch in range(61):
            point = dict(epoch=epoch, train_loss=None if not epoch else .7,
                train_schedule=None if not epoch else dict(
                    sha256=study.hq.epoch_schedule(data['fit'],seed,epoch,True)[2],n_samples=128),
                gradient_norms={} if not epoch else dict(core=core_gradients,common=gradients))
            if epoch%2 == 0:
                point['val'] = val if epoch else metrics(230,quality-1)
            history.append(point)
        probe = dict(n_angles=384,min_radians=-.8,max_radians=.7,mean_abs_radians=.3,saturation_fraction=0.)
        diagnostics = dict(initial_angle_probes=[probe],selected_angle_probes=[probe],
            movement=study.movement(model,initial),probe_names=names[:8],
            gradient_norms={group:dict(value,steps=960,nonzero_steps=value['nonzero_steps']*60)
                for group,value in dict(core=core_gradients,common=gradients).items()})
        row = dict(id='hq_'+kind,kind=kind,seed=seed,learning_rate=rate,model_config=config,
            settings=study.fit_settings(kind,seed,rate,args,data),
            total_params=sum(p.numel() for p in model.parameters()),
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            core_params=sum(p.numel() for p in model.bottleneck.core_parameters()),
            initial_state_sha256=study.state_hash(model.state_dict()),
            initial_shared_state_sha256=study.shared_hash(model),selected_state_sha256=study.state_hash(model.state_dict()),
            best_epoch=2,val=val,history=history,diagnostics=diagnostics,elapsed_s=1.,
            fit_indices=data['fit'],val_indices=data['validation'],fit_names=names[:128],val_names=names[1303:],
            normalization=data['normalization'],n_global_fit=1303,fit_sensor_counts=dict(optical=26,sar=102))
        if reused:
            directory = legacy.expected_directory(row)
            checkpoint = dict(format='hq-direction-v1',kind=kind,seed=seed,best_epoch=2,
                settings=row['settings'],model_config=config,best_state_dict=model.state_dict())
        else:
            row.update(parameter_counts=study.counts(model),trainability=study.trainability(model))
            frozen = study.frozen_hash(model)
            diagnostics['frozen_parameters'] = dict(initial_sha256=frozen,selected_sha256=frozen,unchanged=True)
            diagnostics['angle_probe_scope'] = ('inactive_counterfactual'
                if kind == 'zero_bottleneck' else 'active_fixed_probe')
            directory = str(Path('fits')/row['id']/str(rate)/('seed'+str(seed)))
            checkpoint = dict(format='hq-training-controls-v1',**{key:row[key] for key in
                ('kind','seed','best_epoch','settings','model_config','trainability','parameter_counts')},
                best_state_dict=model.state_dict())
        path = root/directory
        collect.write(path/'metrics.json',row)
        collect.write(path/'history.json',history)
        torch.save(checkpoint,path/'checkpoint.pt')
        entry = dict(origin='reused' if reused else 'fresh',directory=directory,record=row,
            **{name.split('.')[0]+'_sha256':collect.digest(path/name) for name in
                ('metrics.json','history.json','checkpoint.pt')})
        paths[collect.identity(row)] = path
        if reused:
            old_entries.append(dict(directory=directory,record=row))
        entries.append(entry)
        return row
    old_tuning = [one(kind,42,rate,old_root,reused=True) for kind in legacy.study.KINDS for rate in study.RATES]
    for kind in legacy.study.KINDS:
        for seed in study.SEEDS[1:]:
            one(kind,seed,.003,old_root,reused=True)
    old_plan = dict(settings=settings,seeds=list(study.SEEDS),learning_rates=list(study.RATES),
        source_hashes={name:collect.digest(source/name) for name in legacy.study.SOURCE_FILES},
        source_revision='synthetic-old-not-an-experiment',input_identity=inputs,runtime=runtime,
        protocol_sha256=collect.digest(source/legacy.PROTOCOL),
        test_policy='All eight fits finish before six selected checkpoint tests; test already exposed/descriptive',
        augmentation_policy='Stateless D4 by(seed,epoch,original index), independent seeded sample order; train only',
        normalization_policy='All 1303 global fit images, validation excluded; optimization cap separate')
    old_records = []
    for entry in old_entries:
        if entry['record']['learning_rate'] != .003:
            continue
        row = copy.deepcopy(entry['record'])
        row.update(test=metrics(205,40-study.KINDS.index(row['kind'])*5,test=True),test_names=manifest['test'],
            knockout={mode:metrics(205,-index,test=True) for index,mode in enumerate(legacy.study.MODES)},
            checkpoint=str(Path(entry['directory'])/'checkpoint.pt'))
        collect.write(old_root/entry['directory']/'selected_test.json',row)
        old_records.append(row)
    old_audits = []
    for entry in old_entries:
        row = entry['record']
        model,audit = legacy.study.restore_checkpoint(old_root/entry['directory']/'checkpoint.pt',row)
        old_audits.append(dict(id=row['id'],seed=row['seed'],learning_rate=row['learning_rate'],**audit))
        del model
    for filename,value in dict(plan=old_plan,summary=dict(plan=old_plan,selection=dict(hq_quantum=.003,hq_mlp=.003),
        records=old_records,analysis=legacy.study.analysis(old_records)),all_fits=old_entries,
        tuning_candidates=old_tuning,checkpoint_audit=old_audits,
        validation_selection={name:dict(learning_rate=row['learning_rate'],best_epoch=2,val_iou=row['val']['iou'])
            for name,row in legacy.study.pilot.choose_rates(old_tuning).items()}).items():
        collect.write(old_root/(filename+'.json'),value)
    for kind in study.NEW_KINDS:
        for seed,rate in [(42,rate) for rate in study.RATES]+[(seed,.003) for seed in study.SEEDS[1:]]:
            one(kind,seed,rate,raw)
    plan = dict(settings=settings,controls=list(study.KINDS),seeds=list(study.SEEDS),learning_rates=list(study.RATES),
        primary_rate=.003,source_hashes={name:collect.digest(source/name) for name in study.SOURCE_FILES},
        source_revision='synthetic-new-not-an-experiment',input_identity=inputs,runtime=runtime,
        protocol_sha256=collect.digest(source/study.PROTOCOL),legacy_root=str(old_root),
        legacy_plan_sha256=collect.digest(old_root/'plan.json'),legacy_source_hashes=old_plan['source_hashes'],
        legacy_raw_files=collect.file_inventory(old_root),
        primary_policy='Fixed 0.003 for all controls/seeds; no rate reselection',
        test_policy='All 20 fits audited before 15 new primary replays; exposed test descriptive',
        auxiliary_policy='Rate0.001 on seed42 for all five controls; sensitivity only')
    profile = dict(status='passed',source_hashes=plan['source_hashes'],protocol_sha256=plan['protocol_sha256'],
        runtime=runtime,cases=[dict(kind=kind,batch_size=8,strict_restore=True,frozen_unchanged=True) for kind in study.NEW_KINDS])
    collect.write(raw/'resource_profile.json',profile)
    plan['resource_profile_sha256'] = collect.digest(raw/'resource_profile.json')
    collect.write(raw/'plan.json',plan)
    audits, records = [], []
    for entry in entries:
        row = entry['record']
        model,audit = study.restore_checkpoint(paths[collect.identity(row)]/'checkpoint.pt',row)
        audits.append(dict(id=row['id'],seed=row['seed'],learning_rate=row['learning_rate'],origin=entry['origin'],**audit))
        del model
        if row['learning_rate'] != .003:
            continue
        row = copy.deepcopy(row)
        row.update(test=metrics(205,40-study.KINDS.index(row['kind'])*5,test=True),test_names=manifest['test'],
            checkpoint=dict(origin=entry['origin'],path=str(Path(entry['directory'])/'checkpoint.pt')))
        collect.write(raw/'replays'/row['id']/('seed'+str(row['seed'])+'.json'),
            dict(binding=study.replay_binding(entry,collect.digest(raw/'plan.json'),manifest['test']),record=row))
        records.append(row)
    for filename,value in dict(plan=plan,all_fits=entries,fit_progress=entries,checkpoint_audit=audits,
        summary=dict(plan=plan,records=records,analysis=study.analysis(records))).items():
        collect.write(raw/(filename+'.json'),value)
    collect.write(raw/'fit_boundary.json',dict(completed=True,fits=20,fresh=12,reused=8,primary=15,
        all_fits_sha256=collect.digest(raw/'all_fits.json'),checkpoint_audit_sha256=collect.digest(raw/'checkpoint_audit.json')))
    return raw,data_root,source,old_root


def test_complete_fixed_rate_grid_restores_all_checkpoints_and_trainability(campaign):
    raw,data,_,_ = campaign
    result = collect.audit_campaign(raw,data)
    assert len(result['fits']) == len(result['checkpoint_audit']) == 20
    assert len(result['summary']['records']) == 15
    assert sum(entry['origin'] == 'fresh' for entry in result['fits']) == 12
    contrasts = collect.comparison(result)
    assert len(contrasts['paired']) == 24
    assert all(row['wins'] == 3 and row['common_learning_rate'] == .003 for row in contrasts['paired'])
    # For synthetic pooled test counts the quantum/frozen contrast has this
    # direct independent derivation, including the conversion to percentage points.
    row = next(row for row in contrasts['paired'] if row['right'] == 'frozen_quantum' and
        row['split'] == 'test' and row['metric'] == 'iou')
    assert row['mean'] == pytest.approx(100*(2040/3240-2035/3235),abs=1e-14)
    tables = collect.tables(result,contrasts)
    assert {key:len(value) for key,value in tables.items()} == dict(summary=5,per_seed=15,paired=24,tuning=20,trainability=40)
    assert sum(row['role'] == 'auxiliary_rate' for row in tables['tuning']) == 5
    inactive = [row for row in tables['trainability'] if row['control'] in ('frozen_quantum','zero_bottleneck') and row['group'] == 'core']
    assert len(inactive) == 8 and all(row['gradient_max'] == row['movement_l2'] == 0 for row in inactive)


@pytest.mark.parametrize('damage',('source','runtime','primary_rate','boundary','legacy_bytes','missing_fit',
    'history','test_counts','primary_rate_selection','replay','replay_binding','replay_failure',
    'checkpoint','extra_fit','resource_profile','missing_profile'))
def test_corrupt_or_incomplete_evidence_is_rejected(campaign,damage):
    raw,data,source,old_root = campaign
    summary = collect.read(raw/'summary.json')
    selected = next(row for row in summary['records'] if row['kind'] == 'frozen_quantum')
    directory = raw/Path(selected['checkpoint']['path']).parent
    if damage == 'source':
        (source/'utilsTorch/hq_training_controls.py').write_text('stale')
    elif damage in ('runtime','primary_rate'):
        plan = collect.read(raw/'plan.json')
        if damage == 'runtime':
            plan['runtime']['packages']['torch'] = 'different'
        else:
            plan['primary_rate'] = .001
        summary['plan'] = plan
        collect.write(raw/'plan.json',plan)
        collect.write(raw/'summary.json',summary)
    elif damage == 'boundary':
        boundary = collect.read(raw/'fit_boundary.json'); boundary['fresh'] = 11
        collect.write(raw/'fit_boundary.json',boundary)
    elif damage == 'legacy_bytes':
        (old_root/'all_fits.json').write_text('[]')
    elif damage == 'missing_fit':
        (directory/'history.json').unlink()
    elif damage == 'history':
        history = collect.read(directory/'history.json'); history.pop()
        collect.write(directory/'history.json',history)
    elif damage == 'test_counts':
        selected['test']['sar_subset_iou'] += .01
        replay = raw/'replays'/selected['id']/('seed'+str(selected['seed'])+'.json')
        payload = collect.read(replay); payload['record'] = selected; collect.write(replay,payload)
        collect.write(raw/'summary.json',summary)
    elif damage == 'primary_rate_selection':
        selected['learning_rate'] = .001
        collect.write(raw/'summary.json',summary)
    elif damage == 'replay':
        replay = raw/'replays'/selected['id']/('seed'+str(selected['seed'])+'.json')
        payload = collect.read(replay); payload['record']['test_names'].reverse(); collect.write(replay,payload)
    elif damage == 'replay_binding':
        replay = raw/'replays'/selected['id']/('seed'+str(selected['seed'])+'.json')
        payload = collect.read(replay); payload['binding']['checkpoint_sha256'] = 'stale'; collect.write(replay,payload)
    elif damage == 'replay_failure':
        collect.write(raw/'replay_failures'/selected['id']/'seed42.json',dict(error='preserved'))
    elif damage == 'checkpoint':
        (directory/'checkpoint.pt').write_bytes(b'bad checkpoint')
    elif damage == 'resource_profile':
        (raw/'resource_profile.json').write_text('{}')
    elif damage == 'missing_profile':
        (raw/'resource_profile.json').unlink()
    else:
        collect.write(raw/'unplanned/metrics.json',{})
    with pytest.raises((ValueError,FileNotFoundError,RuntimeError,EOFError,KeyError,IndexError)):
        collect.audit_campaign(raw,data)


def test_export_preserves_both_histories_and_regenerates_exact_bytes(campaign,tmp_path):
    raw,data,_,old_root = campaign
    result = collect.audit_campaign(raw,data)
    first,second = tmp_path/'out1',tmp_path/'out2'
    collect.export(result,raw,data,first)
    collect.export(result,raw,data,second)
    hashes = collect.read(first/'artifact_sha256.json')
    assert hashes == collect.read(second/'artifact_sha256.json')
    assert set(hashes) == {str(path.relative_to(first)) for path in first.rglob('*') if path.is_file()}-{'artifact_sha256.json'}
    assert all(collect.digest(first/name) == value for name,value in hashes.items())
    assert (first/'summary.json').read_bytes() == (raw/'summary.json').read_bytes()
    assert (first/'historical_legacy/summary.json').read_bytes() == (old_root/'summary.json').read_bytes()
    assert len(list((first/'raw_campaign/fits').rglob('history.json'))) == 12
    assert len(list((first/'historical_legacy').rglob('history.json'))) == 8
    assert len(list((first/'raw_campaign/replays').rglob('*.json'))) == 15
    assert len(list((first/'historical_legacy').rglob('selected_test.json'))) == 6
    assert not list(first.rglob('*.pt'))
    assert len(list((first/'plots').glob('*.png'))) == len(list((first/'plots').glob('*.pdf'))) == 3
    provenance = collect.read(first/'provenance.json')
    assert (provenance['fresh_fits'],provenance['reused_fits'],provenance['independent_inference_replays']) == (12,8,0)
    with pytest.raises(ValueError,match='empty'):
        collect.export(result,raw,data,first)


def test_output_cannot_alias_either_campaign(campaign):
    raw,data,_,old_root = campaign
    result = collect.audit_campaign(raw,data)
    with pytest.raises(ValueError,match='aliases'):
        collect.export(result,raw,data,old_root/'new-output')


@pytest.mark.parametrize('damage',('trainability','batchnorm','mode'))
def test_checkpoint_audit_rejects_restoration_semantic_damage(campaign,tmp_path,monkeypatch,damage):
    raw,_,_,_ = campaign
    kind = 'zero_bottleneck' if damage == 'mode' else 'frozen_quantum'
    row = next(entry['record'] for entry in collect.read(raw/'all_fits.json')
        if entry['record']['kind'] == kind and entry['record']['seed'] == 42)
    path = raw/'fits'/row['id']/str(row['learning_rate'])/'seed42/checkpoint.pt'
    if damage == 'mode':
        restore = study.restore_checkpoint
        def wrong_mode(*args,**kwargs):
            model,audit = restore(*args,**kwargs)
            model.bottleneck.mode = 'on'
            return model,audit
        monkeypatch.setattr(study,'restore_checkpoint',wrong_mode)
        with pytest.raises(ValueError,match='mode'):
            collect.audit_checkpoint(path,row)
        return
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    if damage == 'trainability':
        name = next(name for name in row['trainability'] if name.startswith('bottleneck.core.'))
        row['trainability'][name] = True
        checkpoint['trainability'][name] = True
        message = 'semantics'
    else:
        name = next(name for name in checkpoint['best_state_dict'] if name.endswith('running_mean'))
        checkpoint['best_state_dict'][name].add_(.5)
        message = 'state'
    damaged = tmp_path/'damaged.pt'
    torch.save(checkpoint,damaged)
    with pytest.raises(ValueError,match=message):
        collect.audit_checkpoint(damaged,row)
