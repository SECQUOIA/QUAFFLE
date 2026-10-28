#!/usr/bin/env python3
"""Recompute metrics and audit exposure for the completed local HQ controls.

Standard library only. This validates saved records and current kept data bytes;
it neither restores weights nor performs training or inference. An exposed-data
audit cannot establish an untouched event holdout or a prospective decision gate.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ('quantum', 'frozen_quantum', 'zero_bottleneck', 'compact_affine', 'mlp')
SEEDS = (42, 77, 123)
RATE = .003
METRICS = ('flood_iou', 'background_iou', 'binary_macro_iou', 'flood_dice', 'accuracy')
SCOPES = ('pooled', 'optical', 'sar')
PRODUCER_SOURCES = frozenset((
    'scripts/run_hq_dfc24_direction.py', 'utilsTorch/hq_direction.py',
    'scripts/run_real_spatial_pilot.py', 'scripts/run_parameter_budget_comparison.py',
    'utilsTorch/__init__.py', 'utilsTorch/unetTorch.py', 'utilsTorch/utilsTorch.py',
    'tools/training_common.py', 'flood_data.py', 'splits/norm_stats.json',
    'scripts/run_hq_training_controls.py', 'utilsTorch/hq_training_controls.py',
    'scripts/collect_hq_dfc24_direction.py',
))
NEW_PROTOCOL = 'docs/HQ_TRAINING_CONTROLS_PROTOCOL.md'
OLD_PROTOCOL = 'docs/HQ_DFC24_DIRECTION_PROTOCOL.md'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    result = json.loads(path.read_text(), parse_constant=invalid, object_pairs_hook=unique)
    def finite(value):
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite(child) for child in value.values())
        if isinstance(value, list):
            return all(finite(child) for child in value)
        return True
    require(finite(result), 'Nonfinite JSON number')
    return result


def inside(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Expected relative artifact name')
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'Artifact path escapes its root')
    return path


def file_hash(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(bool(re.fullmatch('[0-9a-f]{40}', revision)), 'Cannot establish audit source revision')
    return dict(revision=revision, source_sha256=file_hash(Path(__file__)),
                python_version=platform.python_version(), dependencies='Python standard library only')


def verify_packet(root):
    """Check the parent package, including its later retained supplements."""
    manifests = {}
    covered = set()
    for name in ('artifact_sha256.json', 'supplemental_sha256.json'):
        values = read_json(root/name)
        require(isinstance(values, dict) and values, 'Empty parent artifact manifest')
        require(not covered & set(values), 'Overlapping parent artifact manifests')
        for relative, expected in values.items():
            require(file_hash(inside(root, relative)) == expected, 'Parent artifact hash differs: ' + relative)
        covered.update(values)
        manifests[name] = file_hash(root/name)
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual == covered | set(manifests), 'Missing or unbound parent packet files')
    plan = read_json(root/'plan.json')
    provenance = read_json(root/'provenance.json')
    old_plan = read_json(root/'historical_legacy/plan.json')
    require(set(plan['source_hashes']) == PRODUCER_SOURCES, 'Historical producer source inventory differs')
    for field in ('source_hashes', 'protocol_sha256', 'legacy_source_hashes', 'legacy_plan_sha256', 'input_identity'):
        require(provenance[field] == plan[field], 'Parent provenance binding differs: ' + field)
    require(provenance['producing_revision'] == plan['source_revision'], 'Original producer revision differs')
    require(provenance['producer_runtime'] == plan['runtime'] == old_plan['runtime'], 'Original runtime binding differs')
    require(plan['input_identity'] == old_plan['input_identity'], 'Historical/fresh data identities differ')
    require(file_hash(root/'historical_legacy/plan.json') == plan['legacy_plan_sha256'], 'Historical plan hash differs')
    require(plan['legacy_source_hashes'] == old_plan['source_hashes'], 'Legacy source inventory differs')
    require(plan['legacy_raw_files'] == provenance['raw_artifact_sha256']['legacy'], 'Legacy raw inventory binding differs')
    require(all(plan['source_hashes'].get(name) == value for name, value in old_plan['source_hashes'].items()),
            'Legacy executable bytes differ from reused producer')
    expected_sources = dict(plan['source_hashes'])
    expected_sources.update({NEW_PROTOCOL: plan['protocol_sha256'], OLD_PROTOCOL: old_plan['protocol_sha256'],
                            'scripts/collect_hq_training_controls.py': provenance['collector_source_sha256']})
    sources = read_json(root/'source_manifest.json')
    require(set(sources) == set(expected_sources), 'Retained source snapshot inventory differs')
    for name, expected in expected_sources.items():
        require(sources[name]['sha256'] == expected and
                file_hash(inside(root, sources[name]['retained_as'])) == expected,
                'Retained source binding differs: ' + name)
    require(file_hash(root/'input_manifest.json') == plan['input_identity']['manifest_sha256'],
            'Parent input manifest differs')
    for origin, prefix in (('fresh', 'raw_campaign'), ('legacy', 'historical_legacy')):
        for name, expected in provenance['raw_artifact_sha256'][origin].items():
            if name.endswith('.json'):
                require(file_hash(inside(root/prefix, name)) == expected, 'Parent raw JSON binding differs: '+name)
    for name in ('plan.json', 'all_fits.json', 'summary.json'):
        require(file_hash(root/name) == file_hash(root/'raw_campaign'/name), 'Top-level/raw campaign copy differs: '+name)
    profile = root/'raw_campaign/resource_profile.json'
    require(file_hash(profile) == plan['resource_profile_sha256'], 'Retained profile hash differs')
    recorded_profile = read_json(profile)
    require(recorded_profile['status'] == 'passed' and all(recorded_profile[key] == plan[key]
            for key in ('source_hashes', 'protocol_sha256', 'runtime')), 'Retained profile bindings differ')
    receipt = dict(parent_manifest_sha256=manifests, parent_producing_revision=plan['source_revision'],
        historical_producing_revision=old_plan['source_revision'], parent_plan_sha256=file_hash(root/'plan.json'),
        source_manifest_sha256=file_hash(root/'source_manifest.json'), producer_source_hashes=plan['source_hashes'],
        collector_source_sha256=provenance['collector_source_sha256'], profile_sha256=file_hash(profile),
        verified_parent_artifacts=len(covered), input_identity=plan['input_identity'])
    return plan, receipt


def verify_data(root, evidence, plan):
    """Independently hash every kept image and mask, with no raster imports."""
    manifest_path = root/'splits/flood_combined.json'
    require(file_hash(manifest_path) == plan['input_identity']['manifest_sha256'] ==
            file_hash(evidence/'input_manifest.json'), 'Data/drop manifest identity differs')
    manifest = read_json(manifest_path)
    require(manifest['conflict_policy'] == 'drop', 'Expected conflict-drop manifest')
    require(len(manifest['train']) == len(set(manifest['train'])) == 1533 and
            len(manifest['test']) == len(set(manifest['test'])) == 205, 'Incomplete clean data cohorts')
    # Bare filenames may repeat across Training/Testing or sensor tracks.
    # Decoded-pixel identities, checked below, define cross-split duplicates.
    inventory, stable = [], []
    pixels = {}
    for split, directory in (('train', 'Training'), ('test', 'Testing')):
        kept = set(manifest[split])
        records = manifest['hashes'][split]
        require(len({row['name'] for row in records}) == len(records), 'Duplicate hash-record names')
        chosen = [row for row in records if row['name'] in kept]
        require({row['name'] for row in chosen} == kept, 'Kept names missing hash records')
        pixels[split] = {row['image_pix'] for row in chosen}
        require(len(pixels[split]) == len(chosen), 'Duplicate kept decoded image identities')
        for row in chosen:
            for kind, field in (('images', 'image'), ('labels', 'mask')):
                filename = row[field+'_file']
                require(Path(filename).name == filename, 'Image/mask filename must be a basename')
                name = str(Path('flood_combined')/directory/kind/filename)
                path = inside(root, name)
                before = path.stat()
                actual = file_hash(path, 'md5')
                after = path.stat()
                require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                        (after.st_size, after.st_mtime_ns, after.st_ino), 'Data file changed while hashing')
                require(actual == row[field+'_md5'], 'Raw data MD5 differs: ' + name)
                inventory.append(name+':'+actual)
                stable.append((path, after.st_size, after.st_mtime_ns, after.st_ino))
    require(not pixels['train'] & pixels['test'], 'Decoded image leakage across clean splits')
    raw_hash = hashlib.sha256('\n'.join(sorted(inventory)).encode()).hexdigest()
    require(len(inventory) == plan['input_identity']['verified_files'] == 3476 and
            raw_hash == plan['input_identity']['raw_inventory_sha256'], 'Kept data inventory differs')
    for name, expected in plan['input_identity']['stats_files_sha256'].items():
        require(file_hash(inside(root, name)) == expected == file_hash(inside(evidence/'input_metadata', name)),
                'Normalization artifact differs: ' + name)
    fields = sorted({key for records in manifest['hashes'].values() for row in records for key in row})
    return manifest, dict(verified_raw_image_mask_files=3476, raw_inventory_sha256=raw_hash,
        manifest_sha256=file_hash(manifest_path), normalization_metadata_sha256=plan['input_identity']['stats_files_sha256'],
        manifest_hash_record_fields=fields), stable


def derived_metrics(counts):
    require(set(counts) == {'tp', 'fp', 'fn', 'tn'} and
            all(type(value) is int and value >= 0 for value in counts.values()), 'Invalid integer confusion counts')
    tp, fp, fn, tn = (counts[key] for key in ('tp', 'fp', 'fn', 'tn'))
    ratio = lambda numerator, denominator: numerator/denominator if denominator else 0.
    foreground, background = ratio(tp, tp+fp+fn), ratio(tn, tn+fp+fn)
    return dict(flood_iou=foreground, background_iou=background,
        binary_macro_iou=(foreground+background)/2, flood_dice=ratio(2*tp, 2*tp+fp+fn),
        accuracy=ratio(tp+tn, tp+fp+fn+tn))


def validate_saved_metrics(saved, n_images, test=False):
    scopes = {}
    pooled = Counter(dict(tp=0, fp=0, fn=0, tn=0))
    for sensor in ('optical', 'sar'):
        prefix = sensor+'_subset_'
        counts = {key: saved[prefix+key] for key in pooled}
        derived = derived_metrics(counts)
        samples = saved[prefix+'n_samples']
        require(type(samples) is int and samples >= 0 and sum(counts.values()) == samples*128*128,
                'Sensor sample/pixel coverage differs')
        for field, metric in (('iou', 'flood_iou'), ('dice', 'flood_dice'), ('accuracy', 'accuracy')):
            require(type(saved[prefix+field]) in (int, float) and math.isfinite(saved[prefix+field]) and
                    math.isclose(saved[prefix+field], derived[metric], rel_tol=0, abs_tol=1e-12),
                    'Saved sensor metric differs from counts: '+field)
        require(math.isfinite(saved[prefix+'auc']) and 0 <= saved[prefix+'auc'] <= 1, 'Invalid saved sensor AUC')
        scopes[sensor] = dict(n_samples=samples, **counts, **derived)
        pooled.update(counts)
    require(scopes['optical']['n_samples']+scopes['sar']['n_samples'] == n_images,
            'Pooled sensor sample coverage differs')
    if test:
        require((scopes['optical']['n_samples'], scopes['sar']['n_samples']) == (71, 134),
                'Expected all71 Optical and134 SAR test tiles')
    derived = derived_metrics(dict(pooled))
    for field, metric in (('iou', 'flood_iou'), ('dice', 'flood_dice'), ('accuracy', 'accuracy')):
        require(type(saved[field]) in (int, float) and math.isfinite(saved[field]) and
                math.isclose(saved[field], derived[metric], rel_tol=0, abs_tol=1e-12),
                'Saved pooled metric differs from counts: '+field)
    require(math.isfinite(saved['auc']) and 0 <= saved['auc'] <= 1 and
            math.isfinite(saved['loss']) and saved['loss'] >= 0, 'Invalid saved loss/AUC')
    scopes['pooled'] = dict(n_samples=n_images, **pooled, **derived)
    return scopes


def summarize(values):
    values = list(values)
    require(len(values) == 3, 'Expected exactly three paired seed observations')
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values), n=3)


def audit_records(evidence, manifest, plan):
    entries, summary = read_json(evidence/'all_fits.json'), read_json(evidence/'summary.json')
    require(summary['plan'] == plan, 'Summary/plan mismatch')
    require(plan['controls'] == list(CONTROLS) and plan['seeds'] == list(SEEDS) and
            plan['learning_rates'] == [.001, RATE] and plan['primary_rate'] == RATE, 'Historical protocol grid differs')
    require(plan['test_policy'] == 'All 20 fits audited before 15 new primary replays; exposed test descriptive',
            'Historical test exposure policy differs')
    require(plan['settings'] == dict(device='cuda',batch_size=8,epochs=60,fit_cap=128,
            image_size=128,base_channels=8,eval_every=2,augment=True), 'Historical training budget differs')
    grid = {(kind, seed, RATE) for kind in CONTROLS for seed in SEEDS} | {(kind, 42, .001) for kind in CONTROLS}
    identity = lambda row: (row['kind'], row['seed'], row['learning_rate'])
    require(len(entries) == 20 and {identity(entry['record']) for entry in entries} == grid,
            'Missing, duplicated or unplanned fit grid')
    require(Counter(e['origin'] for e in entries) == Counter(fresh=12, reused=8), 'Fresh/reused fit counts differ')
    inventories=read_json(evidence/'provenance.json')['raw_artifact_sha256']
    by_fit = {identity(e['record']): e for e in entries}
    old_entries=read_json(evidence/'historical_legacy/all_fits.json')
    require(len(old_entries)==8,'Incomplete historical baseline fit records')
    old_by_fit={identity(e['record']):e for e in old_entries}
    fit_metrics, seed_exposure, validation_blocks = [], [], 0
    for entry in entries:
        row = entry['record']
        require(row['id']=='hq_'+row['kind'] and type(row['seed']) is int, 'Invalid saved fit identity')
        require(entry['origin'] == ('reused' if row['kind'] in ('quantum', 'mlp') else 'fresh'),
                'Historical baseline mislabeled as fresh')
        if entry['origin']=='reused':
            require(old_by_fit.get(identity(row))==dict(directory=entry['directory'],record=row),
                    'Historical reused fit record differs from original')
        base = evidence/('historical_legacy' if entry['origin'] == 'reused' else 'raw_campaign')
        directory = inside(base, entry['directory'])
        for filename in ('metrics.json','history.json','checkpoint.pt'):
            key=str(Path(entry['directory'])/filename)
            require(entry[filename.split('.')[0]+'_sha256']==inventories['legacy' if entry['origin']=='reused' else 'fresh'][key],
                    'Fit artifact/producer inventory binding differs')
        for filename in ('metrics.json', 'history.json'):
            require(file_hash(directory/filename) == entry[filename.split('.')[0]+'_sha256'], 'Retained raw fit hash differs')
        require(read_json(directory/'metrics.json') == row and read_json(directory/'history.json') == row['history'],
                'Retained fit/history record differs')
        history = row['history']
        require([p['epoch'] for p in history] == list(range(61)), 'Incomplete sixty-epoch history')
        require(history[0]['train_loss'] is None and history[0]['train_schedule'] is None and
                all(type(p['train_loss']) in (int,float) and p['train_loss']>=0 for p in history[1:]),
                'Incomplete training-loss history')
        points = [p for p in history if 'val' in p]
        require([p['epoch'] for p in points] == list(range(0,61,2)), 'Incomplete validation schedule')
        for point in points:
            validate_saved_metrics(point['val'], 230)
            validation_blocks += 1
        best = max(points, key=lambda p:p['val']['iou'])
        require(row['best_epoch'] == best['epoch'] and row['val'] == best['val'], 'First strict validation best differs')
        for scope, values in validate_saved_metrics(row['val'],230).items():
            fit_metrics.append(dict(control=row['kind'],seed=row['seed'],learning_rate=row['learning_rate'],
                role='historical_primary' if row['learning_rate']==RATE else 'historical_auxiliary',
                origin=entry['origin'],scope=scope,**values))
    train = set(manifest['train'])
    optimization_union, validation_union, normalization_union = set(), set(), set()
    for seed in SEEDS:
        cohort = [e['record'] for e in entries if e['record']['seed']==seed]
        row = cohort[0]
        for field in ('fit_names','val_names','fit_indices','val_indices','normalization','n_global_fit','initial_shared_state_sha256'):
            require(all(r[field]==row[field] for r in cohort), 'Unpaired cohort or shared initialization: '+field)
        require(all([p['train_schedule'] for p in r['history']]==[p['train_schedule'] for p in row['history']]
                    for r in cohort),'Unpaired data/augmentation schedule')
        fit, val = set(row['fit_names']), set(row['val_names'])
        require(len(row['fit_names'])==len(fit)==128 and len(row['val_names'])==len(val)==230 and
                fit<=train-val and val<=train, 'Optimization/validation cohorts differ')
        normalization = train-val
        natural = lambda s:[int(t) if t.isdigit() else t for t in re.split(r'(\d+)',s)]
        names_hash = hashlib.sha256('\n'.join(sorted(normalization,key=natural)).encode()).hexdigest()[:16]
        norm = row['normalization']
        require(len(normalization)==row['n_global_fit']==norm['n_fit_train_tiles']==1303 and
                norm['n_excluded_val_tiles']==230 and norm['fit_only'] is True and norm['max_samples'] is None and
                norm['fit_names_sha256']==names_hash and norm['seed']==seed, 'Normalization exposure/provenance differs')
        require(all(r['normalization']==r['settings']['_resolved_norm_provenance'] and
                    r['settings']['max_samples'] is None for r in cohort), 'Fit-only normalization settings differ')
        optimization_union.update(fit); validation_union.update(val); normalization_union.update(normalization)
        seed_exposure.append(dict(seed=seed,optimization_tiles=128,normalization_fit_tiles=1303,validation_tiles=230,
            normalization_fit_names_sha256_16=names_hash,normalization_or_validation_union=len(normalization|val)))
    records = summary['records']
    require(len(records)==15 and {identity(row) for row in records}=={(kind,seed,RATE) for kind in CONTROLS for seed in SEEDS},
            'Missing, duplicated or unplanned selected grid')
    metrics = []
    replay_paths=set()
    for row in records:
        require(type(row['seed']) is int and row['id']=='hq_'+row['kind'],'Invalid selected identity type')
        entry = by_fit[identity(row)]
        require({key:value for key,value in row.items() if key not in ('test','test_names','checkpoint')}==entry['record'],
                'Selected record differs from fit')
        require(row['test_names']==manifest['test'], 'Ordered test cohort differs')
        replay_path=evidence/'raw_campaign/replays'/row['id']/('seed'+str(row['seed'])+'.json')
        replay_paths.add(replay_path)
        replay = read_json(replay_path)
        binding = dict(plan_sha256=file_hash(evidence/'plan.json'),origin=entry['origin'],
            checkpoint_path=str(Path(entry['directory'])/'checkpoint.pt'),checkpoint_sha256=entry['checkpoint_sha256'],
            metrics_sha256=entry['metrics_sha256'],history_sha256=entry['history_sha256'],
            test_names_sha256=hashlib.sha256(json.dumps(row['test_names'],separators=(',',':')).encode()).hexdigest())
        require(replay==dict(binding=binding,record=row), 'Saved replay binding or record differs')
        require(row['checkpoint']==dict(origin=entry['origin'],path=binding['checkpoint_path']), 'Saved checkpoint reference differs')
        for split in ('val','test'):
            for scope,values in validate_saved_metrics(row[split],230 if split=='val' else 205,test=split=='test').items():
                metrics.append(dict(control=row['kind'],seed=row['seed'],split=split,scope=scope,**values))
    require(set((evidence/'raw_campaign/replays').rglob('*.json'))==replay_paths, 'Extra or missing saved primary replays')
    aggregates, paired = [], []
    values = {(r['control'],r['seed'],r['split'],r['scope']):r for r in metrics}
    for kind in CONTROLS:
        for split in ('val','test'):
            for scope in SCOPES:
                aggregate = dict(control=kind,split=split,scope=scope)
                for metric in METRICS:
                    aggregate.update({metric+'_'+key:value for key,value in
                        summarize(values[kind,seed,split,scope][metric] for seed in SEEDS).items()})
                aggregates.append(aggregate)
    for other in CONTROLS[1:]:
        for split in ('val','test'):
            for scope in SCOPES:
                for metric in METRICS:
                    differences = [100*(values['quantum',seed,split,scope][metric]-values[other,seed,split,scope][metric]) for seed in SEEDS]
                    paired.append(dict(left='quantum',right=other,split=split,scope=scope,metric=metric,
                        **summarize(differences),wins=sum(v>0 for v in differences),
                        **{'seed'+str(seed)+'_pp':value for seed,value in zip(SEEDS,differences)}))
    exposure = dict(already_exposed_data=True,untouched_event_holdout_established=False,clean_training_tiles=1533,
        clean_test_tiles=205,test_optical_tiles=71,test_sar_tiles=134,per_seed=seed_exposure,
        optimization_union=len(optimization_union),validation_union=len(validation_union),normalization_union=len(normalization_union),
        training_tiles_exposed_to_normalization_or_validation_per_seed=1533,
        interpretation='Omission from the128-image optimization cap is not an untouched-data reserve. Existing test outcomes informed subsequent research.')
    return dict(primary_metrics=metrics,summary=aggregates,paired=paired,fit_metrics=fit_metrics,exposure=exposure,
        verification=dict(fits=20,fresh_fits=12,reused_historical_fits=8,paired_seeds=3,selected_saved_replays=15,
            historical_validation_metric_blocks=validation_blocks,checkpoint_tensors_loaded=False,new_fits=0,new_inference_calls=0))


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def generate(data_root,evidence_root,output_dir):
    data,evidence,out=(Path(p).resolve() for p in (data_root,evidence_root,output_dir))
    for root in (data,evidence):
        require(out!=root and not out.is_relative_to(root) and not root.is_relative_to(out), 'Output overlaps data/evidence inputs')
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())), 'Output must be new or empty')
    source=source_identity()
    plan,receipt=verify_packet(evidence)
    manifest,inputs,stable=verify_data(data,evidence,plan)
    result=audit_records(evidence,manifest,plan)
    result.update(provenance=dict(audit_source=source,parent_packet=receipt,current_data=inputs),
        metric_definitions=dict(seed_and_summary_units='Fractions',paired_units='Percentage points',
            flood_iou='TP/(TP+FP+FN)',background_iou='TN/(TN+FP+FN)',binary_macro_iou='(flood IoU+background IoU)/2',
            flood_dice='2TP/(2TP+FP+FN)',accuracy='(TP+TN)/(TP+FP+FN+TN)',zero_denominator_result=0,
            aggregation='Sum Optical/SAR confusion counts before pooled metrics; compute each seed metric before mean/sample SD (n-1). No per-image or cross-seed pixel averaging.',
            AUC='Saved AUC ranges checked; AUC cannot be recomputed from confusion counts.'),
        event_metadata_scope=dict(manifest_record_fields=inputs['manifest_hash_record_fields'],tiff_headers_inspected=False,
            authoritative_scene_event_mapping_verified=False,upstream_metadata_inspected=False,
            conclusion='No verifiably untouched event reserve is established by this retained manifest/exposure evidence. This is not a claim that upstream DFC24 lacks geolocation or event metadata.',
            feature_cluster_warning='Feature similarity and client silos do not establish geographic/event independence.'),
        scope='Existing exposed DFC24 records only; no new fits, inference, prospective gate, superiority or equivalence claim.')
    for path,size,mtime,inode in stable:
        stat=path.stat()
        require((stat.st_size,stat.st_mtime_ns,stat.st_ino)==(size,mtime,inode), 'Input changed during audit')
    require(source_identity()==source,'Audit source/revision changed during execution')
    for name,expected in receipt['parent_manifest_sha256'].items():
        require(file_hash(evidence/name)==expected,'Parent manifest changed during audit')
    out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    for name,key in (('per_seed.csv','primary_metrics'),('summary.csv','summary'),('paired.csv','paired'),('all_fit_validation.csv','fit_metrics')):
        write_csv(out/name,result[key])
    lines=['# Existing DFC24 metric and exposure audit','',
        'This audits already exposed data and saved results. It runs no training, checkpoint inference or prospective resource gate. It establishes neither superiority nor equivalence.','',
        'Twenty historical fit records comprise 12 fresh fits from that campaign and 8 reused fits, with 15 selected saved replays and 3 paired seeds. No fits or replays are new in this audit.','',
        'Raw bytes of all 3,476 kept image/mask files were checked against the conflict-drop manifest. Each seed uses 128 optimization tiles, 1,303 normalization-fit tiles and 230 validation tiles. Every cleaned training tile was used for normalization or validation; an unoptimized tile is not thereby untouched. All 205 test tiles were already evaluated.','',
        'Flood IoU is the foreground metric. Binary macro IoU averages flood and non-flood IoU equally. It is not a five-class land-cover score. Pooled metrics first sum Optical/SAR confusion counts; means and sample SD then aggregate the three seed metrics. Zero denominators yield 0. AUC is not recoverable from confusion counts. Paired differences are formed per seed and reported in percentage points; no confidence or significance claim is made.','',
        '| Control | Validation flood IoU | Validation binary macro IoU | Test flood IoU | Test binary macro IoU | Test pixel accuracy |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    pooled={(r['control'],r['split']):r for r in result['summary'] if r['scope']=='pooled'}
    for kind in CONTROLS:
        val,test=pooled[kind,'val'],pooled[kind,'test']
        fields=[(val,'flood_iou'),(val,'binary_macro_iou'),(test,'flood_iou'),(test,'binary_macro_iou'),(test,'accuracy')]
        lines.append('| '+kind+' | '+' | '.join(f"{r[m+'_mean']:.4f} ± {r[m+'_sample_sd']:.4f}" for r,m in fields)+' |')
    lines+=['','No scene/event-independent or genuinely untouched reserve is established. This tool does not inspect TIFF geotags or upstream metadata. A separate authoritative group map and prior-exposure assessment would be needed before a group-holdout claim; feature-cluster silos are insufficient.','',
        'The JSON binds the actual audit Git revision/source hash, original producer/source snapshots, parent artifact manifests and independently rehashed input inventory. Checkpoint tensors are not restored here; saved prediction metrics are reconciled with retained confusion counts.','',
        'Reproduce with the standard-library CLI from the recorded audit revision, using a new/empty output directory:',
        '','```bash','python scripts/audit_hq_dfc24_decision.py --data-root <retained-dataset> --evidence-root <hq-training-controls-packet> --output-dir <new-output>','```','']
    (out/'README.md').write_text('\n'.join(lines))
    hashes={p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--evidence-root',type=Path,default=ROOT/'benchmarks/hq_training_controls_20261028')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    generate(args.data_root,args.evidence_root,args.output_dir)
    print('Verified 3476 retained data files and recomputed 15 saved primary results; no new inference.')


if __name__=='__main__':
    main()
