#!/usr/bin/env python3
"""Render descriptive HQ figures from a frozen, audited numeric cache.

This module deliberately imports no producer, model, fitter or Torch module.
Changing layout therefore never requires repeating checkpoint inference.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

CONTROLS = ('quantum','frozen_quantum','zero_bottleneck','compact_affine')
SEEDS = (42,77,123)
LABELS = ('Learned quantum','Frozen quantum','Zero bottleneck','Compact affine')
COLORS = ('#0072B2','#E69F00','#009E73','#CC79A7')
ERROR_COLORS = ((0.,.62,.45),(.84,.37,0.),(.80,.47,.65))
PROTOCOL = 'docs/HQ_PUBLICATION_FIGURES_PROTOCOL.md'
SELF = Path(__file__).resolve()
RULE = dict(reference='compact_affine',seed=42,learning_rate=.003,
    minimum_ground_truth_flood_fraction=.05,sensors=['optical','sar'],
    ordering=['affine_foreground_iou','validation_index'],
    validation_index='Ordinal position in the saved ordered230 validation subset',
    worst_rank=0,lower_median_rank='(n-1)//2',retain_duplicate_selections=True)


def require(value,message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


LOADED_SELF = digest(SELF)


def read(path):
    def unique(pairs):
        result = {}
        for key,value in pairs:
            require(key not in result,'Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    value = json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=invalid)
    def finite(item):
        if isinstance(item,float):
            require(math.isfinite(item),'Nonfinite JSON number')
        elif isinstance(item,dict):
            for entry in item.values(): finite(entry)
        elif isinstance(item,list):
            for entry in item: finite(entry)
    finite(value)
    return value


def write(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def inventory(root):
    return {str(path.relative_to(root)):digest(path) for path in sorted(Path(root).rglob('*')) if path.is_file()}


def inside(root,name):
    path = (root/name).resolve()
    require(path.is_relative_to(root) and path != root,'Unsafe cache member path')
    return path


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(f'{value.dtype}:{tuple(value.shape)}\n'.encode()+value.tobytes()).hexdigest()


def arrays(path,descriptor,expected):
    require(digest(path) == descriptor['sha256'],'Numeric archive hash differs')
    require(set(descriptor['arrays']) == set(expected),'Numeric array inventory differs')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(sorted(item.filename for item in members) == sorted(name+'.npy' for name in expected),
                'Unexpected or duplicate numeric archive member')
        require(sum(item.file_size for item in members) < 32*1024*1024,'Numeric archive exceeds bounded geometry')
        for item in members:
            name = item.filename[:-4]
            shape,dtype = expected[name]
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1,0),(2,0)),'Unsupported numeric header')
                header = (np.lib.format.read_array_header_1_0(stream) if version == (1,0)
                          else np.lib.format.read_array_header_2_0(stream))
                require(header == (shape,False,np.dtype(dtype)),'Numeric archive shape/dtype/order differs: '+name)
    with np.load(path,allow_pickle=False) as archive:
        result = {name:archive[name] for name in expected}
    for name,value in result.items():
        spec = descriptor['arrays'][name]
        require(spec == dict(dtype=str(value.dtype),shape=list(value.shape),sha256=array_hash(value)),
                'Numeric array identity differs: '+name)
        require(bool(np.isfinite(value).all()),'Nonfinite display or mask values')
        if name in ('predictions','targets'):
            require(bool(((value == 0) | (value == 1)).all()),'Nonbinary segmentation mask')
    return result


def counts(pred,target):
    # Boolean intersections are independently evaluated from the cached masks.
    pred,target = pred.astype(bool),target.astype(bool)
    tp = int(np.count_nonzero(pred & target)); fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target)); tn = int(np.count_nonzero(~pred & ~target))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn)


def iou(row):
    total = row['tp']+row['fp']+row['fn']
    return row['tp']/total if total else 0.


def normalized_display(channels):
    """Use the fixed original normalization range, without per-image stretching."""
    return np.clip((channels.transpose(1,2,0)+1)/2,0,1)


def independently_select(affine,target,names,sensors,indices):
    result = []
    for sensor in ('optical','sar'):
        eligible = [index for index in range(len(names)) if sensors[index] == sensor and np.mean(target[index]) >= .05]
        eligible.sort(key=lambda index:(iou(counts(affine[index],target[index])),index))
        for role,rank in (('worst',0),('lower_median',(len(eligible)-1)//2)):
            row = dict(slot=len(result),sensor=sensor,role=role,eligible_count=len(eligible))
            if eligible:
                index = eligible[rank]
                row.update(status='selected',validation_index=index,global_index=indices[index],name=names[index],
                    affine_foreground_iou=iou(counts(affine[index],target[index])),
                    ground_truth_flood_fraction=float(np.mean(target[index])))
            else:
                row['status'] = 'empty_eligible_set'
            result.append(row)
    return result


def recompute(payload,names,sensors):
    target,predictions = payload['targets'],payload['predictions']
    per_image,pooled,disagreements = [],[],[]
    for control,pred in zip(CONTROLS,predictions):
        for index in range(len(names)):
            row = counts(pred[index],target[index])
            per_image.append(dict(control=control,validation_index=index,name=names[index],sensor=sensors[index],
                **row,foreground_iou=iou(row),ground_truth_flood_fraction=float(np.mean(target[index]))))
        for sensor in ('pooled','optical','sar'):
            subset = list(range(len(names))) if sensor == 'pooled' else [i for i,s in enumerate(sensors) if s == sensor]
            row = counts(pred[subset],target[subset])
            pooled.append(dict(control=control,sensor=sensor,n_samples=len(subset),**row,foreground_iou=iou(row)))
    for left in range(4):
        for right in range(left+1,4):
            for index in range(len(names)):
                changed = int(np.count_nonzero(predictions[left,index] != predictions[right,index]))
                disagreements.append(dict(left=CONTROLS[left],right=CONTROLS[right],validation_index=index,
                    name=names[index],sensor=sensors[index],different_pixels=changed,fraction=changed/16384))
    return dict(per_image=per_image,pooled=pooled,disagreements=disagreements)


def load_cache(cache_dir):
    root = Path(cache_dir).resolve()
    require(not (root/'failure.json').exists(),'Failed prediction attempt cannot be rendered')
    expected = read(root/'artifact_sha256.json')
    actual = inventory(root); actual.pop('artifact_sha256.json',None)
    require(actual == expected,'Cache file inventory or bytes differ')
    cache = read(root/'cache.json')
    require(cache['format'] == 'hq-publication-validation-cache-v1' and cache['complete'] is True,
            'Incomplete or unsupported prediction cache')
    require(cache['controls'] == list(CONTROLS),'Control order differs')
    names,sensors,indices = (cache[key] for key in ('names','sensors','global_indices'))
    require(len(names) == len(set(names)) == 230 and all(isinstance(name,str) for name in names),'Validation names differ')
    require(len(sensors) == 230 and set(sensors) == {'optical','sar'},'Validation sensors differ')
    require(len(indices) == len(set(indices)) == 230 and all(type(i) is int and 0 <= i < 1533 for i in indices),
            'Validation global indices differ')
    documents = {}
    for name in ('plan','selection','histories','metrics','parity','source_manifest'):
        require(digest(root/(name+'.json')) == cache[name+'_sha256'],'Bound JSON differs: '+name)
        documents[name] = read(root/(name+'.json'))
    plan,selection,histories = (documents[key] for key in ('plan','selection','histories'))
    require(plan['format'] == 'hq-validation-prediction-plan-v1' and plan['controls'] == list(CONTROLS)
            and plan['seeds'] == list(SEEDS) and plan['primary_learning_rate'] == .003 and plan['prediction_seed'] == 42,
            'Frozen population differs')
    require(plan['selection_rule'] == selection['rule'] == RULE,'Fixed example-selection rule differs')
    require(selection['format'] == 'hq-validation-example-selection-v1' and selection['plan_sha256'] == cache['plan_sha256'],
            'Selection plan binding differs')
    require((plan['validation_names'],plan['validation_sensors'],plan['validation_global_indices']) == (names,sensors,indices),
            'Cohort bindings differ')
    require(plan['original_campaign'] == dict(fits=20,fresh=12,reused=8) and
            plan['figure_history_population'] == dict(fits=12,fresh=9,reused=3),'Fresh/reused scope differs')
    require(plan['inference_order'] == ['compact_affine',*CONTROLS[:-1]],'Inference order differs')
    require(plan['settings']['batch_size'] == 8 and plan['settings']['image_size'] == 128 and
            plan['settings']['model_eval'] is True and plan['settings']['sequential_validation'] is True,
            'Inference settings differ')
    sources = documents['source_manifest']
    require(set(sources) == set(plan['source_hashes']) | {PROTOCOL},'Archived source inventory differs')
    for name,item in sources.items():
        require(digest(inside(root,item['retained_as'])) == item['sha256'] ==
                (plan['protocol_sha256'] if name == PROTOCOL else plan['source_hashes'][name]),'Archived source differs: '+name)
    require(set(plan['checkpoints']) == set(CONTROLS) and selection['affine_checkpoint_sha256'] ==
            plan['checkpoints']['compact_affine']['checkpoint_sha256'],'Checkpoint binding differs')
    require(cache['payload_file'] == 'validation_predictions.npz','Unexpected payload path')
    payload = arrays(root/cache['payload_file'],cache['payload'],dict(predictions=((4,230,128,128),'uint8'),
        targets=((230,128,128),'uint8'),display_inputs=((4,3,128,128),'float32'),display_valid=((4,),'bool')))
    affine = arrays(root/'affine_predictions.npz',selection['affine_archive'],
                    dict(predictions=((230,128,128),'uint8'),targets=((230,128,128),'uint8')))
    require(np.array_equal(affine['predictions'],payload['predictions'][3]) and
            np.array_equal(affine['targets'],payload['targets']),'Affine phase boundary differs')
    examples = independently_select(affine['predictions'],payload['targets'],names,sensors,indices)
    require(selection['examples'] == examples,'Selected identities/ranks differ')
    for item in examples:
        slot = item['slot']; selected = item['status'] == 'selected'
        require(bool(payload['display_valid'][slot]) == selected,'Display slot validity differs')
        if not selected:
            require(not payload['display_inputs'][slot].any(),'Empty selection contains display input')
    calculated = recompute(payload,names,sensors)
    require(calculated == documents['metrics'],'Saved mask counts/disagreements differ')
    parity = documents['parity']
    require(parity['passed'] is True and parity['loss_or_auc_recovery'] is False and
            len(parity['records']) == 4 and {row['control'] for row in parity['records']} == set(CONTROLS),
            'Original-count parity receipt incomplete')
    for record in parity['records']:
        require(record['exact_sensor_counts'] is True,'Original-count parity not exact')
        for sensor in ('pooled','optical','sar'):
            observed = next(row for row in calculated['pooled'] if row['control'] == record['control'] and row['sensor'] == sensor)
            observed = {key:observed[key] for key in ('n_samples','tp','fp','fn','tn')}
            require(observed == (record['pooled'] if sensor == 'pooled' else record['sensors'][sensor]),
                    'Parity receipt and masks differ')
    require(len(histories) == 12 and {(row['control'],row['seed']) for row in histories} ==
            {(control,seed) for control in CONTROLS for seed in SEEDS},'History grid differs')
    for row in histories:
        require(row['learning_rate'] == .003 and row['origin'] == ('reused' if row['control'] == 'quantum' else 'fresh'),
                'History rate/origin differs')
        points = row['history']
        require([p['epoch'] for p in points] == list(range(61)),'History epoch coverage differs')
        require(points[0]['train_loss'] is None and all(type(p['train_loss']) in (int,float) for p in points[1:]),
                'Training losses incomplete')
        validated = [p for p in points if p.get('val') is not None]
        require([p['epoch'] for p in validated] == list(range(0,61,2)),'Validation schedule differs')
        best = max(validated,key=lambda p:p['val']['iou'])
        require(row['best_epoch'] == best['epoch'],'Selected epoch is not earliest best validation IoU')
        if row['seed'] == 42:
            require(row['checkpoint_sha256'] == plan['checkpoints'][row['control']]['checkpoint_sha256'] and
                    row['best_epoch'] == plan['checkpoints'][row['control']]['best_epoch'], 'History/checkpoint selection differs')
    return dict(root=root,cache=cache,**documents,payload=payload)


def csv_file(path,rows,fields=None):
    fields = fields or list(rows[0])
    with path.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def render(cache_dir,output_dir):
    require(digest(SELF) == LOADED_SELF,'Renderer source changed after import')
    data = load_cache(cache_dir)
    root,out = data['root'],Path(output_dir).resolve()
    require(out != root and not out.is_relative_to(root) and not root.is_relative_to(out),'Render output aliases cache')
    require(not out.exists() or not any(out.iterdir()),'Render output must be new or empty')
    out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'svg.hashsalt':'hq-fixed-validation','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None,'Creator':'HQ frozen-cache renderer'})
        fig.savefig(out/(name+'.png'),dpi=180,metadata={'Software':'HQ frozen-cache renderer'})
        plt.close(fig)
    fig,axes = plt.subplots(3,2,figsize=(7.2,7.5),sharex=True)
    flat = []
    best_rows = []
    for row in data['histories']:
        control,seed = row['control'],row['seed']; c = CONTROLS.index(control); r = SEEDS.index(seed)
        for col,field in enumerate(('train_loss','val')):
            points = [point for point in row['history'] if point.get(field) is not None]
            axes[r,col].plot([p['epoch'] for p in points],
                [p[field]['iou'] if field == 'val' else p[field] for p in points],color=COLORS[c],label=LABELS[c],linewidth=1.3)
        point = row['history'][row['best_epoch']]
        axes[r,1].plot(point['epoch'],point['val']['iou'],'o',color=COLORS[c],markersize=4)
        best_rows.append({key:row[key] for key in ('control','seed','origin','learning_rate','best_epoch','checkpoint_sha256','history_sha256')})
        for point in row['history']:
            flat.append(dict(control=control,seed=seed,origin=row['origin'],learning_rate=.003,epoch=point['epoch'],
                train_loss=point['train_loss'],validation_foreground_iou=point['val']['iou'] if point.get('val') else None,
                selected_best_epoch=point['epoch'] == row['best_epoch']))
    for r,seed in enumerate(SEEDS):
        for col in range(2):
            axes[r,col].set(xlim=(-1,61),xlabel='Epoch',ylabel='Training Dice+CE loss' if col == 0 else 'Validation foreground IoU',
                            title=f'Seed {seed}');axes[r,col].grid(alpha=.2)
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.012),fontsize=8)
    fig.suptitle('Fixed-rate HQ controls: 60 epochs, learning rate 0.003',y=.995,fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.975),h_pad=1.6,w_pad=1.5);save(fig,'learning_curves')
    payload = data['payload']; examples = data['selection']['examples']; displayed = []
    fig,axes = plt.subplots(4,6,figsize=(7.2,6.8))
    for slot,item in enumerate(examples):
        for ax in axes[slot]:
            ax.set_xticks([]);ax.set_yticks([])
        role = 'lower median' if item['role'] == 'lower_median' else item['role']
        axes[slot,0].set_ylabel(f"{item['sensor'].upper()}\n{role}",fontsize=7.5,labelpad=5)
        if item['status'] != 'selected':
            for ax in axes[slot]: ax.text(.5,.5,'No eligible\ntile',ha='center',va='center',transform=ax.transAxes,fontsize=7)
            continue
        index = item['validation_index']; image = normalized_display(payload['display_inputs'][slot])
        target = payload['targets'][index].astype(bool)
        axes[slot,0].imshow(image);axes[slot,0].set_xlabel(f'Validation index {index}',fontsize=7,labelpad=3)
        axes[slot,1].imshow(target,cmap='gray',vmin=0,vmax=1)
        for c,control in enumerate(CONTROLS):
            prediction = payload['predictions'][c,index].astype(bool)
            overlay = image*.28
            overlay[prediction & target] = ERROR_COLORS[0]
            overlay[prediction & ~target] = ERROR_COLORS[1]
            overlay[~prediction & target] = ERROR_COLORS[2]
            values = counts(prediction,target)
            axes[slot,c+2].imshow(overlay);axes[slot,c+2].set_xlabel(f'IoU {iou(values):.3f}',fontsize=7.5,labelpad=3)
            displayed.append(dict(slot=slot,sensor=item['sensor'],role=item['role'],validation_index=index,
                global_index=item['global_index'],name=item['name'],control=control,**values,foreground_iou=iou(values)))
    for ax,title in zip(axes[0],('Normalized\ninput','Ground\ntruth',*(label.replace(' ','\n',1) for label in LABELS))):
        ax.set_title(title,fontsize=7.5,pad=5)
    fig.legend(handles=[Patch(color=color,label=label) for color,label in
        zip(ERROR_COLORS,('True positive','False positive','False negative'))],
        loc='lower center',ncol=3,bbox_to_anchor=(.5,.075),fontsize=8)
    fig.suptitle('Seed 42 validation: examples selected using affine IoU only',y=.985,fontsize=9)
    fig.text(.5,.012,'Affine-conditioned selection emphasizes reference errors; descriptive validation examples, not independent evidence.',
             ha='center',fontsize=7)
    # Fixed grid spacing reserves actual room for every image's index/IoU
    # label; automatic aspect-ratio adjustments can otherwise cover them.
    fig.subplots_adjust(left=.075,right=.995,bottom=.16,top=.89,hspace=.62,wspace=.10)
    save(fig,'qualitative_errors')
    csv_file(out/'learning_curves.csv',flat);csv_file(out/'best_epochs.csv',best_rows)
    for name,rows in data['metrics'].items(): csv_file(out/(name+'.csv'),rows)
    csv_file(out/'selected_examples.csv',examples,fields=('slot','sensor','role','eligible_count','status','validation_index',
        'global_index','name','affine_foreground_iou','ground_truth_flood_fraction'))
    csv_file(out/'displayed_counts.csv',displayed,fields=('slot','sensor','role','validation_index','global_index','name','control',
        'tp','fp','fn','tn','foreground_iou'))
    caption = """Learning curves use the twelve fixed-primary-rate (0.003) histories, four controls and seeds 42, 77 and 123, for 60 epochs with 128 optimization tiles. Nine histories are fresh and three learned-quantum histories reused; the whole audited source campaign contains 20 fits (12 fresh and 8 reused). Circles mark the original earliest best validation IoU epoch; curves do not establish convergence or significance.

Qualitative rows show seed 42 only: each sensor's worst and lower-median affine foreground IoU among validation tiles with at least 5% ground-truth flood. Ties use ordinal validation index. Duplicate selections are retained; empty eligibility is labeled. The affine-conditioned selection bias can emphasize weaknesses of that reference and cannot establish an overall ranking. The rule and identities were frozen before other-arm inference.

Input channels use the fixed linear map from the original normalized range [-1,1] to [0,1], clipped at the endpoints, without per-image stretching; SAR is not natural-color imagery. Ground truth white denotes flood. Model overlays show TP teal, FP orange, FN purple; true-negative pixels show dimmed input. Foreground IoU=TP/(TP+FP+FN), zero for an empty union. Pooled IoU sums counts; disagreements are differing labels, not improvements.

All 230 ordered seed 42 validation tiles were evaluated with batch 8, native fixed checkpoints, and fit-only normalization over 1,303 images excluding validation. Original integer sensor confusion counts were recovered exactly. These validation tiles were already used for checkpoint selection. No new training, test prediction, independent holdout, quantum-superiority, Anvil, hardware or federated-accuracy evidence is added.
"""
    (out/'captions.md').write_text(caption)
    (out/'README.md').write_text('# Descriptive HQ validation figures\n\n'+caption+'\nMachine tables retain full identities and exact counts. Raw prediction archives and checkpoint weights are not included. Renderer source is retained separately; changing layout requires no inference.\n')
    (out/'renderer_source.py.txt').write_bytes(SELF.read_bytes())
    write(out/'provenance.json',dict(format='hq-publication-render-v1',renderer_sha256=LOADED_SELF,
        cache_sha256=digest(root/'cache.json'),cache_inventory_sha256=digest(root/'artifact_sha256.json'),
        plan_sha256=data['cache']['plan_sha256'],selection_sha256=data['cache']['selection_sha256'],
        protocol_sha256=data['plan']['protocol_sha256'],numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
        inference_performed=False,scope='Descriptive validation illustration with affine-conditioned selection bias'))
    require(digest(SELF) == LOADED_SELF,'Renderer source changed during rendering')
    require(inventory(root) == {**read(root/'artifact_sha256.json'),'artifact_sha256.json':digest(root/'artifact_sha256.json')},
            'Cache changed during rendering')
    write(out/'artifact_sha256.json',inventory(out))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(argv)
    render(args.cache_dir,args.output_dir)
    print('Rendered descriptive frozen-cache figures:',args.output_dir)


if __name__ == '__main__':
    main()
