"""Software fixtures only: no scientific fit or own-data inference is run."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from misc import collect_hq_validation_predictions as producer
from misc import plot_hq_publication_figures as renderer


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def rebind(root,name,value):
    dump(root/name,value)
    cache = renderer.read(root/'cache.json')
    if name != 'cache.json':
        cache[name[:-5]+'_sha256'] = renderer.digest(root/name)
        dump(root/'cache.json',cache)
    inventory = renderer.inventory(root);inventory.pop('artifact_sha256.json',None)
    dump(root/'artifact_sha256.json',inventory)


@pytest.fixture
def frozen_cache(tmp_path,monkeypatch):
    work,legacy,data_root = (tmp_path/name for name in ('raw','legacy','data'))
    for path in (work,legacy,data_root):path.mkdir()
    dump(work/'plan.json',dict(legacy_root=str(legacy)))
    (legacy/'untouched.txt').write_text('historical byte fixture')
    names = [f'validation_{i}.tif' for i in range(230)]
    sensors = ['optical']*28+['sar']*202
    global_indices = list(range(1303,1533))
    targets = np.zeros((230,128,128),dtype=np.uint8)
    targets[:,:16,:] = 1
    predictions = np.repeat(targets[None],4,axis=0)
    for control in range(4):
        predictions[control,:,control:control+2,:32] = 0
        predictions[control,:,40:40+control+1,:16] = 1
    # Both modalities have tied affine scores; ordinal order must break ties.
    data = dict(dataset=argparse.Namespace(names=['fit']*1303+names,modalities=['optical']*1303+sensors),
        validation=global_indices,normalization={'fit_count':1303,'fixture':True},all_fit=list(range(1303)),fit=list(range(128)))
    fits = []
    for kind in (*producer.CONTROLS,'mlp'):
        for seed,rate in [(42,.003),(77,.003),(123,.003),(42,.001)]:
            val = {'iou':0.}
            if kind in producer.CONTROLS:
                pred = predictions[producer.CONTROLS.index(kind)]
                for sensor in ('optical','sar'):
                    index = np.flatnonzero(np.asarray(sensors) == sensor)
                    values = producer.counts(pred[index],targets[index])
                    val.update({sensor+'_subset_'+key:value for key,value in values.items()})
                    val[sensor+'_subset_n_samples'] = len(index)
                val['iou'] = producer.score(producer.counts(pred,targets))
            history = [dict(epoch=epoch,train_loss=None if epoch == 0 else 1/(epoch+1),
                val={'iou':.2 if epoch < 2 else .6} if epoch % 2 == 0 else None) for epoch in range(61)]
            row = dict(kind=kind,seed=seed,learning_rate=rate,val=val,val_names=names,val_indices=global_indices,
                normalization=data['normalization'],history=history,best_epoch=2,selected_state_sha256='a'*64)
            directory = f'{kind}_{seed}_{rate}'
            (work/directory).mkdir()
            (work/directory/'checkpoint.pt').write_bytes(b'fixture only: native restore is mocked')
            fits.append(dict(origin='reused' if kind in ('quantum','mlp') else 'fresh',directory=directory,record=row,
                checkpoint_sha256=producer.digest(work/directory/'checkpoint.pt'),metrics_sha256='b'*64,history_sha256='c'*64))
    plan = dict(settings={},input_identity={'fixture':'bytes'},source_revision='fixture-only',source_hashes={'fixture':'x'},runtime={'fixture':True})
    monkeypatch.setattr(producer.historical,'audit_campaign',lambda *args:dict(fits=fits,plan=plan))
    monkeypatch.setattr(producer.historical.study.pilot,'load_training_data',lambda *args:data)
    monkeypatch.setattr(producer.historical.study.pilot,'verify_inputs',lambda *args:plan['input_identity'])
    monkeypatch.setattr(producer.historical.study,'entry_directory',lambda work,legacy,entry:work/entry['directory'])
    monkeypatch.setattr(producer.historical.study.hq,'runtime',lambda device:{'fixture':True})
    calls = []
    output = tmp_path/'cache'
    def inference(control,row,checkpoint,data,device,retain_inputs=False):
        if control != 'compact_affine':
            assert (output/'selection.json').is_file(), 'selection must be durable before other inference'
            assert producer.digest(output/'selection.json') == calls[0][1]
        pred = predictions[producer.CONTROLS.index(control)]
        inputs = np.zeros((230,3,128,128),dtype=np.float32) if retain_inputs else None
        if inputs is not None:
            inputs[:,0] = np.arange(230,dtype=np.float32)[:,None,None]/229
        if control == 'compact_affine':
            calls.append((control,None))
        else:
            calls.append((control,producer.digest(output/'selection.json')))
        return pred.copy(),targets.copy(),inputs,{}
    # Capture the persisted selection hash before the first non-affine call.
    def wrapped(*args,**kwargs):
        if args[0] != 'compact_affine' and calls[0][1] is None:
            calls[0] = (calls[0][0],producer.digest(output/'selection.json'))
        return inference(*args,**kwargs)
    monkeypatch.setattr(producer,'infer',wrapped)
    producer.collect(work,data_root,output,device='cpu',budget_seconds=120)
    return output,calls,predictions,targets


def test_affine_first_selection_persisted_and_strict_count_parity(frozen_cache):
    root,calls,pred,target = frozen_cache
    assert [row[0] for row in calls] == ['compact_affine','quantum','frozen_quantum','zero_bottleneck']
    data = renderer.load_cache(root)
    assert [row['validation_index'] for row in data['selection']['examples']] == [0,13,28,128]
    assert len(data['metrics']['per_image']) == 920
    assert len(data['metrics']['disagreements']) == 1380
    assert data['parity']['passed'] is True
    assert np.array_equal(data['payload']['predictions'],pred)


def test_selection_ties_lower_median_duplicates_and_empty():
    targets = np.ones((5,128,128),dtype=np.uint8)
    affine = targets.copy();affine[0] = 0;affine[1] = 0
    names = ['a','b','c','d','e'];sensors = ['optical']*4+['sar']
    rows = producer.select_examples(affine,targets,names,sensors,[100,99,98,97,96])
    assert [row['validation_index'] for row in rows] == [0,1,4,4]
    assert renderer.independently_select(affine,targets,names,sensors,[100,99,98,97,96]) == rows
    targets[4] = 0
    assert [row['status'] for row in producer.select_examples(affine,targets,names,sensors,list(range(5)))[2:]] == ['empty_eligible_set']*2


def test_exact_five_percent_mask_threshold():
    targets = np.zeros((2,128,128),dtype=np.uint8)
    targets[0].flat[:819] = 1;targets[1].flat[:820] = 1
    rows = producer.select_examples(targets,targets,['a','b'],['optical']*2,[4,3])
    assert rows[0]['eligible_count'] == 1 and rows[0]['validation_index'] == 1


def test_confusion_arithmetic_and_disagreement():
    pred = np.array([1,1,1,1,0,0,0,0,0,0],dtype=np.uint8)
    target = np.array([1,1,1,0,1,1,0,0,0,0],dtype=np.uint8)
    expected = dict(tp=3,fp=1,fn=2,tn=4)
    assert producer.counts(pred,target) == renderer.counts(pred,target) == expected
    assert renderer.iou(expected) == .5
    assert renderer.iou(dict(tp=0,fp=0,fn=0,tn=10)) == 0


@pytest.mark.parametrize('member,mutation',[
    ('selection.json',lambda x:x['examples'][0].update(validation_index=1)),
    ('metrics.json',lambda x:x['per_image'][0].update(tp=99)),
    ('histories.json',lambda x:x[0].update(best_epoch=4)),
    ('histories.json',lambda x:x[0].update(learning_rate=.001)),
    ('parity.json',lambda x:x.update(passed=False)),
    ('plan.json',lambda x:x.update(inference_order=list(producer.CONTROLS))),
])
def test_internally_rehashed_corruption_fails(frozen_cache,member,mutation):
    root,*_ = frozen_cache
    value = renderer.read(root/member);mutation(value);rebind(root,member,value)
    with pytest.raises(ValueError):renderer.load_cache(root)


def test_raw_corruption_and_failed_attempt_rejected(frozen_cache):
    root,*_ = frozen_cache
    (root/'sources/extra.txt').write_text('unexpected')
    with pytest.raises(ValueError,match='inventory'):renderer.load_cache(root)
    (root/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='Failed'):renderer.load_cache(root)


@pytest.mark.parametrize('bad',[
    np.zeros((1,128,128),dtype=np.uint8),
    np.zeros((230,128,128),dtype=np.float32),
    np.full((230,128,128),2,dtype=np.uint8),
    np.empty((230,128,128),dtype=object),
])
def test_numeric_archive_rejects_geometry_dtype_nonbinary_pickle(tmp_path,bad):
    path = tmp_path/'bad.npz'
    descriptor = producer.save_arrays(path,predictions=bad)
    with pytest.raises((ValueError,TypeError)):
        renderer.arrays(path,descriptor,{'predictions':((230,128,128),'uint8')})


def test_numeric_archive_duplicate_member_rejected(tmp_path):
    path = tmp_path/'double.npz'
    good = np.zeros((230,128,128),dtype=np.uint8)
    descriptor = producer.save_arrays(path,predictions=good)
    with zipfile.ZipFile(path,'a') as archive:
        archive.writestr('predictions.npy',b'bad')
    descriptor['sha256'] = producer.digest(path)
    with pytest.raises(ValueError,match='member'):
        renderer.arrays(path,descriptor,{'predictions':((230,128,128),'uint8')})


def test_render_is_deterministic_and_never_repeats_inference(frozen_cache,tmp_path,monkeypatch):
    root,*_ = frozen_cache
    before = renderer.inventory(root)
    monkeypatch.setattr(producer,'infer',lambda *a,**k:pytest.fail('render must not infer'))
    first,second = tmp_path/'render1',tmp_path/'render2'
    renderer.render(root,first);renderer.render(root,second)
    assert renderer.inventory(first) == renderer.inventory(second)
    assert renderer.inventory(root) == before
    for name in ('learning_curves.pdf','learning_curves.png','qualitative_errors.pdf','qualitative_errors.png'):
        assert (first/name).stat().st_size > 1000
    assert 'affine-conditioned selection bias' in (first/'captions.md').read_text()
    assert len((first/'displayed_counts.csv').read_text().splitlines()) == 17
    with pytest.raises(ValueError,match='new or empty'):renderer.render(root,first)
    with pytest.raises(ValueError,match='aliases'):renderer.render(root,root/'render')


def test_renderer_import_has_no_torch_or_model_dependency():
    code = "import sys;from misc import plot_hq_publication_figures;assert 'torch' not in sys.modules;assert not any(x.startswith('utilsTorch') for x in sys.modules)"
    subprocess.run([sys.executable,'-c',code],check=True,cwd=producer.ROOT)


@pytest.mark.parametrize('mask_shape',[(1,128,128),(128,128),(2,128,128)])
def test_native_inference_uses_batch8_and_real_channel_mask(monkeypatch,mask_shape):
    import torch
    from tools import training_common as tc
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):return 230
        def __getitem__(self,index):
            image = torch.zeros((3,128,128));image[0] = index/229
            return image,torch.ones(mask_shape)
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.register_buffer('fixed',torch.ones(1));self.sizes=[]
        def forward(self,x):
            assert not self.training
            self.sizes.append(len(x))
            return torch.stack((torch.zeros_like(x[:,0]),torch.ones_like(x[:,0])),dim=1)
    model = Model()
    monkeypatch.setattr(producer.historical.study,'restore_checkpoint',lambda *a,**k:(model,{}))
    if mask_shape == (1,128,128):
        pred,target,inputs,_ = producer.infer('quantum',{},None,{'val':Dataset()},torch.device('cpu'),retain_inputs=True)
        assert pred.shape == target.shape == (230,128,128) and pred.all() and target.all()
        assert model.sizes == [8]*28+[6] and inputs.shape == (230,3,128,128)
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    else:
        with pytest.raises(ValueError,match='geometry'):
            producer.infer('quantum',{},None,{'val':Dataset()},torch.device('cpu'))


def test_parity_mismatch_is_exact_not_tolerated():
    pred = np.zeros((2,128,128),dtype=np.uint8);targets = pred.copy()
    row = {'val':{'iou':0}}
    for sensor in ('optical','sar'):
        row['val'].update({sensor+'_subset_'+key:value for key,value in dict(tp=0,fp=0,fn=0,tn=16384,n_samples=1).items()})
    producer.parity('quantum',pred,targets,['optical','sar'],row)
    row['val']['sar_subset_tn'] -= 1
    with pytest.raises(ValueError,match='parity'):producer.parity('quantum',pred,targets,['optical','sar'],row)


def test_sources_bound_nonempty_output_and_explicit_resource_validation(frozen_cache,tmp_path):
    root,*_ = frozen_cache
    with pytest.raises(ValueError,match='new or empty'):
        producer.collect(tmp_path/'missing',tmp_path/'missingdata',root,device='cpu')
    with pytest.raises(ValueError,match='Positive finite'):
        with producer.wall_budget(float('inf')):pass
    assert producer.source_hashes() == producer.LOADED_SOURCES


def test_failed_count_parity_preserves_partial_attempt_and_blocks_retry(frozen_cache,tmp_path,monkeypatch):
    root,_,_,target = frozen_cache
    output = tmp_path/'failed'
    monkeypatch.setattr(producer,'infer',lambda *a,**k:(np.zeros_like(target),target.copy(),
        np.zeros((230,3,128,128),dtype=np.float32),{}))
    with pytest.raises(ValueError,match='parity'):
        producer.collect(tmp_path/'raw',tmp_path/'data',output,device='cpu',budget_seconds=120)
    failure = renderer.read(output/'failure.json')
    assert failure['partial_evidence_preserved'] is True and failure['error_type'] == 'ValueError'
    assert (output/'plan.json').exists() and not (output/'cache.json').exists()
    before = renderer.inventory(output)
    with pytest.raises(ValueError,match='new or empty'):
        producer.collect(tmp_path/'raw',tmp_path/'data',output,device='cpu',budget_seconds=120)
    assert renderer.inventory(output) == before


def test_changed_source_and_cpu_allocator_fail_before_output(frozen_cache,tmp_path,monkeypatch):
    root,*_ = frozen_cache
    output = tmp_path/'new'
    with pytest.raises(ValueError,match='CPU prediction'):
        producer.collect(tmp_path/'raw',tmp_path/'data',output,device='cpu',gpu_memory_fraction=.1)
    assert not output.exists()
    monkeypatch.setattr(producer,'source_hashes',lambda:{'unexpected':'source-change'})
    with pytest.raises(ValueError,match='source changed'):
        producer.collect(tmp_path/'raw',tmp_path/'data',output,device='cpu')
    assert not output.exists()


def test_native_evaluation_detects_state_mutation(monkeypatch):
    import torch
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):return 230
        def __getitem__(self,index):return torch.zeros((3,128,128)),torch.zeros((1,128,128))
    class MutatingModel(torch.nn.Module):
        def __init__(self):super().__init__();self.register_buffer('fixed',torch.zeros(1))
        def forward(self,images):
            self.fixed.add_(1)
            return torch.zeros((len(images),2,128,128))
    monkeypatch.setattr(producer.historical.study,'restore_checkpoint',lambda *a,**k:(MutatingModel(),{}))
    with pytest.raises(ValueError,match='BatchNorm buffers'):
        producer.infer('quantum',{},None,{'val':Dataset()},torch.device('cpu'))


def test_fixed_normalized_display_range_preserves_negative_features():
    channels = np.array([[[-2.,-1.,0.,1.,2.]]]*3,dtype=np.float32)
    expected = np.array([0.,0.,.5,1.,1.])
    displayed = renderer.normalized_display(channels)
    assert displayed.shape == (1,5,3)
    for channel in range(3):np.testing.assert_array_equal(displayed[0,:,channel],expected)
