"""Independent five-class arithmetic and bounded native-runner behavior."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch
from torch import nn

from scripts import run_landcover_native as runner
from tools import landcover_metrics as metrics
from utilsTorch import hq_landcover as native


def test_five_class_metrics_have_hand_computable_pooled_basis():
    # Class0:2/4, class1:1/4, class2:1/2, class3:2/3, class4:1/2.
    truth = np.array([0,0,0,1,1,2,2,3,3,4,4], dtype=np.uint8)
    pred = np.array([0,0,1,1,0,2,3,3,3,4,1], dtype=np.uint8)
    counts = metrics.confusion_counts(truth, pred)
    expected = np.array([[2,1,0,0,0],[1,1,0,0,0],[0,0,1,1,0],
                         [0,0,0,2,0],[0,1,0,0,1]])
    np.testing.assert_array_equal(counts, expected)
    row = metrics.metrics_from_counts(counts)
    # Actual class1 has1TP+1FN+2FP=4 union.
    expected_iou = [2/4, 1/4, 1/2, 2/3, 1/2]
    assert list(row['class_iou'].values()) == pytest.approx(expected_iou)
    assert row['miou'] == pytest.approx(sum(expected_iou)/5)
    assert row['foreground_miou'] == pytest.approx(sum(expected_iou[1:])/4)
    assert row['accuracy'] == pytest.approx(7/11)
    assert row['truth_pixels'] == [3,2,2,2,2]
    assert row['prediction_pixels'] == [3,3,1,3,1]
    assert row['absent_union'] == []


@pytest.mark.parametrize('truth,pred', [
    (np.array([5]),np.array([0])), (np.array([-1]),np.array([0])),
    (np.array([.5]),np.array([0])), (np.array([]),np.array([])),
    (np.array([0,1]),np.array([0]))])
def test_invalid_labels_are_not_binarized_or_silently_ignored(truth, pred):
    with pytest.raises(ValueError):
        metrics.confusion_counts(truth, pred)


def test_absent_class_convention_does_not_change_macro_denominator():
    row = metrics.metrics_from_counts(np.diag([2,0,0,0,0]))
    assert row['miou'] == .2 and row['foreground_miou'] == 0
    assert row['accuracy'] == 1 and len(row['absent_union']) == 4


def test_loss_uses_all_five_softmax_channels_and_matches_uniform_reference():
    logits = torch.zeros(1,5,5,5, requires_grad=True)
    masks = (torch.arange(25).reshape(1,5,5) % 5).long()
    value = metrics.segmentation_loss(logits, masks)
    # Uniform predictions: each class intersection1, sums5+5, Dice~.2.
    expected = .5*np.log(5)+.5*(1-(2+1e-6)/(10+1e-6))
    assert float(value.detach()) == pytest.approx(expected, abs=1e-7)
    value.backward()
    assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad).item() == 125
    with pytest.raises(ValueError):
        metrics.segmentation_loss(logits[:,:2], masks)
    with pytest.raises(ValueError):
        metrics.segmentation_loss(logits, masks.float())
    masks[0,0,0] = 5
    with pytest.raises(ValueError):
        metrics.segmentation_loss(logits, masks)


def test_explicit_ce_loss_and_derivatives_match_independent_standard_cpu_loss():
    generator = torch.Generator().manual_seed(39)
    first = torch.randn(2,5,8,8,generator=generator,dtype=torch.double,requires_grad=True)
    second = first.detach().clone().requires_grad_()
    labels = torch.randint(0,5,(2,8,8),generator=generator)
    actual = metrics.segmentation_loss(first,labels)
    probabilities = second.softmax(1)
    onehot = torch.nn.functional.one_hot(labels,5).permute(0,3,1,2).double()
    perclass = []
    for index in range(5):
        p,t = probabilities[:,index],onehot[:,index]
        perclass.append((2*(p*t).sum()+1e-6)/(p.sum()+t.sum()+1e-6))
    expected = .5*torch.nn.functional.cross_entropy(second,labels)+.5*(1-torch.stack(perclass).mean())
    actual.backward(); expected.backward()
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(first.grad,second.grad,rtol=1e-13,atol=1e-13)


def test_json_rejects_duplicate_nonfinite_and_write_preserves_complete_old_file(tmp_path):
    path = tmp_path/'record.json'
    for payload in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'):
        path.write_text(payload)
        with pytest.raises(ValueError):
            runner.read_json(path)
    runner.write_json(path, dict(ok=True))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        runner.write_json(path, dict(x=float('nan')))
    assert path.read_bytes() == before


def test_campaign_writer_lock_rejects_second_writer(tmp_path):
    with runner.campaign_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with runner.campaign_lock(tmp_path):
                pass


class TinyData(torch.utils.data.Dataset):
    def __init__(self, *args, split=None, tile_ids=None, seed=42, **kwargs):
        self.split = args[1] if len(args)>1 else split
        self.seed, self.epoch = seed, 0
        self.names = tile_ids
    def set_epoch(self, epoch):
        self.epoch = epoch
    def __len__(self):
        return 2
    def __getitem__(self, index):
        mask = (torch.arange(64*64).reshape(64,64)+index)%5
        return dict(image=torch.stack([mask.float()/2-1]*3),mask=mask,
                    tile_id=self.names[index],crop_x=self.epoch,crop_y=index,d4=0)


def test_native_fit_uses_five_class_validation_and_saves_strict_checkpoint(tmp_path, monkeypatch):
    from tools import landcover_data
    monkeypatch.setattr(landcover_data, 'NativeLandCoverDataset', TinyData)
    original = native.build_model
    monkeypatch.setattr(native, 'build_model', lambda control,seed,base_channels,image_size=64:
                        original(control,seed=seed,base_channels=1,image_size=64))
    plan = dict(base_channels=1, train_ids=['train0','train1'],val_ids=['val0','val1'],
        source={'fixture':'unpublished-synthetic'},_plan_sha='fixture',batch_size=2,
        epochs=5,evaluation_every=5,training_steps_per_fit=5)
    output = tmp_path/'fit'
    result = runner.fit('compact_affine',42,.001,plan,tmp_path,output,time.time()+120,torch.device('cpu'))
    assert result['status'] == 'complete' and result['optimizer_steps'] == 5
    assert len(result['schedule_hashes']) == 5
    assert [row['epoch'] for row in result['training_history']] == list(range(6))
    assert all(row['train_loss'] is not None for row in result['training_history'][1:])
    assert result['first_gradients']['core_l2'] > 0 and result['core_change_l2'] > 0
    checkpoint = torch.load(output/'best.pt',map_location='cpu',weights_only=True)
    restored = native.build_from_config(checkpoint['model_config'])
    restored.load_state_dict(checkpoint['state_dict'],strict=True)
    expected = checkpoint['validation']
    replayed = runner.evaluate(restored,TinyData(split='val',tile_ids=plan['val_ids']),
                               2,torch.device('cpu'),time.time()+120)
    assert expected['confusion'] == replayed['confusion']
    assert expected['pixels'] == 2*64*64 and len(expected['class_iou']) == 5
    before = {p.name:p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError,match='exists'):
        runner.fit('compact_affine',42,.001,plan,tmp_path,output,time.time()+120,torch.device('cpu'))
    assert {p.name:p.read_bytes() for p in output.iterdir()} == before
    assert runner.verify_completed_fit(output,plan,('compact_affine',42,.001)) == result
    with (output/'best.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='changed'):
        runner.verify_completed_fit(output,plan,('compact_affine',42,.001))


def test_evaluation_preserves_batchnorm_state_and_honors_deadline():
    model = nn.Sequential(nn.Conv2d(3,5,1),nn.BatchNorm2d(5))
    dataset = TinyData(split='val',tile_ids=['a','b'])
    before = runner.state_sha(model.state_dict())
    actual = runner.evaluate(model,dataset,2,torch.device('cpu'),time.time()+60)
    assert actual['pixels'] == 8192 and model.training
    assert runner.state_sha(model.state_dict()) == before
    with pytest.raises(ValueError,match='deadline'):
        runner.evaluate(model,dataset,2,torch.device('cpu'),time.time()-1)


def test_plan_rejects_stale_profile_before_dataset_or_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(runner,'sources',lambda:{'current':'source'})
    profile = tmp_path/'profile.json'
    runner.write_json(profile,dict(status='passed',source={'old':'source'}))
    with pytest.raises(ValueError,match='stale'):
        runner.make_plan(argparse.Namespace(profile=profile,dataset=tmp_path/'absent',
            output=tmp_path/'campaign',epochs=60,train_tiles=256,val_tiles=64,budget_seconds=3600))
    assert not (tmp_path/'campaign').exists()


def test_cost_estimate_amortizes_startup_and_includes_replays_and_io():
    profile = dict(warmup_steps=1,steps=2,actual_optimizer_steps=3,batch_size=4,
        arms={control:dict(warmup_seconds=100.,step_seconds=[2.,3.],eval_batch_seconds=1.)
              for control in runner.CONTROLS})
    cost = runner.estimate_cost(profile,epochs=1,train_tiles=8,val_tiles=2)
    assert cost == dict(startup_seconds=1300.,optimization_seconds=78.,validation_seconds=104.,
        selected_replay_seconds=96.,io_and_collection_allowance_seconds=1800.,total_seconds=3378.)
    profile['arms']['quantum']['step_seconds'][1] = float('nan')
    with pytest.raises(ValueError,match='durations'):
        runner.estimate_cost(profile,1,8,2)
    profile['warmup_steps'] = 0
    with pytest.raises(ValueError,match='warmup'):
        runner.estimate_cost(profile,1,8,2)


def test_state_hash_covers_parameter_names_values_and_batchnorm_buffers():
    state = dict(a=torch.ones(2),bn=torch.tensor(0))
    before = runner.state_sha(state)
    state['bn'] += 1
    assert runner.state_sha(state) != before
    state['bn'] -= 1
    assert runner.state_sha(state) == before
    state['renamed'] = state.pop('a')
    assert runner.state_sha(state) != before


def test_completed_run_audits_all_fit_bytes_without_fit_inference_or_writes(tmp_path,monkeypatch):
    from tools import landcover_data
    output,data = tmp_path/'campaign',tmp_path/'data'
    output.mkdir();data.mkdir()
    runner.write_json(data/'manifest.json',dict(fixture=True))
    source = dict(fixture='synthetic')
    plan = dict(format='landcover-native-plan-v1',source=source,
        dataset_manifest_sha=runner.sha(data/'manifest.json'),runtime=dict(fixture=True),train_ids=[],val_ids=[])
    runner.write_json(output/'plan.json',plan)
    plan_sha=runner.sha(output/'plan.json')
    identities=[('compact_affine',42,.001),('compact_affine',42,.003)]+[
        (c,s,.001) for s in (42,77,123) for c in runner.CONTROLS if (c,s)!=('compact_affine',42)]
    rows=[]
    for control,seed,rate in identities:
        destination=output/f'{control}_seed{seed}_lr{rate:g}';destination.mkdir()
        (destination/'best.pt').write_bytes(b'synthetic bytes; never deserialized')
        for name in ('history.json','training_history.json'):
            runner.write_json(destination/name,[])
        row=dict(control=control,seed=seed,rate=rate,source=source,plan_sha=plan_sha,
            checkpoint_sha=runner.sha(destination/'best.pt'),history_sha=runner.sha(destination/'history.json'),
            training_history_sha=runner.sha(destination/'training_history.json'))
        runner.write_json(destination/'result.json',row)
        runner.write_json(destination/'status.json',dict(status='complete',result_sha=runner.sha(destination/'result.json')))
        rows.append(row)
    runner.write_json(output/'all_fits.json',rows)
    runner.write_json(output/'status.json',dict(status='complete',fits=13,plan_sha=plan_sha,
        all_fits_sha=runner.sha(output/'all_fits.json')))
    (output/'.campaign.lock').touch()
    monkeypatch.setattr(runner,'sources',lambda:source)
    monkeypatch.setattr(runner,'setup',lambda name:torch.device('cpu'))
    monkeypatch.setattr(runner,'runtime',lambda device:dict(fixture=True))
    monkeypatch.setattr(landcover_data,'read_manifest',lambda root:dict(tiles={}))
    def forbidden(*a,**kw):raise AssertionError('Completed run fitted or inferred')
    monkeypatch.setattr(runner,'fit',forbidden);monkeypatch.setattr(runner,'evaluate',forbidden)
    def snapshot():return {str(p.relative_to(output)):(runner.sha(p),p.stat().st_mtime_ns)
                          for p in output.rglob('*') if p.is_file()}
    before=snapshot()
    runner.run(argparse.Namespace(output=output,dataset=data,device='cpu'))
    assert snapshot()==before
    damaged=output/'quantum_seed77_lr0.001'/'best.pt'
    damaged.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        runner.run(argparse.Namespace(output=output,dataset=data,device='cpu'))
