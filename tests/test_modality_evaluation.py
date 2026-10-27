"""Sensor subset metrics must stay attached to samples through batching and subsets."""

import json

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from tools import training_common as tc
from utilsTorch.utilsTorch import evaluate_model


class _SensorTiles(Dataset):
    modalities = ['optical', 'sar', 'optical', 'sar']
    targets = torch.tensor([[1, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    predicted = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1], [1, 1, 1, 0]])

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.full((3, 2, 2), float(index)), self.targets[index].reshape(1, 2, 2).float()


class _LookupModel(torch.nn.Module):
    def forward(self, images):
        indices = images[:, 0, 0, 0].long()
        positive = (_SensorTiles.predicted[indices].to(images.device).reshape(-1, 2, 2).float() * 2 - 1) * 2
        return torch.stack([-positive, positive], dim=1)


@pytest.mark.parametrize('shuffle,num_workers', [(False, 0), (True, 0), (True, 2)])
def test_sensor_confusion_metrics_follow_shuffled_nested_subsets(shuffle, num_workers):
    dataset = Subset(Subset(_SensorTiles(), [3, 0, 2, 1]), [2, 0, 3, 1])
    loader = tc.make_loader(dataset, 3, shuffle, 17, num_workers=num_workers)
    metrics = evaluate_model(_LookupModel(), loader, torch.device('cpu'))
    assert metrics['optical_subset_n_samples'] == metrics['sar_subset_n_samples'] == 2
    assert [metrics['optical_subset_' + k] for k in ('tp', 'fp', 'fn', 'tn')] == [3, 0, 1, 4]
    assert [metrics['sar_subset_' + k] for k in ('tp', 'fp', 'fn', 'tn')] == [2, 3, 0, 3]
    assert metrics['optical_subset_iou'] == pytest.approx(3 / 4)
    assert metrics['sar_subset_iou'] == pytest.approx(2 / 5)
    assert metrics['iou'] == pytest.approx(5 / 9)
    assert metrics['dice'] == pytest.approx(10 / 14)
    weighted = sum(metrics[m + '_subset_iou'] * sum(metrics[m + '_subset_' + k] for k in ('tp', 'fp', 'fn'))
                   for m in ('optical', 'sar')) / 9
    assert metrics['iou'] == pytest.approx(weighted)
    assert metrics['optical_subset_accuracy'] == pytest.approx(7 / 8)
    assert metrics['sar_subset_accuracy'] == pytest.approx(5 / 8)
    for sensor, indices in (('optical', [0, 2]), ('sar', [1, 3])):
        truth = _SensorTiles.targets[indices].numpy().ravel()
        scores = _SensorTiles.predicted[indices].numpy().ravel()
        assert metrics[sensor + '_subset_auc'] == pytest.approx(roc_auc_score(truth, scores))


def test_subset_metrics_persist_and_absent_or_invalid_sensors_are_explicit(tmp_path):
    dataset = _SensorTiles()
    subset = Subset(dataset, [0, 2])
    metrics = evaluate_model(_LookupModel(), DataLoader(subset, batch_size=2), 'cpu', single_class_auc=-1)
    assert metrics['sar_subset_n_samples'] == 0
    assert metrics['sar_subset_auc'] == -1
    assert metrics['optical_subset_iou'] == metrics['iou']
    path = tc.write_metrics_json(str(tmp_path), script='test', training='centralized', model_type='classical',
                                 config={'seed': 42, 'dataset': 'combined'}, val_metrics=metrics, test_metrics=metrics)
    payload = json.loads(open(path).read())
    assert payload['test'] == metrics
    assert payload['val'] == metrics
    dataset.modalities = ['sar']
    with pytest.raises(ValueError, match='one optical/sar tag'):
        evaluate_model(_LookupModel(), DataLoader(dataset), 'cpu')
    bare = TensorDataset(torch.zeros(1, 3, 2, 2), torch.zeros(1, 1, 2, 2))
    result = evaluate_model(_LookupModel(), DataLoader(bare), 'cpu')
    assert set(result) == {'loss', 'accuracy', 'auc', 'iou', 'dice'}


def test_attaching_sensor_metadata_preserves_pooled_metrics():
    tagged = _SensorTiles()
    bare = TensorDataset(torch.stack([tagged[i][0] for i in range(4)]),
                         torch.stack([tagged[i][1] for i in range(4)]))
    loaders = [DataLoader(dataset, batch_size=3, shuffle=True,
                          generator=torch.Generator().manual_seed(22))
               for dataset in (tagged, bare)]
    with_tags = evaluate_model(_LookupModel(), loaders[0], 'cpu')
    without_tags = evaluate_model(_LookupModel(), loaders[1], 'cpu')
    assert {key: with_tags[key] for key in without_tags} == without_tags


def test_tagged_evaluation_rejects_reordering_collator_but_untagged_keeps_it():
    def reverse_collate(items):
        images, masks = zip(*reversed(items))
        return torch.stack(images), torch.stack(masks)

    tagged = _SensorTiles()
    with pytest.raises(ValueError, match='sensor.*default collate'):
        evaluate_model(_LookupModel(), DataLoader(tagged, batch_size=4, collate_fn=reverse_collate), 'cpu')
    bare = TensorDataset(torch.stack([tagged[i][0] for i in range(4)]),
                         torch.stack([tagged[i][1] for i in range(4)]))
    result = evaluate_model(_LookupModel(), DataLoader(bare, batch_size=4, collate_fn=reverse_collate), 'cpu')
    assert result['iou'] == pytest.approx(5 / 9)
    assert set(result) == {'loss', 'accuracy', 'auc', 'iou', 'dice'}


@pytest.mark.parametrize('federated', [False, True])
def test_checkpoint_cli_restores_selected_weights_and_reports_sensor_subsets(tmp_path, federated):
    from scripts import evaluate_checkpoint
    from tools.checkpoint_evaluation import load_checkpoint_model, load_weights
    from tools.make_mini_flood_dataset import generate_mini_flood_dataset
    from utilsTorch.utilsTorch import UNetSegmentation

    data_root = tmp_path / 'data'
    generate_mini_flood_dataset(str(data_root), n_train=8, n_test=2, seed=42, seeds=[42])
    config = dict(base_channels=2, init_dim=2, out_dim=2, dim_mults=[1, 2, 4, 8],
                  resnet_block_groups=1, image_size=32, dataset='combined', seed=42,
                  use_time_emb=False, decoder_skip_gate='none', combined_input_mode='shared3',
                  backbone_conv_mode='ws_conv', model_type='classical', loss='dice_ce')
    model = tc.build_model(UNetSegmentation, config)
    best = tc.clone_state_dict_cpu(model.state_dict())
    final = {key: torch.zeros_like(value) for key, value in best.items()}
    path = tmp_path / 'checkpoint.pt'
    configuration_path = None
    if federated:
        selected = [parameter.detach().numpy().copy() for parameter in model.parameters()]
        configuration_path = tmp_path / 'config.json'
        configuration_path.write_text(json.dumps(config))
        tc.atomic_torch_save(dict(fingerprint={'model_type': 'classical', 'seed': 42, 'dataset': 'combined'},
                                 best_parameters=selected, parameters=[np.zeros_like(a) for a in selected]), str(path))
    else:
        tc.atomic_torch_save(dict(config=config, best_state_dict=best, model_state_dict=final), str(path))
    restored, _, _ = load_checkpoint_model(path, data_root=data_root, config_path=configuration_path)
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in best.items())
    output = tmp_path / 'eval.json'
    argv = ['--checkpoint', str(path), '--data-root', str(data_root), '--out', str(output)]
    if configuration_path:
        argv += ['--config', str(configuration_path)]
    payload = evaluate_checkpoint.main(argv)
    assert payload['n_test_tiles'] == 4
    assert payload['test']['optical_subset_n_samples'] == payload['test']['sar_subset_n_samples'] == 2
    assert json.loads(output.read_text()) == payload
    with pytest.raises(SystemExit):
        evaluate_checkpoint.main(['--checkpoint', str(path), '--data-root', str(data_root), '--out', str(path)])
    assert path.is_file()
    with pytest.raises(ValueError, match='conflicts'):
        load_checkpoint_model(path, data_root=data_root, model_type='quantum')
    parameters = [parameter.detach().numpy().copy() for parameter in restored.parameters()]
    malformed = list(parameters)
    malformed[-1] = np.zeros((999,), dtype=np.float32)
    before = [parameter.detach().clone() for parameter in restored.parameters()]
    with pytest.raises(ValueError, match='shape mismatch'):
        load_weights(restored, {'best_parameters': malformed})
    assert all(torch.equal(parameter, previous) for parameter, previous in zip(restored.parameters(), before))
