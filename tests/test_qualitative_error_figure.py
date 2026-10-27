"""Example selection must depend on the baseline and cover every test tile."""
import json

import numpy as np
import pytest
import torch

from misc import build_qualitative_error_figure as figure
from tools import training_common as tc


def test_fixed_rule_examples_and_pixel_errors():
    records = [dict(index=i, flood_fraction=flood, iou={'classical': iou, 'quantum': 1 - iou})
               for i, (flood, iou) in enumerate([(0.04, 0), (0.05, 0.8), (0.2, 0.2), (0.2, 0.2), (0.4, 1)])]
    assert figure.select_examples(records) == {'worst': 2, 'median': 3}
    assert figure.select_examples(list(reversed(records))) == {'worst': 2, 'median': 3}
    assert figure.select_examples(records[:1]) is None
    target = np.array([[True, False], [True, False]])
    predicted = np.array([[True, True], [False, False]])
    assert figure.pixel_iou(predicted, target) == pytest.approx(1 / 3)
    assert figure.pixel_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 0
    image = np.zeros((2, 2, 3), np.float32)
    overlay = figure.overlay_prediction(image, target, predicted)
    assert overlay[0, 0] == pytest.approx(figure.matplotlib.colors.to_rgb(figure.COLORS['TP']))
    assert overlay[0, 1] == pytest.approx(figure.matplotlib.colors.to_rgb(figure.COLORS['FP']))
    assert overlay[1, 0] == pytest.approx(figure.matplotlib.colors.to_rgb(figure.COLORS['FN']))
    assert np.array_equal(image, np.zeros_like(image))


def test_checkpoint_comparison_smoke_exports_all_tiles_and_fixed_examples(tmp_path):
    from tools.make_mini_flood_dataset import generate_mini_flood_dataset
    from utilsTorch.utilsTorch import UNetSegmentation, QVUNetSegmentation, ClassicalAblationUNetSegmentation

    data_root = tmp_path / 'data'
    generate_mini_flood_dataset(str(data_root), n_train=8, n_test=3, seed=42, seeds=[42])
    config = dict(base_channels=2, init_dim=2, out_dim=2, dim_mults=[1, 2, 4, 8],
        resnet_block_groups=1, image_size=32, dataset='combined', seed=42,
        use_time_emb=False, decoder_skip_gate='none', combined_input_mode='shared3',
        backbone_conv_mode='ws_conv', loss='dice_ce', quantum_channels=16,
        q_ansatz='reupload', q_layers=1, n_qubits=2, q_spatial_mode='global',
        q_share_blocks=True, skip_modulation='affine', quantum_backend='pennylane',
        q_backend='pennylane', q_sim_device='cpu', ablation_mode='mlp', ablation_mlp_mode='full_rank')
    classes = dict(classical=UNetSegmentation, ablation=ClassicalAblationUNetSegmentation, quantum=QVUNetSegmentation)
    for kind, cls in classes.items():
        model = tc.build_model(cls, config)
        tc.atomic_torch_save(dict(settings=config, model_type=kind, model_state_dict=model.state_dict()), str(tmp_path / (kind + '.pt')))
    manifest_path = tmp_path / 'comparison.json'
    manifest_path.write_text(json.dumps({'datasets': [{'id': 'Pooled', 'checkpoints': {k: k + '.pt' for k in classes}}]}))
    output = tmp_path / 'figures'
    payload = figure.main(['--manifest', str(manifest_path), '--data-root', str(data_root), '--out-dir', str(output), '--batch-size', '2'])
    assert json.loads((output / 'qualitative_eval_meta.json').read_text()) == payload
    meta = payload['datasets'][0]
    assert len(meta['tiles']) == 6
    assert {t['modality'] for t in meta['tiles']} == {'optical', 'sar'}
    assert all(set(t['iou']) == set(classes) for t in meta['tiles'])
    assert meta['selected'] == figure.select_examples(meta['tiles'])
    for kind in classes:
        assert meta['checkpoints'][kind]['sha256'] == figure._sha256(tmp_path / (kind + '.pt'))
    for name in ['qualitative_median_worst_fp_fn', 'per_tile_iou_scatter']:
        for extension in ['png', 'pdf']:
            assert (output / (name + '.' + extension)).stat().st_size > 1000
    wrong = dict(config, seed=77)
    model = tc.build_model(UNetSegmentation, wrong)
    tc.atomic_torch_save(dict(settings=wrong, model_type='classical', model_state_dict=model.state_dict()), str(tmp_path / 'classical.pt'))
    with pytest.raises(ValueError, match='identical dataset, seed and preprocessing'):
        figure.main(['--manifest', str(manifest_path), '--data-root', str(data_root), '--out-dir', str(output)])


@pytest.mark.parametrize('policy', ['mixed', 'full', 'fit'])
def test_saved_normalization_must_agree_with_models_and_reloaded_data(tmp_path, policy):
    from tools.make_mini_flood_dataset import generate_mini_flood_dataset
    from tools.compute_norm_stats import build_norm_stats
    from utilsTorch.utilsTorch import UNetSegmentation, QVUNetSegmentation, ClassicalAblationUNetSegmentation

    data = tmp_path / 'data'
    generate_mini_flood_dataset(str(data), n_train=8, n_test=3, seed=42, seeds=[42])
    full_stats = tmp_path / 'full-training.json'
    full_stats.write_text(json.dumps(build_norm_stats(str(data), datasets=['flood_combined'],
                                                    seed=42, fit_only=False, image_size=32)))
    cfg = dict(base_channels=2, init_dim=2, out_dim=2, dim_mults=[1, 2, 4, 8],
        resnet_block_groups=1, image_size=32, dataset='combined', seed=42,
        use_time_emb=False, decoder_skip_gate='none', combined_input_mode='shared3',
        backbone_conv_mode='ws_conv', loss='dice_ce', quantum_channels=16,
        q_ansatz='reupload', q_layers=1, n_qubits=2, q_spatial_mode='global',
        q_share_blocks=True, skip_modulation='affine', quantum_backend='pennylane',
        q_backend='pennylane', q_sim_device='cpu', ablation_mode='mlp', ablation_mlp_mode='full_rank',
        batch_size=2, learning_rate=0.001, data_root=str(data))
    classes = dict(classical=UNetSegmentation, ablation=ClassicalAblationUNetSegmentation, quantum=QVUNetSegmentation)
    for kind, cls in classes.items():
        full = policy == 'full' or (policy == 'mixed' and kind == 'ablation')
        settings = dict(cfg, fit_only_norm=not full)
        if full:
            settings['stats_path'] = str(full_stats)
        model = tc.build_model(cls, settings)
        fingerprint = tc.centralized_fingerprint(settings, kind)
        assert 'fit_only_norm' not in fingerprint and 'stats_path' not in fingerprint
        tc.atomic_torch_save(dict(fingerprint=fingerprint, model_state_dict=model.state_dict()), str(tmp_path / (kind + '.pt')))
    group = dict(id='Pooled', checkpoints={kind: kind + '.pt' for kind in classes})
    if policy == 'fit':
        meta, _, _, _ = figure.evaluate_group(group, data, tmp_path, torch.device('cpu'), 2)
        assert all(record['normalization_verified'] for record in meta['checkpoints'].values())
        assert all(record['saved_normalization']['stats_sha256'] == meta['normalization']['stats_sha256']
                   for record in meta['checkpoints'].values())
    else:
        with pytest.raises(ValueError, match='normalization'):
            figure.evaluate_group(group, data, tmp_path, torch.device('cpu'), 2)


@pytest.mark.parametrize('target', ['manifest', 'checkpoint', 'config'])
def test_output_symlink_cannot_overwrite_protected_input(tmp_path, target):
    manifest = tmp_path / 'comparison.json'
    checkpoint = tmp_path / 'classical.pt'
    configuration = tmp_path / 'model.json'
    checkpoint.write_bytes(b'protected checkpoint bytes')
    configuration.write_text('{}')
    manifest.write_text(json.dumps(dict(datasets=[dict(id='Pooled', checkpoints=dict(
        classical=dict(path='classical.pt', config='model.json'),
        ablation='ablation.pt', quantum='quantum.pt'))])))
    protected = dict(manifest=manifest, checkpoint=checkpoint, config=configuration)[target]
    original = protected.read_bytes()
    output = tmp_path / 'figures'
    output.mkdir()
    (output / 'qualitative_eval_meta.json').symlink_to(protected)
    with pytest.raises(SystemExit):
        figure.main(['--manifest', str(manifest), '--data-root', str(tmp_path / 'data'), '--out-dir', str(output)])
    assert protected.read_bytes() == original
