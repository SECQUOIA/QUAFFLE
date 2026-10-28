"""Load trusted training checkpoints for read-only evaluation and diagnostics."""

import json
from pathlib import Path

import torch

from tools import training_common as tc
from utilsTorch.utilsTorch import (
    ClassicalAblationUNetSegmentation, QVUNetSegmentation, UNetSegmentation,
)


def load_weights(model, checkpoint):
    """Prefer validation-selected weights; validate every federated tensor before copying."""
    _ = sum(parameter.numel() for parameter in model.parameters())
    if isinstance(checkpoint, dict) and any(k in checkpoint for k in ('best_state_dict', 'model_state_dict')):
        state = checkpoint.get('best_state_dict')
        state = checkpoint.get('model_state_dict') if state is None else state
    elif isinstance(checkpoint, dict) and any(k in checkpoint for k in ('best_parameters', 'parameters')):
        values = checkpoint.get('best_parameters')
        values = checkpoint.get('parameters') if values is None else values
        parameters = list(model.parameters())
        if values is None or len(parameters) != len(values):
            raise ValueError('Checkpoint parameter count mismatch')
        validated = []
        for i, (parameter, value) in enumerate(zip(parameters, values)):
            tensor = torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)
            if tensor.shape != parameter.shape:
                raise ValueError(f'Checkpoint parameter shape mismatch at index {i}')
            validated.append(tensor)
        with torch.no_grad():
            for parameter, tensor in zip(parameters, validated):
                parameter.copy_(tensor)
        return model
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint
    if state is None:
        raise ValueError('Checkpoint has no trained weights')
    model.load_state_dict(state)
    return model


def load_checkpoint_model(path, *, data_root, device='cpu', config_path=None, model_type=None):
    """Reconstruct saved architecture without guessing legacy model dimensions.

    Explicit configuration fills missing metadata. Saved fingerprints/settings
    take precedence over sibling metrics and supplied defaults; incompatible
    states fail strict loading rather than partially initializing a model.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = {}
    if config_path is not None:
        config.update(json.loads(Path(config_path).read_text()))
    for sibling in (path.parent / 'results' / 'metrics.json', path.parent / 'metrics.json'):
        if sibling.is_file():
            metrics = json.loads(sibling.read_text())
            config.update(metrics.get('config') or metrics.get('settings') or {})
            config.update({k: v for k, v in metrics.items() if v is not None and k not in ('config', 'settings')})
            break
    if isinstance(checkpoint, dict):
        for key in ('config', 'fingerprint', 'settings'):
            if isinstance(checkpoint.get(key), dict):
                config.update(checkpoint[key])
        if checkpoint.get('model_type') is not None:
            config['model_type'] = checkpoint['model_type']
    saved_type = config.get('model_type')
    if saved_type is not None and model_type is not None and saved_type != model_type:
        raise ValueError('Explicit model type conflicts with checkpoint metadata')
    kind = model_type or saved_type
    classes = dict(classical=UNetSegmentation, quantum=QVUNetSegmentation, ablation=ClassicalAblationUNetSegmentation)
    if kind not in classes:
        raise ValueError('Checkpoint model_type is missing; pass --model-type')
    if config.get('q_ansatz') == 'legacy' and config.get('q_layers') is None:
        config['q_layers'] = 1
    required = ['base_channels', 'image_size', 'resnet_block_groups', 'dataset', 'seed',
                'use_time_emb', 'init_dim', 'out_dim', 'dim_mults', 'decoder_skip_gate',
                'combined_input_mode', 'backbone_conv_mode']
    if kind != 'classical':
        required += ['quantum_channels', 'q_ansatz', 'q_layers', 'q_spatial_mode',
                     'q_share_blocks', 'skip_modulation']
        if config.get('q_ansatz') != 'hqunet':
            required.append('n_qubits')
        else:
            required.append('hqunet_grid')
    if config.get('local_spatial_kernel') is not None:
        required += ['local_spatial_readout', 'local_spatial_channels']
        if kind == 'quantum':
            required.append('q_training_control')
    if kind == 'ablation':
        required += ['ablation_mode', 'ablation_mlp_mode']
    missing = [k for k in required if config.get(k) is None]
    if missing:
        raise ValueError(f'Checkpoint architecture metadata missing {missing}; supply --config')
    config['data_root'] = str(Path(data_root).resolve())
    model = tc.build_model(classes[kind], config).to(torch.device(device))
    load_weights(model, checkpoint)
    model.eval()
    return model, config, kind
