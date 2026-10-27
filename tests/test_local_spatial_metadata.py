"""Production summaries must identify the simulator actually used by local cores."""
import json

import pytest
import torch

from scripts.run_local_spatial_study import study_specs
from tools import training_common as tc
from tools.checkpoint_evaluation import load_checkpoint_model
from utilsTorch.local_spatial import LocalSpatialBottleneckBlock
from utilsTorch.utilsTorch import QVUNetSegmentation, ClassicalAblationUNetSegmentation


@pytest.mark.parametrize('model_id', [spec['id'] for spec in study_specs()])
def test_local_production_simulator_metadata_and_description(model_id):
    spec = next(spec for spec in study_specs() if spec['id'] == model_id)
    config = dict(spec['config'], seed=42)
    cls = QVUNetSegmentation if spec['model_type'] == 'quantum' else ClassicalAblationUNetSegmentation
    model = tc.build_model(cls, config)
    info = tc.quantum_simulator_info(model, config)
    description = tc.format_quantum_simulator(info)
    if spec['model_type'] == 'quantum':
        assert info['q_backend'] == info['family'] == 'pennylane'
        assert info['device'] == 'default.qubit'
        assert info['sim_device'] == 'cpu'
        assert info['interface'] == 'torch' and info['diff_method'] == 'backprop'
        assert 'PennyLane default.qubit' in description and 'on cpu' in description
        assert 'q_backend=pennylane' in description and not description.startswith('none')
    else:
        assert info['q_backend'] is None and info['family'] == 'classical'
        assert description == 'none (quantum family: classical)'
    assert info['local_spatial_kernel'] == config['local_spatial_kernel']
    assert info['local_spatial_readout'] == config['local_spatial_readout']


@pytest.mark.parametrize('kind', ['mlp', 'conv', 'sincos'])
@pytest.mark.parametrize('training', ['centralized', 'federated'])
def test_classical_checkpoint_reader_accepts_its_production_null_control(tmp_path, kind, training):
    import ablation_studies
    spec = next(spec for spec in study_specs() if spec['id'] == 'local_' + kind)
    config = dict(spec['config'], seed=42, image_size=32, batch_size=4,
        learning_rate=0.001, num_epochs=1, local_epochs=1, fed_rounds=1,
        num_clients=2, clients_per_round=2)
    model = tc.build_model(ClassicalAblationUNetSegmentation, config)
    fingerprint = (tc.centralized_fingerprint if training == 'centralized' else tc.run_fingerprint)(config, 'ablation')
    assert fingerprint['q_training_control'] is None
    payload = dict(fingerprint=fingerprint, model_type='ablation', best_epoch=1)
    if training == 'centralized':
        payload['best_state_dict'] = model.state_dict()
    else:
        payload['best_parameters'] = ablation_studies._torch_params_to_numpy(model)
    checkpoint, config_path = tmp_path / 'checkpoint.pt', tmp_path / 'config.json'
    torch.save(payload, checkpoint)
    config_path.write_text(json.dumps(config))
    restored, loaded, model_type = load_checkpoint_model(checkpoint, data_root=tmp_path,
        config_path=config_path, device='cpu')
    assert model_type == 'ablation' and loaded['q_training_control'] is None
    assert restored.model.vertex.q_block.q_training_control == 'trainable'
    assert all(parameter.requires_grad for parameter in restored.parameters())
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize('control', ['frozen', 'no_entanglement', 'invalid', '', False, 0])
def test_null_default_does_not_accept_invalid_classical_quantum_controls(control):
    with pytest.raises(ValueError):
        LocalSpatialBottleneckBlock(4, kind='mlp', q_training_control=control)
