"""Production summaries must identify the simulator actually used by local cores."""
import pytest

from scripts.run_local_spatial_study import study_specs
from tools import training_common as tc
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
