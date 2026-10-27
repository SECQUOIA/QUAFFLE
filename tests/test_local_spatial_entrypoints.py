"""The production ablation CLI must count a quantum reference for local controls."""
import pytest

import ablation_studies
from scripts.run_local_spatial_study import study_specs
from tools import training_common as tc
from utilsTorch.utilsTorch import ClassicalAblationUNetSegmentation


@pytest.mark.parametrize('kind', ['mlp', 'conv', 'sincos'])
def test_ablation_parameter_probe_accepts_the_local_classical_cli_config(kind):
    spec = next(spec for spec in study_specs() if spec['id'] == 'local_' + kind)
    requested = dict(spec['config'], seed=42)
    config, classical_params, quantum_params = ablation_studies._find_matched_torch_config(
        'pennylane', requested)
    assert config['local_spatial_kernel'] == kind
    assert config['local_spatial_readout'] == 'xz_zz'
    actual = tc.build_model(ClassicalAblationUNetSegmentation, config)
    assert classical_params == sum(parameter.numel() for parameter in actual.parameters())
    assert quantum_params == 503294
