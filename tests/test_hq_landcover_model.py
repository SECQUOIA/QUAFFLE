"""Native architectural contracts and independent 16-qubit numerical reference.

These tests verify the diagram interpretation and simulator arithmetic. They do
not establish reproduction of the paper's training procedure or reported scores.
"""
import copy
import pytest
import torch

from utilsTorch import hq_landcover as native
from utilsTorch.hq_direction import HQDirectionUNet
from utilsTorch.unetTorch import torch_hqunet_expvals
from utilsTorch.landcover_circuit import native_hqunet_expvals


@pytest.mark.parametrize('size', [4, 5, 8, 12])
def test_deterministic_pooling_preserves_adaptive_bins_and_input_derivatives(size):
    model = native.build_model('compact_affine',base_channels=1,image_size=64).double()
    first = torch.randn(1,16,size,size,dtype=torch.double,requires_grad=True)
    second = first.detach().clone().requires_grad_()
    actual = model.bottleneck.encode_angles(first)
    expected = model.bottleneck.pre_projection(
        torch.nn.functional.adaptive_avg_pool2d(second,(4,4))).tanh()
    expected = expected.permute(0,2,3,1).reshape(1,16,3)
    coefficients = torch.linspace(-1,1,48,dtype=torch.double).reshape(1,16,3)
    left = torch.autograd.grad((actual*coefficients).sum(),first)[0]
    right = torch.autograd.grad((expected*coefficients).sum(),second)[0]
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(left,right,rtol=1e-13,atol=1e-13)


def _reference_qnode(order):
    """Define all gates and edge order independently from the Torch simulator."""
    qml = pytest.importorskip('pennylane', reason='Independent circuit reference needs PennyLane')
    device = qml.device('default.qubit', wires=16, shots=None)
    if order == 'brickwork':
        # Fig. 2: disjoint first/last neighbor pairs, then the central pair.
        horizontal = [(4*row+a,4*row+b) for row in range(4) for a,b in ((0,1),(2,3),(1,2))]
        vertical = [(4*a+column,4*b+column) for column in range(4) for a,b in ((0,1),(2,3),(1,2))]
    else:
        horizontal = [(4*row+column,4*row+column+1) for row in range(4) for column in range(3)]
        vertical = [(4*row+column,4*(row+1)+column) for row in range(3) for column in range(4)]
    @qml.qnode(device,interface='torch',diff_method='backprop')
    def circuit(angles,weights):
        for q in range(16):
            qml.RX(angles[q,0],wires=q)
            qml.RY(angles[q,1],wires=q)
            qml.RZ(angles[q,2],wires=q)
        for direction,edges in enumerate((horizontal,vertical)):
            for first,second in edges:
                qml.RY(weights[0,direction,0],wires=first)
                qml.RY(weights[0,direction,1],wires=second)
                qml.CNOT(wires=[first,second])
                qml.RY(weights[0,direction,2],wires=first)
                qml.RY(weights[0,direction,3],wires=second)
        return tuple([qml.expval(qml.PauliZ(q)) for q in range(16)]+
                     [qml.expval(qml.PauliX(q)) for q in range(16)])
    return circuit


def _objective(values):
    coefficients = torch.linspace(-.9,1.2,32,dtype=torch.float64)
    curvature = torch.linspace(.1,.4,32,dtype=torch.float64)
    return (values*coefficients+values.square()*curvature).sum()


@pytest.mark.parametrize('order',['brickwork','sequential'])
def test_full_16q_pennylane_outputs_and_both_derivatives(order):
    generator = torch.Generator().manual_seed(8675309)
    angles = (torch.rand((1,16,3),generator=generator,dtype=torch.float64)*1.7-.8).requires_grad_()
    weights = (torch.rand((1,2,4),generator=generator,dtype=torch.float64)*1.9-.9).requires_grad_()
    observed = native_hqunet_expvals(angles,weights,edge_order=order)
    observed_grad = torch.autograd.grad(_objective(observed), (angles,weights))
    ref_angles = angles.detach().clone()[0].requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    expected = torch.stack(_reference_qnode(order)(ref_angles,ref_weights))[None]
    expected_grad = torch.autograd.grad(_objective(expected),(ref_angles,ref_weights))
    print(f'Independent PennyLane 16-qubit {order}: '
          f'output_max_abs_error={float((observed-expected).abs().max().detach()):.17g}, '
          f'angle_gradient_max_abs_error={float((observed_grad[0][0]-expected_grad[0]).abs().max()):.17g}, '
          f'weight_gradient_max_abs_error={float((observed_grad[1]-expected_grad[1]).abs().max()):.17g}')
    torch.testing.assert_close(observed,expected,rtol=2e-11,atol=2e-12)
    torch.testing.assert_close(observed_grad[0][0],expected_grad[0],rtol=2e-10,atol=2e-11)
    torch.testing.assert_close(observed_grad[1],expected_grad[1],rtol=2e-10,atol=2e-11)
    assert observed.shape == (1,32) and torch.isfinite(observed).all()
    assert observed_grad[0].abs().max() > 1e-3 and observed_grad[1].abs().max() > 1e-3


def test_historical_default_is_exactly_sequential_and_brickwork_is_distinct():
    generator = torch.Generator().manual_seed(83)
    angles = torch.randn((1,16,3),generator=generator,dtype=torch.float64).requires_grad_()
    weights = torch.randn((1,2,4),generator=generator,dtype=torch.float64).requires_grad_()
    original = torch_hqunet_expvals(angles,weights,4)
    explicit = native_hqunet_expvals(angles,weights,edge_order='sequential')
    assert torch.equal(original,explicit)
    original_grad = torch.autograd.grad(_objective(original),(angles,weights),retain_graph=False)
    explicit_grad = torch.autograd.grad(_objective(explicit),(angles,weights),retain_graph=False)
    assert all(torch.equal(left,right) for left,right in zip(original_grad,explicit_grad))
    with torch.no_grad():
        reordered = native_hqunet_expvals(angles,weights)
    assert (original-reordered).abs().max() > .01
    with pytest.raises(ValueError,match='edge_order'):
        native_hqunet_expvals(angles,weights,edge_order='unknown')


@pytest.mark.parametrize('control',native.CONTROLS)
def test_five_class_shape_fourth_encoder_conv_and_dense_reconstruction(control):
    model = native.build_model(control,seed=42,base_channels=2,image_size=128).eval()
    events = []
    handles = [model.down4[1].register_forward_hook(lambda m,a,out:events.append(('fourth_conv',tuple(out.shape))))]
    if control != 'zero_bottleneck':
        handles.append(model.bottleneck.pre_projection.register_forward_pre_hook(
            lambda m,a:events.append(('pre_q',tuple(a[0].shape)))))
    with torch.no_grad(): output = model(torch.randn((1,3,128,128)))
    for handle in handles:handle.remove()
    assert output.shape == (1,5,128,128) and torch.isfinite(output).all()
    assert events[0] == ('fourth_conv',(1,32,8,8))
    if control != 'zero_bottleneck':assert events[1] == ('pre_q',(1,32,4,4))
    assert model.bottleneck.pre_projection.in_channels == 32
    assert model.bottleneck.post_projection.in_features == 32
    assert model.bottleneck.post_projection.out_features == 32*8*8
    assert len([m for m in model.modules() if isinstance(m,torch.nn.MaxPool2d)]) == 4
    assert len([m for m in model.modules() if isinstance(m,torch.nn.ConvTranspose2d)]) == 4


def test_native_default_width_and_fiveclass_forward():
    model = native.build_model('compact_affine').eval()
    assert model.base_channels == 64 and model.down4[1][1].out_channels == 1024
    with torch.no_grad():output = model(torch.zeros((1,3,128,128)))
    assert output.shape == (1,5,128,128)
    assert model.bottleneck.post_projection.out_features == 1024*8*8
    assert native.parameter_counts(model)['core_total'] == 1568


def test_common_initialization_and_caller_rng_are_preserved():
    torch.manual_seed(314159)
    before = torch.random.get_rng_state().clone()
    models = [native.build_model(control,seed=77,base_channels=2) for control in native.CONTROLS]
    assert torch.equal(before,torch.random.get_rng_state())
    assert len({native.shared_state_sha256(model) for model in models}) == 1
    reference = models[0].state_dict()
    for model in models[1:]:
        for name,value in model.state_dict().items():
            if not name.startswith('bottleneck.core.'):
                assert torch.equal(value,reference[name]),name
    assert torch.equal(models[0].bottleneck.core.weights,models[1].bottleneck.core.weights)
    assert native.shared_state_sha256(native.build_model('quantum',seed=78,base_channels=2)) != native.shared_state_sha256(models[0])


@pytest.mark.parametrize('control',native.CONTROLS)
def test_native_config_strict_restore_reinstates_training_flags(control):
    model = native.build_model(control,seed=123,base_channels=2)
    with torch.no_grad():model.head.bias.add_(.125)
    config = native.model_config(model)
    assert config['format'] == 'hq-landcover-native-v1' and config['circuit_edge_order'] == 'brickwork'
    restored = native.build_from_config(config)
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    assert native.model_config(restored) == config
    assert native.parameter_counts(restored) == native.parameter_counts(model)
    assert native.shared_state_sha256(restored) == native.shared_state_sha256(model)
    assert {name:p.requires_grad for name,p in restored.named_parameters()} == {name:p.requires_grad for name,p in model.named_parameters()}
    expected_core = 1568 if control == 'compact_affine' else 8
    assert native.parameter_counts(restored)['core_total'] == expected_core
    if control == 'frozen_quantum':
        assert all(not p.requires_grad for p in restored.bottleneck.core_parameters())
        assert restored.bottleneck.pre_projection.weight.requires_grad
    elif control == 'zero_bottleneck':
        assert all(not p.requires_grad for p in restored.bottleneck.parameters())
        restored.bottleneck.mode = 'on'
        assert not restored.bottleneck(torch.randn((1,32,8,8))).any()
    broken = copy.deepcopy(model.state_dict());broken.pop('head.bias')
    with pytest.raises(RuntimeError):restored.load_state_dict(broken,strict=True)


@pytest.mark.parametrize('change',[
    {'format':'hq-binary-v1'},{'out_channels':2},{'in_channels':4},
    {'circuit_edge_order':'sequential'},{'out_channels':5.0},{'seed':True},
    {'base_channels':True},{'image_size':32},{'image_size':127},{'control':'mlp'},
])
def test_native_config_rejects_different_architecture_or_bad_values(change):
    config = native.model_config(native.build_model('compact_affine',base_channels=1))
    config.update(change)
    with pytest.raises(ValueError):native.build_from_config(config)


def test_config_rejects_missing_extra_and_binary_models():
    config = native.model_config(native.build_model('compact_affine',base_channels=1))
    missing = dict(config);missing.pop('format')
    extra = dict(config,unknown=True)
    for bad in (missing,extra,None):
        with pytest.raises(ValueError):native.build_from_config(bad)
    with pytest.raises(TypeError):native.model_config(HQDirectionUNet(kind='mlp',base_channels=1))


@pytest.mark.parametrize('control',['quantum','frozen_quantum'])
def test_actual_quantum_input_gradients_survive_core_freezing(control):
    model = native.build_model(control,seed=42,base_channels=1,image_size=64).eval()
    model.zero_grad(set_to_none=True)
    output = model(torch.randn((1,3,64,64)))
    objective = output.square().mean()+output[:,3].mean()
    objective.backward()
    assert model.bottleneck.pre_projection.weight.grad is not None
    assert model.bottleneck.pre_projection.weight.grad.abs().max() > 0
    assert model.down4[1][1].weight.grad is not None and model.down4[1][1].weight.grad.abs().max() > 0
    weights = model.bottleneck.core.weights
    if control == 'quantum':assert weights.grad is not None and torch.isfinite(weights.grad).all() and weights.grad.abs().max() > 0
    else:assert weights.grad is None and not weights.requires_grad
    assert torch.isfinite(model.bottleneck.pre_projection.weight.grad).all()


def test_pre_q_angles_are_tanh_radians_without_pi_and_core_uses_brickwork():
    model = native.build_model('quantum',base_channels=1,image_size=64)
    with torch.no_grad():
        model.bottleneck.pre_projection.weight.zero_()
        model.bottleneck.pre_projection.bias.copy_(torch.tensor([.2,-.7,1.4]))
    x = torch.randn((1,16,4,4))
    angles = model.bottleneck.encode_angles(x)
    torch.testing.assert_close(angles,torch.tensor([.2,-.7,1.4]).tanh().expand(1,16,3))
    assert angles.abs().max() < 1
    with torch.no_grad():
        output = model.bottleneck.core(angles)
        reference = native_hqunet_expvals(angles,model.bottleneck.core.weights).float()
    torch.testing.assert_close(output,reference,rtol=0,atol=0)


@pytest.mark.parametrize('shape',[(0,3,128,128),(1,4,128,128),(1,3,64,64),(3,128,128)])
def test_native_rejects_invalid_input_geometry(shape):
    model = native.build_model('compact_affine',base_channels=1,image_size=128)
    with pytest.raises(ValueError,match='nonempty'):
        model(torch.empty(shape))


@pytest.mark.parametrize('angle_shape,weight_shape',[
    ((0,16,3),(1,2,4)),((9,16,3),(1,2,4)),((1,4,3),(1,2,4)),
    ((1,16,2),(1,2,4)),((1,16,3),(2,2,4)),((1,16,3),(1,24,4)),
])
def test_native_circuit_rejects_unsupported_geometry_before_state_allocation(angle_shape,weight_shape):
    with pytest.raises(ValueError,match='Expected floating angles'):
        native_hqunet_expvals(torch.zeros(angle_shape),torch.zeros(weight_shape))


def test_native_circuit_rejects_nonfinite_and_integer_angles():
    angles=torch.zeros((1,16,3));weights=torch.zeros((1,2,4))
    with pytest.raises(ValueError,match='Expected floating angles'):
        native_hqunet_expvals(angles.long(),weights)
    angles[0,3,2]=float('nan')
    with pytest.raises(ValueError,match='must be finite'):
        native_hqunet_expvals(angles,weights)
