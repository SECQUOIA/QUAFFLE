"""q_backend='pennylane' (one broadcast QNode call per block forward) must match
q_backend='pennylane_pergroup' (old per-group loop) and q_backend='torch'.

Run with pytest or directly: python tests/test_pennylane_fast.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilsTorch import unetTorch as unet  # noqa: E402

TOL = {torch.float64: (1e-10, 1e-10), torch.float32: (1e-6, 5e-6)}
BACKENDS = ('pennylane', 'pennylane_pergroup', 'torch')
RESULTS = {}


def _blocks(q_ansatz, q_layers, seed=3):
    out = {}
    for qb in BACKENDS:
        out[qb] = unet.QuantumBlock(backend='pennylane', quantum_channels=32, seed=seed,
                                    q_ansatz=q_ansatz, q_layers=q_layers, q_backend=qb)
    ref = out['pennylane_pergroup'].state_dict()
    g = torch.Generator().manual_seed(1)
    if q_ansatz == 'reupload':
        ref = {k: ((torch.rand(v.shape, generator=g) * 2 - 1) * 3.1 if 'quantum_params' in k
                   else 0.5 + torch.rand(v.shape, generator=g)) for k, v in ref.items()}
    for b in out.values():
        b.load_state_dict(ref)
    return out


def _run(block, x0, dtype):
    block = block.to(dtype)
    x = x0.to(dtype).clone().requires_grad_()
    y = block(x)
    cot = torch.randn(y.shape, generator=torch.Generator().manual_seed(7)).to(y.dtype)
    (y * cot).sum().backward()
    return y.detach(), [p.grad.clone() for p in block.parameters()], x.grad.clone()


def _maxdiff(a, b):
    return (a.double() - b.double()).abs().max().item()


def _check(q_ansatz, q_layers, dtype, B=4):
    blocks = _blocks(q_ansatz, q_layers)
    x0 = torch.randn(B, 40, 3, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    res = {k: _run(b, x0, dtype) for k, b in blocks.items()}
    atol, rtol = TOL[dtype]
    new = res['pennylane']
    for other in ('pennylane_pergroup', 'torch'):
        y, gp, gx = res[other]
        key = f"{q_ansatz} L={q_layers} {str(dtype).replace('torch.', '')} B={B} vs {other}"
        RESULTS[key] = dict(out=_maxdiff(new[0], y),
                            grad_params=max(_maxdiff(a, b) for a, b in zip(new[1], gp)),
                            grad_x=_maxdiff(new[2], gx))
        assert (new[0] - y).abs().max() <= atol, (other, (new[0] - y).abs().max())
        assert len(new[1]) == len(gp)
        # Gradients: float32 sums over batch and pixels are taken in a different
        # order by each path, so compare the max abs difference relative to the
        # gradient scale (same criterion as tests/test_quantum_backend.py).
        for a, b in list(zip(new[1], gp)) + [(new[2], gx)]:
            d = _maxdiff(a, b)
            assert d <= max(atol, rtol * max(1.0, b.double().abs().max().item())), (other, d)
    # forward of the single call is bit-identical to the old per-group path
    assert torch.equal(new[0], res['pennylane_pergroup'][0])


def test_legacy_f64(): _check('legacy', 1, torch.float64)
def test_legacy_f32(): _check('legacy', 1, torch.float32)
def test_reupload_l2_f64(): _check('reupload', 2, torch.float64)
def test_reupload_l2_f32(): _check('reupload', 2, torch.float32)
def test_reupload_l3_f64(): _check('reupload', 3, torch.float64)
def test_reupload_l3_f32(): _check('reupload', 3, torch.float32)
def test_batch_one(): _check('reupload', 2, torch.float32, B=1)


def test_pickle():
    import pickle
    blk = unet.QuantumBlock(backend='pennylane', quantum_channels=32, seed=0, q_backend='pennylane')
    blk2 = pickle.loads(pickle.dumps(blk))
    x = torch.randn(2, 32, 4, 4, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(blk(x), blk2(x))


def _small_cfg(**kw):
    cfg = dict(base_channels=8, init_dim=8, out_dim=2, dim_mults=(1, 2, 4, 8),
               resnet_block_groups=4, quantum_channels=32, quantum_backend='pennylane',
               q_ansatz='reupload', q_layers=2, q_share_blocks=True)
    cfg.update(kw)
    return cfg


def test_simulator_info():
    from utilsTorch.utilsTorch import QVUNetSegmentation
    m = QVUNetSegmentation(_small_cfg(), seed=0)  # no q_backend key: default
    info = m.quantum_simulator_info()
    RESULTS_INFO['default model'] = info
    assert info['q_backend'] == 'pennylane', info
    assert info['device'] == 'default.qubit', info
    assert info['diff_method'] == 'backprop', info
    assert info['interface'] == 'torch', info
    assert info['sim_device'] == 'cpu', info
    assert info['broadcast'] == 'batch x groups, one call per block forward', info
    assert info['pennylane_version'], info
    for qb in BACKENDS:
        blk = unet.QuantumBlock(backend='pennylane', quantum_channels=32, seed=0, q_backend=qb)
        RESULTS_INFO[qb] = blk.simulator_info()
        assert blk.simulator_info()['q_backend'] == qb
    assert RESULTS_INFO['torch']['device'] == 'exact torch statevector simulator'
    m2 = QVUNetSegmentation(_small_cfg(q_sim_device='model'), seed=0)
    assert m2.quantum_simulator_info()['sim_device'] == 'model'


def test_bad_options_rejected():
    for kw in (dict(q_backend='lightning'), dict(q_sim_device='cuda')):
        try:
            unet.QuantumBlock(backend='pennylane', quantum_channels=32, seed=0, **kw)
        except ValueError:
            continue
        raise AssertionError(f"{kw} accepted")


def test_sim_device_model_equal_cpu():
    """q_sim_device='cpu' vs 'model' (on the device given by --device, default cpu)."""
    dev = torch.device(DEVICE)
    x0 = torch.randn(3, 32, 4, 4, generator=torch.Generator().manual_seed(5))
    res = {}
    for sd in ('cpu', 'model'):
        blk = unet.QuantumBlock(backend='pennylane', quantum_channels=32, seed=2,
                                q_backend='pennylane', q_sim_device=sd).to(dev)
        x = x0.to(dev).clone().requires_grad_()
        y = blk(x)
        assert y.device == x.device and y.dtype == torch.float32
        y.sum().backward()
        res[sd] = (y.detach().cpu(), x.grad.cpu(), [p.grad.cpu() for p in blk.parameters()])
    d = max([_maxdiff(res['cpu'][0], res['model'][0]), _maxdiff(res['cpu'][1], res['model'][1])]
            + [_maxdiff(a, b) for a, b in zip(res['cpu'][2], res['model'][2])])
    RESULTS[f'q_sim_device cpu vs model on {DEVICE}'] = dict(max_out_or_grad=d)
    if dev.type == 'cpu':
        assert d == 0.0, d
    else:
        assert d < 1e-5, d


def test_full_model_deepcopy_and_pickle():
    import copy
    import pickle
    from utilsTorch.utilsTorch import QVUNetSegmentation
    x = torch.randn(2, 3, 16, 16, generator=torch.Generator().manual_seed(3))
    for share in (True, False):
        m = QVUNetSegmentation(_small_cfg(q_share_blocks=share), seed=4)
        m.eval()
        with torch.no_grad():
            y = m(x)
        m2 = copy.deepcopy(m)
        m3 = pickle.loads(pickle.dumps(m))
        with torch.no_grad():
            assert torch.equal(y, m2(x)), share
            assert torch.equal(y, m3(x)), share
        assert m2.quantum_simulator_info() == m.quantum_simulator_info()


DEVICE = 'cpu'
RESULTS_INFO = {}


if __name__ == '__main__':
    if '--device' in sys.argv:
        DEVICE = sys.argv[sys.argv.index('--device') + 1]
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print("\nmax abs differences (pennylane single call vs reference):")
    for k, v in RESULTS.items():
        print(f"  {k}: " + ", ".join(f"{a}={b:.3e}" for a, b in v.items()))
    print("\nsimulator_info():")
    for k, v in RESULTS_INFO.items():
        print(f"  {k}: {v}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
