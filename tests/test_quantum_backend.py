"""Equivalence tests for the QuantumBlock circuit simulators.

q_backend='pennylane' (one broadcast default.qubit QNode call per forward),
q_backend='pennylane_pergroup' (one QNode per group, the original path) and
q_backend='torch' (exact batched statevector simulation) must agree in outputs
and gradients. The per-group PennyLane path is the reference.

Run with pytest or directly: python tests/test_quantum_backend.py
"""
import copy
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utilsTorch import unetTorch as unet  # noqa: E402

TOL32 = 1e-6
TOL64 = 1e-10
TOL32_GRAD_REL = 5e-6
RESULTS = {}
REF = 'pennylane_pergroup'
BACKENDS = ('pennylane', 'pennylane_pergroup', 'torch')
OTHERS = tuple(b for b in BACKENDS if b != REF)


def _blocks(q_ansatz, q_layers, seed=3, groups=8):
    """One block per q_backend, all with the reference block's parameters."""
    out = {qb: unet.QuantumBlock(backend='pennylane', quantum_channels=4 * groups, seed=seed,
                                 q_ansatz=q_ansatz, q_layers=q_layers, q_backend=qb)
           for qb in BACKENDS}
    for qb in OTHERS:
        out[qb].load_state_dict(out[REF].state_dict())
    return out


def _randomize(block, gen):
    # Random scales too, so the gradient wrt s is tested away from s = 1
    with torch.no_grad():
        for p in block.quantum_params:
            p.copy_((torch.rand(p.shape, generator=gen) * 2 - 1) * 3.1)
        for s in block.input_scales:
            s.copy_(0.5 + torch.rand(s.shape, generator=gen))


def _run(block, x, dtype):
    block = block.to(dtype)
    x = x.clone().to(dtype).requires_grad_(True)
    y = block(x)
    # Weighted sum so every output channel contributes a distinct gradient
    wts = torch.linspace(-1.0, 1.0, y.shape[1], dtype=y.dtype).view(1, -1, 1, 1)
    (y * wts).sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in block.named_parameters()}
    return y.detach(), x.grad.detach().clone(), grads


def _maxdiff(a, b):
    return (a.double() - b.double()).abs().max().item()


def _reldiff(a, b):
    """Max abs difference divided by max(1, max |a|)."""
    return _maxdiff(a, b) / max(1.0, a.double().abs().max().item())


def check_block(q_ansatz, q_layers, dtype, batch=5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    blocks = _blocks(q_ansatz, q_layers)
    _randomize(blocks[REF], gen)
    for qb in OTHERS:
        blocks[qb].load_state_dict(blocks[REF].state_dict())
    x = torch.randn(batch, 32, 6, 6, generator=gen) * 1.5
    ya, gxa, gpa = _run(blocks[REF], x, dtype)
    out = {}
    for qb in OTHERS:
        yb, gxb, gpb = _run(blocks[qb], x, dtype)
        assert set(gpa) == set(gpb)
        wn = [n for n in gpa if 'quantum_params' in n]
        sn = [n for n in gpa if 'input_scales' in n]
        res = dict(
            out=_maxdiff(ya, yb),
            grad_x=_maxdiff(gxa, gxb),
            grad_w=max(_maxdiff(gpa[n], gpb[n]) for n in wn),
            grad_s=max([_maxdiff(gpa[n], gpb[n]) for n in sn] or [0.0]),
            rel_grad_x=_reldiff(gxa, gxb),
            rel_grad_w=max(_reldiff(gpa[n], gpb[n]) for n in wn),
            rel_grad_s=max([_reldiff(gpa[n], gpb[n]) for n in sn] or [0.0]),
            max_abs_grad_w=max(gpa[n].abs().max().item() for n in wn),
        )
        key = f"{q_ansatz} L={q_layers} {str(dtype).replace('torch.', '')} {REF} vs {qb}"
        RESULTS[key] = res
        # All paths cast the block output to float32, so outputs agree to
        # float32 resolution for any parameter dtype.
        assert res['out'] < TOL32, (key, res)
        if qb == 'pennylane':
            # Same QNode gates, only broadcast: forward is bit-identical
            assert torch.equal(ya, yb), key
        if dtype == torch.float64:
            # Exact same math: gradients agree to double precision
            for k in ('grad_x', 'grad_w', 'grad_s'):
                assert res[k] < TOL64, (key, k, res)
        else:
            # float32 gradients are float32 sums over batch and pixels taken in
            # a different order by the paths, so compare relative to their scale.
            for k in ('rel_grad_x', 'rel_grad_w', 'rel_grad_s'):
                assert res[k] < TOL32_GRAD_REL, (key, k, res)
        out[qb] = res
    return out


def test_reupload_l2_f32():
    check_block('reupload', 2, torch.float32)


def test_reupload_l3_f32():
    check_block('reupload', 3, torch.float32)


def test_legacy_f32():
    check_block('legacy', 1, torch.float32)


def test_reupload_l2_f64():
    check_block('reupload', 2, torch.float64)


def test_reupload_l3_f64():
    check_block('reupload', 3, torch.float64)


def test_legacy_f64():
    check_block('legacy', 1, torch.float64)


def test_closed_form_legacy():
    """Legacy outputs equal the closed form products listed in docs/MODEL.md."""
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(7, 3, 4, generator=gen, dtype=torch.float64)
    th = torch.randn(3, 4, generator=gen, dtype=torch.float64)
    c = torch.cos(x) * torch.cos(th)
    ref = torch.stack([c[..., 1] * c[..., 2] * c[..., 3], c[..., 0] * c[..., 1],
                       c[..., 0] * c[..., 1] * c[..., 2], c.prod(-1)], -1)
    d = _maxdiff(unet.torch_legacy_expvals(x, th), ref)
    RESULTS['legacy closed form'] = dict(out=d)
    assert d < TOL64, d


def test_state_dict_and_param_count():
    for ans, L in (('legacy', 1), ('reupload', 2), ('reupload', 3)):
        blocks = _blocks(ans, L)
        a = blocks[REF]
        for qb in OTHERS:
            b = blocks[qb]
            assert list(a.state_dict()) == list(b.state_dict())
            for k in a.state_dict():
                assert a.state_dict()[k].shape == b.state_dict()[k].shape
                assert torch.equal(a.state_dict()[k], b.state_dict()[k])
        counts = {qb: sum(p.numel() for p in m.parameters()) for qb, m in blocks.items()}
        assert set(counts.values()) == {8 * a.n_params_per_group}, (ans, L, counts)
    # Same seed gives the same init for every backend without load_state_dict
    built = {qb: unet.QuantumBlock(backend='pennylane', quantum_channels=32, seed=11, q_backend=qb)
             for qb in BACKENDS}
    for qb in OTHERS:
        for k in built[REF].state_dict():
            assert torch.equal(built[REF].state_dict()[k], built[qb].state_dict()[k])


def test_default_q_backend_is_pennylane():
    blk = unet.QuantumBlock(backend='pennylane', quantum_channels=32, seed=0)
    assert blk.q_backend == 'pennylane' and blk.q_sim_device == 'cpu'
    assert blk.pennylane_circuits == []


def test_no_global_rng_change():
    for qb in BACKENDS:
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        blk = unet.QuantumBlock(backend='pennylane', quantum_channels=32, seed=5, q_backend=qb)
        x = torch.randn(2, 32, 4, 4, generator=torch.Generator().manual_seed(0))
        blk(x)
        blk.set_quantum_mode('random', knockout_seed=4)
        blk(x).sum()
        assert torch.equal(before, torch.get_rng_state()), qb


def test_knockout_modes():
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(3, 32, 4, 4, generator=gen)
    for ans, L in (('legacy', 1), ('reupload', 2)):
        blocks = _blocks(ans, L)
        ref = blocks[REF]
        trained = {k: v.clone() for k, v in ref.state_dict().items()}
        for m in blocks.values():
            m.eval()
        with torch.no_grad():
            on = {qb: m(x) for qb, m in blocks.items()}
            for qb in OTHERS:
                assert _maxdiff(on[REF], on[qb]) < TOL32, (ans, qb)
            for m in blocks.values():
                m.set_quantum_mode('zero')
            for qb, m in blocks.items():
                z = m(x)
                assert torch.count_nonzero(z) == 0 and z.shape == on[REF].shape, qb
                assert z.device == x.device
            for m in blocks.values():
                m.set_quantum_mode('random', knockout_seed=17)
            rnd = {qb: m(x) for qb, m in blocks.items()}
            for qb in OTHERS:
                d = _maxdiff(rnd[REF], rnd[qb])
                RESULTS[f'knockout random {ans} {REF} vs {qb}'] = dict(out=d)
                assert d < TOL32, (qb, d)
            # Same circuit gates in both PennyLane paths
            assert torch.equal(rnd[REF], rnd['pennylane'])
            assert _maxdiff(rnd[REF], on[REF]) > 1e-3
            # Same knockout seed gives the same random circuit again
            for qb, m in blocks.items():
                m.set_quantum_mode('random', knockout_seed=17)
                assert torch.equal(m(x), rnd[qb]), qb
            for m in blocks.values():
                m.set_quantum_mode('on')
            for qb, m in blocks.items():
                assert torch.equal(m(x), on[qb]), qb
        for k, v in ref.state_dict().items():
            assert torch.equal(v, trained[k])


def test_set_q_backend_switch():
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(2, 32, 4, 4, generator=gen)
    blk = unet.QuantumBlock(backend='pennylane', quantum_channels=32, seed=1, q_backend='torch')
    ys = {}
    with torch.no_grad():
        for qb in ('torch', 'pennylane_pergroup', 'pennylane', 'torch'):
            blk.set_q_backend(qb)
            ys.setdefault(qb, blk(x))
    assert blk.pennylane_circuits == []
    for qb in OTHERS:
        assert _maxdiff(ys[REF], ys[qb]) < TOL32
    assert torch.equal(ys[REF], ys['pennylane'])
    # q_sim_device='model' on a CPU input is the same computation
    blk.set_q_backend('pennylane').set_q_sim_device('model')
    with torch.no_grad():
        assert torch.equal(blk(x), ys['pennylane'])


def _full_models(q_ansatz, q_layers, share=True):
    from utilsTorch.utilsTorch import QVUNetSegmentation
    cfg = dict(base_channels=8, init_dim=8, out_dim=2, dim_mults=(1, 2, 4, 8),
               resnet_block_groups=4, quantum_channels=32, quantum_backend='pennylane',
               q_ansatz=q_ansatz, q_layers=q_layers, q_share_blocks=share)
    models = {REF: QVUNetSegmentation(dict(cfg, q_backend=REF), seed=21)}
    for k, qb in enumerate(OTHERS):
        models[qb] = QVUNetSegmentation(dict(cfg, q_backend=qb), seed=99 + k)
        models[qb].load_state_dict(models[REF].state_dict())
    return models


def check_full_model(q_ansatz, q_layers, share=True):
    models = _full_models(q_ansatz, q_layers, share)
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(2, 3, 16, 16, generator=gen)
    masks = (torch.rand(2, 1, 16, 16, generator=gen) > 0.5).long()
    res = {}
    for qb, m in models.items():
        m.train()
        out = m(x)
        loss = torch.nn.functional.cross_entropy(out, masks.squeeze(1))
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
        res[qb] = (out.detach(), loss.item(), grads)
    r0 = res[REF]
    for qb in OTHERS:
        r1 = res[qb]
        d_out = _maxdiff(r0[0], r1[0])
        d_loss = abs(r0[1] - r1[1])
        d_grad = max(_maxdiff(r0[2][n], r1[2][n]) for n in r0[2])
        qnames = [n for n in r0[2] if 'quantum_params' in n or 'input_scales' in n]
        d_qgrad = max(_maxdiff(r0[2][n], r1[2][n]) for n in qnames)
        key = f"full model {q_ansatz} L={q_layers} share={share} {REF} vs {qb}"
        RESULTS[key] = dict(out=d_out, loss=d_loss, grad_all=d_grad, grad_q=d_qgrad)
        assert d_out < 1e-5, (qb, d_out)
        assert d_grad < 1e-5, (qb, d_grad)
        assert d_qgrad < 1e-6, (qb, d_qgrad)
        if qb == 'pennylane':
            assert torch.equal(r0[0], r1[0]), key


def test_full_model_reupload():
    check_full_model('reupload', 2, True)


def test_full_model_reupload_unshared():
    check_full_model('reupload', 2, False)


def test_full_model_legacy():
    check_full_model('legacy', 1, True)


def test_full_param_count_unchanged():
    from utilsTorch.utilsTorch import QVUNetSegmentation
    cfg = dict(base_channels=32, init_dim=32, out_dim=2, dim_mults=(1, 2, 4, 8),
               resnet_block_groups=4, quantum_channels=32, quantum_backend='pennylane',
               q_ansatz='reupload', q_layers=2, q_share_blocks=True)
    n = {}
    for qb in BACKENDS + (None,):
        c = dict(cfg) if qb is None else dict(cfg, q_backend=qb)
        m = QVUNetSegmentation(c, seed=0)
        n[qb or 'default'] = sum(p.numel() for p in m.parameters())
    RESULTS['param count reupload L=2 shared'] = n
    assert set(n.values()) == {8_758_370}, n


def test_state_dict_cross_load():
    """A checkpoint written by a q_backend='torch' model loads into a
    q_backend='pennylane' model with strict=True and gives the same outputs."""
    from utilsTorch.utilsTorch import QVUNetSegmentation
    cfg = dict(base_channels=8, init_dim=8, out_dim=2, dim_mults=(1, 2, 4, 8),
               resnet_block_groups=4, quantum_channels=32, quantum_backend='pennylane',
               q_ansatz='reupload', q_layers=2, q_share_blocks=True)
    mt = QVUNetSegmentation(dict(cfg, q_backend='torch'), seed=5)
    mp = QVUNetSegmentation(dict(cfg, q_backend='pennylane'), seed=77)
    sd = copy.deepcopy(mt.state_dict())
    mp.model.load_state_dict(sd, strict=True)
    x = torch.randn(2, 3, 16, 16, generator=torch.Generator().manual_seed(8))
    mt.eval()
    mp.eval()
    with torch.no_grad():
        d = _maxdiff(mt(x), mp(x))
    RESULTS['state_dict torch -> pennylane'] = dict(out=d)
    assert d < 1e-5, d


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # report and continue
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print("\nmax abs differences:")
    for k, v in RESULTS.items():
        print(f"  {k}: " + ", ".join(f"{a}={b:.3e}" if isinstance(b, float) else f"{a}={b}"
                                      for a, b in v.items()))
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
