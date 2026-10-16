#!/usr/bin/env python3
"""Benchmark the QuantumBlock circuit simulators and a full training step.

Times
  1. QuantumBlock forward + backward (8 groups) for every circuit simulator,
     ansatz legacy and reupload (L=2, L=3), batch 4, 8 (production) and 16.
  2. A full centralized training step (forward, cross entropy, backward, Adam)
     of QVUNetSegmentation with each simulator, and of the parameter matched
     ClassicalAblationUNetSegmentation, at the production configuration
     (base_channels 32, image 128, batch 8). Prints the ratio of every
     quantum step to the ablation step.

Simulators (q_backend, q_sim_device)
  pennylane, cpu       one broadcast default.qubit QNode call per block
                       forward, circuit evaluated on the CPU (default)
  pennylane, model     same call, circuit evaluated on the input's device
                       (only benchmarked when --device is cuda)
  pennylane_pergroup   the previous PennyLane path, one QNode call per
                       4-channel group (legacy: per group and batch item)
  torch                exact torch statevector simulator (cross-check)

Variants that the checked out code does not accept (for example
q_backend='pennylane_pergroup' or q_sim_device before the single-call
PennyLane path exists) are skipped with a message. On such a checkout
'pennylane' is the per-group path, which the printed simulator info shows.

Usage:
    python tools/bench_quantum.py [--device cpu|cuda] [--steps 10] [--quick]
        [--skip-block] [--skip-model] [--threads N] [--json out.json]
"""
import argparse
import inspect
import json
import os
import platform
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utilsTorch import unetTorch as unet  # noqa: E402
from utilsTorch.utilsTorch import (QVUNetSegmentation,  # noqa: E402
                                   ClassicalAblationUNetSegmentation)

ABLATION = 'classical ablation (mlp)'


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _time(fn, device, steps, warmup):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / steps


def _supports_sim_device():
    return 'q_sim_device' in inspect.signature(unet.QuantumBlock.__init__).parameters


def simulators(device):
    """(label, q_backend, q_sim_device or None) in benchmark order."""
    if not _supports_sim_device():
        # Checkout without the single-call path: 'pennylane' is the per-group
        # loop there. Benchmark it under its own label and skip q_sim_device.
        print("note: QuantumBlock has no q_sim_device on this checkout, benchmarking "
              "q_backend='pennylane' as it is (per-group QNode calls) and skipping the "
              "q_sim_device variants", flush=True)
        sims = [('pennylane (checkout)', 'pennylane', None)]
    else:
        sims = [('pennylane cpu', 'pennylane', 'cpu')]
        if device.type == 'cuda':
            sims.append(('pennylane model', 'pennylane', 'model'))
    sims += [('pennylane_pergroup', 'pennylane_pergroup', None),
             ('torch', 'torch', None)]
    return sims


def _block_info(blk):
    info = getattr(blk, 'simulator_info', None)
    if callable(info):
        try:
            return info()
        except Exception as e:  # informational only
            return {'error': repr(e)}
    return {'q_backend': getattr(blk, 'q_backend', None), 'simulator_info': 'not available'}


def _first_quantum_block(model):
    for m in model.modules():
        if isinstance(m, unet.QuantumBlock):
            return m
    return None


def make_block(q_backend, q_sim_device, q_ansatz, q_layers):
    kw = dict(backend='pennylane', quantum_channels=32, seed=0, q_ansatz=q_ansatz,
              q_layers=q_layers, q_backend=q_backend)
    if q_sim_device is not None:
        if not _supports_sim_device():
            raise ValueError('QuantumBlock has no q_sim_device argument on this checkout')
        kw['q_sim_device'] = q_sim_device
    return unet.QuantumBlock(**kw)


def bench_block(device, steps, warmup, batches, ansatzes, sims):
    rows = []
    for ans, L in ansatzes:
        for B in batches:
            for label, qb, sd in sims:
                try:
                    blk = make_block(qb, sd, ans, L).to(device)
                except (ValueError, TypeError, ImportError) as e:
                    print(f"block {ans:8s} L={L} B={B:2d} {label:18s}: skipped ({e})", flush=True)
                    continue
                x = torch.randn(B, 32, 8, 8, device=device, requires_grad=True)

                def step():
                    blk.zero_grad(set_to_none=True)
                    x.grad = None
                    blk(x).sum().backward()

                # The per-group path is slow, time it with fewer repetitions.
                n = max(2, steps // 4) if qb == 'pennylane_pergroup' else steps
                dt = _time(step, device, n, warmup)
                rows.append(dict(kind='block', q_ansatz=ans, q_layers=L, batch=B,
                                 simulator=label, q_backend=qb, q_sim_device=sd, reps=n,
                                 ms=dt * 1e3, info=_block_info(blk)))
                print(f"block {ans:8s} L={L} B={B:2d} {label:18s}: {dt * 1e3:9.2f} ms fwd+bwd",
                      flush=True)
    return rows


def _config(**kw):
    cfg = dict(image_size=128, base_channels=32, init_dim=32, out_dim=2,
               dim_mults=(1, 2, 4, 8), resnet_block_groups=4, quantum_channels=32,
               quantum_backend='pennylane', q_ansatz='reupload', q_layers=2,
               q_share_blocks=True)
    cfg.update(kw)
    return cfg


def model_variants(sims):
    out = []
    for label, qb, sd in sims:
        cfg = _config(q_backend=qb)
        if sd is not None:
            cfg['q_sim_device'] = sd
        out.append(('quantum ' + label, QVUNetSegmentation, cfg, qb, sd))
    out.append((ABLATION, ClassicalAblationUNetSegmentation, _config(), None, None))
    return out


def bench_train_step(device, steps, warmup, batch, image, variants):
    rows = []
    for name, cls, cfg, qb, sd in variants:
        try:
            model = cls(cfg, seed=0).to(device)
        except (ValueError, TypeError, ImportError) as e:
            print(f"train step {name:28s}: skipped ({e})", flush=True)
            continue
        info = None
        if qb is not None:
            blk = _first_quantum_block(model)
            if blk is None or getattr(blk, 'q_backend', None) != qb:
                print(f"train step {name:28s}: skipped (model built q_backend="
                      f"{getattr(blk, 'q_backend', None)!r}, not {qb!r})", flush=True)
                continue
            if sd is not None and getattr(blk, 'q_sim_device', None) != sd:
                print(f"train step {name:28s}: skipped (q_sim_device not supported on this "
                      f"checkout)", flush=True)
                continue
            info = _block_info(blk)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(batch, 3, image, image, generator=gen).to(device)
        y = (torch.rand(batch, 1, image, image, generator=gen) > 0.5).long().to(device)

        def step():
            opt.zero_grad()
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y.squeeze(1))
            loss.backward()
            opt.step()

        n = max(2, steps // 4) if qb == 'pennylane_pergroup' else steps
        dt = _time(step, device, n, warmup)
        rows.append(dict(kind='train_step', model=name, q_backend=qb, q_sim_device=sd,
                         batch=batch, image=image, reps=n, s_per_step=dt,
                         batches_per_s=1.0 / dt, info=info))
        print(f"train step {name:28s}: {dt * 1e3:9.1f} ms  {1.0 / dt:7.2f} batches/s", flush=True)
        del model, opt
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return rows


def ratios_to_ablation(rows):
    ts = {r['model']: r['s_per_step'] for r in rows if r['kind'] == 'train_step'}
    if ABLATION not in ts:
        return {}
    base = ts[ABLATION]
    return {k: v / base for k, v in ts.items() if k != ABLATION}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--batch', type=int, default=8, help='full model batch size')
    ap.add_argument('--image', type=int, default=128, help='full model image size')
    ap.add_argument('--quick', action='store_true', help='reupload L=2 only in the block benchmark')
    ap.add_argument('--skip-block', action='store_true')
    ap.add_argument('--skip-model', action='store_true')
    ap.add_argument('--threads', type=int, default=None, help='torch.set_num_threads(N)')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    sims = simulators(device)
    info = dict(device=str(device), torch=torch.__version__, threads=torch.get_num_threads(),
                host=platform.node(), q_backend_choices=list(getattr(unet, 'Q_BACKEND_CHOICES', [])),
                q_sim_device_supported=_supports_sim_device(),
                cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING', ''))
    try:
        import pennylane as qml
        info['pennylane'] = qml.__version__
    except ImportError:
        info['pennylane'] = None
    if device.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(device)
    print(info, flush=True)

    rows = []
    if not args.skip_block:
        ansatzes = [('reupload', 2)] if args.quick else [('legacy', 1), ('reupload', 2), ('reupload', 3)]
        rows += bench_block(device, args.steps * 5, args.warmup, (4, 8, 16), ansatzes, sims)
    ratios = {}
    if not args.skip_model:
        rows += bench_train_step(device, args.steps, args.warmup, args.batch, args.image,
                                 model_variants(sims))
        ratios = ratios_to_ablation(rows)
        if ratios:
            print("\nfull train step time / classical ablation step time:")
            for k, v in ratios.items():
                print(f"  {k:28s}: {v:5.2f}x")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(dict(info=info, rows=rows, ratio_to_ablation=ratios), f, indent=2,
                      default=str)


if __name__ == '__main__':
    main()
