#!/usr/bin/env python3
"""Benchmark the QuantumBlock circuit simulators and a full training step.

Times
  1. QuantumBlock forward + backward (8 groups) for q_backend 'pennylane' and
     'torch', ansatz legacy and reupload, batch 4 and 16.
  2. A full centralized training step (forward, cross entropy, backward, Adam)
     of QVUNetSegmentation with each q_backend, and of the parameter matched
     ClassicalAblationUNetSegmentation, at the production configuration
     (base_channels 32, image 128, batch 8).

Usage:
    python tools/bench_quantum.py [--device cpu|cuda] [--steps 10] [--quick]
        [--json out.json]
"""
import argparse
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


def bench_block(device, steps, warmup, batches, ansatzes, backends):
    rows = []
    for ans, L in ansatzes:
        for B in batches:
            for qb in backends:
                blk = unet.QuantumBlock(backend='pennylane', quantum_channels=32, seed=0,
                                        q_ansatz=ans, q_layers=L, q_backend=qb).to(device)
                x = torch.randn(B, 32, 8, 8, device=device, requires_grad=True)

                def step():
                    blk.zero_grad(set_to_none=True)
                    x.grad = None
                    blk(x).sum().backward()

                n = steps if qb == 'torch' else max(2, steps // 4)
                dt = _time(step, device, n, warmup)
                rows.append(dict(kind='block', q_ansatz=ans, q_layers=L, batch=B,
                                 q_backend=qb, ms=dt * 1e3))
                print(f"block {ans:8s} L={L} B={B:2d} {qb:9s}: {dt * 1e3:9.2f} ms fwd+bwd",
                      flush=True)
    return rows


def _config(**kw):
    cfg = dict(image_size=128, base_channels=32, init_dim=32, out_dim=2,
               dim_mults=(1, 2, 4, 8), resnet_block_groups=4, quantum_channels=32,
               quantum_backend='pennylane', q_ansatz='reupload', q_layers=2,
               q_share_blocks=True)
    cfg.update(kw)
    return cfg


def bench_train_step(device, steps, warmup, batch, image, variants):
    rows = []
    for name, cls, cfg in variants:
        model = cls(cfg, seed=0).to(device)
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

        dt = _time(step, device, steps, warmup)
        rows.append(dict(kind='train_step', model=name, batch=batch, image=image,
                         s_per_step=dt, batches_per_s=1.0 / dt))
        print(f"train step {name:28s}: {dt * 1e3:9.1f} ms  {1.0 / dt:7.2f} batches/s", flush=True)
    return rows


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
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    info = dict(device=str(device), torch=torch.__version__, threads=torch.get_num_threads(),
                host=platform.node())
    if device.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(device)
    print(info, flush=True)

    rows = []
    if not args.skip_block:
        ansatzes = [('reupload', 2)] if args.quick else [('legacy', 1), ('reupload', 2), ('reupload', 3)]
        rows += bench_block(device, args.steps * 5, args.warmup, (4, 16), ansatzes,
                            ('pennylane', 'torch'))
    if not args.skip_model:
        variants = [
            ('quantum q_backend=pennylane', QVUNetSegmentation, _config(q_backend='pennylane')),
            ('quantum q_backend=torch', QVUNetSegmentation, _config(q_backend='torch')),
            ('classical ablation (mlp)', ClassicalAblationUNetSegmentation, _config()),
        ]
        rows += bench_train_step(device, args.steps, args.warmup, args.batch, args.image, variants)
        ts = {r['model']: r['batches_per_s'] for r in rows if r['kind'] == 'train_step'}
        qp, qt = ts['quantum q_backend=pennylane'], ts['quantum q_backend=torch']
        cl = ts['classical ablation (mlp)']
        print(f"\nspeedup torch vs pennylane (full step): {qt / qp:.2f}x")
        print(f"quantum/classical step time: pennylane {cl / qp:.2f}x, torch {cl / qt:.2f}x")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(dict(info=info, rows=rows), f, indent=2)


if __name__ == '__main__':
    main()
