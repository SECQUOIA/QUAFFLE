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

from tools import training_common as tc  # noqa: E402
from utilsTorch import unetTorch as unet  # noqa: E402
from utilsTorch.utilsTorch import (QVUNetSegmentation,  # noqa: E402
                                   ClassicalAblationUNetSegmentation,
                                   segmentation_loss)

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


def _first_bottleneck_block(model):
    for m in model.modules():
        if isinstance(m, (unet.QuantumBlock, unet.ClassicalBottleneckBlock)):
            return m
    return None


def make_block(
    q_backend,
    q_sim_device,
    q_ansatz,
    q_layers,
    quantum_channels=32,
    q_spatial_mode='global',
    hqunet_grid=2,
    n_qubits=4,
):
    kw = dict(
        backend='pennylane',
        quantum_channels=quantum_channels,
        seed=0,
        q_ansatz=q_ansatz,
        q_layers=q_layers,
        q_backend=q_backend,
        q_spatial_mode=q_spatial_mode,
        hqunet_grid=hqunet_grid,
        n_qubits=int(n_qubits),
    )
    if q_sim_device is not None:
        if not _supports_sim_device():
            raise ValueError('QuantumBlock has no q_sim_device argument on this checkout')
        kw['q_sim_device'] = q_sim_device
    return unet.QuantumBlock(**kw)


def bench_block(
    device,
    steps,
    warmup,
    batches,
    ansatzes,
    sims,
    quantum_channels=32,
    q_spatial_mode='global',
    hqunet_grid=2,
    n_qubits=4,
):
    rows = []
    for ans, L in ansatzes:
        eff_nq = int(n_qubits)
        for B in batches:
            for label, qb, sd in sims:
                try:
                    blk = make_block(
                        qb,
                        sd,
                        ans,
                        L,
                        quantum_channels=quantum_channels,
                        q_spatial_mode=q_spatial_mode,
                        hqunet_grid=hqunet_grid,
                        n_qubits=eff_nq,
                    ).to(device)
                except (ValueError, TypeError, ImportError) as e:
                    print(f"block {ans:8s} L={L} B={B:2d} {label:18s}: skipped ({e})", flush=True)
                    continue
                x = torch.randn(B, quantum_channels, 8, 8, device=device, requires_grad=True)

                def step():
                    blk.zero_grad(set_to_none=True)
                    x.grad = None
                    blk(x).sum().backward()

                # The per-group path is slow, time it with fewer repetitions.
                n = max(1, steps // 4) if qb == 'pennylane_pergroup' else max(1, steps)
                dt = _time(step, device, n, warmup)
                rows.append(dict(kind='block', q_ansatz=ans, q_layers=L, n_qubits=eff_nq,
                                 q_spatial_mode=q_spatial_mode, hqunet_grid=hqunet_grid,
                                 quantum_channels=quantum_channels, batch=B,
                                 simulator=label, q_backend=qb, q_sim_device=sd, reps=n,
                                 ms=dt * 1e3, info=_block_info(blk)))
                print(f"block {ans:8s} L={L} B={B:2d} {label:18s}: {dt * 1e3:9.2f} ms fwd+bwd",
                      flush=True)
    return rows


def _config(**kw):
    bc = int(kw.get('base_channels', 32))
    groups = tc._compatible_group_count(bc, int(kw.get('resnet_block_groups', 4)))
    cfg = dict(image_size=128, base_channels=bc, init_dim=bc, out_dim=2,
               dim_mults=(1, 2, 4, 8), resnet_block_groups=groups, quantum_channels=32,
               quantum_backend='pennylane', q_ansatz='reupload', q_layers=2,
               n_qubits=4, q_spatial_mode='global', hqunet_grid=2,
               q_share_blocks=True)
    cfg.update(kw)
    return cfg


def model_variants(sims, **cfg_kw):
    out = []
    for label, qb, sd in sims:
        cfg = _config(q_backend=qb, **cfg_kw)
        if sd is not None:
            cfg['q_sim_device'] = sd
        out.append(('quantum ' + label, QVUNetSegmentation, cfg, qb, sd))
    ablation_cfg = _config(**cfg_kw)
    if cfg_kw.get('q_ansatz') == 'hqunet' and 'ablation_mode' not in ablation_cfg:
        ablation_cfg['ablation_mode'] = 'hqunet_spatial'
    out.append((ABLATION, ClassicalAblationUNetSegmentation, ablation_cfg, None, None))
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
        b_blk = _first_bottleneck_block(model)
        eff_row_nq = getattr(b_blk, 'n_qubits', int(cfg.get('n_qubits', 4)))
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
        elif b_blk is not None:
            info = {
                'kind': getattr(b_blk, 'kind', None),
                'n_qubits': getattr(b_blk, 'n_qubits', None),
                'num_groups': getattr(b_blk, 'num_groups', None),
                'effective_qc': getattr(b_blk, 'effective_qc', None),
                'bottleneck_target_params': getattr(b_blk, 'bottleneck_target_params', None),
                'bottleneck_actual_params': getattr(b_blk, 'bottleneck_actual_params', None),
            }
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(batch, 3, image, image, generator=gen).to(device)
        y = (torch.rand(batch, 1, image, image, generator=gen) > 0.5).long().to(device)

        loss_mode = str(cfg.get('loss', 'ce'))

        def step():
            opt.zero_grad()
            out = model(x)
            loss = segmentation_loss(out, y.squeeze(1), loss_mode=loss_mode)
            loss.backward()
            opt.step()

        n = max(2, steps // 4) if qb == 'pennylane_pergroup' else steps
        dt = _time(step, device, n, warmup)
        rows.append(dict(kind='train_step', model=name, q_backend=qb, q_sim_device=sd,
                         n_qubits=eff_row_nq, batch=batch, image=image, reps=n, s_per_step=dt,
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--batch', '--batch-size', dest='batch', type=int, default=8,
                    help='full model batch size')
    ap.add_argument('--image', '--image-size', dest='image', type=int, default=128,
                    help='full model image size')
    ap.add_argument('--base-channels', type=int, default=32,
                    help='UNet base channels (default: 32)')
    ap.add_argument('--quantum-channels', '--quantum_channels', dest='quantum_channels',
                    type=int, default=32, help='quantum channels (default: 32)')
    ap.add_argument('--q-ansatz', choices=list(getattr(unet, 'Q_ANSATZ_CHOICES', ('reupload', 'legacy', 'hqunet'))),
                    default=None, help='specific ansatz to benchmark (default: all)')
    ap.add_argument('--q-layers', type=int, default=2, help='circuit layers (default: 2)')
    ap.add_argument('--n-qubits', type=int,
                    choices=list(getattr(unet, 'N_QUBITS_CHOICES', (2, 4, 8, 12, 16))),
                    default=4, help='qubits per group for reupload/legacy (default: 4)')
    ap.add_argument('--q-spatial-mode',
                    choices=list(getattr(unet, 'Q_SPATIAL_MODE_CHOICES', ('global', 'pixel', 'patch2x2', 'patch4x4', 'se_gate'))),
                    default='global', help='spatial bottleneck mode (default: global)')
    ap.add_argument('--hqunet-grid', type=int, choices=[2, 4], default=2,
                    help='2D grid size G in {2, 4} for hqunet (default: 2)')
    ap.add_argument('--skip-modulation', choices=['affine', 'additive'], default='affine',
                    help='quantum bottleneck skip modulation (default: affine)')
    ap.add_argument('--backbone-conv-mode', choices=['standard', 'depthwise_sep'], default='standard',
                    help='UNet backbone convolution mode (default: standard)')
    ap.add_argument('--loss', choices=['ce', 'dice_ce', 'focal_dice'], default='ce',
                    help='segmentation training loss (default: ce)')
    ap.add_argument('--no-time-emb', action='store_true', default=False,
                    help='disable constant time-embedding MLP across ResNet blocks')
    ap.add_argument('--augment', action='store_true', default=False,
                    help='enable geometric training augmentation flag in config')
    ap.add_argument('--quick', action='store_true', help='reupload L=2 only in the block benchmark')
    ap.add_argument('--skip-block', action='store_true')
    ap.add_argument('--skip-model', action='store_true')
    ap.add_argument('--threads', type=int, default=None, help='torch.set_num_threads(N)')
    ap.add_argument('--json', default=None)
    args = ap.parse_args(argv)
    if args.q_ansatz == 'hqunet' and args.n_qubits != 4:
        raise ValueError(
            f"n_qubits={args.n_qubits} is not supported with q_ansatz='hqunet'; "
            "use --hqunet-grid in {2, 4} to control the 2D patch lattice qubit count."
        )

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
        if args.q_ansatz is not None:
            ansatzes = [(args.q_ansatz, 1 if args.q_ansatz == 'legacy' else args.q_layers)]
            batches = (args.batch,) if args.quick else (4, 8, 16)
        elif args.quick:
            ansatzes = [('reupload', 2)]
            batches = (args.batch,) if args.batch != 8 else (4, 8, 16)
        else:
            ansatzes = [('legacy', 1), ('reupload', 2), ('reupload', 3), ('hqunet', 2)]
            batches = (4, 8, 16)
        block_steps = args.steps if args.quick else args.steps * 5
        rows += bench_block(
            device,
            block_steps,
            args.warmup,
            batches,
            ansatzes,
            sims,
            quantum_channels=args.quantum_channels,
            q_spatial_mode=args.q_spatial_mode,
            hqunet_grid=args.hqunet_grid,
            n_qubits=args.n_qubits,
        )
    ratios = {}
    if not args.skip_model:
        model_ansatz = args.q_ansatz or 'reupload'
        model_layers = 1 if model_ansatz == 'legacy' else args.q_layers
        rows += bench_train_step(
            device,
            args.steps,
            args.warmup,
            args.batch,
            args.image,
            model_variants(
                sims,
                image_size=args.image,
                base_channels=args.base_channels,
                quantum_channels=args.quantum_channels,
                q_ansatz=model_ansatz,
                q_layers=model_layers,
                n_qubits=args.n_qubits,
                q_spatial_mode=args.q_spatial_mode,
                hqunet_grid=args.hqunet_grid,
                skip_modulation=args.skip_modulation,
                backbone_conv_mode=args.backbone_conv_mode,
                loss=args.loss,
                use_time_emb=not args.no_time_emb,
                augment=args.augment,
            ),
        )
        ratios = ratios_to_ablation(rows)
        if ratios:
            print("\nfull train step time / classical ablation step time:")
            for k, v in ratios.items():
                print(f"  {k:28s}: {v:5.2f}x")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(dict(info=info, rows=rows, ratio_to_ablation=ratios), f, indent=2,
                      default=str)
    return dict(info=info, rows=rows, ratio_to_ablation=ratios)


if __name__ == '__main__':
    main()
