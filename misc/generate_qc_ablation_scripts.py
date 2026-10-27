#!/usr/bin/env python3
"""Generate SLURM scripts for the quantum channels ablation study.

federated_demoTorchQ.py with --quantum_channels {64,128,256} (or --qc) x
{fed-random, fed-dirichlet} x {optical, sar, combined} x the seeds in
config/seeds.py (or a --seeds subset). qc=32 is the main grid
(misc/generate_grid.py --methods quantum).

Time limits, the script header and the cluster settings (see
config/cluster.example.json) come from misc/generate_grid.py. The circuit runs once per n_qubits-channel group (default 4), so the
quantum overhead in the measured qc=32 runtime is scaled by qc/32. That is
an extrapolation, not a measurement. Run one seed first and check sacct.

Scripts go to misc/grid/qc_<timestamp>/ (or --out-dir), never misc/.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_grid as gg  # noqa: E402

QC_VALUES = [64, 128, 256]
MODES = ["fed-random", "fed-dirichlet"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qc", "--quantum-channels", "--quantum_channels", dest="qc",
                    type=int, nargs="+", default=QC_VALUES)
    ap.add_argument("--modes", nargs="+", choices=gg.ALL_MODES, default=MODES)
    ap.add_argument("--datasets", nargs="+", choices=gg.DATASETS, default=gg.DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=gg.ALL_SEEDS)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--q-ansatz", choices=["reupload", "legacy", "hqunet"], default="reupload")
    ap.add_argument("--q-layers", type=int, default=None)
    ap.add_argument("--n-qubits", type=int, choices=[2, 4, 8, 12, 16], default=None)
    ap.add_argument("--hqunet-grid", type=int, choices=[2, 4], default=None)
    ap.add_argument("--ablation-mode", choices=gg.ABLATION_MODES, default=None)
    ap.add_argument("--ablation-mlp-mode", choices=gg.ABLATION_MLP_MODES, default=None)
    ap.add_argument("--ablation-mlp-hidden", type=int, default=None)
    ap.add_argument("--include-ablation", action="store_true", default=False,
                    help="Also generate parameter-matched classical ablation jobs for each qc")
    ap.add_argument("--q-spatial-mode", choices=["global", "pixel", "patch2x2", "patch4x4", "se_gate"], default=None)
    ap.add_argument("--skip-modulation", choices=["affine", "additive"], default=None)
    ap.add_argument("--backbone-conv-mode", choices=["ws_conv", "standard", "depthwise_sep"], default=None)
    ap.add_argument("--combined-input-mode", choices=["shared3", "concat6", "dual_stem"], default=None)
    ap.add_argument("--loss", choices=["ce", "dice_ce", "focal_dice"], default=None)
    ap.add_argument("--no-time-emb", action="store_true", default=False)
    ap.add_argument("--augment", action="store_true", default=False)
    ap.add_argument("--base-channels", type=int, default=None)
    ap.add_argument("--q-backend", choices=["pennylane", "pennylane_pergroup", "torch"],
                    default=gg.Q_BACKEND)
    ap.add_argument("--no-quantum-knockout-eval", dest="knockout", action="store_false")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--out-dir", default=None)
    gg.add_cluster_args(ap)
    args = ap.parse_args(argv)
    uses_hqunet = (args.q_ansatz == "hqunet") or (args.include_ablation and args.ablation_mode == "hqunet_spatial")
    if uses_hqunet and args.n_qubits is not None and int(args.n_qubits) != 4:
        ap.error(
            "--n-qubits != 4 is not supported with --q-ansatz hqunet or --ablation-mode hqunet_spatial; "
            "use --hqunet-grid in {2, 4}"
        )
    seeds = gg._validate_seeds(ap, args.seeds)
    cluster = gg.resolve_cluster_args(args)
    group_w = 4 if uses_hqunet else int(args.n_qubits or 4)
    bad = [q for q in args.qc if q < group_w or q % group_w]
    if bad:
        ap.error("--qc values must be multiples of %d: %s" % (group_w, bad))
    args.root = os.path.abspath(args.root)
    out_dir = gg.guard_out_dir(ap, args.out_dir or gg.default_out_dir(args.root, "qc_"))
    script = gg.SCRIPT[("quantum", "federated")]
    support = gg.supports_ckpt_every(args.root, script)
    if args.ckpt_every:
        print("ckpt-every: %s %s" % (script, "supported, passing --ckpt-every %d" % args.ckpt_every
                                     if support else "not found in source, not passed"))
    methods = ["quantum", "ablation"] if args.include_ablation else ["quantum"]
    jobs = []
    for qc in args.qc:
        for method in methods:
            for mode in args.modes:
                for dataset in args.datasets:
                    for seed in seeds:
                        jobs.append(gg.build_job(
                            method, mode, dataset, seed, args.root, python=args.python,
                            alpha=args.alpha, q_ansatz=args.q_ansatz, q_layers=args.q_layers,
                            knockout=args.knockout, quantum_channels=qc,
                            ckpt_every=args.ckpt_every, ckpt_support=support,
                            safety=args.safety, floor=args.floor_hours,
                            q_backend=args.q_backend,
                            cpus=cluster["cpus_per_task"], hqunet_grid=args.hqunet_grid,
                            q_spatial_mode=args.q_spatial_mode, base_channels=args.base_channels,
                            skip_modulation=args.skip_modulation,
                            backbone_conv_mode=args.backbone_conv_mode,
                            combined_input_mode=args.combined_input_mode,
                            loss=args.loss, no_time_emb=args.no_time_emb, augment=args.augment,
                            n_qubits=args.n_qubits, ablation_mode=args.ablation_mode,
                            ablation_mlp_mode=args.ablation_mlp_mode,
                            ablation_mlp_hidden=args.ablation_mlp_hidden))
    gg.finish(ap, args, jobs, out_dir)


if __name__ == "__main__":
    main()
