#!/usr/bin/env python3
"""Generate SLURM scripts for the bottleneck-matched classical ablation.

Thin wrapper around misc/generate_grid.py with --methods ablation: the seeds
in config/seeds.py (or a --seeds subset) x {centralized, fed-random,
fed-dirichlet} x {optical, sar, combined}. Time limits, the script header
and the cluster settings (see config/cluster.example.json) come from
generate_grid.py.

The ablation uses ClassicalAblationUNetSegmentation, the fixed architecture
where only the bottleneck QuantumBlock is swapped for a parameter-matched
classical module. This is distinct from the old base_channels=28 "matched"
ablation whose results live under results/ablation_*_torch_classical_matched_*.

Scripts go to misc/grid/ablation_<timestamp>/ (or --out-dir), never misc/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_grid as gg  # noqa: E402


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        print("All options of generate_grid.py are accepted except --methods.\n")
    if any(a == "--methods" or a.startswith("--methods=") for a in argv):
        sys.exit("--methods is fixed to ablation in this wrapper")
    if not any(a == "--out-dir" or a.startswith("--out-dir=") for a in argv):
        root = gg.REPO_DIR
        for i, a in enumerate(argv):
            if a == "--root" and i + 1 < len(argv):
                root = argv[i + 1]
            elif a.startswith("--root="):
                root = a.split("=", 1)[1]
        argv += ["--out-dir", gg.default_out_dir(os.path.abspath(root), "ablation_")]
    gg.main(["--methods", "ablation"] + argv)


if __name__ == "__main__":
    main()
