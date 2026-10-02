#!/usr/bin/env python3
"""Generate SLURM scripts for the new bottleneck-matched ablation study.

Emits scripts for the seeds in config/seeds.py (or a --seeds subset) ×
{centralized, federated-random, federated-dirichlet} × {optical, sar, combined}.
3-hour walltime.

The ablation uses ClassicalAblationUNetSegmentation — the fixed architecture
where only the bottleneck QuantumBlock is swapped for a classical module (new,
parameter-matched to +0.001%). This is distinct from the broken
base_channels=28 "matched" ablation whose results live under
results/abalation_*_torch_classical_matched_*.
"""
import argparse
import os
import sys

MISC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MISC_DIR))
from config.seeds import SEEDS as ALL_SEEDS  # noqa: E402  single pre-declared seed list


def _parse_seeds():
    """--seeds lets you regenerate a subset of the canonical list (for refills).
    Seeds outside config/seeds.py are rejected."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS,
                    help="Subset of config/seeds.py to generate (default: all)")
    seeds = ap.parse_args().seeds
    bad = sorted(set(seeds) - set(ALL_SEEDS))
    if bad:
        ap.error(f"seeds {bad} are not in config/seeds.py {ALL_SEEDS}")
    return seeds

PARTITIONS = ["random", "dirichlet"]
DATASETS = ["optical", "sar", "combined"]
ALPHA = 0.5

TEMPLATE = r"""#!/bin/bash

#SBATCH -A YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH -J {job_name}
#SBATCH -o output_logs/run.o%j
#SBATCH -e error_logs/run.e%j
#SBATCH -p ai
#SBATCH --mail-user=you@example.com
#SBATCH --mail-type=all

module purge
module load modtree/gpu
module load cuda/12.0.1

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quaffle

which python
python --version

module list

export CUDA_LAUNCH_BLOCKING=1
export XLA_FLAGS="--xla_gpu_enable_cudnn_frontend=false"
export TF_CPP_MIN_LOG_LEVEL=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

echo "# Running on GPU node"

cd $SLURM_SUBMIT_DIR

echo "## git: $(git -c safe.directory="$PWD" rev-parse HEAD 2>/dev/null || echo unknown) branch=$(git -c safe.directory="$PWD" rev-parse --abbrev-ref HEAD 2>/dev/null) dirty=$(test -z "$(git -c safe.directory="$PWD" status --porcelain --untracked-files=no 2>/dev/null)" && echo clean || echo dirty)"
echo "## host=$(hostname) SLURM_JOB_ID=$SLURM_JOB_ID"

echo "## Running bottleneck-matched classical ablation ({mode}, {dataset}, {part_tag}, seed {seed})"

python abalation_studies.py {python_args}
"""


def main():
    seeds = _parse_seeds()
    count = 0
    for seed in seeds:
        # Centralized (no partition)
        for dataset in DATASETS:
            args = ["--framework torch", "--mode centralized",
                    f"--dataset {dataset}", f"--seed {seed}"]
            python_args = " ".join(args)
            script_name = f"gpu_job_abalation_bottleneck_torch_central_{dataset}_seed{seed}.sh"
            job_name = f"abl_btn_central_{dataset}_s{seed}"
            content = TEMPLATE.format(
                job_name=job_name, python_args=python_args,
                mode="centralized", dataset=dataset, part_tag="-", seed=seed,
            )
            with open(os.path.join(MISC_DIR, script_name), "w") as f:
                f.write(content)
            count += 1

        # Federated (random and dirichlet)
        for partition in PARTITIONS:
            for dataset in DATASETS:
                args = ["--framework torch", "--mode federated",
                        f"--dataset {dataset}", f"--seed {seed}"]
                if partition == "dirichlet":
                    args.append(f"--partition dirichlet --alpha {ALPHA}")
                python_args = " ".join(args)
                script_name = f"gpu_job_abalation_bottleneck_torch_fed_{partition}_{dataset}_seed{seed}.sh"
                job_name = f"abl_btn_fed_{partition}_{dataset}_s{seed}"
                content = TEMPLATE.format(
                    job_name=job_name, python_args=python_args,
                    mode="federated", dataset=dataset, part_tag=partition, seed=seed,
                )
                with open(os.path.join(MISC_DIR, script_name), "w") as f:
                    f.write(content)
                count += 1

    print(f"Generated {count} ablation SLURM scripts in {MISC_DIR}")


if __name__ == "__main__":
    main()
