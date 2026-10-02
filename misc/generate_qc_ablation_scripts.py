#!/usr/bin/env python3
"""Generate SLURM scripts for quantum channels ablation study.

Creates scripts for qc={64,128,256} x {random,dirichlet} x {optical,sar,combined}
x the seeds in config/seeds.py (or a --seeds subset).
(qc=32 already exists from prior runs.)
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

QC_VALUES = [64, 128, 256]
PARTITIONS = ["random", "dirichlet"]
DATASETS = ["optical", "sar", "combined"]
ALPHA = 0.5

TEMPLATE = r"""#!/bin/bash

#SBATCH -A YOUR_ACCOUNT         # allocation name
#SBATCH --nodes=1             # Total # of nodes
#SBATCH --ntasks-per-node=1   # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=1   # Number of GPUs per node
#SBATCH --mem=64G
#SBATCH --time=48:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J {job_name}               # Job name
#SBATCH -o output_logs/run.o%j        # Name of stdout output file
#SBATCH -e error_logs/run.e%j          # Name of stderr error file
#SBATCH -p ai                # Queue (partition) name
#SBATCH --mail-user=you@example.com
#SBATCH --mail-type=all       # Send email to above address at begin and end of

# Manage processing environment, load compilers, and applications.
module purge
module load modtree/gpu
module load cuda/12.0.1
#module load cudnn/cuda-12.0_8.8

# Python environment

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate quaffle

which python
python --version

module list

export CUDA_LAUNCH_BLOCKING=1
#export XLA_FLAGS=--xla_gpu_use_cudnn=0
export XLA_FLAGS="--xla_gpu_enable_cudnn_frontend=false"
export TF_CPP_MIN_LOG_LEVEL=1

# Reduce CUDA memory fragmentation across Ray workers sharing a GPU.
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# JAX memory management for federated Ray workers
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export JAX_ENABLE_X64=false

echo "# Running on GPU node"

cd $SLURM_SUBMIT_DIR

echo "## git: $(git -c safe.directory="$PWD" rev-parse HEAD 2>/dev/null || echo unknown) branch=$(git -c safe.directory="$PWD" rev-parse --abbrev-ref HEAD 2>/dev/null) dirty=$(test -z "$(git -c safe.directory="$PWD" status --porcelain --untracked-files=no 2>/dev/null)" && echo clean || echo dirty)"
echo "## host=$(hostname) SLURM_JOB_ID=$SLURM_JOB_ID"

echo "## Running Quantum Vertex UNet model (quantum_channels={qc})"

python federated_demoTorchQ.py {python_args}
"""

def main():
    seeds = _parse_seeds()
    count = 0
    for qc in QC_VALUES:
        for partition in PARTITIONS:
            for dataset in DATASETS:
                for seed in seeds:
                    # Build python args
                    args = [f"--quantum_channels {qc}"]
                    if partition == "dirichlet":
                        args.append(f"--partition dirichlet --alpha {ALPHA}")
                    if dataset != "optical":
                        args.append(f"--dataset {dataset}")
                    args.append(f"--seed {seed}")
                    python_args = " ".join(args)

                    # Script name
                    ds_tag = "" if dataset == "optical" else f"_{dataset}"
                    script_name = f"gpu_job_fed_quantum_qc{qc}_{partition}{ds_tag}_seed{seed}.sh"
                    job_name = f"fed_qc{qc}_{dataset}"

                    script_content = TEMPLATE.format(
                        job_name=job_name,
                        qc=qc,
                        python_args=python_args,
                    )

                    path = os.path.join(MISC_DIR, script_name)
                    with open(path, "w") as f:
                        f.write(script_content)
                    count += 1

    print(f"Generated {count} SLURM scripts in {MISC_DIR}")

if __name__ == "__main__":
    main()
