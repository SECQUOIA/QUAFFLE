#!/bin/bash

#SBATCH -A chm250024-gpu         # allocation name
#SBATCH --nodes=1             # Total # of nodes 
#SBATCH --ntasks-per-node=1   # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=1   # Number of GPUs per node
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=20:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J fl_job               # Job name
#SBATCH -o output_logs/run.o%j        # Name of stdout output file
#SBATCH -e error_logs/run.e%j          # Name of stderr error file
#SBATCH -p gpu               # Queue (partition) name
#SBATCH --mail-user=yi161@purdue.edu
#SBATCH --mail-type=all       # Send email to above address at begin and end of

# Manage processing environment, load compilers, and applications.
module purge
module load modtree/gpu
module load cuda/12.0.1
module load cudnn/cuda-12.0_8.8

# Python environment

source /home/x-danoruo/miniconda3/etc/profile.d/conda.sh

conda activate nasa

which python
python --version

module list

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"
export TF_CPP_MIN_LOG_LEVEL=1

echo "# Running on GPU node"

cd $SLURM_SUBMIT_DIR

echo "## Testing JAX setup"

python -c "
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import jax
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
print('Platform:', jax.default_backend())
gpu_available = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in jax.devices())
print('GPU available:', gpu_available)
if not gpu_available:
    print('GPU not detected, will run on CPU')
else:
    print('GPU detected successfully!')
"

echo "## Running Quantum Vertex UNet model"

python demo.py
