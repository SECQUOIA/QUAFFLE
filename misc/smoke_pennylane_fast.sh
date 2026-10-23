#!/bin/bash
# Template SLURM directives. #SBATCH lines cannot read environment variables,
# so either edit YOUR_ACCOUNT here or override it on the sbatch command line
# (command-line flags take precedence over #SBATCH lines), for example
#   sbatch -A <account> -p <gpu partition> misc/smoke_pennylane_fast.sh
# The log paths are relative to the directory sbatch is run from, which must
# already contain output_logs/ and error_logs/.
#SBATCH -A YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -J plfast_smoke
#SBATCH -o output_logs/%x.o%j
#SBATCH -e error_logs/%x.e%j
#
# Smoke and benchmark job for the single-call PennyLane quantum block.
#
# Runs the code of the checkout ROOT but writes every result, log and
# benchmark file under SMOKE_ROOT. The demo scripts write to the relative path
# results/..., and read data/... relative to the working directory, so the
# training commands run with SMOKE_ROOT as the working directory (which holds
# a data symlink to ROOT/data) and call the scripts by absolute path in ROOT.
# Nothing is written to ROOT/results.
#
# Environment (all optional):
#   ROOT          checkout to run (default: the sbatch submit directory)
#   SMOKE_ROOT    output folder (default: $ROOT/smoke_plfast)
#   PY            python interpreter (default: QUAFFLE_PYTHON, else python)
#   QUAFFLE_*     cluster settings. When config/cluster.local.json exists in
#                 ROOT its values are loaded through misc/generate_grid.py
#                 --print-cluster-env. Already exported QUAFFLE_* variables
#                 take precedence. Used here: QUAFFLE_MODULES,
#                 QUAFFLE_CONDA_SH, QUAFFLE_CONDA_ENV, QUAFFLE_PYTHON,
#                 QUAFFLE_GIT_SAFE_DIRECTORY (true: mark ROOT as a git
#                 safe.directory for the provenance line, default off).
# Pass variables with  sbatch --export=ALL,ROOT=/path misc/smoke_pennylane_fast.sh
#
# CUDA_LAUNCH_BLOCKING is NOT exported. Step 1 sets it for one benchmark run
# only, to measure its cost.

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
SMOKE_ROOT=${SMOKE_ROOT:-$ROOT/smoke_plfast}

if [ -f "$ROOT/config/cluster.local.json" ] && command -v python3 >/dev/null 2>&1; then
    eval "$(python3 "$ROOT/misc/generate_grid.py" --print-cluster-env)" || exit 1
fi
PY=${PY:-${QUAFFLE_PYTHON:-python}}

if [ -n "$QUAFFLE_MODULES" ]; then
    module purge
    for m in $QUAFFLE_MODULES; do
        module load "$m"
    done
fi
if [ -n "$QUAFFLE_CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$QUAFFLE_CONDA_SH"
fi
if [ -n "$QUAFFLE_CONDA_ENV" ]; then
    conda activate "$QUAFFLE_CONDA_ENV"
fi
export XLA_FLAGS="--xla_gpu_enable_cudnn_frontend=false"
export TF_CPP_MIN_LOG_LEVEL=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
case "${QUAFFLE_GIT_SAFE_DIRECTORY:-false}" in
    1|true|True|TRUE|yes|on)
        export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$ROOT"
        ;;
esac
export PYTHONDONTWRITEBYTECODE=1
# The scripts run from SMOKE_ROOT, so make the code in ROOT importable for
# the main process and for the Ray workers of the federated simulation.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_LAUNCH_BLOCKING

mkdir -p "$SMOKE_ROOT/bench" "$SMOKE_ROOT/results" || exit 1
if [ ! -e "$SMOKE_ROOT/data" ]; then
    ln -s "$ROOT/data" "$SMOKE_ROOT/data" || exit 1
fi
cd "$SMOKE_ROOT" || exit 1

echo "ROOT=$ROOT SMOKE_ROOT=$SMOKE_ROOT PY=$PY"
echo "git: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown) branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null) dirty=$(test -z "$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null)" && echo clean || echo dirty)"
echo "host=$(hostname) SLURM_JOB_ID=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv

TAG=${SLURM_JOB_ID:-manual}
SUMMARY=()
FAILED=0

step() {
    # step NAME CMD...   run CMD, print and record its elapsed seconds and exit code
    local name=$1
    shift
    echo
    echo "=== $name: $*"
    local t0 t1 rc
    t0=$(date +%s)
    "$@"
    rc=$?
    t1=$(date +%s)
    echo "=== $name: elapsed $((t1 - t0)) s, exit $rc"
    SUMMARY+=("$name elapsed=$((t1 - t0))s exit=$rc")
    if [ "$rc" -ne 0 ]; then
        FAILED=1
    fi
    return "$rc"
}

# 1. Benchmark, once with CUDA_LAUNCH_BLOCKING=1 and once without.
step bench_blocking env CUDA_LAUNCH_BLOCKING=1 "$PY" "$ROOT/tools/bench_quantum.py" \
    --device cuda --json "$SMOKE_ROOT/bench/gpu_pl_fast_blocking_$TAG.json"
step bench_noblocking "$PY" "$ROOT/tools/bench_quantum.py" \
    --device cuda --json "$SMOKE_ROOT/bench/gpu_pl_fast_$TAG.json"

# 2. GPU exactness (q_sim_device cpu vs model vs torch simulator on CUDA).
if [ -f "$ROOT/tests/test_pennylane_fast.py" ]; then
    step exactness_gpu "$PY" "$ROOT/tests/test_pennylane_fast.py" --device cuda
else
    echo "tests/test_pennylane_fast.py not found in $ROOT, GPU exactness step failed"
    SUMMARY+=("exactness_gpu exit=1 (test file missing)")
    FAILED=1
fi

# 3. Centralized training smoke (torch simulator grid run took 0.16 h).
step central_optical "$PY" "$ROOT/centralized_demoTorchQ.py" --dataset optical --seed 17 \
    --q-ansatz reupload --q-backend pennylane --quantum-knockout-eval

# 4. Federated training smoke (torch simulator grid run took 0.65 h).
step federated_random_optical "$PY" "$ROOT/federated_demoTorchQ.py" --partition random \
    --dataset optical --seed 17 --q-ansatz reupload --q-backend pennylane \
    --quantum-knockout-eval --ckpt-every 5

echo
echo "=== go criteria evidence"
for d in "$SMOKE_ROOT"/results/flood_*_seed17; do
    [ -d "$d" ] || continue
    echo "--- $d"
    grep -i "simulator\|Model:" "$d/results/summary.txt" 2>/dev/null
    grep -o '"q_backend": *"[^"]*"' "$d/results/metrics.json" 2>/dev/null
done
echo
echo "=== summary"
printf '%s\n' "${SUMMARY[@]}"
if [ "$FAILED" -ne 0 ]; then
    echo "=== smoke_pennylane_fast: FAILED (one or more required steps exited non-zero)" >&2
    exit 1
fi
