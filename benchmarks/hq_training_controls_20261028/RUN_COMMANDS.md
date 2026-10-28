# Fit, collect and independently audit the control study

The original producer used source `7754773ca2c36020df22a8971e07765b2357cbc8`, the interpreter and package versions in `plan.json`, and the following command. It completed with exit code zero. The existing output directory is retained evidence; choose a different empty work directory for a fresh repeat. `--resume` validates completed fits and replay bindings rather than overwriting them, and refuses an incomplete or failed attempt.

```bash
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
  env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/tmp/quaffle-hq-training-controls-20261028:/tmp/quaffle-wave4a-deps \
  MPLCONFIGDIR=/tmp/quaffle-hqc-mpl \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/bernalde/.gemini/antigravity/brain/1c8d1d6c-6148-446b-a21c-0b9c538c6603/scratch/quaffle-venv/bin/python -u \
  scripts/run_hq_training_controls.py run \
  --data-root /home/bernalde/repos/QUAFFLE/results/local_quantum_followup_20261027/dataset \
  --legacy-root /home/bernalde/repos/QUAFFLE/results/hq_dfc24_direction_20261028/runs/quaffle-hq-dfc24-direction-runs-20261028 \
  --profile /home/bernalde/repos/QUAFFLE/results/hq_training_controls_20261028/profiles/after_review/resource_profile.json \
  --work-dir /home/bernalde/repos/QUAFFLE/results/hq_training_controls_20261028/runs \
  --device cuda --batch-size 8 --epochs 60 --fit-cap 128 \
  --image-size 128 --base-channels 8 --eval-every 2 --augment
```

The eight legacy fits retain their original producer and raw bytes; repeating this follow-up does not make them new replications. The new and historical runtime, source, data and initialization gates must match before reuse. The bound after-review resource probe is retained in `provenance/resource_profile_after_review.json` and in the raw campaign. The earlier probe is historical feasibility evidence; it does not bind the fitted campaign. To generate another probe with the matching environment, use the same bounded environment above with `scripts/run_hq_training_controls.py profile --work-dir <new-empty-profile-directory> --device cuda`. Changed runtime conditions require genuinely matched baselines rather than relabeling the historical observations.

For report regeneration, follow [REPRODUCTION.md](REPRODUCTION.md). Both original collector processes completed with exit zero under separate 3 GiB, zero-swap scopes. Their 114 outputs, including all PNG/PDF figures, were byte-identical without normalization. The retained original generated packets are listed in [the retention inventory](provenance/raw_run_retention.json).

The independent arithmetic audit applies to this captured campaign and a regenerated 114-file packet, before adding supplemental interpretations or review reports:

```bash
systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/bernalde/.gemini/antigravity/brain/1c8d1d6c-6148-446b-a21c-0b9c538c6603/scratch/quaffle-venv/bin/python \
  benchmarks/hq_training_controls_20261028/provenance/independent_numeric_audit.py.txt \
  --work-dir /home/bernalde/repos/QUAFFLE/results/hq_training_controls_20261028/runs \
  --report-dir /tmp/quaffle-hq-training-controls-regenerated \
  --output /tmp/quaffle-hq-training-controls-numeric-audit.json
```

It passed 21,736 checks without importing project or tensor code. The producer and collectors separately establish strict tensor restoration and raw-data identity; this arithmetic audit does not repeat inference. No image data or trained weights are included in Git. Alan owns Anvil execution.
