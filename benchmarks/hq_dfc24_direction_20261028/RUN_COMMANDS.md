# Fit and regenerate the direction screen

The original eight-fit producer used the existing project interpreter and dependency overlay below. Use the same recorded versions in `plan.json`, and choose a new run directory when repeating fits. The study uses local DFC24 data; Alan owns Anvil execution.

```bash
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
  env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/tmp/quaffle-hq-dfc24-direction-20261028:/tmp/quaffle-wave4a-deps \
  MPLCONFIGDIR=/tmp/quaffle-hq-mpl \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/bernalde/.gemini/antigravity/brain/1c8d1d6c-6148-446b-a21c-0b9c538c6603/scratch/quaffle-venv/bin/python \
  scripts/run_hq_dfc24_direction.py run \
  --data-root /tmp/quaffle-dfc24-real \
  --work-dir /tmp/quaffle-hq-dfc24-direction-new-reproduction \
  --device cuda --batch-size 8 --epochs 60 --fit-cap 128 \
  --image-size 128 --base-channels 8 --eval-every 2 --augment
```

The original work directory is `/tmp/quaffle-hq-dfc24-direction-runs-20261028`; its complete persistent copy and all eight checkpoint hashes are recorded in [the retention inventory](provenance/raw_run_retention.json). The new directory above preserves that original evidence. The CLI can resume only compatible completed fits and rejects an incomplete attempt rather than overwriting it.

For report regeneration from the retained checkpoints, follow [REPRODUCTION.md](REPRODUCTION.md). The collector audits saved selected replays and their count-derived metrics; it performs no independent test inference. [INTERPRETATION.md](INTERPRETATION.md) gives the result and its limits.
