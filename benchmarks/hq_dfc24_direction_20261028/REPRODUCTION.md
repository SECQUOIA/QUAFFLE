# Regenerate this report

Run from the source checkout whose producer hashes match `plan.json` and whose collector hash matches `provenance.json`. A later Git revision is allowed when every bound executable byte is unchanged. The exact original source/protocol and collector are retained as inert files under `sources/`.

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/quaffle-hq-collector-mpl PYTHONPATH=/tmp/quaffle-hq-dfc24-direction-20261028:/tmp/quaffle-wave4a-deps /home/bernalde/.gemini/antigravity/brain/1c8d1d6c-6148-446b-a21c-0b9c538c6603/scratch/quaffle-venv/bin/python /tmp/quaffle-hq-dfc24-direction-20261028/scripts/collect_hq_dfc24_direction.py --work-dir /tmp/quaffle-hq-dfc24-direction-runs-20261028 --data-root /tmp/quaffle-dfc24-real --out-dir /tmp/quaffle-hq-dfc24-direction-regenerated
```

Use a new/empty output directory. No weights or images are copied into this report. All eight checkpoint identities and persistent raw locations are in `provenance.json`; weights and data must remain available locally. Data may be relocated only when the complete bound input identity matches; original saved settings retain their historical data-root paths.

The collector reloads training cohorts/fit-only normalization and restores checkpoints on CPU. Saved test predictions are not independently rerun: record equality and count/identity consistency are the audited layer. `artifact_sha256.json` covers every generator output except itself. PDF creation/modification dates are suppressed for reproducibility. Resource profiles, external test logs and subsequent interpretation added by the publication owner are separate evidence.
