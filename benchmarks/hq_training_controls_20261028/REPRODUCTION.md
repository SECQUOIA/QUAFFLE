# Regenerate this report

Use the source checkout matching every producer hash in `plan.json` and the collector hash in `provenance.json`. Both producer source families, protocols and collectors are archived byte-for-byte under inert `sources/*.txt` paths. A later Git revision is acceptable only when those executable bytes still match.

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/quaffle-hq-controls-collector-mpl PYTHONPATH=/tmp/quaffle-hq-training-controls-20261028:/tmp/quaffle-wave4a-deps /home/bernalde/.gemini/antigravity/brain/1c8d1d6c-6148-446b-a21c-0b9c538c6603/scratch/quaffle-venv/bin/python /tmp/quaffle-hq-training-controls-20261028/scripts/collect_hq_training_controls.py --work-dir /home/bernalde/repos/QUAFFLE/results/hq_training_controls_20261028/runs --data-root /home/bernalde/repos/QUAFFLE/results/local_quantum_followup_20261027/dataset --out-dir /tmp/quaffle-hq-training-controls-regenerated
```

Use an empty output directory. The retained original legacy root is required as bound in the plan; all twelve fresh and eight reused checkpoint files must remain available locally. Image data may be relocated only if the complete input identity matches. No trained weights or images are included in this report. Original fit settings retain their historical data-root paths.

The collector audits all twenty checkpoints and fifteen saved primary replays without independently repeating test inference. Six older replays are preserved under `historical_legacy` and are not fresh replications. `artifact_sha256.json` covers every generator output except itself. Both PNG and PDF figures are generated; PDF creation/modification dates are omitted for deterministic bytes. Subsequent interpretations, actual reviewer reports and external execution logs are separate evidence.
