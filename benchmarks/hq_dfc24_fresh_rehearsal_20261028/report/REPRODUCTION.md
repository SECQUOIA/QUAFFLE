# Regenerate the audit

Use the bound producer source and compatible Python/package/Torch build. The collector has its own source binding, separate from fitting dependencies. Original source bytes are retained under inert `sources/*.txt` paths.

```bash
/home/bernalde/.gemini/antigravity/brain/1c8d1d6c-6148-446b-a21c-0b9c538c6603/scratch/quaffle-venv/bin/python /tmp/quaffle-hq-fresh-runner-20261028/scripts/collect_hq_dfc24_fresh.py --work-dir /home/bernalde/repos/QUAFFLE/results/hq_fresh_runner_20261028/runs/rehearsal_v1 --data-root /home/bernalde/repos/QUAFFLE/results/local_quantum_followup_20261027/dataset --output-dir /tmp/quaffle-hq-fresh-regenerated
```

Use an empty output directory and the prescribed OS memory/zero-swap/two-thread bounds. All raw checkpoints must remain available locally. A byte-identical dataset may be relocated without rewriting original producer records; cohort and normalization identities must still match. CPU collection does not require the original GPU host. Raw JSON bytes, source snapshots and derived CSV/JSON are exported; no trained weights or image data are copied. `artifact_sha256.json` covers every output except itself.
