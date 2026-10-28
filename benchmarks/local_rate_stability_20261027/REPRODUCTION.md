# Replaying the local rate study

The committed package retains all outcomes, curves, validation selections,
producer manifests and checkpoint identities. Trained checkpoint bytes are
retained locally, outside Git, under
`results/local_quantum_followup_20261027/runs/` in the primary checkout.
The local retention inventory records original paths and copied file hashes.
Absolute paths inside historical evidence are preserved, not rewritten.

## Use the measured source snapshot

The new matched-rate fits were produced at
`a707b00937faf82ce6c4a72d13c337568bf1fe93`; the fresh 120-epoch fits were produced
at `84464a095834b32e21614dd93f1560887fcf3c54`. The latter adds the collector and
reporting without changing the executable files bound by the earlier plan.
Historical fits retain their own producing revisions and original manifests.

Use a separate worktree at `7259a75cc7e2b6ac267f1240e2560801d8487562` to collect
these runs. This collection snapshot clarifies the report wording about spatial
operations; all nine bound fitting dependencies remain identical. A later maintained checkout can legitimately fail the strict source
inventory after model integration; this is not a reason to bypass that guard.
Source revisions identify the producing programs, not a guarantee of bitwise
CUDA retraining. The retained prefix audit reports actual numerical differences.

The measured environment is Python 3.11.14, PyTorch 2.5.1 with CUDA 12.1,
PennyLane 0.44.1, NumPy 2.4.6 and Rasterio 1.4.3. Figure-library identity is
recorded in `audit.json`. Circuit simulation runs on the CPU; the U-Net runs on
the NVIDIA RTX A1000. The study uses two numerical-library threads and a 12%
per-process CUDA allocator cap. Timings are not controlled throughput benchmarks.

## Collect without retraining

Set `QUAFFLE_PYTHON` to the matching interpreter and the two run variables to
the preserved raw run roots. Historical source run directories and the real
dataset must also remain available at the paths recorded in those plans, or be
restored from their retained copies. The collector verifies current input bytes;
the copied summaries alone do not replace the raw prerequisites.

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$QUAFFLE_PYTHON" scripts/collect_local_rate_stability.py \
  --matched-run "$QUAFFLE_MATCHED_RUN" \
  --long-run "$QUAFFLE_LONG_RUN" \
  --out-dir "$QUAFFLE_NEW_OUTPUT"
```

The output must be new or empty. The collector checks all 30 cells, eight
historical and 22 new fits, configuration/cohort/normalization identities,
complete finite histories, checkpoint and replay hashes, per-sensor confusion
counts, paired statistics, and cross-duration initialization. It refuses stale
or incomplete evidence rather than silently dropping a run.

For reproducibility, repeat collection into another empty directory. Compare
every file listed in `artifact_sha256.json`, including all three PNG/PDF pairs.
The generated inventory covers generator outputs; the final package inventory
also covers curated interpretation and provenance added afterward. Curated text
is not reconstructed by the numerical collector.

## Repeat training

The original protocol is [LOCAL_QUANTUM_FOLLOWUP.md](../../docs/LOCAL_QUANTUM_FOLLOWUP.md).
Use the producing snapshots, the raw input manifests, matching runtime and new
run directories. `run_local_rate_stability.py --phase matched60` requires
`--baseline-run` and `--control-run` with the eight historical cells;
`--phase long120` deliberately forbids reuse and trains all 18 cells fresh.
Both phases require `--data-root`, `--work-dir` and `--device cuda`.
Keep the original six-GiB, no-swap process scope, two threads and the allocator
cap. Every planned fit must finish before that phase's test replay.

Normalization uses the complete global fit partition, excluding validation;
only optimization is capped at 128 images. Test results remain exploratory
because this test set already informed research decisions. Three paired seeds
and longer training do not provide an independent confirmation cohort.

The real dataset is also retained locally at
`results/local_quantum_followup_20261027/dataset/`. Its required input bytes, split
manifest and normalization cache match the study. TIFF images are persistent
hardlinks, and PNG masks/JSON metadata are independent copies; this preserves
files through deletion of temporary names without constituting an independent
backup against in-place image edits. See `provenance/dataset_retention.json`.
