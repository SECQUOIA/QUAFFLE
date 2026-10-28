# Replaying the local spatial study

The package retains every tuning and replication outcome, complete histories,
checkpoint identities, diagnostics, selection rules and source/input manifests.
Trained checkpoint bytes are retained locally, outside Git, under
`results/local_quantum_followup_20261027/runs/` in the primary checkout. Historical
absolute paths are preserved in evidence; restore those paths or supply
equivalent run roots before replay.

## Preserve the producing program

The 60-epoch spatial fits were launched at
`211c2852f470a7b006704f591652939089d4a41f`; the 120-epoch extension was launched
at `057e38e582b39e91190a7ab0f9fc19ed46fa1e7c`. Use collection snapshot
`281a35ee558e503f576334797e0b6b306978e298` for the collector and angle audit.
All ten fitting dependencies retain the producing hashes. This snapshot fixes
the collector's tuple-versus-JSON-list settings comparison without changing
models, measurements or source/runtime guards; the failing attempt and RED/GREEN
regression evidence are retained in provenance. The original collector version
at `aaa5496054c616eb662757291d65aee582624f0c` remains in the branch history.

Use this pre-integration source snapshot to replay the historical study. The
later simulator metadata and null classical-control repairs deliberately change
two source inventories and are tested separately. Their computation audits
explain why the recorded explicit-control fits remain valid. Do not weaken the
source guard to collect old measurements from a changed maintained checkout.

The measured environment is Python 3.11.14, PyTorch 2.5.1 with CUDA 12.1,
PennyLane 0.44.1, NumPy 2.4.6 and Rasterio 1.4.3. The NVIDIA RTX A1000 executes
the U-Net; PennyLane simulates the four-qubit core on the CPU. Numerical-library
threads are limited to two and the CUDA allocator to 12% per process. The
two-step resource profile is not a controlled speed benchmark.

## Audit and collect completed runs

Set `QUAFFLE_PYTHON`, `QUAFFLE_PILOT_RUN`, `QUAFFLE_EXTENSION_RUN`,
`QUAFFLE_DATA_ROOT`, `QUAFFLE_ANGLE_OUTPUT` and `QUAFFLE_NEW_OUTPUT` to explicit
paths. Run each command in a bounded CPU scope, with GPU visibility disabled,
two numerical-library threads, four GiB resident memory and no swap. Keep
concurrent memory caps within the machine's available memory.

```bash
"$QUAFFLE_PYTHON" scripts/audit_local_spatial_angles.py \
  --pilot-dir "$QUAFFLE_PILOT_RUN" \
  --extension-dir "$QUAFFLE_EXTENSION_RUN" \
  --data-root "$QUAFFLE_DATA_ROOT" \
  --output "$QUAFFLE_ANGLE_OUTPUT"

"$QUAFFLE_PYTHON" scripts/collect_local_spatial_study.py \
  --pilot-dir "$QUAFFLE_PILOT_RUN" \
  --extension-dir "$QUAFFLE_EXTENSION_RUN" \
  --angle-audit "$QUAFFLE_ANGLE_OUTPUT" \
  --out-dir "$QUAFFLE_NEW_OUTPUT"
```

The collector requires a new or empty output directory. It verifies all 28
pilot and eight extension fits, including losing learning-rate trials, all
saved states and histories, source/runtime/input identity, shared initialization,
selected rates and finalists, paired cohorts, normalization and sensor counts.
The angle audit uses the first eight fixed optimization tiles per seed and the
15 selected quantum checkpoints. Its single-probe derivatives are not an
accumulated training gradient or proof that an angle improves segmentation.

Regenerate into a second empty directory. Compare the full file inventory and
scientific JSON/CSV values; inspect every figure. PDF timestamps may differ:
report the equality layer actually checked rather than asserting whole-file
identity. `artifact_sha256.json` covers generator outputs, while the final
package inventory includes added interpretation, qualitative views and original
review/provenance evidence.

The retained regeneration audit checks all 29 generated files. Twenty-five
non-PDF outputs are byte-identical; the three PDFs agree byte-for-byte after
normalizing only each PDF Info object's `CreationDate` value. The two artifact
manifests differ only in those verified PDF digests. This is a named metadata
normalization, not whole-PDF byte identity. All three original PNGs and all three
actual PDF exports were also visually inspected. The verifier's unchanged source
and detailed equality records are retained in `provenance/`.

## Repeat training and qualitative evaluation

`"$QUAFFLE_PYTHON" scripts/run_local_spatial_study.py pilot --data-root ... --work-dir ... --device cuda`
runs the seven configurations with two rates on seed 42 and the selected rate
on seeds 77/123. The planned extension is
`"$QUAFFLE_PYTHON" scripts/run_local_spatial_study.py extend --baseline-run ... --data-root ... --work-dir ... --device cuda`.
It selects a trainable quantum readout and classical core using mean paired
validation IoU, then trains eight fresh 120-epoch fits. Use new run roots and
the original six-GiB/no-swap, two-thread scope. No test score selects finalists.

Qualitative views use `misc/build_qualitative_error_figure.py` and a retained
three-seed checkpoint manifest: the original 60-epoch ordinary U-Net is the
reference, alongside the validation-selected spatial classical and quantum
models at the same 60-epoch budget. Median and worst cases are selected by
reference-model tile IoU with at least 5% flood; ties use dataset index.
The scatter includes every test tile. These examples remain descriptive.

The fits use 64-pixel images, base width eight, one GroupNorm group and 128
optimization tiles. Normalization uses all global fit images and excludes
validation. The [Alan handoff](../../docs/LOCAL_SPATIAL_CONFIRMATION.md) changes
model width, normalization groups and resolution. Its separate CPU inference,
GPU batch-one optimizer-step and synthetic federated mechanics checks establish
their recorded execution scopes. The GPU audit measures batch-one allocator
memory; it does not establish full-batch memory, convergence or training accuracy.

The real dataset is also retained locally at
`results/local_quantum_followup_20261027/dataset/`. Its required input bytes, split
manifest and normalization cache match the study. TIFF images are persistent
hardlinks, and PNG masks/JSON metadata are independent copies; this preserves
files through deletion of temporary names without constituting an independent
backup against in-place image edits. See `provenance/dataset_retention.json`.
