# HQ launch package and descriptive validation figures

This packet completes the local launch preparation and standalone publication figures. **No model was retrained, no new test prediction was performed, and no cluster was contacted or job submitted.**

## Launch package

The [operator guide](../../docs/HQ_ANVIL_LAUNCH.md) and [generator](../../misc/generate_hq_campaign_scripts.py) produce separate full-mode profile, run and collect SLURM jobs. Source/configuration bindings, finite budgets, manual acceptance of the passed full profile's exact SHA256, and native full-mode guards precede training. Alan supplies actual cluster settings and confirms resource feasibility.

The [local demonstration](launch_demo/README.md) retains inert copies of three actual generated scripts, manifest and guard checks. It uses invalid synthetic scheduler settings and deliberately tiny limits; these values are not Anvil recommendations. All scripts passed Bash parsing, source/config checks passed, and missing profile acceptance blocked the run. No scheduler command was executed.

## Publication figures

- [Learning curves, PDF](figures/learning_curves.pdf) · [PNG](figures/learning_curves.png)
- [Fixed-rule error overlays, PDF](figures/qualitative_errors.pdf) · [PNG](figures/qualitative_errors.png)
- [Captions](figures/captions.md), [selected examples](figures/selected_examples.csv), [displayed counts](figures/displayed_counts.csv), [all per-image counts](figures/per_image.csv), [pooled/sensor counts](figures/pooled.csv), and [prediction disagreements](figures/disagreements.csv)

Curves show four controls at the existing fixed primary rate 0.003 over 60 epochs, using seeds 42, 77 and 123. These twelve histories comprise nine fresh control fits and three reused quantum fits from the earlier audited study. Fits used 128 optimization tiles, 230 validation tiles, image size 128 and base width 8; normalization uses all 1,303 global-fit images excluding validation. They are separate from the two-epoch fresh-runner rehearsal.

The qualitative cohort is seed 42's complete ordered 230 validation tiles. Compact affine was predicted first. For each sensor, the worst and lower-median eligible affine-IoU tile was selected among masks with at least 5% flood; validation ordinal breaks ties. Selected identities were saved before predicting the other three controls. All models use those same examples. Affine-conditioned selection can emphasize reference errors; these examples cannot establish an overall ranking.

The single validation-prediction run completed in 175.13 seconds. Every model recovered its original sensor confusion counts exactly, and checkpoint/BatchNorm state remained unchanged. Actual tripwires blocked fitting and test-dataset loading. No test score or AUC was reconstructed.

The close learning curves and shared/mixed errors illustrate the existing control results. They add no independent confirmation or quantum-superiority claim. In particular, the selected SAR worst case is missed by all four models.

![Four-control learning curves](figures/learning_curves.png)

![Descriptive fixed-rule validation overlays](figures/qualitative_errors.png)

## Verification and reproduction

Two actual final render invocations produced identical complete output bytes, including PDFs, PNGs, tables and provenance. The renderer imported no model libraries, did no inference, and preserved the prediction cache's bytes and modification times. The initial layout's crowded labels were corrected using the same cache. Final PDFs are 7.2 inches wide; PNGs and actual-PDF rasterizations were directly inspected, with all extracted PDF text inside page bounds.

An [independent audit](verification/independent_audit.json) checked 920 image rows, 12 pooled/sensor rows, 1,380 disagreement rows and 732 curve rows from original integer masks and historical records. It independently recomputed example ranks, reloaded all 230 original targets and the four displayed inputs, and verified historical raw bytes/mtimes stayed unchanged.

The original prediction cache, raw numeric archives, data and native checkpoint weights remain in the persistent local results area; [published cache metadata](cache_metadata/README.md) is not a complete renderable cache. [Reproduction instructions](REPRODUCTION.md) distinguish new validation collection from inference-free rendering. Source snapshots, actual execution logs/bindings and original Astra reports are retained here. No raw TIFF, numeric image archive or model weights are published. The selected display images are included only in the requested figures.

The final repository suite passed **764 tests** (22 retained warnings). 764 passed, 22 warnings in 296.73s (0:04:56)

The inner figures/artifact_sha256.json covers the unchanged renderer output. The outer artifact_sha256.json covers this complete published packet except itself. Alan's actual Anvil profile, scheduler-policy checks, accepted resource budget and full fresh campaign remain unperformed here. Broader issue #42 remains open.
