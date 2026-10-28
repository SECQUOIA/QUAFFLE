# Existing DFC24 metric and exposure audit

This audits already exposed data and saved results. It runs no training, checkpoint inference or prospective resource gate. It establishes neither superiority nor equivalence.

Twenty historical fit records comprise 12 fresh fits from that campaign and 8 reused fits, with 15 selected saved replays and 3 paired seeds. No fits or replays are new in this audit.

Raw bytes of all 3,476 kept image/mask files were checked against the conflict-drop manifest. Each seed uses 128 optimization tiles, 1,303 normalization-fit tiles and 230 validation tiles. Every cleaned training tile was used for normalization or validation; an unoptimized tile is not thereby untouched. All 205 test tiles were already evaluated.

Flood IoU is the foreground metric. Binary macro IoU averages flood and non-flood IoU equally. It is not a five-class land-cover score. Pooled metrics first sum Optical/SAR confusion counts; means and sample SD then aggregate the three seed metrics. Zero denominators yield 0. AUC is not recoverable from confusion counts. Paired differences are formed per seed and reported in percentage points; no confidence or significance claim is made.

| Control | Validation flood IoU | Validation binary macro IoU | Test flood IoU | Test binary macro IoU | Test pixel accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| quantum | 0.6023 ± 0.0493 | 0.7780 ± 0.0274 | 0.6092 ± 0.0122 | 0.7752 ± 0.0077 | 0.9460 ± 0.0029 |
| frozen_quantum | 0.5995 ± 0.0423 | 0.7767 ± 0.0233 | 0.6038 ± 0.0059 | 0.7720 ± 0.0032 | 0.9451 ± 0.0010 |
| zero_bottleneck | 0.6073 ± 0.0455 | 0.7806 ± 0.0258 | 0.6022 ± 0.0052 | 0.7708 ± 0.0030 | 0.9444 ± 0.0019 |
| compact_affine | 0.5993 ± 0.0450 | 0.7766 ± 0.0253 | 0.6052 ± 0.0099 | 0.7732 ± 0.0064 | 0.9461 ± 0.0029 |
| mlp | 0.5997 ± 0.0454 | 0.7761 ± 0.0255 | 0.5872 ± 0.0078 | 0.7604 ± 0.0056 | 0.9394 ± 0.0031 |

No scene/event-independent or genuinely untouched reserve is established. This tool does not inspect TIFF geotags or upstream metadata. A separate authoritative group map and prior-exposure assessment would be needed before a group-holdout claim; feature-cluster silos are insufficient.

The JSON binds the actual audit Git revision/source hash, original producer/source snapshots, parent artifact manifests and independently rehashed input inventory. Checkpoint tensors are not restored here; saved prediction metrics are reconciled with retained confusion counts.

Reproduce with the standard-library CLI from the recorded audit revision, using a new/empty output directory:

```bash
python scripts/audit_hq_dfc24_decision.py --data-root <retained-dataset> --evidence-root <hq-training-controls-packet> --output-dir <new-output>
```
