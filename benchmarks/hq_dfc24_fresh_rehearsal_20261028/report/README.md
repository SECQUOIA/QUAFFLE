# Fresh HQ campaign audit: rehearsal

**Software verification only.** All 16 fresh fits use two epochs and 16 optimization/evaluation-validation tiles. The 12 selected replays use validation images only. No test dataset was evaluated and no scientific gate is calculated: `not_applicable_rehearsal`.

Every arm receives the same two-rate opportunity (0.001 and 0.003) on seed42, then selected-rate follow-ups on seeds77/123. Exact rate ties choose the lower rate; checkpoints use the earliest strict validation maximum, including epoch zero. Seed42 participates in tuning and aggregation. The two follow-ups are displayed separately and are not independent-data confirmation. Different selected rates compare training procedures rather than circuit trainability alone.

All 16 checkpoints were restored strictly on CPU, including BatchNorm buffers, frozen parameters, trainability and permanent-zero mode. Collection audits saved records and count-derived metrics; it neither trains nor repeats inference. CPU restoration can run on another GPU host when Python/package/Torch-build identities match and exact initial state reconstruction succeeds; original GPU identity remains bound to the producer/profile.

Normalization uses all 1,303 global-fit images and excludes all 230 global validation images. The rehearsal subsets do not create a pure16-shot or untouched reserve. Models retain base8,128-pixel inputs,batch8,ordinary skips,full16-qubit computation,tanh-radian encoding without pi,32 X/Z outputs and full reconstruction. These are binary DFC24 controls, not a LandCover.ai replication or quantum-hardware/federated-accuracy result.

| Control | Selected rate | Validation foreground IoU mean ± sample SD | Validation replay foreground IoU mean ± sample SD |
| --- | ---: | ---: | ---: |
| quantum | 0.001 | 0.057191 ± 0.099058 | 0.057191 ± 0.099058 |
| frozen_quantum | 0.001 | 0.057191 ± 0.099058 | 0.057191 ± 0.099058 |
| zero_bottleneck | 0.001 | 0.057191 ± 0.099058 | 0.057191 ± 0.099058 |
| compact_affine | 0.001 | 0.057191 ± 0.099058 | 0.057191 ± 0.099058 |

Metrics are fractions. Pooled scores use summed pixel confusion counts; sensor scores use their own counts. Empty IoU/Dice denominators yield zero. Binary macro IoU is the arithmetic mean of flood and background IoU. AUC is retained only from saved evaluations and cannot be derived from confusion counts.

Paired differences are computed within seed before mean and sample SD, then expressed in percentage points. `paired.csv` retains seed42,77,123 differences and the two-follow-up mean/SD separately. Sample SD is not a confidence interval. Registered/trainable counts are not functional-activity counts: the zero arm retains parameters whose input is permanently absent. Zero-arm angle probes are inactive counterfactual encodings.

See [all candidates](candidates.csv), [selected observations with counts](per_seed.csv), [paired contrasts](paired.csv), [trainability](trainability.csv), [gate and analysis](analysis.json) and [provenance](provenance.json). Original JSON bytes and inert source snapshots are retained; no weights or images are exported.
