# Independent selected LandCover.ai cohort audit

The frozen 256 train / 64 validation cohort was reconstructed directly from official ZIP split lists and the specified SHA256 rankings (subset seeds 742/943). No project selection, preparation or data-loader helper was imported. The cohort was neither filtered nor resampled.

| Split | Tiles | Parent rasters | Class 0 pixels | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 256 | 41 | 38141961 | 517781 | 23089583 | 4322673 | 1036866 |
| val | 64 | 35 | 10066429 | 136361 | 5731888 | 624000 | 218538 |

All five class IDs are present in each cohort. Train and validation tile IDs are disjoint; 35 source parents occur in both selected cohorts. This is not a scene-disjoint geographic holdout. Model seeds 42/77/123 use these same IDs.

Every selected PNG was hashed and decoded, and its mask counts matched the saved manifest. Exact duplicate checks use complete 512x512 uint8 arrays, not compressed PNG bytes:

- RGB-only cross-split duplicate groups: 0; train/validation duplicate pairs: 0.
- RGB-plus-mask cross-split duplicate groups: 0; train/validation duplicate pairs: 0.

All 16 deterministic source-window checks (8 selected train + 8 validation) matched original TIFF RGB, integer mask values and class counts exactly. Coordinates were independently reconstructed by enumerating row-major 512-step x/y positions and preserving skipped partial-edge indices. Stored PNG origins and preparation helpers were not used as the oracle.

No test pixel windows were requested. Test IDs were used only for structural split separation. Archive byte hashing does not evaluate test pixels. No model, fitting, predictions or accuracy calculation was run.

These exact whole-tile checks do not test near duplicates, adjacent/overlapping scene context, or geographic independence. Source fidelity sampled 16 of 320 selected TIFF windows; all 320 saved PNG pairs were decoded and hash/count checked. Detailed IDs, duplicate groups, file/source hashes, sampling rule and runtime are retained in `cohort_audit.json`.

Dataset manifest SHA256: `da33f9a649513e37add550b05086ee56b4dca5a9973fd79d13ba3caf40f049d1`.
Audit source SHA256: `c66799f2e87324885c35dfcd669bd193a33be45b7e2730388503ee63b64c79c9`.
