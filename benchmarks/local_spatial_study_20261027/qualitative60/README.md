# Fixed-rule 60-epoch qualitative comparison

The original renderer labels the roles as follows:

| Figure label | Actual selected model | Learning rate |
| --- | --- | ---: |
| Classical U-Net | Original ordinary U-Net, width7 | 0.001 |
| Classical ablation | Local sine/cosine control (`local_sincos`) | 0.001 |
| QUAFFLE | Local trainable Z-readout circuit (`local_quantum_z`) | 0.003 |

All nine checkpoints use seeds42/77/123,64-pixel inputs,60epochs,128optimizationtiles and matching per-seed global-fit normalization/validation cohorts. The quantum readout and sine/cosine model were selected using the paired validation means. The convolution control has a higher descriptive mean test IoU than this quantum model in the60-epoch pilot, despite lower validation IoU; these panels do not establish superiority over every classical control. See the full seven-model results.

For each seed, the original ordinary U-Net selects the lower-median and worst tile among combined test images with at least5%flood; ties use dataset index. Selection is across the combined eligible set, not separately within each sensor. Quantum scores do not select the examples. Green marks TP boundaries, magenta FP and cyan FN; ground truth is shown separately. The scatter uses all205testtiles(71Optical,134SAR), colored by sensor, with empty-union tile IoU defined as zero. Pooled foreground IoU in the result tables is not the arithmetic mean of these per-tile points.

`qualitative_metric_parity.json` independently verifies exact optical/SAR TP/FP/FN/TN, sample counts and ordered test names against the saved records for all nine models. It retains current transformed-mask/prediction hashes and raw-mask identities. These count checks do not establish historical per-pixel,logit,loss or AUC identity. The original tool source and main/render/evaluation functions remain unchanged; the retained wrapper captures the original returned arrays.

Both original PNGs and both actual PDF raster exports were inspected. These are coarse64-pixel descriptive examples from an already-exposed test split.
