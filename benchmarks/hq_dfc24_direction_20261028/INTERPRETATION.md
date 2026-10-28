# What the DFC24 direction screen tells us

The primary validation comparison gives a small, mixed quantum edge. The descriptive test comparison is positive across all three seeds. Whole-bottleneck knockouts show very little inference dependence, so this screen does not establish a quantum-specific accuracy improvement.

| Arm | Core parameters | Total parameters | Validation foreground IoU | Descriptive test foreground IoU |
| --- | ---: | ---: | ---: | ---: |
| 16-qubit HQ-style core | 8 | 344,480 | 0.6023 ± 0.0493 | 0.6092 ± 0.0122 |
| Classical MLP core | 5,216 | 349,688 | 0.5997 ± 0.0454 | 0.5872 ± 0.0078 |

Values are means and sample standard deviations across three paired seeds. The first seed chose each arm's learning rate from two candidates; both selected 0.003. The selected rates were then used on seeds 77 and 123. Every fit received 60 epochs and 960 optimizer batches. The complete grid contains eight fits, including the two unselected tuning candidates.

| Seed | Quantum minus MLP validation IoU, percentage points | Quantum minus MLP descriptive test IoU, percentage points |
| --- | ---: | ---: |
| 42 | +0.441 | +3.161 |
| 77 | +0.757 | +2.508 |
| 123 | −0.407 | +0.928 |
| Mean ± sample SD | +0.264 ± 0.602 | +2.199 ± 1.148 |

The validation contrast changes sign and is small relative to its observed seed variation. These seeds vary initialization, fit/validation split and optimization subset together; the paired arms share those choices. Three runs do not establish equivalence, statistical significance or a general quantum advantage. Some selected checkpoints occur near the end of the budget, including epoch 60 for both seed-77 arms, so this is not a demonstrated convergence limit.

The test split had already been exposed in earlier research. All eight new fits finished before any selected test replay, and no extra architecture or fit was chosen from these test scores. The +2.199-point test pattern is worth retaining as descriptive evidence; it provides no independent confirmation of the small validation contrast. The dataset is our cleaned binary DFC24 task, rather than the five-class LandCover.ai task in the motivating HQ-UNet paper. This study transfers architectural elements and does not reproduce that paper's reported score.

The knockout result limits a causal interpretation of the test pattern. Zeroing the entire reconstructed quantum bottleneck map changes test IoU by only +0.124, −0.018 and −0.009 percentage points for seeds 42, 77 and 123 respectively (on minus zero). The mean is +0.033 ± 0.080 points, and the largest absolute change is 0.124 points. Replacing only the quantum core coefficients with a fresh deterministic random draw gives a mean change of +0.004 ± 0.008 points. The corresponding MLP whole-bottleneck contrast is −0.028 ± 0.192 points. Ordinary encoder-to-decoder skips remain available in both arms. These interventions support weak inference reliance on the deepest path in this screen.

Training effects remain plausible: changing a core can alter gradients and the learned shared network even when the path can later be ablated. The present two-arm design does not isolate that hypothesis. All four quantum fits recorded nonzero core gradient norms in all 960 optimizer batches, and the selected circuit weights moved from initialization. Those observations establish optimization activity rather than importance to the final prediction. Histories retain epoch training losses and per-epoch gradient summaries; aggregate counters describe batch coverage, without retaining individual batch loss/gradient time series. Angle probes contain 384 values from the same first eight unaugmented optimization images per seed at initialization and the selected checkpoint. Their saturation summaries describe that fixed probe.

The parameter difference is also limited in scope. The quantum core has 8 coefficients versus 5,216 for the MLP, while the complete networks differ by 5,208 coefficients: about 1.49% of the MLP model. Both arms share the encoder, decoder, projection, dense reconstruction and initial BatchNorm state. This is a strong classical transformation control with a larger core, rather than exact parameter-count parity or a comparison with an ordinary convolutional U-Net bottleneck.

The practical conclusion is to retain the candidate and this positive descriptive test pattern, while withholding a quantum-superiority claim. Before a large accuracy-focused campaign, useful controls would isolate training-time effects, a fixed/random core and a compact classical spatial transformation, followed by validation-only and independent-data confirmation. Those controls were not run in this finite screen. Alan retains responsibility for Anvil execution; no cluster result is implied here.

The local circuit simulation ran on CPU with one-sample checkpointing, while the surrounding depthwise U-Net ran on CUDA. The 12% setting limits CUDA allocator memory; it does not cap GPU compute utilization. Recorded fit timings and the random-input resource probe include instrumentation and are not controlled throughput or quantum-hardware benchmarks.

The generated [result tables and figures](README.md), [paired differences](paired.csv), [all tuning candidates](tuning.csv), [knockouts](knockout.csv) and [reproduction instructions](REPRODUCTION.md) preserve the numerical detail. The collector strictly restores all eight checkpoints and checks the six saved selected replays against their records and confusion counts. It does not independently repeat test inference.
