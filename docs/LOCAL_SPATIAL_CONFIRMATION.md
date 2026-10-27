# Full-size spatial confirmation handoff for Alan

The local study develops a four-qubit filter over neighboring encoder positions and compares it with equally tuned spatial classical controls. Read the [local results](../benchmarks/local_spatial_study_20261027/README.md) and their limitations before choosing the Anvil shortlist. Anvil execution belongs to Alan; this repository change submits no jobs.

## Verified locally

The [configuration audit](../benchmarks/local_spatial_study_20261027/provenance/full_model_config_audit.json) parsed all seven production commands, constructed their full-size models, verified identical backbone/projection initialization, and ran one finite random-tensor CPU inference for each. Each output had shape `[1, 2, 128, 128]`. This is a configuration and geometry check, with no data, optimizer update, accuracy result, Anvil execution or full-batch GPU-memory measurement.

Full-size settings are base width 32, 256 active bottleneck channels, GroupNorm with four groups, 128-pixel inputs, four projected spatial feature maps, a shared one-layer four-qubit core, no time embedding and no decoder skip gate. The local fits use base width 8, GroupNorm with one group and 64-pixel inputs; the full-size setting therefore changes more than image resolution.

| Core | Total parameters | Trainable parameters | Core parameters |
| --- | ---: | ---: | ---: |
| local_quantum_z | 6,438,326 | 6,438,326 | 16 |
| local_quantum_xz_zz | 6,438,326 | 6,438,326 | 16 |
| local_quantum_frozen | 6,438,326 | 6,438,310 | 16 |
| local_quantum_no_entanglement | 6,438,326 | 6,438,326 | 16 |
| local_mlp | 6,438,594 | 6,438,594 | 284 |
| local_conv | 6,438,370 | 6,438,370 | 60 |
| local_sincos | 6,438,470 | 6,438,470 | 160 |

The circuit contributes 16 core parameters. Almost all stored parameters remain classical, and the spatial classical controls provide almost the same compression. Z-only also has inactive projection columns; equal stored counts do not mean equal active capacity.

## Production entrypoints

The audit JSON retains every parsed command as an argument array. These two examples use the richer quantum readout and the full-rank spatial MLP. Replace `DATA_ROOT` and the results directory for the actual job. The shown learning rate is an example, not a full-size tuning result.

```bash
python centralized_demoTorchQ.py --dataset combined --seed 42 --base-channels 32 --image-size 128 --batch-size 8 --quantum_channels 256 --q-ansatz reupload --q-layers 1 --n-qubits 4 --q-backend pennylane --q-sim-device cpu --local-spatial-channels 4 --decoder-skip-gate none --skip-modulation affine --no-time-emb --loss dice_ce --epochs 120 --learning-rate 0.001 --eval-every 2 --no-save-samples --data-root DATA_ROOT --results-dir results_spatial_confirmation --local-spatial-kernel quantum --local-spatial-readout xz_zz --q-training-control trainable
```

```bash
python ablation_studies.py --framework torch --mode centralized --quantum-backend pennylane --ablation-mlp-mode full_rank --dataset combined --seed 42 --base-channels 32 --image-size 128 --batch-size 8 --quantum_channels 256 --q-ansatz reupload --q-layers 1 --n-qubits 4 --q-backend pennylane --q-sim-device cpu --local-spatial-channels 4 --decoder-skip-gate none --skip-modulation affine --no-time-emb --loss dice_ce --epochs 120 --learning-rate 0.001 --eval-every 2 --no-save-samples --data-root DATA_ROOT --results-dir results_spatial_confirmation --local-spatial-kernel mlp --local-spatial-readout xz_zz --q-training-control trainable
```

The other audited commands cover Z-only, frozen angles/scales, omitted CNOTs, convolution and sine/cosine controls. Use the new local-kernel flags through the production entrypoints. The existing quantum-channel grid generator does not emit these local options or calibrated local-filter time limits.

## Campaign order and scientific gates

1. Run one full-size forward/backward and short training smoke for each shortlisted quantum/classical configuration on the assigned Anvil node. Measure peak GPU and host memory, CPU circuit cost and wall time before budgeting the larger grid. The existing global/channel-group runtime calibration is not a measured estimate for this local filter.
2. Tune all compared models using the same validation budget at full size. Preserve projections, skips, data/normalization, augmentations, optimizer and training budget across the relevant controls. Keep frozen and separable controls when investigating trainability or entanglement; a selected-rate contrast is not a fixed-rate gate experiment.
3. Replicate paired seeds and retain every outcome. Full-data, higher-resolution, ten-seed, separate-sensor and federated random/Dirichlet comparisons belong to this larger campaign. The local centralized results do not establish their rankings.
4. Choose an independent confirmation cohort before using it to make final superiority claims. The current DFC24 test split has already informed research decisions and remains descriptive. Exclude confirmation/validation images from normalization fitting and record the exact new manifests if a new cohort is introduced.
5. Examine per-sensor accuracy, convergence, inference knockouts and input-angle saturation alongside pooled IoU. A gain over a frozen circuit does not by itself demonstrate an entanglement benefit or superiority to the classical alternatives.

Simulation remains noiseless and classical, using PennyLane on the CPU for circuit evaluation. Exact X/Z/ZZ readouts from a statevector do not measure physical-device sampling cost or noise robustness. A faster GPU or a longer run alone does not imply a quantum accuracy improvement.
