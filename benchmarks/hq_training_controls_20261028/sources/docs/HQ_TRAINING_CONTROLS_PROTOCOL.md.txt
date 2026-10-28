# Local HQ training controls on cleaned DFC24

This finite follow-up asks whether the deepest quantum path affects learning,
despite its weak inference knockout response in the preceding HQ direction
screen. It uses our existing cleaned binary DFC24 data. It is exploratory,
because the dataset and baseline outcomes have already been inspected; it is
not an independent confirmation or a reproduction of the motivating paper.

## Paired models

Keep the existing 16-qubit HQ-style U-Net, including its ordinary encoder to
decoder skips. Preserve its tanh-bounded input radians without pi scaling,
32 X/Z outputs, full dense reconstruction, depthwise BN/ReLU convolutions and
initial non-core state. Compare five controls:

1. `quantum`: the existing learned eight-coefficient circuit.
2. `frozen_quantum`: the exact same initial circuit coefficients, frozen for the
   entire fit. Gradients must still pass through its input to the encoder.
3. `zero_bottleneck`: the complete reconstructed bottleneck output is zero from
   initialization through training and evaluation. All parameters in that unused
   path are frozen. Ordinary U-Net skips remain available.
4. `compact_affine`: a learned 48-to-32 affine map followed by tanh, with 1,568
   coefficients. Its maximum matrix rank is 32; no narrow hidden layer is imposed
   to mimic the circuit's parameter count.
5. `mlp`: the existing strong 48-to-64-to-32 MLP, with 5,216 core coefficients.

The learned/frozen circuit comparison changes core trainability alone. The zero
control tests the contribution of the entire deepest training path. It does not
identify a skip-connection mechanism, and a gain over a frozen circuit would not
by itself establish an entanglement benefit or a general quantum advantage.
Classical controls share the complete input/output interface and all initial
non-core parameters and BatchNorm buffers. Report stored, trainable and inactive
bottleneck counts separately; the zero model retains unused stored parameters.

## Fixed-rate primary comparison and auxiliary rate check

The primary comparison is **all five controls at the same rate 0.003**, on paired
seeds 42, 77 and 123. This rate was selected by both earlier baseline arms before
this follow-up. Do not retune it after inspecting the new controls. Use 60 epochs,
actual batch eight, 128-pixel inputs, base width eight, Adam, Dice plus
cross-entropy, validation every two epochs and deterministic training-only D4.
Use the same 128 optimization tiles and sample/augmentation order per seed.
Normalization uses all 1,303 global fit tiles, excluding 230 validation tiles.
This is not a pure 128-shot learning study.

Retain an auxiliary rate 0.001 fit on seed 42 for every control. Report the
fixed-seed rate sensitivity, without selecting a new primary rate or adding
replications. Thus the complete grid is 20 fits: 15 primary fits and five
auxiliary fits. This is a fixed-rate control experiment, not a separately
optimized performance leaderboard. The best rate for a new control may differ.

Reuse the preceding eight quantum/MLP fits only after checking the exact
executable source bytes, runtime versions, input identities, cohorts,
normalization, initial states, settings, histories and checkpoint hashes. Preserve
their original producer revision and raw records; do not count them as fresh
replications. Train the remaining 12 fits from scratch. Strictly restore every
checkpoint, including auxiliary fits, with BatchNorm state and training-control
semantics. Save full epoch histories and gradient summaries, and verify frozen
parameters remain byte-identical to initialization.

Bind the historical audit implementation as an executable dependency and reject
reuse when the current producing runtime, device, GPU, thread or allocator
identity differs. Resource profiles retain the same complete source binding.

## Evaluation and interpretation

Choose each checkpoint by the first strict maximum validation foreground IoU,
including epoch zero. Complete and audit every grid fit before any new test
replay. Then replay all 15 primary checkpoints on all 205 existing test tiles
(71 Optical and 134 SAR), retaining test names and per-sensor confusion counts.
These tests are descriptive because the test set is already exposed. Keep the
previous six baseline test records as historical evidence; do not conflate them
with the new replays. No new inference knockouts are needed: the new zero arm is
a training intervention, distinct from the prior post-training ablation.
Save each replay with bindings to its frozen plan, raw fit/history/checkpoint
digests and ordered test cohort. Resume validates and reuses a completed replay
without rerunning or overwriting it. Preserve failed/partial replay evidence and
reject an incompatible replay rather than silently replacing measured bytes.

Compute paired differences before mean and sample standard deviation. Report
every primary quantum-versus-control contrast on validation and descriptive
test, plus sensor results, seed wins and all auxiliary outcomes. Examine the
learned-versus-frozen and learned-versus-zero contrasts before attributing gains
to learned circuit coefficients. A small or inconsistent validation difference
does not justify scaling an accuracy-superiority claim. A consistent difference
would motivate an independent confirmation, while still requiring competitive
classical controls. Three seeds establish neither significance nor equivalence.
Angle probes use the fixed first eight unaugmented optimization images per seed.
For the permanently zero arm, these probes are explicitly inactive,
counterfactual values from the unused path, rather than angles used in prediction.

Do not add fits, alter the architecture, or select a favorable rate after seeing
test scores. Negative or mixed outcomes are deliverables. Preserve source, input,
runtime, raw history and checkpoint identities and regenerate a deterministic
report with all outcomes and the original independent Astra review.

## Local execution and limits

Use the existing recorded project environment without changing dependencies.
Keep each heavy process within an OS memory limit with zero swap, two numerical
threads and the 12% CUDA allocator cap. Run the 16-qubit circuit on CPU with
per-sample checkpointing and the surrounding CNN on CUDA. Profile the actual new
controls at batch eight before fitting. These probes establish feasibility, not
controlled throughput. Alan owns Anvil; this work launches no cluster jobs and
makes no full-size, federated-accuracy, hardware or fresh-holdout claim.
