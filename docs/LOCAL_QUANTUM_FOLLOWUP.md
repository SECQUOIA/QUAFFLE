# Local quantum follow-up protocol

This follow-up tests the two uncertainties exposed by the real DFC24 pilot:
whether the circuit benefits from entanglement at a matched learning rate, and
whether a circuit that mixes neighboring spatial positions contributes more
than the current independently evaluated pooled positions. It also checks
whether the 60-epoch ranking changes with a longer training budget.

The protocol is written before the new fits. It is an exploratory extension of
the already inspected pilot, not a formal preregistration or an untouched-test
confirmation. All attempted configurations and outcomes must be retained.

## Rate and duration study

1. Complete the 60-epoch factorial: trainable global circuit and the same circuit
   without CNOT gates, each at learning rates 0.001 and 0.003, for paired seeds
   42, 77 and 123. Audit and reuse the eight existing fits; train the four absent
   combinations. Compare entangled minus separable at each fixed rate, rather
   than attributing independently selected-rate differences to the gates.
2. Train fresh 120-epoch fits for those two models and the patchwise full-rank
   classical MLP, at both rates and all three seeds (18 fits). Use the same
   initialization and data order for matching model/seed/rate combinations.
   Inspect validation-selected checkpoints, late selection, finite loss spikes,
   gradient extremes and the first-60 versus full-120-epoch validation trajectory.
   The patchwise MLP is the strongest observed classical pilot comparator; it is
   not a spatially matched causal control for the global circuit.
3. Report validation and descriptive test IoU, per-sensor results, paired
   differences in percentage points, sample standard deviations, wins and seed
   counts. Test scores do not select rates, architectures or extra experiments.

Both stages use 64-pixel inputs, 128 optimization tiles per seed, batch size 16,
Adam with Dice plus cross-entropy, validation every two epochs, and unchanged
encoder/decoder skips. Normalization uses the global fit partition and excludes
validation; the limited optimization pool does not make this a pure few-image
learning study. Each stage finishes every planned fit before new test replay.

## Spatial/readout pilot

The new opt-in bottleneck uses a shared four-qubit circuit over actual 2 by 2
neighboring encoder locations, initially one layer. It keeps a fixed classical
projection into and out of the feature map, with identical projections and
encoder/decoder initialization across controls for each seed. The original
architecture remains available with unchanged defaults.

The seven planned configurations are:

- trainable spatial circuit with Z readout;
- trainable spatial circuit with Z, X and neighboring-qubit ZZ readout;
- the richer-readout circuit with all angles and input scales frozen;
- the richer-readout circuit with CNOT gates removed;
- a full-rank spatial MLP;
- a spatial convolution;
- a sine/cosine-feature control.

Every configuration receives the same local input window and pre/post projection
dimensions. The Z-only circuit pads unused readout slots with zero; equal stored
parameter counts do not imply that those inactive connections provide capacity.
Classical core capacity is reported explicitly rather than compressed into a
weaker rank constraint to mimic the circuit parameter count.

Profile a forward/backward batch before launching fits. Use the same bounded
64-pixel/128-tile/60-epoch setting as the original pilot. Each configuration gets
the same two-rate tuning on seed 42, followed by replication of its
validation-selected rate on seeds 77 and 123 (28 fits). Report both fixed-seed
readout contrasts and paired selected-rate contrasts; the latter include tuning.
Record gradients, parameter movement, input-angle saturation and zero/random
output knockouts. Do not interpret nonzero group gradients as evidence that
every angle is useful.

Select the richer or Z-only trainable circuit by mean paired validation IoU
(ties prefer Z), and the best of the three trainable classical spatial controls
by mean validation IoU (ties prefer the smaller core). Extend both selected
models to fresh 120-epoch runs with the same two-rate, three-seed tuning and
replication policy (8 fits). This extension is planned regardless of which model
wins the initial pilot, so a disappointing result is not a stopping criterion.
These 120-epoch fits are a duration check, not independent statistical replication.

## Resource and reporting bounds

Run on this workstation only, using classical quantum simulation. Alan owns
Anvil execution. Keep training within a host-memory scope with swap disabled,
two numerical-library threads, and the existing 12% CUDA allocator cap. Test
processes use separate bounded CPU scopes; do not oversubscribe available memory.
If the new circuit exceeds the batch budget, chunk circuit execution while
preserving the mathematical batch and optimizer behavior; record the setting.

Retain exact producing source revisions, executable/input/runtime hashes, raw
curves and checkpoint identities. Reused historical fits retain their original
provenance. Publish reviewable code and measured results through new PRs with
independent Astra review, address findings within the workflow's bounds, and
merge only after live gates pass. State the observed quantum benefits and
failures equally. A claim about data efficiency, robustness, Anvil accuracy or
physical quantum hardware requires additional evidence beyond this protocol.

## Recorded rate-study outcomes

The protocol above was committed before the fits at
`e621dded38bfd96c654a9ea41c5a804f074d843b`; these result links were added afterward.
Read the [interpretation](../benchmarks/local_rate_stability_20261027/INTERPRETATION.md),
[complete generated results](../benchmarks/local_rate_stability_20261027/README.md)
and [reproduction instructions](../benchmarks/local_rate_stability_20261027/REPRODUCTION.md).
The original protocol bytes are retained with the result provenance.
