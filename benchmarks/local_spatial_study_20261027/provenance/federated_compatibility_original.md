# Local spatial bottleneck: federated compatibility probe

Frozen source: `f990aa7564ecb436f9818a1bea461220d6ff455f` in
`/tmp/quaffle-local-spatial-kernel-20261027`. Repository files and Git state were
not changed. The final source worktree remains clean.

The probe used the actual `training_common._run_inprocess_flower_simulation`
Flower Server, production clients and strategies. It started no Ray workers or
network service. Each configuration ran one round with two synthetic clients of
two and four 32-pixel images. Batch size 4 and one local epoch give one optimizer
update per client. Numerical-library/Torch threads were 2; CUDA was hidden. Every
heavy probe used an OS `MemoryMax=3G`, `MemorySwapMax=0` CPU scope.

This verifies local mechanics, not federated segmentation accuracy, convergence,
Anvil readiness as a whole, or a quantum advantage.

| Configuration | Production two-client round | Strict federated checkpoint evaluation | Strict state-dict restore | Total/trainable parameters |
| --- | --- | --- | --- | ---: |
| Trainable local quantum Z/X/ZZ | Passed | Passed | Passed | 503294/503294 |
| Frozen local quantum Z/X/ZZ | Passed | Passed | Passed | 503294/503278 |
| No-entanglement local quantum Z/X/ZZ | Passed | Passed | Passed | 503294/503294 |
| Local classical MLP | Passed through server validation and checkpoint write/resume read | Failed on explicit null training-control metadata | Not reached after that failure | Not claimed by the completed case record |

For all three completed quantum cases:

- Production NumPy/Flower transport carries 207 deduplicated parameter tensors.
  Both bottleneck positions retain the same module identity through transport
  and strict restore; the shared bottleneck is counted once, with 3412 parameters.
- Periodic-angle routing selects transmitted tensor 90 (`quantum_params.0`),
  while tensor 91 (`input_scales.0`) remains Euclidean. Production aggregation
  agrees with the shared weighted aggregation helper within a maximum absolute
  float32 difference of `1.1920928955078125e-07`.
- A branch-cut supplement using the production quantum strategy gives angle
  `-3.1343941688537598` versus independent analytic expectation
  `-3.134394107925865`. Input scales average to `3.3333334922790527`, versus the
  independent rational `20/6`. This exercises circular angle averaging and
  exclusion of input scales from that averaging, rather than relying only on
  tiny ordinary training updates.
- Wrong tensor counts and shapes are rejected before any model parameter changes.
  The fingerprint retains the new local kernel/readout/channel settings and
  quantum training-control setting. Completed-round/global-parameter resume,
  validation-selected checkpoint loading and strict shared state loading pass.
- Frozen flags remain false through local training, transport and reconstruction.
  Frozen tensors change by exactly zero during the local optimizer updates.
  Ordinary float32 server reduction introduces a maximum frozen-tensor difference
  of `1.4901161193847656e-08`; the report preserves that numerical difference.

## Defect found in the classical federated checkpoint reader

The MLP production round completes, including both updates, server aggregation,
server validation, checkpoint write and `load_fed_checkpoint` compatibility read.
`load_checkpoint_model` then fails with `ValueError: Unknown q_training_control`.

The saved ablation fingerprint intentionally contains `q_training_control: null`.
The checkpoint reader merges that fingerprint over the supplied configuration,
overwriting its explicit `trainable` value. The new local factory forwards the
null value to `LocalSpatialBottleneckBlock`, which accepts only the three named
controls. Existing classical bottlenecks did not consume that quantum-only flag.

The coordinator was notified to address this in the separate metadata fix and
verify the real saved MLP checkpoint. No source workaround was applied in this
probe. A fix must preserve rejection of invalid non-null control values; the
classical local factory can normalize the irrelevant null value to `trainable`.

## Coverage limits and retained evidence

Local convolution and sine/cosine controls use the same generic
`ClassicalAblationUNetSegmentation`, `model.parameters()` transport and
`_TorchAblationFedAvg` path as the MLP. Their transport contracts were inspected
statically; they were not trained in this probe. The existing federated
fingerprint records the new local settings, but does not record
`q_share_blocks`; this probe fixes sharing to its default and does not certify an
unshared-block resume. No compression, FedProx, multi-round learning or cluster
execution is claimed.

The separately identified simulator-info standardized-key issue is excluded;
the full command-line main functions were not invoked. This is implementation
validation, not an Astra review round.

Artifacts:

- `/tmp/quaffle-local-spatial-fed-probe.py`: final reproducible probe source.
- `/tmp/quaffle-local-spatial-federated-compatibility.json`: source/package
  hashes, three complete quantum cases and the actual MLP failure traceback.
- `/tmp/quaffle-local-spatial-federated-compatibility.log`: complete final
  attempt log. Session 79046 exited 1 because of the genuine MLP restore defect.
- `/tmp/quaffle-local-spatial-federated-probe-runs-attempt3/`: all four actual
  production round checkpoint/config outputs and completed restore artifacts.
- `/tmp/quaffle-local-spatial-fed-probe-attempt1.py`, corresponding
  `...compatibility-attempt1.log/.json`: preserved first harness-debug attempt,
  which incorrectly re-deserialized a helper's already-decoded parameter list.
- `/tmp/quaffle-local-spatial-fed-probe-attempt2.py`, corresponding
  `...compatibility-attempt2.log/.json`: preserved second harness-debug attempt,
  where NumPy2 weak-scalar comparison rounded the independent expectation to
  float32. Explicit float64 comparison corrected the comparison domain without
  widening its `2e-7` absolute tolerance.
- `/tmp/quaffle-fed-branchcut-debug.py/.log`: isolated production aggregation
  evidence establishing that the second issue was in the probe comparison.

The final full attempt used scope
`run-rc0920145872c4b9aa0cddf333978f188.scope`, invocation
`0a241a5e92d943a59cdfd6cf615ae347`. All worker processes are terminal and
the worker is quiescent. Remaining action is coordinator-owned classical metadata
repair and verification; do not classify every requested restore path as passing.
