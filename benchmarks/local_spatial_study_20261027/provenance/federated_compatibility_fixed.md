# Local spatial federated compatibility after the metadata fix

**All seven configurations passed** actual in-process Flower rounds and both
strict checkpoint reconstruction paths at candidate
`e33f22c53213587b47fb75546dd00a7d267a8ef5` in
`/tmp/quaffle-local-spatial-metadata-20261027`.

This is a mechanics check on six synthetic 32-pixel images, not evidence about
federated segmentation accuracy or quantum advantage. Each configuration used two
clients with two and four examples, batch size 4 and one local epoch: one
optimizer update per client, weighted production server aggregation, client and
server evaluation, and the production completed-round checkpoint path. The
existing in-process Flower Server ran sequentially without Ray or network
services. CPU execution used two threads and an OS 3G/no-swap scope; CUDA was
hidden.

| Configuration | Production round and resume | Strict federated/state restores | Total/trainable parameters | Transport tensors | Shared bottleneck parameters |
| --- | --- | --- | ---: | ---: | ---: |
| Local quantum Z | Passed | Passed | 503294/503294 | 207 | 3412 |
| Local quantum Z/X/ZZ | Passed | Passed | 503294/503294 | 207 | 3412 |
| Frozen local quantum Z/X/ZZ | Passed | Passed | 503294/503278 | 207 | 3412 |
| Separable local quantum Z/X/ZZ | Passed | Passed | 503294/503294 | 207 | 3412 |
| Local MLP | Passed | Passed | 503562/503562 | 209 | 3680 |
| Local convolution | Passed | Passed | 503338/503338 | 207 | 3456 |
| Local sine/cosine features | Passed | Passed | 503438/503438 | 208 | 3556 |

The probe verified the following with the actual production conversion,
client/strategy, aggregation and checkpoint helpers:

- The two bottleneck positions retain one shared module through parameter
  transport and strict state reconstruction. Parameter counts deduplicate that
  shared module. Wrong tensor counts and shapes are rejected atomically.
- Quantum angle tensor 90 is routed to circular aggregation where requested;
  input-scale tensor 91 remains Euclidean. The branch-cut supplement gives angle
  `-3.1343941688537598`, compared with independent analytic expectation
  `-3.134394107925865`; input scales give `3.3333334922790527`, compared with
  `20/6`. Reference comparisons retain the original tolerances and use explicit
  float64 arithmetic for the independent expectations.
- Actual production aggregation agrees with the shared weighted aggregation
  helper within a maximum absolute float32 difference of
  `1.1920928955078125e-07` in every configuration.
- Frozen flags survive local training and reconstruction. Frozen parameters
  change by exactly zero during each local optimizer update. Ordinary float32
  server reduction introduces a maximum difference of
  `1.4901161193847656e-08`; that roundoff is retained rather than labeled exact.
- New local kernel/readout/channel settings and quantum training controls appear
  in federated fingerprints. The completed round, global parameters and
  validation-selected checkpoint survive resume and strict reconstruction.
- The production simulator formatter now reports the quantum path as
  `PennyLane default.qubit, backprop, torch interface, broadcast over batch of local 2x2 windows, on cpu (q_backend=pennylane)`.
  Each classical control is reported as `none (quantum family: classical)`.

## The original failed MLP checkpoint now restores unchanged

The exact checkpoint that failed at frozen source
`f990aa7564ecb436f9818a1bea461220d6ff455f` was loaded with the fixed candidate:

`/tmp/quaffle-local-spatial-federated-probe-runs-attempt3/local_mlp/fed_resume.pt`

Its SHA256 before and after reconstruction is
`60c493f4813f6a6b64d99b55a5ecc20ccace4df3bf9249e12e51f9db6a7e7afc`.
The checkpoint still contains the original null quantum-only control field. The
classical model now reconstructs with trainable parameters and exactly matching
validation-selected parameter values. The old checkpoint, failure JSON, log and
original user-facing report were not overwritten.

## Evidence and limits

- Probe source: `/tmp/quaffle-local-spatial-fed-probe-fixed.py`.
- Structured original output:
  `/tmp/quaffle-local-spatial-federated-compatibility-fixed.json`.
- Full execution log:
  `/tmp/quaffle-local-spatial-federated-compatibility-fixed.log`.
- Actual checkpoints/configurations:
  `/tmp/quaffle-local-spatial-federated-probe-runs-fixed1/`.
- Original failed-source evidence remains in
  `/tmp/quaffle-local-spatial-federated-compatibility.md` and its corresponding
  original JSON/log/attempt files.

Session 73485 exited 0. Scope:
`run-r169486bcbffc4d1c858acb2571b4b0de.scope`, invocation
`75dffa24d04f4ffeb6f1f6dd51b26153`. Both the fixed-candidate and frozen-producer
worktrees remain clean; source hashes and pinned HEAD were verified unchanged by
the successful probe. No repository edits, Git writes, GitHub writes, environment
installation, GPU use or Anvil access were performed.

This does not cover unshared blocks, compression, FedProx, multi-round convergence
or cluster execution. The pre-existing federated fingerprint omission of
`q_share_blocks` remains outside this fixed-sharing probe. These results validate
implementation mechanics and do not replace the planned independent Astra review
of the complete research PR.
