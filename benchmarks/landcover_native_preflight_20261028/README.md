# Native LandCover.ai preflight

This package records the implementation and pre-training checks for the
[frozen native protocol](../../docs/LANDCOVER_NATIVE_PROTOCOL.md). Scientific
training results belong in the subsequent results package; this preflight does
not claim to reproduce the HQ-UNet paper's test score.

The native model follows the diagram's 64/128/256/512/1024 widths, fourth
convolutional down block, five-class head and full 16-qubit brickwork circuit.
It contains 8,154,283 stored parameters; replacing the eight circuit coefficients
with a 1,568-parameter affine core gives 8,155,843 parameters. This close whole-model
comparison leaves all classical interfaces and initial tensors paired. The
frozen circuit retains derivatives into its input; the zero arm removes the
reconstructed deep path while retaining ordinary skips.

`dataset_summary.json` binds the official v1 archive, splitter, split lists and
the prepared manifest. Preparation produced 7,470 train and 1,602 validation tiles
as 18,144 lossless RGB/mask PNGs. The 1,602 test IDs were inspected structurally;
zero test pixel windows were requested and zero test tiles were written. All 41
source parents appear in each official split, so this is not a geographic holdout.
Raw imagery, full local metadata and weights stay outside Git.

The independent PennyLane reference checks the full 16-qubit outputs and both
angle and coefficient derivatives. The entire historical simulator source stays
byte-for-byte unchanged; native brickwork scheduling lives in a separate module.
Additional checks cover loss/pooling equivalence, integer five-class metrics,
lossless source-window alignment, test exclusion, strict checkpoint restoration,
finite replay deadlines, sticky interrupted inference and deterministic exports
without model imports. Retained logs state their actual test cohorts.

Original Astra preflight reports are retained under `reviews/`, separate from
the coordinator's implementation and verification records. The initial proposal
was refined into 13 fits: two affine seed-42 learning-rate candidates followed by
four controls at a common affine-calibrated rate across seeds 42, 77 and 123.
Every fit has 60 epochs on the same 256 training tiles; validation pools all 16
crops of each of 64 fixed tiles. The calibration seed is reported separately
from follow-up seeds. No outcome-based extra fits or test evaluation are planned.

The full native CUDA profile uses a 6 GiB no-swap OS scope, two numerical threads
and a 12% GPU allocator cap. Batch eight exceeded that allocator cap; batch four
fits, preserving all model parameters and training examples. Deterministic
equivalent CE/pooling paths avoid unsupported CUDA kernels. Runtime versions are
recorded as observed; the shared installation and dependency manifests were not
modified. Hosted code CI remains intentionally absent; local validation does not
claim a hosted CI pass.

The accepted `profile.json` records one warmup plus two timed optimizer steps
per arm at batch four. Peak allocated CUDA memory is 683,477,504 bytes
(651.81 MiB). The complete-study estimate is 34,272.84 seconds, including
selected replays and a 30-minute IO/collection allowance; its 2x margin fits
the fixed 24-hour deadline. This is a preliminary cost estimate, not a runtime
guarantee. All four real gradient/movement checks and initialization pairing pass.

Focused verification: 74 isolated/native/historical model tests (including the
previously failing historical source audit), 28 data tests, and 40 final
runner/collector tests pass. The complete suite is being rechecked against
this final source; its terminal evidence will be retained before merge.
