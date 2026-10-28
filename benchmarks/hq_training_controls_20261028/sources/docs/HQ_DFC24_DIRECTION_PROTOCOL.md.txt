# HQ-style bottleneck direction test on DFC24

This bounded study follows the request to test the HQ-UNet direction using our
own cleaned Optical/SAR flood data. It is an architecture-transfer pilot, not a
reproduction of the five-class LandCover.ai score reported by Hossain et al.,
*HQ-UNet*, [arXiv:2604.27206v1](https://arxiv.org/html/2604.27206v1).
The original paper does not fully specify its training protocol; choices below
belong to this study and must not be attributed to that paper.

## Two arms and common architecture

Use a small plain depthwise-separable U-Net, with BatchNorm and ReLU, base width
eight, three input bands and two output classes. Four downsamplings take a
128-by-128 input to an eight-by-eight bottleneck. Replace the deepest classical
DoubleConv with the following adapter:

1. Pool its 64 input channels to a four-by-four grid, project to three channels
   and apply `tanh`. Encode the resulting 16 three-component vectors directly
   as RX/RY/RZ angles in radians, without a pi multiplier or learned input scales.
2. Either evaluate one layer of the existing 16-qubit HQ circuit, with shared
   horizontal/vertical RY-CNOT-RY filters and X/Z readout, or apply a classical
   MLP to the identical 48 encoded values. The MLP uses a 64-wide SiLU hidden
   layer and 32 tanh outputs, without a forced narrow hidden rank.
3. Apply an identical dense 32-to-(128*8*8) reconstruction and reshape to the
   bottleneck map before the ordinary decoder.

There is no bottleneck identity addition, affine modulation or attention block.
Ordinary encoder-to-decoder U-Net skips remain. Share the exact initial encoder,
decoder, preprojection, dense reconstruction and BatchNorm state for each seed;
isolate core initialization from the common random stream. Report core and total
parameter counts separately. The MLP deliberately has more core coefficients;
this is a strong classical transformation control, not exact core-count parity.

Reuse the tested statevector kernel rather than create another simulator. CPU
sample chunking/checkpointing may bound its memory while preserving the complete
model batch and BatchNorm behavior. Keep the circuit at 16 qubits; a four-qubit
substitute would answer a different question.

## Data, selection and finite fit grid

Use `flood_combined`, the committed conflict-policy `drop` manifests and the
existing raw DFC24 tile/mask bytes. Preserve the by-seed stratified split:
1,303 global fit images and 230 validation images. Normalization uses the global
fit partition and excludes validation. Cap optimization to the same deterministic
128-image subset for each paired seed; this is not a pure few-image study.

Use 128-by-128 inputs, binary Dice plus cross-entropy, Adam and deterministic
training-only D4 augmentation. Keep data order and augmentation identical across
paired arms, with independent data RNGs. Validation/test images are unaugmented.

Each arm receives learning rates 0.001 and 0.003 on seed 42. Select its rate by
validation foreground IoU, with ties preferring the lower rate. Then replicate
that rate on seeds 77 and 123: four tuning fits plus four replication fits,
eight fresh fits in total. Retain losing tuning outcomes as well as all six
selected results. Select the first strict validation maximum, including epoch
zero, and retain full histories.

Both models receive **60 epochs, actual batch size eight, validation every two
epochs, base width eight and 128-pixel inputs**. This budget was fixed after the
random-input probe at source `e9e4c3dbfaa03cddc47c39ea82a464b10af63716`, before any
real-data fit or validation/test result in this campaign. Both arms passed at
batches one and eight, including two forward/backward/Adam steps, finite states,
core movement and strict checkpoint restoration. At batch eight the process
reached about 2.52 GiB peak host RSS; maximum PyTorch allocated GPU memory was
146.27 MiB, excluding CUDA context/display memory. The measured quantum batch
steps took 2.36 and 1.64 seconds, including diagnostic instrumentation; these are
resource-planning observations, not controlled throughput benchmarks.
The machine-readable fitting plan must retain this final configuration.

No model's test evaluation begins until every planned fit is finished. Use all
205 existing test images (71 Optical and 134 SAR), retain their ordered names,
pooled foreground metrics and per-sensor confusion counts. This test split has
already informed earlier research decisions: its results remain descriptive.
No new architecture, learning rate or extra fit is selected using test scores.

## Diagnostics and decision

Before fitting, check actual forward/backward/Adam behavior, finite states and
gradients, core movement, shared initialization and strict checkpoint round trips.
Record training group gradients, selected-minus-initial movement and input-angle
distributions. Preserve BatchNorm buffers in checkpoint identities and audits.

On the selected checkpoints evaluate `on`, `zero_core` (zero the 32 features
before reconstruction, retaining its bias), `random_core` (replace only core
weights deterministically) and `zero_bottleneck` (zero the reconstructed map).
Apply the same interventions to both arms. A knockout drop shows dependence on
that information path; it does not establish quantum superiority.

Report all per-seed scores and paired quantum-minus-MLP differences, sample
standard deviations and wins. Different selected learning rates remain part of
the comparison. Validation is the screening signal. A consistent positive
comparison would justify broader controls and independent confirmation; a small
or unstable contrast provides no clear support under this budget. Nonconverged
curves remain inconclusive. Three seeds and one classical core cannot establish
quantum advantage, equivalence or the impossibility of useful quantum models.

The grid ends after all eight fits and their evaluations regardless of outcome.
This protocol does not expand after inspecting the test results. A plain U-Net,
frozen/separable variants and a full LandCover.ai reproduction are outside this
small direction screen.

## Reproduction and completion

Use this workstation only; Alan owns Anvil execution. Bound every heavy process
with an OS memory limit and zero swap allowance, two numerical-library threads,
and the existing 12% CUDA allocator cap when using the GPU. Profile the 16-qubit
model itself rather than extrapolate from the earlier four-qubit campaign.

Freeze exact source/input/runtime/settings identities before fitting. Resume
only matching runs and retain unsuccessful attempts unchanged. Audit all eight
checkpoints, including losing tuning candidates; retain raw curves, diagnostics,
plans and checkpoint/data hashes. Keep trained weights and image data in the
ignored persistent local archive, with reviewable summaries and provenance in
the repository. Verify regenerated scientific artifacts and inspect all exports.

Complete independent Astra review under `gh-issue-to-human-review`, address
actionable findings within its cumulative bounds, then publish and merge the
new PR under `gh-merge-pr` only after refreshing live repository gates. Report
actual local/hosted coverage and leave existing broader Anvil trackers open.
