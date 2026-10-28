# Collection and retention attempts

The 28-fit pilot and eight-fit duration extension completed successfully before
these audits. No training was restarted in response to an observation timeout.

The first angle-audit and extension-preservation launches could not access the
systemd user bus from the filesystem sandbox. Their logs retain that pre-execution
environment failure. Authorized launches outside that sandbox retained the same
4 GiB audit / 2 GiB preservation limits and zero swap allowance.

The first executed angle audit stopped at checkpoint metadata comparison for
`pilot60/tuning/local_conv/0.001/seed42`. The native checkpoint stores `dim_mults`
as a tuple and its JSON fit record stores the same values as a list. Their
unserialized Python equality differs although the JSON representations agree.
The mismatch was diagnosed without rewriting either checkpoint or fit record.
The collector repair at `281a35ee558e503f576334797e0b6b306978e298` canonicalizes only the settings representation; genuine
configuration, model-type, checkpoint-epoch, source, runtime and weight guards
remain required. The regression reproduced before the fix and all 35 focused
collection tests passed afterward. The corrected angle audit and collector both
completed successfully, covering all 36 checkpoints, 27 selected results and
15 individual-angle probes. This is a reporting defect, not a failed training fit.

Extension preservation succeeded for all eight fitted checkpoints. A subsequent
byte audit compared every original and archived file in all six completed run
roots against the retention inventory: 90 fitted checkpoints, comprising 32
historical and 58 new fits. This count includes reused experimental evidence and
is not an independent sample size. The separate failed synthetic compatibility
checkpoint is outside that count.
