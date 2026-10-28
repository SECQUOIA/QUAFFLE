# Coordinator integration dispositions

These checks extend the preflight evidence; they are implementation validation, not additional Astra review rounds. Original probe reports and failed attempts are preserved separately.

- Simulator reporting: the new block omitted the standardized simulator keys, causing production summaries to describe its circuit as absent. Isolated commit e3e77efb85d40cf52f765d54eb8aa288066e0ff4 corrects the metadata. Seven regressions failed before the fix; 39 metadata/kernel/entrypoint tests passed after it.
- Classical checkpoint compatibility: the actual in-process Flower probe completed the MLP round but failed to reload its checkpoint. Production fingerprints represent an irrelevant quantum-control field as null. Isolated commit e33f22c53213587b47fb75546dd00a7d267a8ef5 maps only that null default to trainable; unsupported non-null controls remain rejected. Six centralized/federated checkpoint regressions failed before the guard; all 51 focused tests passed afterward.

Both fixes are isolated from the actively running fitting checkout. The computation audits show that every fitting configuration has an explicit non-null control, and that the remaining circuit, initialization, readout, forward and gradient code is unchanged. Preserve the original fitting/collection snapshot before carrying these commits onto the publication branch. The final results review must inspect both fixes.

The first federated report contains a genuine failed classical restore and is not an all-passing claim. Its seven-configuration rerun on the fixed candidate must complete before final publication. The probe uses only synthetic data and local in-process clients, and supplies no federated accuracy or Anvil result.

The fixed candidate subsequently passed all seven actual in-process Flower round/resume cases and both strict restore paths. The exact original failed MLP checkpoint now reconstructs with identical selected parameters, without changing its bytes (SHA256 `60c493f4813f6a6b64d99b55a5ecc20ccace4df3bf9249e12e51f9db6a7e7afc`). The new original fixed-candidate report and raw structured/log evidence are retained separately from the original failure report. No federated accuracy, multi-round convergence or Anvil result is inferred.
