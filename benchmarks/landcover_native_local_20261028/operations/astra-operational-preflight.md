# Astra operational preflight

Read-only task-local wrapper inspection, not a formal PR review round. Inspected `operational_pipeline.py`, SHA256 `ccc102b91f5dff123d1f0521cb4ad82840140aaabb940aede26e739255dc8981`. No execution, source change or live-campaign intervention was performed.

**Cleared for the described single-coordinator launch. No material operational error found.**

The wrapper waits through active/activating/deactivating unit states and does not equate unit termination with scientific success: it separately requires the frozen plan identity, complete13-fit status and all-fits digest. Unknown unit-state responses fail after bounded retries. Source hashes are checked before dispatch, and the merged collector supplies the complete-grid/checkpoint/data audit before new replay inference.

Replay is limited to the smaller of one hour and remaining persisted campaign headroom minus60seconds; insufficient headroom fails closed. CUDA replay starts only after the named training unit stops. Both exports consume the same completed replay and require complete file-byte equality. Completed-path checks replace fitting/evaluation with failure tripwires and compare campaign/replay bytes and mtimes. Calling the completed training path initializes Torch threading once; the completed replay branch returns through read-only validation without initializing it again.

Interpret the final `within_campaign_deadline` field literally: model-free exports and completed-path bookkeeping occur after the explicit replay-deadline check, so the wrapper may truthfully record `false` if that bookkeeping crosses the campaign deadline. It does not run another fit or inference to recover from failure. Root should retain the wrapper, its hash, logs and terminal evidence, and review actual scientific results separately.

This clearance assumes the stated single wrapper instance and externally enforced6GiB/no-swap/two-thread scope. It does not change the frozen scientific source, plan, rate selection or formal review budget.
