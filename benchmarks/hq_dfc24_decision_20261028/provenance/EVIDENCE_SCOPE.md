# Evidence and reproduction scope

The seven top-level files covered by `artifact_sha256.json` were generated from
the actual retained DFC24 data and committed historical HQ control packet using
`scripts/audit_hq_dfc24_decision.py` at source revision
`3249d63611e2b4a51fd07af2a85d391bdbaebc43`. Two actual runs into separate persistent
directories produced byte-identical files, without normalization or exclusions.
`independent_decision_audit.json` records a second, standard-library-only rational
count oracle checking all 90 primary rows, 30 summary rows, 120 paired contrasts
and 60 all-fit validation rows. Its retained source uses no project imports or
model inference. The public CLI requires access to the original retained input
bytes; publishing aggregate evidence does not distribute the imagery or weights.

The audit freshly checks the 3,476 kept file MD5s, manifest/source/profile/raw-JSON
bindings and saved count-derived metrics. It does not deserialize checkpoints,
repeat predictions, verify AUC from counts, or execute the future study.
Historical checkpoint restoration and inference remain the separately recorded
producer/collector evidence. The original 20 fit records include eight historical
reuses and do not represent 20 independent replications.

The files in this directory and `reviews/` are retained supplements, covered
separately by `supplemental_sha256.json`; the public audit CLI does not regenerate
them. Original reports/logs/helper bytes are preserved. Coordinator summaries and
dispositions are labeled as such.

`retained_archive_audit.json` and `.md` document a bounded read-only inspection of
the retained local archive, including all 1,937 retained combined TIFF headers.
`retained_tiff_header_scanner.py.txt` preserves the executed parser and replay
wrapper. `TIFF_HEADER_SCAN_SCOPE.md` limits the absence claim to those retained
copies. This initial metadata inspection reconstructed the manifest inventory
without rehashing all image/mask bytes; the later public audit performs that full
kept-byte recheck. The two statements describe distinct checks. Positive controls
verified that the scanner detects hand-constructed geolocation and text tags in
both byte orders. This is not a complete TIFF implementation conformance test or
a statement about georeferencing in upstream DFC24 sources.

The manuscript directory and its zip are intentionally Git-ignored. Corrections,
before-copies, full build logs, actual PDFs and the deterministic zip remain in
the user's local workspace. `local_manuscript_summary.json` identifies their
hashes and verification limits; it does not make the manuscript part of the PR
tree or provide enough inputs to rebuild it publicly. The visual inspection
record concerns four actual rendered pages carrying the new section/table, not
every historical page. Compilation does not validate historical scientific claims.

Repository tests are software verification. Synthetic fit mechanics within that
suite are not new real-data scientific fits. No Anvil job, independent holdout,
hardware run or new real-data training/inference was performed in this task.

For the prospective design and exact uncompleted launch gates, read
[the decision protocol](../../../docs/HQ_DFC24_DECISION_PROTOCOL.md). The required
fresh full-fit runner and Anvil resource profile remain future work owned through
Alan's execution handoff; the present PR supplies neither a ready-to-submit
campaign command nor measured cluster feasibility.
