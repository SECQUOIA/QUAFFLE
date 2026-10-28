# Evidence layers

The generator owns the 63 files listed by `artifact_sha256.json` (the manifest excludes itself). Its original bytes remain unchanged during curation. Publication supplements are covered separately by `supplemental_sha256.json`.

The producer is commit `6803241ec9772a99012382fc8ef23be27abf014a`. Later commits add the collector and improve report wording/layout without changing any of the ten fitting dependencies or the frozen protocol. The final collector revision and exact tested sources are recorded in the final test and regeneration bindings.

`result_numeric_audit.json` and its inert source are the original independent stdlib audit of the initial and v2 exports. `release_regeneration.json` carries that evidence forward by checking byte identity of all six CSVs and nine numerical/raw JSON files against the final package. The original audit is not relabeled as a fresh inference run or a formal Astra review. Its historical report inputs are retained under the persistent archive's `report_drafts/` directory.

Histories contain epoch training-loss means and per-epoch gradient summaries. Aggregate counters record all 960 optimizer batches per fit; individual batch loss/gradient time series were not retained. Angle probes cover the first eight unaugmented optimization images (384 encoded values), rather than the entire training or validation set.

Input-integrity hashes cover both training and test file bytes. The test-evaluation boundary concerns model selection and prediction metrics: all eight fits and strict checkpoint audits completed before the six selected test replays. This test set had already been exposed in earlier work and remains descriptive.

Raw weights are retained locally in `runs/quaffle-hq-dfc24-direction-runs-20261028` under the archive identified by `raw_run_retention.json`; data remain in `results/local_quantum_followup_20261027/dataset`. TIFF retention uses persistent hardlinks and assumes immutable inputs. The artifact packet contains weight hashes and data identities rather than trained weights or image tiles. Temporary absolute paths in original evidence are preserved as historical provenance.
