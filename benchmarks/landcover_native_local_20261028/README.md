# Native LandCover.ai local comparison

The frozen study is running; numerical results have not yet been published.
This initial record retains the plan and independent cohort audit before any
quantum comparison outcomes. The implementation and preflight were merged in
[PR #66](https://github.com/SECQUOIA/QUAFFLE/pull/66), merge
`604cef0e164bf1ba2be301b82107ddbb0fb39a90`.

The immutable plan SHA256 is
`a3cae54c1f575fc46722513c8d274daf96ef4b3fc4635c4a3e051c7f6e1b30fc`.
It fixes thirteen fits, four controls, three paired seeds, sixty epochs, native
five-class metrics and zero test predictions. Complete validation replay and
Astra review of the empirical results are still required.

The independent audit reconstructs all selected IDs without project helpers,
checks all 640 prepared PNG hashes and decoded values/counts, finds zero exact
RGB or RGB-plus-mask duplicates, and compares sixteen sampled original TIFF
windows exactly. All thirty-five selected validation parent rasters also occur
in training; this is not a geographic holdout.

`operations/` holds an inert copy of the task-local execution record and its
original Astra assessment. It waits for the live training unit, requires the
complete thirteen-fit grid, then calls the merged collector for twelve selected
validation replays, two deterministic exports and completed-path integrity checks.
It launches no fitting, changes no training condition and performs no GitHub writes.
