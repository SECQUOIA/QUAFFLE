# Independent numeric oracle preparation

The actual campaign audit has **not run**. The helper refuses admission until
`pipeline_status.json` says `stage: complete` and the campaign declares all 13
fits complete with the frozen plan SHA256. It imports no project helpers,
Torch, models, image libraries or NumPy. Before admission it reads only the two
small status JSON files; it never opens partial fit or export records.

Run the low-memory synthetic checks:

```bash
python independent_numeric_audit.py --selfcheck
```

`selfcheck_final.log` retains the final-source result. The hand-check matrix has truth totals
`[6,6,4,4,5]`, prediction totals `[7,5,4,3,6]` and class IoUs
`[5/8,4/7,3/5,2/5,5/6]`. Its all-class mIoU is `509/840`, secondary foreground
mIoU `101/168`, and overall accuracy `19/25`. The checks also establish sample SD
with denominator `n-1`, an exact 1 pp screen pass, exact zero-reference equality,
a one-pixel perturbation below threshold, failure with one tied seed despite an
adequate mean, lower-rate selection on an exact tie, rejection of invalid counts,
and refusal of incomplete mocked pipelines before any fit/plan/export access.

Only after root authorizes the completed-pipeline audit, run under the agreed
6 GiB/no-swap/two-thread CPU scope with the existing project interpreter:

```bash
python independent_numeric_audit.py \
  --task-root /home/bernalde/repos/QUAFFLE/results/landcover_native_20261028 \
  --source-root /tmp/quaffle-landcover-native-20261028 \
  --output-dir /home/bernalde/repos/QUAFFLE/results/landcover_native_20261028/numeric_audit/completed_v1
```

The helper independently checks count-derived metrics for every saved validation
point and all 12 selected replays, paired statistics, all 10 CSV tables, summary
flags, all 13 checkpoint byte hashes, source/plan/data metadata bindings, replay
bindings and the complete 17-file inventories of both exports. Exact rational
count fractions decide the 1 pp screen; float tolerance applies only to serialized
reporting values. Epoch-zero, late-best and calibration/follow-up observations are
kept separate. Checkpoint restoration and forward replay are performed by the
separate retained production pipeline; this helper never loads a checkpoint or
opens dataset/test images. Actual `audit.json` and `REPORT.md` are written only
after successful completed-pipeline admission and audit.
