#!/usr/bin/env python3
"""
Build the headline pooled Optical+SAR comparison table data.

Five rows on the pooled Optical+SAR setting:
  1. Fed U-Net (random)              [baseline for rows 2-3]
  2. Fed Ablation (random)
  3. QUAFFLE (random)
  4. Fed U-Net (Dirichlet alpha=0.5) [baseline for row 5]
  5. QUAFFLE (Dirichlet alpha=0.5)

For each row:
  - mean +/- std of Test_IoU and Test_Dice across the row's seed pool
  - n (number of seeds contributing)
For non-baseline rows:
  - paired ΔIoU and ΔDice against the same-partition Fed U-Net
  - 95% paired-difference CI using Student's t on n-1 dof
  - paired t-test and Wilcoxon signed-rank p-values

Every row uses the single pre-declared seed list in config/seeds.py. Seeds
without a completed run (no Test block) are reported as missing with n,
never silently replaced by a smaller pool.

Source of truth: each run's results/summary.txt, read via
analyze_results.load_from_disk (runs without a Test block are skipped and
logged). Output is a JSON file headline_pooled_table2_data.json in
--out-dir (default: $QUAFFLE_ANALYSIS_OUT_DIR, else <repo>/results).

Usage: python scripts/build_table2_data.py [--out-dir DIR]
"""
import argparse
import os
import sys
import json
from collections import defaultdict
from statistics import mean, stdev

import pandas as pd
from scipy.stats import t as student_t

# Make analyze_results.py importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

import analyze_results as ar  # noqa: E402

from config.seeds import SEEDS  # noqa: E402

OUT_NAME = "headline_pooled_table2_data.json"
OUT_PATH = os.path.join(os.environ.get("QUAFFLE_ANALYSIS_OUT_DIR", os.path.join(REPO_ROOT, "results")),
                        OUT_NAME)
COMMON_SEEDS = set(SEEDS)


def t_ci_half(values):
    n = len(values)
    if n <= 1:
        return None, None
    m = mean(values)
    s = stdev(values)
    tc = float(student_t.ppf(0.975, df=n - 1))
    ci = tc * s / (n ** 0.5)
    return ci, tc


def agg_point(values):
    """Return mean, std, n for the row's metric values."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None}
    m = mean(values)
    s = stdev(values) if n > 1 else 0.0
    return {"n": n, "mean": m, "std": s}


def paired_diff(per_seed_method, per_seed_baseline):
    """Return list of paired (seed, delta) tuples on shared seeds."""
    shared = sorted(set(per_seed_method.keys()) & set(per_seed_baseline.keys()))
    return [(s, per_seed_method[s] - per_seed_baseline[s]) for s in shared]


def agg_paired(deltas):
    """Mean, std, n, 95% t CI, paired t-test and Wilcoxon signed-rank."""
    return ar.paired_stats(deltas)


def main(argv=None):
    global OUT_PATH
    ap = argparse.ArgumentParser(description="Headline pooled Optical+SAR table data.")
    ap.add_argument("--out-dir", default=os.path.dirname(OUT_PATH),
                    help="Output folder (default %(default)s)")
    OUT_PATH = os.path.join(os.path.abspath(ap.parse_args(argv).out_dir), OUT_NAME)
    # Read every run's summary.txt (runs without a Test block are skipped
    # and logged by load_from_disk).
    df = ar.load_from_disk()
    df["Dataset"] = df["Dataset"].fillna("unknown")

    # Restrict to the pooled Optical+SAR setting only.
    pool = df[df["Dataset"] == "combined"].copy()

    # Convert metric columns
    for col in ("Test_IoU", "Test_Dice"):
        pool[col] = pd.to_numeric(pool[col], errors="coerce")
    pool["Seed"] = pd.to_numeric(pool["Seed"], errors="coerce").astype("Int64")

    # One run directory per (Method, Seed), dedupe defensively.
    pool = pool.drop_duplicates(["Method", "Seed"], keep="last")

    # Per-row, per-seed maps
    # All rows use the pre-declared seed list (config/seeds.py).
    METHODS_FOR_TABLE = {
        "fed_unet_random":     ("Fed U-Net (random)",            "Classical Fed (Random)",    True),
        "fed_ablation_random": ("Fed Ablation (random)",         "Ablation Fed (Random)",     True),
        "quaffle_random":      ("QUAFFLE (random)",              "Quantum Fed (Random)",      True),
        "fed_unet_dirichlet":  ("Fed U-Net (Dirichlet a=0.5)",   "Classical Fed (Dirichlet)", True),
        "quaffle_dirichlet":   ("QUAFFLE (Dirichlet a=0.5)",     "Quantum Fed (Dirichlet)",   True),
    }

    per_seed_iou = {}
    per_seed_dice = {}
    for key, (display, internal, is_main) in METHODS_FOR_TABLE.items():
        sub = pool[pool["Method"] == internal]
        sub = sub[sub["Seed"].isin(COMMON_SEEDS)]
        sub = sub.dropna(subset=["Test_IoU", "Test_Dice", "Seed"])
        per_seed_iou[key] = {int(s): float(v) for s, v in zip(sub["Seed"], sub["Test_IoU"])}
        per_seed_dice[key] = {int(s): float(v) for s, v in zip(sub["Seed"], sub["Test_Dice"])}

    # Build output rows
    rows = []
    pairings = {
        "fed_unet_random":     None,                    # baseline row
        "fed_ablation_random": "fed_unet_random",
        "quaffle_random":      "fed_unet_random",
        "fed_unet_dirichlet":  None,                    # baseline row
        "quaffle_dirichlet":   "fed_unet_dirichlet",
    }

    for key, (display, internal, _) in METHODS_FOR_TABLE.items():
        iou_vals = list(per_seed_iou[key].values())
        dice_vals = list(per_seed_dice[key].values())
        row = {
            "key": key,
            "display": display,
            "internal_method": internal,
            "is_baseline": pairings[key] is None,
            "seeds": sorted(per_seed_iou[key].keys()),
            "missing_seeds": sorted(COMMON_SEEDS - set(per_seed_iou[key].keys())),
            "iou":  agg_point(iou_vals),
            "dice": agg_point(dice_vals),
        }
        baseline_key = pairings[key]
        if baseline_key is not None:
            di_pairs = paired_diff(per_seed_iou[key], per_seed_iou[baseline_key])
            dd_pairs = paired_diff(per_seed_dice[key], per_seed_dice[baseline_key])
            row["baseline_key"] = baseline_key
            row["paired_seeds"] = [s for s, _ in di_pairs]
            row["delta_iou"]  = agg_paired([d for _, d in di_pairs])
            row["delta_dice"] = agg_paired([d for _, d in dd_pairs])
        rows.append(row)

    # Pretty-print summary
    print("=== Headline table (pooled Optical+SAR, t-distribution CIs) ===")
    print(f"{'Row':<32}{'n':>3}  {'IoU':>16}  {'Dice':>16}  {'ΔIoU [95% CI]':>32}  {'ΔDice [95% CI]':>32}")
    for r in rows:
        n = r["iou"]["n"]
        iou_str = (f"{r['iou']['mean']:.3f} ± {r['iou']['std']:.3f}"
                   if r["iou"]["mean"] is not None else "—")
        dice_str = (f"{r['dice']['mean']:.3f} ± {r['dice']['std']:.3f}"
                    if r["dice"]["mean"] is not None else "—")
        if r["is_baseline"]:
            di_str = "(baseline)"
            dd_str = "(baseline)"
        else:
            di = r["delta_iou"]
            dd = r["delta_dice"]
            if di["n"] == 0:
                di_str = dd_str = "(no paired seeds)"
            else:
                di_str = f"{di['mean']:+.3f} [{di['mean']-di['ci95_half']:+.3f}, {di['mean']+di['ci95_half']:+.3f}] n={di['n']}"
                dd_str = f"{dd['mean']:+.3f} [{dd['mean']-dd['ci95_half']:+.3f}, {dd['mean']+dd['ci95_half']:+.3f}] n={dd['n']}"
        print(f"  {r['display']:<30}{n:>3}  {iou_str:>16}  {dice_str:>16}  {di_str:>32}  {dd_str:>32}")
        if not r["is_baseline"] and r["delta_iou"]["n"] > 1:
            fmt = lambda v: "NA" if v is None else f"{v:.4f}"
            print(f"      ΔIoU  paired t p={fmt(r['delta_iou']['t_p'])}  Wilcoxon p={fmt(r['delta_iou']['wilcoxon_p'])}")
            print(f"      ΔDice paired t p={fmt(r['delta_dice']['t_p'])}  Wilcoxon p={fmt(r['delta_dice']['wilcoxon_p'])}")
        if r["missing_seeds"]:
            print(f"      WARNING: {r['display']} missing seeds {r['missing_seeds']} "
                  f"(n={n}/{len(COMMON_SEEDS)})")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
