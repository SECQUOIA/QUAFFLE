#!/usr/bin/env python3
"""
Forest plot of paired ΔIoU (quantum bottleneck − classical baseline) across
9 (regime × dataset) cells.

For each cell:
  - centralized      → quantum_label = "Quantum Central",        classical_label = "Classical Central"
  - federated random → quantum_label = "Quantum Fed (Random)",   classical_label = "Classical Fed (Random)"
  - federated dirich → quantum_label = "Quantum Fed (Dirichlet)", classical_label = "Classical Fed (Dirichlet)"

Pairing is by seed across the pre-declared seed list in config/seeds.py.
Cells missing seeds are reported with n and the missing seed ids.
For each common seed s present in BOTH method's results for that cell, compute
    Δ_s = IoU_quantum,s − IoU_classical,s
Then aggregate Δ_mean, Δ_std, n, a paired t-test, a Wilcoxon signed-rank
test and a 95% paired-difference CI using Student's t:
    CI95_half = t_{0.975, n-1} · Δ_std / √n
(Replaces the prior normal approximation 1.96·Δ_std/√n; at n=4 the t critical
value 3.182 is markedly wider than 1.96 and matches the small-sample expectation.)

Outputs:
  - PNG: <out dir>/iou_delta_forest.png (QUAFFLE_FOREST_OUT_DIR, default <repo>/figures/forest)
  - JSON: same dir, iou_delta_forest_data.json (per-cell stats + per-seed Δ)

The JSON is also the data source for the headline mini-table.
"""
import os
import sys
import json
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import t as student_t

REPO_ROOT = os.environ.get("QUAFFLE_REPO_ROOT",
                           os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
OUT_DIR   = os.environ.get("QUAFFLE_FOREST_OUT_DIR",
                           os.path.join(REPO_ROOT, "figures", "forest"))
sys.path.insert(0, REPO_ROOT)
from config.seeds import SEEDS  # noqa: E402
COMMON_SEEDS = set(SEEDS)
_OUT_SUFFIX = ""
DATASETS = ["optical", "sar", "combined"]
DATASET_LABELS = {"optical": "Optical", "sar": "SAR", "combined": "Pooled Opt.+SAR"}

# (Demo, Training, Variant, Partition, Quantum_Channels) → role label
METHOD_MAP = {
    ("Centralized Classical Torch", "centralized", "classical", "centralized", None): ("Classical Central", "main"),
    ("Centralized Quantum Torch",   "centralized", "quantum",   "centralized", 32):   ("Quantum Central",   "main"),
    ("Federated Classical Torch",   "federated",   "classical", "random",      None): ("Classical Fed (Random)",   "main"),
    ("Federated Classical Torch",   "federated",   "classical", "dirichlet",   None): ("Classical Fed (Dirichlet)","main"),
    ("Federated Quantum Torch",     "federated",   "quantum",   "random",      32):   ("Quantum Fed (Random)",     "main"),
    ("Federated Quantum Torch",     "federated",   "quantum",   "dirichlet",   32):   ("Quantum Fed (Dirichlet)",  "main"),
}

# Cells to plot: (regime_key, regime_label, classical_method, quantum_method, color)
# Color palette echoes the PQC circuit figure: sky blue (gate boxes),
# rose pink (CNOT targets), slate gray (neutral third). Markers use these
# slightly-darkened-from-pastel versions to remain visible on white.
CELLS = [
    # Centralized — slate gray (neutral, no quantum/classical association)
    ("centralized", "Centralized",        "Classical Central",         "Quantum Central",         "#5A6B7C"),
    # Federated random — sky blue (matches circuit gate boxes)
    ("fed_random",  "Federated (Random)", "Classical Fed (Random)",    "Quantum Fed (Random)",    "#5A9AB8"),
    # Federated Dirichlet — rose pink (matches circuit CNOT targets)
    ("fed_dirich",  "Federated (Dirichlet)", "Classical Fed (Dirichlet)", "Quantum Fed (Dirichlet)", "#D4729E"),
]


def load_per_seed_iou():
    """Returns {(method_label, dataset): {seed: iou}} for main methods, restricted
    to the 10-seed common pool. Reads each run's results/summary.txt directly via
    analyze_results.load_from_disk — bypasses the Excel logging which has historical
    column-shift bugs for some seeds (notably 7, 42, 123, 256 in the optical sheet)."""
    sys.path.insert(0, REPO_ROOT)
    from analyze_results import load_from_disk
    df = load_from_disk()
    main_methods = {label for (label, kind) in METHOD_MAP.values() if kind == "main"}
    out = defaultdict(dict)
    for _, row in df.iterrows():
        m = row["Method"]
        if m not in main_methods:
            continue
        s = int(row["Seed"])
        if s not in COMMON_SEEDS:
            continue
        ds = row["Dataset"]
        if ds not in DATASETS:
            continue
        iou = row.get("Test_IoU")
        if iou is None or (isinstance(iou, float) and iou != iou):  # NaN check
            continue
        out[(m, ds)][s] = float(iou)
    return dict(out)


def paired_delta(per_seed, quantum_method, classical_method, dataset):
    """Compute paired Δ_s = IoU_quantum,s − IoU_classical,s for shared seeds."""
    q = per_seed.get((quantum_method, dataset), {})
    c = per_seed.get((classical_method, dataset), {})
    shared = sorted(set(q.keys()) & set(c.keys()))
    deltas = [q[s] - c[s] for s in shared]
    return shared, deltas


def aggregate(deltas):
    """Mean, std, n, t CI, paired t-test and Wilcoxon signed-rank p-values."""
    from analyze_results import paired_stats
    return paired_stats(deltas)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading per-seed IoU from {REPO_ROOT}/results/")
    per_seed = load_per_seed_iou()
    print(f"  Loaded {len(per_seed)} (method, dataset) combos")

    # Compute per-cell statistics (9 cells = 3 regimes × 3 datasets)
    cell_stats = []  # list of dicts in plot order (top to bottom)
    for regime_key, regime_label, classical_method, quantum_method, color in CELLS:
        for ds in DATASETS:
            shared, deltas = paired_delta(per_seed, quantum_method, classical_method, ds)
            stats = aggregate(deltas)
            cell = {
                "regime_key": regime_key,
                "regime_label": regime_label,
                "dataset": ds,
                "dataset_label": DATASET_LABELS[ds],
                "quantum_method": quantum_method,
                "classical_method": classical_method,
                "shared_seeds": shared,
                "missing_seeds": sorted(COMMON_SEEDS - set(shared)),
                "deltas": deltas,
                "color": color,
                **stats,
            }
            cell_stats.append(cell)

    # Save JSON with per-cell stats and per-seed Δ
    json_path = os.path.join(OUT_DIR, f"iou_delta_forest_data{_OUT_SUFFIX}.json")
    with open(json_path, "w") as f:
        json.dump(cell_stats, f, indent=2, default=str)
    print(f"Wrote {json_path}")

    # Pretty-print for the headline mini-table
    print(f"\n=== ΔIoU per cell (paired, pre-declared seeds {sorted(COMMON_SEEDS)}) ===")
    print(f"{'Regime':<24}{'Dataset':<10}{'n':>3}  {'Δmean':>8}  {'Δstd':>8}  {'CI95 half':>10}  "
          f"{'t p':>7}  {'Wilcoxon p':>10}  CI overlaps 0?")
    any_missing = False
    for c in cell_stats:
        if c["n"] == 0:
            print(f"  {c['regime_label']:<22}{c['dataset_label']:<18}  (no shared seeds)")
            any_missing = True
            continue
        m, ci = c["mean"], c["ci95_half"]
        overlaps = "YES" if (m - ci) <= 0 <= (m + ci) else "NO"
        tp = "NA" if c["t_p"] is None else f"{c['t_p']:.4f}"
        wp = "NA" if c["wilcoxon_p"] is None else f"{c['wilcoxon_p']:.4f}"
        print(f"  {c['regime_label']:<22}{c['dataset_label']:<18}{c['n']:>3}  {m:>+8.3f}  {c['std']:>8.3f}  "
              f"{ci:>10.3f}  {tp:>7}  {wp:>10}  {overlaps}")
        if c["missing_seeds"]:
            any_missing = True
            print(f"      WARNING: missing seeds {c['missing_seeds']}")
    if any_missing:
        print("!" * 70)
        print("WARNING: some cells use fewer than all pre-declared seeds, see above.")
        print("!" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # Minimalist forest plot — "Regime — Dataset" labels on the left,
    # markers + CI bars in the centre, single n column on the right.
    # Per-regime colours echo the PQC circuit figure: slate gray for
    # Centralized, sky blue for Federated (Random) — same hue as the
    # circuit's RX/RY/measurement gate boxes — and rose pink for Federated
    # (Dirichlet) — same hue as the circuit's CNOT targets. Marker fill
    # encodes significance: filled if the 95% paired-difference CI excludes
    # zero, hollow otherwise.
    # ─────────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Computer Modern Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset":  "cm",
        "axes.linewidth":    0.6,
        "xtick.direction":   "out",
        "xtick.major.size":  3.0,
        "xtick.major.width": 0.5,
        "ytick.major.size":  0.0,
    })

    fig, ax = plt.subplots(figsize=(7.0, 5.4))

    # Y geometry — gap=0.9 between regime groups so the three blocks read as distinct.
    GROUP_GAP = 0.9
    y_row = []
    cur_y = 0.0
    last_regime = None
    for c in cell_stats:
        if last_regime is not None and c["regime_key"] != last_regime:
            cur_y -= GROUP_GAP
        cur_y -= 1.0
        y_row.append(cur_y)
        last_regime = c["regime_key"]

    # X-extent — leave room for the n-column on the right.
    all_lo = min((c["mean"] - c["ci95_half"]) for c in cell_stats if c["n"] > 0)
    all_hi = max((c["mean"] + c["ci95_half"]) for c in cell_stats if c["n"] > 0)
    span   = all_hi - all_lo
    pad    = max(span * 0.18, 0.02)
    x_lo, x_hi = all_lo - pad, all_hi + pad

    # Subtle alternating regime-block shading: shade the MIDDLE regime group
    # (Federated Random) very lightly so the three blocks are visually distinct
    # without needing horizontal rules.
    regime_groups = []
    cur_grp = []
    for c, y in zip(cell_stats, y_row):
        if cur_grp and c["regime_key"] != cur_grp[0][0]["regime_key"]:
            regime_groups.append(cur_grp); cur_grp = []
        cur_grp.append((c, y))
    if cur_grp: regime_groups.append(cur_grp)
    for gi, group in enumerate(regime_groups):
        if gi % 2 == 0:
            continue  # alternate: shade only the middle group
        y_top    = group[0][1]  + 0.5
        y_bottom = group[-1][1] - 0.5
        ax.axhspan(y_bottom, y_top, color="#f4f3ee", zorder=0)

    # Vertical parity line at x=0 — subtle dashed gray.
    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.7, zorder=1)

    # Markers + error bars, regime-colored.
    SIG_MS, NS_MS = 11.0, 11.0
    BAR_LW, EDGE_LW = 1.5, 1.4
    CAP_W = 4.5
    for c, y in zip(cell_stats, y_row):
        if c["n"] == 0:
            ax.text(0, y, " (no shared seeds)", va="center", fontsize=8, color="gray",
                    fontstyle="italic")
            continue
        m, ci = c["mean"], c["ci95_half"]
        col   = c["color"]   # regime hue (slate / sky-blue / rose-pink)
        excludes_zero = (m - ci) > 0 or (m + ci) < 0
        ax.errorbar([m], [y], xerr=[[ci], [ci]], fmt="none",
                    ecolor=col, elinewidth=BAR_LW,
                    capsize=CAP_W, capthick=BAR_LW, zorder=2)
        if excludes_zero:
            ax.plot([m], [y], marker="o", markersize=SIG_MS,
                    markerfacecolor=col, markeredgecolor=col,
                    markeredgewidth=EDGE_LW, zorder=3)
        else:
            ax.plot([m], [y], marker="o", markersize=NS_MS,
                    markerfacecolor="white", markeredgecolor=col,
                    markeredgewidth=EDGE_LW, zorder=3)

    # Y-axis: "Regime — Dataset" labels, right-aligned, breathing room.
    ytick_labels = [f"{c['regime_label']} — {c['dataset_label']}" for c in cell_stats]
    ax.set_yticks(y_row)
    ax.set_yticklabels(ytick_labels, fontsize=10)
    ax.tick_params(axis="y", which="both", length=0, pad=8)
    for tl in ax.get_yticklabels():
        tl.set_horizontalalignment("right")

    # Right-side n column — small, gray, with a small "n" header above the first row.
    n_col_x = x_hi + pad * 0.30
    for c, y in zip(cell_stats, y_row):
        if c["n"] == 0:
            continue
        ax.text(n_col_x, y, f"$n{{=}}{c['n']}$",
                va="center", ha="left", fontsize=9, color="#888888")
    top_y = max(y_row) + 0.7
    ax.text(n_col_x, top_y, "$n$", va="center", ha="left", fontsize=9, color="#888888")

    # X-axis label.
    ax.set_xlabel(r"$\Delta$IoU $\;=\;$ IoU(quantum) $-$ IoU(classical)", fontsize=11)

    # Limits — leave a bit of right-side room for the n column.
    ax.set_xlim(x_lo, x_hi + pad * 0.9)
    ax.set_ylim(min(y_row) - 0.7, max(y_row) + 1.1)

    # Spines: only bottom.
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"iou_delta_forest{_OUT_SUFFIX}.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
