"""
Regression and unit tests for Wave 1D (#7, #8, #9, #11, #28):
  - #7: Table 2 and forest plot include parameter-matched classical ablation
        rows/comparisons and single-seed paired statistics report undefined CI.
  - #8: build_exemplar_figure.py respects QUAFFLE_EXEMPLAR_VARIANT and
        QUAFFLE_EXEMPLAR_QC in both checkpoint paths and model CONFIG.
  - #9: save_training_curves preserves 1-indexed step/round coordinates without
        origin-shifting and labels federated curves as 'Federated Round'.
  - #11: analyze_results.py removes dead Excel schema loaders (_fix_shifted_rows,
         load_dataset_excel, _OLD_COLS, _NEW_COLS), supports --out-dir, and
         ablation_studies.py / visualize_circuit.py docstrings match live code.
  - #28: visualize_circuit.py --q-ansatz compare generates non-empty
         quaffle_circuits_legacy_vs_reupload.png and .pdf schematics.
"""

import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Issue #7: Table 2, forest plot, and paired_stats
# ---------------------------------------------------------------------------

def test_issue7_table2_and_forest_plot_include_ablation_and_valid_single_seed_stats(
    tmp_path, monkeypatch
):
    """Issue #7 triage regression test:
    - Table 2 includes Ablation Fed (Dirichlet) and paired deltas vs ablation.
    - build_forest_plot.CELLS includes Quantum vs Ablation across all 3 regimes.
    - analyze_results.paired_stats returns undefined CI (None) for n=1 and accepts
      both numeric sequences and (seed, delta) tuples.
    """
    import analyze_results as ar
    from scripts import build_forest_plot, build_table2_data

    methods = [
        "Classical Fed (Random)",
        "Ablation Fed (Random)",
        "Quantum Fed (Random)",
        "Classical Fed (Dirichlet)",
        "Ablation Fed (Dirichlet)",
        "Quantum Fed (Dirichlet)",
    ]
    rows = []
    for idx, m in enumerate(methods):
        for s in (42, 123):
            rows.append(
                {
                    "Dataset": "combined",
                    "Method": m,
                    "Seed": s,
                    "Test_IoU": 0.70 + 0.01 * idx + (0.002 if s == 123 else 0.0),
                    "Test_Dice": 0.80 + 0.01 * idx + (0.002 if s == 123 else 0.0),
                }
            )
    monkeypatch.setattr(ar, "load_from_disk", lambda *a, **kw: pd.DataFrame(rows))
    build_table2_data.main(["--out-dir", str(tmp_path)])

    data = json.loads((tmp_path / "headline_pooled_table2_data.json").read_text())
    rows_out = data if isinstance(data, list) else data["rows"]
    by_key = {r["key"]: r for r in rows_out}
    internal_methods = [r["internal_method"] for r in rows_out]
    assert "Ablation Fed (Dirichlet)" in internal_methods, (
        f"Expected Table 2 output to include 'Ablation Fed (Dirichlet)', got {internal_methods}"
    )
    assert "fed_ablation_dirichlet" in by_key
    assert by_key["fed_ablation_dirichlet"]["baseline_key"] == "fed_unet_dirichlet"
    assert by_key["fed_ablation_dirichlet"]["delta_iou"]["n"] == 2

    # Quantum rows must also include paired deltas against same-partition ablation
    for q_key, abl_key in (
        ("quaffle_random", "fed_ablation_random"),
        ("quaffle_dirichlet", "fed_ablation_dirichlet"),
    ):
        q_row = by_key[q_key]
        assert q_row["ablation_baseline_key"] == abl_key
        assert q_row["paired_seeds_vs_ablation"] == [42, 123]
        assert q_row["delta_iou_vs_ablation"]["n"] == 2
        assert q_row["delta_dice_vs_ablation"]["n"] == 2

    # Check build_forest_plot.CELLS includes Quantum vs. Ablation comparisons
    forest_comparators = [cell[3] for cell in build_forest_plot.CELLS]
    assert any("Ablation" in comp for comp in forest_comparators), (
        f"Expected build_forest_plot.CELLS to include Quantum vs. Ablation comparisons, "
        f"got comparators={forest_comparators}"
    )
    assert {"Ablation Central", "Ablation Fed (Random)", "Ablation Fed (Dirichlet)"}.issubset(
        set(forest_comparators)
    )

    # Single-seed paired_stats must not fabricate a 0-width 95% CI
    single_stat_tuple = ar.paired_stats([(42, 0.05)])
    assert single_stat_tuple["n"] == 1
    assert np.isclose(single_stat_tuple["mean"], 0.05)
    assert single_stat_tuple["ci95_half"] is None
    assert single_stat_tuple.get("ci_low") is None or np.isnan(single_stat_tuple.get("ci_low"))
    assert single_stat_tuple.get("ci_high") is None or np.isnan(single_stat_tuple.get("ci_high"))

    single_stat_scalar = ar.paired_stats([0.05])
    assert single_stat_scalar["ci95_half"] is None
    assert single_stat_scalar["ci_low"] is None
    assert single_stat_scalar["ci_high"] is None

    multi_stat = ar.paired_stats([(42, 0.04), (101, 0.06)])
    assert multi_stat["n"] == 2
    assert multi_stat["ci95_half"] is not None and multi_stat["ci95_half"] > 0
    assert np.isclose(multi_stat["ci_low"], multi_stat["mean"] - multi_stat["ci95_half"])
    assert np.isclose(multi_stat["ci_high"], multi_stat["mean"] + multi_stat["ci95_half"])


def test_issue7_forest_plot_end_to_end_18_cells_and_single_seed_handling(
    tmp_path, monkeypatch
):
    """Verify build_forest_plot.main() emits all 18 cells (9 vs Classical + 9 vs Ablation)
    and handles n=1 cells without crashing."""
    import analyze_results as ar
    from scripts import build_forest_plot

    all_methods = [
        "Classical Central",
        "Ablation Central",
        "Quantum Central",
        "Classical Fed (Random)",
        "Ablation Fed (Random)",
        "Quantum Fed (Random)",
        "Classical Fed (Dirichlet)",
        "Ablation Fed (Dirichlet)",
        "Quantum Fed (Dirichlet)",
    ]
    rows = []
    for ds in ("optical", "sar", "combined"):
        for m_idx, m in enumerate(all_methods):
            # Use 1 seed for 'sar' to exercise single-seed CI handling in forest plot
            seeds = (42,) if ds == "sar" else (42, 101)
            for s in seeds:
                rows.append(
                    {
                        "Dataset": ds,
                        "Method": m,
                        "Seed": s,
                        "Test_IoU": 0.60 + 0.01 * m_idx + (0.005 if s == 101 else 0.0),
                        "Test_Dice": 0.72 + 0.01 * m_idx,
                    }
                )
    monkeypatch.setattr(ar, "load_from_disk", lambda *a, **kw: pd.DataFrame(rows))
    forest_out = tmp_path / "forest_out"
    build_forest_plot.main(["--out-dir", str(forest_out)])

    json_path = forest_out / "iou_delta_forest_data.json"
    png_path = forest_out / "iou_delta_forest.png"
    assert json_path.is_file() and json_path.stat().st_size > 0
    assert png_path.is_file() and png_path.stat().st_size > 0

    cells = json.loads(json_path.read_text())
    assert len(cells) == 18
    ablation_cells = [c for c in cells if "Ablation" in c["classical_method"]]
    classical_cells = [c for c in cells if "Classical" in c["classical_method"]]
    assert len(ablation_cells) == 9
    assert len(classical_cells) == 9
    sar_cells = [c for c in cells if c["dataset"] == "sar"]
    assert all(c["n"] == 1 and c["ci95_half"] is None for c in sar_cells)


# ---------------------------------------------------------------------------
# Issue #8: build_exemplar_figure.py respects variant and quantum_channels
# ---------------------------------------------------------------------------

def test_issue8_exemplar_figure_respects_variant_and_qc(monkeypatch):
    """Issue #8 triage regression test:
    build_exemplar_figure must propagate QUAFFLE_EXEMPLAR_VARIANT and
    QUAFFLE_EXEMPLAR_QC to both _fed_ckpt() and CONFIG."""
    import scripts.build_exemplar_figure as bef
    from utilsTorch import utilsTorch as ut

    monkeypatch.setenv("QUAFFLE_EXEMPLAR_VARIANT", "legacy")
    monkeypatch.setenv("QUAFFLE_EXEMPLAR_QC", "16")
    importlib.reload(bef)

    assert bef.CONFIG["quantum_channels"] == 16
    assert bef.CONFIG["q_ansatz"] == "legacy"
    assert bef.CONFIG["q_layers"] == 1

    ckpt_path = bef._fed_ckpt("quantum_federated_torch", "optical", 42)
    assert "_qc16" in ckpt_path and "_legacy" in ckpt_path, (
        f"Expected _fed_ckpt to include '_qc16' and '_legacy', got: {ckpt_path}"
    )

    ckpt_sar = bef._fed_ckpt("_sar", seed=999)
    assert "_sar_qc16_legacy_seed999" in ckpt_sar

    ref_model = ut.QVUNetSegmentation(
        dict(bef.CONFIG, quantum_channels=16, q_ansatz="legacy", q_layers=1),
        seed=0,
    )
    target_model = ut.QVUNetSegmentation(bef.CONFIG, seed=0)
    target_model.load_state_dict(ref_model.state_dict())

    # Also verify reupload_L3 parsing
    monkeypatch.setenv("QUAFFLE_EXEMPLAR_VARIANT", "reupload_L3")
    monkeypatch.setenv("QUAFFLE_EXEMPLAR_QC", "32")
    importlib.reload(bef)
    assert bef.CONFIG["quantum_channels"] == 32
    assert bef.CONFIG["q_ansatz"] == "reupload"
    assert bef.CONFIG["q_layers"] == 3
    ckpt_default_qc = bef._fed_ckpt("", seed=42)
    assert "_qc" not in ckpt_default_qc
    assert "_reupload_L3_seed42" in ckpt_default_qc


# ---------------------------------------------------------------------------
# Issue #9: save_training_curves preserves step/round coordinates and labels
# ---------------------------------------------------------------------------

def test_issue9_save_training_curves_preserves_step_and_federated_round_coordinates(
    tmp_path, monkeypatch
):
    """Issue #9 triage regression test:
    save_training_curves must not subtract first_step from step coordinates and
    must label federated curves as 'Federated Round'."""
    from utilsTorch import utilsTorch as ut

    monkeypatch.setenv("QUAFFLE_SKIP_CURVE_PLOTS", "0")
    recorded_x = []
    recorded_xlabels = []
    orig_plot = plt.plot
    orig_xlabel = plt.xlabel

    def spy_plot(*args, **kwargs):
        if len(args) >= 2:
            recorded_x.append(list(args[0]))
        return orig_plot(*args, **kwargs)

    def spy_xlabel(label, *args, **kwargs):
        recorded_xlabels.append(str(label))
        return orig_xlabel(label, *args, **kwargs)

    monkeypatch.setattr(plt, "plot", spy_plot)
    monkeypatch.setattr(plt, "xlabel", spy_xlabel)

    round_metrics = [
        {"step": 1, "loss": 0.9, "accuracy": 0.7, "auc": 0.75, "iou": 0.5, "dice": 0.65},
        {"step": 2, "loss": 0.8, "accuracy": 0.75, "auc": 0.80, "iou": 0.55, "dice": 0.70},
    ]
    ut.save_training_curves(
        round_metrics,
        str(tmp_path),
        prefix="fed_",
        title_prefix="Federated Validation ",
    )

    assert recorded_x, "plt.plot was not called by save_training_curves"
    assert recorded_x[0] == [1, 2], (
        f"Expected x-coordinates [1, 2] for rounds 1..2, got {recorded_x[0]} "
        "(save_training_curves subtracted first_step)"
    )
    assert all(lbl == "Federated Round" for lbl in recorded_xlabels)

    # Also verify centralized_demoTorchQ records 'epoch' alongside 'step'
    import centralized_demoTorchQ as cdq

    src = inspect.getsource(cdq.train_model)
    assert "val_result['epoch'] = step" in src


# ---------------------------------------------------------------------------
# Issue #11: Dead Excel code removal, --out-dir support, and docstrings
# ---------------------------------------------------------------------------

def test_issue11_analyze_results_parse_args_supports_out_dir_and_no_dead_excel_loader():
    """Issue #11 triage regression test:
    analyze_results.py must not define dead _fix_shifted_rows/load_dataset_excel
    and must support --out-dir in parse_args() and main()."""
    import ablation_studies
    import analyze_results as ar
    import visualize_circuit

    assert not hasattr(ar, "load_dataset_excel") and not hasattr(ar, "_fix_shifted_rows"), (
        "Dead Excel schema loader functions (load_dataset_excel, _fix_shifted_rows) "
        "should be removed from analyze_results.py"
    )
    assert not hasattr(ar, "_OLD_COLS") and not hasattr(ar, "_NEW_COLS")

    args = ar.parse_args(["--out-dir", "/tmp/custom_out"])
    assert getattr(args, "out_dir", None) == "/tmp/custom_out"

    main_src = inspect.getsource(ar.main)
    assert "--out-dir" in main_src, "analyze_results.main() does not reference --out-dir"
    assert "results_summary_*.xlsx" not in main_src

    # Verify docstrings in ablation_studies.py and visualize_circuit.py
    assert "1x1 conv" not in (ablation_studies.__doc__ or "")
    assert "q_backend='pennylane'" in (visualize_circuit.__doc__ or "")


# ---------------------------------------------------------------------------
# Issue #28: visualize_circuit.py --q-ansatz compare
# ---------------------------------------------------------------------------

def test_issue28_visualize_circuit_compare_generates_png_and_pdf(tmp_path):
    """Issue #28: running visualize_circuit.py --q-ansatz compare --out-dir <tmp>
    must exit 0 and generate non-empty quaffle_circuits_legacy_vs_reupload.png and .pdf."""
    import visualize_circuit as vc

    out_dir = tmp_path / "circuits_compare"
    png_path, pdf_path = vc.main(["--q-ansatz", "compare", "--out-dir", str(out_dir)])

    assert Path(png_path).is_file() and Path(png_path).stat().st_size > 1024
    assert Path(pdf_path).is_file() and Path(pdf_path).stat().st_size > 1024
    assert Path(png_path).name == "quaffle_circuits_legacy_vs_reupload.png"
    assert Path(pdf_path).name == "quaffle_circuits_legacy_vs_reupload.pdf"

    # Also test CLI invocation via subprocess
    cli_out = tmp_path / "circuits_cli"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "visualize_circuit.py"),
            "--q-ansatz",
            "compare",
            "--out-dir",
            str(cli_out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"visualize_circuit.py failed: {proc.stderr}"
    assert (cli_out / "quaffle_circuits_legacy_vs_reupload.png").stat().st_size > 1024
    assert (cli_out / "quaffle_circuits_legacy_vs_reupload.pdf").stat().st_size > 1024
