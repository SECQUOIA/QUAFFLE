"""Unit and regression tests for package diagnostics uncovered by the mini-flood workflow (#35):
- D1: results_tracker.py honors config['results_dir'] / QUAFFLE_RESULTS_DIR and isolates mini-run workbooks
- D2: scripts/build_exemplar_figure.py supports mini-dataset / Wave 2 checkpoints, --data-root, --results-dir, and --seed
- D3: visualize_circuit.py supports --q-ansatz hqunet, --hqunet-grid {2,4}, and dual-basis <Z>,<X> barplots
- D4: tools/compare_backends.py parses Wave 2 hyphenated variants (_abl-...) and mini scale tags (_bc..._img..._dr...)
- D5: tools/bench_quantum.py and misc/generate_*.py support Wave 2 (hqunet, --q-spatial-mode, --base-channels, --quantum-channels)
- D6: utilsTorch.visualize_results clips RGB images to [0, 1] and stops evaluating batches once num_samples is reached
"""

import json
import os
import warnings

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# D1: results_tracker.py isolation for --results-dir and mini-scale runs
# ---------------------------------------------------------------------------

def test_d1_results_tracker_honors_results_dir_and_mini_scale(tmp_path, monkeypatch):
    import results_tracker as rt

    monkeypatch.delenv("QUAFFLE_TRACKER_DIR", raising=False)
    monkeypatch.delenv("QUAFFLE_RESULTS_DIR", raising=False)
    monkeypatch.setattr(rt, "_RESULTS_DIR", rt._DEFAULT_RESULTS_DIR)

    mini_results_dir = tmp_path / "mini_results"
    mini_cfg = {
        "seed": 42,
        "base_channels": 4,
        "image_size": 32,
        "data_root": str(tmp_path / "mini_data"),
        "results_dir": str(mini_results_dir),
    }
    path_mini = rt.resolve_tracker_excel_path("optical", config=mini_cfg)
    assert path_mini == str(mini_results_dir / "results_summary_optical_mini.xlsx")

    # Full-scale config with custom results_dir writes results_summary_{dataset}.xlsx inside results_dir
    full_cfg = {
        "seed": 42,
        "base_channels": 32,
        "image_size": 128,
        "results_dir": str(mini_results_dir),
    }
    path_full = rt.resolve_tracker_excel_path("optical", config=full_cfg)
    assert path_full == str(mini_results_dir / "results_summary_optical.xlsx")

    # Write a mini-run row and confirm <repo>/results/ is untouched
    val_m = {"loss": 0.4, "accuracy": 0.85, "auc": 0.9, "iou": 0.7, "dice": 0.82}
    test_m = {"loss": 0.45, "accuracy": 0.83, "auc": 0.88, "iou": 0.68, "dice": 0.80}
    rt.save_results_to_excel(
        "Centralized Quantum Torch",
        "torch",
        "centralized",
        "quantum",
        val_m,
        test_m,
        mini_cfg,
        dataset="optical",
    )
    assert (mini_results_dir / "results_summary_optical_mini.xlsx").is_file()
    assert not (mini_results_dir / "results_summary_optical.xlsx").exists()


# ---------------------------------------------------------------------------
# D2: scripts/build_exemplar_figure.py on mini-dataset and Wave 2 checkpoints
# ---------------------------------------------------------------------------

def test_d2_exemplar_figure_mini_and_wave2_checkpoint_support(tmp_path, monkeypatch):
    from scripts import build_exemplar_figure as bef
    from tools.make_mini_flood_dataset import generate_mini_flood_dataset
    from tools import training_common as tc
    from utilsTorch import utilsTorch as ut

    # 1. Verify _parse_exemplar_variant and _canonical_vtag handle hqunet_G2_L2
    ans, layers = bef._parse_exemplar_variant("hqunet_G2_L2")
    assert ans == "hqunet" and layers == 2
    assert bef._canonical_vtag("hqunet_G2_L2") == "_hqunet_G2_L2"

    # 2. End-to-end mini-dataset exemplar generation with downscaled Wave 2 checkpoints
    data_root = tmp_path / "mini_data"
    results_dir = tmp_path / "mini_results"
    out_dir = tmp_path / "exemplars_out"
    generate_mini_flood_dataset(
        str(data_root),
        n_train=6,
        n_test=3,
        image_size=32,
        seed=42,
        seeds=(42,),
    )

    for ds_name, ds_suffix in (("optical", ""), ("sar", "_sar")):
        cfg = {
            "seed": 42,
            "dataset": ds_name,
            "image_size": 32,
            "base_channels": 4,
            "init_dim": 4,
            "out_dim": 2,
            "dim_mults": (1, 2, 4, 8),
            "resnet_block_groups": 4,
            "quantum_channels": 4,
            "quantum_backend": "pennylane",
            "q_ansatz": "hqunet",
            "q_layers": 1,
            "hqunet_grid": 2,
            "backbone_conv_mode": "depthwise_sep",
            "use_time_emb": False,
            "data_root": str(data_root),
            "results_dir": str(results_dir),
        }
        run_dir = tc.resolve_output_dir(
            f"results/flood_federated_optical_pytorch_pennylane_random{ds_suffix}_hqunet_G2_L1_seed42",
            cfg,
        )
        os.makedirs(run_dir, exist_ok=True)
        model = ut.QVUNetSegmentation(cfg, seed=42)
        _ = sum(p.numel() for p in model.parameters())
        ckpt_payload = {
            "best_parameters": [p.detach().cpu().numpy() for p in model.parameters()],
            "parameters": [p.detach().cpu().numpy() for p in model.parameters()],
            "fingerprint": cfg,
        }
        torch.save(ckpt_payload, os.path.join(run_dir, "fed_resume.pt"))

    bef.main(
        [
            "--results-dir", str(results_dir),
            "--data-root", str(data_root),
            "--out-dir", str(out_dir),
            "--seed", "42",
            "--qc", "4",
            "--variant", "hqunet_G2_L1",
        ]
    )
    assert (out_dir / "exemplars_grid.png").is_file()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert "optical" in meta and "sar" in meta


# ---------------------------------------------------------------------------
# D3: visualize_circuit.py supports --q-ansatz hqunet
# ---------------------------------------------------------------------------

def test_d3_visualize_circuit_supports_hqunet(tmp_path):
    import visualize_circuit as vc

    out_dir = tmp_path / "circuits"
    vc.main(["--q-ansatz", "hqunet", "--hqunet-grid", "2", "--q-layers", "1", "--out", str(out_dir)])
    txt_file = out_dir / "quantum_circuit_hqunet_G2_L1.txt"
    png_file = out_dir / "quantum_circuit_hqunet_G2_L1.png"
    bar_file = out_dir / "quantum_circuit_hqunet_G2_L1_expvals.png"
    assert txt_file.is_file() and txt_file.stat().st_size > 0
    assert png_file.is_file() and png_file.stat().st_size > 0
    assert bar_file.is_file() and bar_file.stat().st_size > 0

    # Verify dual-basis expectation vector has length 2 * G^2 = 8 in [-1, 1]
    x, params = vc._sample_inputs(seed=42, ansatz="hqunet", n_layers=1, hqunet_grid=2)
    vals = vc.eval_hqunet_dual_expvals(x, params, grid=2, n_layers=1)
    assert vals.shape == (8,)
    assert np.all(vals >= -1.0 - 1e-6) and np.all(vals <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# D4: tools/compare_backends.py parses Wave 2 and mini-scale run directories
# ---------------------------------------------------------------------------

def test_d4_compare_backends_parses_wave2_and_mini_scale_runs(tmp_path):
    from tools import compare_backends as cb

    # 1. parse_name handles mini scale tags after _qtorch and hyphenated Wave 2 tags
    p_pl = cb.parse_name(
        "flood_central_optical_pytorch_pennylane_combined_hqunet_G2_L1_depthwise_sep_dice_ce_notime_aug_bc4_img32_dr21c34817_seed42"
    )
    p_qt = cb.parse_name(
        "flood_central_optical_pytorch_pennylane_combined_hqunet_G2_L1_depthwise_sep_dice_ce_notime_aug_qtorch_bc4_img32_dr21c34817_seed42"
    )
    assert p_pl is not None and p_qt is not None
    assert p_pl[:4] == p_qt[:4]  # (mode, ds, seed, variant) match!
    assert p_pl[4] == "" and p_qt[4] == "_qtorch"

    # 2. End-to-end single-directory comparison (root_a == root_b defaults to _qtorch vs '')
    for dname, qb in (
        ("flood_central_optical_pytorch_pennylane_reupload_L1_bc4_img32_dr12345678_seed42", "pennylane"),
        ("flood_central_optical_pytorch_pennylane_reupload_L1_qtorch_bc4_img32_dr12345678_seed42", "torch"),
    ):
        rdir = tmp_path / dname / "results"
        rdir.mkdir(parents=True)
        (rdir / "metrics.json").write_text(
            json.dumps(
                {
                    "q_backend": qb,
                    "test": {"loss": 0.4, "accuracy": 0.85, "auc": 0.9, "iou": 0.72, "dice": 0.83},
                    "knockout": {
                        "zero": {"iou": 0.70, "dice": 0.81, "auc": 0.88},
                        "random": {"iou": 0.69, "dice": 0.80, "auc": 0.87},
                    },
                }
            )
        )
    csv_out = tmp_path / "cmp.csv"
    cb.main([str(tmp_path), "--variant", "_reupload_L1", "--csv", str(csv_out)])
    assert csv_out.is_file() and csv_out.stat().st_size > 0


# ---------------------------------------------------------------------------
# D5: tools/bench_quantum.py and misc/generate_grid.py support Wave 2 options
# ---------------------------------------------------------------------------

def test_d5_bench_quantum_and_generate_grid_support_wave2(tmp_path):
    from tools import bench_quantum as bq
    from misc import generate_grid as gg

    json_out = tmp_path / "bench_hq.json"
    res = bq.main(
        [
            "--device", "cpu",
            "--steps", "1",
            "--warmup", "0",
            "--batch-size", "2",
            "--image-size", "32",
            "--base-channels", "4",
            "--quantum-channels", "4",
            "--q-ansatz", "hqunet",
            "--hqunet-grid", "2",
            "--q-layers", "1",
            "--quick",
            "--json", str(json_out),
        ]
    )
    assert json_out.is_file()
    assert any(r["kind"] == "block" and r["q_ansatz"] == "hqunet" for r in res["rows"])
    assert any(r["kind"] == "train_step" for r in res["rows"])

    # Verify generate_grid supports --q-ansatz hqunet --hqunet-grid 2 and centralized quantum_channels
    j = gg.build_job(
        "quantum",
        "centralized",
        "optical",
        42,
        str(tmp_path),
        q_ansatz="hqunet",
        q_layers=2,
        hqunet_grid=2,
        quantum_channels=16,
    )
    assert "--q-ansatz" in j["args"] and "hqunet" in j["args"]
    assert "--hqunet-grid" in j["args"] and "2" in j["args"]
    assert "--quantum_channels" in j["args"] and "16" in j["args"]
    assert "_hq" in j["job_name"]


# ---------------------------------------------------------------------------
# D6: visualize_results clips RGB images and stops after num_samples
# ---------------------------------------------------------------------------

def test_d6_visualize_results_clips_rgb_and_stops_early(tmp_path):
    from utilsTorch import utilsTorch as ut

    class _CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(self, x):
            self.forward_calls += 1
            B, _, H, W = x.shape
            return torch.zeros(B, 2, H, W, device=x.device)

    # 3 batches of 2 images each, with z-score values outside [-1, 1] (e.g. [-4, +4])
    batches = [
        (torch.linspace(-4.0, 4.0, 2 * 3 * 16 * 16).reshape(2, 3, 16, 16), torch.zeros(2, 1, 16, 16))
        for _ in range(3)
    ]
    model = _CountingModel()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ut.visualize_results(model, batches, torch.device("cpu"), str(tmp_path), num_samples=2)
    clip_warnings = [w for w in caught if "Clipping input data" in str(w.message)]
    assert not clip_warnings, f"Unexpected imshow clipping warning: {clip_warnings}"
    assert model.forward_calls == 1, (
        f"Expected visualize_results to stop after 1 batch for num_samples=2, got {model.forward_calls} forward calls"
    )
    assert (tmp_path / "results" / "sample_1.png").is_file()
    assert (tmp_path / "results" / "sample_2.png").is_file()
