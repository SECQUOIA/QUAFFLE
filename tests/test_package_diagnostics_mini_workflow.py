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
            f"results/flood_federated_optical_pytorch_pennylane_random{ds_suffix}_qc4_hqunet_G2_L1_seed42",
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
    yielded_batches = 0

    def _counting_iter():
        nonlocal yielded_batches
        for b in batches:
            yielded_batches += 1
            yield b

    model = _CountingModel()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ut.visualize_results(model, _counting_iter(), torch.device("cpu"), str(tmp_path), num_samples=2)
    clip_warnings = [w for w in caught if "Clipping input data" in str(w.message)]
    assert not clip_warnings, f"Unexpected imshow clipping warning: {clip_warnings}"
    assert model.forward_calls == 1, (
        f"Expected visualize_results to stop after 1 batch for num_samples=2, got {model.forward_calls} forward calls"
    )
    assert yielded_batches == 1, (
        f"Expected iterator to yield only 1 batch for num_samples=2, got {yielded_batches}"
    )
    assert (tmp_path / "results" / "sample_1.png").is_file()
    assert (tmp_path / "results" / "sample_2.png").is_file()

    # num_samples=0 consumes 0 batches
    yielded_batches = 0
    ut.visualize_results(model, _counting_iter(), torch.device("cpu"), str(tmp_path), num_samples=0)
    assert yielded_batches == 0


# ---------------------------------------------------------------------------
# Round 1 Review Regression Tests (Findings 1-10)
# ---------------------------------------------------------------------------

def test_r1_exemplar_round_checkpoint_top_level_metrics_and_mixed_variants(tmp_path):
    from scripts import build_exemplar_figure as bef
    from tools import training_common as tc
    from utilsTorch import utilsTorch as ut

    results_dir = tmp_path / "results"
    cfg_reup = {
        "seed": 42,
        "dataset": "optical",
        "image_size": 32,
        "base_channels": 6,
        "init_dim": 6,
        "out_dim": 2,
        "dim_mults": (1, 2, 4, 8),
        "resnet_block_groups": 3,
        "quantum_channels": 16,
        "quantum_backend": "pennylane",
        "q_ansatz": "reupload",
        "q_layers": 2,
        "q_spatial_mode": "patch2x2",
        "backbone_conv_mode": "depthwise_sep",
        "use_time_emb": False,
        "data_root": str(tmp_path / "mini_data"),
        "results_dir": str(results_dir),
    }
    cfg_hq = dict(cfg_reup, quantum_channels=32, q_ansatz="hqunet", q_layers=1, hqunet_grid=2, q_spatial_mode="global")

    dir_reup = tc.resolve_output_dir(
        "results/flood_federated_optical_pytorch_pennylane_random_qc16_reupload_L2_seed42",
        cfg_reup,
    )
    dir_hq = tc.resolve_output_dir(
        "results/flood_federated_optical_pytorch_pennylane_random_hqunet_G2_L1_seed42",
        cfg_hq,
    )
    os.makedirs(os.path.join(dir_reup, "results"), exist_ok=True)
    os.makedirs(os.path.join(dir_hq, "results"), exist_ok=True)

    # Save a periodic checkpoint_round_2.pt WITHOUT fingerprint + top-level metrics.json via write_metrics_json
    model_reup = ut.QVUNetSegmentation(cfg_reup, seed=42)
    _ = sum(p.numel() for p in model_reup.parameters())
    ckpt_round2_path = os.path.join(dir_reup, "checkpoint_round_2.pt")
    torch.save({"round": 2, "model_state_dict": model_reup.state_dict()}, ckpt_round2_path)
    tc.write_metrics_json(
        dir_reup,
        script="federated_demoTorchQ.py",
        training="federated",
        model_type="quantum",
        config=cfg_reup,
        val_metrics={"loss": 0.4, "accuracy": 0.8, "auc": 0.85, "iou": 0.65, "dice": 0.78},
        test_metrics={"loss": 0.4, "accuracy": 0.8, "auc": 0.85, "iou": 0.65, "dice": 0.78},
    )

    model_hq = ut.QVUNetSegmentation(cfg_hq, seed=42)
    _ = sum(p.numel() for p in model_hq.parameters())
    torch.save({"round": 2, "model_state_dict": model_hq.state_dict()}, os.path.join(dir_hq, "checkpoint_round_2.pt"))

    # Finding 2: explicit qc=16, variant="reupload_L2" selects dir_reup, never dir_hq
    picked_reup = bef._fed_ckpt("", seed=42, results_dir=str(results_dir), qc=16, variant="reupload_L2")
    assert picked_reup == ckpt_round2_path

    # Requesting a non-existent variant in a mixed directory returns the missing expected path instead of a wrong model
    picked_missing = bef._fed_ckpt("", seed=42, results_dir=str(results_dir), qc=16, variant="legacy")
    assert not os.path.exists(picked_missing)

    # Finding 1 & 7: _resolve_model_config_for_checkpoint reads top-level metrics.json and computes compatible group count (3 for bc=6)
    raw_ckpt = tc.safe_torch_load(ckpt_round2_path, map_location="cpu")
    resolved_cfg = bef._resolve_model_config_for_checkpoint(raw_ckpt, ckpt_path=ckpt_round2_path)
    assert resolved_cfg["base_channels"] == 6
    assert resolved_cfg["init_dim"] == 6
    assert resolved_cfg["resnet_block_groups"] == 3
    assert resolved_cfg["quantum_channels"] == 16
    assert resolved_cfg["q_spatial_mode"] == "patch2x2"
    assert resolved_cfg["backbone_conv_mode"] == "depthwise_sep"
    assert resolved_cfg["use_time_emb"] is False
    rebuilt_model = ut.QVUNetSegmentation(resolved_cfg, seed=0)
    bef._load_checkpoint_into_model(rebuilt_model, raw_ckpt)


def test_r1_compare_backends_max_samples_and_partial_btag(tmp_path):
    import csv
    from tools import compare_backends as cb
    from tools import training_common as tc

    mini_capped_cfg = {
        "seed": 42,
        "base_channels": 4,
        "image_size": 32,
        "max_samples": 8,
        "data_root": str(tmp_path / "mini_data"),
        "results_dir": str(tmp_path),
    }
    pl_dir = tc.resolve_output_dir(
        "results/flood_central_optical_pytorch_pennylane_reupload_L2_seed42",
        mini_capped_cfg,
    )
    qt_dir = tc.resolve_output_dir(
        "results/flood_central_optical_pytorch_pennylane_reupload_L2_qtorch_seed42",
        mini_capped_cfg,
    )
    assert "_max8_" in os.path.basename(pl_dir)
    p_pl = cb.parse_name(os.path.basename(pl_dir))
    p_qt = cb.parse_name(os.path.basename(qt_dir))
    assert p_pl is not None and p_qt is not None
    assert p_pl[:4] == p_qt[:4]
    assert p_pl[4] == "" and p_qt[4] == "_qtorch"

    for rdir, qb, iou_val in ((pl_dir, "pennylane", 0.90), (qt_dir, "torch", 0.50)):
        os.makedirs(os.path.join(rdir, "results"), exist_ok=True)
        with open(os.path.join(rdir, "results", "metrics.json"), "w") as f:
            json.dump(
                {
                    "q_backend": qb,
                    "test": {"loss": 0.3, "accuracy": 0.85, "auc": 0.9, "iou": iou_val, "dice": 0.8},
                },
                f,
            )

    # Finding 4: Passing only --btag-a _qtorch on a single root still defaults btag_b to '' (no self-comparison)
    csv_out = tmp_path / "partial_btag.csv"
    cb.main([str(tmp_path), "--btag-a", "_qtorch", "--variant", "_reupload_L2", "--csv", str(csv_out)])
    with open(csv_out, newline="") as f:
        rows = list(csv.DictReader(f))
    iou_row = next(r for r in rows if r["metric"] == "test_iou")
    assert abs(float(iou_row["diff"]) - 0.40) < 1e-6


def test_r1_bench_and_generate_grid_wave2_flags(tmp_path):
    from misc import generate_grid as gg
    from tools import bench_quantum as bq

    cfg6 = bq._config(base_channels=6)
    assert cfg6["resnet_block_groups"] == 3

    j = gg.build_job(
        "quantum",
        "centralized",
        "optical",
        42,
        str(tmp_path),
        q_ansatz="reupload",
        q_layers=2,
        q_spatial_mode="patch2x2",
        base_channels=8,
        quantum_channels=16,
    )
    assert "--q-spatial-mode" in j["args"] and "patch2x2" in j["args"]
    assert "--base-channels" in j["args"] and "8" in j["args"]
    assert "--quantum_channels" in j["args"] and "16" in j["args"]
    assert "patch2x2" in j["name"] and "bc8" in j["name"] and "qc16" in j["name"]

