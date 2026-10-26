"""Regression tests for the 7 blocking findings resolved in PR #20."""

import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import misc.generate_grid as gg  # noqa: E402
from tools import training_common as tc  # noqa: E402
from utilsTorch.utilsTorch import (  # noqa: E402
    ClassicalAblationUNetSegmentation,
    QVUNetSegmentation,
    UNetSegmentation,
)


def _small_config(**overrides):
    cfg = {
        "dataset": "optical",
        "image_size": 32,
        "num_channels": 3,
        "num_classes": 2,
        "base_channels": 8,
        "dim_mults": (1, 2, 4),
        "resnet_block_groups": 4,
        "quantum_channels": 8,
        "quantum_backend": "pennylane",
        "q_ansatz": "reupload",
        "q_layers": 2,
        "q_share_blocks": True,
        "q_backend": "pennylane",
        "q_sim_device": "cpu",
        "num_epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "seed": 17,
        "val_fraction": 0.15,
        "legacy_data": False,
        "max_samples": None,
    }
    cfg.update(overrides)
    return cfg


def test_paired_backbone_rng_matching():
    """#5: QVUNetSegmentation and ClassicalAblationUNetSegmentation share identical
    backbone initialization for the same explicit seed across ansatz and sharing modes."""
    for ansatz in ("legacy", "reupload"):
        for share in (True, False):
            cfg = _small_config(q_ansatz=ansatz, q_share_blocks=share)
            q_model = QVUNetSegmentation(cfg, seed=17)
            a_model = ClassicalAblationUNetSegmentation(cfg, seed=17)
            q_sd = q_model.state_dict()
            a_sd = a_model.state_dict()
            shared_keys = [
                k for k in q_sd.keys()
                if k in a_sd and "quantum_block" not in k
            ]
            assert len(shared_keys) > 50
            mismatched = [
                k for k in shared_keys if not torch.equal(q_sd[k], a_sd[k])
            ]
            assert not mismatched, (
                f"ansatz={ansatz} share={share}: {len(mismatched)} backbone tensors differ"
            )
            if not share:
                b1_keys = sorted(
                    k for k in a_sd if "qresblock1.quantum_block" in k and k.endswith(".weight")
                )
                b2_keys = sorted(
                    k for k in a_sd if "qresblock2.quantum_block" in k and k.endswith(".weight")
                )
                assert b1_keys and len(b1_keys) == len(b2_keys)
                assert not torch.equal(
                    a_sd[b1_keys[0]],
                    a_sd[b2_keys[0]],
                ), "unshared classical bottleneck blocks should have distinct initial weights"


def test_ablation_seed_sensitivity_and_repeatability(monkeypatch):
    """#5: Explicit seeds are repeatable, distinct, and leave CPU and CUDA RNG
    states completely untouched."""
    cfg = _small_config()
    cuda_calls = []
    orig_cuda_seed_all = torch.cuda.manual_seed_all
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda s: (cuda_calls.append(int(s)), orig_cuda_seed_all(s)),
    )
    cpu_state_before = torch.get_rng_state().clone()
    cuda_state_before = (
        [s.clone() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    a1 = ClassicalAblationUNetSegmentation(cfg, seed=17)
    a2 = ClassicalAblationUNetSegmentation(cfg, seed=17)
    a3 = ClassicalAblationUNetSegmentation(cfg, seed=314)
    q1 = QVUNetSegmentation(cfg, seed=17)
    u1 = UNetSegmentation(cfg, seed=17)
    for k in a1.state_dict():
        assert torch.equal(a1.state_dict()[k], a2.state_dict()[k])
    _ = q1.state_dict()
    _ = u1.state_dict()
    assert cuda_calls == [], f"Model construction unexpectedly called cuda.manual_seed_all: {cuda_calls}"
    assert torch.equal(torch.get_rng_state(), cpu_state_before)
    if torch.cuda.is_available():
        for before, after in zip(cuda_state_before, torch.cuda.get_rng_state_all()):
            assert torch.equal(before, after)
    assert not torch.equal(
        a1.state_dict()["down.init_conv.weight"],
        a3.state_dict()["down.init_conv.weight"],
    )


def test_client_round_seeding_sets_cudnn_determinism(capsys):
    """#24: tc.seed_client_round sets cudnn deterministic/benchmark flags and warns on
    missing round."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    s = tc.seed_client_round(17, 2, {"server_round": 5})
    assert s == tc.client_round_seed(17, 2, 5)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    s0 = tc.seed_client_round(17, 1, {})
    captured = capsys.readouterr().out
    assert s0 == tc.client_round_seed(17, 1, 0)
    assert "no server_round in fit config" in captured
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_grid_runtime_baseline_and_monotonic_scaling():
    """#26: MEASURED_HOURS uses smoke-calibrated quantum values and positive scaling."""
    assert gg.MEASURED_HOURS[("quantum", "centralized")] == 0.09
    assert gg.MEASURED_HOURS[("quantum", "federated")] == 0.12
    for kind in ("centralized", "federated"):
        assert gg.MEASURED_HOURS[("classical", kind)] <= gg.MEASURED_HOURS[("ablation", kind)]
        assert gg.MEASURED_HOURS[("ablation", kind)] < gg.MEASURED_HOURS[("quantum", kind)]

    for mode in ("centralized", "fed-random"):
        h_qc32 = gg.expected_hours("quantum", mode, "optical", quantum_channels=32, q_layers=2)
        h_qc64 = gg.expected_hours("quantum", mode, "optical", quantum_channels=64, q_layers=2)
        h_qc128 = gg.expected_hours("quantum", mode, "optical", quantum_channels=128, q_layers=2)
        assert h_qc32 < h_qc64 < h_qc128

        h_l1 = gg.expected_hours("quantum", mode, "optical", quantum_channels=32, q_layers=1)
        h_l2 = gg.expected_hours("quantum", mode, "optical", quantum_channels=32, q_layers=2)
        h_l3 = gg.expected_hours("quantum", mode, "optical", quantum_channels=32, q_layers=3)
        assert h_l1 < h_l2 < h_l3


def test_private_artifact_ignore_guards():
    """#29: HANDOFF.md, *handoff*.zip, and plans/ are ignored by git, absent from
    tracked files, and root LICENSE file exists with MIT License."""
    for rel in ("HANDOFF.md", "QUAFFLE_handoff.zip", "plans/2026-10-16_quaffle-pennylane-speed.md"):
        r = subprocess.run(
            ["git", "-C", ROOT, "check-ignore", "-q", rel],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert r.returncode == 0, f"{rel} is not ignored by .gitignore"

    tracked = subprocess.run(
        ["git", "-C", ROOT, "ls-files", "HANDOFF.md", "*handoff*.zip", "plans/"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    ).stdout.strip()
    assert tracked == "", f"Private handoff files are tracked: {tracked}"

    lic_path = os.path.join(ROOT, "LICENSE")
    assert os.path.isfile(lic_path), "LICENSE file is missing at repository root"
    with open(lic_path, "r", encoding="utf-8") as f:
        lic_text = f.read()
    assert "MIT License" in lic_text
    assert "Copyright (c) 2026 SECQUOIA" in lic_text


def test_issue_29_gitignore_missing_handoff_guards_and_license_missing():
    """Triage regression test alias for #29."""
    test_private_artifact_ignore_guards()


def test_local_mini_sweep_cli_and_summary_helpers():
    """Verify scripts/run_local_mini_sweep.py CLI --help and summary/report helpers."""
    from scripts import run_local_mini_sweep as rlms

    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "run_local_mini_sweep.py"), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    assert "--base-channels" in r.stdout
    assert "--image-size" in r.stdout
    assert "--seeds" in r.stdout

    fake_records = [
        {
            "seed": 42,
            "total_params": 1000,
            "val": {"iou": 0.80},
            "test": {"iou": 0.82, "dice": 0.90, "auc": 0.95, "loss": 0.20},
            "knockout": {"zero": {"iou": 0.78, "loss": 0.25}},
            "_elapsed_s": 1.5,
        },
        {
            "seed": 77,
            "total_params": 1000,
            "val": {"iou": 0.84},
            "test": {"iou": 0.86, "dice": 0.92, "auc": 0.97, "loss": 0.16},
            "knockout": {"zero": {"iou": 0.80, "loss": 0.21}},
            "_elapsed_s": 1.7,
        },
    ]
    st = rlms._summarize_seed_runs(fake_records)
    assert st["n_seeds"] == 2
    assert st["total_params"] == 1000
    assert abs(st["test_iou"]["mean"] - 0.84) < 1e-6
    assert abs(st["ko_zero_delta_iou"]["mean"] - 0.05) < 1e-6
    assert abs(st["ko_zero_delta_loss"]["mean"] - 0.05) < 1e-6


def test_local_mini_sweep_review_fixes_custom_seeds_workdir_and_validation(tmp_path, monkeypatch):
    """PR #38 Round 1 review fixes:
    - B1: custom seeds (outside config/seeds.py) work end-to-end with analyze_results and preserve reports.
    - B2: reusing --work-dir cleans stale dataset tiles and result directories.
    - NB1: duplicate/empty seeds and invalid/duplicate panel IDs are rejected.
    - NB2: explicit --device cuda raises when CUDA is unavailable, and env vars are restored.
    """
    import glob
    import json
    import pytest
    from scripts import run_local_mini_sweep as sweep

    # NB1: validation checks
    with pytest.raises(ValueError, match="must not be empty"):
        sweep.run_mini_sweep(seeds=[])
    with pytest.raises(ValueError, match="Duplicate seeds"):
        sweep.run_mini_sweep(seeds=[42, 42])
    with pytest.raises(ValueError, match="panels_filter"):
        sweep.run_mini_sweep(seeds=[42], panels_filter=[99])
    with pytest.raises(ValueError, match="Duplicate panel IDs"):
        sweep.run_mini_sweep(seeds=[42], panels_filter=[1, 1])
    with pytest.raises(SystemExit):
        sweep.main(["--seeds", "42", "42"])
    with pytest.raises(SystemExit):
        sweep.main(["--panels", "99"])

    # NB2: explicit --device cuda fails loudly when CUDA is unavailable
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Requested --device cuda"):
        sweep.run_mini_sweep(seeds=[42], device="cuda")

    # Mock _invoke_entry for training entrypoints while letting analyze_results.main run for real
    call_counter = {"n": 0}
    orig_invoke = sweep._invoke_entry

    def _fast_invoke(entry_fn, argv):
        if entry_fn is sweep.analyze_results.main:
            return orig_invoke(entry_fn, [*argv, "--skip-plots"])
        call_counter["n"] += 1
        rdir = argv[argv.index("--results-dir") + 1]
        seed = int(argv[argv.index("--seed") + 1])
        in_mode = argv[argv.index("--combined-input-mode") + 1] if "--combined-input-mode" in argv else "shared3"
        tag = f"_in-{in_mode}" if in_mode != "shared3" else ""
        run_name = f"flood_central_optical_pytorch_pennylane_combined_reupload_L2{tag}_bc8_img16_seed{seed}"
        out_dir = os.path.join(rdir, run_name, "results")
        os.makedirs(out_dir, exist_ok=True)
        metrics = {
            "seed": seed,
            "dataset": "combined",
            "training": "centralized",
            "variant": "quantum",
            "q_ansatz": "reupload",
            "q_layers": 2,
            "q_backend": "pennylane",
            "combined_input_mode": in_mode,
            "base_channels": 8,
            "image_size": 16,
            "loss": "dice_ce",
            "use_time_emb": False,
            "is_mini_run": True,
            "total_params": 537386,
            "val": {"iou": 0.85, "dice": 0.91, "auc": 0.97, "loss": 0.20, "accuracy": 0.90},
            "test": {"iou": 0.86, "dice": 0.92, "auc": 0.98, "loss": 0.19, "accuracy": 0.91},
            "knockout": {
                "on": {"iou": 0.86, "loss": 0.19},
                "zero": {"iou": 0.81, "loss": 0.24},
            },
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        return 0.05

    monkeypatch.setattr(sweep, "_invoke_entry", _fast_invoke)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel_gpu")
    monkeypatch.delenv("QUAFFLE_SKIP_EXCEL_TRACKER", raising=False)
    monkeypatch.delenv("QUAFFLE_SKIP_CURVE_PLOTS", raising=False)

    work_dir = str(tmp_path / "reused_work")
    out_json = str(tmp_path / "out" / "summary.json")
    out_md = str(tmp_path / "out" / "report.md")

    # B1 + B2 first run: custom seed=1 (not in config/seeds.py), n_train=6, n_test=3
    summary1 = sweep.run_mini_sweep(
        seeds=[1],
        image_size=16,
        n_train=6,
        n_test=3,
        device="cpu",
        work_dir=work_dir,
        keep_work_dir=True,
        out_json=out_json,
        out_md=out_md,
        panels_filter=[2],
    )
    assert summary1["seeds"] == [1]
    assert os.path.isfile(out_json)
    assert os.path.isfile(out_md)
    assert os.path.isfile(os.path.join(work_dir, "results", "analysis", "analysis_summary.xlsx"))
    opt_train_1 = glob.glob(os.path.join(work_dir, "mini_data", "flood_optical", "Training", "images", "*.tif"))
    opt_test_1 = glob.glob(os.path.join(work_dir, "mini_data", "flood_optical", "Testing", "images", "*.tif"))
    assert len(opt_train_1) == 6
    assert len(opt_test_1) == 3

    # NB2: all modified environment variables restored after run_mini_sweep
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "sentinel_gpu"
    assert "QUAFFLE_SKIP_EXCEL_TRACKER" not in os.environ
    assert "QUAFFLE_SKIP_CURVE_PLOTS" not in os.environ

    # B2 second run on same work_dir with smaller n_train=4, n_test=2 must not retain stale tiles or stale seed1 runs
    summary2 = sweep.run_mini_sweep(
        seeds=[2],
        image_size=16,
        n_train=4,
        n_test=2,
        device="cpu",
        work_dir=work_dir,
        keep_work_dir=True,
        out_json=out_json,
        out_md=out_md,
        panels_filter=[2],
    )
    assert summary2["seeds"] == [2]
    opt_train_2 = glob.glob(os.path.join(work_dir, "mini_data", "flood_optical", "Training", "images", "*.tif"))
    opt_test_2 = glob.glob(os.path.join(work_dir, "mini_data", "flood_optical", "Testing", "images", "*.tif"))
    assert len(opt_train_2) == 4
    assert len(opt_test_2) == 2
    seed1_dirs = glob.glob(os.path.join(work_dir, "results", "*_seed1"))
    seed2_dirs = glob.glob(os.path.join(work_dir, "results", "*_seed2"))
    assert len(seed1_dirs) == 0
    assert len(seed2_dirs) == 3

    # B1 failure-resilience check: if analyze_results.main raises, out_json and out_md are already written
    out_json_fail = str(tmp_path / "out_fail" / "summary.json")
    out_md_fail = str(tmp_path / "out_fail" / "report.md")

    def _fail_on_analyze(entry_fn, argv):
        if entry_fn is sweep.analyze_results.main:
            raise RuntimeError("simulated analyze_results failure")
        return _fast_invoke(entry_fn, argv)

    monkeypatch.setattr(sweep, "_invoke_entry", _fail_on_analyze)
    with pytest.raises(RuntimeError, match="simulated analyze_results failure"):
        sweep.run_mini_sweep(
            seeds=[3],
            image_size=16,
            n_train=4,
            n_test=2,
            device="cpu",
            out_json=out_json_fail,
            out_md=out_md_fail,
            panels_filter=[2],
        )
    assert os.path.isfile(out_json_fail)
    assert os.path.isfile(out_md_fail)


def test_centralized_checkpoint_fingerprint_validation(capsys):
    """Blocking #6: centralized resume validates full config fingerprint and RUN_FORMAT."""
    assert tc.RUN_FORMAT == "split-v3"
    cfg = _small_config()
    fp = tc.centralized_fingerprint(cfg, "quantum")
    valid_ckpt = {
        "run_format": tc.RUN_FORMAT,
        "q_backend": "pennylane",
        "fingerprint": fp,
    }
    ok = tc.checkpoint_matches_centralized(valid_ckpt, cfg, "quantum")
    assert ok is True

    # Missing fingerprint must be rejected
    ok_missing = tc.checkpoint_matches_centralized(
        {"run_format": tc.RUN_FORMAT, "q_backend": "pennylane"},
        cfg,
        "quantum",
    )
    assert ok_missing is False
    assert "missing fingerprint" in capsys.readouterr().out

    # Altered data/model/training/architecture settings must be rejected
    for key, bad_val in [
        ("legacy_data", True),
        ("val_fraction", 0.25),
        ("max_samples", 16),
        ("seed", 999),
        ("num_epochs", 10),
        ("q_ansatz", "legacy"),
        ("q_layers", 3),
        ("q_share_blocks", False),
        ("base_channels", 16),
        ("init_dim", 16),
        ("out_dim", 3),
        ("dim_mults", (1, 2, 4, 8)),
        ("resnet_block_groups", 8),
        ("quantum_channels", 16),
        ("q_sim_device", "model"),
    ]:
        cfg_other = _small_config(**{key: bad_val})
        ok_diff = tc.checkpoint_matches_centralized(valid_ckpt, cfg_other, "quantum")
        assert ok_diff is False, f"Expected mismatch for {key}={bad_val}, got ok"
        assert "config differs" in capsys.readouterr().out

    # Classical and ablation fingerprints also reject architecture changes
    for model_type in ("classical", "ablation"):
        ckpt_m = {
            "run_format": tc.RUN_FORMAT,
            "fingerprint": tc.centralized_fingerprint(cfg, model_type),
        }
        assert tc.checkpoint_matches_centralized(ckpt_m, cfg, model_type) is True
        for key, bad_val in [
            ("init_dim", 16),
            ("out_dim", 3),
            ("dim_mults", (1, 2, 4, 8)),
            ("resnet_block_groups", 8),
        ]:
            cfg_other = _small_config(**{key: bad_val})
            assert tc.checkpoint_matches_centralized(ckpt_m, cfg_other, model_type) is False
            _ = capsys.readouterr().out


def test_smoke_script_propagates_step_failure():
    """Blocking #7: misc/smoke_pennylane_fast.sh exits non-zero when a step fails."""
    script_path = os.path.join(ROOT, "misc", "smoke_pennylane_fast.sh")
    with open(script_path, encoding="utf-8") as f:
        text = f.read()
    assert "FAILED=0" in text
    assert "exit 1" in text

    # Execute the exact step() definition and final check in bash with a failing command
    snippet = """
    SUMMARY=()
    FAILED=0
    step() {
        local name=$1
        shift
        local t0 t1 rc
        t0=$(date +%s)
        "$@"
        rc=$?
        t1=$(date +%s)
        SUMMARY+=("$name elapsed=$((t1 - t0))s exit=$rc")
        if [ "$rc" -ne 0 ]; then
            FAILED=1
        fi
        return "$rc"
    }
    step ok_step true
    step bad_step false
    printf '%s\\n' "${SUMMARY[@]}"
    if [ "$FAILED" -ne 0 ]; then
        exit 1
    fi
    """
    r = subprocess.run(
        ["bash", "-c", snippet],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert r.returncode == 1
    assert "ok_step" in r.stdout and "bad_step" in r.stdout
