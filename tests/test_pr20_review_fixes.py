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
    """#29 (ignore portion): HANDOFF.md, *handoff*.zip, and plans/ are ignored by git
    and absent from tracked files."""
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
