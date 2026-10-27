"""Regression tests for Issue #44: non-divisible quantum_channels % n_qubits warning
and configurable classical MLP control hidden dimension (--ablation-mlp-hidden / full_rank)."""

import argparse
import importlib.util
import os
import sys
import warnings

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

UNET_PATH = os.path.join(REPO_ROOT, "utilsTorch", "unetTorch.py")

try:
    import utilsTorch.unetTorch as unetTorch
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location("unetTorch", UNET_PATH)
    unetTorch = importlib.util.module_from_spec(spec)
    sys.modules["unetTorch"] = unetTorch
    spec.loader.exec_module(unetTorch)

from tools import training_common as tc
from utilsTorch.utilsTorch import ClassicalAblationUNetSegmentation

ClassicalBottleneckBlock = unetTorch.ClassicalBottleneckBlock
QuantumBlock = unetTorch.QuantumBlock


def test_nondivisible_nqubits_emits_warning_on_quantum_and_classical_blocks():
    """When quantum_channels=32 and n_qubits=12 (32 % 12 == 8 != 0), both QuantumBlock
    and ClassicalBottleneckBlock silently drop 8 channels from the circuit/control
    (effective_qc=24) and must emit a UserWarning alerting the caller."""
    assert os.path.abspath(unetTorch.__file__) == UNET_PATH

    with pytest.warns(UserWarning, match=r"quantum_channels=32.*n_qubits=12.*excluded from circuit/control"):
        qb = QuantumBlock(
            backend="pennylane",
            quantum_channels=32,
            q_ansatz="reupload",
            q_layers=2,
            q_backend="torch",
            n_qubits=12,
            seed=42,
        )
    assert qb.active_channels == 24
    assert qb.effective_qc == 24
    assert qb.remaining_channels == 8

    with pytest.warns(UserWarning, match=r"quantum_channels=32.*n_qubits=12.*excluded from circuit/control"):
        cb = ClassicalBottleneckBlock(
            quantum_channels=32,
            q_ansatz="reupload",
            q_layers=2,
            n_qubits=12,
            seed=42,
        )
    assert cb.active_channels == 24
    assert cb.effective_qc == 24
    assert cb.remaining_channels == 8


def test_divisible_nqubits_does_not_emit_warning():
    """Divisible quantum_channels (e.g., 48 for n_qubits=12, or 32 for n_qubits in {2,4,8,16})
    must not emit a non-divisible UserWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        for nq in (2, 4, 8, 12, 16):
            qc = 48 if nq == 12 else 32
            QuantumBlock(
                backend="pennylane",
                quantum_channels=qc,
                q_ansatz="reupload",
                q_layers=2,
                q_backend="torch",
                n_qubits=nq,
                seed=42,
            )
            ClassicalBottleneckBlock(
                quantum_channels=qc,
                q_ansatz="reupload",
                q_layers=2,
                n_qubits=nq,
                seed=42,
            )
    nondiv_warnings = [
        w for w in caught if "not divisible by group width" in str(w.message)
    ]
    assert nondiv_warnings == []


def test_classical_mlp_control_supports_full_rank_mode():
    """ClassicalBottleneckBlock(kind='mlp') defaults to parameter-matched h=3 at L=2
    across n_qubits in {2, 4, 8, 12, 16}, and must also support ablation_mlp_mode='full_rank'
    (h = n_qubits) to isolate parameter-count matching from rank-3 bottlenecking."""
    for nq in (2, 4, 8, 12, 16):
        qc = 48 if nq == 12 else 32
        cb_param = ClassicalBottleneckBlock(
            quantum_channels=qc,
            q_ansatz="reupload",
            q_layers=2,
            n_qubits=nq,
            seed=42,
        )
        assert cb_param.hidden == 3
        assert cb_param.ablation_mlp_mode == "param_matched"
        assert cb_param.ablation_mlp_hidden is None

        cb_rank = ClassicalBottleneckBlock(
            quantum_channels=qc,
            q_ansatz="reupload",
            q_layers=2,
            n_qubits=nq,
            ablation_mlp_mode="full_rank",
            seed=42,
        )
        assert cb_rank.hidden == nq
        assert cb_rank.ablation_mlp_mode == "full_rank"

        # Verify actual input-output Jacobian rank of a single group MLP
        z = torch.randn(nq, dtype=torch.float32, requires_grad=True)
        jac_param = torch.autograd.functional.jacobian(
            lambda inp: cb_param.per_group[0](inp.unsqueeze(0)).squeeze(0), z
        )
        jac_rank = torch.autograd.functional.jacobian(
            lambda inp: cb_rank.per_group[0](inp.unsqueeze(0)).squeeze(0), z
        )
        assert int(torch.linalg.matrix_rank(jac_param).item()) == min(3, nq)
        assert int(torch.linalg.matrix_rank(jac_rank).item()) == nq


def test_ablation_mlp_cli_model_and_fingerprint_wiring():
    """Verify --ablation-mlp-mode and --ablation-mlp-hidden through CLI, model construction,
    bottleneck_param_counts, variant_tag, and checkpoint fingerprints."""
    parser = argparse.ArgumentParser()
    tc.add_common_args(parser, quantum=True, federated=True)

    args_rank = parser.parse_args(
        ["--n-qubits", "16", "--ablation-mlp-mode", "full_rank"]
    )
    cfg_rank = {
        "image_size": 32,
        "base_channels": 8,
        "init_dim": 8,
        "out_dim": 2,
        "dim_mults": (1, 2, 4, 8),
        "resnet_block_groups": 4,
        "quantum_channels": 32,
        "quantum_backend": "pennylane",
        "q_ansatz": "reupload",
        "q_layers": 2,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "num_clients": 2,
        "fed_rounds": 2,
        "local_epochs": 1,
        "seed": 42,
    }
    tc.apply_common_args(cfg_rank, args_rank)
    assert cfg_rank["ablation_mlp_mode"] == "full_rank"
    assert cfg_rank.get("ablation_mlp_hidden") is None
    assert tc.variant_tag(cfg_rank) == "_reupload_L2_nq16_mlp-full_rank"

    model_rank = tc.build_model(ClassicalAblationUNetSegmentation, cfg_rank)
    blk_rank = model_rank.model.vertex.qresblock1.quantum_block
    assert blk_rank.hidden == 16
    assert blk_rank.bottleneck_target_params == 256
    # 2 groups * ((2*16 + 1)*16 + 16) = 2 * 544 = 1088
    assert blk_rank.bottleneck_actual_params == 1088
    assert tc.bottleneck_param_counts(cfg_rank, "ablation") == (256, 1088)
    assert tc.bottleneck_param_counts(cfg_rank, "ablation", model_rank) == (256, 1088)

    # Explicit --ablation-mlp-hidden overrides --ablation-mlp-mode
    args_h = parser.parse_args(
        ["--n-qubits", "8", "--ablation-mlp-mode", "full_rank", "--ablation-mlp-hidden", "6"]
    )
    cfg_h = dict(cfg_rank)
    tc.apply_common_args(cfg_h, args_h)
    assert cfg_h["ablation_mlp_hidden"] == 6
    assert tc.variant_tag(cfg_h) == "_reupload_L2_nq8_mlph6"
    model_h = tc.build_model(ClassicalAblationUNetSegmentation, cfg_h)
    blk_h = model_h.model.vertex.qresblock1.quantum_block
    assert blk_h.hidden == 6
    # 4 groups * ((2*8 + 1)*6 + 8) = 4 * 110 = 440
    assert tc.bottleneck_param_counts(cfg_h, "ablation") == (256, 440)
    assert tc.bottleneck_param_counts(cfg_h, "ablation", model_h) == (256, 440)

    # Fingerprint & backward compatibility check
    fp_default = tc.centralized_fingerprint(
        {**cfg_rank, "ablation_mlp_mode": "param_matched", "ablation_mlp_hidden": None},
        "ablation",
    )
    fp_rank = tc.centralized_fingerprint(cfg_rank, "ablation")
    assert fp_default["ablation_mlp_mode"] == "param_matched"
    assert fp_rank["ablation_mlp_mode"] == "full_rank"
    assert fp_default != fp_rank

    legacy_saved = {
        k: v
        for k, v in fp_default.items()
        if k not in ("ablation_mlp_mode", "ablation_mlp_hidden")
    }
    assert tc._normalize_saved_fingerprint(legacy_saved, "ablation") == fp_default


def test_ablation_mlp_validation_guards():
    with pytest.raises(ValueError, match="ablation_mlp_mode"):
        unetTorch.normalize_ablation_mlp_mode("invalid_mode")

    with pytest.raises(ValueError, match="ablation_mlp_hidden"):
        unetTorch.normalize_ablation_mlp_hidden(0)

    with pytest.raises(ValueError, match="ablation_mlp_hidden"):
        unetTorch.normalize_ablation_mlp_hidden(True)

    with pytest.raises(ValueError, match="ablation_mlp_hidden"):
        unetTorch.normalize_ablation_mlp_hidden(3.5)

    with pytest.raises(ValueError, match="Conflicting"):
        ClassicalBottleneckBlock(
            quantum_channels=16,
            q_ansatz="reupload",
            q_layers=2,
            n_qubits=4,
            hidden=3,
            ablation_mlp_hidden=4,
        )


def test_downstream_parsers_and_generators_support_ablation_mlp_flags():
    import analyze_results
    from misc import generate_grid, generate_qc_ablation_scripts
    from scripts import run_local_mini_sweep as rlms

    # 1. analyze_results._variant_from_fields, _parse_run_dir_full, _infer_variant_from_summary
    v_full_rank = analyze_results._variant_from_fields(
        "reupload", 2, data={"n_qubits": 16, "ablation_mlp_mode": "full_rank"}
    )
    assert v_full_rank == "reupload_L2_nq16_mlp-full_rank"

    v_mlph = analyze_results._variant_from_fields(
        "reupload", 2, data={"n_qubits": 8, "ablation_mlp_hidden": 6}
    )
    assert v_mlph == "reupload_L2_nq8_mlph6"

    parsed = analyze_results._parse_run_dir_full(
        "ablation_centralized_torch_bottleneck_matched_pennylane_reupload_L2_nq16_mlp-full_rank_seed42"
    )
    assert parsed is not None
    assert parsed[-1] == "reupload_L2_nq16_mlp-full_rank"

    parsed_h = analyze_results._parse_run_dir_full(
        "ablation_centralized_torch_bottleneck_matched_pennylane_reupload_L2_nq8_mlph6_seed42"
    )
    assert parsed_h is not None
    assert parsed_h[-1] == "reupload_L2_nq8_mlph6"

    inferred = analyze_results._infer_variant_from_summary(
        "cli_args: --q-ansatz reupload --q-layers 2 --n-qubits 16 --ablation-mlp-mode full_rank"
    )
    assert inferred == "reupload_L2_nq16_mlp-full_rank"

    # 2. misc/generate_grid.py and misc/generate_qc_ablation_scripts.py
    job = generate_grid.build_job(
        "ablation",
        "centralized",
        "optical",
        42,
        REPO_ROOT,
        n_qubits=16,
        ablation_mlp_mode="full_rank",
    )
    assert "--ablation-mlp-mode full_rank" in job["command"]
    assert "mlp-full_rank" in job["name"]

    job_h = generate_grid.build_job(
        "ablation",
        "fed-random",
        "optical",
        42,
        REPO_ROOT,
        quantum_channels=64,
        n_qubits=8,
        ablation_mlp_hidden=6,
    )
    assert "--ablation-mlp-hidden 6" in job_h["command"]
    assert "mlph6" in job_h["name"]

    # 3. scripts/run_local_mini_sweep.py metadata matching
    meta, explicit = rlms._expected_metadata_from_argv(
        [
            "ablation_studies.py",
            "--mode",
            "centralized",
            "--seed",
            "42",
            "--n-qubits",
            "16",
            "--ablation-mlp-mode",
            "full_rank",
        ]
    )
    assert meta["ablation_mlp_mode"] == "full_rank"
    assert meta["ablation_mlp_hidden"] is None
    assert "ablation_mlp_mode" in explicit
