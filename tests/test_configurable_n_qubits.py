import json
import os
import tempfile
import numpy as np
import pytest
import torch

import ablation_studies
import analyze_results
import federated_demoTorchQ
import visualize_circuit
from misc import generate_grid as gg
from misc import generate_qc_ablation_scripts as gqc
from scripts import build_exemplar_figure as bef
from scripts import run_local_mini_sweep as rlms
from tools import bench_quantum as bq
from tools import compare_backends as cb
from tools import training_common as tc
from utilsTorch import unetTorch as unet
from utilsTorch.utilsTorch import ClassicalAblationUNetSegmentation, QVUNetSegmentation


@pytest.mark.parametrize("n_qubits", [2, 4, 8, 12, 16])
def test_vectorized_cnot_ring_tables_exact_and_no_zsign_matrix(n_qubits):
    perm, zsign = unet._cnot_ring_tables(n_qubits, torch.device("cpu"))
    assert zsign is None
    assert isinstance(perm, torch.Tensor)
    assert perm.shape == (1 << n_qubits,)
    assert perm.dtype == torch.long

    # Verify against reference bit-by-bit definition on a sample of basis states
    sample_indices = (
        list(range(1 << n_qubits))
        if n_qubits <= 8
        else [0, 1, 2, 3, 42, (1 << n_qubits) - 2, (1 << n_qubits) - 1]
    )
    for idx in sample_indices:
        bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        for q in range(n_qubits):
            bits[(q + 1) % n_qubits] ^= bits[q]
        out = 0
        for b in bits:
            out = (out << 1) | b
        assert int(perm[out].item()) == idx


@pytest.mark.parametrize("n_qubits", [2, 4, 8, 12, 16])
@pytest.mark.parametrize("q_ansatz", ["legacy", "reupload"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_simulator_parity_all_widths_ansatzes_precisions_and_backends(n_qubits, q_ansatz, dtype):
    q_layers = 1 if q_ansatz == "legacy" else 2
    num_groups = 2 if n_qubits <= 8 else 1
    qc = num_groups * n_qubits
    B = 2 if n_qubits <= 8 else 1
    H, W = 2, 2
    torch.manual_seed(123 + n_qubits)
    x0 = torch.randn(B, qc, H, W, dtype=dtype)
    weights_loss = torch.linspace(0.3, 1.7, B * qc * H * W, dtype=dtype).reshape(B, qc, H, W)

    outs = {}
    x_grads = {}
    named_grads = {}
    for qb in ("pennylane", "pennylane_pergroup", "torch"):
        blk = unet.QuantumBlock(
            backend="pennylane",
            quantum_channels=qc,
            seed=77,
            q_ansatz=q_ansatz,
            q_layers=q_layers,
            q_backend=qb,
            n_qubits=n_qubits,
        )
        if dtype == torch.float64:
            blk = blk.double()
        else:
            blk = blk.float()
        if q_ansatz == "reupload":
            with torch.no_grad():
                for g_idx, s in enumerate(blk.input_scales):
                    s.copy_(
                        torch.linspace(
                            0.6 + 0.1 * g_idx,
                            1.4 + 0.1 * g_idx,
                            q_layers * n_qubits,
                            dtype=dtype,
                        ).reshape(q_layers, n_qubits)
                    )
        x = x0.clone().requires_grad_(True)
        y = blk(x)
        assert y.dtype == dtype
        loss = (y * weights_loss).sum()
        loss.backward()

        assert x.grad is not None
        outs[qb] = y.detach().clone()
        x_grads[qb] = x.grad.detach().clone()
        grads_dict = {}
        for name, p in blk.named_parameters():
            assert p.grad is not None, f"Missing gradient for {name} on backend={qb}"
            grads_dict[name] = p.grad.detach().clone()
        assert len(grads_dict) > 0
        if q_ansatz == "reupload":
            assert any("input_scales" in k for k in grads_dict)
            assert any("quantum_params" in k for k in grads_dict)
        named_grads[qb] = grads_dict

    atol = 1e-11 if dtype == torch.float64 else 1e-5
    rtol = 1e-9 if dtype == torch.float64 else 1e-4
    ref_keys = set(named_grads["pennylane"].keys())
    for other in ("pennylane_pergroup", "torch"):
        assert set(named_grads[other].keys()) == ref_keys
        torch.testing.assert_close(outs["pennylane"], outs[other], atol=atol, rtol=rtol)
        torch.testing.assert_close(x_grads["pennylane"], x_grads[other], atol=atol, rtol=rtol)
        for k in sorted(ref_keys):
            torch.testing.assert_close(
                named_grads["pennylane"][k], named_grads[other][k], atol=atol, rtol=rtol
            )


@pytest.mark.parametrize("q_backend", ["torch", "pennylane", "pennylane_pergroup"])
@pytest.mark.parametrize("q_ansatz", ["legacy", "reupload"])
def test_chunked_token_and_group_simulation_matches_unchunked(monkeypatch, q_backend, q_ansatz):
    q_layers = 1 if q_ansatz == "legacy" else 2
    blk = unet.QuantumBlock(
        backend="pennylane",
        quantum_channels=16,
        seed=11,
        q_ansatz=q_ansatz,
        q_layers=q_layers,
        q_backend=q_backend,
        q_spatial_mode="patch2x2",
        n_qubits=8,
    )
    if q_ansatz == "reupload":
        with torch.no_grad():
            for g_idx, s in enumerate(blk.input_scales):
                s.copy_(torch.linspace(0.7, 1.3, q_layers * 8).reshape(q_layers, 8) + 0.05 * g_idx)

    torch.manual_seed(99)
    x0 = torch.randn(2, 16, 4, 4, requires_grad=True)
    w_loss = torch.linspace(0.5, 1.5, 2 * 16 * 4 * 4).reshape(2, 16, 4, 4)
    y_full = blk(x0)
    (y_full * w_loss).sum().backward()
    g_full = x0.grad.detach().clone()
    full_grads = {name: p.grad.detach().clone() for name, p in blk.named_parameters()}
    assert all(v is not None for v in full_grads.values())

    blk.zero_grad(set_to_none=True)
    # 1 << 8 = 256 states per group; setting _MAX_SIM_STATES = 256 forces BOTH
    # token chunking (max_tokens=1) AND group chunking (max_groups=1) + activation checkpointing
    monkeypatch.setattr(blk, "_MAX_SIM_STATES", 256)
    x1 = x0.detach().clone().requires_grad_(True)
    y_chunk = blk(x1)
    (y_chunk * w_loss).sum().backward()
    chunk_grads = {name: p.grad.detach().clone() for name, p in blk.named_parameters()}

    assert set(chunk_grads.keys()) == set(full_grads.keys())
    torch.testing.assert_close(y_full, y_chunk, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(g_full, x1.grad, atol=1e-6, rtol=1e-5)
    for name in full_grads:
        torch.testing.assert_close(full_grads[name], chunk_grads[name], atol=1e-6, rtol=1e-5)


def test_quantum_parameter_count_invariance_and_bottleneck_param_counts_agreement():
    # At QC=32, L=2, reupload has 4 * L * QC = 256 unique quantum parameters for n_qubits in {2, 4, 8, 16}
    counts_32 = {}
    for nq in (2, 4, 8, 16):
        blk = unet.QuantumBlock(
            backend="pennylane",
            quantum_channels=32,
            seed=42,
            q_ansatz="reupload",
            q_layers=2,
            q_backend="torch",
            n_qubits=nq,
        )
        counts_32[nq] = sum(p.numel() for p in blk.parameters())
        assert blk.num_groups == 32 // nq
        assert blk.effective_qc == 32
        assert blk.remaining_channels == 0
        assert blk.bottleneck_target_params == 256
        assert blk.bottleneck_actual_params == 256
    assert counts_32 == {2: 256, 4: 256, 8: 256, 16: 256}

    # At QC=48, L=2, reupload has 4 * 2 * 48 = 384 unique quantum parameters across all 5 widths
    for nq in (2, 4, 8, 12, 16):
        blk = unet.QuantumBlock(
            backend="pennylane",
            quantum_channels=48,
            seed=42,
            q_ansatz="reupload",
            q_layers=2,
            q_backend="torch",
            n_qubits=nq,
        )
        assert sum(p.numel() for p in blk.parameters()) == 384

    # Verify bottleneck_param_counts(config, ...) without model matches bottleneck_param_counts(config, ..., model)
    # across spatial modes, HQ-UNet grids, and shared/unshared blocks
    base_cfg = {
        "image_size": 32,
        "base_channels": 8,
        "init_dim": 8,
        "out_dim": 2,
        "dim_mults": (1, 2, 4, 8),
        "resnet_block_groups": 4,
        "quantum_channels": 32,
        "quantum_backend": "pennylane",
        "q_backend": "torch",
        "seed": 42,
    }
    test_cfgs = [
        {**base_cfg, "q_ansatz": "reupload", "q_layers": 2, "n_qubits": 8, "q_spatial_mode": "patch2x2", "q_share_blocks": True},
        {**base_cfg, "q_ansatz": "reupload", "q_layers": 2, "n_qubits": 8, "q_spatial_mode": "se_gate", "q_share_blocks": False},
        {**base_cfg, "q_ansatz": "legacy", "q_layers": 1, "n_qubits": 16, "q_spatial_mode": "pixel", "q_share_blocks": True},
        {**base_cfg, "q_ansatz": "legacy", "ablation_mode": "mlp", "n_qubits": 4, "q_share_blocks": True},
        {**base_cfg, "q_ansatz": "legacy", "ablation_mode": "mlp", "n_qubits": 8, "q_share_blocks": True},
        {**base_cfg, "q_ansatz": "legacy", "ablation_mode": "mlp", "n_qubits": 16, "q_share_blocks": True},
        {**base_cfg, "q_ansatz": "legacy", "ablation_mode": "conv1x1", "n_qubits": 8, "q_share_blocks": False},
        {**base_cfg, "q_ansatz": "legacy", "ablation_mode": "lowrank", "n_qubits": 8, "q_share_blocks": True},
        {**base_cfg, "q_ansatz": "hqunet", "q_layers": 1, "hqunet_grid": 4, "q_share_blocks": True},
        {**base_cfg, "q_ansatz": "hqunet", "q_layers": 2, "hqunet_grid": 2, "q_share_blocks": False},
    ]
    for cfg in test_cfgs:
        q_model = tc.build_model(QVUNetSegmentation, cfg)
        c_model = tc.build_model(ClassicalAblationUNetSegmentation, cfg)
        assert tc.bottleneck_param_counts(cfg, "quantum") == tc.bottleneck_param_counts(cfg, "quantum", q_model)
        assert tc.bottleneck_param_counts(cfg, "ablation") == tc.bottleneck_param_counts(cfg, "ablation", c_model)


def test_validation_guards_for_invalid_quantum_channels_and_n_qubits():
    with pytest.raises(ValueError, match="n_qubits"):
        unet.normalize_n_qubits(6)

    with pytest.raises(ValueError, match="n_qubits"):
        unet.normalize_n_qubits(4.9)

    with pytest.raises(ValueError, match="n_qubits"):
        unet.normalize_n_qubits(True)

    with pytest.raises(ValueError, match="quantum_channels"):
        unet.QuantumBlock(quantum_channels=-4, n_qubits=4)

    with pytest.raises(ValueError, match="quantum_channels"):
        unet.QuantumBlock(quantum_channels=8.5, n_qubits=4)

    with pytest.raises(ValueError, match="quantum_channels"):
        unet.ClassicalBottleneckBlock(quantum_channels=8.5, n_qubits=4)

    with pytest.raises(ValueError, match="quantum_channels"):
        unet.QuantumBlock(quantum_channels=8, n_qubits=12)

    with pytest.raises(ValueError, match="quantum_channels"):
        unet.ClassicalBottleneckBlock(quantum_channels=8, n_qubits=12)

    with pytest.raises(ValueError, match="hqunet"):
        unet.QuantumBlock(quantum_channels=32, q_ansatz="hqunet", n_qubits=8)

    with pytest.raises(ValueError, match="ptlayer"):
        unet.QuantumBlock(backend="ptlayer", quantum_channels=32, n_qubits=8)

    # Zero quantum channels is allowed and returns [B, 0, H, W] for both QuantumBlock and ClassicalBottleneckBlock
    q_zero = unet.QuantumBlock(quantum_channels=0, n_qubits=8, q_backend="torch")
    c_zero = unet.ClassicalBottleneckBlock(quantum_channels=0, n_qubits=8)
    x_empty = torch.randn(2, 0, 4, 4)
    assert q_zero(x_empty).shape == (2, 0, 4, 4)
    assert c_zero(x_empty).shape == (2, 0, 4, 4)


@pytest.mark.parametrize("n_qubits", [2, 4, 8, 12, 16])
@pytest.mark.parametrize("q_share_blocks", [True, False])
def test_paired_seed_backbone_parity_and_non_divisible_groupnorm(n_qubits, q_share_blocks):
    # Use quantum_channels=20 so n_qubits in (8, 12, 16) exercises non-divisible remainder handling
    qc = 20
    cfg = {
        "image_size": 32,
        "base_channels": 8,
        "init_dim": 8,
        "out_dim": 2,
        "dim_mults": (1, 2, 4, 8),
        "resnet_block_groups": 4,
        "quantum_channels": qc,
        "quantum_backend": "pennylane",
        "q_backend": "torch",
        "q_ansatz": "reupload",
        "q_layers": 2,
        "n_qubits": n_qubits,
        "q_share_blocks": q_share_blocks,
        "seed": 101,
    }
    q_model = tc.build_model(QVUNetSegmentation, cfg)
    c_model = tc.build_model(ClassicalAblationUNetSegmentation, cfg)

    q_sd = q_model.state_dict()
    c_sd = c_model.state_dict()
    backbone_keys = [k for k in q_sd if "quantum_block" not in k]
    assert len(backbone_keys) > 0
    for k in backbone_keys:
        torch.testing.assert_close(q_sd[k], c_sd[k], atol=0.0, rtol=0.0)

    # Forward + backward smoke check (verifies GroupNorm divisor when classical_channels = 64 - effective_qc)
    x = torch.randn(2, 3, 32, 32)
    out_q = q_model(x)
    out_c = c_model(x)
    assert out_q.shape == (2, 2, 32, 32)
    assert out_c.shape == (2, 2, 32, 32)
    out_q.sum().backward()
    out_c.sum().backward()


@pytest.mark.parametrize("ablation_mode", ["linear", "mlp", "conv1x1", "depthwise3x3", "lowrank"])
def test_classical_bottleneck_modes_and_knockout_at_n_qubits_8(ablation_mode):
    blk = unet.ClassicalBottleneckBlock(
        quantum_channels=16,
        seed=7,
        q_ansatz="reupload",
        q_layers=2,
        ablation_mode=ablation_mode,
        n_qubits=8,
    )
    assert blk.n_qubits == 8
    assert blk.num_groups == 2
    assert blk.effective_qc == 16
    assert blk.bottleneck_target_params == 128
    assert blk.bottleneck_actual_params > 0
    x = torch.randn(2, 16, 4, 4, requires_grad=True)
    y = blk(x)
    assert y.shape == (2, 16, 4, 4)
    y.sum().backward()
    assert x.grad is not None

    blk.set_quantum_mode("zero")
    y_zero = blk(x.detach())
    assert torch.count_nonzero(y_zero).item() == 0
    blk.set_quantum_mode("random", seed=99)
    y_rand = blk(x.detach())
    assert y_rand.shape == (2, 16, 4, 4)


def test_variant_tags_fingerprints_and_resume_backward_compat():
    cfg_default = {"q_ansatz": "reupload", "q_layers": 2, "n_qubits": 4, "seed": 42, "batch_size": 4, "learning_rate": 1e-4}
    cfg_nq8 = {**cfg_default, "n_qubits": 8}
    cfg_leg8 = {**cfg_default, "q_ansatz": "legacy", "n_qubits": 8}

    assert tc.variant_tag(cfg_default) == "_reupload_L2"
    assert tc.variant_tag(cfg_nq8) == "_reupload_L2_nq8"
    assert tc.variant_tag(cfg_leg8) == "_legacy_nq8"

    fp4 = tc.centralized_fingerprint(cfg_default, "quantum")
    fp8 = tc.centralized_fingerprint(cfg_nq8, "quantum")
    assert fp4["n_qubits"] == 4
    assert fp8["n_qubits"] == 8

    # Historical checkpoint without 'n_qubits' normalizes to n_qubits=4 and matches cfg_default,
    # but is rejected when resuming with n_qubits=8.
    legacy_saved = {k: v for k, v in fp4.items() if k != "n_qubits"}
    ckpt = {"run_format": tc.RUN_FORMAT, "q_backend": tc.DEFAULT_Q_BACKEND, "fingerprint": legacy_saved}
    assert tc.checkpoint_matches_centralized(ckpt, cfg_default, "quantum") is True
    assert tc.checkpoint_matches_centralized(ckpt, cfg_nq8, "quantum") is False


def test_circular_aggregation_and_atomic_param_loading_reject_shape_mismatches():
    a = np.zeros((2, 4, 3), dtype=np.float32)
    b = np.zeros((1, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        tc.circular_mean_angles([a, b], [1.0, 1.0])

    with pytest.raises(ValueError, match="weights length mismatch"):
        tc.circular_mean_angles([a, a], [1.0])

    with pytest.raises(ValueError, match="shape mismatch"):
        tc.aggregate_ndarrays_quantum_aware([([a], 10), ([b], 10)], periodic_indices={0}, q_angle_agg="circular")

    cfg4 = {
        "image_size": 32, "base_channels": 8, "init_dim": 8, "out_dim": 2,
        "dim_mults": (1, 2, 4, 8), "resnet_block_groups": 4, "quantum_channels": 16,
        "quantum_backend": "pennylane", "q_backend": "torch", "q_ansatz": "reupload",
        "q_layers": 1, "n_qubits": 4, "seed": 42,
    }
    cfg8 = {**cfg4, "n_qubits": 8}
    m4 = tc.build_model(QVUNetSegmentation, cfg4)
    m8 = tc.build_model(QVUNetSegmentation, cfg8)
    params8 = federated_demoTorchQ.torch_params_to_numpy(m8)

    # Corrupt the last tensor in a valid m4 parameter list to verify atomic loading (earlier params untouched)
    params4_bad_last = federated_demoTorchQ.torch_params_to_numpy(m4)
    for i in range(len(params4_bad_last) - 1):
        params4_bad_last[i] = params4_bad_last[i] + 99.0
    params4_bad_last[-1] = np.zeros((999,), dtype=np.float32)
    orig_first = next(m4.parameters()).detach().clone()

    with pytest.raises(ValueError, match="shape mismatch"):
        federated_demoTorchQ.numpy_to_torch_params(m4, params4_bad_last)
    torch.testing.assert_close(next(m4.parameters()), orig_first, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="shape mismatch"):
        ablation_studies._numpy_to_torch_params(m4, params4_bad_last)
    torch.testing.assert_close(next(m4.parameters()), orig_first, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="shape mismatch"):
        bef._load_checkpoint_into_model(m4, {"parameters": params4_bad_last})
    torch.testing.assert_close(next(m4.parameters()), orig_first, atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="shape mismatch|count mismatch"):
        federated_demoTorchQ.numpy_to_torch_params(m4, params8)

    # Verify apply_circular_q_angle_agg_to_fit_result with {name: shape} spec rejects
    # identically malformed non-periodic tensors in both aggregate and client
    from flwr.common import FitRes, Status, Code, ndarrays_to_parameters
    param_spec = {name: tuple(p.shape) for name, p in m4.named_parameters()}
    malformed = [p.detach().cpu().numpy().copy() for p in m4.parameters()]
    malformed[0] = np.zeros((999,), dtype=np.float32)  # non-periodic backbone tensor
    bad_params = ndarrays_to_parameters(malformed)
    fit_res = FitRes(status=Status(code=Code.OK, message=""), parameters=bad_params, num_examples=10, metrics={})
    with pytest.raises(ValueError, match="shape mismatch"):
        tc.apply_circular_q_angle_agg_to_fit_result(bad_params, [(None, fit_res)], param_spec, {"q_angle_agg": "circular"})


def test_downstream_parsers_generators_and_visualizers_with_n_qubits():
    # analyze_results
    parsed_q = analyze_results._parse_run_dir_full(
        "flood_central_optical_pytorch_pennylane_qc48_reupload_L2_nq12_seed42"
    )
    assert parsed_q == ("Quantum Central QC=48", "optical", 42, 48, "centralized", "centralized", "reupload_L2_nq12")

    parsed_abl = analyze_results._parse_run_dir_full(
        "ablation_centralized_torch_bottleneck_matched_pennylane_qc64_reupload_L2_nq8_abl-mlp_seed42"
    )
    assert parsed_abl == (
        "Ablation Central QC=64",
        "optical",
        42,
        64,
        "centralized",
        "centralized",
        "reupload_L2_nq8_abl-mlp",
    )

    assert (
        analyze_results._variant_from_fields("reupload", 2, {"n_qubits": 8, "q_spatial_mode": "patch2x2"})
        == "reupload_L2_nq8_patch2x2"
    )
    assert (
        analyze_results._infer_variant_from_summary("cli_args: --q-ansatz reupload --q-layers 2 --n-qubits 16")
        == "reupload_L2_nq16"
    )

    # compare_backends RUN_RE groups simulators with matching _nq8 and separates distinct widths
    m_pl = cb.RUN_RE.match("flood_central_optical_pytorch_pennylane_reupload_L2_nq8_seed42")
    m_pg = cb.RUN_RE.match("flood_central_optical_pytorch_pennylane_reupload_L2_nq8_qpergroup_seed42")
    m_qt = cb.RUN_RE.match("flood_central_optical_pytorch_pennylane_reupload_L2_nq8_qtorch_seed42")
    m_nq16 = cb.RUN_RE.match("flood_central_optical_pytorch_pennylane_reupload_L2_nq16_qtorch_seed42")
    m_nq4 = cb.RUN_RE.match("flood_central_optical_pytorch_pennylane_reupload_L2_qtorch_seed42")
    assert all(m is not None for m in (m_pl, m_pg, m_qt, m_nq16, m_nq4))
    assert m_pl.group("variant") == "_reupload_L2_nq8"
    assert m_pg.group("variant") == "_reupload_L2_nq8"
    assert m_qt.group("variant") == "_reupload_L2_nq8"
    assert m_pl.group("btag") is None
    assert m_pg.group("btag") == "_qpergroup"
    assert m_qt.group("btag") == "_qtorch"
    assert m_nq16.group("variant") == "_reupload_L2_nq16"
    assert m_nq4.group("variant") == "_reupload_L2"
    assert len({m_pl.group("variant"), m_nq16.group("variant"), m_nq4.group("variant")}) == 3

    # build_exemplar_figure
    cfg_ex = bef._apply_variant_to_config({}, "reupload_L2_nq8_patch2x2")
    assert cfg_ex["q_ansatz"] == "reupload"
    assert cfg_ex["q_layers"] == 2
    assert cfg_ex["n_qubits"] == 8
    assert cfg_ex["q_spatial_mode"] == "patch2x2"

    resolved = bef._resolve_model_config_for_checkpoint({"fingerprint": {"n_qubits": 16, "quantum_channels": 32}})
    assert resolved["n_qubits"] == 16

    # generate_grid & generate_qc_ablation_scripts
    job = gg.build_job(
        "quantum", "centralized", "optical", 42, gg.REPO_DIR,
        q_ansatz="reupload", q_layers=2, n_qubits=8, quantum_channels=32,
    )
    assert "--n-qubits" in job["args"]
    assert "8" in job["args"]
    assert job["n_qubits"] == 8
    assert "reuploadL2nq8" in job["name"]
    assert "_nq8" in job["job_name"]
    assert gg.expected_hours("quantum", "centralized", "optical", 32, 2, n_qubits=8) > gg.expected_hours(
        "quantum", "centralized", "optical", 32, 2, n_qubits=4
    )
    with pytest.raises(ValueError, match="hqunet"):
        gg.build_job(
            "quantum", "centralized", "optical", 42, gg.REPO_DIR,
            q_ansatz="hqunet", q_layers=1, n_qubits=8, quantum_channels=32,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_qc_dir = os.path.join(tmpdir, "qc_nq12")
        gqc.main(["--qc", "48", "--n-qubits", "12", "--modes", "fed-random", "--datasets", "optical", "--seeds", "42", "--dry-run", "--out-dir", out_qc_dir])

        # visualize_circuit with --n-qubits 8
        out_viz = os.path.join(tmpdir, "plots")
        visualize_circuit.main(["--q-ansatz", "reupload", "--q-layers", "2", "--n-qubits", "8", "--out-dir", out_viz])
        assert os.path.exists(os.path.join(out_viz, "quantum_circuit_reupload_L2_nq8.txt"))
        assert os.path.exists(os.path.join(out_viz, "quantum_circuit_reupload_L2_nq8.png"))
        assert os.path.exists(os.path.join(out_viz, "quantum_circuit_reupload_L2_nq8_expvals.png"))

        with pytest.raises(ValueError, match="compare"):
            visualize_circuit.main(["--q-ansatz", "compare", "--n-qubits", "8", "--out-dir", out_viz])

        # _find_metrics_json_since strict metadata matching
        run_a = os.path.join(tmpdir, "results", "run_a", "results")
        run_b = os.path.join(tmpdir, "results", "run_b", "results")
        os.makedirs(run_a, exist_ok=True)
        os.makedirs(run_b, exist_ok=True)
        path_a = os.path.join(run_a, "metrics.json")
        path_b = os.path.join(run_b, "metrics.json")
        with open(path_a, "w") as f:
            json.dump({"seed": 42, "model_type": "quantum", "q_ansatz": "reupload", "q_layers": 2, "n_qubits": 4, "quantum_channels": 32}, f)
        snap = rlms._snapshot_metrics_files(os.path.join(tmpdir, "results"))
        with open(path_b, "w") as f:
            json.dump({"seed": 42, "model_type": "quantum", "q_ansatz": "reupload", "q_layers": 2, "n_qubits": 8, "quantum_channels": 32}, f)

        argv_nq8 = ["centralized_demoTorchQ.py", "--q-ansatz", "reupload", "--q-layers", "2", "--n-qubits", "8", "--quantum-channels", "32", "--seed", "42"]
        argv_nq16 = ["centralized_demoTorchQ.py", "--q-ansatz", "reupload", "--q-layers", "2", "--n-qubits", "16", "--quantum-channels", "32", "--seed", "42"]
        argv_default_nq4_qc32 = ["centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "reupload", "--q-layers", "2", "--seed", "42"]
        assert rlms._find_metrics_json_since(os.path.join(tmpdir, "results"), snap, expected_seed=42, expected_argv=argv_nq8) == path_b
        with pytest.raises(FileNotFoundError, match="No metrics.json"):
            rlms._find_metrics_json_since(os.path.join(tmpdir, "results"), snap, expected_seed=42, expected_argv=argv_nq16)
        # Default request (n_qubits=4, qc=32) must reject a candidate with n_qubits=8 (or n_qubits=16, qc=16)
        with pytest.raises(FileNotFoundError, match="No metrics.json"):
            rlms._find_metrics_json_since(os.path.join(tmpdir, "results"), snap, expected_seed=42, expected_argv=argv_default_nq4_qc32)

        # Explicit dual_stem + base_channels=8 request must reject a candidate with shared3 + base_channels=32
        snap2 = rlms._snapshot_metrics_files(os.path.join(tmpdir, "results"))
        run_c = os.path.join(tmpdir, "results", "run_c", "results")
        os.makedirs(run_c, exist_ok=True)
        path_c = os.path.join(run_c, "metrics.json")
        with open(path_c, "w") as f:
            json.dump({
                "seed": 42, "model_type": "quantum", "dataset": "combined",
                "q_ansatz": "reupload", "q_layers": 2, "n_qubits": 4, "quantum_channels": 32,
                "combined_input_mode": "shared3", "base_channels": 32,
            }, f)
        argv_dual_bc8 = [
            "centralized_demoTorchQ.py", "--dataset", "combined", "--q-ansatz", "reupload",
            "--q-layers", "2", "--combined-input-mode", "dual_stem", "--base-channels", "8", "--seed", "42",
        ]
        with pytest.raises(FileNotFoundError, match="No metrics.json"):
            rlms._find_metrics_json_since(os.path.join(tmpdir, "results"), snap2, expected_seed=42, expected_argv=argv_dual_bc8)

        # Round 3 B3 regressions:
        # 1) Explicit --q-spatial-mode pixel must reject a candidate that omits q_spatial_mode and config
        argv_pixel = [
            "centralized_demoTorchQ.py", "--q-ansatz", "reupload", "--q-layers", "2",
            "--n-qubits", "8", "--quantum-channels", "32", "--q-spatial-mode", "pixel", "--seed", "42",
        ]
        with pytest.raises(FileNotFoundError, match="No metrics.json"):
            rlms._find_metrics_json_since(os.path.join(tmpdir, "results"), snap, expected_seed=42, expected_argv=argv_pixel)

        # 2) --base-channels 2 --n-qubits 12 with default QC resolves quantum_channels=12 (group-width rounded)
        snap3 = rlms._snapshot_metrics_files(os.path.join(tmpdir, "results"))
        run_d = os.path.join(tmpdir, "results", "run_d", "results")
        os.makedirs(run_d, exist_ok=True)
        path_d = os.path.join(run_d, "metrics.json")
        with open(path_d, "w") as f:
            json.dump({
                "seed": 42, "model_type": "quantum", "dataset": "optical",
                "q_ansatz": "reupload", "q_layers": 2, "n_qubits": 12,
                "quantum_channels": 12, "base_channels": 2,
            }, f)
        argv_bc2_nq12 = [
            "centralized_demoTorchQ.py", "--dataset", "optical", "--q-ansatz", "reupload",
            "--q-layers", "2", "--base-channels", "2", "--n-qubits", "12", "--seed", "42",
        ]
        assert rlms._find_metrics_json_since(os.path.join(tmpdir, "results"), snap3, expected_seed=42, expected_argv=argv_bc2_nq12) == path_d

        # 3) Supported underscore alias --n_qubits 8 matches n_qubits=8
        argv_nq8_alias = ["centralized_demoTorchQ.py", "--q_ansatz", "reupload", "--q_layers", "2", "--n_qubits", "8", "--quantum_channels", "32", "--seed", "42"]
        assert rlms._find_metrics_json_since(os.path.join(tmpdir, "results"), snap, expected_seed=42, expected_argv=argv_nq8_alias) == path_b

    # bench_quantum
    b_args = bq.make_block("torch", None, "reupload", 2, quantum_channels=16, n_qubits=8)
    assert b_args.n_qubits == 8
    assert b_args.num_groups == 2
    with pytest.raises(ValueError, match="hqunet"):
        bq.main(["--q-ansatz", "hqunet", "--n-qubits", "8", "--skip-block", "--skip-model"])

    # run_local_mini_sweep rejects invalid panel 5, accepts panel 4
    with pytest.raises(ValueError, match="panels_filter"):
        rlms.run_mini_sweep(seeds=[42], panels_filter=[5])
