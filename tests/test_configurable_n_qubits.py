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


@pytest.mark.parametrize("n_qubits", [2, 4, 8])
@pytest.mark.parametrize("q_ansatz", ["legacy", "reupload"])
def test_simulator_parity_float64_and_float32(n_qubits, q_ansatz):
    q_layers = 1 if q_ansatz == "legacy" else 2
    qc = 2 * n_qubits  # 2 groups
    B, H, W = 2, 2, 2
    torch.manual_seed(123)
    x0 = torch.randn(B, qc, H, W, dtype=torch.float64)

    blocks = {}
    outs = {}
    x_grads = {}
    param_grads = {}
    for qb in ("pennylane", "pennylane_pergroup", "torch"):
        blk = unet.QuantumBlock(
            backend="pennylane",
            quantum_channels=qc,
            seed=77,
            q_ansatz=q_ansatz,
            q_layers=q_layers,
            q_backend=qb,
            n_qubits=n_qubits,
        ).double()
        if q_ansatz == "reupload":
            with torch.no_grad():
                for s in blk.input_scales:
                    s.copy_(torch.linspace(0.7, 1.3, q_layers * n_qubits, dtype=torch.float64).reshape(q_layers, n_qubits))
        x = x0.clone().requires_grad_(True)
        y = blk(x)
        loss = (y * torch.arange(1, y.numel() + 1, dtype=torch.float64).reshape_as(y)).sum()
        loss.backward()
        blocks[qb] = blk
        outs[qb] = y.detach()
        x_grads[qb] = x.grad.detach()
        param_grads[qb] = [p.grad.detach().clone() for p in blk.parameters() if p.grad is not None]

    for other in ("pennylane_pergroup", "torch"):
        torch.testing.assert_close(outs["pennylane"], outs[other], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(x_grads["pennylane"], x_grads[other], atol=1e-6, rtol=1e-5)
        for g_ref, g_oth in zip(param_grads["pennylane"], param_grads[other]):
            torch.testing.assert_close(g_ref, g_oth, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("n_qubits", [12, 16])
def test_wide_qubit_torch_and_pennylane_parity(n_qubits):
    qc = n_qubits  # 1 group of 12 or 16 qubits
    x0 = torch.randn(1, qc, 2, 2, dtype=torch.float32)
    blk_torch = unet.QuantumBlock(
        backend="pennylane",
        quantum_channels=qc,
        seed=19,
        q_ansatz="reupload",
        q_layers=1,
        q_backend="torch",
        n_qubits=n_qubits,
    )
    blk_pl = unet.QuantumBlock(
        backend="pennylane",
        quantum_channels=qc,
        seed=19,
        q_ansatz="reupload",
        q_layers=1,
        q_backend="pennylane",
        n_qubits=n_qubits,
    )
    xt = x0.clone().requires_grad_(True)
    xp = x0.clone().requires_grad_(True)
    yt = blk_torch(xt)
    yp = blk_pl(xp)
    torch.testing.assert_close(yt, yp, atol=1e-5, rtol=1e-4)
    yt.sum().backward()
    yp.sum().backward()
    torch.testing.assert_close(xt.grad, xp.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        blk_torch.quantum_params[0].grad,
        blk_pl.quantum_params[0].grad,
        atol=1e-5,
        rtol=1e-4,
    )


def test_chunked_token_simulation_matches_unchunked(monkeypatch):
    blk = unet.QuantumBlock(
        backend="pennylane",
        quantum_channels=8,
        seed=11,
        q_ansatz="reupload",
        q_layers=2,
        q_backend="torch",
        q_spatial_mode="patch2x2",
        n_qubits=4,
    )
    x0 = torch.randn(2, 8, 4, 4, requires_grad=True)
    y_full = blk(x0)
    y_full.sum().backward()
    g_full = x0.grad.detach().clone()
    p_full = blk.quantum_params[0].grad.detach().clone()

    blk.zero_grad(set_to_none=True)
    monkeypatch.setattr(blk, "_MAX_SIM_STATES", 32)  # force token chunking + checkpointing
    x1 = x0.detach().clone().requires_grad_(True)
    y_chunk = blk(x1)
    y_chunk.sum().backward()
    torch.testing.assert_close(y_full, y_chunk, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(g_full, x1.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(p_full, blk.quantum_params[0].grad, atol=1e-6, rtol=1e-5)


def test_quantum_parameter_count_invariance_across_n_qubits():
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


def test_validation_guards_for_invalid_quantum_channels_and_n_qubits():
    with pytest.raises(ValueError, match="n_qubits"):
        unet.normalize_n_qubits(6)

    with pytest.raises(ValueError, match="quantum_channels"):
        unet.QuantumBlock(quantum_channels=-4, n_qubits=4)

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


def test_circular_aggregation_and_param_loading_reject_shape_mismatches():
    a = np.zeros((2, 4, 3), dtype=np.float32)
    b = np.zeros((1, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        tc.circular_mean_angles([a, b], [1.0, 1.0])

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

    with pytest.raises(ValueError, match="shape mismatch|count mismatch"):
        federated_demoTorchQ.numpy_to_torch_params(m4, params8)

    with pytest.raises(ValueError, match="shape mismatch|count mismatch"):
        ablation_studies._numpy_to_torch_params(m4, params8)

    with pytest.raises(ValueError, match="shape mismatch|count mismatch"):
        bef._load_checkpoint_into_model(m4, {"parameters": params8})


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

    # compare_backends RUN_RE groups simulators with matching _nq8
    m_pl = cb.RUN_RE.match("flood_central_optical_pytorch_pennylane_reupload_L2_nq8_seed42")
    m_qt = cb.RUN_RE.match("flood_central_optical_pytorch_pennylane_reupload_L2_nq8_qtorch_seed42")
    assert m_pl is not None and m_qt is not None
    assert m_pl.group(1) == m_qt.group(1)
    assert m_pl.group(3) == m_qt.group(3)

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
    assert "reuploadL2nq8" in job["name"]
    assert "_nq8" in job["job_name"]
    assert gg.expected_hours("quantum", "centralized", "optical", 32, 2, n_qubits=8) > gg.expected_hours(
        "quantum", "centralized", "optical", 32, 2, n_qubits=4
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

    # bench_quantum
    b_args = bq.make_block("torch", None, "reupload", 2, quantum_channels=16, n_qubits=8)
    assert b_args.n_qubits == 8
    assert b_args.num_groups == 2

    # run_local_mini_sweep rejects invalid panel 5, accepts panel 4
    with pytest.raises(ValueError, match="panels_filter"):
        rlms.run_mini_sweep(seeds=[42], panels_filter=[5])

