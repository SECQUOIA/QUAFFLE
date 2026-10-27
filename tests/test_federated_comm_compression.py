"""Regression tests for Issue #48: Federated update compression baselines
(--comm-compression {none,fp16,int8,topk10}) and byte-size telemetry."""

import argparse
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import training_common as tc
from utilsTorch.utilsTorch import (
    ClassicalAblationUNetSegmentation,
    QVUNetSegmentation,
    UNetSegmentation,
)


def _base_fed_config(**overrides):
    cfg = {
        "image_size": 32,
        "base_channels": 4,
        "init_dim": 4,
        "out_dim": 2,
        "dim_mults": (1, 2, 4, 8),
        "resnet_block_groups": 4,
        "quantum_channels": 8,
        "quantum_backend": "pennylane",
        "q_ansatz": "reupload",
        "q_layers": 2,
        "q_backend": "torch",
        "q_sim_device": "cpu",
        "batch_size": 2,
        "learning_rate": 1e-3,
        "num_clients": 2,
        "clients_per_round": 2,
        "fed_rounds": 2,
        "local_epochs": 1,
        "partition": "random",
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


def test_table3_byte_size_accounting_across_all_modes():
    """Verify exact byte-size and MiB accounting for the 9,327,970-parameter classical U-Net
    across 'none', 'fp16', 'int8', and 'topk10' (#48, Table 3)."""
    n_params = 9_327_970

    t_none = tc.compute_comm_compression_telemetry(n_params, "none")
    assert t_none["telemetry_is_exact_per_tensor"] is False
    assert t_none["uncompressed_bytes_per_client_round"] == 37_311_880
    assert t_none["upload_bytes_per_client_round"] == 37_311_880
    assert round(t_none["upload_mib_per_client_round"], 2) == 35.58
    assert t_none["compression_ratio"] == pytest.approx(1.0)
    assert t_none["reduction_fraction"] == pytest.approx(0.0)

    t_fp16 = tc.compute_comm_compression_telemetry(n_params, "fp16")
    assert t_fp16["telemetry_is_exact_per_tensor"] is False
    assert t_fp16["upload_bytes_per_client_round"] == 18_655_940
    assert round(t_fp16["upload_mib_per_client_round"], 2) == 17.79
    assert t_fp16["compression_ratio"] == pytest.approx(0.5)
    assert t_fp16["reduction_fraction"] == pytest.approx(0.5)

    t_int8 = tc.compute_comm_compression_telemetry(n_params, "int8")
    assert t_int8["telemetry_is_exact_per_tensor"] is False
    assert t_int8["upload_bytes_per_client_round"] == 9_327_970
    assert round(t_int8["upload_mib_per_client_round"], 2) == 8.90
    assert t_int8["compression_ratio"] == pytest.approx(0.25)
    assert t_int8["reduction_fraction"] == pytest.approx(0.75)

    t_topk = tc.compute_comm_compression_telemetry(n_params, "topk10")
    assert t_topk["telemetry_is_exact_per_tensor"] is False
    assert t_topk["retained_elements_per_client_round"] == 932_797
    assert t_topk["upload_bytes_per_client_round"] == 8 * 932_797  # 7,462,376
    assert round(t_topk["upload_mib_per_client_round"], 2) == 7.12
    assert t_topk["compression_ratio"] == pytest.approx(0.20)
    assert t_topk["reduction_fraction"] == pytest.approx(0.80)
    assert t_topk["upload_bytes_per_client_round_int16_idx"] == 6 * 932_797  # 5,596,782
    assert round(t_topk["upload_mib_per_client_round_int16_idx"], 2) == 5.34

    # Tensor-list accounting on a live UNetSegmentation model includes per-tensor scale metadata for int8
    cfg_small = _base_fed_config(base_channels=4, init_dim=4, seed=None)
    m_small = tc.build_model(UNetSegmentation, cfg_small)
    n_tensors = len(list(m_small.parameters()))
    n_small_params = sum(p.numel() for p in m_small.parameters())
    t_model_int8 = tc.compute_comm_compression_telemetry(m_small, "int8")
    assert t_model_int8["telemetry_is_exact_per_tensor"] is True
    assert t_model_int8["uncompressed_bytes_per_client_round"] == 4 * n_small_params
    assert t_model_int8["scale_metadata_bytes_per_client_round"] == 4 * n_tensors
    assert t_model_int8["upload_bytes_per_client_round"] == n_small_params + 4 * n_tensors


def test_tensor_and_client_update_compression_mathematics():
    """Verify mathematical accuracy and edge cases for 'none', 'fp16', 'int8', and 'topk10'."""
    with pytest.raises(ValueError, match="comm_compression"):
        tc.normalize_comm_compression("unknown_mode")

    rng = np.random.default_rng(123)
    global_w = rng.normal(0.0, 1.0, size=(10, 10)).astype(np.float32)
    delta = np.linspace(-0.5, 0.5, 100, dtype=np.float32).reshape(10, 10)
    client_w = global_w + delta

    # 1. none: exact identity
    out_none = tc.compress_decompress_tensor_update(client_w, global_w, "none")
    assert np.array_equal(out_none, client_w)

    # 2. fp16: matches float16 cast of delta added to global_w
    out_fp16 = tc.compress_decompress_tensor_update(client_w, global_w, "fp16")
    expected_fp16 = global_w + delta.astype(np.float16).astype(np.float32)
    assert np.allclose(out_fp16, expected_fp16, atol=1e-7)

    # 3. int8: symmetric absmax quantization with max error <= amax / 254
    out_int8 = tc.compress_decompress_tensor_update(client_w, global_w, "int8")
    recon_delta_int8 = out_int8 - global_w
    amax = float(np.max(np.abs(delta)))
    assert np.max(np.abs(recon_delta_int8 - delta)) <= (amax / 127.0) * 0.5 + 1e-6
    # All-zero delta edge case in int8
    out_int8_zero = tc.compress_decompress_tensor_update(global_w, global_w, "int8")
    assert np.array_equal(out_int8_zero, global_w)

    # 4. topk10: retains exactly ceil(0.10 * 100) = 10 largest-magnitude entries of delta
    out_topk = tc.compress_decompress_tensor_update(client_w, global_w, "topk10")
    recon_delta_topk = out_topk - global_w
    nonzero_mask = recon_delta_topk != 0.0
    assert int(np.count_nonzero(nonzero_mask)) == 10
    top10_threshold = np.sort(np.abs(delta.reshape(-1)))[-10]
    assert np.all(np.abs(delta[nonzero_mask]) >= top10_threshold - 1e-7)
    assert np.allclose(recon_delta_topk[nonzero_mask], delta[nonzero_mask], atol=1e-7)

    # Tie-breaking determinism in topk10 (e.g., constant non-zero delta)
    const_delta = np.ones((20,), dtype=np.float32) * 0.25
    out_ties = tc.compress_decompress_tensor_update(const_delta, np.zeros_like(const_delta), "topk10")
    assert int(np.count_nonzero(out_ties)) == 2
    assert np.all(out_ties[:2] == 0.25) and np.all(out_ties[2:] == 0.0)

    # Non-floating-point tensors pass through untouched
    int_arr = np.array([1, 2, 3], dtype=np.int64)
    out_list = tc.compress_decompress_client_update([client_w, int_arr], [global_w, int_arr], "int8")
    assert np.array_equal(out_list[1], int_arr)

    # Non-finite (NaN / Inf) values must raise ValueError across all compressed modes (B1)
    for bad_val in (np.nan, np.inf, -np.inf):
        bad_client = client_w.copy()
        bad_client[0, 0] = bad_val
        bad_global = global_w.copy()
        bad_global[0, 0] = bad_val
        for mode in ("fp16", "int8", "topk10"):
            with pytest.raises(ValueError, match="Non-finite"):
                tc.compress_decompress_tensor_update(bad_client, global_w, mode)
            with pytest.raises(ValueError, match="Non-finite"):
                tc.compress_decompress_tensor_update(client_w, bad_global, mode)
            with pytest.raises(ValueError, match="Non-finite"):
                tc.compress_decompress_tensor_update(bad_client, None, mode)


def test_flower_strategy_round_trip_aggregation_all_modes(tmp_path):
    """Verify RoundOffsetMixin / ClassicalFedAvg, QuantumFedAvg (including circular angle agg),
    and _TorchAblationFedAvg apply --comm-compression during aggregate_fit, including
    multi-round configure_fit baseline refresh and weighted circular aggregation across ±pi."""
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.client_proxy import ClientProxy

    import ablation_studies
    import federated_demoTorchClassical
    import federated_demoTorchQ

    class _DummyProxy(ClientProxy):
        def __init__(self, cid):
            super().__init__(str(cid))

        def get_properties(self, ins, timeout, group_id):
            raise NotImplementedError

        def get_parameters(self, ins, timeout, group_id):
            raise NotImplementedError

        def fit(self, ins, timeout, group_id):
            raise NotImplementedError

        def evaluate(self, ins, timeout, group_id):
            raise NotImplementedError

        def reconnect(self, ins, timeout, group_id):
            raise NotImplementedError

    cm = SimpleClientManager()
    cm.register(_DummyProxy("0"))
    cm.register(_DummyProxy("1"))

    # 1. ClassicalFedAvg 2-round configure_fit -> aggregate_fit across all 4 modes
    cfg_cls = _base_fed_config(seed=42)
    cls_model = tc.build_model(UNetSegmentation, cfg_cls)
    global_nd = [p.detach().cpu().numpy().copy() for p in cls_model.parameters()]
    global_params = ndarrays_to_parameters(global_nd)

    rng = np.random.default_rng(99)
    c1_nd = [arr + rng.normal(0.0, 0.2, size=arr.shape).astype(arr.dtype) for arr in global_nd]
    c2_nd = [arr + rng.normal(0.0, 0.2, size=arr.shape).astype(arr.dtype) for arr in global_nd]

    def _make_results(nd1, nd2, n1=10, n2=10):
        return [
            (
                _DummyProxy("0"),
                FitRes(
                    status=Status(code=Code.OK, message=""),
                    parameters=ndarrays_to_parameters(nd1),
                    num_examples=int(n1),
                    metrics={},
                ),
            ),
            (
                _DummyProxy("1"),
                FitRes(
                    status=Status(code=Code.OK, message=""),
                    parameters=ndarrays_to_parameters(nd2),
                    num_examples=int(n2),
                    metrics={},
                ),
            ),
        ]

    agg_by_mode = {}
    for mode in ("none", "fp16", "int8", "topk10"):
        cfg_m = _base_fed_config(comm_compression=mode)
        strat = federated_demoTorchClassical.ClassicalFedAvg(
            val_loader=None,
            config=cfg_m,
            device_str="cpu",
            output_dir=str(tmp_path / f"cls_{mode}"),
            initial_parameters=global_params,
        )
        # Round 1: configure_fit populates _round_start_parameters automatically
        fit_ins_r1 = strat.configure_fit(1, global_params, cm)
        assert len(fit_ins_r1) == 2
        assert strat._round_start_parameters is global_params
        res_in = _make_results(c1_nd, c2_nd)
        agg_params_r1, _ = strat.aggregate_fit(1, res_in, [])
        agg_nd_r1 = parameters_to_ndarrays(agg_params_r1)
        agg_by_mode[mode] = agg_nd_r1

        # Verify Round 1 strategy aggregation matches manual average of compressed client updates
        exp_c1 = tc.compress_decompress_client_update(c1_nd, global_nd, mode)
        exp_c2 = tc.compress_decompress_client_update(c2_nd, global_nd, mode)
        for a, e1, e2 in zip(agg_nd_r1, exp_c1, exp_c2):
            assert np.allclose(a, 0.5 * (e1 + e2), atol=1e-6)

        # Round 2: configure_fit refreshes _round_start_parameters to agg_params_r1
        strat.configure_fit(2, agg_params_r1, cm)
        assert strat._round_start_parameters is agg_params_r1
        c1_r2_nd = [arr + np.float32(0.15) for arr in agg_nd_r1]
        c2_r2_nd = [arr - np.float32(0.05) for arr in agg_nd_r1]
        agg_params_r2, _ = strat.aggregate_fit(2, _make_results(c1_r2_nd, c2_r2_nd), [])
        agg_nd_r2 = parameters_to_ndarrays(agg_params_r2)
        exp_c1_r2 = tc.compress_decompress_client_update(c1_r2_nd, agg_nd_r1, mode)
        exp_c2_r2 = tc.compress_decompress_client_update(c2_r2_nd, agg_nd_r1, mode)
        for a2, e1_r2, e2_r2 in zip(agg_nd_r2, exp_c1_r2, exp_c2_r2):
            assert np.allclose(a2, 0.5 * (e1_r2 + e2_r2), atol=1e-6)

    assert not tc.ndarrays_equal(agg_by_mode["none"], agg_by_mode["topk10"])
    assert not tc.ndarrays_equal(agg_by_mode["none"], agg_by_mode["int8"])

    # 2. QuantumFedAvg with circular angle aggregation + topk10 compression across ±pi with unequal weights
    cfg_q = _base_fed_config(comm_compression="topk10", q_angle_agg="circular")
    q_model = tc.build_model(QVUNetSegmentation, cfg_q)
    q_global_nd = [p.detach().cpu().numpy().copy() for p in q_model.parameters()]
    q_strat = federated_demoTorchQ.QuantumFedAvg(
        val_loader=None,
        config=cfg_q,
        device_str="cpu",
        output_dir=str(tmp_path / "q_topk10"),
        initial_parameters=ndarrays_to_parameters(q_global_nd),
    )
    q_strat.template_model = q_model
    param_keys = q_strat._ensure_param_keys()
    periodic_idx = tc.quantum_periodic_param_indices(param_keys)
    assert len(periodic_idx) > 0

    # Set periodic quantum parameters to 0.0 at round start and craft client updates straddling ±pi
    for idx in periodic_idx:
        q_global_nd[idx] = np.zeros_like(q_global_nd[idx], dtype=np.float32)
    q_global_params = ndarrays_to_parameters(q_global_nd)

    qc1_nd = []
    qc2_nd = []
    for idx, arr in enumerate(q_global_nd):
        if idx in periodic_idx:
            # Make entry 0 dominant so topk10 deterministically retains it on both clients,
            # with client 1 near +pi (pi - 0.1) and client 2 near -pi (-pi + 0.2)
            u1 = np.zeros_like(arr, dtype=np.float32).reshape(-1)
            u2 = np.zeros_like(arr, dtype=np.float32).reshape(-1)
            u1[0] = np.float32(np.pi - 0.1)
            u2[0] = np.float32(-np.pi + 0.2)
            qc1_nd.append(u1.reshape(arr.shape))
            qc2_nd.append(u2.reshape(arr.shape))
        else:
            qc1_nd.append(arr + rng.normal(0.0, 0.3, size=arr.shape).astype(arr.dtype))
            qc2_nd.append(arr + rng.normal(0.0, 0.3, size=arr.shape).astype(arr.dtype))

    q_strat.configure_fit(1, q_global_params, cm)
    q_agg_params, _ = q_strat.aggregate_fit(1, _make_results(qc1_nd, qc2_nd, n1=10, n2=30), [])
    q_agg_nd = parameters_to_ndarrays(q_agg_params)
    assert len(q_agg_nd) == len(q_global_nd)

    exp_qc1 = tc.compress_decompress_client_update(qc1_nd, q_global_nd, "topk10")
    exp_qc2 = tc.compress_decompress_client_update(qc2_nd, q_global_nd, "topk10")
    for idx, (agg_arr, e1, e2) in enumerate(zip(q_agg_nd, exp_qc1, exp_qc2)):
        euclid_weighted = (10.0 * e1 + 30.0 * e2) / 40.0
        if idx in periodic_idx:
            circ_weighted = tc.circular_mean_angles([e1, e2], [10.0, 30.0])
            assert np.allclose(agg_arr, circ_weighted, atol=1e-6)
            # Entry 0 straddles ±pi, so circular mean (~ -pi + 0.125) differs sharply from Euclidean mean (~ -1.52)
            assert abs(float(agg_arr.reshape(-1)[0]) - float(euclid_weighted.reshape(-1)[0])) > 1.0
        elif np.issubdtype(agg_arr.dtype, np.floating):
            assert np.allclose(agg_arr, euclid_weighted, atol=1e-6)

    # 3. _TorchAblationFedAvg with int8 compression via configure_fit
    cfg_abl = _base_fed_config(comm_compression="int8", ablation_mode="mlp")
    abl_model = tc.build_model(ClassicalAblationUNetSegmentation, cfg_abl)
    abl_global_nd = [p.detach().cpu().numpy().copy() for p in abl_model.parameters()]
    abl_global_params = ndarrays_to_parameters(abl_global_nd)
    ac1_nd = [arr + rng.normal(0.0, 0.2, size=arr.shape).astype(arr.dtype) for arr in abl_global_nd]
    ac2_nd = [arr + rng.normal(0.0, 0.2, size=arr.shape).astype(arr.dtype) for arr in abl_global_nd]

    abl_strat = ablation_studies._TorchAblationFedAvg(
        val_loader=None,
        config=cfg_abl,
        device_str="cpu",
        output_dir=str(tmp_path / "abl_int8"),
        initial_parameters=abl_global_params,
    )
    abl_strat.configure_fit(1, abl_global_params, cm)
    abl_agg_params, _ = abl_strat.aggregate_fit(1, _make_results(ac1_nd, ac2_nd), [])
    abl_agg_nd = parameters_to_ndarrays(abl_agg_params)
    exp_ac1 = tc.compress_decompress_client_update(ac1_nd, abl_global_nd, "int8")
    exp_ac2 = tc.compress_decompress_client_update(ac2_nd, abl_global_nd, "int8")
    for a, e1, e2 in zip(abl_agg_nd, exp_ac1, exp_ac2):
        assert np.allclose(a, 0.5 * (e1 + e2), atol=1e-6)


def test_comm_compression_cli_fingerprints_metrics_and_parsers(tmp_path):
    """Verify --comm-compression through CLI, federated_variant_tag, fingerprints,
    write_metrics_json, generate_grid.py, analyze_results.py, and run_local_mini_sweep.py."""
    import analyze_results
    from misc import generate_grid
    from scripts import run_local_mini_sweep as rlms

    parser = argparse.ArgumentParser()
    tc.add_common_args(parser, quantum=True, federated=True)
    args = parser.parse_args(["--comm-compression", "topk10", "--q-angle-agg", "circular"])
    cfg = _base_fed_config(num_clients=4, clients_per_round=4)
    tc.apply_common_args(cfg, args)
    assert cfg["comm_compression"] == "topk10"
    assert tc.federated_variant_tag(cfg, quantum=True) == "_qagg-circular_comm-topk10"
    assert tc.federated_variant_tag(cfg, quantum=False) == "_comm-topk10"

    # Fingerprints and legacy checkpoint compatibility
    fp_none = tc.run_fingerprint(_base_fed_config(comm_compression="none"), "classical")
    fp_int8 = tc.run_fingerprint(_base_fed_config(comm_compression="int8"), "classical")
    assert fp_none["comm_compression"] == "none"
    assert fp_int8["comm_compression"] == "int8"
    assert fp_none != fp_int8
    legacy_saved = {k: v for k, v in fp_none.items() if k != "comm_compression"}
    assert tc._normalize_saved_fingerprint(legacy_saved, "classical") == fp_none

    # write_metrics_json records upload_bytes_per_client_round and compression_ratio
    run_dir = str(tmp_path / "fed_run")
    val_m = {"loss": 0.3, "accuracy": 0.9, "auc": 0.95, "iou": 0.8, "dice": 0.88}
    m_path = tc.write_metrics_json(
        run_dir,
        script="federated_demoTorchClassical.py",
        training="federated",
        model_type="classical",
        config=_base_fed_config(comm_compression="fp16"),
        val_metrics=val_m,
        test_metrics=val_m,
        total_params=9_327_970,
    )
    with open(m_path) as f:
        m_data = json.load(f)
    assert m_data["comm_compression"] == "fp16"
    assert m_data["uncompressed_bytes_per_client_round"] == 37_311_880
    assert m_data["upload_bytes_per_client_round"] == 18_655_940
    assert round(m_data["upload_mib_per_client_round"], 2) == 17.79
    assert m_data["compression_ratio"] == pytest.approx(0.5)

    # analyze_results
    v_cls = analyze_results._variant_from_fields(
        None, None, data={"training": "federated", "comm_compression": "int8"}
    )
    assert v_cls == "comm-int8"
    parsed_cls = analyze_results._parse_run_dir_full(
        "flood_federated_optical_pytorch_classical_random_comm-int8_seed42"
    )
    assert parsed_cls is not None
    assert parsed_cls[-1] == "comm-int8"

    parsed_q = analyze_results._parse_run_dir_full(
        "flood_federated_optical_pytorch_pennylane_random_reupload_L2_qagg-circular_comm-topk10_seed42"
    )
    assert parsed_q is not None
    assert parsed_q[-1] == "reupload_L2_qagg-circular_comm-topk10"

    inferred = analyze_results._infer_variant_from_summary(
        "cli_args: --partition random --comm-compression fp16",
        quantumish=False,
    )
    assert inferred == "comm-fp16"

    # generate_grid
    job = generate_grid.build_job(
        "classical",
        "fed-random",
        "optical",
        42,
        REPO_ROOT,
        comm_compression="topk10",
    )
    assert "--comm-compression topk10" in job["command"]
    assert "comm-topk10" in job["name"]

    # run_local_mini_sweep (explicit and default federated comm_compression validation, N3)
    meta, explicit = rlms._expected_metadata_from_argv(
        [
            "federated_demoTorchClassical.py",
            "--seed",
            "42",
            "--comm-compression",
            "int8",
        ]
    )
    assert meta["comm_compression"] == "int8"
    assert "comm_compression" in explicit

    meta_def, explicit_def = rlms._expected_metadata_from_argv(
        ["federated_demoTorchClassical.py", "--seed", "42"]
    )
    assert meta_def["comm_compression"] == "none"
    assert "comm_compression" not in explicit_def
    cand_base = dict(meta_def, seed=42)
    assert rlms._matches_expected_metrics(dict(cand_base, comm_compression="none"), 42, meta_def, explicit_def)
    assert not rlms._matches_expected_metrics(dict(cand_base, comm_compression="topk10"), 42, meta_def, explicit_def)

    meta_cent, _ = rlms._expected_metadata_from_argv(
        ["ablation_studies.py", "--mode", "centralized", "--seed", "42"]
    )
    assert "comm_compression" not in meta_cent


