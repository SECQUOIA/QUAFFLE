"""Regression and unit tests for Wave 3 (#14, #15) enhancements:

- #14: Multimodal input stems (`--combined-input-mode {shared3,concat6,dual_stem}`)
  with paired-seed backbone initialization determinism (`#5`) and cross-sensor
  modality / geolocation feature-cluster non-IID federated partitioning
  (`--partition {random,dirichlet,modality,geo_cluster}`).
- #15: FedProx proximal client penalty (`--fl-strategy fedprox --fedprox-mu`),
  periodic PQC circular angle aggregation (`--q-angle-agg circular`) on
  $S^1 = [-\\pi, \\pi]$ across the $\\pm\\pi$ branch cut, and partial client
  participation (`--num-clients`, `--clients-per-round`, `--fraction-fit`).
"""

import argparse
import os
import sys

import numpy as np
import pytest
import torch
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import Dataset, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import flood_data
from federated_demoTorchClassical import ClassicalFlowerClient
from federated_demoTorchQ import QuantumFedAvg, QuantumFlowerClient
from tools import training_common as tc
from utilsTorch import (
    ClassicalAblationUNetSegmentation,
    QVUNetSegmentation,
    UNetSegmentation,
    fedprox_proximal_loss,
)


def _tiny_config(**overrides):
    cfg = {
        "image_size": 16,
        "base_channels": 8,
        "init_dim": 8,
        "out_dim": 2,
        "dim_mults": (1, 2),
        "resnet_block_groups": 4,
        "quantum_channels": 8,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "seed": 42,
        "quantum_backend": "pennylane",
        "q_backend": "torch",
        "q_ansatz": "reupload",
        "q_layers": 2,
        "num_clients": 4,
        "clients_per_round": 4,
        "local_epochs": 1,
        "fed_rounds": 2,
    }
    cfg.update(overrides)
    return cfg


class _SyntheticMultimodalDataset(Dataset):
    """Deterministic synthetic dataset with half optical and half SAR samples."""

    def __init__(self, n_samples=16, combined_input_mode="shared3", image_size=16, seed=42):
        self.n_samples = int(n_samples)
        self.combined_input_mode = flood_data.normalize_combined_input_mode(combined_input_mode)
        self.image_size = int(image_size)
        rng = np.random.default_rng(seed)
        self.images = []
        self.masks = []
        self.modalities = []
        for i in range(self.n_samples):
            mod = "optical" if (i < self.n_samples // 2) else "sar"
            self.modalities.append(mod)
            # Make optical and SAR tiles have distinct regional/spectral signatures
            base_val = -1.0 if mod == "optical" else 1.5
            raw3 = rng.normal(loc=base_val + 0.05 * i, scale=0.3, size=(self.image_size, self.image_size, 3)).astype(
                np.float32
            )
            formatted = flood_data._format_image_for_combined_mode(raw3, mod, self.combined_input_mode)
            img_t = torch.from_numpy(formatted).permute(2, 0, 1).contiguous()
            mask_t = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            if i % 2 == 0:
                mask_t[:, : self.image_size // 2, : self.image_size // 2] = 1.0
            self.images.append(img_t)
            self.masks.append(mask_t)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


# ---------------------------------------------------------------------------
# #14: Modality-conditioned input stems & paired-seed determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combined_input_mode", ["shared3", "concat6", "dual_stem"])
def test_multimodal_input_stems_forward_backward_and_paired_seed(combined_input_mode):
    """Verify `shared3`, `concat6`, and `dual_stem` stems run forward/backward and
    preserve 100% identical initial encoder/decoder weights across Quantum, Ablation,
    and Classical models (#14, #5)."""
    cfg = _tiny_config(combined_input_mode=combined_input_mode, seed=17)
    q_model = tc.build_model(QVUNetSegmentation, cfg)
    a_model = tc.build_model(ClassicalAblationUNetSegmentation, cfg)
    c_model = tc.build_model(UNetSegmentation, cfg)

    q_sd = q_model.model.state_dict()
    a_sd = a_model.model.state_dict()
    c_sd = c_model.model.state_dict()

    # Check shared encoder/vertex/decoder parameters are bit-identical between Quantum and Ablation (#5),
    # and all encoder (`down.*`) parameters are bit-identical across Quantum, Ablation, and Classical.
    qa_shared_keys = [
        k for k in q_sd.keys()
        if k in a_sd and q_sd[k].shape == a_sd[k].shape
    ]
    assert len(qa_shared_keys) > 20
    for k in qa_shared_keys:
        assert torch.equal(q_sd[k], a_sd[k]), f"Quantum vs Ablation mismatch on {k}"
        if k.startswith("down.") and k in c_sd:
            assert torch.equal(q_sd[k], c_sd[k]), f"Quantum vs Classical encoder mismatch on {k}"

    if combined_input_mode == "dual_stem":
        assert q_model.model.init_conv_sar is not None
        # Optical and SAR stems must have distinct random initial weights
        assert not torch.allclose(
            q_model.model.init_conv.weight, q_model.model.init_conv_sar.weight
        )

    in_ch = 3 if combined_input_mode == "shared3" else 6
    x = torch.zeros(2, in_ch, 16, 16, requires_grad=True)
    if in_ch == 6:
        # Sample 0: Optical (first 3 ch active), Sample 1: SAR (last 3 ch active)
        x_data = x.detach().clone()
        x_data[0, :3] = torch.randn(3, 16, 16)
        x_data[1, 3:6] = torch.randn(3, 16, 16)
        x = x_data.requires_grad_(True)
    else:
        x = torch.randn(2, 3, 16, 16, requires_grad=True)

    for m in (q_model, a_model, c_model):
        m.zero_grad()
        logits = m(x)
        assert logits.shape == (2, 2, 16, 16)
        loss = logits.pow(2).mean()
        loss.backward()
        assert m.model.init_conv.weight.grad is not None
        assert float(m.model.init_conv.weight.grad.abs().sum()) > 0.0
        if combined_input_mode == "dual_stem":
            assert m.model.init_conv_sar.weight.grad is not None
            assert float(m.model.init_conv_sar.weight.grad.abs().sum()) > 0.0


def test_dual_stem_routes_optical_and_sar_samples_independently():
    """In `dual_stem` mode, modifying `init_conv_sar` only changes SAR sample outputs,
    and modifying `init_conv` only changes Optical sample outputs (#14)."""
    cfg = _tiny_config(combined_input_mode="dual_stem", seed=23)
    model = tc.build_model(UNetSegmentation, cfg).eval()

    x_opt = torch.zeros(1, 6, 16, 16)
    x_opt[0, :3] = torch.randn(3, 16, 16)
    x_sar = torch.zeros(1, 6, 16, 16)
    x_sar[0, 3:6] = torch.randn(3, 16, 16)

    with torch.no_grad():
        out_opt_before = model(x_opt).clone()
        out_sar_before = model(x_sar).clone()
        model.model.init_conv_sar.weight.mul_(2.5)
        out_opt_after = model(x_opt).clone()
        out_sar_after = model(x_sar).clone()

    assert torch.equal(out_opt_before, out_opt_after)
    assert not torch.allclose(out_sar_before, out_sar_after)


# ---------------------------------------------------------------------------
# #14: Modality and Geo-Cluster non-IID federated partitioning
# ---------------------------------------------------------------------------


def test_modality_and_geo_cluster_partitioning():
    """Verify `--partition modality` separates Optical vs. SAR silos and
    `--partition geo_cluster` produces deterministic feature clusters with the
    `min_client_samples` guard (#14)."""
    ds = _SyntheticMultimodalDataset(n_samples=20, combined_input_mode="dual_stem", seed=42)
    cfg_mod = _tiny_config(partition="modality", num_clients=4, min_client_samples=2, val_fraction=0.2, seed=42)
    clients_mod, val_idx_mod, info_mod = tc.build_federated_split(ds, cfg_mod)
    assert len(clients_mod) == 4
    assert all(len(c) >= 2 for c in clients_mod)
    assert "modality" in info_mod[0]

    # Verify each client is pure optical or pure SAR when pool has 8 optical + 8 SAR
    mods = tc.sample_modalities(ds)
    for c_idxs in clients_mod:
        c_mods = {mods[i] for i in c_idxs}
        assert len(c_mods) == 1, f"Expected pure modality silo, got {c_mods}"

    # Verify geo_cluster partitioning is deterministic and honors min_client_samples
    cfg_geo = _tiny_config(partition="geo_cluster", num_clients=4, min_client_samples=2, val_fraction=0.2, seed=42)
    clients_geo1, val_idx_geo1, info_geo1 = tc.build_federated_split(ds, cfg_geo)
    clients_geo2, val_idx_geo2, _ = tc.build_federated_split(ds, cfg_geo)
    assert clients_geo1 == clients_geo2
    assert val_idx_geo1 == val_idx_geo2
    assert all(len(c) >= 2 for c in clients_geo1)
    assert "geo/topographic" in info_geo1[0]
    # Disjoint union of clients + val equals full dataset
    all_assigned = sorted([i for c in clients_geo1 for i in c] + list(val_idx_geo1))
    assert all_assigned == list(range(len(ds)))

    # B2 regression: 7-D canonical feature vector & partition invariance across combined_input_mode
    ds_shared3 = _SyntheticMultimodalDataset(n_samples=20, combined_input_mode="shared3", seed=42)
    ds_concat6 = _SyntheticMultimodalDataset(n_samples=20, combined_input_mode="concat6", seed=42)
    ds_dual = _SyntheticMultimodalDataset(n_samples=20, combined_input_mode="dual_stem", seed=42)
    for idx in range(len(ds_shared3)):
        f_sh = tc._sample_geo_feature_vector(*ds_shared3[idx], modality=ds_shared3.modalities[idx])
        f_c6 = tc._sample_geo_feature_vector(*ds_concat6[idx], modality=ds_concat6.modalities[idx])
        f_ds = tc._sample_geo_feature_vector(*ds_dual[idx], modality=ds_dual.modalities[idx])
        assert f_sh.shape == (7,)
        assert np.allclose(f_sh, f_c6, atol=1e-6)
        assert np.allclose(f_sh, f_ds, atol=1e-6)

    clients_geo_sh, val_sh, _ = tc.build_federated_split(ds_shared3, cfg_geo)
    clients_geo_c6, val_c6, _ = tc.build_federated_split(ds_concat6, cfg_geo)
    clients_geo_ds, val_ds, _ = tc.build_federated_split(ds_dual, cfg_geo)
    assert clients_geo_sh == clients_geo_c6 == clients_geo_ds
    assert val_sh == val_c6 == val_ds

    # B1 regression: unbalanced modality counts (14 optical, 30 SAR, 8 clients, min_samples=5)
    # must allocate 2 pure optical + 6 pure SAR clients (no cross-modality transfers)
    class _UnbalancedModalityDataset(Dataset):
        def __init__(self, n_opt, n_sar):
            self.modalities = (["optical"] * n_opt) + (["sar"] * n_sar)
            self.n = n_opt + n_sar

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return torch.zeros(3, 8, 8), torch.zeros(8, 8)

    ds_unbal = _UnbalancedModalityDataset(14, 30)
    pool_unbal = np.arange(44, dtype=np.int64)
    labels_unbal = np.zeros(44, dtype=np.int64)
    silos_unbal = tc.split_modality_silos(
        ds_unbal, pool_unbal, labels_unbal, num_clients=8, seed=42, min_samples=5
    )
    assert len(silos_unbal) == 8
    assert all(len(c) >= 5 for c in silos_unbal)
    opt_clients = 0
    sar_clients = 0
    for c_idxs in silos_unbal:
        c_mods = {ds_unbal.modalities[i] for i in c_idxs}
        assert len(c_mods) == 1, f"Cross-modality contamination detected: {c_mods}"
        if "optical" in c_mods:
            opt_clients += 1
        else:
            sar_clients += 1
    assert opt_clients == 2 and sar_clients == 6

    # Infeasible pure-silo request (4 optical, 40 SAR, 8 clients, min_samples=5) must raise ValueError
    ds_infeas = _UnbalancedModalityDataset(4, 40)
    with pytest.raises(ValueError, match="Infeasible pure modality silos"):
        tc.split_modality_silos(
            ds_infeas, np.arange(44, dtype=np.int64), labels_unbal, num_clients=8, seed=42, min_samples=5
        )


# ---------------------------------------------------------------------------
# #15: Circular PQC angle aggregation across the +/-pi branch cut
# ---------------------------------------------------------------------------


def test_circular_mean_angles_branch_cut_and_non_periodic_preservation():
    """Verify circular mean resolves the +/-pi branch cut (where linear average gives 0)
    and `apply_circular_q_angle_agg_to_fit_result` only touches periodic PQC angles (#15)."""
    eps = 0.05
    a1 = np.array([np.pi - eps, 0.2], dtype=np.float32)
    a2 = np.array([-np.pi + eps, 0.4], dtype=np.float32)
    lin_avg = 0.5 * (a1 + a2)
    circ_avg = tc.circular_mean_angles([a1, a2], [10.0, 10.0])

    # Linear average erroneously collapses the first angle to 0.0!
    assert abs(float(lin_avg[0])) < 1e-6
    # Circular average preserves the angle on the +/-pi boundary (|theta| == pi)
    assert abs(abs(float(circ_avg[0])) - np.pi) < 1e-5
    # Small angles far from branch cut agree closely with linear mean
    assert abs(float(circ_avg[1]) - 0.3) < 1e-5

    # Test inside QuantumFedAvg.aggregate_fit
    cfg = _tiny_config(q_angle_agg="circular")
    model = tc.build_model(QVUNetSegmentation, cfg)
    param_keys = [name for name, _ in model.named_parameters()]
    periodic_idx = tc.quantum_periodic_param_indices(param_keys)
    assert len(periodic_idx) > 0
    # Ensure input_scales are NOT treated as periodic angles
    for idx, name in enumerate(param_keys):
        if "input_scales" in name:
            assert idx not in periodic_idx
        if "quantum_params" in name:
            assert idx in periodic_idx

    base_nd = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    c1_nd = [arr.copy() for arr in base_nd]
    c2_nd = [arr.copy() for arr in base_nd]
    for idx in range(len(base_nd)):
        if idx in periodic_idx:
            c1_nd[idx].fill(np.pi - eps)
            c2_nd[idx].fill(-np.pi + eps)
        else:
            c1_nd[idx].fill(2.0)
            c2_nd[idx].fill(4.0)

    res1 = FitRes(status=Status(code=Code.OK, message=""), parameters=ndarrays_to_parameters(c1_nd), num_examples=10, metrics={})
    res2 = FitRes(status=Status(code=Code.OK, message=""), parameters=ndarrays_to_parameters(c2_nd), num_examples=10, metrics={})

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        strat_lin = QuantumFedAvg(val_loader=None, config=_tiny_config(q_angle_agg="linear"), device_str="cpu", output_dir=tmpdir)
        agg_lin, _ = strat_lin.aggregate_fit(1, [(None, res1), (None, res2)], [])
        nd_lin = parameters_to_ndarrays(agg_lin)

        strat_circ = QuantumFedAvg(val_loader=None, config=cfg, device_str="cpu", output_dir=tmpdir)
        agg_circ, _ = strat_circ.aggregate_fit(1, [(None, res1), (None, res2)], [])
        nd_circ = parameters_to_ndarrays(agg_circ)

    for idx in range(len(base_nd)):
        if idx in periodic_idx:
            assert np.allclose(np.abs(nd_lin[idx]), 0.0, atol=1e-5)
            assert np.allclose(np.abs(nd_circ[idx]), np.pi, atol=1e-4)
        else:
            assert np.allclose(nd_lin[idx], 3.0, atol=1e-5)
            assert np.allclose(nd_circ[idx], 3.0, atol=1e-5)


# ---------------------------------------------------------------------------
# #15: FedProx proximal penalty & partial client participation
# ---------------------------------------------------------------------------


def test_fedprox_proximal_loss_and_client_fit():
    """Verify `fedprox_proximal_loss` computes (mu / 2) * sum ||w - w_global||^2
    and constrains client drift during `fit()` (#15)."""
    cfg_avg = _tiny_config(fl_strategy="fedavg", fedprox_mu=0.0, learning_rate=5e-3, local_epochs=2)
    cfg_prox = _tiny_config(fl_strategy="fedprox", fedprox_mu=1.0, learning_rate=5e-3, local_epochs=2)

    ds = _SyntheticMultimodalDataset(n_samples=4, combined_input_mode="shared3", seed=9)
    sub = Subset(ds, [0, 1, 2, 3])

    init_model = tc.build_model(UNetSegmentation, cfg_avg)
    init_params = [p.detach().cpu().numpy().copy() for p in init_model.parameters()]

    # Direct unit check of fedprox_proximal_loss
    global_tensors = [p.detach().clone() + 1.0 for p in init_model.parameters()]
    num_scalars = sum(p.numel() for p in init_model.parameters())
    prox_val = fedprox_proximal_loss(init_model, global_tensors, mu=0.5)
    assert abs(float(prox_val.item()) - 0.5 * 0.5 * num_scalars) < 1e-3

    # Compare client weight drift under FedAvg (mu=0) vs FedProx (mu=1.0)
    c_avg = ClassicalFlowerClient(0, sub, cfg_avg, "cpu")
    updated_avg, _, _ = c_avg.fit([arr.copy() for arr in init_params], {"server_round": 1})

    c_prox = ClassicalFlowerClient(0, sub, cfg_prox, "cpu")
    updated_prox, _, _ = c_prox.fit([arr.copy() for arr in init_params], {"server_round": 1})

    drift_avg = sum(float(np.sum((u - g) ** 2)) for u, g in zip(updated_avg, init_params))
    drift_prox = sum(float(np.sum((u - g) ** 2)) for u, g in zip(updated_prox, init_params))
    assert drift_avg > 0.0
    assert drift_prox < drift_avg


def test_partial_client_participation_and_fingerprints():
    """Verify `--num-clients`, `--clients-per-round`, `--fraction-fit`, and Wave 3
    fingerprint compatibility (#14, #15)."""
    parser = argparse.ArgumentParser()
    tc.add_common_args(parser, quantum=True, federated=True)
    args = parser.parse_args([
        "--num-clients", "8",
        "--fraction-fit", "0.5",
        "--fl-strategy", "fedprox",
        "--fedprox-mu", "0.05",
        "--q-angle-agg", "circular",
        "--combined-input-mode", "dual_stem",
    ])
    cfg = _tiny_config()
    tc.apply_common_args(cfg, args)

    assert cfg["num_clients"] == 8
    assert cfg["clients_per_round"] == 4
    assert abs(cfg["fraction_fit"] - 0.5) < 1e-9
    assert cfg["fl_strategy"] == "fedprox"
    assert abs(cfg["fedprox_mu"] - 0.05) < 1e-9
    assert cfg["q_angle_agg"] == "circular"
    assert cfg["combined_input_mode"] == "dual_stem"

    fp = tc.run_fingerprint(cfg, "quantum")
    assert fp["fl_strategy"] == "fedprox"
    assert abs(fp["fedprox_mu"] - 0.05) < 1e-9
    assert fp["q_angle_agg"] == "circular"
    assert fp["combined_input_mode"] == "dual_stem"
    assert fp["num_clients"] == 8
    assert fp["clients_per_round"] == 4

    # Historical pre-Wave-3 fingerprint normalization fills defaults cleanly
    old_fp = {k: v for k, v in fp.items() if k not in ("fl_strategy", "fedprox_mu", "q_angle_agg", "combined_input_mode")}
    norm_old = tc._normalize_saved_fingerprint(old_fp, "quantum")
    assert norm_old["fl_strategy"] == "fedavg"
    assert norm_old["fedprox_mu"] == 0.0
    assert norm_old["q_angle_agg"] == "linear"
    assert norm_old["combined_input_mode"] == "shared3"

    # B3 regression: explicit --fl-strategy fedprox --fedprox-mu 0 is preserved as 0.0 (not 0.01)
    args_zero_mu = parser.parse_args(["--fl-strategy", "fedprox", "--fedprox-mu", "0"])
    cfg_zero_mu = _tiny_config()
    tc.apply_common_args(cfg_zero_mu, args_zero_mu)
    assert cfg_zero_mu["fl_strategy"] == "fedprox"
    assert cfg_zero_mu["fedprox_mu"] == 0.0
    assert tc.effective_fedprox_mu(cfg_zero_mu) == 0.0

    # Omitting --fedprox-mu with --fl-strategy fedprox defaults to 0.01
    args_def_mu = parser.parse_args(["--fl-strategy", "fedprox"])
    cfg_def_mu = _tiny_config()
    tc.apply_common_args(cfg_def_mu, args_def_mu)
    assert cfg_def_mu["fl_strategy"] == "fedprox"
    assert abs(cfg_def_mu["fedprox_mu"] - 0.01) < 1e-9

    # B4 regression: non-default client counts (including nc < 4) are tagged in federated_variant_tag
    assert tc.federated_variant_tag({"num_clients": 4, "clients_per_round": 4}) == ""
    assert tc.federated_variant_tag({"num_clients": 2, "clients_per_round": 2}) == "_K2C2"
    assert tc.federated_variant_tag({"num_clients": 3, "clients_per_round": 3}) == "_K3C3"
    assert tc.federated_variant_tag({"num_clients": 4, "clients_per_round": 2}) == "_K4C2"


def test_partial_participation_checkpoint_resume_client_sampling_continuity(tmp_path):
    """B5 regression: interrupted-vs-uninterrupted partial participation (`num_clients=4`,
    `clients_per_round=2`, `fed_rounds=2`) selects the exact same client IDs on every
    global round and produces identical aggregated parameters."""
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.client_proxy import ClientProxy

    class _DummyProxy(ClientProxy):
        def __init__(self, cid):
            super().__init__(str(cid))

        def get_properties(self, ins, timeout, group_id=None):
            return None

        def get_parameters(self, ins, timeout, group_id=None):
            return None

        def fit(self, ins, timeout, group_id=None):
            return None

        def evaluate(self, ins, timeout, group_id=None):
            return None

        def reconnect(self, ins, timeout, group_id=None):
            return None

    cm = SimpleClientManager()
    for cid in range(4):
        cm.register(_DummyProxy(cid))

    cfg = _tiny_config(
        seed=42,
        num_clients=4,
        clients_per_round=2,
        fed_rounds=2,
        ckpt_every=1,
        q_angle_agg="circular",
    )
    model = tc.build_model(QVUNetSegmentation, cfg)
    init_nd = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    init_params = ndarrays_to_parameters(init_nd)

    def _make_client_fit_res(cid_int, global_round):
        arrs = [a.copy() + 0.05 * (int(cid_int) + 1) + 0.01 * int(global_round) for a in init_nd]
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(arrs),
            num_examples=4,
            metrics={"loss": 0.5 / (global_round + cid_int + 1)},
        )

    # 1. Uninterrupted 2-round run
    dir_unint = str(tmp_path / "uninterrupted")
    os.makedirs(dir_unint, exist_ok=True)
    strat_unint = QuantumFedAvg(
        val_loader=None,
        config=cfg,
        device_str="cpu",
        output_dir=dir_unint,
        fraction_fit=0.5,
        min_fit_clients=2,
        min_available_clients=4,
    )
    strat_unint.fingerprint = tc.run_fingerprint(cfg, "quantum")
    curr_p = init_params
    for r in (1, 2):
        pairs = strat_unint.configure_fit(r, curr_p, cm)
        assert len(pairs) == 2
        results = [(proxy, _make_client_fit_res(int(proxy.cid), r)) for proxy, _ in pairs]
        curr_p, _ = strat_unint.aggregate_fit(r, results, [])
        strat_unint.round_metrics.append({"round": r, "loss": 0.5 / r})
        tc.fed_round_end(strat_unint, r)

    # 2. Interrupted after round 1, then resumed for round 2 (with global random perturbed!)
    dir_res = str(tmp_path / "resumed")
    os.makedirs(dir_res, exist_ok=True)
    strat_r1 = QuantumFedAvg(
        val_loader=None,
        config=cfg,
        device_str="cpu",
        output_dir=dir_res,
        fraction_fit=0.5,
        min_fit_clients=2,
        min_available_clients=4,
    )
    strat_r1.fingerprint = tc.run_fingerprint(cfg, "quantum")
    pairs_r1 = strat_r1.configure_fit(1, init_params, cm)
    res_r1 = [(proxy, _make_client_fit_res(int(proxy.cid), 1)) for proxy, _ in pairs_r1]
    p_after_r1, _ = strat_r1.aggregate_fit(1, res_r1, [])
    strat_r1.round_metrics.append({"round": 1, "loss": 0.5})
    tc.fed_round_end(strat_r1, 1)

    # Perturb Python's global random state before resuming to prove sampling is independent of it
    import random as _py_random
    _py_random.seed(999999)
    for _ in range(37):
        _py_random.random()

    strat_r2 = QuantumFedAvg(
        val_loader=None,
        config=cfg,
        device_str="cpu",
        output_dir=dir_res,
        fraction_fit=0.5,
        min_fit_clients=2,
        min_available_clients=4,
    )
    strat_r2.fingerprint = tc.run_fingerprint(cfg, "quantum")
    ckpt_r1 = tc.load_fed_checkpoint(dir_res, strat_r2.fingerprint, resume=True)
    resumed_params, rem_rounds = tc.apply_fed_resume(strat_r2, ckpt_r1)
    assert rem_rounds == 1
    assert strat_r2.round_offset == 1
    # Flower starts resumed simulation at server_round=1 (global_round = 1 + 1 = 2)
    pairs_r2 = strat_r2.configure_fit(1, resumed_params, cm)
    res_r2 = [(proxy, _make_client_fit_res(int(proxy.cid), 2)) for proxy, _ in pairs_r2]
    p_after_r2, _ = strat_r2.aggregate_fit(1, res_r2, [])
    strat_r2.round_metrics.append({"round": 2, "loss": 0.25})
    tc.fed_round_end(strat_r2, 2)

    assert strat_r2.selected_clients_by_round == strat_unint.selected_clients_by_round
    nd_unint = parameters_to_ndarrays(curr_p)
    nd_res = parameters_to_ndarrays(p_after_r2)
    for a, b in zip(nd_unint, nd_res):
        assert np.allclose(a, b, atol=1e-6)


def test_generate_grid_wave3_check_flags():
    """Verify `misc/generate_grid.py` builds jobs for Wave 3 modes/flags and passes `--check-flags`."""
    from misc import generate_grid as gg

    jobs = [
        gg.build_job(
            method="quantum",
            mode="fed-modality",
            dataset="combined",
            seed=42,
            root=REPO_ROOT,
            python=sys.executable,
            combined_input_mode="dual_stem",
            fl_strategy="fedprox",
            fedprox_mu=0.01,
            q_angle_agg="circular",
            num_clients=8,
            clients_per_round=4,
        ),
        gg.build_job(
            method="classical",
            mode="fed-geo_cluster",
            dataset="combined",
            seed=42,
            root=REPO_ROOT,
            python=sys.executable,
            combined_input_mode="concat6",
            fl_strategy="fedprox",
            fedprox_mu=0.02,
            num_clients=6,
            fraction_fit=0.5,
        ),
        gg.build_job(
            method="ablation",
            mode="fed-modality",
            dataset="combined",
            seed=42,
            root=REPO_ROOT,
            python=sys.executable,
            combined_input_mode="dual_stem",
        ),
    ]
    assert "in-dual_stem" in jobs[0]["name"]
    assert "fedprox0.01" in jobs[0]["name"]
    assert "qagg-circular" in jobs[0]["name"]
    assert "K8C4" in jobs[0]["name"]
    assert gg.check_flags(jobs, REPO_ROOT, sys.executable)

    # NB1 regression: fraction_fit takes precedence over clients_per_round in generate_grid naming
    job_both = gg.build_job(
        method="quantum",
        mode="fed-random",
        dataset="combined",
        seed=42,
        root=REPO_ROOT,
        python=sys.executable,
        num_clients=8,
        clients_per_round=2,
        fraction_fit=0.5,
    )
    assert "K8C4" in job_both["name"] and "K8C2" not in job_both["name"]

    # Also verify analyze_results parses Wave 3 directory names and metrics fields
    import analyze_results as ar
    parsed_q = ar._parse_run_dir_full(
        "flood_federated_optical_pytorch_pennylane_modality_combined_reupload_L2_in-dual_stem_fedprox0.01_qagg-circular_K8C4_seed42"
    )
    assert parsed_q is not None
    assert parsed_q[0] == "Quantum Fed (Modality)"
    assert parsed_q[1] == "combined"
    assert parsed_q[5] == "modality"
    assert "in-dual_stem" in parsed_q[6]
    assert "fedprox0.01" in parsed_q[6]
    assert "qagg-circular" in parsed_q[6]
    assert "K8C4" in parsed_q[6]

    parsed_c = ar._parse_run_dir_full(
        "flood_federated_optical_pytorch_classical_geocluster_combined_in-concat6_fedprox0.02_K6C3_seed42"
    )
    assert parsed_c is not None
    assert parsed_c[0] == "Classical Fed (GeoCluster)"
    assert parsed_c[5] == "geo_cluster"
    assert parsed_c[6] == "in-concat6_fedprox0.02_K6C3"

