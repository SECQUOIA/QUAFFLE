import argparse
import ast
import inspect
import json
import os
import pathlib
import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tools import training_common as tc
from utilsTorch.unetTorch import (
    AblationResnetBlock,
    ClassicalBottleneckBlock,
    DEFAULT_SKIP_MODULATION,
    QResnetBlock,
    QuantumBlock,
    normalize_skip_modulation,
)
from utilsTorch.utilsTorch import (
    ClassicalAblationUNetSegmentation,
    QVUNetSegmentation,
    evaluate_model,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Issue #6 Triage & Functional Regression Tests
# ---------------------------------------------------------------------------


def test_issue6_best_val_checkpoint_restoration():
    """Verify centralized and federated Torch trainers restore best-validation weights before test_loader evaluation."""
    centralized_files = [
        "centralized_demoTorchQ.py",
        "centralized_demoTorchClassical.py",
        "ablation_studies.py",
    ]
    for rel_path in centralized_files:
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "best_state_dict" in src or "best_model_path" in src, (
            f"{rel_path} never tracks best validation model state (best_state_dict / best_model_path); "
            "test_loader is evaluated on final-epoch weights instead of best-validation weights."
        )

    federated_files = [
        "federated_demoTorchQ.py",
        "federated_demoTorchClassical.py",
        "ablation_studies.py",
    ]
    for rel_path in federated_files:
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "best_parameters" in src, (
            f"{rel_path} CustomFedAvg never snapshots best_parameters across validation rounds; "
            "test_loader is evaluated on strategy.final_parameters instead of best validation parameters."
        )


def test_issue6_select_metric_helpers_and_fingerprint():
    """Verify select-metric helpers handle maximization metrics (iou/dice/auc) and minimization (loss)."""
    for metric in ("iou", "dice", "auc"):
        init = tc.initial_best_val_score(metric)
        assert init == float("-inf")
        assert tc.is_better_val_score(0.5, init, metric)
        assert tc.is_better_val_score(0.6, 0.5, metric)
        assert not tc.is_better_val_score(0.4, 0.5, metric)
        assert not tc.is_better_val_score(float("nan"), 0.5, metric)

    init_loss = tc.initial_best_val_score("loss")
    assert init_loss == float("inf")
    assert tc.is_better_val_score(0.8, init_loss, "loss")
    assert tc.is_better_val_score(0.3, 0.8, "loss")
    assert not tc.is_better_val_score(0.9, 0.8, "loss")

    with pytest.raises(ValueError):
        tc.select_metric_name({"select_metric": "invalid_metric"})

    base_fp_cfg = {
        "seed": 42,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "fed_rounds": 4,
        "local_epochs": 1,
        "num_clients": 2,
        "clients_per_round": 2,
    }
    cfg_iou = {**base_fp_cfg, "select_metric": "iou", "skip_modulation": "affine"}
    cfg_dice = {**base_fp_cfg, "select_metric": "dice", "skip_modulation": "affine"}
    cfg_add = {**base_fp_cfg, "select_metric": "iou", "skip_modulation": "additive"}
    assert tc.centralized_fingerprint(cfg_iou, "quantum") != tc.centralized_fingerprint(cfg_dice, "quantum")
    assert tc.centralized_fingerprint(cfg_iou, "quantum") != tc.centralized_fingerprint(cfg_add, "quantum")
    assert tc.run_fingerprint(cfg_iou, "quantum") != tc.run_fingerprint(cfg_dice, "quantum")
    assert tc.run_fingerprint(cfg_iou, "quantum") != tc.run_fingerprint(cfg_add, "quantum")


def test_issue6_federated_best_parameters_persistence_and_restoration(tmp_path):
    """Verify CustomFedAvg snapshots best_parameters, persists across fed_resume.pt, and restores best round."""
    from flwr.common import ndarrays_to_parameters
    import federated_demoTorchClassical as fed_cls

    cfg = {
        "seed": 7,
        "fed_rounds": 4,
        "ckpt_every": 1,
        "local_epochs": 1,
        "num_clients": 2,
        "clients_per_round": 2,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "select_metric": "iou",
        "skip_modulation": "affine",
    }
    strategy = fed_cls.CustomFedAvg(
        val_loader=None,
        config=cfg,
        device_str="cpu",
        output_dir=str(tmp_path),
    )
    strategy.fingerprint = tc.run_fingerprint(cfg, "classical")

    round1_arrays = [np.array([1.0, 2.0], dtype=np.float32)]
    round2_arrays = [np.array([9.0, 9.0], dtype=np.float32)]
    round3_arrays = [np.array([-3.0, -4.0], dtype=np.float32)]

    # Simulate Round 1 (IoU = 0.40)
    strategy.final_parameters = ndarrays_to_parameters(round1_arrays)
    m1 = {"step": 1, "loss": 0.6, "accuracy": 0.7, "auc": 0.75, "iou": 0.40, "dice": 0.57}
    strategy.round_metrics.append(m1)
    strategy.best_performance = m1["auc"]
    strategy.best_val_score = m1["iou"]
    strategy.best_round = 1
    strategy.best_val_metrics = dict(m1)
    strategy.best_parameters = tc.clone_flower_parameters(strategy.final_parameters)
    tc.fed_round_end(strategy, 1)

    # Simulate Round 2 (best IoU = 0.85)
    strategy.final_parameters = ndarrays_to_parameters(round2_arrays)
    m2 = {"step": 2, "loss": 0.2, "accuracy": 0.9, "auc": 0.92, "iou": 0.85, "dice": 0.92}
    strategy.round_metrics.append(m2)
    strategy.best_performance = m2["auc"]
    strategy.best_val_score = m2["iou"]
    strategy.best_round = 2
    strategy.best_val_metrics = dict(m2)
    strategy.best_parameters = tc.clone_flower_parameters(strategy.final_parameters)
    tc.fed_round_end(strategy, 2)

    # Simulate Round 3 (regressed IoU = 0.30; final_parameters updated, best_parameters kept at Round 2)
    strategy.final_parameters = ndarrays_to_parameters(round3_arrays)
    m3 = {"step": 3, "loss": 0.8, "accuracy": 0.6, "auc": 0.65, "iou": 0.30, "dice": 0.46}
    strategy.round_metrics.append(m3)
    tc.fed_round_end(strategy, 3)

    # Resume into a fresh strategy instance
    resumed_strategy = fed_cls.CustomFedAvg(
        val_loader=None,
        config=cfg,
        device_str="cpu",
        output_dir=str(tmp_path),
    )
    resumed_strategy.fingerprint = strategy.fingerprint
    ckpt = tc.load_fed_checkpoint(str(tmp_path), strategy.fingerprint, resume=True)
    resumed_params, remaining = tc.apply_fed_resume(resumed_strategy, ckpt)
    assert remaining == 1
    assert resumed_strategy.best_round == 2
    assert np.isclose(resumed_strategy.best_val_score, 0.85)

    best_params, final_params, used_best = tc.select_federated_eval_parameters(resumed_strategy)
    assert used_best is True
    assert resumed_strategy.best_round == 2
    assert np.isclose(resumed_strategy.best_val_score, 0.85)
    assert resumed_strategy.best_val_metrics["iou"] == pytest.approx(0.85)
    np.testing.assert_allclose(best_params[0], round2_arrays[0])
    np.testing.assert_allclose(final_params[0], round3_arrays[0])

    # Verify fallback to final_parameters when no validation rounds ran
    no_val_strategy = fed_cls.CustomFedAvg(
        val_loader=None,
        config=cfg,
        device_str="cpu",
        output_dir=str(tmp_path / "noval"),
    )
    no_val_strategy.final_parameters = ndarrays_to_parameters(round3_arrays)
    b_p, f_p, used_best_noval = tc.select_federated_eval_parameters(no_val_strategy)
    assert used_best_noval is False
    assert no_val_strategy.best_round is None
    np.testing.assert_allclose(b_p[0], round3_arrays[0])
    np.testing.assert_allclose(f_p[0], round3_arrays[0])


def test_issue6_write_metrics_json_records_best_and_final_metrics(tmp_path):
    """Verify write_metrics_json records both best-validation and final-epoch test/val metrics."""
    cfg = {"seed": 42, "select_metric": "iou", "skip_modulation": "affine"}
    best_val = {"epoch": 2, "loss": 0.25, "accuracy": 0.90, "auc": 0.93, "iou": 0.82, "dice": 0.90}
    final_val = {"epoch": 4, "loss": 0.55, "accuracy": 0.78, "auc": 0.80, "iou": 0.60, "dice": 0.75}
    best_test = {"loss": 0.28, "accuracy": 0.89, "auc": 0.91, "iou": 0.80, "dice": 0.89}
    final_test = {"loss": 0.58, "accuracy": 0.76, "auc": 0.79, "iou": 0.58, "dice": 0.73}

    path = tc.write_metrics_json(
        str(tmp_path),
        script="centralized_demoTorchQ.py",
        training="centralized",
        model_type="quantum",
        config=cfg,
        val_metrics=best_val,
        best_val_metrics=best_val,
        final_val_metrics=final_val,
        test_metrics=best_test,
        best_val_test_metrics=best_test,
        final_test_metrics=final_test,
        history=[best_val, final_val],
        total_params=1234,
        best_val_score=0.82,
        best_epoch=2,
        best_step=2,
    )
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    assert payload["select_metric"] == "iou"
    assert payload["best_epoch"] == 2
    assert payload["best_val_score"] == pytest.approx(0.82)
    assert payload["val"]["iou"] == pytest.approx(0.82)
    assert payload["best_val_metrics"]["iou"] == pytest.approx(0.82)
    assert payload["final_val_metrics"]["iou"] == pytest.approx(0.60)
    assert payload["test"]["iou"] == pytest.approx(0.80)
    assert payload["best_val_test_metrics"]["iou"] == pytest.approx(0.80)
    assert payload["final_test_metrics"]["iou"] == pytest.approx(0.58)


# ---------------------------------------------------------------------------
# Issue #10 Triage & Functional Regression Tests
# ---------------------------------------------------------------------------


def test_issue10_evaluate_model_no_bare_except_or_pixel_list_extend():
    """Verify evaluate_model avoids pixel-by-pixel list.extend and bare except:, and clients reuse template_model."""
    src = inspect.getsource(evaluate_model)
    tree = ast.parse(src)

    bare_excepts = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is None
    ]
    assert not bare_excepts, "evaluate_model contains a bare `except:` handler"

    extend_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "extend"
    ]
    assert not extend_calls, "evaluate_model uses Python list.extend() per batch instead of NumPy array accumulation"

    fed_cls_src = (REPO_ROOT / "federated_demoTorchClassical.py").read_text(encoding="utf-8")
    assert "copy.deepcopy(self.template_model)" not in fed_cls_src, (
        "federated_demoTorchClassical.py still calls copy.deepcopy(self.template_model) on every round"
    )
    ablation_src = (REPO_ROOT / "ablation_studies.py").read_text(encoding="utf-8")
    assert "copy.deepcopy(self.template_model)" not in ablation_src, (
        "ablation_studies.py still calls copy.deepcopy(self.template_model) on every round"
    )
    fed_q_src = (REPO_ROOT / "federated_demoTorchQ.py").read_text(encoding="utf-8")
    assert "copy.deepcopy(self.template_model)" not in fed_q_src, (
        "federated_demoTorchQ.py still calls copy.deepcopy(self.template_model) during server evaluation"
    )


class _DummyInnerModel(torch.nn.Module):
    def __init__(self, logits_tensor: torch.Tensor):
        super().__init__()
        self._logits = logits_tensor

    def forward(self, x, t):
        B = x.shape[0]
        return self._logits[:B].to(x.device)


class _DummySegWrapper(torch.nn.Module):
    def __init__(self, logits_tensor: torch.Tensor):
        super().__init__()
        self.model = _DummyInnerModel(logits_tensor)

    def forward(self, x, t=None):
        return self.model(x, t)


def test_issue10_evaluate_model_single_class_and_multi_class_auc():
    """Verify evaluate_model computes multi-class AUC and handles single-class masks without warnings or bare except."""
    B, C, H, W = 2, 2, 8, 8
    images = torch.zeros((B, 3, H, W), dtype=torch.float32)

    # Multi-class mask where logits strongly separate class 1 from class 0
    masks = torch.zeros((B, 1, H, W), dtype=torch.float32)
    masks[:, :, :4, :] = 1.0
    logits = torch.zeros((B, C, H, W), dtype=torch.float32)
    logits[:, 0, :, :] = 2.0
    logits[:, 1, :4, :] = 5.0
    logits[:, 0, :4, :] = -2.0

    loader = DataLoader(TensorDataset(images, masks), batch_size=1)
    wrapper = _DummySegWrapper(logits)
    metrics = evaluate_model(wrapper, loader, torch.device("cpu"))
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["dice"] == pytest.approx(1.0)

    # Single-class mask (all zeros): should not emit UndefinedMetricWarning and should return single_class_auc
    all_zero_masks = torch.zeros((B, 1, H, W), dtype=torch.float32)
    zero_loader = DataLoader(TensorDataset(images, all_zero_masks), batch_size=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m_default = evaluate_model(wrapper, zero_loader, torch.device("cpu"))
        m_nan = evaluate_model(wrapper, zero_loader, torch.device("cpu"), single_class_auc=float("nan"))
    assert not any("UndefinedMetricWarning" in str(w.category) for w in caught)
    assert m_default["auc"] == 0.0
    assert np.isnan(m_nan["auc"])


def test_issue10_flower_clients_reuse_template_model_and_gpu_resources():
    """Verify ClassicalFlowerClient and _TorchClassicalFlowerClient reuse self.template_model and align num_gpus=0.2."""
    import federated_demoTorchClassical as fed_cls
    import ablation_studies as abl

    small_cfg = {
        "image_size": 16,
        "channels": 3,
        "num_classes": 2,
        "out_dim": 2,
        "base_channels": 8,
        "init_dim": 8,
        "dim_mults": (1, 2),
        "resnet_block_groups": 2,
        "quantum_channels": 8,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "local_epochs": 1,
        "seed": 42,
        "skip_modulation": "affine",
    }
    ds = TensorDataset(
        torch.zeros((2, 3, 16, 16), dtype=torch.float32),
        torch.zeros((2, 1, 16, 16), dtype=torch.float32),
    )

    cls_client = fed_cls.ClassicalFlowerClient(0, ds, dict(small_cfg), "cpu")
    params, _, _ = cls_client.fit([], {"server_round": 1})
    first_template = cls_client.template_model
    assert cls_client.local_model is first_template
    cls_client.evaluate(params, {"server_round": 1})
    assert cls_client.template_model is first_template

    abl_client = abl._TorchClassicalFlowerClient(0, ds, dict(small_cfg), "cpu")
    params_abl, _, _ = abl_client.fit([], {"server_round": 1})
    first_abl_template = abl_client.template_model
    assert abl_client.local_model is first_abl_template
    abl_client.evaluate(params_abl, {"server_round": 1})
    assert abl_client.template_model is first_abl_template

    for rel_path in ("federated_demoTorchQ.py", "federated_demoTorchClassical.py", "ablation_studies.py"):
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert '"num_gpus": 0.2' in src, f"{rel_path} does not set num_gpus: 0.2 in client_resources"


# ---------------------------------------------------------------------------
# Issue #27 Triage & Functional Regression Tests
# ---------------------------------------------------------------------------


def test_issue_27_qresnet_block_additive_dc_offset_annihilated_by_spatial_centering():
    """
    Regression test for Issue #27:
    QResnetBlock and AblationResnetBlock must modulate spatial features (not just
    shift their spatial DC mean), so that after spatial centering (as performed by
    downstream GroupNorm), quantum_mode='on' and quantum_mode='zero' produce distinct
    spatial feature maps (max_ac_diff > 1e-4).
    """
    torch.manual_seed(42)
    q_block = QuantumBlock(quantum_channels=32, q_ansatz="reupload", q_layers=2, q_backend="torch", seed=42)
    res_block = QResnetBlock(256, 256, time_emb_dim=128, groups=4, quantum_channels=32, q_block=q_block)
    res_block.eval()

    x = torch.randn(2, 256, 8, 8)
    t = torch.randn(2, 128)

    with torch.no_grad():
        q_block.set_quantum_mode("on")
        out_on = res_block(x, t)
        q_block.set_quantum_mode("zero")
        out_zero = res_block(x, t)

    on_ac = out_on - out_on.mean(dim=(-2, -1), keepdim=True)
    zero_ac = out_zero - out_zero.mean(dim=(-2, -1), keepdim=True)
    max_ac_diff = (on_ac - zero_ac).abs().max().item()

    assert max_ac_diff > 1e-4, (
        f"Spatially centered QResnetBlock output is nearly identical between "
        f"quantum_mode='on' and 'zero' (max_ac_diff={max_ac_diff:.3e} <= 1e-4); "
        f"y_q only shifts spatial DC mean and is annihilated by downstream GroupNorm."
    )


def test_issue_27_ablation_resnet_block_and_backward_compatible_additive_mode():
    """Verify AblationResnetBlock affine modulation vs legacy additive mode and paired-seed determinism (#5)."""
    torch.manual_seed(42)
    c_block = ClassicalBottleneckBlock(quantum_channels=32, q_ansatz="reupload", q_layers=2, seed=42)
    abl_affine = AblationResnetBlock(
        256, 256, time_emb_dim=128, groups=4, quantum_channels=32, q_block=c_block, skip_modulation="affine"
    )
    abl_affine.eval()

    x = torch.randn(2, 256, 8, 8)
    t = torch.randn(2, 128)
    with torch.no_grad():
        c_block.set_quantum_mode("on")
        out_on = abl_affine(x, t)
        c_block.set_quantum_mode("zero")
        out_zero = abl_affine(x, t)

    on_ac = out_on - out_on.mean(dim=(-2, -1), keepdim=True)
    zero_ac = out_zero - out_zero.mean(dim=(-2, -1), keepdim=True)
    assert (on_ac - zero_ac).abs().max().item() > 1e-4

    # Legacy additive mode preserves pure DC shift behavior for backward compatibility
    abl_additive = AblationResnetBlock(
        256, 256, time_emb_dim=128, groups=4, quantum_channels=32, q_block=c_block, skip_modulation="additive"
    )
    abl_additive.load_state_dict(abl_affine.state_dict())
    abl_additive.eval()
    with torch.no_grad():
        c_block.set_quantum_mode("on")
        add_on = abl_additive(x, t)
        c_block.set_quantum_mode("zero")
        add_zero = abl_additive(x, t)
    assert (add_on - add_zero).abs().max().item() > 1e-3
    add_on_ac = add_on - add_on.mean(dim=(-2, -1), keepdim=True)
    add_zero_ac = add_zero - add_zero.mean(dim=(-2, -1), keepdim=True)
    assert (add_on_ac - add_zero_ac).abs().max().item() < 1e-5

    assert normalize_skip_modulation(None) == DEFAULT_SKIP_MODULATION == "affine"
    assert normalize_skip_modulation("ADDITIVE") == "additive"
    with pytest.raises(ValueError):
        normalize_skip_modulation("invalid_mode")

    # Paired-seed initialization determinism (#5) preserved with skip_modulation='affine'
    cfg = {
        "image_size": 32,
        "channels": 3,
        "num_classes": 2,
        "out_dim": 2,
        "base_channels": 8,
        "init_dim": 8,
        "dim_mults": (1, 2),
        "resnet_block_groups": 2,
        "quantum_channels": 8,
        "q_backend": "torch",
        "skip_modulation": "affine",
    }
    q_model = QVUNetSegmentation(cfg, seed=123)
    c_model = ClassicalAblationUNetSegmentation(cfg, seed=123)
    q_sd = q_model.state_dict()
    c_sd = c_model.state_dict()
    shared_keys = [k for k in q_sd.keys() if k in c_sd and "q_block" not in k]
    assert len(shared_keys) > 0
    for k in shared_keys:
        torch.testing.assert_close(q_sd[k], c_sd[k], rtol=0, atol=0)
