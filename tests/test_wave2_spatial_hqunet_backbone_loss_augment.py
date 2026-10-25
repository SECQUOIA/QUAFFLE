"""Regression and unit tests for Wave 2 architectural & training enhancements:
- Issue #13: Spatial bottleneck preservation (pixel, patch2x2, patch4x4, se_gate)
  and spatial classical bottleneck baselines (conv1x1, depthwise3x3, lowrank).
- Issue #18: HQ-UNet 2D spatial quanvolutional bottleneck (--q-ansatz hqunet,
  --hqunet-grid {2,4}) and matched spatial-grid classical control (hqunet_spatial).
- Issue #22: Optional depthwise-separable and standard encoder/decoder backbone
  convolutions (--backbone-conv-mode {ws_conv,standard,depthwise_sep}).
- Issue #16: Dice + Cross-Entropy / Focal + Dice segmentation losses
  (--loss {ce,dice_ce,focal_dice}) and optional removal of dead diffusion
  time_mlp parameters (--no-time-emb).
- Issue #25: Seed-reproducible dihedral (D4) geometric data augmentations for
  training splits (--augment).
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import training_common as tc
from utilsTorch import (
    ClassicalAblationUNetSegmentation,
    ClassicalBottleneckBlock,
    DepthwiseSeparableConv,
    QVUNetSegmentation,
    QuantumBlock,
    UNetSegmentation,
    evaluate_model,
    focal_loss,
    segmentation_loss,
    soft_dice_loss,
    train_one_epoch,
)


class _SyntheticFloodDataset(Dataset):
    """Deterministic synthetic dataset where mask is derived directly from image[0]."""

    def __init__(self, n_samples=8, channels=3, size=32):
        g = torch.Generator()
        g.manual_seed(12345)
        self.images = torch.randn(n_samples, channels, size, size, generator=g)
        # Ensure each sample has both flood (1) and non-flood (0) pixels
        # and mask is a deterministic function of image[0] so geometric
        # transform equivariance can be verified bit-for-bit.
        self.masks = (self.images[:, 0:1] > 0.0).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].clone(), self.masks[idx].clone()


# ===========================================================================
# Issue #13: Spatial 4-qubit bottleneck modes & spatial classical baselines
# ===========================================================================


@pytest.mark.parametrize("spatial_mode", ["global", "pixel", "patch2x2", "patch4x4", "se_gate"])
@pytest.mark.parametrize("q_backend", ["torch", "pennylane", "pennylane_pergroup"])
def test_issue13_quantum_spatial_modes_forward_backward_and_knockout(spatial_mode, q_backend):
    torch.manual_seed(42)
    qb = QuantumBlock(
        quantum_channels=8,
        backend="pennylane",
        q_ansatz="reupload",
        q_layers=2,
        q_backend=q_backend,
        q_spatial_mode=spatial_mode,
        seed=42,
    )
    x = torch.randn(2, 8, 4, 4, requires_grad=True)
    y_on = qb(x)
    assert y_on.shape == (2, 8, 4, 4)
    loss = (y_on ** 2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert len(qb.quantum_params) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in qb.quantum_params)

    if spatial_mode == "global":
        assert float(y_on.std(dim=(2, 3)).max().item()) < 1e-6
    else:
        assert float(y_on.std(dim=(2, 3)).max().item()) > 1e-4

    # Knockout modes ('zero' and 'random')
    qb.set_quantum_mode("zero")
    y_zero = qb(x.detach())
    assert torch.allclose(y_zero, torch.zeros_like(y_zero))

    qb.set_quantum_mode("random", seed=99)
    y_rand = qb(x.detach())
    assert not torch.allclose(y_on.detach(), y_rand)
    qb.set_quantum_mode("on")


@pytest.mark.parametrize("spatial_mode", ["pixel", "patch2x2", "patch4x4", "se_gate"])
def test_issue13_quantum_spatial_modes_simulator_agreement(spatial_mode):
    x = torch.randn(2, 8, 4, 4)
    qb_torch = QuantumBlock(
        quantum_channels=8,
        backend="pennylane",
        q_ansatz="reupload",
        q_layers=2,
        q_backend="torch",
        q_spatial_mode=spatial_mode,
        seed=77,
    )
    qb_pl = QuantumBlock(
        quantum_channels=8,
        backend="pennylane",
        q_ansatz="reupload",
        q_layers=2,
        q_backend="pennylane",
        q_spatial_mode=spatial_mode,
        seed=77,
    )
    with torch.no_grad():
        y_t = qb_torch(x)
        y_p = qb_pl(x)
    assert torch.allclose(y_t, y_p, atol=1e-5)


@pytest.mark.parametrize("ablation_mode", ["linear", "mlp", "conv1x1", "depthwise3x3", "lowrank"])
@pytest.mark.parametrize("spatial_mode", ["global", "pixel", "patch2x2", "patch4x4", "se_gate"])
def test_issue13_classical_bottleneck_modes_and_paired_seed(ablation_mode, spatial_mode):
    cb = ClassicalBottleneckBlock(
        quantum_channels=8,
        q_ansatz="reupload",
        q_layers=2,
        spatial_mode=spatial_mode,
        ablation_mode=ablation_mode,
        seed=42,
    )
    x = torch.randn(2, 8, 4, 4, requires_grad=True)
    y = cb(x)
    assert y.shape == (2, 8, 4, 4)
    (y ** 2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    if ablation_mode in ("conv1x1", "depthwise3x3", "lowrank") or spatial_mode != "global":
        assert float(y.std(dim=(2, 3)).max().item()) > 1e-4

    # Paired-seed check (#5): encoder/decoder weights must match QVUNetSegmentation
    cfg = {
        "image_size": 32,
        "base_channels": 8,
        "init_dim": 8,
        "out_dim": 2,
        "dim_mults": (1, 2),
        "resnet_block_groups": 4,
        "quantum_channels": 8,
        "q_ansatz": "reupload",
        "q_layers": 2,
        "q_backend": "torch",
        "q_spatial_mode": spatial_mode,
        "ablation_mode": ablation_mode,
        "seed": 101,
    }
    q_model = tc.build_model(QVUNetSegmentation, cfg)
    c_model = tc.build_model(ClassicalAblationUNetSegmentation, cfg)
    q_sd = q_model.state_dict()
    c_sd = c_model.state_dict()
    shared_keys = [
        k for k in q_sd.keys()
        if k in c_sd and "quantum_block" not in k and q_sd[k].shape == c_sd[k].shape
    ]
    assert len(shared_keys) > 20
    for k in shared_keys:
        assert torch.equal(q_sd[k], c_sd[k]), f"Mismatch on shared parameter {k}"


# ===========================================================================
# Issue #18: HQ-UNet 2D spatial quanvolutional bottleneck & matched control
# ===========================================================================


@pytest.mark.parametrize("grid", [2, 4])
def test_issue18_hqunet_forward_backward_knockout_and_matched_control(grid):
    cfg = {
        "image_size": 32,
        "base_channels": 8,
        "init_dim": 8,
        "out_dim": 2,
        "dim_mults": (1, 2),
        "resnet_block_groups": 4,
        "quantum_channels": 8,
        "q_ansatz": "hqunet",
        "q_layers": 1,
        "hqunet_grid": grid,
        "q_backend": "torch",
        "seed": 55,
    }
    q_model = tc.build_model(QVUNetSegmentation, cfg)
    c_model = tc.build_model(ClassicalAblationUNetSegmentation, cfg)

    q_params = sum(p.numel() for p in q_model.parameters())
    c_params = sum(p.numel() for p in c_model.parameters())
    rel_diff = abs(c_params - q_params) / q_params
    assert rel_diff < 0.005, f"Expected <0.5% param difference, got {rel_diff:.4%}"

    # Verify bit-identical initialization on all shared weights (including pre_q_conv and post_q_proj!)
    q_sd = q_model.state_dict()
    c_sd = c_model.state_dict()
    shared = [k for k in q_sd if k in c_sd and q_sd[k].shape == c_sd[k].shape]
    assert any("pre_q_conv" in k for k in shared)
    assert any("post_q_proj" in k for k in shared)
    for k in shared:
        assert torch.equal(q_sd[k], c_sd[k]), f"Seed mismatch on {k}"

    # Forward & backward pass
    x = torch.randn(2, 3, 32, 32)
    logits_q = q_model(x)
    logits_c = c_model(x)
    assert logits_q.shape == (2, 2, 32, 32)
    assert logits_c.shape == (2, 2, 32, 32)
    (logits_q ** 2).mean().backward()
    qb = q_model.model.vertex.q_block
    assert qb.hqunet_weights.grad is not None and torch.isfinite(qb.hqunet_weights.grad).all()

    # Spatial variance of bottleneck output y_q
    with torch.no_grad():
        x_b = torch.randn(2, 8, 8, 8)
        y_on = qb(x_b)
        assert y_on.shape == (2, 8, 8, 8)
        assert float(y_on.std(dim=(2, 3)).max().item()) > 1e-4

        qb.set_quantum_mode("zero")
        y_zero = qb(x_b)
        assert not torch.allclose(y_on, y_zero)

        qb.set_quantum_mode("random", seed=123)
        y_rand = qb(x_b)
        assert not torch.allclose(y_on, y_rand)
        assert not torch.allclose(y_zero, y_rand)
        qb.set_quantum_mode("on")


def test_issue18_hqunet_grid2_pennylane_matches_torch_simulator():
    x = torch.randn(2, 8, 4, 4)
    qb_torch = QuantumBlock(
        quantum_channels=8,
        backend="pennylane",
        q_ansatz="hqunet",
        q_layers=2,
        hqunet_grid=2,
        q_backend="torch",
        seed=88,
    )
    qb_pl = QuantumBlock(
        quantum_channels=8,
        backend="pennylane",
        q_ansatz="hqunet",
        q_layers=2,
        hqunet_grid=2,
        q_backend="pennylane",
        seed=88,
    )
    with torch.no_grad():
        y_t = qb_torch(x)
        y_p = qb_pl(x)
    assert torch.allclose(y_t, y_p, atol=1e-5)


# ===========================================================================
# Issue #22: Depthwise-separable & standard backbone convolutions
# ===========================================================================


def test_issue22_depthwise_separable_backbone_param_reduction_and_paired_seed():
    base_cfg = {
        "image_size": 64,
        "base_channels": 32,
        "init_dim": 32,
        "out_dim": 2,
        "dim_mults": (1, 2, 4, 8),
        "resnet_block_groups": 4,
        "quantum_channels": 32,
        "q_ansatz": "reupload",
        "q_layers": 2,
        "q_backend": "torch",
        "seed": 42,
    }
    q_ws = tc.build_model(QVUNetSegmentation, {**base_cfg, "backbone_conv_mode": "ws_conv"})
    q_dw = tc.build_model(QVUNetSegmentation, {**base_cfg, "backbone_conv_mode": "depthwise_sep"})
    c_dw = tc.build_model(
        ClassicalAblationUNetSegmentation, {**base_cfg, "backbone_conv_mode": "depthwise_sep"}
    )
    u_ws = tc.build_model(UNetSegmentation, {**base_cfg, "backbone_conv_mode": "ws_conv"})
    u_dw = tc.build_model(UNetSegmentation, {**base_cfg, "backbone_conv_mode": "depthwise_sep"})
    u_std = tc.build_model(UNetSegmentation, {**base_cfg, "backbone_conv_mode": "standard"})

    params_ws = sum(p.numel() for p in q_ws.parameters())
    params_dw = sum(p.numel() for p in q_dw.parameters())
    assert params_dw < 0.35 * params_ws, (
        f"Expected >65% parameter reduction with depthwise_sep, got {params_dw} vs {params_ws}"
    )
    assert sum(p.numel() for p in u_dw.parameters()) < 0.35 * sum(
        p.numel() for p in u_ws.parameters()
    )
    assert sum(p.numel() for p in u_std.parameters()) == sum(
        p.numel() for p in u_ws.parameters()
    )

    # Check DepthwiseSeparableConv instances are present in q_dw
    assert any(isinstance(m, DepthwiseSeparableConv) for m in q_dw.modules())

    # Paired-seed initialization between QVUNetSegmentation and ClassicalAblationUNetSegmentation
    q_sd = q_dw.state_dict()
    c_sd = c_dw.state_dict()
    shared = [k for k in q_sd if k in c_sd and "quantum_block" not in k and q_sd[k].shape == c_sd[k].shape]
    for k in shared:
        assert torch.equal(q_sd[k], c_sd[k]), f"Mismatch on {k} under depthwise_sep"

    # Forward pass smoke test
    x = torch.randn(2, 3, 32, 32)
    assert q_dw(x).shape == (2, 2, 32, 32)
    assert u_dw(x).shape == (2, 2, 32, 32)
    assert u_std(x).shape == (2, 2, 32, 32)


# ===========================================================================
# Issue #16: Dice+CE / Focal+Dice loss & optional removal of dead time_mlp
# ===========================================================================


def test_issue16_segmentation_losses_imbalanced_gradients_and_train_eval():
    torch.manual_seed(42)
    # ~1% flood pixels (strongly imbalanced) with background-biased initial predictions
    masks = torch.zeros(2, 32, 32, dtype=torch.long)
    masks[:, :3, :3] = 1

    base_logits = torch.zeros(2, 2, 32, 32)
    base_logits[:, 0] = 2.0
    base_logits[:, 1] = -2.0

    logits_ce = base_logits.clone().requires_grad_(True)
    loss_ce = segmentation_loss(logits_ce, masks, loss_mode="ce")
    loss_ce.backward()
    flood_grad_ce = logits_ce.grad[:, :, :3, :3].norm().item()

    logits_dc = base_logits.clone().requires_grad_(True)
    loss_dc = segmentation_loss(logits_dc, masks, loss_mode="dice_ce")
    loss_dc.backward()
    flood_grad_dc = logits_dc.grad[:, :, :3, :3].norm().item()

    logits_fd = base_logits.clone().requires_grad_(True)
    loss_fd = segmentation_loss(logits_fd, masks, loss_mode="focal_dice")
    loss_fd.backward()
    flood_grad_fd = logits_fd.grad[:, :, :3, :3].norm().item()

    assert torch.isfinite(loss_ce) and torch.isfinite(loss_dc) and torch.isfinite(loss_fd)
    # Dice component amplifies gradient on rare positive (flood) pixels relative to pure CE
    assert flood_grad_dc > flood_grad_ce * 1.5
    assert flood_grad_fd > flood_grad_ce * 1.5


def test_issue16_no_time_emb_removes_dead_parameters_and_preserves_paired_seed():
    base_cfg = {
        "image_size": 32,
        "base_channels": 32,
        "init_dim": 32,
        "out_dim": 2,
        "dim_mults": (1, 2, 4, 8),
        "resnet_block_groups": 4,
        "quantum_channels": 32,
        "q_ansatz": "reupload",
        "q_layers": 2,
        "q_backend": "torch",
        "seed": 42,
    }
    q_with = tc.build_model(QVUNetSegmentation, {**base_cfg, "use_time_emb": True})
    q_no = tc.build_model(QVUNetSegmentation, {**base_cfg, "use_time_emb": False})
    c_no = tc.build_model(ClassicalAblationUNetSegmentation, {**base_cfg, "use_time_emb": False})
    u_with = tc.build_model(UNetSegmentation, {**base_cfg, "use_time_emb": True})
    u_no = tc.build_model(UNetSegmentation, {**base_cfg, "use_time_emb": False})

    assert any("time_mlp" in k for k, _ in q_with.named_parameters())
    assert not any("time_mlp" in k for k, _ in q_no.named_parameters())
    assert not any("time_mlp" in k for k, _ in c_no.named_parameters())
    assert not any("time_mlp" in k for k, _ in u_no.named_parameters())

    diff_q = sum(p.numel() for p in q_with.parameters()) - sum(p.numel() for p in q_no.parameters())
    diff_u = sum(p.numel() for p in u_with.parameters()) - sum(p.numel() for p in u_no.parameters())
    assert diff_q == 524352, f"Expected 524,352 time_mlp/init_mlp params removed in QVUNet, got {diff_q}"
    assert diff_u == 540864, f"Expected 540,864 time_mlp/init_mlp params removed in UNet, got {diff_u}"

    # Paired-seed initialization preserved when use_time_emb=False
    q_sd = q_no.state_dict()
    c_sd = c_no.state_dict()
    shared = [k for k in q_sd if k in c_sd and "quantum_block" not in k and q_sd[k].shape == c_sd[k].shape]
    for k in shared:
        assert torch.equal(q_sd[k], c_sd[k]), f"Mismatch on {k} with use_time_emb=False"

    # Train + eval smoke test with loss_mode='dice_ce' and use_time_emb=False
    ds = _SyntheticFloodDataset(n_samples=4, size=32)
    loader = tc.make_loader(ds, batch_size=2, shuffle=True, seed=42)
    q_no.loss_mode = "dice_ce"
    opt = torch.optim.Adam(q_no.parameters(), lr=1e-3)
    loss_val, acc_val = train_one_epoch(q_no, loader, opt, torch.device("cpu"))
    assert np.isfinite(loss_val) and 0.0 <= acc_val <= 1.0
    eval_res = evaluate_model(q_no, loader, torch.device("cpu"))
    assert np.isfinite(eval_res["loss"]) and 0.0 <= eval_res["iou"] <= 1.0


# ===========================================================================
# Issue #25: Seed-reproducible D4 dihedral geometric data augmentations
# ===========================================================================


def test_issue25_dihedral_transforms_equivariance_reproducibility_and_resume():
    ds = _SyntheticFloodDataset(n_samples=6, size=32)
    img0, mask0 = ds[0]

    # Check all 8 dihedral group elements preserve exact (img, mask) correspondence
    seen_orientations = set()
    for k in range(8):
        img_k, mask_k = tc.apply_dihedral_transform(img0, mask0, k)
        assert img_k.shape == img0.shape
        assert mask_k.shape == mask0.shape
        expected_mask = (img_k[0:1] > 0.0).float()
        assert torch.equal(mask_k, expected_mask), f"Mask misaligned for dihedral k={k}"
        seen_orientations_key = tuple(img_k[0, :4, :4].flatten().tolist())
        seen_orientations.add(seen_orientations_key)
    assert len(seen_orientations) == 8, "Expected 8 distinct D4 orientations on asymmetric input"

    # Seed reproducibility across two independent loaders + rng_state resume
    cfg = {"augment": True, "seed": 42}
    aug_ds_1 = tc.maybe_wrap_augment(Subset(ds, [0, 1, 2, 3]), cfg, seed=42)
    aug_ds_2 = tc.maybe_wrap_augment(Subset(ds, [0, 1, 2, 3]), cfg, seed=42)
    loader_1 = tc.make_loader(aug_ds_1, batch_size=2, shuffle=True, seed=42)
    loader_2 = tc.make_loader(aug_ds_2, batch_size=2, shuffle=True, seed=42)

    b1_epoch1 = list(loader_1)
    saved_state = tc.rng_state(loader_1)
    b1_epoch2 = list(loader_1)

    b2_epoch1 = list(loader_2)
    for (i1, m1), (i2, m2) in zip(b1_epoch1, b2_epoch1):
        assert torch.equal(i1, i2)
        assert torch.equal(m1, m2)

    # Restore loader_2 from saved_state after epoch 1 and verify epoch 2 matches bit-for-bit
    tc.restore_rng_state(saved_state, loader_2)
    b2_epoch2 = list(loader_2)
    for (i1, m1), (i2, m2) in zip(b1_epoch2, b2_epoch2):
        assert torch.equal(i1, i2)
        assert torch.equal(m1, m2)

    # unwrap_dataset strips DihedralAugmentDataset so eval loaders are never augmented
    unwrapped = tc.unwrap_dataset(aug_ds_1)
    assert not isinstance(unwrapped, tc.DihedralAugmentDataset)
    raw_img, raw_mask = unwrapped[0]
    orig_img, orig_mask = ds[0]
    assert torch.equal(raw_img, orig_img) and torch.equal(raw_mask, orig_mask)


# ===========================================================================
# Variant tags, fingerprints, and metrics.json recording
# ===========================================================================


def test_wave2_variant_tags_fingerprints_and_metrics_json(tmp_path):
    default_cfg = {"seed": 42}
    assert tc.variant_tag(default_cfg) == "_reupload_L2"
    assert tc.classical_variant_tag(default_cfg) == ""

    wave2_q_cfg = {
        "seed": 42,
        "q_ansatz": "hqunet",
        "q_layers": 2,
        "hqunet_grid": 4,
        "ablation_mode": "hqunet_spatial",
        "backbone_conv_mode": "depthwise_sep",
        "loss": "dice_ce",
        "use_time_emb": False,
        "augment": True,
        "fed_rounds": 5,
        "local_epochs": 1,
        "num_clients": 2,
        "batch_size": 4,
        "learning_rate": 1e-4,
    }
    vtag = tc.variant_tag(wave2_q_cfg)
    assert "_hqunet_G4_L2" in vtag
    assert "_abl-hqunet_spatial" in vtag
    assert "_depthwise_sep" in vtag
    assert "_dice_ce" in vtag
    assert "_notime" in vtag
    assert "_aug" in vtag

    ctag = tc.classical_variant_tag(wave2_q_cfg)
    assert "_depthwise_sep" in ctag
    assert "_dice_ce" in ctag
    assert "_notime" in ctag
    assert "_aug" in ctag

    fp_c = tc.centralized_fingerprint(wave2_q_cfg, "quantum")
    fp_f = tc.run_fingerprint(wave2_q_cfg, "quantum")
    for fp in (fp_c, fp_f):
        assert fp["hqunet_grid"] == 4
        assert fp["ablation_mode"] is None
        assert fp["backbone_conv_mode"] == "depthwise_sep"
        assert fp["loss"] == "dice_ce"
        assert fp["use_time_emb"] is False
        assert fp["augment"] is True

    fp_abl_c = tc.centralized_fingerprint(wave2_q_cfg, "ablation")
    fp_abl_f = tc.run_fingerprint(wave2_q_cfg, "ablation")
    for fp in (fp_abl_c, fp_abl_f):
        assert fp["ablation_mode"] == "hqunet_spatial"

    mpath = tc.write_metrics_json(
        str(tmp_path),
        script="centralized_demoTorchQ.py",
        training="centralized",
        model_type="quantum",
        config=wave2_q_cfg,
        val_metrics={"loss": 0.5, "accuracy": 0.8, "auc": 0.85, "iou": 0.6, "dice": 0.75},
        test_metrics={"loss": 0.4, "accuracy": 0.82, "auc": 0.87, "iou": 0.62, "dice": 0.77},
        history=[],
        total_params=1234,
    )
    with open(mpath) as f:
        rec = json.load(f)
    assert rec["hqunet_grid"] == 4
    assert rec["ablation_mode"] is None
    assert rec["backbone_conv_mode"] == "depthwise_sep"
    assert rec["loss"] == "dice_ce"
    assert rec["use_time_emb"] is False
    assert rec["augment"] is True

    mpath_abl = tc.write_metrics_json(
        str(tmp_path / "abl"),
        script="ablation_studies.py",
        training="centralized",
        model_type="ablation",
        config=wave2_q_cfg,
        val_metrics={"loss": 0.5, "accuracy": 0.8, "auc": 0.85, "iou": 0.6, "dice": 0.75},
        test_metrics={"loss": 0.4, "accuracy": 0.82, "auc": 0.87, "iou": 0.62, "dice": 0.77},
        history=[],
        total_params=1234,
    )
    with open(mpath_abl) as f:
        rec_abl = json.load(f)
    assert rec_abl["ablation_mode"] == "hqunet_spatial"


def test_wave2_review_fixes_shared_hqunet_filters_and_grid4_pennylane():
    # Finding 1: Shared 4-param horizontal F_R and 4-param vertical F_C per layer (8*L params for both G=2 and G=4)
    for grid in (2, 4):
        qb = QuantumBlock(
            quantum_channels=8,
            backend="pennylane",
            q_ansatz="hqunet",
            q_layers=2,
            hqunet_grid=grid,
            q_backend="pennylane",
            q_sim_device="cpu",
            seed=88,
        )
        cb = ClassicalBottleneckBlock(
            quantum_channels=8,
            q_ansatz="hqunet",
            q_layers=2,
            hqunet_grid=grid,
            seed=88,
        )
        assert qb.hqunet_weights.shape == (2, 2, 4)
        assert qb.hqunet_weights.numel() == 16
        assert cb.hqunet_classical_weights.shape == (2, 2, 4)
        assert cb.hqunet_classical_weights.numel() == 16

        # Finding 3: Both G=2 and G=4 use PennyLane default.qubit when q_backend='pennylane'
        sim_info = qb.simulator_info()
        assert sim_info["q_backend"] == "pennylane"
        assert sim_info["device"] == "default.qubit"
        assert sim_info["sim_device"] == "cpu"

        qb_torch = QuantumBlock(
            quantum_channels=8,
            backend="pennylane",
            q_ansatz="hqunet",
            q_layers=2,
            hqunet_grid=grid,
            q_backend="torch",
            seed=88,
        )
        x = torch.randn(2, 8, 4, 4)
        with torch.no_grad():
            y_pl = qb(x)
            y_th = qb_torch(x)
        assert torch.allclose(y_pl, y_th, atol=1e-5)


def test_wave2_review_fixes_standard_vs_ws_conv_and_lowrank_knockout():
    from utilsTorch.unetTorch import WeightStandardizedConv

    base_cfg = {
        "image_size": 32,
        "base_channels": 8,
        "init_dim": 8,
        "out_dim": 2,
        "dim_mults": (1, 2),
        "resnet_block_groups": 4,
        "seed": 42,
    }
    u_ws = tc.build_model(UNetSegmentation, {**base_cfg, "backbone_conv_mode": "ws_conv"})
    u_std = tc.build_model(UNetSegmentation, {**base_cfg, "backbone_conv_mode": "standard"})

    assert any(isinstance(m, WeightStandardizedConv) for m in u_ws.model.down.modules())
    assert not any(isinstance(m, WeightStandardizedConv) for m in u_std.model.down.modules())
    assert tc.classical_variant_tag({"backbone_conv_mode": "standard"}) == "_standard"

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_ws = u_ws(x)
        out_std = u_std(x)
    assert not torch.allclose(out_ws, out_std)

    # Finding 6: lowrank random knockout must not register a submodule or alter parameter count / state_dict
    cb = ClassicalBottleneckBlock(
        quantum_channels=8,
        q_ansatz="reupload",
        q_layers=2,
        ablation_mode="lowrank",
        seed=42,
    )
    orig_param_count = sum(p.numel() for p in cb.parameters())
    orig_sd_keys = set(cb.state_dict().keys())
    cb.set_quantum_mode("random", seed=99)
    assert sum(p.numel() for p in cb.parameters()) == orig_param_count
    assert set(cb.state_dict().keys()) == orig_sd_keys
    y_rand = cb(torch.randn(2, 8, 4, 4))
    assert y_rand.shape == (2, 8, 4, 4)
    cb.set_quantum_mode("on")
    assert sum(p.numel() for p in cb.parameters()) == orig_param_count
    assert set(cb.state_dict().keys()) == orig_sd_keys


def test_wave2_review_fixes_ablation_hqunet_grid_and_legacy_checkpoint_resume(tmp_path):
    # Finding 4: --ablation-mode hqunet_spatial with q_ansatz='reupload' must distinguish hqunet_grid=2 vs 4
    cfg_g2 = {
        "seed": 42,
        "q_ansatz": "reupload",
        "q_layers": 2,
        "ablation_mode": "hqunet_spatial",
        "hqunet_grid": 2,
        "fed_rounds": 4,
        "local_epochs": 1,
        "num_clients": 2,
        "batch_size": 2,
        "learning_rate": 1e-4,
    }
    cfg_g4 = {**cfg_g2, "hqunet_grid": 4}
    assert tc.variant_tag(cfg_g2) != tc.variant_tag(cfg_g4)
    assert "_abl-hqunet_spatial_G2" in tc.variant_tag(cfg_g2)
    assert "_abl-hqunet_spatial_G4" in tc.variant_tag(cfg_g4)
    assert tc.centralized_fingerprint(cfg_g2, "ablation")["hqunet_grid"] == 2
    assert tc.centralized_fingerprint(cfg_g4, "ablation")["hqunet_grid"] == 4
    assert tc.run_fingerprint(cfg_g2, "ablation")["hqunet_grid"] == 2
    assert tc.run_fingerprint(cfg_g4, "ablation")["hqunet_grid"] == 4

    # Finding 7: Pre-Wave-2 checkpoint fingerprint migration
    default_q_cfg = {
        "seed": 42,
        "q_ansatz": "reupload",
        "q_layers": 2,
        "q_backend": "pennylane",
        "num_epochs": 5,
        "fed_rounds": 4,
        "local_epochs": 1,
        "num_clients": 2,
        "batch_size": 2,
        "learning_rate": 1e-4,
    }
    full_c_fp = tc.centralized_fingerprint(default_q_cfg, "quantum")
    pre_wave2_c_fp = {
        k: v
        for k, v in full_c_fp.items()
        if k
        not in (
            "q_spatial_mode",
            "hqunet_grid",
            "ablation_mode",
            "backbone_conv_mode",
            "loss",
            "use_time_emb",
            "augment",
        )
    }
    ckpt = {
        "run_format": tc.RUN_FORMAT,
        "q_backend": "pennylane",
        "fingerprint": pre_wave2_c_fp,
    }
    assert tc.checkpoint_matches_centralized(ckpt, default_q_cfg, "quantum")
    assert not tc.checkpoint_matches_centralized(
        ckpt, {**default_q_cfg, "loss": "dice_ce"}, "quantum"
    )

    # Finding 8: Worker seeding for DihedralAugmentDataset
    ds = _SyntheticFloodDataset(n_samples=6, size=16)
    aug_ds = tc.DihedralAugmentDataset(ds, seed=42)
    loader_mp = tc.make_loader(aug_ds, batch_size=2, shuffle=True, seed=42, num_workers=2)
    batches = list(loader_mp)
    assert len(batches) == 3

    # Finding 9: model.loss_mode attribute takes precedence over model.config['loss']
    u_model = tc.build_model(
        UNetSegmentation,
        {
            "image_size": 16,
            "base_channels": 8,
            "init_dim": 8,
            "out_dim": 2,
            "dim_mults": (1, 2),
            "resnet_block_groups": 4,
            "loss": "ce",
            "seed": 42,
        },
    )
    u_model.loss_mode = "focal_dice"
    from utilsTorch.utilsTorch import _resolve_model_loss_mode
    assert _resolve_model_loss_mode(u_model) == "focal_dice"


def test_wave2_review_fixes_analyze_results_variant_separation(tmp_path):
    import analyze_results as ar

    base_cfg = {
        "seed": 42,
        "base_channels": 32,
        "image_size": 128,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 5,
    }
    val_m = {"loss": 0.4, "accuracy": 0.85, "auc": 0.9, "iou": 0.7, "dice": 0.82}

    # 1. Quantum default (reupload_L2, global) vs spatial pixel vs hqunet G2 vs hqunet G4
    configs = [
        (
            "flood_central_optical_pytorch_pennylane_reupload_L2_seed42",
            "quantum",
            {**base_cfg, "q_ansatz": "reupload", "q_layers": 2, "q_spatial_mode": "global"},
            0.70,
        ),
        (
            "flood_central_optical_pytorch_pennylane_reupload_L2_pixel_seed42",
            "quantum",
            {**base_cfg, "q_ansatz": "reupload", "q_layers": 2, "q_spatial_mode": "pixel"},
            0.75,
        ),
        (
            "flood_central_optical_pytorch_pennylane_hqunet_G2_L2_seed42",
            "quantum",
            {**base_cfg, "q_ansatz": "hqunet", "q_layers": 2, "hqunet_grid": 2},
            0.78,
        ),
        (
            "flood_central_optical_pytorch_pennylane_hqunet_G4_L2_seed42",
            "quantum",
            {**base_cfg, "q_ansatz": "hqunet", "q_layers": 2, "hqunet_grid": 4},
            0.81,
        ),
        (
            "flood_central_optical_pytorch_classical_seed42",
            "classical",
            {**base_cfg},
            0.68,
        ),
        (
            "flood_central_optical_pytorch_classical_depthwise_sep_dice_ce_notime_aug_seed42",
            "classical",
            {
                **base_cfg,
                "backbone_conv_mode": "depthwise_sep",
                "loss": "dice_ce",
                "use_time_emb": False,
                "augment": True,
            },
            0.74,
        ),
    ]
    for dirname, mtype, cfg, test_iou in configs:
        tc.write_metrics_json(
            str(tmp_path / dirname),
            script="centralized_demoTorchQ.py" if mtype == "quantum" else "centralized_demoTorchClassical.py",
            training="centralized",
            model_type=mtype,
            config=cfg,
            val_metrics=val_m,
            test_metrics={**val_m, "iou": test_iou},
            history=[],
            total_params=1000,
        )

    df = ar.load_from_disk(results_dir=str(tmp_path))
    assert len(df) == 6
    assert len(df["Method"].unique()) == 6
    dedup = ar.per_seed_summary(df)
    assert len(dedup) == 6
    assert set(np.round(dedup["Test_IoU"].values, 2)) == {0.68, 0.70, 0.74, 0.75, 0.78, 0.81}


def test_wave2_review_r2_hqunet_ablation_spatial_and_legacy_hqunet_layers(tmp_path):
    import analyze_results as ar

    base_cfg = {
        "seed": 42,
        "base_channels": 32,
        "image_size": 128,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "num_epochs": 5,
        "fed_rounds": 4,
        "local_epochs": 1,
        "num_clients": 2,
    }
    val_m = {"loss": 0.4, "accuracy": 0.85, "auc": 0.9, "iou": 0.7, "dice": 0.82}

    # 1. Finding 5 edge case: q_ansatz='hqunet' + ablation_mode='mlp' with q_spatial_mode='global' vs 'pixel'
    cfg_hq_mlp_global = {
        **base_cfg,
        "q_ansatz": "hqunet",
        "q_layers": 2,
        "hqunet_grid": 2,
        "ablation_mode": "mlp",
        "q_spatial_mode": "global",
    }
    cfg_hq_mlp_pixel = {**cfg_hq_mlp_global, "q_spatial_mode": "pixel"}
    vtag_global = tc.variant_tag(cfg_hq_mlp_global)
    vtag_pixel = tc.variant_tag(cfg_hq_mlp_pixel)
    assert vtag_global != vtag_pixel
    assert "_pixel_" in vtag_pixel

    # 2. Finding N1: q_ansatz='legacy' + ablation_mode='hqunet_spatial' with q_layers=1 vs 2
    cfg_leg_hq_l1 = {
        **base_cfg,
        "q_ansatz": "legacy",
        "q_layers": 1,
        "hqunet_grid": 2,
        "ablation_mode": "hqunet_spatial",
    }
    cfg_leg_hq_l2 = {**cfg_leg_hq_l1, "q_layers": 2}
    assert tc.variant_tag(cfg_leg_hq_l1) != tc.variant_tag(cfg_leg_hq_l2)
    assert "_abl-hqunet_spatial_G2_L1" in tc.variant_tag(cfg_leg_hq_l1)
    assert "_abl-hqunet_spatial_G2_L2" in tc.variant_tag(cfg_leg_hq_l2)

    fp_c_l1 = tc.centralized_fingerprint(cfg_leg_hq_l1, "ablation")
    fp_c_l2 = tc.centralized_fingerprint(cfg_leg_hq_l2, "ablation")
    assert fp_c_l1["q_layers"] == 1
    assert fp_c_l2["q_layers"] == 2
    ckpt_l1 = {"run_format": tc.RUN_FORMAT, "fingerprint": fp_c_l1}
    assert tc.checkpoint_matches_centralized(ckpt_l1, cfg_leg_hq_l1, "ablation")
    assert not tc.checkpoint_matches_centralized(ckpt_l1, cfg_leg_hq_l2, "ablation")

    fp_f_l1 = tc.run_fingerprint(cfg_leg_hq_l1, "ablation")
    fp_f_l2 = tc.run_fingerprint(cfg_leg_hq_l2, "ablation")
    assert fp_f_l1["q_layers"] == 1
    assert fp_f_l2["q_layers"] == 2
    fed_dir = tmp_path / "fed_leg_hq"
    tc.atomic_torch_save(
        {"run_format": tc.RUN_FORMAT, "fingerprint": fp_f_l1, "completed_round": 2},
        tc.fed_checkpoint_path(str(fed_dir)),
    )
    assert tc.load_fed_checkpoint(str(fed_dir), fp_f_l1, resume=True) is not None
    assert tc.load_fed_checkpoint(str(fed_dir), fp_f_l2, resume=True) is None

    # 3. Verify analyze_results.py separates all 4 ablation runs cleanly
    ablation_runs = [
        (
            f"ablation_centralized_torch_bottleneck_matched_pennylane{vtag_global}_seed42",
            cfg_hq_mlp_global,
            0.70,
        ),
        (
            f"ablation_centralized_torch_bottleneck_matched_pennylane{vtag_pixel}_seed42",
            cfg_hq_mlp_pixel,
            0.90,
        ),
        (
            f"ablation_centralized_torch_bottleneck_matched_pennylane{tc.variant_tag(cfg_leg_hq_l1)}_seed42",
            cfg_leg_hq_l1,
            0.65,
        ),
        (
            f"ablation_centralized_torch_bottleneck_matched_pennylane{tc.variant_tag(cfg_leg_hq_l2)}_seed42",
            cfg_leg_hq_l2,
            0.85,
        ),
    ]
    res_dir = tmp_path / "results_r2"
    for dirname, cfg, test_iou in ablation_runs:
        tc.write_metrics_json(
            str(res_dir / dirname),
            script="ablation_studies.py",
            training="centralized",
            model_type="ablation",
            config=cfg,
            val_metrics=val_m,
            test_metrics={**val_m, "iou": test_iou},
            history=[],
            total_params=1000,
        )

    df = ar.load_from_disk(results_dir=str(res_dir))
    assert len(df) == 4
    dedup = ar.per_seed_summary(df)
    assert len(dedup) == 4
    assert set(np.round(dedup["Test_IoU"].values, 2)) == {0.65, 0.70, 0.85, 0.90}
