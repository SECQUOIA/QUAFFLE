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
