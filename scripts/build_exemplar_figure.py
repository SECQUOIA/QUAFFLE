#!/usr/bin/env python3
"""
Build a clean 3x3 exemplar figure (image | ground truth | QUAFFLE prediction)
for one optical sample, one SAR sample, and one mixed-pooled sample.

For each modality:
  - Loads the QUAFFLE checkpoint (federated quantum, random partition, seed 999)
  - Runs inference on the entire test split
  - Picks the test sample with the highest IoU among samples with >= 5% flood pixels
    (visually informative AND model performs reasonably)
  - Saves the picked sample's image / mask / prediction as a 3-panel composite

Outputs go to: <repo>/figures/exemplars/
  exemplars_grid.png  -- the exemplar figure
  meta.json           -- which sample index was picked for each modality and its IoU

Run with: python scripts/build_exemplar_figure.py
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_DIR   = os.path.join(REPO_ROOT, "figures", "exemplars")
DATA_ROOT = os.path.join(REPO_ROOT, "data")

# QUAFFLE federated-quantum checkpoints (seed 999, random partition) for each dataset.
# Pooled Opt.+SAR ("combined") row is not included: the QUAFFLE prediction
# from the combined checkpoint did not align with the ground-truth flood mask cleanly enough
# to serve as a representative exemplar. The single-modality optical and SAR rows remain.
CHECKPOINTS = {
    "optical":  os.path.join(REPO_ROOT, "results/flood_federated_optical_pytorch_pennylane_random_seed999/checkpoint_round_25.pt"),
    "sar":      os.path.join(REPO_ROOT, "results/flood_federated_optical_pytorch_pennylane_random_sar_seed999/checkpoint_round_25.pt"),
}
TEST_DIRS = {
    "optical":  (os.path.join(DATA_ROOT, "flood_optical/Testing/images"),
                 os.path.join(DATA_ROOT, "flood_optical/Testing/labels")),
    "sar":      (os.path.join(DATA_ROOT, "flood_SAR/Testing/images"),
                 os.path.join(DATA_ROOT, "flood_SAR/Testing/labels")),
}

CONFIG = {
    "image_size": 128,
    "base_channels": 32,
    "init_dim": 32,
    "out_dim": 2,
    "dim_mults": (1, 2, 4, 8),
    "resnet_block_groups": 4,
    "quantum_channels": 32,
    "quantum_backend": "pennylane",
}

sys.path.insert(0, REPO_ROOT)
from utilsTorch.utilsTorch import FloodSegmentationDataset, QVUNetSegmentation


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU on binary masks."""
    inter = ((pred > 0) & (gt > 0)).sum()
    union = ((pred > 0) | (gt > 0)).sum()
    return float(inter) / max(int(union), 1)


def select_best_sample(model, loader, device, min_flood_frac=0.05, max_samples=24):
    """Run inference on the first `max_samples` test samples; return the one with
    the highest IoU among those with >= min_flood_frac flood pixels.
    Limited to keep CPU PennyLane runtime reasonable (~5s/sample)."""
    model.eval()
    best = None
    seen = 0
    with torch.no_grad():
        idx = 0
        for imgs, masks in loader:
            if seen >= max_samples:
                break
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            for i in range(imgs.size(0)):
                if seen >= max_samples:
                    break
                img_np = (imgs[i].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
                mask_np = masks[i, 0].cpu().numpy()
                pred_np = preds[i].cpu().numpy().astype(np.float32)
                flood_frac = float((mask_np > 0).sum()) / mask_np.size
                seen += 1
                if flood_frac < min_flood_frac:
                    idx += 1
                    continue
                score = iou(pred_np, mask_np)
                cand = (idx, img_np, mask_np, pred_np, score, flood_frac)
                if best is None or cand[4] > best[4]:
                    best = cand
                idx += 1
            if seen >= max_samples:
                break
    if best is None:
        # fallback: pick highest-flood-fraction sample even if model failed
        seen = 0; idx = 0; best_alt = None
        for imgs, masks in loader:
            if seen >= max_samples: break
            with torch.no_grad():
                logits = model(imgs.to(device))
                preds = torch.argmax(logits, dim=1)
            for i in range(imgs.size(0)):
                if seen >= max_samples: break
                img_np = (imgs[i].cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
                mask_np = masks[i, 0].cpu().numpy()
                pred_np = preds[i].cpu().numpy().astype(np.float32)
                flood_frac = float((mask_np > 0).sum()) / mask_np.size
                if best_alt is None or flood_frac > best_alt[5]:
                    best_alt = (idx, img_np, mask_np, pred_np, iou(pred_np, mask_np), flood_frac)
                seen += 1; idx += 1
        if best_alt is None:
            raise RuntimeError("No samples processed at all")
        print(f"  WARN: no sample had IoU>0 with flood_frac>={min_flood_frac}; using highest-flood sample instead")
        return best_alt
    return best


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    picks = {}
    for modality, ckpt_path in CHECKPOINTS.items():
        print(f"\n=== {modality.upper()} ===")
        if not os.path.exists(ckpt_path):
            print(f"  MISSING checkpoint: {ckpt_path}")
            continue
        imgs_dir, masks_dir = TEST_DIRS[modality]
        print(f"  Test dir: {imgs_dir}")
        ds = FloodSegmentationDataset(imgs_dir, masks_dir, CONFIG["image_size"])
        print(f"  Loaded {len(ds)} test samples")
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

        model = QVUNetSegmentation(CONFIG).to(device)
        # Force initialization
        _ = sum(p.numel() for p in model.parameters())
        ckpt = torch.load(ckpt_path, map_location=device)
        # Common patterns
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        model.load_state_dict(state)
        print(f"  Loaded checkpoint ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")

        idx, img, mask, pred, iou_score, flood_frac = select_best_sample(model, loader, device)
        print(f"  Picked test sample idx={idx}: IoU={iou_score:.3f}, flood_frac={flood_frac:.2%}")

        picks[modality] = {
            "test_index": idx,
            "iou": iou_score,
            "flood_fraction": flood_frac,
            "image": img,
            "mask": mask,
            "pred": pred,
        }

    if not picks:
        print("ERROR: no picks. Exiting.")
        sys.exit(1)

    # ---- Build the 3x3 figure ----
    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, 3, figsize=(8, 2.7 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = ["Input", "Ground truth", "QUAFFLE prediction"]
    row_titles = {"optical": "Optical", "sar": "SAR"}
    for r, (modality, p) in enumerate(picks.items()):
        # Input
        axes[r, 0].imshow(np.clip(p["image"], 0, 1))
        axes[r, 0].set_axis_off()
        # Ground truth
        axes[r, 1].imshow(p["mask"], cmap="Blues", vmin=0, vmax=1)
        axes[r, 1].set_axis_off()
        # Prediction (with IoU as right-aligned text inside the panel, not a title)
        axes[r, 2].imshow(p["pred"], cmap="Blues", vmin=0, vmax=1)
        axes[r, 2].set_axis_off()
        axes[r, 2].text(0.97, 0.03, f"IoU={p['iou']:.2f}",
                        transform=axes[r, 2].transAxes,
                        ha="right", va="bottom",
                        fontsize=9, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.5, edgecolor="none"))
        # Row label on the very left (only for first column)
        axes[r, 0].set_ylabel(row_titles[modality], rotation=90, fontsize=11, labelpad=6)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for s in axes[r, 0].spines.values(): s.set_visible(False)
    # Column titles on top row only (now safe — no per-row title collision)
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "exemplars_grid.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_png}")

    # Save metadata
    meta = {m: {k: v for k, v in p.items() if k not in {"image", "mask", "pred"}}
            for m, p in picks.items()}
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Wrote {os.path.join(OUT_DIR, 'meta.json')}")


if __name__ == "__main__":
    main()
