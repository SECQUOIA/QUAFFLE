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

Outputs go to --out-dir (default: $QUAFFLE_FIGURES_DIR/exemplars, else
<repo>/figures/exemplars/):
  exemplars_grid.png  -- the exemplar figure
  meta.json           -- which sample index was picked for each modality and its IoU

Checkpoints are read from $QUAFFLE_RESULTS_DIR (default <repo>/results) and the
test tiles from <repo>/data. Run from anywhere:
  python scripts/build_exemplar_figure.py [--out-dir DIR]
"""
import argparse
import os
import re
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS_DIR = os.environ.get("QUAFFLE_RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
OUT_DIR = os.path.join(os.environ.get("QUAFFLE_FIGURES_DIR", os.path.join(REPO_ROOT, "figures")),
                       "exemplars")
DATA_ROOT = os.environ.get("QUAFFLE_DATA_ROOT", os.path.join(REPO_ROOT, "data"))

# QUAFFLE federated-quantum checkpoints (seed 999, random partition) for each dataset.
# Pooled Opt.+SAR ("combined") row is not included: the QUAFFLE prediction
# from the combined checkpoint did not align with the ground-truth flood mask cleanly enough
# to serve as a representative exemplar. The single-modality optical and SAR rows remain.
# Result dirs now carry an optional _qc<N> tag (when QC != 32) and a quantum
# variant tag (e.g. _reupload_L2, _legacy) before _seed<N>.
# QUAFFLE_EXEMPLAR_QC and QUAFFLE_EXEMPLAR_VARIANT configure both the checkpoint
# path lookup and the QVUNetSegmentation architecture in CONFIG.
EXEMPLAR_QC = int(os.environ.get("QUAFFLE_EXEMPLAR_QC", "32"))
EXEMPLAR_VARIANT = os.environ.get("QUAFFLE_EXEMPLAR_VARIANT", "reupload_L2")
EXEMPLAR_SEED = int(os.environ.get("QUAFFLE_EXEMPLAR_SEED", "999"))


def _parse_exemplar_variant(variant_str: str):
    v = (variant_str or "reupload_L2").strip()
    if v == "legacy" or v.startswith("legacy_"):
        return "legacy", 1
    m_hq = re.match(r"^hqunet(?:_G(\d+))?_L(\d+)(?:_.*)?$", v)
    if m_hq:
        return "hqunet", int(m_hq.group(2))
    if v == "hqunet":
        return "hqunet", 2
    m = re.match(r"^([A-Za-z0-9]+)_L(\d+)(?:_.*)?$", v)
    if m:
        return m.group(1), int(m.group(2))
    if v == "reupload":
        return "reupload", 2
    return v, 2


_EXEMPLAR_ANSATZ, _EXEMPLAR_LAYERS = _parse_exemplar_variant(EXEMPLAR_VARIANT)


def _normalize_ds_suffix(dataset_or_suffix: str) -> str:
    s = (dataset_or_suffix or "").strip().lower()
    if s in ("", "optical"):
        return ""
    if s in ("sar", "_sar"):
        return "_sar"
    if s in ("combined", "_combined"):
        return "_combined"
    return s if s.startswith("_") else f"_{s}"


def _canonical_vtag(variant_str: str = None) -> str:
    v = EXEMPLAR_VARIANT if variant_str is None else variant_str
    if not v:
        return ""
    v_clean = v.strip().lstrip("_")
    if not v_clean:
        return ""
    if v_clean == "legacy" or v_clean.startswith("legacy_"):
        return f"_{v_clean}"
    if v_clean == "hqunet":
        return "_hqunet_G2_L2"
    m_hq = re.match(r"^hqunet(?:_G(\d+))?_L(\d+)(.*)$", v_clean)
    if m_hq:
        grid = int(m_hq.group(1)) if m_hq.group(1) else 2
        layers = int(m_hq.group(2))
        suffix = m_hq.group(3) or ""
        return f"_hqunet_G{grid}_L{layers}{suffix}"
    if v_clean == "reupload":
        return "_reupload_L2"
    m = re.match(r"^([A-Za-z0-9]+)_L(\d+)(.*)$", v_clean)
    if m:
        return f"_{m.group(1)}_L{int(m.group(2))}{m.group(3) or ''}"
    return f"_{v_clean}"


def _apply_variant_to_config(cfg: dict, variant_str: str) -> dict:
    out = dict(cfg)
    ansatz, layers = _parse_exemplar_variant(variant_str)
    out["q_ansatz"] = ansatz
    out["q_layers"] = layers
    v = (variant_str or "").strip().lstrip("_")
    m_hq = re.match(r"^hqunet_G(\d+)_L(\d+)", v)
    if m_hq:
        out["hqunet_grid"] = int(m_hq.group(1))
    tokens = set(v.split("_"))
    for sm in ("pixel", "patch2x2", "patch4x4", "se_gate"):
        if sm in tokens or f"_{sm}" in f"_{v}":
            out["q_spatial_mode"] = sm
            break
    if "addskip" in tokens:
        out["skip_modulation"] = "additive"
    if "depthwise_sep" in v:
        out["backbone_conv_mode"] = "depthwise_sep"
    if "notime" in tokens:
        out["use_time_emb"] = False
    return out


def _find_ckpt_in_dir(run_dir: str):
    """Return the preferred checkpoint path inside `run_dir`, or None."""
    if not os.path.isdir(run_dir):
        return None
    # Prefer highest checkpoint_round_<N>.pt, then fed_resume.pt, then centralized checkpoints
    round_ckpts = []
    try:
        for fname in os.listdir(run_dir):
            m = re.match(r"^checkpoint_round_(\d+)\.pt$", fname)
            if m:
                round_ckpts.append((int(m.group(1)), os.path.join(run_dir, fname)))
    except OSError:
        pass
    if round_ckpts:
        round_ckpts.sort(key=lambda x: x[0], reverse=True)
        return round_ckpts[0][1]
    for name in ("checkpoint_round_25.pt", "fed_resume.pt", "best_model.pt", "checkpoint.pt", "final_model.pt"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p
    return None


_SCALE_AND_SEED_RE = re.compile(
    r"(?:_qtorch|_qpergroup)?"
    r"(?:_bc\d+)?(?:_img\d+)?(?:_max\d+|_n\d+)?(?:_dr[0-9a-f]{6,16})?"
    r"_seed(?P<seed>\d+)$"
)


def _fed_ckpt(*args, seed=None, results_dir=None, qc=None, variant=None):
    explicit_seed = seed is not None
    explicit_qc = qc is not None
    explicit_variant = variant is not None
    if seed is None:
        seed = EXEMPLAR_SEED
    if len(args) == 1:
        ds_suffix = _normalize_ds_suffix(args[0])
    elif len(args) == 2:
        if isinstance(args[1], int):
            ds_suffix = _normalize_ds_suffix(args[0])
            seed = int(args[1])
            explicit_seed = True
        else:
            ds_suffix = _normalize_ds_suffix(args[1])
    elif len(args) >= 3:
        ds_suffix = _normalize_ds_suffix(args[1])
        seed = int(args[2])
        explicit_seed = True
    else:
        ds_suffix = ""
    eff_results_dir = results_dir or RESULTS_DIR
    eff_qc = EXEMPLAR_QC if qc is None else int(qc)
    eff_variant = EXEMPLAR_VARIANT if variant is None else variant
    is_default_qc = (not explicit_qc) and (eff_qc == 32) and ("QUAFFLE_EXEMPLAR_QC" not in os.environ)
    is_default_variant = (
        (not explicit_variant)
        and (eff_variant == "reupload_L2")
        and ("QUAFFLE_EXEMPLAR_VARIANT" not in os.environ)
    )
    qc_suffix = "" if eff_qc == 32 else f"_qc{eff_qc}"
    vtag = _canonical_vtag(eff_variant)
    raw_vtag = "" if not eff_variant else f"_{eff_variant.strip().lstrip('_')}"
    base = os.path.join(
        eff_results_dir,
        f"flood_federated_optical_pytorch_pennylane_random{ds_suffix}{qc_suffix}",
    )
    candidates = [f"{base}{vtag}_seed{seed}"]
    if raw_vtag and raw_vtag != vtag:
        candidates.append(f"{base}{raw_vtag}_seed{seed}")
    if is_default_variant:
        candidates.append(f"{base}_seed{seed}")
    for d in candidates:
        found = _find_ckpt_in_dir(d)
        if found:
            return found

    # Fallback discovery for scale-tagged mini-dataset runs in eff_results_dir
    if os.path.isdir(eff_results_dir):
        fed_prefix = f"flood_federated_optical_pytorch_pennylane_random{ds_suffix}"
        cent_prefix = f"flood_central_optical_pytorch_pennylane{ds_suffix}"
        other_ds = ("_sar", "_combined") if ds_suffix == "" else ()
        try:
            entries = sorted(os.listdir(eff_results_dir))
        except OSError:
            entries = []

        def _parse_entry(entry_name: str, prefix: str):
            if not entry_name.startswith(prefix):
                return None, None
            rest = entry_name[len(prefix):]
            for od in other_ds:
                if rest.startswith(od):
                    return None, None
            m = _SCALE_AND_SEED_RE.search(rest)
            if not m:
                return None, None
            var_part = rest[:m.start()]
            return var_part, int(m.group("seed"))

        expected_var_parts = [f"{qc_suffix}{vtag}"]
        if raw_vtag and f"{qc_suffix}{raw_vtag}" not in expected_var_parts:
            expected_var_parts.append(f"{qc_suffix}{raw_vtag}")
        if is_default_variant and qc_suffix not in expected_var_parts:
            expected_var_parts.append(qc_suffix)

        # Pass 1: exact (qc, variant) match on scale-tagged federated or centralized runs
        for prefix in (fed_prefix, cent_prefix):
            for want_var in expected_var_parts:
                # 1a. Exact seed match
                for entry in entries:
                    var_part, entry_seed = _parse_entry(entry, prefix)
                    if var_part == want_var and entry_seed == seed:
                        found = _find_ckpt_in_dir(os.path.join(eff_results_dir, entry))
                        if found:
                            return found
                # 1b. Any seed if seed was not explicitly forced
                if not explicit_seed:
                    for entry in entries:
                        var_part, entry_seed = _parse_entry(entry, prefix)
                        if var_part == want_var and entry_seed is not None:
                            found = _find_ckpt_in_dir(os.path.join(eff_results_dir, entry))
                            if found:
                                return found

        # Pass 2: only when neither --qc nor --variant (nor QUAFFLE_EXEMPLAR_*) was explicitly requested,
        # fall back to an unambiguous single available variant for that modality
        if is_default_qc and is_default_variant:
            for prefix in (fed_prefix, cent_prefix):
                seed_matches = []
                any_matches = []
                for entry in entries:
                    var_part, entry_seed = _parse_entry(entry, prefix)
                    if var_part is None:
                        continue
                    found = _find_ckpt_in_dir(os.path.join(eff_results_dir, entry))
                    if not found:
                        continue
                    any_matches.append((var_part, found))
                    if entry_seed == seed:
                        seed_matches.append((var_part, found))
                if len({vp for vp, _ in seed_matches}) == 1:
                    return seed_matches[0][1]
                if not explicit_seed and not seed_matches and len({vp for vp, _ in any_matches}) == 1:
                    return any_matches[0][1]

    return os.path.join(candidates[0], "checkpoint_round_25.pt")


CHECKPOINTS = {
    "optical":  _fed_ckpt(""),
    "sar":      _fed_ckpt("_sar"),
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
    "quantum_channels": EXEMPLAR_QC,
    "quantum_backend": "pennylane",
    "q_ansatz": _EXEMPLAR_ANSATZ,
    "q_layers": _EXEMPLAR_LAYERS,
}

sys.path.insert(0, REPO_ROOT)
from tools import training_common as tc
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


def _load_checkpoint_into_model(model, ckpt):
    """Load a centralized or federated checkpoint dict/state_dict into `model`.
    Prefers `best_state_dict` over `model_state_dict` and `best_parameters` over
    `parameters` when present and non-None."""
    # Ensure lazy QVUNetSegmentation wrapper is initialized before parameter copy
    _ = sum(p.numel() for p in model.parameters())
    if isinstance(ckpt, dict) and ("best_state_dict" in ckpt or "model_state_dict" in ckpt):
        state = (
            ckpt["best_state_dict"]
            if ckpt.get("best_state_dict") is not None
            else ckpt.get("model_state_dict")
        )
    elif isinstance(ckpt, dict) and ("best_parameters" in ckpt or "parameters" in ckpt):
        # fed_resume.pt stores the global parameters as a list of arrays
        # in model.parameters() order (prefer best_parameters when available).
        params_list = (
            ckpt["best_parameters"]
            if ckpt.get("best_parameters") is not None
            else ckpt.get("parameters")
        )
        with torch.no_grad():
            for p, arr in zip(model.parameters(), params_list):
                t_arr = torch.as_tensor(arr, dtype=p.dtype, device=p.device)
                p.copy_(t_arr)
        state = None
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    if state is not None:
        model.load_state_dict(state)
    return model


def _resolve_model_config_for_checkpoint(ckpt, ckpt_path=None, base_config=None):
    """Build a QVUNetSegmentation config dict matching the saved checkpoint architecture.

    Reads architecture keys from sibling `results/metrics.json` (including top-level
    fields written by `write_metrics_json`), sibling `fed_resume.pt` (when loading a
    periodic `checkpoint_round_*.pt`), and `ckpt['fingerprint']` / `ckpt['config']`,
    falling back to `base_config` / `CONFIG` for legacy checkpoints.
    """
    cfg = dict(base_config or CONFIG)
    sources = []
    if ckpt_path:
        run_dir = os.path.dirname(os.path.abspath(ckpt_path))
        metrics_path = os.path.join(run_dir, "results", "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    mdata = json.load(f)
                if isinstance(mdata, dict):
                    sources.append(mdata)
                    if isinstance(mdata.get("config"), dict):
                        sources.append(mdata["config"])
                    if isinstance(mdata.get("fingerprint"), dict):
                        sources.append(mdata["fingerprint"])
            except Exception:
                pass
        # If loading checkpoint_round_*.pt (which has no fingerprint), also check sibling fed_resume.pt
        if not (isinstance(ckpt, dict) and isinstance(ckpt.get("fingerprint"), dict)):
            resume_path = os.path.join(run_dir, "fed_resume.pt")
            if os.path.exists(resume_path) and os.path.abspath(resume_path) != os.path.abspath(ckpt_path):
                try:
                    rckpt = tc.safe_torch_load(resume_path, map_location="cpu")
                    if isinstance(rckpt, dict):
                        if isinstance(rckpt.get("config"), dict):
                            sources.append(rckpt["config"])
                        if isinstance(rckpt.get("fingerprint"), dict):
                            sources.append(rckpt["fingerprint"])
                except Exception:
                    pass
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("config"), dict):
            sources.append(ckpt["config"])
        if isinstance(ckpt.get("fingerprint"), dict):
            sources.append(ckpt["fingerprint"])

    int_keys = (
        "image_size",
        "base_channels",
        "init_dim",
        "resnet_block_groups",
        "quantum_channels",
        "q_layers",
        "hqunet_grid",
    )
    str_keys = (
        "q_ansatz",
        "q_spatial_mode",
        "skip_modulation",
        "backbone_conv_mode",
    )
    bool_keys = ("use_time_emb",)

    explicit_init_dim = False
    explicit_groups = False
    for src in sources:
        for k in int_keys:
            if src.get(k) is not None:
                cfg[k] = int(src[k])
                if k == "init_dim":
                    explicit_init_dim = True
                elif k == "resnet_block_groups":
                    explicit_groups = True
        for k in str_keys:
            if src.get(k) is not None:
                cfg[k] = str(src[k])
        for k in bool_keys:
            if src.get(k) is not None:
                cfg[k] = bool(src[k])

    bc = int(cfg.get("base_channels", 32))
    if not explicit_init_dim:
        cfg["init_dim"] = bc
    if not explicit_groups:
        cfg["resnet_block_groups"] = tc._compatible_group_count(
            bc, int(cfg.get("resnet_block_groups", 4))
        )
    return cfg


def main(argv=None):
    global OUT_DIR
    ap = argparse.ArgumentParser(description="Build the exemplar figure.")
    ap.add_argument("--out-dir", "--out", dest="out_dir", default=OUT_DIR,
                    help="Output folder (default %(default)s)")
    ap.add_argument("--results-dir", default=RESULTS_DIR,
                    help="Results folder containing checkpoints (default %(default)s)")
    ap.add_argument("--data-root", default=DATA_ROOT,
                    help="Data root containing flood_optical/ and flood_SAR/ (default %(default)s)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Checkpoint seed to load (default: $QUAFFLE_EXEMPLAR_SEED or 999, with mini-run fallback)")
    ap.add_argument("--qc", type=int, default=None,
                    help="Quantum channels (default: $QUAFFLE_EXEMPLAR_QC or 32)")
    ap.add_argument("--variant", default=None,
                    help="Quantum variant tag, e.g. reupload_L2, legacy, hqunet_G2_L2 (default: $QUAFFLE_EXEMPLAR_VARIANT or reupload_L2)")
    args = ap.parse_args(argv)
    OUT_DIR = os.path.abspath(args.out_dir)
    results_dir = os.path.abspath(args.results_dir)
    data_root = os.path.abspath(args.data_root)
    eff_qc = EXEMPLAR_QC if args.qc is None else int(args.qc)
    eff_variant = EXEMPLAR_VARIANT if args.variant is None else args.variant
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_cfg = _apply_variant_to_config(
        dict(CONFIG, quantum_channels=eff_qc),
        eff_variant,
    )
    checkpoints = {
        "optical": _fed_ckpt("", seed=args.seed, results_dir=results_dir, qc=args.qc, variant=args.variant),
        "sar": _fed_ckpt("_sar", seed=args.seed, results_dir=results_dir, qc=args.qc, variant=args.variant),
    }
    test_dirs = {
        "optical": (os.path.join(data_root, "flood_optical/Testing/images"),
                    os.path.join(data_root, "flood_optical/Testing/labels")),
        "sar": (os.path.join(data_root, "flood_SAR/Testing/images"),
                os.path.join(data_root, "flood_SAR/Testing/labels")),
    }

    picks = {}
    for modality, ckpt_path in checkpoints.items():
        print(f"\n=== {modality.upper()} ===")
        if not os.path.exists(ckpt_path):
            print(f"  MISSING checkpoint: {ckpt_path}")
            continue
        imgs_dir, masks_dir = test_dirs[modality]
        if not os.path.isdir(imgs_dir) or not os.path.isdir(masks_dir):
            print(f"  MISSING test split: {imgs_dir}")
            continue
        print(f"  Test dir: {imgs_dir}")
        ckpt = tc.safe_torch_load(ckpt_path, map_location=device)
        model_cfg = _resolve_model_config_for_checkpoint(ckpt, ckpt_path=ckpt_path, base_config=base_cfg)
        ds = FloodSegmentationDataset(imgs_dir, masks_dir, model_cfg["image_size"], data_root=data_root)
        print(f"  Loaded {len(ds)} test samples")
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

        model = QVUNetSegmentation(model_cfg).to(device)
        _load_checkpoint_into_model(model, ckpt)
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
