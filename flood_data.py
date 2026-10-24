"""Framework-agnostic data loading for the DFC 2024 flood tiles.

Used by utilsTorch.FloodSegmentationDataset and utilsJAX.load_dataset.

Two modes:
  clean (default)
    * honors the committed split manifest splits/<dataset>.json for
      flood_optical, flood_SAR and flood_combined (drops test tiles that are
      pixel-identical to a training tile and collapses duplicates, excluding
      conflicting-mask duplicate groups by default)
    * explicit band selection by name (see BAND_NAMES, DEFAULT_BANDS)
    * dataset-level normalization with per-band statistics computed on the
      fit-only training split (splits/norm_stats.json), mapped to [-1, 1]
  legacy
    * the original behavior: every file in the folder, first three bands,
      per-image 2-98 percentile stretch. Reproduces pre-fix numbers.

Select legacy with legacy=True on the loader functions, with
set_data_defaults(legacy=True), or with the environment variable
QUAFFLE_LEGACY_DATA=1.
"""

from collections import defaultdict
import glob
import hashlib
import json
import os
import re

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(REPO_ROOT, "splits")
STATS_PATH = os.path.join(SPLITS_DIR, "norm_stats.json")

MANIFEST_DATASETS = ("flood_optical", "flood_SAR", "flood_combined")
SPLIT_FOLDERS = {"Training": "train", "Testing": "test"}

# Band order of the DFC 2024 tiles, from the official contest page
# (grss-ieee.org, 2024 IEEE GRSS Data Fusion Contest, "format of the data
# sets" figure) and checked against value ranges in the local tiles
# (see docs/DATA.md).
BAND_NAMES = {
    "sar": ["VV", "VH", "MERIT_DEM", "COP_DEM", "WORLDCOVER", "WATER_OCC"],
    "optical": ["COASTAL", "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2",
                "QA", "MERIT_DEM", "COP_DEM", "WORLDCOVER", "WATER_OCC"],
}
MODALITY_BY_COUNT = {6: "sar", 12: "optical"}

# Three-channel defaults (in_channels=3 is unchanged).
DEFAULT_BANDS = {
    "sar": ["VV", "VH", "COP_DEM"],
    "optical": ["GREEN", "NIR", "SWIR1"],
}

# Bands that get log1p before scaling (heavy tailed, saturate at 32765).
LOG_BANDS = {"VV", "VH"}
# Values treated as missing (excluded from statistics, mapped to the low end).
NODATA = -9999
STAT_PERCENTILES = (1.0, 99.0)
STRIDE = {"sar": 8, "optical": 2}
DEFAULT_VAL_FRACTION = 0.15
FLOOD_LABEL_THRESHOLD = 0.05

_DEFAULTS = {"legacy": None, "bands": None, "data_root": None}
_FIT_STATS_CACHE = {}
_LAST_TRAIN_STATS_BY_DIR = {}
_LAST_LOADED_PROVENANCE_BY_DIR = {}


def get_last_loaded_stats_provenance(images_dir):
    if images_dir is None:
        return None
    return _LAST_LOADED_PROVENANCE_BY_DIR.get(os.path.abspath(images_dir))


def _fit_names_digest(names):
    ordered = sorted((str(n) for n in names), key=natural_key)
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()[:16]


def _stats_band_digest(stats_dict):
    if not isinstance(stats_dict, dict):
        return None
    parts = []
    for mod in sorted(k for k in stats_dict.keys() if not k.startswith("_") and k != "by_seed"):
        mval = stats_dict.get(mod)
        if not isinstance(mval, dict) or "bands" not in mval:
            continue
        for bname in sorted(mval["bands"].keys()):
            b = mval["bands"][bname]
            parts.append(f"{mod}:{bname}:{b.get('lo')}:{b.get('hi')}:{b.get('log1p')}")
    if not parts:
        return None
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def stats_provenance(stats_dict):
    """Extract a canonical provenance dict from a loaded normalization stats dict."""
    if not isinstance(stats_dict, dict):
        return {"fit_only": False, "source": "none"}
    prov = dict(stats_dict.get("_provenance") or {})
    n_tiles = None
    for mod in ("optical", "sar"):
        if isinstance(stats_dict.get(mod), dict) and "n_train_tiles" in stats_dict[mod]:
            n_tiles = int(stats_dict[mod]["n_train_tiles"])
            break
    fit_only = bool(prov.get("fit_only", False))
    return {
        "mode": "fit_only_train_split" if fit_only else "full_train_split",
        "fit_only": fit_only,
        "seed": prov.get("seed"),
        "val_fraction": prov.get("val_fraction"),
        "image_size": prov.get("image_size"),
        "max_samples": prov.get("max_samples"),
        "n_manifest_train_tiles": prov.get("n_manifest_train_tiles", n_tiles),
        "n_fit_train_tiles": prov.get("n_fit_train_tiles", n_tiles),
        "n_excluded_val_tiles": prov.get("n_excluded_val_tiles", 0),
        "fit_names_sha256": prov.get("fit_names_sha256"),
        "stats_sha256": _stats_band_digest(stats_dict),
    }


def set_data_defaults(legacy=None, bands=None, data_root=None):
    """Set process-wide defaults (e.g. from a --legacy-data CLI flag)."""
    if legacy is not None:
        _DEFAULTS["legacy"] = bool(legacy)
    if bands is not None:
        _DEFAULTS["bands"] = bands
    if data_root is not None:
        _DEFAULTS["data_root"] = data_root


def resolve_legacy(legacy=None):
    if legacy is not None:
        return bool(legacy)
    if _DEFAULTS["legacy"] is not None:
        return _DEFAULTS["legacy"]
    return os.environ.get("QUAFFLE_LEGACY_DATA", "0").lower() in ("1", "true", "yes")


def resolve_bands(bands=None):
    """Return {modality: [band names]} from None, a dict, or a flat list
    (a flat list applies to every modality whose band set contains it)."""
    bands = bands if bands is not None else _DEFAULTS["bands"]
    out = {m: list(v) for m, v in DEFAULT_BANDS.items()}
    if bands is None:
        return out
    if isinstance(bands, str):
        bands = [b.strip() for b in bands.split(",") if b.strip()]
    if isinstance(bands, dict):
        for m, v in bands.items():
            out[m] = list(v)
    else:
        for m in out:
            if all(b in BAND_NAMES[m] for b in bands):
                out[m] = list(bands)
    for m, v in out.items():
        bad = [b for b in v if b not in BAND_NAMES[m]]
        if bad:
            raise ValueError(f"Unknown {m} band(s) {bad}; valid: {BAND_NAMES[m]}")
    return out


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _derive_seed(*parts):
    ss = np.random.SeedSequence([int(p) for p in parts])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def split_train_val_indices(n, val_fraction=DEFAULT_VAL_FRACTION, seed=42, labels=None, order=None):
    """Deterministic stratified train/val index split matching training_common.holdout_validation."""
    n = int(n)
    if order is None:
        order = np.arange(n, dtype=np.int64)
    else:
        order = np.asarray(order, dtype=np.int64)
    if n <= 1 or val_fraction is None or float(val_fraction) <= 0.0:
        return sorted(order.tolist()), []
    if labels is None:
        labels = np.zeros(n, dtype=np.int64)
    else:
        labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(_derive_seed(seed, 1))
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = order[labels[order] == c]
        idx = rng.permutation(idx)
        n_val = int(round(float(val_fraction) * len(idx)))
        if len(idx) > 1:
            n_val = min(max(n_val, 1), len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def find_mask(masks_dir, name):
    for ext in [".png", ".jpg", ".tif"]:
        p = os.path.join(masks_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def list_pairs(images_dir, masks_dir, sort=True):
    """[(name, image_path, mask_path)] for every image that has a mask."""
    paths = glob.glob(os.path.join(images_dir, "*.tif")) + \
        glob.glob(os.path.join(images_dir, "*.tiff"))
    pairs = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        m = find_mask(masks_dir, name)
        if m is not None:
            pairs.append((name, p, m))
    if sort:
        pairs.sort(key=lambda t: natural_key(t[0]))
    return pairs


def infer_dataset_split(images_dir):
    """('flood_SAR', 'train') from .../flood_SAR/Training/images, else (None, None)."""
    parts = os.path.normpath(os.path.abspath(images_dir)).split(os.sep)
    if len(parts) >= 3 and parts[-2] in SPLIT_FOLDERS:
        return parts[-3], SPLIT_FOLDERS[parts[-2]]
    return None, None


def _resolve_splits_dir(images_dir=None, data_root=None):
    """Locate the splits/ directory associated with data_root or images_dir,
    falling back to the repository's committed splits/ directory."""
    candidates = []
    eff_root = data_root if data_root is not None else _DEFAULTS.get("data_root")
    if eff_root is not None:
        abs_root = os.path.abspath(eff_root)
        candidates.append(os.path.join(abs_root, "splits"))
        candidates.append(os.path.join(os.path.dirname(abs_root), "splits"))
    if images_dir is not None:
        cur = os.path.abspath(images_dir)
        for _ in range(5):
            cur = os.path.dirname(cur)
            if not cur or cur == os.sep:
                break
            candidates.append(os.path.join(cur, "splits"))
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return SPLITS_DIR


def _infer_data_root_from_images_dir(images_dir, data_root=None):
    if data_root is not None:
        return data_root
    if _DEFAULTS.get("data_root") is not None:
        return _DEFAULTS["data_root"]
    if images_dir is None:
        return None
    parts = os.path.normpath(os.path.abspath(images_dir)).split(os.sep)
    if len(parts) >= 4 and parts[-2] in SPLIT_FOLDERS:
        return os.sep.join(parts[:-3]) or os.sep
    return None


def _resolve_training_dirs(images_dir=None, masks_dir=None, dataset=None, data_root=None):
    """Resolve `(train_images_dir, train_masks_dir)` across parent-level data_root,
    dataset-level data_root, Training/images, or sibling Testing/images."""
    if images_dir is not None:
        ds_from_img, split_kind = infer_dataset_split(images_dir)
        if dataset is None:
            dataset = ds_from_img
        abs_img = os.path.abspath(images_dir)
        if split_kind == "train" and os.path.isdir(abs_img):
            lbl = (
                os.path.abspath(masks_dir)
                if masks_dir is not None
                else os.path.join(os.path.dirname(abs_img), "labels")
            )
            return abs_img, lbl
        if split_kind == "test":
            ds_dir = os.path.dirname(os.path.dirname(abs_img))
            cand_img = os.path.join(ds_dir, "Training", "images")
            cand_lbl = os.path.join(ds_dir, "Training", "labels")
            if os.path.isdir(cand_img) and os.path.isdir(cand_lbl):
                return cand_img, cand_lbl

    eff_root = _infer_data_root_from_images_dir(images_dir, data_root=data_root)
    if eff_root is not None:
        abs_root = os.path.abspath(eff_root)
        candidates = []
        if dataset:
            candidates.append(
                (
                    os.path.join(abs_root, dataset, "Training", "images"),
                    os.path.join(abs_root, dataset, "Training", "labels"),
                )
            )
        candidates.append(
            (
                os.path.join(abs_root, "Training", "images"),
                os.path.join(abs_root, "Training", "labels"),
            )
        )
        for cand_img, cand_lbl in candidates:
            if os.path.isdir(cand_img) and os.path.isdir(cand_lbl):
                return cand_img, cand_lbl
    return None, None


def load_manifest(dataset, splits_dir=None, data_root=None, images_dir=None):
    eff_root = _infer_data_root_from_images_dir(images_dir, data_root=data_root)
    if splits_dir is None and eff_root is not None:
        abs_root = os.path.abspath(eff_root)
        for local_cand in (
            os.path.join(abs_root, dataset, "manifest.json"),
            os.path.join(abs_root, "manifest.json") if os.path.basename(abs_root) == dataset else "",
        ):
            if local_cand and os.path.exists(local_cand):
                with open(local_cand) as f:
                    return json.load(f)
    sdir = splits_dir or _resolve_splits_dir(images_dir=images_dir, data_root=data_root)
    path = os.path.join(sdir, f"{dataset}.json")
    if not os.path.exists(path) and sdir != SPLITS_DIR:
        fallback = os.path.join(SPLITS_DIR, f"{dataset}.json")
        if os.path.exists(fallback):
            path = fallback
    with open(path) as f:
        return json.load(f)


def _filter_conflicting_from_manifest_hashes(names_set, manifest, split, conflict_policy="drop"):
    if conflict_policy != "drop":
        return names_set
    hashes = (manifest.get("hashes") or {}).get(split)
    if not hashes:
        return names_set
    groups = defaultdict(set)
    name_to_pix = {}
    for r in hashes:
        groups[r["image_pix"]].add(r["mask_pix"])
        name_to_pix[r["name"]] = r["image_pix"]
    conflicting = {
        name for name, pix in name_to_pix.items() if len(groups[pix]) > 1
    }
    return names_set - conflicting


def manifest_names(images_dir, manifest=None, data_root=None, conflict_policy=None):
    """Names allowed by the manifest for this folder, or None if no manifest
    applies. Raises if a known flood dataset has no manifest file."""
    if manifest is False:
        return None
    dataset, split = infer_dataset_split(images_dir)
    if isinstance(manifest, str):
        with open(manifest) as f:
            manifest = json.load(f)
    if manifest is None:
        if dataset not in MANIFEST_DATASETS:
            return None
        eff_root = _infer_data_root_from_images_dir(images_dir, data_root=data_root)
        local_manifest = None
        if eff_root is not None:
            abs_root = os.path.abspath(eff_root)
            for cand in (
                os.path.join(abs_root, dataset, "manifest.json"),
                os.path.join(abs_root, "manifest.json") if os.path.basename(abs_root) == dataset else "",
            ):
                if cand and os.path.exists(cand):
                    local_manifest = cand
                    break
        if local_manifest is not None:
            path = local_manifest
        else:
            sdir = _resolve_splits_dir(images_dir=images_dir, data_root=data_root)
            path = os.path.join(sdir, f"{dataset}.json")
            if not os.path.exists(path) and sdir != SPLITS_DIR:
                path = os.path.join(SPLITS_DIR, f"{dataset}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Split manifest {path} missing. Run tools/dedup_splits.py or use legacy=True.")
        with open(path) as f:
            manifest = json.load(f)
    if split is None:
        raise ValueError(f"Cannot infer split (Training/Testing) from {images_dir}")
    allowed = set(manifest[split])
    policy = conflict_policy or manifest.get("conflict_policy", "drop")
    return _filter_conflicting_from_manifest_hashes(allowed, manifest, split, conflict_policy=policy)


def read_raw(img_path):
    import rasterio
    with rasterio.open(img_path) as src:
        return src.read()


def load_mask(mask_path, image_size):
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    return (mask_array > 0.001).astype(np.float32)[..., np.newaxis]


def _slice_indices_preload(items, max_samples, sample_seed=42, name_fn=None):
    """Deterministically select `max_samples` indices in canonical order.

    When `items` contains both non-' copy' and ' copy' stems (as in
    `flood_combined` where optical and SAR tiles share stems via ' copy'),
    samples proportionally from both groups when `max_samples >= 2` so capped
    combined splits retain both modalities.
    """
    n = len(items)
    if max_samples is None or int(max_samples) <= 0 or int(max_samples) >= n:
        return list(range(n))
    k = int(max_samples)
    eff_seed = 42 if sample_seed is None else int(sample_seed)
    get_name = name_fn or (lambda x: x)
    canon_order = sorted(range(n), key=lambda i: os.path.basename(str(get_name(items[i]))))
    rng = np.random.default_rng(_derive_seed(eff_seed, 7))
    primary_idx = [i for i in canon_order if " copy" not in os.path.basename(str(get_name(items[i])))]
    copy_idx = [i for i in canon_order if " copy" in os.path.basename(str(get_name(items[i])))]
    if primary_idx and copy_idx and k >= 2:
        k_primary = min(len(primary_idx), max(1, (k + 1) // 2))
        k_copy = min(len(copy_idx), max(1, k - k_primary))
        perm_p = rng.permutation(primary_idx).tolist()
        perm_c = rng.permutation(copy_idx).tolist()
        chosen = perm_p[:k_primary] + perm_c[:k_copy]
        if len(chosen) < k:
            rem = perm_p[k_primary:] + perm_c[k_copy:]
            chosen.extend(rem[: k - len(chosen)])
        return sorted(chosen)
    return sorted(rng.permutation(canon_order)[:k].tolist())


def compute_fit_only_names(
    images_dir,
    masks_dir,
    train_names,
    *,
    seed=None,
    val_fraction=None,
    train_indices=None,
    include_names=None,
    exclude_names=None,
    image_size=128,
    max_samples=None,
    sample_seed=None,
    masks_by_name=None,
):
    """Return the fit-only subset of `train_names` (excluding validation holdout tiles)."""
    ordered = sorted(list(train_names), key=natural_key)
    eff_image_size = 128 if image_size is None else int(image_size)
    if max_samples is not None and int(max_samples) > 0 and int(max_samples) < len(ordered):
        keep_idx = _slice_indices_preload(ordered, max_samples, sample_seed=sample_seed)
        ordered = [ordered[i] for i in keep_idx]

    if train_indices is not None:
        idx_set = {int(i) for i in train_indices}
        selected = [n for i, n in enumerate(ordered) if i in idx_set]
    elif seed is not None or (val_fraction is not None and float(val_fraction) > 0.0):
        eff_seed = 42 if seed is None else int(seed)
        eff_vf = DEFAULT_VAL_FRACTION if val_fraction is None else float(val_fraction)
        if eff_vf > 0.0 and len(ordered) > 1:
            labels = []
            for n in ordered:
                if masks_by_name is not None and n in masks_by_name:
                    m = masks_by_name[n]
                    labels.append(1 if float(m.mean()) > FLOOD_LABEL_THRESHOLD else 0)
                else:
                    mp = find_mask(masks_dir, n) if masks_dir else None
                    if mp is not None:
                        m = load_mask(mp, eff_image_size)
                        labels.append(1 if float(m.mean()) > FLOOD_LABEL_THRESHOLD else 0)
                    else:
                        labels.append(0)
            # Match training_common.canonical_order sorting (lexicographic on str(name))
            canon_order = sorted(range(len(ordered)), key=lambda i: os.path.basename(str(ordered[i])))
            tr_idx, _ = split_train_val_indices(
                len(ordered),
                val_fraction=eff_vf,
                seed=eff_seed,
                labels=np.asarray(labels, dtype=np.int64),
                order=canon_order,
            )
            tr_set = set(tr_idx)
            selected = [n for i, n in enumerate(ordered) if i in tr_set]
        else:
            selected = list(ordered)
    else:
        selected = list(ordered)

    if include_names is not None:
        inc = set(include_names)
        selected = [n for n in selected if n in inc]
    if exclude_names is not None:
        exc = set(exclude_names)
        selected = [n for n in selected if n not in exc]
    return selected


def _default_modality_band_stats(modality, provenance=None):
    """Physical prior band bounds when a combined split's tiny capped fit subset
    contains zero tiles of `modality` and no precomputed stats file exists."""
    bands = {}
    for b in BAND_NAMES[modality]:
        if modality == "sar" and b in LOG_BANDS:
            bands[b] = {
                "lo": -28.0,
                "hi": -4.0,
                "mean": -15.0,
                "std": 5.0,
                "log1p": False,
                "auto_scale_from_prior": True,
            }
        elif b in ("MERIT_DEM", "COP_DEM"):
            bands[b] = {"lo": 0.0, "hi": 1000.0, "mean": 150.0, "std": 100.0, "log1p": False}
        elif b in ("WORLDCOVER", "WATER_OCC"):
            bands[b] = {"lo": 0.0, "hi": 100.0, "mean": 40.0, "std": 25.0, "log1p": False}
        else:
            bands[b] = {"lo": 0.0, "hi": 4000.0, "mean": 1000.0, "std": 600.0, "log1p": False}
    entry = {
        "n_train_tiles": 0,
        "pixel_stride": int(STRIDE[modality]),
        "n_pixels_per_band": 0,
        "bands": bands,
        "modality_fallback": "default_physical_prior_zero_fit_tiles",
    }
    if provenance:
        entry.update(provenance)
    return entry


def compute_modality_band_stats(
    images_dir,
    masks_dir,
    tile_names,
    *,
    stride_map=None,
    provenance=None,
    raw_by_name=None,
    sub_by_name=None,
    ensure_modalities=None,
    fallback_stats=None,
):
    """Compute per-modality, per-band (lo, hi, mean, std, log1p) over `tile_names`."""
    smap = stride_map or STRIDE
    buf = {}
    n_tiles = {}
    for name in tile_names:
        if sub_by_name is not None and name in sub_by_name:
            mod, sub = sub_by_name[name]
        else:
            if raw_by_name is not None and name in raw_by_name:
                raw = raw_by_name[name]
            else:
                p = os.path.join(images_dir, name + ".tif")
                if not os.path.exists(p):
                    p = os.path.join(images_dir, name + ".tiff")
                if masks_dir is not None:
                    assert find_mask(masks_dir, name) is not None, name
                raw = read_raw(p)
            mod = MODALITY_BY_COUNT[raw.shape[0]]
            s = max(1, min(int(smap[mod]), raw.shape[1], raw.shape[2]))
            sub = raw[:, ::s, ::s].reshape(raw.shape[0], -1)
        buf.setdefault(mod, []).append(sub)
        n_tiles[mod] = n_tiles.get(mod, 0) + 1
    out = {}
    for mod, arrs in buf.items():
        a = np.concatenate(arrs, axis=1)
        bands = {}
        for i, b in enumerate(BAND_NAMES[mod]):
            v = a[i].astype(np.float64)
            v = v[v != NODATA]
            if v.size == 0:
                v = np.zeros(1, dtype=np.float64)
            # Apply log1p to positive linear-amplitude SAR backscatter, but preserve
            # negative dB-scale SAR backscatter directly without clipping to 0.
            use_log1p = (b in LOG_BANDS) and not (v.size > 0 and float(np.median(v)) < 0.0)
            if use_log1p:
                v = np.log1p(np.clip(v, 0, None))
            lo, hi = np.percentile(v, STAT_PERCENTILES)
            bands[b] = {
                "lo": round(float(lo), 6),
                "hi": round(float(hi), 6),
                "mean": round(float(v.mean()), 6),
                "std": round(float(v.std()), 6),
                "log1p": bool(use_log1p),
            }
        entry = {
            "n_train_tiles": n_tiles[mod],
            "pixel_stride": int(max(1, min(int(smap[mod]), a.shape[1])) if a.ndim > 1 else smap[mod]),
            "n_pixels_per_band": int(a.shape[1]),
            "bands": bands,
        }
        if provenance:
            entry.update(provenance)
        out[mod] = entry
    if ensure_modalities:
        for req_mod in ensure_modalities:
            if req_mod not in out:
                if isinstance(fallback_stats, dict) and isinstance(fallback_stats.get(req_mod), dict):
                    fb_entry = dict(fallback_stats[req_mod])
                    if provenance:
                        fb_entry.update(provenance)
                    fb_entry["n_train_tiles"] = 0
                    fb_entry["modality_fallback"] = "precomputed_norm_stats_zero_fit_tiles"
                    out[req_mod] = fb_entry
                else:
                    out[req_mod] = _default_modality_band_stats(req_mod, provenance=provenance)
    if provenance:
        out["_provenance"] = dict(provenance)
    return out


def _entry_matches_fit_request(
    entry,
    *,
    eff_seed,
    eff_vf,
    eff_image_size,
    max_samples,
    expected_fit_digest=None,
    expected_pool_size=None,
):
    if not isinstance(entry, dict):
        return False
    prov = entry.get("_provenance")
    if not isinstance(prov, dict) or prov.get("fit_only") is not True:
        return False
    if prov.get("seed") != eff_seed:
        return False
    if prov.get("val_fraction") is None or abs(float(prov["val_fraction"]) - eff_vf) >= 1e-6:
        return False
    if max_samples is not None:
        if prov.get("max_samples") != int(max_samples):
            return False
    elif prov.get("max_samples") is not None:
        return False
    if prov.get("image_size") is None or int(prov["image_size"]) != int(eff_image_size):
        return False
    if prov.get("fit_names_sha256") is None:
        return False
    if expected_pool_size is not None:
        if prov.get("n_manifest_train_tiles") is None or int(prov["n_manifest_train_tiles"]) != int(expected_pool_size):
            return False
    if expected_fit_digest is not None:
        if prov.get("fit_names_sha256") != expected_fit_digest:
            return False
    return True


def load_stats(
    dataset,
    stats_path=None,
    *,
    seed=None,
    val_fraction=None,
    fit_only=None,
    data_root=None,
    images_dir=None,
    masks_dir=None,
    train_indices=None,
    exclude_names=None,
    image_size=None,
    max_samples=None,
    sample_seed=None,
    preloaded_train_names=None,
    preloaded_raw_by_name=None,
    preloaded_sub_by_name=None,
    preloaded_masks_by_name=None,
    _probe_only=False,
):
    sdir = _resolve_splits_dir(images_dir=images_dir, data_root=data_root)
    path = stats_path
    if path is None:
        eff_root = _infer_data_root_from_images_dir(images_dir, data_root=data_root)
        local_stats = (
            os.path.join(os.path.abspath(eff_root), "norm_stats.json")
            if eff_root is not None
            else None
        )
        cand = os.path.join(sdir, "norm_stats.json")
        if os.path.exists(cand):
            path = cand
        elif local_stats is not None and os.path.exists(local_stats):
            path = local_stats
        else:
            path = STATS_PATH

    stats = None
    if os.path.exists(path):
        with open(path) as f:
            stats = json.load(f)

    eff_seed = 42 if seed is None else int(seed)
    eff_vf = DEFAULT_VAL_FRACTION if val_fraction is None else float(val_fraction)
    eff_image_size = 128 if image_size is None else int(image_size)
    eff_sseed = eff_seed if sample_seed is None else int(sample_seed)
    want_fit_only = bool(
        fit_only
        or seed is not None
        or val_fraction is not None
        or train_indices is not None
        or exclude_names is not None
        or max_samples is not None
    )

    ds_stats = None
    if stats is not None:
        key = dataset if dataset in stats else ("_default" if "_default" in stats else None)
        if key is not None:
            ds_stats = stats[key]

    train_img_dir, train_lbl_dir = _resolve_training_dirs(
        images_dir=images_dir, masks_dir=masks_dir, dataset=dataset, data_root=data_root
    )
    _, split_kind = infer_dataset_split(images_dir) if images_dir else (None, None)

    # If loading Testing split after Training in the same process, reuse the paired
    # Training split's active stats when (seed, val_fraction, image_size) match.
    if (
        split_kind == "test"
        and train_img_dir is not None
        and train_indices is None
        and exclude_names is None
    ):
        paired = _LAST_TRAIN_STATS_BY_DIR.get((os.path.abspath(train_img_dir), dataset))
        if paired is not None:
            p_prov = paired.get("_provenance") or {}
            if (
                p_prov.get("fit_only") is True
                and p_prov.get("seed") == eff_seed
                and abs(float(p_prov.get("val_fraction", eff_vf)) - eff_vf) < 1e-6
                and int(p_prov.get("image_size", eff_image_size)) == eff_image_size
            ):
                return paired

    # Determine the actual training pool and fit-only names if local training tiles exist
    train_pool_names = None
    fit_names = None
    fit_digest = None
    if want_fit_only and fit_only is not False and train_img_dir is not None and os.path.isdir(train_img_dir):
        if preloaded_train_names is not None:
            train_pool_names = sorted(list(preloaded_train_names), key=natural_key)
        else:
            inferred_root = _infer_data_root_from_images_dir(train_img_dir, data_root=data_root)
            try:
                allowed = manifest_names(train_img_dir, data_root=inferred_root)
            except Exception:
                allowed = None
            pairs = list_pairs(train_img_dir, train_lbl_dir, sort=True)
            if allowed is not None:
                all_names = [p[0] for p in pairs if p[0] in allowed]
            else:
                all_names = [p[0] for p in pairs]
            if max_samples is not None and int(max_samples) > 0 and int(max_samples) < len(all_names):
                keep_idx = _slice_indices_preload(all_names, max_samples, sample_seed=eff_sseed)
                train_pool_names = [all_names[i] for i in keep_idx]
            else:
                train_pool_names = all_names
        if train_pool_names:
            fit_names = compute_fit_only_names(
                train_img_dir,
                train_lbl_dir,
                train_pool_names,
                seed=eff_seed,
                val_fraction=eff_vf,
                train_indices=train_indices,
                exclude_names=exclude_names,
                image_size=eff_image_size,
                masks_by_name=preloaded_masks_by_name,
            )
            fit_digest = _fit_names_digest(fit_names)

    # 1. Check precomputed per-seed fit-only entry in norm_stats.json
    if (
        ds_stats is not None
        and want_fit_only
        and fit_only is not False
        and train_indices is None
        and exclude_names is None
    ):
        by_seed = ds_stats.get("by_seed") or {}
        seed_keys = [
            f"seed_{eff_seed}_val_{eff_vf:.4f}",
            f"seed_{eff_seed}",
            str(eff_seed),
        ]
        for sk in seed_keys:
            if sk in by_seed:
                cand_entry = by_seed[sk]
                if _entry_matches_fit_request(
                    cand_entry,
                    eff_seed=eff_seed,
                    eff_vf=eff_vf,
                    eff_image_size=eff_image_size,
                    max_samples=max_samples,
                    expected_fit_digest=fit_digest,
                    expected_pool_size=len(train_pool_names) if train_pool_names is not None else None,
                ):
                    if split_kind == "train" and train_img_dir is not None:
                        _LAST_TRAIN_STATS_BY_DIR[(os.path.abspath(train_img_dir), dataset)] = cand_entry
                    return cand_entry
        if _entry_matches_fit_request(
            ds_stats,
            eff_seed=eff_seed,
            eff_vf=eff_vf,
            eff_image_size=eff_image_size,
            max_samples=max_samples,
            expected_fit_digest=fit_digest,
            expected_pool_size=len(train_pool_names) if train_pool_names is not None else None,
        ):
            if split_kind == "train" and train_img_dir is not None:
                _LAST_TRAIN_STATS_BY_DIR[(os.path.abspath(train_img_dir), dataset)] = ds_stats
            return ds_stats

    # 2. If fit-only stats are requested and local Training tiles exist on disk,
    #    compute fit-only stats excluding val_idx on the fly (cached).
    if want_fit_only and fit_only is not False and train_img_dir is not None and fit_names:
        cache_key = (
            os.path.abspath(train_img_dir),
            dataset,
            eff_seed,
            eff_vf,
            eff_image_size,
            None if max_samples is None else int(max_samples),
            eff_sseed if max_samples is not None else None,
            fit_digest,
        )
        if cache_key in _FIT_STATS_CACHE:
            cached = _FIT_STATS_CACHE[cache_key]
            if split_kind == "train":
                _LAST_TRAIN_STATS_BY_DIR[(os.path.abspath(train_img_dir), dataset)] = cached
            return cached
        if _probe_only and preloaded_raw_by_name is None and preloaded_sub_by_name is None:
            return None
        prov = {
            "fit_only": True,
            "seed": eff_seed,
            "val_fraction": eff_vf,
            "image_size": eff_image_size,
            "max_samples": int(max_samples) if max_samples is not None else None,
            "n_manifest_train_tiles": len(train_pool_names),
            "n_fit_train_tiles": len(fit_names),
            "n_excluded_val_tiles": len(train_pool_names) - len(fit_names),
            "fit_names_sha256": fit_digest,
        }
        req_mods = ("optical", "sar") if dataset in ("combined", "flood_combined") else None
        computed = compute_modality_band_stats(
            train_img_dir,
            train_lbl_dir,
            fit_names,
            provenance=prov,
            raw_by_name=preloaded_raw_by_name,
            sub_by_name=preloaded_sub_by_name,
            ensure_modalities=req_mods,
            fallback_stats=ds_stats,
        )
        _FIT_STATS_CACHE[cache_key] = computed
        if split_kind == "train":
            _LAST_TRAIN_STATS_BY_DIR[(os.path.abspath(train_img_dir), dataset)] = computed
        return computed

    if stats is None:
        raise FileNotFoundError(
            f"Normalization stats {path} missing. Run tools/compute_norm_stats.py or use legacy=True."
        )
    if ds_stats is None:
        raise KeyError(f"No normalization stats for {dataset} in {path}")
    if want_fit_only and fit_only is not False:
        raise KeyError(
            f"No matching fit-only normalization stats for dataset={dataset!r} "
            f"(seed={eff_seed}, val_fraction={eff_vf}, image_size={eff_image_size}, "
            f"max_samples={max_samples}) in {path}, and local training tiles were not found."
        )
    return ds_stats


def _resize_float(chw, image_size):
    out = np.empty((chw.shape[0], image_size, image_size), dtype=np.float32)
    for c in range(chw.shape[0]):
        im = Image.fromarray(np.ascontiguousarray(chw[c], dtype=np.float32))
        out[c] = np.asarray(im.resize((image_size, image_size), Image.LANCZOS))
    return out


def normalize_bands(raw, modality, band_names, mstats):
    """Select bands by name and map to [-1, 1] with training-split stats."""
    names = BAND_NAMES[modality]
    out = []
    for b in band_names:
        i = names.index(b)
        x = raw[i].astype(np.float32)
        s = mstats["bands"][b]
        nod = x == NODATA
        use_log1p = s.get("log1p", b in LOG_BANDS)
        lo, hi = s["lo"], s["hi"]
        if s.get("auto_scale_from_prior", False) and b in LOG_BANDS:
            valid = x[~nod]
            if valid.size > 0 and float(np.median(valid)) >= 0.0:
                use_log1p = True
                lo, hi = 0.0, 9.5
        if use_log1p:
            x = np.log1p(np.clip(x, 0, None))
        x = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
        x = np.clip(x, 0.0, 1.0)
        x[nod] = 0.0
        out.append(x * 2.0 - 1.0)
    return np.stack(out)


def _normalize_raw_clean(raw, image_size, bands, stats, img_path="<raw>"):
    modality = MODALITY_BY_COUNT.get(raw.shape[0])
    if modality is None:
        raise ValueError(f"{img_path}: unexpected band count {raw.shape[0]}")
    x = normalize_bands(raw, modality, bands[modality], stats[modality])
    x = np.clip(_resize_float(x, image_size), -1.0, 1.0)
    return np.transpose(x, (1, 2, 0))


def load_image_clean(img_path, image_size, bands, stats):
    """HWC float32 in [-1, 1]."""
    raw = read_raw(img_path)
    return _normalize_raw_clean(raw, image_size, bands, stats, img_path=img_path)


def load_image_legacy(img_path, image_size):
    """Original reader: first three bands, per-image percentile stretch."""
    try:
        import rasterio
        with rasterio.open(img_path) as src:
            img_data = src.read()[:3]
            img_array = np.transpose(img_data, (1, 2, 0))
            if img_array.dtype == np.uint8:
                img_array = img_array.astype(np.float32) / 255.0
            elif img_array.dtype == np.uint16:
                img_array = img_array.astype(np.float32) / 65535.0
            else:
                img_array = img_array.astype(np.float32)
                for c in range(3):
                    channel = img_array[:, :, c]
                    low, high = np.percentile(channel, [2, 98])
                    if high > low:
                        img_array[:, :, c] = np.clip((channel - low) / (high - low), 0, 1)
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            img = img.resize((image_size, image_size), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
    except ImportError:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((image_size, image_size), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array * 2.0 - 1.0


def _slice_pairs_preload(pairs, max_samples, sample_seed=42):
    """Deterministically slice `pairs` to at most `max_samples` BEFORE reading TIFFs."""
    if max_samples is None or int(max_samples) <= 0 or int(max_samples) >= len(pairs):
        return pairs
    keep_idx = _slice_indices_preload(pairs, max_samples, sample_seed=sample_seed, name_fn=lambda p: p[0])
    return [pairs[i] for i in keep_idx]


def load_samples(
    images_dir,
    masks_dir,
    image_size,
    legacy=None,
    bands=None,
    manifest=None,
    stats_path=None,
    progress=None,
    max_samples=None,
    sample_seed=42,
    data_root=None,
    seed=None,
    val_fraction=None,
    fit_only=None,
):
    """Return a list of (image HWC float32 in [-1,1], mask HW1 float32) and
    the list of sample names, in deterministic order.

    legacy:       True reproduces the original loader exactly (see module doc).
    bands:        dict {modality: [names]}, flat list, or comma string.
    manifest:     None (auto from path), False (no filtering), path, or dict.
    max_samples:  Optional int cap applied to the manifest-filtered pair list
                  BEFORE reading every TIFF from disk. Full-manifest validation
                  remains strict when max_samples is None.
    sample_seed:  Seed used for pre-load max_samples subset selection.
    """
    legacy = resolve_legacy(legacy)
    wrap = progress or (lambda it, **kw: it)
    samples, names = [], []
    eff_sample_seed = sample_seed if sample_seed is not None else (seed if seed is not None else 42)
    if legacy:
        pairs = list_pairs(images_dir, masks_dir, sort=False)
        if not pairs and not glob.glob(os.path.join(images_dir, "*.tif*")):
            raise ValueError(f"No TIFF images found in {images_dir}")
        pairs = _slice_pairs_preload(pairs, max_samples, sample_seed=eff_sample_seed)
        for name, img_path, mask_path in wrap(pairs, desc="Loading data (legacy)"):
            try:
                samples.append((load_image_legacy(img_path, image_size),
                                load_mask(mask_path, image_size)))
                names.append(name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        if not samples:
            raise ValueError("No valid image-mask pairs loaded")
        return samples, names

    pairs = list_pairs(images_dir, masks_dir, sort=True)
    if not pairs:
        raise ValueError(f"No TIFF image/mask pairs found in {images_dir}")
    allowed = manifest_names(images_dir, manifest, data_root=data_root)
    if allowed is not None:
        present = {p[0] for p in pairs}
        missing = allowed - present
        if missing and max_samples is None:
            raise ValueError(f"{len(missing)} manifest entries missing in {images_dir}, "
                             f"e.g. {sorted(missing)[:3]}")
        pairs = [p for p in pairs if p[0] in allowed]
        if not pairs:
            raise ValueError(f"No manifest-allowed image/mask pairs found in {images_dir}")
    pairs = _slice_pairs_preload(pairs, max_samples, sample_seed=eff_sample_seed)
    dataset, split_kind = infer_dataset_split(images_dir)
    band_sel = resolve_bands(bands)

    if split_kind != "train":
        stats = load_stats(
            dataset,
            stats_path,
            seed=seed,
            val_fraction=val_fraction,
            fit_only=fit_only,
            data_root=data_root,
            images_dir=images_dir,
            masks_dir=masks_dir,
            image_size=image_size,
            max_samples=None,
            sample_seed=eff_sample_seed,
        )
        _LAST_LOADED_PROVENANCE_BY_DIR[os.path.abspath(images_dir)] = stats_provenance(stats)
        for name, img_path, mask_path in wrap(pairs, desc="Loading data"):
            samples.append((load_image_clean(img_path, image_size, band_sel, stats),
                            load_mask(mask_path, image_size)))
            names.append(name)
        return samples, names

    pair_names = [p[0] for p in pairs]
    masks_by_name = {name: load_mask(mask_path, image_size) for name, _, mask_path in pairs}
    stats = load_stats(
        dataset,
        stats_path,
        seed=seed,
        val_fraction=val_fraction,
        fit_only=fit_only,
        data_root=data_root,
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        max_samples=max_samples,
        sample_seed=eff_sample_seed,
        preloaded_train_names=pair_names,
        preloaded_masks_by_name=masks_by_name,
        _probe_only=True,
    )
    if stats is not None or len(pairs) > 64:
        if stats is None:
            stats = load_stats(
                dataset,
                stats_path,
                seed=seed,
                val_fraction=val_fraction,
                fit_only=fit_only,
                data_root=data_root,
                images_dir=images_dir,
                masks_dir=masks_dir,
                image_size=image_size,
                max_samples=max_samples,
                sample_seed=eff_sample_seed,
                preloaded_train_names=pair_names,
                preloaded_masks_by_name=masks_by_name,
            )
        _LAST_LOADED_PROVENANCE_BY_DIR[os.path.abspath(images_dir)] = stats_provenance(stats)
        for name, img_path, _ in wrap(pairs, desc="Loading data"):
            samples.append((load_image_clean(img_path, image_size, band_sel, stats),
                            masks_by_name[name]))
            names.append(name)
        return samples, names

    # Single-pass path when stats are not precomputed and split is small/capped (<= 64 tiles):
    # reads each capped TIFF once and passes strided pixel subsamples to load_stats.
    raw_items = []
    sub_by_name = {}
    for name, img_path, _ in wrap(pairs, desc="Loading data"):
        raw = read_raw(img_path)
        mask = masks_by_name[name]
        raw_items.append((name, img_path, raw, mask))
        mod = MODALITY_BY_COUNT.get(raw.shape[0])
        if mod is not None:
            s = max(1, min(int(STRIDE[mod]), raw.shape[1], raw.shape[2]))
            sub_by_name[name] = (mod, raw[:, ::s, ::s].reshape(raw.shape[0], -1))

    stats = load_stats(
        dataset,
        stats_path,
        seed=seed,
        val_fraction=val_fraction,
        fit_only=fit_only,
        data_root=data_root,
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        max_samples=max_samples,
        sample_seed=eff_sample_seed,
        preloaded_train_names=pair_names,
        preloaded_sub_by_name=sub_by_name,
        preloaded_masks_by_name=masks_by_name,
    )
    _LAST_LOADED_PROVENANCE_BY_DIR[os.path.abspath(images_dir)] = stats_provenance(stats)
    for name, img_path, raw, mask in raw_items:
        samples.append((_normalize_raw_clean(raw, image_size, band_sel, stats, img_path=img_path), mask))
        names.append(name)
    return samples, names
