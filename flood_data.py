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


def load_manifest(dataset, splits_dir=None, data_root=None, images_dir=None):
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
):
    """Return the fit-only subset of `train_names` (excluding validation holdout tiles)."""
    ordered = sorted(list(train_names), key=natural_key)
    if train_indices is not None:
        idx_set = {int(i) for i in train_indices}
        selected = [n for i, n in enumerate(ordered) if i in idx_set]
    elif seed is not None or (val_fraction is not None and float(val_fraction) > 0.0):
        eff_seed = 42 if seed is None else int(seed)
        eff_vf = DEFAULT_VAL_FRACTION if val_fraction is None else float(val_fraction)
        if eff_vf > 0.0 and len(ordered) > 1:
            labels = []
            for n in ordered:
                mp = find_mask(masks_dir, n) if masks_dir else None
                if mp is not None:
                    m = load_mask(mp, image_size)
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


def compute_modality_band_stats(
    images_dir,
    masks_dir,
    tile_names,
    *,
    stride_map=None,
    provenance=None,
):
    """Compute per-modality, per-band (lo, hi, mean, std, log1p) over `tile_names`."""
    smap = stride_map or STRIDE
    buf = {}
    n_tiles = {}
    for name in tile_names:
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
            if b in LOG_BANDS:
                v = np.log1p(np.clip(v, 0, None))
            lo, hi = np.percentile(v, STAT_PERCENTILES)
            bands[b] = {
                "lo": round(float(lo), 6),
                "hi": round(float(hi), 6),
                "mean": round(float(v.mean()), 6),
                "std": round(float(v.std()), 6),
                "log1p": b in LOG_BANDS,
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
    if provenance:
        out["_provenance"] = dict(provenance)
    return out


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
):
    sdir = _resolve_splits_dir(images_dir=images_dir, data_root=data_root)
    path = stats_path
    if path is None:
        cand = os.path.join(sdir, "norm_stats.json")
        path = cand if os.path.exists(cand) else STATS_PATH

    stats = None
    if os.path.exists(path):
        with open(path) as f:
            stats = json.load(f)

    eff_vf = DEFAULT_VAL_FRACTION if val_fraction is None else float(val_fraction)
    want_fit_only = bool(
        fit_only
        or seed is not None
        or val_fraction is not None
        or train_indices is not None
        or exclude_names is not None
    )

    ds_stats = None
    if stats is not None:
        key = dataset if dataset in stats else ("_default" if "_default" in stats else None)
        if key is not None:
            ds_stats = stats[key]

    # 1. Check precomputed per-seed fit-only entry in norm_stats.json
    if (
        ds_stats is not None
        and want_fit_only
        and train_indices is None
        and exclude_names is None
        and seed is not None
    ):
        by_seed = ds_stats.get("by_seed") or {}
        seed_keys = [
            f"seed_{int(seed)}_val_{eff_vf:.4f}",
            f"seed_{int(seed)}",
            str(int(seed)),
        ]
        for sk in seed_keys:
            if sk in by_seed:
                cand_entry = by_seed[sk]
                prov = cand_entry.get("_provenance") or {}
                if prov.get("val_fraction") is None or abs(float(prov["val_fraction"]) - eff_vf) < 1e-6:
                    return cand_entry
        top_prov = ds_stats.get("_provenance") or {}
        if (
            top_prov.get("fit_only") is True
            and top_prov.get("seed") == int(seed)
            and abs(float(top_prov.get("val_fraction", eff_vf)) - eff_vf) < 1e-6
        ):
            return ds_stats

    # 2. If fit-only stats are requested and local Training tiles exist on disk,
    #    compute fit-only stats excluding val_idx on the fly (cached).
    if want_fit_only and fit_only is not False:
        inferred_root = _infer_data_root_from_images_dir(images_dir, data_root=data_root)
        train_img_dir = None
        train_lbl_dir = None
        if inferred_root and dataset:
            cand_img = os.path.join(inferred_root, dataset, "Training", "images")
            cand_lbl = os.path.join(inferred_root, dataset, "Training", "labels")
            if os.path.isdir(cand_img) and os.path.isdir(cand_lbl):
                train_img_dir, train_lbl_dir = cand_img, cand_lbl
        if train_img_dir is None and images_dir is not None:
            _, split_kind = infer_dataset_split(images_dir)
            if split_kind == "train" and os.path.isdir(images_dir):
                train_img_dir = images_dir
                train_lbl_dir = masks_dir
        if train_img_dir is not None and os.path.isdir(train_img_dir):
            cache_key = (
                os.path.abspath(train_img_dir),
                dataset,
                None if seed is None else int(seed),
                eff_vf,
                None if train_indices is None else tuple(int(i) for i in train_indices),
                None if exclude_names is None else tuple(sorted(exclude_names)),
            )
            if cache_key in _FIT_STATS_CACHE:
                return _FIT_STATS_CACHE[cache_key]
            try:
                allowed = manifest_names(train_img_dir, data_root=inferred_root)
            except Exception:
                allowed = None
            pairs = list_pairs(train_img_dir, train_lbl_dir, sort=True)
            if allowed is not None:
                train_names = [p[0] for p in pairs if p[0] in allowed]
            else:
                train_names = [p[0] for p in pairs]
            if train_names:
                fit_names = compute_fit_only_names(
                    train_img_dir,
                    train_lbl_dir,
                    train_names,
                    seed=42 if seed is None else int(seed),
                    val_fraction=eff_vf,
                    train_indices=train_indices,
                    exclude_names=exclude_names,
                )
                if fit_names:
                    prov = {
                        "fit_only": True,
                        "seed": 42 if seed is None else int(seed),
                        "val_fraction": eff_vf,
                        "n_manifest_train_tiles": len(train_names),
                        "n_fit_train_tiles": len(fit_names),
                        "n_excluded_val_tiles": len(train_names) - len(fit_names),
                    }
                    computed = compute_modality_band_stats(
                        train_img_dir,
                        train_lbl_dir,
                        fit_names,
                        provenance=prov,
                    )
                    _FIT_STATS_CACHE[cache_key] = computed
                    return computed

    if stats is None:
        raise FileNotFoundError(
            f"Normalization stats {path} missing. Run tools/compute_norm_stats.py or use legacy=True."
        )
    if ds_stats is None:
        raise KeyError(f"No normalization stats for {dataset} in {path}")
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
        if b in LOG_BANDS:
            x = np.log1p(np.clip(x, 0, None))
        lo, hi = s["lo"], s["hi"]
        x = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
        x = np.clip(x, 0.0, 1.0)
        x[nod] = 0.0
        out.append(x * 2.0 - 1.0)
    return np.stack(out)


def load_image_clean(img_path, image_size, bands, stats):
    """HWC float32 in [-1, 1]."""
    raw = read_raw(img_path)
    modality = MODALITY_BY_COUNT.get(raw.shape[0])
    if modality is None:
        raise ValueError(f"{img_path}: unexpected band count {raw.shape[0]}")
    x = normalize_bands(raw, modality, bands[modality], stats[modality])
    x = np.clip(_resize_float(x, image_size), -1.0, 1.0)
    return np.transpose(x, (1, 2, 0))


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
    k = int(max_samples)
    eff_seed = 42 if sample_seed is None else int(sample_seed)
    # Use the same canonical name order and SeedSequence(seed, 7) as training_common.cap_samples
    canon_order = sorted(range(len(pairs)), key=lambda i: os.path.basename(str(pairs[i][0])))
    rng = np.random.default_rng(_derive_seed(eff_seed, 7))
    keep_idx = sorted(rng.permutation(canon_order)[:k].tolist())
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
    dataset, _ = infer_dataset_split(images_dir)
    stats = load_stats(
        dataset,
        stats_path,
        seed=seed,
        val_fraction=val_fraction,
        fit_only=fit_only,
        data_root=data_root,
        images_dir=images_dir,
        masks_dir=masks_dir,
    )
    band_sel = resolve_bands(bands)
    for name, img_path, mask_path in wrap(pairs, desc="Loading data"):
        samples.append((load_image_clean(img_path, image_size, band_sel, stats),
                        load_mask(mask_path, image_size)))
        names.append(name)
    return samples, names
