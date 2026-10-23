"""Framework-agnostic data loading for the DFC 2024 flood tiles.

Used by utilsTorch.FloodSegmentationDataset and utilsJAX.load_dataset.

Two modes:
  clean (default)
    * honors the committed split manifest splits/<dataset>.json for
      flood_optical, flood_SAR and flood_combined (drops test tiles that are
      pixel-identical to a training tile and collapses duplicates)
    * explicit band selection by name (see BAND_NAMES, DEFAULT_BANDS)
    * dataset-level normalization with per-band statistics computed on the
      training split only (splits/norm_stats.json), mapped to [-1, 1]
  legacy
    * the original behavior: every file in the folder, first three bands,
      per-image 2-98 percentile stretch. Reproduces pre-fix numbers.

Select legacy with legacy=True on the loader functions, with
set_data_defaults(legacy=True), or with the environment variable
QUAFFLE_LEGACY_DATA=1.
"""

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

_DEFAULTS = {"legacy": None, "bands": None}


def set_data_defaults(legacy=None, bands=None):
    """Set process-wide defaults (e.g. from a --legacy-data CLI flag)."""
    if legacy is not None:
        _DEFAULTS["legacy"] = bool(legacy)
    if bands is not None:
        _DEFAULTS["bands"] = bands


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


def load_manifest(dataset):
    path = os.path.join(SPLITS_DIR, f"{dataset}.json")
    with open(path) as f:
        return json.load(f)


def manifest_names(images_dir, manifest=None):
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
        path = os.path.join(SPLITS_DIR, f"{dataset}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Split manifest {path} missing. Run tools/dedup_splits.py or use legacy=True.")
        manifest = load_manifest(dataset)
    if split is None:
        raise ValueError(f"Cannot infer split (Training/Testing) from {images_dir}")
    return set(manifest[split])


def load_stats(dataset, stats_path=None):
    path = stats_path or STATS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Normalization stats {path} missing. Run tools/compute_norm_stats.py or use legacy=True.")
    with open(path) as f:
        stats = json.load(f)
    key = dataset if dataset in stats else "_default"
    if key not in stats:
        raise KeyError(f"No normalization stats for {dataset} in {path}")
    return stats[key]


def read_raw(img_path):
    import rasterio
    with rasterio.open(img_path) as src:
        return src.read()


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


def load_mask(mask_path, image_size):
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    return (mask_array > 0.001).astype(np.float32)[..., np.newaxis]


def load_samples(images_dir, masks_dir, image_size, legacy=None, bands=None,
                 manifest=None, stats_path=None, progress=None):
    """Return a list of (image HWC float32 in [-1,1], mask HW1 float32) and
    the list of sample names, in deterministic order.

    legacy:     True reproduces the original loader exactly (see module doc).
    bands:      dict {modality: [names]}, flat list, or comma string.
    manifest:   None (auto from path), False (no filtering), path, or dict.
    """
    legacy = resolve_legacy(legacy)
    wrap = progress or (lambda it, **kw: it)
    samples, names = [], []
    if legacy:
        pairs = list_pairs(images_dir, masks_dir, sort=False)
        if not pairs and not glob.glob(os.path.join(images_dir, "*.tif*")):
            raise ValueError(f"No TIFF images found in {images_dir}")
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
    allowed = manifest_names(images_dir, manifest)
    if allowed is not None:
        present = {p[0] for p in pairs}
        missing = allowed - present
        if missing:
            raise ValueError(f"{len(missing)} manifest entries missing in {images_dir}, "
                             f"e.g. {sorted(missing)[:3]}")
        pairs = [p for p in pairs if p[0] in allowed]
    dataset, _ = infer_dataset_split(images_dir)
    stats = load_stats(dataset, stats_path)
    band_sel = resolve_bands(bands)
    for name, img_path, mask_path in wrap(pairs, desc="Loading data"):
        samples.append((load_image_clean(img_path, image_size, band_sel, stats),
                        load_mask(mask_path, image_size)))
        names.append(name)
    return samples, names
