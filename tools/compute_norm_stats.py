#!/usr/bin/env python3
"""Compute per-band normalization statistics on the TRAINING split only.

For every flood dataset, iterates over the training names in
splits/<dataset>.json (the deduplicated manifest), reads each tile, and
records per band the 1st and 99th percentile (lo, hi) plus mean and std.
VV and VH are log1p transformed first. Pixels equal to -9999 (nodata) are
excluded. To bound memory, a deterministic regular grid of pixels
(every STRIDE-th row and column) is sampled from each tile.

The loader maps each band to [-1, 1] as clip((x - lo) / (hi - lo), 0, 1) * 2 - 1.

Usage:
    python tools/compute_norm_stats.py            # writes splits/norm_stats.json
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from flood_data import (BAND_NAMES, LOG_BANDS, MODALITY_BY_COUNT, NODATA,  # noqa: E402
                        STAT_PERCENTILES, find_mask, load_manifest, read_raw)

DATASETS = ["flood_optical", "flood_SAR", "flood_combined"]
STRIDE = {"sar": 8, "optical": 2}


def compute(data_root, dataset):
    man = load_manifest(dataset)
    images_dir = os.path.join(data_root, dataset, "Training", "images")
    masks_dir = os.path.join(data_root, dataset, "Training", "labels")
    buf = {}
    n_tiles = {}
    for name in man["train"]:
        p = os.path.join(images_dir, name + ".tif")
        if not os.path.exists(p):
            p = os.path.join(images_dir, name + ".tiff")
        assert find_mask(masks_dir, name) is not None, name
        raw = read_raw(p)
        mod = MODALITY_BY_COUNT[raw.shape[0]]
        s = STRIDE[mod]
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
            if b in LOG_BANDS:
                v = np.log1p(np.clip(v, 0, None))
            lo, hi = np.percentile(v, STAT_PERCENTILES)
            bands[b] = {"lo": round(float(lo), 6), "hi": round(float(hi), 6),
                        "mean": round(float(v.mean()), 6), "std": round(float(v.std()), 6),
                        "log1p": b in LOG_BANDS}
        out[mod] = {"n_train_tiles": n_tiles[mod], "pixel_stride": STRIDE[mod],
                    "n_pixels_per_band": int(a.shape[1]), "bands": bands}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "splits", "norm_stats.json"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    stats = {"_info": {
        "generated_by": "tools/compute_norm_stats.py",
        "split": "training split of splits/<dataset>.json only",
        "percentiles": list(STAT_PERCENTILES),
        "nodata": NODATA,
        "log1p_bands": sorted(LOG_BANDS),
    }}
    for ds in args.datasets:
        stats[ds] = compute(args.data_root, ds)
        for mod, m in stats[ds].items():
            print(f"[{ds}/{mod}] tiles={m['n_train_tiles']}")
            for b, s in m["bands"].items():
                print(f"    {b:11s} lo={s['lo']:.4f} hi={s['hi']:.4f} mean={s['mean']:.4f} std={s['std']:.4f}")
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=1)
        f.write("\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
