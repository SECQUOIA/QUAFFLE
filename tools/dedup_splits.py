#!/usr/bin/env python3
"""Build deterministic, leakage-free split manifests for the flood datasets.

The raw Training/Testing folders under data/ are left untouched. This tool
hashes every image/mask pair and writes splits/<dataset>.json, which the
loaders in flood_data.py honor by default.

Rules (applied per dataset):
  1. Identity of a sample = hash of the decoded image pixels (dtype, shape and
     raw array bytes, so GeoTIFF metadata is ignored). File md5s are also
     recorded for reference.
  2. Training: duplicate images are collapsed to one copy (first name in
     natural sort order is kept).
  3. Testing: any test image whose pixel hash occurs anywhere in Training is
     dropped. Remaining duplicates inside Testing are collapsed to one copy.
  4. Near duplicates (identical after a flip/rotation, or matching an 8x8
     average hash with near-identical pixels) are only reported, never
     excluded, unless they are pixel-identical (rule 1).

Usage:
    python tools/dedup_splits.py                  # all three flood datasets
    python tools/dedup_splits.py --datasets flood_SAR --data-root data
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from flood_data import list_pairs, natural_key  # noqa: E402

DATASETS = ["flood_optical", "flood_SAR", "flood_combined"]
SPLITS = {"train": "Training", "test": "Testing"}


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_md5(arr):
    arr = np.ascontiguousarray(arr)
    h = hashlib.md5()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def read_image(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read()


def read_mask(path):
    from PIL import Image
    return np.array(Image.open(path).convert("L"))


def dihedral_hashes(img):
    """Hashes of the 7 non-identity flips/rotations of a CHW array."""
    out = []
    for k in range(4):
        r = np.rot90(img, k, axes=(1, 2))
        for flip in (False, True):
            if k == 0 and not flip:
                continue
            out.append(array_md5(r[:, :, ::-1] if flip else r))
    return out


def ahash(img, size=8):
    """64-bit average hash of the mean over the first two bands."""
    a = img[: min(2, img.shape[0])].astype(np.float64).mean(axis=0)
    h, w = a.shape
    a = a[: h - h % size, : w - w % size]
    a = a.reshape(size, a.shape[0] // size, size, a.shape[1] // size).mean(axis=(1, 3))
    return (a > np.median(a)).flatten()


def small(img, size=64):
    """Block-mean of the first two bands at size x size, for cheap correlation."""
    a = img[: min(2, img.shape[0])].astype(np.float32)
    c, h, w = a.shape
    s = min(size, h, w)
    a = a[:, : h - h % s, : w - w % s]
    return a.reshape(c, s, a.shape[1] // s, s, a.shape[2] // s).mean(axis=(2, 4))


def scan_split(data_root, dataset, split_dir):
    images_dir = os.path.join(data_root, dataset, split_dir, "images")
    masks_dir = os.path.join(data_root, dataset, split_dir, "labels")
    pairs = list_pairs(images_dir, masks_dir)
    recs = []
    for name, img_path, mask_path in pairs:
        img = read_image(img_path)
        mask = read_mask(mask_path)
        recs.append({
            "name": name,
            "image_file": os.path.basename(img_path),
            "mask_file": os.path.basename(mask_path),
            "image_md5": file_md5(img_path),
            "mask_md5": file_md5(mask_path),
            "image_pix": array_md5(img),
            "mask_pix": array_md5(mask),
            "_dihedral": dihedral_hashes(img),
            "_ahash": ahash(img),
            "_small": small(img),
            "_shape": img.shape,
        })
    n_images = len([f for f in os.listdir(images_dir) if f.lower().endswith((".tif", ".tiff"))])
    return recs, n_images


def collapse(recs):
    """Keep first record (natural sort) of each image pixel hash."""
    groups = defaultdict(list)
    for r in recs:
        groups[r["image_pix"]].append(r)
    keep, dropped, label_conflicts = [], [], 0
    for g in groups.values():
        g.sort(key=lambda r: natural_key(r["name"]))
        keep.append(g[0])
        for r in g[1:]:
            dropped.append({"name": r["name"], "duplicate_of": g[0]["name"],
                            "same_mask": r["mask_pix"] == g[0]["mask_pix"]})
        if len({r["mask_pix"] for r in g}) > 1:
            label_conflicts += 1
    keep.sort(key=lambda r: natural_key(r["name"]))
    dropped.sort(key=lambda d: natural_key(d["name"]))
    return keep, dropped, len(groups), label_conflicts


def near_dups(test_recs, train_recs):
    """Report test items that are not pixel-identical to a training image but
    are a flipped/rotated copy, or have an identical 8x8 average hash and a
    downsampled (64x64) correlation above 0.99 with some training image of the same shape."""
    train_pix = {r["image_pix"]: r["name"] for r in train_recs}
    dihedral, ahash_close = [], []
    for t in test_recs:
        if t["image_pix"] in train_pix:
            continue
        hit = next((train_pix[h] for h in t["_dihedral"] if h in train_pix), None)
        if hit is not None:
            dihedral.append({"test": t["name"], "train": hit})
            continue
        for r in train_recs:
            if r["_shape"] != t["_shape"]:
                continue
            if np.count_nonzero(r["_ahash"] != t["_ahash"]) > 0:
                continue
            a = t["_small"].astype(np.float64).ravel()
            b = r["_small"].astype(np.float64).ravel()
            if a.std() == 0 or b.std() == 0:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if c > 0.99:
                ahash_close.append({"test": t["name"], "train": r["name"], "corr": round(c, 5)})
                break
    return dihedral, ahash_close


def build(data_root, dataset):
    train, n_train_files = scan_split(data_root, dataset, SPLITS["train"])
    test, n_test_files = scan_split(data_root, dataset, SPLITS["test"])

    train_keep, train_dropped, train_unique, train_conflicts = collapse(train)

    train_pix = {}
    for r in sorted(train, key=lambda r: natural_key(r["name"])):
        train_pix.setdefault(r["image_pix"], r)
    test_clean, test_leaked = [], []
    for r in test:
        m = train_pix.get(r["image_pix"])
        if m is None:
            test_clean.append(r)
        else:
            test_leaked.append({"name": r["name"], "train_match": m["name"],
                                "same_mask": r["mask_pix"] == m["mask_pix"]})
    test_leaked.sort(key=lambda d: natural_key(d["name"]))
    test_keep, test_dropped, _, test_conflicts = collapse(test_clean)

    dihedral, ahash_close = near_dups(test_keep, train_keep)

    def strip(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    manifest = {
        "dataset": dataset,
        "generated_by": "tools/dedup_splits.py",
        "identity": "md5 of decoded image pixels (dtype, shape, bytes); metadata ignored",
        "counts": {
            "train_files": n_train_files,
            "train_pairs": len(train),
            "train_unique_images": train_unique,
            "train_kept": len(train_keep),
            "train_dropped_internal_duplicates": len(train_dropped),
            "train_duplicate_groups_with_conflicting_masks": train_conflicts,
            "test_files": n_test_files,
            "test_pairs": len(test),
            "test_dropped_leaked_from_train": len(test_leaked),
            "test_leaked_with_identical_mask": sum(d["same_mask"] for d in test_leaked),
            "test_dropped_internal_duplicates": len(test_dropped),
            "test_duplicate_groups_with_conflicting_masks": test_conflicts,
            "test_kept": len(test_keep),
            "near_dup_test_vs_train_flip_or_rotation": len(dihedral),
            "near_dup_test_vs_train_ahash_corr_gt_0.99": len(ahash_close),
        },
        "train": [r["name"] for r in train_keep],
        "test": [r["name"] for r in test_keep],
        "excluded": {
            "train_internal_duplicates": train_dropped,
            "test_leaked_from_train": test_leaked,
            "test_internal_duplicates": test_dropped,
        },
        "near_duplicates_reported_not_excluded": {
            "flip_or_rotation_of_train": dihedral,
            "ahash_equal_corr_gt_0.99": ahash_close,
        },
        "hashes": {
            "train": [strip(r) for r in sorted(train, key=lambda r: natural_key(r["name"]))],
            "test": [strip(r) for r in sorted(test, key=lambda r: natural_key(r["name"]))],
        },
    }
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "splits"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for ds in args.datasets:
        m = build(args.data_root, ds)
        out = os.path.join(args.out_dir, f"{ds}.json")
        with open(out, "w") as f:
            json.dump(m, f, indent=1, sort_keys=False)
            f.write("\n")
        print(f"[{ds}] -> {out}")
        for k, v in m["counts"].items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
