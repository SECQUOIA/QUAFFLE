#!/usr/bin/env python3
"""Build deterministic, leakage-free split manifests for the flood datasets.

The raw Training/Testing folders under data/ are left untouched. This tool
hashes every image/mask pair and writes splits/<dataset>.json, which the
loaders in flood_data.py honor by default.

Rules (applied per dataset):
  1. Identity of a sample = hash of the decoded image pixels (dtype, shape and
     raw array bytes, so GeoTIFF metadata is ignored). File md5s are also
     recorded for reference.
  2. Training: duplicate images are collapsed according to --conflict-policy:
       * drop (default): duplicate groups whose masks disagree are excluded
         entirely from the kept training split; non-conflicting duplicate
         groups keep the first name in natural sort order.
       * majority: duplicate groups whose masks disagree keep the first tile
         of the majority mask hash (ties broken deterministically by natural
         sort order of tile names).
       * first: legacy behavior — keeps the first name in natural sort order
         regardless of mask agreement.
  3. Testing: any test image whose pixel hash occurs anywhere in Training is
     dropped. Remaining duplicates inside Testing are collapsed with the same
     conflict policy.
  4. Near duplicates (identical after a flip/rotation, or matching an 8x8
     average hash with near-identical pixels) are only reported, never
     excluded, unless they are pixel-identical (rule 1).

Usage:
    python tools/dedup_splits.py                  # all three flood datasets (default: --conflict-policy drop)
    python tools/dedup_splits.py --datasets flood_SAR --data-root data --conflict-policy drop
    python tools/dedup_splits.py --from-manifests --conflict-policy drop
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from flood_data import list_pairs, natural_key  # noqa: E402

DATASETS = ["flood_optical", "flood_SAR", "flood_combined"]
SPLITS = {"train": "Training", "test": "Testing"}
CONFLICT_POLICIES = ("drop", "majority", "first")
DEFAULT_CONFLICT_POLICY = "drop"


def _normalize_conflict_policy(conflict_policy):
    if conflict_policy is None:
        return DEFAULT_CONFLICT_POLICY
    if conflict_policy == "keep_first":
        return "first"
    if conflict_policy not in CONFLICT_POLICIES:
        raise ValueError(
            f"Unknown conflict_policy {conflict_policy!r}; expected one of {CONFLICT_POLICIES}"
        )
    return conflict_policy


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


def collapse(recs, conflict_policy=DEFAULT_CONFLICT_POLICY, return_details=False):
    """Collapse duplicate image pixel hashes according to conflict_policy.

    Policies for duplicate groups whose masks disagree (len(set(mask_pix)) > 1):
      * 'drop' (default): exclude the entire conflicting group from `keep`.
      * 'majority': keep the first record (in natural sort order) whose mask_pix
        has the highest frequency in the group (ties broken by natural sort order).
      * 'first' (or 'keep_first'): keep the first record in natural sort order.
    """
    policy = _normalize_conflict_policy(conflict_policy)
    groups = defaultdict(list)
    for r in recs:
        groups[r["image_pix"]].append(r)
    keep, dropped, label_conflicts = [], [], 0
    conflict_groups, conflict_dropped = [], []
    for image_pix, g in groups.items():
        g.sort(key=lambda r: natural_key(r["name"]))
        mask_hashes = [r["mask_pix"] for r in g]
        has_conflict = len(set(mask_hashes)) > 1
        if not has_conflict:
            keep.append(g[0])
            for r in g[1:]:
                dropped.append({
                    "name": r["name"],
                    "duplicate_of": g[0]["name"],
                    "same_mask": True,
                })
            continue

        label_conflicts += 1
        group_names = [r["name"] for r in g]
        if policy == "drop":
            for r in g:
                entry = {
                    "name": r["name"],
                    "duplicate_of": g[0]["name"],
                    "same_mask": r["mask_pix"] == g[0]["mask_pix"],
                    "conflict_policy": "drop",
                    "reason": "conflicting_mask_group",
                }
                dropped.append(entry)
                conflict_dropped.append(dict(entry))
            conflict_groups.append({
                "image_pix": image_pix,
                "names": group_names,
                "mask_pix": mask_hashes,
                "kept": None,
                "dropped": list(group_names),
                "policy": "drop",
            })
        elif policy == "majority":
            mask_counts = defaultdict(int)
            for r in g:
                mask_counts[r["mask_pix"]] += 1
            winner = min(
                g,
                key=lambda r: (-mask_counts[r["mask_pix"]], natural_key(r["name"])),
            )
            keep.append(winner)
            for r in g:
                if r is winner:
                    continue
                entry = {
                    "name": r["name"],
                    "duplicate_of": winner["name"],
                    "same_mask": r["mask_pix"] == winner["mask_pix"],
                }
                dropped.append(entry)
                if r["mask_pix"] != winner["mask_pix"]:
                    conflict_dropped.append({
                        **entry,
                        "conflict_policy": "majority",
                        "reason": "conflicting_mask_minority",
                    })
            conflict_groups.append({
                "image_pix": image_pix,
                "names": group_names,
                "mask_pix": mask_hashes,
                "kept": winner["name"],
                "dropped": [r["name"] for r in g if r is not winner],
                "policy": "majority",
            })
        elif policy == "first":
            winner = g[0]
            keep.append(winner)
            for r in g[1:]:
                entry = {
                    "name": r["name"],
                    "duplicate_of": winner["name"],
                    "same_mask": r["mask_pix"] == winner["mask_pix"],
                }
                dropped.append(entry)
                if r["mask_pix"] != winner["mask_pix"]:
                    conflict_dropped.append({
                        **entry,
                        "conflict_policy": "first",
                        "reason": "conflicting_mask_subsequent",
                    })
            conflict_groups.append({
                "image_pix": image_pix,
                "names": group_names,
                "mask_pix": mask_hashes,
                "kept": winner["name"],
                "dropped": [r["name"] for r in g[1:]],
                "policy": "first",
            })

    keep.sort(key=lambda r: natural_key(r["name"]))
    dropped.sort(key=lambda d: natural_key(d["name"]))
    conflict_dropped.sort(key=lambda d: natural_key(d["name"]))
    conflict_groups.sort(key=lambda cg: natural_key(cg["names"][0]))
    if return_details:
        details = {
            "conflict_policy": policy,
            "conflict_groups": conflict_groups,
            "conflict_dropped": conflict_dropped,
        }
        return keep, dropped, len(groups), label_conflicts, details
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
        if "_dihedral" not in t:
            continue
        hit = next((train_pix[h] for h in t["_dihedral"] if h in train_pix), None)
        if hit is not None:
            dihedral.append({"test": t["name"], "train": hit})
            continue
        for r in train_recs:
            if "_shape" not in r or "_shape" not in t:
                continue
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


def _assemble_manifest(
    dataset,
    train,
    n_train_files,
    test,
    n_test_files,
    conflict_policy=DEFAULT_CONFLICT_POLICY,
    near_dup_override=None,
):
    policy = _normalize_conflict_policy(conflict_policy)
    (
        train_keep,
        train_dropped,
        train_unique,
        train_conflicts,
        train_details,
    ) = collapse(train, conflict_policy=policy, return_details=True)

    train_pix = {}
    for r in sorted(train, key=lambda r: natural_key(r["name"])):
        train_pix.setdefault(r["image_pix"], r)
    test_clean, test_leaked = [], []
    for r in test:
        m = train_pix.get(r["image_pix"])
        if m is None:
            test_clean.append(r)
        else:
            test_leaked.append({
                "name": r["name"],
                "train_match": m["name"],
                "same_mask": r["mask_pix"] == m["mask_pix"],
            })
    test_leaked.sort(key=lambda d: natural_key(d["name"]))
    (
        test_keep,
        test_dropped,
        _,
        test_conflicts,
        test_details,
    ) = collapse(test_clean, conflict_policy=policy, return_details=True)

    if near_dup_override is not None:
        dihedral = list(near_dup_override.get("flip_or_rotation_of_train", []))
        ahash_close = list(near_dup_override.get("ahash_equal_corr_gt_0.99", []))
    else:
        dihedral, ahash_close = near_dups(test_keep, train_keep)

    def strip(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    train_internal_nonconflict = [
        d for d in train_dropped if d.get("reason") != "conflicting_mask_group"
    ]

    manifest = {
        "dataset": dataset,
        "generated_by": "tools/dedup_splits.py",
        "identity": "md5 of decoded image pixels (dtype, shape, bytes); metadata ignored",
        "conflict_policy": policy,
        "conflict_groups": train_details["conflict_groups"],
        "conflict_dropped": train_details["conflict_dropped"],
        "counts": {
            "train_files": n_train_files,
            "train_pairs": len(train),
            "train_unique_images": train_unique,
            "train_kept": len(train_keep),
            "train_dropped_internal_duplicates": len(train_dropped),
            "train_dropped_nonconflicting_duplicates": len(train_internal_nonconflict),
            "train_duplicate_groups_with_conflicting_masks": train_conflicts,
            "conflict_groups": train_conflicts,
            "conflict_dropped": len(train_details["conflict_dropped"]),
            "test_files": n_test_files,
            "test_pairs": len(test),
            "test_dropped_leaked_from_train": len(test_leaked),
            "test_leaked_with_identical_mask": sum(d["same_mask"] for d in test_leaked),
            "test_dropped_internal_duplicates": len(test_dropped),
            "test_duplicate_groups_with_conflicting_masks": test_conflicts,
            "test_conflict_groups": test_conflicts,
            "test_conflict_dropped": len(test_details["conflict_dropped"]),
            "test_kept": len(test_keep),
            "near_dup_test_vs_train_flip_or_rotation": len(dihedral),
            "near_dup_test_vs_train_ahash_corr_gt_0.99": len(ahash_close),
        },
        "train": [r["name"] for r in train_keep],
        "test": [r["name"] for r in test_keep],
        "excluded": {
            "train_internal_duplicates": train_dropped,
            "conflict_groups": train_details["conflict_groups"],
            "conflict_dropped": train_details["conflict_dropped"],
            "test_leaked_from_train": test_leaked,
            "test_internal_duplicates": test_dropped,
            "test_conflict_groups": test_details["conflict_groups"],
            "test_conflict_dropped": test_details["conflict_dropped"],
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


def build_from_manifest(manifest, conflict_policy=DEFAULT_CONFLICT_POLICY):
    """Deterministically rebuild a split manifest from its committed per-tile hashes."""
    if isinstance(manifest, str):
        with open(manifest) as f:
            manifest = json.load(f)
    dataset = manifest["dataset"]
    train = [dict(r) for r in manifest["hashes"]["train"]]
    test = [dict(r) for r in manifest["hashes"]["test"]]
    counts = manifest.get("counts", {})
    n_train_files = int(counts.get("train_files", len(train)))
    n_test_files = int(counts.get("test_files", len(test)))
    near_dup_override = manifest.get("near_duplicates_reported_not_excluded", {
        "flip_or_rotation_of_train": [],
        "ahash_equal_corr_gt_0.99": [],
    })
    return _assemble_manifest(
        dataset,
        train,
        n_train_files,
        test,
        n_test_files,
        conflict_policy=conflict_policy,
        near_dup_override=near_dup_override,
    )


def validate_manifest(manifest, expected_policy=None):
    """Verify manifest internal consistency against its per-pair hashes."""
    if isinstance(manifest, str):
        with open(manifest) as f:
            manifest = json.load(f)
    policy = _normalize_conflict_policy(
        expected_policy if expected_policy is not None else manifest.get("conflict_policy", DEFAULT_CONFLICT_POLICY)
    )
    rebuilt = build_from_manifest(manifest, conflict_policy=policy)
    if manifest.get("train") != rebuilt["train"]:
        raise ValueError(
            f"{manifest.get('dataset')}: kept train list does not match rebuilt list under policy={policy!r}"
        )
    if manifest.get("test") != rebuilt["test"]:
        raise ValueError(
            f"{manifest.get('dataset')}: kept test list does not match rebuilt list under policy={policy!r}"
        )
    if policy == "drop":
        groups = defaultdict(set)
        name_to_pix = {}
        for r in manifest["hashes"]["train"]:
            groups[r["image_pix"]].add(r["mask_pix"])
            name_to_pix[r["name"]] = r["image_pix"]
        conflicting_names = [
            n for n in manifest["train"] if len(groups[name_to_pix[n]]) > 1
        ]
        if conflicting_names:
            raise ValueError(
                f"{manifest.get('dataset')}: conflicting-mask tiles retained in train under drop policy: "
                f"{conflicting_names[:5]}"
            )
    return True


def build(data_root, dataset, conflict_policy=DEFAULT_CONFLICT_POLICY, manifest_dir=None):
    train_images_dir = os.path.join(data_root, dataset, SPLITS["train"], "images")
    if os.path.isdir(train_images_dir):
        train, n_train_files = scan_split(data_root, dataset, SPLITS["train"])
        test, n_test_files = scan_split(data_root, dataset, SPLITS["test"])
        return _assemble_manifest(
            dataset,
            train,
            n_train_files,
            test,
            n_test_files,
            conflict_policy=conflict_policy,
        )
    search_dirs = []
    if manifest_dir:
        search_dirs.append(manifest_dir)
    search_dirs.append(os.path.join(REPO_ROOT, "splits"))
    for d in search_dirs:
        cand = os.path.join(d, f"{dataset}.json")
        if os.path.exists(cand):
            return build_from_manifest(cand, conflict_policy=conflict_policy)
    raise FileNotFoundError(
        f"Neither raw directory {train_images_dir} nor existing manifest for {dataset} found."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "splits"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument(
        "--conflict-policy",
        choices=list(CONFLICT_POLICIES),
        default=DEFAULT_CONFLICT_POLICY,
        help="How to resolve duplicate image groups whose masks disagree (default: drop)",
    )
    ap.add_argument(
        "--from-manifests",
        action="store_true",
        help="Rebuild manifests deterministically from the per-pair hashes in existing manifest JSONs",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing manifests in --out-dir against --conflict-policy without writing",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for ds in args.datasets:
        out = os.path.join(args.out_dir, f"{ds}.json")
        if args.validate:
            validate_manifest(out, expected_policy=args.conflict_policy)
            print(f"[{ds}] VALIDATED ({args.conflict_policy}): {out}")
            continue
        if args.from_manifests:
            src = out if os.path.exists(out) else os.path.join(REPO_ROOT, "splits", f"{ds}.json")
            m = build_from_manifest(src, conflict_policy=args.conflict_policy)
        else:
            m = build(
                args.data_root,
                ds,
                conflict_policy=args.conflict_policy,
                manifest_dir=args.out_dir,
            )
        validate_manifest(m, expected_policy=args.conflict_policy)
        with open(out, "w") as f:
            json.dump(m, f, indent=1, sort_keys=False)
            f.write("\n")
        print(f"[{ds}] -> {out} (conflict_policy={m['conflict_policy']})")
        for k, v in m["counts"].items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
