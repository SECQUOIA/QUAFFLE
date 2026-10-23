#!/usr/bin/env python3
"""Compute per-band normalization statistics on the fit-only TRAINING split.

For every flood dataset, iterates over the deduplicated training names in
splits/<dataset>.json, excludes the held-out validation split (`val_idx` for a
given `seed` and `val_fraction`, or explicit `train_indices` / `exclude_names`)
so validation tile distributions never leak into normalization percentiles,
reads each fit-only training tile, and records per band the 1st and 99th
percentile (lo, hi) plus mean and std. VV and VH are log1p transformed first.
Pixels equal to -9999 (nodata) are excluded. To bound memory, a deterministic
regular grid of pixels (every STRIDE-th row and column) is sampled from each
tile.

The loader maps each band to [-1, 1] as clip((x - lo) / (hi - lo), 0, 1) * 2 - 1.

Usage:
    python tools/compute_norm_stats.py            # writes splits/norm_stats.json
    python tools/compute_norm_stats.py --seed 42 --val-fraction 0.15 --seeds 42 17 123
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from flood_data import (  # noqa: E402
    BAND_NAMES,
    DEFAULT_VAL_FRACTION,
    LOG_BANDS,
    MODALITY_BY_COUNT,
    NODATA,
    STAT_PERCENTILES,
    STRIDE,
    compute_fit_only_names,
    compute_modality_band_stats,
    find_mask,
    load_manifest,
    read_raw,
)

DATASETS = ["flood_optical", "flood_SAR", "flood_combined"]


def compute(
    data_root,
    dataset,
    seed=42,
    val_fraction=DEFAULT_VAL_FRACTION,
    train_indices=None,
    exclude_names=None,
    include_names=None,
    manifest=None,
    fit_only=True,
    stride=None,
    image_size=128,
):
    """Compute per-modality, per-band normalization statistics for `dataset`.

    By default (`fit_only=True`, `seed=42`, `val_fraction=0.15`), excludes the
    stratified validation holdout (`val_idx`) from the manifest's `train` list.
    Callers may also pass explicit `train_indices`, `exclude_names`, or
    `include_names`, or set `fit_only=False` / `val_fraction=0.0` to compute
    over all manifest training tiles.
    """
    if isinstance(manifest, str):
        with open(manifest) as f:
            man = json.load(f)
    elif isinstance(manifest, dict):
        man = manifest
    else:
        man = load_manifest(dataset, data_root=data_root)

    images_dir = os.path.join(data_root, dataset, "Training", "images")
    masks_dir = os.path.join(data_root, dataset, "Training", "labels")
    all_train_names = list(man["train"])

    if not fit_only and train_indices is None and exclude_names is None and include_names is None:
        selected_names = all_train_names
        eff_vf = 0.0
        eff_seed = None
    elif (train_indices is not None or exclude_names is not None or include_names is not None) and (
        seed is None and val_fraction is None
    ):
        selected_names = compute_fit_only_names(
            images_dir,
            masks_dir,
            all_train_names,
            seed=None,
            val_fraction=0.0,
            train_indices=train_indices,
            include_names=include_names,
            exclude_names=exclude_names,
            image_size=image_size,
        )
        eff_vf = 0.0
        eff_seed = None
    else:
        eff_seed = 42 if seed is None else int(seed)
        eff_vf = DEFAULT_VAL_FRACTION if val_fraction is None else float(val_fraction)
        # If caller explicitly passed train_indices or exclude_names without wanting
        # an additional random holdout split on top, honor their explicit filter directly:
        if train_indices is not None or exclude_names is not None:
            selected_names = compute_fit_only_names(
                images_dir,
                masks_dir,
                all_train_names,
                seed=None,
                val_fraction=0.0,
                train_indices=train_indices,
                include_names=include_names,
                exclude_names=exclude_names,
                image_size=image_size,
            )
        else:
            selected_names = compute_fit_only_names(
                images_dir,
                masks_dir,
                all_train_names,
                seed=eff_seed,
                val_fraction=eff_vf,
                include_names=include_names,
                image_size=image_size,
            )

    if not selected_names:
        raise ValueError(f"No training tiles selected for {dataset} in {images_dir}")

    prov = {
        "fit_only": bool(len(selected_names) < len(all_train_names) or fit_only),
        "seed": eff_seed,
        "val_fraction": float(eff_vf),
        "n_manifest_train_tiles": len(all_train_names),
        "n_fit_train_tiles": len(selected_names),
        "n_excluded_val_tiles": len(all_train_names) - len(selected_names),
    }
    return compute_modality_band_stats(
        images_dir,
        masks_dir,
        selected_names,
        stride_map=stride or STRIDE,
        provenance=prov,
    )


def build_norm_stats(
    data_root,
    datasets=DATASETS,
    *,
    seed=42,
    val_fraction=DEFAULT_VAL_FRACTION,
    seeds=None,
    fit_only=True,
    image_size=128,
):
    """Build a complete norm_stats.json payload with backward-compatible top-level
    dataset keys and optional per-seed fit-only entries under `by_seed`."""
    eff_vf = 0.0 if not fit_only else float(val_fraction)
    seed_list = list(seeds) if seeds is not None else ([int(seed)] if seed is not None else [])
    if seed is not None and int(seed) not in seed_list:
        seed_list.insert(0, int(seed))
    stats = {
        "_info": {
            "generated_by": "tools/compute_norm_stats.py",
            "split": (
                f"fit-only training split of splits/<dataset>.json "
                f"(val_fraction={eff_vf} excluded)"
                if fit_only and eff_vf > 0
                else "training split of splits/<dataset>.json only"
            ),
            "fit_only": bool(fit_only and eff_vf > 0),
            "default_seed": int(seed) if seed is not None else None,
            "val_fraction": eff_vf,
            "seeds": seed_list,
            "percentiles": list(STAT_PERCENTILES),
            "nodata": NODATA,
            "log1p_bands": sorted(LOG_BANDS),
        }
    }
    for ds in datasets:
        primary = compute(
            data_root,
            ds,
            seed=seed,
            val_fraction=eff_vf,
            fit_only=fit_only,
            image_size=image_size,
        )
        if fit_only and eff_vf > 0 and seed_list:
            by_seed = {}
            for s in seed_list:
                s_entry = (
                    primary
                    if (seed is not None and int(s) == int(seed))
                    else compute(
                        data_root,
                        ds,
                        seed=int(s),
                        val_fraction=eff_vf,
                        fit_only=True,
                        image_size=image_size,
                    )
                )
                by_seed[str(int(s))] = s_entry
                by_seed[f"seed_{int(s)}_val_{eff_vf:.4f}"] = s_entry
            primary = dict(primary)
            primary["by_seed"] = by_seed
        stats[ds] = primary
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "splits", "norm_stats.json"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seed", type=int, default=42, help="Primary seed for fit-only holdout exclusion (default: 42)")
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of seeds to precompute under dataset['by_seed']",
    )
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help=f"Validation holdout fraction excluded from fit-only stats (default: {DEFAULT_VAL_FRACTION})",
    )
    ap.add_argument(
        "--full-train",
        action="store_true",
        help="Compute over all manifest training tiles without excluding validation holdout",
    )
    args = ap.parse_args()
    stats = build_norm_stats(
        args.data_root,
        datasets=args.datasets,
        seed=args.seed,
        val_fraction=args.val_fraction,
        seeds=args.seeds,
        fit_only=not args.full_train,
    )
    for ds in args.datasets:
        for mod, m in stats[ds].items():
            if mod.startswith("_") or mod == "by_seed" or not isinstance(m, dict) or "bands" not in m:
                continue
            print(f"[{ds}/{mod}] tiles={m['n_train_tiles']}")
            for b, s in m["bands"].items():
                print(
                    f"    {b:11s} lo={s['lo']:.4f} hi={s['hi']:.4f} "
                    f"mean={s['mean']:.4f} std={s['std']:.4f}"
                )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=1)
        f.write("\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
