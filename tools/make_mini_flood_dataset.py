#!/usr/bin/env python3
"""Deterministic synthetic mini-flood dataset generator for local verification.

Creates a self-contained mini dataset directory with:
  * flood_optical/{Training,Testing}/{images,labels}/
      - 12-band float32 GeoTIFFs matching BAND_NAMES['optical']
      - DEM bands (MERIT_DEM, COP_DEM) containing -9999.0 nodata pixels
  * flood_SAR/{Training,Testing}/{images,labels}/
      - 6-band float32 GeoTIFFs matching BAND_NAMES['sar'] (backscatter/dB +
        DEM with -9999.0 nodata + WORLDCOVER + WATER_OCC)
  * flood_combined/{Training,Testing}/{images,labels}/
      - Optical + SAR combined folders (SAR name collisions suffixed with ' copy')
  * splits/{flood_optical,flood_SAR,flood_combined}.json
      - Deterministic split manifests built with tools/dedup_splits.py
        (--conflict-policy drop)
  * splits/norm_stats.json
      - Fit-only per-band normalization statistics built with
        tools/compute_norm_stats.py (excluding validation holdout val_idx)

Usage:
    python tools/make_mini_flood_dataset.py --out-dir /tmp/quaffle_mini
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flood_data import BAND_NAMES, DEFAULT_VAL_FRACTION, NODATA  # noqa: E402
from tools import compute_norm_stats, dedup_splits  # noqa: E402


def _write_geotiff(path, chw_array):
    import rasterio
    from rasterio.transform import Affine

    arr = np.ascontiguousarray(chw_array, dtype=np.float32)
    c, h, w = arr.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(h))
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(arr)


def _write_mask_png(path, mask_hw):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im = Image.fromarray((mask_hw > 0).astype(np.uint8) * 255, mode="L")
    im.save(path)


def _make_flood_mask_and_terrain(rng, image_size, is_flood_dominant):
    """Synthesize a spatially coherent elevation map and binary flood mask.

    - `is_flood_dominant=True` produces ~18-35% flood pixels (> 0.05 threshold).
    - `is_flood_dominant=False` produces ~1-3.5% flood pixels (<= 0.05 threshold,
      still > 0 positive pixels so per-tile and per-batch ROC AUC is well-defined).
    """
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, image_size, dtype=np.float32),
        np.linspace(-1.0, 1.0, image_size, dtype=np.float32),
        indexing="ij",
    )
    angle = float(rng.uniform(0.0, np.pi))
    phase = float(rng.uniform(-0.3, 0.3))
    u = xx * np.cos(angle) + yy * np.sin(angle) - phase
    v = -xx * np.sin(angle) + yy * np.cos(angle)
    meander = 0.22 * np.sin(np.pi * v + float(rng.uniform(0.0, 2.0 * np.pi)))
    dist = np.abs(u - meander)

    if is_flood_dominant:
        width = float(rng.uniform(0.22, 0.36))
    else:
        width = float(rng.uniform(0.015, 0.032))

    mask = (dist < width).astype(np.uint8)
    # Guarantee at least 2 flood pixels and at least 2 dry pixels
    if mask.sum() < 2:
        mask[image_size // 2, image_size // 2] = 1
        mask[image_size // 2, max(0, image_size // 2 - 1)] = 1
    if is_flood_dominant and mask.mean() <= 0.06:
        r0 = max(0, image_size // 4)
        r1 = min(image_size, 3 * image_size // 4)
        mask[r0:r1, r0:r1] = 1
    elif not is_flood_dominant and mask.mean() > 0.048:
        mask.fill(0)
        mask[image_size // 2, image_size // 2] = 1
        mask[image_size // 2, max(0, image_size // 2 - 1)] = 1

    # Elevation in meters: lower inside channel, higher on banks
    base_elev = float(rng.uniform(115.0, 155.0))
    dem = base_elev + 25.0 * dist + rng.normal(0.0, 1.2, size=(image_size, image_size)).astype(np.float32)
    return mask, dem.astype(np.float32)


def _synthesize_optical_tile(rng, image_size, mask, dem, tile_idx):
    """12-band float32 Optical tile matching BAND_NAMES['optical']."""
    h = w = image_size
    water = mask.astype(np.float32)
    dry = 1.0 - water
    bands = []
    # 1..7: COASTAL, BLUE, GREEN, RED, NIR, SWIR1, SWIR2
    # Water has higher GREEN/BLUE relative to strongly absorbed NIR/SWIR1
    spec_profiles = [
        (420.0, 310.0),   # COASTAL
        (520.0, 390.0),   # BLUE
        (780.0, 640.0),   # GREEN
        (950.0, 320.0),   # RED
        (2600.0, 240.0),  # NIR
        (2100.0, 150.0),  # SWIR1
        (1450.0, 110.0),  # SWIR2
    ]
    tile_offset = float(tile_idx) * 3.5
    for dry_mean, water_mean in spec_profiles:
        signal = dry * dry_mean + water * water_mean + tile_offset
        noise = rng.normal(0.0, 25.0, size=(h, w)).astype(np.float32)
        bands.append((signal + noise).astype(np.float32))

    # 8: QA bitfield
    qa = np.full((h, w), 64.0, dtype=np.float32)
    qa[water > 0.5] = 128.0
    bands.append(qa)

    # 9..10: MERIT_DEM and COP_DEM with deterministic -9999 nodata pixels
    merit_dem = dem.copy()
    cop_dem = (dem + rng.normal(0.0, 0.4, size=(h, w)).astype(np.float32)).astype(np.float32)
    merit_dem[0, :2] = float(NODATA)
    cop_dem[0, :2] = float(NODATA)
    bands.append(merit_dem)
    bands.append(cop_dem)

    # 11: WORLDCOVER (multiples of 10 in 10..90; 80 = permanent water)
    wc = np.where(water > 0.5, 80.0, 30.0).astype(np.float32)
    bands.append(wc)

    # 12: WATER_OCC (0..100)
    wocc = np.clip(
        water * 85.0 + dry * 5.0 + rng.uniform(0.0, 8.0, size=(h, w)).astype(np.float32),
        0.0,
        100.0,
    ).astype(np.float32)
    bands.append(wocc)

    arr = np.stack(bands, axis=0).astype(np.float32)
    assert arr.shape == (len(BAND_NAMES["optical"]), h, w)
    return arr


def _synthesize_sar_tile(rng, image_size, mask, dem, tile_idx):
    """6-band float32 SAR tile matching BAND_NAMES['sar'] (VV, VH, MERIT_DEM, COP_DEM, WORLDCOVER, WATER_OCC).

    VV and VH are synthesized in realistic Sentinel-1 dB backscatter units
    (float32 negative dB in [-28, -4] dB), where specular flood water has
    significantly lower backscatter (~ -21 dB VV, ~ -26 dB VH) than rough dry
    terrain (~ -9.5 dB VV, ~ -15.5 dB VH).
    """
    h = w = image_size
    water = mask.astype(np.float32)
    dry = 1.0 - water
    tile_offset_db = float(int(tile_idx) - 6) * 0.25

    vv = np.clip(
        dry * (-9.5 + tile_offset_db)
        + water * (-21.0 + 0.5 * tile_offset_db)
        + rng.normal(0.0, 1.1, size=(h, w)).astype(np.float32),
        -29.0,
        -2.5,
    ).astype(np.float32)
    vh = np.clip(
        dry * (-15.5 + tile_offset_db)
        + water * (-26.0 + 0.5 * tile_offset_db)
        + rng.normal(0.0, 1.1, size=(h, w)).astype(np.float32),
        -30.0,
        -6.0,
    ).astype(np.float32)

    merit_dem = dem.copy()
    cop_dem = (dem + rng.normal(0.0, 0.4, size=(h, w)).astype(np.float32)).astype(np.float32)
    merit_dem[0, :2] = float(NODATA)
    cop_dem[0, :2] = float(NODATA)

    wc = np.where(water > 0.5, 80.0, 40.0).astype(np.float32)
    wocc = np.clip(
        water * 88.0 + dry * 4.0 + rng.uniform(0.0, 6.0, size=(h, w)).astype(np.float32),
        0.0,
        100.0,
    ).astype(np.float32)

    arr = np.stack(
        [
            vv.astype(np.float32),
            vh.astype(np.float32),
            merit_dem.astype(np.float32),
            cop_dem.astype(np.float32),
            wc,
            wocc,
        ],
        axis=0,
    ).astype(np.float32)
    assert arr.shape == (len(BAND_NAMES["sar"]), h, w)
    return arr


def generate_mini_flood_dataset(
    out_dir,
    *,
    n_train=16,
    n_test=4,
    image_size=32,
    seed=42,
    val_fraction=DEFAULT_VAL_FRACTION,
    seeds=(42, 17, 123),
):
    """Generate the synthetic mini-flood dataset and splits/ manifests under `out_dir`."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(int(seed))

    n_train = int(n_train)
    n_test = int(n_test)
    image_size = int(image_size)
    seed_list = list(dict.fromkeys([int(seed)] + [int(s) for s in (seeds or ())]))

    for ds_idx, (ds_name, synth_fn) in enumerate(
        [
            ("flood_optical", _synthesize_optical_tile),
            ("flood_SAR", _synthesize_sar_tile),
        ]
    ):
        for split_idx, (split_folder, count) in enumerate(
            [("Training", n_train), ("Testing", n_test)]
        ):
            img_dir = os.path.join(out_dir, ds_name, split_folder, "images")
            lbl_dir = os.path.join(out_dir, ds_name, split_folder, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            for i in range(count):
                tile_rng = np.random.default_rng(
                    int(seed) * 10000 + ds_idx * 1000 + split_idx * 500 + i
                )
                # Alternate flood-dominant and non-flood-dominant tiles so every
                # stratified holdout and Dirichlet split has both classes
                is_flood_dom = (i % 2 == 0)
                mask, dem = _make_flood_mask_and_terrain(
                    tile_rng, image_size, is_flood_dominant=is_flood_dom
                )
                offset_idx = i + (100 if split_folder == "Testing" else 0) + ds_idx * 200
                arr = synth_fn(tile_rng, image_size, mask, dem, offset_idx)
                name = f"{i:03d}"
                _write_geotiff(os.path.join(img_dir, f"{name}.tif"), arr)
                _write_mask_png(os.path.join(lbl_dir, f"{name}.png"), mask)

    # Build flood_combined by copying flood_optical and flood_SAR (with ' copy' suffix on clashes)
    for split_folder in ("Training", "Testing"):
        for kind in ("images", "labels"):
            dst = os.path.join(out_dir, "flood_combined", split_folder, kind)
            os.makedirs(dst, exist_ok=True)
            opt_src = os.path.join(out_dir, "flood_optical", split_folder, kind)
            sar_src = os.path.join(out_dir, "flood_SAR", split_folder, kind)
            for fname in sorted(os.listdir(opt_src)):
                shutil.copy2(os.path.join(opt_src, fname), os.path.join(dst, fname))
            for fname in sorted(os.listdir(sar_src)):
                stem, ext = os.path.splitext(fname)
                target_name = f"{stem} copy{ext}" if os.path.exists(os.path.join(opt_src, fname)) else fname
                shutil.copy2(os.path.join(sar_src, fname), os.path.join(dst, target_name))

    # Build and validate splits/<dataset>.json manifests
    splits_dir = os.path.join(out_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    manifests = {}
    for ds_name in ("flood_optical", "flood_SAR", "flood_combined"):
        man = dedup_splits.build(out_dir, ds_name, conflict_policy="drop")
        dedup_splits.validate_manifest(man, expected_policy="drop")
        man_path = os.path.join(splits_dir, f"{ds_name}.json")
        with open(man_path, "w") as f:
            json.dump(man, f, indent=1)
            f.write("\n")
        manifests[ds_name] = man

    # Build fit-only splits/norm_stats.json (excluding validation holdout per seed)
    stats = compute_norm_stats.build_norm_stats(
        out_dir,
        datasets=["flood_optical", "flood_SAR", "flood_combined"],
        seed=int(seed),
        val_fraction=float(val_fraction),
        seeds=seed_list,
        fit_only=True,
        image_size=image_size,
    )
    stats_path = os.path.join(splits_dir, "norm_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=1)
        f.write("\n")

    return {
        "data_root": out_dir,
        "splits_dir": splits_dir,
        "stats_path": stats_path,
        "manifests": manifests,
        "norm_stats": stats,
    }


make_mini_flood_dataset = generate_mini_flood_dataset
build_mini_flood_dataset = generate_mini_flood_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", "--data-root", dest="out_dir", default=os.path.join(REPO_ROOT, "data_mini"),
                    help="Target directory for the generated mini dataset")
    ap.add_argument("--n-train", type=int, default=16, help="Training tiles per modality (default: 16)")
    ap.add_argument("--n-test", type=int, default=4, help="Testing tiles per modality (default: 4)")
    ap.add_argument("--image-size", type=int, default=32, help="Tile spatial size H=W (default: 32)")
    ap.add_argument("--seed", type=int, default=42, help="Generator and primary split seed (default: 42)")
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                    help=f"Validation holdout fraction excluded from fit-only norm stats (default: {DEFAULT_VAL_FRACTION})")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 17, 123],
                    help="Seeds precomputed in norm_stats.json['by_seed']")
    args = ap.parse_args()
    info = generate_mini_flood_dataset(
        args.out_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        image_size=args.image_size,
        seed=args.seed,
        val_fraction=args.val_fraction,
        seeds=args.seeds,
    )
    print(f"Generated mini flood dataset at: {info['data_root']}")
    for ds_name, man in info["manifests"].items():
        print(f"  [{ds_name}] train={len(man['train'])}, test={len(man['test'])}")
    print(f"  [norm_stats] {info['stats_path']}")


if __name__ == "__main__":
    main()
