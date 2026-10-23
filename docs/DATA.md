# Flood data: splits, bands and normalization

This document covers the three DFC 2024 flood datasets under `data/`
(`flood_optical`, `flood_SAR`, `flood_combined`). Nothing under `data/` is
modified. All fixes live in committed files that the loaders read.

## Files

| File | Purpose |
|---|---|
| `flood_data.py` | Shared loader used by `utilsTorch.FloodSegmentationDataset`, `utilsTorch.get_dataloaders` and `utilsJAX.load_dataset` |
| `tools/dedup_splits.py` | Hashes every image/mask pair and writes `splits/<dataset>.json` |
| `tools/compute_norm_stats.py` | Computes per-band statistics on the training split only and writes `splits/norm_stats.json` |
| `splits/flood_optical.json`, `splits/flood_SAR.json`, `splits/flood_combined.json` | Split manifests (kept names, exclusions with reasons, per-file hashes) |
| `splits/norm_stats.json` | Training-split per-band statistics |

Regenerate with:

```
python tools/dedup_splits.py
python tools/compute_norm_stats.py
```

Both tools are deterministic. The manifests store file md5s and decoded
pixel md5s for every pair, so a changed dataset can be detected by rerunning
the tool and diffing the JSON.

## Building flood_combined

`flood_combined` is not a separate download. It holds every file of
`flood_optical` and `flood_SAR`, unchanged, in one folder per split
(`Training`, `Testing`) and kind (`images`, `labels`):

* Optical files keep their names.
* A SAR file whose name already exists in the optical folder of the same
  split and kind gets ` copy` before the extension. For example SAR
  `Training/images/164.tif` becomes `Training/images/164 copy.tif` and its
  label `Training/labels/164.png` becomes `Training/labels/164 copy.png`.
* All other SAR files keep their names.

With the tiles used here this renames 142 SAR training pairs (names 164 to
305) and 71 SAR test pairs (names 0 to 70), which gives 1702 training and 235
test pairs. The file md5s in the three manifests confirm that every
combined file is a byte-identical copy of its optical or SAR source. The
loader checks the folder against `splits/flood_combined.json` and stops with
an error if a listed name is missing.

From inside `data/`:

```bash
for split in Training Testing; do
  for kind in images labels; do
    dst=flood_combined/$split/$kind
    mkdir -p "$dst"
    cp flood_optical/$split/$kind/* "$dst/"
    for f in flood_SAR/$split/$kind/*; do
      b=$(basename "$f")
      if [ -e "flood_optical/$split/$kind/$b" ]; then
        b="${b%.*} copy.${b##*.}"
      fi
      cp "$f" "$dst/$b"
    done
  done
done
```

## Train/test leakage

### Method

* Identity of a tile is the md5 of its decoded pixel array (dtype, shape and
  raw bytes), so GeoTIFF metadata does not matter. File md5s are recorded too.
* Training: duplicate tiles are collapsed to one copy. The first name in
  natural sort order is kept.
* Testing: every test tile whose pixel hash appears anywhere in Training is
  dropped. Remaining duplicates inside Testing are collapsed to one copy.
* Masks are hashed as well. Mask agreement is recorded per exclusion but the
  decision uses the image hash only (an identical input in train and test is
  leakage regardless of its label).

### Counts (image/mask pairs)

| Dataset | Split | Before | Dropped | Reason | After |
|---|---|---|---|---|---|
| flood_optical | Training | 235 | 0 | none | 235 |
| flood_optical | Testing | 71 | 0 | none | 71 |
| flood_SAR | Training | 1467 | 151 | duplicate of another training tile | 1316 |
| flood_SAR | Testing | 164 | 29 | pixel-identical to a training tile | |
| flood_SAR | Testing | | 1 | duplicate inside Testing (128 == 98) | 134 |
| flood_combined | Training | 1702 | 151 | duplicate of another training tile | 1551 |
| flood_combined | Testing | 235 | 29 | pixel-identical to a training tile | |
| flood_combined | Testing | | 1 | duplicate inside Testing (128 == 98) | 205 |

Details:

* Of the 29 leaked SAR test tiles, 28 also have a byte-identical mask. The
  remaining one (Testing 41 vs Training 300) has the same image with a
  different mask.
* Among the 151 duplicate training tiles there are 18 duplicate groups whose
  masks disagree. One copy is kept, chosen by name order.
* `flood_combined` is optical plus SAR (name clashes resolved with a ` copy`
  suffix on the SAR file, see "Building flood_combined"), so it inherits exactly the SAR leakage and duplicates. The
  optical part of combined is clean.

### Near-duplicate check

Run on the tiles that survive the exact-hash step, test against train:

* Flipped or rotated copies (7 dihedral transforms, exact pixel hash): 0 in
  every dataset.
* Identical 8x8 average hash plus correlation above 0.99 on a 64x64
  block-mean of the first two bands: 0 in every dataset.
* A looser exploratory scan (best 64x64 correlation per test tile) found no
  test tile above 0.99. The highest values were 0.977 (SAR) and 0.975
  (optical). 15 SAR test tiles and 2 optical test tiles exceed 0.95. These
  look like overlapping or neighbouring patches of the same flood events, not
  copies, so they are reported and not excluded. Event-level overlap between
  the provided Training and Testing folders cannot be ruled out without the
  original tile geolocation, which the tiles do not carry (no geotransform).

## Input bands and normalization

### Band order

The tiles have no band descriptions or tags (checked with rasterio, all
`None`) and no georeferencing. The band order below is taken from the
official contest page, figure "format of the data sets according to the
different tracks" on
https://www.grss-ieee.org/community/technical-committees/2024-ieee-grss-data-fusion-contest/
(image `wp-content/uploads/2023/12/formattracks.png`) together with the
track 2 band table on the same page (`Table.png`).

SAR, 6 bands, 512 x 512, 10 m (flood_SAR):

| Index | Name in code | Content | Observed range in the tiles |
|---|---|---|---|
| 1 | VV | Sentinel-1 VV | 0 to 32765, heavy tail, saturates at 32765 |
| 2 | VH | Sentinel-1 VH | 0 to 32765 |
| 3 | MERIT_DEM | MERIT DEM | metres, e.g. 120 to 160 |
| 4 | COP_DEM | Copernicus DEM | metres, tracks band 3 closely |
| 5 | WORLDCOVER | ESA WorldCover class | multiples of 10, 10 to 90 |
| 6 | WATER_OCC | Global Surface Water occurrence | 0 to 100 |

Optical, 12 bands, 128 x 128 in the copy used here (the contest figure says 512
pixels at 30 m) (flood_optical):

| Index | Name in code | Content | Observed range |
|---|---|---|---|
| 1 to 7 | COASTAL, BLUE, GREEN, RED, NIR, SWIR1, SWIR2 | HLS surface reflectance, scaled integers | about -1400 to 8700 |
| 8 | QA | HLS quality band (bit field) | 28 distinct values, 64 to 240 |
| 9 | MERIT_DEM | MERIT DEM | includes -9999 nodata |
| 10 | COP_DEM | Copernicus DEM | tracks band 9 |
| 11 | WORLDCOVER | ESA WorldCover class | 10 to 90, 8 distinct values |
| 12 | WATER_OCC | Global Surface Water occurrence | 0 to 105 |

What is verified and what is not:

* Verified: the number and order of layers per track against the official
  figure. The observed value ranges match every layer type (two DEM bands
  that track each other, a class map in steps of 10, an occurrence layer in
  0 to 100, a bit-field QA band, reflectance-like spectral bands).
* Not verified: which of the two DEM bands is MERIT and which is Copernicus
  (they are numerically almost identical, so the choice barely matters), and
  the exact order of the seven optical spectral bands. The figure lists
  coastal aerosol, blue, green, red, NIR, SWIR1, SWIR2 with ellipses, and
  this order matches HLS conventions, but no per-band metadata in the files
  confirms it.
* The old loader used bands 1 to 3. For SAR that is VV, VH and the MERIT
  DEM, for optical it is coastal aerosol, blue and green. The value range
  of SAR band 3 (about 120 to 160) is consistent with it being a DEM.

### Default band selection (in_channels stays 3)

| Modality | Default bands | Rationale |
|---|---|---|
| SAR | VV, VH, COP_DEM | Both polarisations plus terrain, same physical content as the legacy first three bands but with the DEM chosen explicitly |
| Optical | GREEN, NIR, SWIR1 | Standard water indices (NDWI, MNDWI) are built from these |

Override with `bands=` on the loader functions (flat list such as
`["RED", "GREEN", "BLUE"]`, comma string, or a dict per modality), with
`set_data_defaults(bands=...)`, or with `config.bands` for the JAX loader.
Changing the number of bands requires a matching `in_channels`.

### Normalization

* Statistics come only from the deduplicated training split of each dataset
  (`splits/norm_stats.json`, built by `tools/compute_norm_stats.py`).
  A regular pixel grid is sampled (every 8th row and column for SAR, every
  2nd for optical).
* VV and VH are transformed with log1p first (they are heavy tailed and
  saturate at 32765).
* Each band is mapped with fixed dataset-level bounds:
  `clip((x - p1) / (p99 - p1), 0, 1) * 2 - 1`, where p1 and p99 are the
  training percentiles. Pixels equal to -9999 are excluded from the
  statistics and mapped to -1.
* Resizing to `image_size` is done in float32 with Lanczos, then clipped to
  [-1, 1]. The old path quantised to uint8 before resizing.
* This keeps absolute intensity, which the old per-image 2 to 98 percentile
  stretch removed (a tile of open water and a tile of dry land both used
  the full range).
* For `flood_combined` the SAR and optical tiles each use their own
  modality statistics, computed from the training tiles of that modality in
  combined. Note that combined still mixes two sensors with different
  semantics per channel position.

## Opting out (legacy behaviour)

The legacy path reproduces the original loader bit for bit: every file in
the folder, unsorted glob order, first three bands, per-image percentile
stretch, uint8 round trip. Any of these selects it:

* `FloodSegmentationDataset(..., legacy=True)`, `get_dataloaders(..., legacy=True)`, `load_dataset(..., legacy=True)`
* `config.legacy_data = True` for the JAX loader
* `set_data_defaults(legacy=True)` once at start-up (exported from both
  `utilsTorch` and `utilsJAX`)
* environment variable `QUAFFLE_LEGACY_DATA=1` (used only when neither of the
  above is set)

`manifest=False` keeps the new bands and normalization but disables the
split manifest. `manifest=<path or dict>` uses a different manifest.

Directories that are not one of the three flood datasets (or not under a
`Training`/`Testing` folder) are loaded without manifest filtering.
Clean mode still needs normalization statistics for them. Add an entry to
`splits/norm_stats.json` (or a `_default` entry) or use `legacy=True`.
