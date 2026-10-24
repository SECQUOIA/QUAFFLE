"""Unit and regression tests for Wave 1B (#19, #23, #30)."""

import argparse
import inspect
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import flood_data  # noqa: E402
from tools import compute_norm_stats, dedup_splits, make_mini_flood_dataset  # noqa: E402
from tools import training_common as tc  # noqa: E402
from utilsTorch import FloodSegmentationDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Issue #19: Duplicate image groups with conflicting masks in dedup_splits.py
# ---------------------------------------------------------------------------


def test_issue_19_dedup_splits_conflicting_masks_retained():
    """Triage reproduction test for #19: default collapse() drops duplicate groups
    whose ground-truth masks disagree."""
    records = [
        {"name": "tile_a", "image_pix": "img_hash_1", "mask_pix": "mask_hash_1"},
        {"name": "tile_b", "image_pix": "img_hash_1", "mask_pix": "mask_hash_2"},
    ]
    keep, dropped, n_groups, label_conflicts = dedup_splits.collapse(records)
    assert n_groups == 1
    assert label_conflicts == 1
    assert len(keep) == 0, (
        f"Expected conflicting duplicate group to be excluded or resolved, "
        f"but kept {[r['name'] for r in keep]}"
    )
    assert [d["name"] for d in dropped] == ["tile_a", "tile_b"]


def test_issue_19_conflict_policies_drop_majority_first():
    """Verify --conflict-policy {drop,majority,first} and return_details metadata."""
    records = [
        {"name": "tile_1", "image_pix": "img_conflict_3", "mask_pix": "mask_minority"},
        {"name": "tile_2", "image_pix": "img_conflict_3", "mask_pix": "mask_majority"},
        {"name": "tile_10", "image_pix": "img_conflict_3", "mask_pix": "mask_majority"},
        {"name": "tile_20", "image_pix": "img_clean", "mask_pix": "mask_clean"},
        {"name": "tile_21", "image_pix": "img_clean", "mask_pix": "mask_clean"},
    ]

    # 1. drop (default)
    keep_d, dropped_d, n_groups_d, conflicts_d, details_d = dedup_splits.collapse(
        records, conflict_policy="drop", return_details=True
    )
    assert n_groups_d == 2
    assert conflicts_d == 1
    assert [r["name"] for r in keep_d] == ["tile_20"]
    assert details_d["conflict_policy"] == "drop"
    assert len(details_d["conflict_groups"]) == 1
    assert details_d["conflict_groups"][0]["kept"] is None
    assert details_d["conflict_groups"][0]["dropped"] == ["tile_1", "tile_2", "tile_10"]
    assert [d["name"] for d in details_d["conflict_dropped"]] == ["tile_1", "tile_2", "tile_10"]

    # 2. majority (strict majority selects tile_2 with mask_majority over tile_1)
    keep_m, dropped_m, _, conflicts_m, details_m = dedup_splits.collapse(
        records, conflict_policy="majority", return_details=True
    )
    assert conflicts_m == 1
    assert [r["name"] for r in keep_m] == ["tile_2", "tile_20"]
    assert details_m["conflict_groups"][0]["kept"] == "tile_2"
    assert [d["name"] for d in details_m["conflict_dropped"]] == ["tile_1"]

    # 3. majority tie-breaking (1 vs 1 tie broken deterministically by natural sort order)
    tie_records = [
        {"name": "tile_10", "image_pix": "img_tie", "mask_pix": "mask_b"},
        {"name": "tile_2", "image_pix": "img_tie", "mask_pix": "mask_a"},
    ]
    keep_tie, _, _, _, details_tie = dedup_splits.collapse(
        tie_records, conflict_policy="majority", return_details=True
    )
    assert [r["name"] for r in keep_tie] == ["tile_2"]
    assert details_tie["conflict_groups"][0]["kept"] == "tile_2"

    # 4. first (and keep_first alias)
    for pol in ("first", "keep_first"):
        keep_f, _, _, conflicts_f, details_f = dedup_splits.collapse(
            records, conflict_policy=pol, return_details=True
        )
        assert conflicts_f == 1
        assert [r["name"] for r in keep_f] == ["tile_1", "tile_20"]
        assert details_f["conflict_policy"] == "first"
        assert details_f["conflict_groups"][0]["kept"] == "tile_1"


def test_issue_19_committed_split_manifests_and_offline_rebuild():
    """Verify committed splits/flood_*.json manifests have 0 conflicting groups in train
    under conflict_policy='drop' and can be deterministically rebuilt from hashes."""
    expected = {
        "flood_optical": {"train_kept": 235, "test_kept": 71, "conflict_groups": 0, "conflict_dropped": 0},
        "flood_SAR": {"train_kept": 1298, "test_kept": 134, "conflict_groups": 18, "conflict_dropped": 36},
        "flood_combined": {"train_kept": 1533, "test_kept": 205, "conflict_groups": 18, "conflict_dropped": 36},
    }
    for ds_name, exp in expected.items():
        man_path = os.path.join(ROOT, "splits", f"{ds_name}.json")
        with open(man_path) as f:
            man = json.load(f)
        assert man["conflict_policy"] == "drop"
        assert len(man["train"]) == exp["train_kept"]
        assert len(man["test"]) == exp["test_kept"]
        assert man["counts"]["train_kept"] == exp["train_kept"]
        assert man["counts"]["conflict_groups"] == exp["conflict_groups"]
        assert man["counts"]["conflict_dropped"] == exp["conflict_dropped"]
        assert len(man["conflict_groups"]) == exp["conflict_groups"]
        assert len(man["conflict_dropped"]) == exp["conflict_dropped"]
        assert dedup_splits.validate_manifest(man, expected_policy="drop") is True

        # Offline rebuild under 'first' recovers legacy counts (1316 SAR, 1551 combined)
        man_first = dedup_splits.build_from_manifest(man, conflict_policy="first")
        assert len(man_first["train"]) == exp["train_kept"] + exp["conflict_groups"]
        assert dedup_splits.validate_manifest(man_first, expected_policy="first") is True

    # Verify docs/DATA.md documents --conflict-policy and 1298 / 1533 counts
    with open(os.path.join(ROOT, "docs", "DATA.md")) as f:
        data_doc = f.read()
    assert "--conflict-policy" in data_doc
    assert "1298" in data_doc
    assert "1533" in data_doc


# ---------------------------------------------------------------------------
# Issue #23: Fit-only normalization statistics without validation holdout leakage
# ---------------------------------------------------------------------------


def test_issue_23_compute_norm_stats_includes_validation_holdout():
    """Triage reproduction test for #23: compute_norm_stats.compute accepts fit-only
    holdout exclusion parameters."""
    sig = inspect.signature(compute_norm_stats.compute)
    params = set(sig.parameters.keys())
    assert any(
        p in params for p in ("exclude_names", "val_fraction", "seed", "train_indices")
    ), (
        f"compute_norm_stats.compute({list(params)}) has no parameters to exclude "
        f"validation holdout indices/names before computing normalization percentiles"
    )


def test_issue_23_fit_only_norm_stats_excludes_validation_outliers(tmp_path):
    """Verify validation holdout tiles never leak into fit-only normalization percentiles."""
    data_root = str(tmp_path / "mini_leak_check")
    make_mini_flood_dataset.generate_mini_flood_dataset(
        data_root,
        n_train=12,
        n_test=4,
        image_size=16,
        seed=42,
        val_fraction=0.25,
        seeds=(42,),
    )
    train_img_dir = os.path.join(data_root, "flood_optical", "Training", "images")
    train_lbl_dir = os.path.join(data_root, "flood_optical", "Training", "labels")

    # Determine which tiles belong to train_idx vs val_idx for seed=42, val_fraction=0.25
    ds_initial = FloodSegmentationDataset(
        train_img_dir, train_lbl_dir, 16, data_root=data_root, seed=42, val_fraction=0.25
    )
    train_idx, val_idx = tc.split_train_val(ds_initial, val_fraction=0.25, seed=42)
    assert len(val_idx) >= 2
    val_names = [ds_initial.names[i] for i in val_idx]
    train_names = [ds_initial.names[i] for i in train_idx]

    # Inject extreme outlier values (500,000.0) into every validation tile on disk
    import rasterio

    for vname in val_names:
        tif_path = os.path.join(train_img_dir, f"{vname}.tif")
        with rasterio.open(tif_path) as src:
            arr = src.read()
            profile = src.profile.copy()
        arr[:] = 500000.0
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(arr)

    flood_data._FIT_STATS_CACHE.clear()

    # Fit-only stats (by seed/val_fraction, by train_indices, and by exclude_names)
    stats_seed = compute_norm_stats.compute(
        data_root, "flood_optical", seed=42, val_fraction=0.25, image_size=16
    )
    stats_idx = compute_norm_stats.compute(
        data_root, "flood_optical", train_indices=train_idx, image_size=16
    )
    stats_excl = compute_norm_stats.compute(
        data_root, "flood_optical", exclude_names=val_names, image_size=16
    )
    # Full-train stats (including validation holdout outliers)
    stats_leaked = compute_norm_stats.compute(
        data_root, "flood_optical", fit_only=False, val_fraction=0.0, image_size=16
    )

    assert stats_seed["optical"]["n_train_tiles"] == len(train_names)
    assert stats_seed["optical"]["bands"] == stats_idx["optical"]["bands"]
    assert stats_seed["optical"]["bands"] == stats_excl["optical"]["bands"]

    # Fit-only NIR upper bound is around normal reflectance (~2700), whereas leaked is 500,000
    nir_hi_fit = stats_seed["optical"]["bands"]["NIR"]["hi"]
    nir_hi_leaked = stats_leaked["optical"]["bands"]["NIR"]["hi"]
    assert nir_hi_fit < 10000.0
    assert nir_hi_leaked > 100000.0

    # Check centralized_fingerprint and run_fingerprint record norm_provenance
    cfg = {
        "dataset": "optical",
        "data_root": data_root,
        "seed": 42,
        "val_fraction": 0.25,
        "legacy_data": False,
        "num_epochs": 2,
        "fed_rounds": 2,
        "local_epochs": 1,
        "num_clients": 2,
        "clients_per_round": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "base_channels": 8,
        "image_size": 16,
    }
    cfp = tc.centralized_fingerprint(cfg, "classical")
    rfp = tc.run_fingerprint(cfg, "classical")
    assert cfp["norm_provenance"]["fit_only"] is True
    assert cfp["norm_provenance"]["seed"] == 42
    assert cfp["norm_provenance"]["val_fraction"] == 0.25
    assert rfp["norm_provenance"] == cfp["norm_provenance"]


# ---------------------------------------------------------------------------
# Issue #30: Pre-load manifest slicing, CLI overrides, mini dataset, and harness
# ---------------------------------------------------------------------------


def test_issue_30_local_mini_dataset_and_downscaled_cli_missing():
    """Triage reproduction test for #30: CLI flags and local verification scripts exist."""
    parser = argparse.ArgumentParser()
    tc.add_common_args(parser, quantum=True, federated=False)
    help_text = parser.format_help()

    missing_flags = [
        flag
        for flag in ("--data-root", "--base-channels", "--image-size", "--batch-size")
        if flag not in help_text
    ]
    assert not missing_flags, f"Missing CLI flags: {missing_flags}"
    assert os.path.exists(os.path.join(ROOT, "tools", "make_mini_flood_dataset.py"))
    assert os.path.exists(os.path.join(ROOT, "scripts", "verify_local_method.py"))


def test_issue_30_preload_manifest_slicing_and_strict_full_manifest_validation(tmp_path, monkeypatch):
    """Verify max_samples slices the manifest list BEFORE reading every TIFF from disk,
    without weakening full-manifest validation when max_samples=None."""
    data_root = str(tmp_path / "mini_slice")
    make_mini_flood_dataset.generate_mini_flood_dataset(
        data_root,
        n_train=10,
        n_test=4,
        image_size=16,
        seed=42,
        val_fraction=0.15,
        seeds=(42,),
    )
    train_img_dir = os.path.join(data_root, "flood_optical", "Training", "images")
    train_lbl_dir = os.path.join(data_root, "flood_optical", "Training", "labels")
    stats_path = os.path.join(data_root, "splits", "norm_stats.json")

    read_calls = []
    orig_read_raw = flood_data.read_raw

    def spy_read_raw(path):
        read_calls.append(os.path.basename(path))
        return orig_read_raw(path)

    monkeypatch.setattr(flood_data, "read_raw", spy_read_raw)

    samples, names = flood_data.load_samples(
        train_img_dir,
        train_lbl_dir,
        16,
        stats_path=stats_path,
        data_root=data_root,
        max_samples=3,
        sample_seed=42,
        seed=42,
        val_fraction=0.15,
    )
    assert len(samples) == 3
    assert len(names) == 3
    assert len(read_calls) == 3, (
        f"Expected pre-load slicing to read only 3 TIFFs from disk, but read {len(read_calls)}: {read_calls}"
    )

    # Full-manifest validation when max_samples=None must still raise if a manifest entry is missing
    os.remove(os.path.join(train_img_dir, "009.tif"))
    with pytest.raises(ValueError, match="manifest entries missing"):
        flood_data.load_samples(
            train_img_dir,
            train_lbl_dir,
            16,
            stats_path=stats_path,
            data_root=data_root,
            max_samples=None,
            seed=42,
            val_fraction=0.15,
        )


def test_issue_30_mini_dataset_bands_nodata_and_cli_overrides(tmp_path):
    """Verify synthetic mini dataset has 12-band float32 Optical + 6-band float32 SAR
    with -9999 DEM nodata and that apply_common_args wires all CLI overrides."""
    data_root = str(tmp_path / "mini_bands")
    info = make_mini_flood_dataset.generate_mini_flood_dataset(
        data_root, n_train=8, n_test=3, image_size=24, seed=17
    )

    opt_tif = os.path.join(data_root, "flood_optical", "Training", "images", "000.tif")
    sar_tif = os.path.join(data_root, "flood_SAR", "Training", "images", "000.tif")
    opt_raw = flood_data.read_raw(opt_tif)
    sar_raw = flood_data.read_raw(sar_tif)

    assert opt_raw.shape == (12, 24, 24)
    assert opt_raw.dtype == np.float32
    assert np.any(opt_raw[8] == flood_data.NODATA)  # MERIT_DEM nodata
    assert np.any(opt_raw[9] == flood_data.NODATA)  # COP_DEM nodata

    assert sar_raw.shape == (6, 24, 24)
    assert sar_raw.dtype == np.float32
    assert float(sar_raw[0].max()) < 0.0 and float(sar_raw[1].max()) < 0.0  # VV, VH in dB scale
    assert np.any(sar_raw[2] == flood_data.NODATA)  # MERIT_DEM nodata
    assert np.any(sar_raw[3] == flood_data.NODATA)  # COP_DEM nodata
    assert os.path.isfile(info["stats_path"])
    with open(info["stats_path"]) as f:
        st_json = json.load(f)
    assert st_json["flood_SAR"]["sar"]["bands"]["VV"]["log1p"] is False
    assert st_json["flood_SAR"]["sar"]["bands"]["VH"]["log1p"] is False

    # Verify CLI overrides and fingerprint sensitivity
    parser = argparse.ArgumentParser()
    tc.add_common_args(parser, quantum=True, federated=False)
    args = parser.parse_args(
        [
            "--data-root", data_root,
            "--base-channels", "8",
            "--image-size", "24",
            "--batch-size", "2",
            "--epochs", "3",
        ]
    )
    cfg = {
        "image_size": 128,
        "base_channels": 32,
        "init_dim": 32,
        "resnet_block_groups": 4,
        "quantum_channels": 32,
        "batch_size": 8,
        "num_epochs": 100,
        "eval_every": 10,
        "log_every": 10,
        "learning_rate": 1e-4,
        "seed": 17,
    }
    tc.apply_common_args(cfg, args)
    assert cfg["data_root"] == data_root
    assert cfg["base_channels"] == 8
    assert cfg["init_dim"] == 8
    assert cfg["image_size"] == 24
    assert cfg["batch_size"] == 2
    assert cfg["num_epochs"] == 3
    assert cfg["eval_every"] == 3

    # Changing data_root invalidates centralized and federated checkpoint fingerprints
    fp1 = tc.centralized_fingerprint(cfg, "quantum")
    cfg2 = dict(cfg, data_root=str(tmp_path / "other_root"))
    fp2 = tc.centralized_fingerprint(cfg2, "quantum")
    assert fp1["data_root"] != fp2["data_root"]
    assert tc.checkpoint_matches_centralized(
        {"run_format": tc.RUN_FORMAT, "q_backend": "pennylane", "fingerprint": fp1},
        cfg2,
        "quantum",
    ) is False


def test_review_r1_max_samples_image_size_dataset_root_and_manifest_validation(tmp_path, monkeypatch):
    """Regression test for Round 1 review findings:
      1. --max-samples bounds TIFF reads to max_samples and excludes capped val_idx from norm stats.
      2. --image-size stratification changes invalidate 128-px precomputed stats and recompute fit-only stats.
      3. Dataset-level --data-root resolves sibling Training/ for Testing/ stats, and explicit fit_only=True
         without local tiles or matching precomputed stats raises KeyError instead of falling back silently.
      4. validate_manifest() rejects tampered conflict_policy, conflict_groups, counts, and excluded metadata.
    """
    data_root = str(tmp_path / "r1_mini")
    info = make_mini_flood_dataset.generate_mini_flood_dataset(
        data_root, n_train=12, n_test=4, image_size=32, seed=42, val_fraction=0.15, seeds=(42,)
    )
    ds_root = os.path.join(data_root, "flood_optical")
    train_img_dir = os.path.join(ds_root, "Training", "images")
    train_lbl_dir = os.path.join(ds_root, "Training", "labels")
    test_img_dir = os.path.join(ds_root, "Testing", "images")
    test_lbl_dir = os.path.join(ds_root, "Testing", "labels")

    # 1. With max_samples=4, load_samples reads at most 4 TIFFs and excludes the 4-tile validation holdout
    flood_data._FIT_STATS_CACHE.clear()
    flood_data._LAST_TRAIN_STATS_BY_DIR.clear()
    read_calls = []
    orig_read_raw = flood_data.read_raw

    def spy_read_raw(path):
        read_calls.append(os.path.basename(path))
        return orig_read_raw(path)

    monkeypatch.setattr(flood_data, "read_raw", spy_read_raw)
    samples, names = flood_data.load_samples(
        train_img_dir,
        train_lbl_dir,
        32,
        stats_path=info["stats_path"],
        data_root=ds_root,  # dataset-level data_root
        max_samples=4,
        sample_seed=42,
        seed=42,
        val_fraction=0.15,
        fit_only=True,
    )
    assert len(samples) == 4
    assert len(read_calls) == 4, f"Expected exactly 4 TIFF reads for max_samples=4, got {len(read_calls)}"
    prov = flood_data.get_last_loaded_stats_provenance(train_img_dir)
    assert prov["fit_only"] is True
    assert prov["max_samples"] == 4
    assert prov["n_manifest_train_tiles"] == 4
    assert prov["n_fit_train_tiles"] < 4

    # Paired Testing load with dataset-level data_root reuses the exact capped training stats without reading training TIFFs
    read_calls.clear()
    test_samples, _ = flood_data.load_samples(
        test_img_dir,
        test_lbl_dir,
        32,
        stats_path=info["stats_path"],
        data_root=ds_root,
        seed=42,
        val_fraction=0.15,
        fit_only=True,
    )
    assert len(test_samples) == 4
    assert len(read_calls) == 4  # Only the 4 testing TIFFs read
    test_prov = flood_data.get_last_loaded_stats_provenance(test_img_dir)
    assert test_prov["stats_sha256"] == prov["stats_sha256"]
    assert test_prov["n_fit_train_tiles"] == prov["n_fit_train_tiles"]

    # 2. Explicit fit_only=True for a missing seed without local training tiles must raise KeyError
    with pytest.raises(KeyError, match="No matching fit-only normalization stats"):
        flood_data.load_stats(
            "flood_SAR",
            stats_path=os.path.join(ROOT, "splits", "norm_stats.json"),
            seed=99999,
            val_fraction=0.15,
            fit_only=True,
            data_root=str(tmp_path / "nonexistent_root"),
        )

    # 3. validate_manifest() rejects tampered metadata
    tampered = json.loads(json.dumps(info["manifests"]["flood_SAR"]))
    tampered["conflict_groups"] = 999
    with pytest.raises(ValueError, match="conflict_groups"):
        dedup_splits.validate_manifest(tampered, expected_policy="drop")
    tampered2 = json.loads(json.dumps(info["manifests"]["flood_SAR"]))
    tampered2["conflict_policy"] = "first"
    with pytest.raises(ValueError, match="conflict_policy"):
        dedup_splits.validate_manifest(tampered2, expected_policy="drop")
