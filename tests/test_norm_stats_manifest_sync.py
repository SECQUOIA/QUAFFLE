"""Regression tests for Issue #45: committed splits/norm_stats.json alignment
with conflict_policy='drop' manifests and fit-only validation exclusion."""

import json
import os
import pytest

import flood_data
from config.seeds import SEEDS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(REPO_ROOT, "splits")


def test_committed_manifests_use_drop_conflict_policy():
    """Verify committed split manifests enforce conflict_policy='drop' and 1,298 SAR train tiles."""
    with open(os.path.join(SPLITS_DIR, "flood_SAR.json")) as f:
        sar_man = json.load(f)
    with open(os.path.join(SPLITS_DIR, "flood_combined.json")) as f:
        comb_man = json.load(f)

    assert sar_man["conflict_policy"] == "drop"
    assert len(sar_man["train"]) == 1298
    assert sar_man["counts"]["train_kept"] == 1298

    assert comb_man["conflict_policy"] == "drop"
    assert len(comb_man["train"]) == 1533
    assert comb_man["counts"]["train_kept"] == 1533


def test_committed_norm_stats_aligned_with_drop_manifests_and_fit_only():
    """Committed splits/norm_stats.json must not retain stale 1,316-tile keep_first
    counts and must include fit-only provenance so flood_data.load_stats(..., fit_only=True)
    resolves offline and matches the 1,298-tile SAR drop manifest."""
    assert os.path.abspath(flood_data.__file__) == os.path.join(REPO_ROOT, "flood_data.py")

    with open(os.path.join(SPLITS_DIR, "norm_stats.json")) as f:
        norm_stats = json.load(f)
    with open(os.path.join(SPLITS_DIR, "flood_SAR.json")) as f:
        sar_man = json.load(f)

    # 1. _info must record fit_only=True rather than unsplit full-training stats
    info = norm_stats.get("_info", {})
    assert info.get("fit_only") is True, (
        f"Expected splits/norm_stats.json _info['fit_only'] is True, got {info!r}"
    )

    # 2. SAR tile count in norm_stats.json must not exceed the 1,298 kept SAR train tiles
    # in splits/flood_SAR.json (1,316 was the pre-drop keep_first count with 18 conflicting pairs)
    sar_entry = norm_stats["flood_SAR"]["sar"]
    assert sar_entry["n_train_tiles"] <= len(sar_man["train"]), (
        f"flood_SAR norm_stats n_train_tiles={sar_entry['n_train_tiles']} exceeds "
        f"splits/flood_SAR.json kept train count={len(sar_man['train'])} (stale keep_first=1316 stats)"
    )

    # 3. load_stats with fit_only=True for default seed=42, val_fraction=0.15 must resolve
    # from splits/norm_stats.json without requiring raw GeoTIFFs on disk
    resolved = flood_data.load_stats(
        "flood_SAR",
        stats_path=os.path.join(SPLITS_DIR, "norm_stats.json"),
        seed=42,
        val_fraction=0.15,
        fit_only=True,
    )
    assert resolved.get("_provenance", {}).get("fit_only") is True
    assert resolved["_provenance"]["n_manifest_train_tiles"] == len(sar_man["train"])


@pytest.mark.parametrize('dataset,modalities', [
    ('flood_optical', ['optical']), ('flood_SAR', ['sar']),
    ('flood_combined', ['optical', 'sar']),
])
@pytest.mark.parametrize('seed', SEEDS)
def test_all_campaign_seeds_resolve_committed_fit_only_stats_offline(tmp_path, monkeypatch, dataset, modalities, seed):
    def raw_statistics_forbidden(*args, **kwargs):
        pytest.fail('Committed campaign statistics must resolve without reading raw tiles')

    monkeypatch.setattr(flood_data, 'compute_modality_band_stats', raw_statistics_forbidden)
    with open(os.path.join(SPLITS_DIR, 'norm_stats.json')) as stream:
        committed = json.load(stream)
    with open(os.path.join(SPLITS_DIR, dataset + '.json')) as stream:
        manifest = json.load(stream)
    assert committed['_info']['seeds'] == SEEDS
    resolved = flood_data.load_stats(
        dataset, stats_path=os.path.join(SPLITS_DIR, 'norm_stats.json'),
        data_root=str(tmp_path / 'absent-data'), seed=seed, val_fraction=0.15,
        fit_only=True, image_size=128,
    )
    entry = committed[dataset]['by_seed'][str(seed)]
    assert resolved == entry == committed[dataset]['by_seed'][f'seed_{seed}_val_0.1500']
    provenance = resolved['_provenance']
    assert provenance['fit_only'] is True
    assert provenance['seed'] == seed
    assert provenance['val_fraction'] == 0.15
    assert provenance['image_size'] == 128
    assert provenance['max_samples'] is None
    assert provenance['n_manifest_train_tiles'] == len(manifest['train'])
    assert 0 < provenance['n_excluded_val_tiles'] < len(manifest['train'])
    assert provenance['n_fit_train_tiles'] + provenance['n_excluded_val_tiles'] == len(manifest['train'])
    assert len(provenance['fit_names_sha256']) == 16
    assert sum(resolved[modality]['n_train_tiles'] for modality in modalities) == provenance['n_fit_train_tiles']
    if seed == 42:
        primary = {key: value for key, value in committed[dataset].items() if key != 'by_seed'}
        assert primary == resolved
