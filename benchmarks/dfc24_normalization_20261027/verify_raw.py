"""Audit committed fit identities and selected band statistics against real tiles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import flood_data as fd
from config.seeds import SEEDS


def verify(data_root):
    stats_path = REPO / 'splits/norm_stats.json'
    stats = json.loads(stats_path.read_text())
    results = []
    for dataset in ('flood_optical', 'flood_SAR', 'flood_combined'):
        manifest = json.loads((REPO / 'splits' / (dataset + '.json')).read_text())
        images = data_root / dataset / 'Training/images'
        labels = data_root / dataset / 'Training/labels'
        masks = {name: fd.load_mask(fd.find_mask(str(labels), name), 128) for name in manifest['train']}
        default_names = None
        identities = {}
        for seed in SEEDS:
            names = fd.compute_fit_only_names(str(images), str(labels), manifest['train'],
                seed=seed, val_fraction=0.15, image_size=128, masks_by_name=masks)
            digest = fd._fit_names_digest(names)
            entry = stats[dataset]['by_seed'][str(seed)]['_provenance']
            assert digest == entry['fit_names_sha256']
            assert len(names) == entry['n_fit_train_tiles']
            identities[str(seed)] = dict(n_fit=len(names), fit_names_sha256=digest)
            if seed == 42:
                default_names = names
        # Independent arithmetic on selected raw bands: no call to the generator's
        # compute_modality_band_stats or compute_norm_stats.compute implementation.
        vectors = {}
        for name in default_names:
            with rasterio.open(images / (name + '.tif')) as source:
                modality = fd.MODALITY_BY_COUNT[source.count]
                band = 'GREEN' if modality == 'optical' else 'VV'
                index = fd.BAND_NAMES[modality].index(band) + 1
                stride = fd.STRIDE[modality]
                values = source.read(index)[::stride, ::stride].astype(np.float64).ravel()
            vectors.setdefault((modality, band), []).append(values[values != fd.NODATA])
        references = {}
        for (modality, band), chunks in vectors.items():
            values = np.concatenate(chunks)
            logged = band in fd.LOG_BANDS and float(np.median(values)) >= 0
            if logged:
                values = np.log1p(np.maximum(values, 0))
            lo, hi = np.percentile(values, [1, 99])
            actual = dict(lo=round(float(lo), 6), hi=round(float(hi), 6),
                mean=round(float(values.mean()), 6), std=round(float(values.std()), 6), log1p=logged)
            assert actual == stats[dataset][modality]['bands'][band], (dataset, modality, band)
            references[modality + '/' + band] = actual
        results.append(dict(dataset=dataset, fit_identities=identities, raw_reference_bands=references))
    return dict(norm_stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(), datasets=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(verify(args.data_root), indent=2, allow_nan=False) + '\n')
