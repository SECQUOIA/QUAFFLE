"""Native multiclass/window semantics using synthetic source GeoTIFFs only."""
import hashlib
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import torch

from scripts import prepare_landcover_native as prep
from tools import landcover_data as data


def tiff_bytes(array):
    bands = array[None] if array.ndim == 2 else array
    with MemoryFile() as memory:
        with memory.open(driver='GTiff', width=bands.shape[2], height=bands.shape[1], count=bands.shape[0],
                         dtype=bands.dtype, transform=from_origin(1, 1026, 1, 1)) as dataset:
            dataset.write(bands)
        return memory.read()


@pytest.fixture
def archive_factory(tmp_path, monkeypatch):
    serial = [0]
    def make(*, invalid_label=False, invalid_test_label=False, mask_shape=False, rgb_bands=3,
             overlap=False, extra=None, missing_pair=False, omitted_tile=False):
        serial[0] += 1
        y, x = np.indices((1025, 1025))
        labels = ((x//128+y//128) % 5).astype(np.uint8)
        rgb = np.stack((labels*50, (x % 256).astype(np.uint8), (y % 256).astype(np.uint8)))[:rgb_bands]
        if invalid_label:
            labels[0, 0] = 5
        if invalid_test_label:
            labels[512:1024, :512] = 99  # ONLY source tile k3, the test tile.
        if mask_shape:
            labels = labels[:1024]
        # k increments on partial right edges: complete windows are0,1,3,4.
        train = ['scene_0', 'scene_4']
        val = ['scene_0' if overlap else 'scene_1']
        test = ['scene_3']
        if omitted_tile:
            train.pop()
        script = b"raise RuntimeError('archive code must never execute')\n"
        path = tmp_path/f'synthetic-{serial[0]}.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('images/scene.tif', tiff_bytes(rgb))
            if not missing_pair:
                archive.writestr('masks/scene.tif', tiff_bytes(labels))
            for split, names in [('train', train), ('val', val), ('test', test)]:
                archive.writestr(split+'.txt', '\n'.join(names)+'\n')
            archive.writestr('split.py', script)
            if extra:
                archive.writestr(extra, b'unsafe or unexpected')
        # Fixtures explicitly replace the official identity. Production has no
        # CLI/API override and always checks the fixed published v1 MD5.
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        monkeypatch.setattr(prep, 'OFFICIAL_MD5', md5)
        monkeypatch.setattr(data, 'EXPECTED_ARCHIVE_MD5', md5)
        monkeypatch.setattr(prep, 'OFFICIAL_SPLIT_SHA256', hashlib.sha256(script).hexdigest())
        monkeypatch.setattr(prep, 'OFFICIAL_COUNTS', dict(train=len(train), val=1, test=1))
        return path, tmp_path/f'prepared-{serial[0]}', rgb, labels
    return make


def test_lossless_class_ids_partial_edge_numbering_and_no_test_pixel_reads(archive_factory, monkeypatch):
    path, out, original_rgb, original_mask = archive_factory(invalid_test_label=True)
    actual_open = rasterio.open
    requests = []
    class ObservedRaster:
        def __init__(self, dataset, name):
            self.dataset, self.name = dataset, name
        def __enter__(self):
            self.dataset.__enter__(); return self
        def __exit__(self, *args):
            return self.dataset.__exit__(*args)
        def __getattr__(self, key):
            return getattr(self.dataset, key)
        def read(self, *args, **kwargs):
            window = kwargs.get('window')
            requests.append((self.name, window.col_off, window.row_off, window.width, window.height))
            return self.dataset.read(*args, **kwargs)
    monkeypatch.setattr(rasterio, 'open', lambda name, *args, **kwargs:
                        ObservedRaster(actual_open(name, *args, **kwargs), str(name)))
    manifest = prep.prepare_archive(path, out)
    assert len(requests) == 6  # RGB+mask for exactly the3 train/val windows.
    assert {(x, y, w, h) for _, x, y, w, h in requests} == {(0, 0, 512, 512), (512, 0, 512, 512), (512, 512, 512, 512)}
    assert not any((x, y) == (0, 512) for _, x, y, _, _ in requests)
    assert manifest['tile_origins']['scene_4']['x'] == 512
    assert manifest['tile_origins']['scene_4']['y'] == 512
    assert manifest['test_pixel_windows_read'] == manifest['test_tiles_written'] == 0
    assert not (out/'test').exists() and set(manifest['tiles']) == {'scene_0', 'scene_1', 'scene_4'}
    assert manifest['archive']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(manifest['sources']) == 2
    for name, row in manifest['tiles'].items():
        x, y = row['source_window']['x'], row['source_window']['y']
        rgb, mask = np.array(Image.open(out/row['image'])), np.array(Image.open(out/row['mask']))
        np.testing.assert_array_equal(rgb, original_rgb[:, y:y+512, x:x+512].transpose(1, 2, 0))
        np.testing.assert_array_equal(mask, original_mask[y:y+512, x:x+512])
        assert set(np.unique(mask)) == set(range(5))
        assert row['class_pixel_counts'] == np.bincount(mask.reshape(-1), minlength=5).tolist()
        assert sum(row['class_pixel_counts']) == 512*512


@pytest.mark.parametrize('case, match', [
    ({'invalid_label': True}, 'class label'), ({'mask_shape': True}, 'shapes differ'),
    ({'rgb_bands': 1}, 'RGB uint8'), ({'overlap': True}, 'overlapping'),
    ({'missing_pair': True}, 'Unpaired'), ({'omitted_tile': True}, 'cover exactly'),
    ({'extra': '../escape.txt'}, 'Unsafe ZIP'), ({'extra': '/absolute.txt'}, 'Unsafe ZIP'),
    ({'extra': 'images\\escape.tif'}, 'Unsafe ZIP'), ({'extra': 'execute.sh'}, 'Unexpected'),
])
def test_preparation_refuses_corrupt_or_unsafe_inputs(archive_factory, case, match):
    path, out, _, _ = archive_factory(**case)
    with pytest.raises(ValueError, match=match):
        prep.prepare_archive(path, out)
    assert not (out/'manifest.json').exists()
    assert not (out.parent/'escape.txt').exists()


def test_official_version_checksum_and_script_semantics_guards(archive_factory, monkeypatch):
    path, out, _, _ = archive_factory()
    monkeypatch.setattr(prep, 'OFFICIAL_MD5', '0'*32)
    with pytest.raises(ValueError, match='checksum mismatch'):
        prep.prepare_archive(path, out)
    assert not out.exists()
    monkeypatch.setattr(prep, 'OFFICIAL_MD5', hashlib.md5(path.read_bytes()).hexdigest())
    monkeypatch.setattr(prep, 'OFFICIAL_SPLIT_SHA256', '0'*64)
    with pytest.raises(ValueError, match='coordinate semantics'):
        prep.prepare_archive(path, out)
    assert not out.exists()


@pytest.mark.parametrize('name', ['scene_2', 'scene_5', 'scene_9', 'scene_-1', '../scene_0', 'scene_00'])
def test_tile_ids_cannot_name_skipped_edges_or_escape(name):
    with pytest.raises(ValueError):
        prep.tile_origin(name, 1025, 1025)


def test_train_crop_pairing_rng_independence_epoch_and_validation_grid(archive_factory):
    path, out, _, _ = archive_factory()
    prep.prepare_archive(path, out)
    first = data.NativeLandCoverDataset(out, seed=77, epoch=9)
    duplicate = data.NativeLandCoverDataset(out, seed=77, epoch=9)
    a = first[0]
    random.seed(991); np.random.seed(992); torch.manual_seed(993)
    b = duplicate[0]
    assert a['tile_id'] == b['tile_id'] and a['crop_x'] == b['crop_x'] and a['crop_y'] == b['crop_y'] and a['d4'] == b['d4']
    assert torch.equal(a['image'], b['image']) and torch.equal(a['mask'], b['mask'])
    assert a['image'].shape == (3, 128, 128) and a['image'].dtype == torch.float32
    assert a['mask'].shape == (128, 128) and a['mask'].dtype == torch.long
    assert float(a['image'].min()) >= -1 and float(a['image'].max()) <= 1
    # The first RGB channel independently encodes the ORIGINAL label value.
    recovered = torch.round((a['image'][0]+1)*127.5/50).long()
    assert torch.equal(recovered, a['mask'])
    initial = (a['crop_x'], a['crop_y'], a['d4'])
    first.set_epoch(10)
    later = first[0]
    assert (later['crop_x'], later['crop_y'], later['d4']) != initial
    val = data.NativeLandCoverDataset(out, split='val')
    assert len(val) == 16 and len(first) == 2
    coords = [(val[i]['crop_x'], val[i]['crop_y']) for i in range(16)]
    assert coords == [(x, y) for y in (0, 128, 256, 384) for x in (0, 128, 256, 384)]
    before = val[5]; val.set_epoch(999); after = val[5]
    assert torch.equal(before['image'], after['image']) and before['d4'] == after['d4'] == 0


def test_cached_pixels_still_reject_changed_png_and_manifest_counts(archive_factory):
    path, out, _, _ = archive_factory()
    manifest = prep.prepare_archive(path, out)
    val = data.NativeLandCoverDataset(out, split='val')
    val[0]  # Populate decoded cache.
    image_path = out/manifest['tiles']['scene_1']['image']
    with image_path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='PNG bytes changed'):
        val[1]
    # A semantically corrupt count row is rejected even when totals still match.
    train_row = manifest['tiles']['scene_0']
    train_row['class_pixel_counts'][0] -= 1; train_row['class_pixel_counts'][1] += 1
    (out/'manifest.json').write_text(json.dumps(manifest))
    train = data.NativeLandCoverDataset(out)
    with pytest.raises(ValueError, match='class counts differ'):
        train[0]


@pytest.mark.parametrize('kwargs', [dict(split='test'), dict(split='val', augment=True),
    dict(tile_ids=['scene_3']), dict(tile_ids=['scene_0', 'scene_0']),
    dict(crop_size=129), dict(seed=True), dict(epoch=-1)])
def test_dataset_rejects_wrong_split_subset_and_bounds(archive_factory, kwargs):
    path, out, _, _ = archive_factory()
    prep.prepare_archive(path, out)
    with pytest.raises(ValueError):
        data.NativeLandCoverDataset(out, **kwargs)


def test_prepared_version_subset_order_and_failed_attempt_preservation(archive_factory, monkeypatch):
    path, out, _, _ = archive_factory()
    prep.prepare_archive(path, out)
    original = (out/'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='partial attempts are preserved'):
        prep.prepare_archive(path, out)
    assert (out/'manifest.json').read_bytes() == original
    names = ['a', 'b', 'c', 'd', 'e', 'f']
    selected = data.deterministic_subset(names, 3, seed=42)
    random.seed(939)
    assert data.deterministic_subset(names, 3, seed=42) == selected
    assert len(selected) == 3 and [name for name in names if name in selected] == selected
    with pytest.raises(ValueError):
        data.deterministic_subset(names, 7)
    monkeypatch.setattr(data, 'EXPECTED_ARCHIVE_MD5', '0'*32)
    with pytest.raises(ValueError, match='archive version'):
        data.NativeLandCoverDataset(out)
