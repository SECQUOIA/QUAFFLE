"""Native five-class LandCover.ai tiles; no binary flood-data transformations.

Prepared train/validation tiles retain 512x512 RGB bytes and class IDs 0..4.
Training draws one reproducible 128 crop per tile/epoch; validation enumerates
the full row-major nonoverlapping 128 grid. This module never opens test data.
"""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

FORMAT = 'landcover-ai-native-v1'
EXPECTED_ARCHIVE_MD5 = '3268c89070e8734b4e91d531c0617e03'
TILE_SIZE = 512
CLASS_IDS = tuple(range(5))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read_manifest(root):
    def unique(pairs):
        value = {}
        for key, child in pairs:
            require(key not in value, 'Duplicate manifest key: ' + key)
            value[key] = child
        return value
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(), object_pairs_hook=unique, parse_constant=invalid)
    require(manifest['format'] == FORMAT, 'Unknown native LandCover manifest format')
    require(manifest['archive']['md5'] == EXPECTED_ARCHIVE_MD5 and manifest['class_ids'] == list(CLASS_IDS)
            and manifest['test_pixel_windows_read'] == 0 and manifest['test_tiles_written'] == 0,
            'Prepared archive version/classes/test-access contract differs')
    splits = manifest['splits']
    require(set(splits) == {'train', 'val', 'test'}, 'Expected the three official splits')
    seen = set()
    for split, names in splits.items():
        require(isinstance(names, list) and names and all(isinstance(name, str) and name for name in names),
                'Invalid official tile ID list: ' + split)
        require(len(names) == len(set(names)) and not seen.intersection(names), 'Duplicate/overlapping official splits')
        seen.update(names)
    rows = manifest['tiles']
    require(set(rows) == set(splits['train']) | set(splits['val']), 'Prepared train/val grid differs')
    for split in ('train', 'val'):
        for name in splits[split]:
            row = rows[name]
            require(row['split'] == split and row['shape'] == [512, 512], 'Invalid prepared tile identity/shape')
            for key in ('image', 'mask'):
                relative = row[key]
                require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Expected relative PNG path')
                path = (root/relative).resolve()
                require(path.is_relative_to(root) and path.suffix.lower() == '.png', 'Prepared path escapes root')
            counts = row['class_pixel_counts']
            require(isinstance(counts, list) and len(counts) == 5
                    and all(type(value) is int and value >= 0 for value in counts)
                    and sum(counts) == 512*512, 'Invalid class pixel counts')
    return manifest


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest()[:8], 'little')


def deterministic_subset(tile_ids, count, seed=42):
    """Choose by a stable seeded ranking and retain the official list order."""
    require(type(seed) is int, 'Seed must be an integer')
    names = list(tile_ids)
    require(names and len(names) == len(set(names)) and all(isinstance(name, str) for name in names),
            'Expected unique tile IDs')
    require(type(count) is int and 0 < count <= len(names), 'Invalid subset size')
    selected = set(sorted(names, key=lambda name: (_seed(seed, name, 'subset'), name))[:count])
    return [name for name in names if name in selected]


class NativeLandCoverDataset(Dataset):
    """Return image/mask plus tile/crop identity; class0 stays class0.

    ``augment=None`` enables D4 only for training. ``set_epoch`` controls the
    deterministic training schedule independently of model/global RNG state.
    Validation has 16 examples per selected 512 tile at the default crop size.
    ``test`` is deliberately unsupported by this local-pilot loader.
    """
    def __init__(self, root, split='train', tile_ids=None, seed=42, epoch=0,
                 crop_size=128, augment=None):
        require(split in ('train', 'val'), 'Only official train/val data may be loaded')
        require(type(seed) is int, 'Seed must be an integer')
        require(type(crop_size) is int and 0 < crop_size <= TILE_SIZE and TILE_SIZE % crop_size == 0,
                'Crop size must be a positive divisor of 512')
        if augment is None:
            augment = split == 'train'
        require(type(augment) is bool and (split == 'train' or not augment), 'D4 is training-only')
        self.root, self.split = Path(root).resolve(), split
        self.manifest = read_manifest(self.root)
        names = list(self.manifest['splits'][split]) if tile_ids is None else list(tile_ids)
        require(names and len(names) == len(set(names))
                and set(names) <= set(self.manifest['splits'][split]), 'Invalid/duplicate selected tile IDs')
        self.tile_ids, self.seed, self.crop_size, self.augment = names, seed, crop_size, augment
        self._decoded = OrderedDict()
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        require(type(epoch) is int and epoch >= 0, 'Epoch must be a nonnegative integer')
        self.epoch = epoch

    def __len__(self):
        return len(self.tile_ids) * ((TILE_SIZE//self.crop_size)**2 if self.split == 'val' else 1)

    def __getitem__(self, index):
        require(type(index) is int and 0 <= index < len(self), 'Dataset index outside selected crop grid')
        n_side = TILE_SIZE//self.crop_size
        tile_index = index//(n_side*n_side) if self.split == 'val' else index
        tile_id = self.tile_ids[tile_index]
        row = self.manifest['tiles'][tile_id]
        image_path, mask_path = (self.root/row[key] for key in ('image', 'mask'))
        require(file_sha256(image_path) == row['image_sha256']
                and file_sha256(mask_path) == row['mask_sha256'], 'Prepared PNG bytes changed')
        # SHA checks above remain mandatory on EVERY access, including cache hits.
        # Four decoded tiles avoid repeated PNG decoding across a validation grid.
        if tile_id in self._decoded:
            image, mask = self._decoded.pop(tile_id)
        else:
            with Image.open(image_path) as opened:
                require(opened.mode == 'RGB' and opened.size == (512, 512), 'Expected lossless 512 RGB tile')
                image = np.array(opened)
            with Image.open(mask_path) as opened:
                require(opened.mode in ('L', 'I;16', 'I') and opened.size == (512, 512), 'Expected integer 512 mask')
                mask = np.array(opened)
            require(image.dtype == np.uint8 and np.issubdtype(mask.dtype, np.integer)
                    and mask.ndim == 2 and int(mask.min()) >= 0 and int(mask.max()) <= 4,
                    'Expected RGB uint8 and integer class IDs 0..4')
            require(np.bincount(mask.reshape(-1).astype(np.int64), minlength=5).tolist() == row['class_pixel_counts'],
                    'Prepared class counts differ from mask')
        self._decoded[tile_id] = (image, mask)
        if len(self._decoded) > 4:
            self._decoded.popitem(last=False)
        if self.split == 'train':
            rng = random.Random(_seed(self.seed, self.epoch, tile_id, 'crop'))
            x, y = rng.randrange(TILE_SIZE-self.crop_size+1), rng.randrange(TILE_SIZE-self.crop_size+1)
            d4 = _seed(self.seed, self.epoch, tile_id, 'D4') % 8 if self.augment else 0
        else:
            slot = index % (n_side*n_side)
            x, y = (slot % n_side)*self.crop_size, (slot//n_side)*self.crop_size
            d4 = 0
        image = image[y:y+self.crop_size, x:x+self.crop_size]
        mask = mask[y:y+self.crop_size, x:x+self.crop_size]
        image, mask = np.rot90(image, d4 % 4), np.rot90(mask, d4 % 4)
        if d4 >= 4:
            image, mask = np.fliplr(image), np.fliplr(mask)
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)/127.5-1
        mask = np.ascontiguousarray(mask, dtype=np.int64)
        return dict(image=torch.from_numpy(image), mask=torch.from_numpy(mask), tile_id=tile_id,
                    crop_x=x, crop_y=y, d4=d4, split=self.split)
