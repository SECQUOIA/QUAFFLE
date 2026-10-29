#!/usr/bin/env python3
"""Prepare official LandCover.ai v1 train/val windows as lossless native PNGs.

No archive code is executed or extracted. The inspected official split.py uses
row-major k, incremented even for incomplete edge windows. Test IDs and raster
headers are inspected for split completeness; no test pixel window is requested.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OFFICIAL_MD5 = '3268c89070e8734b4e91d531c0617e03'
OFFICIAL_SPLIT_SHA256 = '15ee4ca9e3fd187957addfa8f0d74ac31bc928a966f76926e11b3c33ea76daa1'
OFFICIAL_COUNTS = dict(train=7470, val=1602, test=1602)
TILE_SIZE = 512
FORMAT = 'landcover-ai-native-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            md5.update(block); sha.update(block)
    return dict(md5=md5.hexdigest(), sha256=sha.hexdigest())


def stream_sha256(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''):
        value.update(block)
    return value.hexdigest()


def tile_origin(tile_id, width, height):
    require(type(width) is int and type(height) is int and width > 0 and height > 0,
            'Invalid source raster dimensions')
    match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9-]*)_(0|[1-9]\d*)', tile_id)
    require(match is not None, 'Invalid official tile ID: ' + str(tile_id))
    source, index = match[1], int(match[2])
    columns = (width+511)//512
    x, y = (index % columns)*512, (index//columns)*512
    require(x+512 <= width and y+512 <= height, 'Tile ID names an incomplete/out-of-bounds window: ' + tile_id)
    return dict(source=source, index=index, x=x, y=y, width=512, height=512)


def _members(archive):
    infos = archive.infolist()
    names = [info.filename for info in infos]
    require(len(names) == len(set(names)), 'Duplicate ZIP members')
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        require(name and not path.is_absolute() and '..' not in path.parts and '\\' not in name
                and not any(ord(c) < 32 for c in name) and path.as_posix() == name.rstrip('/'),
                'Unsafe ZIP member path: ' + repr(name))
        require(stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK and not info.flag_bits & 1,
                'Symlink/encrypted ZIP members are unsupported')
        if not info.is_dir():
            require(name in ('split.py', 'train.txt', 'val.txt', 'test.txt')
                    or re.fullmatch(r'(?:images|masks)/[A-Za-z0-9][A-Za-z0-9-]*\.tif', name),
                    'Unexpected official archive member: ' + name)
    require({'split.py', 'train.txt', 'val.txt', 'test.txt'} <= set(names), 'Missing official split metadata')
    images = {Path(name).stem for name in names if name.startswith('images/') and name.endswith('.tif')}
    masks = {Path(name).stem for name in names if name.startswith('masks/') and name.endswith('.tif')}
    require(images and images == masks, 'Unpaired original image/mask TIFFs')
    return sorted(images)


def _splits(archive):
    result, bindings, seen = {}, {}, set()
    for split in ('train', 'val', 'test'):
        raw = archive.read(split+'.txt')
        names = raw.decode('utf-8').splitlines()
        require(len(names) == OFFICIAL_COUNTS[split] and all(name and name == name.strip() for name in names),
                'Unexpected official split length/empty tile ID: ' + split)
        require(len(set(names)) == len(names) and not seen.intersection(names), 'Duplicate/overlapping official splits')
        seen.update(names)
        result[split] = names
        bindings[split] = dict(member=split+'.txt', sha256=hashlib.sha256(raw).hexdigest(), count=len(names))
    return result, bindings


def _raster_path(archive_path, member):
    return 'zip://'+str(archive_path)+'!'+member


def prepare_archive(archive_path, output_dir):
    """Verify fixed v1 identity, then prepare all 7470 train and 1602 val tiles.

    An existing nonempty output is refused, including a failed partial attempt.
    Source files are byte-hashed; pixel decoding requests only train/val windows.
    """
    archive_path, output_dir = Path(archive_path).resolve(), Path(output_dir).resolve()
    require(not archive_path.is_relative_to(output_dir), 'Output aliases the source ZIP')
    require(not output_dir.exists() or (output_dir.is_dir() and not any(output_dir.iterdir())),
            'Use a new or empty native dataset directory; partial attempts are preserved')
    identity = hashes(archive_path)
    require(identity['md5'] == OFFICIAL_MD5, 'Official LandCover.ai v1 MD5 checksum mismatch')
    import numpy as np
    from PIL import Image
    import rasterio
    from rasterio.windows import Window

    with zipfile.ZipFile(archive_path) as archive, rasterio.Env(GDAL_CACHEMAX=64*1024*1024):
        sources = _members(archive)
        script = archive.read('split.py')
        script_sha = hashlib.sha256(script).hexdigest()
        require(script_sha == OFFICIAL_SPLIT_SHA256, 'Unrecognized official split.py coordinate semantics')
        splits, split_bindings = _splits(archive)
        dimensions, source_bindings = {}, {}
        for source in sources:
            image_member, mask_member = 'images/'+source+'.tif', 'masks/'+source+'.tif'
            with rasterio.open(_raster_path(archive_path, image_member)) as image, \
                    rasterio.open(_raster_path(archive_path, mask_member)) as mask:
                require(image.count == 3 and image.dtypes == ('uint8',)*3, 'Expected native RGB uint8 TIFF')
                require(mask.count == 1 and all(np.issubdtype(np.dtype(dtype), np.integer) for dtype in mask.dtypes),
                        'Expected single-band integer class-mask TIFF')
                require((image.width, image.height) == (mask.width, mask.height), 'Source RGB/mask shapes differ')
                dimensions[source] = dict(width=image.width, height=image.height)
            for member in (image_member, mask_member):
                info = archive.getinfo(member)
                with archive.open(member) as stream:
                    sha = stream_sha256(stream)
                source_bindings[member] = dict(sha256=sha, uncompressed_bytes=info.file_size,
                                              zip_crc32=f'{info.CRC:08x}')
        origins, expected = {}, set()
        for source, shape in dimensions.items():
            columns = (shape['width']+511)//512
            expected.update(f'{source}_{row*columns+column}' for row in range(shape['height']//512)
                            for column in range(shape['width']//512))
        for split, names in splits.items():
            for name in names:
                match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9-]*)_(0|[1-9]\d*)', name)
                require(match is not None and match[1] in dimensions, 'Unknown tile source: ' + name)
                origins[name] = dict(split=split, **tile_origin(name, **dimensions[match[1]]))
        require(set(origins) == expected, 'Official splits do not cover exactly the complete 512 source windows')
        output_dir.mkdir(parents=True, exist_ok=True)
        rows, totals = {}, {split: Counter() for split in ('train', 'val')}
        for source in sources:
            selected = sorted((name for split in ('train', 'val') for name in splits[split]
                               if origins[name]['source'] == source), key=lambda name: origins[name]['index'])
            if not selected:
                continue
            image_member, mask_member = 'images/'+source+'.tif', 'masks/'+source+'.tif'
            with rasterio.open(_raster_path(archive_path, image_member)) as image, \
                    rasterio.open(_raster_path(archive_path, mask_member)) as mask:
                for name in selected:
                    origin = origins[name]
                    window = Window(origin['x'], origin['y'], 512, 512)
                    rgb = image.read((1, 2, 3), window=window).transpose(1, 2, 0)
                    labels = mask.read(1, window=window)
                    require(rgb.shape == (512, 512, 3) and rgb.dtype == np.uint8, 'Native RGB window differs')
                    require(labels.shape == (512, 512) and np.issubdtype(labels.dtype, np.integer)
                            and int(labels.min()) >= 0 and int(labels.max()) <= 4,
                            'Invalid native class label; expected integer IDs 0..4: ' + name)
                    counts = np.bincount(labels.astype(np.int64).reshape(-1), minlength=5).tolist()
                    split = origin['split']
                    relative_image, relative_mask = f'{split}/images/{name}.png', f'{split}/masks/{name}.png'
                    for relative, array in ((relative_image, rgb), (relative_mask, labels.astype(np.uint8))):
                        target = output_dir/relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(array).save(target, format='PNG', compress_level=6, optimize=False)
                    rows[name] = dict(split=split, shape=[512, 512], image=relative_image, mask=relative_mask,
                        image_sha256=hashes(output_dir/relative_image)['sha256'],
                        mask_sha256=hashes(output_dir/relative_mask)['sha256'], class_pixel_counts=counts,
                        source_image=image_member, source_mask=mask_member,
                        source_window={key: origin[key] for key in ('x', 'y', 'width', 'height', 'index')})
                    totals[split].update({str(i): count for i, count in enumerate(counts)})
            print('Prepared '+source+': '+str(len(selected))+' train/val tiles', flush=True)
        require(hashes(archive_path) == identity, 'Source ZIP changed during preparation')
        manifest = dict(format=FORMAT, archive=dict(filename=archive_path.name, bytes=archive_path.stat().st_size, **identity),
            official_split_files=split_bindings, splits=splits,
            split_script=dict(member='split.py', sha256=script_sha,
                coordinate_rule='row-major k with ceil(source_width/512) columns; increment k even at partial edges'),
            sources=source_bindings, source_dimensions=dimensions, tile_origins=origins, tiles=rows,
            class_pixel_counts={split: [totals[split][str(i)] for i in range(5)] for split in ('train', 'val')},
            class_ids=[0, 1, 2, 3, 4], image_encoding='lossless RGB uint8 PNG; original TIFF sample values',
            mask_encoding='single-band integer PNG preserving class IDs 0..4; no resizing or remapping',
            test_pixel_windows_read=0, test_tiles_written=0,
            split_independence='Official random tile splits can share parent rasters; geographic/event independence is not established.',
            pixel_access_scope='Only train/val raster windows were requested. Test IDs and parent headers establish structural coverage; encoded source-file byte hashing is not test-window decoding.',
            preparation_source_sha256=hashes(Path(__file__))['sha256'])
        (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_archive(args.archive, args.output_dir)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print('Prepared '+str(len(manifest['splits']['train']))+' train and '+str(len(manifest['splits']['val']))+
          ' validation tiles; no test pixel windows requested.')


if __name__ == '__main__':
    main()
