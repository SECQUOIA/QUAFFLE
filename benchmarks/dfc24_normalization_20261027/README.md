# Real DFC24 fit-only normalization

The raw source files were recovered from the organizer-linked public Drive
folders recorded in `provenance.json`. All 3,874 original file MD5s match the
committed Optical/SAR manifests. The 3,874 combined files are byte-identical
hardlinks constructed with the documented collision naming rule. Raw images
and masks are not redistributed here.

The existing generator was run against those files with the committed `drop`
manifests, default seed 42, validation fraction 0.15, image size 128, and every
campaign seed. It excludes each seed's validation tiles before sampling pixels.
The command, source hashes, manifest hashes, package versions and resulting
normalization hash are in `provenance.json`.

The original regression fails on base `2d9907e` because the old artifact lacks
fit-only metadata. With the regenerated artifact, 32 focused offline checks
pass, including every seed/dataset combination without raw-data fallback.
The full suite passes 257 tests.

`verify_raw.py` separately checks all 30 fit-name identities against the raw
training masks and independently recomputes default-seed GREEN/Optical and
VV/SAR percentiles, means and standard deviations from sampled raw pixels.
Run with the verified dataset root:

```bash
python benchmarks/dfc24_normalization_20261027/verify_raw.py \
  --data-root /path/to/data --out /tmp/raw-normalization-audit.json
```

The corresponding retained result is `raw_audit.json`. This validates the
normalization artifact; it is not a new trained-model accuracy result.
