# Local DFC24 data and exposure audit

No verifiably untouched scene/event partition was identified in the inspected retained local archive. This conclusion is bounded to `/home/bernalde/repos/QUAFFLE/results/local_quantum_followup_20261027/dataset` and the recorded HQ campaign, not all possible DFC24 sources.

The original combined directory contains 1,937 tile pairs: 1,702 training and 235 testing. The cleaned manifest retains 1,533 training and 205 testing pairs, binding 3,476 image/mask files. Sensor-specific directories contain the same image corpus; they add no new pixel identities. The archive has 3,874 TIFF paths but 1,937 distinct device/inode pairs.

For each of seeds42,77,123, all 1,533 cleaned training tiles already belong either to the 1,303-image normalization-fit pool or the 230-image validation pool. Optimization uses128 tiles per seed (359 unique across these three seeds). Tiles omitted from optimization are therefore not an untouched reserve. The current15 primary replays evaluated all205 cleaned test tiles (71 Optical/134 SAR); the protocol already labels these exposed outcomes descriptive.

The raw exclusions comprise 169 training and 30 testing tile names. There are 18 excluded training image identities absent from the clean corpus; these arise in the manifest's conflicting-mask audit, not a verified clean independent-event reserve. Do not restore known duplicate/leakage/conflict exclusions as new test data.

Manifest hash records contain only `name`, image/mask filenames, file MD5s and decoded-pixel hashes. The only metadata sidecars are three split manifests and `norm_stats.json`. A stdlib header-only scan covered all1,937 original combined TIFFs and 1,937 IFDs: 0 files had any checked geolocation/GPS tags; 0 ASCII/text metadata entries were found. No raster pixels were decoded. Numeric filenames and `copy` collision suffixes are not verified scene/event identifiers.

`geo_cluster` is a seven-feature grouping of three channel means, three channel standard deviations and mask flood fraction. It uses seeded k-means and can move individual tiles to satisfy minimum client sizes. It does not use coordinates or event IDs and cannot certify event/geographic independence. Current held-out validation is stratified on a flood label, rather than grouped by event.

A defensible group holdout first requires an authoritative tile-to-scene/event/acquisition map verified against the manifest. Then reserve entire groups and recompute normalization using training groups only. A recovered map cannot undo prior exposure: without documented untouched data/outcomes, describe a new split as a prospective stress test on previously exposed DFC24. If group metadata cannot be recovered, do not relabel feature clusters as geographic or event holdouts. Alan retains Anvil execution responsibility.

The JSON records exact paths, manifest/file digests, source-function locations, all TIFF header metadata, exposure counts and scope limits. Its reconstructed inventory matches the recorded3476-input identity, but this audit did not independently rehash complete raw image/mask bytes; that byte check remains the earlier producer/collector evidence.
