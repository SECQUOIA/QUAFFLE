# Retained TIFF metadata scan: provenance and scope

The `tiff_metadata` parser function and its `sizes`, `geo_tags`, and `text_names` definitions are preserved exactly as executed for the original exposure report. The small CLI wrapper was added afterward to retain and replay that same parser.

- Parser/CLI source: `retained_tiff_header_scanner.py`; SHA256 `8c13dbb45739a3ea86097f8dca542d11e33f19884804abf7db3baecf0de18bbd`.
- Replay output: `retained_tiff_header_replay.json`; SHA256 `063b3b9c05d2dcb07175a0ccb0dd5a6dc412e96976e93df4e1a2128617cc0484`.
- Original report JSON: SHA256 `f8ce76e97b063f5de83404a6dc0298046626048a9884e8dbbdb4f128c6e91d8d`.
- Original report Markdown: SHA256 `aef8c9a5e2e916f7edb683b33f5b767bda1acce004be44a912e6c984acef6a8f`.
- Canonical per-file header record SHA256: `cf63716966e0df043e3259ee41787d7969550035ab3ffa5a183ac48b4a9feb8f`.

The replay's complete per-file records are exactly equal to those in the original report. It covers 1,937 retained local combined TIFF files and 1,937 IFDs, with zero checked geolocation/GPS tags and zero ASCII/text metadata entries. No raster pixels were decoded or full image bytes rehashed. The 18 excluded training image identities absent from the clean corpus match the manifest's 18 conflicting-mask image groups exactly.

**Scope:** these are the TIFF copies retained in our local archive. Their lack of embedded geolocation/event metadata says nothing about whether the upstream DFC24 distribution, original scene products, or organizer-maintained metadata contain georeferencing or event identifiers. No upstream source was inspected. No event groups were inferred from numeric filenames, image appearance, sensor, elevation, or feature clusters.

The original exposure report remains unchanged. Its 3,476-file byte-verification statement continues to refer to earlier producer/collector evidence; current metadata/path checks are not mislabeled as a fresh full-byte rehash.
