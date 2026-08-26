# Change log

## 2026-08-25

- Adopted the historical-weather training contract: backfill the Open-Meteo
  Historical Weather API ECMWF IFS (`ecmwf_ifs`) at FIRMS-seeded candidate
  tiles and each row's UTC hourly prediction anchor. These values are retained
  as `historical_analysis` with raw-response, retrieval, tile-mapping, model,
  and valid-hour provenance; they are not reconstructed issued forecasts.
- Retained the optional Open-Meteo Single Runs capture contract for future
  operational research. It requires an operator-selected model/run, preserves
  request pacing/retries and the deliberate 429 pause, archives immutable
  responses and candidate cell-to-tile mappings, and records successful-
  response availability time.
- Kept the completed 2026-05-31 through 2026-08-10 release explicitly
  weather-free as a past artifact. The planned 2026-05-11 through 2026-08-22
  rebuild may add retrospective historical-weather features, while only
  contemporaneous Single Runs captures can support an operational as-of claim.

## 2026-08-24

- Added the manifest-selected no-weather FIRMS/FEDS candidate-table pipeline,
  notebook runner, chunk-merge support, and self-contained gzip JSONL release
  exporter. It publishes an atomic completed candidate-view manifest and never
  globs historical immutable artifacts.
- Built and verified the retained 2026-05-31 through 2026-08-10 release:
  305,528 candidate rows (19,528 supported weak positives and 286,000
  FIRMS-seeded weak-negative proxies), plus 11,848 unscored FIRMS-uncovered
  positives. The release includes schema, inventory, and SHA-256 checksums.
- Explicitly kept weather out of model inputs because no forecast values with
  retained run and availability provenance were captured. The release remains
  a no-weather weak-label research baseline.

## 2026-08-20

- Added a hard FIRMS product/day coverage gate to archive-backed training-row
  assembly. The 24-hour lookback required one leading 2026-05-30 FIRMS
  collection day; the completed three-product collection added 2,043 source
  rows with no retry gaps. The rebuilt view contains 31,376 positive rows and
  is published through a completed-build manifest, so interrupted immutable
  artifacts cannot be read as a partial dataset.
- Hardened the tabular baseline's time split: FEDS rows can now group by
  `source_snapshot_time`, keeping all cell-local cutoffs from one source
  snapshot on the same side of the chronological holdout. The persisted
  baseline contract is versioned accordingly.
- Accepted [ADR 0002](adr/0002-first-1km-12hour-weak-label-baseline.md): the first experiment uses a canonical 1 km ESRI:102008 lattice, a 12-hour horizon, CONUS + Canada scope, FEDS satellite-weak positive labels, and a chronological tabular histogram-gradient-boosting baseline. Alaska is excluded pending its own FEDS time-alignment policy.
- Added a resumable, quota-admitted NASA FEDS perimeter collector and a FEDS perimeter-difference label builder. The first 2026-05-31 through 2026-08-10 FEDS capture recorded 12,148 provider features as immutable evidence. FEDS remains explicitly weak satellite evidence, not operational perimeter history or independent truth from FIRMS.
- Hardened FEDS snapshot provenance around the provider primary-key timestamp and retained the provider time field alongside it. Added a no-network raw-response replayer so the already archived range can be rebuilt into primary-key snapshot partitions without recollection. Label anchors use a documented per-cell local-solar-to-UTC estimate rather than falsely treating every FEDS snapshot as one global UTC observation time.
- Added the fixed grid contract, availability-gated FIRMS cell/3×3 feature builder, on-demand ETOPO cell-centre sampler, positive-only training-table assembler, and leakage-aware chronological tabular baseline component. Added the deterministic FIRMS-only candidate/weak-negative sampler: it retains proxy/unknown semantics and unscored positives rather than inventing clear/no-burn evidence. Candidate feature-view publication and an end-to-end binary-train command remain pending.
- Added the explicit [training pipeline](training-pipeline.md) and rewrote the collection guide as command → exact data produced. Both documents make the weather boundary clear: HRDPS is only a retrieval plan today, so the baseline has no weather or wind-direction feature yet.
- Amended [ADR 0001](adr/0001-bounded-20gb-local-dataset.md) to admit compact FEDS weak-label evidence while keeping the whole data tree under the 20 GB contract.

## 2026-08-18

- Amended [ADR 0001](adr/0001-bounded-20gb-local-dataset.md) after adversarial storage reviews: retained paired-L2 capacity is 3.0 GB, static capacity is 5.5 GB, and derived-view capacity is 1.5 GB. This preserves a compact L2 reserve while admitting the highest-value static evidence that can be collected without misleading historical reconstruction.
- Added quota-admitted NOAA NCEI ETOPO 2022 v1 15-arc-second terrain collection. The completed 2026-05-31 through 2026-08-10 run retains raw source subsets, normalized provenance, and compact elevation/slope/aspect NPZ blocks for 47 10° WGS84 source blocks covering 2,762 FIRMS 96 km context tiles and 271,166,584 source-grid cells.
- Added a streamed, resumable CEC NALCMS 2020 v2 country-source collector and archived the Canada and United States releases as immutable source evidence. The archive records Canada 2020, CONUS 2019, and Alaska 2021 component timing; it does not create or imply a model-ready categorical land-cover grid.
- Hardened the NALCMS collector so full ZIP staging must be outside `data/`, source responses cannot exceed their quota-admitted identity `Content-Length`, and a second admission occurs before raw archival. It records malformed or oversized responses as failed coverage rather than allowing them to bypass the cap.
- Added static-data runbook steps and corrected architecture/feature documentation to distinguish collected terrain, retained NALCMS source evidence, and still-reserved L2/weather inputs. The final audit uses 4,964,605,150 of 20,000,000,000 bytes; static evidence uses 4,540,056,932 of its 5,500,000,000-byte allocation.

## 2026-08-16

- Accepted [ADR 0001](adr/0001-bounded-20gb-local-dataset.md): the complete local `data/` tree, including pre-existing CSVs, is hard-capped at 20,000,000,000 bytes. The policy reserves bounded capacity for compact training evidence and never silently evicts existing files.
- Added machine-readable storage allocations, quota admission checks, and `inspect_storage_budget`, which writes a scored `data/retention/storage_budget.csv` inventory without altering provider artifacts.
- Added a resumable NIFC WFIGS reference-perimeter collector with immutable paginated GeoJSON evidence, coverage checkpoints, normalized source/timing provenance, and `final_reference`/`label_quality_score` fields. The 2026-05-31 through 2026-08-10 backfill recorded 2,019 reference perimeters.
- Added a quota-admitted CWFIS active-fire record-history collector. It retains the provider's `record_start`/`record_end` intervals as Canadian operational incident context and collected 10,136 record versions for 2026-05-31 through 2026-08-10; it never represents those points as spread labels.
- Added leakage-safe, 96 km pre-grid weather-tile planning with conservative FIRMS availability latency, explicit candidate/cap reasons, `fire_evidence_score`, `forecast_availability_score`, and `retention_priority_score`. The stored HRDPS plan covers the still-public 2026-07-17 through 2026-08-10 slice with 56,497 candidates and 3,200 selected tiles; it is not a forecast-value archive.
- Made repeat WFIGS range runs idempotent by default; `--refresh` explicitly captures a new current/reference view. Fixed coverage-ledger ordering when multiple immutable entries share a supplied timestamp.
- Added quota admission to the FIRMS response path, so a full or category-capped `data/` tree yields a retryable `partial` record before new FIRMS bytes are written.
- Corrected Level-2 provenance: `VNP14IMG`/`VJ114IMG`/`VJ214IMG` hold fire mask and algorithm QA, while `VNP03IMG`/`VJ103IMG`/`VJ203IMG` hold geolocation. A fire-file-only full download is now an explicit legacy override and is incompatible with the compact local policy.
- Added a CMR-first Collection 2 VIIRS active-fire inventory collector for SNPP `VNP14IMG`, NOAA-20 `VJ114IMG`, and NOAA-21 `VJ214IMG`.
- Added a discovery workflow that archives unmodified CMR responses with redacted Earthdata authentication, platform/product/version, observation interval, footprint, response, and inventory-artifact provenance. The old full-file downloader remains available only as an explicit legacy override.
- Added granule-level integrity checks and coverage semantics for the legacy downloader: invalid/login payloads are retained but failed, historical empty inventories can be confirmed, and date/product coverage completes only after every listed fire file is archived. Compact paired cutouts remain pending.
- Documented Earthdata setup and the requested 2026-05-31 through 2026-08-10 inventory/retry commands.
- Added [the planned-layout FIRMS re-collection guide](collecting-data.md), including configuration, PYTHONPATH, module invocation, and coverage retry checks.
- Added the U.S. and Canada wildfire spread-forecasting collection and provenance contract.
- Defined raw, normalized, and derived data layers; forecast-time semantics; source/coverage quality requirements; and label tiers.
- Added a feature and label map and linked both documents from the README.
- Added an immutable raw-artifact store with SHA-256 identities, credential-redacted provenance manifests, and an append-only coverage ledger with complete, empty-confirmed, partial, and failed states.
- Added lossless normalized JSON Lines storage, a FIRMS normalizer, and a FIRMS collection path that archives unfiltered responses before the notebook applies its visualization threshold.
- Added explicit request-failure coverage records, generic snapshot archiving for future source adapters, a U.S./Canada source catalog, and leakage-safe issued-at forecast-weather records.
- Updated the notebook to preserve raw FIRMS rows, retain all provider fields in weather/map exports, and write archive/coverage artifacts to `data/`.
- Added collection-window planning so scheduled source coverage can be enumerated and incomplete windows retried instead of silently leaving gaps.
- Redacted FIRMS API keys embedded in request paths before raw collection manifests are persisted.
- Added the `src/wildfire_data/collect_firms.py` command-line range collector, which reads the existing environment key, retries ordinary HTTP failures, and records daily failures without losing the rest of a collection run.
- Added a raw-plus-normalized forecast collection path so future HRRR/HRDPS adapters can retain issued-at model values without weather leakage.
