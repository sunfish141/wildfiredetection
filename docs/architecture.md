# Fire spread forecasting data architecture

## Scope and intent

This contract defines the data needed to forecast where an existing wildfire is likely to newly burn next in the United States and Canada.

The first prediction unit is a fixed 1 km North America Albers equal-area cell
(ESRI:102008) at a 12-hour horizon. The service returns calibrated
probabilities and the corresponding cell-centroid latitude/longitude
coordinates, rather than treating arbitrary point coordinates as direct
regression targets.

The notebook remains one presentation input stream: NASA FIRMS detections
enriched with selected hourly weather values. It is not the operational weather
source. The repository also has a separate FEDS weak-label path, a completed
FIRMS-only candidate-table builder, and a self-contained release exporter. A
future trained model remains a research baseline, not an operational forecast.

## Planned collection layout

The planned package layout places environment configuration in config/.env, pinned dependencies in config/requirements.txt, durable collection artifacts in data/, and collectors in src/wildfire_data. The step-by-step re-collection procedure, including coverage recovery, is in [Collecting data](collecting-data.md).

## Current repository implementation

The repository implements the collection foundation, bounded-storage policy,
FEDS satellite-weak labels, WFIGS reference-perimeter and CWFIS
incident-context collectors, a CMR-first VIIRS inventory collector, compact
ETOPO terrain collection, and bounded NALCMS source-archive collection.
Compact paired-L2, issued-forecast values, and categorical land-cover features
remain to be added:

- `src/wildfire_data/data_archive.py` stores immutable gzip-compressed raw bytes by SHA-256, writes secret-redacted manifests, and records append-only coverage outcomes.
- `src/wildfire_data/firms_collection.py` archives each successful or failed FIRMS daily request; successful CSV rows are normalized losslessly before the legacy brightness-filtered view is created.
- `src/wildfire_data/collect_firms.py` is the non-notebook entry point for durable, daily FIRMS range collection and continues across failed days while recording retryable coverage gaps.
- src/wildfire_data/feds_collection.py and src/wildfire_data/collect_feds.py archive public NASA FEDS perimeter response pages for each requested 12-hour source interval. src/wildfire_data/rebuild_feds_normalization.py can replay retained raw pages without an API call to normalize by the timestamp embedded in each source primary key. The evidence is weak_satellite, defaults to CONUS + Canada, and is not represented as an operational perimeter revision history.
- src/wildfire_data/feds_labels.py and src/wildfire_data/build_feds_labels.py derive positive-only 1 km cells from consecutive FEDS perimeter differences. The source time, cell-local-solar-to-UTC estimate, overlap fraction, raw IDs, and weak-label tier remain in the label record.
- src/wildfire_data/training_grid.py defines the canonical ESRI:102008 1 km lattice and cell-centroid conversion. src/wildfire_data/fire_state_features.py builds availability-gated FIRMS centre/3x3 state summaries, src/wildfire_data/terrain_features.py samples retained ETOPO blocks at a cell centre, and src/wildfire_data/tabular_baseline.py provides the chronological histogram-gradient-boosting baseline.
- src/wildfire_data/training_dataset.py and src/wildfire_data/build_training_dataset.py persist a bounded positive-only FEDS training view with cutoff-safe FIRMS/terrain features, raw-artifact lineage, and explicit weather missingness. They require terminal FIRMS product/day coverage through each usable feature interval and expose rows only through a completed-build manifest, so missing or interrupted partitions cannot read as zero evidence.
- src/wildfire_data/candidate_sampling.py creates deterministic FIRMS-only candidate cells from cutoff-eligible detections. It retains FEDS positives within candidate support, records non-positive candidates only as explicit weak-negative proxies, and separately reports positives with no FIRMS candidate support. It does not claim a clear/no-burn observation mask.
- `src/wildfire_data/candidate_dataset.py` and `build_candidate_dataset.py` turn exactly one completed positive-view manifest into cutoff-safe FIRMS/terrain candidate rows and atomically publish a completed candidate-view manifest. `export_candidate_dataset.py` materializes that one manifest as a checksum-protected upload directory; it never globs historical artifacts. The first completed view covers 2026-05-31 through 2026-08-10 and excludes retrospective Open-Meteo exports.
- `src/wildfire_data/storage_budget.py` and `config/storage_budget.json` account for every byte under `data/`, reject supported writes that would exceed the whole or category caps, and write a scored storage report without changing source records.
- `src/wildfire_data/wfigs_collection.py` and `src/wildfire_data/collect_wfigs.py` archive paginated WFIGS GeoJSON responses and normalize their reference perimeters with source/timing provenance. This backfill is a `final_reference` label tier; it is not a substitute for historical revision snapshots.
- `src/wildfire_data/cwfis_active_fires.py` and `src/wildfire_data/collect_cwfis_active_fires.py` archive CWFIS active-fire record versions in deterministic `record_start,id` order. `record_start`/`record_end` are retained as Canadian operational incident context, never converted into a perimeter/spread label.
- `src/wildfire_data/viirs_l2_observability.py` discovers Collection 2 SNPP (`VNP14IMG`), NOAA-20 (`VJ114IMG`), and NOAA-21 (`VJ214IMG`) active-fire files through NASA CMR and archives the unmodified inventory response. The active-fire products contain fire mask and algorithm QA; their required `VNP03IMG`/`VJ103IMG`/`VJ203IMG` companion products contain latitude/longitude.
- `src/wildfire_data/collect_viirs_l2.py` is the corresponding command-line inventory collector. A full fire-file download is explicitly marked legacy because it lacks the geolocation partner and is disallowed by the compact 20 GB policy.
- `src/wildfire_data/firms_normalization.py` preserves every provider field and records the TI4 threshold only as derived metadata.
- `src/wildfire_data/normalized_storage.py` writes content-addressed, lossless JSON Lines records under the normalized layer. Analytics-specific Parquet/GeoParquet views are derived outputs and can be regenerated.
- `src/wildfire_data/forecast_weather.py` defines issued-at forecast records and enforces forecast availability at an as-of time.
- `src/wildfire_data/forecast_tile_planning.py` and `src/wildfire_data/plan_forecast_tiles.py` create a bounded, scored HRDPS/HRRR pre-grid tile plan from FIRMS evidence available before each model run. The current 2026-07-17 through 2026-08-10 HRDPS plan has 56,497 candidates and 3,200 selected tiles; it does not claim that forecast values have been collected.
- `src/wildfire_data/forecast_collection.py` archives the provider response before normalizing long-form forecast measurements with model-run, publication, valid, and retrieval times.
- `src/wildfire_data/etopo_terrain.py` and `src/wildfire_data/collect_static_terrain.py` select all FIRMS-context tiles for a requested range, group them into quota-admitted ETOPO 2022 v1 15-arc-second source subsets, and store immutable raw subsets plus compact elevation/slope/aspect NPZ blocks.
- `src/wildfire_data/nalcms_collection.py` and `src/wildfire_data/collect_nalcms_land_cover.py` stream the fixed public CEC NALCMS country ZIPs through an outside-`data/` staging area after Content-Length quota admission. They retain raw source evidence and provenance only; a categorical model-feature derivation is intentionally pending.
- `src/wildfire_data/collection_catalog.py` and `src/wildfire_data/source_snapshots.py` define the next source adapters and give them the same immutable evidence and coverage-ledger path.
- `src/wildfire_data/collection_planning.py` expands a source cadence into explicit expected windows and reports missing, failed, or partial windows for retry.

The `open_meteo_weather_*.csv` workflow remains a presentation cache; it does not identify forecast issue/run time and must not be used as the operational forecast feature source.

The binding storage contract is [ADR 0001](adr/0001-bounded-20gb-local-dataset.md). It reserves space for compact source-paired L2 cutouts and forecast tiles, rather than full native swaths and full national forecast grids. It counts legacy CSV exports too and never silently removes existing evidence to create capacity.

## Logical layers

### Raw layer: immutable source evidence

Store each admitted source response unchanged and append-only. Retain all available FIRMS records before any TI4 threshold, together with the request range and product/version. Under the local cap, retain the unmodified CMR inventory response; the policy permits later selected, paired active-fire/geolocation cutouts with source pair IDs, checksums, crop geometry, pixel row/column, fire-mask value, and QA. The implemented archive uses `data/raw/<source>/<sha256>.gz` plus a new capture manifest for each attempted collection. Full native L2 swaths are explicitly outside the local policy.

Source categories are:

- Satellite active-fire detections and their coverage or quality metadata.
- Operational incident locations and perimeter revisions.
- Satellite-derived modeled progression products.
- Final agency and post-fire perimeter or burned-area products.
- Weather observations, forecast runs, and static terrain/fuel layers.

Static evidence currently has two deliberately different forms: ETOPO source subsets also have compact terrain features, while CEC NALCMS country ZIPs are retained only as versioned source evidence. The latter must not be joined to training rows until a target grid and categorical mode/fraction aggregation contract are versioned. Its component years are Canada 2020, CONUS 2019, and Alaska 2021.

For every raw snapshot, retain source name, endpoint and request parameters, response headers, source license/terms, product/schema version, raw-byte hash, collection time, and whether the fetch was empty or failed. Redact credentials before manifest persistence. A source response must never be overwritten by a corrected or later response.

### Normalized layer: source-faithful records

Normalize records without losing native identifiers or source timing:

- Detection: satellite/platform, acquisition time, coordinates, brightness, FRP, confidence, scan/track footprint, day/night, and quality flags.
- Incident: native incident ID, cross-source IDs, agency, name, cause, discovery/out/containment/control times, size, and status.
- Perimeter version: native geometry, source record ID, incident IDs, map method, geometry capture time, source create/edit time, source, acres, and snapshot ID.
- Weather: location/grid identifier, variable, valid time, issue/run time, model/version, fetch time, observation/forecast flag, and quality metadata.
- Coverage: satellite swath, cloud/smoke or missing-data indicators, valid-observation window, and source quality flags.

The incident crosswalk is versioned. It may link identifiers, but must not silently merge or replace source records.

### Derived layer: reproducible training examples

Derived tables are reproducible outputs from versioned raw and normalized inputs:

- Incident-time snapshots: the state of one incident and local context as known at an as-of time.
- Spatial feature cubes: dynamic inputs aligned to a fixed cell system and static terrain/fuel inputs.
- Labels: cells newly burned in a declared future interval, with label tier and observation coverage.
- Model-ready splits, predictions, calibration outputs, and evaluation records.

Each derived row records upstream snapshot IDs, transformation version, spatial resolution, horizon, and the exact feature cutoff.

## Forecast-time semantics

Every record must distinguish these UTC timestamps when applicable:

| Timestamp | Meaning |
| --- | --- |
| event or observation time | When the fire, satellite, or weather condition occurred. |
| geometry capture time | When a perimeter represents conditions on the ground. |
| source create/edit time | When the provider created or revised the record. |
| publication time | When the record became available to a user. |
| ingest time | When this project retrieved it. |
| forecast issue/run and valid time | When a weather prediction was issued and the time it predicts. |

For a sample at as-of time t, every feature must have been available no later than t. Weather features for a future horizon must come from a forecast run issued no later than t, never from a later reanalysis or a live endpoint queried after the event.

The primary spread label is the observed difference between two time-stamped perimeter states:

newly burned from t to t+h = perimeter at t+h minus perimeter at t

Only make that label when both geometry times and observation coverage are credible. Keep final perimeters as validation or end-state references, rather than assigning their full area to an earlier time.

## Collection priorities

Capture changing operational sources now because their revision histories are often not retained by final archives:

1. U.S. NIFC WFIGS current perimeters and IRWIN incident locations, including every observed geometry revision.
2. NASA FEDS perimeter, active-front, and new-fire-pixel products at their 12-hour cadence.
3. Canadian CWFIS active-fire and Fire M3 perimeter-estimate layers, plus direct provincial operational layers where available.
4. Raw multi-satellite FIRMS detections and Collection 2 VIIRS Level-2 coverage/quality cutouts. The compact implementation must pair SNPP `VNP14IMG` with `VNP03IMG`, NOAA-20 `VJ114IMG` with `VJ103IMG`, and NOAA-21 `VJ214IMG` with `VJ203IMG`; a legacy fire-file-only command does not meet this requirement.

Backfill stable reference products separately: U.S. WFIGS/IFPH and FODR, MTBS, Canadian NFDB and NBAC, and MODIS/VIIRS burned-area products. These products improve historical coverage and validation, but usually do not provide operational progression timestamps.

The collection catalog deliberately lists source *categories* and expected cadence without embedding external credentials or silently starting downloads. FIRMS, FEDS perimeters, WFIGS reference perimeters, CMR L2 inventory, ETOPO terrain, and NALCMS source archives are source-specific implementations; paired-cutout, issued-forecast, categorical static-feature, and training-table adapters will use the same immutable evidence, quota-admission, and coverage-ledger pattern.

## Quality, coverage, and label tiers

Treat absence of a detection as unknown unless valid observation coverage supports a negative label. Cloud, smoke, canopy, satellite cadence, sensor geometry, missing data, and quality flags must be explicit features or censoring masks.

Assign every perimeter or label one of these tiers:

- Operational: agency or incident mapping with a captured geometry time.
- Final reference: post-season or final mapped extent.
- Satellite-derived: active-fire or burned-area algorithm output.

Satellite-derived products such as FEDS and Fire M3 are valuable weak labels,
but can share input observations with FIRMS. They are not independent ground
truth and must not use information after the as-of time in the feature set.
The first FEDS builder emits positives only; missing or unchanged FEDS cells
cannot be treated as clear/no-burn negatives without a later candidate and
observation-coverage policy.

## Evaluation boundary

Split data by incident and use rolling time and held-out regions. Do not randomly split neighboring cells or consecutive updates from the same fire across training and evaluation. Preserve source availability and forecast-issue times in evaluation so offline scores represent operational use.
