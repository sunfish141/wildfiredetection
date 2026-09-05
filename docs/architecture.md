# Fire spread forecasting data architecture

## Scope and intent

This contract defines the data needed to forecast where an existing wildfire is likely to newly burn next in the United States and Canada.

The first prediction unit is a fixed 1 km North America Albers equal-area cell
(ESRI:102008) at a 12-hour horizon. The service returns calibrated
probabilities and the corresponding cell-centroid latitude/longitude
coordinates, rather than treating arbitrary point coordinates as direct
regression targets.

The notebook is the operator-controlled entry point for NASA FIRMS collection,
historical-weather backfill, and optional forward forecast capture. The
historical path intentionally creates retrospective weather-analysis evidence;
it does not claim to reconstruct a forecast that was available at a past
cutoff. The repository also has a separate FEDS weak-label path, a completed
FIRMS-only candidate-table builder, and a self-contained release exporter. A
future trained model remains a research baseline, not an operational forecast.

## Planned collection layout

The planned package layout places environment configuration in config/.env, pinned dependencies in config/requirements.txt, durable collection artifacts in data/, and collectors in src/wildfire_data. The step-by-step re-collection procedure, including coverage recovery, is in [Collecting data](collecting-data.md).

## Current repository implementation

The local [Wildfire Atlas web app](web-app.md) adds `web_app.py` as a FastAPI
entry point and `web/` as a static Leaflet frontend. Browser history owns each
scenario; stateless seed/step endpoints wrap the retained incident model.
Model inference and cached ETOPO sampling are serialized in worker threads.
`live_firms.py` provides bounded, transient observations from three VIIRS feeds
using server-held credentials. Full-region CSV streams aggregate directly to
1 km cells; display-only grouping keeps dense maps manageable. Playback has
no fixed horizon, and the browser retains a rolling 128-frame history while
preserving the full current burned-cell mask. Preview data and scenarios do not mutate the
collection archive, release, or fitted models.

The repository implements the collection foundation, bounded-storage policy,
FEDS satellite-weak labels, WFIGS reference-perimeter and CWFIS
incident-context collectors, a CMR-first VIIRS inventory collector, compact
ETOPO terrain collection, and bounded NALCMS source-archive collection.
Compact paired-L2 and categorical land-cover features remain to be added. The
forward forecast-capture path is implemented; the historical-weather contract
uses Open-Meteo Historical Weather API ECMWF IFS values at candidate tiles and
hourly anchors. Neither path changes the immutable current no-weather release:

- `src/wildfire_data/data_archive.py` stores immutable gzip-compressed raw bytes by SHA-256, writes secret-redacted manifests, and records append-only coverage outcomes.
- `src/wildfire_data/firms_collection.py` archives each successful or failed FIRMS daily request; successful CSV rows are normalized losslessly without applying a presentation-only brightness filter.
- `src/wildfire_data/collect_firms.py` is the non-notebook entry point for durable, daily FIRMS range collection and continues across failed days while recording retryable coverage gaps.
- src/wildfire_data/feds_collection.py and src/wildfire_data/collect_feds.py archive public NASA FEDS perimeter response pages for each requested 12-hour source interval. src/wildfire_data/rebuild_feds_normalization.py can replay retained raw pages without an API call to normalize by the timestamp embedded in each source primary key. The evidence is weak_satellite, defaults to CONUS + Canada, and is not represented as an operational perimeter revision history.
- src/wildfire_data/feds_labels.py and src/wildfire_data/build_feds_labels.py derive positive-only 1 km cells from consecutive FEDS perimeter differences. The source time, cell-local-solar-to-UTC estimate, overlap fraction, raw IDs, and weak-label tier remain in the label record.
- src/wildfire_data/training_grid.py defines the canonical ESRI:102008 1 km lattice and cell-centroid conversion. src/wildfire_data/fire_state_features.py builds availability-gated FIRMS centre/3x3 state summaries, src/wildfire_data/terrain_features.py samples retained ETOPO blocks at a cell centre, and src/wildfire_data/tabular_baseline.py provides the chronological histogram-gradient-boosting baseline.
- src/wildfire_data/training_dataset.py and src/wildfire_data/build_training_dataset.py persist a bounded positive-only FEDS training view with cutoff-safe FIRMS/terrain features, raw-artifact lineage, and explicit weather missingness. They require terminal FIRMS product/day coverage through each usable feature interval and expose rows only through a completed-build manifest, so missing or interrupted partitions cannot read as zero evidence.
- src/wildfire_data/candidate_sampling.py creates deterministic FIRMS-only candidate cells from cutoff-eligible detections. It retains FEDS positives within candidate support, records non-positive candidates only as explicit weak-negative proxies, and separately reports positives with no FIRMS candidate support. It does not claim a clear/no-burn observation mask.
- `src/wildfire_data/candidate_dataset.py` and `build_candidate_dataset.py` turn exactly one completed positive-view manifest into cutoff-safe FIRMS/terrain candidate rows and atomically publish a completed candidate-view manifest. `merge_candidate_dataset.py` combines contiguous completed chunks without globbing. `export_candidate_dataset.py` materializes that one manifest as a checksum-protected CSV/JSONL upload directory. The completed active view covers 2026-05-11 through 2026-08-22 and has no weather features.
- `src/wildfire_data/storage_budget.py` and `config/storage_budget.json` account for every byte under `data/`, reject supported writes that would exceed the whole or category caps, and write a scored storage report without changing source records.
- `src/wildfire_data/wfigs_collection.py` and `src/wildfire_data/collect_wfigs.py` archive paginated WFIGS GeoJSON responses and normalize their reference perimeters with source/timing provenance. This backfill is a `final_reference` label tier; it is not a substitute for historical revision snapshots.
- `src/wildfire_data/cwfis_active_fires.py` and `src/wildfire_data/collect_cwfis_active_fires.py` archive CWFIS active-fire record versions in deterministic `record_start,id` order. `record_start`/`record_end` are retained as Canadian operational incident context, never converted into a perimeter/spread label.
- `src/wildfire_data/viirs_l2_observability.py` discovers Collection 2 SNPP (`VNP14IMG`), NOAA-20 (`VJ114IMG`), and NOAA-21 (`VJ214IMG`) active-fire files through NASA CMR and archives the unmodified inventory response. The active-fire products contain fire mask and algorithm QA; their required `VNP03IMG`/`VJ103IMG`/`VJ203IMG` companion products contain latitude/longitude.
- `src/wildfire_data/collect_viirs_l2.py` is the corresponding command-line inventory collector. A full fire-file download is explicitly marked legacy because it lacks the geolocation partner and is disallowed by the compact 20 GB policy.
- `src/wildfire_data/firms_normalization.py` preserves every provider field and records the TI4 threshold only as derived metadata.
- `src/wildfire_data/normalized_storage.py` writes content-addressed, lossless JSON Lines records under the normalized layer. Analytics-specific Parquet/GeoParquet views are derived outputs and can be regenerated.
- `src/wildfire_data/forecast_weather.py` defines issued-at forecast records and enforces forecast availability at an as-of time.
- `src/wildfire_data/forecast_tile_planning.py` and `src/wildfire_data/plan_forecast_tiles.py` create a bounded, scored HRDPS/HRRR pre-grid tile plan from FIRMS evidence available before each model run. The current 2026-07-17 through 2026-08-10 HRDPS plan has 56,497 candidates and 3,200 selected tiles; it does not claim that forecast values have been collected.
- `src/wildfire_data/weather_rate_limit.py` preserves bounded request pacing, retry handling, a `Retry-After`-aware 429 cooldown, and the deliberate pause after two consecutive 429 responses.
- `src/wildfire_data/open_meteo_single_run.py` plans candidate-cell tiles from newly archived FIRMS evidence and captures one explicitly chosen Open-Meteo Single Runs model/run. It retains immutable provider bytes, a mapping that records the FIRMS seed/raw-artifact IDs, acquisition time, and all candidate cells (including weak-negative proxies), and only values valid after the successful response was captured.
- `src/wildfire_data/open_meteo_historical.py` rate-limits historical ECMWF IFS tile/date capture, archives raw API responses and candidate-cell mappings, floors each candidate anchor to its UTC hour, binds the backfill to an immutable base-manifest identity, and publishes a complete-or-partial manifest that can reuse completed dates on resume.
- `src/wildfire_data/weather_candidate_dataset.py` requires one complete historical backfill and the exact matching completed base candidate view, joins only the matching tile/hour and example lineage, and publishes or exports a separate weather-bearing view without mutating its no-weather spine. The compact tile cover defaults to a 10 km maximum candidate-to-requested-tile distance before provider grid snapping.
- `src/wildfire_data/forecast_collection.py` archives the provider response before normalizing long-form forecast measurements with model-run, valid, retrieval, and captured-availability times.
- `src/wildfire_data/etopo_terrain.py` and `src/wildfire_data/collect_static_terrain.py` select all FIRMS-context tiles for a requested range, group them into quota-admitted ETOPO 2022 v1 15-arc-second source subsets, and store immutable raw subsets plus compact elevation/slope/aspect NPZ blocks.
- `src/wildfire_data/nalcms_collection.py` and `src/wildfire_data/collect_nalcms_land_cover.py` stream the fixed public CEC NALCMS country ZIPs through an outside-`data/` staging area after Content-Length quota admission. They retain raw source evidence and provenance only; a categorical model-feature derivation is intentionally pending.
- `src/wildfire_data/collection_catalog.py` and `src/wildfire_data/source_snapshots.py` define the next source adapters and give them the same immutable evidence and coverage-ledger path.
- `src/wildfire_data/collection_planning.py` expands a source cadence into explicit expected windows and reports missing, failed, or partial windows for retry.

The completed 2026-05-11 through 2026-08-22 release is explicitly no-weather;
only after it is verified may a separate historical rebuild use
archived Open-Meteo Historical Weather API ECMWF IFS (`ecmwf_ifs`) values at
candidate tiles and hourly prediction anchors, explicitly labelled
`historical_analysis`. See [the no-weather POC guide](no-weather-poc.md).
Separately, only forward Single Runs capture artifacts can support an
operational issued-forecast as-of feature join.

The binding storage contract is [ADR 0001](adr/0001-bounded-20gb-local-dataset.md). It reserves space for compact source-paired L2 cutouts and forecast tiles, rather than full native swaths and full national forecast grids. It counts every retained file and never silently removes existing evidence to create capacity.

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
- Weather: location/grid identifier, variable, valid hour, explicit model,
  raw-response ID, candidate-cell/tile mapping, feature mode, and quality
  metadata. Historical-analysis records also retain request range, retrieval
  time, and hourly-anchor rule. Issued-forecast records additionally retain
  the model run and captured availability time.
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
| captured availability time | When this collector successfully received the exact requested forecast run; for Open-Meteo Single Runs this is the evidence that the selected run was usable. |

For an operational sample at as-of time t, every feature must have been
available no later than t. An issued Open-Meteo forecast feature requires an
explicit model/run, immutable raw response, and captured availability time no
later than t; its valid time must be later than that captured availability
time. A model-run initialization time alone does not prove publication or
availability.

Historical Open-Meteo ECMWF IFS values are a distinct `historical_analysis`
mode. Backfill them at the retained candidate tile and the prediction cutoff
floored to its UTC hour, retaining raw response and retrieval provenance. The
value must not be from an hour after the cutoff, but its later retrieval is
expected. It describes retrospective conditions and cannot be used to assert
operational forecast availability or to compare directly with an issued-
forecast model.

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

Split data by incident and use rolling time and held-out regions. Do not randomly split neighboring cells or consecutive updates from the same fire across training and evaluation. Preserve feature mode throughout evaluation: historical-analysis scores are retrospective research results, while operational scores require source availability and forecast-issue times.
