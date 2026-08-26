# Wildfire spread-pipeline handoff

This document is the operational handoff for the compact first wildfire
spread-prediction pipeline. Read it before changing collection, labels,
features, or training behavior.

## Goal and current boundary

The product takes wildfire evidence—FIRMS locations/TI4 plus weather—and
returns 1 km cells where fire is likely to spread in the next 12 hours. Cell
centroids provide the output latitude/longitude. Historical training analysis
uses retrospective weather; a separate optional path captures issued forecasts
for future operational experiments.

The repository has a completed **no-weather weak-label candidate dataset**,
not a trained operational binary spread predictor. It contains FIRMS-supported
weak positives and explicit weak-negative proxies; do not describe it as a
weather-aware forecast or a verified clear/no-burn dataset.

## Settled decisions

| Topic | Agreed decision |
| --- | --- |
| Model geometry | Fixed 1 km North America Albers equal-area grid (`ESRI:102008`); a cell ID is the durable prediction key and its centroid is the returned lat/lon. |
| Horizon | 12 hours. Each row has an explicit UTC feature cutoff and `target_end_at = cutoff + 12 h`. |
| Initial geography | CONUS + Canada. Alaska is excluded until FEDS local-solar alignment is separately validated. |
| First target | FEDS perimeter at `t + 12 h` minus FEDS perimeter at `t`, rasterized to positive cells only. It is `weak_satellite`, not independent ground truth. |
| FEDS time | The authoritative snapshot timestamp is embedded in the provider primary key. `t` is not substituted as the snapshot identity. A source snapshot becomes a documented per-cell local-solar-to-UTC cutoff estimate. |
| Current fire features | Unfiltered FIRMS from SNPP, NOAA-20, and NOAA-21. Use a 24-hour lookback and conservative 3-hour availability lag. Preserve all source fields; the 305 K TI4 threshold is not a collection filter. |
| Static feature | Sample retained ETOPO elevation, slope, and aspect at the 1 km cell centre. NALCMS is source evidence only, not a feature yet. |
| Weather | The next historical rebuild backfills Open-Meteo Historical Weather API ECMWF IFS (`ecmwf_ifs`) values at each FIRMS-seeded candidate tile and UTC hourly prediction anchor. Mark them `historical_analysis`; they are retrospective conditions, not reconstructed issued forecasts. The separate Open-Meteo Single Runs collector may capture one manually selected model/run for an operational experiment. The completed release remains weather-free as a past artifact. Retain wind as U/V components (or cyclic derivatives), never a raw 0–360° scalar. |
| First model | A simple `HistGradientBoostingClassifier` tabular baseline after a valid binary candidate table exists. It is intentionally not a spatial deep-learning cube. |
| Evaluation split | Chronological and grouped by FEDS `source_snapshot_time`, so all cells from one source snapshot stay on one side of holdout. Never randomly split neighboring cells or source snapshots. |
| Storage | The complete `data/` tree is hard-capped at 20,000,000,000 bytes. Existing files are counted and never silently evicted. |

The binding rationale is in [ADR 0001](adr/0001-bounded-20gb-local-dataset.md)
and [ADR 0002](adr/0002-first-1km-12hour-weak-label-baseline.md).

## Non-negotiable pipeline rules

1. Preserve raw provider responses unchanged and append provenance/coverage
   records. Never edit raw data or coverage ledger entries by hand.
2. Do not use information that was unavailable at the row's cutoff for an
   operational feature. In particular, do not use retrospective ingestion
   time, final WFIGS geometry, or future FEDS state. Retrospective ECMWF IFS
   weather is allowed only in explicitly labelled `historical_analysis` rows;
   it must never be represented as an issued forecast.
3. FEDS absence is **not** a no-spread label. Missing, partial, cloudy,
   unprocessed, or otherwise unobserved evidence is unknown until a valid
   observability policy says otherwise.
4. The archive-backed training builder requires every FIRMS product/day needed
   through the usable `[cutoff - lookback, cutoff - availability lag]` interval
   to be `complete` or `empty-confirmed`. A missing date may never become a
   zero FIRMS count.
5. Read training rows only through a completed training-view manifest. An
   interrupted build may leave immutable row artifacts behind, but those files
   are deliberately invisible to `iter_training_examples` until a final
   manifest is published.
6. Keep label, target, source snapshot, geometry, raw artifact, identifier,
   and future-time metadata out of model feature columns. The baseline's
   leakage guard enforces this for known fields.
7. Every future candidate/negative row must retain the candidate-selection
   reason, label tier/quality, observation state, feature availability policy,
   and raw-artifact lineage.
8. Never store a full Level-2 swath, full forecast grid, unpaired VIIRS fire
   file, or extracted national land-cover TIFF under the 20 GB policy. Run
   archive-writing commands one at a time and keep background workers below
   10 on this WSL environment.
9. A historical weather value is usable only when an archived ECMWF IFS
   response, candidate-cell/tile mapping, and valid hour match the row's UTC
   hourly prediction anchor at or before the cutoff; retain its retrieval time
   and `historical_analysis` mode. An operational issued-forecast value instead
   requires an explicit model/run and captured availability timestamp at or
   before the cutoff, with a valid time after that availability time.

## What is complete and verified

### Source evidence and derived data

- FIRMS was retained for the target period **2026-05-31 through 2026-08-10**,
  plus the required leading context day **2026-05-30**. The final context-day
  collection added 3 responses and 2,043 source rows with no retry gaps.
- FEDS raw responses were replayed through the primary-key normalizer rather
  than recollected. Do not union two captures: select one coherent capture;
  semantically conflicting records across captures are real revisions, not
  duplicates.
- FEDS labels are built from primary-key snapshot partitions. A no-expansion
  interval is `partial`, never an `empty-confirmed` global no-spread state.
- The current completed v3 training view contains **31,376** positive rows in
  **143** immutable artifacts. Every row has the explicit weather status
  `unavailable-no-issued-forecast-features`.
- The completed manifest for the current full range is
  `data/manifests/training-dataset-builds/2026/08/21/034320260310_3eef915e5aaa483c94908d9204f9b459.json`.
  It records the source date range, FIRMS products/region, 24-hour lookback,
  180-minute lag, artifact list, and row count.
- The completed no-weather candidate view is
  `data/manifests/candidate-dataset-builds/2026/08/24/040059951477_76e075f405f24287ba9531bea436cbe5.json`.
  It has **305,528** candidate rows: **19,528** supported weak positives and
  **286,000** weak-negative proxies. **11,848** positives outside FIRMS
  candidate support are retained as unscored diagnostics.
- The uploadable, self-contained release is
  `releases/wildfire-spread-firms-feds-no-weather-2026-05-31_to_2026-08-10/`.
  Its streamed row counts and every SHA-256 checksum were verified.
- At the latest audit, `data/` used **5,356,093,539 / 20,000,000,000 bytes**.

### Implemented code

| Component | Responsibility |
| --- | --- |
| `training_grid.py` | Canonical 1 km grid, cell IDs, centroids, and example keys. |
| `feds_collection.py` / `rebuild_feds_normalization.py` | Raw FEDS capture/replay using provider primary-key snapshot time. |
| `feds_labels.py` / `build_feds_labels.py` | Positive-only 1 km FEDS perimeter-difference labels. |
| `fire_state_features.py` | Availability-gated FIRMS centre and 3×3 km fire-state features. |
| `terrain_features.py` | On-demand retained ETOPO cell-centre sampling. |
| `training_dataset.py` / `build_training_dataset.py` | Coverage-gated positive-only assembly and atomic completed-view publication. |
| `candidate_sampling.py` / `candidate_dataset.py` / `build_candidate_dataset.py` | Deterministic FIRMS-supported weak candidates, cutoff-safe features, atomic candidate manifest, and unscored-positive diagnostics. |
| `export_candidate_dataset.py` | Self-contained gzip JSONL upload release with schema, inventory, and SHA-256 checksums. |
| `tabular_baseline.py` | Leakage-gated chronological tabular trainer, metrics, calibration, and persisted feature contract. |

The test suite covers the candidate build, source-range refusal, manifest
selection, release checksums, and chunk merge behavior. Run it again after any
source or policy change.

## Remaining work before a credible operational predictor

### 1. Candidate source and weak-negative policy

The first candidate source is now **FIRMS-only**. It matches the intended
predictor input and avoids requiring a FEDS perimeter at inference.
`candidate_sampling.py` expands a deterministic square radius around each
FIRMS detection that was available at the candidate cell's local-solar aligned
cutoff. Exact FEDS-positive cells inside that support retain target=1; capped
non-positive cells retain target=0 only as `weak_negative_proxy`, with the
explicit `unobserved-no-clear-no-burn-mask` observability state. A positive
outside FIRMS support is retained as `unscored-positive-no-firms-candidate`,
not discarded or relabeled.

The existing positive table shows why this choice matters: only 15,302 of
31,376 FEDS-positive cells (48.8%) have an eligible FIRMS detection in the
immediate 3×3 km context at cutoff. A FIRMS-only design must seed/expand an
incident or fire cluster wider than that local window, or explicitly retain
the uncovered positives as unscored/excluded cases.

### 2. Define valid target=0 / observation handling

The default prototype radius is 2 km and proxy cap is 2,000 cells per FEDS
snapshot. Both are explicit sampler inputs recorded in the completed
candidate-view manifest. The policy does *not* claim a verified
clear/no-burn negative; paired VIIRS observation cutouts remain necessary for
that stronger label.

The completed view publishes candidate features and source lineage atomically.
It can now be passed to `train_tabular_baseline` with the manifest feature
allowlist and `split_group_column="source_snapshot_time"`. Any resulting
score remains a weak-label experiment until paired observation coverage and
independent validation are added.

### 3. Backfill historical weather; optionally capture issued forecasts

For the planned 2026-05-11 through 2026-08-22 source rebuild, turn the
FIRMS-seeded candidate cells into compact weather tiles, then retrieve hourly
Open-Meteo Historical Weather API ECMWF IFS (`ecmwf_ifs`) values for each tile
and each row's prediction cutoff floored to UTC hour. Preserve raw responses,
the tile/candidate mapping, model, valid hour, retrieval time, and
`historical_analysis` feature mode. Include target=0 proxy candidate cells in
the mapping, not only positive detections. This gives the training table
retrospective weather conditions; it does not claim the conditions were known
as a forecast at the cutoff. The backfill and join must receive the exact same
completed candidate manifest; its path, build ID, and content hash are retained.
The default compact weather cover is a 10 km spatial approximation before the
provider grid snap, so it is not literal 1 km meteorology.

The Open-Meteo Single Runs capture remains optional and only while the forecast
is operational. The operator explicitly supplies the model and
UTC model-run time; the collector archives the exact response plus
candidate-cell/tile mappings and preserves the
600-location-unit-per-minute pacing/retry/429-pause behavior. A captured
response establishes availability at its response time, not at model
initialization time. Retain wind as U/V components in either feature mode.

### 4. Improve observations and static features

- Collect paired VIIRS fire-mask/QA **and matching geolocation** cutouts before
  claiming observation-aware no-fire labels.
- Define target-grid categorical compaction for NALCMS (for example class
  fractions or mode); never average categorical IDs.
- Add fuel moisture, drought/soil moisture, water, roads, and suppression
  context only with versioned spatial/time semantics and a storage admission.

### 5. Strengthen validation

Use later-time, held-incident, and held-region evaluation. Treat FEDS/FIRMS
dependence as a weak-label limitation; do not present a FEDS-trained/
FIRMS-featured score as independent ground truth.

## Commands to resume safely

Run from the repository root:

```bash
# Verify code and contracts
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests

# Inspect remaining storage before any collection
PYTHONPATH=src .venv/bin/python -m wildfire_data.inspect_storage_budget --data-root data

# Rebuild the current full positive-only view after an intentional source update
PYTHONPATH=src .venv/bin/python -m wildfire_data.build_training_dataset \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --data-root data

# Build and export the coherent no-weather candidate release
PYTHONPATH=src .venv/bin/python -m wildfire_data.build_candidate_dataset \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --data-root data
PYTHONPATH=src .venv/bin/python -m wildfire_data.export_candidate_dataset \
  --data-root data \
  --candidate-manifest data/manifests/candidate-dataset-builds/2026/08/24/040059951477_76e075f405f24287ba9531bea436cbe5.json \
  --output releases/wildfire-spread-firms-feds-no-weather-2026-05-31_to_2026-08-10
```

The build commands are idempotent at the immutable-artifact level and publish
new completed manifests only if all source coverage checks pass. For a full
from-scratch collection, use the ordered commands in
[Collecting data](collecting-data.md); it includes the May 30 FIRMS context
day and August 11 FEDS boundary snapshot.

To create the requested weather-bearing 2026-05-11 through 2026-08-22 range,
first retain FIRMS for 2026-05-10 through 2026-08-22 and FEDS source snapshots
through 2026-08-23. Rebuild FEDS labels, terrain blocks, the positive view,
and a completed base candidate view. Then
`open_meteo_historical.backfill_open_meteo_historical_weather` backfills hourly
Open-Meteo Historical Weather API ECMWF IFS values at every candidate tile and
row anchor. If it pauses, pass its partial manifest as `resume_manifest` to
reuse complete dates. Pass only the resulting complete manifest, together with
the same base candidate manifest, to
`weather_candidate_dataset.build_weather_candidate_dataset`, then export the
separate weather-bearing view. The base candidate view is immutable; the
current builder deliberately refuses to extend beyond its completed
positive-view manifest.

## Useful references

- [Architecture](architecture.md)
- [Data collection runbook](collecting-data.md)
- [Training-pipeline contract](training-pipeline.md)
- [Uploadable dataset contract](uploadable-dataset.md)
- [Feature and label map](feature-map.md)
- [Change log](change-log.md)
