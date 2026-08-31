# Wildfire spread-pipeline handoff

This document is the operational handoff for the compact first wildfire
spread-prediction pipeline. Read it before changing collection, labels,
features, or training behavior.

## Goal and current boundary

The product takes wildfire evidence and returns 1 km cells where fire is likely
to spread in the next 12 hours. Cell centroids provide the output
latitude/longitude. The active first proof of concept uses FIRMS and terrain
without weather; retrospective weather and issued forecasts remain separate
future experiments.

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
| Weather | The active 2026-05-11 through 2026-08-22 POC deliberately does not collect, join, or train on weather. Its rows retain explicit unavailable/missing declarations. After the POC is verified, a separate candidate view may backfill Open-Meteo Historical Weather API ECMWF IFS (`ecmwf_ifs`) values at FIRMS-seeded tiles and UTC hourly anchors as `historical_analysis`; it is not a reconstructed issued forecast. The optional Single Runs collector remains a separate operational experiment. |
| First model | A simple `HistGradientBoostingClassifier` tabular baseline after a valid binary candidate table exists. It is intentionally not a spatial deep-learning cube. |
| Evaluation split | Chronological and grouped by FEDS `source_snapshot_time`, so all cells from one source snapshot stay on one side of holdout. Never randomly split neighboring cells or source snapshots. |
| Storage | The complete `data/` tree is hard-capped at 20,000,000,000 bytes. Existing files are counted and never silently evicted. |

The binding rationale is in [ADR 0001](adr/0001-bounded-20gb-local-dataset.md)
and [ADR 0002](adr/0002-first-1km-12hour-weak-label-baseline.md).

The active POC contract, including its May 10 FIRMS context day, August 23
FEDS boundary evidence, schema-v2 CSV release, provenance checks, and
chronological notebook baseline, is [documented separately](no-weather-poc.md).

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

- FIRMS is complete for the POC context/target period **2026-05-10 through
  2026-08-22**: 98 SNPP, 83 NOAA-20, and 105 NOAA-21 complete daily
  artifacts, with the remaining SNPP/NOAA-20 days explicitly
  `empty-confirmed`.
- FEDS observed/raw source snapshots are complete through **2026-08-23**.
  The label rebuild selects one latest normalized source artifact per snapshot
  (it never unions historical revisions). It published 203 complete source
  windows and five expected `partial` no-positive-expansion windows.
- The completed positive-only manifest is
  `data/manifests/training-dataset-builds/2026/08/31/013701963916_7b82974c02774071afbf3b815dd0112a.json`.
  It contains **35,580** rows in **203** selected immutable artifacts, all
  with explicit no-weather declarations.
- ETOPO terrain coverage is complete for the 2,933 FIRMS context tiles; WFIGS
  (3,139 reference perimeters), CWFIS (12,222 record versions), NALCMS source
  archives, and a VIIRS Level-2 metadata inventory (18,354 granules) are also
  retained as non-model context/observability evidence.
- The completed no-weather candidate view is
  `data/manifests/candidate-dataset-builds/2026/08/31/021729652268_27f6f72166ca43c2a2d92c2691deb4eb.json`.
  It has **428,656** candidate rows: **22,656** supported weak positives and
  **406,000** weak-negative proxies. **12,924** positives outside FIRMS
  candidate support are retained as unscored diagnostics.
- The uploadable, self-contained schema-v2 CSV/JSONL release is
  `releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22/`.
  All SHA-256 checksums, CSV row counts, unique example IDs, weather exclusion,
  and finite numeric feature values were verified.
- The chronological no-weather tabular baseline was run and persisted under
  `artifacts/tabular-baseline-201db0d293c56f51/`. On 89,210 validation rows,
  it obtained ROC-AUC 0.9163 and PR-AUC 0.5408. These are weak-label POC
  metrics, not operational performance claims.

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
| `export_candidate_dataset.py` | Self-contained gzip CSV/JSONL upload release with schema, inventory, and SHA-256 checksums. |
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

The sampler expands an eligible FIRMS seed into a bounded incident context.
It deliberately retains FEDS positives without that support as unscored rather
than silently treating them as no-spread cases.

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

After the no-weather 2026-05-11 through 2026-08-22 POC is complete, turn its
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

# Open `wildfire_firms_analysis.ipynb` in a Jupyter-compatible environment
# only when intentionally recollecting its full 2026-05-11 through 2026-08-22
# range. Set WILDFIRE_RUN_NON_WEATHER_PIPELINE=1 in that notebook environment.
# It builds seven-day candidate chunks with a 24-block terrain cache and merges
# their manifests. Then run `notebooks/train_tabular_baseline.ipynb` to verify
# and fit the already published upload release.
```

The build commands are idempotent at the immutable-artifact level and publish
new completed manifests only if all source coverage checks pass. For a full
from-scratch collection, use the ordered commands in
[the no-weather POC guide](no-weather-poc.md); it includes the May 10 FIRMS
context day and August 23 FEDS boundary snapshot.

To create a later weather-bearing version of the 2026-05-11 through 2026-08-22
range, first complete and verify the no-weather POC: FIRMS must cover
2026-05-10 through 2026-08-22, FEDS source snapshots must cover through
2026-08-23, and the positive/candidate views must have completed manifests. Then
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
