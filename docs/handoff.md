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

The repository has a completed **no-weather weak-label candidate dataset**, a
verified one-step tabular baseline, and an experimental recursive frontier
baseline. The recursive model can advance synthetic fire state, but its
generated states have not been admitted to training and its open-loop quality
degrades sharply after the first step. None of these artifacts is an
operational predictor, a weather-aware forecast, or a verified clear/no-burn
dataset.

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
| Recursive experiment | A separate no-centre frontier classifier consumes seven synthetic/observed 3×3 FIRMS-like fields plus six terrain fields. Twelve-hour state recursion is explicit and heuristic until future fire-state targets are learned. |
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
- The separate recursive frontier baseline is persisted under
  `artifacts/recursive-frontier-baseline-201db0d293c56f51/`. It excludes all
  centre-detection features and trains only on rows with
  `firms_center_has_detection = 0`. On 78,598 chronological validation rows it
  obtained ROC-AUC 0.9101, PR-AUC 0.4653, and Brier score 0.0429.
- The 203 source-snapshot transitions form five gap-safe temporal sequences of
  1, 5, 13, 9, and 175 snapshots. Missing FEDS observation windows split a
  sequence; weak-negative rows retain missing observation identity rather than
  being promoted to observed negatives.
- The persisted open-loop evaluation starts from 455 observed validation
  active cells. Frontier-domain coverage falls from 29.0% at 12 hours to 3.3%
  at 96 hours; recall falls from 35.4% to 2.3%, while candidates outside the
  historical label domain grow from 5,979 to 28,058. This is recorded in
  `artifacts/recursive-frontier-baseline-201db0d293c56f51/open_loop_evaluation.json`.
- The training-only one-step augmentation inspection processed 157 consecutive
  snapshot pairs and published 34,249 matched synthetic rows, including 3,968
  positives. It covers 11.6% of the historical frontier and 37.8% of its
  positives, while 770,659 model candidates lie outside the corresponding
  historical frontier. Its manifest deliberately declares
  `training_admitted = false`.
- The separate **renderer v2** inspection is now complete under
  `artifacts/recursive-renderer-v2-201db0d293c56f51/one-step-augmentation/`.
  It calibrates counts on 34,462 observed centre rows from the 162 training
  snapshots, preserves observed age, advances ages by 12 hours, and excludes
  evidence outside the 3--24-hour window. It generated **32,044** matched rows
  (**2,619** positives) across the same 157 training pairs. Frontier coverage
  is **10.85%**, positive-frontier coverage **24.92%**, and 761,097 candidates
  lie outside the historical frontier. Four of seven renderer feature checks
  fail; `training_admitted` remains **false**. See the
  [renderer experiment report](recursive-renderer-v2.md).
- Exact-contract v2 open-loop replay is persisted at
  `artifacts/recursive-renderer-v2-201db0d293c56f51/open_loop_evaluation.json`.
  With the same classifier, origin, 455 active cells, and evaluation domains,
  12-hour recall rises from 35.4% to 82.1%, but 96-hour recall drops from 2.3%
  to 0%, and 96-hour domain coverage from 3.34% to 0.87%. This experiment is
  **not promoted** and does not justify a scheduled-sampling fit.

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
| `recursive_transition.py` / `train_recursive_transition.py` | Synthetic FIRMS-compatible cell state, deterministic 12-hour transitions, and the separately persisted no-centre frontier classifier. |
| `recursive_calibration.py` | Training-only intensity-to-count calibration, snapshot split checks, and versioned calibration provenance. |
| `rollout_sequences.py` | Gap-safe temporal grouping that preserves whole FEDS source snapshots and cell-specific cutoffs. |
| `rollout_evaluation.py` | Fixed-origin, fully open-loop validation at 12, 24, 48, and 96 hours without inventing labels outside the released candidate domain. |
| `rollout_augmentation.py` | Training-only one-step synthetic-state generation, matched-domain coverage diagnostics, feature-distribution comparison, and an explicit training-admission guard. |

The 204-test suite covers collection/build contracts, the candidate release,
both tabular baselines, recursive transitions, sequence gaps, open-loop
evaluation, age eligibility, calibration isolation/replay, and guarded
augmentation persistence. Run it again after any
source, state-transition, feature, or policy change.

## Remaining work before a credible operational predictor

### 1. Resolve remaining renderer drift before retraining

The requested first correction is implemented and the guarded augmentation
has been regenerated. Both the original v1 inspection and the new v2
inspection remain unadmitted. The original v1 comparison showed:

- synthetic 3×3 detection count averages 1.22 versus 4.38 in matched observed
  rows;
- synthetic hours since last detection is always 0 versus an observed mean of
  15.8 hours; and
- synthetic platform count averages 0.45 versus 1.03 observed.

Renderer v2 carries observed age through initialization, advances it by 12
hours per step, and assigns new ignitions an explicit 7.5-hour age (the
midpoint of eligible acquisition ages [3, 12] in the preceding step). Five
fixed intensity bins calibrate detection/platform counts from training centre
observations only. Release and model checksums, calibration, scenario
parameters, cutoffs, and original row IDs are retained; a new output path is
required. Validation augmentation and inconsistent snapshot splits are rejected.

In the v2 matched cohort, synthetic/observed means are 1.99/3.46 detections,
12.52/15.90 hours since detection, and 0.47/0.90 platforms. The screen still
fails brightness max/mean, recency, and platform count, including a
12.66-percentage-point missingness gap. Positive-frontier coverage declined
from 37.75% to 24.92%. Count calibration and explicit age alone do not solve
the state-distribution mismatch.

The next experiment should address brightness/observation-history loss,
platform-diversity approximation, and frontier support. The current slider
clips brightness to 305--367 K and decays it with simulated intensity; counts
are conditional means, and platform identities are not retained in the
released row aggregates. Those remain explicit heuristics, not future-state
targets. Do not relax the screen or admit rows merely to obtain a fit.
The fixed-origin replay also fails the multi-step comparison despite improved
12-hour scores. The next retraining and application prerequisites remain unmet.

### 2. Run one controlled scheduled-sampling fit

Only after the renderer comparison is acceptable, publish a new immutable
mixed training view containing the original observed frontier rows plus a
bounded, explicitly weighted subset of synthetic rows. Never modify the base
candidate release or current model bundle in place, and never generate
augmentation from validation snapshots.

Fit a new model artifact, then rerun the same fixed-origin open-loop evaluation
at 12, 24, 48, and 96 hours. Compare it directly with
`open_loop_evaluation.json`; do not promote the new model if one-step metrics
improve while multi-step coverage, drift, or outside-domain expansion worsens.

### 3. Add incident- and region-held-out evaluation

The current upload release retains `contributing_fire_count` but not a durable
per-row FEDS incident identity. Preserve or derive a versioned incident key
before incident-held-out training. Then report later-time, held-incident, and
held-region results in addition to the chronological snapshot holdout. Treat
FEDS/FIRMS dependence as a weak-label limitation; do not present these scores
as independent ground truth.

### 4. Improve observation handling and valid target=0 evidence

The completed FIRMS-only candidate policy and weak-negative proxy contract do
not need to be redesigned before the next experiment. They still do *not*
establish a verified no-fire label. Collect paired VIIRS fire-mask/QA and
matching geolocation cutouts before claiming observation-aware negatives, and
keep cloudy, missing, partial, or unprocessed areas unknown.

### 5. Backfill historical weather; optionally capture issued forecasts

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

### 6. Improve static and operational-context features

- Define target-grid categorical compaction for NALCMS (for example class
  fractions or mode); never average categorical IDs.
- Add fuel moisture, drought/soil moisture, water, roads, and suppression
  context only with versioned spatial/time semantics and a storage admission.

### 7. Build the interactive inference application

After a recursive model passes the controlled comparison, add the application
wrapper that accepts either a user ignition or current FIRMS detections, maps
them to canonical 1 km cells, samples terrain, advances versioned 12-hour
state, and emits cell probabilities/centroids for the map. The interface must
retain scenario parameters and label recursive results as experimental
simulation output.

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
