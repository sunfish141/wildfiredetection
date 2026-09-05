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
verified one-step tabular baseline, experimental recursive frontier models,
and a completed **incident-sequence two-pass scheduled-sampling experiment**.
A new bounded, weighted synthetic view has been admitted to that research
fit; the earlier failed renderer augmentation artifacts remain unadmitted.
Open-loop results remain mixed and no recursive model is promoted. None of these artifacts is an
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
| Evaluation split | Original tabular baselines group chronological holdout by FEDS `source_snapshot_time`. The incident experiment assigns whole spatially separated FEDS complexes before fitting, with region holdouts, a region feature halo, and a later-time test period. All cells from one incident snapshot stay together. Never randomly split neighboring cells. |
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

### Interactive product

- Added the [Wildfire Atlas FastAPI app](web-app.md): click-to-place intensity
  seeds or current FIRMS loading, 12-hour predictions, 1/3/5/10-second
  playback, pause, single-step, timeline history, and clickable cell inspection.
- The app uses the saved incident pass-2 model and retained terrain, keeping
  calibration and rollout bounds unchanged. It is an explicitly experimental
  preview with no playback time limit, not a model promotion. The rolling
  timeline retains 128 frames while the current state retains its full burned
  mask. FIRMS loading defaults to the notebook's full collection rectangle,
  with streaming 1 km aggregation and no zoom/cell-count restriction.
- FIRMS keys stay server-side; bounded previews require valid responses from
  all three VIIRS feeds and preserve the 3–24-hour observation window.
  Browser state and transient previews do not modify source/training archives.

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
  **not promoted**; its generated rows remain unadmitted for training.
- The subsequent, separately versioned incident experiment is complete at
  `artifacts/incident-sequences-v1-201db0d293c56f51-halo/manifest.json` and
  `artifacts/incident-two-pass-v1-201db0d293c56f51/run_manifest.json`.
  Current FEDS geometry associates 78,154 candidates into 243 conservative
  incident complexes; 350,502 candidates, including 53 weak positives, remain
  explicitly unassigned. Whole-incident, region, and later-time splits are
  established before fitting or augmentation. The region check includes the
  3×3 feature halo.
- Pass 1 fits 3,266 observed frontier rows; pass 2 adds 447 synthetic rows
  (247 positives) at weight 0.25. Both use separate sigmoid probability
  calibration on 329 frontier rows from six calibration complexes. Generated
  rows come only from training fragments, with predicted-state fractions
  increasing from 0.25 to 0.50 to 0.75. The earlier failed augmentation views
  are not inputs to this fit and their admission flags remain unchanged.
- Both passes have fully open-loop 12/24/48/96-hour evaluations on identical
  origins: 9 held-incident, 21 held-region, and 65 later-time fragments.
  Reports include accuracy, spatial precision/recall, coverage, cumulative
  new-burned-area error, and front distance with empty-front diagnostics.
  Results are mixed, with 22 metric regressions beyond 12 hours; neither model
  is promoted. See the [incident experiment report](incident-scheduled-sampling.md)
  for results, scope, artifact checks, and reproducible commands.

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
| `incident_sequences.py` | Versioned FEDS complex association, whole-incident/region/later-time partitions, and gap-safe incident sequence manifests. |
| `incident_transition.py` | Preserved observed FIRMS aggregates, scheduled observation/prediction mixing, calibrated estimator bundles, and bounded frontier/persistence rules. |
| `scheduled_sampling.py` | Two supervised fits, training-only bounded weighted augmentation, separate probability calibration, immutable training views, and verified model loading. |
| `incident_evaluation.py` | Fully open-loop per-incident 12/24/48/96-hour spatial, area, front-distance, and probability metrics. |

The 233-test suite covers collection/build contracts, the candidate release,
both tabular baselines, recursive transitions, sequence gaps, open-loop
evaluation, age eligibility, calibration isolation/replay, and guarded
augmentation persistence, plus incident grouping/splits, feature halos,
scheduled sampling, mixed-view checksums/lineage, and spatial metrics. Run it again after any
source, state-transition, feature, or policy change.

## Remaining work before a credible operational predictor

### 1. Improve future fire-state dynamics

The incident renderer now preserves observed centre counts, maximum/mean
brightness, platform counts, and recency. Simulated intensity decay no longer
rewrites historical brightness. This corrects part of the earlier renderer
loss, but future brightness, persistence/extinction, and platform diversity
still use explicit heuristics. Learn or improve those state targets from
training sequences; do not interpret observation absence as proof of no burn.

The two-pass model still loses most frontier coverage by 96 hours. Examine
state extinction, observation expiry, and candidate support before adding
model complexity. Keep the first 1,414.2 m step-distance bound, burned mask,
probability calibration, and candidate/growth caps versioned in each run.

### 2. Improve the controlled dataset-aggregation experiment

The requested two-pass scheduled-sampling baseline is implemented and fitted.
Its new mixed view was admitted only for a bounded research comparison, with
renderer drift retained as a diagnostic; this supersedes the previous plan
to postpone every fit until the old renderer screen passed. The old failed
inspection artifacts remain unadmitted and immutable.

Further small rounds or changes to mixing/growth controls should be selected
using internal training development splits. Preserve the complete incident,
region, and later-time exclusions; never generate training rows from them.
Compare fully open-loop cases at all four horizons and do not promote a
model based on a one-step improvement alone. Both current fits remain
research artifacts with mixed multi-step results.

### 3. Strengthen incident identity and sequence coverage

The original upload release stays unchanged; a checksum-pinned sidecar now
supplies versioned FEDS incident-complex identities and partitions. The keys
and conservative spatial grouping protect known neighbouring fire context,
but are not an independent operational incident crosswalk. Validate identity
across provider merges and longer periods, and recover fuller observed
sequences without inventing missing-window zero labels.

Only 45 complexes remain in training under the current strict split, and
calibration uses six. Area/front metrics are restricted to released weak-label
domains, with outside-domain and empty-front diagnostics. The later-time
period was withheld from this experiment's fitting but was used by earlier
chronological research; a future prospective period is still needed for a
pristine final test. FEDS/FIRMS dependence remains a weak-label limitation.

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

### 7. Extend the completed interactive application

The user authorized building the application as a research preview before
model promotion. The [FastAPI map app](web-app.md) now implements placed
ignitions and current FIRMS initialization, terrain sampling, 12-hour model
steps, playback/pause, history, and cell inspection. All four requested product
flows are complete. Further product work can add portable scenario save/load
and richer perimeter rendering; model promotion still depends on stronger
held-out rollout evidence.

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
