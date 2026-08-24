# First tabular training pipeline

This is the contract for the first model, not a claim that an operational
spread predictor is already trained. It turns a prediction cell and an
as-of time into a probability that the cell will newly burn in the following
12 hours, then returns that cell's centroid latitude/longitude.

## Fixed choices

| Choice | First implementation |
| --- | --- |
| Geography | CONUS + Canada; Alaska excluded from the first FEDS-aligned dataset |
| Spatial key | 1 km North America Albers equal-area cell (`ESRI:102008`) |
| Cell ID | `naea-1km:x=<integer>:y=<integer>` |
| Prediction horizon | 12 hours |
| Label | FEDS perimeter at `t + 12 h` minus FEDS perimeter at `t` |
| Label tier | `weak_satellite` positives; FIRMS-seeded `weak_negative_proxy` target=0 rows |
| First model | `HistGradientBoostingClassifier` tabular baseline |
| Split | Strict chronological holdout; no random neighboring-cell split |

The canonical grid is defined in `src/wildfire_data/training_grid.py`. A
collection tile, FIRMS point, or weather-plan tile is never itself a model
cell.

## Data flow

```text
FIRMS detections ── availability gate ──> FIRMS-only candidate cells + 1 km features ─┐
                                                                                       ├─> candidate row ─> tabular baseline
ETOPO terrain blocks ── cell-centre sample ──> elevation / slope / aspect ────────┘

FEDS snapshots(t) + FEDS snapshots(t+12 h) ──> 1 km newly-burned positives

Issued forecast weather ──> not collected yet; no weather feature is joined today
```

`FEDS` labels share satellite evidence with FIRMS and are therefore not
independent ground truth. They are useful for an initial weak-label baseline,
but each output must retain label tier, source IDs, source snapshot times, and
time-alignment method.

## What the current code does

1. `collect_feds` stores raw FEDS perimeter responses and source-faithful
   normalized snapshots at the provider's 12-hour cadence.
2. `rebuild_feds_normalization` replays an already retained FEDS capture
   without an API request and groups snapshots by the authoritative timestamp
   embedded in the provider primary key.
3. `build_feds_labels` calculates `perimeter(t + 12 h) - perimeter(t)`,
   rasterizes only changed area to the canonical 1 km cells, and stores
   positive weak labels. It uses an explicit cell-local-solar-to-UTC estimate
   for FEDS by default; the source timestamp is never silently rewritten.
4. `fire_state_features.py` builds leakage-safe FIRMS features. A detection
   contributes only when its acquisition time plus the chosen availability lag
   is at or before the feature cutoff. It produces centre-cell and 3 km × 3 km
   counts, TI4 summaries, platform count, active-cell count, and recency.
5. `terrain_features.py` samples the retained ETOPO source pixel at the
   canonical cell centre and returns elevation, slope, aspect sine/cosine, and
   coverage/provenance fields without materializing a continental cache.
6. `tabular_baseline.py` trains a histogram gradient-boosting classifier,
   rejects known leakage columns, and reports ROC-AUC, PR-AUC, Brier score,
   ECE, and MCE. For FEDS rows, the chronological split groups on
   `source_snapshot_time`, not cell-local `anchor_at`, so all cells sourced
   from one satellite snapshot remain wholly in train or holdout.
7. `build_training_dataset` writes a bounded positive-only training view.
   It joins each FEDS-positive cell to cutoff-safe FIRMS features and sampled
   terrain, carries raw-artifact lineage, and writes explicit weather
   unavailable/missing fields. Before assembly it requires terminal
   (`complete` or `empty-confirmed`) coverage for every configured FIRMS
   product/day intersecting the usable feature interval. It atomically
   publishes a completed-view manifest only after every selected partition
   succeeds; interrupted partial artifacts are not readable as a dataset.
   It does not create zero targets.
8. `candidate_dataset.py` reads exactly one completed positive view, rechecks
   terminal FIRMS coverage for each source snapshot, materializes candidate
   features, and atomically publishes the candidate view. It writes target=0
   only as `weak_negative_proxy` and puts FIRMS-uncovered positives in a
   separate diagnostic stream. `export_candidate_dataset.py` creates the
   checksum-protected upload directory from that manifest alone.

## What to run to create the positive-only training view

Follow the collection commands in this order:

1. [Collect FIRMS](collecting-data.md#step-2-collect-unfiltered-firms-detections) — dynamic fire evidence.
2. [Collect FEDS](collecting-data.md#step-3-collect-feds-12-hour-perimeter-snapshots) — source snapshots.
3. [Rebuild FEDS snapshots](collecting-data.md#step-4-rebuild-feds-primary-key-snapshot-partitions-no-api-request) — primary-key time normalization, no API call.
4. [Build FEDS labels](collecting-data.md#step-5-build-feds-weak-positive-labels-at-1-km-and-12-hours) — weak positives.
5. [Collect terrain](collecting-data.md#step-10-collect-terrain-source-blocks) — static terrain values.
6. [Build the positive-only view](collecting-data.md#step-11-build-the-positive-only-tabular-training-view) — joined, lineage-rich positive rows.
7. [Build the uploadable candidate view](collecting-data.md#step-11a-build-the-uploadable-no-weather-candidate-view) — binary weak-label rows and release.
8. [Audit the cap](collecting-data.md#step-14-audit-the-finished-local-package) — make sure the full data tree remains under 20 GB.

WFIGS and CWFIS collection add validation and incident context, but neither
creates the initial 12-hour target. The Level-2 inventory and NALCMS archives
remain correctly retained source evidence, not inputs to the first table.

## Current boundary: completed no-weather candidate view

The positive-only table assembler is implemented. It writes one target=1 row
per FEDS-positive cell with FIRMS/terrain lineage and explicit weather
missingness. A complete manifest selects exactly one consistent derived view;
do not glob together old immutable `training-examples` artifacts. The
FIRMS-only candidate policy is implemented in `candidate_sampling.py`:
cutoff-eligible FIRMS detections expand a fixed deterministic square radius;
FEDS positives within that support are target=1 and other selected cells are
target=0 only as explicitly named `weak_negative_proxy` rows. Positives
outside FIRMS support remain separate unscored diagnostics. This does not
claim clear/no-burn coverage.

That wiring is complete for the retained archive. The completed view has
305,528 rows: 19,528 supported weak positives and 286,000 weak-negative
proxies, plus 11,848 FIRMS-uncovered positives in a separate diagnostic file.
It has a whole-source-snapshot chronological split and an explicit numeric
feature allowlist. Load only the manifest-selected rows, pass that allowlist to
`train_tabular_baseline`, and use `split_group_column="source_snapshot_time"`.

Do not use absent FEDS labels as zeros, use `ingested_at` as a feature, join
final WFIGS perimeters as earlier state, or train on notebook
`open_meteo_weather_*.csv` files.

## Weather and wind direction

Weather is **not in the first table yet**. The checked-in HRDPS artifact is a
candidate/retrieval plan, not forecast measurements. It has no temperature,
humidity, precipitation, wind speed, wind direction, or gust values.

The future issued-forecast extractor must retain, for every weather value:

- grid/tile identifier and variable;
- model run/issue/publication time;
- valid time and retrieval time; and
- the availability decision at the row's as-of cutoff.

Wind direction should be represented as provider U/V components (and, if
needed, derived `sin(direction)`/`cos(direction)`), rather than an unwrapped
0–360° scalar. Until such issued-at values are retained, the baseline is an
explicit no-weather ablation—not an operational weather-aware forecast.

## Acceptance criteria for the first assembled table

Before fitting the baseline, verify that every row has:

- a canonical cell ID, cell-centre coordinates, actual UTC cutoff, and
  `target_end_at = cutoff + 12 h`;
- source IDs/transformation versions for label, FIRMS, and terrain inputs;
- terminal coverage for every FIRMS product/day needed through the availability
  cutoff, plus the completed training-view manifest that selected the row;
- `weak_satellite` target=1 or explicitly named `weak_negative_proxy` target=0
  tier, plus a label-observability field;
- no evidence made available after the cutoff;
- a deterministic candidate/negative-selection reason; and
- a time-ordered train/validation split grouped by source snapshot, with no
  duplicated example ID.

The model artifact must store its feature list, cutoff/availability policy,
grid and label versions, metrics, and calibration output. A calibrated result
can be converted to latitude/longitude only through the predicted canonical
cell centroids.
