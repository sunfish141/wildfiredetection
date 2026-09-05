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
Historical ECMWF IFS ── tile + hourly-anchor join ──> retrospective weather features ─┼─> candidate row ─> tabular baseline
ETOPO terrain blocks ── cell-centre sample ──> elevation / slope / aspect ─────────────┘

FEDS snapshots(t) + FEDS snapshots(t+12 h) ──> 1 km newly-burned positives

Optional issued forecast weather ──> separate forward-only operational experiment
```

## Active no-weather proof of concept

The current build target is a **no-weather** CSV candidate release for source
snapshots from 2026-05-11 through 2026-08-22. It uses FIRMS from 2026-05-10
through 2026-08-22, FEDS source snapshots through 2026-08-23, and sampled
ETOPO terrain. Open-Meteo collection and joins are deliberately deferred; a
POC row declares weather unavailable rather than substituting an unarchived
value.

The completed candidate manifest is exported as schema-v2
`candidate_examples.csv.gz` (with an equivalent JSONL stream), then
[`notebooks/train_tabular_baseline.ipynb`](../notebooks/train_tabular_baseline.ipynb)
verifies the release before training. It reads the manifest's exact feature
allowlist and keeps whole `source_snapshot_time` groups on one side of the
chronological holdout. See [the no-weather POC guide](no-weather-poc.md) for
the reproducible build/release sequence and its weak-label limits.

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
   checksum-protected upload directory from that manifest alone. Schema-v2
   exports contain `candidate_examples.csv.gz` for tabular fitting plus
   matching JSONL and unscored-diagnostic payloads.
9. `open_meteo_historical.py` reads a completed base candidate view, plans
   compact candidate weather tiles, and captures rate-limited hourly Open-Meteo
   Historical Weather API ECMWF IFS values per candidate date. It writes raw
   responses, mappings, and a complete-or-partial immutable backfill manifest.
10. `weather_candidate_dataset.py` refuses an incomplete backfill, joins each
    base row only to its mapped tile and UTC hour at or before its anchor, and
    writes a separate immutable weather-bearing candidate view and export. It
    does not modify the base no-weather view.

## What to run to create the positive-only training view

Follow the collection commands in this order:

1. [Collect FIRMS](collecting-data.md#step-2-collect-unfiltered-firms-detections) — dynamic fire evidence.
2. [Collect FEDS](collecting-data.md#step-3-collect-feds-12-hour-perimeter-snapshots) — source snapshots.
3. [Rebuild FEDS snapshots](collecting-data.md#step-4-rebuild-feds-primary-key-snapshot-partitions-no-api-request) — primary-key time normalization, no API call.
4. [Build FEDS labels](collecting-data.md#step-5-build-feds-weak-positive-labels-at-1-km-and-12-hours) — weak positives.
5. [Collect terrain](collecting-data.md#step-10-collect-terrain-source-blocks) — static terrain values.
6. [Build the positive-only view](collecting-data.md#step-11-build-the-positive-only-tabular-training-view) — joined, lineage-rich positive rows.
7. [Build the base candidate view](collecting-data.md#step-11a-build-a-candidate-view-current-release-command) — binary weak-label rows and immutable spine.
8. [Export the candidate release](no-weather-poc.md#build-and-export) — schema-v2 CSV plus provenance/checksums.
9. [Run the tabular-baseline notebook](../notebooks/train_tabular_baseline.ipynb) — verify the release and fit the chronological weak-label baseline.
10. [Audit the cap](collecting-data.md#step-14-audit-the-finished-local-package) — make sure the full data tree remains under 20 GB.

Historical-weather backfill is intentionally outside this POC and is only a
later, separate candidate view.

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

That wiring is complete for the retained archive. The completed May 11–August
22 view has 428,656 rows: 22,656 supported weak positives and 406,000
weak-negative proxies, plus 12,924 FIRMS-uncovered positives in a separate
diagnostic file.
It has a whole-source-snapshot chronological split and an explicit numeric
feature allowlist. Load only the manifest-selected rows, pass that allowlist to
`train_tabular_baseline`, and use `split_group_column="source_snapshot_time"`.

Do not use absent FEDS labels as zeros, use `ingested_at` as a feature, or join
final WFIGS perimeters as earlier state. Historical weather is permitted only
through the contracted, archived Open-Meteo Historical Weather API/ECMWF IFS
path and must remain explicitly marked as retrospective analysis; do not
substitute an unversioned reanalysis or latest-endpoint lookup.

## Weather and wind direction — after the POC

Weather is deliberately absent from the active 2026-05-11 through 2026-08-22
POC as well as the completed 2026-05-31 through 2026-08-10 release. Only after
the POC is verified may a separate candidate view backfill the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
with `models=ecmwf_ifs` at every planned candidate tile and each row's hourly
weather anchor. `floor_weather_hour` derives the feature's `weather_observed_at`
by flooring the UTC prediction cutoff to the start of the hour, so a feature
never uses a weather hour after the cutoff.

The compact request cover defaults to a 10 km maximum candidate-to-requested-
tile distance before the provider's weather-grid snap. This is a deliberate
spatial approximation, not 1 km meteorology; each row retains the mapping and
distance. The backfill and join require the same immutable base candidate
manifest identity.

Every historical-analysis measurement must retain:

- `historical_analysis` feature mode, provider/model, grid/tile identifier,
  candidate-cell mapping, variable, units, and valid hour;
- immutable raw-response artifact ID, retrieval time, request range, and the
  deterministic weather-anchor alignment rule; and
- a join decision showing that the tile and valid hour match the candidate row.

The mapping must include every selected candidate cell, not just positive
FIRMS/FEDS examples, so target=0 weak-negative proxies use the same feature
contract. Wind direction should be retained as provider U/V components (and,
if needed, derived `sin(direction)`/`cos(direction)`), rather than an
unwrapped 0–360° scalar. The rate-limited collector paces calls, retries
ordinary transient failures, honours a 429 cooldown, and pauses after two
consecutive 429 responses.

Historical ECMWF IFS values describe retrospective conditions for offline
training analysis; they are not a reconstructed forecast and must not support
an operational as-of claim. For that separate experiment, the optional
forward Open-Meteo Single Runs capture retains an explicit model/run, raw
response, captured availability timestamp, and only values valid after that
availability time.

## Acceptance criteria for the first assembled table

Before fitting the baseline, verify that every row has:

- a canonical cell ID, cell-centre coordinates, actual UTC cutoff, and
  `target_end_at = cutoff + 12 h`;
- source IDs/transformation versions for label, FIRMS, and terrain inputs;
- terminal coverage for every FIRMS product/day needed through the availability
  cutoff, plus the completed training-view manifest that selected the row;
- `weak_satellite` target=1 or explicitly named `weak_negative_proxy` target=0
  tier, plus a label-observability field;
- no operational feature made available after the cutoff; a
  `historical_analysis` weather feature may be retrieved later, but its valid
  hour must match the documented hourly anchor at or before the cutoff;
- a deterministic candidate/negative-selection reason; and
- a time-ordered train/validation split grouped by source snapshot, with no
  duplicated example ID.

The model artifact must store its feature list, cutoff/availability policy,
grid and label versions, metrics, and calibration output. A calibrated result
can be converted to latitude/longitude only through the predicted canonical
cell centroids.

## Experimental recursive transition baseline

`recursive_transition.py` wraps the completed no-weather classifier in the
smallest reproducible state transition needed by an interactive map. A user
ignition supplies a canonical cell and intensity from zero to one. At every
12-hour step, active cells are converted to synthetic FIRMS-compatible
brightness, detection-count, recency, and 3-by-3 neighbourhood features;
unburned cells in the existing two-cell candidate radius are scored; and a
fixed probability threshold determines which cells become active next.

The recursive classifier is fitted only on historical candidate rows where
`firms_center_has_detection = 0`, and it removes all six centre-detection
features from its input contract. This matches rollout semantics: a candidate
is unburned before the transition, so it cannot already supply its own active
FIRMS detection. The model uses the remaining seven 3-by-3 fire-state fields
and six terrain fields.

This is an application baseline, not a newly trained multi-step model. The
following transition behavior is deliberately heuristic and versioned:

- intensity maps linearly from 305 K to 367 K synthetic TI4 brightness;
- without calibration, intensity maps to one through three synthetic
  detections and one provider-neutral observation stream;
- renderer v2 optionally uses training-only, five-bin intensity calibration
  for detection and platform counts (see below); platform diversity remains
  a count proxy, because per-row platform identities are unavailable;
- active state retains observation age, advances it by 12 hours per step,
  and renders evidence only in the inclusive 3--24-hour eligibility window;
- new ignitions use a configurable age of 7.5 hours, the midpoint of the
  eligible acquisition ages [3, 12] within the preceding step;
- active cells persist for two steps by default and retain 85% intensity per
  step;
- the default deterministic ignition threshold is 0.05 (approximately 20.3%
  precision and 90.4% recall on the frontier chronological holdout); and
- burned cells cannot ignite again during the scenario.

The classifier still supplies only the probability of new burning. Future
FIRMS brightness, fire persistence, and intensity have not yet been learned or
validated, and recursive rollouts can accumulate error. The web application
must label results as an experimental simulation and retain the transition
parameters with each scenario.

Train and persist this separate frontier model with:

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.train_recursive_transition \
  --release releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22 \
  --output artifacts/recursive-frontier-baseline-201db0d293c56f51
```

### Incremental rollout-training work

Increment 1 adds only the temporal sequence contract in
`rollout_sequences.py`; it does not retrain either baseline. Candidate rows
are grouped by whole FEDS `source_snapshot_time`, with every cell from one
snapshot kept in the same `RolloutSnapshot`. Distinct snapshots continue a
sequence only when they are exactly 12 hours apart. A larger whole-12-hour gap
starts a new sequence, while an irregular cadence is rejected.

The sequence metadata stores integer dataframe row positions rather than
copying examples. This preserves every cell's local-solar-aligned `anchor_at`
and exact `target_end_at = anchor_at + 12 hours` while allowing later training
to retrieve a complete snapshot through `snapshot_frame`. It also validates
that every present FEDS target snapshot is exactly 12 hours after its source
and that a cell appears at most once in a source snapshot. Weak-negative proxy
rows correctly retain a missing observed `target_snapshot_time`; sequence
metadata derives the intended transition endpoint as source plus 12 hours
without turning that missing observation into a verified negative.

This is preparation for supervised autoregressive rollouts. It does not yet
feed predictions back into training, generate synthetic examples, or use held
out snapshots for augmentation.

### Open-loop baseline before augmentation

Increment 2 adds `rollout_evaluation.py` without changing either fitted model.
It selects the first run of eight consecutive validation snapshots, initializes
the origin from observed FIRMS centre detections, and then advances entirely
with model-generated state. Metrics are recorded after 1, 2, 4, and 8 steps
(12, 24, 48, and 96 hours).

Each horizon is evaluated on that historical snapshot's rows with
`firms_center_has_detection = 0`, matching the recursive frontier training
contract. A labelled candidate cell that the recursive frontier fails to reach
receives probability zero and can become a false negative. Model candidates
outside the released snapshot's candidate domain are counted separately and
remain unscored; they are never converted into false positives because the
release provides no label or observation claim for those cells.

Run and persist the diagnostic with:

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.rollout_evaluation \
  --release releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22 \
  --model-bundle artifacts/recursive-frontier-baseline-201db0d293c56f51/recursive_frontier_baseline.joblib \
  --data-root data \
  --output artifacts/recursive-frontier-baseline-201db0d293c56f51/open_loop_evaluation.json
```

The resulting weak-label metrics establish how quickly the current heuristic
rollout degrades before any generated state is admitted to training.

### One-step augmentation inspection

Increment 3 adds `rollout_augmentation.py`, but still does not fit or overwrite
a model. For each consecutive pair wholly inside the training split, it starts
from observed FIRMS centre detections at the first snapshot, advances the
current recursive model once, and derives the following predicted frontier.
Synthetic feature rows are created only where that frontier intersects the
next snapshot's no-centre historical candidate domain. The next snapshot's
existing target is retained as supervised truth.

Historical frontier cells the model did not reach and predicted candidates
outside the historical domain remain manifest diagnostics. They are not
silently discarded as successful predictions, converted to negative labels,
or admitted to the generated CSV. The manifest compares synthetic and observed
feature distributions and sets `training_admitted = false` so inspection is a
required separate decision before retraining.

The original v1 inspection is retained unchanged. Generate a **new** v2
inspection directory with:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python -m wildfire_data.rollout_augmentation \
  --release releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22 \
  --model-bundle artifacts/recursive-frontier-baseline-201db0d293c56f51/recursive_frontier_baseline.joblib \
  --data-root data \
  --reference-manifest artifacts/recursive-frontier-baseline-201db0d293c56f51/one-step-augmentation/manifest.json \
  --output artifacts/recursive-renderer-v2-201db0d293c56f51/one-step-augmentation
```

### Renderer v2 calibration and admission screen

`recursive_calibration.py` fits mean observed centre detection and platform
counts in five fixed equal-width intensity bins. It reads feature values only
from `train` rows with centre detections, and uses the global training mean
for an empty bin. The renderer rounds means half-up to integer counts.
Neighbour platform-count proxies combine by maximum, bounded by three; this
is not a reconstruction of satellite identities or overpass schedules.

Observed initialization now requires the released centre observation age.
It cannot substitute the new-ignition age for missing observed recency. Ages
outside [3, 24] fail initialization, and surviving evidence ages past 24 hours
become missing rendered observations even while the simulated cell remains
active. The geometric frontier still uses active simulated cells. No new
observation is inferred merely because a cell survives.

The CLI verifies the release CSV checksum and row count, checks whole-snapshot
chronological splits, and fits counts only on training observations. The model
bundle's training snapshot boundary must match calibration. Augmentation
rejects validation generation, missing splits, mixed snapshots, duplicate
example IDs, and supplied sequences that differ from the source frame.

The v2 inspection manifest retains the full renderer/scenario contract,
calibration training snapshots and row count, release/model checksums, row
lineage, pair coverage, class balance, missingness, means, standard deviations,
and quartiles. Each generated row retains its original example ID, training
split, cutoff, and target end; resolve its original evidence/label provenance
through the checksum-pinned base release. An existing output directory is
rejected, and only the final manifest denotes completion.

Before examining v2 results, the engineering screen was fixed at a maximum
10-percentage-point missingness gap and 0.5 observed standard deviations of
mean or quartile difference for each of the seven FIRMS features. Empty
comparisons fail. This is a training-distribution diagnostic, not an
operational validation criterion. `training_admitted` remains false even if
the screen passes; the next step would be a separate bounded, weighted
scheduled-sampling experiment.

Replay the exact inspected renderer with the unchanged classifier at the
original validation origin:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python -m wildfire_data.rollout_evaluation \
  --release releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22 \
  --model-bundle artifacts/recursive-frontier-baseline-201db0d293c56f51/recursive_frontier_baseline.joblib \
  --renderer-manifest artifacts/recursive-renderer-v2-201db0d293c56f51/one-step-augmentation/manifest.json \
  --data-root data \
  --output artifacts/recursive-renderer-v2-201db0d293c56f51/open_loop_evaluation.json
```

This comparison isolates renderer changes; it is not a scheduled-sampling
fit. Original v1 artifacts remain historical results, and running current v2
code without a calibration manifest does not reproduce their v1 semantics.
