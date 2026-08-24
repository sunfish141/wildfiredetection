# Re-collect the compact wildfire dataset

Run these commands from the repository root, one at a time. They rebuild the
U.S./Canada **prediction/label range** for **2026-05-31 through 2026-08-10**,
inclusive. The commands deliberately include a leading FIRMS context date and
a following FEDS source date where those are required to construct that range;
those boundary dates do not extend the target range. Every archive-writing
command writes under `data/`, so do not run two of them concurrently.

The package has a hard **20,000,000,000-byte** limit, including existing CSV
exports and notebook caches. The collectors preserve existing evidence and
stop before an admitted write would exceed the policy; they never delete old
files to make space.

## One-time setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r config/requirements.txt
```

Create `config/.env` before Step 2:

```text
NASA_FIRMS_API_KEY=replace-with-your-key
```

## Step 1: Inspect the 20 GB budget

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.inspect_storage_budget --data-root data
```

You get `data/retention/storage_budget.csv`: every retained file, its storage
category, and the remaining whole-package/category capacity. This downloads
nothing.

## Step 2: Collect unfiltered FIRMS detections

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_firms \
  --start 2026-05-30 \
  --end 2026-08-10 \
  --product VIIRS_SNPP_NRT \
  --product VIIRS_NOAA20_NRT \
  --product VIIRS_NOAA21_NRT
```

You get immutable FIRMS responses and lossless normalized rows for every
returned detection: coordinates, acquisition time, TI4/TI5 brightness, FRP,
confidence, scan/track footprint, day/night, platform, and all provider
fields. You also get one coverage record per product/day.

This is the dynamic fire-state input for the first baseline. The `305 K` TI4
threshold is only a derived pass/fail field; it does **not** discard lower
brightness source rows. A complete run ends with `0 coverage windows need
retry.`

The extra **2026-05-30** day supplies the full 24-hour FIRMS lookback for
labels whose FEDS source time begins on 2026-05-31. With the configured
three-hour FIRMS availability lag, the last eligible FIRMS acquisition for
the 2026-08-10 source date remains on 2026-08-10, so an 2026-08-11 FIRMS
collection is not required. Recalculate this padding if the lookback, time
alignment, geography, or availability-lag policy changes.

### If FIRMS was already collected for 2026-05-31 through 2026-08-10

Add only the missing leading context day; do not re-download the already
retained range:

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_firms \
  --start 2026-05-30 \
  --end 2026-05-30 \
  --product VIIRS_SNPP_NRT \
  --product VIIRS_NOAA20_NRT \
  --product VIIRS_NOAA21_NRT
```

After it completes, rebuild the Step 11 training view. Do not concatenate the
older edge-incomplete view with the rebuilt view: they contain the same
logical examples with different available FIRMS context. Use the rebuilt view
selected by the training-dataset version/coverage policy.

## Step 3: Collect FEDS 12-hour perimeter snapshots

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_feds \
  --start 2026-05-31 \
  --end 2026-08-11 \
  --archive-root data
```

You get immutable NASA FEDS response pages, source-faithful perimeter records,
and coverage records for 12-hour snapshot windows. The default scope is
**CONUS + Canada**. Do not add `--include-alaska` to the first training
dataset: FEDS timestamps use a local-solar convention that the initial
UTC-feature alignment does not support there.

FEDS is a satellite-derived source, not an agency perimeter history. Its
native source identity and timestamp are retained, and every downstream label
is marked `weak_satellite`; do not treat it as independent ground truth from
FIRMS.

The final **2026-08-11** FEDS source date is boundary evidence for comparing
the final 2026-08-10 source snapshot to its next 12-hour snapshot. It is not
an additional prediction/label date.

## Step 4: Rebuild FEDS primary-key snapshot partitions (no API request)

~~~bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.rebuild_feds_normalization \
  --start 2026-05-31 \
  --end 2026-08-11 \
  --data-root data
~~~

You get the v2 source-faithful FEDS snapshot partitions and observed-snapshot
coverage records derived only from the immutable response pages in Step 3.
The command makes **no network request** and does not recollect FEDS. It uses
the timestamp embedded in each provider primary key as the snapshot identity,
while preserving the provider time field for audit.

If several FEDS captures exist, it selects the largest, newest coherent
capture and prints its timestamp. To reproduce a specific replay, add
`--captured-at <that-ISO-8601-timestamp>`.

Run this after the already captured FEDS range, and after any intentional
FEDS refresh. It prevents the provider wall-clock time field from being used
as the sole snapshot grouping key.

The one extra source date is required so the final 12-hour label ending on
2026-08-11 can be formed for the requested 2026-08-10 prediction interval.

## Step 5: Build FEDS weak-positive labels at 1 km and 12 hours

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.build_feds_labels \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --data-root data
```

You get derived JSON Lines label records for canonical 1 km North America
equal-area cells. A record means that a cell newly entered the future FEDS
perimeter during the next 12-hour interval:

```text
FEDS perimeter(t + 12 h) − FEDS perimeter(t)
```

Step 4 retained the following source date so this command can include the
final collection date. The output is **positive-only**:
an omitted cell is not a no-spread label. Labels retain the FEDS source time,
the estimated cell-local-solar-to-UTC anchor, source IDs, overlap fraction,
and the `weak_satellite` tier.

## Step 6: Collect U.S. final/reference perimeters

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_wfigs \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --archive-root data
```

You get paginated WFIGS GeoJSON, normalized U.S. perimeter geometry, incident
identifiers, source times, acreage, and provenance. These are final/reference
geometries for incident matching and validation—**not** 12-hour spread labels.

## Step 7: Collect Canadian incident context

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_cwfis_active_fires \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --archive-root data
```

You get CWFIS active-fire record versions: agency incident IDs, status,
reported size, point geometry, and `record_start`/`record_end`. These are
Canadian incident-context features/provenance, not perimeter labels.

## Step 8: Save the VIIRS Level-2 inventory

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_viirs_l2 \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --platform snpp \
  --platform noaa20 \
  --platform noaa21 \
  --dry-run
```

You get CMR inventory evidence for SNPP, NOAA-20, and NOAA-21 active-fire
files: granule ID, observation interval, footprint, version, and metadata.
This is **not** pixel data. No fire-mask/QA arrays or matching geolocation
arrays are downloaded, so it cannot create clear-no-fire or cloud-aware
negative labels yet.

Never use `--legacy-fire-files-only` in this 20 GB package. A fire product
without its matching geolocation product is not a complete observation.

## Step 9: Make the issued-weather retrieval plan

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.plan_forecast_tiles \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --model hrdps \
  --data-root data
```

You get a scored HRDPS fire-context tile plan under
`data/weather/forecast-tile-plans/`. It records selected/capped 96 km tiles
and `fire_evidence_score`, `forecast_availability_score`, and
`retention_priority_score`.

You do **not** get temperature, humidity, precipitation, wind speed, wind
direction, or gust values. This repository does not yet have the compact
issued-forecast tile extractor, so no weather column is eligible for the first
training table. The notebook's Open-Meteo CSVs are visualization caches and
must not be substituted here.

## Step 10: Collect terrain source blocks

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_static_terrain \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --data-root data
```

You get immutable NOAA ETOPO source-subset TIFF bytes, normalized provenance,
and compressed terrain blocks in `data/static/etopo-2022-15s/`. Each retained
source block has elevation, slope, and downhill aspect. The training pipeline
samples those source blocks at canonical 1 km cell centres; it does not create
a second continental training grid.

## Step 11: Build the positive-only tabular training view

~~~bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.build_training_dataset \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --data-root data
~~~

You get immutable, lineage-rich rows under data/normalized/training-examples/.
Each row is one FEDS-positive canonical 1 km cell, with its actual UTC cutoff,
12-hour target end, FIRMS features from the trailing 24 hours (default
three-hour availability lag), sampled ETOPO terrain, FEDS/FIRMS raw-artifact
IDs, and explicit weather missingness. The command first verifies that every
required FIRMS product/day is `complete` or `empty-confirmed`; it stops with
the exact product/dates to collect if a feature-window edge is missing.

Its final line names a completed training-view manifest under
`data/manifests/training-dataset-builds/`. That manifest is the only safe
selector for the immutable row artifacts. If a run is interrupted, its
partial artifacts remain retained but are not part of a readable training
view; rerun this exact command to publish a complete one.

This is deliberately a **positive-only** training view. It does not invent
zero targets, candidates, weather, or wind direction, and therefore cannot
fit a binary classifier by itself. Step 11a applies the retained FIRMS-only
candidate/weak-negative policy before the tabular baseline can be trained.

## Step 11a: Build the uploadable no-weather candidate view

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.build_candidate_dataset \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --split-start 2026-05-31 \
  --split-end 2026-08-10 \
  --data-root data
```

This reads one completed positive-only manifest, rechecks FIRMS product/day
coverage for every source snapshot, adds cutoff-safe FIRMS and terrain
features to deterministic FIRMS-only candidate cells, and atomically publishes
a completed candidate-view manifest. It creates target=0 only as explicitly
named `weak_negative_proxy` rows; it does not claim clear/no-burn evidence.

Export the selected view outside `data/` so the portable copy does not consume
the 20 GB archive budget:

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.export_candidate_dataset \
  --data-root data \
  --candidate-manifest data/manifests/candidate-dataset-builds/2026/08/24/040059951477_76e075f405f24287ba9531bea436cbe5.json \
  --output releases/wildfire-spread-firms-feds-no-weather-2026-05-31_to_2026-08-10
```

The current export has 305,528 candidate rows, 11,848 unscored positives,
JSON Lines gzip files, a schema, manifest, inventory, and SHA-256 checksums.
For a notebook version of these same commands, use
`notebooks/build_uploadable_dataset.ipynb`.

## Step 12: Archive Canada land-cover source evidence

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_nalcms_land_cover \
  --data-root data \
  --release canada
```

You get one immutable CEC NALCMS Canada source archive and its coverage and
provenance. It is source evidence only—there is no model-ready land-cover
feature yet. Leave about 2 GB free in `/tmp` for the temporary download.

## Step 13: Archive U.S. land-cover source evidence

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_nalcms_land_cover \
  --data-root data \
  --release united-states
```

You get one immutable CEC NALCMS United States source archive and provenance.
It is source evidence only; do not extract a national TIFF into `data/` or
average categorical class IDs. Leave about 1.7 GB free in `/tmp` for staging.

## Step 14: Audit the finished local package

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.inspect_storage_budget --data-root data
```

You get the final byte-level inventory. It must report no more than
20,000,000,000 bytes. Unused capacity is deliberate reserve for correctly
implemented paired Level-2 cutouts and issued-weather value tiles—not a reason
to add unpaired swaths or retrospective weather caches.

## What this supplies to training today

| Training role | Data now available | Status |
| --- | --- | --- |
| Dynamic fire state | All three FIRMS products, availability-gated at the feature cutoff | Ready input evidence |
| Spread target | FEDS 1 km, 12-hour perimeter-difference positives | Weak positive labels only |
| Static terrain | ETOPO elevation, slope, aspect sampled at the cell centre | Ready feature source |
| Training view | FEDS-positive rows joined to cutoff-safe FIRMS and terrain | Ready positive-only source view |
| Candidate dataset | FIRMS-supported 1 km candidate rows with terrain/FIRMS features | Ready no-weather weak-label baseline; target=0 is proxy only |
| Weather | HRDPS candidate/retrieval plan only | **No value data; no wind direction** |
| Negative/observation mask | Level-2 inventory only | Not available yet |
| Land cover/fuels | NALCMS source archives only | Not feature-ready yet |
| Final validation | WFIGS final/reference perimeters | Not a progression target |

The code contains the canonical grid, FIRMS feature builder, terrain sampler,
FEDS-label/positive-view assembler, candidate sampler/view publisher, release
exporter, and tabular baseline. The completed no-weather release is a research
baseline, not an operational predictor; see
[the training pipeline](training-pipeline.md) for the exact boundary.

## Retry rules

Rerun the same command after a transient failure. Raw evidence is immutable;
the append-only coverage ledger records complete, empty, partial, and failed
attempts. Do not delete files or edit coverage records by hand.

For FIRMS, retry only dates whose latest product/day coverage is `partial` or
`failed`; NRT responses can change. Use `--refresh` for FEDS only when a new
current provider view is intentionally wanted—the ordinary command is
resumable and skips terminal coverage.
