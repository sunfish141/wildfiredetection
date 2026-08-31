# Re-collect the compact wildfire dataset

Run these commands from the repository root, one at a time. The checked-in
commands below retain the older U.S./Canada **2026-05-31 through 2026-08-10**
release as reproducible historical evidence. The completed active POC range is
**2026-05-11 through 2026-08-22**; its exact bounded chunk-and-merge build is
in [the no-weather POC guide](no-weather-poc.md). Every archive-writing command
writes under `data/`, so do not run two of them concurrently.

The active May 11–Aug 22 rebuild is a **no-weather proof of concept**. Collect
the non-weather evidence first and do not run Step 11b until the coherent CSV
release and tabular baseline have been verified. The exact POC source
boundaries, export contract, and training step are in [the no-weather POC guide](no-weather-poc.md).

The package has a hard **20,000,000,000-byte** limit, including every existing
retained artifact. The collectors preserve existing evidence and
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

## Step 9: Optionally capture a forward Open-Meteo forecast run

Forward issued-forecast research remains optional. Use
`wildfire_firms_analysis.ipynb` while the forecast is operational, set the
capture cell's explicit `model` and exact UTC `model_run_at`, review the
planned candidate-cell tiles, then enable the capture flag. The Single Runs
capture stores an immutable response before normalizing it and retains the
model/run, returned grid location, valid time, raw artifact ID, and timestamp
at which that exact response was successfully received. Only values valid
after that captured availability time can pass an operational as-of join.

Keep the rate limit enabled: the default is 600 location units per minute.
Transient errors retry; HTTP 429 waits at least 90 seconds or a longer
`Retry-After`; two consecutive 429 responses pause the run rather than
retrying indefinitely. Already archived batches and the final 429 response
remain immutable, with the latter recorded as failed coverage. A later
continuation starts a new immutable attempt; it is not a mutable cache resume.
The historical backfill in Step 11b uses the same pacing and pause behaviour,
but its explicit partial-manifest resume reuses already complete dates in a new
immutable backfill manifest.

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

## Step 11a: Legacy candidate-view command

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
The command below reproduces the legacy no-weather release. The completed
active POC uses the seven-day chunk/merge process in the POC guide instead.

Export the selected view outside `data/` so the portable copy does not consume
the 20 GB archive budget:

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.export_candidate_dataset \
  --data-root data \
  --candidate-manifest data/manifests/candidate-dataset-builds/2026/08/24/040059951477_76e075f405f24287ba9531bea436cbe5.json \
  --output releases/wildfire-spread-firms-feds-no-weather-2026-05-31_to_2026-08-10
```

The legacy export has 305,528 candidate rows, 11,848 unscored positives,
JSON Lines gzip files, a schema, manifest, inventory, and SHA-256 checksums.
The completed May 11–Aug 22 POC instead has 428,656 candidate rows, 12,924
unscored positives, and schema-v2 CSV/JSONL at
`releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22/`.
Run `notebooks/train_tabular_baseline.ipynb` against that verified release.

## Step 11b: Later only — backfill historical weather and publish a weather-bearing view

After the no-weather POC is verified, a separate weather-bearing experiment
may pass the completed candidate manifest from Step 11a as its immutable spine.
`open_meteo_historical.py`
records that manifest's path, build ID, and content hash, then derives
compact FIRMS-seeded weather tiles from that view and calls the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
with `models=ecmwf_ifs` and `timezone=UTC`, one candidate date at a time. It
requests hourly `temperature_2m`, `relative_humidity_2m`, `precipitation`,
`weather_code`, `wind_speed_10m`, and `wind_direction_10m`. Batched locations
are supported, but every location counts toward rate-limit pacing.

The default compact-cover distance is 10 km: a request tile may be up to that
far from a 1 km candidate centre, followed by Open-Meteo's own grid snap. The
mapping retains the candidate-to-request distance and both locations; lower
the distance only when the additional rate-limited requests are justified.

For each candidate row, `floor_weather_hour` derives the UTC weather hour at
or before its prediction cutoff; the joined row records it as
`weather_observed_at`. The collector archives the raw response, normalizes its
values, and persists a candidate-cell-to-weather-tile mapping with FIRMS
raw-artifact and example lineage. It includes target=0 `weak_negative_proxy`
cells. Wind direction is converted to U/V components for the model feature
set.

`backfill_open_meteo_historical_weather` writes an immutable per-date
backfill manifest. If the rate limiter or storage budget pauses it, the
manifest is partial; pass it as `--resume-manifest` with the same base candidate
manifest to reuse completed dates and retry the first unfinished date rather
than assembling a partially weathered dataset.
It retains the same 600-location-unit-per-minute pacing, `Retry-After`
cooldown, retry, and two-consecutive-429 pause policy as Step 9.
Only a complete backfill manifest may be passed to
`weather_candidate_dataset.build_weather_candidate_dataset`, which joins the
exact tile and hourly anchor to the same completed base candidate view (its
identity must match the backfill manifest) and publishes
a distinct immutable weather-bearing candidate view. Use
`export_weather_candidate_dataset_release` to create its upload directory;
the base no-weather view is never mutated.

```bash
# Substitute the completed manifest printed by Step 11a.
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_historical_weather \
  --data-root data \
  --candidate-manifest data/manifests/candidate-dataset-builds/<base>.json \
  --start 2026-05-11 \
  --end 2026-08-22

# If that command reports a partial backfill, keep the same base manifest and
# continue with the partial manifest it printed.
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_historical_weather \
  --data-root data \
  --candidate-manifest data/manifests/candidate-dataset-builds/<base>.json \
  --resume-manifest data/manifests/open-meteo-historical-weather-backfills/<partial>.json

PYTHONPATH=src .venv/bin/python -m wildfire_data.build_weather_candidate_dataset \
  --data-root data \
  --candidate-manifest data/manifests/candidate-dataset-builds/<base>.json \
  --weather-backfill-manifest data/manifests/open-meteo-historical-weather-backfills/<complete>.json

PYTHONPATH=src .venv/bin/python -m wildfire_data.export_weather_candidate_dataset \
  --data-root data \
  --weather-candidate-manifest data/manifests/weather-candidate-dataset-builds/<weather>.json \
  --output releases/wildfire-spread-firms-feds-weather-2026-05-11_to_2026-08-22
```

This is a **retrospective weather-analysis** feature. It describes historical
conditions for offline training analysis, not the forecast an operator had at
the cutoff. Mark it `historical_analysis`; do not use it for an operational
as-of claim.

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
implemented paired Level-2 cutouts and compact weather tiles—not a reason to
add unpaired swaths or unproven, unmapped weather data.

## What this supplies to training today

| Training role | Data now available | Status |
| --- | --- | --- |
| Dynamic fire state | All three FIRMS products, availability-gated at the feature cutoff | Ready input evidence |
| Spread target | FEDS 1 km, 12-hour perimeter-difference positives | Weak positive labels only |
| Static terrain | ETOPO elevation, slope, aspect sampled at the cell centre | Ready feature source |
| Training view | FEDS-positive rows joined to cutoff-safe FIRMS and terrain | Ready positive-only source view |
| Candidate dataset | FIRMS-supported 1 km candidate rows with terrain/FIRMS features | Completed no-weather weak-label baseline is a past release; target=0 is proxy only |
| Weather | Open-Meteo Historical Weather API ECMWF IFS backfill at candidate tiles/hourly anchors; optional forward Single Runs | Historical analysis is the next training feature path; issued-forecast availability requirements apply only to the optional forward mode |
| Negative/observation mask | Level-2 inventory only | Not available yet |
| Land cover/fuels | NALCMS source archives only | Not feature-ready yet |
| Final validation | WFIGS final/reference perimeters | Not a progression target |

The code contains the canonical grid, FIRMS feature builder, terrain sampler,
FEDS-label/positive-view assembler, candidate sampler/view publisher, release
exporter, and tabular baseline. The completed no-weather release is a past
research baseline, not an operational predictor; see
[the training pipeline](training-pipeline.md) for the exact boundary.

## Retry rules

Rerun the same command after a transient failure. Raw evidence is immutable;
the append-only coverage ledger records complete, empty, partial, and failed
attempts. Do not delete files or edit coverage records by hand.

For FIRMS, retry only dates whose latest product/day coverage is `partial` or
`failed`; NRT responses can change. Use `--refresh` for FEDS only when a new
current provider view is intentionally wanted—the ordinary command is
resumable and skips terminal coverage.
