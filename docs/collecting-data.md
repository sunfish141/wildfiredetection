# Re-collect the wildfire dataset

Run these commands from the repository root, one at a time. This is the exact order for rebuilding the current U.S./Canada dataset for **2026-05-31 through 2026-08-10**.

For a different collection period, replace both dates everywhere they appear. Do not run two archive-writing commands at the same time against `data/`.

## One-time setup

Create the environment if it does not already exist:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r config/requirements.txt
```

Create `config/.env` with your FIRMS API key:

```text
NASA_FIRMS_API_KEY=replace-with-your-key
```

## Step 1 — Check available storage

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.inspect_storage_budget --data-root data
```

You get a report of every file already under `data/`, its storage category, and remaining capacity. It also writes `data/retention/storage_budget.csv`.

This command downloads nothing. Do not delete old CSVs or raw files to make space; the collectors enforce the 20 GB policy.

## Step 2 — Collect raw FIRMS fire detections

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_firms \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --product VIIRS_SNPP_NRT \
  --product VIIRS_NOAA20_NRT \
  --product VIIRS_NOAA21_NRT
```

You get:

- Every FIRMS response for every requested day and satellite product, including low-brightness and duplicate detections.
- Lossless normalized detection rows: latitude, longitude, acquisition time, brightness, FRP, confidence, scan/track footprint, day/night, and all other provider fields.
- A coverage record for every product/day, including explicit empty, partial, and failed responses.

The notebook’s TI4 brightness threshold is **not** applied to this source archive. The command is complete only when its final line reports `0 coverage windows need retry.`

## Step 3 — Collect U.S. reference perimeters

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_wfigs \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --archive-root data
```

You get paginated WFIGS GeoJSON and normalized U.S. perimeter geometries, incident identifiers, timestamps, acreage, and source fields.

These are **final/reference perimeters** for incident matching and validation. They are not a historical sequence of where a fire spread at each hour or day.

## Step 4 — Collect Canadian incident record history

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_cwfis_active_fires \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --archive-root data
```

You get CWFIS active-fire record versions: agency incident IDs, status, reported size, point geometry, and `record_start`/`record_end` timing.

This is Canadian incident context. It is **not** a fire perimeter and does not create a `newly_burned` spread label.

## Step 5 — Save the VIIRS Level-2 inventory

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_viirs_l2 \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --platform snpp \
  --platform noaa20 \
  --platform noaa21 \
  --dry-run
```

You get CMR inventory evidence for the SNPP, NOAA-20, and NOAA-21 active-fire Level-2 files: granule identity, time, footprint, version, and source metadata.

You do **not** get Level-2 pixels yet. In particular, this step does not download fire mask/QA arrays or their matching geolocation arrays. That paired-cutout collector is not implemented yet.

Never run `--legacy-fire-files-only` for this 20 GB dataset. It downloads unpaired fire files, cannot provide geolocated no-fire coverage, and does not fit the storage policy.

## Step 6 — Create the weather retrieval plan

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.plan_forecast_tiles \
  --start 2026-07-17 \
  --end 2026-08-10 \
  --model hrdps \
  --data-root data
```

You get `data/weather/forecast-tile-plans/hrdps_20260717_20260810.csv.gz`: scored 96 km fire-context tile candidates, selected/capped status, and `fire_evidence_score`, `forecast_availability_score`, and `retention_priority_score`.

This is a **plan**, not weather data. It contains no temperature, humidity, wind, precipitation, or gust values. Do not join it to a model as weather.

## Step 7 — Collect terrain for each fire context

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_static_terrain \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --data-root data
```

You get immutable NOAA ETOPO source subsets, normalized source metadata, and compressed terrain blocks in `data/static/etopo-2022-15s/`.

Each terrain block provides:

- `elevation_m`
- `slope_degrees_x2` — slope in 0.5° increments
- `aspect_degrees_x2` — downhill aspect in 2° increments

The terrain source is a WGS84 15-arc-second grid. It is useful static context, but it is not yet the final model cell grid.

## Step 8 — Archive Canada land-cover source data

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_nalcms_land_cover \
  --data-root data \
  --release canada
```

You get one immutable, content-addressed copy of the public CEC NALCMS Canada source release plus its provenance and coverage record. The source represents Canada in 2020.

The download stages temporarily in `/tmp`, so leave roughly 2 GB free there. The temporary file is removed when archival succeeds.

## Step 9 — Archive U.S. land-cover source data

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.collect_nalcms_land_cover \
  --data-root data \
  --release united-states
```

You get one immutable, content-addressed copy of the public CEC NALCMS U.S. source release plus its provenance and coverage record. Its component years are CONUS 2019 and Alaska 2021.

The download stages temporarily in `/tmp`, so leave roughly 1.7 GB free there. The temporary file is removed when archival succeeds.

Steps 8 and 9 preserve the original 30 m categorical source data for later processing. They do **not** create model-ready land-cover features. Do not extract national TIFFs into `data/` or average land-cover class IDs; a future compactor must use an explicit target grid and class mode/fraction rule.

## Step 10 — Verify the finished dataset

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.inspect_storage_budget --data-root data
```

You get the final storage inventory. For the currently collected 2026-05-31 through 2026-08-10 package, the expected total is about **4.965 GB of the 20 GB cap**.

The unused allocation is deliberate. It is reserved for correctly implemented paired Level-2 cutouts and issued forecast-value tiles; it is not filled with unpaired satellite files, retrospective weather, or invented perimeter history.

## If a command needs retrying

Run the same command again. Successful raw artifacts remain immutable, completed WFIGS/CWFIS/NALCMS scopes are skipped, and every failed or partial attempt stays in the coverage ledger for audit.

Do not delete source files or edit coverage records by hand. If a FIRMS run reports coverage windows needing retry, rerun its exact date range first; then retry a single date only if necessary.

## Do not treat these as training inputs yet

- Notebook `open_meteo_weather_*.csv` files are visualization caches, not issued-at forecasts.
- The HRDPS CSV is a retrieval plan, not weather values.
- VIIRS Level-2 inventory records are not cloud/no-fire observation pixels.
- WFIGS is final/reference geometry, not fire progression history.
- CWFIS active-fire records are incident context, not perimeter labels.
- NALCMS archives are retained source rasters, not model-ready categorical features.
