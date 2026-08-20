# Wildfire Detection Starter

This workspace contains a Jupyter notebook for pulling near-real-time fire detections from the NASA FIRMS API and visualizing them on a scatter plot. Each successful notebook collection now archives the exact unfiltered FIRMS response and writes a lossless normalized record set before creating the filtered weather-enrichment view.

The notebook excludes FIRMS detections with TI4 brightness below 305 only from the weather/map view. The raw archive retains those rows and every other FIRMS field, so a later model can use a different threshold or feature set without recollecting.

## Spread-forecasting data contract

The current notebook is a fire-detection and weather-enrichment prototype; it does not yet produce fire-spread labels or predictions. The proposed U.S. and Canada collection, provenance, feature, and label contract is documented in:

- [Architecture and collection contract](architecture.md)
- [Feature and label map](feature-map.md)
- [Collection runbook](collecting-data.md)
- [20 GB local-dataset decision](adr/0001-bounded-20gb-local-dataset.md)
- [Change log](change-log.md)

The local collection root is `data/` (ignored by Git): immutable provider bytes live under `data/raw/`, lossless normalized JSON Lines under `data/normalized/`, and append-only raw/coverage manifests under `data/manifests/`. The notebook and `collect_firms` implement this path for FIRMS.

The local package has a hard **20,000,000,000-byte** limit, including all existing CSV exports and caches. It is therefore a compact training package, not a complete native archive. It currently contains unfiltered FIRMS evidence, WFIGS reference perimeters, CWFIS active-fire record history, Collection 2 VIIRS inventory evidence, an HRDPS retrieval plan, compact ETOPO terrain blocks, and one immutable CEC NALCMS land-cover source ZIP for each of Canada and the United States. Those NALCMS ZIPs are source evidence, not model-ready fuel features. Paired VIIRS pixels and issued forecast values remain reserved-but-uncollected allocations. The policy and per-category usage report are [documented in ADR 0001](adr/0001-bounded-20gb-local-dataset.md) and enforced before compact collectors persist new artifacts. Existing files are counted and never silently evicted.

The current L2 command is deliberately **inventory-only by default** under this policy. `VNP14IMG`/`VJ114IMG`/`VJ214IMG` provide fire mask and algorithm QA; matching `VNP03IMG`/`VJ103IMG`/`VJ203IMG` products provide geolocation. Downloading the former alone is not a complete observation and requires an explicit legacy override outside the 20 GB policy.

`open_meteo_weather_*.csv` remains a notebook cache/export for visualization. It is not a reproducible issued-at forecast archive; use `src/wildfire_data/forecast_weather.py` records for operational forecast data.

The repository now also stores a compact HRDPS candidate plan for 2026-07-17 through 2026-08-10. It contains scored, selected and capped fire-context tiles—not weather values—and keeps the historical forecast-publication uncertainty visible until a value-tile extractor is used.

## Weather lookup behaviour

The notebook selects a deterministic subset of input fire locations that keeps every detection within 2 km of a selected weather source. It converts coordinates to WGS84 Earth-Centered, Earth-Fixed (ECEF) metres and uses a KD-tree to identify nearby locations. This bounded-time selection does not attempt a globally minimum source count.

The notebook first displays the consolidated selected-source table and saves the planned source/hour lookups to `data/weather/open_meteo_weather_requests.csv`, without making a network request. It keeps a separate fire-to-source mapping so weather can later be attached to every original detection. Its following cell fetches weather once per selected source and UTC day, rather than once per fire hour.

Each successful batch is atomically checkpointed to `data/weather/open_meteo_weather_cache.csv`, so rerunning after an interruption fetches only missing lookups. The lookup paces individual source/day calls at 600 per minute. If Open-Meteo returns HTTP 429, it waits at least 90 seconds (or longer when instructed by `Retry-After`) and retries the same batch. When complete, the notebook saves the enriched detections to a date-labelled file under `data/exports/`, such as `fires_with_weather_2026-07-26_to_2026-07-29.csv`; this label always represents the queried coverage window, even if no fire occurs on a boundary date. Rows are ordered by oldest acquisition time and then nearest weather source.

After two consecutive HTTP 429 responses for a batch, the weather fetch pauses gracefully instead of retrying indefinitely. Completed batches remain checkpointed, the FIRMS range is not advanced, and the notebook's `weather-resume` cell resumes only the remaining lookups.

Before each weather fetch, the cache is rewritten to retain only entries belonging to the current FIRMS range; the request manifest is overwritten too. This bounds both `open_meteo_*.csv` files to the active range while still allowing an interrupted range to resume.

FIRMS collection ranges persist in `data/state/firms_collection_range_state.json`. The notebook collects four inclusive days at a time and moves backward from the last successfully exported range; date arithmetic continues correctly across months and years. If no state exists yet, it seeds the next range from the oldest date-labelled weather export.

## Setup

1. Install the Python dependencies:
   ```bash
   python3 -m pip install -r config/requirements.txt
   ```
2. Create `config/.env` with your NASA FIRMS API key:
   ```env
   NASA_FIRMS_API_KEY=your_key_here
   ```
3. Open and run the notebook: `wildfire_firms_analysis.ipynb`

To collect a durable FIRMS range without opening the notebook, run:

```bash
PYTHONPATH=src python3 -m wildfire_data.collect_firms --start 2026-07-01 --end 2026-07-31
```

It reads `NASA_FIRMS_API_KEY` (or the legacy `MAP_KEY`) from the environment, saves raw/normalized evidence under `data/`, and records coverage failures for retry.

To inspect the whole-data budget, run:

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.inspect_storage_budget --data-root data
```

For the compact paired-L2 plan and safe inventory command, follow [Step 9 of the collection runbook](collecting-data.md#9-inventory-viirs-level-2-and-plan-paired-cutouts).

For terrain and bounded land-cover source collection, follow [Step 12](collecting-data.md#12-collect-static-terrain-for-the-firms-context) and [Step 13](collecting-data.md#13-archive-nalcms-land-cover-source-evidence).
