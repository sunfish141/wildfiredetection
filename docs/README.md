# Wildfire Detection Starter

The interactive **Wildfire Atlas** FastAPI app supports fire placement,
current FIRMS loading, adjustable playback, and point inspection.
See [Web app setup and controls](web-app.md) to run it locally.

This workspace contains a Jupyter notebook for pulling near-real-time fire detections from the NASA FIRMS API. It also defines two weather paths: a retrospective Open-Meteo Historical Weather API backfill for training analysis and an optional live Open-Meteo Single Runs capture for issued-forecast research. The active proof of concept deliberately defers both weather paths. Each successful FIRMS collection archives the exact unfiltered response and writes a lossless normalized record set. The source archive retains every FIRMS field, so a later model can choose a different threshold or feature set without recollecting.

## Spread-forecasting data contract

The notebook is a collection tool; it does not itself produce model labels or
predictions. The repository also contains a manifest-selected FIRMS/FEDS
candidate-table builder and an uploadable release exporter. The active proof
of concept has rebuilt the coherent **no-weather** range 2026-05-11 through
2026-08-22 and fitted its first baseline. A completed
release manifest—not the presence of loose artifacts—establishes that this
range is ready.
The U.S./Canada collection and training contract is documented in:

- [Architecture and collection contract](architecture.md)
- [Feature and label map](feature-map.md)
- [Collection runbook](collecting-data.md)
- [First tabular training pipeline](training-pipeline.md)
- [Uploadable candidate dataset](uploadable-dataset.md)
- [No-weather May 11–Aug 22 proof of concept](no-weather-poc.md)
- [Pipeline handoff](handoff.md)
- [Incident sequences and two-pass scheduled sampling](incident-scheduled-sampling.md)
- [20 GB local-dataset decision](adr/0001-bounded-20gb-local-dataset.md)
- [1 km / 12-hour weak-label-baseline decision](adr/0002-first-1km-12hour-weak-label-baseline.md)
- [Change log](change-log.md)

The local collection root is `data/` (ignored by Git): immutable provider bytes live under `data/raw/`, lossless normalized JSON Lines under `data/normalized/`, and append-only raw/coverage manifests under `data/manifests/`. The notebook and `collect_firms` implement this path for FIRMS.

The local package has a hard **20,000,000,000-byte** limit, including every
pre-existing retained file. It is therefore a compact training package,
not a complete native archive. It currently contains unfiltered FIRMS
evidence, FEDS satellite-weak perimeter snapshots/derived positives, WFIGS
reference perimeters, CWFIS active-fire record history, Collection 2 VIIRS
inventory evidence, an HRDPS retrieval plan, compact ETOPO terrain blocks, and
one immutable CEC NALCMS land-cover source ZIP for each of Canada and the
United States. Those NALCMS ZIPs are source evidence, not model-ready fuel
features. Paired VIIRS pixels and issued forecast values remain
reserved-but-uncollected allocations. The policy and per-category usage report
are [documented in ADR 0001](adr/0001-bounded-20gb-local-dataset.md) and
enforced before compact collectors persist new artifacts. Existing files are
counted and never silently evicted.

The current L2 command is deliberately **inventory-only by default** under this policy. `VNP14IMG`/`VJ114IMG`/`VJ214IMG` provide fire mask and algorithm QA; matching `VNP03IMG`/`VJ103IMG`/`VJ203IMG` products provide geolocation. Downloading the former alone is not a complete observation and requires an explicit legacy override outside the 20 GB policy.

The existing HRDPS plan is still only a bounded candidate plan, not weather
measurements. The completed 2026-05-11 through 2026-08-22 POC release and
baseline deliberately keep weather out.

## Historical weather backfill and optional forward capture

Weather is deferred from the active 2026-05-11 through 2026-08-22 proof of
concept. Do not invoke either weather collector or include weather features in
that POC's CSV release or baseline. The contract below applies only to a later,
separate weather-bearing experiment after the no-weather release is verified.

For a later weather-bearing experiment derived from the completed POC candidate
view, backfill hourly weather from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
with the pinned ECMWF IFS model (`ecmwf_ifs`). Pass one completed base candidate
manifest to the backfill; it records that manifest's path, build ID, and content
hash before requesting each retained tile for the range needed by its candidate
rows. Join a tile value only to the row's deterministic UTC hourly weather
anchor: floor the row's prediction cutoff to the start of its hour, never
round into a later hour. The tile mapping must cover all selected candidate
cells, including `weak_negative_proxy` target=0 cells, not just detection
points or positive labels. The default compact cover may place a request up to
10 km from a 1 km candidate centre before Open-Meteo snaps it to its weather
grid; the candidate-to-request distance and returned grid location are retained.

This is a **retrospective weather-analysis** feature: it describes conditions
for offline training analysis after the event. Retain the model, requested
tile, raw response, retrieval time, valid hour, and time-alignment rule, and
label the feature mode `historical_analysis`. It must never be presented as a
reconstructed issued forecast or as evidence that the weather value was known
at the historical prediction cutoff.

`src/wildfire_data/open_meteo_single_run.py` remains available for optional
forward forecast capture. Before enabling it, the operator
must set one explicit Open-Meteo model and exact model-run timestamp in UTC.
The collector requests that single run rather than an undated latest-forecast
view.

The notebook expands newly archived FIRMS evidence into the same candidate
cells used by the candidate policy, including cells that later become
`weak_negative_proxy` target=0 rows. It deterministically assigns those cells
to bounded forecast tiles, saves the assignment mapping, and requests only
forecast times after the successful response's captured availability time.
Each admitted response is immutable raw evidence; normalized measurements keep
the provider/model, requested run, returned grid/tile, valid time, capture
availability time, raw-artifact ID, and mapping lineage. The mapping records
the FIRMS detection IDs, source raw-artifact IDs, and latest acquisition time
that seeded each candidate cell. A model run timestamp is not treated as proof
that the run was already available: the successful captured response supplies
that availability evidence.

The rate limit is intentionally retained for both weather paths. The default
pacing is 600 location units per minute, normal transient failures are
retried, and HTTP 429 waits at least 90 seconds or the longer `Retry-After`
interval. After two consecutive 429 responses, the run pauses gracefully so
already archived batches remain available and the final 429 response is
recorded as failed coverage. Pass a historical backfill's partial manifest as
`--resume-manifest` to publish a new manifest that reuses completed dates and
retries the unfinished date; forward Single Runs remain separate immutable
capture attempts.

The active FIRMS/FEDS rebuild spans **2026-05-11 through 2026-08-22** and is
explicitly no-weather. Only a later, separately published candidate view may
include the retrospective ECMWF IFS weather-analysis features above. The
separately captured Single Runs artifacts remain the only weather source
eligible for an operational, issued-forecast as-of experiment.

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

For the safe VIIRS L2 inventory command, follow [Step 8 of the collection runbook](collecting-data.md#step-8-save-the-viirs-level-2-inventory).

For terrain, positive-only training-view, and bounded land-cover source
collection, follow [Step 10](collecting-data.md#step-10-collect-terrain-source-blocks),
[Step 11](collecting-data.md#step-11-build-the-positive-only-tabular-training-view),
[Step 12](collecting-data.md#step-12-archive-canada-land-cover-source-evidence),
and [Step 13](collecting-data.md#step-13-archive-us-land-cover-source-evidence).
