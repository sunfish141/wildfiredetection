# Wildfire Atlas

The local FastAPI app provides an interactive map for the retained incident
spread model. Place fires with a chosen intensity, or load current FIRMS
observations; play, pause, single-step, and inspect individual cells.

## Start

From the repository root:

```bash
.venv/bin/python -m pip install -r config/requirements.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src \
  .venv/bin/python -m uvicorn wildfire_data.web_app:app \
  --host 127.0.0.1 --port 8000
```

Open [localhost:8000](http://127.0.0.1:8000). The default model is pass 2 from
`artifacts/incident-two-pass-v1-201db0d293c56f51/run_manifest.json`. It loads
through the existing checksum-verified model loader. Retained ETOPO blocks
come from `data/`. Missing model artifacts produce an actionable page instead
of fabricated predictions. These ignored local artifacts must be supplied
separately on a fresh checkout.

Optional server settings:

| Variable | Purpose |
| --- | --- |
| `WILDFIRE_RUN_MANIFEST` | Completed, trusted local two-pass run manifest |
| `WILDFIRE_MODEL_PASS` | `pass_1` or `pass_2`; default `pass_2` |
| `WILDFIRE_DATA_ROOT` | Retained terrain directory; default repository `data/` |
| `NASA_FIRMS_API_KEY` or `MAP_KEY` | FIRMS key, read from the environment or `config/.env` |

This is a local, unauthenticated research application. Browser tabs hold
independent scenarios, and reloading the page clears that tab's history.

## Using the map

1. Set **Starting intensity**, choose **Place on map**, then click one or
   more points. Placement snaps to the canonical 1 km cell. Repeated points
   in one cell use their maximum intensity. Coordinates are also supported.
2. Alternatively, choose **FIRMS detections → Load current FIRMS**. The default
   fetch covers the notebook's full United States/Canada collection extent
   (`-179,24,-52,84`), independent of map zoom. **Visible map area** is optional.
   There is no zoom requirement or FIRMS cell-count cap. Loading replaces the
   scenario and fits the map to the loaded fires.
3. Press **Play**. The default is three seconds between completed 12-hour
   steps, plus inference time. Select 1, 3, 5, or 10 seconds, or use **+12 h**.
4. **Pause** immediately freezes the displayed state, including while a
   prediction request is in flight. Clicking any visible cell also pauses
   and opens the inspector. Space toggles playback when focus is outside
   form controls. Switching away from the tab pauses playback.
5. Drag the timeline to revisit the latest 128 completed steps. Future steps must first be
   simulated. Resuming from an earlier step replays saved states, then
   continues from the latest result. Reset clears the scenario for new seeds.

Active fire, burned cells, and optional spread candidates use separate map
layers. Inspect intensity, cell coordinates, source, and the last-step spread
probability when available; FIRMS cells also expose observed brightness and
detection count. Cell centroids are points, not precise fire perimeter edges.
Playback has no forecast-horizon limit: it continues until paused, including
after extinction (the empty fire stays extinct). The timeline expands as
steps complete and retains a rolling 128-frame window. Discarding old display
frames never removes burned cells from the current state or permits reignition.
The model's evaluation still covers 12–96 hours; longer playback does not
extend the evidence for predictive accuracy.

Large map views group nearby markers by screen position and state. Click a
group to expand it; individual cells remain inspectable at closer zooms.
Grouping affects display only. Inference retains every aggregated 1 km cell,
and panning redraws the visible cells without fetching FIRMS again.

## FIRMS semantics

The server requests two calendar days from the three NRT VIIRS area feeds,
then filters by the precise UTC acquisition window **3–24 hours before the
preview origin**. All three feeds must return valid CSV before replacing a
scenario. Authentication, connectivity, and malformed-feed failures are
errors, not empty-fire states. Successful header-only CSV is an empty feed.
The response reports how many newer detections were excluded by the model's
three-hour availability lag. Brightness is not threshold-filtered.

The key stays on the server and credential-bearing URLs are omitted from
client errors. Identical bounds reuse an in-memory preview for five minutes;
the displayed snapshot timestamp stays with that cached response. Requests
are serialized, CSV rows are streamed directly into per-cell aggregates, and
the cache holds at most 16 previews. There is no 5 MB response cutoff or
500-cell truncation. This viewer keeps observations transient and does not
write to the immutable collection or training archive. Use the existing
collector for durable source evidence.

The endpoint follows the [NASA FIRMS area API](https://firms.modaps.eosdis.nasa.gov/api/area/).
Map controls use vendored [Leaflet 1.9.4](https://leafletjs.com/reference.html),
with its license retained alongside the files. OpenStreetMap tiles and optional
Google Fonts need an internet connection; local scripts, API calls, and
coordinate placement work without either external asset service. The app
serves its assets using [FastAPI StaticFiles](https://fastapi.tiangolo.com/tutorial/static-files/).

## Model and API boundaries

The model remains a research preview: no weather inputs, heuristic persistence
and extinction, and mixed held-out rollout results. Scenario intensity is a
0–100% synthetic scale, not measured FRP. Using pass 2 in the interface does
not promote it or change any training artifact. The model uses its retained
threshold, probability calibration, adjacent-cell candidate bounds, growth
caps, and burned-cell masking unchanged. Missing terrain enters the model
through its existing missing-feature handling and is reported on the map.

| Endpoint | Behavior |
| --- | --- |
| `GET /api/config` | Model readiness, FIRMS availability, playback limits |
| `POST /api/seed` | Validate points/intensities and build a starting state |
| `POST /api/step` | Accept full state and UTC origin; return the next state, candidate probabilities, and inspectable points |
| `POST /api/firms` | Load the notebook's full collection area by default, or the requested rectangle |

State requests preserve the observed FIRMS aggregates needed for subsequent
steps. The server is stateless with respect to simulations: no background
clock mutates a session. The browser cancels and invalidates pending requests
on pause, reset, history navigation, and source changes. Serialized inference
protects shared terrain caches; model and terrain work run outside the ASGI
event loop.

## Validation

All 233 repository tests passed. Browser checks passed with the retained
model, including a delayed request during pause, actual inference to 144 hours,
rolling-history navigation, the full-region request, and display grouping.
A fresh full-region FIRMS request on September 5, 2026 completed in 1.92 seconds:
2,445 eligible detections combined into 1,006 active cells, with 591 newer
detections excluded by the existing availability policy. The response was
560,524 bytes. The measured fetch is recorded in
`artifacts/web-preview/full-firms-verification.json`.

Run the repository tests with:

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python -m unittest discover -s tests
```

Optional browser checks require Playwright and Chromium:

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install chromium
# If your OS lacks browser libraries: python -m playwright install-deps chromium
.venv/bin/python tests/browser_web_app.py
```

The browser script uses the running server for model inference and a fixture
for the FIRMS UI response. It checks map placement, intensity inspection,
single-step, history, pause during an in-flight request, resume, and mobile
layout, and writes screenshots to `artifacts/web-preview/`. Provider parsing,
age eligibility, aggregation, and failures have separate offline unit tests.
Regression coverage includes 2,000 FIRMS cells, input states and burned masks
exceeding 1,600 cells, continuation beyond 96 hours, empty-state continuation,
the growing timeline, and rolling history.
