# Feature and label map

## Interactive spread explorer

The [FastAPI web app](web-app.md) maps user-placed intensity seeds or current
FIRMS observations onto the existing 1 km grid. It provides 12-hour model
steps, adjustable play/pause, saved-step inspection, and active/burned/candidate
map layers. The inspector shows state and available probabilities for each
cell. FIRMS loading preserves observed aggregates and the 3–24-hour eligibility
window. Full notebook-region FIRMS loading works at any zoom; dense map
markers group for display while inference retains the individual 1 km cells.
Playback has no time limit, with a rolling 128-step inspection timeline and
the full current burned mask. No new model features or training labels are introduced.

## First-model contract

Each row represents one canonical 1 km North America Albers equal-area cell
(ESRI:102008) at a real UTC feature cutoff. Its target is whether the cell is
observed to newly enter a FEDS perimeter in the following 12 hours. Output is a
probability plus that cell's centroid latitude/longitude—not a direct
regression to an arbitrary point.

The first scope is CONUS + Canada. FEDS has a local-solar source-time
convention, so each FEDS-derived row retains its native time and the documented
cell-local-solar-to-UTC anchor estimate. Alaska is excluded until that
alignment is separately validated.

| Family | Data used now | Eligibility at the cutoff | Important limit |
| --- | --- | --- | --- |
| Fire state | Unfiltered FIRMS detections: platform, acquisition time, TI4/TI5, FRP, confidence, scan/track, day/night, and source fields | Acquisition time plus the declared availability lag must be no later than the cutoff | Keep all raw detections; the 305 K TI4 rule is a downstream field, not a collection filter. |
| Static terrain | Retained ETOPO elevation, slope, and downhill aspect sampled at the 1 km cell centre | Static source version retained for the package | It is a sampled source pixel, not a dense 1 km terrain cache. |
| Weak spread target | Consecutive FEDS perimeter snapshots plus FIRMS candidate support | Source snapshots t and t + 12 h must both be retained and paired for the same fire | Positive cells are `weak_satellite`; target=0 is only a FIRMS-seeded `weak_negative_proxy`, never confirmed clear/no-burn. |
| Weather | Historical Open-Meteo API, ECMWF IFS (`ecmwf_ifs`), at FIRMS-seeded candidate tiles; optional forward Open-Meteo Single Runs | Historical-analysis values must match the candidate tile and a deterministic UTC hourly anchor at or before the cutoff; issued-forecast values additionally require model/run and captured availability | The completed release has no weather values. Historical values are retrospective analysis, not reconstructed issued forecasts. A mapping must cover all selected candidate cells, including target=0 weak-negative proxies. |
| Observation coverage | VIIRS Level-2 inventory | None: paired fire-mask/QA + geolocation cutouts are not stored | Cannot prove clear/no-fire or produce coverage-aware negatives yet. |
| Land cover/fuels | Canada/U.S. NALCMS source archives | None: no categorical aggregation is implemented | Never average land-cover class IDs. |
| Incident/reference | CWFIS incident context and WFIGS final/reference perimeters | Only source facts available by cutoff | They validate/match context; they are not 12-hour spread targets. |

## Label tiers

| Label tier | Definition | Use | Restriction |
| --- | --- | --- | --- |
| Operational progression | Difference between time-stamped agency perimeter versions | Preferred future supervised target | Not available in the current backfill. |
| Satellite weak progression | FEDS perimeter(t + 12 h) minus FEDS perimeter(t), rasterized to 1 km cells | First tabular-baseline positives | Keep source IDs/time alignment and weak_satellite tier. Never infer zeros from absent cells. |
| Burn-date raster | MODIS/VIIRS burn date with uncertainty/QA | Future historical weak labels/coverage | Retrospective, not operational. |
| Final extent | WFIGS/IFPH, MTBS, NFDB, NBAC, certified final perimeter | End-state validation/reference | Do not reconstruct spread timing from a final polygon. |

## Current FIRMS feature fields

The first feature builder produces, separately for the target cell and its
3 km × 3 km neighbourhood:

- detection present/count;
- maximum and mean TI4 brightness;
- number of platforms;
- hours since the most recent eligible detection; and
- active-cell count for the neighbourhood.

Each record also preserves feature-build version, cutoff, lookback start,
availability lag, and latest eligible acquisition time. It must not use
ingested-at or future FEDS information.

## Terrain feature fields

The terrain sampler returns:

- terrain-valid and coverage status;
- elevation in metres;
- slope in degrees;
- downhill aspect encoded as sine/cosine plus an aspect-defined flag; and
- source-block/pixel/sampling provenance.

## Minimum training-row lineage

Every persisted training candidate must include:

- example ID, cell ID, cell-centre latitude/longitude, cutoff, horizon, and
  target-end time;
- upstream raw/normalized IDs and transformation versions;
- feature availability policy and observation/label-coverage status;
- label value, tier, source time interval, time-alignment method, and quality
  flags;
- deterministic candidate/negative-selection reason; and
- split/model/feature versions once fitted.

For a weather-bearing row, also retain the returned tile/grid, raw-response
artifact ID, candidate-cell-to-tile mapping, valid hour, and feature mode. A
`historical_analysis` row records `ecmwf_ifs`, retrieval time, request range,
and the deterministic hourly weather-anchor rule; it is valid only when its
tile and hour match the candidate row. A separate issued-forecast row records
the selected model/run, captured availability timestamp and basis, and the
result of the operational as-of availability decision.

## Current implementation boundary

The canonical grid, FEDS collection/positive-label builder, leakage-gated
FIRMS feature builder, ETOPO sampler, positive-only training-table builder,
chronological tabular baseline, FIRMS-only candidate sampler, candidate-view
publisher, and release exporter are implemented. The completed candidate view
makes target=0 rows only as named weak-negative proxies and preserves positives
without FIRMS candidate support as unscored diagnostics. The completed active
range is 2026-05-11 through 2026-08-22. Even with the completed wiring:

- no absent FEDS label may become a zero;
- no unarchived or unversioned weather lookup may become a weather feature;
  the contracted Historical Weather API/ECMWF IFS path is permitted only as
  explicitly labelled retrospective analysis, never as an issued forecast;
- no WFIGS final perimeter may become earlier fire state; and
- the completed release must be described as a no-weather, satellite-weak
  baseline rather than an operational wildfire-spread predictor; a later
  historical-weather model remains retrospective analysis, not an issued-
  forecast model.

The active 2026-05-11 through 2026-08-22 proof of concept deliberately keeps
weather unavailable while it rebuilds a coherent CSV candidate release and
fits the first baseline. Its required FIRMS/FEDS boundaries, CSV contract,
label policy, and notebook workflow are in [the no-weather POC guide](no-weather-poc.md).

See [the training pipeline](training-pipeline.md) and
[ADR 0002](adr/0002-first-1km-12hour-weak-label-baseline.md) for the fixed
first-model choices.
