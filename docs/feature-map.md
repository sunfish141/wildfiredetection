# Feature and label map

## Contract

Each training example represents an incident-local area at an as-of time t and a forecast horizon h. Its output is a probability that each candidate cell newly burns during the interval from t to t+h.

The table below describes required source categories. It deliberately separates input evidence from labels and final validation data.

| Family | Raw and normalized content | Eligibility at time t | Quality requirement |
| --- | --- | --- | --- |
| Fire state | Unfiltered FIRMS active-fire records; platform, acquisition time, TI4/TI5, FRP, confidence, scan/track, day/night, and source quality. Operational perimeter versions and incident links when available. | Only records published by t. | Keep all raw detections; any TI4 threshold is a downstream policy, not a collection filter. Preserve duplicate, missing, and low-confidence evidence. |
| Weather and fire environment | Temperature, humidity, precipitation, wind U/V direction and speed, gusts, antecedent moisture, drought/fire-weather indices, and forecast model/run metadata. | Observed variables must be available by t; horizon variables must come from a run issued by t. | Store valid time separately from issue time. The current HRDPS candidate CSV is a scored retrieval plan, not weather values; its historical publication timing is explicitly uncertain. |
| Static landscape | ETOPO 2022 v1 source subsets plus compact elevation, slope, and aspect; versioned fuel/land-cover, canopy, water, roads, and built-area sources. NALCMS is currently retained as raw source evidence only, not feature values. | Static version valid at t. | Version rasters and record spatial resolution, CRS, source date, and resampling method. Never average categorical class IDs during aggregation. |
| Observation coverage | Satellite swath/overpass, cloud or smoke mask, valid-pixel status, source latency, and data outages. | Coverage information known by t or for the label observation interval. | Under the 20 GB policy, retain selected *paired* SNPP/NOAA-20/NOAA-21 fire-mask/QA and geolocation cutouts, not complete swaths. Distinguish no observation from observed-not-burning when a later derived mask is built. |
| Incident operations | Incident ID, discovery time, agency, fire status, size, containment/control status, map method, and suppression context when available. | Operational report was published by t. | Preserve native IDs and source update times; do not backfill final incident values into earlier samples. |

## Label map

| Label tier | Definition | Primary use | Caveat |
| --- | --- | --- | --- |
| Operational progression | Difference between two time-stamped agency perimeter versions for the same incident. | Preferred supervised spread label. | Mapping time and geometry quality vary; retain map method and capture time. |
| Satellite progression | New FEDS pixels/active-front geometry or a Fire M3 progression estimate after t. | Weak labels where operational geometry is absent. | Derived from active-fire detections, potentially overlapping FIRMS inputs; label provenance must be visible. |
| Burn-date raster | MODIS or VIIRS cell burn date, uncertainty, QA, and valid-observation bounds. | Daily historical weak labels and coverage masks. | Monthly retrospective product; burn date is estimated, not an operational update. |
| Final extent | WFIGS/IFPH, MTBS, NFDB, NBAC, or certified final perimeter. | Validation, incident end state, and long-history reference. | Do not infer within-event progression from a final polygon alone. |

## Minimum model-ready record

A derived incident-time cell record must include:

- Incident and source IDs plus the source snapshot IDs used.
- As-of time, horizon, cell identifier, centroid, CRS, and spatial resolution.
- A feature cutoff timestamp and weather issue/run plus valid times.
- Observation coverage status for the input and label intervals.
- Label value, label tier, label time interval, and label-quality flags.
- Upstream raw/normalized schema and transformation versions.

## U.S. and Canada source map

| Area | Operational collection | Historical/final reference |
| --- | --- | --- |
| United States | NIFC WFIGS current perimeter and IRWIN incident snapshots; selected state/agency feeds when they provide better timing. | WFIGS full history/IFPH, FODR, MTBS, and burned-area products. |
| Canada | CWFIS active fires and Fire M3 progression; provincial or territorial perimeter feeds where available. | CNFDB agency points/polygons, NBAC, and burned-area products. |

The boundary is U.S. and Canada. A separate contract is required before adding Mexico or other regions, including source authority, licensing, incident IDs, cadence, and coverage assessment.

## Output contract

Predictions must carry:

- As-of time and horizon.
- Cell geometry and centroid latitude/longitude.
- Probability, rank, calibration version, and uncertainty or coverage status.
- Feature and model version IDs.

This makes a list of predicted latitude/longitude locations traceable to the spatial cells, time horizon, and evidence used to produce it.

## Implementation status

The FIRMS raw/normalized path, a 20 GB admission policy, WFIGS reference-perimeter collection, CWFIS historical active-fire incident-context collection, the issued-at forecast record contract, a scored compact weather-tile planner, a Collection 2 VIIRS L2 inventory collector, and ETOPO terrain collection are implemented. The terrain blocks are on the source’s WGS84 15-arc-second grid: `elevation_m` is `int16` metres (`-32768` unavailable), `slope_degrees_x2` is `uint8` in 0.5° increments, and `aspect_degrees_x2` is downhill clockwise-from-north in 2° increments (`255` undefined). NALCMS country releases are retained with 30 m/19-class provenance only; a target-grid categorical feature cube has not been created. The CWFIS records preserve agency-report intervals and statuses but are not spread labels. The HRDPS plan records every selected and capped candidate using only FIRMS evidence available before the run under a conservative latency policy; it does not contain forecast values. The L2 inventory identifies the active-fire side of each required source pair but does not yet create paired geolocated cutouts or labels. WFIGS reference data validates end state but does not supply historical progression snapshots. Progression products and compact forecast-value tiles remain unimplemented. Until those inputs and perimeter snapshots are collected, the repository must not describe a model as predicting physical wildfire spread.
