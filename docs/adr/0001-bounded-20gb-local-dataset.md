# ADR 0001: bounded 20 GB local wildfire dataset

## Status

Accepted — 2026-08-16; amended — 2026-08-18 and 2026-08-20.

## Context

The local `data/` directory must hold the entire 2026-05-31 through 2026-08-10 dataset, including existing CSV files, raw evidence, normalized records, manifests, caches, and derived training views, within 20,000,000,000 bytes.

The competing full-evidence design cannot meet that constraint. The discovered SNPP, NOAA-20, and NOAA-21 VIIRS Level-2 fire files alone total about 19.56 GB before matching geolocation files, FIRMS, operational labels, weather, static layers, or filesystem overhead. Full HRRR and HRDPS forecast grids are orders of magnitude larger.

Three adversarial reviews informed this decision:

- Keeping native inputs maximizes later reprocessing and auditability, but full L2 and forecast grids consume the budget before labels and weather fit.
- Keeping only detections is compact, but loses valid-no-fire, cloud, water, and unavailable-observation states needed for defensible negatives.
- Keeping source-derived tiles and deltas preserves the features needed for spread prediction, but makes future feature research outside retained tiles non-reproducible without recollection.

## Decision

Treat `data/` as a bounded canonical training package, not an unlimited raw archive. Enforce the whole-directory limit with [the machine-readable budget](../../config/storage_budget.json). Existing files are pinned and counted; no collector may silently delete them to make room.

Use these allocations:

| Local category | Cap | Decision |
| --- | ---: | --- |
| FIRMS and detection evidence | 0.75 GB | Retain raw FIRMS CSVs and one compact canonical detection representation. |
| Operational labels and progression | 3.0 GB | Retain revision/delta responses from WFIGS/IRWIN, FEDS, CWFIS active fires, and Fire M3 perimeters. Do not retain redundant Fire M3 hotspots. |
| VIIRS L2 paired cutouts | 3.0 GB | Retain selected paired fire-mask/QA and geolocation cutouts plus complete pair provenance. |
| Issued weather tiles | 5.0 GB | Retain fixed-variable HRRR/HRDPS tiles selected from evidence available at the as-of time. |
| Static cell features | 5.5 GB | Retain compact ETOPO terrain blocks and at most one immutable CEC NALCMS source archive per country release; prohibit extracted TIFF copies and categorical grids before target semantics are fixed. |
| Derived training views | 1.5 GB | Retain compact, lineage-verified model partitions; these alone may be evicted. |
| Manifests, staging, and headroom | 1.25 GB | Reserve space before any atomic pair/tile write. |

This reallocates unused paired-L2 and derived-view capacity to the static evidence that can be collected correctly now, while retaining a 3.0 GB reserve for future paired L2 cutouts.

### Required compact records

Every training candidate must preserve evidence quality rather than a simplistic binary label:

- `fire_evidence_score`: strength and agreement of FIRMS evidence; never a replacement for source fields.
- `observation_coverage_score`: valid-land observation fraction, with cloud/water/glint/unprocessed fractions and raw QA retained.
- `forecast_availability_score`: source-run availability, variable completeness, lead time, and run age.
- `label_quality_score`: operational geometry versus satellite weak-label provenance and temporal quality.
- `retention_priority_score`: deterministic reason a tile or record was admitted under the cap.

These belong in compact derived/candidate tables, never by mutating immutable raw provider CSVs.

### L2 source-pair rule

The active-fire product supplies fire mask and algorithm QA, while the matching geolocation product supplies latitude and longitude. The source atom is therefore a pair:

| Fire product | Geolocation product |
| --- | --- |
| `VNP14IMG.002` | `VNP03IMG.002` |
| `VJ114IMG.002` | `VJ103IMG.002` |
| `VJ214IMG.002` | `VJ203IMG.002` |

Never admit, evict, or describe one partner as a complete L2 observation. A compact derived tile records both native IDs, versions, URLs/checksums, observation time, crop geometry, pixel row/column, fire-mask class, and raw QA value.

### Selection rules

1. Select perimeter, FIRMS, weather, and L2 tiles only from information available at the tile anchor/as-of time. Do not select a tile because of later spread.
2. Use fixed fire-context windows and a capped, deterministic score to include both positive and observed-no-fire cells.
3. Record every non-admission or eviction as `capped`, `unavailable`, `missing`, or `evicted` with a reason. Never turn it into a negative fire label.
4. Reserve enough space for an entire source pair and its staging before downloading it. Do not leave an orphaned L2 partner after a quota failure.
5. Do not archive full native L2 swaths, full HRRR/HRDPS grids, repeated whole WFIGS snapshots, redundant Fire M3 hotspots, or uncontrolled/duplicate nationwide static copies. The static exception is one immutable CEC NALCMS source ZIP per country release, retained without extracting the national TIFF into `data/`.

## Consequences

The local package can fit within 20 GB and remains suitable for a compact operational spread model. It loses full-domain retrospective feature generation and whole-swath observation coverage outside retained context tiles. A future experiment needing an unretained L2 pixel, forecast variable, lead time, or national raster other than the retained NALCMS source releases must recollect it or obtain it from external archival storage.

The package is therefore locally reproducible only for its declared compact feature set. Each model release must state this limitation and reference the budget-policy version used to select its data.

## Admission decisions and evidence status

The priority order was reviewed adversarially for label value, leakage risk, source persistence, and bytes. Space is intentionally left unused when the only available source would create a misleading historical record.

| Source | Decision | Reason and present status |
| --- | --- | --- |
| Unfiltered FIRMS, all three VIIRS NRT products | Admit | Core fire evidence. Retain raw responses and lossless normalized rows. The requested 72-day range is complete or explicitly empty by product/day. |
| WFIGS Year-to-Date perimeters | Admit as reference only | 2,019 U.S. geometries for 2026-05-31 through 2026-08-10 were collected. They are `final_reference` and cannot be treated as a historical operational revision sequence. |
| CWFIS active fires | Admit as incident context | The new CWFIF layer preserves `record_start`/`record_end` per agency-record version. The requested range yielded 10,136 compact Canadian context records. They provide IDs, status, size, and source timing, never a spread geometry label. |
| CWFIS Fire M3 perimeters | Reserve, do not backfill as snapshots | The examined Fire M3 July query contained geometry whose source fields reached August, so it is not safe to call it a July as-issued snapshot. Begin daily captures for future ranges or use a dated provider archive; never backfill an invented progression series. Fire M3 hotspot points are excluded as redundant with FIRMS/VIIRS. |
| NASA FEDS NRT perimeters | Admit as compact satellite-weak label evidence | The public service is now captured as immutable response pages for the requested range (12,148 returned provider features in the first capture). It is not an operational perimeter history: source identity/timestamp provenance is retained, FEDS-to-FIRMS dependence is explicit, labels are positive-only, and Alaska remains outside the first local-solar time-aligned dataset. |
| VIIRS L2 fire mask/QA + geolocation | Reserve for paired cutouts | CMR inventory is retained, but the compact pair adapter and Earthdata access are required before any L2 pixel is admitted. Full fire files alone are about 19.56 GB and fail both the pairing and storage rules. |
| HRDPS issued forecast | Admit plan, defer values pending a compact extractor | A scored plan for the still-public 2026-07-17 through 2026-08-10 slice stores 56,497 candidates and 3,200 selected tiles under the weather allocation. It uses only FIRMS evidence available under a conservative three-hour latency policy. It is not weather data and explicitly marks historical publication availability as uncertain. |
| HRRR issued forecast | Reserve | NOAA's archive is persistent, unlike the time-limited HRDPS source. Defer download until the same compact extractor has been validated. |
| ETOPO 2022 v1 terrain | Admit | Retain 47 quota-admitted 10° WGS84 surface-elevation subsets selected from 2,762 FIRMS 96 km Web-Mercator context tiles. Their 271,166,584 source-grid cells are preserved as raw NOAA subsets, compact NPZ elevation/slope/aspect blocks, and normalized provenance. The terrain-only audit used 859,270,132 bytes; it is not a uniform equal-area model grid. |
| CEC NALCMS 2020 v2 land cover | Admitted as raw source evidence only | Verified Canada and U.S. country ZIPs are retained as immutable, content-addressed provider bytes with 30 m, 19-class FAO-LCCS provenance; national TIFFs are not extracted into `data/`. The Canada component year is 2020; the U.S. release combines CONUS 2019 and Alaska 2021, so “2020” is a release name rather than a uniform observation year. Categorical compaction remains pending an explicit target grid and mode/fraction rule; class IDs must never be averaged. The post-admission audit used 4,540,056,932 of the 5,500,000,000 static bytes and 4,964,605,150 of the 20,000,000,000 whole-data bytes. |

The HRDPS plan is deliberately not a claim that the historical forecast files were originally published at the specified anchor. It carries `forecast_availability_score=0.6` and a documented latency assumption. A value tile may only be joined to a training row after its model-run, valid time, source retrieval, and conservative availability policy pass the as-of check.
