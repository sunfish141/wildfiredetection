# ADR 0002: first 1 km / 12-hour weak-label tabular baseline

## Status

Accepted for the first implementation — 2026-08-20.

## Context

The product must forecast where a known wildfire is likely to spread next and
return latitude/longitude locations. Raw FIRMS records are irregular satellite
points, not a stable output geometry or a progression target. The retained
WFIGS data are final/reference geometry and CWFIS records are incident
context; neither is a historical, time-resolved spread-label sequence.

NASA FEDS supplies time-stamped satellite-derived perimeter snapshots that can
form an initial target, but it shares satellite evidence with FIRMS. FEDS is
therefore useful as a weak label, not independent operational truth. Its
timestamp convention is local solar time with a UTC date, so treating all
snapshot timestamps as a global UTC observation instant would misalign
features and labels.

The complete local dataset, including raw evidence and derived tables, is
hard-capped at 20 GB by [ADR 0001](0001-bounded-20gb-local-dataset.md). A
full-grid cube, all native satellite swaths, and full forecast grids are out
of scope for the first implementation.

## Decision

1. Use a fixed 1,000 m North America Albers equal-area lattice in
   ESRI:102008. A cell ID is naea-1km:x=<integer>:y=<integer> and is the
   durable key for labels, features, model rows, evaluation, and prediction.
   Latitude/longitude is derived from that cell's centroid for output.
2. Use a 12-hour forecast horizon. A prediction row has an explicit UTC
   anchor_at/feature cutoff and target_end_at = anchor_at + 12 h; it is not
   required to fall on a global 00:00/12:00 UTC phase.
3. Build the first target from consecutive FEDS snapshots:

   ~~~text
   newly_burned(t, t + 12 h) = FEDS_perimeter(t + 12 h) − FEDS_perimeter(t)
   ~~~

   Rasterize changed area to 1 km cells and retain only positives. Missing
   FEDS coverage or cells outside the changed area are not implicitly negative.
   Each label carries weak_satellite, source snapshot IDs/times, geometry
   provenance, overlap fraction, and the time-alignment method.
4. Default to CONUS + Canada. Exclude Alaska from this first label dataset
   because the local-solar FEDS convention needs a separately validated
   time-alignment policy there.
5. Convert FEDS' source time to an estimated per-cell UTC overpass cutoff
   using longitude and a documented approximate local-solar overpass phase.
   Preserve the source timestamp and the estimate side by side. Do not use the
   simpler nominal-UTC mode for the baseline unless it is explicitly requested
   for an ablation.
6. Start with a compact tabular HistGradientBoostingClassifier, using
   leakage-gated FIRMS fire-state summaries and sampled ETOPO terrain.
   Train/validation splits are chronological and group FEDS rows by
   `source_snapshot_time`, so cell-local estimated cutoffs from one source
   snapshot cannot straddle train and holdout. The artifact must persist its
   feature list, metrics, grid version, label version, split group, and
   availability policy.
7. Treat weather as absent from the first table until issued-at forecast values
   are actually archived with run/issue, valid, publication/retrieval, and
   availability times. The HRDPS plan and notebook Open-Meteo CSVs are not
   weather features. Wind direction will enter as retained U/V components (or
   cyclic derivatives), not as an unwrapped degree scalar.
8. Persist the currently valid positive-only view now: one FEDS-positive cell
   joined to availability-gated FIRMS and sampled terrain, with raw-artifact
   lineage and explicit weather missingness. Archive-backed assembly requires
   terminal coverage for every FIRMS product/day needed in a row's usable
   lookback interval; missing source partitions are never zero fire evidence.
   Publish rows through one completed-build manifest so interrupted immutable
   artifacts cannot form a partial training view. Do not fit the binary
   classifier until a separate versioned candidate/negative policy supplies
   valid target=0 rows and keeps unknown observation states distinct.

## Consequences

The first model can produce a reproducible no-weather, satellite-weak
baseline while making its limitations visible. It cannot be described as an
operational weather-aware wildfire-spread forecast, and its score cannot be
interpreted as independent validation against FIRMS.

The remaining binary-training decision is deliberate: a versioned candidate
region and weak-negative/observation policy must be chosen before a classifier
is fitted. The positive-only training view is useful for lineage and feature
validation, but the training code must never convert an absent FEDS positive
into a zero label without that policy and, later, valid observation coverage.

The 20 GB cap is protected because terrain is sampled on demand, label
partitions are compact derived data, and the design does not materialize a
continental 1 km feature cube. Derived views remain subject to ADR 0001's
1.5 GB allocation and lineage requirements.

## Rejected alternatives

| Alternative | Reason not selected for v1 |
| --- | --- |
| Predict arbitrary latitude/longitude directly | It has no stable area semantics, makes deduplication/evaluation ambiguous, and cannot express a coherent spatial probability surface. |
| Global 00:00/12:00 UTC FEDS anchors | FEDS uses a local-solar source convention, so this would falsely imply one physical observation time across North America. |
| Use WFIGS final perimeters as time-series labels | Final/reference geometry does not reconstruct when land newly burned. |
| Treat FEDS absence as a no-spread negative | No FEDS perimeter change is not a valid clear/no-burn observation mask. |
| Include notebook Open-Meteo weather | It lacks issued-at forecast provenance and would leak post-event knowledge into an operational claim. |
| Start with a spatial deep-learning cube | It requires a candidate/negative policy and large dense feature storage that the current 20 GB package does not yet support. |
