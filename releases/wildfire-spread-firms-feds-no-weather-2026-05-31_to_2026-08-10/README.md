# Wildfire spread weak-label candidate dataset

This uploadable release contains one manifest-selected no-weather candidate
table for 2026-05-31 through
2026-08-10. It is a research baseline for 1 km,
12-hour spread prediction, not an operational fire-spread forecast.

Files:

- `candidate_examples.jsonl.gz`: fit-eligible weak positives and FIRMS-seeded
  weak-negative proxies.
- `unscored_positives.jsonl.gz`: FEDS positives outside FIRMS candidate
  support; retain these for coverage diagnostics rather than treating them as
  negatives or dropping them.
- `dataset_manifest.json`, `schema.json`, `file_inventory.json`, and
  `SHA256SUMS`: version, schema, and integrity information.

Rows: 305,528 candidate rows and
11,848 unscored positives.

## Model feature allowlist

- `firms_center_has_detection`
- `firms_center_detection_count`
- `firms_center_bright_ti4_max`
- `firms_center_bright_ti4_mean`
- `firms_center_platform_count`
- `firms_center_hours_since_last_detection`
- `firms_local_3x3_has_detection`
- `firms_local_3x3_detection_count`
- `firms_local_3x3_bright_ti4_max`
- `firms_local_3x3_bright_ti4_mean`
- `firms_local_3x3_platform_count`
- `firms_local_3x3_hours_since_last_detection`
- `firms_local_3x3_active_cell_count`
- `terrain_valid`
- `terrain_elevation_m`
- `terrain_slope_degrees`
- `terrain_aspect_defined`
- `terrain_aspect_sin`
- `terrain_aspect_cos`

## Limitations

- target=0 means a FIRMS-seeded weak-negative proxy, not observed clear/no-burn.
- FEDS weak labels and FIRMS features share satellite evidence and are not independent ground truth.
- FIRMS-uncovered positives are retained in unscored_positives.jsonl.gz and excluded from the candidate table.
- Only source snapshots represented by FEDS-positive labels are included; no all-zero FEDS window is invented.
- Weather is absent from this immutable past release. It predates the
  retrospective ECMWF IFS weather-analysis backfill contract; its contents and
  checksums are unchanged.
- The chronological split groups whole source snapshots but is not incident-held-out or region-held-out validation.
