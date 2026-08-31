# No-weather proof of concept: 2026-05-11 to 2026-08-22

This is the active first proof of concept: a coherent, uploadable candidate
table and reproducible tabular baseline **without weather**. It is a weak-label
research baseline, not an operational spread forecast.

Do not run either Open-Meteo collector for this POC. A later weather experiment
must publish a separate immutable candidate view; it must not modify this one.

## Scope and source boundaries

| Role | Required period | POC use |
| --- | --- | --- |
| FIRMS SNPP, NOAA-20, NOAA-21 | 2026-05-10 through 2026-08-22 | Cutoff-safe fire-state features; May 10 supplies the 24-hour lookback. |
| FEDS source snapshots | 2026-05-11 through 2026-08-23 | 12-hour perimeter-difference labels; Aug 23 is boundary evidence only. |
| Labels, derived views, and release | 2026-05-11 through 2026-08-22 | Prediction/label range. |
| ETOPO terrain | Selected candidate cells | Static elevation, slope, and aspect. |
| WFIGS, CWFIS, VIIRS L2 inventory | Retained POC range | Reference/context/observability evidence only—not inputs or targets. |
| NALCMS | Existing source archives | Not yet a model feature. |

All required source collections and selected derived views need their intended
coverage/build manifests. A FEDS interval with no positive expansion is the
documented exception: it stays `partial` and excluded from positive rows; it is
never turned into a global no-spread/zero label.

## Build and export

Collect non-weather evidence one archive-writing command at a time. Build the
positive view once, then construct candidate rows as consecutive seven-day
chunks (do not run chunks concurrently). A bounded terrain cache keeps this
long range within memory while avoiding repeated ETOPO block reads. Every
chunk must share the *global* split dates:

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.build_training_dataset \
  --start 2026-05-11 --end 2026-08-22 --data-root data

PYTHONPATH=src .venv/bin/python -m wildfire_data.build_candidate_dataset \
  --start 2026-05-11 --end 2026-05-17 \
  --split-start 2026-05-11 --split-end 2026-08-22 \
  --positive-view-manifest data/manifests/training-dataset-builds/<positive-view>.json \
  --max-cached-terrain-blocks 24 \
  --data-root data

# Repeat for contiguous seven-day date ranges through 2026-08-22, then merge
# the printed completed chunk manifests in chronological order.
PYTHONPATH=src .venv/bin/python -m wildfire_data.merge_candidate_dataset \
  --start 2026-05-11 --end 2026-08-22 --data-root data \
  --input-manifest <chunk-1-manifest.json> \
  --input-manifest <chunk-2-manifest.json>

PYTHONPATH=src .venv/bin/python -m wildfire_data.export_candidate_dataset \
  --data-root data \
  --candidate-manifest data/manifests/candidate-dataset-builds/<candidate-view>.json \
  --output releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22
```

The expected output is
`releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22/`.
Its `dataset_manifest.json`—not a glob of retained artifacts—establishes the
release identity, date range, row counts, feature allowlist, and weather
status.

The root [collection notebook](../wildfire_firms_analysis.ipynb) automates
this exact chunk-and-merge sequence with a seven-day chunk size and a
24-block terrain cache.

Schema-v2 releases contain:

- `candidate_examples.csv.gz`: the fit-eligible tabular input.
- `candidate_examples.jsonl.gz`: the equivalent lossless row stream.
- `unscored_positives.csv.gz` and `.jsonl.gz`: FEDS positives outside FIRMS
  candidate support. They are diagnostics, never training negatives.
- `dataset_manifest.json`, `schema.json`, `file_inventory.json`, and
  `SHA256SUMS`: the provenance and integrity contract.

Nested CSV list/dictionary values are canonical compact, sorted-key JSON
strings. `schema.json` declares columns and encoding; the manifest records
payload digests and row counts. Verify `SHA256SUMS` and the manifest/schema
before loading the CSV. Do not combine files from different candidate manifests.

The 2026-05-31 to 2026-08-10 release is a legacy JSONL-only schema-v1
artifact. It remains historical evidence, but it is not this POC's CSV input.

## Completed POC release

`releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22/`
has been built and checksum-verified. Its immutable candidate build is
`27f6f72166ca43c2a2d92c2691deb4eb` and contains 428,656 candidate rows:
22,656 supported weak positives, 406,000 weak-negative proxies, and 12,924
unscored positives. It declares 19 model input fields and no weather inputs.

## Labels and no-weather contract

`target_newly_burned_12h = 1` is a FEDS perimeter-difference positive with
label tier `weak_satellite`. `target_newly_burned_12h = 0` is a deterministic,
FIRMS-seeded `weak_negative_proxy`, not observed clear/no-burn. An omitted
FEDS cell never becomes zero. FEDS positives outside FIRMS support are retained
as unscored diagnostics, not discarded or relabelled.

Each candidate preserves its cell identity/centroid, cutoff and target end,
FEDS/FIRMS/terrain lineage, availability policy, selection reason, label
observability, and split metadata. FEDS labels and FIRMS features share
satellite evidence, so scores are not independent ground-truth validation.

Weather fields must remain explicit missingness/provenance declarations:

- `weather_available = 0` and `weather_missing_indicator = 1`
- `weather_feature_status = unavailable-no-issued-forecast-features`
- `weather_input_policy = exclude-open-meteo-retrospective-exports/v1`

They are not model inputs. The release manifest must have
`weather.available = false`; do not fill, fetch, impute, or train on weather.

## Training notebook

Run [the tabular-baseline notebook](../notebooks/train_tabular_baseline.ipynb)
against the completed release. It verifies the checksum, declared CSV columns,
and row count before loading `candidate_examples.csv.gz`; enforces unique
`example_id` and binary `target_newly_burned_12h`; and uses only the ordered
`model_feature_columns` from the manifest.

It fits `HistGradientBoostingClassifier` with a chronological holdout grouped
by `source_snapshot_time`, never a random neighbouring-cell split, then
persists the model, feature contract, metrics, and a run manifest referencing
the exact input release/build. IDs, timestamps, labels, geometry, source/raw
lineage, selection metadata, weather declarations, and `dataset_split` are not
model features.

The score is a POC sanity check only. It does not establish operational
readiness, independent validation, clear/no-burn classification, or a
weather-aware model.

## After the POC

Only after this release and baseline are verified may a weather-bearing
experiment begin. It must start from this completed base candidate manifest,
retain separate weather mapping/raw-response provenance, obey the existing
rate-limit/429-pause policy, and publish a distinct immutable release.
