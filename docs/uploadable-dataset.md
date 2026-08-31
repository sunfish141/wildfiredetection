# Uploadable wildfire spread candidate dataset

## Active no-weather POC release

The active proof of concept rebuilds the no-weather range 2026-05-11 through
2026-08-22 and exports it to:

```text
releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22/
```

This target directory is only a completed release after its own
`dataset_manifest.json` has been published. Its schema-v2 export uses
`candidate_examples.csv.gz` as the fit-eligible tabular input, with the same
rows retained in JSONL. `unscored_positives.csv.gz` is diagnostic evidence,
not extra training data. The POC is explicitly weather-free; it must not run
Open-Meteo collection or add weather columns. See [the no-weather POC guide](no-weather-poc.md)
for source boundaries, build commands, label semantics, and the training
notebook workflow.

### Schema-v2 CSV contract

`schema.json` lists the exact CSV columns and documents canonical compact,
sorted-key JSON strings for nested values. `dataset_manifest.json` records the
candidate build identity, CSV payload digests, row counts, weather status, and
ordered `model_feature_columns`; `SHA256SUMS` verifies every released file.
The trainer must verify those files before loading
`candidate_examples.csv.gz`, use only the manifest feature allowlist, and
keep `unscored_positives` out of fitting.

## Completed active release

The current self-contained release is:

```text
releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22/
```

It is a no-weather, 1 km, 12-hour weak-label research dataset covering source
snapshots from 2026-05-11 through 2026-08-22. It was built only from the
completed positive-view manifest and is suitable for upload without the local
raw archive.

| Item | Count |
| --- | ---: |
| Candidate rows | 428,656 |
| FEDS/FIRMS-supported target=1 rows | 22,656 |
| FIRMS-seeded target=0 weak-negative proxies | 406,000 |
| Unscored FEDS positives outside FIRMS candidate support | 12,924 |
| Model input fields | 19 |

The release contains:

- `candidate_examples.jsonl.gz` — fit-eligible weak positives and weak-negative
  proxies, with full row lineage and the explicit model-feature allowlist in
  `schema.json`.
- `unscored_positives.jsonl.gz` — FEDS positives that have no FIRMS candidate
  support. Keep these for coverage analysis; do not relabel or discard them.
- `dataset_manifest.json`, `schema.json`, `file_inventory.json`, and
  `SHA256SUMS` — provenance, contract, and integrity data.

It is a schema-v2 CSV/JSONL release. The prior May 31–August 10 schema-v1
JSONL release remains immutable legacy evidence and is not rewritten.

## Reproduce or rebuild

For the active no-weather POC, follow [the POC guide](no-weather-poc.md) and
then use [the tabular-baseline notebook](../notebooks/train_tabular_baseline.ipynb).
To reproduce the past no-weather release, run:

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.build_candidate_dataset \
  --start 2026-05-31 \
  --end 2026-08-10 \
  --split-start 2026-05-31 \
  --split-end 2026-08-10 \
  --data-root data
```

Then export the exact completed candidate manifest printed by that command:

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.export_candidate_dataset \
  --data-root data \
  --candidate-manifest <completed-candidate-manifest.json> \
  --output releases/<new-release-name>
```

The exporter reuses an existing release only after verifying its logical
payloads are byte-equivalent; it never overwrites different content. Verify an
upload by hashing each path named in `SHA256SUMS` and comparing it to the
listed digest.

## Scope and limitations

- Weather is deliberately absent from this immutable past release. No forward
  forecast capture with explicit model/run, immutable raw response,
  candidate-cell/tile mapping, and captured availability provenance was made
  for it. Later collection cannot alter this release, although a newly built
  range may include retrospective historical-weather analysis features.
- `target=0` is not observed clear/no-burn; it is a capped, deterministic
  FIRMS-seeded weak-negative proxy. Use it only as a first research baseline.
- FEDS labels and FIRMS features share satellite evidence, so the labels are
  not independent ground truth.
- The chronological split keeps whole FEDS source snapshots together, but it
  is not an incident-held-out or region-held-out evaluation.
- The release has only source snapshots with FEDS-positive labels; it does not
  invent all-zero FEDS time windows.

## Legacy reproduction

The old 2026-05-31 through 2026-08-10 artifact can be reproduced using its
legacy commands above. For the completed summer POC, use the bounded chunk and
merge instructions in [the no-weather POC guide](no-weather-poc.md), then run
`notebooks/train_tabular_baseline.ipynb` against its verified CSV release.

Weather is intentionally deferred until this proof of concept is complete. A
later weather experiment must take the completed base candidate manifest,
backfill mapped tile/hour values through the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
and publish a distinct weather-bearing release. The base no-weather view
remains unchanged; it is not a reconstructed issued-forecast data set.
