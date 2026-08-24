# Uploadable wildfire spread candidate dataset

## Completed release

The current self-contained release is:

```text
releases/wildfire-spread-firms-feds-no-weather-2026-05-31_to_2026-08-10/
```

It is a no-weather, 1 km, 12-hour weak-label research dataset covering source
snapshots from 2026-05-31 through 2026-08-10. It was built only from the
completed positive-view manifest and is suitable for upload without the local
raw archive.

| Item | Count |
| --- | ---: |
| Candidate rows | 305,528 |
| FEDS/FIRMS-supported target=1 rows | 19,528 |
| FIRMS-seeded target=0 weak-negative proxies | 286,000 |
| Unscored FEDS positives outside FIRMS candidate support | 11,848 |
| Model input fields | 19 |

The release directory contains:

- `candidate_examples.jsonl.gz` — fit-eligible weak positives and weak-negative
  proxies, with full row lineage and the explicit model-feature allowlist in
  `schema.json`.
- `unscored_positives.jsonl.gz` — FEDS positives that have no FIRMS candidate
  support. Keep these for coverage analysis; do not relabel or discard them.
- `dataset_manifest.json`, `schema.json`, `file_inventory.json`, and
  `SHA256SUMS` — provenance, contract, and integrity data.

## Reproduce or rebuild

Run the notebook [build_uploadable_dataset.ipynb](../notebooks/build_uploadable_dataset.ipynb)
from the repository root, or run:

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

The exporter reuses an existing release only after verifying its logical JSONL
payload is byte-equivalent; it never overwrites different content. Verify an
upload by hashing each path named in `SHA256SUMS` and comparing it to the
listed digest.

## Scope and limitations

- Weather is deliberately absent. The existing Open-Meteo files are
  retrospective visualization caches with no forecast run, issue/publication,
  or cutoff-availability provenance.
- `target=0` is not observed clear/no-burn; it is a capped, deterministic
  FIRMS-seeded weak-negative proxy. Use it only as a first research baseline.
- FEDS labels and FIRMS features share satellite evidence, so the labels are
  not independent ground truth.
- The chronological split keeps whole FEDS source snapshots together, but it
  is not an incident-held-out or region-held-out evaluation.
- The release has only source snapshots with FEDS-positive labels; it does not
  invent all-zero FEDS time windows.

## Extending to the requested summer

The retained source archive cannot honestly produce a no-weather data set for
2026-05-11 through 2026-08-22 yet. Before requesting that range, collect and
verify FIRMS for 2026-05-10 through 2026-08-22 and FEDS snapshots for
2026-05-11 through 2026-08-23. Then rebuild FEDS labels, terrain blocks, the
positive view, and the candidate view. The date guard in the builder prevents
mixing a longer request with the current shorter manifest.
