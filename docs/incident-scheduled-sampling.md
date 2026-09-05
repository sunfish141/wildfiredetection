# Incident sequences and two-pass scheduled sampling

The first incident-sequence builder and two-pass supervised baseline are
implemented and have been run on the retained no-weather release. Pass 2
shows mixed open-loop results and is **not promoted**. This is dataset
aggregation with direct FEDS supervision; there is no reinforcement learning
policy or reward.

## Completed artifacts

- Incident view:
  `artifacts/incident-sequences-v1-201db0d293c56f51-halo/manifest.json`.
- Two-pass run:
  `artifacts/incident-two-pass-v1-201db0d293c56f51/run_manifest.json`.
- Mixed training view:
  `artifacts/incident-two-pass-v1-201db0d293c56f51/mixed_training_manifest.json`.
- Models and evaluations: `pass_1.joblib`, `pass_2.joblib`,
  `pass_1_evaluation.json`, and `pass_2_evaluation.json` in the run directory.
- Direct comparison: `comparison.json`; independent artifact/lineage checks:
  `verification.json` in the run directory.

The base release manifest SHA-256 is
`201db0d293c56f5133f3b7edd65c5789b4191fea8147de6ef7ad557809160c84`.
The original release, chronological models, and failed renderer inspection
artifacts remain unchanged. An initial incident view without the region
feature halo is retained as an intermediate diagnostic; it was not used for
either fit. The `-halo` manifest above is the completed training input.

## Incident identity and split contract

`incident_sequences.py` associates candidate cell centres with **current**
FEDS perimeters within 5 km. FEDS IDs are normalized to
`feds-incident/v1:<region>:<year>:<integer-fire-id>`. Each assignment retains
its provider record IDs. The source loader uses completed coverage-ledger
selections, verifies the normalized content hashes, and pins both source
artifacts and coverage selections in the new manifest. Current context pages
that did not contribute positive labels are retained separately as split
evidence; they do not supply additional labels.

Ambiguous associations are grouped together. Whole-history candidate bounding
boxes within 20 km also join their FEDS IDs into one incident complex. This
conservatively keeps nearby fires, shared context, and spatially adjacent
provider-ID changes on the same side of a split. Complex identity is a hash
of the sorted member keys. These are retrospective grouping identities,
not independently verified operational incident identities.

Whole-history grouping and split assignment are offline metadata. Neither
FEDS geometry, incident identity, region identity, future state, nor cutoff
metadata enters the classifier's feature columns. Unassociated candidates
remain explicitly unassigned, rather than becoming fabricated incidents.

Splits are fixed before either model, calibration, or augmentation is fitted:

1. Any complex with candidate rows at or after `2026-08-02T12:00:00Z` is
   assigned wholly to `later_time`. Its earlier rows are withheld from all
   fitting and excluded from later-time scoring.
2. Of the remaining complexes, any touching a held 1,000 km Albers region is
   assigned wholly to `held_region`. The seeded region hash selects 20% of
   possible blocks. Region checks include the 1 km feature halo, so a
   neighbouring held-region cell cannot enter training's 3×3 context.
3. A separate seeded incident hash assigns 20% of remaining complexes to
   `held_incident`, then 15% of the remainder to internal `calibration`.
4. The remainder is `train`. No neighbouring-cell random split is used.

| Partition | Complexes | Candidate rows |
| --- | ---: | ---: |
| Train | 45 | 4,029 |
| Calibration | 6 | 382 |
| Held incident | 18 | 1,596 |
| Held region | 46 | 8,439 |
| Later time, including withheld earlier context | 128 | 63,708 |
| Unassigned | — | 350,502 |

The 78,154 associated rows include all but 53 of the release's 22,656 weak
positives. Association therefore changes the population substantially:
results are conditional on FEDS-associated fire contexts, not on all FIRMS
candidates. Strict whole-incident separation also removes long-running fires
from training if they extend into the final period.

The resulting 670 sequence fragments preserve cell-specific cutoffs and split
on every missing 12-hour source window. Missing or no-positive-only source
windows are not filled with invented zero labels. These are complete observed
fragments, not complete historical lifecycles for every fire.

## Training and state construction

Both fits use the existing 13 no-centre FIRMS/terrain features and historical
rows with `firms_center_has_detection = 0`. There are 3,266 observed training
frontier rows and 329 probability-calibration rows, the latter containing 83
positives across six separate complexes.

Pass 1 fits `HistGradientBoostingClassifier` on observed training rows only:
200 iterations, 15 leaves, learning rate 0.08, L2 regularization 1, seed 0,
and no automatic early-stopping split. A regularized sigmoid of the model's
log odds is fitted on the separate calibration complexes. The final holdouts
do not enter either fit or calibration.

`incident_transition.py` initializes each training fragment with real FIRMS
centre aggregates. Observed detection count, maximum/mean brightness,
platform count, and observation age are retained; historical brightness is
not clipped to the synthetic slider range or changed by simulated intensity
decay. Counts for newly generated cells use intensity-bin calibration fitted
only on the 763 observed training centre rows.

Fragments are limited to eight target windows and longer sequences are
reseeded from observations at the next fragment. At successive state updates,
the predicted branch probability rises from 0.25 to 0.50 to 0.75. A seeded,
deterministic cell-level choice selects either the actual state or the
predicted state, including absence and burned masks. At every generated
feature snapshot, its matching historical FEDS next-12-hour target supplies
the label. An actual correction may undo a model false positive during
training; evaluation applies no such corrections.

Synthetic examples are saved only inside the intersection with that training
snapshot's historical frontier. They retain the original example ID, target,
cell-specific cutoff, target end, incident membership, label quality and
observation declarations, candidate-selection reason, and raw-artifact
lineage. Unreached labels and outside-domain predictions remain diagnostics.

The deterministic synthetic subset is capped at 50% of the observed training
row count and each synthetic row receives weight 0.25. The actual run produced
447 admitted synthetic rows, including 247 positives: total synthetic weight
111.75 versus observed weight 3,266. The manifest records the generator model
checksum and full policy. Pass 2 reads the newly completed, checksum-verified
mixed view, fits a new classifier, and fits its own sigmoid on the same six
calibration complexes.

The previous renderer screen remains a diagnostic. The new state retains
observed brightness and recency more faithfully; four of seven screen fields
pass, but count, maximum brightness, and active-cell count still drift. This
controlled experiment admits only its newly generated, bounded weighted
view. It does not change admission on the old failed augmentation artifacts
or imply that a model is ready for promotion.

## Rollout bounds

- Candidate cells must be immediate eight-neighbours of active cells: at most
  1,414.2 m centre-to-centre per 12-hour step. There is no distance-two jump.
- At most 5,000 candidates are retained, ranked by neighbouring intensity and
  then stable cell ID.
- Calibrated ignition probability must reach the fixed scenario threshold
  0.20. Among eligible cells, admit at most
  `min(128, ceil(0.5 * active_cell_count))`, ranked by probability and cell ID.
- Active cells persist for two steps, retain 85% simulated intensity per step,
  then become masked burned cells. An empty active state cannot spontaneously
  restart. These persistence/extinction rules are **heuristics**, not learned
  future-FIRMS targets.
- Observation age advances by 12 hours and only ages in [3, 24] are rendered.
  New ignitions use the documented 7.5-hour synthetic age. Platform diversity
  is still a count proxy; the released aggregates cannot reconstruct exact
  platform identities or individual observation expiry.
- Training candidates cannot enter held-region feature halos. Eight-step
  fragments and the 20 km grouping separation protect the known held-incident
  contexts without clipping inference to future FEDS geometry.

## Open-loop evaluation

Both models are scored on identical first eligible origins with eight
consecutive target windows and an observed FIRMS seed. Actual scored cases:
9 held-incident fragments from 8 complexes; 21 held-region fragments from 16
complexes; and 65 later-time fragments from 56 complexes. Short fragments and
fragments without a suitable observed seed are listed as exclusions. No
FIRMS or FEDS correction enters state after the origin.

Each evaluation JSON reports 12, 24, 48, and 96 hours, including spatial
precision/recall, accuracy, Brier score, domain coverage, new-burned-area
error, cumulative new-burned-area error, and front distances. Area/front
metrics describe **new growth since origin within the released label
domains**, not total physical burned area or a complete observed perimeter.
Outside-domain candidates and ignitions remain unknown diagnostics.

Front distance compares the boundaries of cumulative newly-burned 1 km cell
masks, using mean symmetric nearest-boundary distance and Hausdorff distance.
If either front is empty, distance is undefined, never zero. Reports retain
the number of defined and undefined cases. Mean distances from different
models can therefore have different support. Precision/recall use a zero
division convention in JSON; consult confusion counts. The table marks recall
undefined where there are no positive labels.

| Holdout | Horizon | Recall, pass 1 → 2 | Coverage, pass 1 → 2 | Cumulative area MAE km², pass 1 → 2 |
| --- | --- | ---: | ---: | ---: |
| Incident | 12 h | 11.43% → 11.43% | 41.18% → 41.18% | 3.44 → 3.33 |
| Incident | 24 h | 0.00% → 2.94% | 21.67% → 20.00% | 5.56 → 5.33 |
| Incident | 48 h | 0.00% → 7.69% | 15.38% → 23.08% | 8.11 → 7.44 |
| Incident | 96 h | — → — | 3.03% → 15.15% | 12.89 → 11.33 |
| Region | 12 h | 6.12% → 3.40% | 32.13% → 32.13% | 6.52 → 6.67 |
| Region | 24 h | 2.27% → 6.06% | 17.67% → 18.61% | 11.19 → 10.90 |
| Region | 48 h | 0.00% → 0.00% | 10.39% → 10.39% | 20.14 → 20.24 |
| Region | 96 h | 0.00% → 0.00% | 5.46% → 4.44% | 38.43 → 37.43 |
| Later time | 12 h | 7.56% → 7.84% | 31.92% → 31.92% | 5.03 → 5.06 |
| Later time | 24 h | 5.43% → 3.88% | 20.86% → 20.70% | 5.60 → 5.69 |
| Later time | 48 h | 4.67% → 4.00% | 12.17% → 10.20% | 11.08 → 11.06 |
| Later time | 96 h | 2.55% → 3.57% | 8.49% → 7.69% | 19.26 → 19.28 |

These results do not establish a better multi-step predictor. The comparison
records 22 metric regressions beyond 12 hours, and neither model is promoted.
The small training/calibration population, missing historical windows, and
remaining heuristic state dynamics constrain the result. FEDS and FIRMS share
satellite evidence; proxy zeros are not independently observed no-burn truth.
The later-time period was untouched by this experiment's training/generation,
but was used in earlier chronological baseline research; it is not a pristine
prospective test set.

## Reproduction and model loading

Run one archive-writing command at a time and use fresh output paths:

```bash
PYTHONPATH=src .venv/bin/python -m wildfire_data.incident_sequences \
  --release releases/wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22 \
  --data-root data \
  --output artifacts/incident-sequences-v1-201db0d293c56f51-halo

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python -m wildfire_data.scheduled_sampling \
  --incident-manifest artifacts/incident-sequences-v1-201db0d293c56f51-halo/manifest.json \
  --data-root data \
  --output artifacts/incident-two-pass-v1-201db0d293c56f51
```

`load_incident_view` and `load_mixed_training_view` reject incomplete or
checksum-mismatched inputs. `load_pass_model(run_manifest_path, "pass_2")`
loads a trusted local model through its completed run manifest, restores
calibration and transition parameters, and verifies the bundle checksum.
Use the returned model's `step` with retained terrain sampling for replay.

Both bundles were loaded in a fresh Python process and produced finite,
normalized probabilities. Training parents, targets, cutoffs, weights,
incident isolation, region halos, and identical evaluation cases were audited.
The full repository suite passes all 221 tests, including incident isolation,
scheduled-state generation, calibration-bin boundaries, and open-loop checks.

Next work should learn or improve future fire-state persistence/observation
semantics and test less restrictive but training-selected growth controls.
Tune further experiments with internal training development splits; preserve
external holdouts for a final locked comparison. Recover fuller observation
sequences before interpreting sparse long-horizon results as incident skill.
