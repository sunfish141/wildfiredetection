# Recursive renderer v2 experiment

The observation-age/count correction is implemented, but the regenerated
augmentation **does not pass its renderer screen**. Training admission remains
false. No scheduled-sampling model was fitted and no model was promoted.

## Reproducible inputs

- Base release: `wildfire-spread-firms-feds-no-weather-2026-05-11_to_2026-08-22`.
- Release manifest SHA-256:
  `201db0d293c56f5133f3b7edd65c5789b4191fea8147de6ef7ad557809160c84`.
- Classifier: the unchanged
  `artifacts/recursive-frontier-baseline-201db0d293c56f51/recursive_frontier_baseline.joblib`.
- New inspection:
  `artifacts/recursive-renderer-v2-201db0d293c56f51/one-step-augmentation/manifest.json`.
- Original inspection:
  `artifacts/recursive-frontier-baseline-201db0d293c56f51/one-step-augmentation/manifest.json`.

The new manifest includes the classifier checksum, generated CSV checksum,
complete renderer contract, calibration training snapshots, all 157 pair
reports, feature distributions, and comparison to the original inspection.
The [pipeline guide](training-pipeline.md#renderer-v2-calibration-and-admission-screen)
contains generation and exact-contract replay commands. Use fresh output
paths when repeating them; existing artifacts cannot be overwritten.

## Changed state and calibration

Observed initialization requires the centre detection age. Surviving state
adds 12 hours to that age; new ignitions receive 7.5 hours, the midpoint of
eligible observation ages [3, 12] within the previous step. Rendering applies
the inclusive 3--24-hour FIRMS availability/lookback window. An active
simulated cell can therefore persist without currently eligible evidence.

Calibration uses 34,462 detected centre rows from 162 training snapshots,
ending at `2026-08-02T00:00:00Z`. Holdout feature values and targets do not
enter calibration. Five equal-width intensity bins have these fitted means:

| Intensity bin | Detection count | Platform count |
| --- | ---: | ---: |
| [0, 0.2) | 2.010 | 1.448 |
| [0.2, 0.4) | 3.863 | 1.839 |
| [0.4, 0.6) | 3.342 | 1.605 |
| [0.6, 0.8) | 5.655 | 1.982 |
| [0.8, 1] | 9.252 | 2.213 |

The renderer rounds these means half-up and sums detection counts locally.
It combines platform counts by maximum. This is a proxy for platform
diversity, not a reconstruction of platform identity or acquisition history.
Brightness, intensity decay, and active duration remain heuristics.

## Training inspection results

| Diagnostic | Original v1 | Renderer v2 |
| --- | ---: | ---: |
| Training snapshot pairs | 157 | 157 |
| Matched synthetic rows | 34,249 | 32,044 |
| Matched positives | 3,968 | 2,619 |
| Synthetic positive rate | 11.59% | 8.17% |
| Historical frontier coverage | 11.59% | 10.85% |
| Historical positive-frontier coverage | 37.75% | 24.92% |
| Candidates outside historical frontier | 770,659 | 761,097 |

The historical frontier positive rate is 3.56%, so synthetic rows still have
a different class balance. The matched cell cohort changes with the renderer;
each feature comparison below uses the observed rows matching its own v2
synthetic cohort.

| Local 3×3 feature | Synthetic mean | Matched observed mean | Screen |
| --- | ---: | ---: | --- |
| Has detection | 0.353 | 0.479 | Pass |
| Detection count | 1.992 | 3.459 | Pass |
| Brightness maximum, K | 318.108 | 331.306 | Fail |
| Brightness mean, K | 317.488 | 319.432 | Fail |
| Platform count | 0.475 | 0.896 | Fail |
| Hours since last detection | 12.520 | 15.898 | Fail |
| Active cell count | 0.667 | 0.867 | Pass |

Brightness and age means exclude missing values. Those fields are missing
in 20,739 synthetic rows versus 16,682 observed rows, a 12.66-percentage-point
gap. The predeclared engineering screen allows at most 10 percentage points
of missingness difference and 0.5 observed standard deviations of mean or
quartile difference per feature. Four of seven fields fail. This screen is
an inspection aid, not proof of operational performance or independent
ground truth.

## Decision and next work

The fixed-origin open-loop comparison is persisted at
`artifacts/recursive-renderer-v2-201db0d293c56f51/open_loop_evaluation.json`.
It uses the exact inspected renderer and the unchanged classifier. Origin
`2026-08-02T12:00:00Z`, all 455 initial active cells, evaluation snapshots,
domain row counts, and positive counts match the original evaluation.

| Horizon | Domain coverage, v1 → v2 | Recall, v1 → v2 | PR-AUC, v1 → v2 | Outside-domain candidates, v1 → v2 |
| --- | ---: | ---: | ---: | ---: |
| 12 h | 28.96% → 28.96% | 35.44% → 82.11% | 0.4591 → 0.6939 | 5,979 → 5,979 |
| 24 h | 19.97% → 17.73% | 20.65% → 21.74% | 0.1118 → 0.0632 | 9,922 → 8,572 |
| 48 h | 8.06% → 3.49% | 4.32% → 3.60% | 0.1020 → 0.0753 | 14,654 → 11,107 |
| 96 h | 3.34% → 0.87% | 2.31% → 0.00% | 0.0814 → 0.0667 | 28,058 → 17,883 |

The 12-hour improvement does not persist through recursion. Fewer candidates
outside the label domain accompany substantially poorer later coverage; that
reduction is not evidence of a better multi-step predictor. These are
weak-label, single-origin diagnostics, not independent operational scores.

Keep both augmentation manifests unadmitted. The next experiment should
address lost observation history and brightness summaries, platform diversity,
and spatial frontier support before attempting a bounded weighted fit.
The current clipped brightness slider and decayed intensity cannot fully
represent the released observed distributions. Count calibration alone
cannot correct these missing state semantics.

The handoff's scheduled-sampling fit depends on an acceptable renderer
comparison, and its interactive application depends on an accepted recursive
model. Those prerequisites remain unmet. Incident/region holdouts, paired
VIIRS observation evidence, weather, and richer static/context features
remain the separately described research work in the handoff; this experiment
does not establish an operational predictor.

Validation: the 204-test suite passes. The generated CSV checksum, all 32,044
unique original training IDs, no-centre frontier membership, targets,
cell-specific cutoffs, 12-hour origin/feature pairing, and finite-or-missing
numeric features were checked against the base release.
