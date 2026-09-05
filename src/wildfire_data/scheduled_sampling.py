"""Two-pass supervised dataset aggregation with incident-held-out evaluation.

Pass 1 fits observed training frontier rows. Training-only incident fragments
then mix actual and predicted cell state, with increasing predicted fractions.
Pass 2 fits observed rows plus a deterministic bounded, weighted synthetic
subset. All external holdouts remain untouched until both fits are complete.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .incident_sequences import (IncidentPolicy, build_incident_sequences, region_is_held,
                                 FEATURE_COLUMNS, stable_fraction)
from .incident_transition import (IncidentTransitionModel, observed_incident_state,
                                  mix_observed_and_predicted, OBSERVATION_COLUMNS,
                                  CalibratedSpreadEstimator, probability_logit)
from .incident_evaluation import probability_metrics, evaluate_incident_rollouts
from .recursive_calibration import fit_observation_calibration
from .recursive_transition import RECURSIVE_MODEL_FEATURE_COLUMNS, SyntheticObservationCalibration
from .rollout_augmentation import compare_renderer_features
from .rollout_sequences import snapshot_frame
from .tabular_baseline import _numeric_features, _binary_target, _validate_feature_columns
from .terrain_features import TerrainFeatureSampler
from .train_recursive_transition import _sha256_file, _verify_candidate_checksum, _atomic_json


TRAINING_VERSION = "incident-two-pass-scheduled-sampling/v1"
TARGET = "target_newly_burned_12h"
FEATURES = RECURSIVE_MODEL_FEATURE_COLUMNS
LINEAGE_COLUMNS = ("candidate_selection_reason", "label_tier", "label_quality_score",
                   "label_observability", "label_raw_artifact_ids", "firms_raw_artifact_ids",
                   "firms_feature_policy", "feature_cutoff_at")


@dataclass(frozen=True)
class SamplingPolicy:
    rollout_steps: int = 8
    predicted_fractions: tuple[float, ...] = (.25, .5, .75)
    max_synthetic_fraction: float = .5
    synthetic_weight: float = .25
    ignition_threshold: float = .2
    random_state: int = 0

    def __post_init__(self):
        object.__setattr__(self, "predicted_fractions", tuple(self.predicted_fractions))
        if (not isinstance(self.rollout_steps, int) or not 2 <= self.rollout_steps <= 8
                or not self.predicted_fractions
                or tuple(sorted(self.predicted_fractions)) != tuple(self.predicted_fractions)
                or any(not 0 <= x <= 1 for x in self.predicted_fractions)
                or not 0 < self.max_synthetic_fraction <= 1
                or not 0 < self.synthetic_weight <= 1
                or not 0 < self.ignition_threshold <= 1):
            raise ValueError("invalid bounded scheduled-sampling policy")

    def fraction(self, offset):
        if not 1 <= offset < self.rollout_steps:
            raise ValueError("scheduled offset outside fragment")
        return self.predicted_fractions[min(len(self.predicted_fractions) - 1,
            (offset - 1) * len(self.predicted_fractions) // (self.rollout_steps - 1))]


def fit_pass(observed_train, calibration_rows, *, synthetic=None, random_state=0):
    """Fit only designated training groups, calibrate only calibration groups."""
    if not observed_train.incident_split.eq("train").all() or observed_train.empty:
        raise ValueError("fit requires nonempty training-only rows")
    if not calibration_rows.incident_split.eq("calibration").all() or calibration_rows.empty:
        raise ValueError("probability calibration requires separate calibration incidents")
    train_groups = set(observed_train.incident_group_id)
    if train_groups & set(calibration_rows.incident_group_id):
        raise ValueError("training/calibration incident overlap")
    _validate_feature_columns(observed_train, target_column=TARGET, anchor_column="anchor_at",
                              split_group_column="source_snapshot_time", feature_columns=FEATURES)
    training = observed_train.copy()
    training["sample_weight"] = 1.
    if synthetic is not None and not synthetic.empty:
        if (not synthetic.incident_split.eq("train").all()
                or not set(synthetic.incident_group_id).issubset(train_groups)
                or not synthetic.sample_weight.between(0, 1, inclusive="right").all()):
            raise ValueError("synthetic rows violate training incident/weight policy")
        if (not synthetic.original_example_id.is_unique
                or not set(synthetic.original_example_id).issubset(set(observed_train.example_id))):
            raise ValueError("synthetic parents must be unique observed training rows")
        parents = observed_train.set_index("example_id").loc[synthetic.original_example_id]
        for column in ("incident_group_id", "cell_id", "source_snapshot_time", "anchor_at", "target_end_at", TARGET):
            if parents[column].tolist() != synthetic[column].tolist():
                raise ValueError(f"synthetic parent lineage/target mismatch: {column}")
        training = pd.concat([training, synthetic], ignore_index=True)
    y = _binary_target(training[TARGET], target_column=TARGET)
    yc = _binary_target(calibration_rows[TARGET], target_column=TARGET)
    if len(np.unique(y)) != 2 or len(np.unique(yc)) != 2:
        raise ValueError("training and calibration splits must both contain both classes")
    x = _numeric_features(training, FEATURES).to_numpy()
    xc = _numeric_features(calibration_rows, FEATURES).to_numpy()
    estimator = HistGradientBoostingClassifier(learning_rate=.08, max_iter=200, max_leaf_nodes=15,
                  l2_regularization=1., early_stopping=False, random_state=random_state)
    estimator.fit(x, y, sample_weight=training.sample_weight.to_numpy())
    raw = estimator.predict_proba(xc)[:, 1]
    calibrator = LogisticRegression(C=1., random_state=random_state)
    calibrator.fit(probability_logit(raw), yc)
    fitted = CalibratedSpreadEstimator(estimator, calibrator)
    return fitted, {"calibration_method": "regularized-logit-sigmoid; separate calibration incident groups",
                     "training_rows": len(training), "training_weight_sum": float(training.sample_weight.sum()),
                     "calibration_groups": sorted(set(calibration_rows.incident_group_id)),
                     "calibration_before": probability_metrics(yc, raw),
                     "calibration_after": probability_metrics(yc, fitted.predict_proba(xc)[:, 1])}


def generate_scheduled_examples(model, examples, sequences, *, terrain_provider, policy=SamplingPolicy()):
    """Generate only within training incidents, reseeding bounded 8-step fragments."""
    # Validate the caller's cached sequence positions before touching features.
    rows, reports = [], []
    for sequence in sequences:
        if sequence.split != "train":
            continue
        for snapshot in sequence.snapshots:
            source = snapshot_frame(examples, snapshot)
            if not source.incident_split.eq("train").all() or not source.incident_group_id.eq(sequence.incident_group_id).all():
                raise ValueError("sequence includes a held-out or different incident")
        for start in range(0, len(sequence.snapshots), policy.rollout_steps):
            fragment = sequence.snapshots[start:start + policy.rollout_steps]
            origin = snapshot_frame(examples, fragment[0])
            state = observed_incident_state(model, origin)
            if not state.active_cells or len(fragment) < 2:
                reports.append({"sequence_id": sequence.sequence_id, "origin": fragment[0].source_snapshot_time.isoformat(),
                                "status": "skipped-no-observed-seed-or-single-window"})
                continue
            for offset, snapshot in enumerate(fragment):
                frame = snapshot_frame(examples, snapshot)
                if offset:
                    if snapshot.source_snapshot_time - fragment[offset - 1].source_snapshot_time != pd.Timedelta(hours=12):
                        raise ValueError("scheduled sampling cannot bridge a missing snapshot")
                    observed = observed_incident_state(model, frame, step_index=state.step_index)
                    fraction = policy.fraction(offset)
                    state = mix_observed_and_predicted(observed, state, predicted_fraction=fraction,
                        key=sequence.sequence_id + ":" + str(start + offset) + ":" + str(policy.random_state))
                    historical = frame.loc[frame.firms_center_has_detection.eq(0)]
                    historical_ids = set(historical.cell_id)
                    candidates = {c.cell_id for c in model.candidate_cells(state)}
                    matched = historical.loc[historical.cell_id.isin(candidates)].set_index("cell_id", drop=False)
                    feature_rows = model.candidate_feature_rows(state, terrain_provider=terrain_provider,
                                                               include_cell_ids=historical_ids)
                    for cell, features in feature_rows:
                        row = matched.loc[cell.cell_id].to_dict()
                        original_id = row["example_id"]
                        row.update(features)
                        row.update({"original_example_id": original_id,
                                    "example_id": hashlib.sha256((TRAINING_VERSION + sequence.sequence_id + original_id).encode()).hexdigest(),
                                    "row_kind": "scheduled-synthetic-state",
                                    "sequence_id": sequence.sequence_id, "synthetic_state_steps": offset,
                                    "generation_origin_source_snapshot_time": fragment[0].source_snapshot_time.isoformat(),
                                    "predicted_state_fraction": fraction, "sample_weight": policy.synthetic_weight})
                        rows.append(row)
                    reports.append({"sequence_id": sequence.sequence_id, "status": "generated",
                        "origin": fragment[0].source_snapshot_time.isoformat(), "source_snapshot_time": snapshot.source_snapshot_time.isoformat(),
                        "synthetic_state_steps": offset, "predicted_state_fraction": fraction,
                        "historical_frontier_rows": len(historical), "matched_rows": len(matched),
                        "matched_positives": int(matched[TARGET].sum()),
                        "outside_historical_domain_candidates": len(candidates - historical_ids)})
                if offset < len(fragment) - 1:
                    state = model.step(state, terrain_provider=terrain_provider).state
    generated = pd.DataFrame(rows)
    observed_count = int((examples.incident_split.eq("train") & examples.firms_center_has_detection.eq(0)).sum())
    limit = int(observed_count * policy.max_synthetic_fraction)
    if not generated.empty:
        if not generated.original_example_id.is_unique:
            raise ValueError("synthetic parent IDs must be unique")
        generated["_rank"] = generated.original_example_id.map(lambda x: stable_fraction(str(policy.random_state) + ":subset:" + x))
        generated = generated.sort_values(["_rank", "original_example_id"]).head(limit).drop(columns="_rank").reset_index(drop=True)
    return generated, reports


def load_incident_view(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("kind") != "completed-incident-sequence-view" or manifest.get("status") != "complete":
        raise ValueError("training requires a completed incident manifest")
    release = Path(manifest["source_release"])
    if _sha256_file(release / "dataset_manifest.json") != manifest["source_release_manifest_sha256"]:
        raise ValueError("incident source release changed")
    artifact = manifest["assignment_artifact"]
    if _sha256_file(Path(artifact["path"])) != artifact["sha256"]:
        raise ValueError("incident assignments checksum mismatch")
    annotations = pd.read_csv(artifact["path"], keep_default_na=False)
    _verify_candidate_checksum(release, release / "candidate_examples.csv.gz")
    columns = list(dict.fromkeys([*FEATURE_COLUMNS, *OBSERVATION_COLUMNS, *FEATURES, *LINEAGE_COLUMNS]))
    examples = pd.read_csv(release / "candidate_examples.csv.gz", usecols=columns)
    if len(examples) != artifact["row_count"] or len(annotations) != len(examples):
        raise ValueError("incident assignment row count mismatch")
    examples = examples.merge(annotations, on="example_id", validate="one_to_one", how="left")
    if examples.incident_split.isna().any():
        raise ValueError("missing incident row assignments")
    policy = IncidentPolicy(**manifest["policy"])
    for group_id, group in examples.loc[examples.incident_group_id.ne("")].groupby("incident_group_id"):
        contract = manifest["groups"][group_id]
        if set(group.incident_split) != {contract["split"]}:
            raise ValueError("incident assignment disagrees with split manifest")
        if contract["split"] in ("train", "calibration"):
            if (pd.to_datetime(group.source_snapshot_time, utc=True, format="mixed") >= pd.Timestamp(policy.later_test_at)).any():
                raise ValueError("training/calibration incident reaches final later-time period")
            if any(region_is_held(c, policy) for c in group.cell_id):
                raise ValueError("training/calibration incident reaches held region")
    sequences = build_incident_sequences(examples, later_test_at=policy.later_test_at)
    declared = {s["sequence_id"]: s["snapshot_count"] for s in manifest["sequences"]}
    if {s.sequence_id: len(s.snapshots) for s in sequences} != declared:
        raise ValueError("sequence manifest differs from reconstructed view")
    return examples, sequences, manifest


def make_transition(estimator, calibration, sampling_policy, *, allowed_cell=None):
    return IncidentTransitionModel(estimator, feature_columns=FEATURES,
        observation_calibration=calibration, ignition_threshold=sampling_policy.ignition_threshold,
        allowed_cell=allowed_cell)


def load_mixed_training_view(path, examples, *, incident_manifest_sha256):
    """Read only completed, checksum-verified weighted training views."""
    manifest = json.loads(Path(path).read_text())
    if (manifest.get("kind") != "completed-incident-scheduled-sampling-training-view"
            or manifest.get("status") != "complete" or manifest.get("training_admitted") is not True
            or manifest.get("source_incident_manifest_sha256") != incident_manifest_sha256):
        raise ValueError("requires completed admitted mixed view for this incident manifest")
    if _sha256_file(Path(manifest["source_generator_model"])) != manifest["source_generator_model_sha256"]:
        raise ValueError("generator model checksum mismatch")
    frames = {}
    for name in ("observed", "synthetic"):
        artifact = manifest[name + "_artifact"]
        if _sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("mixed training artifact checksum mismatch")
        frames[name] = pd.read_csv(artifact["path"])
    references, synthetic = frames["observed"], frames["synthetic"]
    policy = SamplingPolicy(**manifest["policy"])
    if (len(references) != manifest["observed_row_count"] or len(synthetic) != manifest["synthetic_row_count"]
            or not references.example_id.is_unique or not references.sample_weight.eq(1.).all()
            or len(synthetic) > int(len(references) * policy.max_synthetic_fraction)
            or not synthetic.sample_weight.eq(policy.synthetic_weight).all()):
        raise ValueError("mixed training counts/weights violate the published policy")
    observed = examples.set_index("example_id", drop=False).loc[references.example_id].reset_index(drop=True)
    if (not observed.incident_split.eq("train").all() or not observed.firms_center_has_detection.eq(0).all()
            or observed.incident_group_id.tolist() != references.incident_group_id.tolist()):
        raise ValueError("observed training references violate incident/frontier policy")
    return observed, synthetic


def _save_pass_bundle(path, estimator, calibration, sampling_policy, train_groups, incident_manifest_sha256):
    if path.exists():
        raise ValueError("model bundle already exists")
    model = make_transition(estimator, calibration, sampling_policy)
    joblib.dump({"model": estimator, "feature_contract": {"feature_columns": list(FEATURES),
                 "training_version": TRAINING_VERSION, "source_incident_manifest_sha256": incident_manifest_sha256,
                 "training_incident_groups": sorted(train_groups)},
                 "transition_contract": model.transition_contract(), "observation_calibration": asdict(calibration)}, path)


def one_step_metrics(estimator, examples, policy):
    reports = {}
    for split in ("held_incident", "held_region", "later_time"):
        selected = examples.incident_split.eq(split) & examples.firms_center_has_detection.eq(0)
        if split == "later_time":
            selected &= pd.to_datetime(examples.source_snapshot_time, utc=True, format="mixed") >= pd.Timestamp(policy.later_test_at)
        frame = examples.loc[selected]
        probabilities = estimator.predict_proba(_numeric_features(frame, FEATURES).to_numpy())[:, 1] if len(frame) else []
        reports[split] = probability_metrics(frame[TARGET], probabilities)
    return reports


def compare_passes(first, second):
    def cases(evaluation):
        return sorted((r["sequence_id"], r["origin"], r["horizon_hours"], r["row_count"], r["positive_count"])
                      for r in evaluation["reports"])
    if cases(first) != cases(second):
        raise ValueError("open-loop comparison must use the same origins and target domains")
    before = {(r["split"], r["horizon_hours"]): r for r in first["summary"]}
    after = {(r["split"], r["horizon_hours"]): r for r in second["summary"]}
    if set(before) != set(after):
        raise ValueError("open-loop comparison must use the same cases")
    changes, regressions = [], []
    for key in sorted(before):
        a, b = before[key], after[key]
        for metric, higher_better in (("spatial_precision", True), ("spatial_recall", True), ("domain_coverage", True),
                                      ("cumulative_newly_burned_area_mae_km2", False), ("outside_domain_ignitions", False),
                                      ("mean_front_distance_m", False), ("undefined_front_distance_count", False)):
            old, new = a[metric], b[metric]
            if old is None or new is None:
                continue
            delta = new - old
            row = {"split": key[0], "horizon_hours": key[1], "metric": metric,
                   "pass_1": old, "pass_2": new, "delta": delta}
            changes.append(row)
            if key[1] > 12 and (delta < -1e-12 if higher_better else delta > 1e-12):
                regressions.append(row)
    return {"changes": changes, "multi_step_regressions": regressions,
            "promoted": False, "status": "research-only; multi-step regressions" if regressions else "research-only; further validation required"}


def train_two_pass(manifest_path: Path, output: Path, *, data_root: Path, sampling_policy=SamplingPolicy()):
    if output.exists():
        raise ValueError("two-pass output must be a new directory")
    examples, sequences, incident_manifest = load_incident_view(manifest_path)
    split_policy = IncidentPolicy(**incident_manifest["policy"])
    frontier = examples.firms_center_has_detection.eq(0)
    train = examples.loc[examples.incident_split.eq("train") & frontier]
    calibration_rows = examples.loc[examples.incident_split.eq("calibration") & frontier]
    observation_train = examples.loc[examples.incident_split.eq("train")].copy()
    observation_train["dataset_split"] = "train"
    observation_calibration = fit_observation_calibration(observation_train,
        release_manifest_sha256=incident_manifest["source_release_manifest_sha256"])
    print(f"Pass 1: {len(train)} observed training rows; {len(calibration_rows)} calibration rows", flush=True)
    pass_one, fit_one = fit_pass(train, calibration_rows, random_state=sampling_policy.random_state)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "observation_calibration.json", asdict(observation_calibration))
    _save_pass_bundle(output / "pass_1.joblib", pass_one, observation_calibration, sampling_policy,
                      set(train.incident_group_id), _sha256_file(manifest_path))
    sampler = TerrainFeatureSampler(data_root, max_cached_blocks=8)
    @lru_cache(maxsize=200000)
    def terrain(cell_id):
        return sampler.sample_cell(cell_id)
    generator = make_transition(pass_one, observation_calibration, sampling_policy,
                                allowed_cell=lambda cell_id: not region_is_held(cell_id, split_policy))
    print("Generating scheduled states from training incidents only", flush=True)
    synthetic, generation_reports = generate_scheduled_examples(generator, examples, sequences,
        terrain_provider=terrain, policy=sampling_policy)
    if synthetic.empty:
        raise ValueError("no matched synthetic training rows; no second pass can be fitted")
    synthetic_path = output / "synthetic_training_examples.csv.gz"
    synthetic.to_csv(synthetic_path, index=False, compression={"method": "gzip", "mtime": 0})
    observed_path = output / "observed_training_references.csv.gz"
    references = train[["example_id", "incident_group_id", "incident_split"]].copy()
    references["sample_weight"] = 1.
    references.to_csv(observed_path, index=False, compression={"method": "gzip", "mtime": 0})
    paired_observed = examples.set_index("example_id").loc[synthetic.original_example_id]
    drift = compare_renderer_features(synthetic, paired_observed)
    # Publish a new view for this explicitly bounded experiment. Prior failed
    # renderer artifacts are never loaded or have their admission changed.
    mixed_manifest = {"kind": "completed-incident-scheduled-sampling-training-view", "status": "complete",
        "training_version": TRAINING_VERSION, "source_incident_manifest_sha256": _sha256_file(manifest_path),
        "source_release_manifest_sha256": incident_manifest["source_release_manifest_sha256"],
        "source_generator_model": str(output / "pass_1.joblib"),
        "source_generator_model_sha256": _sha256_file(output / "pass_1.joblib"),
        "policy": asdict(sampling_policy), "training_admitted": True,
        "admission_scope": "bounded two-pass research experiment; no model promotion",
        "renderer_screen": drift, "observed_row_count": len(train), "synthetic_row_count": len(synthetic),
        "synthetic_positive_count": int(synthetic[TARGET].sum()), "synthetic_weight_sum": float(synthetic.sample_weight.sum()),
        "observed_artifact": {"path": str(observed_path), "sha256": _sha256_file(observed_path)},
        "synthetic_artifact": {"path": str(synthetic_path), "sha256": _sha256_file(synthetic_path)},
        "generation_reports": generation_reports}
    _atomic_json(output / "mixed_training_manifest.json", mixed_manifest)
    train, synthetic = load_mixed_training_view(output / "mixed_training_manifest.json", examples,
                                                incident_manifest_sha256=_sha256_file(manifest_path))
    print(f"Pass 2: adding {len(synthetic)} synthetic rows at weight {sampling_policy.synthetic_weight}", flush=True)
    pass_two, fit_two = fit_pass(train, calibration_rows, synthetic=synthetic, random_state=sampling_policy.random_state)
    artifacts, evaluations = {}, {}
    for name, estimator, fit_report in (("pass_1", pass_one, fit_one), ("pass_2", pass_two, fit_two)):
        model = make_transition(estimator, observation_calibration, sampling_policy)
        path = output / f"{name}.joblib"
        if name == "pass_2":
            _save_pass_bundle(path, estimator, observation_calibration, sampling_policy,
                              set(train.incident_group_id), _sha256_file(manifest_path))
        print(f"Evaluating {name}: fully open-loop incident/region/later-time holdouts", flush=True)
        evaluation = evaluate_incident_rollouts(model, examples, sequences, terrain_provider=terrain)
        evaluation["one_step_observed_metrics"] = one_step_metrics(estimator, examples, split_policy)
        evaluation["fit_report"] = fit_report
        _atomic_json(output / f"{name}_evaluation.json", evaluation)
        evaluations[name] = evaluation
        artifacts[name] = {"path": str(path), "sha256": _sha256_file(path),
                           "evaluation_path": str(output / f"{name}_evaluation.json"),
                           "evaluation_sha256": _sha256_file(output / f"{name}_evaluation.json")}
    comparison = compare_passes(evaluations["pass_1"], evaluations["pass_2"])
    _atomic_json(output / "comparison.json", comparison)
    manifest = {"kind": "completed-incident-two-pass-training-run", "status": "complete",
                "training_version": TRAINING_VERSION, "source_incident_manifest": str(manifest_path),
                "source_incident_manifest_sha256": _sha256_file(manifest_path),
                "mixed_training_manifest_sha256": _sha256_file(output / "mixed_training_manifest.json"),
                "policy": asdict(sampling_policy), "artifacts": artifacts,
                "comparison_sha256": _sha256_file(output / "comparison.json"),
                "promoted": False, "limitations": ["FEDS/FIRMS dependent weak labels; proxies are not verified negatives",
                    "incident complexes are retrospective grouping metadata, not independently verified incident identities",
                    "FIRMS availability is an estimate; source sequences are incomplete observed fragments",
                    "persistence, extinction, and newly ignited state remain bounded heuristics"]}
    _atomic_json(output / "run_manifest.json", manifest)
    return manifest


def load_pass_model(run_manifest_path: Path, pass_name="pass_2"):
    """Load a trusted local fitted bundle only through its completed run manifest."""
    manifest = json.loads(run_manifest_path.read_text())
    if manifest.get("kind") != "completed-incident-two-pass-training-run" or manifest.get("status") != "complete":
        raise ValueError("model loading requires a completed two-pass run")
    if pass_name not in ("pass_1", "pass_2"):
        raise ValueError("pass_name must be pass_1 or pass_2")
    artifact = manifest["artifacts"][pass_name]
    if _sha256_file(Path(artifact["path"])) != artifact["sha256"]:
        raise ValueError("model bundle checksum mismatch")
    # Joblib uses pickle; never use this loader on untrusted uploaded files.
    bundle = joblib.load(artifact["path"])
    if (tuple(bundle["feature_contract"]["feature_columns"]) != FEATURES
            or bundle["feature_contract"].get("source_incident_manifest_sha256") != manifest["source_incident_manifest_sha256"]):
        raise ValueError("persisted model feature contract mismatch")
    contract = bundle["transition_contract"]
    from .incident_transition import INCIDENT_TRANSITION_VERSION
    if contract["transition_version"] != INCIDENT_TRANSITION_VERSION:
        raise ValueError("unsupported incident transition version")
    return IncidentTransitionModel(bundle["model"], feature_columns=FEATURES,
        observation_calibration=SyntheticObservationCalibration(**bundle["observation_calibration"]),
        ignition_threshold=contract["ignition_threshold"], active_duration_steps=contract["active_duration_steps"],
        intensity_retention=contract["intensity_retention"], new_ignition_age_hours=contract["new_ignition_age_hours"],
        max_new_cells_per_step=contract["max_new_cells_per_step"], max_candidates=contract["max_candidates"],
        growth_fraction=contract["growth_fraction"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incident-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    train_two_pass(args.incident_manifest, args.output, data_root=args.data_root)


if __name__ == "__main__":
    main()
