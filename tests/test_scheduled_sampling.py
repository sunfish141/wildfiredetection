import unittest
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from wildfire_data.incident_sequences import build_incident_sequences
from wildfire_data.incident_transition import (EvidenceCell, IncidentTransitionModel,
    observed_incident_state, mix_observed_and_predicted)
from wildfire_data.recursive_transition import RECURSIVE_MODEL_FEATURE_COLUMNS, RecursiveFireState
from wildfire_data.scheduled_sampling import SamplingPolicy, fit_pass, generate_scheduled_examples, load_mixed_training_view, compare_passes
from wildfire_data.train_recursive_transition import _sha256_file
from wildfire_data.incident_evaluation import front_distance, evaluate_incident_rollouts, score_spatial_horizon, probability_metrics


FEATURES = RECURSIVE_MODEL_FEATURE_COLUMNS


class EvidenceClassifier:
    def predict_proba(self, values):
        index = FEATURES.index("firms_local_3x3_active_cell_count")
        p = np.where(np.asarray(values)[:, index] > 0, .8, .01)
        return np.column_stack([1-p, p])


def terrain(_cell_id):
    return {"terrain_valid": True, "terrain_elevation_m": 500., "terrain_slope_degrees": 5.,
            "terrain_aspect_defined": True, "terrain_aspect_sin": 0., "terrain_aspect_cos": 1.}


def frame(split="train", steps=8):
    rows = []
    for step in range(steps):
        timestamp = pd.Timestamp("2026-07-01T00:00:00Z") + pd.Timedelta(hours=12*step)
        for x in (0, 1, 2):
            row = {"example_id": f"{split}-{step}-{x}", "cell_id": f"naea-1km:x={x}:y=0",
                   "source_snapshot_time": timestamp.isoformat(), "target_snapshot_time": None,
                   "anchor_at": timestamp.isoformat(), "target_end_at": (timestamp+pd.Timedelta(hours=12)).isoformat(),
                   "dataset_split": "train", "incident_split": split, "incident_group_id": "group-"+split,
                   "target_newly_burned_12h": int(x == 1), "firms_center_has_detection": int(x == 0),
                   "firms_center_detection_count": 4 if x == 0 else 0,
                   "firms_center_bright_ti4_max": 290. if x == 0 else None,
                   "firms_center_bright_ti4_mean": 280. if x == 0 else None,
                   "firms_center_platform_count": 2 if x == 0 else 0,
                   "firms_center_hours_since_last_detection": 10. if x == 0 else None,
                   **terrain(None)}
            for name in FEATURES:
                row.setdefault(name, float(x == 1))
            rows.append(row)
    return pd.DataFrame(rows)


def model(**kwargs):
    return IncidentTransitionModel(EvidenceClassifier(), feature_columns=FEATURES,
                                   ignition_threshold=.5, **kwargs)


class IncidentTransitionTests(unittest.TestCase):
    def test_preserves_actual_brightness_and_counts_when_observations_age(self):
        m = model()
        state = observed_incident_state(m, frame().iloc[:3])
        self.assertIsInstance(state.active_cells[0], EvidenceCell)
        features = m.candidate_feature_rows(state, terrain_provider=terrain,
                                           include_cell_ids=["naea-1km:x=1:y=0"])[0][1]
        self.assertEqual(features["firms_local_3x3_bright_ti4_max"], 290.)
        self.assertEqual(features["firms_local_3x3_bright_ti4_mean"], 280.)
        self.assertEqual(features["firms_local_3x3_detection_count"], 4.)
        advanced = m.step(state, terrain_provider=terrain)
        survivor = next(c for c in advanced.state.active_cells if c.cell_id == "naea-1km:x=0:y=0")
        self.assertEqual(survivor.observation_age_hours, 22.)
        self.assertEqual(survivor.bright_ti4_max, 290.)
        self.assertEqual(survivor.detection_count, 4)

    def test_distance_candidate_ignition_caps_and_extinction(self):
        m = model(max_new_cells_per_step=1, max_candidates=3)
        state = observed_incident_state(m, frame().iloc[:3])
        result = m.step(state, terrain_provider=terrain)
        self.assertLessEqual(len(result.predictions), 3)
        self.assertLessEqual(sum(p.will_ignite for p in result.predictions), 1)
        for p in result.predictions:
            self.assertNotIn("x=2:", p.cell_id)
        empty = RecursiveFireState(0, ())
        self.assertEqual(m.step(empty, terrain_provider=terrain).state.active_cells, ())
        m = model()
        m.ignition_threshold = 1.
        first = m.step(state, terrain_provider=terrain)
        second = m.step(first.state, terrain_provider=terrain)
        self.assertFalse(second.state.active_cells)
        self.assertIn("naea-1km:x=0:y=0", second.state.burned_cell_ids)

    def test_sampling_endpoints_replace_state_and_are_deterministic(self):
        m = model()
        observed = observed_incident_state(m, frame().iloc[:3], step_index=1)
        predicted = m.step(replace(observed, step_index=0), terrain_provider=terrain).state
        self.assertEqual(mix_observed_and_predicted(observed, predicted, predicted_fraction=0., key="a"), observed)
        # Compare sorted cells because the underlying wrapper sorts by grid index.
        mixed = mix_observed_and_predicted(observed, predicted, predicted_fraction=1., key="a")
        self.assertEqual(set(mixed.active_cells), set(predicted.active_cells))
        self.assertEqual(mixed.burned_cell_ids, predicted.burned_cell_ids)
        self.assertEqual(mix_observed_and_predicted(observed, predicted, predicted_fraction=.5, key="a"),
                         mix_observed_and_predicted(observed, predicted, predicted_fraction=.5, key="a"))


class ScheduledSamplingTests(unittest.TestCase):
    def test_generation_is_bounded_weighted_and_never_reads_holdout_features(self):
        examples = pd.concat([frame(), frame("held_incident")], ignore_index=True)
        sequences = build_incident_sequences(examples, later_test_at="2026-08-01T00:00:00Z")
        policy = SamplingPolicy()
        before, reports = generate_scheduled_examples(model(), examples, sequences, terrain_provider=terrain, policy=policy)
        self.assertGreater(len(before), 0)
        self.assertLessEqual(len(before), 8)
        self.assertTrue(before.sample_weight.eq(.25).all())
        self.assertTrue(before.incident_split.eq("train").all())
        self.assertTrue(before.original_example_id.str.startswith("train-").all())
        generated = [r for r in reports if r["status"] == "generated"]
        self.assertEqual(sorted(set(r["predicted_state_fraction"] for r in generated)), [.25, .5, .75])
        original = examples.set_index("example_id").loc[before.original_example_id]
        self.assertEqual(original.target_newly_burned_12h.tolist(), before.target_newly_burned_12h.tolist())
        for column in [c for c in examples if c.startswith(("firms_", "terrain_"))]:
            examples[column] = examples[column].astype(float)
            examples.loc[examples.incident_split.eq("held_incident"), column] = np.nan
        after, _ = generate_scheduled_examples(model(), examples, sequences, terrain_provider=terrain, policy=policy)
        pd.testing.assert_frame_equal(before, after, check_dtype=False)

    def test_forged_training_sequence_cannot_include_held_out_rows(self):
        examples = frame("held_incident")
        sequences = build_incident_sequences(examples, later_test_at="2026-08-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "held-out"):
            generate_scheduled_examples(model(), examples, [replace(sequences[0], split="train")], terrain_provider=terrain)

    def test_fit_and_calibration_use_disjoint_incidents(self):
        train = frame().loc[lambda x: x.firms_center_has_detection.eq(0)]
        calibration = frame("calibration").loc[lambda x: x.firms_center_has_detection.eq(0)]
        estimator, report = fit_pass(train, calibration)
        self.assertEqual(report["training_rows"], len(train))
        self.assertTrue(np.isfinite(estimator.predict_proba(train[list(FEATURES)].to_numpy())).all())
        calibration.incident_group_id = "group-train"
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_pass(train, calibration)

    def test_second_pass_checks_real_targets_and_only_reads_completed_verified_views(self):
        examples = frame()
        sequences = build_incident_sequences(examples, later_test_at="2026-08-01T00:00:00Z")
        synthetic, _ = generate_scheduled_examples(model(), examples, sequences, terrain_provider=terrain)
        train = examples.loc[examples.firms_center_has_detection.eq(0)]
        refs = train[["example_id", "incident_group_id", "incident_split"]].copy()
        refs["sample_weight"] = 1.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = root / "generator.joblib"
            generator.write_bytes(b"trusted test generator reference")
            observed_path, synthetic_path = root / "observed.csv.gz", root / "synthetic.csv.gz"
            refs.to_csv(observed_path, index=False)
            synthetic.to_csv(synthetic_path, index=False)
            manifest = {"kind": "completed-incident-scheduled-sampling-training-view", "status": "complete",
                        "training_admitted": True, "source_incident_manifest_sha256": "a"*64,
                        "source_generator_model": str(generator), "source_generator_model_sha256": _sha256_file(generator),
                        "policy": asdict(SamplingPolicy()), "observed_row_count": len(refs), "synthetic_row_count": len(synthetic),
                        "observed_artifact": {"path": str(observed_path), "sha256": _sha256_file(observed_path)},
                        "synthetic_artifact": {"path": str(synthetic_path), "sha256": _sha256_file(synthetic_path)}}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            observed, loaded = load_mixed_training_view(path, examples, incident_manifest_sha256="a"*64)
            calibration = frame("calibration").loc[lambda x: x.firms_center_has_detection.eq(0)]
            _, report = fit_pass(observed, calibration, synthetic=loaded)
            self.assertEqual(report["training_rows"], len(train)+len(synthetic))
            loaded.loc[0, "target_newly_burned_12h"] = 1-loaded.loc[0, "target_newly_burned_12h"]
            with self.assertRaisesRegex(ValueError, "lineage/target"):
                fit_pass(observed, calibration, synthetic=loaded)
            manifest["status"] = "partial"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "completed"):
                load_mixed_training_view(path, examples, incident_manifest_sha256="a"*64)
            manifest["status"] = "complete"
            path.write_text(json.dumps(manifest))
            synthetic_path.write_bytes(synthetic_path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_mixed_training_view(path, examples, incident_manifest_sha256="a"*64)


class SpatialEvaluationTests(unittest.TestCase):
    def test_calibration_bins_do_not_overlap_at_decimal_boundaries(self):
        self.assertAlmostEqual(probability_metrics([0], [.3])["ece"], .3)
        self.assertAlmostEqual(probability_metrics([0, 1], [0., 1.])["ece"], 0.)

    def test_front_distances_and_empty_fronts(self):
        a = {"naea-1km:x=0:y=0"}; b = {"naea-1km:x=3:y=0"}
        self.assertEqual(front_distance(a, b)["front_hausdorff_distance_m"], 3000.)
        self.assertEqual(front_distance(a, a)["symmetric_front_mean_distance_m"], 0.)
        self.assertIsNone(front_distance(set(), b)["symmetric_front_mean_distance_m"])

    def test_outside_domain_predictions_remain_unknown_and_area_error_is_scoped(self):
        m = model()
        snapshot = frame().iloc[:3]
        predictions = m.step(observed_incident_state(m, snapshot), terrain_provider=terrain).predictions
        result = score_spatial_horizon(snapshot, predictions, cumulative_predicted=set(), cumulative_actual=set(),
                                       cumulative_domain=set(), initial_active={"naea-1km:x=0:y=0"})
        self.assertGreater(result["outside_domain_candidates"], 0)
        self.assertEqual(result["false_positive"], 0)
        self.assertEqual(result["cumulative_newly_burned_area_error_km2"], -1)

    def test_evaluation_is_fully_open_loop_and_reports_all_four_horizons(self):
        examples = frame("held_incident")
        sequences = build_incident_sequences(examples, later_test_at="2026-08-01T00:00:00Z")
        m = model()
        states = []
        original_step = m.step
        def capture(state, **kwargs):
            states.append(state)
            return original_step(state, **kwargs)
        m.step = capture
        result = evaluate_incident_rollouts(m, examples, sequences, terrain_provider=terrain)
        before = list(states); states.clear()
        examples.loc[3:, "firms_center_has_detection"] = 0
        evaluate_incident_rollouts(m, examples, sequences, terrain_provider=terrain)
        self.assertEqual(states, before)
        self.assertEqual([r["horizon_hours"] for r in result["reports"]], [12, 24, 48, 96])
        different_origin = {**result, "reports": [{**r, "origin": "2026-07-02T00:00:00Z"} for r in result["reports"]]}
        with self.assertRaisesRegex(ValueError, "same origins"):
            compare_passes(result, different_origin)


if __name__ == "__main__":
    unittest.main()
