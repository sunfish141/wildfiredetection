import tempfile
import json
import unittest
from pathlib import Path

import joblib
import numpy as np

from wildfire_data.recursive_transition import (
    ActiveFireCell,
    RECURSIVE_MODEL_FEATURE_COLUMNS,
    RecursiveFireState,
    RecursiveTransitionError,
    RecursiveTransitionModel,
    SyntheticObservationCalibration,
)


class LocalEvidenceClassifier:
    def __init__(self):
        self.last_values = None
        self.local_active_index = RECURSIVE_MODEL_FEATURE_COLUMNS.index(
            "firms_local_3x3_active_cell_count"
        )

    def predict_proba(self, values):
        self.last_values = np.asarray(values)
        positive = np.where(self.last_values[:, self.local_active_index] > 0, 0.8, 0.1)
        return np.column_stack((1.0 - positive, positive))


def terrain(_cell_id):
    return {
        "terrain_valid": True,
        "terrain_elevation_m": 800.0,
        "terrain_slope_degrees": 12.0,
        "terrain_aspect_defined": True,
        "terrain_aspect_sin": 0.6,
        "terrain_aspect_cos": 0.8,
    }


class RecursiveTransitionTests(unittest.TestCase):
    def _model(self, **parameters):
        return RecursiveTransitionModel(
            LocalEvidenceClassifier(),
            feature_columns=RECURSIVE_MODEL_FEATURE_COLUMNS,
            **parameters,
        )

    def test_one_step_renders_firms_compatible_features_and_ignites_neighbours(self):
        model = self._model(ignition_threshold=0.5, active_duration_steps=2)
        state = model.initial_state({"naea-1km:x=10:y=20": 1.0})

        result = model.step(state, terrain_provider=terrain)

        self.assertEqual(len(result.predictions), 24)
        self.assertEqual(sum(item.will_ignite for item in result.predictions), 8)
        self.assertEqual(len(result.state.active_cells), 9)
        self.assertEqual(result.state.burned_cell_ids, ())
        self.assertEqual(result.state.step_index, 1)
        neighbour = next(
            item for item in result.predictions if item.cell_id == "naea-1km:x=11:y=20"
        )
        self.assertEqual(neighbour.ignition_probability, 0.8)
        self.assertAlmostEqual(neighbour.next_intensity, 0.85)

        values = model.estimator.last_values
        local_count_index = RECURSIVE_MODEL_FEATURE_COLUMNS.index(
            "firms_local_3x3_detection_count"
        )
        local_brightness_index = RECURSIVE_MODEL_FEATURE_COLUMNS.index(
            "firms_local_3x3_bright_ti4_max"
        )
        self.assertIn(3.0, values[:, local_count_index])
        self.assertIn(367.0, values[:, local_brightness_index])

    def test_active_cells_decay_then_become_burned_and_cannot_reignite(self):
        model = self._model(ignition_threshold=1.0, active_duration_steps=2)
        initial = model.initial_state({"naea-1km:x=0:y=0": 0.8})

        first = model.step(initial, terrain_provider=terrain)
        self.assertEqual(first.state.active_cells[0].remaining_active_steps, 1)
        self.assertAlmostEqual(first.state.active_cells[0].intensity, 0.68)
        self.assertEqual(first.state.active_cells[0].observation_age_hours, 19.5)
        second = model.step(first.state, terrain_provider=terrain)

        self.assertEqual(second.state.active_cells, ())
        self.assertEqual(second.state.burned_cell_ids, ("naea-1km:x=0:y=0",))
        third = model.step(second.state, terrain_provider=terrain)
        self.assertEqual(third.predictions, ())
        self.assertEqual(third.state.burned_cell_ids, second.state.burned_cell_ids)

    def test_candidate_feature_rows_filters_only_after_deriving_the_frontier(self):
        model = self._model()
        state = model.initial_state({"naea-1km:x=0:y=0": 1.0})

        rows = model.candidate_feature_rows(
            state,
            terrain_provider=terrain,
            include_cell_ids=("naea-1km:x=1:y=0", "naea-1km:x=20:y=20"),
        )

        self.assertEqual([cell.cell_id for cell, _features in rows], ["naea-1km:x=1:y=0"])
        self.assertEqual(tuple(rows[0][1]), RECURSIVE_MODEL_FEATURE_COLUMNS)

    def test_initial_points_merge_inside_one_cell_using_strongest_intensity(self):
        model = self._model()
        state = model.initial_state_from_points(
            ((53.5461, -113.4938, 0.2), (53.5461, -113.4938, 0.9))
        )
        self.assertEqual(len(state.active_cells), 1)
        self.assertEqual(state.active_cells[0].intensity, 0.9)

    def test_loads_a_trusted_baseline_bundle_and_rejects_wrong_feature_contract(self):
        bundle = {
            "model": LocalEvidenceClassifier(),
            "feature_contract": {"feature_columns": list(RECURSIVE_MODEL_FEATURE_COLUMNS)},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(bundle, path)
            loaded = RecursiveTransitionModel.from_model_bundle(path)
        self.assertEqual(loaded.feature_columns, RECURSIVE_MODEL_FEATURE_COLUMNS)

        with self.assertRaisesRegex(RecursiveTransitionError, "feature contract"):
            RecursiveTransitionModel(
                LocalEvidenceClassifier(),
                feature_columns=("terrain_elevation_m",),
            )

    def test_rejects_invalid_or_overlapping_state(self):
        with self.assertRaisesRegex(RecursiveTransitionError, "between zero and one"):
            ActiveFireCell("naea-1km:x=0:y=0", 1.1, 1)
        with self.assertRaisesRegex(RecursiveTransitionError, "both active and burned"):
            RecursiveFireState(
                step_index=0,
                active_cells=(ActiveFireCell("naea-1km:x=0:y=0", 0.5, 1),),
                burned_cell_ids=("naea-1km:x=0:y=0",),
            )

    def test_observation_age_advances_and_only_eligible_evidence_is_rendered(self):
        model = self._model(ignition_threshold=1.0, active_duration_steps=3)
        cell_id = "naea-1km:x=0:y=0"
        state = model.initial_state({cell_id: .8}, observation_ages={cell_id: 23.})
        next_state = model.step(state, terrain_provider=terrain).state
        self.assertEqual(next_state.active_cells[0].observation_age_hours, 35.)
        rows = model.candidate_feature_rows(next_state, terrain_provider=terrain)
        self.assertTrue(rows)
        for _cell, features in rows:
            self.assertEqual(features["firms_local_3x3_detection_count"], 0)
            self.assertIsNone(features["firms_local_3x3_hours_since_last_detection"])

        for age, eligible in ((2.9, False), (3., True), (24., True), (24.1, False)):
            state = model.initial_state({cell_id: .8}, observation_ages={cell_id: age})
            features = dict((cell.cell_id, values) for cell, values in
                            model.candidate_feature_rows(state, terrain_provider=terrain))
            local = features["naea-1km:x=1:y=0"]
            self.assertEqual(local["firms_local_3x3_has_detection"], float(eligible))
            self.assertEqual(local["firms_local_3x3_hours_since_last_detection"], age if eligible else None)

    def test_new_ignitions_use_documented_recency_and_invalid_ages_fail(self):
        model = self._model(ignition_threshold=.5)
        state = model.initial_state({"naea-1km:x=0:y=0": 1.})
        result = model.step(state, terrain_provider=terrain)
        self.assertTrue(all(cell.observation_age_hours == 7.5 for cell in result.state.active_cells
                            if cell.cell_id != "naea-1km:x=0:y=0"))
        for age in (-1., float("nan"), float("inf")):
            with self.assertRaises(RecursiveTransitionError):
                ActiveFireCell("naea-1km:x=0:y=0", 1., 1, age)

    def test_calibrated_counts_recency_and_contract_round_trip(self):
        calibration = SyntheticObservationCalibration(
            (2., 4., 6., 8., 10.), (1., 1., 2., 2., 3.), 100,
            ("2026-07-01T00:00:00Z",), "a" * 64,
        )
        model = self._model(observation_calibration=calibration)
        state = model.initial_state({"naea-1km:x=0:y=0": 1., "naea-1km:x=2:y=0": .2},
            observation_ages={"naea-1km:x=0:y=0": 20., "naea-1km:x=2:y=0": 11.})
        rows = model.candidate_feature_rows(state, terrain_provider=terrain,
                                          include_cell_ids=["naea-1km:x=1:y=0"])
        features = rows[0][1]
        self.assertEqual(features["firms_local_3x3_detection_count"], 14.)
        self.assertEqual(features["firms_local_3x3_platform_count"], 3.)
        self.assertEqual(features["firms_local_3x3_hours_since_last_detection"], 11.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            bundle = {"model": LocalEvidenceClassifier(), "feature_contract": {
                "feature_columns": RECURSIVE_MODEL_FEATURE_COLUMNS,
                "split_group_column": "source_snapshot_time",
                "chronological_split_cutoff_at": "2026-07-01T00:00:00Z",
            }}
            joblib.dump(bundle, path)
            loaded = RecursiveTransitionModel.from_model_bundle(path,
                renderer_contract=json.loads(json.dumps(model.transition_contract())))
            self.assertEqual(loaded.transition_contract(), model.transition_contract())
            self.assertEqual(loaded.candidate_feature_rows(state, terrain_provider=terrain),
                             model.candidate_feature_rows(state, terrain_provider=terrain))
            bundle["feature_contract"]["chronological_split_cutoff_at"] = "2026-08-01T00:00:00Z"
            joblib.dump(bundle, path)
            with self.assertRaisesRegex(RecursiveTransitionError, "boundary differ"):
                RecursiveTransitionModel.from_model_bundle(path, observation_calibration=calibration)


if __name__ == "__main__":
    unittest.main()
