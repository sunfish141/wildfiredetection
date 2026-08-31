import tempfile
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
        second = model.step(first.state, terrain_provider=terrain)

        self.assertEqual(second.state.active_cells, ())
        self.assertEqual(second.state.burned_cell_ids, ("naea-1km:x=0:y=0",))
        third = model.step(second.state, terrain_provider=terrain)
        self.assertEqual(third.predictions, ())
        self.assertEqual(third.state.burned_cell_ids, second.state.burned_cell_ids)

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


if __name__ == "__main__":
    unittest.main()
