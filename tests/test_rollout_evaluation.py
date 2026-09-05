import unittest

import numpy as np
import pandas as pd

from wildfire_data.recursive_transition import (
    RECURSIVE_MODEL_FEATURE_COLUMNS,
    RecursiveTransitionModel,
)
from wildfire_data.rollout_evaluation import (
    RolloutEvaluationError,
    evaluate_open_loop,
    first_split_origin,
    initial_state_from_observed_firms,
)
from wildfire_data.rollout_sequences import build_rollout_sequences


class LocalEvidenceClassifier:
    def predict_proba(self, values):
        active_index = RECURSIVE_MODEL_FEATURE_COLUMNS.index(
            "firms_local_3x3_active_cell_count"
        )
        positive = np.where(np.asarray(values)[:, active_index] > 0, 0.8, 0.1)
        return np.column_stack((1.0 - positive, positive))


def terrain(_cell_id):
    return {
        "terrain_valid": True,
        "terrain_elevation_m": 500.0,
        "terrain_slope_degrees": 5.0,
        "terrain_aspect_defined": True,
        "terrain_aspect_sin": 0.0,
        "terrain_aspect_cos": 1.0,
    }


class RolloutEvaluationTests(unittest.TestCase):
    def _model(self):
        return RecursiveTransitionModel(
            LocalEvidenceClassifier(),
            feature_columns=RECURSIVE_MODEL_FEATURE_COLUMNS,
            ignition_threshold=0.5,
        )

    def _examples(self, snapshot_count=3):
        rows = []
        start = pd.Timestamp("2026-08-03T00:00:00Z")
        for snapshot_index in range(snapshot_count):
            source = start + pd.Timedelta(hours=12 * snapshot_index)
            target = source + pd.Timedelta(hours=12)
            specifications = (
                (f"naea-1km:x={snapshot_index}:y=0", 1, 0, 367.0),
                (f"naea-1km:x={snapshot_index + 1}:y=0", 0, 1, None),
                (f"naea-1km:x=100:y={snapshot_index}", 0, 1, None),
                (f"naea-1km:x=101:y={snapshot_index}", 0, 0, None),
            )
            for cell_offset, (cell_id, detected, label, brightness) in enumerate(specifications):
                anchor = source + pd.Timedelta(minutes=cell_offset)
                rows.append(
                    {
                        "source_snapshot_time": source.isoformat(),
                        "target_snapshot_time": target.isoformat() if label else None,
                        "anchor_at": anchor.isoformat(),
                        "target_end_at": (anchor + pd.Timedelta(hours=12)).isoformat(),
                        "cell_id": cell_id,
                        "dataset_split": "validation",
                        "target_newly_burned_12h": label,
                        "firms_center_has_detection": detected,
                        "firms_center_bright_ti4_max": brightness,
                        "firms_center_hours_since_last_detection": 11.0 if detected else None,
                    }
                )
        return pd.DataFrame(rows)

    def test_scores_predictions_only_inside_frontier_domain_and_counts_missed_coverage(self):
        examples = self._examples()
        sequence = build_rollout_sequences(examples)[0]

        evaluation = evaluate_open_loop(
            self._model(),
            examples,
            sequence,
            start_snapshot_index=0,
            terrain_provider=terrain,
            horizons=(1,),
        )

        self.assertEqual(evaluation.initial_active_cell_count, 1)
        metrics = evaluation.horizons[0]
        self.assertEqual(metrics.horizon_hours, 12)
        self.assertEqual(metrics.evaluation_row_count, 3)
        self.assertEqual(metrics.evaluation_positive_count, 2)
        self.assertEqual(metrics.candidates_inside_evaluation_domain, 1)
        self.assertGreater(metrics.candidates_outside_evaluation_domain, 0)
        self.assertEqual(metrics.true_positive_count, 1)
        self.assertEqual(metrics.false_negative_count, 1)
        self.assertEqual(metrics.false_positive_count, 0)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 0.5)

    def test_initial_state_converts_observed_brightness_to_bounded_intensity(self):
        examples = self._examples().iloc[:4].copy()
        state = initial_state_from_observed_firms(self._model(), examples)

        self.assertEqual(len(state.active_cells), 1)
        self.assertEqual(state.active_cells[0].intensity, 1.0)
        self.assertEqual(state.active_cells[0].observation_age_hours, 11.0)

    def test_observed_initialization_rejects_missing_and_unavailable_recency(self):
        examples = self._examples().iloc[:4].copy()
        for age in (None, float("inf"), 2.9, 24.1):
            with self.subTest(age=age):
                examples.loc[0, "firms_center_hours_since_last_detection"] = age
                with self.assertRaisesRegex(RolloutEvaluationError, "ages"):
                    initial_state_from_observed_firms(self._model(), examples)

    def test_finds_first_long_enough_split_run(self):
        examples = self._examples(snapshot_count=4)
        examples.loc[:3, "dataset_split"] = "train"
        sequences = build_rollout_sequences(examples)

        sequence, start = first_split_origin(
            examples, sequences, split_name="validation", required_snapshot_count=3
        )

        self.assertIs(sequence, sequences[0])
        self.assertEqual(start, 1)

    def test_rejects_insufficient_horizon_and_mixed_split_snapshot(self):
        examples = self._examples(snapshot_count=2)
        sequence = build_rollout_sequences(examples)[0]
        with self.assertRaisesRegex(RolloutEvaluationError, "every requested rollout horizon"):
            evaluate_open_loop(
                self._model(),
                examples,
                sequence,
                start_snapshot_index=0,
                terrain_provider=terrain,
                horizons=(1, 4),
            )

        examples.loc[0, "dataset_split"] = "train"
        with self.assertRaisesRegex(RolloutEvaluationError, "entirely"):
            evaluate_open_loop(
                self._model(),
                examples,
                sequence,
                start_snapshot_index=0,
                terrain_provider=terrain,
                horizons=(1,),
            )


if __name__ == "__main__":
    unittest.main()
