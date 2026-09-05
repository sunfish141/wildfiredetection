import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_data.recursive_transition import (
    RECURSIVE_MODEL_FEATURE_COLUMNS,
    RecursiveTransitionModel,
)
from wildfire_data.rollout_augmentation import (
    RolloutAugmentationError,
    generate_one_step_augmentation,
    persist_rollout_augmentation,
    compare_renderer_features,
)
from wildfire_data.recursive_calibration import RecursiveCalibrationError
from wildfire_data.rollout_sequences import build_rollout_sequences


class LocalEvidenceClassifier:
    def predict_proba(self, values):
        index = RECURSIVE_MODEL_FEATURE_COLUMNS.index(
            "firms_local_3x3_active_cell_count"
        )
        positive = np.where(np.asarray(values)[:, index] > 0, 0.8, 0.1)
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


class RolloutAugmentationTests(unittest.TestCase):
    def _model(self):
        return RecursiveTransitionModel(
            LocalEvidenceClassifier(),
            feature_columns=RECURSIVE_MODEL_FEATURE_COLUMNS,
            ignition_threshold=0.5,
        )

    def _examples(self):
        rows = []
        start = pd.Timestamp("2026-07-01T00:00:00Z")
        for snapshot_index in range(3):
            source = start + pd.Timedelta(hours=12 * snapshot_index)
            target = source + pd.Timedelta(hours=12)
            split = "train" if snapshot_index < 2 else "validation"
            specifications = (
                (f"naea-1km:x={snapshot_index}:y=0", 1, 0, 367.0),
                (f"naea-1km:x={snapshot_index + 1}:y=0", 0, 1, None),
                (f"naea-1km:x=100:y={snapshot_index}", 0, 1, None),
                (f"naea-1km:x=101:y={snapshot_index}", 0, 0, None),
            )
            for row_index, (cell_id, detected, label, brightness) in enumerate(specifications):
                anchor = source + pd.Timedelta(minutes=row_index)
                row = {
                    "source_snapshot_time": source.isoformat(),
                    "target_snapshot_time": target.isoformat() if label else None,
                    "anchor_at": anchor.isoformat(),
                    "target_end_at": (anchor + pd.Timedelta(hours=12)).isoformat(),
                    "cell_id": cell_id,
                    "example_id": f"example-{snapshot_index}-{row_index}",
                    "dataset_split": split,
                    "target_newly_burned_12h": label,
                    "firms_center_has_detection": detected,
                    "firms_center_bright_ti4_max": brightness,
                    "firms_center_hours_since_last_detection": 11.0 if detected else None,
                    "firms_center_detection_count": detected * 3,
                    "firms_center_platform_count": detected,
                    **terrain(cell_id),
                }
                for name in RECURSIVE_MODEL_FEATURE_COLUMNS:
                    row.setdefault(name, 1.0)
                rows.append(row)
        return pd.DataFrame(rows)

    def test_generates_only_reached_next_snapshot_training_rows(self):
        examples = self._examples()
        result = generate_one_step_augmentation(
            self._model(),
            examples,
            build_rollout_sequences(examples),
            terrain_provider=terrain,
        )

        self.assertEqual(result.pair_count, 1)
        self.assertEqual(result.examples["cell_id"].tolist(), ["naea-1km:x=2:y=0"])
        self.assertEqual(result.examples["target_newly_burned_12h"].tolist(), [1])
        self.assertEqual(
            result.examples["synthetic_feature_source_snapshot_time"].tolist(),
            ["2026-07-01T12:00:00Z"],
        )
        report = result.pair_reports[0]
        self.assertEqual(report.historical_frontier_row_count, 3)
        self.assertEqual(report.matched_row_count, 1)
        self.assertEqual(report.unreached_positive_count, 1)
        self.assertGreater(report.candidates_outside_historical_frontier, 0)

    def test_persists_checksum_and_keeps_training_admission_false(self):
        examples = self._examples()
        result = generate_one_step_augmentation(
            self._model(),
            examples,
            build_rollout_sequences(examples),
            terrain_provider=terrain,
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            model_path.write_bytes(b"test source model")
            manifest = persist_rollout_augmentation(
                result,
                Path(directory) / "inspection",
                release_manifest_sha256="a" * 64,
                model_bundle_path=str(model_path),
            )
            path = Path(directory) / "inspection" / "synthetic_frontier_examples.csv.gz"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            self.assertEqual(manifest["artifact"]["file_sha256"], digest)
            self.assertFalse(manifest["training_admitted"])
            self.assertEqual(len(pd.read_csv(path)), 1)
            with self.assertRaisesRegex(RolloutAugmentationError, "immutable"):
                persist_rollout_augmentation(result, path.parent,
                    release_manifest_sha256="a" * 64, model_bundle_path=str(model_path))

    def test_refuses_a_split_without_consecutive_pairs(self):
        examples = self._examples()
        examples.loc[examples["dataset_split"] == "train", "dataset_split"] = "validation"
        with self.assertRaisesRegex(RecursiveCalibrationError, "no training"):
            generate_one_step_augmentation(
                self._model(),
                examples,
                build_rollout_sequences(examples),
                terrain_provider=terrain,
                split_name="train",
            )

    def test_validation_augmentation_and_missing_split_are_rejected(self):
        examples = self._examples()
        with self.assertRaisesRegex(RolloutAugmentationError, "restricted"):
            generate_one_step_augmentation(self._model(), examples,
                build_rollout_sequences(examples), terrain_provider=terrain, split_name="validation")
        examples.loc[0, "dataset_split"] = None
        with self.assertRaisesRegex(RecursiveCalibrationError, "missing"):
            generate_one_step_augmentation(self._model(), examples,
                build_rollout_sequences(examples), terrain_provider=terrain)

    def test_renderer_screen_detects_recency_and_missingness_drift(self):
        observed = pd.DataFrame({name: [3., 10., 17., 24.] for name in RECURSIVE_MODEL_FEATURE_COLUMNS})
        self.assertTrue(compare_renderer_features(observed, observed)["passed"])
        synthetic = observed.copy()
        synthetic["firms_local_3x3_hours_since_last_detection"] = 0.
        self.assertFalse(compare_renderer_features(synthetic, observed)["passed"])
        synthetic = observed.copy()
        synthetic.loc[0, "firms_local_3x3_bright_ti4_mean"] = np.nan
        self.assertFalse(compare_renderer_features(synthetic, observed)["passed"])
        self.assertFalse(compare_renderer_features(observed.iloc[:0], observed.iloc[:0])["passed"])

    def test_validation_feature_changes_cannot_change_generated_training_rows(self):
        examples = self._examples()
        before = generate_one_step_augmentation(self._model(), examples,
            build_rollout_sequences(examples), terrain_provider=terrain)
        columns = [name for name in examples if name.startswith(("firms_", "terrain_"))]
        examples = examples.astype({name: float for name in columns})
        # A future snapshot may have arbitrary missing feature evidence: no
        # augmentation computation may touch it, even for an error check.
        examples.loc[examples.dataset_split == "validation", columns] = np.nan
        after = generate_one_step_augmentation(self._model(), examples,
            build_rollout_sequences(examples), terrain_provider=terrain)
        pd.testing.assert_frame_equal(before.examples, after.examples)
        self.assertEqual(before.pair_reports, after.pair_reports)


if __name__ == "__main__":
    unittest.main()
