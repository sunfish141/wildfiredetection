import unittest

import pandas as pd

from wildfire_data.recursive_calibration import RecursiveCalibrationError, fit_observation_calibration
from wildfire_data.recursive_transition import SyntheticObservationCalibration


class RecursiveCalibrationTests(unittest.TestCase):
    def examples(self):
        return pd.DataFrame({
            "source_snapshot_time": ["2026-07-01T00:00:00Z"] * 2 + ["2026-07-02T00:00:00Z"],
            "dataset_split": ["train", "train", "validation"],
            "firms_center_has_detection": [1, 1, 1],
            "firms_center_bright_ti4_max": [305., 367., 330.],
            "firms_center_detection_count": [2., 10., 50.],
            "firms_center_platform_count": [1., 3., 3.],
            "firms_center_hours_since_last_detection": [10., 20., 15.],
        })

    def test_fits_counts_and_empty_bin_fallback_only_from_training(self):
        examples = self.examples()
        calibrated = fit_observation_calibration(examples, release_manifest_sha256="a" * 64)
        self.assertEqual(calibrated.detection_count_by_bin, (2., 6., 6., 6., 10.))
        self.assertEqual(calibrated.counts(1.), (10, 3))
        self.assertEqual(calibrated.training_row_count, 2)
        examples.loc[2, "firms_center_detection_count"] = float("nan")
        examples.loc[2, "firms_center_bright_ti4_max"] = -10000
        self.assertEqual(fit_observation_calibration(examples, release_manifest_sha256="a" * 64), calibrated)
        self.assertEqual(SyntheticObservationCalibration(**calibrated.__dict__), calibrated)

    def test_rejects_mixed_nonchronological_or_missing_splits(self):
        for column, value in (("source_snapshot_time", "2026-07-01T00:00:00Z"),
                              ("source_snapshot_time", "2026-06-01T00:00:00Z"),
                              ("dataset_split", None)):
            examples = self.examples()
            examples.loc[2, column] = value
            with self.assertRaises(RecursiveCalibrationError):
                fit_observation_calibration(examples, release_manifest_sha256="a" * 64)

    def test_rejects_training_counts_or_ages_outside_observation_contract(self):
        for column, value in (("firms_center_detection_count", 1.5),
                              ("firms_center_platform_count", 4.),
                              ("firms_center_hours_since_last_detection", 0.)):
            examples = self.examples()
            examples.loc[0, column] = value
            with self.assertRaises(RecursiveCalibrationError):
                fit_observation_calibration(examples, release_manifest_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
