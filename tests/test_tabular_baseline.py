import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from wildfire_data.tabular_baseline import (
    TABULAR_BASELINE_VERSION,
    TabularBaselineError,
    leakage_metadata_columns,
    persist_tabular_baseline,
    train_tabular_baseline,
)


class TabularBaselineTests(unittest.TestCase):
    def _examples(self) -> pd.DataFrame:
        rows = []
        for time_index in range(10):
            anchor = pd.Timestamp("2026-07-01T00:00:00Z") + pd.Timedelta(hours=12 * time_index)
            for cell_index in range(24):
                target = int(cell_index % 4 == 0)
                rows.append(
                    {
                        "example_id": f"example-{time_index}-{cell_index}",
                        "cell_id": f"naea-1km:x={cell_index}:y={time_index}",
                        "anchor_at": anchor.isoformat(),
                        "target_newly_burned_12h": target,
                        "firms_count_trailing_12h": target * 5 + (cell_index % 3),
                        "mean_slope_degrees": float((cell_index + time_index) % 10),
                        "weather_missing_indicator": float(cell_index % 2),
                    }
                )
        frame = pd.DataFrame(rows)
        # HistGradientBoosting handles this natively; the feature contract must
        # retain a missingness count rather than hide it through imputation.
        frame.loc[3, "mean_slope_degrees"] = float("nan")
        return frame.sample(frac=1.0, random_state=8).reset_index(drop=True)

    def _train(self):
        return train_tabular_baseline(
            self._examples(),
            target_column="target_newly_burned_12h",
            feature_columns=(
                "firms_count_trailing_12h",
                "mean_slope_degrees",
                "weather_missing_indicator",
            ),
            split_fraction=0.8,
            calibration_bins=5,
            model_parameters={"max_iter": 30, "min_samples_leaf": 3},
        )

    def test_trains_on_explicit_features_and_holds_out_later_anchor_times(self):
        result = self._train()

        self.assertIsInstance(result.model, HistGradientBoostingClassifier)
        self.assertEqual(
            result.feature_contract.feature_columns,
            (
                "firms_count_trailing_12h",
                "mean_slope_degrees",
                "weather_missing_indicator",
            ),
        )
        self.assertLess(
            result.feature_contract.training_anchor_end_at,
            result.feature_contract.evaluation_anchor_start_at,
        )
        self.assertEqual(result.feature_contract.training_missing_value_counts["mean_slope_degrees"], 1)
        self.assertEqual(
            sum(bin_record.count for bin_record in result.metrics.calibration_bins),
            result.feature_contract.evaluation_row_count,
        )
        self.assertGreaterEqual(result.metrics.roc_auc, 0.0)
        self.assertLessEqual(result.metrics.roc_auc, 1.0)
        self.assertGreaterEqual(result.metrics.pr_auc, 0.0)
        self.assertLessEqual(result.metrics.pr_auc, 1.0)
        self.assertGreaterEqual(result.metrics.brier_score, 0.0)

    def test_split_group_keeps_every_source_snapshot_on_one_side_of_holdout(self):
        """Split logical FEDS snapshots, not individually adjusted cell times."""
        rows = []
        base = pd.Timestamp("2026-07-01T00:00:00Z")
        # The uneven row counts make an accidental row-level/anchor-level
        # split observable.  With five source snapshots and a 60% split, the
        # first three whole snapshots contain 12 rows; splitting their 23
        # distinct cell anchor times would instead put 13 rows in training.
        for snapshot_index, row_count in enumerate((3, 5, 4, 6, 5)):
            source_snapshot = base + pd.Timedelta(hours=12 * snapshot_index)
            for cell_index in range(row_count):
                target = int(cell_index % 2 == 0)
                rows.append(
                    {
                        "example_id": f"group-{snapshot_index}-cell-{cell_index}",
                        "cell_id": f"naea-1km:x={cell_index}:y={snapshot_index}",
                        # In the real table this time is cell-local, so all
                        # cells from one FEDS snapshot must be split together.
                        "anchor_at": (
                            source_snapshot + pd.Timedelta(minutes=cell_index + 1)
                        ).isoformat(),
                        "source_snapshot_time": source_snapshot.isoformat(),
                        "target_newly_burned_12h": target,
                        "firms_count_trailing_12h": float(3 * target + cell_index),
                        "mean_slope_degrees": float(snapshot_index + cell_index),
                    }
                )
        frame = pd.DataFrame(rows).sample(frac=1.0, random_state=31).reset_index(drop=True)
        common_arguments = {
            "target_column": "target_newly_burned_12h",
            "feature_columns": ("firms_count_trailing_12h", "mean_slope_degrees"),
            "split_fraction": 0.6,
            "calibration_bins": 3,
            "model_parameters": {"max_iter": 10, "min_samples_leaf": 2},
        }

        grouped = train_tabular_baseline(
            frame,
            split_group_column="source_snapshot_time",
            **common_arguments,
        )
        anchor_split = train_tabular_baseline(frame, **common_arguments)

        self.assertEqual(grouped.feature_contract.split_group_column, "source_snapshot_time")
        self.assertEqual(
            grouped.feature_contract.chronological_split_cutoff_at,
            "2026-07-02T00:00:00Z",
        )
        self.assertEqual(grouped.feature_contract.training_row_count, 12)
        self.assertEqual(grouped.feature_contract.evaluation_row_count, 11)
        self.assertEqual(anchor_split.feature_contract.split_group_column, "anchor_at")
        self.assertEqual(anchor_split.feature_contract.training_row_count, 13)

    def test_rejects_label_source_time_and_identifier_metadata_as_features(self):
        frame = self._examples()
        frame["label_quality_score"] = 0.7
        frame["feds_event_id"] = "feds-1"

        self.assertEqual(
            leakage_metadata_columns(
                ("firms_count_trailing_12h", "label_quality_score", "feds_event_id", "cell_id")
            ),
            ("label_quality_score", "feds_event_id", "cell_id"),
        )
        with self.assertRaisesRegex(TabularBaselineError, "leakage or metadata"):
            train_tabular_baseline(
                frame,
                target_column="target_newly_burned_12h",
                feature_columns=("firms_count_trailing_12h", "label_quality_score"),
                model_parameters={"max_iter": 10, "min_samples_leaf": 3},
            )
        with self.assertRaisesRegex(TabularBaselineError, "anchor_column"):
            train_tabular_baseline(
                frame,
                target_column="target_newly_burned_12h",
                feature_columns=("firms_count_trailing_12h", "anchor_at"),
                model_parameters={"max_iter": 10, "min_samples_leaf": 3},
            )

    def test_rejects_a_chronological_holdout_that_has_only_one_target_class(self):
        frame = self._examples()
        latest_anchor = frame["anchor_at"].max()
        frame.loc[frame["anchor_at"] == latest_anchor, "target_newly_burned_12h"] = 1

        with self.assertRaisesRegex(TabularBaselineError, "evaluation split must contain both"):
            train_tabular_baseline(
                frame,
                target_column="target_newly_burned_12h",
                feature_columns=("firms_count_trailing_12h", "mean_slope_degrees"),
                split_fraction=0.9,
                model_parameters={"max_iter": 10, "min_samples_leaf": 3},
            )

    def test_persists_a_joblib_bundle_and_readable_feature_contract(self):
        result = self._train()

        with tempfile.TemporaryDirectory() as directory:
            persisted = persist_tabular_baseline(result, directory, basename="feds-baseline")

            self.assertTrue(persisted.model_bundle_path.exists())
            self.assertTrue(persisted.feature_contract_path.exists())
            self.assertTrue(persisted.metrics_path.exists())
            bundle = joblib.load(persisted.model_bundle_path)
            self.assertEqual(bundle["baseline_version"], TABULAR_BASELINE_VERSION)
            self.assertEqual(
                bundle["feature_contract"]["feature_columns"],
                ["firms_count_trailing_12h", "mean_slope_degrees", "weather_missing_indicator"],
            )
            contract = json.loads(Path(persisted.feature_contract_path).read_text(encoding="utf-8"))
            metrics = json.loads(Path(persisted.metrics_path).read_text(encoding="utf-8"))
            self.assertEqual(contract["split_group_column"], "anchor_at")
            self.assertEqual(contract["evaluation_row_count"], result.metrics.evaluation_row_count)
            self.assertEqual(metrics["evaluation_row_count"], result.metrics.evaluation_row_count)


if __name__ == "__main__":
    unittest.main()
