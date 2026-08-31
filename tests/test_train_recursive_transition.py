import unittest

import pandas as pd

from wildfire_data.recursive_transition import RECURSIVE_MODEL_FEATURE_COLUMNS
from wildfire_data.tabular_baseline import TabularBaselineError
from wildfire_data.train_recursive_transition import train_recursive_frontier_baseline


class TrainRecursiveTransitionTests(unittest.TestCase):
    def _examples(self):
        rows = []
        for snapshot in range(5):
            source_time = pd.Timestamp("2026-06-01T00:00:00Z") + pd.Timedelta(
                hours=12 * snapshot
            )
            for index in range(8):
                target = index % 2
                row = {
                    "anchor_at": (source_time + pd.Timedelta(minutes=index)).isoformat(),
                    "source_snapshot_time": source_time.isoformat(),
                    "target_newly_burned_12h": target,
                    "firms_center_has_detection": int(index == 7),
                }
                for feature_index, name in enumerate(RECURSIVE_MODEL_FEATURE_COLUMNS):
                    row[name] = float(target * 3 + snapshot + index + feature_index / 10)
                rows.append(row)
        return pd.DataFrame(rows)

    def test_filters_center_detections_and_keeps_whole_chronological_groups(self):
        result = train_recursive_frontier_baseline(self._examples())

        self.assertEqual(result.source_row_count, 40)
        self.assertEqual(result.frontier_row_count, 35)
        self.assertEqual(result.excluded_center_detection_row_count, 5)
        self.assertEqual(
            result.baseline.feature_contract.feature_columns,
            RECURSIVE_MODEL_FEATURE_COLUMNS,
        )
        self.assertEqual(
            result.baseline.feature_contract.split_group_column,
            "source_snapshot_time",
        )

    def test_rejects_missing_or_nonbinary_filter(self):
        examples = self._examples().drop(columns=["firms_center_has_detection"])
        with self.assertRaisesRegex(TabularBaselineError, "row-filter column"):
            train_recursive_frontier_baseline(examples)

        examples = self._examples()
        examples.loc[0, "firms_center_has_detection"] = 2
        with self.assertRaisesRegex(TabularBaselineError, "binary 0/1"):
            train_recursive_frontier_baseline(examples)


if __name__ == "__main__":
    unittest.main()
