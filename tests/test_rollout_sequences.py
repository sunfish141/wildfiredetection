import unittest

import pandas as pd

from wildfire_data.rollout_sequences import (
    ROLLOUT_SEQUENCE_VERSION,
    RolloutSequenceError,
    build_rollout_sequences,
    snapshot_frame,
)


class RolloutSequenceTests(unittest.TestCase):
    def _rows(self, snapshot_hours=(0, 12, 24)):
        rows = []
        start = pd.Timestamp("2026-06-01T00:00:00Z")
        for snapshot_index, hours in enumerate(snapshot_hours):
            source = start + pd.Timedelta(hours=hours)
            target = source + pd.Timedelta(hours=12)
            for cell_index in range(2):
                anchor = source + pd.Timedelta(minutes=20 * cell_index)
                rows.append(
                    {
                        "source_snapshot_time": source.isoformat(),
                        "target_snapshot_time": target.isoformat(),
                        "anchor_at": anchor.isoformat(),
                        "target_end_at": (anchor + pd.Timedelta(hours=12)).isoformat(),
                        "cell_id": f"naea-1km:x={cell_index}:y={snapshot_index}",
                        "value": f"snapshot-{snapshot_index}-cell-{cell_index}",
                    }
                )
        return pd.DataFrame(rows, index=range(100, 100 + len(rows)))

    def test_builds_one_sequence_and_preserves_cell_specific_anchor_rows(self):
        examples = self._rows()

        sequences = build_rollout_sequences(examples)

        self.assertEqual(ROLLOUT_SEQUENCE_VERSION, "feds-snapshot-rollout-sequences/v1")
        self.assertEqual(len(sequences), 1)
        sequence = sequences[0]
        self.assertEqual(sequence.sequence_index, 0)
        self.assertEqual(sequence.transition_count, 3)
        self.assertEqual(sequence.row_count, 6)
        self.assertEqual(sequence.snapshots[0].row_positions, (0, 1))
        self.assertEqual(
            sequence.snapshots[0].anchor_end_at - sequence.snapshots[0].anchor_start_at,
            pd.Timedelta(minutes=20),
        )
        selected = snapshot_frame(examples, sequence.snapshots[1])
        self.assertEqual(selected["value"].tolist(), ["snapshot-1-cell-0", "snapshot-1-cell-1"])
        self.assertEqual(selected.index.tolist(), [102, 103])

    def test_splits_sequences_at_missing_twelve_hour_snapshots(self):
        sequences = build_rollout_sequences(self._rows(snapshot_hours=(0, 12, 36, 48, 84)))

        self.assertEqual([sequence.transition_count for sequence in sequences], [2, 2, 1])
        self.assertEqual([sequence.sequence_index for sequence in sequences], [0, 1, 2])

    def test_rejects_non_twelve_hour_cadence_and_invalid_targets(self):
        with self.assertRaisesRegex(RolloutSequenceError, "whole multiple of 12 hours"):
            build_rollout_sequences(self._rows(snapshot_hours=(0, 18)))

        examples = self._rows()
        examples.loc[100, "target_snapshot_time"] = "2026-06-01T18:00:00Z"
        with self.assertRaisesRegex(RolloutSequenceError, "target_snapshot_time"):
            build_rollout_sequences(examples)

        examples = self._rows()
        examples.loc[100, "target_end_at"] = "2026-06-01T13:00:00Z"
        with self.assertRaisesRegex(RolloutSequenceError, "target_end_at"):
            build_rollout_sequences(examples)

    def test_allows_unobserved_target_snapshot_time_for_weak_negative_rows(self):
        examples = self._rows()
        examples.loc[100, "target_snapshot_time"] = None

        sequences = build_rollout_sequences(examples)

        self.assertEqual(
            sequences[0].snapshots[0].target_snapshot_time,
            pd.Timestamp("2026-06-01T12:00:00Z"),
        )

    def test_rejects_duplicate_cells_within_one_snapshot(self):
        examples = self._rows()
        examples.loc[101, "cell_id"] = examples.loc[100, "cell_id"]

        with self.assertRaisesRegex(RolloutSequenceError, "duplicate cell_id"):
            build_rollout_sequences(examples)

    def test_rejects_missing_columns_empty_frames_and_invalid_timestamps(self):
        with self.assertRaisesRegex(RolloutSequenceError, "must not be empty"):
            build_rollout_sequences(pd.DataFrame())
        with self.assertRaisesRegex(RolloutSequenceError, "missing required columns"):
            build_rollout_sequences(self._rows().drop(columns=["anchor_at"]))

        examples = self._rows()
        examples.loc[100, "source_snapshot_time"] = "not-a-time"
        with self.assertRaisesRegex(RolloutSequenceError, "UTC-parseable"):
            build_rollout_sequences(examples)


if __name__ == "__main__":
    unittest.main()
