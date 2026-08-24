import unittest
from datetime import datetime, timedelta, timezone

from wildfire_data.candidate_sampling import (
    UNSCORED_POSITIVE_STATUS,
    WEAK_NEGATIVE_PROXY_OBSERVABILITY,
    build_firms_only_candidates,
)
from wildfire_data.feds_labels import estimate_feds_observation_at
from wildfire_data.training_grid import TrainingExampleKey, cell_from_wgs84


def _label(cell, *, snapshot_at):
    anchor_at = estimate_feds_observation_at(snapshot_at, longitude=cell.center_wgs84[1])
    key = TrainingExampleKey(cell_id=cell.cell_id, anchor_at=anchor_at)
    return {
        "example_id": key.example_id,
        "cell_id": cell.cell_id,
        "anchor_at": anchor_at.isoformat().replace("+00:00", "Z"),
        "target_end_at": key.target_end_at.isoformat().replace("+00:00", "Z"),
        "source_snapshot_time": snapshot_at.isoformat().replace("+00:00", "Z"),
        "target_newly_burned_12h": 1,
        "label_status": "positive-observed",
        "label_observability": "satellite-weak-positive-only",
        "label_tier": "weak_satellite",
    }


def _detection(cell, *, identifier, acquired_at):
    latitude, longitude = cell.center_wgs84
    return {
        "detection_id": identifier,
        "latitude": latitude,
        "longitude": longitude,
        "acquired_at": acquired_at.isoformat().replace("+00:00", "Z"),
        "provenance": {"raw_artifact_id": f"raw-{identifier}"},
    }


class CandidateSamplingTests(unittest.TestCase):
    def test_firms_seed_expansion_marks_non_positive_cells_as_proxies(self):
        snapshot = datetime(2026, 7, 1, tzinfo=timezone.utc)
        cell = cell_from_wgs84(latitude=50.0, longitude=-110.0)
        anchor = estimate_feds_observation_at(snapshot, longitude=cell.center_wgs84[1])
        result = build_firms_only_candidates(
            [_label(cell, snapshot_at=snapshot)],
            [_detection(cell, identifier="seed", acquired_at=anchor - timedelta(hours=4))],
            radius_cells=1,
        )

        self.assertEqual(result.positive_candidate_count, 1)
        self.assertEqual(result.weak_negative_proxy_count, 8)
        self.assertEqual(result.unscored_positive_rows, ())
        proxy = next(row for row in result.candidate_rows if row["target_newly_burned_12h"] == 0)
        self.assertEqual(proxy["label_observability"], WEAK_NEGATIVE_PROXY_OBSERVABILITY)
        self.assertEqual(proxy["candidate_seed_detection_ids"], ["seed"])
        self.assertEqual(proxy["candidate_seed_raw_artifact_ids"], ["raw-seed"])

    def test_positive_without_an_available_firms_seed_is_retained_unscored(self):
        snapshot = datetime(2026, 7, 1, tzinfo=timezone.utc)
        positive = _label(cell_from_wgs84(latitude=50.0, longitude=-110.0), snapshot_at=snapshot)
        result = build_firms_only_candidates(
            [positive],
            [],
            radius_cells=1,
        )

        self.assertEqual(result.candidate_rows, ())
        self.assertEqual(len(result.unscored_positive_rows), 1)
        self.assertEqual(result.unscored_positive_rows[0]["candidate_selection_reason"], UNSCORED_POSITIVE_STATUS)

    def test_proxy_cap_is_deterministic_and_never_removes_a_positive(self):
        snapshot = datetime(2026, 7, 1, tzinfo=timezone.utc)
        cell = cell_from_wgs84(latitude=50.0, longitude=-110.0)
        anchor = estimate_feds_observation_at(snapshot, longitude=cell.center_wgs84[1])
        arguments = ([ _label(cell, snapshot_at=snapshot) ], [_detection(cell, identifier="seed", acquired_at=anchor - timedelta(hours=4))])
        first = build_firms_only_candidates(*arguments, radius_cells=1, max_weak_negative_proxies_per_snapshot=2)
        second = build_firms_only_candidates(*arguments, radius_cells=1, max_weak_negative_proxies_per_snapshot=2)

        self.assertEqual(first.candidate_rows, second.candidate_rows)
        self.assertEqual(first.positive_candidate_count, 1)
        self.assertEqual(first.weak_negative_proxy_count, 2)


if __name__ == "__main__":
    unittest.main()
