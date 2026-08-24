import unittest
from datetime import datetime, timedelta, timezone

from wildfire_data.fire_state_features import (
    FireStateFeatureError,
    build_firms_fire_state_features,
)
from wildfire_data.training_grid import GridCell, cell_from_wgs84


UTC = timezone.utc


def _detection(*, detection_id, acquired_at, latitude, longitude, bright_ti4, satellite="N"):
    return {
        "record_type": "firms_detection",
        "detection_id": detection_id,
        "acquired_at": acquired_at,
        "latitude": latitude,
        "longitude": longitude,
        "bright_ti4": bright_ti4,
        "raw_source_fields": {"satellite": satellite},
    }


def _at_cell(cell, **values):
    latitude, longitude = cell.center_wgs84
    return _detection(latitude=latitude, longitude=longitude, **values)


class FirmsFireStateFeatureTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = datetime(2026, 7, 2, 12, tzinfo=UTC)
        self.target = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)

    def _build(self, detections, **overrides):
        return build_firms_fire_state_features(
            detections,
            cell_id=self.target.cell_id,
            cutoff_at=self.cutoff,
            lookback=overrides.pop("lookback", timedelta(hours=24)),
            availability_lag=overrides.pop("availability_lag", timedelta(hours=2)),
            **overrides,
        )

    def test_deduplicates_ids_and_excludes_not_yet_available_or_old_evidence(self):
        recent = _at_cell(
            self.target,
            detection_id="same-detection",
            acquired_at="2026-07-02T09:00:00Z",
            bright_ti4=320.0,
        )
        features = self._build(
            [
                recent,
                dict(recent),
                _at_cell(
                    self.target,
                    detection_id="not-available-yet",
                    acquired_at="2026-07-02T11:00:00Z",
                    bright_ti4=340.0,
                ),
                _at_cell(
                    self.target,
                    detection_id="future",
                    acquired_at="2026-07-02T13:00:00Z",
                    bright_ti4=350.0,
                ),
                _at_cell(
                    self.target,
                    detection_id="too-old",
                    acquired_at="2026-07-01T11:59:00Z",
                    bright_ti4=360.0,
                ),
            ]
        )

        self.assertEqual(features["latest_eligible_acquisition_at"], "2026-07-02T10:00:00Z")
        self.assertEqual(features["firms_center_detection_count"], 1)
        self.assertEqual(features["firms_center_bright_ti4_max"], 320.0)
        self.assertEqual(features["firms_center_bright_ti4_mean"], 320.0)
        self.assertEqual(features["firms_center_platform_count"], 1)
        self.assertEqual(features["firms_center_hours_since_last_detection"], 3.0)

    def test_reports_center_and_3_by_3_context_separately(self):
        east = GridCell(self.target.x_index + 1, self.target.y_index)
        north_east = GridCell(self.target.x_index + 1, self.target.y_index + 1)
        outside = GridCell(self.target.x_index + 2, self.target.y_index)
        features = self._build(
            [
                _at_cell(
                    self.target,
                    detection_id="centre",
                    acquired_at="2026-07-02T08:00:00Z",
                    bright_ti4=300.0,
                    satellite="N",
                ),
                _at_cell(
                    east,
                    detection_id="east",
                    acquired_at="2026-07-02T10:00:00Z",
                    bright_ti4=330.0,
                    satellite="N20",
                ),
                _at_cell(
                    north_east,
                    detection_id="north-east",
                    acquired_at="2026-07-02T07:00:00Z",
                    bright_ti4=310.0,
                    satellite="N21",
                ),
                _at_cell(
                    outside,
                    detection_id="outside",
                    acquired_at="2026-07-02T09:00:00Z",
                    bright_ti4=999.0,
                ),
            ]
        )

        self.assertEqual(features["firms_center_detection_count"], 1)
        self.assertEqual(features["firms_local_3x3_detection_count"], 3)
        self.assertEqual(features["firms_local_3x3_active_cell_count"], 3)
        self.assertEqual(features["firms_local_3x3_bright_ti4_max"], 330.0)
        self.assertAlmostEqual(features["firms_local_3x3_bright_ti4_mean"], 940.0 / 3.0)
        self.assertEqual(features["firms_local_3x3_platform_count"], 3)
        self.assertEqual(features["firms_local_3x3_hours_since_last_detection"], 2.0)

    def test_empty_evidence_uses_null_summaries_not_zero_brightness(self):
        features = self._build([])

        self.assertEqual(features["firms_center_has_detection"], 0)
        self.assertEqual(features["firms_center_detection_count"], 0)
        self.assertIsNone(features["firms_center_bright_ti4_max"])
        self.assertIsNone(features["firms_center_bright_ti4_mean"])
        self.assertIsNone(features["firms_center_hours_since_last_detection"])
        self.assertEqual(features["firms_local_3x3_active_cell_count"], 0)

    def test_rejects_conflicting_duplicate_detection_revisions(self):
        first = _at_cell(
            self.target,
            detection_id="conflict",
            acquired_at="2026-07-02T09:00:00Z",
            bright_ti4=320.0,
        )
        second = dict(first)
        second["bright_ti4"] = 321.0

        with self.assertRaisesRegex(FireStateFeatureError, "conflicting"):
            self._build([first, second])

    def test_requires_explicit_nonnegative_availability_lag(self):
        with self.assertRaisesRegex(FireStateFeatureError, "availability_lag"):
            self._build([], availability_lag=timedelta(hours=-1))


if __name__ == "__main__":
    unittest.main()
