import tempfile
import unittest
from datetime import datetime, timezone

from wildfire_data.collection_catalog import targets_for_entity
from wildfire_data.collection_planning import plan_collection_windows, windows_needing_collection
from wildfire_data.data_archive import CoverageLedger, CoverageStatus


class CollectionPlanningTests(unittest.TestCase):
    def test_builds_explicit_windows_at_the_target_cadence(self):
        target = targets_for_entity("operational_perimeter")[0]

        windows = plan_collection_windows(
            target,
            coverage_start=datetime(2026, 7, 26, 0, tzinfo=timezone.utc),
            coverage_end=datetime(2026, 7, 26, 0, 35, tzinfo=timezone.utc),
        )

        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[-1].coverage_end, datetime(2026, 7, 26, 0, 35, tzinfo=timezone.utc))
        self.assertIn("wfigs_current_perimeters", windows[0].expected_coverage_id)

    def test_identifies_missing_failed_and_partial_windows_for_retry(self):
        target = targets_for_entity("operational_perimeter")[0]
        windows = plan_collection_windows(
            target,
            coverage_start=datetime(2026, 7, 26, 0, tzinfo=timezone.utc),
            coverage_end=datetime(2026, 7, 26, 0, 45, tzinfo=timezone.utc),
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = CoverageLedger(directory)
            for window, status in zip(
                windows[:2], [CoverageStatus.COMPLETE, CoverageStatus.FAILED]
            ):
                ledger.record(
                    source=window.target.provider,
                    product=window.target.key,
                    coverage_start=window.coverage_start,
                    coverage_end=window.coverage_end,
                    region=window.region,
                    expected_coverage_id=window.expected_coverage_id,
                    status=status,
                )

            pending = windows_needing_collection(ledger, windows)

        self.assertEqual(pending, (windows[1], windows[2]))
