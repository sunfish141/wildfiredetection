import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from shapely.geometry import Polygon

from wildfire_data.feds_collection import FEDS_NORMALIZATION_PARTITION
from wildfire_data.feds_labels import (
    DEFAULT_TIME_ALIGNMENT_MODE,
    FedsLabelError,
    build_feds_weak_positive_labels,
    build_and_store_feds_weak_labels,
    esri_rings_to_polygonal_geometry,
    estimate_feds_observation_at,
    rasterize_positive_cells,
)
from wildfire_data.normalized_storage import write_normalized_jsonl
from wildfire_data.storage_budget import load_storage_budget


def _record(*, timestamp, rings, source_id, raw_id, fire_id=1.0, region="CONUS"):
    return {
        "record_type": "feds_perimeter_snapshot",
        "source_record_id": source_id,
        "raw_artifact_id": raw_id,
        "region": region,
        "fire_id": fire_id,
        "source_snapshot_time": timestamp,
        "time_alignment_eligible": True,
        "geometry": {
            "encoding": "esri-rings-wgs84/v1",
            "spatial_reference": "EPSG:4326",
            "rings": rings,
        },
        "source_fields": {"n_newpixels": 2, "flinelen": 1.5},
    }


def _policy(path):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "whole_data_cap_bytes": 10_000_000,
                "whole_data_cap_label": "test",
                "scope": "test",
                "categories": [
                    {
                        "key": "derived_training_views",
                        "cap_bytes": 10_000_000,
                        "priority_score": 1,
                        "pinned": False,
                        "retention": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return load_storage_budget(path)


def _store_snapshot(root, record, *, snapshot_time):
    source = datetime.fromisoformat(snapshot_time.replace("Z", "+00:00"))
    raw_ids = [record["raw_artifact_id"]]
    write_normalized_jsonl(
        root,
        entity="fire-progression",
        records=[record],
        partitions={
            "normalization_version": FEDS_NORMALIZATION_PARTITION,
            "source": "feds-nrt-perimeters",
            "snapshot_start": snapshot_time,
            "snapshot_end": (source + timedelta(hours=12)).isoformat().replace("+00:00", "Z"),
        },
        raw_artifact_ids=raw_ids,
        transformation_version="feds-nrt-perimeters/v2-primarykey-time",
    )


class FedsLabelsTests(unittest.TestCase):
    def test_local_solar_timestamp_estimate_accounts_for_longitude(self):
        source_time = datetime(2026, 7, 1, tzinfo=timezone.utc)
        self.assertEqual(
            estimate_feds_observation_at(source_time, longitude=-105.0),
            datetime(2026, 7, 1, 8, 30, tzinfo=timezone.utc),
        )

    def test_esri_ring_nesting_preserves_a_hole(self):
        geometry = esri_rings_to_polygonal_geometry(
            [
                [[-110.0, 50.0], [-109.8, 50.0], [-109.8, 50.2], [-110.0, 50.2], [-110.0, 50.0]],
                [[-109.95, 50.05], [-109.85, 50.05], [-109.85, 50.15], [-109.95, 50.15], [-109.95, 50.05]],
            ]
        )
        self.assertGreater(geometry.area, 0)
        self.assertLess(geometry.area, 0.04)

    def test_esri_geometry_ignores_a_degenerate_auxiliary_ring(self):
        geometry = esri_rings_to_polygonal_geometry(
            [
                [[-110.0, 50.0], [-109.8, 50.0], [-109.8, 50.2], [-110.0, 50.2], [-110.0, 50.0]],
                [[-109.9, 50.1], [-109.89, 50.11]],
            ]
        )
        self.assertGreater(geometry.area, 0)

    def test_builds_positive_cells_from_future_minus_current_perimeter(self):
        current_time = "2026-07-01T00:00:00Z"
        future_time = "2026-07-01T12:00:00Z"
        current = _record(
            timestamp=current_time,
            source_id="CONUS|1|current",
            raw_id="raw-current",
            rings=[[[-110.00, 50.00], [-109.99, 50.00], [-109.99, 50.01], [-110.00, 50.01], [-110.00, 50.00]]],
        )
        future = _record(
            timestamp=future_time,
            source_id="CONUS|1|future",
            raw_id="raw-future",
            rings=[[[-110.00, 50.00], [-109.96, 50.00], [-109.96, 50.02], [-110.00, 50.02], [-110.00, 50.00]]],
        )
        labels, paired = build_feds_weak_positive_labels(
            [current],
            [future],
            source_snapshot_time=datetime(2026, 7, 1, tzinfo=timezone.utc),
            positive_overlap_fraction=0.01,
        )

        self.assertEqual(paired, 1)
        self.assertGreater(len(labels), 0)
        self.assertTrue(all(label["target_newly_burned_12h"] == 1 for label in labels))
        self.assertTrue(all(label["time_alignment_mode"] == DEFAULT_TIME_ALIGNMENT_MODE for label in labels))
        self.assertTrue(all(label["label_observability"] == "satellite-weak-positive-only" for label in labels))
        self.assertTrue(all(label["contributing_fires"][0]["future_n_newpixels"] == 2.0 for label in labels))

    def test_rejects_conflicting_duplicate_fire_snapshot_revisions(self):
        timestamp = "2026-07-01T00:00:00Z"
        first = _record(
            timestamp=timestamp,
            source_id="CONUS|1|current",
            raw_id="one",
            rings=[[[-110.0, 50.0], [-109.99, 50.0], [-109.99, 50.01], [-110.0, 50.01], [-110.0, 50.0]]],
        )
        second = _record(
            timestamp=timestamp,
            source_id="CONUS|1|current",
            raw_id="two",
            rings=[[[-110.0, 50.0], [-109.98, 50.0], [-109.98, 50.01], [-110.0, 50.01], [-110.0, 50.0]]],
        )
        with self.assertRaises(FedsLabelError):
            build_feds_weak_positive_labels(
                [first, second],
                [],
                source_snapshot_time=datetime(2026, 7, 1, tzinfo=timezone.utc),
            )

    def test_rasterizer_requires_a_real_overlap_threshold(self):
        with self.assertRaises(FedsLabelError):
            rasterize_positive_cells(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), positive_overlap_fraction=0)

    def test_storage_builder_uses_v2_primarykey_partitions_without_old_query_coverage(self):
        current = _record(
            timestamp="2026-07-01T00:00:00Z",
            source_id="CONUS|1|2026-07-01T00:00:00",
            raw_id="a" * 64,
            rings=[[[-110.00, 50.00], [-109.99, 50.00], [-109.99, 50.01], [-110.00, 50.01], [-110.00, 50.00]]],
        )
        future = _record(
            timestamp="2026-07-01T12:00:00Z",
            source_id="CONUS|1|2026-07-01T12:00:00",
            raw_id="b" * 64,
            rings=[[[-110.00, 50.00], [-109.96, 50.00], [-109.96, 50.02], [-110.00, 50.02], [-110.00, 50.00]]],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _store_snapshot(root, current, snapshot_time="2026-07-01T00:00:00Z")
            _store_snapshot(root, future, snapshot_time="2026-07-01T12:00:00Z")
            reports = build_and_store_feds_weak_labels(
                root,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                storage_budget=_policy(Path(directory) / "budget.json"),
                positive_overlap_fraction=0.01,
                generated_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
            )

        self.assertEqual(reports[0].status.value, "complete")
        self.assertGreater(reports[0].positive_cell_count, 0)
        # The later source window has no retained t+12 primary-key partition;
        # it remains unknown rather than becoming an empty/no-spread label.
        self.assertEqual(reports[1].status.value, "partial")

    def test_storage_builder_keeps_no_expansion_partial_not_empty_confirmed(self):
        rings = [[[-110.00, 50.00], [-109.99, 50.00], [-109.99, 50.01], [-110.00, 50.01], [-110.00, 50.00]]]
        current = _record(
            timestamp="2026-07-01T00:00:00Z",
            source_id="CONUS|1|2026-07-01T00:00:00",
            raw_id="a" * 64,
            rings=rings,
        )
        future = _record(
            timestamp="2026-07-01T12:00:00Z",
            source_id="CONUS|1|2026-07-01T12:00:00",
            raw_id="b" * 64,
            rings=rings,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _store_snapshot(root, current, snapshot_time="2026-07-01T00:00:00Z")
            _store_snapshot(root, future, snapshot_time="2026-07-01T12:00:00Z")
            reports = build_and_store_feds_weak_labels(
                root,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                storage_budget=_policy(Path(directory) / "budget.json"),
                generated_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
            )

        self.assertEqual(reports[0].status.value, "partial")
        self.assertEqual(reports[0].positive_cell_count, 0)


if __name__ == "__main__":
    unittest.main()
