import gzip
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from wildfire_data.data_archive import CoverageLedger, CoverageStatus
from wildfire_data.feds_labels import FEDS_LABEL_BUILD_VERSION
from wildfire_data.normalized_storage import write_normalized_jsonl
from wildfire_data.storage_budget import load_storage_budget
from wildfire_data.terrain_features import TerrainFeatureSampler
from wildfire_data.training_dataset import (
    DEFAULT_FIRMS_AVAILABILITY_LAG,
    TRAINING_DATASET_BUILD_VERSION,
    TrainingDatasetError,
    assemble_feds_weak_positive_examples,
    build_and_store_feds_weak_positive_training_dataset,
    iter_feds_weak_positive_label_paths,
    iter_training_examples,
)
from wildfire_data.training_grid import TrainingExampleKey, cell_from_wgs84


UTC = timezone.utc


def _label(cell, *, anchor_at="2026-07-02T12:00:00Z", **changes):
    anchor = datetime.fromisoformat(anchor_at.replace("Z", "+00:00"))
    example = TrainingExampleKey(cell_id=cell.cell_id, anchor_at=anchor)
    record = {
        "schema_version": 1,
        "example_id": example.example_id,
        "cell_id": cell.cell_id,
        "anchor_at": anchor_at,
        "target_end_at": example.target_end_at.isoformat().replace("+00:00", "Z"),
        "target_newly_burned_12h": 1,
        "label_status": "positive-observed",
        "label_observability": "satellite-weak-positive-only",
        "label_tier": "weak_satellite",
        "label_source": "NASA FEDS",
        "label_quality_score": 0.45,
        "label_build_version": "feds-perimeter-difference-1km/v3-primarykey-observed",
        "positive_overlap_fraction": 0.25,
        "source_snapshot_time": "2026-07-02T00:00:00Z",
        "target_snapshot_time": "2026-07-02T12:00:00Z",
        "source_time_semantics": "local-solar-time-wall-clock-with-utc-date/v1",
        "time_alignment_mode": "estimated-local-solar-to-utc/v1",
        "contributing_fire_count": 1,
        "contributing_fires": [
            {
                "current_raw_artifact_id": "feds-current",
                "future_raw_artifact_id": "feds-future",
            }
        ],
    }
    record.update(changes)
    return record


def _detection(cell, *, identifier, acquired_at, bright_ti4=320.0, raw_artifact_id="firms-raw"):
    latitude, longitude = cell.center_wgs84
    return {
        "record_type": "firms_detection",
        "detection_id": identifier,
        "acquired_at": acquired_at,
        "latitude": latitude,
        "longitude": longitude,
        "bright_ti4": bright_ti4,
        "raw_source_fields": {"satellite": "N20"},
        "provenance": {"raw_artifact_id": raw_artifact_id},
    }


class TrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)

    def test_assembles_positive_row_with_only_cutoff_available_firms_and_explicit_weather_missingness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _write_terrain_block(root, self.cell)
            rows = assemble_feds_weak_positive_examples(
                [_label(self.cell)],
                firms_detections=[
                    _detection(
                        self.cell,
                        identifier="eligible",
                        acquired_at="2026-07-02T08:00:00Z",
                        raw_artifact_id="firms-eligible",
                    ),
                    _detection(
                        self.cell,
                        identifier="not-yet-available",
                        acquired_at="2026-07-02T10:00:01Z",
                        bright_ti4=390.0,
                        raw_artifact_id="firms-late",
                    ),
                    _detection(
                        self.cell,
                        identifier="future",
                        acquired_at="2026-07-02T13:00:00Z",
                        bright_ti4=400.0,
                        raw_artifact_id="firms-future",
                    ),
                ],
                terrain_sampler=TerrainFeatureSampler(root),
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["target_newly_burned_12h"], 1)
        self.assertEqual(row["prediction_horizon_hours"], 12)
        self.assertEqual(row["feature_cutoff_at"], row["anchor_at"])
        self.assertEqual(row["firms_center_detection_count"], 1)
        self.assertEqual(row["firms_center_bright_ti4_max"], 320.0)
        self.assertEqual(row["firms_raw_artifact_ids"], ["firms-eligible"])
        self.assertEqual(row["terrain_elevation_m"], 777.0)
        self.assertEqual(row["weather_available"], 0)
        self.assertEqual(row["weather_missing_indicator"], 1)
        self.assertEqual(row["weather_feature_status"], "unavailable-no-issued-forecast-features")
        self.assertEqual(
            row["binary_training_status"],
            "positive-only-requires-explicit-negative-or-observability-labels",
        )
        self.assertNotIn("wind_direction", row)

    def test_rejects_a_label_that_claims_a_non_12_hour_target_or_a_zero_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _write_terrain_block(root, self.cell)
            sampler = TerrainFeatureSampler(root)
            incorrect_target_end = _label(self.cell, target_end_at="2026-07-03T12:00:00Z")
            with self.assertRaisesRegex(TrainingDatasetError, "exactly 12 hours"):
                assemble_feds_weak_positive_examples(
                    [incorrect_target_end],
                    firms_detections=[],
                    terrain_sampler=sampler,
                )
            with self.assertRaisesRegex(TrainingDatasetError, "positive"):
                assemble_feds_weak_positive_examples(
                    [_label(self.cell, target_newly_burned_12h=0)],
                    firms_detections=[],
                    terrain_sampler=sampler,
                )
            with self.assertRaisesRegex(TrainingDatasetError, "different source-time"):
                assemble_feds_weak_positive_examples(
                    [_label(self.cell, label_build_version="feds-perimeter-difference-1km/v2-primarykey-time")],
                    firms_detections=[],
                    terrain_sampler=sampler,
                )

    def test_archive_builder_reads_bounded_partitions_and_persists_source_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _write_terrain_block(root, self.cell)
            label = _label(self.cell)
            write_normalized_jsonl(
                root,
                entity="training_labels",
                records=[label],
                partitions={"source": "feds-perimeter-difference", "source_snapshot": "2026-07-02"},
                raw_artifact_ids=["feds-current", "feds-future"],
                transformation_version="feds-perimeter-difference-1km/v3-primarykey-observed",
            )
            write_normalized_jsonl(
                root,
                entity="fire_detections",
                records=[
                    _detection(
                        self.cell,
                        identifier="in-range",
                        acquired_at="2026-07-02T08:00:00Z",
                        raw_artifact_id="firms-in-range",
                    )
                ],
                partitions={"acq_date": "2026-07-02"},
                raw_artifact_ids=["firms-in-range"],
                transformation_version="firms-normalized/v1",
            )
            # A corrupt artifact outside the necessary 24-hour lookback must
            # not be opened.  This is a regression check against archive-wide
            # scans per label batch.
            corrupt = root / "normalized" / "fire-detections" / "acq-date=2020-01-01" / "bad.jsonl.gz"
            corrupt.parent.mkdir(parents=True)
            with gzip.open(corrupt, "wt", encoding="utf-8") as destination:
                destination.write("not-json\n")
            policy_path = Path(directory) / "budget.json"
            policy_path.write_text(
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
            with self.assertRaisesRegex(TrainingDatasetError, "FIRMS coverage is incomplete"):
                build_and_store_feds_weak_positive_training_dataset(
                    root,
                    storage_budget=load_storage_budget(policy_path),
                    start_date=date(2026, 7, 2),
                    end_date=date(2026, 7, 2),
                )
            _record_terminal_firms_coverage(
                root,
                dates=(date(2026, 7, 1), date(2026, 7, 2)),
            )
            result = build_and_store_feds_weak_positive_training_dataset(
                root,
                storage_budget=load_storage_budget(policy_path),
                start_date=date(2026, 7, 2),
                end_date=date(2026, 7, 2),
            )
            stored_rows = list(iter_training_examples(root))
            original_directory = Path.cwd()
            try:
                os.chdir(root.parent)
                prefixed_manifest_path = Path("data") / result.manifest_path.relative_to(root)
                prefixed_rows = list(
                    iter_training_examples(
                        Path("data"),
                        manifest_path=prefixed_manifest_path,
                    )
                )
            finally:
                os.chdir(original_directory)

        self.assertEqual(result.input_label_count, 1)
        self.assertEqual(result.training_row_count, 1)
        self.assertEqual(result.normalized_artifact_count, 1)
        self.assertEqual(len(stored_rows), 1)
        self.assertEqual(len(prefixed_rows), 1)
        self.assertEqual(stored_rows[0]["label_raw_artifact_ids"], ["feds-current", "feds-future"])
        self.assertEqual(stored_rows[0]["firms_raw_artifact_ids"], ["firms-in-range"])
        self.assertEqual(stored_rows[0]["firms_center_detection_count"], 1)

    def test_archive_builder_selects_latest_completed_feds_label_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _write_terrain_block(root, self.cell)
            first_artifact = write_normalized_jsonl(
                root,
                entity="training_labels",
                records=[_label(self.cell, positive_overlap_fraction=0.25)],
                partitions={
                    "source": "feds-perimeter-difference",
                    "source_snapshot": "2026-07-02T00:00:00Z",
                    "target_snapshot": "2026-07-02T12:00:00Z",
                    "grid": "naea-1km",
                },
                raw_artifact_ids=["feds-first-current", "feds-first-future"],
                transformation_version=FEDS_LABEL_BUILD_VERSION,
            )
            refreshed_artifact = write_normalized_jsonl(
                root,
                entity="training_labels",
                records=[_label(self.cell, positive_overlap_fraction=0.50)],
                partitions={
                    "source": "feds-perimeter-difference",
                    "source_snapshot": "2026-07-02T00:00:00Z",
                    "target_snapshot": "2026-07-02T12:00:00Z",
                    "grid": "naea-1km",
                },
                raw_artifact_ids=["feds-refreshed-current", "feds-refreshed-future"],
                transformation_version=FEDS_LABEL_BUILD_VERSION,
            )
            expected_coverage_id = (
                f"feds-weak-labels:{FEDS_LABEL_BUILD_VERSION}:United States and Canada:"
                "2026-07-02T00:00:00Z:estimated-local-solar-to-utc/v1:overlap=0.100000"
            )
            ledger = CoverageLedger(root)
            ledger.record(
                source="wildfire-data training pipeline",
                product="feds-weak-labels",
                coverage_start="2026-07-02T00:00:00Z",
                coverage_end="2026-07-02T12:00:00Z",
                region="United States and Canada",
                expected_coverage_id=expected_coverage_id,
                status=CoverageStatus.COMPLETE,
                detail={"normalized_artifact_id": first_artifact.normalized_artifact_id},
                recorded_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
            ledger.record(
                source="wildfire-data training pipeline",
                product="feds-weak-labels",
                coverage_start="2026-07-02T00:00:00Z",
                coverage_end="2026-07-02T12:00:00Z",
                region="United States and Canada",
                expected_coverage_id=expected_coverage_id,
                status=CoverageStatus.COMPLETE,
                detail={"normalized_artifact_id": refreshed_artifact.normalized_artifact_id},
                recorded_at=datetime(2026, 8, 2, tzinfo=UTC),
            )
            write_normalized_jsonl(
                root,
                entity="fire_detections",
                records=[
                    _detection(
                        self.cell,
                        identifier="in-range",
                        acquired_at="2026-07-02T08:00:00Z",
                        raw_artifact_id="firms-in-range",
                    )
                ],
                partitions={"acq_date": "2026-07-02"},
                raw_artifact_ids=["firms-in-range"],
                transformation_version="firms-normalized/v1",
            )
            _record_terminal_firms_coverage(
                root,
                dates=(date(2026, 7, 1), date(2026, 7, 2)),
            )
            policy_path = Path(directory) / "budget.json"
            policy_path.write_text(
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

            self.assertEqual(
                iter_feds_weak_positive_label_paths(root),
                (refreshed_artifact.artifact_path,),
            )
            result = build_and_store_feds_weak_positive_training_dataset(
                root,
                storage_budget=load_storage_budget(policy_path),
                start_date=date(2026, 7, 2),
                end_date=date(2026, 7, 2),
            )

        self.assertEqual(result.input_label_count, 1)
        self.assertEqual(result.training_row_count, 1)
        self.assertEqual(result.normalized_artifact_count, 1)

    def test_uses_the_same_three_hour_default_lag_as_weather_tile_planning(self):
        self.assertEqual(DEFAULT_FIRMS_AVAILABILITY_LAG, timedelta(hours=3))

    def test_iterator_ignores_current_version_artifacts_until_a_completed_manifest_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            write_normalized_jsonl(
                root,
                entity="training_examples",
                records=[
                    {
                        "record_type": "training_example",
                        "example_id": "interrupted-build-row",
                        "training_dataset_build_version": TRAINING_DATASET_BUILD_VERSION,
                    }
                ],
                partitions={"dataset_build": TRAINING_DATASET_BUILD_VERSION},
                raw_artifact_ids=["feds-current"],
                transformation_version=TRAINING_DATASET_BUILD_VERSION,
            )

            self.assertEqual(list(iter_training_examples(root)), [])


def _write_terrain_block(root: Path, cell) -> None:
    latitude, longitude = cell.center_wgs84
    resolution = 0.01
    path = root / "static" / "etopo-2022-15s" / "main.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        elevation_m=np.full((3, 3), 777, dtype=np.int16),
        slope_degrees_x2=np.full((3, 3), 10, dtype=np.uint8),
        aspect_degrees_x2=np.full((3, 3), 45, dtype=np.uint8),
        grid_west=np.float64(longitude - resolution),
        grid_north=np.float64(latitude + resolution),
        pixel_width_degrees=np.float64(resolution),
        pixel_height_degrees=np.float64(resolution),
    )


def _record_terminal_firms_coverage(root: Path, *, dates: tuple[date, ...]) -> None:
    ledger = CoverageLedger(root)
    for coverage_date in dates:
        for product in ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"):
            ledger.record(
                source="NASA FIRMS",
                product=product,
                coverage_start=coverage_date,
                coverage_end=coverage_date,
                region="United States and Canada",
                expected_coverage_id=(
                    f"firms:{product}:United States and Canada:{coverage_date.isoformat()}"
                ),
                status=CoverageStatus.EMPTY_CONFIRMED,
            )


if __name__ == "__main__":
    unittest.main()
