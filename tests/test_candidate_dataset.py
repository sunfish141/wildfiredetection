import gzip
import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from wildfire_data.candidate_dataset import (
    CANDIDATE_DATASET_BUILD_VERSION,
    CandidateDatasetError,
    build_and_store_firms_candidate_dataset,
    export_candidate_dataset_release,
    iter_candidate_examples,
    merge_candidate_dataset_builds,
)
from wildfire_data.data_archive import CoverageLedger, CoverageStatus, write_atomic_json
from wildfire_data.feds_labels import estimate_feds_observation_at
from wildfire_data.normalized_storage import write_normalized_jsonl
from wildfire_data.storage_budget import load_storage_budget
from wildfire_data.training_dataset import TRAINING_DATASET_BUILD_VERSION
from wildfire_data.training_grid import TrainingExampleKey, cell_from_wgs84


UTC = timezone.utc


class CandidateDatasetTests(unittest.TestCase):
    def test_builds_manifest_selected_no_weather_rows_and_self_contained_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)
            _write_terrain_block(root, cell)
            snapshot = datetime(2026, 7, 2, tzinfo=UTC)
            positive = _positive_row(cell, snapshot=snapshot)
            artifact = write_normalized_jsonl(
                root,
                entity="training_examples",
                records=[positive],
                partitions={"dataset_build": TRAINING_DATASET_BUILD_VERSION, "anchor_date": "2026-07-02"},
                raw_artifact_ids=["feds-current", "feds-future"],
                transformation_version=TRAINING_DATASET_BUILD_VERSION,
            )
            positive_manifest = _write_positive_manifest(root, artifact, snapshot.date())
            _write_detection(root, cell, acquired_at=_parse(positive["anchor_at"]) - timedelta(hours=4))
            _record_coverage(root, snapshot.date())
            policy = _policy(Path(directory) / "budget.json")

            result = build_and_store_firms_candidate_dataset(
                root,
                storage_budget=policy,
                start_date=snapshot.date(),
                end_date=snapshot.date(),
                positive_view_manifest=positive_manifest,
                radius_cells=1,
                max_weak_negative_proxies_per_snapshot=2,
            )
            rows = list(iter_candidate_examples(root, manifest_path=result.manifest_path))
            release = export_candidate_dataset_release(
                root,
                Path(directory) / "release",
                candidate_manifest=result.manifest_path,
            )

            self.assertEqual(result.candidate_row_count, 3)
            self.assertEqual(result.supported_positive_count, 1)
            self.assertEqual(result.weak_negative_proxy_count, 2)
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["target_newly_burned_12h"] for row in rows}, {0, 1})
            self.assertTrue(all(row["weather_available"] == 0 for row in rows))
            self.assertTrue(
                all(
                    row["weather_feature_status"]
                    == "unavailable-no-issued-forecast-features"
                    for row in rows
                )
            )
            self.assertTrue(all(row["dataset_split"] == "train" for row in rows))
            self.assertTrue(
                all(row["candidate_dataset_build_version"] == CANDIDATE_DATASET_BUILD_VERSION for row in rows)
            )
            self.assertTrue((release.directory / "candidate_examples.jsonl.gz").is_file())
            self.assertTrue((release.directory / "dataset_manifest.json").is_file())
            self.assertEqual(_gzip_record_count(release.directory / "candidate_examples.jsonl.gz"), 3)
            _assert_checksums(self, release.directory)

            repeated = build_and_store_firms_candidate_dataset(
                root,
                storage_budget=policy,
                start_date=snapshot.date(),
                end_date=snapshot.date(),
                positive_view_manifest=positive_manifest,
                radius_cells=1,
                max_weak_negative_proxies_per_snapshot=2,
            )
            self.assertNotEqual(repeated.manifest_path, result.manifest_path)
            reused = export_candidate_dataset_release(
                root,
                release.directory,
                candidate_manifest=repeated.manifest_path,
            )
            self.assertEqual(reused.directory, release.directory)
            self.assertEqual(reused.candidate_row_count, release.candidate_row_count)

    def test_refuses_to_claim_an_uncovered_source_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)
            _write_terrain_block(root, cell)
            snapshot = datetime(2026, 7, 2, tzinfo=UTC)
            positive = _positive_row(cell, snapshot=snapshot)
            artifact = write_normalized_jsonl(
                root,
                entity="training_examples",
                records=[positive],
                partitions={"dataset_build": TRAINING_DATASET_BUILD_VERSION, "anchor_date": "2026-07-02"},
                raw_artifact_ids=["feds-current", "feds-future"],
                transformation_version=TRAINING_DATASET_BUILD_VERSION,
            )
            manifest = _write_positive_manifest(root, artifact, snapshot.date())
            with self.assertRaisesRegex(CandidateDatasetError, "outside the completed positive"):
                build_and_store_firms_candidate_dataset(
                    root,
                    storage_budget=_policy(Path(directory) / "budget.json"),
                    start_date=date(2026, 7, 1),
                    end_date=snapshot.date(),
                    positive_view_manifest=manifest,
                )

    def test_merges_contiguous_chunks_with_one_global_snapshot_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)
            _write_terrain_block(root, cell)
            first_snapshot = datetime(2026, 7, 2, tzinfo=UTC)
            second_snapshot = datetime(2026, 7, 3, tzinfo=UTC)
            first_positive = _positive_row(cell, snapshot=first_snapshot)
            second_positive = _positive_row(cell, snapshot=second_snapshot)
            artifact = write_normalized_jsonl(
                root,
                entity="training_examples",
                records=[first_positive, second_positive],
                partitions={"dataset_build": TRAINING_DATASET_BUILD_VERSION, "anchor_date": "2026-07"},
                raw_artifact_ids=["feds-current", "feds-future"],
                transformation_version=TRAINING_DATASET_BUILD_VERSION,
            )
            positive_manifest = _write_positive_manifest(
                root, artifact, first_snapshot.date(), end_day=second_snapshot.date()
            )
            _write_detection(
                root,
                cell,
                identifier="first",
                acquired_at=_parse(first_positive["anchor_at"]) - timedelta(hours=4),
            )
            _write_detection(
                root,
                cell,
                identifier="second",
                acquired_at=_parse(second_positive["anchor_at"]) - timedelta(hours=4),
            )
            _record_coverage(root, first_snapshot.date())
            _record_coverage(root, second_snapshot.date())
            policy = _policy(Path(directory) / "budget.json")
            common = {
                "storage_budget": policy,
                "positive_view_manifest": positive_manifest,
                "radius_cells": 1,
                "max_weak_negative_proxies_per_snapshot": 1,
                "split_start_date": first_snapshot.date(),
                "split_end_date": second_snapshot.date(),
            }
            first = build_and_store_firms_candidate_dataset(
                root, start_date=first_snapshot.date(), end_date=first_snapshot.date(), **common
            )
            second = build_and_store_firms_candidate_dataset(
                root, start_date=second_snapshot.date(), end_date=second_snapshot.date(), **common
            )
            merged = merge_candidate_dataset_builds(
                root,
                input_manifests=[first.manifest_path, second.manifest_path],
                start_date=first_snapshot.date(),
                end_date=second_snapshot.date(),
            )
            rows = list(iter_candidate_examples(root, manifest_path=merged))

            self.assertEqual(len(rows), 4)
            self.assertEqual(
                {row["source_snapshot_time"]: row["dataset_split"] for row in rows},
                {
                    "2026-07-02T00:00:00Z": "train",
                    "2026-07-03T00:00:00Z": "validation",
                },
            )


def _positive_row(cell, *, snapshot):
    anchor = estimate_feds_observation_at(snapshot, longitude=cell.center_wgs84[1])
    key = TrainingExampleKey(cell_id=cell.cell_id, anchor_at=anchor)
    return {
        "record_type": "training_example",
        "training_dataset_build_version": TRAINING_DATASET_BUILD_VERSION,
        "example_id": key.example_id,
        "cell_id": cell.cell_id,
        "anchor_at": anchor.isoformat().replace("+00:00", "Z"),
        "feature_cutoff_at": anchor.isoformat().replace("+00:00", "Z"),
        "target_end_at": key.target_end_at.isoformat().replace("+00:00", "Z"),
        "target_newly_burned_12h": 1,
        "source_snapshot_time": snapshot.isoformat().replace("+00:00", "Z"),
        "label_status": "positive-observed",
        "label_observability": "satellite-weak-positive-only",
        "label_tier": "weak_satellite",
        "label_raw_artifact_ids": ["feds-current", "feds-future"],
    }


def _write_detection(root, cell, *, identifier="seed", acquired_at):
    latitude, longitude = cell.center_wgs84
    record = {
        "record_type": "firms_detection",
        "detection_id": identifier,
        "acquired_at": acquired_at.isoformat().replace("+00:00", "Z"),
        "latitude": latitude,
        "longitude": longitude,
        "bright_ti4": 320.0,
        "raw_source_fields": {"satellite": "N20"},
        "provenance": {"raw_artifact_id": "firms-raw"},
    }
    write_normalized_jsonl(
        root,
        entity="fire_detections",
        records=[record],
        partitions={"acq_date": acquired_at.date().isoformat()},
        raw_artifact_ids=["firms-raw"],
        transformation_version="firms-normalized/v1",
    )


def _record_coverage(root, day):
    ledger = CoverageLedger(root)
    for coverage_day in (day - timedelta(days=1), day):
        for product in ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"):
            ledger.record(
                source="NASA FIRMS",
                product=product,
                coverage_start=coverage_day,
                coverage_end=coverage_day,
                region="United States and Canada",
                expected_coverage_id=f"firms:{product}:United States and Canada:{coverage_day.isoformat()}",
                status=CoverageStatus.COMPLETE if coverage_day == day else CoverageStatus.EMPTY_CONFIRMED,
            )


def _write_positive_manifest(root, artifact, day, *, end_day=None):
    path = root / "manifests" / "training-dataset-builds" / "2026" / "07" / "02" / "view.json"
    write_atomic_json(
        path,
        {
            "schema_version": 1,
            "kind": "completed-training-dataset-build",
            "status": "complete",
            "build_id": "positive-view",
            "completed_at": "2026-08-23T00:00:00Z",
            "training_dataset_build_version": TRAINING_DATASET_BUILD_VERSION,
            "source_snapshot_start_date": day.isoformat(),
            "source_snapshot_end_date": (end_day or day).isoformat(),
            "artifact_relative_paths": [artifact.artifact_path.relative_to(root).as_posix()],
        },
    )
    return path


def _write_terrain_block(root, cell):
    latitude, longitude = cell.center_wgs84
    path = root / "static" / "etopo-2022-15s" / "main.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        elevation_m=np.full((9, 9), 777, dtype=np.int16),
        slope_degrees_x2=np.full((9, 9), 10, dtype=np.uint8),
        aspect_degrees_x2=np.full((9, 9), 45, dtype=np.uint8),
        grid_west=np.float64(longitude - 0.04),
        grid_north=np.float64(latitude + 0.04),
        pixel_width_degrees=np.float64(0.01),
        pixel_height_degrees=np.float64(0.01),
    )


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


def _gzip_record_count(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def _assert_checksums(test, directory):
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split("  ", 1)
        actual = hashlib.sha256((directory / relative_path).read_bytes()).hexdigest()
        test.assertEqual(actual, expected)


def _parse(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


if __name__ == "__main__":
    unittest.main()
