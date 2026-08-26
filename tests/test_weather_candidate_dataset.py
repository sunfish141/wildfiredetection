import gzip
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from wildfire_data.candidate_dataset import CANDIDATE_DATASET_BUILD_VERSION
from wildfire_data.normalized_storage import write_normalized_jsonl
from wildfire_data.open_meteo_historical import (
    OPEN_METEO_HISTORICAL_FEATURE_MODE,
    OPEN_METEO_HISTORICAL_WEATHER_KIND,
)
from wildfire_data.storage_budget import StorageBudgetCategory, StorageBudgetPolicy
from wildfire_data.weather_candidate_dataset import (
    WEATHER_CANDIDATE_DATASET_BUILD_VERSION,
    WEATHER_FEATURE_STATUS,
    WeatherCandidateDatasetError,
    build_weather_candidate_dataset,
    export_weather_candidate_dataset_release,
    iter_weather_candidate_examples,
)


def _storage_policy():
    return StorageBudgetPolicy(
        schema_version=1,
        whole_data_cap_bytes=10_000_000,
        whole_data_cap_label="test",
        scope="test root",
        categories=(
            StorageBudgetCategory(
                key="derived_training_views",
                cap_bytes=10_000_000,
                priority_score=50,
                pinned=False,
                retention="derived",
            ),
        ),
    )


def _candidate():
    return {
        "example_id": "cell-001@2026-06-01T10:45:00Z",
        "cell_id": "naea-1km:000000:000000",
        "anchor_at": "2026-06-01T10:45:00Z",
        "target_end_at": "2026-06-01T22:45:00Z",
        "target_newly_burned_12h": 1,
        "firms_raw_artifact_ids": ["firms-raw"],
        "feds_snapshot_context_raw_artifact_ids": ["feds-raw"],
        "candidate_dataset_build_version": CANDIDATE_DATASET_BUILD_VERSION,
    }


def _measurement(variable, value):
    return {
        "weather_measurement_id": variable,
        "weather_tile_id": "tile-001",
        "observed_at": "2026-06-01T10:00:00Z",
        "variable": variable,
        "value": value,
        "provider": "Open-Meteo",
        "model": "ecmwf_ifs",
        "raw_artifact_id": "weather-raw",
    }


def _write_completed_candidate_manifest(root, *, build_id="candidate-build-a"):
    artifact = write_normalized_jsonl(
        root,
        entity="candidate_examples",
        records=[_candidate()],
        partitions={"anchor_date": "2026-06-01"},
        raw_artifact_ids=["firms-raw"],
        transformation_version=CANDIDATE_DATASET_BUILD_VERSION,
        generated_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )
    path = root / "manifests" / "candidate-dataset-builds" / f"{build_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "kind": "completed-firms-candidate-dataset-build",
        "status": "complete",
        "build_id": build_id,
        "completed_at": "2026-08-25T00:00:00Z",
        "candidate_dataset_build_version": CANDIDATE_DATASET_BUILD_VERSION,
        "source_snapshot_start_date": "2026-06-01",
        "source_snapshot_end_date": "2026-06-01",
        "candidate_artifact_relative_paths": [artifact.artifact_path.relative_to(root).as_posix()],
        "unscored_artifact_relative_paths": [],
    }
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path, {
        "relative_path": path.relative_to(root).as_posix(),
        "build_id": build_id,
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_backfill(root, *, candidate_identity, omit=()):
    variables = {
        "temperature_2m": 21.5,
        "relative_humidity_2m": 35.0,
        "precipitation": 0.4,
        "wind_u_10m": -2.0,
        "wind_v_10m": 3.0,
    }
    measurements = [
        _measurement(variable, value)
        for variable, value in variables.items()
        if variable not in set(omit)
    ]
    measurement_artifact = write_normalized_jsonl(
        root,
        entity="historical_weather",
        records=measurements,
        partitions={"weather_date": "2026-06-01", "model": "ecmwf_ifs"},
        raw_artifact_ids=["weather-raw"],
        transformation_version="test",
        generated_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )
    mapping_artifact = write_normalized_jsonl(
        root,
        entity="open_meteo_historical_weather_tile_assignments",
        records=[
            {
                "candidate_cell_id": "naea-1km:000000:000000",
                "source_example_ids": ["cell-001@2026-06-01T10:45:00Z"],
                "weather_tile_id": "tile-001",
                "weather_tile_distance_m": 800.0,
                "source_grid_id": "open-meteo:tile-001",
                "raw_artifact_id": "weather-raw",
            }
        ],
        partitions={"weather_date": "2026-06-01", "model": "ecmwf_ifs"},
        raw_artifact_ids=["weather-raw", "firms-raw"],
        transformation_version="test",
        generated_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )
    document = {
        "kind": "open-meteo-historical-weather-backfill",
        "status": "complete",
        "candidate_manifest": candidate_identity,
        "provider": "Open-Meteo",
        "product_kind": OPEN_METEO_HISTORICAL_WEATHER_KIND,
        "model": "ecmwf_ifs",
        "feature_mode": OPEN_METEO_HISTORICAL_FEATURE_MODE,
        "feature_hour_policy": "floor-anchor-to-utc-hour/v1",
        "reports": [
            {
                "weather_date": "2026-06-01",
                "status": "complete",
                "measurement_artifact_relative_paths": [
                    measurement_artifact.artifact_path.relative_to(root).as_posix()
                ],
                "assignment_artifact_relative_paths": [
                    mapping_artifact.artifact_path.relative_to(root).as_posix()
                ],
            }
        ],
    }
    path = root / "manifests" / "weather-backfill.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class WeatherCandidateDatasetTests(unittest.TestCase):
    def test_builds_and_exports_a_complete_weather_bearing_candidate_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            candidate_manifest, candidate_identity = _write_completed_candidate_manifest(root)
            backfill = _write_backfill(root, candidate_identity=candidate_identity)
            result = build_weather_candidate_dataset(
                root,
                storage_budget=_storage_policy(),
                weather_backfill_manifest=backfill,
                candidate_manifest=candidate_manifest,
            )
            records = list(
                iter_weather_candidate_examples(root, manifest_path=result.manifest_path)
            )
            release = export_weather_candidate_dataset_release(
                root,
                Path(directory) / "release",
                weather_candidate_manifest=result.manifest_path,
            )
            with gzip.open(release.directory / "candidate_examples.jsonl.gz", "rt") as source:
                exported = json.loads(source.readline())
            checksum_exists = (release.directory / "SHA256SUMS").is_file()

        self.assertEqual(result.candidate_row_count, 1)
        self.assertEqual(records[0]["weather_candidate_dataset_build_version"], WEATHER_CANDIDATE_DATASET_BUILD_VERSION)
        self.assertEqual(records[0]["weather_feature_status"], WEATHER_FEATURE_STATUS)
        self.assertEqual(records[0]["weather_observed_at"], "2026-06-01T10:00:00Z")
        self.assertEqual(records[0]["weather_temperature_2m"], 21.5)
        self.assertEqual(records[0]["weather_wind_u_10m"], -2.0)
        self.assertEqual(exported["weather_precipitation"], 0.4)
        self.assertTrue(checksum_exists)

    def test_refuses_to_publish_when_a_required_weather_value_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            candidate_manifest, candidate_identity = _write_completed_candidate_manifest(root)
            backfill = _write_backfill(
                root,
                candidate_identity=candidate_identity,
                omit=("wind_v_10m",),
            )
            with self.assertRaisesRegex(WeatherCandidateDatasetError, "missing weather variables"):
                build_weather_candidate_dataset(
                    root,
                    storage_budget=_storage_policy(),
                    weather_backfill_manifest=backfill,
                    candidate_manifest=candidate_manifest,
                )

    def test_refuses_a_candidate_manifest_that_does_not_match_the_backfill_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            source_manifest, source_identity = _write_completed_candidate_manifest(
                root, build_id="candidate-build-a"
            )
            other_manifest, _ = _write_completed_candidate_manifest(
                root, build_id="candidate-build-b"
            )
            backfill = _write_backfill(root, candidate_identity=source_identity)

            self.assertNotEqual(source_manifest, other_manifest)
            with self.assertRaisesRegex(
                WeatherCandidateDatasetError, "candidate manifest identity"
            ):
                build_weather_candidate_dataset(
                    root,
                    storage_budget=_storage_policy(),
                    weather_backfill_manifest=backfill,
                    candidate_manifest=other_manifest,
                )


if __name__ == "__main__":
    unittest.main()
