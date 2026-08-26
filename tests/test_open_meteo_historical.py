import gzip
import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from wildfire_data.data_archive import CoverageLedger, CoverageStatus
from wildfire_data.candidate_dataset import CANDIDATE_DATASET_BUILD_VERSION
from wildfire_data.open_meteo_historical import (
    OPEN_METEO_HISTORICAL_WEATHER_URL,
    backfill_open_meteo_historical_weather,
    capture_open_meteo_historical_weather,
    floor_weather_hour,
)
from wildfire_data.normalized_storage import write_normalized_jsonl
from wildfire_data.open_meteo_single_run import plan_candidate_example_weather_tiles
from wildfire_data.storage_budget import StorageBudgetCategory, StorageBudgetPolicy
from wildfire_data.training_grid import cell_from_wgs84


class _Response:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    content = b'{"provider":"fake-open-meteo"}'

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _RateLimitedResponse:
    status_code = 429
    headers = {"Retry-After": "0"}
    content = b'{"reason":"rate limited"}'

    def raise_for_status(self):
        raise RuntimeError("HTTP 429")


class _Session:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return next(self._responses)


def _storage_policy():
    return StorageBudgetPolicy(
        schema_version=1,
        whole_data_cap_bytes=10_000_000,
        whole_data_cap_label="test",
        scope="test root",
        categories=(
            StorageBudgetCategory(
                key="issued_weather_tiles",
                cap_bytes=10_000_000,
                priority_score=85,
                pinned=True,
                retention="weather",
            ),
        ),
    )


def _example(latitude=54.0, longitude=-106.0, identifier="example-001"):
    return {
        "cell_id": cell_from_wgs84(latitude=latitude, longitude=longitude).cell_id,
        "example_id": identifier,
        "firms_raw_artifact_ids": ["a" * 64],
    }


def _payload(latitude=54.0, longitude=-106.0, weather_date=date(2026, 6, 1)):
    weather_day = weather_date.isoformat()
    return [
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly_units": {
                "temperature_2m": "°C",
                "relative_humidity_2m": "%",
                "precipitation": "mm",
                "weather_code": "wmo code",
                "wind_speed_10m": "m/s",
                "wind_direction_10m": "°",
            },
            "hourly": {
                "time": [f"{weather_day}T10:00", f"{weather_day}T11:00"],
                "temperature_2m": [20.0, 21.0],
                "relative_humidity_2m": [35.0, 34.0],
                "precipitation": [0.0, 1.5],
                "weather_code": [1, 2],
                "wind_speed_10m": [5.0, 10.0],
                "wind_direction_10m": [0.0, 90.0],
            },
        }
    ]


def _records(artifacts):
    output = []
    for artifact in artifacts:
        with gzip.open(artifact.artifact_path, "rt", encoding="utf-8") as source:
            output.extend(json.loads(line) for line in source if line.strip())
    return output


def _candidate_row(
    *,
    latitude=54.0,
    longitude=-106.0,
    identifier="example-001",
    anchor_at="2026-06-01T10:45:00Z",
):
    return {
        **_example(latitude=latitude, longitude=longitude, identifier=identifier),
        "anchor_at": anchor_at,
        "candidate_dataset_build_version": CANDIDATE_DATASET_BUILD_VERSION,
    }


def _write_completed_candidate_manifest(root, rows, *, build_id="candidate-build-a"):
    anchors = [datetime.fromisoformat(row["anchor_at"].replace("Z", "+00:00")) for row in rows]
    artifact = write_normalized_jsonl(
        root,
        entity="candidate_examples",
        records=rows,
        partitions={"anchor_date": "2026-06"},
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
        "source_snapshot_start_date": min(anchor.date() for anchor in anchors).isoformat(),
        "source_snapshot_end_date": max(anchor.date() for anchor in anchors).isoformat(),
        "candidate_artifact_relative_paths": [artifact.artifact_path.relative_to(root).as_posix()],
        "unscored_artifact_relative_paths": [],
    }
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path, {
        "relative_path": path.relative_to(root).as_posix(),
        "build_id": build_id,
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class OpenMeteoHistoricalWeatherTests(unittest.TestCase):
    def test_collects_hourly_weather_at_candidate_tiles_and_keeps_example_mapping(self):
        plan = plan_candidate_example_weather_tiles(pd.DataFrame([_example()]))
        session = _Session([_Response(_payload())])
        retrieved_at = datetime(2026, 8, 25, 12, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            result = capture_open_meteo_historical_weather(
                root,
                plan,
                weather_date=date(2026, 6, 1),
                storage_policy=_storage_policy(),
                requests_per_minute=100_000,
                session=session,
                retrieved_at=retrieved_at,
            )
            measurements = _records(result.measurement_artifacts)
            mappings = _records(result.assignment_artifacts)

        self.assertEqual(session.calls[0]["url"], OPEN_METEO_HISTORICAL_WEATHER_URL)
        self.assertEqual(session.calls[0]["params"]["models"], "ecmwf_ifs")
        self.assertEqual(session.calls[0]["params"]["start_date"], "2026-06-01")
        self.assertEqual(session.calls[0]["params"]["end_date"], "2026-06-01")
        self.assertEqual(result.measurement_count, 16)
        self.assertEqual(len(measurements), 16)
        self.assertEqual(
            {record["variable"] for record in measurements},
            {
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_u_10m",
                "wind_v_10m",
            },
        )
        self.assertTrue(all(record["retrieved_at"] == "2026-08-25T12:00:00Z" for record in measurements))
        self.assertEqual(mappings[0]["source_example_ids"], ["example-001"])
        self.assertEqual(mappings[0]["source_grid_id"], measurements[0]["source_grid_id"])

    def test_rejects_any_historical_weather_model_other_than_ecmwf_ifs(self):
        plan = plan_candidate_example_weather_tiles(pd.DataFrame([_example()]))
        session = _Session([])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "ecmwf_ifs"):
                capture_open_meteo_historical_weather(
                    Path(directory) / "data",
                    plan,
                    weather_date=date(2026, 6, 1),
                    storage_policy=_storage_policy(),
                    model="gfs_seamless",
                    session=session,
                )
        self.assertEqual(session.calls, [])

    def test_second_consecutive_429_is_retained_and_pauses(self):
        first = _example(54.0, -106.0, "example-001")
        second = _example(60.0, -120.0, "example-002")
        plan = plan_candidate_example_weather_tiles(
            pd.DataFrame([first, second]), max_tile_distance_m=2_000
        )
        session = _Session([_Response(_payload()), _RateLimitedResponse(), _RateLimitedResponse()])

        with tempfile.TemporaryDirectory() as directory, patch(
            "wildfire_data.weather_rate_limit.time.sleep"
        ):
            root = Path(directory) / "data"
            result = capture_open_meteo_historical_weather(
                root,
                plan,
                weather_date=date(2026, 6, 1),
                storage_policy=_storage_policy(),
                requests_per_minute=100_000,
                batch_size=1,
                rate_limit_cooldown_seconds=1,
                session=session,
                retrieved_at=datetime(2026, 8, 25, 12, tzinfo=timezone.utc),
            )
            coverage = CoverageLedger(root).entries()

        self.assertTrue(result.paused_for_rate_limit)
        self.assertEqual(result.captured_tile_count, 1)
        self.assertEqual(result.rate_limit_retries, 2)
        self.assertEqual([entry.status for entry in coverage], [CoverageStatus.COMPLETE, CoverageStatus.FAILED])

    def test_backfill_requires_completed_candidate_manifest_and_records_immutable_identity(self):
        rows = [_candidate_row()]
        session = _Session([_Response(_payload())])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            with self.assertRaisesRegex(TypeError, "candidate_manifest"):
                backfill_open_meteo_historical_weather(
                    root,
                    storage_policy=_storage_policy(),
                    session=_Session([]),
                )
            candidate_manifest, expected_identity = _write_completed_candidate_manifest(root, rows)
            result = backfill_open_meteo_historical_weather(
                root,
                storage_policy=_storage_policy(),
                candidate_manifest=candidate_manifest,
                requests_per_minute=100_000,
                session=session,
                retrieved_at=datetime(2026, 8, 25, 12, tzinfo=timezone.utc),
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertTrue(result.complete)
        self.assertEqual(result.weather_date_count, 1)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["reports"][0]["weather_date"], "2026-06-01")
        self.assertEqual(manifest["candidate_manifest"], expected_identity)

    def test_missing_required_weather_at_an_anchor_marks_backfill_incomplete(self):
        rows = [_candidate_row(anchor_at="2026-06-01T10:45:00Z")]
        payload = _payload()
        payload[0]["hourly"]["wind_direction_10m"][0] = None
        session = _Session([_Response(payload)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            candidate_manifest, _ = _write_completed_candidate_manifest(root, rows)
            result = backfill_open_meteo_historical_weather(
                root,
                storage_policy=_storage_policy(),
                candidate_manifest=candidate_manifest,
                requests_per_minute=100_000,
                session=session,
                retrieved_at=datetime(2026, 8, 25, 12, tzinfo=timezone.utc),
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertFalse(result.complete)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["reports"][0]["status"], "failed")

    def test_storage_budget_pause_publishes_a_resumable_partial_manifest(self):
        rows = [_candidate_row()]
        constrained_policy = StorageBudgetPolicy(
            schema_version=1,
            whole_data_cap_bytes=1,
            whole_data_cap_label="test",
            scope="test root",
            categories=(
                StorageBudgetCategory(
                    key="issued_weather_tiles",
                    cap_bytes=1,
                    priority_score=85,
                    pinned=True,
                    retention="weather",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            candidate_manifest, _ = _write_completed_candidate_manifest(root, rows)
            session = _Session([])
            result = backfill_open_meteo_historical_weather(
                root,
                storage_policy=constrained_policy,
                candidate_manifest=candidate_manifest,
                session=session,
                retrieved_at=datetime(2026, 8, 25, 12, tzinfo=timezone.utc),
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertFalse(result.complete)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["reports"][0]["status"], "failed")
        self.assertEqual(session.calls, [])

    def test_resume_retains_completed_reports_and_only_collects_unfinished_dates(self):
        rows = [
            _candidate_row(anchor_at="2026-06-01T10:45:00Z"),
            _candidate_row(
                latitude=60.0,
                longitude=-120.0,
                identifier="example-002",
                anchor_at="2026-06-02T10:45:00Z",
            ),
        ]
        first_session = _Session(
            [
                _Response(_payload(weather_date=date(2026, 6, 1))),
                _RateLimitedResponse(),
                _RateLimitedResponse(),
            ]
        )
        resumed_session = _Session([_Response(_payload(weather_date=date(2026, 6, 2)))])
        with tempfile.TemporaryDirectory() as directory, patch(
            "wildfire_data.weather_rate_limit.time.sleep"
        ):
            root = Path(directory) / "data"
            candidate_manifest, _ = _write_completed_candidate_manifest(root, rows)
            partial = backfill_open_meteo_historical_weather(
                root,
                storage_policy=_storage_policy(),
                candidate_manifest=candidate_manifest,
                requests_per_minute=100_000,
                rate_limit_cooldown_seconds=1,
                session=first_session,
                retrieved_at=datetime(2026, 8, 25, 12, tzinfo=timezone.utc),
            )
            partial_manifest = json.loads(partial.manifest_path.read_text(encoding="utf-8"))
            resumed = backfill_open_meteo_historical_weather(
                root,
                storage_policy=_storage_policy(),
                candidate_manifest=candidate_manifest,
                resume_manifest=partial.manifest_path,
                requests_per_minute=100_000,
                session=resumed_session,
                retrieved_at=datetime(2026, 8, 25, 13, tzinfo=timezone.utc),
            )
            resumed_manifest = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))

        self.assertFalse(partial.complete)
        self.assertEqual(partial_manifest["reports"][0]["status"], "complete")
        self.assertEqual(partial_manifest["reports"][1]["status"], "partial")
        self.assertTrue(resumed.complete)
        self.assertEqual(
            [report["weather_date"] for report in resumed_manifest["reports"]],
            ["2026-06-01", "2026-06-02"],
        )
        self.assertEqual(resumed_manifest["reports"][0], partial_manifest["reports"][0])
        self.assertEqual(len(resumed_session.calls), 1)
        self.assertEqual(resumed_session.calls[0]["params"]["start_date"], "2026-06-02")

    def test_floor_weather_hour_never_moves_to_a_later_hour(self):
        self.assertEqual(
            floor_weather_hour("2026-06-01T10:59:59Z"),
            datetime(2026, 6, 1, 10, tzinfo=timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
