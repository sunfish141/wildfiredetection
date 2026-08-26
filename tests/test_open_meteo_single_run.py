import gzip
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

from wildfire_data.data_archive import CoverageLedger, CoverageStatus
from wildfire_data.open_meteo_single_run import (
    OPEN_METEO_SINGLE_RUNS_URL,
    OpenMeteoSingleRunError,
    capture_open_meteo_single_run,
    plan_firms_candidate_weather_tiles,
)
from wildfire_data.storage_budget import StorageBudgetCategory, StorageBudgetPolicy
from wildfire_data.training_grid import cell_from_wgs84


class _FakeResponse:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    content = b'{"provider":"fake-open-meteo"}'

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return _FakeResponse(self._payload)


class _RateLimitedResponse:
    status_code = 429
    headers = {"Retry-After": "0"}
    content = b'{"reason":"rate limited"}'

    def raise_for_status(self):
        raise RuntimeError("HTTP 429")


class _BadRequestResponse:
    status_code = 400
    headers = {"Content-Type": "application/json"}
    content = b'{"reason":"unknown model run"}'

    def raise_for_status(self):
        error = requests.HTTPError("HTTP 400")
        error.response = self
        raise error


class _SequencedSession:
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
                retention="issued forecast tiles",
            ),
        ),
    )


def _normalized_records(collection_results):
    records = []
    for collection in collection_results:
        for artifact in collection.normalized_artifacts:
            with gzip.open(artifact.artifact_path, "rt", encoding="utf-8") as source:
                records.extend(json.loads(line) for line in source if line.strip())
    return records


class OpenMeteoSingleRunTests(unittest.TestCase):
    def test_plans_tiles_from_candidate_grid_cells_not_only_detection_points(self):
        fires = pd.DataFrame(
            {
                "latitude": [54.0],
                "longitude": [-106.0],
                "detection_id": ["firms-001"],
                "acquired_at": ["2026-08-10T02:40:00Z"],
                "raw_artifact_id": ["a" * 64],
            }
        )

        plan = plan_firms_candidate_weather_tiles(
            fires,
            candidate_radius_cells=1,
            max_tile_distance_m=2_000,
        )

        detection_cell = cell_from_wgs84(latitude=54.0, longitude=-106.0).cell_id
        self.assertGreater(len(plan.assignments), len(fires))
        self.assertIn(detection_cell, set(plan.assignments["candidate_cell_id"]))
        self.assertTrue(
            set(plan.assignments["forecast_tile_id"]).issubset(
                set(plan.tiles["forecast_tile_id"])
            )
        )
        self.assertTrue((plan.assignments["forecast_tile_distance_m"] <= 2_000).all())
        self.assertTrue(
            all(ids == ["firms-001"] for ids in plan.assignments["source_firms_detection_ids"])
        )
        self.assertTrue(
            all(
                value == "2026-08-10T02:40:00Z"
                for value in plan.assignments["latest_source_firms_acquired_at"]
            )
        )
        self.assertTrue(
            all(ids == ["a" * 64] for ids in plan.assignments["source_firms_raw_artifact_ids"])
        )

    def test_archives_only_future_valid_hours_with_capture_availability_provenance(self):
        fires = pd.DataFrame(
            {"latitude": [54.0], "longitude": [-106.0], "raw_artifact_id": ["b" * 64]}
        )
        plan = plan_firms_candidate_weather_tiles(
            fires,
            candidate_radius_cells=0,
            max_tile_distance_m=2_000,
        )
        # The first timestamp is the retrieval cutoff and must not be usable
        # as a future forecast value.  The second one is retained.
        response_payload = [
            {
                "latitude": 54.0,
                "longitude": -106.0,
                "hourly_units": {
                    "temperature_2m": "°C",
                    "relative_humidity_2m": "%",
                    "precipitation": "mm",
                    "wind_speed_10m": "m/s",
                    "wind_direction_10m": "°",
                },
                "hourly": {
                    "time": ["2026-08-10T03:00", "2026-08-10T04:00"],
                    "temperature_2m": [20.0, 21.0],
                    "relative_humidity_2m": [35.0, 34.0],
                    "precipitation": [0.0, 1.5],
                    "wind_speed_10m": [5.0, 10.0],
                    "wind_direction_10m": [0.0, 90.0],
                },
            }
        ]
        session = _FakeSession(response_payload)
        retrieved_at = datetime(2026, 8, 10, 3, 0, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as directory:
            result = capture_open_meteo_single_run(
                Path(directory) / "data",
                plan,
                model="ecmwf_ifs",
                model_run_at="2026-08-10T00:00:00Z",
                forecast_horizon_hours=2,
                storage_policy=_storage_policy(),
                requests_per_minute=100_000,
                batch_size=50,
                session=session,
                retrieved_at=retrieved_at,
            )
            records = _normalized_records(result.collection_results)
            assignment_manifest = json.loads(
                result.assignment_artifacts[0].manifest_path.read_text(encoding="utf-8")
            )

        self.assertEqual(session.calls[0]["url"], OPEN_METEO_SINGLE_RUNS_URL)
        self.assertEqual(session.calls[0]["params"]["models"], "ecmwf_ifs")
        self.assertIn("run", session.calls[0]["params"])
        # Forecast hours are relative to the named run (00:00), not the
        # response availability (03:00). The request reaches the 05:00
        # boundary needed for a two-hour forward window after capture.
        self.assertEqual(session.calls[0]["params"]["forecast_hours"], 6)
        self.assertEqual(result.measurement_count, 7)
        self.assertEqual(len(records), 7)
        self.assertTrue(all(record["valid_at"] == "2026-08-10T04:00:00Z" for record in records))
        self.assertTrue(all(record["valid_at"] > record["availability_at"] for record in records))
        self.assertTrue(
            all(record["availability_at"] == "2026-08-10T03:00:00Z" for record in records)
        )
        self.assertTrue(all("published_at" not in record for record in records))

        by_variable = {record["variable"]: record for record in records}
        self.assertEqual(set(by_variable), {
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_u_10m",
            "wind_v_10m",
        })
        self.assertEqual(by_variable["temperature_2m"]["value"], 21.0)
        self.assertAlmostEqual(by_variable["wind_u_10m"]["value"], -10.0, places=6)
        self.assertAlmostEqual(by_variable["wind_v_10m"]["value"], 0.0, places=6)
        self.assertIn("b" * 64, assignment_manifest["raw_artifact_ids"])
        self.assertIn(
            result.collection_results[0].raw_artifact.raw_artifact_id,
            assignment_manifest["raw_artifact_ids"],
        )

    def test_requires_archived_firms_lineage_before_making_a_weather_request(self):
        plan = plan_firms_candidate_weather_tiles(
            pd.DataFrame({"latitude": [54.0], "longitude": [-106.0]}),
            candidate_radius_cells=0,
        )
        session = _FakeSession([])

        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "source FIRMS raw_artifact_id"
        ):
            capture_open_meteo_single_run(
                Path(directory) / "data",
                plan,
                model="ecmwf_ifs",
                model_run_at="2026-08-10T00:00:00Z",
                forecast_horizon_hours=2,
                storage_policy=_storage_policy(),
                session=session,
                retrieved_at=datetime(2026, 8, 10, 3, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(session.calls, [])

    def test_preserves_a_bad_provider_response_before_raising_a_parse_error(self):
        plan = plan_firms_candidate_weather_tiles(
            pd.DataFrame(
                {
                    "latitude": [54.0],
                    "longitude": [-106.0],
                    "raw_artifact_id": ["c" * 64],
                }
            ),
            candidate_radius_cells=0,
        )
        session = _FakeSession([{"latitude": 54.0, "longitude": -106.0, "hourly": {}}])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            with self.assertRaises(OpenMeteoSingleRunError):
                capture_open_meteo_single_run(
                    root,
                    plan,
                    model="ecmwf_ifs",
                    model_run_at="2026-08-10T00:00:00Z",
                    forecast_horizon_hours=2,
                    storage_policy=_storage_policy(),
                    session=session,
                    retrieved_at=datetime(2026, 8, 10, 3, 0, tzinfo=timezone.utc),
                )
            coverage = CoverageLedger(root).entries()
            raw_artifacts = list((root / "raw" / "open-meteo-single-runs").glob("*.gz"))

        self.assertEqual([entry.status for entry in coverage], [CoverageStatus.FAILED])
        self.assertEqual(len(raw_artifacts), 1)

    def test_archives_the_final_429_response_and_returns_a_paused_result(self):
        plan = plan_firms_candidate_weather_tiles(
            pd.DataFrame(
                {
                    "latitude": [54.0, 60.0],
                    "longitude": [-106.0, -120.0],
                    "raw_artifact_id": ["d" * 64, "e" * 64],
                }
            ),
            candidate_radius_cells=0,
            max_tile_distance_m=2_000,
        )
        success_payload = [
            {
                "latitude": 54.0,
                "longitude": -106.0,
                "hourly_units": {"temperature_2m": "°C"},
                "hourly": {"time": ["2026-08-10T04:00"], "temperature_2m": [21.0]},
            }
        ]
        session = _SequencedSession(
            [_FakeResponse(success_payload), _RateLimitedResponse(), _RateLimitedResponse()]
        )

        with tempfile.TemporaryDirectory() as directory, patch(
            "wildfire_data.weather_rate_limit.time.sleep"
        ):
            result = capture_open_meteo_single_run(
                Path(directory) / "data",
                plan,
                model="ecmwf_ifs",
                model_run_at="2026-08-10T00:00:00Z",
                forecast_horizon_hours=2,
                storage_policy=_storage_policy(),
                requests_per_minute=100_000,
                batch_size=1,
                rate_limit_cooldown_seconds=1,
                session=session,
                retrieved_at=datetime(2026, 8, 10, 3, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(result.paused_for_rate_limit)
        self.assertEqual(result.captured_tile_count, 1)
        self.assertEqual(result.rate_limit_retries, 2)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(
            [collection.coverage.status for collection in result.collection_results],
            [CoverageStatus.COMPLETE, CoverageStatus.FAILED],
        )

    def test_preserves_an_unsuccessful_provider_response_before_raising(self):
        plan = plan_firms_candidate_weather_tiles(
            pd.DataFrame(
                {
                    "latitude": [54.0],
                    "longitude": [-106.0],
                    "raw_artifact_id": ["f" * 64],
                }
            ),
            candidate_radius_cells=0,
        )
        session = _SequencedSession([_BadRequestResponse()])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            with self.assertRaises(OpenMeteoSingleRunError):
                capture_open_meteo_single_run(
                    root,
                    plan,
                    model="ecmwf_ifs",
                    model_run_at="2026-08-10T00:00:00Z",
                    forecast_horizon_hours=2,
                    storage_policy=_storage_policy(),
                    session=session,
                    retrieved_at=datetime(2026, 8, 10, 3, 0, tzinfo=timezone.utc),
                )
            coverage = CoverageLedger(root).entries()
            raw_artifacts = list((root / "raw" / "open-meteo-single-runs").glob("*.gz"))

        self.assertEqual([entry.status for entry in coverage], [CoverageStatus.FAILED])
        self.assertEqual(len(raw_artifacts), 1)

    def test_refuses_a_model_run_that_was_not_yet_available_at_capture(self):
        fires = pd.DataFrame({"latitude": [54.0], "longitude": [-106.0]})
        plan = plan_firms_candidate_weather_tiles(fires, candidate_radius_cells=0)
        session = _FakeSession([])
        retrieved_at = datetime(2026, 8, 10, 3, 0, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "model_run_at.*capture"
        ):
            capture_open_meteo_single_run(
                Path(directory) / "data",
                plan,
                model="ecmwf_ifs",
                model_run_at="2026-08-10T04:00:00Z",
                forecast_horizon_hours=2,
                storage_policy=_storage_policy(),
                requests_per_minute=100_000,
                session=session,
                retrieved_at=retrieved_at,
            )

        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()
