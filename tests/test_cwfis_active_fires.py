import gzip
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from wildfire_data.cwfis_active_fires import (
    CWFIS_ACTIVE_FIRES_WFS_URL,
    collect_cwfis_active_fire_history,
    cwfis_query_parameters,
    cwfis_record_start_filter,
)
from wildfire_data.data_archive import CoverageStatus
from wildfire_data.storage_budget import load_storage_budget


class _FakeResponse:
    def __init__(self, content, *, status_code=200):
        self.content = content
        self.status_code = status_code
        self.headers = {"Content-Type": "application/geo+json"}
        self.url = f"{CWFIS_ACTIVE_FIRES_WFS_URL}?token=private"


class _FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def _policy(path, *, total=10_000_000, labels_cap=10_000_000):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "whole_data_cap_bytes": total,
                "whole_data_cap_label": "test",
                "scope": "test data",
                "categories": [
                    {
                        "key": "operational_labels_and_progression",
                        "cap_bytes": labels_cap,
                        "priority_score": 95,
                        "pinned": True,
                        "retention": "labels",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return load_storage_budget(path)


def _feature(identifier):
    return {
        "type": "Feature",
        "properties": {
            "id": identifier,
            "national_fire_id": f"2026_ON_TEST_{identifier}",
            "record_start": "2026-07-01T00:45:00Z",
            "record_end": "2026-07-01T19:45:00Z",
            "status_date": "2026-07-01T00:45:00Z",
            "fire_size": 10,
        },
        "geometry": {"type": "Point", "coordinates": [-91.4, 48.5]},
    }


def _payload(features, *, total):
    return json.dumps({"type": "FeatureCollection", "features": features, "totalFeatures": total}).encode()


class CwfisActiveFireTests(unittest.TestCase):
    def test_builds_an_inclusive_range_record_start_filter(self):
        where = cwfis_record_start_filter(date(2026, 5, 31), date(2026, 8, 10))
        parameters = cwfis_query_parameters(
            date(2026, 5, 31), date(2026, 8, 10), start_index=1_000
        )

        self.assertIn("2026-05-31T00:00:00Z", where)
        self.assertIn("2026-08-11T00:00:00Z", where)
        self.assertEqual(parameters["startIndex"], 1_000)
        self.assertEqual(parameters["sortBy"], "record_start,id")
        self.assertEqual(parameters["typeNames"], "public:cwfif_national_activefires")

    def test_archives_record_intervals_as_incident_context_not_spread_labels(self):
        session = _FakeSession([_FakeResponse(_payload([_feature(1)], total=1))])
        with tempfile.TemporaryDirectory() as directory:
            result = collect_cwfis_active_fire_history(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=_policy(Path(directory) / "budget.json"),
                session=session,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            artifact = result.pages[0].normalized_artifact
            with gzip.open(artifact.artifact_path, "rt", encoding="utf-8") as source:
                normalized = json.loads(source.readline())
            raw_manifest = result.pages[0].raw_artifact.manifest_path.read_text(encoding="utf-8")

        self.assertEqual(result.coverage.status, CoverageStatus.COMPLETE)
        self.assertEqual(result.feature_count, 1)
        self.assertEqual(normalized["record_role"], "operational_incident_context")
        self.assertTrue(normalized["historical_record_interval_preserved"])
        self.assertEqual(normalized["incident_context_quality_score"], 0.8)
        self.assertNotIn("private", raw_manifest)

    def test_paginates_using_the_provider_matched_feature_count(self):
        session = _FakeSession(
            [
                _FakeResponse(_payload([_feature(1)], total=2)),
                _FakeResponse(_payload([_feature(2)], total=2)),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = collect_cwfis_active_fire_history(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=_policy(Path(directory) / "budget.json"),
                page_size=1,
                session=session,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )

        self.assertEqual(result.feature_count, 2)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[1][1]["params"]["startIndex"], 1)

    def test_records_partial_before_persisting_when_quota_would_be_exceeded(self):
        session = _FakeSession([_FakeResponse(_payload([_feature(1)], total=1))])
        with tempfile.TemporaryDirectory() as directory:
            result = collect_cwfis_active_fire_history(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=_policy(Path(directory) / "budget.json", total=100, labels_cap=100),
                session=session,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )

        self.assertEqual(result.coverage.status, CoverageStatus.PARTIAL)
        self.assertEqual(result.pages, ())

    def test_retains_a_non_successful_provider_response_as_raw_evidence(self):
        session = _FakeSession([_FakeResponse(b"provider error", status_code=400)])
        with tempfile.TemporaryDirectory() as directory:
            result = collect_cwfis_active_fire_history(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=_policy(Path(directory) / "budget.json"),
                session=session,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            raw_paths = list((Path(directory) / "raw" / "cwfis-cwfis-active-fires").glob("*.gz"))
            with gzip.open(raw_paths[0], "rb") as source:
                raw_payload = source.read()

        self.assertEqual(result.coverage.status, CoverageStatus.FAILED)
        self.assertEqual(raw_payload, b"provider error")

    def test_skips_already_terminal_history_by_default(self):
        first_session = _FakeSession([_FakeResponse(_payload([_feature(1)], total=1))])
        repeat_session = _FakeSession([])
        with tempfile.TemporaryDirectory() as directory:
            policy = _policy(Path(directory) / "budget.json")
            collect_cwfis_active_fire_history(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=policy,
                session=first_session,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            result = collect_cwfis_active_fire_history(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=policy,
                session=repeat_session,
                retrieved_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
            )

        self.assertTrue(result.skipped_terminal_coverage)
        self.assertEqual(repeat_session.calls, [])


if __name__ == "__main__":
    unittest.main()
