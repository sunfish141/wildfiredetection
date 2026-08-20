import gzip
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from wildfire_data.data_archive import CoverageLedger, CoverageStatus
from wildfire_data.storage_budget import load_storage_budget
from wildfire_data.wfigs_collection import (
    WFIGS_YEAR_TO_DATE_QUERY_URL,
    collect_wfigs_year_to_date,
    wfigs_query_parameters,
    wfigs_where_clause,
)


class _FakeResponse:
    def __init__(self, *, status_code=200, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"Content-Type": "application/geo+json"}
        self.url = f"{WFIGS_YEAR_TO_DATE_QUERY_URL}?token=private"


class _FakeSession:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.outcomes)


def _policy(path, *, total=10_000_000, labels_cap=10_000_000):
    document = {
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
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_storage_budget(path)


def _feature(identifier):
    return {
        "type": "Feature",
        "properties": {
            "poly_IRWINID": identifier,
            "poly_PolygonDateTime": 1780272000000,
            "poly_MapMethod": "GPS",
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-110.0, 50.0], [-109.9, 50.0], [-109.9, 50.1], [-110.0, 50.0]]],
        },
    }


def _payload(features, *, has_next=False):
    result = {"type": "FeatureCollection", "features": features}
    if has_next:
        result["exceededTransferLimit"] = True
    return json.dumps(result).encode("utf-8")


class WfigsCollectionTests(unittest.TestCase):
    def test_builds_a_half_open_date_filter(self):
        where = wfigs_where_clause(date(2026, 5, 31), date(2026, 8, 10))
        parameters = wfigs_query_parameters(date(2026, 5, 31), date(2026, 8, 10), offset=2_000)

        self.assertIn("2026-05-31", where)
        self.assertIn("2026-08-11", where)
        self.assertEqual(parameters["resultOffset"], 2_000)
        self.assertEqual(parameters["f"], "geojson")

    def test_archives_and_normalizes_reference_geometry_with_quality_and_retention_scores(self):
        session = _FakeSession([_FakeResponse(content=_payload([_feature("incident-1")]))])
        retrieved_at = datetime(2026, 8, 16, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            policy = _policy(Path(directory) / "budget.json")
            result = collect_wfigs_year_to_date(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=policy,
                session=session,
                retrieved_at=retrieved_at,
            )
            artifact = result.pages[0].normalized_artifact
            with gzip.open(artifact.artifact_path, "rt", encoding="utf-8") as file:
                normalized = json.loads(file.readline())
            raw_manifest = result.pages[0].raw_artifact.manifest_path.read_text(encoding="utf-8")
            entries = CoverageLedger(directory).entries()

        self.assertEqual(result.coverage.status, CoverageStatus.COMPLETE)
        self.assertEqual(result.feature_count, 1)
        self.assertEqual(normalized["label_tier"], "final_reference")
        self.assertEqual(normalized["label_quality_score"], 0.55)
        self.assertEqual(normalized["retention_priority_score"], 95)
        self.assertFalse(normalized["historical_snapshot_recreated"])
        self.assertNotIn("private", raw_manifest)
        self.assertEqual(
            [entry.status for entry in entries].count(CoverageStatus.COMPLETE),
            2,
        )

    def test_records_partial_without_writing_a_page_when_the_quota_would_be_exceeded(self):
        payload = _payload([_feature("incident-1")])
        session = _FakeSession([_FakeResponse(content=payload)])
        with tempfile.TemporaryDirectory() as directory:
            policy = _policy(Path(directory) / "budget.json", total=100, labels_cap=100)
            result = collect_wfigs_year_to_date(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=policy,
                session=session,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            coverage_text = result.coverage.path.read_text(encoding="utf-8")

        self.assertEqual(result.coverage.status, CoverageStatus.PARTIAL)
        self.assertEqual(result.pages, ())
        self.assertIn("Cannot admit", coverage_text)

    def test_skips_an_already_complete_range_unless_explicitly_refreshed(self):
        first_session = _FakeSession([_FakeResponse(content=_payload([_feature("incident-1")]))])
        repeated_session = _FakeSession([])
        with tempfile.TemporaryDirectory() as directory:
            policy = _policy(Path(directory) / "budget.json")
            collect_wfigs_year_to_date(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=policy,
                session=first_session,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            result = collect_wfigs_year_to_date(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=policy,
                session=repeated_session,
                retrieved_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
            )

        self.assertEqual(result.coverage.status, CoverageStatus.COMPLETE)
        self.assertEqual(result.pages, ())
        self.assertTrue(result.skipped_terminal_coverage)
        self.assertEqual(repeated_session.calls, [])

    def test_paginates_until_the_service_no_longer_indicates_more_features(self):
        session = _FakeSession(
            [
                _FakeResponse(content=_payload([_feature("one")], has_next=True)),
                _FakeResponse(content=_payload([_feature("two")])),
                _FakeResponse(content=_payload([])),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            policy = _policy(Path(directory) / "budget.json")
            result = collect_wfigs_year_to_date(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 8, 10),
                storage_budget=policy,
                session=session,
                page_size=1,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )

        self.assertEqual(result.feature_count, 2)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(session.calls[1][1]["params"]["resultOffset"], 1)


if __name__ == "__main__":
    unittest.main()
