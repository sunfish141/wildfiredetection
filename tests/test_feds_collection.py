import gzip
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from wildfire_data.data_archive import CoverageLedger, CoverageStatus, write_raw_artifact
from wildfire_data.feds_collection import (
    FEDS_PERIMETERS_LAYER_URL,
    FEDS_PERIMETERS_QUERY_URL,
    _observed_snapshot_expected_coverage_id,
    collect_feds_perimeters,
    feds_query_parameters,
    iter_feds_snapshot_windows,
    rebuild_feds_primarykey_normalization,
)
from wildfire_data.feds_labels import load_feds_snapshot_records
from wildfire_data.storage_budget import load_storage_budget


class _FakeResponse:
    def __init__(self, *, status_code=200, content=b"", headers=None, url="https://example.test"):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"Content-Type": "application/json"}
        self.url = url


class _FakeSession:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.outcomes)


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
                        "priority_score": 90,
                        "pinned": True,
                        "retention": "labels",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return load_storage_budget(path)


def _metadata(start=1782864000000, end=1782907200000):
    return json.dumps({"timeInfo": {"timeExtent": [start, end]}}).encode("utf-8")


def _feature(timestamp, identifier="CONUS|1|2026-07-01T00:00:00"):
    return {
        "attributes": {
            "primarykey": identifier,
            "region": "CONUS",
            "fireid": 1.0,
            "t": timestamp,
            "n_newpixels": 5,
            "geom_counts": float("nan"),
        },
        "geometry": {
            "rings": [[[-110.0, 50.0], [-109.9, 50.0], [-109.9, 50.1], [-110.0, 50.0]]]
        },
    }


def _page(features, *, has_next=False):
    return json.dumps(
        {"features": features, "exceededTransferLimit": has_next},
        allow_nan=True,
    ).encode("utf-8")


def _archive_feds_page(directory, *, capture_at, features, snapshot_start, snapshot_end):
    return write_raw_artifact(
        directory,
        source="NASA FEDS",
        payload=_page(features),
        retrieved_at=capture_at,
        media_type="application/json",
        provenance={
            "stage": "query-batch-page",
            "region": "CONUS+Canada",
            "snapshot_start": snapshot_start,
            "snapshot_end": snapshot_end,
        },
    )


class FedsCollectionTests(unittest.TestCase):
    def test_builds_source_aligned_12_hour_windows_and_query_parameters(self):
        windows = iter_feds_snapshot_windows(date(2026, 7, 1), date(2026, 7, 1))
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0][0], datetime(2026, 7, 1, tzinfo=timezone.utc))
        parameters = feds_query_parameters(*windows[0], offset=2_000, region_names=("CONUS", "Canada"))
        self.assertEqual(parameters["time"], "1782864000000,1782907199999")
        self.assertEqual(parameters["resultOffset"], 2_000)
        self.assertEqual(parameters["where"], "region IN ('CONUS', 'Canada')")
        self.assertEqual(parameters["f"], "json")

    def test_archives_normalizes_and_marks_feds_as_a_weak_satellite_tier(self):
        session = _FakeSession(
            [
                _FakeResponse(content=_metadata(), url=f"{FEDS_PERIMETERS_LAYER_URL}?f=json"),
                _FakeResponse(content=_page([_feature(1782864000000)]), url=FEDS_PERIMETERS_QUERY_URL),
                _FakeResponse(
                    content=_page([_feature(1782907200000, "CONUS|1|2026-07-01T12:00:00")]),
                    url=FEDS_PERIMETERS_QUERY_URL,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = collect_feds_perimeters(
                directory,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                storage_budget=_policy(Path(directory) / "budget.json"),
                session=session,
                snapshot_windows_per_request=1,
                retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )
            artifact = result.windows[0].pages[0].normalized_artifact
            with gzip.open(artifact.artifact_path, "rt", encoding="utf-8") as file:
                record = json.loads(file.readline())
            entries = CoverageLedger(directory).entries()

        self.assertEqual(result.coverage.status, CoverageStatus.COMPLETE)
        self.assertEqual(result.feature_count, 2)
        self.assertEqual(record["label_tier"], "weak_satellite")
        self.assertEqual(record["source_snapshot_time"], "2026-07-01T00:00:00Z")
        self.assertEqual(record["geometry"]["encoding"], "esri-rings-wgs84/v1")
        self.assertIsNone(record["source_fields"]["geom_counts"])
        self.assertTrue(record["time_alignment_eligible"])
        self.assertEqual(len(session.calls), 3)
        # Two requested query windows, two actual primary-key snapshot
        # observations, and the requested range are all explicitly logged.
        self.assertEqual([entry.status for entry in entries].count(CoverageStatus.COMPLETE), 5)

    def test_marks_windows_beyond_advertised_source_time_as_partial_not_empty_labels(self):
        session = _FakeSession(
            [
                _FakeResponse(content=_metadata(start=1782864000000, end=1782864000000)),
                _FakeResponse(content=_page([_feature(1782864000000)]), url=FEDS_PERIMETERS_QUERY_URL),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = collect_feds_perimeters(
                directory,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                storage_budget=_policy(Path(directory) / "budget.json"),
                session=session,
                snapshot_windows_per_request=1,
                retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )

        self.assertEqual(result.windows[0].coverage.status, CoverageStatus.COMPLETE)
        self.assertEqual(result.windows[1].coverage.status, CoverageStatus.PARTIAL)
        self.assertEqual(result.coverage.status, CoverageStatus.PARTIAL)
        self.assertEqual(len(session.calls), 2)

    def test_batches_consecutive_source_windows_but_normalizes_them_separately(self):
        session = _FakeSession(
            [
                _FakeResponse(content=_metadata(), url=f"{FEDS_PERIMETERS_LAYER_URL}?f=json"),
                _FakeResponse(
                    content=_page(
                        [
                            _feature(1782864000000),
                            _feature(1782907200000, "CONUS|1|2026-07-01T12:00:00"),
                        ]
                    ),
                    url=FEDS_PERIMETERS_QUERY_URL,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = collect_feds_perimeters(
                directory,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                storage_budget=_policy(Path(directory) / "budget.json"),
                session=session,
                retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )

        self.assertEqual(result.coverage.status, CoverageStatus.COMPLETE)
        self.assertEqual(result.feature_count, 2)
        self.assertEqual(len(result.windows[0].pages), 1)
        self.assertEqual(len(result.windows[1].pages), 1)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[1][1]["params"]["time"], "1782864000000,1782950399999")

    def test_uses_primarykey_timestamp_when_provider_t_is_a_query_window_value(self):
        feature = _feature(1782864000000, "CONUS|1|2026-07-01T12:00:00")
        session = _FakeSession(
            [
                _FakeResponse(content=_metadata()),
                _FakeResponse(content=_page([feature]), url=FEDS_PERIMETERS_QUERY_URL),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = collect_feds_perimeters(
                directory,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                storage_budget=_policy(Path(directory) / "budget.json"),
                session=session,
                retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )
            artifact = result.windows[1].pages[0].normalized_artifact
            with gzip.open(artifact.artifact_path, "rt", encoding="utf-8") as file:
                record = json.loads(file.readline())
            entries = {
                entry.expected_coverage_id: entry
                for entry in CoverageLedger(directory).entries()
                if entry.expected_coverage_id is not None
            }

        self.assertEqual(result.windows[0].feature_count, 0)
        self.assertEqual(result.windows[0].coverage.status, CoverageStatus.PARTIAL)
        self.assertEqual(result.windows[1].feature_count, 1)
        self.assertEqual(result.windows[1].coverage.status, CoverageStatus.COMPLETE)
        self.assertEqual(result.coverage.status, CoverageStatus.PARTIAL)
        self.assertEqual(record["source_snapshot_time"], "2026-07-01T12:00:00Z")
        self.assertEqual(record["source_snapshot_time_source"], "primarykey")
        self.assertFalse(record["provider_t_matches_primarykey"])
        observed = entries[_observed_snapshot_expected_coverage_id(
            datetime(2026, 7, 1, 12, tzinfo=timezone.utc), "CONUS+Canada"
        )]
        self.assertEqual(observed.status, CoverageStatus.COMPLETE)

    def test_skips_a_terminal_range_without_more_requests(self):
        first_session = _FakeSession(
            [
                _FakeResponse(content=_metadata()),
                _FakeResponse(content=_page([_feature(1782864000000)])),
                _FakeResponse(
                    content=_page([_feature(1782907200000, "CONUS|1|2026-07-01T12:00:00")])
                ),
            ]
        )
        repeated_session = _FakeSession([])
        with tempfile.TemporaryDirectory() as directory:
            policy = _policy(Path(directory) / "budget.json")
            collect_feds_perimeters(
                directory,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                storage_budget=policy,
                session=first_session,
                snapshot_windows_per_request=1,
                retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )
            result = collect_feds_perimeters(
                directory,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                storage_budget=policy,
                session=repeated_session,
                snapshot_windows_per_request=1,
                retrieved_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
            )

        self.assertTrue(result.skipped_terminal_coverage)
        self.assertEqual(repeated_session.calls, [])

    def test_raw_rebuild_selects_one_latest_capture_instead_of_merging_revisions(self):
        source_start = "2026-07-01T00:00:00Z"
        source_end = "2026-07-01T12:00:00Z"
        with tempfile.TemporaryDirectory() as directory:
            policy = _policy(Path(directory) / "budget.json")
            _archive_feds_page(
                directory,
                capture_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
                features=[_feature(1782864000000, "CONUS|1|2026-07-01T00:00:00")],
                snapshot_start=source_start,
                snapshot_end=source_end,
            )
            latest = _archive_feds_page(
                directory,
                capture_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
                features=[_feature(1782864000000, "CONUS|2|2026-07-01T12:00:00")],
                snapshot_start=source_start,
                snapshot_end=source_end,
            )
            report = rebuild_feds_primarykey_normalization(
                directory,
                storage_budget=policy,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                generated_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
            )
            pinned_report = rebuild_feds_primarykey_normalization(
                directory,
                storage_budget=policy,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                captured_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
                generated_at=datetime(2026, 8, 22, 1, tzinfo=timezone.utc),
            )
            records = load_feds_snapshot_records(
                directory,
                source_snapshot_time=datetime(2026, 7, 1, 12, tzinfo=timezone.utc),
            )

        self.assertEqual(report.status, CoverageStatus.COMPLETE)
        self.assertEqual(report.selected_capture_at, "2026-08-21T00:00:00Z")
        self.assertEqual(report.feature_count, 1)
        self.assertEqual(pinned_report.selected_capture_at, "2026-08-20T00:00:00Z")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_record_id"], "CONUS|2|2026-07-01T12:00:00")
        self.assertEqual(records[0]["raw_artifact_id"], latest.raw_artifact_id)

    def test_raw_rebuild_accepts_volatile_esri_oid_duplicates_but_marks_real_conflicts_partial(self):
        source_start = "2026-07-01T00:00:00Z"
        source_end = "2026-07-01T12:00:00Z"
        capture_at = datetime(2026, 8, 21, tzinfo=timezone.utc)
        duplicate_first = _feature(1782864000000)
        duplicate_second = _feature(1782864000000)
        duplicate_first["attributes"]["ESRI_OID"] = 1
        duplicate_second["attributes"]["ESRI_OID"] = 2
        conflict = _feature(1782864000000)
        conflict["geometry"]["rings"][0][1][0] = -109.8
        with tempfile.TemporaryDirectory() as directory:
            policy = _policy(Path(directory) / "budget.json")
            _archive_feds_page(
                directory,
                capture_at=capture_at,
                features=[duplicate_first],
                snapshot_start=source_start,
                snapshot_end=source_end,
            )
            _archive_feds_page(
                directory,
                capture_at=capture_at,
                features=[duplicate_second],
                snapshot_start=source_start,
                snapshot_end=source_end,
            )
            first_report = rebuild_feds_primarykey_normalization(
                directory,
                storage_budget=policy,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                generated_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
            )
            _archive_feds_page(
                directory,
                capture_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
                features=[duplicate_first, conflict],
                snapshot_start=source_start,
                snapshot_end=source_end,
            )
            second_report = rebuild_feds_primarykey_normalization(
                directory,
                storage_budget=policy,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 1),
                generated_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
            )

        self.assertEqual(first_report.status, CoverageStatus.COMPLETE)
        self.assertEqual(first_report.duplicate_record_count, 1)
        self.assertEqual(first_report.conflicting_record_count, 0)
        self.assertEqual(second_report.status, CoverageStatus.PARTIAL)
        self.assertEqual(second_report.conflicting_record_count, 1)


if __name__ == "__main__":
    unittest.main()
