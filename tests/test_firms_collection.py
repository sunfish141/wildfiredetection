import gzip
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from wildfire_data.data_archive import CoverageLedger, CoverageStatus
from wildfire_data.firms_collection import (
    FirmsCollectionError,
    archive_firms_csv_response,
    record_firms_collection_failure,
    redact_firms_source_url,
)
from wildfire_data.storage_budget import load_storage_budget


class FirmsCollectionTests(unittest.TestCase):
    def _policy(self, path, *, total=10_000_000, firms_cap=10_000_000):
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "whole_data_cap_bytes": total,
                    "whole_data_cap_label": "test",
                    "scope": "test data",
                    "categories": [
                        {
                            "key": "firms_and_detection_evidence",
                            "cap_bytes": firms_cap,
                            "priority_score": 100,
                            "pinned": True,
                            "retention": "FIRMS",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return load_storage_budget(path)

    def _collect(self, directory, payload, **changes):
        arguments = {
            "payload": payload,
            "product": "VIIRS_SNPP_NRT",
            "coverage_date": date(2026, 7, 26),
            "region": "United States and Canada",
            "source_url": "https://example.test/firms?MAP_KEY=never-persist",
            "response_status_code": 200,
            "retrieved_at": datetime(2026, 7, 26, 1, tzinfo=timezone.utc),
        }
        arguments.update(changes)
        return archive_firms_csv_response(directory, **arguments)

    def test_archives_unfiltered_rows_and_records_threshold_as_derived_metadata(self):
        payload = (
            b"latitude,longitude,bright_ti4,acq_date,acq_time,frp,future_field\n"
            b"54.1,-106.2,304.9,2026-07-26,0040,5.2,retained\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = self._collect(directory, payload)
            artifact = result.normalized_artifacts[0]
            with gzip.open(artifact.artifact_path, "rt", encoding="utf-8") as file:
                normalized = json.loads(file.readline())
            raw_manifest_text = result.raw_artifact.manifest_path.read_text(encoding="utf-8")

        self.assertEqual(result.coverage.status, CoverageStatus.COMPLETE)
        self.assertEqual(result.record_count, 1)
        self.assertEqual(normalized["raw_source_fields"]["future_field"], "retained")
        self.assertEqual(normalized["raw_source_fields"]["frp"], "5.2")
        self.assertFalse(normalized["derived"]["ti4_threshold"]["passes"])
        self.assertEqual(normalized["provenance"]["raw_artifact_id"], result.raw_artifact.raw_artifact_id)
        self.assertNotIn("never-persist", raw_manifest_text)

    def test_records_an_empty_successful_response_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._collect(
                directory,
                b"latitude,longitude,bright_ti4,acq_date,acq_time\n",
            )

        self.assertEqual(result.coverage.status, CoverageStatus.EMPTY_CONFIRMED)
        self.assertEqual(result.normalized_artifacts, ())

    def test_records_non_success_responses_without_attempting_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._collect(directory, b"service unavailable", response_status_code=503)

        self.assertEqual(result.coverage.status, CoverageStatus.FAILED)
        self.assertEqual(result.record_count, 0)

    def test_records_partial_without_writing_firms_bytes_when_the_quota_would_be_exceeded(self):
        payload = b"latitude,longitude,bright_ti4,acq_date,acq_time\n54.1,-106.2,320,2026-07-26,0040\n"
        with tempfile.TemporaryDirectory() as directory:
            policy = self._policy(Path(directory) / "budget.json", total=100, firms_cap=100)
            result = self._collect(directory, payload, storage_budget=policy)

        self.assertIsNone(result.raw_artifact)
        self.assertEqual(result.coverage.status, CoverageStatus.PARTIAL)
        self.assertEqual(result.record_count, 0)

    def test_keeps_raw_evidence_and_marks_malformed_successful_data_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FirmsCollectionError):
                self._collect(directory, b"not,a,firms,csv\n1,2,3,4\n")
            latest = list(CoverageLedger(directory).latest_by_coverage().values())

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].status, CoverageStatus.FAILED)

    def test_records_request_failures_that_do_not_have_a_response_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            coverage = record_firms_collection_failure(
                directory,
                product="VIIRS_SNPP_NRT",
                coverage_date=date(2026, 7, 26),
                region="United States and Canada",
                error="connection timed out",
                retrieved_at=datetime(2026, 7, 26, 1, tzinfo=timezone.utc),
            )

        self.assertEqual(coverage.status, CoverageStatus.FAILED)
        self.assertEqual(
            coverage.expected_coverage_id,
            "firms:VIIRS_SNPP_NRT:United States and Canada:2026-07-26",
        )

    def test_redacts_the_path_embedded_firms_key_before_manifest_persistence(self):
        redacted = redact_firms_source_url(
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/private-key/"
            "VIIRS_SNPP_NRT/-179,24,-52,84/1/2026-07-26"
        )

        self.assertNotIn("private-key", redacted)
        self.assertIn("/<redacted>/VIIRS_SNPP_NRT/", redacted)
