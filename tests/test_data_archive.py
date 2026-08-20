import gzip
import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from wildfire_data.data_archive import (
    ArchiveIntegrityError,
    CoverageLedger,
    CoverageStatus,
    write_atomic_json,
    write_raw_artifact,
    write_raw_artifact_from_file,
)


class DataArchiveTests(unittest.TestCase):
    def test_raw_artifact_preserves_exact_bytes_and_redacts_provenance(self):
        payload = b"FIRMS response\n\x00with exact bytes\n"
        retrieved_at = datetime(2026, 8, 16, 12, 30, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as directory:
            artifact = write_raw_artifact(
                directory,
                source="NASA FIRMS / VIIRS",
                payload=payload,
                retrieved_at=retrieved_at,
                capture_id="firms-20260816-001",
                provenance={
                    "source_url": (
                        "https://example.test/firms?api_key=do-not-store&"
                        "MAP_KEY=also-do-not-store&area=canada"
                    ),
                    "status_code": 200,
                    "request_parameters": {"days": 4, "token": "also-do-not-store"},
                    "headers": {"Authorization": "Bearer private-token"},
                },
            )

            with gzip.open(artifact.artifact_path, "rb") as raw_file:
                self.assertEqual(raw_file.read(), payload)
            manifest_text = artifact.manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)

        expected_digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(artifact.raw_artifact_id, expected_digest)
        self.assertEqual(artifact["raw_artifact_id"], expected_digest)
        self.assertEqual(artifact["artifact_path"], str(artifact.artifact_path))
        self.assertEqual(artifact.content_sha256, expected_digest)
        self.assertEqual(manifest["artifact"]["content_sha256"], expected_digest)
        self.assertEqual(manifest["artifact"]["content_bytes"], len(payload))
        self.assertEqual(manifest["provenance"]["status_code"], 200)
        self.assertNotIn("do-not-store", manifest_text)
        self.assertNotIn("also-do-not-store", manifest_text)
        self.assertNotIn("private-token", manifest_text)
        self.assertEqual(manifest["provenance"]["request_parameters"]["token"], "<redacted>")

    def test_repeated_payload_reuses_immutable_raw_file_and_keeps_capture_history(self):
        payload = b"same response"
        retrieved_at = datetime(2026, 8, 16, 13, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as directory:
            first = write_raw_artifact(
                directory,
                source="FIRMS",
                payload=payload,
                retrieved_at=retrieved_at,
                capture_id="first-capture",
            )
            second = write_raw_artifact(
                directory,
                source="FIRMS",
                payload=payload,
                retrieved_at=retrieved_at,
                capture_id="second-capture",
            )

            self.assertTrue(first.created)
            self.assertFalse(second.created)
            self.assertEqual(first.artifact_path, second.artifact_path)
            self.assertNotEqual(first.manifest_path, second.manifest_path)
            with gzip.open(second.artifact_path, "rb") as raw_file:
                self.assertEqual(raw_file.read(), payload)

    def test_streams_a_large_source_file_into_the_same_immutable_raw_contract(self):
        payload = (b"NALCMS source bytes\n" * 100_000) + b"end"
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.zip"
            source_path.write_bytes(payload)
            artifact = write_raw_artifact_from_file(
                directory,
                source="CEC NALCMS land cover",
                source_path=source_path,
                retrieved_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
                provenance={"source_url": "https://example.test/file.zip?token=private"},
            )
            with gzip.open(artifact.artifact_path, "rb") as source:
                restored = source.read()
            manifest = artifact.manifest_path.read_text(encoding="utf-8")

        self.assertEqual(restored, payload)
        self.assertEqual(artifact.byte_count, len(payload))
        self.assertEqual(artifact.content_sha256, hashlib.sha256(payload).hexdigest())
        self.assertNotIn("private", manifest)

    def test_detects_an_existing_artifact_that_no_longer_matches_its_digest(self):
        payload = b"original"
        retrieved_at = datetime(2026, 8, 16, 14, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as directory:
            first = write_raw_artifact(
                directory,
                source="FIRMS",
                payload=payload,
                retrieved_at=retrieved_at,
                capture_id="original-capture",
            )
            first.artifact_path.write_bytes(gzip.compress(b"corrupted", mtime=0))

            with self.assertRaises(ArchiveIntegrityError):
                write_raw_artifact(
                    directory,
                    source="FIRMS",
                    payload=payload,
                    retrieved_at=retrieved_at,
                    capture_id="retry-capture",
                )

    def test_atomic_json_replaces_a_complete_document_without_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manifests" / "current.json"
            write_atomic_json(target, {"state": "first"})
            write_atomic_json(target, {"state": "second", "count": 2})

            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"count": 2, "state": "second"},
            )
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_coverage_ledger_records_each_explicit_outcome_and_latest_status(self):
        statuses = [
            CoverageStatus.COMPLETE,
            CoverageStatus.EMPTY_CONFIRMED,
            CoverageStatus.PARTIAL,
            CoverageStatus.FAILED,
        ]
        recorded_at = datetime(2026, 8, 16, 15, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as directory:
            ledger = CoverageLedger(directory)
            records = []
            for offset, status in enumerate(statuses):
                records.append(
                    ledger.record(
                        source="NASA FIRMS",
                        product="VIIRS-NRT",
                        coverage_start=date(2026, 8, 12),
                        coverage_end=date(2026, 8, 15),
                        region="Canada",
                        tile="northwest",
                        expected_coverage_id="firms-ca-nw-2026-08-12-to-15",
                        status=status,
                        detail={"batch": offset, "api_key": "do-not-store"},
                        message="collection attempt",
                        error="Bearer private-token" if status is CoverageStatus.FAILED else None,
                        recorded_at=recorded_at + timedelta(minutes=offset),
                    )
                )

            persisted = ledger.entries()
            latest = ledger.latest_by_coverage()
            failed_text = records[-1].path.read_text(encoding="utf-8")

        self.assertEqual([record.status for record in persisted], statuses)
        self.assertEqual({record.path for record in persisted}, {record.path for record in records})
        self.assertEqual(records[0].expected_coverage_id, "firms-ca-nw-2026-08-12-to-15")
        self.assertEqual(latest[records[0].coverage_key].status, CoverageStatus.FAILED)
        self.assertNotIn("do-not-store", failed_text)
        self.assertNotIn("private-token", failed_text)

    def test_coverage_ledger_rejects_unknown_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = CoverageLedger(directory)

            with self.assertRaises(ValueError):
                ledger.record(
                    source="FIRMS",
                    product="VIIRS",
                    coverage_start="2026-08-15",
                    coverage_end="2026-08-15",
                    region="Canada",
                    status="missing",
                )

    def test_coverage_ledger_uses_append_order_when_attempts_share_a_timestamp(self):
        recorded_at = datetime(2026, 8, 16, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            ledger = CoverageLedger(directory)
            first = ledger.record(
                source="NIFC WFIGS",
                product="wfigs_current_perimeters",
                coverage_start="2026-05-31",
                coverage_end="2026-08-10",
                region="United States",
                expected_coverage_id="wfigs-range",
                status=CoverageStatus.PARTIAL,
                recorded_at=recorded_at,
            )
            second = ledger.record(
                source="NIFC WFIGS",
                product="wfigs_current_perimeters",
                coverage_start="2026-05-31",
                coverage_end="2026-08-10",
                region="United States",
                expected_coverage_id="wfigs-range",
                status=CoverageStatus.COMPLETE,
                recorded_at=recorded_at,
            )
            entries = ledger.entries()
            latest = ledger.latest_by_coverage()

        self.assertEqual([entry.entry_id for entry in entries], [first.entry_id, second.entry_id])
        self.assertEqual(latest[first.coverage_key].status, CoverageStatus.COMPLETE)
