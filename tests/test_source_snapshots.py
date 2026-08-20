import json
import tempfile
import unittest
from datetime import datetime, timezone

import requests

from wildfire_data.collection_catalog import targets_for_entity
from wildfire_data.data_archive import CoverageLedger, CoverageStatus
from wildfire_data.source_snapshots import (
    SourceSnapshotRequestError,
    archive_source_snapshot,
    fetch_http_source_snapshot,
)


class _FakeResponse:
    def __init__(self, status_code=200, content=b"{}"):
        self.status_code = status_code
        self.content = content
        self.url = "https://example.test/snapshot?token=private"
        self.headers = {"Content-Type": "application/json"}


class _FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def get(self, _url, *, params, headers, timeout):
        if self.error:
            raise self.error
        return self.response


class SourceSnapshotTests(unittest.TestCase):
    def test_archives_a_perimeter_snapshot_and_its_coverage_outcome(self):
        target = targets_for_entity("operational_perimeter")[0]
        with tempfile.TemporaryDirectory() as directory:
            receipt = archive_source_snapshot(
                directory,
                target=target,
                payload=b'{"features": []}',
                coverage_start="2026-07-26T00:00:00Z",
                coverage_end="2026-07-26T00:15:00Z",
                source_url="https://example.test/perimeters?token=secret",
                status=CoverageStatus.EMPTY_CONFIRMED,
                response_status_code=200,
                expected_coverage_id="wfigs-2026-07-26T00:00Z",
                retrieved_at=datetime(2026, 7, 26, 0, 16, tzinfo=timezone.utc),
            )
            manifest = json.loads(receipt.raw_artifact.manifest_path.read_text(encoding="utf-8"))
            entries = CoverageLedger(directory).entries()

        self.assertEqual(receipt.coverage.status, CoverageStatus.EMPTY_CONFIRMED)
        self.assertEqual(entries, (receipt.coverage,))
        self.assertNotIn("secret", json.dumps(manifest))
        self.assertEqual(manifest["provenance"]["response_status_code"], 200)

    def test_fetches_and_archives_a_successful_http_snapshot(self):
        target = targets_for_entity("operational_perimeter")[0]
        with tempfile.TemporaryDirectory() as directory:
            receipt = fetch_http_source_snapshot(
                directory,
                target=target,
                source_url="https://example.test/snapshot",
                coverage_start="2026-07-26T00:00:00Z",
                coverage_end="2026-07-26T00:15:00Z",
                session=_FakeSession(_FakeResponse(content=b'{"features":[]}')),
                retrieved_at=datetime(2026, 7, 26, 0, 16, tzinfo=timezone.utc),
            )

        self.assertEqual(receipt.coverage.status, CoverageStatus.COMPLETE)

    def test_records_request_errors_without_a_response(self):
        target = targets_for_entity("operational_perimeter")[0]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SourceSnapshotRequestError) as caught:
                fetch_http_source_snapshot(
                    directory,
                    target=target,
                    source_url="https://example.test/snapshot",
                    coverage_start="2026-07-26T00:00:00Z",
                    coverage_end="2026-07-26T00:15:00Z",
                    session=_FakeSession(error=requests.ConnectionError("no route")),
                    retrieved_at=datetime(2026, 7, 26, 0, 16, tzinfo=timezone.utc),
                )

        self.assertEqual(caught.exception.coverage.status, CoverageStatus.FAILED)
