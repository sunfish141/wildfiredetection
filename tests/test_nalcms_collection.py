import gzip
import io
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from wildfire_data.data_archive import CoverageLedger, CoverageStatus
from wildfire_data.nalcms_collection import NalcmsRelease, collect_nalcms_land_cover
from wildfire_data.storage_budget import load_storage_budget


class _FakeResponse:
    def __init__(self, content=b"", *, status_code=200, headers=None, url="https://example.test/source.zip"):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/zip", "Content-Length": str(len(content))}
        self.url = url

    def iter_content(self, chunk_size):
        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]


class _FakeSession:
    def __init__(self, head, get):
        self.head_response = head
        self.get_response = get
        self.head_calls = []
        self.get_calls = []

    def head(self, url, **kwargs):
        self.head_calls.append((url, kwargs))
        return self.head_response

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.get_response


def _zip_payload():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("land_cover.tif", b"fake GeoTIFF source bytes")
    return output.getvalue()


def _policy(path, *, total=100_000_000, static_cap=100_000_000):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "whole_data_cap_bytes": total,
                "whole_data_cap_label": "test",
                "scope": "test data",
                "categories": [
                    {
                        "key": "static_cell_features",
                        "cap_bytes": static_cap,
                        "priority_score": 70,
                        "pinned": True,
                        "retention": "static",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return load_storage_budget(path)


class NalcmsCollectionTests(unittest.TestCase):
    def _release(self):
        return NalcmsRelease(
            key="canada",
            country="Canada",
            source_url="https://example.test/nalcms.zip?token=private",
            component_years={"Canada": 2020},
        )

    def test_archives_a_streamed_zip_and_normalizes_static_source_provenance(self):
        payload = _zip_payload()
        session = _FakeSession(
            _FakeResponse(headers={"Content-Length": str(len(payload)), "ETag": "abc"}),
            _FakeResponse(payload, url="https://example.test/nalcms.zip?token=private"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            result = collect_nalcms_land_cover(
                root,
                storage_budget=_policy(Path(directory) / "budget.json"),
                releases=(self._release(),),
                session=session,
                staging_directory=Path(directory) / "staging-outside-data",
                retrieved_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
            )
            release_result = result.releases[0]
            with gzip.open(release_result.raw_artifact.artifact_path, "rb") as source:
                restored = source.read()
            normalized_path = release_result.normalized_artifact.artifact_path
            with gzip.open(normalized_path, "rt", encoding="utf-8") as source:
                normalized = json.loads(source.readline())
            raw_manifest = release_result.raw_artifact.manifest_path.read_text(encoding="utf-8")

        self.assertEqual(release_result.coverage.status, CoverageStatus.COMPLETE)
        self.assertEqual(restored, payload)
        self.assertEqual(normalized["record_type"], "static_land_cover_source_release")
        self.assertEqual(normalized["component_years"], {"Canada": 2020})
        self.assertEqual(normalized["retention_priority_score"], 70)
        self.assertNotIn("private", raw_manifest)
        self.assertEqual(len(session.get_calls), 1)
        self.assertEqual(session.head_calls[0][1]["headers"], {"Accept-Encoding": "identity"})
        self.assertEqual(session.get_calls[0][1]["headers"], {"Accept-Encoding": "identity"})

    def test_skips_complete_release_without_another_metadata_or_download_request(self):
        payload = _zip_payload()
        first_session = _FakeSession(
            _FakeResponse(headers={"Content-Length": str(len(payload))}), _FakeResponse(payload)
        )
        repeat_session = _FakeSession(_FakeResponse(), _FakeResponse())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            policy = _policy(Path(directory) / "budget.json")
            collect_nalcms_land_cover(
                root,
                storage_budget=policy,
                releases=(self._release(),),
                session=first_session,
                retrieved_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
            )
            result = collect_nalcms_land_cover(
                root,
                storage_budget=policy,
                releases=(self._release(),),
                session=repeat_session,
                retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )

        self.assertTrue(result.releases[0].skipped_terminal_coverage)
        self.assertEqual(repeat_session.head_calls, [])
        self.assertEqual(repeat_session.get_calls, [])

    def test_records_partial_before_download_when_the_static_budget_cannot_admit_source(self):
        payload = _zip_payload()
        session = _FakeSession(
            _FakeResponse(headers={"Content-Length": str(len(payload))}), _FakeResponse(payload)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            result = collect_nalcms_land_cover(
                root,
                storage_budget=_policy(Path(directory) / "budget.json", total=1_000, static_cap=1_000),
                releases=(self._release(),),
                session=session,
                retrieved_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
            )
            entries = CoverageLedger(root).entries()

        self.assertEqual(result.releases[0].coverage.status, CoverageStatus.PARTIAL)
        self.assertEqual(session.get_calls, [])
        self.assertEqual(entries[-1].status, CoverageStatus.PARTIAL)

    def test_records_raw_evidence_but_fails_a_same_length_non_zip_response(self):
        payload = b"not a zip"
        session = _FakeSession(
            _FakeResponse(headers={"Content-Length": str(len(payload))}), _FakeResponse(payload)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            result = collect_nalcms_land_cover(
                root,
                storage_budget=_policy(Path(directory) / "budget.json"),
                releases=(self._release(),),
                session=session,
                retrieved_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
            )

        self.assertEqual(result.releases[0].coverage.status, CoverageStatus.FAILED)
        self.assertIsNotNone(result.releases[0].raw_artifact)

    def test_rejects_an_oversized_download_before_it_can_enter_the_archive(self):
        payload = _zip_payload()
        session = _FakeSession(
            _FakeResponse(headers={"Content-Length": str(len(payload) - 1)}), _FakeResponse(payload)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            result = collect_nalcms_land_cover(
                root,
                storage_budget=_policy(Path(directory) / "budget.json"),
                releases=(self._release(),),
                session=session,
                staging_directory=Path(directory) / "staging-outside-data",
                retrieved_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
            )

            raw_archives = list((root / "raw" / "cec-nalcms-land-cover").glob("*.gz"))

        self.assertEqual(result.releases[0].coverage.status, CoverageStatus.FAILED)
        self.assertIsNone(result.releases[0].raw_artifact)
        self.assertEqual(raw_archives, [])

    def test_rejects_staging_inside_the_governed_data_root(self):
        payload = _zip_payload()
        session = _FakeSession(
            _FakeResponse(headers={"Content-Length": str(len(payload))}), _FakeResponse(payload)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            with self.assertRaisesRegex(ValueError, "staging_directory must be outside data-root"):
                collect_nalcms_land_cover(
                    root,
                    storage_budget=_policy(Path(directory) / "budget.json"),
                    releases=(self._release(),),
                    session=session,
                    staging_directory=root / "staging",
                    retrieved_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
                )

        self.assertEqual(session.head_calls, [])


if __name__ == "__main__":
    unittest.main()
