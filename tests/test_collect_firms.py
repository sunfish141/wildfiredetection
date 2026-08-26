import json
import tempfile
import unittest
from datetime import date, datetime, timezone

import requests

from wildfire_data.collect_firms import collect_firms_range, firms_area_url
from wildfire_data.data_archive import CoverageLedger, CoverageStatus


class _FakeResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": "text/csv"}


class _FakeSession:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.urls = []

    def get(self, url, *, timeout):
        self.urls.append(url)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class CollectFirmsTests(unittest.TestCase):
    def test_builds_a_daily_firms_url(self):
        url = firms_area_url(
            api_key="key",
            product="VIIRS_SNPP_NRT",
            bbox="-179,24,-52,84",
            request_date=date(2026, 7, 26),
        )

        self.assertTrue(url.endswith("/key/VIIRS_SNPP_NRT/-179,24,-52,84/1/2026-07-26"))

    def test_collects_each_day_and_preserves_a_coverage_retry_for_a_request_error(self):
        payload = (
            b"latitude,longitude,bright_ti4,acq_date,acq_time\n"
            b"54.1,-106.2,306,2026-07-26,0040\n"
        )
        session = _FakeSession([_FakeResponse(content=payload), requests.ConnectionError("offline")])
        with tempfile.TemporaryDirectory() as directory:
            result = collect_firms_range(
                directory,
                api_key="do-not-persist",
                start_date=date(2026, 7, 26),
                end_date=date(2026, 7, 27),
                session=session,
                retrieved_at=datetime(2026, 7, 26, 1, tzinfo=timezone.utc),
            )
            entries = CoverageLedger(directory).entries()
            manifests = list((__import__("pathlib").Path(directory) / "manifests" / "raw").rglob("*.json"))
            manifest_text = "\n".join(path.read_text(encoding="utf-8") for path in manifests)

        self.assertEqual(len(result.responses), 1)
        self.assertEqual(len(result.request_failures), 1)
        self.assertEqual(result.failed_count, 1)
        self.assertCountEqual(
            [entry.status for entry in entries], [CoverageStatus.COMPLETE, CoverageStatus.FAILED]
        )
        self.assertNotIn("do-not-persist", manifest_text)
        self.assertEqual(len(session.urls), 2)

    def test_skips_terminal_product_day_unless_refresh_is_requested(self):
        payload = (
            b"latitude,longitude,bright_ti4,acq_date,acq_time\n"
            b"54.1,-106.2,306,2026-07-26,0040\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            first_session = _FakeSession([_FakeResponse(content=payload)])
            collect_firms_range(
                directory,
                api_key="do-not-persist",
                start_date=date(2026, 7, 26),
                end_date=date(2026, 7, 26),
                session=first_session,
                retrieved_at=datetime(2026, 7, 26, 1, tzinfo=timezone.utc),
            )
            resumed_session = _FakeSession([])
            resumed = collect_firms_range(
                directory,
                api_key="do-not-persist",
                start_date=date(2026, 7, 26),
                end_date=date(2026, 7, 26),
                session=resumed_session,
                retrieved_at=datetime(2026, 7, 26, 2, tzinfo=timezone.utc),
            )
            refreshed_session = _FakeSession([_FakeResponse(content=payload)])
            refreshed = collect_firms_range(
                directory,
                api_key="do-not-persist",
                start_date=date(2026, 7, 26),
                end_date=date(2026, 7, 26),
                session=refreshed_session,
                retrieved_at=datetime(2026, 7, 26, 3, tzinfo=timezone.utc),
                refresh=True,
            )

        self.assertEqual(resumed.skipped_terminal_count, 1)
        self.assertEqual(resumed.responses, ())
        self.assertEqual(resumed_session.urls, [])
        self.assertEqual(len(refreshed.responses), 1)
        self.assertEqual(len(refreshed_session.urls), 1)
