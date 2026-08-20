import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone

import requests

from wildfire_data.data_archive import CoverageLedger, CoverageStatus
from wildfire_data.viirs_l2_observability import (
    CMR_GRANULES_URL,
    ViirsL2InventoryError,
    cmr_inventory_parameters,
    collect_viirs_l2_range,
    parse_cmr_inventory_payload,
    product_for_platform,
)


NETCDF4_HEADER = b"\x89HDF\r\n\x1a\nfull-level-2-content"


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code=200,
        content=b"",
        url="https://example.test/response",
        headers=None,
    ):
        self.status_code = status_code
        self.content = content
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}


class _FakeSession:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _inventory_payload(*, hits=1, granule_id="VNP14IMG.A2026151.0006.002.2026151103258"):
    return json.dumps(
        {
            "hits": hits,
            "items": [
                {
                    "meta": {"collection-concept-id": "C2734202914-LPCLOUD"},
                    "umm": {
                        "GranuleUR": granule_id,
                        "CollectionReference": {"ShortName": "VNP14IMG", "Version": "002"},
                        "TemporalExtent": {
                            "RangeDateTime": {
                                "BeginningDateTime": "2026-05-31T00:06:00.000Z",
                                "EndingDateTime": "2026-05-31T00:12:00.000Z",
                            }
                        },
                        "SpatialExtent": {
                            "HorizontalSpatialDomain": {
                                "Geometry": {"BoundingRectangles": [{"WestBoundingCoordinate": -174.5}]}
                            }
                        },
                        "RelatedUrls": [
                            {
                                "URL": (
                                    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
                                    f"VNP14IMG.002/{granule_id}/{granule_id}.nc"
                                ),
                                "Type": "GET DATA",
                            }
                        ],
                        "DataGranule": {
                            "ArchiveAndDistributionInformation": [
                                {"Size": 1.5, "SizeUnit": "MB"}
                            ]
                        },
                    },
                }
            ],
        }
    ).encode("utf-8")


def _empty_inventory_payload():
    return b'{"hits": 0, "items": []}'


class ViirsL2ObservabilityTests(unittest.TestCase):
    def test_builds_a_cmr_query_and_parses_the_embedded_observability_asset(self):
        product = product_for_platform("SNPP")

        parameters = cmr_inventory_parameters(
            product,
            coverage_date=date(2026, 5, 31),
            bbox="-179,24,-52,84",
        )
        hits, granules = parse_cmr_inventory_payload(_inventory_payload(), product=product)

        self.assertEqual(parameters["short_name"], "VNP14IMG")
        self.assertEqual(parameters["temporal"], "2026-05-31T00:00:00Z,2026-06-01T00:00:00Z")
        self.assertEqual(hits, 1)
        self.assertEqual(granules[0].native_filename, "VNP14IMG.A2026151.0006.002.2026151103258.nc")
        self.assertEqual(granules[0].reported_size_bytes, 1_500_000)

    def test_rejects_an_inventory_item_from_an_unexpected_product(self):
        wrong_product = json.loads(_inventory_payload())
        wrong_product["items"][0]["umm"]["CollectionReference"]["ShortName"] = "VJ114IMG"

        with self.assertRaises(ViirsL2InventoryError):
            parse_cmr_inventory_payload(json.dumps(wrong_product).encode("utf-8"), product=product_for_platform("snpp"))

    def test_archives_inventory_and_whole_netcdf_granule_without_persisting_the_token(self):
        session = _FakeSession(
            [
                _FakeResponse(content=_inventory_payload(), url=f"{CMR_GRANULES_URL}?page_num=1"),
                _FakeResponse(
                    content=NETCDF4_HEADER,
                    url="https://data.lpdaac.earthdatacloud.nasa.gov/download?token=private",
                    headers={"Content-Type": "application/x-netcdf", "Content-Length": str(len(NETCDF4_HEADER))},
                ),
            ]
        )
        retrieved_at = datetime(2026, 8, 16, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            result = collect_viirs_l2_range(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 5, 31),
                platforms=["snpp"],
                earthdata_token="do-not-persist",
                session=session,
                retrieved_at=retrieved_at,
            )
            manifests = list((__import__("pathlib").Path(directory) / "manifests" / "raw").rglob("*.json"))
            manifest_text = "\n".join(path.read_text(encoding="utf-8") for path in manifests)
            entries = CoverageLedger(directory).entries()

        self.assertEqual(result.inventory_response_count, 1)
        self.assertEqual(result.discovered_granule_count, 1)
        self.assertEqual(result.archived_granule_count, 1)
        self.assertEqual(result.incomplete_window_count, 0)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0][0], CMR_GRANULES_URL)
        self.assertEqual(session.calls[1][1]["headers"]["Authorization"], "Bearer do-not-persist")
        self.assertNotIn("do-not-persist", manifest_text)
        self.assertNotIn("private", manifest_text)
        self.assertIn("fire_mask_and_algorithm_qa", manifest_text)
        self.assertIn("VNP03IMG", manifest_text)
        self.assertCountEqual(
            [entry.status for entry in entries], [CoverageStatus.COMPLETE, CoverageStatus.COMPLETE]
        )

    def test_retries_only_a_failed_granule_and_completes_its_parent_window(self):
        retrieved_at = datetime(2026, 8, 16, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            first_session = _FakeSession(
                [
                    _FakeResponse(content=_inventory_payload()),
                    _FakeResponse(status_code=503, content=b"temporarily unavailable"),
                ]
            )
            first = collect_viirs_l2_range(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 5, 31),
                platforms=["snpp"],
                earthdata_token="test-token",
                session=first_session,
                retrieved_at=retrieved_at,
            )
            retry_session = _FakeSession(
                [
                    _FakeResponse(content=_inventory_payload()),
                    _FakeResponse(
                        content=NETCDF4_HEADER,
                        headers={"Content-Length": str(len(NETCDF4_HEADER))},
                    ),
                ]
            )
            second = collect_viirs_l2_range(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 5, 31),
                platforms=["snpp"],
                earthdata_token="test-token",
                session=retry_session,
                retrieved_at=retrieved_at + timedelta(minutes=1),
            )
            entries = CoverageLedger(directory).entries()

        self.assertEqual(first.incomplete_window_count, 1)
        self.assertEqual(second.archived_granule_count, 1)
        self.assertEqual(second.incomplete_window_count, 0)
        self.assertEqual(len(retry_session.calls), 2)
        self.assertEqual([entry.status for entry in entries].count(CoverageStatus.FAILED), 1)
        self.assertEqual(entries[-1].status, CoverageStatus.COMPLETE)

    def test_dry_run_archives_discovery_but_leaves_downloads_retryable(self):
        session = _FakeSession([_FakeResponse(content=_inventory_payload())])
        with tempfile.TemporaryDirectory() as directory:
            result = collect_viirs_l2_range(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 5, 31),
                platforms=["snpp"],
                dry_run=True,
                session=session,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )

        self.assertEqual(result.archived_granule_count, 0)
        self.assertEqual(result.incomplete_window_count, 1)
        self.assertEqual(len(session.calls), 1)

    def test_empty_historical_inventory_is_confirmed_but_recent_inventory_is_retryable(self):
        retrieved_at = datetime(2026, 8, 16, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            historical = collect_viirs_l2_range(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 5, 31),
                platforms=["snpp"],
                dry_run=True,
                session=_FakeSession([_FakeResponse(content=_empty_inventory_payload())]),
                retrieved_at=retrieved_at,
            )
            recent = collect_viirs_l2_range(
                directory,
                start_date=date(2026, 8, 16),
                end_date=date(2026, 8, 16),
                platforms=["snpp"],
                dry_run=True,
                session=_FakeSession([_FakeResponse(content=_empty_inventory_payload())]),
                retrieved_at=retrieved_at,
            )

        self.assertEqual(historical.days[0].coverage.status, CoverageStatus.EMPTY_CONFIRMED)
        self.assertEqual(recent.days[0].coverage.status, CoverageStatus.PARTIAL)

    def test_invalid_download_payload_is_retained_but_marked_failed(self):
        session = _FakeSession(
            [
                _FakeResponse(content=_inventory_payload()),
                _FakeResponse(content=b"<html>Earthdata login page</html>"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = collect_viirs_l2_range(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 5, 31),
                platforms=["snpp"],
                earthdata_token="test-token",
                session=session,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )
            receipts = result.days[0].granule_receipts

        self.assertEqual(receipts[0].coverage.status, CoverageStatus.FAILED)
        self.assertIsNotNone(receipts[0].raw_artifact)
        self.assertEqual(result.incomplete_window_count, 1)

    def test_records_a_request_failure_without_suppressing_other_collection_state(self):
        session = _FakeSession([requests.ConnectionError("offline")])
        with tempfile.TemporaryDirectory() as directory:
            result = collect_viirs_l2_range(
                directory,
                start_date=date(2026, 5, 31),
                end_date=date(2026, 5, 31),
                platforms=["snpp"],
                dry_run=True,
                session=session,
                retrieved_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )

        self.assertEqual(result.days[0].coverage.status, CoverageStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
