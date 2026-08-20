"""Command-line and programmatic collection of durable daily FIRMS archives."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .data_archive import CoverageRecord, CoverageStatus
from .firms_collection import (
    FirmsCollectionError,
    FirmsCollectionResult,
    archive_firms_csv_response,
    record_firms_collection_failure,
)
from .storage_budget import DEFAULT_POLICY_PATH, StorageBudgetPolicy, load_storage_budget


DEFAULT_BBOX = "-179,24,-52,84"
DEFAULT_REGION = "United States and Canada"
DEFAULT_PRODUCT = "VIIRS_SNPP_NRT"


@dataclass(frozen=True)
class FirmsRangeCollection:
    """Every durable outcome from a FIRMS range collection attempt."""

    responses: tuple[FirmsCollectionResult, ...]
    request_failures: tuple[CoverageRecord, ...]
    normalization_failures: tuple[str, ...]

    @property
    def failed_count(self) -> int:
        terminal_statuses = {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED}
        return (
            len(self.request_failures)
            + len(self.normalization_failures)
            + sum(response.coverage.status not in terminal_statuses for response in self.responses)
        )


def firms_area_url(*, api_key: str, product: str, bbox: str, request_date: date) -> str:
    """Build the FIRMS daily-area URL; callers must not log this unredacted URL."""
    if not api_key.strip() or not product.strip() or not bbox.strip():
        raise ValueError("api_key, product, and bbox must be non-empty")
    return (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{api_key}/{product}/{bbox}/1/{request_date.isoformat()}"
    )


def collect_firms_range(
    archive_root: str,
    *,
    api_key: str,
    start_date: date,
    end_date: date,
    products: Iterable[str] = (DEFAULT_PRODUCT,),
    bbox: str = DEFAULT_BBOX,
    region: str = DEFAULT_REGION,
    minimum_bright_ti4: float | None = 305.0,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (10, 120),
    retrieved_at: datetime | None = None,
    storage_budget: StorageBudgetPolicy | None = None,
) -> FirmsRangeCollection:
    """Collect every requested day/product while recording failures for retry.

    The function deliberately continues after one request or normalization
    failure.  The coverage ledger tells a scheduler exactly which daily
    product windows need another attempt.
    """
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    product_values = tuple(product.strip() for product in products if product.strip())
    if not product_values:
        raise ValueError("products must contain at least one non-empty product")
    if not region.strip():
        raise ValueError("region must be non-empty")

    owns_session = session is None
    active_session = session or _retrying_session()
    responses = []
    request_failures = []
    normalization_failures = []
    try:
        request_day = start_date
        while request_day <= end_date:
            for product in product_values:
                url = firms_area_url(
                    api_key=api_key,
                    product=product,
                    bbox=bbox,
                    request_date=request_day,
                )
                attempted_at = _retrieved_at(retrieved_at)
                try:
                    response = active_session.get(url, timeout=timeout)
                except requests.RequestException as exc:
                    request_failures.append(
                        record_firms_collection_failure(
                            archive_root,
                            product=product,
                            coverage_date=request_day,
                            region=region,
                            error=str(exc),
                            retrieved_at=attempted_at,
                        )
                    )
                    continue
                try:
                    responses.append(
                        archive_firms_csv_response(
                            archive_root,
                            payload=response.content,
                            product=product,
                            coverage_date=request_day,
                            region=region,
                            source_url=url,
                            response_status_code=response.status_code,
                            response_headers=dict(response.headers),
                            request_parameters={"bbox": bbox, "days": 1},
                            retrieved_at=attempted_at,
                            minimum_bright_ti4=minimum_bright_ti4,
                            storage_budget=storage_budget,
                        )
                    )
                except FirmsCollectionError as exc:
                    normalization_failures.append(str(exc))
            request_day += timedelta(days=1)
    finally:
        if owns_session:
            active_session.close()
    return FirmsRangeCollection(
        responses=tuple(responses),
        request_failures=tuple(request_failures),
        normalization_failures=tuple(normalization_failures),
    )


def _retrying_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _retrieved_at(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return resolved.astimezone(timezone.utc)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: list[str] | None = None) -> int:
    """Run a range collection without writing API credentials to disk."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path("config/.env"))
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="inclusive UTC date")
    parser.add_argument("--end", required=True, type=_parse_date, help="inclusive UTC date")
    parser.add_argument("--archive-root", default="data")
    parser.add_argument("--bbox", default=DEFAULT_BBOX)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--product", action="append", dest="products")
    parser.add_argument("--minimum-bright-ti4", type=float, default=305.0)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    arguments = parser.parse_args(argv)
    api_key = (os.getenv("NASA_FIRMS_API_KEY") or os.getenv("MAP_KEY") or "").strip()
    if not api_key:
        parser.error("Set NASA_FIRMS_API_KEY or MAP_KEY before collecting FIRMS data.")

    result = collect_firms_range(
        arguments.archive_root,
        api_key=api_key,
        start_date=arguments.start,
        end_date=arguments.end,
        products=arguments.products or [DEFAULT_PRODUCT],
        bbox=arguments.bbox,
        region=arguments.region,
        minimum_bright_ti4=arguments.minimum_bright_ti4,
        storage_budget=load_storage_budget(arguments.policy),
    )
    completed_rows = sum(response.record_count for response in result.responses)
    print(
        f"Recorded {len(result.responses):,} HTTP responses and {completed_rows:,} source rows; "
        f"{result.failed_count:,} coverage windows need retry."
    )
    return 1 if result.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
