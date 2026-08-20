"""Quota-admitted WFIGS perimeter backfill for the compact local dataset.

The public Year-to-Date service is a current/reference view, not a historical
revision archive.  This collector preserves its returned geometry and source
fields as a compact final-reference label tier, and never claims to recreate
an operational snapshot from the requested date.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .collection_catalog import target_by_key
from .data_archive import CoverageLedger, CoverageRecord, CoverageStatus, RawArtifact, write_raw_artifact
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .storage_budget import StorageBudgetError, StorageBudgetPolicy, require_admission


WFIGS_YEAR_TO_DATE_QUERY_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/ArcGIS/rest/services/"
    "WFIGS_Interagency_Perimeters_YearToDate/FeatureServer/0/query"
)
DEFAULT_REGION = "United States"
DEFAULT_PAGE_SIZE = 2_000
WFIGS_COMPACT_NORMALIZATION_VERSION = "wfigs-year-to-date-reference/v1"
WFIGS_LABEL_QUALITY_SCORE = 0.55
WFIGS_RETENTION_PRIORITY_SCORE = 95


@dataclass(frozen=True)
class WfigsPageCollection:
    """One persisted WFIGS query page and its compact normalized records."""

    page_number: int
    raw_artifact: RawArtifact
    normalized_artifact: NormalizedArtifact | None
    feature_count: int


@dataclass(frozen=True)
class WfigsRangeCollection:
    """Every durable outcome from one reference-perimeter backfill attempt."""

    pages: tuple[WfigsPageCollection, ...]
    coverage: CoverageRecord
    skipped_terminal_coverage: bool = False

    @property
    def feature_count(self) -> int:
        return sum(page.feature_count for page in self.pages)


def wfigs_where_clause(start_date: date, end_date: date) -> str:
    """Return the inclusive-date WFIGS filter using a half-open UTC interval."""
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    exclusive_end = end_date + timedelta(days=1)
    return (
        f"poly_DateCurrent >= DATE '{start_date.isoformat()}' "
        f"AND poly_DateCurrent < DATE '{exclusive_end.isoformat()}'"
    )


def wfigs_query_parameters(
    start_date: date,
    end_date: date,
    *,
    offset: int = 0,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, str | int | bool]:
    """Build one deterministic GeoJSON page query for WFIGS Year-to-Date."""
    if offset < 0:
        raise ValueError("offset must not be negative")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return {
        "where": wfigs_where_clause(start_date, end_date),
        "outFields": "*",
        "returnGeometry": "true",
        "resultOffset": offset,
        "resultRecordCount": page_size,
        "f": "geojson",
    }


def collect_wfigs_year_to_date(
    archive_root: str,
    *,
    start_date: date,
    end_date: date,
    storage_budget: StorageBudgetPolicy,
    region: str = DEFAULT_REGION,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (10, 180),
    page_size: int = DEFAULT_PAGE_SIZE,
    retrieved_at: datetime | None = None,
    refresh: bool = False,
) -> WfigsRangeCollection:
    """Archive all quota-admitted WFIGS reference pages for an inclusive range.

    The whole response is preserved before compact normalization.  Admission is
    evaluated before each page is written, using a conservative raw-plus-
    normalized estimate; a quota refusal is recorded as ``partial`` and no
    source page is silently discarded or substituted.
    """
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    if not region.strip():
        raise ValueError("region must be non-empty")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    target = target_by_key("wfigs_current_perimeters")
    retrieved = _utc_now_or_value(retrieved_at)
    ledger = CoverageLedger(archive_root)
    expected_id = _expected_coverage_id(start_date, end_date, region)
    latest_by_expected_id = {
        record.expected_coverage_id: record
        for record in ledger.entries()
        if record.expected_coverage_id is not None
    }
    existing_coverage = latest_by_expected_id.get(expected_id)
    if not refresh and existing_coverage is not None and existing_coverage.status in {
        CoverageStatus.COMPLETE,
        CoverageStatus.EMPTY_CONFIRMED,
    }:
        return WfigsRangeCollection(
            pages=(),
            coverage=existing_coverage,
            skipped_terminal_coverage=True,
        )
    pages = []
    artifact_hashes = []
    offset = 0
    page_number = 1
    owns_session = session is None
    active_session = session or _retrying_session()
    try:
        while True:
            parameters = wfigs_query_parameters(
                start_date,
                end_date,
                offset=offset,
                page_size=page_size,
            )
            try:
                response = active_session.get(
                    WFIGS_YEAR_TO_DATE_QUERY_URL,
                    params=parameters,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                coverage = _record_range_coverage(
                    archive_root,
                    target=target,
                    start_date=start_date,
                    end_date=end_date,
                    region=region,
                    expected_id=expected_id,
                    status=CoverageStatus.FAILED,
                    artifact_sha256s=artifact_hashes,
                    detail={"page_count": len(pages), "failed_page_number": page_number},
                    error=f"WFIGS request failed: {exc}",
                    retrieved_at=retrieved,
                )
                return WfigsRangeCollection(tuple(pages), coverage)

            if not 200 <= response.status_code < 300:
                artifact = _admit_and_archive_page(
                    archive_root,
                    storage_budget=storage_budget,
                    payload=response.content,
                    target_key=target.key,
                    parameters=parameters,
                    response_url=getattr(response, "url", WFIGS_YEAR_TO_DATE_QUERY_URL),
                    response_headers=dict(response.headers),
                    response_status_code=response.status_code,
                    start_date=start_date,
                    end_date=end_date,
                    page_number=page_number,
                    retrieved_at=retrieved,
                )
                if artifact is not None:
                    artifact_hashes.append(artifact.raw_artifact_id)
                coverage = _record_range_coverage(
                    archive_root,
                    target=target,
                    start_date=start_date,
                    end_date=end_date,
                    region=region,
                    expected_id=expected_id,
                    status=CoverageStatus.FAILED,
                    artifact_sha256s=artifact_hashes,
                    detail={"page_count": len(pages), "failed_page_number": page_number},
                    error=f"WFIGS returned HTTP {response.status_code}",
                    retrieved_at=retrieved,
                )
                return WfigsRangeCollection(tuple(pages), coverage)

            try:
                page_document = _geojson_document(response.content)
            except ValueError as exc:
                artifact = _admit_and_archive_page(
                    archive_root,
                    storage_budget=storage_budget,
                    payload=response.content,
                    target_key=target.key,
                    parameters=parameters,
                    response_url=getattr(response, "url", WFIGS_YEAR_TO_DATE_QUERY_URL),
                    response_headers=dict(response.headers),
                    response_status_code=response.status_code,
                    start_date=start_date,
                    end_date=end_date,
                    page_number=page_number,
                    retrieved_at=retrieved,
                )
                if artifact is not None:
                    artifact_hashes.append(artifact.raw_artifact_id)
                coverage = _record_range_coverage(
                    archive_root,
                    target=target,
                    start_date=start_date,
                    end_date=end_date,
                    region=region,
                    expected_id=expected_id,
                    status=CoverageStatus.FAILED,
                    artifact_sha256s=artifact_hashes,
                    detail={"page_count": len(pages), "failed_page_number": page_number},
                    error=str(exc),
                    retrieved_at=retrieved,
                )
                return WfigsRangeCollection(tuple(pages), coverage)

            estimated_bytes = _conservative_page_bytes(response.content)
            try:
                require_admission(
                    storage_budget,
                    archive_root,
                    category="operational_labels_and_progression",
                    requested_bytes=estimated_bytes,
                )
            except StorageBudgetError as exc:
                coverage = _record_range_coverage(
                    archive_root,
                    target=target,
                    start_date=start_date,
                    end_date=end_date,
                    region=region,
                    expected_id=expected_id,
                    status=CoverageStatus.PARTIAL,
                    artifact_sha256s=artifact_hashes,
                    detail={
                        "page_count": len(pages),
                        "capped_page_number": page_number,
                        "estimated_page_bytes": estimated_bytes,
                        "retention_priority_score": WFIGS_RETENTION_PRIORITY_SCORE,
                    },
                    error=str(exc),
                    retrieved_at=retrieved,
                )
                return WfigsRangeCollection(tuple(pages), coverage)

            artifact = _archive_page(
                archive_root,
                payload=response.content,
                target_key=target.key,
                parameters=parameters,
                response_url=getattr(response, "url", WFIGS_YEAR_TO_DATE_QUERY_URL),
                response_headers=dict(response.headers),
                response_status_code=response.status_code,
                start_date=start_date,
                end_date=end_date,
                page_number=page_number,
                retrieved_at=retrieved,
            )
            artifact_hashes.append(artifact.raw_artifact_id)
            _record_page_coverage(
                archive_root,
                target=target,
                start_date=start_date,
                end_date=end_date,
                region=region,
                parent_expected_id=expected_id,
                page_number=page_number,
                status=CoverageStatus.PARTIAL,
                artifact_sha256s=[artifact.raw_artifact_id],
                detail={"stage": "raw_archived", "feature_count": len(page_document["features"])},
                retrieved_at=retrieved,
            )
            features = page_document["features"]
            normalized_artifact = (
                write_normalized_jsonl(
                    archive_root,
                    entity="operational-perimeters",
                    records=(
                        _compact_perimeter_record(
                            feature,
                            raw_artifact_id=artifact.raw_artifact_id,
                            retrieval_time=retrieved,
                            start_date=start_date,
                            end_date=end_date,
                        )
                        for feature in features
                    ),
                    partitions={
                        "source": "wfigs-year-to-date",
                        "coverage_start": start_date.isoformat(),
                        "coverage_end": end_date.isoformat(),
                        "page": str(page_number),
                    },
                    raw_artifact_ids=[artifact.raw_artifact_id],
                    transformation_version=WFIGS_COMPACT_NORMALIZATION_VERSION,
                    generated_at=retrieved,
                )
                if features
                else None
            )
            pages.append(
                WfigsPageCollection(
                    page_number=page_number,
                    raw_artifact=artifact,
                    normalized_artifact=normalized_artifact,
                    feature_count=len(features),
                )
            )
            _record_page_coverage(
                archive_root,
                target=target,
                start_date=start_date,
                end_date=end_date,
                region=region,
                parent_expected_id=expected_id,
                page_number=page_number,
                status=CoverageStatus.COMPLETE,
                artifact_sha256s=[artifact.raw_artifact_id],
                detail={
                    "stage": "raw_archived_and_normalized",
                    "feature_count": len(features),
                    "normalized_artifact_id": (
                        normalized_artifact.normalized_artifact_id
                        if normalized_artifact is not None
                        else None
                    ),
                },
                retrieved_at=retrieved,
            )
            _record_range_coverage(
                archive_root,
                target=target,
                start_date=start_date,
                end_date=end_date,
                region=region,
                expected_id=expected_id,
                status=CoverageStatus.PARTIAL,
                artifact_sha256s=artifact_hashes,
                detail={
                    "page_count": len(pages),
                    "feature_count": sum(page.feature_count for page in pages),
                    "next_page_number": page_number + 1,
                    "checkpoint": True,
                },
                message="WFIGS page checkpointed; collection has not yet reached a terminal page.",
                retrieved_at=retrieved,
            )
            if not _has_next_page(page_document, feature_count=len(features), page_size=page_size):
                break
            offset += len(features)
            page_number += 1
    finally:
        if owns_session:
            active_session.close()

    total_features = sum(page.feature_count for page in pages)
    status = CoverageStatus.COMPLETE if total_features else CoverageStatus.EMPTY_CONFIRMED
    coverage = _record_range_coverage(
        archive_root,
        target=target,
        start_date=start_date,
        end_date=end_date,
        region=region,
        expected_id=expected_id,
        status=status,
        artifact_sha256s=artifact_hashes,
        detail={
            "page_count": len(pages),
            "feature_count": total_features,
            "label_tier": "final_reference",
            "label_quality_score": WFIGS_LABEL_QUALITY_SCORE,
            "retention_priority_score": WFIGS_RETENTION_PRIORITY_SCORE,
            "historical_snapshot_recreated": False,
        },
        message=(
            "WFIGS Year-to-Date reference geometries archived and compactly normalized."
            if pages
            else "WFIGS Year-to-Date query returned no reference geometries."
        ),
        retrieved_at=retrieved,
    )
    return WfigsRangeCollection(tuple(pages), coverage)


def _admit_and_archive_page(
    archive_root: str,
    *,
    storage_budget: StorageBudgetPolicy,
    payload: bytes,
    target_key: str,
    parameters: Mapping[str, Any],
    response_url: str,
    response_headers: Mapping[str, Any],
    response_status_code: int,
    start_date: date,
    end_date: date,
    page_number: int,
    retrieved_at: datetime,
) -> RawArtifact | None:
    """Preserve a non-success/invalid response when capacity permits it."""
    try:
        require_admission(
            storage_budget,
            archive_root,
            category="operational_labels_and_progression",
            requested_bytes=len(payload) + 16_384,
        )
    except StorageBudgetError:
        return None
    return _archive_page(
        archive_root,
        payload=payload,
        target_key=target_key,
        parameters=parameters,
        response_url=response_url,
        response_headers=response_headers,
        response_status_code=response_status_code,
        start_date=start_date,
        end_date=end_date,
        page_number=page_number,
        retrieved_at=retrieved_at,
    )


def _archive_page(
    archive_root: str,
    *,
    payload: bytes,
    target_key: str,
    parameters: Mapping[str, Any],
    response_url: str,
    response_headers: Mapping[str, Any],
    response_status_code: int,
    start_date: date,
    end_date: date,
    page_number: int,
    retrieved_at: datetime,
) -> RawArtifact:
    return write_raw_artifact(
        archive_root,
        source="NIFC WFIGS",
        payload=payload,
        retrieved_at=retrieved_at,
        media_type="application/geo+json",
        provenance={
            "source_url": response_url,
            "request_parameters": dict(parameters),
            "response_headers": dict(response_headers),
            "response_status_code": response_status_code,
            "target_key": target_key,
            "source_view": "WFIGS Year-to-Date current/reference view",
            "coverage_start": start_date,
            "coverage_end": end_date,
            "page_number": page_number,
            "label_tier": "final_reference",
            "historical_snapshot_recreated": False,
            "retention_priority_score": WFIGS_RETENTION_PRIORITY_SCORE,
        },
    )


def _record_range_coverage(
    archive_root: str,
    *,
    target,
    start_date: date,
    end_date: date,
    region: str,
    expected_id: str,
    status: CoverageStatus,
    artifact_sha256s: list[str],
    detail: Mapping[str, Any],
    retrieved_at: datetime,
    message: str | None = None,
    error: str | None = None,
) -> CoverageRecord:
    return CoverageLedger(archive_root).record(
        source=target.provider,
        product=target.key,
        coverage_start=start_date,
        coverage_end=end_date,
        region=region,
        expected_coverage_id=expected_id,
        status=status,
        artifact_sha256s=artifact_sha256s,
        detail=dict(detail),
        message=message,
        error=error,
        recorded_at=retrieved_at,
    )


def _record_page_coverage(
    archive_root: str,
    *,
    target,
    start_date: date,
    end_date: date,
    region: str,
    parent_expected_id: str,
    page_number: int,
    status: CoverageStatus,
    artifact_sha256s: list[str],
    detail: Mapping[str, Any],
    retrieved_at: datetime,
) -> CoverageRecord:
    return CoverageLedger(archive_root).record(
        source=target.provider,
        product=target.key,
        coverage_start=start_date,
        coverage_end=end_date,
        region=region,
        tile=f"page-{page_number}",
        expected_coverage_id=f"{parent_expected_id}:page:{page_number}",
        status=status,
        artifact_sha256s=artifact_sha256s,
        detail=dict(detail),
        recorded_at=retrieved_at,
    )


def _compact_perimeter_record(
    feature: Mapping[str, Any],
    *,
    raw_artifact_id: str,
    retrieval_time: datetime,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    if not isinstance(feature, Mapping):
        raise ValueError("WFIGS GeoJSON features must be objects")
    properties = feature.get("properties")
    geometry = feature.get("geometry")
    if not isinstance(properties, Mapping) or not isinstance(geometry, Mapping):
        raise ValueError("WFIGS GeoJSON feature must contain properties and geometry")
    return {
        "source": "NIFC WFIGS",
        "source_view": "year-to-date-current-reference",
        "label_tier": "final_reference",
        "label_quality_score": WFIGS_LABEL_QUALITY_SCORE,
        "retention_priority_score": WFIGS_RETENTION_PRIORITY_SCORE,
        "historical_snapshot_recreated": False,
        "coverage_start": start_date.isoformat(),
        "coverage_end": end_date.isoformat(),
        "retrieved_at": retrieval_time.isoformat().replace("+00:00", "Z"),
        "raw_artifact_id": raw_artifact_id,
        "geometry": dict(geometry),
        "source_fields": dict(properties),
    }


def _geojson_document(payload: bytes) -> dict[str, Any]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("WFIGS response is not valid GeoJSON") from exc
    if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
        raise ValueError("WFIGS response is not a GeoJSON FeatureCollection")
    features = document.get("features")
    if not isinstance(features, list):
        raise ValueError("WFIGS response has no feature list")
    if not all(isinstance(feature, dict) and feature.get("type") == "Feature" for feature in features):
        raise ValueError("WFIGS response contains an invalid GeoJSON feature")
    return document


def _has_next_page(document: Mapping[str, Any], *, feature_count: int, page_size: int) -> bool:
    return bool(document.get("exceededTransferLimit")) or feature_count >= page_size


def _conservative_page_bytes(payload: bytes) -> int:
    """Reserve raw evidence, compact normalization, and their manifests."""
    return len(payload) * 2 + 65_536


def _expected_coverage_id(start_date: date, end_date: date, region: str) -> str:
    return f"wfigs-ytd-reference:{region}:{start_date.isoformat()}:{end_date.isoformat()}"


def _utc_now_or_value(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return resolved.astimezone(timezone.utc)


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
