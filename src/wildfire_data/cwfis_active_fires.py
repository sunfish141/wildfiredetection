"""Quota-admitted historical CWFIS active-fire incident-context collection.

The CWFIF active-fire WFS exposes `record_start` and `record_end` for each
agency-reported record version.  This collector retains those record intervals
as operational incident context.  It deliberately does not turn points,
status, or estimated size into a fire-spread geometry label.
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


CWFIS_ACTIVE_FIRES_WFS_URL = "https://geoserver.cwfif.nrcan.gc.ca/geoserver/public/ows"
CWFIS_ACTIVE_FIRES_LAYER = "public:cwfif_national_activefires"
DEFAULT_REGION = "Canada"
DEFAULT_PAGE_SIZE = 1_000
CWFIS_NORMALIZATION_VERSION = "cwfis-active-fire-record-history/v1"
CWFIS_INCIDENT_CONTEXT_QUALITY_SCORE = 0.8
CWFIS_RETENTION_PRIORITY_SCORE = 95


@dataclass(frozen=True)
class CwfisPageCollection:
    """One archived CWFIS WFS result page and its normalized records."""

    page_number: int
    raw_artifact: RawArtifact
    normalized_artifact: NormalizedArtifact | None
    feature_count: int


@dataclass(frozen=True)
class CwfisRangeCollection:
    """Durable result of a historical CWFIS active-fire range collection."""

    pages: tuple[CwfisPageCollection, ...]
    coverage: CoverageRecord
    skipped_terminal_coverage: bool = False

    @property
    def feature_count(self) -> int:
        return sum(page.feature_count for page in self.pages)


def cwfis_record_start_filter(start_date: date, end_date: date) -> str:
    """Return a WFS temporal predicate for all record versions in the range."""
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    exclusive_end = end_date + timedelta(days=1)
    return (
        "record_start DURING "
        f"{start_date.isoformat()}T00:00:00Z/{exclusive_end.isoformat()}T00:00:00Z"
    )


def cwfis_query_parameters(
    start_date: date,
    end_date: date,
    *,
    start_index: int = 0,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, str | int]:
    """Build one deterministic historical-record WFS GeoJSON request."""
    if start_index < 0:
        raise ValueError("start_index must not be negative")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": CWFIS_ACTIVE_FIRES_LAYER,
        "outputFormat": "application/json",
        "count": page_size,
        "startIndex": start_index,
        "sortBy": "record_start,id",
        "CQL_FILTER": cwfis_record_start_filter(start_date, end_date),
    }


def collect_cwfis_active_fire_history(
    archive_root: str,
    *,
    start_date: date,
    end_date: date,
    storage_budget: StorageBudgetPolicy,
    region: str = DEFAULT_REGION,
    page_size: int = DEFAULT_PAGE_SIZE,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (10, 180),
    retrieved_at: datetime | None = None,
    refresh: bool = False,
) -> CwfisRangeCollection:
    """Collect record-version history without reinterpreting it as a perimeter.

    Every page undergoes admission for its raw and normalized representations
    before any provider bytes are persisted.  A storage refusal remains a
    retryable `partial` coverage state with an explicit cap reason.
    """
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    if not region.strip():
        raise ValueError("region must be non-empty")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    target = target_by_key("cwfis_active_fires")
    retrieved = _utc_now_or_value(retrieved_at)
    expected_id = _expected_coverage_id(start_date, end_date, region)
    ledger = CoverageLedger(archive_root)
    latest_by_expected_id = {
        record.expected_coverage_id: record
        for record in ledger.entries()
        if record.expected_coverage_id is not None
    }
    existing = latest_by_expected_id.get(expected_id)
    if not refresh and _is_terminal(existing):
        return CwfisRangeCollection((), existing, skipped_terminal_coverage=True)

    pages: list[CwfisPageCollection] = []
    artifact_ids: list[str] = []
    start_index = 0
    page_number = 1
    owns_session = session is None
    active_session = session or _retrying_session()
    try:
        while True:
            parameters = cwfis_query_parameters(
                start_date,
                end_date,
                start_index=start_index,
                page_size=page_size,
            )
            try:
                response = active_session.get(
                    CWFIS_ACTIVE_FIRES_WFS_URL,
                    params=parameters,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                return _failed_range(
                    archive_root,
                    target=target,
                    start_date=start_date,
                    end_date=end_date,
                    region=region,
                    expected_id=expected_id,
                    artifact_ids=artifact_ids,
                    pages=pages,
                    page_number=page_number,
                    error=f"CWFIS active-fire request failed: {exc}",
                    retrieved_at=retrieved,
                )
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
                    artifact_ids=artifact_ids,
                    detail={
                        "page_count": len(pages),
                        "capped_page_number": page_number,
                        "estimated_page_bytes": estimated_bytes,
                        "retention_priority_score": CWFIS_RETENTION_PRIORITY_SCORE,
                    },
                    error=str(exc),
                    retrieved_at=retrieved,
                )
                return CwfisRangeCollection(tuple(pages), coverage)

            artifact = _archive_response(
                archive_root,
                target=target,
                response=response,
                parameters=parameters,
                retrieved_at=retrieved,
                start_date=start_date,
                end_date=end_date,
                page_number=page_number,
            )
            artifact_ids.append(artifact.raw_artifact_id)
            if not 200 <= response.status_code < 300:
                return _failed_range(
                    archive_root,
                    target=target,
                    start_date=start_date,
                    end_date=end_date,
                    region=region,
                    expected_id=expected_id,
                    artifact_ids=artifact_ids,
                    pages=pages,
                    page_number=page_number,
                    error=f"CWFIS active-fire service returned HTTP {response.status_code}",
                    retrieved_at=retrieved,
                )
            try:
                document = _geojson_document(response.content)
            except ValueError as exc:
                return _failed_range(
                    archive_root,
                    target=target,
                    start_date=start_date,
                    end_date=end_date,
                    region=region,
                    expected_id=expected_id,
                    artifact_ids=artifact_ids,
                    pages=pages,
                    page_number=page_number,
                    error=str(exc),
                    retrieved_at=retrieved,
                )
            features = document["features"]
            normalized_artifact = (
                write_normalized_jsonl(
                    archive_root,
                    entity="incident-snapshots",
                    records=(
                        _incident_context_record(
                            feature,
                            raw_artifact_id=artifact.raw_artifact_id,
                            retrieved_at=retrieved,
                        )
                        for feature in features
                    ),
                    partitions={
                        "source": "cwfis-active-fire-history",
                        "coverage_start": start_date.isoformat(),
                        "coverage_end": end_date.isoformat(),
                        "page": str(page_number),
                    },
                    raw_artifact_ids=[artifact.raw_artifact_id],
                    transformation_version=CWFIS_NORMALIZATION_VERSION,
                    generated_at=retrieved,
                )
                if features
                else None
            )
            pages.append(
                CwfisPageCollection(
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
                artifact_ids=[artifact.raw_artifact_id],
                detail={
                    "feature_count": len(features),
                    "normalized_artifact_id": (
                        normalized_artifact.normalized_artifact_id
                        if normalized_artifact is not None
                        else None
                    ),
                },
                retrieved_at=retrieved,
            )
            total_features = _total_features(document)
            if start_index + len(features) >= total_features:
                break
            if not features:
                return _failed_range(
                    archive_root,
                    target=target,
                    start_date=start_date,
                    end_date=end_date,
                    region=region,
                    expected_id=expected_id,
                    artifact_ids=artifact_ids,
                    pages=pages,
                    page_number=page_number,
                    error="CWFIS active-fire service returned an empty page before all matched features",
                    retrieved_at=retrieved,
                )
            start_index += len(features)
            page_number += 1
    finally:
        if owns_session:
            active_session.close()

    feature_count = sum(page.feature_count for page in pages)
    coverage = _record_range_coverage(
        archive_root,
        target=target,
        start_date=start_date,
        end_date=end_date,
        region=region,
        expected_id=expected_id,
        status=CoverageStatus.COMPLETE if feature_count else CoverageStatus.EMPTY_CONFIRMED,
        artifact_ids=artifact_ids,
        detail={
            "page_count": len(pages),
            "feature_count": feature_count,
            "record_role": "operational_incident_context",
            "incident_context_quality_score": CWFIS_INCIDENT_CONTEXT_QUALITY_SCORE,
            "retention_priority_score": CWFIS_RETENTION_PRIORITY_SCORE,
            "historical_record_interval_preserved": True,
        },
        message="CWFIS agency-reported active-fire record history archived and normalized.",
        retrieved_at=retrieved,
    )
    return CwfisRangeCollection(tuple(pages), coverage)


def _archive_response(
    archive_root: str,
    *,
    target,
    response,
    parameters: Mapping[str, Any],
    retrieved_at: datetime,
    start_date: date,
    end_date: date,
    page_number: int,
) -> RawArtifact:
    response_headers = dict(response.headers)
    content_type = response_headers.get("Content-Type")
    media_type = (
        content_type.split(";", maxsplit=1)[0].strip()
        if isinstance(content_type, str) and content_type.strip()
        else "application/octet-stream"
    )
    return write_raw_artifact(
        archive_root,
        source="CWFIS:cwfis-active-fires",
        payload=response.content,
        retrieved_at=retrieved_at,
        media_type=media_type,
        provenance={
            "source_url": getattr(response, "url", CWFIS_ACTIVE_FIRES_WFS_URL),
            "request_parameters": dict(parameters),
            "response_headers": response_headers,
            "response_status_code": response.status_code,
            "target_key": target.key,
            "source_layer": CWFIS_ACTIVE_FIRES_LAYER,
            "historical_record_interval_preserved": True,
            "coverage_start": start_date,
            "coverage_end": end_date,
            "page_number": page_number,
            "retention_priority_score": CWFIS_RETENTION_PRIORITY_SCORE,
        },
    )


def _failed_range(
    archive_root: str,
    *,
    target,
    start_date: date,
    end_date: date,
    region: str,
    expected_id: str,
    artifact_ids: list[str],
    pages: list[CwfisPageCollection],
    page_number: int,
    error: str,
    retrieved_at: datetime,
) -> CwfisRangeCollection:
    coverage = _record_range_coverage(
        archive_root,
        target=target,
        start_date=start_date,
        end_date=end_date,
        region=region,
        expected_id=expected_id,
        status=CoverageStatus.FAILED,
        artifact_ids=artifact_ids,
        detail={"page_count": len(pages), "failed_page_number": page_number},
        error=error,
        retrieved_at=retrieved_at,
    )
    return CwfisRangeCollection(tuple(pages), coverage)


def _geojson_document(payload: bytes) -> dict[str, Any]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CWFIS active-fire response is not valid GeoJSON") from exc
    if not isinstance(document, dict):
        raise ValueError("CWFIS active-fire response is not a GeoJSON FeatureCollection")
    features = document.get("features")
    if document.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise ValueError("CWFIS active-fire response is not a GeoJSON FeatureCollection")
    if not all(isinstance(feature, dict) and feature.get("type") == "Feature" for feature in features):
        raise ValueError("CWFIS active-fire response contains an invalid GeoJSON feature")
    _total_features(document)
    return document


def _total_features(document: Mapping[str, Any]) -> int:
    value = document.get("totalFeatures", document.get("numberMatched"))
    try:
        total = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CWFIS active-fire response has no valid matched-feature count") from exc
    if total < 0:
        raise ValueError("CWFIS active-fire response has a negative matched-feature count")
    return total


def _incident_context_record(
    feature: Mapping[str, Any],
    *,
    raw_artifact_id: str,
    retrieved_at: datetime,
) -> dict[str, Any]:
    properties = feature.get("properties")
    geometry = feature.get("geometry")
    if not isinstance(properties, Mapping) or not isinstance(geometry, Mapping):
        raise ValueError("CWFIS active-fire feature must contain properties and geometry")
    national_fire_id = properties.get("national_fire_id")
    source_record_id = properties.get("id")
    if national_fire_id in (None, "") and source_record_id in (None, ""):
        raise ValueError("CWFIS active-fire feature has no incident identifier")
    return {
        "record_type": "cwfis_active_fire_incident_context",
        "source": "CWFIS",
        "source_layer": CWFIS_ACTIVE_FIRES_LAYER,
        "record_role": "operational_incident_context",
        "incident_id": national_fire_id or str(source_record_id),
        "source_record_id": source_record_id,
        "record_start": properties.get("record_start"),
        "record_end": properties.get("record_end"),
        "situation_report_date": properties.get("situation_report_date"),
        "status_date": properties.get("status_date"),
        "incident_context_quality_score": CWFIS_INCIDENT_CONTEXT_QUALITY_SCORE,
        "retention_priority_score": CWFIS_RETENTION_PRIORITY_SCORE,
        "historical_record_interval_preserved": True,
        "retrieved_at": _format_utc(retrieved_at),
        "raw_artifact_id": raw_artifact_id,
        "geometry": dict(geometry),
        "source_fields": dict(properties),
    }


def _record_range_coverage(
    archive_root: str,
    *,
    target,
    start_date: date,
    end_date: date,
    region: str,
    expected_id: str,
    status: CoverageStatus,
    artifact_ids: list[str],
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
        artifact_sha256s=artifact_ids,
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
    artifact_ids: list[str],
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
        artifact_sha256s=artifact_ids,
        detail=dict(detail),
        recorded_at=retrieved_at,
    )


def _conservative_page_bytes(payload: bytes) -> int:
    return len(payload) * 2 + 65_536


def _expected_coverage_id(start_date: date, end_date: date, region: str) -> str:
    return f"cwfis-active-fire-history:{region}:{start_date.isoformat()}:{end_date.isoformat()}"


def _is_terminal(record: CoverageRecord | None) -> bool:
    return record is not None and record.status in {
        CoverageStatus.COMPLETE,
        CoverageStatus.EMPTY_CONFIRMED,
    }


def _utc_now_or_value(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return resolved.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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
