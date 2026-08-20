"""Generic, provenance-preserving snapshot ingestion for non-FIRMS sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import requests

from .collection_catalog import CollectionTarget
from .data_archive import CoverageLedger, CoverageRecord, CoverageStatus, RawArtifact, write_raw_artifact


@dataclass(frozen=True)
class SourceSnapshotReceipt:
    """Raw evidence and coverage outcome for one external-source snapshot."""

    target: CollectionTarget
    raw_artifact: RawArtifact
    coverage: CoverageRecord


class SourceSnapshotRequestError(RuntimeError):
    """A request failure that has been durably recorded in the coverage ledger."""

    def __init__(self, coverage: CoverageRecord, cause: requests.RequestException) -> None:
        super().__init__(f"Snapshot request failed; coverage ledger entry: {coverage.path}")
        self.coverage = coverage
        self.__cause__ = cause


def archive_source_snapshot(
    archive_root: str,
    *,
    target: CollectionTarget,
    payload: bytes,
    coverage_start: str | date | datetime,
    coverage_end: str | date | datetime,
    source_url: str,
    status: CoverageStatus | str,
    response_status_code: int | None = None,
    region: str | None = None,
    tile: str | None = None,
    expected_coverage_id: str | None = None,
    request_parameters: Mapping[str, Any] | None = None,
    response_headers: Mapping[str, Any] | None = None,
    message: str | None = None,
    error: str | None = None,
    detail: Mapping[str, Any] | None = None,
    retrieved_at: datetime | None = None,
) -> SourceSnapshotReceipt:
    """Archive a source response and its explicit collection outcome.

    Provider adapters perform parsing separately.  This keeps source bytes,
    snapshot timing, status, and error evidence even when a parser changes.
    """
    artifact = write_raw_artifact(
        archive_root,
        source=f"{target.provider}:{target.key}",
        payload=payload,
        retrieved_at=retrieved_at,
        media_type="application/octet-stream",
        provenance={
            "source_url": source_url,
            "request_parameters": dict(request_parameters or {}),
            "response_headers": dict(response_headers or {}),
            "response_status_code": response_status_code,
            "target": {
                "key": target.key,
                "entity": target.entity,
                "label_tier": target.label_tier,
            },
        },
    )
    coverage = CoverageLedger(archive_root).record(
        source=target.provider,
        product=target.key,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        region=region or target.region,
        tile=tile,
        expected_coverage_id=expected_coverage_id,
        status=status,
        artifact_sha256s=[artifact.raw_artifact_id],
        message=message,
        error=error,
        detail=detail,
        recorded_at=retrieved_at,
    )
    return SourceSnapshotReceipt(target=target, raw_artifact=artifact, coverage=coverage)


def fetch_http_source_snapshot(
    archive_root: str,
    *,
    target: CollectionTarget,
    source_url: str,
    coverage_start: str | date | datetime,
    coverage_end: str | date | datetime,
    timeout: int | tuple[int, int] = 90,
    request_parameters: Mapping[str, Any] | None = None,
    request_headers: Mapping[str, str] | None = None,
    region: str | None = None,
    tile: str | None = None,
    expected_coverage_id: str | None = None,
    session: requests.Session | None = None,
    retrieved_at: datetime | None = None,
) -> SourceSnapshotReceipt:
    """Fetch one HTTP source response and archive it before any parsing.

    A successful HTTP response is recorded as ``complete`` because a generic
    adapter cannot safely infer a provider's empty-response semantics.  A
    source-specific adapter can call :func:`archive_source_snapshot` directly
    when it can prove ``empty-confirmed`` or ``partial``.
    """
    owns_session = session is None
    active_session = session or requests.Session()
    try:
        try:
            response = active_session.get(
                source_url,
                params=dict(request_parameters or {}),
                headers=dict(request_headers or {}),
                timeout=timeout,
            )
        except requests.RequestException as exc:
            coverage = record_source_snapshot_failure(
                archive_root,
                target=target,
                coverage_start=coverage_start,
                coverage_end=coverage_end,
                region=region,
                tile=tile,
                expected_coverage_id=expected_coverage_id,
                error=str(exc),
                retrieved_at=retrieved_at,
            )
            raise SourceSnapshotRequestError(coverage, exc) from exc
        return archive_source_snapshot(
            archive_root,
            target=target,
            payload=response.content,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            source_url=response.url,
            status=(
                CoverageStatus.COMPLETE
                if 200 <= response.status_code < 300
                else CoverageStatus.FAILED
            ),
            response_status_code=response.status_code,
            region=region,
            tile=tile,
            expected_coverage_id=expected_coverage_id,
            request_parameters=request_parameters,
            response_headers=dict(response.headers),
            error=(None if 200 <= response.status_code < 300 else f"HTTP {response.status_code}"),
            retrieved_at=retrieved_at,
        )
    finally:
        if owns_session:
            active_session.close()


def record_source_snapshot_failure(
    archive_root: str,
    *,
    target: CollectionTarget,
    coverage_start: str | date | datetime,
    coverage_end: str | date | datetime,
    error: str,
    region: str | None = None,
    tile: str | None = None,
    expected_coverage_id: str | None = None,
    retrieved_at: datetime | None = None,
) -> CoverageRecord:
    """Record a failed source request that returned no archivable response."""
    return CoverageLedger(archive_root).record(
        source=target.provider,
        product=target.key,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        region=region or target.region,
        tile=tile,
        expected_coverage_id=expected_coverage_id,
        status=CoverageStatus.FAILED,
        error=error,
        detail={"failure_stage": "request", "target_entity": target.entity},
        recorded_at=retrieved_at,
    )
