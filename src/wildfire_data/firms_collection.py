"""Archive and normalize one NASA FIRMS CSV response without data loss."""

from __future__ import annotations

import csv
import io
import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .data_archive import CoverageLedger, CoverageRecord, CoverageStatus, RawArtifact, write_raw_artifact
from .firms_normalization import normalize_firms_detection
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .storage_budget import StorageBudgetError, StorageBudgetPolicy, load_storage_budget, require_admission


FIRMS_NORMALIZATION_VERSION = "firms-normalized/v1"


@dataclass(frozen=True)
class FirmsCollectionResult:
    """Artifacts and coverage outcome created from one FIRMS response."""

    raw_artifact: RawArtifact | None
    normalized_artifacts: tuple[NormalizedArtifact, ...]
    coverage: CoverageRecord
    record_count: int


def archive_firms_csv_response(
    archive_root: str,
    *,
    payload: bytes,
    product: str,
    coverage_date: date,
    region: str,
    source_url: str,
    response_status_code: int,
    response_headers: Mapping[str, Any] | None = None,
    request_parameters: Mapping[str, Any] | None = None,
    retrieved_at: datetime | None = None,
    ingestion_id: str | None = None,
    minimum_bright_ti4: float | None = 305.0,
    storage_budget: StorageBudgetPolicy | None = None,
) -> FirmsCollectionResult:
    """Capture raw FIRMS bytes, normalize all rows, and record coverage.

    A non-success response is still retained as raw evidence and marked
    ``failed`` in the ledger.  Successful, header-only CSV responses are
    explicitly marked ``empty-confirmed``.  Any parse or normalization error
    leaves the raw evidence intact and records the collection as failed.
    """
    if not product.strip() or not region.strip() or not source_url.strip():
        raise ValueError("product, region, and source_url must be non-empty")
    if not isinstance(response_status_code, int):
        raise TypeError("response_status_code must be an integer")
    retrieved = _utc_now_or_value(retrieved_at)
    resolved_ingestion_id = ingestion_id or uuid.uuid4().hex
    expected_coverage_id = f"firms:{product}:{region}:{coverage_date.isoformat()}"
    resolved_storage_budget = storage_budget or load_storage_budget()
    ledger = CoverageLedger(archive_root)
    estimated_bytes = _conservative_response_bytes(payload)
    try:
        require_admission(
            resolved_storage_budget,
            archive_root,
            category="firms_and_detection_evidence",
            requested_bytes=estimated_bytes,
        )
    except StorageBudgetError as exc:
        coverage = ledger.record(
            source="NASA FIRMS",
            product=product,
            coverage_start=coverage_date,
            coverage_end=coverage_date,
            region=region,
            expected_coverage_id=expected_coverage_id,
            status=CoverageStatus.PARTIAL,
            error=str(exc),
            detail={
                "ingestion_id": resolved_ingestion_id,
                "failure_stage": "storage_admission",
                "estimated_response_and_normalization_bytes": estimated_bytes,
                "retention_priority_score": 100,
            },
            recorded_at=retrieved,
        )
        return FirmsCollectionResult(None, (), coverage, 0)
    artifact = write_raw_artifact(
        archive_root,
        source="NASA FIRMS",
        payload=payload,
        retrieved_at=retrieved,
        media_type="text/csv",
        provenance={
            "source_url": redact_firms_source_url(source_url),
            "request_parameters": dict(request_parameters or {}),
            "response_headers": dict(response_headers or {}),
            "response_status_code": response_status_code,
            "product": product,
            "coverage_date": coverage_date.isoformat(),
            "region": region,
        },
    )
    if not 200 <= response_status_code < 300:
        coverage = ledger.record(
            source="NASA FIRMS",
            product=product,
            coverage_start=coverage_date,
            coverage_end=coverage_date,
            region=region,
            expected_coverage_id=expected_coverage_id,
            status=CoverageStatus.FAILED,
            artifact_sha256s=[artifact.raw_artifact_id],
            error=f"HTTP {response_status_code}",
            detail={"ingestion_id": resolved_ingestion_id},
            recorded_at=retrieved,
        )
        return FirmsCollectionResult(artifact, (), coverage, 0)

    try:
        source_rows = _parse_csv_rows(payload)
        normalized_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row_offset, source_fields in enumerate(source_rows, start=1):
            normalized = normalize_firms_detection(
                source_fields,
                provenance={
                    "provider": "NASA FIRMS",
                    "product": product,
                    "schema_version": FIRMS_NORMALIZATION_VERSION,
                    "raw_artifact_id": artifact.raw_artifact_id,
                    "raw_record_offset": row_offset,
                    "ingestion_id": resolved_ingestion_id,
                    "ingested_at": retrieved,
                },
                minimum_bright_ti4=minimum_bright_ti4,
            )
            normalized_by_date[normalized["acquired_at"][:10]].append(normalized)
        normalized_artifacts = tuple(
            write_normalized_jsonl(
                archive_root,
                entity="fire_detections",
                records=records,
                partitions={"acq_date": acquisition_date},
                raw_artifact_ids=[artifact.raw_artifact_id],
                transformation_version=FIRMS_NORMALIZATION_VERSION,
                generated_at=retrieved,
            )
            for acquisition_date, records in sorted(normalized_by_date.items())
        )
    except Exception as exc:
        coverage = ledger.record(
            source="NASA FIRMS",
            product=product,
            coverage_start=coverage_date,
            coverage_end=coverage_date,
            region=region,
            expected_coverage_id=expected_coverage_id,
            status=CoverageStatus.FAILED,
            artifact_sha256s=[artifact.raw_artifact_id],
            error=str(exc),
            detail={"ingestion_id": resolved_ingestion_id},
            recorded_at=retrieved,
        )
        raise FirmsCollectionError(
            f"Could not normalize FIRMS response for {coverage_date.isoformat()}; "
            f"coverage ledger entry: {coverage.path}"
        ) from exc

    record_count = len(source_rows)
    coverage = ledger.record(
        source="NASA FIRMS",
        product=product,
        coverage_start=coverage_date,
        coverage_end=coverage_date,
        region=region,
        expected_coverage_id=expected_coverage_id,
        status=CoverageStatus.COMPLETE if record_count else CoverageStatus.EMPTY_CONFIRMED,
        artifact_sha256s=[artifact.raw_artifact_id],
        message="FIRMS CSV archived and normalized",
        detail={
            "ingestion_id": resolved_ingestion_id,
            "record_count": record_count,
            "normalized_artifact_ids": [
                normalized_artifact.normalized_artifact_id
                for normalized_artifact in normalized_artifacts
            ],
        },
        recorded_at=retrieved,
    )
    return FirmsCollectionResult(artifact, normalized_artifacts, coverage, record_count)


def record_firms_collection_failure(
    archive_root: str,
    *,
    product: str,
    coverage_date: date,
    region: str,
    error: str,
    retrieved_at: datetime | None = None,
    ingestion_id: str | None = None,
) -> CoverageRecord:
    """Record a failed attempt that did not receive an HTTP response to archive."""
    if not product.strip() or not region.strip() or not error.strip():
        raise ValueError("product, region, and error must be non-empty")
    retrieved = _utc_now_or_value(retrieved_at)
    resolved_ingestion_id = ingestion_id or uuid.uuid4().hex
    return CoverageLedger(archive_root).record(
        source="NASA FIRMS",
        product=product,
        coverage_start=coverage_date,
        coverage_end=coverage_date,
        region=region,
        expected_coverage_id=f"firms:{product}:{region}:{coverage_date.isoformat()}",
        status=CoverageStatus.FAILED,
        error=error,
        detail={"ingestion_id": resolved_ingestion_id, "failure_stage": "request"},
        recorded_at=retrieved,
    )


class FirmsCollectionError(RuntimeError):
    """Raised after a malformed successful response is retained and marked failed."""


def redact_firms_source_url(source_url: str) -> str:
    """Remove a path-embedded FIRMS map key before persisting a request URL."""
    parsed = urlsplit(source_url)
    path_segments = parsed.path.split("/")
    try:
        csv_position = path_segments.index("csv")
    except ValueError:
        # A path-shaped key cannot be identified safely in an unexpected URL.
        # Preserve the useful endpoint origin but never risk persisting it.
        return urlunsplit((parsed.scheme, parsed.netloc, "<redacted-path>", "", ""))
    key_position = csv_position + 1
    if key_position >= len(path_segments):
        return urlunsplit((parsed.scheme, parsed.netloc, "<redacted-path>", "", ""))
    path_segments[key_position] = "<redacted>"
    return urlunsplit((parsed.scheme, parsed.netloc, "/".join(path_segments), parsed.query, parsed.fragment))


def _parse_csv_rows(payload: bytes) -> list[dict[str, str]]:
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("FIRMS CSV response must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise ValueError("FIRMS CSV response is missing a header row")
    return [dict(row) for row in reader]


def _conservative_response_bytes(payload: bytes) -> int:
    """Reserve raw CSV, normalized JSONL, and manifests before accepting bytes."""
    return len(payload) * 4 + 262_144


def _utc_now_or_value(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return resolved.astimezone(timezone.utc)
