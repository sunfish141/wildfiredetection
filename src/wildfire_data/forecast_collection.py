"""Archive raw weather forecast responses with issued-at measurement records."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from .data_archive import (
    CoverageLedger,
    CoverageRecord,
    CoverageStatus,
    RawArtifact,
    sanitize_manifest_value,
    write_raw_artifact,
)
from .forecast_weather import normalize_forecast_measurement
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl


FORECAST_NORMALIZATION_VERSION = "forecast-normalized/v1"


@dataclass(frozen=True)
class ForecastCollectionResult:
    """Archive/normalization result for one issued weather forecast response."""

    raw_artifact: RawArtifact
    normalized_artifacts: tuple[NormalizedArtifact, ...]
    coverage: CoverageRecord
    measurement_count: int


def archive_forecast_response(
    archive_root: str,
    *,
    payload: bytes,
    measurements: Iterable[Mapping[str, object]],
    provider: str,
    product: str,
    model: str,
    model_run_at: object,
    source_uri: str,
    coverage_start: str | date | datetime,
    coverage_end: str | date | datetime,
    region: str,
    response_status_code: int,
    response_headers: Mapping[str, Any] | None = None,
    request_parameters: Mapping[str, Any] | None = None,
    model_version: str | None = None,
    published_at: object | None = None,
    availability_at: object | None = None,
    availability_basis: str | None = None,
    coverage_identity: str | None = None,
    parser_error: str | None = None,
    retrieved_at: datetime | None = None,
    ingestion_id: str | None = None,
) -> ForecastCollectionResult:
    """Store source bytes and normalize long-form measurements without leakage.

    The caller supplies provider-specific parsed measurements. Keeping the raw
    payload first means future parsers or variables can be rebuilt without
    re-downloading an issued forecast run. If provider parsing fails before
    this function can receive measurements, ``parser_error`` records that
    failure while preserving the raw payload as evidence.
    """
    if not provider.strip() or not product.strip() or not model.strip() or not region.strip():
        raise ValueError("provider, product, model, and region must be non-empty")
    if not source_uri.strip():
        raise ValueError("source_uri must be non-empty")
    if parser_error is not None and (not isinstance(parser_error, str) or not parser_error.strip()):
        raise ValueError("parser_error must be a non-empty string when supplied")
    retrieved = _utc_now_or_value(retrieved_at)
    resolved_ingestion_id = ingestion_id or uuid.uuid4().hex
    expected_coverage_id = ":".join(
        (
            "forecast",
            provider,
            model,
            str(model_run_at),
            str(coverage_start),
            str(coverage_end),
            region,
            coverage_identity or "response",
        )
    )
    artifact = write_raw_artifact(
        archive_root,
        source=f"{provider}:{product}",
        payload=payload,
        retrieved_at=retrieved,
        media_type="application/octet-stream",
        provenance={
            "source_url": source_uri,
            "request_parameters": dict(request_parameters or {}),
            "response_headers": dict(response_headers or {}),
            "response_status_code": response_status_code,
            "provider": provider,
            "product": product,
            "model": model,
            "model_version": model_version,
            "model_run_at": model_run_at,
            "published_at": published_at,
            "availability_at": availability_at,
            "availability_basis": availability_basis,
        },
    )
    ledger = CoverageLedger(archive_root)
    if not 200 <= response_status_code < 300:
        coverage = ledger.record(
            source=provider,
            product=product,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            region=region,
            expected_coverage_id=expected_coverage_id,
            status=CoverageStatus.FAILED,
            artifact_sha256s=[artifact.raw_artifact_id],
            error=f"HTTP {response_status_code}",
            detail={"ingestion_id": resolved_ingestion_id, "model": model},
            recorded_at=retrieved,
        )
        return ForecastCollectionResult(artifact, (), coverage, 0)

    if parser_error is not None:
        coverage = ledger.record(
            source=provider,
            product=product,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            region=region,
            expected_coverage_id=expected_coverage_id,
            status=CoverageStatus.FAILED,
            artifact_sha256s=[artifact.raw_artifact_id],
            error=parser_error,
            detail={
                "ingestion_id": resolved_ingestion_id,
                "model": model,
                "failure_stage": "provider-response-parsing",
                "availability_at": availability_at,
                "availability_basis": availability_basis,
            },
            recorded_at=retrieved,
        )
        return ForecastCollectionResult(artifact, (), coverage, 0)

    try:
        normalized_by_valid_date: dict[str, list[dict[str, object]]] = defaultdict(list)
        safe_uri = sanitize_manifest_value(source_uri)
        if not isinstance(safe_uri, str):
            raise ValueError("source_uri must normalize to text")
        for measurement in measurements:
            normalized = normalize_forecast_measurement(
                measurement,
                provider=provider,
                model=model,
                model_run_at=model_run_at,
                retrieved_at=retrieved,
                raw_artifact_id=artifact.raw_artifact_id,
                ingestion_id=resolved_ingestion_id,
                source_uri=safe_uri,
                published_at=published_at,
                availability_at=availability_at,
                availability_basis=availability_basis,
                model_version=model_version,
            )
            normalized_by_valid_date[normalized["valid_at"][:10]].append(normalized)
        normalized_artifacts = tuple(
            write_normalized_jsonl(
                archive_root,
                entity="forecast_weather",
                records=records,
                partitions={
                    "valid_date": valid_date,
                    "model_run_date": str(model_run_at)[:10],
                },
                raw_artifact_ids=[artifact.raw_artifact_id],
                transformation_version=FORECAST_NORMALIZATION_VERSION,
                generated_at=retrieved,
            )
            for valid_date, records in sorted(normalized_by_valid_date.items())
        )
    except Exception as exc:
        coverage = ledger.record(
            source=provider,
            product=product,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            region=region,
            expected_coverage_id=expected_coverage_id,
            status=CoverageStatus.FAILED,
            artifact_sha256s=[artifact.raw_artifact_id],
            error=str(exc),
            detail={
                "ingestion_id": resolved_ingestion_id,
                "model": model,
                "availability_at": availability_at,
                "availability_basis": availability_basis,
            },
            recorded_at=retrieved,
        )
        raise ForecastCollectionError(
            f"Could not normalize forecast response; coverage ledger entry: {coverage.path}"
        ) from exc

    measurement_count = sum(len(records) for records in normalized_by_valid_date.values())
    coverage = ledger.record(
        source=provider,
        product=product,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        region=region,
        expected_coverage_id=expected_coverage_id,
        status=CoverageStatus.COMPLETE if measurement_count else CoverageStatus.EMPTY_CONFIRMED,
        artifact_sha256s=[artifact.raw_artifact_id],
        message="Forecast response archived and normalized",
        detail={
            "ingestion_id": resolved_ingestion_id,
            "model": model,
            "availability_at": availability_at,
            "availability_basis": availability_basis,
            "measurement_count": measurement_count,
            "normalized_artifact_ids": [
                normalized_artifact.normalized_artifact_id
                for normalized_artifact in normalized_artifacts
            ],
        },
        recorded_at=retrieved,
    )
    return ForecastCollectionResult(artifact, normalized_artifacts, coverage, measurement_count)


class ForecastCollectionError(RuntimeError):
    """Raised after bad parsed measurements are preserved as raw evidence."""


def _utc_now_or_value(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return resolved.astimezone(timezone.utc)
