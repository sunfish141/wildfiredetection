"""Quota-admitted archival of the public NALCMS 2020 land-cover release.

The NALCMS release is retained as raw, versioned static evidence so the
cross-border land-cover/fuel-form proxy can be compacted later without asking
the provider to serve the large national rasters again.  This module does not
pretend that a 30 m categorical raster is already a model-ready 500 m fuel
grid: categorical reprojection and the target grid contract remain explicit
downstream work.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .data_archive import (
    CoverageLedger,
    CoverageRecord,
    CoverageStatus,
    RawArtifact,
    write_raw_artifact_from_file,
)
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .storage_budget import StorageBudgetError, StorageBudgetPolicy, require_admission


NALCMS_RELEASE_NAME = "North American Land Change Monitoring System 2020 Land Cover v2"
NALCMS_RELEASE_VERSION = "2020 v2"
NALCMS_RESOLUTION_METRES = 30
NALCMS_CLASSIFICATION = "19-class FAO Land Cover Classification System"
NALCMS_RETENTION_PRIORITY_SCORE = 70
NALCMS_SOURCE_QUALITY_SCORE = 0.85
NALCMS_NORMALIZATION_VERSION = "nalcms-2020-v2-source-release/v1"
NALCMS_SOURCE = "CEC:nalcms land cover"
STAGING_ALLOWANCE_BYTES = 67_108_864
SOURCE_REQUEST_HEADERS = {"Accept-Encoding": "identity"}


@dataclass(frozen=True)
class NalcmsRelease:
    """One immutable, country-specific NALCMS source archive."""

    key: str
    country: str
    source_url: str
    component_years: Mapping[str, int]


NALCMS_RELEASES = (
    NalcmsRelease(
        key="canada",
        country="Canada",
        source_url=(
            "https://www.cec.org/files/atlas_layers/1_terrestrial_ecosystems/"
            "1_01_0_land_cover_2020_30m/can_land_cover_2020v2_30m_tif.zip"
        ),
        component_years={"Canada": 2020},
    ),
    NalcmsRelease(
        key="united-states",
        country="United States",
        source_url=(
            "https://www.cec.org/files/atlas_layers/1_terrestrial_ecosystems/"
            "1_01_0_land_cover_2020_30m/usa_land_cover_2020v2_30m_tif.zip"
        ),
        component_years={"CONUS": 2019, "Alaska": 2021},
    ),
)


@dataclass(frozen=True)
class NalcmsReleaseCollection:
    """Durable result for one NALCMS source release."""

    release: NalcmsRelease
    raw_artifact: RawArtifact | None
    normalized_artifact: NormalizedArtifact | None
    coverage: CoverageRecord
    skipped_terminal_coverage: bool = False


@dataclass(frozen=True)
class NalcmsCollection:
    """Results for every requested country source release."""

    releases: tuple[NalcmsReleaseCollection, ...]

    @property
    def complete_count(self) -> int:
        return sum(release.coverage.status == CoverageStatus.COMPLETE for release in self.releases)

    @property
    def skipped_count(self) -> int:
        return sum(release.skipped_terminal_coverage for release in self.releases)

    @property
    def partial_or_failed_count(self) -> int:
        return sum(
            release.coverage.status not in {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED}
            for release in self.releases
        )


def collect_nalcms_land_cover(
    archive_root: str | Path,
    *,
    storage_budget: StorageBudgetPolicy,
    releases: Iterable[NalcmsRelease] = NALCMS_RELEASES,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (20, 900),
    retrieved_at: datetime | None = None,
    staging_directory: str | Path | None = None,
) -> NalcmsCollection:
    """Archive quota-admitted NALCMS country ZIPs without loading them in RAM.

    Source files stream into temporary staging outside ``data/``.  Only the
    immutable raw evidence and small provenance record enter the governed
    archive.  A later categorical compactor must use an explicit grid and
    mode/fraction rule; it must not average land-cover class identifiers.
    """
    root = Path(archive_root)
    selected_releases = tuple(releases)
    if not selected_releases:
        raise ValueError("releases must not be empty")
    if len({release.key for release in selected_releases}) != len(selected_releases):
        raise ValueError("release keys must be unique")
    retrieved = _utc_now_or_value(retrieved_at)
    ledger = CoverageLedger(root)
    latest_by_expected = {
        entry.expected_coverage_id: entry
        for entry in ledger.entries()
        if entry.expected_coverage_id is not None
    }
    active_session = session or _retrying_session()
    owns_session = session is None
    temporary_root = Path(staging_directory) if staging_directory is not None else Path(tempfile.gettempdir())
    temporary_root.mkdir(parents=True, exist_ok=True)
    _require_staging_outside_archive(root, temporary_root)
    results = []
    try:
        for release in selected_releases:
            expected_id = _expected_coverage_id(release)
            existing = latest_by_expected.get(expected_id)
            if existing is not None and existing.status in {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED}:
                results.append(NalcmsReleaseCollection(release, None, None, existing, True))
                continue

            try:
                head = active_session.head(
                    release.source_url,
                    allow_redirects=True,
                    headers=SOURCE_REQUEST_HEADERS,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                coverage = _record_coverage(
                    ledger,
                    release=release,
                    expected_id=expected_id,
                    status=CoverageStatus.FAILED,
                    retrieved_at=retrieved,
                    error=f"NALCMS metadata request failed: {exc}",
                    detail={"stage": "head"},
                )
                results.append(NalcmsReleaseCollection(release, None, None, coverage))
                continue
            if not 200 <= head.status_code < 300:
                coverage = _record_coverage(
                    ledger,
                    release=release,
                    expected_id=expected_id,
                    status=CoverageStatus.FAILED,
                    retrieved_at=retrieved,
                    error=f"NALCMS metadata request returned HTTP {head.status_code}",
                    detail={"stage": "head", "response_headers": dict(head.headers)},
                )
                results.append(NalcmsReleaseCollection(release, None, None, coverage))
                continue
            try:
                advertised_bytes = _content_length(head.headers)
            except ValueError as exc:
                coverage = _record_coverage(
                    ledger,
                    release=release,
                    expected_id=expected_id,
                    status=CoverageStatus.PARTIAL,
                    retrieved_at=retrieved,
                    error=str(exc),
                    detail={"stage": "head", "response_headers": dict(head.headers)},
                )
                results.append(NalcmsReleaseCollection(release, None, None, coverage))
                continue
            try:
                require_admission(
                    storage_budget,
                    root,
                    category="static_cell_features",
                    requested_bytes=advertised_bytes + STAGING_ALLOWANCE_BYTES,
                )
            except StorageBudgetError as exc:
                coverage = _record_coverage(
                    ledger,
                    release=release,
                    expected_id=expected_id,
                    status=CoverageStatus.PARTIAL,
                    retrieved_at=retrieved,
                    error=str(exc),
                    detail={
                        "stage": "storage_admission_before_download",
                        "advertised_bytes": advertised_bytes,
                    },
                )
                results.append(NalcmsReleaseCollection(release, None, None, coverage))
                break

            temporary_path: Path | None = None
            response: requests.Response | None = None
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f"nalcms-{release.key}-", suffix=".zip", dir=temporary_root
                )
                temporary_path = Path(temporary_name)
                with os.fdopen(descriptor, "wb") as destination:
                    response = active_session.get(
                        release.source_url,
                        stream=True,
                        headers=SOURCE_REQUEST_HEADERS,
                        timeout=timeout,
                    )
                    if not 200 <= response.status_code < 300:
                        coverage = _record_coverage(
                            ledger,
                            release=release,
                            expected_id=expected_id,
                            status=CoverageStatus.FAILED,
                            retrieved_at=retrieved,
                            error=f"NALCMS download returned HTTP {response.status_code}",
                            detail={
                                "stage": "download",
                                "response_headers": dict(response.headers),
                                "body_archived": False,
                            },
                        )
                        results.append(NalcmsReleaseCollection(release, None, None, coverage))
                        continue
                    received_bytes = 0
                    exceeded_advertised_length = False
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            if received_bytes + len(chunk) > advertised_bytes:
                                exceeded_advertised_length = True
                                break
                            destination.write(chunk)
                            received_bytes += len(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
                if exceeded_advertised_length:
                    coverage = _record_coverage(
                        ledger,
                        release=release,
                        expected_id=expected_id,
                        status=CoverageStatus.FAILED,
                        retrieved_at=retrieved,
                        error=(
                            "NALCMS download exceeded the advertised Content-Length; "
                            "the untrusted oversized body was not archived"
                        ),
                        detail={
                            "stage": "download",
                            "advertised_bytes": advertised_bytes,
                            "received_bytes_before_rejected_chunk": received_bytes,
                            "body_archived": False,
                        },
                    )
                    results.append(NalcmsReleaseCollection(release, None, None, coverage))
                    continue
                try:
                    require_admission(
                        storage_budget,
                        root,
                        category="static_cell_features",
                        requested_bytes=received_bytes + STAGING_ALLOWANCE_BYTES,
                    )
                except StorageBudgetError as exc:
                    coverage = _record_coverage(
                        ledger,
                        release=release,
                        expected_id=expected_id,
                        status=CoverageStatus.PARTIAL,
                        retrieved_at=retrieved,
                        error=str(exc),
                        detail={
                            "stage": "storage_admission_before_archive",
                            "received_bytes": received_bytes,
                        },
                    )
                    results.append(NalcmsReleaseCollection(release, None, None, coverage))
                    continue
                if received_bytes != advertised_bytes:
                    raw_artifact = write_raw_artifact_from_file(
                        root,
                        source=NALCMS_SOURCE,
                        source_path=temporary_path,
                        retrieved_at=retrieved,
                        media_type=_response_media_type(response.headers),
                        provenance=_raw_provenance(
                            release,
                            source_url=getattr(response, "url", release.source_url),
                            response_headers=dict(response.headers),
                            response_status_code=response.status_code,
                            advertised_bytes=advertised_bytes,
                        ),
                    )
                    coverage = _record_coverage(
                        ledger,
                        release=release,
                        expected_id=expected_id,
                        status=CoverageStatus.FAILED,
                        retrieved_at=retrieved,
                        artifact_sha256s=[raw_artifact.raw_artifact_id],
                        error=(
                            "NALCMS download length differs from the advertised Content-Length: "
                            f"{received_bytes} != {advertised_bytes}"
                        ),
                        detail={"stage": "download", "received_bytes": received_bytes},
                    )
                    results.append(NalcmsReleaseCollection(release, raw_artifact, None, coverage))
                    continue
                if not zipfile.is_zipfile(temporary_path):
                    raw_artifact = write_raw_artifact_from_file(
                        root,
                        source=NALCMS_SOURCE,
                        source_path=temporary_path,
                        retrieved_at=retrieved,
                        media_type=_response_media_type(response.headers),
                        provenance=_raw_provenance(
                            release,
                            source_url=getattr(response, "url", release.source_url),
                            response_headers=dict(response.headers),
                            response_status_code=response.status_code,
                            advertised_bytes=advertised_bytes,
                        ),
                    )
                    coverage = _record_coverage(
                        ledger,
                        release=release,
                        expected_id=expected_id,
                        status=CoverageStatus.FAILED,
                        retrieved_at=retrieved,
                        artifact_sha256s=[raw_artifact.raw_artifact_id],
                        error="NALCMS response has the advertised length but is not a ZIP archive",
                        detail={"stage": "download", "received_bytes": received_bytes},
                    )
                    results.append(NalcmsReleaseCollection(release, raw_artifact, None, coverage))
                    continue
                raw_artifact = write_raw_artifact_from_file(
                    root,
                    source=NALCMS_SOURCE,
                    source_path=temporary_path,
                    retrieved_at=retrieved,
                    media_type=_response_media_type(response.headers),
                    provenance=_raw_provenance(
                        release,
                        source_url=getattr(response, "url", release.source_url),
                        response_headers=dict(response.headers),
                        response_status_code=response.status_code,
                        advertised_bytes=advertised_bytes,
                    ),
                )
            except (OSError, requests.RequestException, ValueError) as exc:
                coverage = _record_coverage(
                    ledger,
                    release=release,
                    expected_id=expected_id,
                    status=CoverageStatus.FAILED,
                    retrieved_at=retrieved,
                    error=f"NALCMS streaming download failed: {exc}",
                    detail={"stage": "download"},
                )
                results.append(NalcmsReleaseCollection(release, None, None, coverage))
                continue
            finally:
                _close_response(response)
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

            normalized_artifact = write_normalized_jsonl(
                root,
                entity="static_cell_features",
                records=[_normalized_release_record(release, raw_artifact, advertised_bytes)],
                partitions={"dataset": "nalcms-2020-v2", "country": release.key},
                raw_artifact_ids=[raw_artifact.raw_artifact_id],
                transformation_version=NALCMS_NORMALIZATION_VERSION,
                generated_at=retrieved,
            )
            coverage = _record_coverage(
                ledger,
                release=release,
                expected_id=expected_id,
                status=CoverageStatus.COMPLETE,
                retrieved_at=retrieved,
                artifact_sha256s=[raw_artifact.raw_artifact_id],
                detail={
                    "advertised_bytes": advertised_bytes,
                    "raw_artifact_id": raw_artifact.raw_artifact_id,
                    "normalized_artifact_id": normalized_artifact.normalized_artifact_id,
                    "compact_feature_derivation": "pending explicit categorical target-grid contract",
                },
            )
            results.append(NalcmsReleaseCollection(release, raw_artifact, normalized_artifact, coverage))
    finally:
        if owns_session:
            active_session.close()
    return NalcmsCollection(tuple(results))


def _expected_coverage_id(release: NalcmsRelease) -> str:
    return f"nalcms:2020-v2:30m:{release.key}:source-zip"


def _content_length(headers: Mapping[str, Any]) -> int:
    value = headers.get("Content-Length") or headers.get("content-length")
    try:
        length = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("NALCMS source response has no valid Content-Length") from exc
    if length <= 0:
        raise ValueError("NALCMS source Content-Length must be positive")
    return length


def _require_staging_outside_archive(archive_root: Path, staging_root: Path) -> None:
    """Keep full provider ZIP staging outside the governed ``data/`` tree."""
    resolved_archive = archive_root.resolve()
    resolved_staging = staging_root.resolve()
    if resolved_staging == resolved_archive or resolved_archive in resolved_staging.parents:
        raise ValueError(
            "staging_directory must be outside data-root so a full source ZIP cannot bypass the storage budget"
        )


def _close_response(response: Any | None) -> None:
    """Close a streamed response when the real or test response supports it."""
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _raw_provenance(
    release: NalcmsRelease,
    *,
    source_url: str,
    response_headers: Mapping[str, Any],
    response_status_code: int,
    advertised_bytes: int,
) -> dict[str, object]:
    return {
        "source_url": source_url,
        "response_headers": dict(response_headers),
        "response_status_code": response_status_code,
        "provider": "Commission for Environmental Cooperation",
        "release_name": NALCMS_RELEASE_NAME,
        "release_version": NALCMS_RELEASE_VERSION,
        "country": release.country,
        "component_years": dict(release.component_years),
        "advertised_bytes": advertised_bytes,
        "asset_role": "versioned_land_cover_source_archive",
    }


def _normalized_release_record(
    release: NalcmsRelease,
    raw_artifact: RawArtifact,
    advertised_bytes: int,
) -> dict[str, object]:
    return {
        "record_type": "static_land_cover_source_release",
        "source": "CEC NALCMS",
        "release_name": NALCMS_RELEASE_NAME,
        "release_version": NALCMS_RELEASE_VERSION,
        "country": release.country,
        "component_years": dict(release.component_years),
        "resolution_metres": NALCMS_RESOLUTION_METRES,
        "classification": NALCMS_CLASSIFICATION,
        "source_url": release.source_url,
        "advertised_bytes": advertised_bytes,
        "raw_artifact_id": raw_artifact.raw_artifact_id,
        "compact_feature_derivation": {
            "status": "pending",
            "requirement": "reproject categorical classes with mode/fraction rules; never average class identifiers",
        },
        "static_source_quality_score": NALCMS_SOURCE_QUALITY_SCORE,
        "retention_priority_score": NALCMS_RETENTION_PRIORITY_SCORE,
    }


def _record_coverage(
    ledger: CoverageLedger,
    *,
    release: NalcmsRelease,
    expected_id: str,
    status: CoverageStatus,
    retrieved_at: datetime,
    artifact_sha256s: Iterable[str] = (),
    error: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> CoverageRecord:
    return ledger.record(
        source="CEC",
        product="NALCMS_2020_v2_30m_land_cover",
        coverage_start="2019",
        coverage_end="2021",
        region=release.country,
        tile=release.key,
        expected_coverage_id=expected_id,
        status=status,
        artifact_sha256s=artifact_sha256s,
        error=error,
        detail={
            "release_name": NALCMS_RELEASE_NAME,
            "release_version": NALCMS_RELEASE_VERSION,
            "component_years": dict(release.component_years),
            **dict(detail or {}),
        },
        recorded_at=retrieved_at,
    )


def _response_media_type(headers: Mapping[str, Any]) -> str:
    value = headers.get("Content-Type") or headers.get("content-type")
    if isinstance(value, str) and value.strip():
        return value.split(";", 1)[0].strip()
    return "application/zip"


def _retrying_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def _utc_now_or_value(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return resolved.astimezone(timezone.utc)
