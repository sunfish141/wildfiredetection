"""Quota-admitted NASA FEDS perimeter snapshots for satellite-weak labels.

FEDS is not an operational perimeter archive.  It is a VIIRS/NOAA-20-derived
fire-tracking product, so every record written here is explicitly labelled as
``weak_satellite``.  The service exposes 12-hour perimeter snapshots with
per-snapshot new-pixel and active-front metrics; later label construction can
derive a weak ``newly_burned`` target by comparing successive snapshots.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .collection_catalog import target_by_key
from .data_archive import CoverageLedger, CoverageRecord, CoverageStatus, RawArtifact, write_raw_artifact
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .storage_budget import StorageBudgetError, StorageBudgetPolicy, require_admission


FEDS_PERIMETERS_MAPSERVER_URL = (
    "https://gis.earthdata.nasa.gov/image/rest/services/FireTracking/"
    "Fire_Events_Data_Suite_Fire_Perimeters_nrt/MapServer"
)
FEDS_PERIMETERS_LAYER_URL = f"{FEDS_PERIMETERS_MAPSERVER_URL}/0"
FEDS_PERIMETERS_QUERY_URL = f"{FEDS_PERIMETERS_LAYER_URL}/query"
FEDS_PERIMETERS_LAYER_NAME = "veda.public.eis_fire_lf_perimeter_nrt"
FEDS_NORMALIZATION_VERSION = "feds-nrt-perimeters/v2-primarykey-time"
FEDS_NORMALIZATION_PARTITION = "v2-primarykey-time"
FEDS_QUERY_COVERAGE_VERSION = "v2-primarykey-time"
FEDS_OBSERVED_SNAPSHOT_PRODUCT = "feds-nrt-observed-primarykey-snapshots"
FEDS_LABEL_QUALITY_SCORE = 0.45
FEDS_RETENTION_PRIORITY_SCORE = 90
FEDS_SNAPSHOT_INTERVAL = timedelta(hours=12)
FEDS_SOURCE_TIME_SEMANTICS = "local-solar-time-wall-clock-with-utc-date/v1"
DEFAULT_REGION_NAMES = ("CONUS", "Canada")
DEFAULT_REGION_LABEL = "CONUS+Canada"
DEFAULT_PAGE_SIZE = 2_000
DEFAULT_SNAPSHOT_WINDOWS_PER_REQUEST = 14
_SAFE_REGION_NAME = re.compile(r"^[A-Za-z0-9 _-]+$")


@dataclass(frozen=True)
class FedsPageCollection:
    """One durable FEDS response page and its normalized records."""

    page_number: int
    raw_artifact: RawArtifact
    normalized_artifact: NormalizedArtifact | None
    feature_count: int


@dataclass(frozen=True)
class FedsWindowCollection:
    """All results for one source-aligned 12-hour snapshot window."""

    snapshot_start: datetime
    snapshot_end: datetime
    pages: tuple[FedsPageCollection, ...]
    coverage: CoverageRecord
    skipped_terminal_coverage: bool = False

    @property
    def feature_count(self) -> int:
        return sum(page.feature_count for page in self.pages)


@dataclass(frozen=True)
class FedsRangeCollection:
    """Durable outcome of a FEDS range collection attempt."""

    metadata_artifact: RawArtifact | None
    windows: tuple[FedsWindowCollection, ...]
    coverage: CoverageRecord
    skipped_terminal_coverage: bool = False

    @property
    def feature_count(self) -> int:
        return sum(window.feature_count for window in self.windows)


@dataclass(frozen=True)
class FedsNormalizationRebuildReport:
    """Outcome of rebuilding primary-key snapshots from immutable raw pages."""

    snapshot_count: int
    feature_count: int
    duplicate_record_count: int
    conflicting_record_count: int
    invalid_record_count: int
    raw_artifact_count: int
    normalized_artifact_count: int
    status: CoverageStatus
    selected_capture_at: str | None


def iter_feds_snapshot_windows(start_date: date, end_date: date) -> tuple[tuple[datetime, datetime], ...]:
    """Return every 00:00/12:00 source interval for inclusive UTC dates."""
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    windows = []
    current = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    final = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)
    while current < final:
        windows.append((current, current + FEDS_SNAPSHOT_INTERVAL))
        current += FEDS_SNAPSHOT_INTERVAL
    return tuple(windows)


def feds_query_parameters(
    snapshot_start: datetime,
    snapshot_end: datetime,
    *,
    offset: int = 0,
    page_size: int = DEFAULT_PAGE_SIZE,
    region_names: Iterable[str] = DEFAULT_REGION_NAMES,
) -> dict[str, str | int | bool]:
    """Build one source-aligned ArcGIS JSON query page.

    The current FEDS MapServer advertises GeoJSON but returns HTTP 400 for
    valid GeoJSON requests.  ArcGIS JSON is reliable and preserves the exact
    provider geometry; normalized records therefore retain ``rings`` rather
    than pretending this source returned GeoJSON.
    """
    start = _as_utc(snapshot_start, "snapshot_start")
    end = _as_utc(snapshot_end, "snapshot_end")
    duration = end - start
    if duration <= timedelta(0) or duration.total_seconds() % FEDS_SNAPSHOT_INTERVAL.total_seconds():
        raise ValueError("FEDS query ranges must contain an exact positive number of 12-hour windows")
    if offset < 0:
        raise ValueError("offset must not be negative")
    if page_size <= 0 or page_size > DEFAULT_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {DEFAULT_PAGE_SIZE}")
    regions = _validated_region_names(region_names)
    return {
        "where": _region_where_clause(regions),
        "outFields": "*",
        "returnGeometry": "true",
        "resultOffset": offset,
        "resultRecordCount": page_size,
        # ArcGIS time ranges are inclusive.  The final millisecond makes this
        # a half-open 12-hour window for the source's displayed wall-clock t.
        "time": f"{_epoch_milliseconds(start)},{_epoch_milliseconds(end) - 1}",
        "timeRelation": "esriTimeRelationOverlaps",
        "f": "json",
    }


def collect_feds_perimeters(
    archive_root: str,
    *,
    start_date: date,
    end_date: date,
    storage_budget: StorageBudgetPolicy,
    region_names: Iterable[str] = DEFAULT_REGION_NAMES,
    region_label: str = DEFAULT_REGION_LABEL,
    page_size: int = DEFAULT_PAGE_SIZE,
    snapshot_windows_per_request: int = DEFAULT_SNAPSHOT_WINDOWS_PER_REQUEST,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (20, 600),
    retrieved_at: datetime | None = None,
    refresh: bool = False,
) -> FedsRangeCollection:
    """Archive FEDS 12-hour perimeter snapshots for the requested dates.

    A source metadata response is captured first.  Its advertised time extent
    governs coverage status: a requested source window outside that extent is
    recorded as ``partial`` rather than silently treated as a no-fire label.
    Existing terminal 12-hour windows are skipped on rerun, so a transient
    service failure never requires re-downloading successful windows.
    """
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    if not region_label.strip():
        raise ValueError("region_label must be non-empty")
    if page_size <= 0 or page_size > DEFAULT_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {DEFAULT_PAGE_SIZE}")
    if snapshot_windows_per_request <= 0:
        raise ValueError("snapshot_windows_per_request must be positive")
    regions = _validated_region_names(region_names)
    target = target_by_key("feds_nrt")
    retrieved = _utc_now_or_value(retrieved_at)
    ledger = CoverageLedger(archive_root)
    range_expected_id = _range_expected_coverage_id(start_date, end_date, region_label)
    existing = _latest_expected_coverage(ledger).get(range_expected_id)
    if not refresh and _is_terminal(existing):
        return FedsRangeCollection(
            metadata_artifact=None,
            windows=(),
            coverage=existing,
            skipped_terminal_coverage=True,
        )

    owns_session = session is None
    active_session = session or _retrying_session()
    metadata_artifact: RawArtifact | None = None
    try:
        metadata_document, metadata_artifact, metadata_error = _fetch_metadata(
            archive_root,
            storage_budget=storage_budget,
            target=target,
            active_session=active_session,
            timeout=timeout,
            retrieved_at=retrieved,
            start_date=start_date,
            end_date=end_date,
            region_label=region_label,
        )
        if metadata_error is not None:
            coverage = _record_range_coverage(
                archive_root,
                target=target,
                start_date=start_date,
                end_date=end_date,
                region=region_label,
                expected_id=range_expected_id,
                status=CoverageStatus.FAILED if metadata_artifact is not None else CoverageStatus.PARTIAL,
                artifact_ids=[metadata_artifact.raw_artifact_id] if metadata_artifact else [],
                detail={
                    "stage": "layer-metadata",
                    "label_tier": "weak_satellite",
                    "retention_priority_score": FEDS_RETENTION_PRIORITY_SCORE,
                },
                error=metadata_error,
                retrieved_at=retrieved,
            )
            return FedsRangeCollection(metadata_artifact, (), coverage)
        assert metadata_document is not None
        available_extent = _metadata_time_extent(metadata_document)
        requested_windows = iter_feds_snapshot_windows(start_date, end_date)
        # Always use the grouping-aware implementation, even for one-window
        # requests.  FEDS filters its ``t`` field (detection time), whereas a
        # perimeter snapshot's durable timestamp is in ``primarykey``.  A
        # response can therefore contain several primary-key snapshots.
        windows = list(
            _collect_batched_windows(
                archive_root,
                target=target,
                storage_budget=storage_budget,
                requested_windows=requested_windows,
                available_extent=available_extent,
                region_names=regions,
                region_label=region_label,
                page_size=page_size,
                snapshot_windows_per_request=snapshot_windows_per_request,
                active_session=active_session,
                timeout=timeout,
                retrieved_at=retrieved,
                refresh=refresh,
            )
        )
        status = _range_status(windows)
        artifact_ids = [metadata_artifact.raw_artifact_id]
        artifact_ids.extend(
            page.raw_artifact.raw_artifact_id
            for window in windows
            for page in window.pages
        )
        coverage = _record_range_coverage(
            archive_root,
            target=target,
            start_date=start_date,
            end_date=end_date,
            region=region_label,
            expected_id=range_expected_id,
            status=status,
            artifact_ids=artifact_ids,
            detail={
                "window_count": len(windows),
                "feature_count": sum(window.feature_count for window in windows),
                "snapshot_interval_hours": int(FEDS_SNAPSHOT_INTERVAL.total_seconds() / 3_600),
                "snapshot_windows_per_request": snapshot_windows_per_request,
                "available_time_extent": _format_extent(available_extent),
                "region_names": list(regions),
                "label_tier": "weak_satellite",
                "label_quality_score": FEDS_LABEL_QUALITY_SCORE,
                "source_time_semantics": FEDS_SOURCE_TIME_SEMANTICS,
                "alaska_time_alignment_excluded": "Alaska" not in regions,
                "retention_priority_score": FEDS_RETENTION_PRIORITY_SCORE,
            },
            message=(
                "FEDS 12-hour VIIRS/NOAA-20 perimeter snapshots archived as satellite-weak labels."
            ),
            retrieved_at=retrieved,
        )
        return FedsRangeCollection(metadata_artifact, tuple(windows), coverage)
    finally:
        if owns_session:
            active_session.close()


def rebuild_feds_primarykey_normalization(
    archive_root: str | Path,
    *,
    storage_budget: StorageBudgetPolicy,
    start_date: date | None = None,
    end_date: date | None = None,
    region_label: str = DEFAULT_REGION_LABEL,
    generated_at: datetime | None = None,
    captured_at: datetime | None = None,
) -> FedsNormalizationRebuildReport:
    """Rebuild v2 FEDS snapshots from already-retained raw query responses.

    The FEDS MapServer's query ``time`` field selects fire-detection time
    (``t``), not the timestamp of each cumulative perimeter state.  This
    replay therefore selects one coherent retained collection run, groups its
    raw features by the timestamp in their documented ``primarykey``, and
    writes only the requested logical date range.  It needs no network access
    and never uses an absent query result as evidence of an empty snapshot.

    When a provider returns conflicting versions of a primary-key record, the
    raw evidence remains intact, the stable records are retained, and that
    snapshot is marked ``partial``.  The label builder will then exclude it
    rather than selecting a revision silently.
    """
    if start_date is not None and end_date is not None and end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    if not region_label.strip():
        raise ValueError("region_label must be non-empty")

    root = Path(archive_root)
    generated = _utc_now_or_value(generated_at)
    requested_capture_at = _as_utc(captured_at, "captured_at") if captured_at is not None else None
    target = target_by_key("feds_nrt")
    grouped: dict[datetime, dict[str, dict[str, Any]]] = defaultdict(dict)
    raw_ids_by_snapshot: dict[datetime, set[str]] = defaultdict(set)
    conflict_ids_by_snapshot: dict[datetime, set[str]] = defaultdict(set)
    invalid_counts_by_snapshot: dict[datetime, int] = defaultdict(int)
    raw_artifact_ids: set[str] = set()
    duplicate_record_count = 0
    invalid_record_count = 0

    selected_capture_at, raw_manifests = _select_feds_raw_capture(
        root,
        region_label=region_label,
        start_date=start_date,
        end_date=end_date,
        captured_at=requested_capture_at,
    )
    for manifest_path, manifest in raw_manifests:
        artifact = manifest["artifact"]
        provenance = manifest["provenance"]
        raw_artifact_id = artifact["content_sha256"]
        raw_path = root / artifact["relative_path"]
        try:
            with gzip.open(raw_path, "rt", encoding="utf-8") as source:
                document = json.load(source)
            features = _arcgis_feature_document_from_raw(document, context=str(manifest_path))["features"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            # There is no trustworthy source timestamp to attach to a broken
            # raw page, so retain the raw evidence and fail the rebuild rather
            # than treating it as a no-fire response.
            raise ValueError(f"Cannot replay retained FEDS raw page {manifest_path}: {exc}") from exc

        retrieved_at = _manifest_timestamp(manifest.get("retrieved_at"), field="retrieved_at")
        query_start = _manifest_timestamp(
            provenance.get("snapshot_start"),
            field=f"{manifest_path}: provenance.snapshot_start",
        )
        query_end = _manifest_timestamp(
            provenance.get("snapshot_end"),
            field=f"{manifest_path}: provenance.snapshot_end",
        )
        for feature in features:
            attributes = feature.get("attributes") if isinstance(feature, Mapping) else None
            try:
                if not isinstance(attributes, Mapping):
                    raise ValueError("FEDS feature must contain attributes")
                timestamp_text, _timestamp_source = _authoritative_source_timestamp(attributes)
                if timestamp_text is None:
                    raise ValueError("FEDS feature has no primarykey or source timestamp t")
                snapshot_start = _parse_utc_timestamp(timestamp_text, field="FEDS primarykey timestamp")
                if not _snapshot_in_requested_dates(snapshot_start, start_date=start_date, end_date=end_date):
                    continue
                record = _feds_perimeter_record(
                    feature,
                    raw_artifact_id=raw_artifact_id,
                    retrieved_at=retrieved_at,
                    query_snapshot_start=query_start,
                    query_snapshot_end=query_end,
                )
            except ValueError:
                invalid_record_count += 1
                # A malformed feature can usually still disclose its
                # primary-key time.  If so, make that source snapshot unsafe
                # for labels instead of losing the fact that replay was
                # incomplete.
                snapshot_start = _best_effort_feature_snapshot_start(attributes)
                if snapshot_start is not None and _snapshot_in_requested_dates(
                    snapshot_start, start_date=start_date, end_date=end_date
                ):
                    invalid_counts_by_snapshot[snapshot_start] += 1
                    raw_ids_by_snapshot[snapshot_start].add(raw_artifact_id)
                continue

            raw_artifact_ids.add(raw_artifact_id)
            raw_ids_by_snapshot[snapshot_start].add(raw_artifact_id)
            source_record_id = record["source_record_id"]
            semantic_fingerprint = _feds_snapshot_semantic_fingerprint(record)
            existing = grouped[snapshot_start].get(source_record_id)
            if existing is None:
                grouped[snapshot_start][source_record_id] = {
                    "record": record,
                    "semantic_fingerprint": semantic_fingerprint,
                    "raw_artifact_ids": {raw_artifact_id},
                }
                continue
            existing["raw_artifact_ids"].add(raw_artifact_id)
            if existing["semantic_fingerprint"] == semantic_fingerprint:
                duplicate_record_count += 1
                if raw_artifact_id < existing["record"]["raw_artifact_id"]:
                    existing["record"] = record
                continue
            conflict_ids_by_snapshot[snapshot_start].add(source_record_id)

    reports: list[CoverageStatus] = []
    normalized_artifact_count = 0
    feature_count = 0
    conflicting_record_count = sum(len(values) for values in conflict_ids_by_snapshot.values())
    snapshot_starts = sorted(set(grouped) | set(invalid_counts_by_snapshot))
    for snapshot_start in snapshot_starts:
        candidates = grouped.get(snapshot_start, {})
        records = []
        for source_record_id in sorted(candidates):
            candidate = candidates[source_record_id]
            record = dict(candidate["record"])
            record["equivalent_raw_artifact_ids"] = sorted(candidate["raw_artifact_ids"])
            records.append(record)
        source_raw_ids = sorted(raw_ids_by_snapshot[snapshot_start])
        status = (
            CoverageStatus.PARTIAL
            if conflict_ids_by_snapshot[snapshot_start] or invalid_counts_by_snapshot[snapshot_start]
            else CoverageStatus.COMPLETE
        )
        artifact: NormalizedArtifact | None = None
        error: str | None = None
        if records:
            try:
                require_admission(
                    storage_budget,
                    root,
                    category="operational_labels_and_progression",
                    requested_bytes=_conservative_normalized_snapshot_bytes(records),
                )
                artifact = write_normalized_jsonl(
                    root,
                    entity="fire-progression",
                    records=records,
                    partitions={
                        "normalization_version": FEDS_NORMALIZATION_PARTITION,
                        "source": "feds-nrt-perimeters",
                        "snapshot_start": _format_utc(snapshot_start),
                        "snapshot_end": _format_utc(snapshot_start + FEDS_SNAPSHOT_INTERVAL),
                    },
                    raw_artifact_ids=source_raw_ids,
                    transformation_version=FEDS_NORMALIZATION_VERSION,
                    generated_at=generated,
                )
                normalized_artifact_count += 1
                feature_count += len(records)
            except StorageBudgetError as exc:
                status = CoverageStatus.PARTIAL
                error = str(exc)
        else:
            status = CoverageStatus.PARTIAL
            error = "No valid FEDS records could be rebuilt for the observed primary-key snapshot"

        _record_observed_snapshot_coverage(
            root,
            target=target,
            snapshot_start=snapshot_start,
            region=region_label,
            artifact_ids=source_raw_ids,
            status=status,
            detail={
                "rebuild_from_raw": True,
                "selected_raw_capture_at": _format_utc(selected_capture_at),
                "normalization_version": FEDS_NORMALIZATION_VERSION,
                "normalized_artifact_id": artifact.normalized_artifact_id if artifact else None,
                "feature_count": len(records),
                "duplicate_record_count": sum(
                    max(0, len(candidate["raw_artifact_ids"]) - 1)
                    for candidate in candidates.values()
                ),
                "conflicting_source_record_ids": sorted(conflict_ids_by_snapshot[snapshot_start]),
                "invalid_record_count": invalid_counts_by_snapshot[snapshot_start],
                "snapshot_timestamp_source": "primarykey",
                "snapshot_completeness": "not-proven",
                "no_fire_inferred": False,
                "source_time_semantics": FEDS_SOURCE_TIME_SEMANTICS,
                "label_tier": "weak_satellite",
                "retention_priority_score": FEDS_RETENTION_PRIORITY_SCORE,
            },
            error=error,
            retrieved_at=generated,
        )
        reports.append(status)

    overall_status = (
        CoverageStatus.PARTIAL
        if not reports or any(status is not CoverageStatus.COMPLETE for status in reports)
        else CoverageStatus.COMPLETE
    )
    return FedsNormalizationRebuildReport(
        snapshot_count=len(snapshot_starts),
        feature_count=feature_count,
        duplicate_record_count=duplicate_record_count,
        conflicting_record_count=conflicting_record_count,
        invalid_record_count=invalid_record_count,
        raw_artifact_count=len(raw_artifact_ids),
        normalized_artifact_count=normalized_artifact_count,
        status=overall_status,
        selected_capture_at=_format_utc(selected_capture_at) if selected_capture_at else None,
    )


def _fetch_metadata(
    archive_root: str,
    *,
    storage_budget: StorageBudgetPolicy,
    target,
    active_session: requests.Session,
    timeout: tuple[int, int],
    retrieved_at: datetime,
    start_date: date,
    end_date: date,
    region_label: str,
) -> tuple[dict[str, Any] | None, RawArtifact | None, str | None]:
    parameters = {"f": "json"}
    try:
        response = active_session.get(FEDS_PERIMETERS_LAYER_URL, params=parameters, timeout=timeout)
    except requests.RequestException as exc:
        return None, None, f"FEDS metadata request failed: {exc}"
    artifact, admission_error = _admit_and_archive_response(
        archive_root,
        storage_budget=storage_budget,
        target=target,
        response=response,
        parameters=parameters,
        stage="layer-metadata",
        snapshot_start=None,
        snapshot_end=None,
        page_number=None,
        retrieved_at=retrieved_at,
        start_date=start_date,
        end_date=end_date,
        region_label=region_label,
    )
    if admission_error is not None:
        return None, None, admission_error
    assert artifact is not None
    if not 200 <= response.status_code < 300:
        return None, artifact, f"FEDS metadata service returned HTTP {response.status_code}"
    try:
        return _json_document(response.content, context="FEDS metadata"), artifact, None
    except ValueError as exc:
        return None, artifact, str(exc)


def _collect_batched_windows(
    archive_root: str,
    *,
    target,
    storage_budget: StorageBudgetPolicy,
    requested_windows: tuple[tuple[datetime, datetime], ...],
    available_extent: tuple[datetime, datetime] | None,
    region_names: tuple[str, ...],
    region_label: str,
    page_size: int,
    snapshot_windows_per_request: int,
    active_session: requests.Session,
    timeout: tuple[int, int],
    retrieved_at: datetime,
    refresh: bool,
) -> tuple[FedsWindowCollection, ...]:
    """Collect unresolved source windows in fewer, multi-snapshot queries.

    NASA's MapServer has sizeable request setup latency.  A query batch reduces
    that overhead, but the raw response and the normalized records remain
    attributed to their exact 12-hour source snapshot.  This preserves the
    retry/coverage guarantees of the one-window mode.
    """
    ledger = CoverageLedger(archive_root)
    latest = _latest_expected_coverage(ledger)
    resolved: dict[datetime, FedsWindowCollection] = {}
    pending: list[tuple[datetime, datetime]] = []
    for snapshot_start, snapshot_end in requested_windows:
        expected_id = _window_expected_coverage_id(snapshot_start, region_label)
        existing = latest.get(expected_id)
        if not refresh and _is_terminal(existing):
            resolved[snapshot_start] = FedsWindowCollection(
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                pages=(),
                coverage=existing,
                skipped_terminal_coverage=True,
            )
        elif available_extent is not None and not _window_intersects_extent(
            snapshot_start, snapshot_end, available_extent
        ):
            coverage = _record_window_coverage(
                archive_root,
                target=target,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                region=region_label,
                expected_id=expected_id,
                status=CoverageStatus.PARTIAL,
                artifact_ids=[],
                detail={
                    "reason": "requested-source-window-outside-advertised-time-extent",
                    "available_time_extent": _format_extent(available_extent),
                    "label_tier": "weak_satellite",
                },
                retrieved_at=retrieved_at,
            )
            resolved[snapshot_start] = FedsWindowCollection(snapshot_start, snapshot_end, (), coverage)
        else:
            pending.append((snapshot_start, snapshot_end))

    batch: list[tuple[datetime, datetime]] = []
    for window in pending:
        if batch and (
            window[0] != batch[-1][1] or len(batch) >= snapshot_windows_per_request
        ):
            for result in _collect_window_batch(
                archive_root,
                target=target,
                storage_budget=storage_budget,
                windows=tuple(batch),
                region_names=region_names,
                region_label=region_label,
                page_size=page_size,
                active_session=active_session,
                timeout=timeout,
                retrieved_at=retrieved_at,
            ):
                resolved[result.snapshot_start] = result
            batch = []
        batch.append(window)
    if batch:
        for result in _collect_window_batch(
            archive_root,
            target=target,
            storage_budget=storage_budget,
            windows=tuple(batch),
            region_names=region_names,
            region_label=region_label,
            page_size=page_size,
            active_session=active_session,
            timeout=timeout,
            retrieved_at=retrieved_at,
        ):
            resolved[result.snapshot_start] = result
    return tuple(resolved[start] for start, _end in requested_windows)


def _collect_window_batch(
    archive_root: str,
    *,
    target,
    storage_budget: StorageBudgetPolicy,
    windows: tuple[tuple[datetime, datetime], ...],
    region_names: tuple[str, ...],
    region_label: str,
    page_size: int,
    active_session: requests.Session,
    timeout: tuple[int, int],
    retrieved_at: datetime,
) -> tuple[FedsWindowCollection, ...]:
    """Fetch one contiguous batch and retain per-snapshot normalized outputs."""
    if not windows:
        return ()
    query_start = windows[0][0]
    query_end = windows[-1][1]
    pages_by_start: dict[datetime, list[FedsPageCollection]] = defaultdict(list)
    artifact_ids_by_start: dict[datetime, list[str]] = defaultdict(list)
    all_artifact_ids: list[str] = []
    offset = 0
    page_number = 1
    while True:
        parameters = feds_query_parameters(
            query_start,
            query_end,
            offset=offset,
            page_size=page_size,
            region_names=region_names,
        )
        try:
            response = active_session.get(FEDS_PERIMETERS_QUERY_URL, params=parameters, timeout=timeout)
        except requests.RequestException as exc:
            return _failed_batch_windows(
                archive_root,
                target=target,
                windows=windows,
                region_label=region_label,
                pages_by_start=pages_by_start,
                artifact_ids_by_start=artifact_ids_by_start,
                error=f"FEDS perimeter request failed: {exc}",
                retrieved_at=retrieved_at,
            )
        artifact, admission_error = _admit_and_archive_response(
            archive_root,
            storage_budget=storage_budget,
            target=target,
            response=response,
            parameters=parameters,
            stage="query-batch-page",
            snapshot_start=query_start,
            snapshot_end=query_end,
            page_number=page_number,
            retrieved_at=retrieved_at,
            start_date=query_start.date(),
            end_date=(query_end - timedelta(microseconds=1)).date(),
            region_label=region_label,
        )
        if admission_error is not None:
            return _partial_batch_windows(
                archive_root,
                target=target,
                windows=windows,
                region_label=region_label,
                pages_by_start=pages_by_start,
                artifact_ids_by_start=artifact_ids_by_start,
                error=admission_error,
                retrieved_at=retrieved_at,
            )
        assert artifact is not None
        all_artifact_ids.append(artifact.raw_artifact_id)
        if not 200 <= response.status_code < 300:
            return _failed_batch_windows(
                archive_root,
                target=target,
                windows=windows,
                region_label=region_label,
                pages_by_start=pages_by_start,
                artifact_ids_by_start=artifact_ids_by_start,
                error=f"FEDS perimeter service returned HTTP {response.status_code}",
                retrieved_at=retrieved_at,
            )
        try:
            document = _arcgis_feature_document(response.content)
            groups = _features_by_snapshot_start(document["features"])
        except ValueError as exc:
            return _failed_batch_windows(
                archive_root,
                target=target,
                windows=windows,
                region_label=region_label,
                pages_by_start=pages_by_start,
                artifact_ids_by_start=artifact_ids_by_start,
                error=str(exc),
                retrieved_at=retrieved_at,
            )
        # ArcGIS applies the query's ``time`` interval to the provider's
        # detection-time field ``t``.  It does *not* limit the primary-key
        # timestamp of the cumulative perimeter states that accompany that
        # detection.  Retain and normalize every returned primary-key group;
        # rejecting those additional snapshots would make the raw archive
        # unusable and would incorrectly turn an API detail into a failure.
        for snapshot_start, features in groups.items():
            snapshot_end = snapshot_start + FEDS_SNAPSHOT_INTERVAL
            normalized_artifact = write_normalized_jsonl(
                archive_root,
                entity="fire-progression",
                records=(
                    _feds_perimeter_record(
                        feature,
                        raw_artifact_id=artifact.raw_artifact_id,
                        retrieved_at=retrieved_at,
                        query_snapshot_start=query_start,
                        query_snapshot_end=query_end,
                    )
                    for feature in features
                ),
                partitions={
                    "normalization_version": FEDS_NORMALIZATION_PARTITION,
                    "source": "feds-nrt-perimeters",
                    "snapshot_start": _format_utc(snapshot_start),
                    "snapshot_end": _format_utc(snapshot_end),
                    "page": str(page_number),
                },
                raw_artifact_ids=[artifact.raw_artifact_id],
                transformation_version=FEDS_NORMALIZATION_VERSION,
                generated_at=retrieved_at,
            )
            pages_by_start[snapshot_start].append(
                FedsPageCollection(
                    page_number=page_number,
                    raw_artifact=artifact,
                    normalized_artifact=normalized_artifact,
                    feature_count=len(features),
                )
            )
            artifact_ids_by_start[snapshot_start].append(artifact.raw_artifact_id)
        feature_count = len(document["features"])
        if not _has_next_page(document, feature_count=feature_count, page_size=page_size):
            break
        if not feature_count:
            return _failed_batch_windows(
                archive_root,
                target=target,
                windows=windows,
                region_label=region_label,
                pages_by_start=pages_by_start,
                artifact_ids_by_start=artifact_ids_by_start,
                error="FEDS returned an empty page before its transfer limit was cleared",
                retrieved_at=retrieved_at,
            )
        offset += feature_count
        page_number += 1

    # This is the only ledger entry that asserts a usable source snapshot.
    # A query response lacking a given primary-key timestamp cannot establish
    # that the snapshot was empty, because ``time`` filters detection time.
    for observed_start, pages in pages_by_start.items():
        _record_observed_snapshot_coverage(
            archive_root,
            target=target,
            snapshot_start=observed_start,
            region=region_label,
            artifact_ids=artifact_ids_by_start[observed_start],
            detail={
                "page_count": len(pages),
                "feature_count": sum(page.feature_count for page in pages),
                "query_batch_start": _format_utc(query_start),
                "query_batch_end": _format_utc(query_end),
                "source_time_semantics": FEDS_SOURCE_TIME_SEMANTICS,
                "snapshot_timestamp_source": "primarykey",
                "label_tier": "weak_satellite",
                "retention_priority_score": FEDS_RETENTION_PRIORITY_SCORE,
            },
            retrieved_at=retrieved_at,
        )

    results = []
    for snapshot_start, snapshot_end in windows:
        pages = tuple(pages_by_start[snapshot_start])
        feature_count = sum(page.feature_count for page in pages)
        observed = feature_count > 0
        coverage = _record_window_coverage(
            archive_root,
            target=target,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            region=region_label,
            expected_id=_window_expected_coverage_id(snapshot_start, region_label),
            # This coverage represents whether an observed primary-key group
            # was persisted.  A missing group is unknown, never a confirmed
            # no-fire snapshot, because the API filtered detection time ``t``.
            status=CoverageStatus.COMPLETE if observed else CoverageStatus.PARTIAL,
            artifact_ids=artifact_ids_by_start[snapshot_start] or all_artifact_ids,
            detail={
                "page_count": len(pages),
                "matching_primarykey_feature_count": feature_count,
                "query_batch_start": _format_utc(query_start),
                "query_batch_end": _format_utc(query_end),
                "coverage_semantics": "observed-primarykey-group/v2",
                "source_snapshot_observability": "use-observed-primarykey-ledger",
                "no_fire_inferred": False,
                "label_tier": "weak_satellite",
                "label_quality_score": FEDS_LABEL_QUALITY_SCORE,
                "source_time_semantics": FEDS_SOURCE_TIME_SEMANTICS,
                "retention_priority_score": FEDS_RETENTION_PRIORITY_SCORE,
            },
            retrieved_at=retrieved_at,
        )
        results.append(FedsWindowCollection(snapshot_start, snapshot_end, pages, coverage))
    return tuple(results)


def _features_by_snapshot_start(
    features: Iterable[Mapping[str, Any]],
) -> dict[datetime, list[Mapping[str, Any]]]:
    groups: dict[datetime, list[Mapping[str, Any]]] = defaultdict(list)
    for feature in features:
        attributes = feature.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError("FEDS feature must contain attributes")
        timestamp_text, _timestamp_source = _authoritative_source_timestamp(attributes)
        if timestamp_text is None:
            raise ValueError("FEDS feature has no primarykey or source timestamp t")
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00")).astimezone(timezone.utc)
        groups[timestamp].append(feature)
    return groups


def _failed_batch_windows(
    archive_root: str,
    *,
    target,
    windows: tuple[tuple[datetime, datetime], ...],
    region_label: str,
    pages_by_start: Mapping[datetime, list[FedsPageCollection]],
    artifact_ids_by_start: Mapping[datetime, list[str]],
    error: str,
    retrieved_at: datetime,
) -> tuple[FedsWindowCollection, ...]:
    return tuple(
        FedsWindowCollection(
            snapshot_start,
            snapshot_end,
            tuple(pages_by_start.get(snapshot_start, [])),
            _record_window_coverage(
                archive_root,
                target=target,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                region=region_label,
                expected_id=_window_expected_coverage_id(snapshot_start, region_label),
                status=CoverageStatus.FAILED,
                artifact_ids=list(artifact_ids_by_start.get(snapshot_start, [])),
                detail={"page_count": len(pages_by_start.get(snapshot_start, [])), "query_batch": True},
                error=error,
                retrieved_at=retrieved_at,
            ),
        )
        for snapshot_start, snapshot_end in windows
    )


def _partial_batch_windows(
    archive_root: str,
    *,
    target,
    windows: tuple[tuple[datetime, datetime], ...],
    region_label: str,
    pages_by_start: Mapping[datetime, list[FedsPageCollection]],
    artifact_ids_by_start: Mapping[datetime, list[str]],
    error: str,
    retrieved_at: datetime,
) -> tuple[FedsWindowCollection, ...]:
    return tuple(
        FedsWindowCollection(
            snapshot_start,
            snapshot_end,
            tuple(pages_by_start.get(snapshot_start, [])),
            _record_window_coverage(
                archive_root,
                target=target,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                region=region_label,
                expected_id=_window_expected_coverage_id(snapshot_start, region_label),
                status=CoverageStatus.PARTIAL,
                artifact_ids=list(artifact_ids_by_start.get(snapshot_start, [])),
                detail={"page_count": len(pages_by_start.get(snapshot_start, [])), "query_batch": True},
                error=error,
                retrieved_at=retrieved_at,
            ),
        )
        for snapshot_start, snapshot_end in windows
    )


def _collect_window(
    archive_root: str,
    *,
    target,
    storage_budget: StorageBudgetPolicy,
    snapshot_start: datetime,
    snapshot_end: datetime,
    available_extent: tuple[datetime, datetime] | None,
    region_names: tuple[str, ...],
    region_label: str,
    page_size: int,
    active_session: requests.Session,
    timeout: tuple[int, int],
    retrieved_at: datetime,
    refresh: bool,
) -> FedsWindowCollection:
    expected_id = _window_expected_coverage_id(snapshot_start, region_label)
    ledger = CoverageLedger(archive_root)
    existing = _latest_expected_coverage(ledger).get(expected_id)
    if not refresh and _is_terminal(existing):
        return FedsWindowCollection(
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            pages=(),
            coverage=existing,
            skipped_terminal_coverage=True,
        )
    if available_extent is not None and not _window_intersects_extent(
        snapshot_start, snapshot_end, available_extent
    ):
        coverage = _record_window_coverage(
            archive_root,
            target=target,
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            region=region_label,
            expected_id=expected_id,
            status=CoverageStatus.PARTIAL,
            artifact_ids=[],
            detail={
                "reason": "requested-source-window-outside-advertised-time-extent",
                "available_time_extent": _format_extent(available_extent),
                "label_tier": "weak_satellite",
            },
            retrieved_at=retrieved_at,
        )
        return FedsWindowCollection(snapshot_start, snapshot_end, (), coverage)

    pages: list[FedsPageCollection] = []
    artifact_ids: list[str] = []
    offset = 0
    page_number = 1
    while True:
        parameters = feds_query_parameters(
            snapshot_start,
            snapshot_end,
            offset=offset,
            page_size=page_size,
            region_names=region_names,
        )
        try:
            response = active_session.get(FEDS_PERIMETERS_QUERY_URL, params=parameters, timeout=timeout)
        except requests.RequestException as exc:
            return _failed_window(
                archive_root,
                target=target,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                region=region_label,
                expected_id=expected_id,
                pages=pages,
                artifact_ids=artifact_ids,
                page_number=page_number,
                error=f"FEDS perimeter request failed: {exc}",
                retrieved_at=retrieved_at,
            )
        artifact, admission_error = _admit_and_archive_response(
            archive_root,
            storage_budget=storage_budget,
            target=target,
            response=response,
            parameters=parameters,
            stage="query-page",
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
            page_number=page_number,
            retrieved_at=retrieved_at,
            start_date=snapshot_start.date(),
            end_date=(snapshot_end - timedelta(microseconds=1)).date(),
            region_label=region_label,
        )
        if admission_error is not None:
            coverage = _record_window_coverage(
                archive_root,
                target=target,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                region=region_label,
                expected_id=expected_id,
                status=CoverageStatus.PARTIAL,
                artifact_ids=artifact_ids,
                detail={
                    "page_count": len(pages),
                    "capped_page_number": page_number,
                    "retention_priority_score": FEDS_RETENTION_PRIORITY_SCORE,
                },
                error=admission_error,
                retrieved_at=retrieved_at,
            )
            return FedsWindowCollection(snapshot_start, snapshot_end, tuple(pages), coverage)
        assert artifact is not None
        artifact_ids.append(artifact.raw_artifact_id)
        if not 200 <= response.status_code < 300:
            return _failed_window(
                archive_root,
                target=target,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                region=region_label,
                expected_id=expected_id,
                pages=pages,
                artifact_ids=artifact_ids,
                page_number=page_number,
                error=f"FEDS perimeter service returned HTTP {response.status_code}",
                retrieved_at=retrieved_at,
            )
        try:
            document = _arcgis_feature_document(response.content)
        except ValueError as exc:
            return _failed_window(
                archive_root,
                target=target,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                region=region_label,
                expected_id=expected_id,
                pages=pages,
                artifact_ids=artifact_ids,
                page_number=page_number,
                error=str(exc),
                retrieved_at=retrieved_at,
            )
        features = document["features"]
        normalized_artifact = (
            write_normalized_jsonl(
                archive_root,
                entity="fire-progression",
                records=(
                    _feds_perimeter_record(
                        feature,
                        raw_artifact_id=artifact.raw_artifact_id,
                        retrieved_at=retrieved_at,
                        query_snapshot_start=snapshot_start,
                        query_snapshot_end=snapshot_end,
                    )
                    for feature in features
                ),
                partitions={
                    "normalization_version": FEDS_NORMALIZATION_PARTITION,
                    "source": "feds-nrt-perimeters",
                    "snapshot_start": _format_utc(snapshot_start),
                    "snapshot_end": _format_utc(snapshot_end),
                    "page": str(page_number),
                },
                raw_artifact_ids=[artifact.raw_artifact_id],
                transformation_version=FEDS_NORMALIZATION_VERSION,
                generated_at=retrieved_at,
            )
            if features
            else None
        )
        pages.append(
            FedsPageCollection(
                page_number=page_number,
                raw_artifact=artifact,
                normalized_artifact=normalized_artifact,
                feature_count=len(features),
            )
        )
        if not _has_next_page(document, feature_count=len(features), page_size=page_size):
            break
        if not features:
            return _failed_window(
                archive_root,
                target=target,
                snapshot_start=snapshot_start,
                snapshot_end=snapshot_end,
                region=region_label,
                expected_id=expected_id,
                pages=pages,
                artifact_ids=artifact_ids,
                page_number=page_number,
                error="FEDS returned an empty page before its transfer limit was cleared",
                retrieved_at=retrieved_at,
            )
        offset += len(features)
        page_number += 1

    feature_count = sum(page.feature_count for page in pages)
    coverage = _record_window_coverage(
        archive_root,
        target=target,
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
        region=region_label,
        expected_id=expected_id,
        status=CoverageStatus.COMPLETE if feature_count else CoverageStatus.EMPTY_CONFIRMED,
        artifact_ids=artifact_ids,
        detail={
            "page_count": len(pages),
            "feature_count": feature_count,
            "label_tier": "weak_satellite",
            "label_quality_score": FEDS_LABEL_QUALITY_SCORE,
            "source_time_semantics": FEDS_SOURCE_TIME_SEMANTICS,
            "retention_priority_score": FEDS_RETENTION_PRIORITY_SCORE,
        },
        retrieved_at=retrieved_at,
    )
    return FedsWindowCollection(snapshot_start, snapshot_end, tuple(pages), coverage)


def _admit_and_archive_response(
    archive_root: str,
    *,
    storage_budget: StorageBudgetPolicy,
    target,
    response,
    parameters: Mapping[str, Any],
    stage: str,
    snapshot_start: datetime | None,
    snapshot_end: datetime | None,
    page_number: int | None,
    retrieved_at: datetime,
    start_date: date,
    end_date: date,
    region_label: str,
) -> tuple[RawArtifact | None, str | None]:
    estimated_bytes = _conservative_response_bytes(response.content)
    try:
        require_admission(
            storage_budget,
            archive_root,
            category="operational_labels_and_progression",
            requested_bytes=estimated_bytes,
        )
    except StorageBudgetError as exc:
        return None, str(exc)
    headers = dict(getattr(response, "headers", {}))
    content_type = headers.get("Content-Type")
    media_type = (
        content_type.split(";", maxsplit=1)[0].strip()
        if isinstance(content_type, str) and content_type.strip()
        else "application/json"
    )
    artifact = write_raw_artifact(
        archive_root,
        source="NASA FEDS",
        payload=response.content,
        retrieved_at=retrieved_at,
        media_type=media_type,
        provenance={
            "source_url": getattr(response, "url", FEDS_PERIMETERS_QUERY_URL),
            "request_parameters": dict(parameters),
            "response_headers": headers,
            "response_status_code": response.status_code,
            "target_key": target.key,
            "source_layer": FEDS_PERIMETERS_LAYER_NAME,
            "source_service": "Fire_Events_Data_Suite_Fire_Perimeters_nrt",
            "stage": stage,
            "coverage_start": start_date,
            "coverage_end": end_date,
            "region": region_label,
            "snapshot_start": _format_utc(snapshot_start) if snapshot_start else None,
            "snapshot_end": _format_utc(snapshot_end) if snapshot_end else None,
            "page_number": page_number,
            "label_tier": "weak_satellite",
            "source_time_semantics": FEDS_SOURCE_TIME_SEMANTICS,
            "retention_priority_score": FEDS_RETENTION_PRIORITY_SCORE,
        },
    )
    return artifact, None


def _feds_perimeter_record(
    feature: Mapping[str, Any],
    *,
    raw_artifact_id: str,
    retrieved_at: datetime,
    query_snapshot_start: datetime,
    query_snapshot_end: datetime,
) -> dict[str, Any]:
    attributes = feature.get("attributes")
    geometry = feature.get("geometry")
    if not isinstance(attributes, Mapping) or not isinstance(geometry, Mapping):
        raise ValueError("FEDS feature must contain attributes and an ArcGIS geometry")
    rings = geometry.get("rings")
    if not isinstance(rings, list) or not rings:
        raise ValueError("FEDS perimeter feature has no ArcGIS polygon rings")
    region = _optional_text(attributes.get("region"))
    primary_key = _optional_text(attributes.get("primarykey"))
    source_snapshot_time, timestamp_source = _authoritative_source_timestamp(attributes)
    provider_t_time = _source_timestamp(attributes.get("t"))
    source_record_id = primary_key or _fallback_source_record_id(attributes, source_snapshot_time)
    return {
        "record_type": "feds_perimeter_snapshot",
        "source": "NASA FEDS",
        "source_service": "Fire_Events_Data_Suite_Fire_Perimeters_nrt",
        "source_layer": FEDS_PERIMETERS_LAYER_NAME,
        "record_role": "weak_satellite_perimeter_snapshot",
        "label_tier": "weak_satellite",
        "label_quality_score": FEDS_LABEL_QUALITY_SCORE,
        "retention_priority_score": FEDS_RETENTION_PRIORITY_SCORE,
        "source_record_id": source_record_id,
        "fire_id": _json_safe(attributes.get("fireid")),
        "region": region,
        "source_snapshot_time": source_snapshot_time,
        "source_snapshot_time_source": timestamp_source,
        "primarykey_snapshot_time": _primarykey_timestamp(primary_key),
        "provider_t_epoch_ms": _json_safe(attributes.get("t")),
        "provider_t_snapshot_time": provider_t_time,
        "provider_t_matches_primarykey": (
            provider_t_time == source_snapshot_time if provider_t_time and source_snapshot_time else None
        ),
        "source_time_semantics": FEDS_SOURCE_TIME_SEMANTICS,
        "time_alignment_eligible": region in DEFAULT_REGION_NAMES,
        "query_snapshot_start": _format_utc(query_snapshot_start),
        "query_snapshot_end": _format_utc(query_snapshot_end),
        "retrieved_at": _format_utc(retrieved_at),
        "raw_artifact_id": raw_artifact_id,
        "geometry": {
            "encoding": "esri-rings-wgs84/v1",
            "spatial_reference": "EPSG:4326",
            "rings": _json_safe(rings),
        },
        "source_fields": _json_safe(dict(attributes)),
    }


def _fallback_source_record_id(attributes: Mapping[str, Any], source_snapshot_time: str | None) -> str:
    values = (
        _optional_text(attributes.get("region")) or "unknown-region",
        str(_json_safe(attributes.get("fireid"))),
        source_snapshot_time or str(_json_safe(attributes.get("ESRI_OID"))),
    )
    return "|".join(values)


def _authoritative_source_timestamp(attributes: Mapping[str, Any]) -> tuple[str | None, str]:
    """Use the documented timestamp in FEDS' globally unique primary key.

    The current MapServer can return a query-window ``t`` value for multiple
    features whose ``primarykey`` timestamps differ.  NASA documents the
    primary key as ``region|fireid|timestamp``; it is therefore the durable
    identity/timestamp for grouping source snapshots.  Preserve provider ``t``
    separately for audit rather than silently discarding the discrepancy.
    """
    primary_time = _primarykey_timestamp(_optional_text(attributes.get("primarykey")))
    if primary_time is not None:
        return primary_time, "primarykey"
    return _source_timestamp(attributes.get("t")), "provider-t-fallback"


def _primarykey_timestamp(primary_key: str | None) -> str | None:
    if primary_key is None:
        return None
    parts = primary_key.rsplit("|", maxsplit=1)
    if len(parts) != 2 or not parts[1].strip():
        raise ValueError("FEDS primarykey does not contain its documented timestamp")
    raw_timestamp = parts[1].strip()
    try:
        parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("FEDS primarykey timestamp is not ISO-8601") from exc
    # The source explicitly defines this as a local-solar wall clock.  Attach
    # UTC only as a nominal serialization; the semantic field stays explicit.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return _format_utc(parsed)


def _source_timestamp(value: object) -> str | None:
    if value is None or value == "":
        return None
    try:
        milliseconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("FEDS feature timestamp t must be epoch milliseconds") from exc
    if not math.isfinite(milliseconds):
        raise ValueError("FEDS feature timestamp t must be finite")
    return _format_utc(datetime.fromtimestamp(milliseconds / 1_000, tz=timezone.utc))


def _arcgis_feature_document(payload: bytes) -> dict[str, Any]:
    document = _json_document(payload, context="FEDS query")
    return _arcgis_feature_document_from_raw(document, context="FEDS query")


def _arcgis_feature_document_from_raw(document: object, *, context: str) -> dict[str, Any]:
    """Validate a decoded FEDS ArcGIS page for live and replayed inputs."""
    if not isinstance(document, dict):
        raise ValueError(f"{context} response is not a JSON object")
    if "error" in document:
        raise ValueError(f"FEDS query returned an error: {_json_safe(document['error'])}")
    features = document.get("features")
    if not isinstance(features, list):
        raise ValueError(f"{context} response has no feature list")
    if not all(isinstance(feature, Mapping) for feature in features):
        raise ValueError(f"{context} response contains an invalid feature")
    return document


def _select_feds_raw_capture(
    root: Path,
    *,
    region_label: str,
    start_date: date | None,
    end_date: date | None,
    captured_at: datetime | None,
) -> tuple[datetime | None, tuple[tuple[Path, dict[str, Any]], ...]]:
    """Choose one complete-looking raw FEDS collection run for replay.

    NRT perimeters can legitimately be revised between retrievals.  Merging
    two collection runs would manufacture duplicate primary keys and require
    an arbitrary revision choice.  The largest requested-query coverage wins;
    capture time breaks ties in favour of the newest coherent run.
    """
    manifest_root = root / "manifests" / "raw" / "nasa-feds"
    captures: dict[datetime, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for manifest_path in sorted(manifest_root.rglob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid FEDS raw manifest {manifest_path}: {exc}") from exc
        if manifest.get("kind") != "raw-artifact" or manifest.get("source") != "NASA FEDS":
            continue
        provenance = manifest.get("provenance")
        artifact = manifest.get("artifact")
        if not isinstance(provenance, Mapping) or not isinstance(artifact, Mapping):
            continue
        if provenance.get("stage") not in {"query-page", "query-batch-page"}:
            continue
        manifest_region = provenance.get("region")
        if manifest_region is not None and manifest_region != region_label:
            continue
        raw_artifact_id = artifact.get("content_sha256")
        relative_path = artifact.get("relative_path")
        if not isinstance(raw_artifact_id, str) or not re.fullmatch(r"[0-9a-f]{64}", raw_artifact_id):
            raise ValueError(f"FEDS raw manifest has an invalid content SHA-256: {manifest_path}")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"FEDS raw manifest has no artifact path: {manifest_path}")
        capture_at = _manifest_timestamp(manifest.get("retrieved_at"), field="retrieved_at")
        captures[capture_at].append((manifest_path, {
            "artifact": dict(artifact),
            "provenance": dict(provenance),
            "retrieved_at": manifest.get("retrieved_at"),
        }))
    if not captures:
        return None, ()

    def score(item: tuple[datetime, list[tuple[Path, dict[str, Any]]]]) -> tuple[int, datetime]:
        capture_at, manifests = item
        query_windows = set()
        for _manifest_path, manifest in manifests:
            provenance = manifest["provenance"]
            start = _manifest_timestamp(provenance.get("snapshot_start"), field="snapshot_start")
            end = _manifest_timestamp(provenance.get("snapshot_end"), field="snapshot_end")
            if end <= start:
                raise ValueError("FEDS raw manifest query window has a non-positive duration")
            if _query_window_intersects_requested_dates(
                start,
                end,
                start_date=start_date,
                end_date=end_date,
            ):
                query_windows.add((start, end))
        return len(query_windows), capture_at

    if captured_at is not None:
        selected_capture_at = _as_utc(captured_at, "captured_at")
        selected = captures.get(selected_capture_at)
        if selected is None:
            available = ", ".join(_format_utc(value) for value in sorted(captures))
            raise ValueError(
                f"No retained FEDS query capture exists at {_format_utc(selected_capture_at)}; "
                f"available captures: {available}"
            )
    else:
        selected_capture_at, selected = max(captures.items(), key=score)
    unique: dict[str, tuple[Path, dict[str, Any]]] = {}
    for manifest_path, manifest in selected:
        raw_artifact_id = manifest["artifact"]["content_sha256"]
        # A repeated response in the same collection run contains identical
        # evidence.  Keep its lexically first manifest so replay stays stable.
        unique.setdefault(raw_artifact_id, (manifest_path, manifest))
    return selected_capture_at, tuple(unique[key] for key in sorted(unique))


def _query_window_intersects_requested_dates(
    query_start: datetime,
    query_end: datetime,
    *,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    if start_date is None and end_date is None:
        return True
    if start_date is not None:
        start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
        if query_end <= start:
            return False
    if end_date is not None:
        end = datetime.combine(end_date, time.min, tzinfo=timezone.utc) + timedelta(days=1)
        if query_start >= end:
            return False
    return True


def _manifest_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"FEDS raw manifest has no {field}")
    return _parse_utc_timestamp(value, field=field)


def _parse_utc_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _snapshot_in_requested_dates(
    snapshot_start: datetime,
    *,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    value = _as_utc(snapshot_start, "snapshot_start").date()
    return (start_date is None or value >= start_date) and (end_date is None or value <= end_date)


def _best_effort_feature_snapshot_start(attributes: Mapping[str, Any] | None) -> datetime | None:
    if attributes is None:
        return None
    try:
        timestamp_text, _timestamp_source = _authoritative_source_timestamp(attributes)
        return _parse_utc_timestamp(timestamp_text, field="FEDS primarykey timestamp") if timestamp_text else None
    except ValueError:
        return None


def _feds_snapshot_semantic_fingerprint(record: Mapping[str, Any]) -> str:
    """Compare provider state while intentionally ignoring query-time ``t``.

    The same FEDS perimeter state can be returned from different detection-time
    queries, whose provider ``t`` and query provenance differ by design.  Those
    copies are equivalent evidence, not conflicting source revisions.
    """
    source_fields = record.get("source_fields")
    filtered_fields = dict(source_fields) if isinstance(source_fields, Mapping) else {}
    filtered_fields.pop("t", None)
    filtered_fields.pop("ESRI_OID", None)
    value = {
        "source_record_id": record.get("source_record_id"),
        "fire_id": record.get("fire_id"),
        "region": record.get("region"),
        "source_snapshot_time": record.get("source_snapshot_time"),
        "geometry": record.get("geometry"),
        "source_fields_without_t": filtered_fields,
    }
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_document(payload: bytes, *, context: str) -> dict[str, Any]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} response is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{context} response is not a JSON object")
    return document


def _metadata_time_extent(document: Mapping[str, Any]) -> tuple[datetime, datetime] | None:
    time_info = document.get("timeInfo")
    if not isinstance(time_info, Mapping):
        return None
    extent = time_info.get("timeExtent")
    if not isinstance(extent, list) or len(extent) != 2:
        return None
    try:
        start = datetime.fromtimestamp(float(extent[0]) / 1_000, tz=timezone.utc)
        end = datetime.fromtimestamp(float(extent[1]) / 1_000, tz=timezone.utc)
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError("FEDS metadata time extent is invalid") from exc
    if end < start:
        raise ValueError("FEDS metadata time extent ends before it starts")
    return start, end


def _window_intersects_extent(
    snapshot_start: datetime,
    snapshot_end: datetime,
    extent: tuple[datetime, datetime],
) -> bool:
    available_start, available_end = extent
    # The source end is an observed snapshot instant, not an exclusive bound.
    return snapshot_end > available_start and snapshot_start <= available_end


def _has_next_page(document: Mapping[str, Any], *, feature_count: int, page_size: int) -> bool:
    return bool(document.get("exceededTransferLimit")) or feature_count >= page_size


def _failed_window(
    archive_root: str,
    *,
    target,
    snapshot_start: datetime,
    snapshot_end: datetime,
    region: str,
    expected_id: str,
    pages: list[FedsPageCollection],
    artifact_ids: list[str],
    page_number: int,
    error: str,
    retrieved_at: datetime,
) -> FedsWindowCollection:
    coverage = _record_window_coverage(
        archive_root,
        target=target,
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
        region=region,
        expected_id=expected_id,
        status=CoverageStatus.FAILED,
        artifact_ids=artifact_ids,
        detail={"page_count": len(pages), "failed_page_number": page_number},
        error=error,
        retrieved_at=retrieved_at,
    )
    return FedsWindowCollection(snapshot_start, snapshot_end, tuple(pages), coverage)


def _record_window_coverage(
    archive_root: str,
    *,
    target,
    snapshot_start: datetime,
    snapshot_end: datetime,
    region: str,
    expected_id: str,
    status: CoverageStatus,
    artifact_ids: list[str],
    detail: Mapping[str, Any],
    retrieved_at: datetime,
    error: str | None = None,
) -> CoverageRecord:
    return CoverageLedger(archive_root).record(
        source=target.provider,
        product=target.key,
        coverage_start=snapshot_start,
        coverage_end=snapshot_end,
        region=region,
        expected_coverage_id=expected_id,
        status=status,
        artifact_sha256s=artifact_ids,
        detail=dict(detail),
        error=error,
        recorded_at=retrieved_at,
    )


def _record_observed_snapshot_coverage(
    archive_root: str,
    *,
    target,
    snapshot_start: datetime,
    region: str,
    artifact_ids: list[str],
    detail: Mapping[str, Any],
    retrieved_at: datetime,
    status: CoverageStatus = CoverageStatus.COMPLETE,
    error: str | None = None,
) -> CoverageRecord:
    """Record evidence for one actual primary-key snapshot.

    This intentionally has distinct product and expected-ID namespaces from
    API-query coverage.  In particular, an absent primary-key group is never
    inferred to be an empty fire snapshot.
    """
    snapshot_end = snapshot_start + FEDS_SNAPSHOT_INTERVAL
    return CoverageLedger(archive_root).record(
        source=target.provider,
        product=FEDS_OBSERVED_SNAPSHOT_PRODUCT,
        coverage_start=snapshot_start,
        coverage_end=snapshot_end,
        region=region,
        expected_coverage_id=_observed_snapshot_expected_coverage_id(snapshot_start, region),
        status=status,
        artifact_sha256s=artifact_ids,
        detail=dict(detail),
        error=error,
        recorded_at=retrieved_at,
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


def _range_status(windows: Iterable[FedsWindowCollection]) -> CoverageStatus:
    values = tuple(window.coverage.status for window in windows)
    if any(value is CoverageStatus.FAILED for value in values):
        return CoverageStatus.FAILED
    if any(value is CoverageStatus.PARTIAL for value in values):
        return CoverageStatus.PARTIAL
    if values and all(value is CoverageStatus.EMPTY_CONFIRMED for value in values):
        return CoverageStatus.EMPTY_CONFIRMED
    return CoverageStatus.COMPLETE


def _latest_expected_coverage(ledger: CoverageLedger) -> dict[str, CoverageRecord]:
    return {
        record.expected_coverage_id: record
        for record in ledger.entries()
        if record.expected_coverage_id is not None
    }


def _is_terminal(record: CoverageRecord | None) -> bool:
    return record is not None and record.status in {
        CoverageStatus.COMPLETE,
        CoverageStatus.EMPTY_CONFIRMED,
    }


def _range_expected_coverage_id(start_date: date, end_date: date, region_label: str) -> str:
    return (
        f"feds-nrt-query:{FEDS_QUERY_COVERAGE_VERSION}:{region_label}:"
        f"{start_date.isoformat()}:{end_date.isoformat()}"
    )


def _window_expected_coverage_id(snapshot_start: datetime, region_label: str) -> str:
    return f"feds-nrt-query:{FEDS_QUERY_COVERAGE_VERSION}:{region_label}:{_format_utc(snapshot_start)}"


def _observed_snapshot_expected_coverage_id(snapshot_start: datetime, region_label: str) -> str:
    return (
        f"feds-nrt-primarykey-snapshot-observed:v1:{region_label}:"
        f"{_format_utc(snapshot_start)}"
    )


def _validated_region_names(values: Iterable[str]) -> tuple[str, ...]:
    regions = tuple(value.strip() for value in values if isinstance(value, str) and value.strip())
    if not regions:
        raise ValueError("region_names must contain at least one source region")
    if len(set(regions)) != len(regions):
        raise ValueError("region_names must not contain duplicates")
    if any(_SAFE_REGION_NAME.fullmatch(region) is None for region in regions):
        raise ValueError("region_names contain unsupported characters")
    return regions


def _region_where_clause(regions: tuple[str, ...]) -> str:
    quoted = ", ".join("'" + region.replace("'", "''") + "'" for region in regions)
    return f"region IN ({quoted})"


def _epoch_milliseconds(value: datetime) -> int:
    return int(_as_utc(value, "value").timestamp() * 1_000)


def _format_utc(value: datetime) -> str:
    return _as_utc(value, "value").isoformat().replace("+00:00", "Z")


def _format_extent(value: tuple[datetime, datetime] | None) -> dict[str, str] | None:
    if value is None:
        return None
    return {"start": _format_utc(value[0]), "end": _format_utc(value[1])}


def _as_utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be an offset-aware datetime")
    return value.astimezone(timezone.utc)


def _utc_now_or_value(value: datetime | None) -> datetime:
    return _as_utc(value or datetime.now(timezone.utc), "retrieved_at")


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _conservative_normalized_snapshot_bytes(records: Iterable[Mapping[str, Any]]) -> int:
    """Reserve compact derived bytes plus an intentionally generous margin."""
    encoded_bytes = sum(
        len(
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        + 1
        for record in records
    )
    return encoded_bytes * 2 + 65_536


def _conservative_response_bytes(payload: bytes) -> int:
    return len(payload) * 2 + 65_536


def _json_safe(value: Any) -> Any:
    """Convert provider non-finite values while preserving raw bytes separately."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


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
