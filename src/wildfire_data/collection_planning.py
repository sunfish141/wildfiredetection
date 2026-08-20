"""Plan expected source windows and identify coverage that needs retrying."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .collection_catalog import CollectionTarget
from .data_archive import CoverageLedger, CoverageStatus


@dataclass(frozen=True)
class CollectionWindow:
    """One explicit time/region/tile collection obligation."""

    target: CollectionTarget
    coverage_start: datetime
    coverage_end: datetime
    region: str
    tile: str | None
    expected_coverage_id: str


def plan_collection_windows(
    target: CollectionTarget,
    *,
    coverage_start: datetime,
    coverage_end: datetime,
    region: str | None = None,
    tile: str | None = None,
) -> tuple[CollectionWindow, ...]:
    """Split a UTC interval into the target's collection cadence windows."""
    start = _utc(coverage_start, "coverage_start")
    end = _utc(coverage_end, "coverage_end")
    if end <= start:
        raise ValueError("coverage_end must be after coverage_start")
    resolved_region = (region or target.region).strip()
    if not resolved_region:
        raise ValueError("region must be non-empty")
    interval = timedelta(minutes=target.cadence_minutes)
    windows = []
    current = start
    while current < end:
        next_boundary = min(current + interval, end)
        expected_id = _expected_id(target, current, next_boundary, resolved_region, tile)
        windows.append(
            CollectionWindow(
                target=target,
                coverage_start=current,
                coverage_end=next_boundary,
                region=resolved_region,
                tile=tile,
                expected_coverage_id=expected_id,
            )
        )
        current = next_boundary
    return tuple(windows)


def windows_needing_collection(
    ledger: CoverageLedger, windows: tuple[CollectionWindow, ...]
) -> tuple[CollectionWindow, ...]:
    """Return missing, failed, or partial windows; complete/empty ones are done."""
    latest_by_expected_id = {}
    for record in ledger.entries():
        if record.expected_coverage_id:
            latest_by_expected_id[record.expected_coverage_id] = record
    completed = {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED}
    return tuple(
        window
        for window in windows
        if latest_by_expected_id.get(window.expected_coverage_id) is None
        or latest_by_expected_id[window.expected_coverage_id].status not in completed
    )


def _expected_id(
    target: CollectionTarget,
    coverage_start: datetime,
    coverage_end: datetime,
    region: str,
    tile: str | None,
) -> str:
    tile_component = tile or "all"
    return ":".join(
        (
            target.key,
            region,
            tile_component,
            coverage_start.isoformat().replace("+00:00", "Z"),
            coverage_end.isoformat().replace("+00:00", "Z"),
        )
    )


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset")
    return value.astimezone(timezone.utc)
