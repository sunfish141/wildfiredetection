"""Leakage-safe, bounded candidate plans for compact issued-weather tiles.

This module plans *which* forecast tiles a later HRRR/HRDPS adapter may
retrieve.  It intentionally does not fetch a weather field: a plan is useful
only when every candidate can be traced to fire evidence that was available at
the forecast model run.  Keeping selected and capped candidates makes the
20 GB weather allocation auditable before any large provider response is read.
"""

from __future__ import annotations

import bisect
import csv
import gzip
import io
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from .storage_budget import StorageBudgetPolicy, require_admission


FORECAST_TILE_PLAN_SCHEMA_VERSION = 1
DEFAULT_TILE_KILOMETRES = 96.0
DEFAULT_ANCHOR_HOURS = 24
DEFAULT_FIRMS_AVAILABILITY_LAG_MINUTES = 180
DEFAULT_HRDPS_RUN_HOURS = (0, 6, 12, 18)
DEFAULT_HRDPS_TILES_PER_RUN = 32
WEATHER_RETENTION_PRIORITY_SCORE = 85
HISTORICAL_FORECAST_AVAILABILITY_SCORE = 0.6


@dataclass(frozen=True)
class ForecastTilePlan:
    """One selected-or-capped weather tile candidate for a model run."""

    model: str
    model_run_at: datetime
    tile_id: str
    tile_center_latitude: float
    tile_center_longitude: float
    anchor_window_start: datetime
    anchor_window_end: datetime
    candidate_detection_count: int
    max_bright_ti4: float
    latest_evidence_at: datetime
    representative_detection_id: str
    fire_evidence_score: float
    forecast_availability_score: float
    retention_priority_score: int
    selected: bool
    selection_rank: int | None
    non_admission_reason: str | None
    availability_policy: str

    def as_row(self) -> dict[str, object]:
        """Return a CSV-safe, stable representation of the plan row."""
        return {
            "schema_version": FORECAST_TILE_PLAN_SCHEMA_VERSION,
            "model": self.model,
            "model_run_at": _format_utc(self.model_run_at),
            "tile_id": self.tile_id,
            "tile_center_latitude": _format_float(self.tile_center_latitude),
            "tile_center_longitude": _format_float(self.tile_center_longitude),
            "anchor_window_start": _format_utc(self.anchor_window_start),
            "anchor_window_end": _format_utc(self.anchor_window_end),
            "candidate_detection_count": self.candidate_detection_count,
            "max_bright_ti4": _format_float(self.max_bright_ti4),
            "latest_evidence_at": _format_utc(self.latest_evidence_at),
            "representative_detection_id": self.representative_detection_id,
            "fire_evidence_score": _format_float(self.fire_evidence_score),
            "forecast_availability_score": _format_float(self.forecast_availability_score),
            "retention_priority_score": self.retention_priority_score,
            "selected": str(self.selected).lower(),
            "selection_rank": self.selection_rank if self.selection_rank is not None else "",
            "non_admission_reason": self.non_admission_reason or "",
            "availability_policy": self.availability_policy,
        }


class ForecastTilePlanError(ValueError):
    """Raised when a plan would have ambiguous time or spatial semantics."""


def forecast_model_runs(
    *,
    start_date: date,
    end_date: date,
    run_hours: Sequence[int] = DEFAULT_HRDPS_RUN_HOURS,
) -> tuple[datetime, ...]:
    """Return all UTC model-run anchors in an inclusive date range."""
    if end_date < start_date:
        raise ForecastTilePlanError("end_date must not be before start_date")
    normalized_hours = _validated_run_hours(run_hours)
    runs = []
    current = start_date
    while current <= end_date:
        for hour in normalized_hours:
            runs.append(datetime.combine(current, time(hour=hour), tzinfo=timezone.utc))
        current += timedelta(days=1)
    return tuple(runs)


def plan_forecast_tiles(
    detections: Iterable[Mapping[str, object]],
    *,
    model: str,
    start_date: date,
    end_date: date,
    run_hours: Sequence[int] = DEFAULT_HRDPS_RUN_HOURS,
    max_tiles_per_run: int = DEFAULT_HRDPS_TILES_PER_RUN,
    tile_kilometres: float = DEFAULT_TILE_KILOMETRES,
    anchor_hours: int = DEFAULT_ANCHOR_HOURS,
    availability_lag_minutes: int = DEFAULT_FIRMS_AVAILABILITY_LAG_MINUTES,
    forecast_availability_score: float = HISTORICAL_FORECAST_AVAILABILITY_SCORE,
) -> tuple[ForecastTilePlan, ...]:
    """Score and cap fire-evidence weather-tile candidates without leakage.

    A detection contributes only after an explicit conservative FIRMS latency
    has elapsed and only while it falls in the preceding anchor window.  Tile
    coordinates use a 96 km Web-Mercator planning grid; a source-specific
    collector must later resolve each center to native HRRR/HRDPS grid cells.
    """
    model_name = _required_text(model, "model")
    if max_tiles_per_run <= 0:
        raise ForecastTilePlanError("max_tiles_per_run must be positive")
    if tile_kilometres <= 0:
        raise ForecastTilePlanError("tile_kilometres must be positive")
    if anchor_hours <= 0:
        raise ForecastTilePlanError("anchor_hours must be positive")
    if availability_lag_minutes < 0:
        raise ForecastTilePlanError("availability_lag_minutes must not be negative")
    if not 0.0 <= forecast_availability_score <= 1.0:
        raise ForecastTilePlanError("forecast_availability_score must be between zero and one")

    runs = forecast_model_runs(start_date=start_date, end_date=end_date, run_hours=run_hours)
    if not runs:
        return ()
    anchor_window = timedelta(hours=anchor_hours)
    availability_lag = timedelta(minutes=availability_lag_minutes)
    minimum_evidence_at = runs[0] - anchor_window
    maximum_evidence_at = runs[-1]
    buckets: dict[tuple[datetime, int, int], _CandidateBucket] = {}
    run_values = list(runs)

    for detection in detections:
        evidence = _detection_evidence(detection)
        evidence = replace(evidence, available_at=evidence.acquired_at + availability_lag)
        if evidence.acquired_at < minimum_evidence_at or evidence.acquired_at > maximum_evidence_at:
            continue
        first_run_index = bisect.bisect_left(run_values, evidence.available_at)
        final_run_at = evidence.acquired_at + anchor_window
        while first_run_index < len(run_values):
            run_at = run_values[first_run_index]
            if run_at > final_run_at:
                break
            tile_x, tile_y = _tile_indices(
                latitude=evidence.latitude,
                longitude=evidence.longitude,
                tile_kilometres=tile_kilometres,
            )
            key = (run_at, tile_x, tile_y)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = _CandidateBucket(
                    detection_count=0,
                    max_bright_ti4=evidence.bright_ti4,
                    latest_evidence_at=evidence.acquired_at,
                    representative_detection_id=evidence.detection_id,
                )
                buckets[key] = bucket
            bucket.add(evidence)
            first_run_index += 1

    policy = (
        "FIRMS acquisition time plus "
        f"{availability_lag_minutes} minute conservative availability lag; "
        "historical forecast publication time remains unverified"
    )
    planned = []
    for run_at in runs:
        candidates = []
        for (candidate_run, tile_x, tile_y), bucket in buckets.items():
            if candidate_run != run_at:
                continue
            center_latitude, center_longitude = _tile_center(
                tile_x=tile_x,
                tile_y=tile_y,
                tile_kilometres=tile_kilometres,
            )
            candidates.append(
                (
                    _fire_evidence_score(
                        max_bright_ti4=bucket.max_bright_ti4,
                        detection_count=bucket.detection_count,
                        latest_evidence_at=bucket.latest_evidence_at,
                        run_at=run_at,
                        anchor_hours=anchor_hours,
                    ),
                    _tile_id(tile_x, tile_y, tile_kilometres),
                    center_latitude,
                    center_longitude,
                    bucket,
                )
            )
        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
        for position, (score, tile_id, latitude, longitude, bucket) in enumerate(candidates, start=1):
            selected = position <= max_tiles_per_run
            planned.append(
                ForecastTilePlan(
                    model=model_name,
                    model_run_at=run_at,
                    tile_id=tile_id,
                    tile_center_latitude=latitude,
                    tile_center_longitude=longitude,
                    anchor_window_start=run_at - anchor_window,
                    anchor_window_end=run_at,
                    candidate_detection_count=bucket.detection_count,
                    max_bright_ti4=bucket.max_bright_ti4,
                    latest_evidence_at=bucket.latest_evidence_at,
                    representative_detection_id=bucket.representative_detection_id,
                    fire_evidence_score=score,
                    forecast_availability_score=forecast_availability_score,
                    retention_priority_score=WEATHER_RETENTION_PRIORITY_SCORE,
                    selected=selected,
                    selection_rank=position if selected else None,
                    non_admission_reason=None if selected else "per-run-tile-cap",
                    availability_policy=policy,
                )
            )
    return tuple(planned)


def iter_normalized_firms_detections(
    data_root: str | Path,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> Iterable[dict[str, object]]:
    """Yield source-faithful FIRMS detections from selected date partitions."""
    if start_date is not None and end_date is not None and end_date < start_date:
        raise ForecastTilePlanError("end_date must not be before start_date")
    root = Path(data_root) / "normalized" / "fire-detections"
    for path in sorted(root.rglob("*.jsonl.gz")):
        partition_date = _partition_date(path)
        if start_date is not None and (partition_date is None or partition_date < start_date):
            continue
        if end_date is not None and (partition_date is None or partition_date > end_date):
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        record = json.loads(line)
                        if isinstance(record, dict) and record.get("record_type") == "firms_detection":
                            yield record
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ForecastTilePlanError(f"Could not read normalized FIRMS artifact: {path}") from exc


def _partition_date(path: Path) -> date | None:
    for parent in path.parents:
        if parent.name.startswith("acq-date="):
            try:
                return date.fromisoformat(parent.name.removeprefix("acq-date="))
            except ValueError as exc:
                raise ForecastTilePlanError(f"Invalid FIRMS acquisition-date partition: {parent}") from exc
    return None


def write_forecast_tile_plan(
    data_root: str | Path,
    *,
    plan: Iterable[ForecastTilePlan],
    output_path: str | Path,
    storage_budget: StorageBudgetPolicy,
) -> Path:
    """Write a compressed, quota-admitted candidate plan without changing evidence."""
    resolved_plan = tuple(plan)
    rows = [candidate.as_row() for candidate in resolved_plan]
    fieldnames = [
        "schema_version",
        "model",
        "model_run_at",
        "tile_id",
        "tile_center_latitude",
        "tile_center_longitude",
        "anchor_window_start",
        "anchor_window_end",
        "candidate_detection_count",
        "max_bright_ti4",
        "latest_evidence_at",
        "representative_detection_id",
        "fire_evidence_score",
        "forecast_availability_score",
        "retention_priority_score",
        "selected",
        "selection_rank",
        "non_admission_reason",
        "availability_policy",
    ]
    text_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(text_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    compressed = gzip.compress(text_buffer.getvalue().encode("utf-8"), mtime=0)
    root = Path(data_root)
    target = Path(output_path)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ForecastTilePlanError("output_path must be below data_root") from exc
    require_admission(
        storage_budget,
        root,
        category="issued_weather_tiles",
        requested_bytes=len(compressed) + 65_536,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(compressed)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return target


@dataclass
class _CandidateBucket:
    detection_count: int
    max_bright_ti4: float
    latest_evidence_at: datetime
    representative_detection_id: str

    def add(self, evidence: "_DetectionEvidence") -> None:
        self.detection_count += 1
        if evidence.acquired_at > self.latest_evidence_at:
            self.latest_evidence_at = evidence.acquired_at
        if (evidence.bright_ti4, evidence.detection_id) > (
            self.max_bright_ti4,
            self.representative_detection_id,
        ):
            self.max_bright_ti4 = evidence.bright_ti4
            self.representative_detection_id = evidence.detection_id


@dataclass(frozen=True)
class _DetectionEvidence:
    detection_id: str
    acquired_at: datetime
    available_at: datetime
    latitude: float
    longitude: float
    bright_ti4: float


def _detection_evidence(detection: Mapping[str, object]) -> _DetectionEvidence:
    if not isinstance(detection, Mapping):
        raise ForecastTilePlanError("detections must be mappings")
    acquired_at = _parse_utc(detection.get("acquired_at"), "acquired_at")
    detection_id = _required_text(detection.get("detection_id"), "detection_id")
    latitude = _finite_float(detection.get("latitude"), "latitude")
    longitude = _finite_float(detection.get("longitude"), "longitude")
    bright_ti4 = _finite_float(detection.get("bright_ti4"), "bright_ti4")
    if not -85.0 < latitude < 85.0:
        raise ForecastTilePlanError("latitude must be between -85 and 85 for the planning grid")
    if not -180.0 <= longitude <= 180.0:
        raise ForecastTilePlanError("longitude must be between -180 and 180")
    # ``available_at`` is filled by the caller because it is a plan policy,
    # not a provider claim embedded in the FIRMS raw record.
    return _DetectionEvidence(
        detection_id=detection_id,
        acquired_at=acquired_at,
        available_at=acquired_at,
        latitude=latitude,
        longitude=longitude,
        bright_ti4=bright_ti4,
    )


def _tile_indices(*, latitude: float, longitude: float, tile_kilometres: float) -> tuple[int, int]:
    tile_metres = tile_kilometres * 1_000
    radius = 6_378_137.0
    x = radius * math.radians(longitude)
    y = radius * math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2))
    return math.floor(x / tile_metres), math.floor(y / tile_metres)


def _tile_center(*, tile_x: int, tile_y: int, tile_kilometres: float) -> tuple[float, float]:
    tile_metres = tile_kilometres * 1_000
    radius = 6_378_137.0
    x = (tile_x + 0.5) * tile_metres
    y = (tile_y + 0.5) * tile_metres
    longitude = math.degrees(x / radius)
    latitude = math.degrees(2 * math.atan(math.exp(y / radius)) - math.pi / 2)
    return latitude, longitude


def _tile_id(tile_x: int, tile_y: int, tile_kilometres: float) -> str:
    return f"webmercator-{tile_kilometres:g}km-x{tile_x}-y{tile_y}"


def _fire_evidence_score(
    *,
    max_bright_ti4: float,
    detection_count: int,
    latest_evidence_at: datetime,
    run_at: datetime,
    anchor_hours: int,
) -> float:
    brightness = max(0.0, min(1.0, max_bright_ti4 / 400.0))
    density = min(1.0, math.log1p(detection_count) / math.log(11.0))
    age_hours = max(0.0, (run_at - latest_evidence_at).total_seconds() / 3_600)
    recency = max(0.0, 1.0 - age_hours / anchor_hours)
    return round(0.55 * brightness + 0.25 * density + 0.20 * recency, 6)


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ForecastTilePlanError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForecastTilePlanError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ForecastTilePlanError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _finite_float(value: object, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ForecastTilePlanError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ForecastTilePlanError(f"{label} must be finite")
    return parsed


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForecastTilePlanError(f"{label} must be non-empty text")
    return value.strip()


def _validated_run_hours(values: Sequence[int]) -> tuple[int, ...]:
    hours = tuple(values)
    if not hours or any(not isinstance(hour, int) or isinstance(hour, bool) or not 0 <= hour <= 23 for hour in hours):
        raise ForecastTilePlanError("run_hours must contain integer UTC hours between 0 and 23")
    if len(hours) != len(set(hours)):
        raise ForecastTilePlanError("run_hours must not contain duplicates")
    return tuple(sorted(hours))


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _format_float(value: float) -> str:
    return f"{value:.6f}"
