"""Leakage-safe FIRMS fire-state features for one canonical training cell.

FIRMS detections are observations, not a fire perimeter.  This module keeps
that distinction explicit: it aggregates only detections that were both
acquired within a caller-selected lookback window and operationally available
before the requested UTC cutoff.  It does not create labels, read files, or
infer availability from a retrospective ``ingested_at`` timestamp.

The functions accept the normalized FIRMS record shape produced by
``firms_normalization`` (or an equivalent mapping) so a dataset builder can
stream partitions rather than materialising an archive-specific dataframe.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .training_grid import GridCell, TrainingGridError, cell_from_id, cell_from_wgs84, format_utc


FIRMS_FIRE_STATE_FEATURE_SCHEMA_VERSION = 1
FIRMS_FIRE_STATE_FEATURE_BUILD_VERSION = "firms-cell-state-1km/v1"
NEIGHBOURHOOD_RADIUS_CELLS = 1


class FireStateFeatureError(ValueError):
    """Raised when a feature cutoff or normalized-like detection is unsafe."""


@dataclass(frozen=True)
class _DetectionEvidence:
    """Validated input fields needed for a leakage-safe aggregate."""

    detection_id: str
    acquired_at: datetime
    latitude: float
    longitude: float
    bright_ti4: float
    platform: str

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        """Fields that must agree if a detection is retained more than once."""
        return (
            self.detection_id,
            self.acquired_at,
            self.latitude,
            self.longitude,
            self.bright_ti4,
            self.platform,
        )


def build_firms_fire_state_features(
    detections: Iterable[Mapping[str, Any]],
    *,
    cell_id: str,
    cutoff_at: datetime,
    lookback: timedelta,
    availability_lag: timedelta,
) -> dict[str, Any]:
    """Aggregate recent FIRMS evidence for a cell and its 3-by-3 context.

    ``cutoff_at`` is the time at which a model is allowed to make a
    prediction.  A detection is eligible only when all of the following hold:

    * its acquisition time is in ``[cutoff_at - lookback, cutoff_at]``;
    * ``acquired_at + availability_lag <= cutoff_at``; and
    * its mapped canonical cell is the requested cell or one of its eight
      immediate neighbours.

    ``availability_lag`` is deliberately required.  Backfilled archive
    ``ingested_at`` values prove only when this project obtained a record, not
    when a real-time model could have known it.  Callers must record their
    source-specific operational latency assumption alongside the resulting
    feature records.

    Duplicate normalized artifacts are common when collection ranges overlap.
    Identical ``detection_id`` values count once.  If two eligible records use
    the same ID but disagree on model-relevant fields, this function raises
    instead of choosing an arbitrary revision.

    Brightness summaries are ``None`` when no evidence is eligible; the count
    and ``has_detection`` fields let downstream tabular models distinguish a
    missing numeric summary from a zero-valued physical measurement.
    """
    target = _canonical_cell(cell_id)
    cutoff = _as_utc(cutoff_at, "cutoff_at")
    lookback_value = _nonnegative_duration(lookback, "lookback")
    availability_lag_value = _nonnegative_duration(availability_lag, "availability_lag")
    window_start = cutoff - lookback_value
    latest_eligible_acquisition = cutoff - availability_lag_value

    center_evidence: list[_DetectionEvidence] = []
    local_evidence: list[tuple[_DetectionEvidence, GridCell]] = []
    seen_by_id: dict[str, _DetectionEvidence] = {}
    for record in detections:
        evidence = _parse_evidence(record)
        if evidence.acquired_at < window_start or evidence.acquired_at > cutoff:
            continue
        # This is the key time-travel guard.  Do not substitute retrospective
        # retrieval/ingestion timestamps here: they are not source availability.
        if evidence.acquired_at > latest_eligible_acquisition:
            continue
        mapped_cell = _mapped_cell(evidence)
        if not _is_in_local_neighbourhood(target, mapped_cell):
            continue
        existing = seen_by_id.get(evidence.detection_id)
        if existing is not None:
            if existing.fingerprint != evidence.fingerprint:
                raise FireStateFeatureError(
                    "duplicate detection_id has conflicting model-relevant fields: "
                    f"{evidence.detection_id!r}"
                )
            continue
        seen_by_id[evidence.detection_id] = evidence
        local_evidence.append((evidence, mapped_cell))
        if mapped_cell == target:
            center_evidence.append(evidence)

    # Sorting makes floating point means and output deterministic even when an
    # upstream partition iterator has a different order on a later rebuild.
    center_evidence.sort(key=lambda item: item.detection_id)
    local_evidence.sort(key=lambda item: item[0].detection_id)
    center = _aggregate(center_evidence, cutoff_at=cutoff)
    local = _aggregate((item for item, _cell in local_evidence), cutoff_at=cutoff)
    local_active_cell_count = len({cell.cell_id for _item, cell in local_evidence})

    return {
        "schema_version": FIRMS_FIRE_STATE_FEATURE_SCHEMA_VERSION,
        "feature_build_version": FIRMS_FIRE_STATE_FEATURE_BUILD_VERSION,
        "feature_source": "NASA FIRMS normalized detections",
        "cell_id": target.cell_id,
        "cutoff_at": format_utc(cutoff),
        "lookback_start_at": format_utc(window_start),
        "lookback_hours": _duration_hours(lookback_value),
        "availability_lag_hours": _duration_hours(availability_lag_value),
        "latest_eligible_acquisition_at": format_utc(latest_eligible_acquisition),
        "firms_center_has_detection": center["has_detection"],
        "firms_center_detection_count": center["detection_count"],
        "firms_center_bright_ti4_max": center["bright_ti4_max"],
        "firms_center_bright_ti4_mean": center["bright_ti4_mean"],
        "firms_center_platform_count": center["platform_count"],
        "firms_center_hours_since_last_detection": center["hours_since_last_detection"],
        "firms_local_3x3_has_detection": local["has_detection"],
        "firms_local_3x3_detection_count": local["detection_count"],
        "firms_local_3x3_bright_ti4_max": local["bright_ti4_max"],
        "firms_local_3x3_bright_ti4_mean": local["bright_ti4_mean"],
        "firms_local_3x3_platform_count": local["platform_count"],
        "firms_local_3x3_hours_since_last_detection": local["hours_since_last_detection"],
        "firms_local_3x3_active_cell_count": local_active_cell_count,
    }


def _aggregate(
    evidence: Iterable[_DetectionEvidence], *, cutoff_at: datetime
) -> dict[str, int | float | None]:
    values = tuple(evidence)
    if not values:
        return {
            "has_detection": 0,
            "detection_count": 0,
            "bright_ti4_max": None,
            "bright_ti4_mean": None,
            "platform_count": 0,
            "hours_since_last_detection": None,
        }
    brightnesses = [item.bright_ti4 for item in values]
    most_recent = max(item.acquired_at for item in values)
    return {
        "has_detection": 1,
        "detection_count": len(values),
        "bright_ti4_max": max(brightnesses),
        "bright_ti4_mean": math.fsum(brightnesses) / len(brightnesses),
        "platform_count": len({item.platform for item in values}),
        "hours_since_last_detection": (cutoff_at - most_recent).total_seconds() / 3600.0,
    }


def _parse_evidence(record: Mapping[str, Any]) -> _DetectionEvidence:
    if not isinstance(record, Mapping):
        raise FireStateFeatureError("each FIRMS detection must be a mapping")
    detection_id = record.get("detection_id")
    if not isinstance(detection_id, str) or not detection_id.strip():
        raise FireStateFeatureError("FIRMS detection has no non-empty detection_id")
    acquired_at = _timestamp(record.get("acquired_at"), "acquired_at")
    latitude = _finite_coordinate(record.get("latitude"), "latitude", minimum=-90.0, maximum=90.0)
    longitude = _finite_coordinate(record.get("longitude"), "longitude", minimum=-180.0, maximum=180.0)
    brightness = record.get("bright_ti4")
    if brightness is None:
        raw = record.get("raw_source_fields")
        brightness = raw.get("bright_ti4") if isinstance(raw, Mapping) else None
    bright_ti4 = _finite_number(brightness, "bright_ti4")
    return _DetectionEvidence(
        detection_id=detection_id.strip(),
        acquired_at=acquired_at,
        latitude=latitude,
        longitude=longitude,
        bright_ti4=bright_ti4,
        platform=_platform_identifier(record),
    )


def _platform_identifier(record: Mapping[str, Any]) -> str:
    """Return a stable sensor/platform identity without using ingestion IDs."""
    raw = record.get("raw_source_fields")
    provenance = record.get("provenance")
    candidates = (
        record.get("platform"),
        record.get("satellite"),
        raw.get("satellite") if isinstance(raw, Mapping) else None,
        raw.get("platform") if isinstance(raw, Mapping) else None,
        provenance.get("product") if isinstance(provenance, Mapping) else None,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().casefold()
    # Keep a data-quality issue visible without fabricating a per-record
    # platform from source IDs or retrieval metadata.
    return "unknown"


def _mapped_cell(evidence: _DetectionEvidence) -> GridCell:
    try:
        return cell_from_wgs84(latitude=evidence.latitude, longitude=evidence.longitude)
    except TrainingGridError as exc:
        raise FireStateFeatureError("FIRMS detection cannot be mapped to a canonical cell") from exc


def _canonical_cell(cell_id: str) -> GridCell:
    try:
        return cell_from_id(cell_id)
    except TrainingGridError as exc:
        raise FireStateFeatureError("cell_id is not a canonical 1 km training cell") from exc


def _is_in_local_neighbourhood(target: GridCell, candidate: GridCell) -> bool:
    return (
        abs(candidate.x_index - target.x_index) <= NEIGHBOURHOOD_RADIUS_CELLS
        and abs(candidate.y_index - target.y_index) <= NEIGHBOURHOOD_RADIUS_CELLS
    )


def _as_utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise FireStateFeatureError(f"{label} must be an offset-aware datetime")
    return value.astimezone(timezone.utc)


def _timestamp(value: Any, label: str) -> datetime:
    if isinstance(value, datetime):
        return _as_utc(value, label)
    if not isinstance(value, str) or not value.strip():
        raise FireStateFeatureError(f"{label} must be an offset-aware ISO-8601 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise FireStateFeatureError(f"{label} must be an offset-aware ISO-8601 timestamp") from exc
    return _as_utc(parsed, label)


def _nonnegative_duration(value: timedelta, label: str) -> timedelta:
    if not isinstance(value, timedelta):
        raise FireStateFeatureError(f"{label} must be a timedelta")
    if value < timedelta(0):
        raise FireStateFeatureError(f"{label} must not be negative")
    return value


def _finite_coordinate(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    numeric = _finite_number(value, label)
    if not minimum <= numeric <= maximum:
        raise FireStateFeatureError(f"{label} must be between {minimum:g} and {maximum:g}")
    return numeric


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise FireStateFeatureError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise FireStateFeatureError(f"{label} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise FireStateFeatureError(f"{label} must be a finite number")
    return numeric


def _duration_hours(value: timedelta) -> float:
    return value.total_seconds() / 3600.0
