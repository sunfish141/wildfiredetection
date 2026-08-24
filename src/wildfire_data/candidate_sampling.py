"""Deterministic FIRMS-only candidates for the first weak-label baseline.

The module intentionally separates *where a model may score* from what is
known to have burned.  Candidate cells are expanded only from FIRMS detections
that would have been available at a FEDS-aligned cutoff.  FEDS positives are
then joined when present; every other retained candidate is a clearly marked
weak-negative proxy, not a clear/no-burn observation.

This is the label/candidate policy layer.  Feature assembly and immutable
table publication remain separate so callers can retain the source labels,
FIRMS evidence, terrain sampler, and completed-view manifest as lineage.
"""

from __future__ import annotations

import hashlib
import heapq
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .feds_labels import estimate_feds_observation_at
from .training_grid import (
    GridCell,
    TrainingExampleKey,
    cell_from_id,
    cell_from_wgs84,
    cells_in_square_radius,
    format_utc,
)


CANDIDATE_SAMPLER_VERSION = "firms-only-radius-weak-negative-proxy/v1"
WEAK_NEGATIVE_PROXY_OBSERVABILITY = "unobserved-no-clear-no-burn-mask"
WEAK_NEGATIVE_PROXY_STATUS = "weak-negative-proxy-unobserved"
UNSCORED_POSITIVE_STATUS = "unscored-positive-no-firms-candidate"
MAX_RETAINED_SEED_DETECTION_IDS = 128
# A 2 km square candidate expansion changes local-solar UTC alignment by only
# seconds in the CONUS/Canada scope.  The wider margin avoids an expensive
# inverse projection for clearly eligible/ineligible detections while routing
# every near-boundary decision through the exact cell-centre calculation.
_FAST_ELIGIBILITY_MARGIN = timedelta(minutes=30)


class CandidateSamplingError(ValueError):
    """Raised when inputs cannot form a deterministic FIRMS candidate view."""


@dataclass(frozen=True)
class CandidateSamplingResult:
    """Rows eligible for a weak binary fit and positives outside its support."""

    candidate_rows: tuple[dict[str, Any], ...]
    unscored_positive_rows: tuple[dict[str, Any], ...]

    @property
    def positive_candidate_count(self) -> int:
        return sum(row["target_newly_burned_12h"] == 1 for row in self.candidate_rows)

    @property
    def weak_negative_proxy_count(self) -> int:
        return sum(row["target_newly_burned_12h"] == 0 for row in self.candidate_rows)


@dataclass(frozen=True)
class _Detection:
    detection_id: str
    acquired_at: datetime
    cell: GridCell
    raw_artifact_id: str | None


def build_firms_only_candidates(
    positive_labels: Iterable[Mapping[str, Any]],
    firms_detections: Iterable[Mapping[str, Any]],
    *,
    radius_cells: int = 2,
    max_weak_negative_proxies_per_snapshot: int = 2_000,
    firms_lookback: timedelta = timedelta(hours=24),
    firms_availability_lag: timedelta = timedelta(hours=3),
) -> CandidateSamplingResult:
    """Create FIRMS-seeded candidate labels without inventing observability.

    For each retained FEDS source snapshot, a candidate is a canonical cell
    within ``radius_cells`` of a FIRMS detection that is usable at that cell's
    local-solar aligned cutoff.  An exact FEDS-positive cell becomes target 1.
    Other selected cells receive target 0 only as a named ``weak_negative``
    proxy; no row claims clear/no-burn observation coverage.  Positives that
    have no FIRMS candidate are preserved separately as unscored diagnostics.

    The snapshot-specific deterministic hash cap controls only proxy rows.
    Positives are never capped away.
    """
    if not isinstance(radius_cells, int) or isinstance(radius_cells, bool) or radius_cells < 0:
        raise CandidateSamplingError("radius_cells must be a non-negative integer")
    if (
        not isinstance(max_weak_negative_proxies_per_snapshot, int)
        or isinstance(max_weak_negative_proxies_per_snapshot, bool)
        or max_weak_negative_proxies_per_snapshot < 0
    ):
        raise CandidateSamplingError("max_weak_negative_proxies_per_snapshot must be non-negative")
    lookback = _nonnegative_duration(firms_lookback, "firms_lookback")
    availability_lag = _nonnegative_duration(firms_availability_lag, "firms_availability_lag")

    labels = tuple(_validated_positive_label(record) for record in positive_labels)
    detections = tuple(_validated_detection(record) for record in firms_detections)
    positives_by_snapshot: dict[datetime, dict[str, dict[str, Any]]] = defaultdict(dict)
    for label in labels:
        snapshot = _parse_utc(label["source_snapshot_time"], "source_snapshot_time")
        cell_id = label["cell_id"]
        if cell_id in positives_by_snapshot[snapshot]:
            raise CandidateSamplingError("duplicate positive cell for one FEDS source snapshot")
        positives_by_snapshot[snapshot][cell_id] = label

    rows: list[dict[str, Any]] = []
    unscored: list[dict[str, Any]] = []
    for snapshot_time in sorted(positives_by_snapshot):
        positives = positives_by_snapshot[snapshot_time]
        positive_cell_ids = set(positives)
        candidate_cell_ids, detections_by_seed_cell = _select_candidate_cell_ids(
            snapshot_time=snapshot_time,
            detections=detections,
            positive_cell_ids=positive_cell_ids,
            radius_cells=radius_cells,
            max_weak_negative_proxies=max_weak_negative_proxies_per_snapshot,
            lookback=lookback,
            availability_lag=availability_lag,
        )
        supported_positive_ids = positive_cell_ids & candidate_cell_ids
        proxy_cell_ids = sorted(candidate_cell_ids - positive_cell_ids, key=lambda cell_id: _rank(snapshot_time, cell_id))
        selected_cell_ids = sorted(supported_positive_ids) + proxy_cell_ids
        for cell_id in selected_cell_ids:
            # Recovering the cell from its source-independent ID avoids using a
            # detection's point location as a model-cell coordinate.
            cell = cell_from_id(cell_id)
            anchor_at = _anchor_for_cell(snapshot_time, cell)
            seeds = _candidate_seeds(
                cell=cell,
                cutoff_at=anchor_at,
                detections_by_seed_cell=detections_by_seed_cell,
                radius_cells=radius_cells,
                lookback=lookback,
                availability_lag=availability_lag,
            )
            positive = positives.get(cell_id)
            rows.append(
                _candidate_row(
                    cell=cell,
                    snapshot_time=snapshot_time,
                    anchor_at=anchor_at,
                    seeds=seeds,
                    positive_label=positive,
                    radius_cells=radius_cells,
                    lookback=lookback,
                    availability_lag=availability_lag,
                )
            )
        for cell_id in sorted(positive_cell_ids - supported_positive_ids):
            label = positives[cell_id]
            unscored.append(
                {
                    **label,
                    "candidate_sampler_version": CANDIDATE_SAMPLER_VERSION,
                    "candidate_selection_reason": UNSCORED_POSITIVE_STATUS,
                    "binary_training_eligible": 0,
                    "binary_training_status": UNSCORED_POSITIVE_STATUS,
                }
            )
    return CandidateSamplingResult(
        candidate_rows=tuple(sorted(rows, key=lambda row: (row["source_snapshot_time"], row["cell_id"]))),
        unscored_positive_rows=tuple(sorted(unscored, key=lambda row: (row["source_snapshot_time"], row["cell_id"]))),
    )


def _select_candidate_cell_ids(
    *,
    snapshot_time: datetime,
    detections: tuple[_Detection, ...],
    positive_cell_ids: set[str],
    radius_cells: int,
    max_weak_negative_proxies: int,
    lookback: timedelta,
    availability_lag: timedelta,
) -> tuple[set[str], dict[str, list[_Detection]]]:
    """Return supported positives plus a bounded deterministic proxy sample.

    This deliberately avoids holding every FIRMS-radius cell in memory.  A
    source snapshot can have millions of raw detections, but the output has at
    most all supported positives plus the declared proxy cap.  The retained
    detection-by-cell index is later used to recover complete seed lineage for
    only those selected cells.
    """
    supported_positive_ids: set[str] = set()
    selected_proxy_ranks: dict[str, int] = {}
    # Max heap represented by negative rank: the root is the currently worst
    # selected proxy and is replaced only by a lower deterministic hash rank.
    proxy_heap: list[tuple[int, str]] = []
    detections_by_seed_cell: dict[str, list[_Detection]] = defaultdict(list)
    anchor_cache: dict[str, datetime] = {}

    def anchor_for(cell: GridCell) -> datetime:
        return anchor_cache.setdefault(cell.cell_id, _anchor_for_cell(snapshot_time, cell))

    # First collapse repeated FIRMS pixels. Candidate geometry depends on a
    # seed cell, not on how many platforms/retrievals reported that same cell.
    # We retain every potentially eligible detection in the value lists so the
    # selected rows still receive complete seed lineage later.
    for detection in detections:
        seed_anchor = anchor_for(detection.cell)
        if not _possibly_eligible(
            detection,
            cutoff_at=seed_anchor,
            lookback=lookback,
            availability_lag=availability_lag,
        ):
            continue
        detections_by_seed_cell[detection.cell.cell_id].append(detection)

    for seed_cell_id, seed_detections in detections_by_seed_cell.items():
        seed_cell = seed_detections[0].cell
        seed_anchor = anchor_for(seed_cell)
        definitely_eligible = any(
            _definitely_eligible(
                detection,
                cutoff_at=seed_anchor,
                lookback=lookback,
                availability_lag=availability_lag,
            )
            for detection in seed_detections
        )
        for candidate in cells_in_square_radius(seed_cell, radius_cells=radius_cells):
            if not definitely_eligible and not any(
                _eligible(
                    detection,
                    cutoff_at=anchor_for(candidate),
                    lookback=lookback,
                    availability_lag=availability_lag,
                )
                for detection in seed_detections
            ):
                continue
            candidate_id = candidate.cell_id
            if candidate_id in positive_cell_ids:
                supported_positive_ids.add(candidate_id)
                continue
            _consider_proxy_cell(
                candidate_id,
                snapshot_time=snapshot_time,
                limit=max_weak_negative_proxies,
                selected_ranks=selected_proxy_ranks,
                max_heap=proxy_heap,
            )
    return supported_positive_ids | set(selected_proxy_ranks), dict(detections_by_seed_cell)


def _possibly_eligible(
    detection: _Detection,
    *,
    cutoff_at: datetime,
    lookback: timedelta,
    availability_lag: timedelta,
) -> bool:
    return (
        cutoff_at - lookback - _FAST_ELIGIBILITY_MARGIN <= detection.acquired_at
        and detection.acquired_at + availability_lag <= cutoff_at + _FAST_ELIGIBILITY_MARGIN
    )


def _definitely_eligible(
    detection: _Detection,
    *,
    cutoff_at: datetime,
    lookback: timedelta,
    availability_lag: timedelta,
) -> bool:
    return (
        cutoff_at - lookback + _FAST_ELIGIBILITY_MARGIN <= detection.acquired_at
        and detection.acquired_at + availability_lag <= cutoff_at - _FAST_ELIGIBILITY_MARGIN
    )


def _consider_proxy_cell(
    cell_id: str,
    *,
    snapshot_time: datetime,
    limit: int,
    selected_ranks: dict[str, int],
    max_heap: list[tuple[int, str]],
) -> None:
    if limit == 0 or cell_id in selected_ranks:
        return
    rank = int(_rank(snapshot_time, cell_id), 16)
    if len(selected_ranks) < limit:
        selected_ranks[cell_id] = rank
        heapq.heappush(max_heap, (-rank, cell_id))
        return
    worst_negative_rank, worst_cell_id = max_heap[0]
    worst_rank = -worst_negative_rank
    if rank >= worst_rank:
        return
    heapq.heapreplace(max_heap, (-rank, cell_id))
    selected_ranks.pop(worst_cell_id)
    selected_ranks[cell_id] = rank


def _candidate_seeds(
    *,
    cell: GridCell,
    cutoff_at: datetime,
    detections_by_seed_cell: Mapping[str, list[_Detection]],
    radius_cells: int,
    lookback: timedelta,
    availability_lag: timedelta,
) -> tuple[_Detection, ...]:
    seeds = {
        detection.detection_id: detection
        for neighbour in cells_in_square_radius(cell, radius_cells=radius_cells)
        for detection in detections_by_seed_cell.get(neighbour.cell_id, ())
        if _eligible(
            detection,
            cutoff_at=cutoff_at,
            lookback=lookback,
            availability_lag=availability_lag,
        )
    }
    return tuple(sorted(seeds.values(), key=lambda item: item.detection_id))


def _candidate_row(*, cell: GridCell, snapshot_time: datetime, anchor_at: datetime, seeds: tuple[_Detection, ...], positive_label: Mapping[str, Any] | None, radius_cells: int, lookback: timedelta, availability_lag: timedelta) -> dict[str, Any]:
    key = TrainingExampleKey(cell_id=cell.cell_id, anchor_at=anchor_at)
    latitude, longitude = cell.center_wgs84
    all_seed_ids = [seed.detection_id for seed in seeds]
    seed_ids = all_seed_ids[:MAX_RETAINED_SEED_DETECTION_IDS]
    raw_ids = sorted({seed.raw_artifact_id for seed in seeds if seed.raw_artifact_id})
    common = {
        "example_id": key.example_id,
        "cell_id": cell.cell_id,
        "cell_center_latitude": latitude,
        "cell_center_longitude": longitude,
        "anchor_at": format_utc(anchor_at),
        "feature_cutoff_at": format_utc(anchor_at),
        "target_end_at": format_utc(key.target_end_at),
        "source_snapshot_time": format_utc(snapshot_time),
        "candidate_sampler_version": CANDIDATE_SAMPLER_VERSION,
        "candidate_source": "firms-only",
        "candidate_selection_reason": f"firms-eligible-seed-square-radius-{radius_cells}km",
        "candidate_seed_detection_ids": seed_ids,
        "candidate_seed_detection_id_count": len(all_seed_ids),
        "candidate_seed_detection_ids_truncated": int(
            len(all_seed_ids) > MAX_RETAINED_SEED_DETECTION_IDS
        ),
        "candidate_seed_detection_ids_sha256": hashlib.sha256(
            "\x1f".join(all_seed_ids).encode("utf-8")
        ).hexdigest(),
        "candidate_seed_raw_artifact_ids": raw_ids,
        "candidate_seed_count": len(seed_ids),
        "firms_feature_policy": {
            "lookback_hours": lookback.total_seconds() / 3_600,
            "availability_lag_minutes": availability_lag.total_seconds() / 60,
        },
    }
    if positive_label is not None:
        return {
            **dict(positive_label),
            **common,
            "target_newly_burned_12h": 1,
            "binary_training_eligible": 1,
            "binary_training_status": "weak-positive-within-firms-candidate-support",
        }
    return {
        **common,
        "target_newly_burned_12h": 0,
        "label_status": WEAK_NEGATIVE_PROXY_STATUS,
        "label_observability": WEAK_NEGATIVE_PROXY_OBSERVABILITY,
        "label_tier": "weak_negative_proxy",
        "label_quality_score": 0.0,
        "binary_training_eligible": 1,
        "binary_training_status": "weak-negative-proxy-not-clear-no-burn",
    }


def _validated_positive_label(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise CandidateSamplingError("each positive label must be a mapping")
    if record.get("target_newly_burned_12h") != 1 or isinstance(record.get("target_newly_burned_12h"), bool):
        raise CandidateSamplingError("candidate sampler accepts explicit FEDS positives only")
    for key in ("cell_id", "source_snapshot_time", "anchor_at", "target_end_at", "example_id"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise CandidateSamplingError(f"positive label has no {key}")
    try:
        cell = cell_from_id(record["cell_id"])
    except Exception as exc:
        raise CandidateSamplingError("positive label cell_id is not canonical") from exc
    source_snapshot_time = _parse_utc(record["source_snapshot_time"], "source_snapshot_time")
    anchor_at = _parse_utc(record["anchor_at"], "anchor_at")
    if anchor_at != _anchor_for_cell(source_snapshot_time, cell):
        raise CandidateSamplingError("positive label anchor_at does not match its FEDS local-solar cutoff")
    key = TrainingExampleKey(cell_id=cell.cell_id, anchor_at=anchor_at)
    if record["example_id"] != key.example_id:
        raise CandidateSamplingError("positive label example_id does not match cell_id and anchor_at")
    if _parse_utc(record["target_end_at"], "target_end_at") != key.target_end_at:
        raise CandidateSamplingError("positive label target_end_at is not 12 hours after anchor_at")
    return dict(record)


def _validated_detection(record: Mapping[str, Any]) -> _Detection:
    if not isinstance(record, Mapping):
        raise CandidateSamplingError("each FIRMS detection must be a mapping")
    detection_id = record.get("detection_id")
    if not isinstance(detection_id, str) or not detection_id.strip():
        raise CandidateSamplingError("FIRMS detection has no detection_id")
    try:
        cell = cell_from_wgs84(latitude=float(record["latitude"]), longitude=float(record["longitude"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateSamplingError("FIRMS detection has invalid latitude/longitude") from exc
    raw_id = record.get("raw_artifact_id")
    if raw_id is None and isinstance(record.get("provenance"), Mapping):
        raw_id = record["provenance"].get("raw_artifact_id")
    return _Detection(
        detection_id=detection_id.strip(),
        acquired_at=_parse_utc(record.get("acquired_at"), "acquired_at"),
        cell=cell,
        raw_artifact_id=raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else None,
    )


def _anchor_for_cell(snapshot_time: datetime, cell: GridCell) -> datetime:
    _latitude, longitude = cell.center_wgs84
    return estimate_feds_observation_at(snapshot_time, longitude=longitude)


def _eligible(detection: _Detection, *, cutoff_at: datetime, lookback: timedelta, availability_lag: timedelta) -> bool:
    return cutoff_at - lookback <= detection.acquired_at and detection.acquired_at + availability_lag <= cutoff_at


def _rank(snapshot_time: datetime, cell_id: str) -> str:
    return hashlib.sha256(f"{format_utc(snapshot_time)}|{cell_id}".encode("utf-8")).hexdigest()


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise CandidateSamplingError(f"{label} must be an ISO-8601 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CandidateSamplingError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CandidateSamplingError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _nonnegative_duration(value: timedelta, label: str) -> timedelta:
    if not isinstance(value, timedelta) or value < timedelta(0):
        raise CandidateSamplingError(f"{label} must be a non-negative timedelta")
    return value
