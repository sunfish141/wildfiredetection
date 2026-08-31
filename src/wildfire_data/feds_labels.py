"""Build leakage-aware 1 km satellite-weak labels from retained FEDS snapshots.

The FEDS API supplies cumulative perimeters, rather than operational burn
progression.  For a fire with consecutive source snapshots ``t`` and
``t + 12h``, this module emits only cells intersecting
``perimeter(t + 12h) - perimeter(t)``.  They are positive observations, not
proof that every other cell was observed clear and unburned.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from pyproj import Transformer
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union

from .data_archive import CoverageLedger, CoverageStatus
from .feds_collection import (
    DEFAULT_REGION_LABEL,
    DEFAULT_REGION_NAMES,
    FEDS_LABEL_QUALITY_SCORE,
    FEDS_NORMALIZATION_PARTITION,
    FEDS_SNAPSHOT_INTERVAL,
    FEDS_SOURCE_TIME_SEMANTICS,
    _observed_snapshot_expected_coverage_id,
    iter_feds_snapshot_windows,
)
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .storage_budget import StorageBudgetError, StorageBudgetPolicy, require_admission
from .training_grid import (
    DEFAULT_CELL_SIZE_METRES,
    TRAINING_GRID_CRS,
    TrainingExampleKey,
    cell_from_id,
    format_utc,
)


FEDS_LABEL_SCHEMA_VERSION = 1
FEDS_LABEL_BUILD_VERSION = "feds-perimeter-difference-1km/v3-primarykey-observed"
FEDS_TIME_ALIGNMENT_ESTIMATED_LOCAL_SOLAR_UTC = "estimated-local-solar-to-utc/v1"
FEDS_TIME_ALIGNMENT_NOMINAL_UTC = "nominal-source-t-as-utc/v1"
DEFAULT_TIME_ALIGNMENT_MODE = FEDS_TIME_ALIGNMENT_ESTIMATED_LOCAL_SOLAR_UTC
FEDS_APPROXIMATE_OVERPASS_HOUR_LOCAL_SOLAR = 1.5
DEFAULT_POSITIVE_OVERLAP_FRACTION = 0.10
DEFAULT_ELIGIBLE_REGIONS = DEFAULT_REGION_NAMES
_COMPONENT_PATTERN = re.compile(r"[^a-z0-9]+")


class FedsLabelError(ValueError):
    """Raised when FEDS source evidence cannot produce a safe weak label."""


@dataclass(frozen=True)
class FedsLabelBuildReport:
    """One source-window label-building outcome."""

    source_snapshot_time: datetime
    target_snapshot_time: datetime
    status: CoverageStatus
    paired_fire_count: int
    positive_cell_count: int
    missing_current_coverage: bool
    missing_future_coverage: bool
    raw_artifact_ids: tuple[str, ...]
    normalized_artifact: NormalizedArtifact | None = None


def estimate_feds_observation_at(source_snapshot_time: datetime, *, longitude: float) -> datetime:
    """Estimate the physical UTC overpass from FEDS' local-solar timestamp.

    FEDS stores the UTC *date* with a local-solar 00:00/12:00 wall-clock time.
    Its VIIRS observations occur at approximately 01:30/13:30 local solar
    time.  This conversion is intentionally an estimate, and label records
    retain both it and the unmodified source timestamp.
    """
    nominal = _as_utc(source_snapshot_time, "source_snapshot_time")
    try:
        longitude_value = float(longitude)
    except (TypeError, ValueError) as exc:
        raise FedsLabelError("longitude must be numeric") from exc
    if not math.isfinite(longitude_value) or not -180 <= longitude_value <= 180:
        raise FedsLabelError("longitude must be finite and between -180 and 180")
    utc_offset_hours = FEDS_APPROXIMATE_OVERPASS_HOUR_LOCAL_SOLAR - longitude_value / 15.0
    return nominal + timedelta(hours=utc_offset_hours)


def build_feds_weak_positive_labels(
    current_records: Iterable[Mapping[str, Any]],
    future_records: Iterable[Mapping[str, Any]],
    *,
    source_snapshot_time: datetime,
    time_alignment_mode: str = DEFAULT_TIME_ALIGNMENT_MODE,
    positive_overlap_fraction: float = DEFAULT_POSITIVE_OVERLAP_FRACTION,
    eligible_regions: Iterable[str] = DEFAULT_ELIGIBLE_REGIONS,
) -> tuple[list[dict[str, Any]], int]:
    """Return deduplicated positive labels and the number of paired fires.

    Both inputs must be source-faithful normalized FEDS records for exactly
    consecutive snapshots.  The result purposefully contains no zero labels:
    a satellite no-detection is not global no-burn evidence without a paired
    observability product.  Candidate generation later creates explicitly
    marked weak negatives inside a defined prediction window.
    """
    source_time = _as_utc(source_snapshot_time, "source_snapshot_time")
    target_time = source_time + FEDS_SNAPSHOT_INTERVAL
    mode = _validated_time_alignment_mode(time_alignment_mode)
    threshold = _validated_overlap_fraction(positive_overlap_fraction)
    regions = frozenset(_validated_regions(eligible_regions))
    current = _snapshot_by_fire(current_records, expected_time=source_time, eligible_regions=regions)
    future = _snapshot_by_fire(future_records, expected_time=target_time, eligible_regions=regions)
    labels_by_key: dict[tuple[str, datetime], dict[str, Any]] = {}
    paired_fire_count = 0
    for fire_key in sorted(set(current).intersection(future)):
        current_record = current[fire_key]
        future_record = future[fire_key]
        current_geometry = feds_record_geometry(current_record)
        future_geometry = feds_record_geometry(future_record)
        newly_burned = _polygonal_geometry(future_geometry.difference(current_geometry))
        paired_fire_count += 1
        if newly_burned.is_empty:
            continue
        for cell_id, overlap_fraction in rasterize_positive_cells(
            newly_burned,
            positive_overlap_fraction=threshold,
        ):
            cell = cell_from_id(cell_id)
            latitude, longitude = cell.center_wgs84
            anchor_at = _label_anchor_at(source_time, longitude=longitude, mode=mode)
            example_key = TrainingExampleKey(cell_id=cell_id, anchor_at=anchor_at)
            key = (cell_id, anchor_at)
            source = _label_source_summary(current_record, future_record)
            existing = labels_by_key.get(key)
            if existing is None:
                labels_by_key[key] = {
                    "schema_version": FEDS_LABEL_SCHEMA_VERSION,
                    "example_id": example_key.example_id,
                    "cell_id": cell_id,
                    "cell_center_latitude": latitude,
                    "cell_center_longitude": longitude,
                    "anchor_at": format_utc(anchor_at),
                    "target_end_at": format_utc(example_key.target_end_at),
                    "target_newly_burned_12h": 1,
                    "label_status": "positive-observed",
                    "label_observability": "satellite-weak-positive-only",
                    "label_tier": "weak_satellite",
                    "label_source": "NASA FEDS",
                    "label_quality_score": FEDS_LABEL_QUALITY_SCORE,
                    "label_build_version": FEDS_LABEL_BUILD_VERSION,
                    "positive_overlap_fraction": overlap_fraction,
                    "source_snapshot_time": format_utc(source_time),
                    "target_snapshot_time": format_utc(target_time),
                    "source_time_semantics": FEDS_SOURCE_TIME_SEMANTICS,
                    "time_alignment_mode": mode,
                    "contributing_fire_count": 1,
                    "contributing_fires": [source],
                }
            else:
                existing["positive_overlap_fraction"] = max(
                    float(existing["positive_overlap_fraction"]), overlap_fraction
                )
                contributing = existing["contributing_fires"]
                if source not in contributing:
                    contributing.append(source)
                    existing["contributing_fire_count"] = len(contributing)
    labels = [labels_by_key[key] for key in sorted(labels_by_key)]
    return labels, paired_fire_count


def feds_record_geometry(record: Mapping[str, Any]) -> BaseGeometry:
    """Convert a retained ``esri-rings-wgs84/v1`` FEDS geometry to NA Albers."""
    geometry = record.get("geometry")
    if not isinstance(geometry, Mapping):
        raise FedsLabelError("FEDS record has no geometry")
    if geometry.get("encoding") != "esri-rings-wgs84/v1":
        raise FedsLabelError("FEDS record geometry is not esri-rings-wgs84/v1")
    rings = geometry.get("rings")
    if not isinstance(rings, list):
        raise FedsLabelError("FEDS geometry has no rings")
    wgs84 = esri_rings_to_polygonal_geometry(rings)
    projected = transform(_to_training_grid().transform, wgs84)
    return _polygonal_geometry(projected)


def esri_rings_to_polygonal_geometry(rings: list[Any]) -> BaseGeometry:
    """Convert ArcGIS polygon rings without assuming GeoJSON ring ordering.

    ArcGIS represents outer rings, holes, and disjoint parts in one flat list.
    Ring nesting gives a robust topology even when a provider does not preserve
    orientation consistently.  Nested islands (depth 2) become new polygons.
    """
    ring_polygons = []
    for ring in rings:
        try:
            coordinates = _ring_coordinates(ring)
        except FedsLabelError:
            # FEDS occasionally includes a degenerate auxiliary ring beside a
            # valid outer perimeter (for example, a two-point sliver).  It
            # cannot carry burn area, so preserve the raw record but exclude
            # only that unusable ring from this derived geometry.
            continue
        polygon = _polygonal_geometry(Polygon(coordinates))
        if polygon.is_empty or polygon.area <= 0:
            continue
        # A malformed ring can become multiple polygons after validation; each
        # component is a separate candidate in the containment hierarchy.
        ring_polygons.extend(_polygon_components(polygon))
    if not ring_polygons:
        raise FedsLabelError("FEDS geometry contains no usable polygon rings")
    parents: dict[int, int | None] = {}
    for index, polygon in enumerate(ring_polygons):
        point = polygon.representative_point()
        containing = [
            candidate
            for candidate, outer in enumerate(ring_polygons)
            if candidate != index and outer.area > polygon.area and outer.contains(point)
        ]
        parents[index] = min(containing, key=lambda candidate: ring_polygons[candidate].area) if containing else None
    depths: dict[int, int] = {}
    for index in range(len(ring_polygons)):
        depth = 0
        parent = parents[index]
        visited = {index}
        while parent is not None:
            if parent in visited:
                raise FedsLabelError("FEDS polygon ring containment is cyclic")
            visited.add(parent)
            depth += 1
            parent = parents[parent]
        depths[index] = depth
    polygon_parts = []
    for index, outer in enumerate(ring_polygons):
        if depths[index] % 2:
            continue
        holes = [
            list(ring_polygons[child].exterior.coords)
            for child, parent in parents.items()
            if parent == index and depths[child] == depths[index] + 1
        ]
        polygon_parts.extend(_polygon_components(_polygonal_geometry(Polygon(outer.exterior.coords, holes))))
    return _polygonal_geometry(unary_union(polygon_parts))


def rasterize_positive_cells(
    geometry: BaseGeometry,
    *,
    positive_overlap_fraction: float = DEFAULT_POSITIVE_OVERLAP_FRACTION,
) -> tuple[tuple[str, float], ...]:
    """Return 1 km cells with enough newly-burned area to be weak positives."""
    threshold = _validated_overlap_fraction(positive_overlap_fraction)
    polygonal = _polygonal_geometry(geometry)
    if polygonal.is_empty:
        return ()
    xmin, ymin, xmax, ymax = polygonal.bounds
    size = DEFAULT_CELL_SIZE_METRES
    x_start = math.floor(xmin / size)
    y_start = math.floor(ymin / size)
    x_end = math.ceil(xmax / size) - 1
    y_end = math.ceil(ymax / size) - 1
    if x_end < x_start or y_end < y_start:
        return ()
    values = []
    cell_area = float(size * size)
    for y_index in range(y_start, y_end + 1):
        for x_index in range(x_start, x_end + 1):
            cell_geometry = box(
                x_index * size,
                y_index * size,
                (x_index + 1) * size,
                (y_index + 1) * size,
            )
            overlap_fraction = polygonal.intersection(cell_geometry).area / cell_area
            if overlap_fraction + 1e-12 >= threshold:
                cell_id = f"naea-1km:x={x_index}:y={y_index}"
                values.append((cell_id, float(min(1.0, overlap_fraction))))
    return tuple(values)


def load_feds_snapshot_records(
    data_root: str | Path,
    *,
    source_snapshot_time: datetime,
    region_label: str = DEFAULT_REGION_LABEL,
) -> list[dict[str, Any]]:
    """Load one selected FEDS source snapshot without materializing all history.

    A raw FEDS capture can be replayed more than once as the provider revises
    its NRT perimeters.  The observed-snapshot coverage ledger records the
    normalized artifact selected by each successful replay.  Prefer that
    explicit artifact rather than globbing every immutable revision for a
    snapshot, which would manufacture a conflicting source state.

    Older archives without that selection metadata retain the original
    manifest-free fallback so they remain readable.
    """
    source_time = _as_utc(source_snapshot_time, "source_snapshot_time")
    token = _storage_component(format_utc(source_time))
    # The raw replay writes one combined source-snapshot artifact.  Live
    # collection may additionally retain page-level artifacts, so support
    # both deterministic partition layouts without scanning full history.
    root = Path(data_root) / "normalized" / "fire-progression"
    end_token = _storage_component(format_utc(source_time + FEDS_SNAPSHOT_INTERVAL))
    patterns = (
        root
        / f"normalization-version={FEDS_NORMALIZATION_PARTITION}"
        / f"snapshot-end={end_token}"
        / f"snapshot-start={token}"
        / "source=feds-nrt-perimeters"
        / "*.jsonl.gz",
        root
        / f"normalization-version={FEDS_NORMALIZATION_PARTITION}"
        / "page=*"
        / f"snapshot-end={end_token}"
        / f"snapshot-start={token}"
        / "source=feds-nrt-perimeters"
        / "*.jsonl.gz",
    )
    selected_artifact_id = _selected_normalized_snapshot_artifact_id(
        data_root,
        source_time=source_time,
        region_label=region_label,
    )
    if selected_artifact_id is not None:
        paths = tuple(pattern.parent / f"{selected_artifact_id}.jsonl.gz" for pattern in patterns)
        paths = tuple(path for path in paths if path.is_file())
        if len(paths) != 1:
            raise FedsLabelError(
                "selected FEDS normalized artifact is unavailable or ambiguous for "
                f"{format_utc(source_time)}: {selected_artifact_id}"
            )
    else:
        paths = tuple(
            sorted({path for pattern in patterns for path in root.glob(str(pattern.relative_to(root)))})
        )
    records = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                if (
                    record.get("record_type") == "feds_perimeter_snapshot"
                    and record.get("source_snapshot_time") == format_utc(source_time)
                ):
                    records.append(record)
    return records


def _selected_normalized_snapshot_artifact_id(
    data_root: str | Path,
    *,
    source_time: datetime,
    region_label: str,
) -> str | None:
    """Return the newest successful replay's selected artifact for one snapshot."""
    expected_coverage_id = _observed_snapshot_expected_coverage_id(source_time, region_label)
    for record in reversed(CoverageLedger(data_root).entries()):
        if record.expected_coverage_id != expected_coverage_id or record.status is not CoverageStatus.COMPLETE:
            continue
        try:
            document = json.loads(record.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FedsLabelError(f"could not read FEDS snapshot coverage record: {record.path}") from exc
        detail = document.get("detail")
        artifact_id = detail.get("normalized_artifact_id") if isinstance(detail, Mapping) else None
        if artifact_id is None:
            continue
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            raise FedsLabelError(
                "FEDS snapshot coverage has an invalid normalized artifact identity for "
                f"{format_utc(source_time)}"
            )
        return artifact_id
    return None


def build_and_store_feds_weak_labels(
    data_root: str | Path,
    *,
    start_date: date,
    end_date: date,
    storage_budget: StorageBudgetPolicy,
    region_label: str = DEFAULT_REGION_LABEL,
    time_alignment_mode: str = DEFAULT_TIME_ALIGNMENT_MODE,
    positive_overlap_fraction: float = DEFAULT_POSITIVE_OVERLAP_FRACTION,
    eligible_regions: Iterable[str] = DEFAULT_ELIGIBLE_REGIONS,
    generated_at: datetime | None = None,
    refresh: bool = False,
) -> tuple[FedsLabelBuildReport, ...]:
    """Derive/store positive weak-label cells, one source window at a time.

    This is intentionally streaming by 12-hour source snapshots.  Even when
    the raw FEDS archive grows, it holds only two snapshot batches in memory.
    It requires actual v2 primary-key snapshot partitions.  A missing side of
    a comparison stays incomplete rather than becoming an empty target; FEDS
    is a positive-only satellite source, not a global no-fire observation.
    """
    if end_date < start_date:
        raise FedsLabelError("end_date must not be before start_date")
    mode = _validated_time_alignment_mode(time_alignment_mode)
    threshold = _validated_overlap_fraction(positive_overlap_fraction)
    regions = _validated_regions(eligible_regions)
    generated = _as_utc(generated_at or datetime.now(timezone.utc), "generated_at")
    ledger = CoverageLedger(data_root)
    latest = {
        record.expected_coverage_id: record
        for record in ledger.entries()
        if record.expected_coverage_id is not None
    }
    reports = []
    for source_time, target_time in iter_feds_snapshot_windows(start_date, end_date):
        label_expected_id = _label_expected_coverage_id(source_time, region_label, mode, threshold)
        existing_label = latest.get(label_expected_id)
        # Only an actual positive-label artifact is terminal.  An old
        # positive-only build that wrote `empty-confirmed` cannot suppress a
        # later rebuild once more source snapshots are available.
        if not refresh and existing_label is not None and existing_label.status is CoverageStatus.COMPLETE:
            reports.append(
                FedsLabelBuildReport(
                    source_snapshot_time=source_time,
                    target_snapshot_time=target_time,
                    status=existing_label.status,
                    paired_fire_count=0,
                    positive_cell_count=0,
                    missing_current_coverage=False,
                    missing_future_coverage=False,
                    raw_artifact_ids=(),
                    normalized_artifact=None,
                )
            )
            continue
        current_records = load_feds_snapshot_records(
            data_root,
            source_snapshot_time=source_time,
            region_label=region_label,
        )
        future_records = load_feds_snapshot_records(
            data_root,
            source_snapshot_time=target_time,
            region_label=region_label,
        )
        current_coverage = latest.get(
            _observed_snapshot_expected_coverage_id(source_time, region_label)
        )
        future_coverage = latest.get(
            _observed_snapshot_expected_coverage_id(target_time, region_label)
        )
        missing_current = not current_records or (
            current_coverage is not None and current_coverage.status is not CoverageStatus.COMPLETE
        )
        missing_future = not future_records or (
            future_coverage is not None and future_coverage.status is not CoverageStatus.COMPLETE
        )
        available_records = current_records + future_records
        available_raw_artifact_ids = _raw_artifact_ids_from_records(available_records)
        if missing_current or missing_future:
            coverage = _record_label_coverage(
                data_root,
                source_time=source_time,
                target_time=target_time,
                region_label=region_label,
                expected_id=label_expected_id,
                status=CoverageStatus.PARTIAL,
                artifact_ids=list(available_raw_artifact_ids),
                detail={
                    "reason": "no-observed-primarykey-snapshot",
                    "missing_current_coverage": missing_current,
                    "missing_future_coverage": missing_future,
                    "current_primarykey_snapshot_status": (
                        current_coverage.status.value if current_coverage is not None else None
                    ),
                    "future_primarykey_snapshot_status": (
                        future_coverage.status.value if future_coverage is not None else None
                    ),
                    "time_alignment_mode": mode,
                    "positive_overlap_fraction": threshold,
                },
                generated_at=generated,
            )
            reports.append(
                FedsLabelBuildReport(
                    source_time,
                    target_time,
                    coverage.status,
                    0,
                    0,
                    missing_current,
                    missing_future,
                    available_raw_artifact_ids,
                )
            )
            continue
        raw_artifact_ids = available_raw_artifact_ids
        try:
            labels, paired_fire_count = build_feds_weak_positive_labels(
                current_records,
                future_records,
                source_snapshot_time=source_time,
                time_alignment_mode=mode,
                positive_overlap_fraction=threshold,
                eligible_regions=regions,
            )
        except FedsLabelError as exc:
            coverage = _record_label_coverage(
                data_root,
                source_time=source_time,
                target_time=target_time,
                region_label=region_label,
                expected_id=label_expected_id,
                status=CoverageStatus.PARTIAL,
                artifact_ids=list(raw_artifact_ids),
                detail={
                    "reason": "conflicting-primarykey-source-revisions",
                    "time_alignment_mode": mode,
                    "positive_overlap_fraction": threshold,
                },
                error=str(exc),
                generated_at=generated,
            )
            reports.append(
                FedsLabelBuildReport(
                    source_time,
                    target_time,
                    coverage.status,
                    0,
                    0,
                    False,
                    False,
                    raw_artifact_ids,
                )
            )
            continue
        artifact = None
        if labels:
            estimate = _conservative_label_bytes(labels)
            try:
                require_admission(
                    storage_budget,
                    data_root,
                    category="derived_training_views",
                    requested_bytes=estimate,
                )
            except StorageBudgetError as exc:
                coverage = _record_label_coverage(
                    data_root,
                    source_time=source_time,
                    target_time=target_time,
                    region_label=region_label,
                    expected_id=label_expected_id,
                    status=CoverageStatus.PARTIAL,
                    artifact_ids=list(raw_artifact_ids),
                    detail={"paired_fire_count": paired_fire_count, "positive_cell_count": len(labels)},
                    error=str(exc),
                    generated_at=generated,
                )
                reports.append(
                    FedsLabelBuildReport(
                        source_time,
                        target_time,
                        coverage.status,
                        paired_fire_count,
                        len(labels),
                        False,
                        False,
                        raw_artifact_ids,
                    )
                )
                continue
            artifact = write_normalized_jsonl(
                data_root,
                entity="training-labels",
                records=labels,
                partitions={
                    "source": "feds-perimeter-difference",
                    "source_snapshot": format_utc(source_time),
                    "target_snapshot": format_utc(target_time),
                    "grid": "naea-1km",
                },
                raw_artifact_ids=raw_artifact_ids,
                transformation_version=FEDS_LABEL_BUILD_VERSION,
                generated_at=generated,
            )
        # With a positive-only satellite source, zero rasterized cells are not
        # a negative/no-spread target.  Preserve the comparison provenance but
        # leave the window partial for future stronger observability labels.
        status = CoverageStatus.COMPLETE if labels else CoverageStatus.PARTIAL
        artifact_ids = list(raw_artifact_ids)
        coverage = _record_label_coverage(
            data_root,
            source_time=source_time,
            target_time=target_time,
            region_label=region_label,
            expected_id=label_expected_id,
            status=status,
            artifact_ids=artifact_ids,
            detail={
                "paired_fire_count": paired_fire_count,
                "positive_cell_count": len(labels),
                "normalized_artifact_id": artifact.normalized_artifact_id if artifact else None,
                "label_tier": "weak_satellite",
                "label_status": "positive-observed",
                "reason": (
                    None
                    if labels
                    else (
                        "no-common-fire-across-observed-snapshots"
                        if paired_fire_count == 0
                        else "no-positive-expansion-from-positive-only-source"
                    )
                ),
                "no_negative_or_no_spread_inferred": not bool(labels),
                "time_alignment_mode": mode,
                "positive_overlap_fraction": threshold,
                "label_build_version": FEDS_LABEL_BUILD_VERSION,
            },
            generated_at=generated,
        )
        reports.append(
            FedsLabelBuildReport(
                source_time,
                target_time,
                coverage.status,
                paired_fire_count,
                len(labels),
                False,
                False,
                raw_artifact_ids,
                artifact,
            )
        )
    return tuple(reports)


def _snapshot_by_fire(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_time: datetime,
    eligible_regions: frozenset[str],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    expected = format_utc(expected_time)
    values: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        if record.get("record_type") != "feds_perimeter_snapshot":
            continue
        if record.get("source_snapshot_time") != expected:
            continue
        if record.get("region") not in eligible_regions:
            continue
        if record.get("time_alignment_eligible") is False:
            continue
        region = record.get("region")
        fire_id = record.get("fire_id")
        if not isinstance(region, str) or fire_id is None:
            raise FedsLabelError("FEDS record has no region/fire_id identity")
        key = (region, str(fire_id))
        existing = values.get(key)
        if existing is not None and _record_fingerprint(existing) != _record_fingerprint(record):
            raise FedsLabelError(
                "FEDS source contains conflicting revisions for the same fire/snapshot; "
                "select one explicit source revision before label generation"
            )
        values[key] = record
    return values


def _record_fingerprint(record: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        record.get("source_record_id"),
        json.dumps(record.get("geometry"), sort_keys=True, separators=(",", ":")),
    )


def _raw_artifact_ids_from_records(records: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    raw_ids: set[str] = set()
    for record in records:
        raw_artifact_id = record.get("raw_artifact_id")
        if isinstance(raw_artifact_id, str) and raw_artifact_id:
            raw_ids.add(raw_artifact_id)
        equivalent = record.get("equivalent_raw_artifact_ids")
        if isinstance(equivalent, list):
            raw_ids.update(value for value in equivalent if isinstance(value, str) and value)
    return tuple(sorted(raw_ids))


def _label_source_summary(current: Mapping[str, Any], future: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "region": current.get("region"),
        "fire_id": current.get("fire_id"),
        "current_source_record_id": current.get("source_record_id"),
        "future_source_record_id": future.get("source_record_id"),
        "current_raw_artifact_id": current.get("raw_artifact_id"),
        "future_raw_artifact_id": future.get("raw_artifact_id"),
        "future_n_newpixels": _number_or_none(_source_field(future, "n_newpixels")),
        "future_active_front_length_km": _number_or_none(_source_field(future, "flinelen")),
    }


def _source_field(record: Mapping[str, Any], name: str) -> object:
    fields = record.get("source_fields")
    return fields.get(name) if isinstance(fields, Mapping) else None


def _number_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _label_anchor_at(source_time: datetime, *, longitude: float, mode: str) -> datetime:
    if mode == FEDS_TIME_ALIGNMENT_NOMINAL_UTC:
        return source_time
    return estimate_feds_observation_at(source_time, longitude=longitude)


def _ring_coordinates(ring: Any) -> list[tuple[float, float]]:
    if not isinstance(ring, list) or len(ring) < 3:
        raise FedsLabelError("FEDS polygon ring must have at least three points")
    coordinates = []
    for coordinate in ring:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
            raise FedsLabelError("FEDS polygon coordinate is invalid")
        try:
            longitude = float(coordinate[0])
            latitude = float(coordinate[1])
        except (TypeError, ValueError) as exc:
            raise FedsLabelError("FEDS polygon coordinate is not numeric") from exc
        if not all(math.isfinite(value) for value in (longitude, latitude)):
            raise FedsLabelError("FEDS polygon coordinate is not finite")
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise FedsLabelError("FEDS polygon coordinate is outside WGS84 bounds")
        coordinates.append((longitude, latitude))
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    return coordinates


def _polygonal_geometry(value: BaseGeometry) -> BaseGeometry:
    repaired = make_valid(value) if not value.is_valid else value
    components = _polygon_components(repaired)
    if not components:
        return GeometryCollection()
    return unary_union(components)


def _polygon_components(value: BaseGeometry) -> list[Polygon]:
    if value.is_empty:
        return []
    if isinstance(value, Polygon):
        return [value]
    if isinstance(value, MultiPolygon):
        return list(value.geoms)
    if isinstance(value, GeometryCollection):
        result: list[Polygon] = []
        for component in value.geoms:
            result.extend(_polygon_components(component))
        return result
    return []


def _to_training_grid() -> Transformer:
    return Transformer.from_crs("EPSG:4326", TRAINING_GRID_CRS, always_xy=True)


def _terminal_feds_window(record) -> bool:
    return record is not None and record.status in {
        CoverageStatus.COMPLETE,
        CoverageStatus.EMPTY_CONFIRMED,
    }


def _record_label_coverage(
    data_root: str | Path,
    *,
    source_time: datetime,
    target_time: datetime,
    region_label: str,
    expected_id: str,
    status: CoverageStatus,
    artifact_ids: list[str],
    detail: Mapping[str, Any],
    generated_at: datetime,
    error: str | None = None,
):
    return CoverageLedger(data_root).record(
        source="wildfire-data training pipeline",
        product="feds-weak-labels",
        coverage_start=source_time,
        coverage_end=target_time,
        region=region_label,
        expected_coverage_id=expected_id,
        status=status,
        artifact_sha256s=artifact_ids,
        detail=dict(detail),
        error=error,
        recorded_at=generated_at,
    )


def _label_expected_coverage_id(
    source_time: datetime,
    region_label: str,
    time_alignment_mode: str,
    positive_overlap_fraction: float,
) -> str:
    return (
        "feds-weak-labels:"
        f"{FEDS_LABEL_BUILD_VERSION}:{region_label}:{format_utc(source_time)}:{time_alignment_mode}:"
        f"overlap={positive_overlap_fraction:.6f}"
    )


def _conservative_label_bytes(records: Iterable[Mapping[str, Any]]) -> int:
    encoded = sum(
        len(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")) + 1
        for record in records
    )
    return encoded * 2 + 65_536


def _validated_time_alignment_mode(value: str) -> str:
    if value not in {
        FEDS_TIME_ALIGNMENT_ESTIMATED_LOCAL_SOLAR_UTC,
        FEDS_TIME_ALIGNMENT_NOMINAL_UTC,
    }:
        raise FedsLabelError("unsupported FEDS time alignment mode")
    return value


def _validated_overlap_fraction(value: float) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise FedsLabelError("positive_overlap_fraction must be numeric") from exc
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise FedsLabelError("positive_overlap_fraction must be in (0, 1]")
    return fraction


def _validated_regions(values: Iterable[str]) -> tuple[str, ...]:
    regions = tuple(value.strip() for value in values if isinstance(value, str) and value.strip())
    if not regions:
        raise FedsLabelError("eligible_regions must not be empty")
    return regions


def _storage_component(value: str) -> str:
    return _COMPONENT_PATTERN.sub("-", value.lower()).strip("-")


def _as_utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise FedsLabelError(f"{label} must be an offset-aware datetime")
    return value.astimezone(timezone.utc)
