"""Build and publish a coherent, no-weather weak-label wildfire dataset.

The source archive intentionally keeps immutable evidence in many normalized
partitions.  This module is the boundary that turns one *completed* positive
training view into a single, manifest-selected candidate table suitable for
upload and a first weak-label experiment.

The resulting ``target_newly_burned_12h=0`` rows are not clear/no-burn
observations.  They are explicitly named FIRMS-seeded weak-negative proxies.
Weather is deliberately absent because the retained Open-Meteo exports cannot
prove forecast availability at an example's cutoff.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .candidate_sampling import (
    CANDIDATE_SAMPLER_VERSION,
    CandidateSamplingError,
    build_firms_only_candidates,
)
from .data_archive import CoverageLedger, CoverageRecord, CoverageStatus, write_atomic_json
from .fire_state_features import FireStateFeatureError, build_firms_fire_state_features
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .storage_budget import StorageBudgetPolicy, require_admission
from .terrain_features import TerrainFeatureError, TerrainFeatureSampler
from .training_dataset import (
    DEFAULT_FIRMS_AVAILABILITY_LAG,
    DEFAULT_FIRMS_LOOKBACK,
    DEFAULT_FIRMS_PRODUCTS,
    DEFAULT_FIRMS_REGION,
    TRAINING_DATASET_BUILD_VERSION,
    TrainingDatasetError,
    _selected_training_dataset_manifest,
    iter_training_examples,
)
from .training_grid import (
    DEFAULT_HORIZON_HOURS,
    GridCell,
    TrainingGridError,
    cell_from_id,
    cell_from_wgs84,
    cells_in_square_radius,
    format_utc,
)


CANDIDATE_DATASET_SCHEMA_VERSION = 1
CANDIDATE_DATASET_BUILD_VERSION = "firms-feds-weak-candidate-tabular-1km-12h/v1"
CANDIDATE_DATASET_MANIFEST_SCHEMA_VERSION = 1
# v2 adds a CSV representation alongside the original JSONL payload.  Keep
# accepting v1 releases when a caller asks to reuse an existing destination:
# historical exports remain immutable and should not have to be regenerated
# just to add an optional convenience format.
CANDIDATE_DATASET_RELEASE_SCHEMA_VERSION = 2
LEGACY_CANDIDATE_DATASET_RELEASE_SCHEMA_VERSION = 1
DEFAULT_CANDIDATE_RADIUS_CELLS = 2
DEFAULT_MAX_WEAK_NEGATIVE_PROXIES_PER_SNAPSHOT = 2_000
# The retained package currently contains 47 compact source blocks.  Keeping
# all of them in this build-only LRU avoids repeatedly decompressing the same
# ETOPO arrays for candidate snapshots scattered across North America.
DEFAULT_TERRAIN_CACHE_BLOCKS = 64
NO_WEATHER_STATUS = "unavailable-no-issued-forecast-features"
NO_WEATHER_POLICY = "exclude-open-meteo-retrospective-exports/v1"
SOURCE_WINDOW_POLICY = "positive-feds-snapshot-windows-only/v1"
SPLIT_POLICY = "chronological-source-snapshot-80-20/v1"
DEFAULT_TRAIN_FRACTION = 0.80

# Explicit model inputs.  Every other field in a row is lineage, a label,
# selection metadata, geometry, timing, or a missingness declaration.
DEFAULT_MODEL_FEATURE_COLUMNS = (
    "firms_center_has_detection",
    "firms_center_detection_count",
    "firms_center_bright_ti4_max",
    "firms_center_bright_ti4_mean",
    "firms_center_platform_count",
    "firms_center_hours_since_last_detection",
    "firms_local_3x3_has_detection",
    "firms_local_3x3_detection_count",
    "firms_local_3x3_bright_ti4_max",
    "firms_local_3x3_bright_ti4_mean",
    "firms_local_3x3_platform_count",
    "firms_local_3x3_hours_since_last_detection",
    "firms_local_3x3_active_cell_count",
    "terrain_valid",
    "terrain_elevation_m",
    "terrain_slope_degrees",
    "terrain_aspect_defined",
    "terrain_aspect_sin",
    "terrain_aspect_cos",
)


class CandidateDatasetError(ValueError):
    """Raised when retained evidence cannot form a coherent candidate view."""


@dataclass(frozen=True)
class CandidateDatasetSnapshotReport:
    """Counts and immutable artifacts produced for one FEDS source snapshot."""

    source_snapshot_time: datetime
    input_positive_count: int
    candidate_row_count: int
    supported_positive_count: int
    weak_negative_proxy_count: int
    unscored_positive_count: int
    candidate_artifact: NormalizedArtifact
    unscored_artifact: NormalizedArtifact | None


@dataclass(frozen=True)
class CandidateDatasetBuildResult:
    """Completed build result selected by one atomic candidate manifest."""

    reports: tuple[CandidateDatasetSnapshotReport, ...]
    manifest_path: Path

    @property
    def candidate_row_count(self) -> int:
        return sum(report.candidate_row_count for report in self.reports)

    @property
    def supported_positive_count(self) -> int:
        return sum(report.supported_positive_count for report in self.reports)

    @property
    def weak_negative_proxy_count(self) -> int:
        return sum(report.weak_negative_proxy_count for report in self.reports)

    @property
    def unscored_positive_count(self) -> int:
        return sum(report.unscored_positive_count for report in self.reports)


@dataclass(frozen=True)
class DatasetRelease:
    """A self-contained, uploadable materialization of one candidate view."""

    directory: Path
    manifest_path: Path
    candidate_row_count: int
    unscored_positive_count: int


def build_and_store_firms_candidate_dataset(
    data_root: str | Path,
    *,
    storage_budget: StorageBudgetPolicy,
    start_date: date,
    end_date: date,
    positive_view_manifest: str | Path | None = None,
    split_start_date: date | None = None,
    split_end_date: date | None = None,
    radius_cells: int = DEFAULT_CANDIDATE_RADIUS_CELLS,
    max_weak_negative_proxies_per_snapshot: int = DEFAULT_MAX_WEAK_NEGATIVE_PROXIES_PER_SNAPSHOT,
    firms_lookback: timedelta = DEFAULT_FIRMS_LOOKBACK,
    firms_availability_lag: timedelta = DEFAULT_FIRMS_AVAILABILITY_LAG,
    firms_products: Sequence[str] = DEFAULT_FIRMS_PRODUCTS,
    firms_region: str = DEFAULT_FIRMS_REGION,
    max_cached_terrain_blocks: int = DEFAULT_TERRAIN_CACHE_BLOCKS,
) -> CandidateDatasetBuildResult:
    """Create an atomic FIRMS-only candidate feature view from retained data.

    The input is *one* completed positive-only training manifest.  The builder
    never globs all historical artifacts and refuses a requested range outside
    that manifest's verified source window.  FIRMS coverage is rechecked for
    every candidate snapshot, so an absent product/day cannot become zero
    fire evidence for a candidate located outside the original positive cells.
    """
    _validate_date_range(start_date, end_date)
    resolved_split_start = split_start_date or start_date
    resolved_split_end = split_end_date or end_date
    _validate_date_range(resolved_split_start, resolved_split_end)
    if resolved_split_start > start_date or resolved_split_end < end_date:
        raise CandidateDatasetError(
            "split source range must include the candidate build source range"
        )
    _validate_nonnegative_int(radius_cells, "radius_cells")
    _validate_nonnegative_int(
        max_weak_negative_proxies_per_snapshot,
        "max_weak_negative_proxies_per_snapshot",
    )
    _validate_nonnegative_int(max_cached_terrain_blocks, "max_cached_terrain_blocks")
    lookback = _nonnegative_duration(firms_lookback, "firms_lookback")
    availability_lag = _nonnegative_duration(
        firms_availability_lag, "firms_availability_lag"
    )
    products = _validated_products(firms_products)
    region = _required_text(firms_region, "firms_region")
    root = Path(data_root).resolve()

    selected_positive_manifest = _selected_training_dataset_manifest(
        root, manifest_path=positive_view_manifest
    )
    if selected_positive_manifest is None:
        raise CandidateDatasetError("no completed positive-only training manifest is available")
    positive_manifest_path = selected_positive_manifest["path"]
    positive_manifest = _read_json(positive_manifest_path, "positive training manifest")
    _require_positive_manifest_range(
        positive_manifest,
        start_date=start_date,
        end_date=end_date,
    )
    _require_positive_manifest_range(
        positive_manifest,
        start_date=resolved_split_start,
        end_date=resolved_split_end,
    )
    split_positive_rows = _selected_positive_rows(
        root,
        manifest_path=positive_manifest_path,
        start_date=resolved_split_start,
        end_date=resolved_split_end,
    )
    positive_rows = tuple(
        row
        for row in split_positive_rows
        if start_date
        <= _parse_utc(row.get("source_snapshot_time"), "source_snapshot_time").date()
        <= end_date
    )
    if not positive_rows:
        raise CandidateDatasetError(
            "the selected positive training manifest contains no rows in the requested source range"
        )
    labels_by_snapshot = _group_rows_by_source_snapshot(positive_rows)
    split_labels_by_snapshot = _group_rows_by_source_snapshot(split_positive_rows)
    snapshot_splits = _chronological_snapshot_splits(tuple(sorted(split_labels_by_snapshot)))
    latest_firms_coverage = _latest_firms_coverage(CoverageLedger(root))
    sampler = TerrainFeatureSampler(root, max_cached_blocks=max_cached_terrain_blocks)
    reports: list[CandidateDatasetSnapshotReport] = []

    for snapshot_time, positive_rows_for_snapshot in sorted(labels_by_snapshot.items()):
        required_dates = _candidate_firms_dates(snapshot_time)
        _require_terminal_firms_coverage(
            snapshot_time=snapshot_time,
            dates=required_dates,
            products=products,
            region=region,
            latest_by_expected_id=latest_firms_coverage,
        )
        detections = tuple(_iter_firms_detections(root, dates=required_dates))
        try:
            sampled = build_firms_only_candidates(
                positive_rows_for_snapshot,
                detections,
                radius_cells=radius_cells,
                max_weak_negative_proxies_per_snapshot=max_weak_negative_proxies_per_snapshot,
                firms_lookback=lookback,
                firms_availability_lag=availability_lag,
            )
        except CandidateSamplingError as exc:
            raise CandidateDatasetError(
                f"could not sample FIRMS candidates for {format_utc(snapshot_time)}: {exc}"
            ) from exc
        context_label_raw_ids = _snapshot_label_raw_artifact_ids(positive_rows_for_snapshot)
        indexed_detections = _index_detections_by_cell(detections)
        candidate_rows = [
            _assemble_candidate_feature_row(
                row,
                detections_by_cell=indexed_detections,
                terrain_sampler=sampler,
                firms_lookback=lookback,
                firms_availability_lag=availability_lag,
                snapshot_label_raw_artifact_ids=context_label_raw_ids,
                split=snapshot_splits[snapshot_time],
            )
            for row in sampled.candidate_rows
        ]
        if not candidate_rows:
            # A source snapshot with only unscored positives is still an
            # important diagnostic, but cannot form a fit-ready artifact.
            raise CandidateDatasetError(
                "FIRMS-only candidate support is empty for source snapshot "
                f"{format_utc(snapshot_time)}; no binary rows can be published"
            )
        unscored_rows = [
            _assemble_unscored_positive_row(
                row,
                snapshot_label_raw_artifact_ids=context_label_raw_ids,
                split=snapshot_splits[snapshot_time],
            )
            for row in sampled.unscored_positive_rows
        ]
        candidate_artifact = _store_candidate_rows(
            root,
            rows=candidate_rows,
            source_snapshot_time=snapshot_time,
            storage_budget=storage_budget,
        )
        unscored_artifact = (
            _store_unscored_rows(
                root,
                rows=unscored_rows,
                source_snapshot_time=snapshot_time,
                storage_budget=storage_budget,
            )
            if unscored_rows
            else None
        )
        reports.append(
            CandidateDatasetSnapshotReport(
                source_snapshot_time=snapshot_time,
                input_positive_count=len(positive_rows_for_snapshot),
                candidate_row_count=len(candidate_rows),
                supported_positive_count=sampled.positive_candidate_count,
                weak_negative_proxy_count=sampled.weak_negative_proxy_count,
                unscored_positive_count=len(unscored_rows),
                candidate_artifact=candidate_artifact,
                unscored_artifact=unscored_artifact,
            )
        )

    manifest_path = _write_completed_candidate_manifest(
        root,
        reports=tuple(reports),
        input_positive_manifest_path=positive_manifest_path,
        start_date=start_date,
        end_date=end_date,
        split_start_date=resolved_split_start,
        split_end_date=resolved_split_end,
        radius_cells=radius_cells,
        max_weak_negative_proxies_per_snapshot=max_weak_negative_proxies_per_snapshot,
        firms_lookback=lookback,
        firms_availability_lag=availability_lag,
        firms_products=products,
        firms_region=region,
        snapshot_splits=snapshot_splits,
    )
    return CandidateDatasetBuildResult(reports=tuple(reports), manifest_path=manifest_path)


def iter_candidate_examples(
    data_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one completed candidate view, never an arbitrary artifact glob."""
    root = Path(data_root).resolve()
    manifest = _selected_candidate_manifest(root, manifest_path=manifest_path)
    if manifest is None:
        return
    seen_example_ids: set[str] = set()
    for relative_path in manifest["candidate_artifact_relative_paths"]:
        for record in _iter_jsonl_records(root / relative_path, context="candidate example"):
            if record.get("candidate_dataset_build_version") != CANDIDATE_DATASET_BUILD_VERSION:
                raise CandidateDatasetError(
                    "candidate artifact has a mismatched build version: "
                    f"{root / relative_path}"
                )
            example_id = _required_text(record.get("example_id"), "candidate example example_id")
            if example_id in seen_example_ids:
                raise CandidateDatasetError(
                    "duplicate example_id across completed candidate view: " + example_id
                )
            seen_example_ids.add(example_id)
            yield record


def iter_unscored_positive_diagnostics(
    data_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield FIRMS-uncovered positives selected by one completed candidate view."""
    root = Path(data_root).resolve()
    manifest = _selected_candidate_manifest(root, manifest_path=manifest_path)
    if manifest is None:
        return
    for relative_path in manifest["unscored_artifact_relative_paths"]:
        yield from _iter_jsonl_records(root / relative_path, context="unscored positive")


def export_candidate_dataset_release(
    data_root: str | Path,
    output_directory: str | Path,
    *,
    candidate_manifest: str | Path | None = None,
) -> DatasetRelease:
    """Materialize a self-contained, uploadable release from one manifest.

    ``output_directory`` is normally outside ``data/`` so the portable copy
    does not consume the governed local archive budget. Existing output is
    accepted when it was generated from the same candidate build or contains
    byte-equivalent logical records from a repeat build. Different existing
    content is never overwritten.
    """
    root = Path(data_root).resolve()
    manifest = _selected_candidate_manifest(root, manifest_path=candidate_manifest)
    if manifest is None:
        raise CandidateDatasetError("no completed candidate dataset manifest is available")
    destination = Path(output_directory)
    if destination.exists():
        existing = destination / "dataset_manifest.json"
        if existing.is_file():
            document = _read_json(existing, "existing release manifest")
            if _release_is_equivalent(
                root,
                destination,
                document,
                manifest,
            ):
                return DatasetRelease(
                    directory=destination,
                    manifest_path=existing,
                    candidate_row_count=_nonnegative_int(
                        document.get("candidate_row_count"), "candidate_row_count"
                    ),
                    unscored_positive_count=_nonnegative_int(
                        document.get("unscored_positive_count"), "unscored_positive_count"
                    ),
                )
        raise CandidateDatasetError(
            f"release destination already exists with different or invalid content: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    )
    try:
        candidate_path = stage / "candidate_examples.jsonl.gz"
        diagnostic_path = stage / "unscored_positives.jsonl.gz"
        candidate_csv_path = stage / "candidate_examples.csv.gz"
        diagnostic_csv_path = stage / "unscored_positives.csv.gz"
        candidate_count, candidate_content_sha256 = _concatenate_jsonl_gzip(
            root,
            manifest["candidate_artifact_relative_paths"],
            candidate_path,
        )
        unscored_count, unscored_content_sha256 = _concatenate_jsonl_gzip(
            root,
            manifest["unscored_artifact_relative_paths"],
            diagnostic_path,
        )
        (
            candidate_csv_count,
            candidate_csv_content_sha256,
            candidate_csv_columns,
        ) = _concatenate_csv_gzip(
            root,
            manifest["candidate_artifact_relative_paths"],
            candidate_csv_path,
        )
        (
            unscored_csv_count,
            unscored_csv_content_sha256,
            unscored_csv_columns,
        ) = _concatenate_csv_gzip(
            root,
            manifest["unscored_artifact_relative_paths"],
            diagnostic_csv_path,
        )
        if candidate_csv_count != candidate_count or unscored_csv_count != unscored_count:
            raise CandidateDatasetError(
                "CSV release row counts do not match the corresponding JSONL payloads"
            )
        schema = _release_schema(
            candidate_csv_columns=candidate_csv_columns,
            unscored_csv_columns=unscored_csv_columns,
        )
        _write_json_file(stage / "schema.json", schema)
        release_manifest = {
            "schema_version": CANDIDATE_DATASET_RELEASE_SCHEMA_VERSION,
            "kind": "wildfire-spread-candidate-dataset-release",
            "candidate_build_id": manifest["build_id"],
            "candidate_build_manifest_relative_path": manifest["path"].relative_to(root).as_posix(),
            "candidate_dataset_build_version": CANDIDATE_DATASET_BUILD_VERSION,
            "source_snapshot_start_date": manifest["source_snapshot_start_date"],
            "source_snapshot_end_date": manifest["source_snapshot_end_date"],
            "candidate_row_count": candidate_count,
            "unscored_positive_count": unscored_count,
            "candidate_examples_content_sha256": candidate_content_sha256,
            "unscored_positives_content_sha256": unscored_content_sha256,
            "candidate_examples_csv_content_sha256": candidate_csv_content_sha256,
            "unscored_positives_csv_content_sha256": unscored_csv_content_sha256,
            "candidate_examples_csv_columns": list(candidate_csv_columns),
            "unscored_positives_csv_columns": list(unscored_csv_columns),
            "weather": {
                "available": False,
                "status": NO_WEATHER_STATUS,
                "policy": NO_WEATHER_POLICY,
            },
            "model_feature_columns": list(DEFAULT_MODEL_FEATURE_COLUMNS),
            "limitations": _release_limitations(),
        }
        _write_json_file(stage / "dataset_manifest.json", release_manifest)
        (stage / "README.md").write_text(_release_readme(release_manifest), encoding="utf-8")
        _write_json_file(
            stage / "file_inventory.json",
            _file_inventory(stage, exclude={"file_inventory.json", "SHA256SUMS"}),
        )
        # Include the now-fixed inventory in the final checksum list. The list
        # itself is deliberately not self-referential.
        _write_checksums(stage)
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return DatasetRelease(
        directory=destination,
        manifest_path=destination / "dataset_manifest.json",
        candidate_row_count=candidate_count,
        unscored_positive_count=unscored_count,
    )


def merge_candidate_dataset_builds(
    data_root: str | Path,
    *,
    input_manifests: Sequence[str | Path],
    start_date: date,
    end_date: date,
) -> Path:
    """Atomically combine contiguous completed candidate build chunks.

    This supports resumable local builds without ever globbing immutable
    candidate artifacts. Every input must use the exact same positive view,
    candidate policy, global chronological split, feature allowlist, and
    no-weather policy. The combined document is itself a normal completed
    candidate manifest, so existing readers and the release exporter need no
    special path.
    """
    _validate_date_range(start_date, end_date)
    if isinstance(input_manifests, (str, Path)) or not isinstance(input_manifests, Sequence):
        raise CandidateDatasetError("input_manifests must be a non-empty sequence of manifest paths")
    if not input_manifests:
        raise CandidateDatasetError("input_manifests must not be empty")
    root = Path(data_root).resolve()
    manifests = []
    for value in input_manifests:
        selected = _selected_candidate_manifest(root, manifest_path=value)
        if selected is None:  # pragma: no cover - required=True already raises
            raise CandidateDatasetError("input candidate manifest is unavailable")
        document = _read_json(selected["path"], "input candidate dataset manifest")
        manifests.append((selected, document))
    _require_merge_compatibility(manifests)
    _require_contiguous_manifest_ranges(manifests, start_date=start_date, end_date=end_date)

    reports: list[dict[str, Any]] = []
    candidate_paths: list[str] = []
    unscored_paths: list[str] = []
    normalized_ids: list[str] = []
    seen_snapshots: set[datetime] = set()
    for _selected, document in manifests:
        for report in document.get("snapshot_reports", []):
            if not isinstance(report, dict):
                raise CandidateDatasetError("input candidate manifest has an invalid snapshot report")
            snapshot = _parse_utc(report.get("source_snapshot_time"), "snapshot report source_snapshot_time")
            if not start_date <= snapshot.date() <= end_date:
                raise CandidateDatasetError(
                    "input candidate manifest contains a snapshot outside the requested merge range"
                )
            if snapshot in seen_snapshots:
                raise CandidateDatasetError(
                    "input candidate manifests overlap at source snapshot " + format_utc(snapshot)
                )
            seen_snapshots.add(snapshot)
            candidate_path = _validated_relative_paths(
                [report.get("candidate_artifact_relative_path")],
                prefix="normalized/candidate-examples/",
                label="snapshot candidate artifact",
            )[0]
            unscored_value = report.get("unscored_artifact_relative_path")
            unscored_path = None
            if unscored_value is not None:
                unscored_path = _validated_relative_paths(
                    [unscored_value],
                    prefix="normalized/candidate-unscored-positives/",
                    label="snapshot unscored artifact",
                )[0]
            reports.append(dict(report))
            candidate_paths.append(candidate_path)
            if unscored_path is not None:
                unscored_paths.append(unscored_path)
        normalized_ids.extend(
            _required_text(value, "normalized artifact id")
            for value in document.get("normalized_artifact_ids", [])
        )
    if not reports:
        raise CandidateDatasetError("input candidate manifests contain no snapshot reports")
    reports.sort(key=lambda report: report["source_snapshot_time"])
    candidate_paths = [report["candidate_artifact_relative_path"] for report in reports]
    unscored_paths = [
        report["unscored_artifact_relative_path"]
        for report in reports
        if report.get("unscored_artifact_relative_path") is not None
    ]
    base = manifests[0][1]
    total_positive = sum(_nonnegative_int(report.get("input_positive_count"), "input_positive_count") for report in reports)
    supported_positive = sum(_nonnegative_int(report.get("supported_positive_count"), "supported_positive_count") for report in reports)
    completed_at = datetime.now(timezone.utc)
    build_id = uuid.uuid4().hex
    document = {
        "schema_version": CANDIDATE_DATASET_MANIFEST_SCHEMA_VERSION,
        "kind": "completed-firms-candidate-dataset-build",
        "status": "complete",
        "build_id": build_id,
        "completed_at": format_utc(completed_at),
        "candidate_dataset_build_version": CANDIDATE_DATASET_BUILD_VERSION,
        "merged_from_candidate_build_ids": [selected["build_id"] for selected, _ in manifests],
        "input_positive_view_manifest_relative_path": base["input_positive_view_manifest_relative_path"],
        "input_positive_view_build_version": base["input_positive_view_build_version"],
        "source_snapshot_start_date": start_date.isoformat(),
        "source_snapshot_end_date": end_date.isoformat(),
        "split_source_snapshot_start_date": base["split_source_snapshot_start_date"],
        "split_source_snapshot_end_date": base["split_source_snapshot_end_date"],
        "candidate_policy": base["candidate_policy"],
        "firms_feature_policy": base["firms_feature_policy"],
        "weather": base["weather"],
        "model_feature_columns": base["model_feature_columns"],
        "split_policy": base["split_policy"],
        "snapshot_splits": base["snapshot_splits"],
        "candidate_artifact_relative_paths": candidate_paths,
        "unscored_artifact_relative_paths": unscored_paths,
        "normalized_artifact_ids": sorted(set(normalized_ids)),
        "input_positive_count": total_positive,
        "candidate_row_count": sum(_nonnegative_int(report.get("candidate_row_count"), "candidate_row_count") for report in reports),
        "supported_positive_count": supported_positive,
        "weak_negative_proxy_count": sum(_nonnegative_int(report.get("weak_negative_proxy_count"), "weak_negative_proxy_count") for report in reports),
        "unscored_positive_count": sum(_nonnegative_int(report.get("unscored_positive_count"), "unscored_positive_count") for report in reports),
        "supported_positive_rate": supported_positive / total_positive if total_positive else 0.0,
        "snapshot_reports": reports,
    }
    destination = (
        root
        / "manifests"
        / "candidate-dataset-builds"
        / completed_at.strftime("%Y/%m/%d")
        / f"{completed_at.strftime('%H%M%S%f')}_{build_id}.json"
    )
    return write_atomic_json(destination, document)


def _require_merge_compatibility(
    manifests: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    keys = (
        "input_positive_view_manifest_relative_path",
        "input_positive_view_build_version",
        "split_source_snapshot_start_date",
        "split_source_snapshot_end_date",
        "candidate_policy",
        "firms_feature_policy",
        "weather",
        "model_feature_columns",
        "split_policy",
        "snapshot_splits",
    )
    baseline = manifests[0][1]
    for _selected, document in manifests[1:]:
        for key in keys:
            if _canonical_json(document.get(key)) != _canonical_json(baseline.get(key)):
                raise CandidateDatasetError(
                    "input candidate manifests disagree on " + key + "; they cannot be merged"
                )


def _require_contiguous_manifest_ranges(
    manifests: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    start_date: date,
    end_date: date,
) -> None:
    ranges = []
    for _selected, document in manifests:
        try:
            manifest_start = date.fromisoformat(
                _required_text(document.get("source_snapshot_start_date"), "source_snapshot_start_date")
            )
            manifest_end = date.fromisoformat(
                _required_text(document.get("source_snapshot_end_date"), "source_snapshot_end_date")
            )
        except ValueError as exc:
            raise CandidateDatasetError("input candidate manifest has invalid source date bounds") from exc
        ranges.append((manifest_start, manifest_end))
    ranges.sort()
    if ranges[0][0] != start_date or ranges[-1][1] != end_date:
        raise CandidateDatasetError("input candidate manifests do not span the requested merge range")
    previous_end = ranges[0][1]
    for current_start, current_end in ranges[1:]:
        if current_start != previous_end + timedelta(days=1):
            raise CandidateDatasetError("input candidate manifests are not contiguous date chunks")
        previous_end = current_end


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _selected_positive_rows(
    data_root: Path,
    *,
    manifest_path: Path,
    start_date: date,
    end_date: date,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for row in iter_training_examples(data_root, manifest_path=manifest_path):
        snapshot_time = _parse_utc(row.get("source_snapshot_time"), "source_snapshot_time")
        if start_date <= snapshot_time.date() <= end_date:
            rows.append(row)
    return tuple(rows)


def _group_rows_by_source_snapshot(
    rows: Iterable[Mapping[str, Any]],
) -> dict[datetime, tuple[dict[str, Any], ...]]:
    grouped: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        snapshot_time = _parse_utc(row.get("source_snapshot_time"), "source_snapshot_time")
        grouped[snapshot_time].append(dict(row))
    return {
        snapshot: tuple(sorted(values, key=lambda row: row["cell_id"]))
        for snapshot, values in grouped.items()
    }


def _candidate_firms_dates(snapshot_time: datetime) -> tuple[date, ...]:
    """Return both UTC dates that can affect a Canada/CONUS candidate cutoff."""
    resolved = _as_utc(snapshot_time, "source_snapshot_time")
    return ((resolved - timedelta(days=1)).date(), resolved.date())


def _iter_firms_detections(data_root: Path, *, dates: Sequence[date]) -> Iterator[dict[str, Any]]:
    seen_paths: set[Path] = set()
    for coverage_date in dates:
        partition = data_root / "normalized" / "fire-detections" / f"acq-date={coverage_date.isoformat()}"
        if not partition.exists():
            continue
        for path in sorted(partition.glob("*.jsonl.gz")):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            for record in _iter_jsonl_records(path, context="FIRMS detection"):
                if record.get("record_type") == "firms_detection":
                    yield record


def _latest_firms_coverage(ledger: CoverageLedger) -> dict[str, CoverageRecord]:
    latest: dict[str, CoverageRecord] = {}
    for record in ledger.entries():
        if (
            record.source == "NASA FIRMS"
            and record.expected_coverage_id is not None
            and record.expected_coverage_id.startswith("firms:")
        ):
            latest[record.expected_coverage_id] = record
    return latest


def _require_terminal_firms_coverage(
    *,
    snapshot_time: datetime,
    dates: Sequence[date],
    products: Sequence[str],
    region: str,
    latest_by_expected_id: Mapping[str, CoverageRecord],
) -> None:
    terminal = {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED}
    incomplete = []
    for coverage_date in dates:
        for product in products:
            expected_id = f"firms:{product}:{region}:{coverage_date.isoformat()}"
            record = latest_by_expected_id.get(expected_id)
            if record is None:
                incomplete.append(f"{product}/{coverage_date.isoformat()} (missing)")
            elif record.status not in terminal:
                incomplete.append(f"{product}/{coverage_date.isoformat()} ({record.status.value})")
    if incomplete:
        raise CandidateDatasetError(
            "FIRMS coverage is incomplete for FIRMS-only candidate snapshot "
            f"{format_utc(snapshot_time)}: {', '.join(incomplete)}"
        )


def _index_detections_by_cell(
    detections: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        try:
            cell = cell_from_wgs84(
                latitude=_finite_float(detection.get("latitude"), "FIRMS latitude"),
                longitude=_finite_float(detection.get("longitude"), "FIRMS longitude"),
            )
        except TrainingGridError as exc:
            raise CandidateDatasetError("FIRMS detection cannot be mapped to the training grid") from exc
        indexed[cell.cell_id].append(dict(detection))
    return dict(indexed)


def _assemble_candidate_feature_row(
    row: Mapping[str, Any],
    *,
    detections_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    terrain_sampler: TerrainFeatureSampler,
    firms_lookback: timedelta,
    firms_availability_lag: timedelta,
    snapshot_label_raw_artifact_ids: tuple[str, ...],
    split: str,
) -> dict[str, Any]:
    try:
        cell = cell_from_id(_required_text(row.get("cell_id"), "candidate cell_id"))
        cutoff_at = _parse_utc(row.get("anchor_at"), "candidate anchor_at")
        local_detections = [
            detection
            for neighbour in cells_in_square_radius(cell, radius_cells=1)
            for detection in detections_by_cell.get(neighbour.cell_id, ())
        ]
        firms_features = build_firms_fire_state_features(
            local_detections,
            cell_id=cell.cell_id,
            cutoff_at=cutoff_at,
            lookback=firms_lookback,
            availability_lag=firms_availability_lag,
        )
        terrain_features = terrain_sampler.sample_cell(cell)
    except (TrainingGridError, FireStateFeatureError, TerrainFeatureError) as exc:
        raise CandidateDatasetError(
            f"could not assemble candidate features for {row.get('example_id')}: {exc}"
        ) from exc
    result = dict(row)
    result.update(dict(firms_features))
    result.update(dict(terrain_features))
    result.update(
        {
            "record_type": "candidate_training_example",
            "candidate_dataset_schema_version": CANDIDATE_DATASET_SCHEMA_VERSION,
            "candidate_dataset_build_version": CANDIDATE_DATASET_BUILD_VERSION,
            "training_grid": "naea-1km",
            "prediction_horizon_hours": DEFAULT_HORIZON_HOURS,
            "dataset_split": split,
            "source_window_policy": SOURCE_WINDOW_POLICY,
            "feds_snapshot_context_raw_artifact_ids": list(snapshot_label_raw_artifact_ids),
            "firms_raw_artifact_ids": list(
                _eligible_firms_raw_artifact_ids(
                    local_detections,
                    cutoff_at=cutoff_at,
                    lookback=firms_lookback,
                    availability_lag=firms_availability_lag,
                )
            ),
            "weather_available": 0,
            "weather_missing_indicator": 1,
            "weather_feature_status": NO_WEATHER_STATUS,
            "weather_input_policy": NO_WEATHER_POLICY,
        }
    )
    return result


def _assemble_unscored_positive_row(
    row: Mapping[str, Any],
    *,
    snapshot_label_raw_artifact_ids: tuple[str, ...],
    split: str,
) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "record_type": "unscored_positive_diagnostic",
            "candidate_dataset_schema_version": CANDIDATE_DATASET_SCHEMA_VERSION,
            "candidate_dataset_build_version": CANDIDATE_DATASET_BUILD_VERSION,
            "dataset_split": split,
            "source_window_policy": SOURCE_WINDOW_POLICY,
            "feds_snapshot_context_raw_artifact_ids": list(snapshot_label_raw_artifact_ids),
            "weather_available": 0,
            "weather_missing_indicator": 1,
            "weather_feature_status": NO_WEATHER_STATUS,
            "weather_input_policy": NO_WEATHER_POLICY,
        }
    )
    return result


def _eligible_firms_raw_artifact_ids(
    detections: Iterable[Mapping[str, Any]],
    *,
    cutoff_at: datetime,
    lookback: timedelta,
    availability_lag: timedelta,
) -> tuple[str, ...]:
    window_start = cutoff_at - lookback
    latest_eligible = cutoff_at - availability_lag
    identifiers = set()
    for detection in detections:
        acquired_at = _parse_utc(detection.get("acquired_at"), "FIRMS acquired_at")
        if not window_start <= acquired_at <= latest_eligible:
            continue
        provenance = detection.get("provenance")
        if isinstance(provenance, Mapping):
            raw_id = provenance.get("raw_artifact_id")
            if isinstance(raw_id, str) and raw_id.strip():
                identifiers.add(raw_id.strip())
    return tuple(sorted(identifiers))


def _snapshot_label_raw_artifact_ids(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value.strip()
                for row in rows
                for value in row.get("label_raw_artifact_ids", [])
                if isinstance(value, str) and value.strip()
            }
        )
    )


def _store_candidate_rows(
    data_root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    source_snapshot_time: datetime,
    storage_budget: StorageBudgetPolicy,
) -> NormalizedArtifact:
    raw_ids = _artifact_ids_for_rows(rows)
    if not raw_ids:
        raise CandidateDatasetError("candidate rows have no retained raw-artifact lineage")
    require_admission(
        storage_budget,
        data_root,
        category="derived_training_views",
        requested_bytes=_conservative_bytes(rows),
    )
    return write_normalized_jsonl(
        data_root,
        entity="candidate_examples",
        records=rows,
        partitions={
            "dataset_build": CANDIDATE_DATASET_BUILD_VERSION,
            "source": "firms-only-candidates-feds-weak-labels",
            "source_snapshot": format_utc(source_snapshot_time),
            "grid": "naea-1km",
        },
        raw_artifact_ids=raw_ids,
        transformation_version=CANDIDATE_DATASET_BUILD_VERSION,
    )


def _store_unscored_rows(
    data_root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    source_snapshot_time: datetime,
    storage_budget: StorageBudgetPolicy,
) -> NormalizedArtifact:
    raw_ids = _artifact_ids_for_rows(rows)
    if not raw_ids:
        raise CandidateDatasetError("unscored positive rows have no retained raw-artifact lineage")
    require_admission(
        storage_budget,
        data_root,
        category="derived_training_views",
        requested_bytes=_conservative_bytes(rows),
    )
    return write_normalized_jsonl(
        data_root,
        entity="candidate_unscored_positives",
        records=rows,
        partitions={
            "dataset_build": CANDIDATE_DATASET_BUILD_VERSION,
            "source": "firms-only-candidates-feds-weak-labels",
            "source_snapshot": format_utc(source_snapshot_time),
            "grid": "naea-1km",
        },
        raw_artifact_ids=raw_ids,
        transformation_version=CANDIDATE_DATASET_BUILD_VERSION,
    )


def _artifact_ids_for_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    keys = (
        "label_raw_artifact_ids",
        "firms_raw_artifact_ids",
        "candidate_seed_raw_artifact_ids",
        "feds_snapshot_context_raw_artifact_ids",
    )
    return tuple(
        sorted(
            {
                value.strip()
                for row in rows
                for key in keys
                for value in row.get(key, [])
                if isinstance(value, str) and value.strip()
            }
        )
    )


def _write_completed_candidate_manifest(
    data_root: Path,
    *,
    reports: tuple[CandidateDatasetSnapshotReport, ...],
    input_positive_manifest_path: Path,
    start_date: date,
    end_date: date,
    split_start_date: date,
    split_end_date: date,
    radius_cells: int,
    max_weak_negative_proxies_per_snapshot: int,
    firms_lookback: timedelta,
    firms_availability_lag: timedelta,
    firms_products: Sequence[str],
    firms_region: str,
    snapshot_splits: Mapping[datetime, str],
) -> Path:
    if not reports:
        raise CandidateDatasetError("cannot publish a completed candidate view without snapshots")
    try:
        positive_manifest_relative_path = input_positive_manifest_path.relative_to(data_root).as_posix()
        candidate_paths = [
            report.candidate_artifact.artifact_path.relative_to(data_root).as_posix()
            for report in reports
        ]
        unscored_paths = [
            report.unscored_artifact.artifact_path.relative_to(data_root).as_posix()
            for report in reports
            if report.unscored_artifact is not None
        ]
    except ValueError as exc:
        raise CandidateDatasetError("candidate source artifact lies outside data_root") from exc
    completed_at = datetime.now(timezone.utc)
    build_id = uuid.uuid4().hex
    total_positive = sum(report.input_positive_count for report in reports)
    supported_positive = sum(report.supported_positive_count for report in reports)
    document = {
        "schema_version": CANDIDATE_DATASET_MANIFEST_SCHEMA_VERSION,
        "kind": "completed-firms-candidate-dataset-build",
        "status": "complete",
        "build_id": build_id,
        "completed_at": format_utc(completed_at),
        "candidate_dataset_build_version": CANDIDATE_DATASET_BUILD_VERSION,
        "input_positive_view_manifest_relative_path": positive_manifest_relative_path,
        "input_positive_view_build_version": TRAINING_DATASET_BUILD_VERSION,
        "source_snapshot_start_date": start_date.isoformat(),
        "source_snapshot_end_date": end_date.isoformat(),
        "split_source_snapshot_start_date": split_start_date.isoformat(),
        "split_source_snapshot_end_date": split_end_date.isoformat(),
        "candidate_policy": {
            "source": "firms-only",
            "sampler_version": CANDIDATE_SAMPLER_VERSION,
            "radius_cells": radius_cells,
            "max_weak_negative_proxies_per_snapshot": max_weak_negative_proxies_per_snapshot,
            "target_zero_semantics": "weak-negative-proxy-not-clear-no-burn",
            "positive_outside_support_policy": "retain-unscored-positive-diagnostic",
            "source_window_policy": SOURCE_WINDOW_POLICY,
        },
        "firms_feature_policy": {
            "products": list(firms_products),
            "region": firms_region,
            "lookback_hours": firms_lookback.total_seconds() / 3_600,
            "availability_lag_minutes": firms_availability_lag.total_seconds() / 60,
            "coverage_policy": "require-terminal-product-day-for-candidate-window/v1",
        },
        "weather": {
            "available": False,
            "status": NO_WEATHER_STATUS,
            "policy": NO_WEATHER_POLICY,
        },
        "model_feature_columns": list(DEFAULT_MODEL_FEATURE_COLUMNS),
        "split_policy": SPLIT_POLICY,
        "snapshot_splits": {
            format_utc(snapshot): split for snapshot, split in sorted(snapshot_splits.items())
        },
        "candidate_artifact_relative_paths": candidate_paths,
        "unscored_artifact_relative_paths": unscored_paths,
        "normalized_artifact_ids": [report.candidate_artifact.normalized_artifact_id for report in reports],
        "input_positive_count": total_positive,
        "candidate_row_count": sum(report.candidate_row_count for report in reports),
        "supported_positive_count": supported_positive,
        "weak_negative_proxy_count": sum(report.weak_negative_proxy_count for report in reports),
        "unscored_positive_count": sum(report.unscored_positive_count for report in reports),
        "supported_positive_rate": supported_positive / total_positive if total_positive else 0.0,
        "snapshot_reports": [
            {
                "source_snapshot_time": format_utc(report.source_snapshot_time),
                "input_positive_count": report.input_positive_count,
                "candidate_row_count": report.candidate_row_count,
                "supported_positive_count": report.supported_positive_count,
                "weak_negative_proxy_count": report.weak_negative_proxy_count,
                "unscored_positive_count": report.unscored_positive_count,
                "candidate_artifact_relative_path": report.candidate_artifact.artifact_path.relative_to(data_root).as_posix(),
                "unscored_artifact_relative_path": (
                    report.unscored_artifact.artifact_path.relative_to(data_root).as_posix()
                    if report.unscored_artifact is not None
                    else None
                ),
            }
            for report in reports
        ],
    }
    destination = (
        data_root
        / "manifests"
        / "candidate-dataset-builds"
        / completed_at.strftime("%Y/%m/%d")
        / f"{completed_at.strftime('%H%M%S%f')}_{build_id}.json"
    )
    return write_atomic_json(destination, document)


def _selected_candidate_manifest(
    data_root: Path,
    *,
    manifest_path: str | Path | None,
) -> dict[str, Any] | None:
    if manifest_path is not None:
        path = Path(manifest_path)
        if not path.is_absolute():
            working_directory_path = path.resolve()
            path = (
                working_directory_path
                if working_directory_path.is_file()
                and working_directory_path.is_relative_to(data_root)
                else data_root / path
            )
        return _read_completed_candidate_manifest(path, required=True)
    root = data_root / "manifests" / "candidate-dataset-builds"
    if not root.exists():
        return None
    manifests = [
        document
        for path in sorted(root.rglob("*.json"))
        if (document := _read_completed_candidate_manifest(path, required=False)) is not None
    ]
    if not manifests:
        return None
    return max(manifests, key=lambda document: (document["completed_at"], document["build_id"]))


def _read_completed_candidate_manifest(path: Path, *, required: bool) -> dict[str, Any] | None:
    try:
        document = _read_json(path, "candidate dataset manifest")
        if document.get("kind") != "completed-firms-candidate-dataset-build":
            return None
        if document.get("status") != "complete":
            raise CandidateDatasetError("candidate dataset manifest is not complete")
        if document.get("schema_version") != CANDIDATE_DATASET_MANIFEST_SCHEMA_VERSION:
            raise CandidateDatasetError("candidate dataset manifest has an unsupported schema version")
        if document.get("candidate_dataset_build_version") != CANDIDATE_DATASET_BUILD_VERSION:
            raise CandidateDatasetError("candidate dataset manifest has a different build version")
        candidate_paths = _validated_relative_paths(
            document.get("candidate_artifact_relative_paths"),
            prefix="normalized/candidate-examples/",
            label="candidate artifact",
        )
        unscored_paths = _validated_relative_paths(
            document.get("unscored_artifact_relative_paths"),
            prefix="normalized/candidate-unscored-positives/",
            label="unscored artifact",
            allow_empty=True,
        )
        return {
            "path": path,
            "build_id": _required_text(document.get("build_id"), "candidate build_id"),
            "completed_at": _parse_utc(document.get("completed_at"), "candidate completed_at"),
            "source_snapshot_start_date": _required_text(
                document.get("source_snapshot_start_date"), "source_snapshot_start_date"
            ),
            "source_snapshot_end_date": _required_text(
                document.get("source_snapshot_end_date"), "source_snapshot_end_date"
            ),
            "candidate_artifact_relative_paths": candidate_paths,
            "unscored_artifact_relative_paths": unscored_paths,
        }
    except (CandidateDatasetError, OSError, json.JSONDecodeError) as exc:
        if required:
            if isinstance(exc, CandidateDatasetError):
                raise
            raise CandidateDatasetError(f"could not read candidate dataset manifest: {path}") from exc
        return None


def _validated_relative_paths(
    value: object,
    *,
    prefix: str,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise CandidateDatasetError(f"{label} paths must be a {'possibly empty' if allow_empty else 'non-empty'} list")
    paths = []
    for item in value:
        path = Path(_required_text(item, f"{label} path"))
        if path.is_absolute() or ".." in path.parts:
            raise CandidateDatasetError(f"{label} path must be relative")
        normalized = path.as_posix()
        if not normalized.startswith(prefix):
            raise CandidateDatasetError(f"{label} path is outside {prefix}")
        paths.append(normalized)
    if len(set(paths)) != len(paths):
        raise CandidateDatasetError(f"{label} paths must not contain duplicates")
    return tuple(paths)


def _require_positive_manifest_range(
    manifest: Mapping[str, Any],
    *,
    start_date: date,
    end_date: date,
) -> None:
    try:
        available_start = date.fromisoformat(
            _required_text(manifest.get("source_snapshot_start_date"), "source_snapshot_start_date")
        )
        available_end = date.fromisoformat(
            _required_text(manifest.get("source_snapshot_end_date"), "source_snapshot_end_date")
        )
    except ValueError as exc:
        raise CandidateDatasetError("positive training manifest has invalid source date bounds") from exc
    if start_date < available_start or end_date > available_end:
        raise CandidateDatasetError(
            "requested source range is outside the completed positive training view: "
            f"requested {start_date.isoformat()} through {end_date.isoformat()}, "
            f"available {available_start.isoformat()} through {available_end.isoformat()}. "
            "Collect/rebuild FIRMS, FEDS labels, terrain, and the positive view before extending it."
        )


def _chronological_snapshot_splits(snapshots: tuple[datetime, ...]) -> dict[datetime, str]:
    if not snapshots:
        raise CandidateDatasetError("candidate dataset requires at least one source snapshot")
    split_index = max(1, int(len(snapshots) * DEFAULT_TRAIN_FRACTION))
    if split_index >= len(snapshots):
        split_index = len(snapshots)
    return {
        snapshot: "train" if index < split_index else "validation"
        for index, snapshot in enumerate(snapshots)
    }


def _conservative_bytes(rows: Iterable[Mapping[str, Any]]) -> int:
    encoded = sum(
        len(json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")) + 1
        for row in rows
    )
    return encoded * 2 + 65_536


def _release_is_equivalent(
    data_root: Path,
    destination: Path,
    existing: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> bool:
    """Return whether an existing release has the same logical payload.

    Re-running a deterministic immutable build gets a new manifest ID even
    when every candidate artifact is reused. Compare the uncompressed JSONL
    bytes (and, for v2 releases, the canonical CSV bytes) in that case,
    rather than forcing users to create duplicate upload directories. The
    files themselves are hashed; an existing manifest's claims are not
    trusted as proof.

    Schema v1 is intentionally accepted with its original JSONL-only
    contract. Reusing such an immutable historical release preserves the
    exporter behaviour that predated the CSV convenience files.
    """
    existing_schema_version = existing.get("schema_version")
    if existing_schema_version not in {
        LEGACY_CANDIDATE_DATASET_RELEASE_SCHEMA_VERSION,
        CANDIDATE_DATASET_RELEASE_SCHEMA_VERSION,
    }:
        return False
    source_document = _read_json(
        candidate_manifest["path"], "candidate dataset manifest"
    )
    try:
        expected_candidate_count = _nonnegative_int(
            source_document.get("candidate_row_count"), "candidate_row_count"
        )
        expected_unscored_count = _nonnegative_int(
            source_document.get("unscored_positive_count"), "unscored_positive_count"
        )
        existing_candidate_count = _nonnegative_int(
            existing.get("candidate_row_count"), "existing candidate_row_count"
        )
        existing_unscored_count = _nonnegative_int(
            existing.get("unscored_positive_count"), "existing unscored_positive_count"
        )
    except CandidateDatasetError:
        return False
    if (
        existing.get("candidate_dataset_build_version")
        != CANDIDATE_DATASET_BUILD_VERSION
        or existing.get("source_snapshot_start_date")
        != candidate_manifest.get("source_snapshot_start_date")
        or existing.get("source_snapshot_end_date")
        != candidate_manifest.get("source_snapshot_end_date")
        or existing.get("model_feature_columns") != list(DEFAULT_MODEL_FEATURE_COLUMNS)
        or existing.get("weather")
        != {
            "available": False,
            "status": NO_WEATHER_STATUS,
            "policy": NO_WEATHER_POLICY,
        }
        or existing_candidate_count != expected_candidate_count
        or existing_unscored_count != expected_unscored_count
    ):
        return False
    expected_candidate_digest, candidate_count = _jsonl_artifact_payload_digest(
        data_root,
        candidate_manifest["candidate_artifact_relative_paths"],
    )
    expected_unscored_digest, unscored_count = _jsonl_artifact_payload_digest(
        data_root,
        candidate_manifest["unscored_artifact_relative_paths"],
    )
    if candidate_count != expected_candidate_count or unscored_count != expected_unscored_count:
        raise CandidateDatasetError("candidate manifest row counts do not match its artifacts")
    try:
        actual_candidate_digest = _gzip_payload_sha256(
            destination / "candidate_examples.jsonl.gz"
        )
        actual_unscored_digest = _gzip_payload_sha256(
            destination / "unscored_positives.jsonl.gz"
        )
    except (OSError, EOFError):
        return False
    jsonl_is_equivalent = (
        actual_candidate_digest == expected_candidate_digest
        and actual_unscored_digest == expected_unscored_digest
    )
    if not jsonl_is_equivalent:
        return False
    if existing_schema_version == LEGACY_CANDIDATE_DATASET_RELEASE_SCHEMA_VERSION:
        return True

    (
        expected_candidate_csv_digest,
        candidate_csv_count,
        candidate_csv_columns,
    ) = _csv_artifact_payload_digest(
        data_root,
        candidate_manifest["candidate_artifact_relative_paths"],
    )
    (
        expected_unscored_csv_digest,
        unscored_csv_count,
        unscored_csv_columns,
    ) = _csv_artifact_payload_digest(
        data_root,
        candidate_manifest["unscored_artifact_relative_paths"],
    )
    if candidate_csv_count != expected_candidate_count or unscored_csv_count != expected_unscored_count:
        raise CandidateDatasetError("candidate manifest row counts do not match its CSV payload")
    if (
        existing.get("candidate_examples_csv_content_sha256")
        != expected_candidate_csv_digest
        or existing.get("unscored_positives_csv_content_sha256")
        != expected_unscored_csv_digest
        or existing.get("candidate_examples_csv_columns") != list(candidate_csv_columns)
        or existing.get("unscored_positives_csv_columns") != list(unscored_csv_columns)
    ):
        return False
    try:
        actual_candidate_csv_digest = _gzip_payload_sha256(
            destination / "candidate_examples.csv.gz"
        )
        actual_unscored_csv_digest = _gzip_payload_sha256(
            destination / "unscored_positives.csv.gz"
        )
        actual_candidate_csv_columns, actual_candidate_csv_count = _gzip_csv_layout(
            destination / "candidate_examples.csv.gz"
        )
        actual_unscored_csv_columns, actual_unscored_csv_count = _gzip_csv_layout(
            destination / "unscored_positives.csv.gz"
        )
    except (CandidateDatasetError, OSError, EOFError, UnicodeDecodeError, csv.Error):
        return False
    return (
        actual_candidate_csv_digest == expected_candidate_csv_digest
        and actual_unscored_csv_digest == expected_unscored_csv_digest
        and actual_candidate_csv_columns == candidate_csv_columns
        and actual_unscored_csv_columns == unscored_csv_columns
        and actual_candidate_csv_count == expected_candidate_count
        and actual_unscored_csv_count == expected_unscored_count
    )


def _concatenate_jsonl_gzip(
    data_root: Path,
    relative_paths: Sequence[str],
    destination: Path,
) -> tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    with gzip.GzipFile(destination, mode="wb", mtime=0) as compressed:
        for relative_path in relative_paths:
            path = data_root / relative_path
            for record in _iter_jsonl_records(path, context="release record"):
                line = _release_jsonl_line(record)
                compressed.write(line)
                digest.update(line)
                count += 1
    return count, digest.hexdigest()


def _jsonl_artifact_payload_digest(
    data_root: Path,
    relative_paths: Sequence[str],
) -> tuple[str, int]:
    """Hash the exact logical JSONL bytes an upload export would contain."""
    digest = hashlib.sha256()
    count = 0
    for relative_path in relative_paths:
        path = data_root / relative_path
        for record in _iter_jsonl_records(path, context="release record"):
            digest.update(_release_jsonl_line(record))
            count += 1
    return digest.hexdigest(), count


def _concatenate_csv_gzip(
    data_root: Path,
    relative_paths: Sequence[str],
    destination: Path,
) -> tuple[int, str, tuple[str, ...]]:
    """Write canonical CSV rows for one JSONL artifact sequence.

    CSV is a convenience representation of the self-contained candidate
    release. The JSONL copies remain the lossless source-format payload. Each
    nested list or object is encoded as compact, sorted-key JSON so lineage
    fields can be safely loaded as strings without Python repr differences.
    """
    column_names = _csv_columns_from_artifacts(data_root, relative_paths)
    with gzip.GzipFile(destination, mode="wb", mtime=0) as compressed:
        text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        try:
            count = _write_csv_payload(
                text,
                data_root=data_root,
                relative_paths=relative_paths,
                column_names=column_names,
            )
            text.flush()
        finally:
            # Do not let TextIOWrapper close the GzipFile before its footer is
            # emitted by the surrounding context manager.
            text.detach()
    return count, _gzip_payload_sha256(destination), column_names


def _csv_artifact_payload_digest(
    data_root: Path,
    relative_paths: Sequence[str],
) -> tuple[str, int, tuple[str, ...]]:
    """Hash the exact canonical CSV bytes an upload export would contain."""
    column_names = _csv_columns_from_artifacts(data_root, relative_paths)
    sink = _CsvPayloadDigest()
    count = _write_csv_payload(
        sink,
        data_root=data_root,
        relative_paths=relative_paths,
        column_names=column_names,
    )
    return sink.hexdigest(), count, column_names


def _csv_columns_from_artifacts(
    data_root: Path,
    relative_paths: Sequence[str],
) -> tuple[str, ...]:
    """Return one deterministic CSV header for the selected artifact rows."""
    columns: set[str] = set()
    for relative_path in relative_paths:
        path = data_root / relative_path
        for record in _iter_jsonl_records(path, context="release record"):
            for key in record:
                if not isinstance(key, str) or not key:
                    raise CandidateDatasetError("release CSV columns must be non-empty strings")
                columns.add(key)
    return tuple(sorted(columns))


def _write_csv_payload(
    stream: Any,
    *,
    data_root: Path,
    relative_paths: Sequence[str],
    column_names: Sequence[str],
) -> int:
    """Write canonical CSV header and rows to a text stream; return row count."""
    writer = csv.DictWriter(
        stream,
        fieldnames=tuple(column_names),
        extrasaction="raise",
        lineterminator="\n",
    )
    if column_names:
        writer.writeheader()
    count = 0
    for relative_path in relative_paths:
        path = data_root / relative_path
        for record in _iter_jsonl_records(path, context="release record"):
            writer.writerow(
                {
                    column: _canonical_csv_value(record.get(column))
                    for column in column_names
                }
            )
            count += 1
    return count


def _canonical_csv_value(value: Any) -> str:
    """Return one stable CSV field while preserving nested JSON lineage."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    if isinstance(value, (list, tuple)):
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    raise CandidateDatasetError(
        "release CSV cannot encode a value of type " + type(value).__name__
    )


class _CsvPayloadDigest:
    """Text writer accepted by :mod:`csv` that hashes each emitted UTF-8 byte."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()

    def write(self, value: str) -> int:
        self._digest.update(value.encode("utf-8"))
        return len(value)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _release_jsonl_line(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _gzip_payload_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _gzip_csv_layout(path: Path) -> tuple[tuple[str, ...], int]:
    """Return a CSV payload's header and row count without inferring types."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        try:
            header = tuple(next(reader))
        except StopIteration:
            return (), 0
        if not header or len(set(header)) != len(header):
            raise CandidateDatasetError("release CSV has an invalid header")
        return header, sum(1 for _ in reader)


def _release_schema(
    *,
    candidate_csv_columns: Sequence[str],
    unscored_csv_columns: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": CANDIDATE_DATASET_RELEASE_SCHEMA_VERSION,
        "format": "gzip-compressed JSON Lines and CSV",
        "formats": {
            "candidate_examples": {
                "jsonl_gzip_path": "candidate_examples.jsonl.gz",
                "csv_gzip_path": "candidate_examples.csv.gz",
                "csv_columns": list(candidate_csv_columns),
            },
            "unscored_positives": {
                "jsonl_gzip_path": "unscored_positives.jsonl.gz",
                "csv_gzip_path": "unscored_positives.csv.gz",
                "csv_columns": list(unscored_csv_columns),
            },
        },
        "csv_encoding": {
            "nested_values": "canonical-json-sorted-keys-compact/v1",
            "null_values": "empty-field",
            "boolean_values": "lowercase-json-literals",
        },
        "candidate_dataset_build_version": CANDIDATE_DATASET_BUILD_VERSION,
        "model_feature_columns": list(DEFAULT_MODEL_FEATURE_COLUMNS),
        "target_column": "target_newly_burned_12h",
        "split_column": "dataset_split",
        "not_model_features": [
            "all timestamps", "all identifiers", "all source/raw lineage", "all label metadata",
            "candidate-selection metadata", "weather missingness metadata", "dataset_split",
        ],
    }


def _release_limitations() -> list[str]:
    return [
        "target=0 means a FIRMS-seeded weak-negative proxy, not observed clear/no-burn.",
        "FEDS weak labels and FIRMS features share satellite evidence and are not independent ground truth.",
        "FIRMS-uncovered positives are retained in unscored_positives.jsonl.gz and excluded from the candidate table.",
        "Only source snapshots represented by FEDS-positive labels are included; no all-zero FEDS window is invented.",
        "Weather is absent. Open-Meteo exports are retrospective visualization data and are excluded.",
        "The chronological split groups whole source snapshots but is not incident-held-out or region-held-out validation.",
    ]


def _release_readme(manifest: Mapping[str, Any]) -> str:
    limitations = "\n".join(f"- {item}" for item in manifest["limitations"])
    features = "\n".join(f"- `{item}`" for item in manifest["model_feature_columns"])
    return f"""# Wildfire spread weak-label candidate dataset

This uploadable release contains one manifest-selected no-weather candidate
table for {manifest['source_snapshot_start_date']} through
{manifest['source_snapshot_end_date']}. It is a research baseline for 1 km,
12-hour spread prediction, not an operational fire-spread forecast.

Files:

- `candidate_examples.jsonl.gz`: fit-eligible weak positives and FIRMS-seeded
  weak-negative proxies.
- `candidate_examples.csv.gz`: the same candidate rows in a tabular format;
  nested lineage values use canonical compact JSON strings.
- `unscored_positives.jsonl.gz`: FEDS positives outside FIRMS candidate
  support; retain these for coverage diagnostics rather than treating them as
  negatives or dropping them.
- `unscored_positives.csv.gz`: the same diagnostic rows in CSV form.
- `dataset_manifest.json`, `schema.json`, `file_inventory.json`, and
  `SHA256SUMS`: version, schema, and integrity information.

Rows: {manifest['candidate_row_count']:,} candidate rows and
{manifest['unscored_positive_count']:,} unscored positives.

## Model feature allowlist

{features}

## Limitations

{limitations}
"""


def _write_checksums(directory: Path, *, destination: Path | None = None) -> Path:
    target = destination if destination is not None else directory / "SHA256SUMS"
    entries = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path == target:
            continue
        entries.append(f"{_sha256(path)}  {path.relative_to(directory).as_posix()}\n")
    target.write_text("".join(entries), encoding="utf-8")
    return target


def _file_inventory(directory: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in exclude:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_file(path: Path, document: Mapping[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateDatasetError(f"could not read {label}: {path}") from exc
    if not isinstance(document, dict):
        raise CandidateDatasetError(f"{label} must be a JSON object: {path}")
    return document


def _iter_jsonl_records(path: Path, *, context: str) -> Iterator[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CandidateDatasetError(
                        f"invalid {context} JSON at {path}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise CandidateDatasetError(
                        f"{context} record at {path}:{line_number} is not an object"
                    )
                yield record
    except (OSError, EOFError) as exc:
        raise CandidateDatasetError(f"could not read {context} artifact: {path}") from exc


def _validate_date_range(start_date: date, end_date: date) -> None:
    if not isinstance(start_date, date) or isinstance(start_date, datetime):
        raise CandidateDatasetError("start_date must be a date")
    if not isinstance(end_date, date) or isinstance(end_date, datetime):
        raise CandidateDatasetError("end_date must be a date")
    if end_date < start_date:
        raise CandidateDatasetError("end_date must not be before start_date")


def _validate_nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CandidateDatasetError(f"{label} must be a non-negative integer")
    return value


def _validated_products(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise CandidateDatasetError("firms_products must be a sequence")
    products = tuple(_required_text(value, "FIRMS product") for value in values)
    if not products or len(set(products)) != len(products):
        raise CandidateDatasetError("firms_products must be a non-empty unique sequence")
    return products


def _nonnegative_duration(value: timedelta, label: str) -> timedelta:
    if not isinstance(value, timedelta) or value < timedelta(0):
        raise CandidateDatasetError(f"{label} must be a non-negative timedelta")
    return value


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateDatasetError(f"{label} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CandidateDatasetError(f"{label} must be a non-negative integer")
    return value


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CandidateDatasetError(f"{label} must be numeric") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise CandidateDatasetError(f"{label} must be finite")
    return result


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise CandidateDatasetError(f"{label} must be an offset-aware ISO-8601 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CandidateDatasetError(f"{label} must be an offset-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CandidateDatasetError(f"{label} must be an offset-aware ISO-8601 timestamp")
    return parsed.astimezone(timezone.utc)


def _as_utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CandidateDatasetError(f"{label} must be an offset-aware datetime")
    return value.astimezone(timezone.utc)
