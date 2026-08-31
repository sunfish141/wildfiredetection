"""Build bounded, leakage-safe rows for the first wildfire tabular baseline.

This module is deliberately narrower than a full training pipeline.  It turns
the current FEDS *positive-only* weak labels into one row per canonical 1 km /
12 hour example, joins only FIRMS evidence available by that row's cutoff, and
samples retained terrain.  It does **not** manufacture negative labels or
weather values.

Issued forecast values need a native-grid adapter with explicit model/run,
captured availability, valid-time, and candidate-cell mapping provenance before
they can become model inputs. Until then every row records explicit weather
missingness. This no-weather builder intentionally does not read weather data:
the current release has no contemporaneously captured forecasts eligible at its
prediction cutoffs.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .data_archive import CoverageLedger, CoverageRecord, CoverageStatus, write_atomic_json
from .feds_labels import FEDS_LABEL_BUILD_VERSION
from .fire_state_features import FireStateFeatureError, build_firms_fire_state_features
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .storage_budget import StorageBudgetPolicy, require_admission
from .terrain_features import TerrainFeatureError, TerrainFeatureSampler
from .training_grid import (
    DEFAULT_HORIZON_HOURS,
    GridCell,
    TrainingExampleKey,
    TrainingGridError,
    cell_from_id,
    cell_from_wgs84,
    cells_in_square_radius,
    format_utc,
)


TRAINING_DATASET_SCHEMA_VERSION = 1
# v3 makes daily FIRMS coverage a precondition of archive-backed assembly and
# publishes a completed view only after every selected label partition has
# finished. Earlier artifacts remain immutable evidence but are never selected
# as a current training table.
TRAINING_DATASET_BUILD_VERSION = "feds-weak-positive-tabular-1km-12h/v3-coverage-manifest"
TRAINING_DATASET_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_FIRMS_LOOKBACK = timedelta(hours=24)
DEFAULT_FIRMS_AVAILABILITY_LAG = timedelta(minutes=180)
DEFAULT_FIRMS_PRODUCTS = (
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
)
DEFAULT_FIRMS_REGION = "United States and Canada"
FIRMS_COVERAGE_POLICY = "require-terminal-product-day-through-availability-cutoff/v1"
DEFAULT_LABEL_BATCH_SIZE = 5_000
DEFAULT_TERRAIN_CACHE_BLOCKS = 2

FEDS_LABEL_COVERAGE_SOURCE = "wildfire-data training pipeline"
FEDS_LABEL_COVERAGE_PRODUCT = "feds-weak-labels"
_NORMALIZED_ARTIFACT_ID_PATTERN = re.compile(r"[0-9a-f]{64}")

FEDS_WEAK_POSITIVE_OBSERVABILITY = "satellite-weak-positive-only"
WEATHER_FEATURE_STATUS = "unavailable-no-issued-forecast-features"
WEATHER_MISSING_REASON = "issued-forecast-features-not-collected"
WEATHER_INPUT_POLICY = "do-not-substitute-retrospective-or-open-meteo-weather/v1"
POSITIVE_ONLY_TRAINING_STATUS = (
    "positive-only-requires-explicit-negative-or-observability-labels"
)


class TrainingDatasetError(ValueError):
    """Raised when source evidence cannot safely form a training row."""


@dataclass(frozen=True)
class TrainingDatasetBuildReport:
    """The immutable training views derived from one FEDS label artifact."""

    label_artifact_path: Path
    input_label_count: int
    training_row_count: int
    normalized_artifacts: tuple[NormalizedArtifact, ...]


@dataclass(frozen=True)
class TrainingDatasetBuildResult:
    """A bounded assembly outcome suitable for logs or a later trainer."""

    reports: tuple[TrainingDatasetBuildReport, ...]
    manifest_path: Path | None = None

    @property
    def input_label_count(self) -> int:
        return sum(report.input_label_count for report in self.reports)

    @property
    def training_row_count(self) -> int:
        return sum(report.training_row_count for report in self.reports)

    @property
    def normalized_artifact_count(self) -> int:
        return sum(len(report.normalized_artifacts) for report in self.reports)


@dataclass(frozen=True)
class _ValidatedFedsLabel:
    """The minimum source-label contract used by the row assembler."""

    record: Mapping[str, Any]
    example_key: TrainingExampleKey
    cell: GridCell
    source_snapshot_time: datetime
    label_raw_artifact_ids: tuple[str, ...]


def assemble_feds_weak_positive_examples(
    labels: Iterable[Mapping[str, Any]],
    *,
    firms_detections: Iterable[Mapping[str, Any]],
    terrain_sampler: TerrainFeatureSampler,
    firms_lookback: timedelta = DEFAULT_FIRMS_LOOKBACK,
    firms_availability_lag: timedelta = DEFAULT_FIRMS_AVAILABILITY_LAG,
) -> list[dict[str, Any]]:
    """Assemble a bounded batch of FEDS weak-positive rows.

    Callers should pass a deliberately small label batch (the archive builder
    uses :data:`DEFAULT_LABEL_BATCH_SIZE`).  FIRMS input is streamed once and
    only retained if it can affect a requested cell's 3-by-3 context during
    the batch's lookback range.  This avoids materialising a continental
    detection archive or scanning it once per label.

    The returned rows deliberately all have a target of one.  Absence from
    FEDS is not a negative label, so this function never invents zero-valued
    targets.  A later candidate/observability stage must provide negatives
    before :mod:`wildfire_data.tabular_baseline` can fit a binary classifier.
    """
    if not isinstance(terrain_sampler, TerrainFeatureSampler):
        raise TypeError("terrain_sampler must be a TerrainFeatureSampler")
    lookback = _nonnegative_timedelta(firms_lookback, "firms_lookback")
    availability_lag = _nonnegative_timedelta(
        firms_availability_lag, "firms_availability_lag"
    )
    validated_labels = tuple(_validated_feds_label(label) for label in labels)
    if not validated_labels:
        return []
    _require_unique_example_ids(validated_labels)
    detections_by_cell = _index_local_firms_detections(
        firms_detections,
        labels=validated_labels,
        lookback=lookback,
        availability_lag=availability_lag,
    )
    rows = []
    for label in sorted(
        validated_labels,
        key=lambda item: (item.example_key.anchor_at, item.example_key.cell_id),
    ):
        local_detections = _local_detections_for_cell(detections_by_cell, label.cell)
        try:
            firms_features = build_firms_fire_state_features(
                local_detections,
                cell_id=label.example_key.cell_id,
                cutoff_at=label.example_key.anchor_at,
                lookback=lookback,
                availability_lag=availability_lag,
            )
        except FireStateFeatureError as exc:
            raise TrainingDatasetError(
                f"Could not build FIRMS features for {label.example_key.example_id}: {exc}"
            ) from exc
        try:
            terrain_features = terrain_sampler.sample_cell(label.cell)
        except TerrainFeatureError as exc:
            raise TrainingDatasetError(
                f"Could not sample terrain for {label.example_key.example_id}: {exc}"
            ) from exc
        rows.append(
            _assembled_row(
                label,
                firms_features=firms_features,
                terrain_features=terrain_features,
                firms_raw_artifact_ids=_eligible_firms_raw_artifact_ids(
                    local_detections,
                    cutoff_at=label.example_key.anchor_at,
                    lookback=lookback,
                    availability_lag=availability_lag,
                ),
            )
        )
    return rows


def build_and_store_feds_weak_positive_training_dataset(
    data_root: str | Path,
    *,
    storage_budget: StorageBudgetPolicy,
    start_date: date,
    end_date: date,
    firms_lookback: timedelta = DEFAULT_FIRMS_LOOKBACK,
    firms_availability_lag: timedelta = DEFAULT_FIRMS_AVAILABILITY_LAG,
    firms_products: Sequence[str] = DEFAULT_FIRMS_PRODUCTS,
    firms_region: str = DEFAULT_FIRMS_REGION,
    label_batch_size: int = DEFAULT_LABEL_BATCH_SIZE,
    max_cached_terrain_blocks: int = DEFAULT_TERRAIN_CACHE_BLOCKS,
) -> TrainingDatasetBuildResult:
    """Build quota-admitted training rows from archived FEDS weak labels.

    ``start_date`` and ``end_date`` select the FEDS *source snapshot* dates,
    not retrieval dates.  The function streams one label artifact and one
    bounded label batch at a time. It requires a terminal FIRMS coverage
    record for every configured product/day intersecting a batch's usable
    ``[cutoff - lookback, cutoff - availability_lag]`` interval, then loads
    only those date partitions. An absent archive partition is never treated
    as a zero-fire feature value.

    The output entity is ``normalized/training-examples``.  It is an immutable
    JSONL view with full source-artifact lineage; source evidence is never
    changed or compacted by this builder.
    """
    if end_date < start_date:
        raise TrainingDatasetError("end_date must not be before start_date")
    lookback = _nonnegative_timedelta(firms_lookback, "firms_lookback")
    availability_lag = _nonnegative_timedelta(
        firms_availability_lag, "firms_availability_lag"
    )
    products = _validated_firms_products(firms_products)
    region = _required_text(firms_region, "firms_region")
    if not isinstance(label_batch_size, int) or isinstance(label_batch_size, bool):
        raise TrainingDatasetError("label_batch_size must be a positive integer")
    if label_batch_size <= 0:
        raise TrainingDatasetError("label_batch_size must be a positive integer")

    root = Path(data_root)
    sampler = TerrainFeatureSampler(root, max_cached_blocks=max_cached_terrain_blocks)
    latest_firms_coverage = _latest_firms_coverage(CoverageLedger(root))
    reports = []
    for label_path in iter_feds_weak_positive_label_paths(root):
        input_count = 0
        output_count = 0
        artifacts = []
        for labels in _iter_selected_label_batches(
            label_path,
            start_date=start_date,
            end_date=end_date,
            batch_size=label_batch_size,
        ):
            input_count += len(labels)
            validated_labels = tuple(_validated_feds_label(label) for label in labels)
            _require_terminal_firms_coverage(
                validated_labels,
                lookback=lookback,
                availability_lag=availability_lag,
                products=products,
                region=region,
                latest_by_expected_id=latest_firms_coverage,
            )
            rows = assemble_feds_weak_positive_examples(
                labels,
                firms_detections=_iter_local_firms_detections_from_archive(
                    root,
                    labels=validated_labels,
                    lookback=lookback,
                    availability_lag=availability_lag,
                ),
                terrain_sampler=sampler,
                firms_lookback=lookback,
                firms_availability_lag=availability_lag,
            )
            output_count += len(rows)
            artifacts.extend(
                _store_training_rows_by_anchor_date(
                    root,
                    rows=rows,
                    storage_budget=storage_budget,
                )
            )
        if input_count:
            reports.append(
                TrainingDatasetBuildReport(
                    label_artifact_path=label_path,
                    input_label_count=input_count,
                    training_row_count=output_count,
                    normalized_artifacts=tuple(artifacts),
                )
            )
    completed_reports = tuple(reports)
    result = TrainingDatasetBuildResult(reports=completed_reports)
    if result.training_row_count == 0:
        return result
    manifest_path = _write_completed_training_dataset_manifest(
        root,
        reports=completed_reports,
        start_date=start_date,
        end_date=end_date,
        firms_lookback=lookback,
        firms_availability_lag=availability_lag,
        firms_products=products,
        firms_region=region,
    )
    return TrainingDatasetBuildResult(
        reports=completed_reports,
        manifest_path=manifest_path,
    )


def iter_feds_weak_positive_label_paths(data_root: str | Path) -> tuple[Path, ...]:
    """Return the current FEDS weak-label artifacts in deterministic path order.

    FEDS label artifacts are immutable, so a refreshed source window leaves
    its earlier revision on disk.  When the coverage ledger has selected
    normalized artifact identities, it is the authoritative current view:
    choose the newest successful artifact for each expected label scope
    rather than globbing every historical revision.  Legacy archives created
    before label coverage recorded artifact identities keep the original
    filesystem fallback.
    """
    root = Path(data_root) / "normalized" / "training-labels"
    selected_artifact_ids = _selected_feds_weak_positive_label_artifact_ids(data_root)
    if selected_artifact_ids is None:
        if not root.exists():
            return ()
        return tuple(sorted(path for path in root.rglob("*.jsonl.gz") if path.is_file()))
    return _resolve_selected_feds_weak_positive_label_paths(root, selected_artifact_ids)


def _selected_feds_weak_positive_label_artifact_ids(
    data_root: str | Path,
) -> tuple[str, ...] | None:
    """Return ledger-selected label artifact IDs, or ``None`` for legacy data.

    A later partial outcome cannot replace a prior completed positive-label
    artifact: partial FEDS windows intentionally have no artifact when they
    show no positive expansion.  This mirrors the source-snapshot selection
    rule and considers only complete coverage entries with an explicit
    immutable artifact identity.
    """
    latest_by_expected_id: dict[str, str] = {}
    found_selection_metadata = False
    expected_prefix = f"feds-weak-labels:{FEDS_LABEL_BUILD_VERSION}:"
    for record in CoverageLedger(data_root).entries():
        if (
            record.source != FEDS_LABEL_COVERAGE_SOURCE
            or record.product != FEDS_LABEL_COVERAGE_PRODUCT
            or record.status is not CoverageStatus.COMPLETE
            or record.expected_coverage_id is None
            or not record.expected_coverage_id.startswith(expected_prefix)
        ):
            continue
        artifact_id = _normalized_artifact_id_from_feds_label_coverage(record)
        if artifact_id is None:
            continue
        found_selection_metadata = True
        latest_by_expected_id[record.expected_coverage_id] = artifact_id
    if not found_selection_metadata:
        return None
    return tuple(sorted(set(latest_by_expected_id.values())))


def _normalized_artifact_id_from_feds_label_coverage(record: CoverageRecord) -> str | None:
    """Read one selected label artifact identity from an immutable ledger entry."""
    try:
        document = json.loads(record.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDatasetError(
            f"could not read FEDS weak-label coverage record: {record.path}"
        ) from exc
    detail = document.get("detail")
    artifact_id = detail.get("normalized_artifact_id") if isinstance(detail, Mapping) else None
    if artifact_id is None:
        return None
    if not isinstance(artifact_id, str) or not _NORMALIZED_ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
        raise TrainingDatasetError(
            "FEDS weak-label coverage has an invalid normalized artifact identity: "
            f"{record.path}"
        )
    return artifact_id


def _resolve_selected_feds_weak_positive_label_paths(
    root: Path,
    artifact_ids: Sequence[str],
) -> tuple[Path, ...]:
    """Resolve each ledger-selected immutable label artifact exactly once."""
    expected_names = {f"{artifact_id}.jsonl.gz": artifact_id for artifact_id in artifact_ids}
    matches: dict[str, list[Path]] = {artifact_id: [] for artifact_id in artifact_ids}
    if root.exists():
        for path in root.rglob("*.jsonl.gz"):
            artifact_id = expected_names.get(path.name)
            if artifact_id is not None and path.is_file():
                matches[artifact_id].append(path)
    resolved = []
    for artifact_id in artifact_ids:
        paths = matches[artifact_id]
        if len(paths) != 1:
            state = "missing" if not paths else "ambiguous"
            raise TrainingDatasetError(
                f"selected FEDS weak-label artifact is {state}: {artifact_id}"
            )
        resolved.append(paths[0])
    return tuple(sorted(resolved))


def iter_training_examples(
    data_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield rows from one completed coverage-verified derived view.

    Previous content-addressed views are intentionally retained for lineage,
    but mixing them with a rebuilt version would duplicate examples and could
    reintroduce rows made before FIRMS edge coverage was checked. By default,
    the newest completed current-version manifest is selected. Supply its
    path explicitly when reproducing an older completed build.
    """
    root = Path(data_root)
    selected_manifest = _selected_training_dataset_manifest(root, manifest_path=manifest_path)
    if selected_manifest is None:
        return
    seen_example_ids: set[str] = set()
    for relative_path in selected_manifest["artifact_relative_paths"]:
        path = _training_artifact_path(root, relative_path)
        for record in _iter_jsonl_records(path, context="training example"):
            if record.get("training_dataset_build_version") != TRAINING_DATASET_BUILD_VERSION:
                raise TrainingDatasetError(
                    f"training example artifact has a mismatched build version: {path}"
                )
            example_id = _required_text(record.get("example_id"), "training example example_id")
            if example_id in seen_example_ids:
                raise TrainingDatasetError(
                    f"duplicate example_id across completed training view: {example_id}"
                )
            seen_example_ids.add(example_id)
            yield record


def _write_completed_training_dataset_manifest(
    data_root: Path,
    *,
    reports: Sequence[TrainingDatasetBuildReport],
    start_date: date,
    end_date: date,
    firms_lookback: timedelta,
    firms_availability_lag: timedelta,
    firms_products: Sequence[str],
    firms_region: str,
) -> Path:
    """Atomically publish one complete selection of immutable row artifacts.

    Individual JSONL artifacts are immutable and may be left behind by an
    interrupted run. This manifest is the single commit point: readers select
    only artifact paths listed here, and it is written only after every source
    label batch has completed.
    """
    artifacts = sorted(
        {
            artifact
            for report in reports
            for artifact in report.normalized_artifacts
        },
        key=lambda artifact: artifact.artifact_path.as_posix(),
    )
    if not artifacts:
        raise TrainingDatasetError("cannot publish a completed training view without row artifacts")
    try:
        artifact_relative_paths = [
            artifact.artifact_path.relative_to(data_root).as_posix() for artifact in artifacts
        ]
        label_relative_paths = sorted(
            {report.label_artifact_path.relative_to(data_root).as_posix() for report in reports}
        )
    except ValueError as exc:
        raise TrainingDatasetError("training artifact path is outside data_root") from exc
    completed_at = datetime.now(timezone.utc)
    build_id = uuid.uuid4().hex
    document = {
        "schema_version": TRAINING_DATASET_MANIFEST_SCHEMA_VERSION,
        "kind": "completed-training-dataset-build",
        "status": "complete",
        "build_id": build_id,
        "completed_at": format_utc(completed_at),
        "training_dataset_build_version": TRAINING_DATASET_BUILD_VERSION,
        "source_snapshot_start_date": start_date.isoformat(),
        "source_snapshot_end_date": end_date.isoformat(),
        "firms_feature_policy": {
            "coverage_policy": FIRMS_COVERAGE_POLICY,
            "products": list(firms_products),
            "region": firms_region,
            "lookback_hours": firms_lookback.total_seconds() / 3_600,
            "availability_lag_minutes": firms_availability_lag.total_seconds() / 60,
        },
        "input_label_artifact_relative_paths": label_relative_paths,
        "artifact_relative_paths": artifact_relative_paths,
        "normalized_artifact_ids": [artifact.normalized_artifact_id for artifact in artifacts],
        "input_label_count": sum(report.input_label_count for report in reports),
        "training_row_count": sum(report.training_row_count for report in reports),
    }
    destination = (
        data_root
        / "manifests"
        / "training-dataset-builds"
        / completed_at.strftime("%Y/%m/%d")
        / f"{completed_at.strftime('%H%M%S%f')}_{build_id}.json"
    )
    return write_atomic_json(destination, document)


def _selected_training_dataset_manifest(
    data_root: Path,
    *,
    manifest_path: str | Path | None,
) -> dict[str, Any] | None:
    """Return one valid, completed manifest or no view when none exists."""
    if manifest_path is not None:
        candidate = Path(manifest_path)
        if not candidate.is_absolute():
            working_directory_path = candidate.resolve()
            resolved_data_root = data_root.resolve()
            candidate = (
                working_directory_path
                if working_directory_path.is_file()
                and working_directory_path.is_relative_to(resolved_data_root)
                else data_root / candidate
            )
        return _read_completed_training_dataset_manifest(candidate, required=True)
    manifest_root = data_root / "manifests" / "training-dataset-builds"
    if not manifest_root.exists():
        return None
    candidates = []
    for path in sorted(manifest_root.rglob("*.json")):
        document = _read_completed_training_dataset_manifest(path, required=False)
        if document is not None:
            candidates.append(document)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda document: (document["completed_at"], document["build_id"]),
    )


def _read_completed_training_dataset_manifest(
    path: Path,
    *,
    required: bool,
) -> dict[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if required:
            raise TrainingDatasetError(f"could not read training dataset manifest: {path}") from exc
        raise TrainingDatasetError(f"invalid training dataset manifest: {path}") from exc
    if not isinstance(document, dict):
        if required:
            raise TrainingDatasetError(f"training dataset manifest must be an object: {path}")
        return None
    if document.get("kind") != "completed-training-dataset-build":
        return None
    if document.get("status") != "complete":
        if required:
            raise TrainingDatasetError(f"training dataset manifest is not complete: {path}")
        return None
    if document.get("schema_version") != TRAINING_DATASET_MANIFEST_SCHEMA_VERSION:
        if required:
            raise TrainingDatasetError(f"training dataset manifest has an unsupported schema: {path}")
        return None
    if document.get("training_dataset_build_version") != TRAINING_DATASET_BUILD_VERSION:
        if required:
            raise TrainingDatasetError(f"training dataset manifest has a different build version: {path}")
        return None
    try:
        completed_at = _parse_utc(document.get("completed_at"), "manifest completed_at")
        build_id = _required_text(document.get("build_id"), "manifest build_id")
        paths = document.get("artifact_relative_paths")
        if not isinstance(paths, list) or not paths:
            raise TrainingDatasetError("manifest artifact_relative_paths must be a non-empty list")
        relative_paths = [_validated_training_artifact_relative_path(item) for item in paths]
    except TrainingDatasetError:
        if required:
            raise
        return None
    if len(set(relative_paths)) != len(relative_paths):
        if required:
            raise TrainingDatasetError("manifest artifact_relative_paths must not contain duplicates")
        return None
    return {
        "completed_at": completed_at,
        "build_id": build_id,
        "artifact_relative_paths": tuple(relative_paths),
        "path": path,
    }


def _validated_training_artifact_relative_path(value: object) -> str:
    path = Path(_required_text(value, "manifest artifact path"))
    if path.is_absolute() or ".." in path.parts:
        raise TrainingDatasetError("manifest artifact path must be relative to data_root")
    normalized = path.as_posix()
    if not normalized.startswith("normalized/training-examples/"):
        raise TrainingDatasetError("manifest artifact path is outside normalized/training-examples")
    return normalized


def _training_artifact_path(data_root: Path, relative_path: str) -> Path:
    path = data_root / relative_path
    if not path.is_file():
        raise TrainingDatasetError(f"training view artifact is missing: {path}")
    return path


def _iter_selected_label_batches(
    path: Path,
    *,
    start_date: date,
    end_date: date,
    batch_size: int,
) -> Iterator[tuple[dict[str, Any], ...]]:
    batch: list[dict[str, Any]] = []
    for record in _iter_jsonl_records(path, context="FEDS label"):
        label = _validated_feds_label(record)
        source_date = label.source_snapshot_time.date()
        if source_date < start_date or source_date > end_date:
            continue
        batch.append(record)
        if len(batch) >= batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _validated_firms_products(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TrainingDatasetError("firms_products must be a non-empty sequence of product names")
    products = tuple(_required_text(value, "FIRMS product") for value in values)
    if not products:
        raise TrainingDatasetError("firms_products must not be empty")
    if len(set(products)) != len(products):
        raise TrainingDatasetError("firms_products must not contain duplicates")
    return products


def _latest_firms_coverage(ledger: CoverageLedger) -> dict[str, CoverageRecord]:
    """Return the newest ledger outcome for each explicit FIRMS product/day."""
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
    labels: Sequence[_ValidatedFedsLabel],
    *,
    lookback: timedelta,
    availability_lag: timedelta,
    products: Sequence[str],
    region: str,
    latest_by_expected_id: Mapping[str, CoverageRecord],
) -> None:
    """Reject archive-backed rows whose usable FIRMS interval is unobserved.

    Daily FIRMS responses are source coverage records, not merely optional
    file partitions. A missing response must remain unknown rather than turn
    into a zero count. ``empty-confirmed`` is valid evidence; ``partial`` and
    ``failed`` are not.
    """
    required_dates = _required_firms_coverage_dates(
        labels,
        lookback=lookback,
        availability_lag=availability_lag,
    )
    terminal = {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED}
    incomplete: list[str] = []
    for coverage_date in required_dates:
        for product in products:
            expected_id = _firms_expected_coverage_id(product, region, coverage_date)
            record = latest_by_expected_id.get(expected_id)
            if record is None:
                incomplete.append(f"{product}/{coverage_date.isoformat()} (missing)")
            elif record.status not in terminal:
                incomplete.append(f"{product}/{coverage_date.isoformat()} ({record.status.value})")
    if incomplete:
        detail = ", ".join(incomplete[:12])
        if len(incomplete) > 12:
            detail = f"{detail}, … ({len(incomplete):,} incomplete product-days total)"
        raise TrainingDatasetError(
            "FIRMS coverage is incomplete for availability-gated feature windows under "
            f"{FIRMS_COVERAGE_POLICY}: {detail}. Collect/retry those exact product-days "
            "before building training rows."
        )


def _required_firms_coverage_dates(
    labels: Sequence[_ValidatedFedsLabel],
    *,
    lookback: timedelta,
    availability_lag: timedelta,
) -> tuple[date, ...]:
    if not labels:
        return ()
    earliest = min(label.example_key.anchor_at for label in labels) - lookback
    latest = max(label.example_key.anchor_at for label in labels) - availability_lag
    if latest < earliest:
        return ()
    dates = []
    current = earliest.date()
    final = latest.date()
    while current <= final:
        dates.append(current)
        current += timedelta(days=1)
    return tuple(dates)


def _firms_expected_coverage_id(product: str, region: str, coverage_date: date) -> str:
    return f"firms:{product}:{region}:{coverage_date.isoformat()}"


def _iter_local_firms_detections_from_archive(
    data_root: Path,
    *,
    labels: Sequence[_ValidatedFedsLabel],
    lookback: timedelta,
    availability_lag: timedelta,
) -> Iterator[dict[str, Any]]:
    """Yield only date partitions that could affect the supplied label batch.

    The upper bound is the latest *available* acquisition, not the raw
    cutoff. That avoids reading a next-day partition containing detections
    which the configured FIRMS latency would forbid using anyway.
    """
    if not labels:
        return
    earliest = min(label.example_key.anchor_at for label in labels) - lookback
    latest = max(label.example_key.anchor_at for label in labels) - availability_lag
    if latest < earliest:
        return
    current_date = earliest.date()
    final_date = latest.date()
    root = data_root / "normalized" / "fire-detections"
    while current_date <= final_date:
        partition = root / f"acq-date={current_date.isoformat()}"
        if partition.exists():
            for path in sorted(partition.glob("*.jsonl.gz")):
                for record in _iter_jsonl_records(path, context="FIRMS detection"):
                    if record.get("record_type") == "firms_detection":
                        yield record
        current_date += timedelta(days=1)


def _index_local_firms_detections(
    detections: Iterable[Mapping[str, Any]],
    *,
    labels: Sequence[_ValidatedFedsLabel],
    lookback: timedelta,
    availability_lag: timedelta,
) -> dict[str, list[Mapping[str, Any]]]:
    """Index only detection cells that can affect this bounded label batch."""
    candidate_cells = {
        neighbour.cell_id
        for label in labels
        for neighbour in cells_in_square_radius(label.cell, radius_cells=1)
    }
    earliest = min(label.example_key.anchor_at for label in labels) - lookback
    latest = max(label.example_key.anchor_at for label in labels) - availability_lag
    if latest < earliest:
        return {}
    indexed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in detections:
        if not isinstance(record, Mapping):
            raise TrainingDatasetError("each FIRMS detection must be a mapping")
        if record.get("record_type") != "firms_detection":
            continue
        acquired_at = _parse_utc(record.get("acquired_at"), "FIRMS acquired_at")
        if acquired_at < earliest or acquired_at > latest:
            continue
        try:
            latitude = _finite_float(record.get("latitude"), "FIRMS latitude")
            longitude = _finite_float(record.get("longitude"), "FIRMS longitude")
            cell = cell_from_wgs84(latitude=latitude, longitude=longitude)
        except TrainingGridError as exc:
            raise TrainingDatasetError("FIRMS detection cannot be mapped to the training grid") from exc
        if cell.cell_id in candidate_cells:
            indexed[cell.cell_id].append(record)
    return dict(indexed)


def _local_detections_for_cell(
    detections_by_cell: Mapping[str, Sequence[Mapping[str, Any]]], cell: GridCell
) -> list[Mapping[str, Any]]:
    return [
        detection
        for neighbour in cells_in_square_radius(cell, radius_cells=1)
        for detection in detections_by_cell.get(neighbour.cell_id, ())
    ]


def _assembled_row(
    label: _ValidatedFedsLabel,
    *,
    firms_features: Mapping[str, Any],
    terrain_features: Mapping[str, Any],
    firms_raw_artifact_ids: tuple[str, ...],
) -> dict[str, Any]:
    record = label.record
    latitude, longitude = label.cell.center_wgs84
    row = {
        "record_type": "training_example",
        "training_dataset_schema_version": TRAINING_DATASET_SCHEMA_VERSION,
        "training_dataset_build_version": TRAINING_DATASET_BUILD_VERSION,
        "training_grid": "naea-1km",
        "prediction_horizon_hours": DEFAULT_HORIZON_HOURS,
        "example_id": label.example_key.example_id,
        "cell_id": label.cell.cell_id,
        "cell_center_latitude": latitude,
        "cell_center_longitude": longitude,
        "anchor_at": format_utc(label.example_key.anchor_at),
        "feature_cutoff_at": format_utc(label.example_key.anchor_at),
        "target_end_at": format_utc(label.example_key.target_end_at),
        "target_newly_burned_12h": 1,
        "label_status": _required_text(record.get("label_status"), "label_status"),
        "label_observability": FEDS_WEAK_POSITIVE_OBSERVABILITY,
        "label_tier": _required_text(record.get("label_tier"), "label_tier"),
        "label_source": _required_text(record.get("label_source"), "label_source"),
        "label_quality_score": _finite_float(record.get("label_quality_score"), "label_quality_score"),
        "label_build_version": _required_text(record.get("label_build_version"), "label_build_version"),
        "positive_overlap_fraction": _finite_float(
            record.get("positive_overlap_fraction"), "positive_overlap_fraction"
        ),
        "source_snapshot_time": format_utc(label.source_snapshot_time),
        "target_snapshot_time": format_utc(
            _parse_utc(record.get("target_snapshot_time"), "target_snapshot_time")
        ),
        "source_time_semantics": _required_text(
            record.get("source_time_semantics"), "source_time_semantics"
        ),
        "time_alignment_mode": _required_text(
            record.get("time_alignment_mode"), "time_alignment_mode"
        ),
        "contributing_fire_count": _nonnegative_int(
            record.get("contributing_fire_count"), "contributing_fire_count"
        ),
        "label_raw_artifact_ids": list(label.label_raw_artifact_ids),
        "firms_raw_artifact_ids": list(firms_raw_artifact_ids),
        "weather_available": 0,
        "weather_missing_indicator": 1,
        "weather_feature_status": WEATHER_FEATURE_STATUS,
        "weather_missing_reason": WEATHER_MISSING_REASON,
        "weather_input_policy": WEATHER_INPUT_POLICY,
        "binary_training_status": POSITIVE_ONLY_TRAINING_STATUS,
    }
    row.update(dict(firms_features))
    row.update(dict(terrain_features))
    return row


def _validated_feds_label(record: Mapping[str, Any]) -> _ValidatedFedsLabel:
    if not isinstance(record, Mapping):
        raise TrainingDatasetError("each FEDS label must be a mapping")
    target = record.get("target_newly_burned_12h")
    if isinstance(target, bool) or target != 1:
        raise TrainingDatasetError(
            "FEDS weak-label assembler accepts only explicit positive newly-burned labels"
        )
    if record.get("label_observability") != FEDS_WEAK_POSITIVE_OBSERVABILITY:
        raise TrainingDatasetError("FEDS label does not declare positive-only satellite observability")
    if _required_text(record.get("label_build_version"), "label_build_version") != FEDS_LABEL_BUILD_VERSION:
        raise TrainingDatasetError(
            "FEDS label was built with a different source-time/observability contract"
        )
    if _required_text(record.get("label_tier"), "label_tier") != "weak_satellite":
        raise TrainingDatasetError("FEDS label tier must be weak_satellite")
    if _required_text(record.get("label_status"), "label_status") != "positive-observed":
        raise TrainingDatasetError("FEDS label status must be positive-observed")
    try:
        cell = cell_from_id(_required_text(record.get("cell_id"), "cell_id"))
        anchor_at = _parse_utc(record.get("anchor_at"), "anchor_at")
        example_key = TrainingExampleKey(cell_id=cell.cell_id, anchor_at=anchor_at)
    except TrainingGridError as exc:
        raise TrainingDatasetError("FEDS label has an invalid canonical training key") from exc
    example_id = _required_text(record.get("example_id"), "example_id")
    if example_id != example_key.example_id:
        raise TrainingDatasetError("FEDS label example_id does not match its cell_id and anchor_at")
    target_end_at = _parse_utc(record.get("target_end_at"), "target_end_at")
    if target_end_at != example_key.target_end_at:
        raise TrainingDatasetError("FEDS label target_end_at is not exactly 12 hours after anchor_at")
    source_snapshot_time = _parse_utc(record.get("source_snapshot_time"), "source_snapshot_time")
    target_snapshot_time = _parse_utc(record.get("target_snapshot_time"), "target_snapshot_time")
    if target_snapshot_time - source_snapshot_time != timedelta(hours=DEFAULT_HORIZON_HOURS):
        raise TrainingDatasetError("FEDS label snapshots must be exactly 12 hours apart")
    raw_artifact_ids = _label_raw_artifact_ids(record)
    return _ValidatedFedsLabel(
        record=record,
        example_key=example_key,
        cell=cell,
        source_snapshot_time=source_snapshot_time,
        label_raw_artifact_ids=raw_artifact_ids,
    )


def _require_unique_example_ids(labels: Sequence[_ValidatedFedsLabel]) -> None:
    seen: set[str] = set()
    for label in labels:
        example_id = label.example_key.example_id
        if example_id in seen:
            raise TrainingDatasetError(f"duplicate FEDS label example_id in one batch: {example_id}")
        seen.add(example_id)


def _label_raw_artifact_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    identifiers = set()
    direct = record.get("raw_artifact_ids")
    if isinstance(direct, (list, tuple)):
        identifiers.update(_nonempty_strings(direct))
    contributors = record.get("contributing_fires")
    if isinstance(contributors, list):
        for contributor in contributors:
            if not isinstance(contributor, Mapping):
                raise TrainingDatasetError("FEDS contributing_fires must contain mappings")
            identifiers.update(
                _nonempty_strings(
                    (
                        contributor.get("current_raw_artifact_id"),
                        contributor.get("future_raw_artifact_id"),
                    )
                )
            )
    return tuple(sorted(identifiers))


def _eligible_firms_raw_artifact_ids(
    detections: Iterable[Mapping[str, Any]],
    *,
    cutoff_at: datetime,
    lookback: timedelta,
    availability_lag: timedelta,
) -> tuple[str, ...]:
    """Return lineage only for detections that the FIRMS feature gate can use."""
    cutoff = _parse_utc(cutoff_at, "cutoff_at")
    window_start = cutoff - lookback
    latest_eligible = cutoff - availability_lag
    identifiers = set()
    for detection in detections:
        acquired_at = _parse_utc(detection.get("acquired_at"), "FIRMS acquired_at")
        if acquired_at < window_start or acquired_at > latest_eligible:
            continue
        provenance = detection.get("provenance")
        if isinstance(provenance, Mapping):
            identifiers.update(_nonempty_strings((provenance.get("raw_artifact_id"),)))
    return tuple(sorted(identifiers))


def _store_training_rows_by_anchor_date(
    data_root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    storage_budget: StorageBudgetPolicy,
) -> tuple[NormalizedArtifact, ...]:
    by_anchor_date: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        anchor_at = _parse_utc(row.get("anchor_at"), "training row anchor_at")
        by_anchor_date[anchor_at.date().isoformat()].append(row)
    artifacts = []
    for anchor_date, date_rows in sorted(by_anchor_date.items()):
        raw_artifact_ids = sorted(
            {
                artifact_id
                for row in date_rows
                for artifact_id in _nonempty_strings(
                    [
                        *row.get("label_raw_artifact_ids", []),
                        *row.get("firms_raw_artifact_ids", []),
                    ]
                )
            }
        )
        if not raw_artifact_ids:
            raise TrainingDatasetError(
                "cannot persist training rows without retained FEDS or FIRMS raw-artifact lineage"
            )
        require_admission(
            storage_budget,
            data_root,
            category="derived_training_views",
            requested_bytes=_conservative_training_view_bytes(date_rows),
        )
        artifacts.append(
            write_normalized_jsonl(
                data_root,
                entity="training_examples",
                records=date_rows,
                partitions={
                    "source": "feds-weak-positive-labels",
                    "dataset_build": TRAINING_DATASET_BUILD_VERSION,
                    "anchor_date": anchor_date,
                    "grid": "naea-1km",
                },
                raw_artifact_ids=raw_artifact_ids,
                transformation_version=TRAINING_DATASET_BUILD_VERSION,
            )
        )
    return tuple(artifacts)


def _conservative_training_view_bytes(records: Iterable[Mapping[str, Any]]) -> int:
    encoded = sum(
        len(json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        + 1
        for record in records
    )
    # The uncompressed JSONL, gzip artifact, and a small manifest allowance.
    # This is intentionally conservative before a governed write occurs.
    return encoded * 2 + 65_536


def _iter_jsonl_records(path: Path, *, context: str) -> Iterator[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TrainingDatasetError(
                        f"{context} artifact {path} has a non-object JSON record at line {line_number}"
                    )
                yield record
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingDatasetError(f"Could not read {context} artifact: {path}") from exc


def _parse_utc(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise TrainingDatasetError(f"{label} must be an offset-aware ISO-8601 timestamp") from exc
    else:
        raise TrainingDatasetError(f"{label} must be an offset-aware ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise TrainingDatasetError(f"{label} must be an offset-aware ISO-8601 timestamp")
    return parsed.astimezone(timezone.utc)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingDatasetError(f"{label} must be non-empty text")
    return value.strip()


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise TrainingDatasetError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingDatasetError(f"{label} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise TrainingDatasetError(f"{label} must be a finite number")
    return numeric


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TrainingDatasetError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_timedelta(value: timedelta, label: str) -> timedelta:
    if not isinstance(value, timedelta) or value < timedelta(0):
        raise TrainingDatasetError(f"{label} must be a non-negative timedelta")
    return value


def _nonempty_strings(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip()
    )
