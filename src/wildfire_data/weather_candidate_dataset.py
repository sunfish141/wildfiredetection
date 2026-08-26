"""Build and export a weather-bearing candidate dataset without mutating its spine.

The existing FIRMS/FEDS candidate view remains immutable.  This module joins a
*complete* historical-weather backfill to exactly one such view using the
candidate cell, the candidate example ID, and the UTC hour at or before the
prediction anchor.  The result is a separate uploadable data product with a
small explicit weather feature allowlist.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .candidate_dataset import DEFAULT_MODEL_FEATURE_COLUMNS, iter_candidate_examples
from .data_archive import write_atomic_json
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .open_meteo_historical import (
    OPEN_METEO_HISTORICAL_FEATURE_MODE,
    OPEN_METEO_HISTORICAL_WEATHER_KIND,
    OPEN_METEO_HISTORICAL_WEATHER_MODEL,
    floor_weather_hour,
    required_model_weather_variables,
)
from .storage_budget import StorageBudgetPolicy, require_admission


WEATHER_CANDIDATE_DATASET_BUILD_VERSION = "firms-feds-weather-candidate-tabular-1km-12h/v1"
WEATHER_CANDIDATE_DATASET_MANIFEST_SCHEMA_VERSION = 1
WEATHER_CANDIDATE_DATASET_RELEASE_SCHEMA_VERSION = 1
WEATHER_FEATURE_STATUS = "complete-open-meteo-historical-weather/v1"
WEATHER_INPUT_POLICY = "historical-weather-at-or-before-anchor-hour/v1"
WEATHER_FEATURE_COLUMNS = (
    "weather_temperature_2m",
    "weather_relative_humidity_2m",
    "weather_precipitation",
    "weather_wind_u_10m",
    "weather_wind_v_10m",
)
WEATHER_MODEL_FEATURE_COLUMNS = (*DEFAULT_MODEL_FEATURE_COLUMNS, *WEATHER_FEATURE_COLUMNS)


class WeatherCandidateDatasetError(ValueError):
    """The selected candidate or weather artifacts cannot form one dataset."""


@dataclass(frozen=True)
class BaseCandidateManifestIdentity:
    """The exact immutable base view selected for a weather dataset build."""

    path: Path
    relative_path: str
    build_id: str
    content_sha256: str


@dataclass(frozen=True)
class WeatherCandidateDatasetBuildResult:
    """The completed immutable weather-bearing candidate view."""

    manifest_path: Path
    candidate_row_count: int
    weather_date_count: int
    normalized_artifacts: tuple[NormalizedArtifact, ...]


@dataclass(frozen=True)
class WeatherCandidateDatasetRelease:
    """A portable export of one completed weather-bearing candidate view."""

    directory: Path
    manifest_path: Path
    candidate_row_count: int


def build_weather_candidate_dataset(
    data_root: str | Path,
    *,
    storage_budget: StorageBudgetPolicy,
    weather_backfill_manifest: str | Path,
    candidate_manifest: str | Path,
) -> WeatherCandidateDatasetBuildResult:
    """Attach complete hourly weather to an existing candidate view.

    No row can silently fall back to a later hour, another candidate's tile,
    or an old exploratory export.  A missing required weather field aborts the
    build before a completed manifest is published.
    """
    root = Path(data_root).resolve()
    candidate_identity = _candidate_manifest_identity(root, candidate_manifest)
    backfill_path = _resolve_path(root, weather_backfill_manifest)
    backfill = _read_json(backfill_path, "weather backfill manifest")
    reports = _validated_complete_backfill(backfill, root)
    _require_matching_candidate_identity(backfill, candidate_identity)
    candidates_by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for candidate in iter_candidate_examples(root, manifest_path=candidate_identity.path):
        anchor_at = _parse_utc(candidate.get("anchor_at"), "candidate anchor_at")
        candidates_by_date[anchor_at.date()].append(candidate)
    if not candidates_by_date:
        raise WeatherCandidateDatasetError("selected candidate view contains no examples")

    missing_dates = sorted(set(candidates_by_date).difference(reports))
    if missing_dates:
        rendered = ", ".join(value.isoformat() for value in missing_dates[:5])
        raise WeatherCandidateDatasetError(
            "complete weather backfill has no report for candidate date(s): " + rendered
        )

    artifacts: list[NormalizedArtifact] = []
    total_rows = 0
    for weather_date, candidates in sorted(candidates_by_date.items()):
        report = reports[weather_date]
        mappings = _load_mappings(root, report)
        weather = _load_weather(root, report)
        enriched: list[dict[str, Any]] = []
        missing: list[str] = []
        for candidate in candidates:
            try:
                enriched.append(
                    _join_candidate_weather(candidate, mappings=mappings, weather=weather)
                )
            except WeatherCandidateDatasetError as exc:
                missing.append(f"{candidate.get('example_id')}: {exc}")
                if len(missing) >= 5:
                    break
        if missing:
            raise WeatherCandidateDatasetError(
                f"weather is incomplete for {weather_date.isoformat()}: " + "; ".join(missing)
            )
        raw_ids = _source_raw_artifact_ids(enriched)
        if not raw_ids:
            raise WeatherCandidateDatasetError("weather-enriched rows lack raw-artifact lineage")
        require_admission(
            storage_budget,
            root,
            category="derived_training_views",
            requested_bytes=_conservative_bytes(enriched),
        )
        artifact = write_normalized_jsonl(
            root,
            entity="weather_candidate_examples",
            records=enriched,
            partitions={"anchor_date": weather_date.isoformat()},
            raw_artifact_ids=sorted(raw_ids),
            transformation_version=WEATHER_CANDIDATE_DATASET_BUILD_VERSION,
        )
        artifacts.append(artifact)
        total_rows += len(enriched)

    manifest_path = _write_completed_manifest(
        root,
        candidate_manifest=candidate_identity,
        weather_backfill_manifest=backfill_path,
        backfill=backfill,
        artifacts=artifacts,
        candidate_row_count=total_rows,
    )
    return WeatherCandidateDatasetBuildResult(
        manifest_path=manifest_path,
        candidate_row_count=total_rows,
        weather_date_count=len(artifacts),
        normalized_artifacts=tuple(artifacts),
    )


def iter_weather_candidate_examples(
    data_root: str | Path,
    *,
    manifest_path: str | Path,
) -> Iterator[dict[str, Any]]:
    """Yield only records selected by one completed weather dataset manifest."""
    root = Path(data_root).resolve()
    manifest = _read_completed_manifest(_resolve_path(root, manifest_path), root)
    seen: set[str] = set()
    for relative_path in manifest["candidate_artifact_relative_paths"]:
        for record in _iter_jsonl(root / relative_path, "weather candidate example"):
            if record.get("weather_candidate_dataset_build_version") != (
                WEATHER_CANDIDATE_DATASET_BUILD_VERSION
            ):
                raise WeatherCandidateDatasetError("weather candidate artifact has a mismatched version")
            example_id = _required_text(record.get("example_id"), "example_id")
            if example_id in seen:
                raise WeatherCandidateDatasetError(
                    "duplicate example_id across weather candidate view: " + example_id
                )
            seen.add(example_id)
            yield record


def export_weather_candidate_dataset_release(
    data_root: str | Path,
    output_directory: str | Path,
    *,
    weather_candidate_manifest: str | Path,
) -> WeatherCandidateDatasetRelease:
    """Export a new self-contained weather-bearing upload directory."""
    root = Path(data_root).resolve()
    manifest = _read_completed_manifest(_resolve_path(root, weather_candidate_manifest), root)
    destination = Path(output_directory)
    if destination.exists():
        raise WeatherCandidateDatasetError(
            f"weather release destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent))
    try:
        candidate_path = stage / "candidate_examples.jsonl.gz"
        candidate_count, content_sha256 = _write_release_jsonl(
            root,
            manifest["candidate_artifact_relative_paths"],
            candidate_path,
        )
        schema = {
            "schema_version": WEATHER_CANDIDATE_DATASET_RELEASE_SCHEMA_VERSION,
            "format": "gzip-compressed JSON Lines",
            "candidate_dataset_build_version": WEATHER_CANDIDATE_DATASET_BUILD_VERSION,
            "model_feature_columns": list(WEATHER_MODEL_FEATURE_COLUMNS),
            "target_column": "target_newly_burned_12h",
            "split_column": "dataset_split",
        }
        _write_json(stage / "schema.json", schema)
        release_manifest = {
            "schema_version": WEATHER_CANDIDATE_DATASET_RELEASE_SCHEMA_VERSION,
            "kind": "wildfire-spread-weather-candidate-dataset-release",
            "weather_candidate_build_id": manifest["build_id"],
            "candidate_row_count": candidate_count,
            "candidate_examples_content_sha256": content_sha256,
            "source_candidate_manifest": manifest["candidate_manifest"],
            "source_weather_backfill_manifest": manifest["weather_backfill_manifest"],
            "weather": manifest["weather"],
            "model_feature_columns": list(WEATHER_MODEL_FEATURE_COLUMNS),
            "limitations": [
                "target=0 remains a FIRMS-seeded weak-negative proxy, not observed clear/no-burn.",
                "Weather is retrospective Open-Meteo historical weather at the UTC hour at or before the anchor; it is not a preserved issued forecast.",
                "A live model must use an equivalent contemporaneous weather feed or a separately trained forecast-vintage feature set.",
            ],
        }
        _write_json(stage / "dataset_manifest.json", release_manifest)
        (stage / "README.md").write_text(_release_readme(release_manifest), encoding="utf-8")
        _write_json(stage / "file_inventory.json", _file_inventory(stage, exclude={"file_inventory.json", "SHA256SUMS"}))
        _write_checksums(stage)
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return WeatherCandidateDatasetRelease(
        directory=destination,
        manifest_path=destination / "dataset_manifest.json",
        candidate_row_count=candidate_count,
    )


def _validated_complete_backfill(
    document: Mapping[str, Any],
    root: Path,
) -> dict[date, Mapping[str, Any]]:
    if document.get("kind") != "open-meteo-historical-weather-backfill":
        raise WeatherCandidateDatasetError("weather backfill manifest has the wrong kind")
    if document.get("status") != "complete":
        raise WeatherCandidateDatasetError(
            "weather backfill is not complete; resume it before building a weather dataset"
        )
    if document.get("product_kind") != OPEN_METEO_HISTORICAL_WEATHER_KIND:
        raise WeatherCandidateDatasetError("weather backfill has an unsupported weather product")
    if document.get("model") != OPEN_METEO_HISTORICAL_WEATHER_MODEL:
        raise WeatherCandidateDatasetError("weather backfill is not pinned to ECMWF IFS")
    if document.get("feature_mode") != OPEN_METEO_HISTORICAL_FEATURE_MODE:
        raise WeatherCandidateDatasetError("weather backfill has the wrong feature mode")
    reports = document.get("reports")
    if not isinstance(reports, list) or not reports:
        raise WeatherCandidateDatasetError("weather backfill must contain non-empty reports")
    indexed: dict[date, Mapping[str, Any]] = {}
    for report in reports:
        if not isinstance(report, Mapping) or report.get("status") != "complete":
            raise WeatherCandidateDatasetError("weather backfill contains an incomplete date report")
        try:
            weather_date = date.fromisoformat(_required_text(report.get("weather_date"), "weather_date"))
        except ValueError as exc:
            raise WeatherCandidateDatasetError("weather backfill report has an invalid date") from exc
        if weather_date in indexed:
            raise WeatherCandidateDatasetError("weather backfill has duplicate date reports")
        _relative_paths(report.get("measurement_artifact_relative_paths"), root, "historical_weather")
        _relative_paths(
            report.get("assignment_artifact_relative_paths"),
            root,
            "open_meteo_historical_weather_tile_assignments",
        )
        indexed[weather_date] = report
    return indexed


def _load_mappings(root: Path, report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    mappings: dict[str, dict[str, Any]] = {}
    for relative_path in _relative_paths(
        report.get("assignment_artifact_relative_paths"),
        root,
        "open_meteo_historical_weather_tile_assignments",
    ):
        for record in _iter_jsonl(root / relative_path, "weather tile assignment"):
            cell_id = _required_text(record.get("candidate_cell_id"), "candidate_cell_id")
            prior = mappings.get(cell_id)
            if prior is not None and (
                prior.get("weather_tile_id") != record.get("weather_tile_id")
                or set(prior.get("source_example_ids", []))
                != set(record.get("source_example_ids", []))
            ):
                raise WeatherCandidateDatasetError(
                    f"candidate cell {cell_id} has conflicting weather assignments"
                )
            mappings[cell_id] = dict(record)
    return mappings


def _load_weather(
    root: Path,
    report: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for relative_path in _relative_paths(
        report.get("measurement_artifact_relative_paths"), root, "historical_weather"
    ):
        for record in _iter_jsonl(root / relative_path, "historical weather measurement"):
            tile_id = _required_text(record.get("weather_tile_id"), "weather_tile_id")
            observed_at = _required_text(record.get("observed_at"), "observed_at")
            variable = _required_text(record.get("variable"), "weather variable")
            variables = indexed.setdefault((tile_id, observed_at), {})
            prior = variables.get(variable)
            if prior is not None and prior.get("value") != record.get("value"):
                raise WeatherCandidateDatasetError(
                    f"weather tile {tile_id} has conflicting {variable} values at {observed_at}"
                )
            variables[variable] = dict(record)
    return indexed


def _join_candidate_weather(
    candidate: Mapping[str, Any],
    *,
    mappings: Mapping[str, Mapping[str, Any]],
    weather: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    cell_id = _required_text(candidate.get("cell_id"), "candidate cell_id")
    example_id = _required_text(candidate.get("example_id"), "candidate example_id")
    mapping = mappings.get(cell_id)
    if mapping is None:
        raise WeatherCandidateDatasetError("no weather tile assignment")
    if example_id not in set(mapping.get("source_example_ids", [])):
        raise WeatherCandidateDatasetError("weather tile assignment is not tied to this example")
    weather_hour = floor_weather_hour(candidate.get("anchor_at"))
    values = weather.get((str(mapping.get("weather_tile_id")), _format_utc(weather_hour)))
    if values is None:
        raise WeatherCandidateDatasetError("no weather values at or before the anchor hour")
    required = required_model_weather_variables()
    missing = [variable for variable in required if variable not in values]
    if missing:
        raise WeatherCandidateDatasetError("missing weather variables: " + ", ".join(missing))
    result = dict(candidate)
    result.update(
        {
            "weather_candidate_dataset_build_version": WEATHER_CANDIDATE_DATASET_BUILD_VERSION,
            "weather_available": 1,
            "weather_missing_indicator": 0,
            "weather_feature_status": WEATHER_FEATURE_STATUS,
            "weather_input_policy": WEATHER_INPUT_POLICY,
            "weather_feature_mode": OPEN_METEO_HISTORICAL_FEATURE_MODE,
            "weather_provider": values[required[0]]["provider"],
            "weather_model": values[required[0]]["model"],
            "weather_observed_at": _format_utc(weather_hour),
            "weather_tile_id": mapping["weather_tile_id"],
            "weather_source_grid_id": mapping["source_grid_id"],
            "weather_tile_distance_m": mapping["weather_tile_distance_m"],
            "weather_temperature_2m": _finite_value(values["temperature_2m"]),
            "weather_relative_humidity_2m": _finite_value(values["relative_humidity_2m"]),
            "weather_precipitation": _finite_value(values["precipitation"]),
            "weather_wind_u_10m": _finite_value(values["wind_u_10m"]),
            "weather_wind_v_10m": _finite_value(values["wind_v_10m"]),
            "weather_raw_artifact_ids": sorted(
                {
                    _required_text(values[variable].get("raw_artifact_id"), "weather raw_artifact_id")
                    for variable in required
                }
            ),
            "weather_mapping_raw_artifact_id": _required_text(
                mapping.get("raw_artifact_id"), "weather mapping raw_artifact_id"
            ),
        }
    )
    return result


def _write_completed_manifest(
    root: Path,
    *,
    candidate_manifest: BaseCandidateManifestIdentity,
    weather_backfill_manifest: Path,
    backfill: Mapping[str, Any],
    artifacts: Sequence[NormalizedArtifact],
    candidate_row_count: int,
) -> Path:
    now = datetime.now(timezone.utc)
    build_id = uuid.uuid4().hex
    document = {
        "schema_version": WEATHER_CANDIDATE_DATASET_MANIFEST_SCHEMA_VERSION,
        "kind": "completed-weather-candidate-dataset-build",
        "status": "complete",
        "build_id": build_id,
        "completed_at": _format_utc(now),
        "weather_candidate_dataset_build_version": WEATHER_CANDIDATE_DATASET_BUILD_VERSION,
        "candidate_manifest": {
            "relative_path": candidate_manifest.relative_path,
            "build_id": candidate_manifest.build_id,
            "content_sha256": candidate_manifest.content_sha256,
        },
        "weather_backfill_manifest": weather_backfill_manifest.relative_to(root).as_posix(),
        "weather": {
            "available": True,
            "status": WEATHER_FEATURE_STATUS,
            "policy": WEATHER_INPUT_POLICY,
            "provider": backfill["provider"],
            "product_kind": backfill["product_kind"],
            "model": backfill["model"],
            "feature_hour_policy": backfill["feature_hour_policy"],
            "feature_mode": backfill["feature_mode"],
        },
        "candidate_row_count": candidate_row_count,
        "model_feature_columns": list(WEATHER_MODEL_FEATURE_COLUMNS),
        "candidate_artifact_relative_paths": [
            artifact.artifact_path.relative_to(root).as_posix() for artifact in artifacts
        ],
        "normalized_artifact_ids": [artifact.normalized_artifact_id for artifact in artifacts],
    }
    destination = (
        root
        / "manifests"
        / "weather-candidate-dataset-builds"
        / now.strftime("%Y/%m/%d")
        / f"{now.strftime('%H%M%S%f')}_{build_id}.json"
    )
    return write_atomic_json(destination, document)


def _read_completed_manifest(path: Path, root: Path) -> dict[str, Any]:
    document = _read_json(path, "weather candidate dataset manifest")
    if document.get("kind") != "completed-weather-candidate-dataset-build":
        raise WeatherCandidateDatasetError("weather candidate manifest has the wrong kind")
    if document.get("status") != "complete":
        raise WeatherCandidateDatasetError("weather candidate manifest is not complete")
    if document.get("weather_candidate_dataset_build_version") != (
        WEATHER_CANDIDATE_DATASET_BUILD_VERSION
    ):
        raise WeatherCandidateDatasetError("weather candidate manifest has an unsupported version")
    paths = _relative_paths(
        document.get("candidate_artifact_relative_paths"), root, "weather_candidate_examples"
    )
    return {
        "build_id": _required_text(document.get("build_id"), "weather build_id"),
        "candidate_manifest": _candidate_identity_from_document(
            document.get("candidate_manifest"), "weather candidate manifest"
        ),
        "candidate_artifact_relative_paths": paths,
        "weather_backfill_manifest": _required_text(
            document.get("weather_backfill_manifest"), "weather_backfill_manifest"
        ),
        "weather": document.get("weather"),
    }


def _source_raw_artifact_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    identifiers: set[str] = set()
    for row in rows:
        for field in (
            "firms_raw_artifact_ids",
            "feds_snapshot_context_raw_artifact_ids",
            "weather_raw_artifact_ids",
        ):
            value = row.get(field, [])
            if isinstance(value, (list, tuple, set)):
                identifiers.update(
                    item.strip() for item in value if isinstance(item, str) and item.strip()
                )
        mapping_id = row.get("weather_mapping_raw_artifact_id")
        if isinstance(mapping_id, str) and mapping_id.strip():
            identifiers.add(mapping_id.strip())
    return identifiers


def _conservative_bytes(rows: Iterable[Mapping[str, Any]]) -> int:
    encoded = sum(
        len(json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        + 1
        for row in rows
    )
    return encoded * 2 + 65_536


def _relative_paths(value: object, root: Path, entity: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise WeatherCandidateDatasetError(f"{entity} artifact paths must be a non-empty list")
    paths: list[str] = []
    prefix = f"normalized/{entity.replace('_', '-')}/"
    for item in value:
        path = Path(_required_text(item, f"{entity} artifact path"))
        if path.is_absolute() or ".." in path.parts:
            raise WeatherCandidateDatasetError(f"{entity} artifact path must be relative")
        normalized = path.as_posix()
        if not normalized.startswith(prefix):
            raise WeatherCandidateDatasetError(f"{entity} artifact path is outside {prefix}")
        if not (root / normalized).is_file():
            raise WeatherCandidateDatasetError(f"{entity} artifact is missing: {normalized}")
        paths.append(normalized)
    if len(set(paths)) != len(paths):
        raise WeatherCandidateDatasetError(f"{entity} artifact paths must not repeat")
    return tuple(paths)


def _iter_jsonl(path: Path, context: str) -> Iterator[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise WeatherCandidateDatasetError(
                        f"{context} record {number} is not a JSON object"
                    )
                yield value
    except (OSError, json.JSONDecodeError) as exc:
        raise WeatherCandidateDatasetError(f"could not read {context}: {path}") from exc


def _write_release_jsonl(
    root: Path,
    relative_paths: Sequence[str],
    destination: Path,
) -> tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    with gzip.GzipFile(destination, mode="wb", mtime=0) as compressed:
        for relative_path in relative_paths:
            for record in _iter_jsonl(root / relative_path, "weather candidate release record"):
                line = _canonical_json_line(record)
                compressed.write(line)
                digest.update(line)
                count += 1
    return count, digest.hexdigest()


def _canonical_json_line(record: Mapping[str, Any]) -> bytes:
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


def _write_checksums(directory: Path) -> None:
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{_sha256(path)}  {path.relative_to(directory).as_posix()}\n")
    (directory / "SHA256SUMS").write_text("".join(entries), encoding="utf-8")


def _file_inventory(directory: Path, *, exclude: set[str]) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in exclude
    ]


def _release_readme(manifest: Mapping[str, Any]) -> str:
    features = "\n".join(f"- `{field}`" for field in manifest["model_feature_columns"])
    limitations = "\n".join(f"- {item}" for item in manifest["limitations"])
    return f"""# Wildfire spread weather candidate dataset

This uploadable table contains weather at each candidate tile at the UTC hour
at or before its prediction anchor. It covers {manifest['candidate_row_count']:,}
candidate rows.

## Model feature allowlist

{features}

## Limitations

{limitations}
"""


def _write_json(path: Path, value: Mapping[str, Any] | list[dict[str, object]]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = path.resolve()
    if direct.is_file() and direct.is_relative_to(root):
        return direct
    return root / path


def _candidate_manifest_identity(
    root: Path,
    value: str | Path,
) -> BaseCandidateManifestIdentity:
    path = _resolve_path(root, value).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise WeatherCandidateDatasetError(
            "candidate manifest must be an existing file below data_root"
        )
    document = _read_json(path, "candidate manifest")
    if document.get("kind") != "completed-firms-candidate-dataset-build":
        raise WeatherCandidateDatasetError("candidate manifest has the wrong kind")
    if document.get("status") != "complete":
        raise WeatherCandidateDatasetError("candidate manifest is not complete")
    return BaseCandidateManifestIdentity(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        build_id=_required_text(document.get("build_id"), "candidate manifest build_id"),
        content_sha256=_sha256(path),
    )


def _candidate_identity_from_document(
    value: object,
    label: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise WeatherCandidateDatasetError(f"{label} has no immutable candidate identity")
    return {
        "relative_path": _required_text(value.get("relative_path"), f"{label} relative_path"),
        "build_id": _required_text(value.get("build_id"), f"{label} build_id"),
        "content_sha256": _required_text(
            value.get("content_sha256"), f"{label} content_sha256"
        ),
    }


def _require_matching_candidate_identity(
    backfill: Mapping[str, Any],
    candidate_identity: BaseCandidateManifestIdentity,
) -> None:
    recorded = _candidate_identity_from_document(
        backfill.get("candidate_manifest"), "weather backfill"
    )
    expected = {
        "relative_path": candidate_identity.relative_path,
        "build_id": candidate_identity.build_id,
        "content_sha256": candidate_identity.content_sha256,
    }
    if recorded != expected:
        raise WeatherCandidateDatasetError(
            "weather backfill candidate manifest identity does not match the selected immutable view"
        )


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WeatherCandidateDatasetError(f"could not read {context}: {path}") from exc
    if not isinstance(value, dict):
        raise WeatherCandidateDatasetError(f"{context} must be a JSON object")
    return value


def _parse_utc(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise WeatherCandidateDatasetError(f"{label} must be ISO-8601") from exc
    else:
        raise WeatherCandidateDatasetError(f"{label} must be ISO-8601")
    if parsed.tzinfo is None:
        raise WeatherCandidateDatasetError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_value(record: Mapping[str, Any]) -> float:
    try:
        value = float(record.get("value"))
    except (TypeError, ValueError) as exc:
        raise WeatherCandidateDatasetError("weather value must be numeric") from exc
    if value != value or value in {float("inf"), float("-inf")}:
        raise WeatherCandidateDatasetError("weather value must be finite")
    return value


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WeatherCandidateDatasetError(f"{label} must be non-empty")
    return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
