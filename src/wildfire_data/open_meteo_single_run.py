"""Open-Meteo Single Runs capture for candidate wildfire cells.

The adapter supports two explicit modes:

* a forward capture whose availability is proven by the successful response
  time; and
* a historical backfill of an archived, named model run whose availability is
  supplied from a versioned provider schedule.

Both modes retain immutable provider bytes and candidate-cell mappings.  The
caller, rather than the current wall clock, defines the availability contract
for an archived run so a later retrieval cannot masquerade as a forecast that
was unavailable at the prediction anchor.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .forecast_collection import ForecastCollectionResult, archive_forecast_response
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .storage_budget import StorageBudgetPolicy, require_admission
from .training_grid import cell_from_id, cell_from_wgs84, cells_in_square_radius, format_utc
from .weather_rate_limit import WeatherRateLimitPause, WeatherRequestPacer, get_with_retries
from .weather_source_selection import minimum_covering_sources


OPEN_METEO_SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
)
OPEN_METEO_AVAILABILITY_BASIS = "collector-captured-single-run-response/v1"
OPEN_METEO_TILE_MAPPING_VERSION = "open-meteo-single-run-candidate-mapping/v1"
DEFAULT_CANDIDATE_RADIUS_CELLS = 2
DEFAULT_MAX_TILE_DISTANCE_METRES = 10_000.0
DEFAULT_FORECAST_HORIZON_HOURS = 12
DEFAULT_BATCH_SIZE = 50
# Kept from the prior weather notebook: batched locations count as units, so
# a 50-location request is paced as 50 units rather than as one free call.
DEFAULT_REQUESTS_PER_MINUTE = 600
DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 90
DEFAULT_MAX_CONSECUTIVE_RATE_LIMITS = 2

_TILE_COLUMNS = (
    "forecast_tile_id",
    "forecast_tile_latitude",
    "forecast_tile_longitude",
    "representative_candidate_cell_id",
)
_ASSIGNMENT_COLUMNS = (
    "candidate_cell_id",
    "candidate_cell_latitude",
    "candidate_cell_longitude",
    "source_firms_detection_ids",
    "source_firms_detection_count",
    "latest_source_firms_acquired_at",
    "source_firms_raw_artifact_ids",
    "source_example_ids",
    "source_example_count",
    "forecast_tile_id",
    "forecast_tile_latitude",
    "forecast_tile_longitude",
    "forecast_tile_distance_m",
)
_FIELD_METADATA = {
    "temperature_2m": ("2m", "°C"),
    "relative_humidity_2m": ("2m", "%"),
    "precipitation": ("surface", "mm"),
    "wind_speed_10m": ("10m", "km/h"),
    "wind_direction_10m": ("10m", "°"),
}


@dataclass(frozen=True)
class OpenMeteoForecastTilePlan:
    """A spatial cover of model candidate cells with retained source lineage."""

    tiles: pd.DataFrame
    assignments: pd.DataFrame
    input_kind: str = "firms-detection-grid"


@dataclass(frozen=True)
class OpenMeteoSingleRunCaptureResult:
    """Immutable weather and mapping artifacts written by one capture attempt."""

    collection_results: tuple[ForecastCollectionResult, ...]
    assignment_artifacts: tuple[NormalizedArtifact, ...]
    planned_tile_count: int
    captured_tile_count: int
    candidate_cell_count: int
    measurement_count: int
    http_attempts: int
    api_call_units: int
    rate_limit_retries: int
    paused_for_rate_limit: bool


class OpenMeteoSingleRunError(RuntimeError):
    """Raised after a provider response was retained but could not be parsed."""


def plan_firms_candidate_weather_tiles(
    fires: pd.DataFrame,
    *,
    candidate_radius_cells: int = DEFAULT_CANDIDATE_RADIUS_CELLS,
    max_tile_distance_m: float = DEFAULT_MAX_TILE_DISTANCE_METRES,
) -> OpenMeteoForecastTilePlan:
    """Map FIRMS-seeded candidate cells to a compact weather-location cover.

    The selection starts from every canonical 1 km candidate cell, rather than
    just the FIRMS point.  This is important because the training set includes
    nearby target=0 weak-negative proxies that need a weather mapping too.
    """
    if not isinstance(fires, pd.DataFrame):
        raise TypeError("fires must be a pandas DataFrame")
    if not isinstance(candidate_radius_cells, int) or candidate_radius_cells < 0:
        raise ValueError("candidate_radius_cells must be a non-negative integer")
    if not math.isfinite(max_tile_distance_m) or max_tile_distance_m <= 0:
        raise ValueError("max_tile_distance_m must be positive and finite")
    required_columns = {"latitude", "longitude"}
    missing_columns = required_columns.difference(fires.columns)
    if missing_columns:
        raise ValueError(f"fires is missing required columns: {sorted(missing_columns)}")
    if fires.empty:
        return _empty_plan()

    location_columns = ["latitude", "longitude"]
    if "detection_id" in fires.columns:
        location_columns.append("detection_id")
    if "acquired_at" in fires.columns:
        location_columns.append("acquired_at")
    if "raw_artifact_id" in fires.columns:
        location_columns.append("raw_artifact_id")
    locations = fires.loc[:, location_columns].copy()
    locations["latitude"] = pd.to_numeric(locations["latitude"], errors="coerce")
    locations["longitude"] = pd.to_numeric(locations["longitude"], errors="coerce")
    valid = (
        np.isfinite(locations["latitude"])
        & np.isfinite(locations["longitude"])
        & locations["latitude"].between(-90, 90)
        & locations["longitude"].between(-180, 180)
    )
    if not valid.all():
        invalid_count = int((~valid).sum())
        raise ValueError(f"fires contains {invalid_count:,} invalid latitude/longitude rows")
    if "acquired_at" in locations:
        raw_acquired_at = locations["acquired_at"]
        locations["acquired_at"] = pd.to_datetime(raw_acquired_at, utc=True, errors="coerce")
        invalid_acquired_count = int(
            (raw_acquired_at.notna() & locations["acquired_at"].isna()).sum()
        )
        if invalid_acquired_count:
            raise ValueError(
                f"fires contains {invalid_acquired_count:,} invalid acquired_at timestamps"
            )
    else:
        locations["acquired_at"] = pd.NaT
    if "detection_id" not in locations:
        locations["detection_id"] = None
    if "raw_artifact_id" not in locations:
        locations["raw_artifact_id"] = None
    locations = locations.sort_values(["longitude", "latitude"], kind="stable")

    candidate_cells: dict[str, dict[str, object]] = {}
    for location in locations.itertuples(index=False):
        seed = cell_from_wgs84(
            latitude=float(location.latitude), longitude=float(location.longitude)
        )
        for candidate in cells_in_square_radius(seed, radius_cells=candidate_radius_cells):
            metadata = candidate_cells.setdefault(
                candidate.cell_id,
                {
                    "cell": candidate,
                    "detection_ids": set(),
                    "detection_count": 0,
                    "latest_acquired_at": None,
                    "raw_artifact_ids": set(),
                },
            )
            metadata["detection_count"] = int(metadata["detection_count"]) + 1
            detection_id = location.detection_id
            if detection_id is not None and not pd.isna(detection_id) and str(detection_id).strip():
                metadata["detection_ids"].add(str(detection_id).strip())
            raw_artifact_id = location.raw_artifact_id
            if (
                raw_artifact_id is not None
                and not pd.isna(raw_artifact_id)
                and str(raw_artifact_id).strip()
            ):
                metadata["raw_artifact_ids"].add(str(raw_artifact_id).strip())
            acquired_at = location.acquired_at
            if not pd.isna(acquired_at):
                latest = metadata["latest_acquired_at"]
                if latest is None or acquired_at > latest:
                    metadata["latest_acquired_at"] = acquired_at
    candidate_rows = []
    for candidate_cell_id, metadata in sorted(candidate_cells.items()):
        candidate = metadata["cell"]
        latitude, longitude = candidate.center_wgs84
        latest_acquired_at = metadata["latest_acquired_at"]
        candidate_rows.append(
            {
                "candidate_cell_id": candidate_cell_id,
                "candidate_cell_latitude": latitude,
                "candidate_cell_longitude": longitude,
                "source_firms_detection_ids": sorted(metadata["detection_ids"]),
                "source_firms_detection_count": int(metadata["detection_count"]),
                "latest_source_firms_acquired_at": (
                    latest_acquired_at.isoformat().replace("+00:00", "Z")
                    if latest_acquired_at is not None
                    else None
                ),
                "source_firms_raw_artifact_ids": sorted(metadata["raw_artifact_ids"]),
                "source_example_ids": [],
                "source_example_count": 0,
            }
        )
    return _plan_candidate_rows(candidate_rows, input_kind="firms-detection-grid", max_tile_distance_m=max_tile_distance_m)


def plan_candidate_example_weather_tiles(
    examples: pd.DataFrame,
    *,
    max_tile_distance_m: float = DEFAULT_MAX_TILE_DISTANCE_METRES,
) -> OpenMeteoForecastTilePlan:
    """Plan weather locations for already-selected candidate training cells.

    This is the historical-backfill counterpart to
    :func:`plan_firms_candidate_weather_tiles`.  Its inputs are candidate
    rows—not a later weather observation—so tile selection remains a purely
    spatial compression of the exact examples that need forecasts.  Every
    candidate must retain at least one FIRMS raw-artifact id, and the mapping
    keeps the example IDs needed to audit the eventual join.
    """
    if not isinstance(examples, pd.DataFrame):
        raise TypeError("examples must be a pandas DataFrame")
    if not math.isfinite(max_tile_distance_m) or max_tile_distance_m <= 0:
        raise ValueError("max_tile_distance_m must be positive and finite")
    required_columns = {"cell_id", "example_id", "firms_raw_artifact_ids"}
    missing_columns = required_columns.difference(examples.columns)
    if missing_columns:
        raise ValueError(f"examples is missing required columns: {sorted(missing_columns)}")
    if examples.empty:
        return _empty_plan(input_kind="candidate-example-grid")

    candidate_cells: dict[str, dict[str, object]] = {}
    for example in examples.loc[:, list(required_columns)].itertuples(index=False):
        cell_id = _required_text(example.cell_id, "candidate cell_id")
        example_id = _required_text(example.example_id, "candidate example_id")
        try:
            cell = cell_from_id(cell_id)
        except Exception as exc:
            raise ValueError(f"candidate cell_id is invalid: {cell_id}") from exc
        raw_artifact_ids = _artifact_id_list(
            example.firms_raw_artifact_ids,
            "candidate firms_raw_artifact_ids",
        )
        if not raw_artifact_ids:
            raise ValueError(
                f"candidate example {example_id} has no FIRMS raw-artifact lineage"
            )
        metadata = candidate_cells.setdefault(
            cell.cell_id,
            {
                "cell": cell,
                "example_ids": set(),
                "raw_artifact_ids": set(),
            },
        )
        metadata["example_ids"].add(example_id)
        metadata["raw_artifact_ids"].update(raw_artifact_ids)

    candidate_rows = []
    for candidate_cell_id, metadata in sorted(candidate_cells.items()):
        latitude, longitude = metadata["cell"].center_wgs84
        example_ids = sorted(metadata["example_ids"])
        candidate_rows.append(
            {
                "candidate_cell_id": candidate_cell_id,
                "candidate_cell_latitude": latitude,
                "candidate_cell_longitude": longitude,
                "source_firms_detection_ids": [],
                "source_firms_detection_count": 0,
                "latest_source_firms_acquired_at": None,
                "source_firms_raw_artifact_ids": sorted(metadata["raw_artifact_ids"]),
                "source_example_ids": example_ids,
                "source_example_count": len(example_ids),
            }
        )
    return _plan_candidate_rows(
        candidate_rows,
        input_kind="candidate-example-grid",
        max_tile_distance_m=max_tile_distance_m,
    )


def _plan_candidate_rows(
    candidate_rows: list[dict[str, object]],
    *,
    input_kind: str,
    max_tile_distance_m: float,
) -> OpenMeteoForecastTilePlan:
    if not candidate_rows:
        return _empty_plan(input_kind=input_kind)
    candidates = pd.DataFrame(candidate_rows)
    cover = minimum_covering_sources(
        candidates[["candidate_cell_longitude", "candidate_cell_latitude"]].to_numpy(),
        radius_m=max_tile_distance_m,
    )
    candidates["_candidate_index"] = candidates.index
    source_indices = [int(index) for index in cover.source_indices]
    sources = candidates.iloc[source_indices].copy().reset_index(drop=True)
    sources["forecast_tile_id"] = [
        f"open-meteo-{candidate_cell_id}" for candidate_cell_id in sources["candidate_cell_id"]
    ]
    tiles = sources.rename(
        columns={
            "candidate_cell_id": "representative_candidate_cell_id",
            "candidate_cell_latitude": "forecast_tile_latitude",
            "candidate_cell_longitude": "forecast_tile_longitude",
        }
    )
    tiles = tiles.loc[:, list(_TILE_COLUMNS)].sort_values("forecast_tile_id").reset_index(drop=True)

    source_by_candidate_index = sources.set_index("_candidate_index")
    assignments = candidates.drop(columns="_candidate_index").copy()
    assigned = source_by_candidate_index.loc[cover.assigned_source_indices]
    assignments["forecast_tile_id"] = assigned["forecast_tile_id"].to_numpy()
    assignments["forecast_tile_latitude"] = assigned["candidate_cell_latitude"].to_numpy()
    assignments["forecast_tile_longitude"] = assigned["candidate_cell_longitude"].to_numpy()
    assignments["forecast_tile_distance_m"] = cover.distances_m.astype(float)
    assignments = (
        assignments.loc[:, list(_ASSIGNMENT_COLUMNS)]
        .sort_values("candidate_cell_id")
        .reset_index(drop=True)
    )
    return OpenMeteoForecastTilePlan(
        tiles=tiles,
        assignments=assignments,
        input_kind=input_kind,
    )


def capture_open_meteo_single_run(
    data_root: str | Path,
    plan: OpenMeteoForecastTilePlan,
    *,
    model: str,
    model_run_at: object,
    forecast_horizon_hours: int = DEFAULT_FORECAST_HORIZON_HOURS,
    availability_at: object | None = None,
    availability_basis: str | None = None,
    valid_end_at: object | None = None,
    storage_policy: StorageBudgetPolicy,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    rate_limit_cooldown_seconds: int = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    max_consecutive_rate_limits: int = DEFAULT_MAX_CONSECUTIVE_RATE_LIMITS,
    session: requests.Session | None = None,
    pacer: WeatherRequestPacer | None = None,
    retrieved_at: datetime | None = None,
) -> OpenMeteoSingleRunCaptureResult:
    """Capture one named Open-Meteo forecast run with explicit availability.

    Each successful response is archived before the next batch begins.  A
    two-429 pause therefore leaves complete immutable batches available for a
    later, separately auditable capture attempt.  When ``availability_at`` is
    omitted, the successful response time is the availability proof used for a
    forward capture.  An archived historical run must provide its separately
    documented availability time and basis; today’s retrieval time remains
    provenance only.
    """
    if not isinstance(plan, OpenMeteoForecastTilePlan):
        raise TypeError("plan must be an OpenMeteoForecastTilePlan")
    model_name = _required_text(model, "model")
    run_at = _parse_utc(model_run_at, "model_run_at")
    requested_capture_at = _utc_now_or_value(retrieved_at)
    if run_at > requested_capture_at:
        raise ValueError("model_run_at must not be after the capture time")
    if not isinstance(forecast_horizon_hours, int) or forecast_horizon_hours <= 0:
        raise ValueError("forecast_horizon_hours must be a positive integer")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if availability_at is None:
        resolved_availability_at = requested_capture_at
        resolved_availability_basis = OPEN_METEO_AVAILABILITY_BASIS
    else:
        resolved_availability_at = _parse_utc(availability_at, "availability_at")
        resolved_availability_basis = _required_text(
            availability_basis,
            "availability_basis",
        )
    if resolved_availability_at < run_at:
        raise ValueError("availability_at must not precede model_run_at")
    if resolved_availability_at > requested_capture_at:
        raise ValueError("availability_at must not be after the response capture time")
    resolved_valid_end_at = (
        _parse_utc(valid_end_at, "valid_end_at")
        if valid_end_at is not None
        else resolved_availability_at + timedelta(hours=forecast_horizon_hours)
    )
    if resolved_valid_end_at <= resolved_availability_at:
        raise ValueError("valid_end_at must be after availability_at")
    requested_forecast_hours = _forecast_hours_through(run_at, resolved_valid_end_at)
    _validate_plan(plan)
    missing_firms_lineage = plan.assignments["source_firms_raw_artifact_ids"].map(
        lambda identifiers: not isinstance(identifiers, (list, tuple)) or not identifiers
    )
    if bool(missing_firms_lineage.any()):
        raise ValueError(
            "every capture candidate must include at least one source FIRMS raw_artifact_id"
        )

    tiles = plan.tiles.sort_values("forecast_tile_id").reset_index(drop=True)
    if tiles.empty:
        return OpenMeteoSingleRunCaptureResult(
            collection_results=(),
            assignment_artifacts=(),
            planned_tile_count=0,
            captured_tile_count=0,
            candidate_cell_count=0,
            measurement_count=0,
            http_attempts=0,
            api_call_units=0,
            rate_limit_retries=0,
            paused_for_rate_limit=False,
        )

    root = Path(data_root)
    active_pacer = pacer or WeatherRequestPacer(requests_per_minute)
    initial_request_count = active_pacer.request_count
    initial_api_call_units = active_pacer.api_call_units
    initial_rate_limit_count = active_pacer.rate_limit_count
    owns_session = session is None
    active_session = session or requests.Session()
    collection_results = []
    assignment_artifacts = []
    captured_tile_count = 0
    measurement_count = 0
    paused_for_rate_limit = False
    run_text = _format_model_run(run_at)
    try:
        for batch in _batches(tiles, batch_size):
            batch_assignments = plan.assignments[
                plan.assignments["forecast_tile_id"].isin(batch["forecast_tile_id"])
            ]
            require_admission(
                storage_policy,
                root,
                category="issued_weather_tiles",
                requested_bytes=_conservative_batch_bytes(
                    tile_count=len(batch),
                    candidate_cell_count=len(batch_assignments),
                    forecast_horizon_hours=requested_forecast_hours,
                ),
            )
            parameters = _request_parameters(
                batch,
                model=model_name,
                run_text=run_text,
                forecast_hours=requested_forecast_hours,
            )
            batch_identity = _batch_identity(model_name, run_at, batch)
            try:
                response = get_with_retries(
                    active_session,
                    url=OPEN_METEO_SINGLE_RUNS_URL,
                    params=parameters,
                    pacer=active_pacer,
                    timeout=timeout,
                    max_attempts=max_attempts,
                    rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
                    api_call_units=len(batch),
                    max_consecutive_rate_limits=max_consecutive_rate_limits,
                )
            except WeatherRateLimitPause as pause:
                if pause.response is not None:
                    response_captured_at = _utc_now_or_value(retrieved_at)
                    collection_results.append(
                        _archive_open_meteo_response(
                            root,
                            response=pause.response,
                            measurements=(),
                            model=model_name,
                            model_run_at=run_at,
                            response_captured_at=response_captured_at,
                            availability_at=resolved_availability_at,
                            availability_basis=resolved_availability_basis,
                            valid_end_at=resolved_valid_end_at,
                            parameters=parameters,
                            batch_identity=batch_identity,
                        )
                    )
                paused_for_rate_limit = True
                break
            except requests.HTTPError as exc:
                if exc.response is None:
                    raise
                response_captured_at = _utc_now_or_value(retrieved_at)
                failed_collection = _archive_open_meteo_response(
                    root,
                    response=exc.response,
                    measurements=(),
                    model=model_name,
                    model_run_at=run_at,
                    response_captured_at=response_captured_at,
                    availability_at=resolved_availability_at,
                    availability_basis=resolved_availability_basis,
                    valid_end_at=resolved_valid_end_at,
                    parameters=parameters,
                    batch_identity=batch_identity,
                )
                collection_results.append(failed_collection)
                raise OpenMeteoSingleRunError(
                    "Open-Meteo returned an unsuccessful response; raw evidence was retained at "
                    f"{failed_collection.raw_artifact.artifact_path}"
                ) from exc

            response_captured_at = _utc_now_or_value(retrieved_at)
            try:
                response_payloads = _response_payloads(response, expected_count=len(batch))
                measurements = _measurements_from_response_payloads(
                    response_payloads,
                    batch,
                    model=model_name,
                    model_run_at=run_at,
                    available_at=resolved_availability_at,
                    valid_end_at=resolved_valid_end_at,
                )
            except Exception as exc:
                failed_collection = _archive_open_meteo_response(
                    root,
                    response=response,
                    measurements=(),
                    model=model_name,
                    model_run_at=run_at,
                    response_captured_at=response_captured_at,
                    availability_at=resolved_availability_at,
                    availability_basis=resolved_availability_basis,
                    valid_end_at=resolved_valid_end_at,
                    parameters=parameters,
                    batch_identity=batch_identity,
                    parser_error=f"{type(exc).__name__}: {exc}",
                )
                collection_results.append(failed_collection)
                raise OpenMeteoSingleRunError(
                    "Could not parse Open-Meteo response; raw evidence was retained at "
                    f"{failed_collection.raw_artifact.artifact_path}"
                ) from exc
            collection = _archive_open_meteo_response(
                root,
                response=response,
                measurements=measurements,
                model=model_name,
                model_run_at=run_at,
                response_captured_at=response_captured_at,
                availability_at=resolved_availability_at,
                availability_basis=resolved_availability_basis,
                valid_end_at=resolved_valid_end_at,
                parameters=parameters,
                batch_identity=batch_identity,
            )
            collection_results.append(collection)
            assignment_artifacts.append(
                _write_assignment_artifact(
                    root,
                    assignments=batch_assignments,
                    input_kind=plan.input_kind,
                    model=model_name,
                    model_run_at=run_at,
                    availability_at=resolved_availability_at,
                    availability_basis=resolved_availability_basis,
                    response_captured_at=response_captured_at,
                    raw_artifact_id=collection.raw_artifact.raw_artifact_id,
                )
            )
            captured_tile_count += len(batch)
            measurement_count += collection.measurement_count
    finally:
        if owns_session:
            active_session.close()

    return OpenMeteoSingleRunCaptureResult(
        collection_results=tuple(collection_results),
        assignment_artifacts=tuple(assignment_artifacts),
        planned_tile_count=len(tiles),
        captured_tile_count=captured_tile_count,
        candidate_cell_count=len(plan.assignments),
        measurement_count=measurement_count,
        http_attempts=active_pacer.request_count - initial_request_count,
        api_call_units=active_pacer.api_call_units - initial_api_call_units,
        rate_limit_retries=active_pacer.rate_limit_count - initial_rate_limit_count,
        paused_for_rate_limit=paused_for_rate_limit,
    )


def _archive_open_meteo_response(
    root: Path,
    *,
    response: Any,
    measurements: list[dict[str, object]] | tuple[Mapping[str, object], ...],
    model: str,
    model_run_at: datetime,
    response_captured_at: datetime,
    availability_at: datetime,
    availability_basis: str,
    valid_end_at: datetime,
    parameters: Mapping[str, object],
    batch_identity: str,
    parser_error: str | None = None,
) -> ForecastCollectionResult:
    return archive_forecast_response(
        str(root),
        payload=bytes(response.content),
        measurements=measurements,
        provider="Open-Meteo",
        product="Single Runs",
        model=model,
        model_run_at=format_utc(model_run_at),
        source_uri=OPEN_METEO_SINGLE_RUNS_URL,
        coverage_start=availability_at,
        coverage_end=valid_end_at,
        region="United States and Canada",
        response_status_code=response.status_code,
        response_headers=dict(response.headers),
        request_parameters=parameters,
        availability_at=availability_at,
        availability_basis=availability_basis,
        coverage_identity=batch_identity,
        parser_error=parser_error,
        retrieved_at=response_captured_at,
    )


def _empty_plan(*, input_kind: str = "firms-detection-grid") -> OpenMeteoForecastTilePlan:
    return OpenMeteoForecastTilePlan(
        tiles=pd.DataFrame(columns=_TILE_COLUMNS),
        assignments=pd.DataFrame(columns=_ASSIGNMENT_COLUMNS),
        input_kind=input_kind,
    )


def _validate_plan(plan: OpenMeteoForecastTilePlan) -> None:
    if not isinstance(plan.tiles, pd.DataFrame) or not isinstance(plan.assignments, pd.DataFrame):
        raise TypeError("plan tiles and assignments must be pandas DataFrames")
    _required_text(plan.input_kind, "plan input_kind")
    missing_tile_columns = set(_TILE_COLUMNS).difference(plan.tiles.columns)
    missing_assignment_columns = set(_ASSIGNMENT_COLUMNS).difference(plan.assignments.columns)
    if missing_tile_columns or missing_assignment_columns:
        raise ValueError(
            "plan is missing required columns: "
            f"tiles={sorted(missing_tile_columns)}, "
            f"assignments={sorted(missing_assignment_columns)}"
        )
    tile_ids = set(plan.tiles["forecast_tile_id"])
    assignment_tile_ids = set(plan.assignments["forecast_tile_id"])
    if not assignment_tile_ids.issubset(tile_ids):
        raise ValueError("plan assignments reference unknown forecast tiles")
    if tile_ids != assignment_tile_ids:
        raise ValueError("every forecast tile must be referenced by a candidate-cell assignment")
    if len(plan.tiles) and plan.assignments.empty:
        raise ValueError("a non-empty tile plan must have candidate-cell assignments")


def _batches(frame: pd.DataFrame, size: int):
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size].copy()


def _request_parameters(
    batch: pd.DataFrame,
    *,
    model: str,
    run_text: str,
    forecast_hours: int,
) -> dict[str, object]:
    return {
        "latitude": ",".join(_coordinate_text(value) for value in batch["forecast_tile_latitude"]),
        "longitude": ",".join(
            _coordinate_text(value) for value in batch["forecast_tile_longitude"]
        ),
        "hourly": ",".join(OPEN_METEO_HOURLY_VARIABLES),
        "models": model,
        "run": run_text,
        # ``forecast_hours`` is measured from the named model run, not from
        # today’s response time.  The caller has already extended it through
        # the latest required valid hour.
        "forecast_hours": forecast_hours,
        "timezone": "UTC",
        "timeformat": "iso8601",
        "elevation": "nan",
        "cell_selection": "nearest",
    }


def _response_payloads(response: Any, *, expected_count: int) -> list[Mapping[str, Any]]:
    try:
        payloads = response.json()
    except (TypeError, ValueError, requests.RequestException) as exc:
        raise ValueError("Open-Meteo response was not valid JSON") from exc
    if isinstance(payloads, Mapping):
        payloads = [payloads]
    if not isinstance(payloads, list) or len(payloads) != expected_count:
        count = len(payloads) if isinstance(payloads, list) else "an invalid number of"
        raise ValueError(
            f"Open-Meteo returned {count} location payloads for {expected_count} tiles"
        )
    if not all(isinstance(payload, Mapping) for payload in payloads):
        raise ValueError("Open-Meteo response includes a non-object location payload")
    return list(payloads)


def _measurements_from_response_payloads(
    payloads: list[Mapping[str, Any]],
    batch: pd.DataFrame,
    *,
    model: str,
    model_run_at: datetime,
    available_at: datetime,
    valid_end_at: datetime,
) -> list[dict[str, object]]:
    measurements = []
    for payload, tile in zip(payloads, batch.itertuples(index=False)):
        latitude = _response_coordinate(
            payload.get("latitude"), tile.forecast_tile_latitude, "latitude"
        )
        longitude = _response_coordinate(
            payload.get("longitude"), tile.forecast_tile_longitude, "longitude"
        )
        source_grid_id = ":".join(
            (
                "open-meteo-single-runs",
                model,
                _format_model_run(model_run_at),
                f"latitude={latitude:.5f}",
                f"longitude={longitude:.5f}",
            )
        )
        hourly = payload.get("hourly")
        if not isinstance(hourly, Mapping):
            raise ValueError("Open-Meteo payload is missing an hourly object")
        times = hourly.get("time")
        if not isinstance(times, list):
            raise ValueError("Open-Meteo hourly payload is missing time values")
        units = payload.get("hourly_units")
        if not isinstance(units, Mapping):
            units = {}
        for index, raw_time in enumerate(times):
            valid_at = _parse_open_meteo_utc(raw_time)
            if valid_at <= available_at or valid_at > valid_end_at:
                continue
            point_values: dict[str, float] = {}
            for field in OPEN_METEO_HOURLY_VARIABLES:
                values = hourly.get(field)
                if not isinstance(values, list) or index >= len(values):
                    continue
                value = _optional_finite_float(values[index])
                if value is None:
                    continue
                point_values[field] = value
                level, default_unit = _FIELD_METADATA[field]
                unit = _unit_text(units.get(field), default_unit)
                measurements.append(
                    {
                        "valid_at": format_utc(valid_at),
                        "source_grid_id": source_grid_id,
                        "latitude_wgs84": latitude,
                        "longitude_wgs84": longitude,
                        "variable": field,
                        "level": level,
                        "member": "deterministic",
                        "value": value,
                        "unit": unit,
                        "forecast_tile_id": tile.forecast_tile_id,
                        "open_meteo_hourly_field": field,
                    }
                )
            wind_speed = point_values.get("wind_speed_10m")
            wind_direction = point_values.get("wind_direction_10m")
            if wind_speed is not None and wind_direction is not None:
                speed_unit = _unit_text(units.get("wind_speed_10m"), "km/h")
                speed_m_per_s = _wind_speed_metres_per_second(wind_speed, speed_unit)
                direction_radians = math.radians(wind_direction)
                for variable, value in (
                    ("wind_u_10m", -speed_m_per_s * math.sin(direction_radians)),
                    ("wind_v_10m", -speed_m_per_s * math.cos(direction_radians)),
                ):
                    measurements.append(
                        {
                            "valid_at": format_utc(valid_at),
                            "source_grid_id": source_grid_id,
                            "latitude_wgs84": latitude,
                            "longitude_wgs84": longitude,
                            "variable": variable,
                            "level": "10m",
                            "member": "deterministic",
                            "value": value,
                            "unit": "m/s",
                            "forecast_tile_id": tile.forecast_tile_id,
                            "wind_direction_convention": "meteorological-from-degrees/v1",
                        }
                    )
    return measurements


def _write_assignment_artifact(
    root: Path,
    *,
    assignments: pd.DataFrame,
    input_kind: str,
    model: str,
    model_run_at: datetime,
    availability_at: datetime,
    availability_basis: str,
    response_captured_at: datetime,
    raw_artifact_id: str,
) -> NormalizedArtifact:
    records = []
    for assignment in assignments.itertuples(index=False):
        records.append(
            {
                "schema_version": 1,
                "candidate_cell_id": str(assignment.candidate_cell_id),
                "candidate_cell_latitude": float(assignment.candidate_cell_latitude),
                "candidate_cell_longitude": float(assignment.candidate_cell_longitude),
                "source_firms_detection_ids": list(assignment.source_firms_detection_ids),
                "source_firms_detection_count": int(assignment.source_firms_detection_count),
                "latest_source_firms_acquired_at": (
                    str(assignment.latest_source_firms_acquired_at)
                    if pd.notna(assignment.latest_source_firms_acquired_at)
                    else None
                ),
                "source_firms_raw_artifact_ids": list(assignment.source_firms_raw_artifact_ids),
                "source_example_ids": list(assignment.source_example_ids),
                "source_example_count": int(assignment.source_example_count),
                "forecast_tile_id": str(assignment.forecast_tile_id),
                "forecast_tile_latitude": float(assignment.forecast_tile_latitude),
                "forecast_tile_longitude": float(assignment.forecast_tile_longitude),
                "forecast_tile_distance_m": float(assignment.forecast_tile_distance_m),
                "provider": "Open-Meteo",
                "product_kind": "single-run-forecast-tile-assignment",
                "plan_input_kind": input_kind,
                "model": model,
                "model_run_at": format_utc(model_run_at),
                "availability_at": format_utc(availability_at),
                "availability_basis": availability_basis,
                "retrieved_at": format_utc(response_captured_at),
                "raw_artifact_id": raw_artifact_id,
            }
        )
    source_artifact_ids = {raw_artifact_id}
    for assignment in assignments.itertuples(index=False):
        source_artifact_ids.update(str(value) for value in assignment.source_firms_raw_artifact_ids)
    return write_normalized_jsonl(
        root,
        entity="open_meteo_forecast_tile_assignments",
        records=records,
        partitions={
            "capture_date": response_captured_at.date().isoformat(),
            "model_run_date": model_run_at.date().isoformat(),
        },
        raw_artifact_ids=sorted(source_artifact_ids),
        transformation_version=OPEN_METEO_TILE_MAPPING_VERSION,
        generated_at=response_captured_at,
    )


def _conservative_batch_bytes(
    *, tile_count: int, candidate_cell_count: int, forecast_horizon_hours: int
) -> int:
    # Reserve raw JSON, long-form weather rows, assignments, manifests, and a
    # wide safety margin before any provider response is persisted.
    expected_records = tile_count * (forecast_horizon_hours + 1) * (
        len(OPEN_METEO_HOURLY_VARIABLES) + 2
    )
    return max(262_144, expected_records * 1_024 + candidate_cell_count * 1_024 + 262_144)


def _forecast_hours_through(model_run_at: datetime, valid_end_at: datetime) -> int:
    """Return a run-relative hourly request large enough to include ``valid_end_at``."""
    seconds = (valid_end_at - model_run_at).total_seconds()
    if seconds <= 0:
        raise ValueError("valid_end_at must be after model_run_at")
    # Open-Meteo returns a clock-aligned series beginning at the run boundary.
    # Keep one extra hour so a non-hour-aligned end never loses its final
    # eligible forecast value.
    return max(1, math.ceil(seconds / 3_600) + 1)


def _batch_identity(model: str, model_run_at: datetime, batch: pd.DataFrame) -> str:
    identifiers = "\x1f".join(sorted(str(value) for value in batch["forecast_tile_id"]))
    digest = hashlib.sha256(
        "\x1f".join((model, format_utc(model_run_at), identifiers)).encode("utf-8")
    ).hexdigest()
    return f"open-meteo-single-run-{digest}"


def _coordinate_text(value: object) -> str:
    numeric = _finite_float(value, "coordinate")
    return format(numeric, ".7f")


def _response_coordinate(value: object, fallback: object, label: str) -> float:
    try:
        return _finite_float(value, label)
    except ValueError:
        return _finite_float(fallback, label)


def _optional_finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return _finite_float(value, "weather value")
    except ValueError:
        return None


def _finite_float(value: object, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _unit_text(value: object, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _wind_speed_metres_per_second(value: float, unit: str) -> float:
    normalized = unit.casefold().replace(" ", "")
    if normalized in {"m/s", "ms", "metrespersecond", "meterspersecond"}:
        return value
    if normalized in {"km/h", "kmh", "kilometresperhour", "kilometersperhour"}:
        return value / 3.6
    raise ValueError(f"unsupported Open-Meteo wind speed unit: {unit!r}")


def _parse_open_meteo_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Open-Meteo hourly time must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Open-Meteo hourly time must be ISO-8601") from exc
    if parsed.tzinfo is None:
        # The request explicitly asks for UTC, so the API's zone-less response
        # is unambiguous and can be represented as UTC here.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_utc(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _utc_now_or_value(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return resolved.astimezone(timezone.utc)


def _format_model_run(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value.strip()


def _artifact_id_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{label} must be a list of raw-artifact IDs")
    identifiers = tuple(sorted({_required_text(item, label) for item in value}))
    return identifiers
