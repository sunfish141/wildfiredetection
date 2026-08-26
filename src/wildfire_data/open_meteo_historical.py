"""Rate-limited historical Open-Meteo weather collection for candidate cells.

This module deliberately answers the simple retrospective data question:
given a candidate tile and a prediction timestamp, retain the hourly weather
condition at that tile at or immediately before that timestamp.  It uses the
Open-Meteo Historical Weather API with one pinned model, batches locations,
and stores the provider response plus an immutable candidate-to-weather-tile
mapping.

It is not an issued-forecast reconstruction.  The separate Single Runs
collector remains available when a future model needs forecast-vintage
features rather than historical weather conditions.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .candidate_dataset import iter_candidate_examples
from .data_archive import (
    CoverageLedger,
    CoverageRecord,
    CoverageStatus,
    RawArtifact,
    write_atomic_json,
    write_raw_artifact,
)
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .open_meteo_single_run import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_CONSECUTIVE_RATE_LIMITS,
    DEFAULT_MAX_TILE_DISTANCE_METRES,
    DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    DEFAULT_REQUESTS_PER_MINUTE,
    DEFAULT_TIMEOUT_SECONDS,
    OpenMeteoForecastTilePlan,
    plan_candidate_example_weather_tiles,
)
from .storage_budget import StorageBudgetError, StorageBudgetPolicy, require_admission
from .training_grid import format_utc
from .weather_rate_limit import WeatherRateLimitPause, WeatherRequestPacer, get_with_retries


OPEN_METEO_HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HISTORICAL_WEATHER_MODEL = "ecmwf_ifs"
OPEN_METEO_HISTORICAL_WEATHER_PRODUCT = "Historical Weather API"
OPEN_METEO_HISTORICAL_WEATHER_KIND = "historical-weather-analysis/v1"
OPEN_METEO_HISTORICAL_FEATURE_MODE = "historical_analysis"
OPEN_METEO_HISTORICAL_WEATHER_MAPPING_VERSION = "open-meteo-historical-weather-mapping/v1"
OPEN_METEO_HISTORICAL_WEATHER_BACKFILL_VERSION = "open-meteo-historical-weather-backfill/v1"
OPEN_METEO_HISTORICAL_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
    "wind_direction_10m",
)
_FIELD_METADATA = {
    "temperature_2m": ("2m", "°C"),
    "relative_humidity_2m": ("2m", "%"),
    "precipitation": ("surface", "mm"),
    "weather_code": ("surface", "wmo code"),
    "wind_speed_10m": ("10m", "km/h"),
    "wind_direction_10m": ("10m", "°"),
}
_REQUIRED_MODEL_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_u_10m",
    "wind_v_10m",
)


class OpenMeteoHistoricalWeatherError(RuntimeError):
    """A historical weather response was retained but cannot be used."""


@dataclass(frozen=True)
class CandidateManifestIdentity:
    """The immutable base candidate view a weather backfill is allowed to use."""

    path: Path
    relative_path: str
    build_id: str
    content_sha256: str


@dataclass(frozen=True)
class HistoricalWeatherCaptureResult:
    """Immutable artifacts emitted while collecting one UTC weather date."""

    raw_artifacts: tuple[RawArtifact, ...]
    measurement_artifacts: tuple[NormalizedArtifact, ...]
    assignment_artifacts: tuple[NormalizedArtifact, ...]
    coverage_records: tuple[CoverageRecord, ...]
    planned_tile_count: int
    captured_tile_count: int
    candidate_cell_count: int
    measurement_count: int
    http_attempts: int
    api_call_units: int
    rate_limit_retries: int
    paused_for_rate_limit: bool


@dataclass(frozen=True)
class HistoricalWeatherBackfillResult:
    """One complete or intentionally paused candidate-weather backfill."""

    manifest_path: Path
    complete: bool
    weather_date_count: int
    captured_tile_count: int
    candidate_cell_count: int
    measurement_count: int
    api_call_units: int
    paused_for_rate_limit: bool


def capture_open_meteo_historical_weather(
    data_root: str | Path,
    plan: OpenMeteoForecastTilePlan,
    *,
    weather_date: date,
    storage_policy: StorageBudgetPolicy,
    model: str = OPEN_METEO_HISTORICAL_WEATHER_MODEL,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    rate_limit_cooldown_seconds: int = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    max_consecutive_rate_limits: int = DEFAULT_MAX_CONSECUTIVE_RATE_LIMITS,
    session: requests.Session | None = None,
    pacer: WeatherRequestPacer | None = None,
    retrieved_at: datetime | None = None,
    required_weather_hours_by_example: Mapping[str, object] | None = None,
) -> HistoricalWeatherCaptureResult:
    """Collect one date of hourly weather for a compact candidate tile plan.

    The weather hour later used as a feature is selected by
    :func:`floor_weather_hour`: it is never after the candidate's anchor.
    Request pacing counts every coordinate in a batch, preserving the existing
    600 location-units/minute default and deliberate two-429 pause behaviour.
    """
    if not isinstance(plan, OpenMeteoForecastTilePlan):
        raise TypeError("plan must be an OpenMeteoForecastTilePlan")
    if not isinstance(weather_date, date):
        raise TypeError("weather_date must be a datetime.date")
    model_name = _required_text(model, "model")
    if model_name != OPEN_METEO_HISTORICAL_WEATHER_MODEL:
        raise ValueError(
            "historical weather is pinned to "
            f"{OPEN_METEO_HISTORICAL_WEATHER_MODEL!r}, not {model_name!r}"
        )
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(plan.tiles, pd.DataFrame) or not isinstance(plan.assignments, pd.DataFrame):
        raise TypeError("plan tiles and assignments must be pandas DataFrames")
    required_assignment_columns = {
        "candidate_cell_id",
        "source_firms_raw_artifact_ids",
        "source_example_ids",
        "forecast_tile_id",
    }
    missing_columns = required_assignment_columns.difference(plan.assignments.columns)
    if missing_columns:
        raise ValueError(f"plan assignments are missing columns: {sorted(missing_columns)}")
    missing_lineage = plan.assignments["source_firms_raw_artifact_ids"].map(
        lambda value: not isinstance(value, (list, tuple)) or not value
    )
    if bool(missing_lineage.any()):
        raise ValueError("every weather candidate must include FIRMS raw-artifact lineage")
    required_hours = _validated_required_weather_hours(
        plan.assignments,
        required_weather_hours_by_example,
    )

    tiles = plan.tiles.sort_values("forecast_tile_id").reset_index(drop=True)
    if tiles.empty:
        return HistoricalWeatherCaptureResult(
            raw_artifacts=(),
            measurement_artifacts=(),
            assignment_artifacts=(),
            coverage_records=(),
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
    raw_artifacts: list[RawArtifact] = []
    measurement_artifacts: list[NormalizedArtifact] = []
    assignment_artifacts: list[NormalizedArtifact] = []
    coverage_records: list[CoverageRecord] = []
    captured_tile_count = 0
    measurement_count = 0
    paused_for_rate_limit = False
    try:
        for batch in _batches(tiles, batch_size):
            assignments = plan.assignments[
                plan.assignments["forecast_tile_id"].isin(batch["forecast_tile_id"])
            ]
            require_admission(
                storage_policy,
                root,
                category="issued_weather_tiles",
                requested_bytes=_conservative_batch_bytes(
                    tile_count=len(batch),
                    candidate_cell_count=len(assignments),
                ),
            )
            parameters = _request_parameters(batch, weather_date=weather_date, model=model_name)
            batch_identity = _batch_identity(model_name, weather_date, batch)
            try:
                response = get_with_retries(
                    active_session,
                    url=OPEN_METEO_HISTORICAL_WEATHER_URL,
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
                    captured = _utc_now_or_value(retrieved_at)
                    raw, coverage = _archive_failed_response(
                        root,
                        response=pause.response,
                        weather_date=weather_date,
                        model=model_name,
                        parameters=parameters,
                        batch_identity=batch_identity,
                        retrieved_at=captured,
                    )
                    raw_artifacts.append(raw)
                    coverage_records.append(coverage)
                paused_for_rate_limit = True
                break
            except requests.HTTPError as exc:
                if exc.response is None:
                    raise
                captured = _utc_now_or_value(retrieved_at)
                raw, coverage = _archive_failed_response(
                    root,
                    response=exc.response,
                    weather_date=weather_date,
                    model=model_name,
                    parameters=parameters,
                    batch_identity=batch_identity,
                    retrieved_at=captured,
                )
                raw_artifacts.append(raw)
                coverage_records.append(coverage)
                raise OpenMeteoHistoricalWeatherError(
                    "Open-Meteo Historical Weather returned an unsuccessful response; "
                    f"raw evidence was retained at {raw.artifact_path}"
                ) from exc

            captured = _utc_now_or_value(retrieved_at)
            raw = _write_raw_response(
                root,
                response=response,
                weather_date=weather_date,
                model=model_name,
                parameters=parameters,
                retrieved_at=captured,
            )
            raw_artifacts.append(raw)
            try:
                payloads = _response_payloads(response, expected_count=len(batch))
                measurements, source_locations = _measurements_from_payloads(
                    payloads,
                    batch,
                    weather_date=weather_date,
                    model=model_name,
                    raw_artifact_id=raw.raw_artifact_id,
                    retrieved_at=captured,
                )
                artifact = (
                    _write_measurements(
                        root,
                        measurements=measurements,
                        weather_date=weather_date,
                        model=model_name,
                        raw_artifact_id=raw.raw_artifact_id,
                        retrieved_at=captured,
                    )
                    if measurements
                    else None
                )
                assignment_artifact = _write_assignment_artifact(
                    root,
                    assignments=assignments,
                    source_locations=source_locations,
                    weather_date=weather_date,
                    model=model_name,
                    raw_artifact_id=raw.raw_artifact_id,
                    retrieved_at=captured,
                )
                missing_required_weather = _missing_required_weather_measurements(
                    assignments=assignments,
                    measurements=measurements,
                    required_hours_by_example=required_hours,
                )
            except Exception as exc:
                coverage = _record_coverage(
                    root,
                    weather_date=weather_date,
                    model=model_name,
                    batch_identity=batch_identity,
                    status=CoverageStatus.FAILED,
                    raw_artifact_id=raw.raw_artifact_id,
                    retrieved_at=captured,
                    error=f"{type(exc).__name__}: {exc}",
                )
                coverage_records.append(coverage)
                raise OpenMeteoHistoricalWeatherError(
                    "Could not parse Open-Meteo Historical Weather response; raw evidence was "
                    f"retained at {raw.artifact_path}"
                ) from exc

            if missing_required_weather:
                detail = "; ".join(missing_required_weather[:5])
                coverage = _record_coverage(
                    root,
                    weather_date=weather_date,
                    model=model_name,
                    batch_identity=batch_identity,
                    status=CoverageStatus.FAILED,
                    raw_artifact_id=raw.raw_artifact_id,
                    retrieved_at=captured,
                    error="required historical weather values are missing: " + detail,
                    detail={"missing_required_weather": missing_required_weather[:20]},
                )
                coverage_records.append(coverage)
                raise OpenMeteoHistoricalWeatherError(
                    "Open-Meteo Historical Weather is missing required values for "
                    f"the candidate anchor hour; raw evidence was retained at {raw.artifact_path}: "
                    f"{detail}"
                )

            coverage = _record_coverage(
                root,
                weather_date=weather_date,
                model=model_name,
                batch_identity=batch_identity,
                status=CoverageStatus.COMPLETE if measurements else CoverageStatus.EMPTY_CONFIRMED,
                raw_artifact_id=raw.raw_artifact_id,
                retrieved_at=captured,
                detail={
                    "measurement_count": len(measurements),
                    "measurement_artifact_id": (
                        artifact.normalized_artifact_id if artifact is not None else None
                    ),
                    "assignment_artifact_id": assignment_artifact.normalized_artifact_id,
                },
            )
            coverage_records.append(coverage)
            if artifact is not None:
                measurement_artifacts.append(artifact)
            assignment_artifacts.append(assignment_artifact)
            captured_tile_count += len(batch)
            measurement_count += len(measurements)
    finally:
        if owns_session:
            active_session.close()

    return HistoricalWeatherCaptureResult(
        raw_artifacts=tuple(raw_artifacts),
        measurement_artifacts=tuple(measurement_artifacts),
        assignment_artifacts=tuple(assignment_artifacts),
        coverage_records=tuple(coverage_records),
        planned_tile_count=len(tiles),
        captured_tile_count=captured_tile_count,
        candidate_cell_count=len(plan.assignments),
        measurement_count=measurement_count,
        http_attempts=active_pacer.request_count - initial_request_count,
        api_call_units=active_pacer.api_call_units - initial_api_call_units,
        rate_limit_retries=active_pacer.rate_limit_count - initial_rate_limit_count,
        paused_for_rate_limit=paused_for_rate_limit,
    )


def backfill_open_meteo_historical_weather(
    data_root: str | Path,
    *,
    storage_policy: StorageBudgetPolicy,
    candidate_manifest: str | Path,
    start_date: date | None = None,
    end_date: date | None = None,
    resume_manifest: str | Path | None = None,
    model: str = OPEN_METEO_HISTORICAL_WEATHER_MODEL,
    max_tile_distance_m: float = DEFAULT_MAX_TILE_DISTANCE_METRES,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    rate_limit_cooldown_seconds: int = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    max_consecutive_rate_limits: int = DEFAULT_MAX_CONSECUTIVE_RATE_LIMITS,
    session: requests.Session | None = None,
    retrieved_at: datetime | None = None,
) -> HistoricalWeatherBackfillResult:
    """Backfill weather at every selected candidate row's hourly anchor.

    The input is a completed no-weather candidate view.  It is not mutated:
    this produces a separately versioned weather backfill manifest that a
    weather-feature builder can require in full before publishing an upload.
    A 429 or storage pause records a partial manifest and returns normally.
    Pass that manifest as ``resume_manifest`` to reuse its completed dates and
    retry its first incomplete date without weakening the base-view lineage.
    """
    if (start_date is None) != (end_date is None):
        raise ValueError("start_date and end_date must be supplied together")
    if start_date is not None and end_date is not None and end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    root = Path(data_root).resolve()
    model_name = _required_text(model, "model")
    if model_name != OPEN_METEO_HISTORICAL_WEATHER_MODEL:
        raise ValueError(
            "historical weather is pinned to "
            f"{OPEN_METEO_HISTORICAL_WEATHER_MODEL!r}, not {model_name!r}"
        )
    candidate_identity = _candidate_manifest_identity(root, candidate_manifest)
    resolved_start = start_date
    resolved_end = end_date
    prior_reports: dict[date, dict[str, object]] = {}
    if resume_manifest is not None:
        resume_document = _read_backfill_manifest(root, resume_manifest)
        resume_start, resume_end = _backfill_date_range(resume_document)
        if resolved_start is None:
            resolved_start, resolved_end = resume_start, resume_end
        elif (resolved_start, resolved_end) != (resume_start, resume_end):
            raise ValueError(
                "resume manifest range does not match the requested weather date range"
            )
        prior_reports = _completed_resume_reports(
            root,
            document=resume_document,
            candidate_identity=candidate_identity,
            model=model_name,
            max_tile_distance_m=max_tile_distance_m,
        )

    rows_by_date: dict[date, list[dict[str, object]]] = {}
    for row in iter_candidate_examples(root, manifest_path=candidate_identity.path):
        anchor_at = _parse_utc(row.get("anchor_at"), "candidate anchor_at")
        weather_day = anchor_at.date()
        if resolved_start is not None and not resolved_start <= weather_day <= resolved_end:
            continue
        rows_by_date.setdefault(weather_day, []).append(
            {
                "cell_id": row.get("cell_id"),
                "example_id": row.get("example_id"),
                "firms_raw_artifact_ids": row.get("firms_raw_artifact_ids"),
                "anchor_at": format_utc(anchor_at),
            }
        )
    if not rows_by_date:
        raise OpenMeteoHistoricalWeatherError(
            "the selected candidate view has no examples in the requested weather date range"
        )
    if resolved_start is None:
        resolved_start, resolved_end = min(rows_by_date), max(rows_by_date)
    assert resolved_end is not None
    _validate_resume_report_dates(prior_reports, expected_dates=set(rows_by_date))

    active_pacer = WeatherRequestPacer(requests_per_minute)
    owns_session = session is None
    active_session = session or requests.Session()
    reports: list[dict[str, object]] = []
    complete = True
    paused_for_rate_limit = False
    try:
        for weather_day, rows in sorted(rows_by_date.items()):
            if weather_day in prior_reports:
                reports.append(prior_reports[weather_day])
                continue
            plan = plan_candidate_example_weather_tiles(
                pd.DataFrame(rows),
                max_tile_distance_m=max_tile_distance_m,
            )
            required_hours_by_example = {
                _required_text(row["example_id"], "candidate example_id"): row["anchor_at"]
                for row in rows
            }
            try:
                result = capture_open_meteo_historical_weather(
                    root,
                    plan,
                    weather_date=weather_day,
                    storage_policy=storage_policy,
                    model=model_name,
                    batch_size=batch_size,
                    timeout=timeout,
                    max_attempts=max_attempts,
                    rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
                    max_consecutive_rate_limits=max_consecutive_rate_limits,
                    session=active_session,
                    pacer=active_pacer,
                    retrieved_at=retrieved_at,
                    required_weather_hours_by_example=required_hours_by_example,
                )
            except (OpenMeteoHistoricalWeatherError, StorageBudgetError, requests.RequestException) as exc:
                reports.append(
                    {
                        "weather_date": weather_day.isoformat(),
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                complete = False
                break
            reports.append(_backfill_report(weather_day, result, root))
            if result.paused_for_rate_limit:
                complete = False
                paused_for_rate_limit = True
                break
    finally:
        if owns_session:
            active_session.close()

    manifest_path = _write_backfill_manifest(
        root,
        candidate_manifest=candidate_identity,
        model=model_name,
        start_date=resolved_start,
        end_date=resolved_end,
        max_tile_distance_m=max_tile_distance_m,
        requests_per_minute=requests_per_minute,
        reports=sorted(reports, key=lambda report: str(report["weather_date"])),
        complete=complete and len(reports) == len(rows_by_date),
        paused_for_rate_limit=paused_for_rate_limit,
        generated_at=_utc_now_or_value(retrieved_at),
    )
    captured_tile_count = sum(int(report.get("captured_tile_count", 0)) for report in reports)
    candidate_cell_count = sum(int(report.get("candidate_cell_count", 0)) for report in reports)
    measurement_count = sum(int(report.get("measurement_count", 0)) for report in reports)
    return HistoricalWeatherBackfillResult(
        manifest_path=manifest_path,
        complete=complete and len(reports) == len(rows_by_date),
        weather_date_count=len(reports),
        captured_tile_count=captured_tile_count,
        candidate_cell_count=candidate_cell_count,
        measurement_count=measurement_count,
        api_call_units=active_pacer.api_call_units,
        paused_for_rate_limit=paused_for_rate_limit,
    )


def floor_weather_hour(value: object) -> datetime:
    """Return the UTC hour at or before an acquisition/prediction timestamp."""
    parsed = _parse_utc(value, "anchor_at")
    return parsed.replace(minute=0, second=0, microsecond=0)


def required_model_weather_variables() -> tuple[str, ...]:
    """Return the weather variables a complete candidate feature row requires."""
    return _REQUIRED_MODEL_VARIABLES


def _validated_required_weather_hours(
    assignments: pd.DataFrame,
    supplied: Mapping[str, object] | None,
) -> dict[str, datetime]:
    if supplied is None:
        return {}
    if not isinstance(supplied, Mapping):
        raise TypeError("required_weather_hours_by_example must be a mapping")
    expected_example_ids: set[str] = set()
    for value in assignments["source_example_ids"]:
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("weather assignment source_example_ids must be a sequence")
        expected_example_ids.update(_required_text(item, "candidate example_id") for item in value)
    normalized_supplied: dict[str, object] = {}
    for identifier, value in supplied.items():
        normalized_identifier = _required_text(identifier, "candidate example_id")
        if normalized_identifier in normalized_supplied:
            raise ValueError("required weather hours repeat a candidate example_id")
        normalized_supplied[normalized_identifier] = value
    supplied_example_ids = set(normalized_supplied)
    if supplied_example_ids != expected_example_ids:
        missing = sorted(expected_example_ids.difference(supplied_example_ids))
        unexpected = sorted(supplied_example_ids.difference(expected_example_ids))
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:3]))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected[:3]))
        raise ValueError("required weather hours do not match plan examples: " + "; ".join(details))
    return {
        identifier: floor_weather_hour(normalized_supplied[identifier])
        for identifier in sorted(supplied_example_ids)
    }


def _missing_required_weather_measurements(
    *,
    assignments: pd.DataFrame,
    measurements: list[dict[str, object]],
    required_hours_by_example: Mapping[str, datetime],
) -> list[str]:
    if not required_hours_by_example:
        return []
    available = {
        (
            _required_text(record.get("weather_tile_id"), "weather_tile_id"),
            _required_text(record.get("observed_at"), "observed_at"),
            _required_text(record.get("variable"), "weather variable"),
        )
        for record in measurements
    }
    missing: list[str] = []
    for assignment in assignments.itertuples(index=False):
        tile_id = _required_text(assignment.forecast_tile_id, "weather_tile_id")
        source_example_ids = assignment.source_example_ids
        if not isinstance(source_example_ids, (list, tuple, set)):
            raise ValueError("weather assignment source_example_ids must be a sequence")
        for example_id in source_example_ids:
            resolved_example_id = _required_text(example_id, "candidate example_id")
            observed_at = format_utc(required_hours_by_example[resolved_example_id])
            for variable in _REQUIRED_MODEL_VARIABLES:
                if (tile_id, observed_at, variable) not in available:
                    missing.append(
                        f"{resolved_example_id}/{variable}@{observed_at} (tile {tile_id})"
                    )
    return missing


def _request_parameters(
    batch: pd.DataFrame,
    *,
    weather_date: date,
    model: str,
) -> dict[str, object]:
    return {
        "latitude": ",".join(_coordinate_text(value) for value in batch["forecast_tile_latitude"]),
        "longitude": ",".join(_coordinate_text(value) for value in batch["forecast_tile_longitude"]),
        "start_date": weather_date.isoformat(),
        "end_date": weather_date.isoformat(),
        "hourly": ",".join(OPEN_METEO_HISTORICAL_HOURLY_VARIABLES),
        "models": model,
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


def _measurements_from_payloads(
    payloads: list[Mapping[str, Any]],
    batch: pd.DataFrame,
    *,
    weather_date: date,
    model: str,
    raw_artifact_id: str,
    retrieved_at: datetime,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    measurements: list[dict[str, object]] = []
    source_locations: dict[str, dict[str, object]] = {}
    for payload, tile in zip(payloads, batch.itertuples(index=False)):
        latitude = _response_coordinate(
            payload.get("latitude"), tile.forecast_tile_latitude, "latitude"
        )
        longitude = _response_coordinate(
            payload.get("longitude"), tile.forecast_tile_longitude, "longitude"
        )
        source_grid_id = ":".join(
            (
                "open-meteo-historical-weather",
                model,
                f"latitude={latitude:.5f}",
                f"longitude={longitude:.5f}",
            )
        )
        source_locations[str(tile.forecast_tile_id)] = {
            "source_grid_id": source_grid_id,
            "source_grid_latitude": latitude,
            "source_grid_longitude": longitude,
        }
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
            observed_at = _parse_open_meteo_utc(raw_time)
            if observed_at.date() != weather_date:
                continue
            point_values: dict[str, float] = {}
            for field in OPEN_METEO_HISTORICAL_HOURLY_VARIABLES:
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
                    _measurement_record(
                        weather_date=weather_date,
                        observed_at=observed_at,
                        weather_tile_id=str(tile.forecast_tile_id),
                        source_grid_id=source_grid_id,
                        latitude=latitude,
                        longitude=longitude,
                        variable=field,
                        level=level,
                        value=value,
                        unit=unit,
                        model=model,
                        raw_artifact_id=raw_artifact_id,
                        retrieved_at=retrieved_at,
                    )
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
                        _measurement_record(
                            weather_date=weather_date,
                            observed_at=observed_at,
                            weather_tile_id=str(tile.forecast_tile_id),
                            source_grid_id=source_grid_id,
                            latitude=latitude,
                            longitude=longitude,
                            variable=variable,
                            level="10m",
                            value=value,
                            unit="m/s",
                            model=model,
                            raw_artifact_id=raw_artifact_id,
                            retrieved_at=retrieved_at,
                        )
                    )
    return measurements, source_locations


def _measurement_record(
    *,
    weather_date: date,
    observed_at: datetime,
    weather_tile_id: str,
    source_grid_id: str,
    latitude: float,
    longitude: float,
    variable: str,
    level: str,
    value: float,
    unit: str,
    model: str,
    raw_artifact_id: str,
    retrieved_at: datetime,
) -> dict[str, object]:
    identity = "\x1f".join(
        (
            model,
            weather_tile_id,
            format_utc(observed_at),
            variable,
            source_grid_id,
        )
    )
    return {
        "schema_version": 1,
        "weather_measurement_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "provider": "Open-Meteo",
        "feature_mode": OPEN_METEO_HISTORICAL_FEATURE_MODE,
        "product_kind": OPEN_METEO_HISTORICAL_WEATHER_KIND,
        "model": model,
        "weather_date": weather_date.isoformat(),
        "observed_at": format_utc(observed_at),
        "weather_tile_id": weather_tile_id,
        "source_grid_id": source_grid_id,
        "latitude_wgs84": latitude,
        "longitude_wgs84": longitude,
        "variable": variable,
        "level": level,
        "value": value,
        "unit": unit,
        "raw_artifact_id": raw_artifact_id,
        "retrieved_at": format_utc(retrieved_at),
        "source_uri": OPEN_METEO_HISTORICAL_WEATHER_URL,
    }


def _write_measurements(
    root: Path,
    *,
    measurements: list[dict[str, object]],
    weather_date: date,
    model: str,
    raw_artifact_id: str,
    retrieved_at: datetime,
) -> NormalizedArtifact:
    return write_normalized_jsonl(
        root,
        entity="historical_weather",
        records=measurements,
        partitions={"weather_date": weather_date.isoformat(), "model": model},
        raw_artifact_ids=[raw_artifact_id],
        transformation_version=OPEN_METEO_HISTORICAL_WEATHER_KIND,
        generated_at=retrieved_at,
    )


def _write_assignment_artifact(
    root: Path,
    *,
    assignments: pd.DataFrame,
    source_locations: Mapping[str, Mapping[str, object]],
    weather_date: date,
    model: str,
    raw_artifact_id: str,
    retrieved_at: datetime,
) -> NormalizedArtifact:
    records: list[dict[str, object]] = []
    raw_ids = {raw_artifact_id}
    for assignment in assignments.itertuples(index=False):
        tile_id = str(assignment.forecast_tile_id)
        source = source_locations.get(tile_id)
        if source is None:
            raise ValueError(f"missing source location for weather tile {tile_id}")
        firms_raw_ids = _artifact_ids(assignment.source_firms_raw_artifact_ids)
        raw_ids.update(firms_raw_ids)
        records.append(
            {
                "schema_version": 1,
                "weather_date": weather_date.isoformat(),
                "candidate_cell_id": str(assignment.candidate_cell_id),
                "candidate_cell_latitude": float(assignment.candidate_cell_latitude),
                "candidate_cell_longitude": float(assignment.candidate_cell_longitude),
                "source_example_ids": list(assignment.source_example_ids),
                "source_example_count": int(assignment.source_example_count),
                "source_firms_raw_artifact_ids": list(firms_raw_ids),
                "weather_tile_id": tile_id,
                "weather_tile_latitude": float(assignment.forecast_tile_latitude),
                "weather_tile_longitude": float(assignment.forecast_tile_longitude),
                "weather_tile_distance_m": float(assignment.forecast_tile_distance_m),
                "source_grid_id": str(source["source_grid_id"]),
                "source_grid_latitude": float(source["source_grid_latitude"]),
                "source_grid_longitude": float(source["source_grid_longitude"]),
                "provider": "Open-Meteo",
                "feature_mode": OPEN_METEO_HISTORICAL_FEATURE_MODE,
                "product_kind": OPEN_METEO_HISTORICAL_WEATHER_KIND,
                "model": model,
                "raw_artifact_id": raw_artifact_id,
                "retrieved_at": format_utc(retrieved_at),
            }
        )
    return write_normalized_jsonl(
        root,
        entity="open_meteo_historical_weather_tile_assignments",
        records=records,
        partitions={"weather_date": weather_date.isoformat(), "model": model},
        raw_artifact_ids=sorted(raw_ids),
        transformation_version=OPEN_METEO_HISTORICAL_WEATHER_MAPPING_VERSION,
        generated_at=retrieved_at,
    )


def _write_raw_response(
    root: Path,
    *,
    response: Any,
    weather_date: date,
    model: str,
    parameters: Mapping[str, object],
    retrieved_at: datetime,
) -> RawArtifact:
    return write_raw_artifact(
        root,
        source="Open-Meteo:Historical Weather",
        payload=bytes(response.content),
        retrieved_at=retrieved_at,
        media_type="application/json",
        provenance={
            "source_url": OPEN_METEO_HISTORICAL_WEATHER_URL,
            "request_parameters": dict(parameters),
            "response_headers": dict(getattr(response, "headers", {}) or {}),
            "response_status_code": int(response.status_code),
            "provider": "Open-Meteo",
            "product": OPEN_METEO_HISTORICAL_WEATHER_PRODUCT,
            "product_kind": OPEN_METEO_HISTORICAL_WEATHER_KIND,
            "model": model,
            "weather_date": weather_date.isoformat(),
        },
    )


def _archive_failed_response(
    root: Path,
    *,
    response: Any,
    weather_date: date,
    model: str,
    parameters: Mapping[str, object],
    batch_identity: str,
    retrieved_at: datetime,
) -> tuple[RawArtifact, CoverageRecord]:
    raw = _write_raw_response(
        root,
        response=response,
        weather_date=weather_date,
        model=model,
        parameters=parameters,
        retrieved_at=retrieved_at,
    )
    return raw, _record_coverage(
        root,
        weather_date=weather_date,
        model=model,
        batch_identity=batch_identity,
        status=CoverageStatus.FAILED,
        raw_artifact_id=raw.raw_artifact_id,
        retrieved_at=retrieved_at,
        error=f"HTTP {response.status_code}",
    )


def _record_coverage(
    root: Path,
    *,
    weather_date: date,
    model: str,
    batch_identity: str,
    status: CoverageStatus,
    raw_artifact_id: str,
    retrieved_at: datetime,
    error: str | None = None,
    detail: Mapping[str, object] | None = None,
) -> CoverageRecord:
    start = datetime.combine(weather_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(weather_date, time.max, tzinfo=timezone.utc)
    return CoverageLedger(root).record(
        source="Open-Meteo",
        product=OPEN_METEO_HISTORICAL_WEATHER_PRODUCT,
        coverage_start=start,
        coverage_end=end,
        region="United States and Canada",
        expected_coverage_id=(
            f"open-meteo-historical-weather:{model}:{weather_date.isoformat()}:{batch_identity}"
        ),
        status=status,
        artifact_sha256s=[raw_artifact_id],
        error=error,
        detail=dict(detail or {}),
        recorded_at=retrieved_at,
    )


def _backfill_report(
    weather_date: date,
    result: HistoricalWeatherCaptureResult,
    root: Path,
) -> dict[str, object]:
    return {
        "weather_date": weather_date.isoformat(),
        "status": "partial" if result.paused_for_rate_limit else "complete",
        "planned_tile_count": result.planned_tile_count,
        "captured_tile_count": result.captured_tile_count,
        "candidate_cell_count": result.candidate_cell_count,
        "measurement_count": result.measurement_count,
        "http_attempts": result.http_attempts,
        "api_call_units": result.api_call_units,
        "rate_limit_retries": result.rate_limit_retries,
        "raw_artifact_ids": [artifact.raw_artifact_id for artifact in result.raw_artifacts],
        "measurement_artifact_relative_paths": [
            artifact.artifact_path.relative_to(root).as_posix()
            for artifact in result.measurement_artifacts
        ],
        "assignment_artifact_relative_paths": [
            artifact.artifact_path.relative_to(root).as_posix()
            for artifact in result.assignment_artifacts
        ],
        "coverage_relative_paths": [
            coverage.path.relative_to(root).as_posix() for coverage in result.coverage_records
        ],
    }


def _write_backfill_manifest(
    root: Path,
    *,
    candidate_manifest: CandidateManifestIdentity,
    model: str,
    start_date: date,
    end_date: date,
    max_tile_distance_m: float,
    requests_per_minute: int,
    reports: list[dict[str, object]],
    complete: bool,
    paused_for_rate_limit: bool,
    generated_at: datetime,
) -> Path:
    backfill_id = uuid.uuid4().hex
    document = {
        "schema_version": 1,
        "kind": "open-meteo-historical-weather-backfill",
        "status": "complete" if complete else "partial",
        "backfill_id": backfill_id,
        "backfill_version": OPEN_METEO_HISTORICAL_WEATHER_BACKFILL_VERSION,
        "generated_at": format_utc(generated_at),
        "candidate_manifest": {
            "relative_path": candidate_manifest.relative_path,
            "build_id": candidate_manifest.build_id,
            "content_sha256": candidate_manifest.content_sha256,
        },
        "provider": "Open-Meteo",
        "product": OPEN_METEO_HISTORICAL_WEATHER_PRODUCT,
        "product_kind": OPEN_METEO_HISTORICAL_WEATHER_KIND,
        "source_url": OPEN_METEO_HISTORICAL_WEATHER_URL,
        "model": model,
        "hourly_variables": list(OPEN_METEO_HISTORICAL_HOURLY_VARIABLES),
        "feature_hour_policy": "floor-anchor-to-utc-hour/v1",
        "feature_mode": OPEN_METEO_HISTORICAL_FEATURE_MODE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "max_tile_distance_m": max_tile_distance_m,
        "request_rate_location_units_per_minute": requests_per_minute,
        "paused_for_rate_limit": paused_for_rate_limit,
        "reports": reports,
    }
    destination = (
        root
        / "manifests"
        / "open-meteo-historical-weather-backfills"
        / generated_at.strftime("%Y/%m/%d")
        / f"{generated_at.strftime('%H%M%S%f')}_{backfill_id}.json"
    )
    return write_atomic_json(destination, document)


def _candidate_manifest_identity(
    root: Path,
    value: str | Path,
) -> CandidateManifestIdentity:
    path = _resolve_data_file(root, value, "candidate manifest")
    document = _read_json_document(path, "candidate manifest")
    if document.get("kind") != "completed-firms-candidate-dataset-build":
        raise OpenMeteoHistoricalWeatherError("candidate manifest has the wrong kind")
    if document.get("status") != "complete":
        raise OpenMeteoHistoricalWeatherError("candidate manifest is not complete")
    return CandidateManifestIdentity(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        build_id=_required_text(document.get("build_id"), "candidate manifest build_id"),
        content_sha256=_sha256_file(path),
    )


def _read_backfill_manifest(root: Path, value: str | Path) -> dict[str, object]:
    path = _resolve_data_file(root, value, "resume manifest")
    document = _read_json_document(path, "resume manifest")
    if document.get("kind") != "open-meteo-historical-weather-backfill":
        raise OpenMeteoHistoricalWeatherError("resume manifest has the wrong kind")
    return document


def _backfill_date_range(document: Mapping[str, object]) -> tuple[date, date]:
    try:
        start = date.fromisoformat(_required_text(document.get("start_date"), "start_date"))
        end = date.fromisoformat(_required_text(document.get("end_date"), "end_date"))
    except ValueError as exc:
        raise OpenMeteoHistoricalWeatherError("resume manifest has invalid date bounds") from exc
    if end < start:
        raise OpenMeteoHistoricalWeatherError("resume manifest end date precedes its start date")
    return start, end


def _completed_resume_reports(
    root: Path,
    *,
    document: Mapping[str, object],
    candidate_identity: CandidateManifestIdentity,
    model: str,
    max_tile_distance_m: float,
) -> dict[date, dict[str, object]]:
    candidate = document.get("candidate_manifest")
    if not isinstance(candidate, Mapping):
        raise OpenMeteoHistoricalWeatherError("resume manifest has no immutable candidate identity")
    if _required_text(candidate.get("build_id"), "resume candidate build_id") != candidate_identity.build_id:
        raise OpenMeteoHistoricalWeatherError("resume manifest belongs to a different candidate view")
    if _required_text(candidate.get("content_sha256"), "resume candidate content_sha256") != (
        candidate_identity.content_sha256
    ):
        raise OpenMeteoHistoricalWeatherError("resume candidate manifest contents do not match")
    if _required_text(document.get("model"), "resume model") != model:
        raise OpenMeteoHistoricalWeatherError("resume manifest uses a different weather model")
    if document.get("product_kind") != OPEN_METEO_HISTORICAL_WEATHER_KIND:
        raise OpenMeteoHistoricalWeatherError("resume manifest has an unsupported weather product")
    if document.get("feature_mode") != OPEN_METEO_HISTORICAL_FEATURE_MODE:
        raise OpenMeteoHistoricalWeatherError("resume manifest has the wrong feature mode")
    try:
        resume_distance = float(document.get("max_tile_distance_m"))
    except (TypeError, ValueError) as exc:
        raise OpenMeteoHistoricalWeatherError(
            "resume manifest has an invalid max_tile_distance_m"
        ) from exc
    if not math.isfinite(resume_distance) or resume_distance != max_tile_distance_m:
        raise OpenMeteoHistoricalWeatherError(
            "resume manifest uses a different weather tile-distance policy"
        )
    reports = document.get("reports")
    if not isinstance(reports, list):
        raise OpenMeteoHistoricalWeatherError("resume manifest reports must be a list")
    complete: dict[date, dict[str, object]] = {}
    seen_dates: set[date] = set()
    for report in reports:
        if not isinstance(report, Mapping):
            raise OpenMeteoHistoricalWeatherError("resume manifest contains an invalid date report")
        try:
            weather_date = date.fromisoformat(
                _required_text(report.get("weather_date"), "resume weather_date")
            )
        except ValueError as exc:
            raise OpenMeteoHistoricalWeatherError(
                "resume manifest contains an invalid weather date"
            ) from exc
        if weather_date in seen_dates:
            raise OpenMeteoHistoricalWeatherError("resume manifest repeats a weather date")
        seen_dates.add(weather_date)
        status = _required_text(report.get("status"), "resume weather status")
        if status == "complete":
            _validate_reusable_report_paths(root, report)
            complete[weather_date] = dict(report)
        elif status not in {"partial", "failed"}:
            raise OpenMeteoHistoricalWeatherError(
                f"resume manifest has an unsupported weather date status: {status!r}"
            )
    return complete


def _validate_reusable_report_paths(root: Path, report: Mapping[str, object]) -> None:
    for field in (
        "measurement_artifact_relative_paths",
        "assignment_artifact_relative_paths",
    ):
        values = report.get(field)
        if not isinstance(values, list) or not values:
            raise OpenMeteoHistoricalWeatherError(
                f"resume complete report has no {field}"
            )
        for value in values:
            path = _relative_data_file(root, value, field)
            if not path.is_file():
                raise OpenMeteoHistoricalWeatherError(
                    f"resume complete report references a missing artifact: {path.relative_to(root)}"
                )


def _validate_resume_report_dates(
    reports: Mapping[date, Mapping[str, object]],
    *,
    expected_dates: set[date],
) -> None:
    unexpected = sorted(set(reports).difference(expected_dates))
    if unexpected:
        rendered = ", ".join(value.isoformat() for value in unexpected[:5])
        raise OpenMeteoHistoricalWeatherError(
            "resume manifest has completed weather outside the selected candidate dates: " + rendered
        )


def _resolve_data_file(root: Path, value: str | Path, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise OpenMeteoHistoricalWeatherError(f"{label} must be a path")
    provided = Path(value)
    candidates = (provided.resolve(),) if provided.is_absolute() else (
        provided.resolve(),
        (root / provided).resolve(),
    )
    for path in candidates:
        if path.is_file() and path.is_relative_to(root):
            return path
    raise OpenMeteoHistoricalWeatherError(f"{label} must be an existing file below data_root")


def _relative_data_file(root: Path, value: object, label: str) -> Path:
    relative = Path(_required_text(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise OpenMeteoHistoricalWeatherError(f"{label} must be a relative data-root path")
    return root / relative


def _read_json_document(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenMeteoHistoricalWeatherError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise OpenMeteoHistoricalWeatherError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _conservative_batch_bytes(*, tile_count: int, candidate_cell_count: int) -> int:
    expected_records = tile_count * 24 * (len(OPEN_METEO_HISTORICAL_HOURLY_VARIABLES) + 2)
    return max(262_144, expected_records * 768 + candidate_cell_count * 1_024 + 262_144)


def _batch_identity(model: str, weather_date: date, batch: pd.DataFrame) -> str:
    identifiers = "\x1f".join(sorted(str(value) for value in batch["forecast_tile_id"]))
    digest = hashlib.sha256(
        "\x1f".join((model, weather_date.isoformat(), identifiers)).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def _batches(frame: pd.DataFrame, size: int):
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size].copy()


def _parse_open_meteo_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Open-Meteo hourly time must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Open-Meteo hourly time must be ISO-8601") from exc
    if parsed.tzinfo is None:
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


def _coordinate_text(value: object) -> str:
    return format(_finite_float(value, "coordinate"), ".7f")


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
    return value.strip() if isinstance(value, str) and value.strip() else default


def _wind_speed_metres_per_second(value: float, unit: str) -> float:
    normalized = unit.casefold().replace(" ", "")
    if normalized in {"m/s", "ms", "metrespersecond", "meterspersecond"}:
        return value
    if normalized in {"km/h", "kmh", "kilometresperhour", "kilometersperhour"}:
        return value / 3.6
    raise ValueError(f"unsupported Open-Meteo wind speed unit: {unit!r}")


def _artifact_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("raw-artifact IDs must be a sequence")
    return tuple(sorted({_required_text(item, "raw_artifact_id") for item in value}))


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value.strip()
