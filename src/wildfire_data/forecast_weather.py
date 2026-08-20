"""Canonical, leakage-safe records for issued weather forecasts.

The live weather lookup used by the notebook is useful for visualization, but
it does not identify the forecast run that was available when a prediction
would have been made.  This module keeps those concepts explicit so forecast
data can be collected once and safely reused for training and inference.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


FORECAST_SCHEMA_VERSION = 1


class ForecastRecordError(ValueError):
    """Raised when a weather forecast record lacks reliable provenance."""


def _parse_utc(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ForecastRecordError(f"{label} must be an ISO-8601 timestamp") from exc
    else:
        raise ForecastRecordError(f"{label} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise ForecastRecordError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: object, label: str) -> str:
    return _parse_utc(value, label).isoformat().replace("+00:00", "Z")


def _required_text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ForecastRecordError(f"forecast record is missing required field: {field}")
    return value.strip()


def _optional_float(record: Mapping[str, object], field: str) -> float | None:
    value = record.get(field)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ForecastRecordError(f"{field} must be numeric") from exc


def _stable_id(parts: Iterable[object]) -> str:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return sha256(encoded).hexdigest()


def normalize_forecast_measurement(
    measurement: Mapping[str, object],
    *,
    provider: str,
    model: str,
    model_run_at: object,
    published_at: object,
    retrieved_at: object,
    raw_artifact_id: str,
    ingestion_id: str,
    source_uri: str,
    model_version: str | None = None,
    schema_version: int = FORECAST_SCHEMA_VERSION,
) -> dict[str, object]:
    """Return one normalized weather measurement with forecast provenance.

    Measurements are intentionally long-form: retaining a variable, level,
    unit and value per row means new forecast variables do not require a
    schema migration.  ``raw_fields`` keeps provider-specific content intact.
    """
    if not provider.strip() or not model.strip():
        raise ForecastRecordError("provider and model must be non-empty")
    if not raw_artifact_id or not ingestion_id or not source_uri:
        raise ForecastRecordError(
            "raw_artifact_id, ingestion_id, and source_uri must be non-empty"
        )

    run_at = _parse_utc(model_run_at, "model_run_at")
    available_at = _parse_utc(published_at, "published_at")
    retrieved = _parse_utc(retrieved_at, "retrieved_at")
    if available_at < run_at:
        raise ForecastRecordError("published_at must not precede model_run_at")

    valid_at = _utc_text(measurement.get("valid_at"), "valid_at")
    grid_id = _required_text(measurement, "source_grid_id")
    variable = _required_text(measurement, "variable")
    unit = _required_text(measurement, "unit")
    value = _optional_float(measurement, "value")
    if value is None:
        raise ForecastRecordError("forecast record is missing required field: value")

    latitude = _optional_float(measurement, "latitude_wgs84")
    longitude = _optional_float(measurement, "longitude_wgs84")
    if (latitude is None) != (longitude is None):
        raise ForecastRecordError("latitude_wgs84 and longitude_wgs84 must be supplied together")
    if latitude is not None and not -90 <= latitude <= 90:
        raise ForecastRecordError("latitude_wgs84 must be between -90 and 90")
    if longitude is not None and not -180 <= longitude <= 180:
        raise ForecastRecordError("longitude_wgs84 must be between -180 and 180")

    level = str(measurement.get("level", "surface"))
    member = str(measurement.get("member", "deterministic"))
    record_id = _stable_id(
        (
            provider,
            model,
            model_version or "",
            run_at.isoformat(),
            valid_at,
            grid_id,
            variable,
            level,
            member,
        )
    )
    valid = _parse_utc(valid_at, "valid_at")
    return {
        "weather_snapshot_id": record_id,
        "provider": provider.strip(),
        "product_kind": "forecast",
        "model": model.strip(),
        "model_version": model_version,
        "model_run_at": run_at.isoformat().replace("+00:00", "Z"),
        "published_at": available_at.isoformat().replace("+00:00", "Z"),
        "retrieved_at": retrieved.isoformat().replace("+00:00", "Z"),
        "valid_at": valid_at,
        "lead_hours": (valid - run_at).total_seconds() / 3_600,
        "source_grid_id": grid_id,
        "latitude_wgs84": latitude,
        "longitude_wgs84": longitude,
        "variable": variable,
        "level": level,
        "member": member,
        "value": value,
        "unit": unit,
        "raw_artifact_id": raw_artifact_id,
        "ingestion_id": ingestion_id,
        "source_uri": source_uri,
        "schema_version": schema_version,
        "raw_fields": dict(measurement),
    }


def forecasts_available_at(
    records: Iterable[Mapping[str, object]], *, anchor_at: object
) -> list[Mapping[str, object]]:
    """Return measurements provably published by a prediction cutoff.

    This deliberately does not require ``valid_at <= anchor_at`` because the
    point of a forecast is to make a prediction for a future valid time.
    """
    anchor = _parse_utc(anchor_at, "anchor_at")
    available = []
    for record in records:
        run_at = _parse_utc(record.get("model_run_at"), "model_run_at")
        published_at = _parse_utc(record.get("published_at"), "published_at")
        if run_at <= published_at <= anchor:
            available.append(record)
    return available


def latest_forecasts_as_of(
    records: Iterable[Mapping[str, object]], *, anchor_at: object
) -> list[Mapping[str, object]]:
    """Choose the latest available run for each forecast value identity."""
    newest: dict[tuple[object, ...], Mapping[str, object]] = {}
    for record in forecasts_available_at(records, anchor_at=anchor_at):
        key = (
            record.get("provider"),
            record.get("model"),
            record.get("valid_at"),
            record.get("source_grid_id"),
            record.get("variable"),
            record.get("level"),
            record.get("member"),
        )
        prior = newest.get(key)
        if prior is None or _parse_utc(record.get("published_at"), "published_at") > _parse_utc(
            prior.get("published_at"), "published_at"
        ):
            newest[key] = record
    return list(newest.values())
