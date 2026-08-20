"""Lossless normalization for individual NASA FIRMS fire-detection records.

The normalizer deliberately keeps filtering decisions out of collection.  It
returns every input source field and provenance value in JSON-safe form, then
adds validated core fields and derived metadata alongside them.  This makes a
record reproducible when FIRMS adds columns or a later model needs a field that
is not currently interpreted by the application.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
from numbers import Integral, Real
from typing import Any


DEFAULT_MINIMUM_BRIGHT_TI4 = 305.0
REQUIRED_CORE_FIELDS = (
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "bright_ti4",
)


class FirmsRecordValidationError(ValueError):
    """Raised when a FIRMS detection cannot be normalized safely."""


def normalize_firms_detection(
    source_fields: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    minimum_bright_ti4: float | None = DEFAULT_MINIMUM_BRIGHT_TI4,
) -> dict[str, Any]:
    """Return one lossless, JSON-safe normalized FIRMS detection record.

    ``source_fields`` must contain the core FIRMS fields listed in
    :data:`REQUIRED_CORE_FIELDS`.  All supplied source fields, including
    fields unknown to this module, are kept under ``raw_source_fields``.

    ``provenance`` must include ``provider`` and ``product``.  Other keys are
    retained verbatim (after JSON-safe conversion), including collector-level
    metadata such as ``raw_artifact_id``, ``raw_record_offset``,
    ``ingestion_id``, and ``ingested_at``.

    The returned record is never filtered by brightness.  The configured TI4
    threshold is represented only by ``derived.ti4_threshold``.
    """
    raw_source_fields = _json_safe_mapping(source_fields, name="source_fields")
    normalized_provenance = _json_safe_mapping(provenance, name="provenance")
    _require_provenance_value(normalized_provenance, "provider")
    _require_provenance_value(normalized_provenance, "product")

    core = _validate_core_fields(raw_source_fields)
    threshold = _validate_threshold(minimum_bright_ti4)
    source_id = _derive_source_id(normalized_provenance)
    detection_id = _derive_detection_id(
        source_id=source_id,
        raw_source_fields=raw_source_fields,
        provenance=normalized_provenance,
    )

    passes_threshold = None if threshold is None else core["bright_ti4"] >= threshold
    return {
        "record_type": "firms_detection",
        "source_id": source_id,
        "detection_id": detection_id,
        "acquired_at": core["acquired_at"],
        "latitude": core["latitude"],
        "longitude": core["longitude"],
        "bright_ti4": core["bright_ti4"],
        "raw_source_fields": raw_source_fields,
        "provenance": normalized_provenance,
        "derived": {
            "ti4_threshold": {
                "minimum_bright_ti4": threshold,
                "passes": passes_threshold,
            }
        },
    }


def _validate_core_fields(source_fields: Mapping[str, Any]) -> dict[str, Any]:
    missing_fields = [field for field in REQUIRED_CORE_FIELDS if field not in source_fields]
    if missing_fields:
        missing = ", ".join(repr(field) for field in missing_fields)
        raise FirmsRecordValidationError(f"source_fields is missing required field(s): {missing}")

    latitude = _parse_finite_number(source_fields["latitude"], field="latitude")
    if not -90.0 <= latitude <= 90.0:
        raise FirmsRecordValidationError("latitude must be between -90 and 90")

    longitude = _parse_finite_number(source_fields["longitude"], field="longitude")
    if not -180.0 <= longitude <= 180.0:
        raise FirmsRecordValidationError("longitude must be between -180 and 180")

    bright_ti4 = _parse_finite_number(source_fields["bright_ti4"], field="bright_ti4")
    acquired_at = _parse_acquired_at(
        source_fields["acq_date"],
        source_fields["acq_time"],
    )
    return {
        "latitude": latitude,
        "longitude": longitude,
        "bright_ti4": bright_ti4,
        "acquired_at": _format_utc_datetime(acquired_at),
    }


def _parse_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise FirmsRecordValidationError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise FirmsRecordValidationError(f"{field} must be a finite number") from error
    if not math.isfinite(parsed):
        raise FirmsRecordValidationError(f"{field} must be a finite number")
    return parsed


def _parse_acquired_at(acq_date: Any, acq_time: Any) -> datetime:
    parsed_date = _parse_acquisition_date(acq_date)
    hour, minute = _parse_acquisition_time(acq_time)
    return datetime(
        parsed_date.year,
        parsed_date.month,
        parsed_date.day,
        hour,
        minute,
        tzinfo=timezone.utc,
    )


def _parse_acquisition_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise FirmsRecordValidationError("acq_date must be an ISO-8601 date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise FirmsRecordValidationError("acq_date must be an ISO-8601 date") from error


def _parse_acquisition_time(value: Any) -> tuple[int, int]:
    if isinstance(value, bool) or value is None:
        raise FirmsRecordValidationError("acq_time must be a valid HHMM value")
    if isinstance(value, Integral):
        digits = str(int(value))
    elif isinstance(value, Real):
        if not math.isfinite(float(value)) or not float(value).is_integer():
            raise FirmsRecordValidationError("acq_time must be a valid HHMM value")
        digits = str(int(value))
    elif isinstance(value, str):
        digits = value.strip()
    else:
        raise FirmsRecordValidationError("acq_time must be a valid HHMM value")

    if not digits.isascii() or not digits.isdigit() or not 1 <= len(digits) <= 4:
        raise FirmsRecordValidationError("acq_time must be a valid HHMM value")

    padded = digits.zfill(4)
    hour = int(padded[:2])
    minute = int(padded[2:])
    if hour > 23 or minute > 59:
        raise FirmsRecordValidationError("acq_time must be a valid HHMM value")
    return hour, minute


def _validate_threshold(value: float | None) -> float | None:
    if value is None:
        return None
    return _parse_finite_number(value, field="minimum_bright_ti4")


def _derive_source_id(provenance: Mapping[str, Any]) -> str:
    # Ingestion and retrieval identifiers intentionally do not participate in
    # source identity: downloading the same FIRMS product again must produce
    # the same source ID.
    identity_keys = ("provider", "product", "version", "product_version", "schema_version")
    identity = {key: provenance[key] for key in identity_keys if key in provenance}
    return _stable_identifier("firms-source", identity)


def _derive_detection_id(
    *,
    source_id: str,
    raw_source_fields: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    source_native_id = provenance.get("source_native_id")
    if _has_nonempty_value(source_native_id):
        identity: dict[str, Any] = {
            "source_id": source_id,
            "source_native_id": source_native_id,
        }
    else:
        # FIRMS CSV detections do not universally expose a native detection ID.
        # Hashing the complete original record prevents newly added columns from
        # being silently discarded and remains deterministic across reprocessing.
        identity = {
            "source_id": source_id,
            "raw_source_fields": raw_source_fields,
        }
    return _stable_identifier("firms-detection", identity)


def _stable_identifier(namespace: str, value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = sha256(serialized.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def _require_provenance_value(provenance: Mapping[str, Any], key: str) -> None:
    if not _has_nonempty_value(provenance.get(key)):
        raise FirmsRecordValidationError(f"provenance is missing required value: {key!r}")


def _has_nonempty_value(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _json_safe_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings for JSON-safe output")
        result[key] = _json_safe_value(item)
    return result


def _json_safe_value(value: Any) -> Any:
    """Convert common tabular values to standard JSON-compatible values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, datetime):
        return _format_datetime(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Mapping):
        return _json_safe_mapping(value, name="nested mapping")
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        serialized_items = [_json_safe_value(item) for item in value]
        return sorted(
            serialized_items,
            key=lambda item: json.dumps(item, allow_nan=False, sort_keys=True),
        )

    # NumPy and pandas scalar values commonly expose ``item`` without requiring
    # this module to depend on either library.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return _json_safe_value(scalar)

    # A string representation is preferable to dropping an as-yet unknown
    # provider value; the resulting record remains valid JSON and auditable.
    return str(value)


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        return value.isoformat()
    return _format_utc_datetime(value.astimezone(timezone.utc))


def _format_utc_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
