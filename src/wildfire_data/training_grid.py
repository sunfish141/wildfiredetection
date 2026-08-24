"""Canonical spatial and temporal keys for model-ready wildfire examples.

Collection tiles are deliberately not training cells: they were chosen to
bound downloads and have variable ground size.  This module defines the one
fixed 1 km equal-area lattice shared by labels, features, splits, inference,
and evaluation.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from pyproj import Transformer


TRAINING_GRID_SCHEMA_VERSION = 1
TRAINING_GRID_CRS = "ESRI:102008"
WGS84_CRS = "EPSG:4326"
DEFAULT_CELL_SIZE_METRES = 1_000
DEFAULT_HORIZON_HOURS = 12
_CELL_ID_PATTERN = re.compile(r"^naea-1km:x=(-?\d+):y=(-?\d+)$")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class TrainingGridError(ValueError):
    """Raised when a spatial or temporal training key is ambiguous."""


@dataclass(frozen=True)
class GridCell:
    """One 1 km by 1 km cell in the canonical equal-area lattice."""

    x_index: int
    y_index: int
    cell_size_metres: int = DEFAULT_CELL_SIZE_METRES

    def __post_init__(self) -> None:
        if self.cell_size_metres != DEFAULT_CELL_SIZE_METRES:
            raise TrainingGridError(
                f"v{TRAINING_GRID_SCHEMA_VERSION} only supports "
                f"{DEFAULT_CELL_SIZE_METRES:,} m cells"
            )

    @property
    def cell_id(self) -> str:
        """Return a stable, human-readable, projection-specific cell id."""
        return f"naea-1km:x={self.x_index}:y={self.y_index}"

    @property
    def bounds_projected(self) -> tuple[float, float, float, float]:
        """Return ``(xmin, ymin, xmax, ymax)`` in ``ESRI:102008`` metres."""
        xmin = self.x_index * self.cell_size_metres
        ymin = self.y_index * self.cell_size_metres
        return (
            float(xmin),
            float(ymin),
            float(xmin + self.cell_size_metres),
            float(ymin + self.cell_size_metres),
        )

    @property
    def center_projected(self) -> tuple[float, float]:
        """Return the cell centre in ``ESRI:102008`` metres."""
        xmin, ymin, xmax, ymax = self.bounds_projected
        return ((xmin + xmax) / 2, (ymin + ymax) / 2)

    @property
    def center_wgs84(self) -> tuple[float, float]:
        """Return the cell-centre ``(latitude, longitude)`` in WGS84."""
        x, y = self.center_projected
        longitude, latitude = _to_wgs84().transform(x, y)
        return (float(latitude), float(longitude))


@dataclass(frozen=True)
class TrainingExampleKey:
    """Identity of one prediction made at a 12-hour cutoff for one cell.

    FEDS labels use a cell-specific local-solar overpass phase.  The key does
    not force cutoffs to a global UTC clock; it instead preserves the actual
    UTC estimate used to select past evidence and issued forecasts.
    """

    cell_id: str
    anchor_at: datetime
    horizon: timedelta = timedelta(hours=DEFAULT_HORIZON_HOURS)

    def __post_init__(self) -> None:
        cell_from_id(self.cell_id)
        anchor = _as_utc(self.anchor_at, "anchor_at")
        if self.horizon != timedelta(hours=DEFAULT_HORIZON_HOURS):
            raise TrainingGridError(
                f"v{TRAINING_GRID_SCHEMA_VERSION} only supports a "
                f"{DEFAULT_HORIZON_HOURS}-hour horizon"
            )
        object.__setattr__(self, "anchor_at", anchor)

    @property
    def target_end_at(self) -> datetime:
        """Return the exclusive end of the future label window."""
        return self.anchor_at + self.horizon

    @property
    def example_id(self) -> str:
        """Return a durable key suitable for deduplicating derived records."""
        encoded = "\x1f".join(
            (
                str(TRAINING_GRID_SCHEMA_VERSION),
                self.cell_id,
                format_utc(self.anchor_at),
                str(int(self.horizon.total_seconds())),
            )
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def cell_from_wgs84(*, latitude: float, longitude: float) -> GridCell:
    """Map a WGS84 point to exactly one canonical 1 km training cell."""
    latitude_value = _finite_float(latitude, "latitude")
    longitude_value = _finite_float(longitude, "longitude")
    if not -90 <= latitude_value <= 90:
        raise TrainingGridError("latitude must be between -90 and 90")
    if not -180 <= longitude_value <= 180:
        raise TrainingGridError("longitude must be between -180 and 180")
    x, y = _to_projected().transform(longitude_value, latitude_value)
    return GridCell(
        x_index=math.floor(x / DEFAULT_CELL_SIZE_METRES),
        y_index=math.floor(y / DEFAULT_CELL_SIZE_METRES),
    )


def cell_from_id(cell_id: str) -> GridCell:
    """Parse a canonical cell id and reject collection-tile identifiers."""
    if not isinstance(cell_id, str):
        raise TrainingGridError("cell_id must be a string")
    match = _CELL_ID_PATTERN.fullmatch(cell_id)
    if match is None:
        raise TrainingGridError("cell_id is not a canonical naea-1km id")
    return GridCell(x_index=int(match.group(1)), y_index=int(match.group(2)))


def cells_in_square_radius(cell: GridCell, *, radius_cells: int) -> tuple[GridCell, ...]:
    """Return cells in a deterministic square neighbourhood around ``cell``.

    The centre cell is included.  A radius of one therefore represents the
    3 km by 3 km local context used by the first tabular baseline.
    """
    if not isinstance(radius_cells, int) or radius_cells < 0:
        raise TrainingGridError("radius_cells must be a non-negative integer")
    return tuple(
        GridCell(x_index=x_index, y_index=y_index)
        for y_index in range(cell.y_index - radius_cells, cell.y_index + radius_cells + 1)
        for x_index in range(cell.x_index - radius_cells, cell.x_index + radius_cells + 1)
    )


def anchor_time_bin(value: datetime, *, horizon: timedelta = timedelta(hours=DEFAULT_HORIZON_HOURS)) -> datetime:
    """Floor an aware timestamp to the canonical 12-hour UTC anchor."""
    resolved = _as_utc(value, "value")
    if horizon != timedelta(hours=DEFAULT_HORIZON_HOURS):
        raise TrainingGridError(
            f"v{TRAINING_GRID_SCHEMA_VERSION} only supports a "
            f"{DEFAULT_HORIZON_HOURS}-hour horizon"
        )
    seconds = (resolved - _EPOCH).total_seconds()
    interval_seconds = horizon.total_seconds()
    floored_seconds = math.floor(seconds / interval_seconds) * interval_seconds
    return _EPOCH + timedelta(seconds=floored_seconds)


def format_utc(value: datetime) -> str:
    """Format an aware datetime as a stable UTC ISO-8601 string."""
    return _as_utc(value, "value").isoformat().replace("+00:00", "Z")


@lru_cache(maxsize=1)
def _to_projected() -> Transformer:
    return Transformer.from_crs(WGS84_CRS, TRAINING_GRID_CRS, always_xy=True)


@lru_cache(maxsize=1)
def _to_wgs84() -> Transformer:
    return Transformer.from_crs(TRAINING_GRID_CRS, WGS84_CRS, always_xy=True)


def _as_utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TrainingGridError(f"{label} must be an offset-aware datetime")
    return value.astimezone(timezone.utc)


def _finite_float(value: float, label: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingGridError(f"{label} must be numeric") from exc
    if not math.isfinite(resolved):
        raise TrainingGridError(f"{label} must be finite")
    return resolved
