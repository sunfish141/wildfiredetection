"""Sample compact ETOPO terrain inputs at canonical training-cell centres.

The terrain collector stores bounded WGS84 ETOPO source blocks, including a
one-pixel halo around most neighbouring blocks. Those blocks are collection
artifacts, not a second model grid. This module resolves a canonical 1 km
``GridCell`` to exactly one retained source pixel at request time and returns
small JSON-serialisable feature records. It deliberately does not materialise
or persist a continental 1 km cache.
"""

from __future__ import annotations

import math
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .etopo_terrain import NO_DATA_ASPECT_DEGREES_X2, NO_DATA_ELEVATION_METRES
from .training_grid import GridCell, cell_from_id


TERRAIN_FEATURE_SCHEMA_VERSION = 1
TERRAIN_SAMPLING_METHOD = "containing-etopo-15arcsec-pixel-at-1km-cell-centre/v1"
TERRAIN_ASPECT_CONVENTION = "downhill-clockwise-from-north"
_ETOPO_DIRECTORY = Path("static") / "etopo-2022-15s"
_GRID_EDGE_TOLERANCE_PIXELS = 1.0e-9


class TerrainFeatureError(ValueError):
    """Raised when a compact terrain artifact does not satisfy its contract."""


@dataclass(frozen=True)
class _TerrainBlock:
    """Metadata required to locate a source pixel without loading its arrays."""

    path: Path
    relative_path: str
    source_block_id: str
    width: int
    height: int
    grid_west: float
    grid_north: float
    pixel_width_degrees: float
    pixel_height_degrees: float

    def source_pixel_at(self, *, latitude: float, longitude: float) -> tuple[int, int] | None:
        """Return a half-open-grid ``(row, column)`` for a WGS84 point.

        ETOPO arrays are north-up. A point exactly on a west or north grid
        edge belongs to the pixel on its east/south side; a point exactly on
        the east or south outer edge is outside. Near-integer ratios are
        snapped first so harmless projection round-off cannot move a centre
        across a source-pixel boundary.
        """
        column_ratio = _snap_grid_ratio((longitude - self.grid_west) / self.pixel_width_degrees)
        row_ratio = _snap_grid_ratio((self.grid_north - latitude) / self.pixel_height_degrees)
        if not (0.0 <= column_ratio < self.width and 0.0 <= row_ratio < self.height):
            return None
        return math.floor(row_ratio), math.floor(column_ratio)

    def edge_distance_pixels(self, *, row: int, column: int) -> int:
        """Return distance to the nearest block edge, used to prefer non-halo data."""
        return min(row, column, self.height - 1 - row, self.width - 1 - column)


@dataclass(frozen=True)
class _BlockSample:
    """A deterministic candidate source pixel from one compact ETOPO block."""

    block: _TerrainBlock
    row: int
    column: int

    @property
    def rank(self) -> tuple[int, float, str, str]:
        """Sort key: prefer interiors, then finer sources, then stable paths.

        Adjacent collection blocks overlap only in their halo. Preferring the
        candidate farthest from its edge selects the owning block whenever it
        is available. The remaining keys make a tie deterministic even if two
        overlap artifacts disagree unexpectedly.
        """
        return (
            -self.block.edge_distance_pixels(row=self.row, column=self.column),
            self.block.pixel_width_degrees * self.block.pixel_height_degrees,
            self.block.source_block_id,
            self.block.relative_path,
        )


class TerrainFeatureSampler:
    """Bounded-memory sampler for the retained compact ETOPO source blocks.

    ``max_cached_blocks`` is an in-memory LRU bound, not a derived dataset
    cache. Metadata discovery reads only NPZ headers and scalar grid metadata;
    terrain arrays are decompressed only for the selected source block.
    """

    def __init__(self, data_root: str | Path, *, max_cached_blocks: int = 2) -> None:
        if not isinstance(max_cached_blocks, int) or isinstance(max_cached_blocks, bool):
            raise TerrainFeatureError("max_cached_blocks must be a non-negative integer")
        if max_cached_blocks < 0:
            raise TerrainFeatureError("max_cached_blocks must be a non-negative integer")
        self._data_root = Path(data_root)
        self._max_cached_blocks = max_cached_blocks
        self._arrays_by_path: OrderedDict[Path, Mapping[str, np.ndarray]] = OrderedDict()
        artifact_root = self._data_root / _ETOPO_DIRECTORY
        self._blocks = tuple(
            _read_block_metadata(path, data_root=self._data_root)
            for path in sorted(artifact_root.glob("*.npz"))
        )

    @property
    def source_block_count(self) -> int:
        """Return the number of retained compact source blocks discovered."""
        return len(self._blocks)

    def sample_cell(self, cell: GridCell | str) -> dict[str, object]:
        """Return terrain features sampled at one canonical 1 km cell centre."""
        resolved = cell_from_id(cell) if isinstance(cell, str) else cell
        if not isinstance(resolved, GridCell):
            raise TerrainFeatureError("cell must be a GridCell or canonical cell_id")
        latitude, longitude = resolved.center_wgs84
        return self.sample_wgs84(latitude=latitude, longitude=longitude)

    def sample_wgs84(self, *, latitude: float, longitude: float) -> dict[str, object]:
        """Return terrain features for a point that is already a cell centre.

        This public helper is useful for testing and for pipeline code which
        has decoded a canonical cell centre already. Callers building model
        examples should normally use :meth:`sample_cell` so the canonical grid
        remains the source of the coordinate.
        """
        latitude_value = _finite_coordinate(latitude, "latitude", minimum=-90.0, maximum=90.0)
        longitude_value = _finite_coordinate(longitude, "longitude", minimum=-180.0, maximum=180.0)
        candidates = []
        for block in self._blocks:
            source_pixel = block.source_pixel_at(latitude=latitude_value, longitude=longitude_value)
            if source_pixel is not None:
                row, column = source_pixel
                candidates.append(_BlockSample(block=block, row=row, column=column))
        if not candidates:
            status = "no-terrain-artifacts" if not self._blocks else "outside-retained-terrain-coverage"
            return _unavailable_features(status=status)

        selected = min(candidates, key=lambda candidate: candidate.rank)
        arrays = self._arrays_for(selected.block)
        elevation = int(arrays["elevation_m"][selected.row, selected.column])
        slope_x2 = int(arrays["slope_degrees_x2"][selected.row, selected.column])
        aspect_x2 = int(arrays["aspect_degrees_x2"][selected.row, selected.column])
        metadata = _sample_metadata(selected)
        if elevation == int(NO_DATA_ELEVATION_METRES):
            return _unavailable_features(status="source-no-data", **metadata)

        aspect_defined = aspect_x2 != int(NO_DATA_ASPECT_DEGREES_X2)
        if aspect_defined:
            aspect_radians = math.radians(aspect_x2 * 2.0)
            aspect_sin = float(math.sin(aspect_radians))
            aspect_cos = float(math.cos(aspect_radians))
        else:
            # A flat slope has no direction. The explicit flag keeps this
            # neutral vector distinct from a valid north/south aspect.
            aspect_sin = 0.0
            aspect_cos = 0.0
        return {
            "terrain_valid": True,
            "terrain_elevation_m": float(elevation),
            "terrain_slope_degrees": float(slope_x2) / 2.0,
            "terrain_aspect_sin": aspect_sin,
            "terrain_aspect_cos": aspect_cos,
            "terrain_aspect_defined": aspect_defined,
            "terrain_coverage_status": "sampled",
            **metadata,
        }

    def _arrays_for(self, block: _TerrainBlock) -> Mapping[str, np.ndarray]:
        cached = self._arrays_by_path.pop(block.path, None)
        if cached is not None:
            self._arrays_by_path[block.path] = cached
            return cached
        with np.load(block.path, allow_pickle=False) as archive:
            arrays = {
                "elevation_m": np.ascontiguousarray(archive["elevation_m"]),
                "slope_degrees_x2": np.ascontiguousarray(archive["slope_degrees_x2"]),
                "aspect_degrees_x2": np.ascontiguousarray(archive["aspect_degrees_x2"]),
            }
        _validate_loaded_arrays(arrays, block=block)
        if self._max_cached_blocks:
            self._arrays_by_path[block.path] = arrays
            while len(self._arrays_by_path) > self._max_cached_blocks:
                self._arrays_by_path.popitem(last=False)
        return arrays


def sample_terrain_features(
    data_root: str | Path,
    *,
    cell: GridCell | str,
    max_cached_blocks: int = 2,
) -> dict[str, object]:
    """Sample terrain once without creating a persisted or global cache."""
    return TerrainFeatureSampler(data_root, max_cached_blocks=max_cached_blocks).sample_cell(cell)


def _read_block_metadata(path: Path, *, data_root: Path) -> _TerrainBlock:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "elevation_m",
                "slope_degrees_x2",
                "aspect_degrees_x2",
                "grid_west",
                "grid_north",
                "pixel_width_degrees",
                "pixel_height_degrees",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise TerrainFeatureError(f"terrain artifact {path} is missing {', '.join(missing)}")
            elevation_shape, elevation_dtype = _npz_array_header(path, "elevation_m")
            slope_shape, slope_dtype = _npz_array_header(path, "slope_degrees_x2")
            aspect_shape, aspect_dtype = _npz_array_header(path, "aspect_degrees_x2")
            _validate_array_headers(
                path,
                elevation_shape=elevation_shape,
                elevation_dtype=elevation_dtype,
                slope_shape=slope_shape,
                slope_dtype=slope_dtype,
                aspect_shape=aspect_shape,
                aspect_dtype=aspect_dtype,
            )
            grid_west = _finite_scalar(archive["grid_west"], "grid_west", path=path)
            grid_north = _finite_scalar(archive["grid_north"], "grid_north", path=path)
            pixel_width = _finite_scalar(
                archive["pixel_width_degrees"], "pixel_width_degrees", path=path
            )
            pixel_height = _finite_scalar(
                archive["pixel_height_degrees"], "pixel_height_degrees", path=path
            )
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        if isinstance(exc, TerrainFeatureError):
            raise
        raise TerrainFeatureError(f"cannot read terrain artifact {path}: {exc}") from exc

    if pixel_width <= 0.0 or pixel_height <= 0.0:
        raise TerrainFeatureError(f"terrain artifact {path} has non-positive pixel dimensions")
    relative_path = _relative_path(path, root=data_root)
    return _TerrainBlock(
        path=path,
        relative_path=relative_path,
        source_block_id=path.stem,
        width=elevation_shape[1],
        height=elevation_shape[0],
        grid_west=grid_west,
        grid_north=grid_north,
        pixel_width_degrees=pixel_width,
        pixel_height_degrees=pixel_height,
    )


def _npz_array_header(path: Path, array_name: str) -> tuple[tuple[int, ...], np.dtype]:
    """Read one NPZ member header without expanding its terrain array."""
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(f"{array_name}.npy") as source:
                version = np.lib.format.read_magic(source)
                readers = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                    (3, 0): np.lib.format.read_array_header_2_0,
                }
                reader = readers.get(version)
                if reader is None:
                    raise TerrainFeatureError(
                        f"terrain artifact {path} has unsupported NPY version {version!r}"
                    )
                shape, _fortran_order, dtype = reader(source)
    except KeyError as exc:
        raise TerrainFeatureError(f"terrain artifact {path} has no {array_name}.npy member") from exc
    return tuple(int(dimension) for dimension in shape), np.dtype(dtype)


def _validate_array_headers(
    path: Path,
    *,
    elevation_shape: tuple[int, ...],
    elevation_dtype: np.dtype,
    slope_shape: tuple[int, ...],
    slope_dtype: np.dtype,
    aspect_shape: tuple[int, ...],
    aspect_dtype: np.dtype,
) -> None:
    if len(elevation_shape) != 2 or min(elevation_shape) < 1:
        raise TerrainFeatureError(f"terrain artifact {path} elevation_m must be a non-empty 2-D array")
    if slope_shape != elevation_shape or aspect_shape != elevation_shape:
        raise TerrainFeatureError(f"terrain artifact {path} feature arrays must share a shape")
    expected_dtypes = {
        "elevation_m": np.dtype(np.int16),
        "slope_degrees_x2": np.dtype(np.uint8),
        "aspect_degrees_x2": np.dtype(np.uint8),
    }
    actual_dtypes = {
        "elevation_m": elevation_dtype,
        "slope_degrees_x2": slope_dtype,
        "aspect_degrees_x2": aspect_dtype,
    }
    mismatches = [
        f"{name}={actual_dtypes[name]}"
        for name in sorted(expected_dtypes)
        if actual_dtypes[name] != expected_dtypes[name]
    ]
    if mismatches:
        raise TerrainFeatureError(
            f"terrain artifact {path} has unexpected dtypes ({', '.join(mismatches)})"
        )


def _validate_loaded_arrays(arrays: Mapping[str, np.ndarray], *, block: _TerrainBlock) -> None:
    expected_shape = (block.height, block.width)
    expected_dtypes = {
        "elevation_m": np.dtype(np.int16),
        "slope_degrees_x2": np.dtype(np.uint8),
        "aspect_degrees_x2": np.dtype(np.uint8),
    }
    for name, expected_dtype in expected_dtypes.items():
        array = arrays[name]
        if array.shape != expected_shape or array.dtype != expected_dtype:
            raise TerrainFeatureError(f"terrain artifact {block.path} changed after metadata discovery")


def _sample_metadata(sample: _BlockSample) -> dict[str, object]:
    block = sample.block
    return {
        "terrain_feature_schema_version": TERRAIN_FEATURE_SCHEMA_VERSION,
        "terrain_sampling_method": TERRAIN_SAMPLING_METHOD,
        "terrain_aspect_convention": TERRAIN_ASPECT_CONVENTION,
        "terrain_source_block_id": block.source_block_id,
        "terrain_source_artifact": block.relative_path,
        "terrain_source_pixel_row": sample.row,
        "terrain_source_pixel_column": sample.column,
        "terrain_source_resolution_degrees": block.pixel_width_degrees,
    }


def _unavailable_features(*, status: str, **metadata: object) -> dict[str, object]:
    return {
        "terrain_valid": False,
        "terrain_elevation_m": None,
        "terrain_slope_degrees": None,
        "terrain_aspect_sin": None,
        "terrain_aspect_cos": None,
        "terrain_aspect_defined": False,
        "terrain_coverage_status": status,
        "terrain_feature_schema_version": TERRAIN_FEATURE_SCHEMA_VERSION,
        "terrain_sampling_method": TERRAIN_SAMPLING_METHOD,
        "terrain_aspect_convention": TERRAIN_ASPECT_CONVENTION,
        "terrain_source_block_id": None,
        "terrain_source_artifact": None,
        "terrain_source_pixel_row": None,
        "terrain_source_pixel_column": None,
        "terrain_source_resolution_degrees": None,
        **metadata,
    }


def _snap_grid_ratio(value: float) -> float:
    nearest = round(value)
    if abs(value - nearest) <= _GRID_EDGE_TOLERANCE_PIXELS:
        return float(nearest)
    return value


def _finite_scalar(value: np.ndarray, label: str, *, path: Path) -> float:
    if value.shape != ():
        raise TerrainFeatureError(f"terrain artifact {path} {label} must be a scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise TerrainFeatureError(f"terrain artifact {path} {label} must be finite")
    return resolved


def _finite_coordinate(value: float, label: str, *, minimum: float, maximum: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TerrainFeatureError(f"{label} must be numeric") from exc
    if not math.isfinite(resolved) or not minimum <= resolved <= maximum:
        raise TerrainFeatureError(f"{label} must be finite and between {minimum} and {maximum}")
    return resolved


def _relative_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
