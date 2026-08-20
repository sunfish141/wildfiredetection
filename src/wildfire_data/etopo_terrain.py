"""Bounded ETOPO 2022 terrain collection for wildfire-context tiles.

ETOPO is a static source, so it is safe to select its coverage from the full
training-period FIRMS footprint.  This adapter keeps every requested NOAA
subset as immutable raw evidence, then stores a compact terrain block with
elevation, slope, and aspect.  It deliberately does not infer fuel, land
cover, or water classes from elevation alone.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .data_archive import CoverageLedger, CoverageRecord, CoverageStatus, RawArtifact, write_raw_artifact
from .normalized_storage import NormalizedArtifact, write_normalized_jsonl
from .storage_budget import StorageBudgetError, StorageBudgetPolicy, require_admission


ETOPO_EXPORT_URL = "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/ImageServer/exportImage"
ETOPO_DATASET = "ETOPO_2022_v1_15s_surface_elev"
ETOPO_DATASET_VERSION = "ETOPO 2022 v1"
ETOPO_DOI = "10.25921/fd45-gt74"
ETOPO_SOURCE_DATE = "2022-10-01"
ETOPO_RESOLUTION_DEGREES = 1.0 / 240.0
ETOPO_SOURCE = "NOAA NCEI:etopo terrain"
DEFAULT_CONTEXT_TILE_KILOMETRES = 96.0
DEFAULT_SOURCE_BLOCK_DEGREES = 10.0
DEFAULT_HALO_PIXELS = 1
STATIC_TERRAIN_NORMALIZATION_VERSION = "etopo-2022-15s-terrain/v1"
STATIC_RETENTION_PRIORITY_SCORE = 70
STATIC_SOURCE_QUALITY_SCORE = 0.8
NO_DATA_ELEVATION_METRES = np.int16(-32768)
NO_DATA_ASPECT_DEGREES_X2 = np.uint8(255)

_WEB_MERCATOR_TILE_PATTERN = re.compile(
    r"^webmercator-(?P<kilometres>[0-9]+(?:\.[0-9]+)?)km-x(?P<x>-?[0-9]+)-y(?P<y>-?[0-9]+)$"
)


class EtopoTerrainError(RuntimeError):
    """Raised when a terrain source response cannot be preserved correctly."""


@dataclass(frozen=True)
class EtopoSourceBlock:
    """One NOAA subset request covering one or more FIRMS-context tiles."""

    block_id: str
    x_index: int
    y_index: int
    west: float
    south: float
    east: float
    north: float
    request_west: float
    request_south: float
    request_east: float
    request_north: float
    width: int
    height: int
    halo_pixels: int
    context_tile_ids: tuple[str, ...]

    @property
    def cell_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class DecodedEtopoRaster:
    """A validated ETOPO raster and its actual GeoTIFF grid transform."""

    elevation_m: np.ndarray
    west: float
    north: float
    pixel_width_degrees: float
    pixel_height_degrees: float


@dataclass(frozen=True)
class TerrainFeatures:
    """Quantized static terrain inputs aligned to one ETOPO source block."""

    elevation_m: np.ndarray
    slope_degrees_x2: np.ndarray
    aspect_degrees_x2: np.ndarray


@dataclass(frozen=True)
class EtopoBlockCollection:
    """Durable result for one ETOPO source block."""

    block: EtopoSourceBlock
    raw_artifact: RawArtifact | None
    normalized_artifact: NormalizedArtifact | None
    static_artifact_path: Path | None
    coverage: CoverageRecord
    skipped_terminal_coverage: bool = False


@dataclass(frozen=True)
class EtopoTerrainCollection:
    """All source-block outcomes for a static terrain collection run."""

    blocks: tuple[EtopoBlockCollection, ...]
    context_tile_count: int

    @property
    def complete_count(self) -> int:
        return sum(block.coverage.status == CoverageStatus.COMPLETE for block in self.blocks)

    @property
    def skipped_count(self) -> int:
        return sum(block.skipped_terminal_coverage for block in self.blocks)

    @property
    def partial_or_failed_count(self) -> int:
        return sum(
            block.coverage.status not in {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED}
            for block in self.blocks
        )


def webmercator_context_tile_id(
    latitude: float,
    longitude: float,
    *,
    tile_kilometres: float = DEFAULT_CONTEXT_TILE_KILOMETRES,
) -> str:
    """Return the deterministic planning-tile ID for a WGS84 coordinate."""
    if not math.isfinite(latitude) or not -85.0 < latitude < 85.0:
        raise ValueError("latitude must be finite and between -85 and 85")
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ValueError("longitude must be finite and between -180 and 180")
    if not math.isfinite(tile_kilometres) or tile_kilometres <= 0:
        raise ValueError("tile_kilometres must be positive")
    tile_metres = tile_kilometres * 1_000.0
    radius = 6_378_137.0
    x = radius * math.radians(longitude)
    y = radius * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0))
    return f"webmercator-{tile_kilometres:g}km-x{math.floor(x / tile_metres)}-y{math.floor(y / tile_metres)}"


def webmercator_tile_bounds(tile_id: str) -> tuple[float, float, float, float]:
    """Return ``west, south, east, north`` WGS84 bounds for a planning tile."""
    match = _WEB_MERCATOR_TILE_PATTERN.fullmatch(tile_id)
    if match is None:
        raise ValueError("tile_id must be a webmercator-<km>km-x<index>-y<index> value")
    tile_kilometres = float(match.group("kilometres"))
    tile_x = int(match.group("x"))
    tile_y = int(match.group("y"))
    tile_metres = tile_kilometres * 1_000.0
    radius = 6_378_137.0
    west = math.degrees((tile_x * tile_metres) / radius)
    east = math.degrees(((tile_x + 1) * tile_metres) / radius)
    south = math.degrees(2.0 * math.atan(math.exp((tile_y * tile_metres) / radius)) - math.pi / 2.0)
    north = math.degrees(
        2.0 * math.atan(math.exp(((tile_y + 1) * tile_metres) / radius)) - math.pi / 2.0
    )
    return west, south, east, north


def context_tile_ids_from_detections(detections: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    """Select every unique 96 km static context tile represented by FIRMS."""
    tile_ids = set()
    for detection in detections:
        if not isinstance(detection, Mapping):
            raise ValueError("detections must contain mappings")
        try:
            latitude = float(detection["latitude"])
            longitude = float(detection["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every detection needs numeric latitude and longitude") from exc
        tile_ids.add(webmercator_context_tile_id(latitude, longitude))
    return tuple(sorted(tile_ids))


def etopo_source_blocks(
    context_tile_ids: Iterable[str],
    *,
    block_degrees: float = DEFAULT_SOURCE_BLOCK_DEGREES,
    halo_pixels: int = DEFAULT_HALO_PIXELS,
) -> tuple[EtopoSourceBlock, ...]:
    """Group static-context tiles into bounded WGS84 ETOPO subset requests."""
    _validate_block_degrees(block_degrees)
    if not isinstance(halo_pixels, int) or isinstance(halo_pixels, bool) or halo_pixels < 0:
        raise ValueError("halo_pixels must be a non-negative integer")

    grouped: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    for tile_id in sorted(set(context_tile_ids)):
        west, south, east, north = webmercator_tile_bounds(tile_id)
        if west < -180.0 or east > 180.0 or south < -90.0 or north > 90.0:
            raise ValueError(f"context tile is outside ETOPO WGS84 bounds: {tile_id}")
        x_start = _block_index(west, origin=-180.0, block_degrees=block_degrees)
        x_end = _block_index(math.nextafter(east, -math.inf), origin=-180.0, block_degrees=block_degrees)
        y_start = _block_index(south, origin=-90.0, block_degrees=block_degrees)
        y_end = _block_index(math.nextafter(north, -math.inf), origin=-90.0, block_degrees=block_degrees)
        for x_index in range(x_start, x_end + 1):
            for y_index in range(y_start, y_end + 1):
                grouped[(x_index, y_index)].add(tile_id)

    blocks = []
    for (x_index, y_index), tile_ids in sorted(grouped.items()):
        west = -180.0 + x_index * block_degrees
        east = west + block_degrees
        south = -90.0 + y_index * block_degrees
        north = south + block_degrees
        request_west = max(-180.0, west - halo_pixels * ETOPO_RESOLUTION_DEGREES)
        request_east = min(180.0, east + halo_pixels * ETOPO_RESOLUTION_DEGREES)
        request_south = max(-90.0, south - halo_pixels * ETOPO_RESOLUTION_DEGREES)
        request_north = min(90.0, north + halo_pixels * ETOPO_RESOLUTION_DEGREES)
        width = _grid_dimension(request_west, request_east)
        height = _grid_dimension(request_south, request_north)
        blocks.append(
            EtopoSourceBlock(
                block_id=f"etopo-2022-15s-wgs84-{block_degrees:g}deg-x{x_index}-y{y_index}",
                x_index=x_index,
                y_index=y_index,
                west=west,
                south=south,
                east=east,
                north=north,
                request_west=request_west,
                request_south=request_south,
                request_east=request_east,
                request_north=request_north,
                width=width,
                height=height,
                halo_pixels=halo_pixels,
                context_tile_ids=tuple(sorted(tile_ids)),
            )
        )
    return tuple(blocks)


def etopo_request_parameters(block: EtopoSourceBlock) -> dict[str, str]:
    """Build the exact deterministic NOAA ArcGIS ImageServer request."""
    return {
        "bbox": ",".join(
            f"{value:.12f}"
            for value in (block.request_west, block.request_south, block.request_east, block.request_north)
        ),
        "bboxSR": "4326",
        "size": f"{block.width},{block.height}",
        "imageSR": "4326",
        "format": "tiff",
        "pixelType": "S16",
        "interpolation": "RSP_NearestNeighbor",
        "compression": "LZ77",
        "renderingRule": json.dumps({"rasterFunction": "none"}, separators=(",", ":")),
        "mosaicRule": json.dumps(
            {"where": f"Name='{ETOPO_DATASET}'"}, separators=(",", ":")
        ),
        "f": "image",
    }


def collect_etopo_terrain(
    archive_root: str | Path,
    *,
    context_tile_ids: Iterable[str],
    storage_budget: StorageBudgetPolicy,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (15, 300),
    block_degrees: float = DEFAULT_SOURCE_BLOCK_DEGREES,
    halo_pixels: int = DEFAULT_HALO_PIXELS,
    retrieved_at: datetime | None = None,
) -> EtopoTerrainCollection:
    """Collect quota-admitted ETOPO terrain blocks for FIRMS-context tiles.

    Each persisted block has its raw ImageServer response, a compact NPZ
    feature cube, a normalized provenance record, and a coverage-ledger entry.
    A completed block is idempotently skipped on later runs.
    """
    root = Path(archive_root)
    tile_ids = tuple(sorted(set(context_tile_ids)))
    blocks = etopo_source_blocks(
        tile_ids,
        block_degrees=block_degrees,
        halo_pixels=halo_pixels,
    )
    retrieved = _utc_now_or_value(retrieved_at)
    ledger = CoverageLedger(root)
    latest_by_expected_id = {
        entry.expected_coverage_id: entry
        for entry in ledger.entries()
        if entry.expected_coverage_id is not None
    }
    active_session = session or _retrying_session()
    owns_session = session is None
    results = []
    try:
        for block in blocks:
            expected_id = _expected_coverage_id(block)
            prior = latest_by_expected_id.get(expected_id)
            static_path = _static_artifact_path(root, block)
            if (
                prior is not None
                and prior.status in {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED}
                and static_path.exists()
            ):
                results.append(
                    EtopoBlockCollection(
                        block=block,
                        raw_artifact=None,
                        normalized_artifact=None,
                        static_artifact_path=static_path,
                        coverage=prior,
                        skipped_terminal_coverage=True,
                    )
                )
                continue

            estimated_bytes = _conservative_block_bytes(block)
            try:
                require_admission(
                    storage_budget,
                    root,
                    category="static_cell_features",
                    requested_bytes=estimated_bytes,
                )
            except StorageBudgetError as exc:
                coverage = _record_coverage(
                    ledger,
                    block=block,
                    expected_id=expected_id,
                    status=CoverageStatus.PARTIAL,
                    retrieved_at=retrieved,
                    error=str(exc),
                    detail={
                        "stage": "storage_admission_before_request",
                        "estimated_persisted_bytes": estimated_bytes,
                    },
                )
                results.append(EtopoBlockCollection(block, None, None, None, coverage))
                break

            parameters = etopo_request_parameters(block)
            source_uri = requests.Request("GET", ETOPO_EXPORT_URL, params=parameters).prepare().url
            try:
                response = active_session.get(
                    ETOPO_EXPORT_URL,
                    params=parameters,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                coverage = _record_coverage(
                    ledger,
                    block=block,
                    expected_id=expected_id,
                    status=CoverageStatus.FAILED,
                    retrieved_at=retrieved,
                    error=f"ETOPO request failed: {exc}",
                    detail={"stage": "request", "source_uri": source_uri},
                )
                results.append(EtopoBlockCollection(block, None, None, None, coverage))
                continue

            payload = bytes(response.content)
            actual_estimate = len(payload) + _derived_block_bytes(block) + 1_048_576
            try:
                require_admission(
                    storage_budget,
                    root,
                    category="static_cell_features",
                    requested_bytes=actual_estimate,
                )
            except StorageBudgetError as exc:
                coverage = _record_coverage(
                    ledger,
                    block=block,
                    expected_id=expected_id,
                    status=CoverageStatus.PARTIAL,
                    retrieved_at=retrieved,
                    error=str(exc),
                    detail={
                        "stage": "storage_admission_after_response",
                        "response_bytes_not_persisted": len(payload),
                        "estimated_persisted_bytes": actual_estimate,
                    },
                )
                results.append(EtopoBlockCollection(block, None, None, None, coverage))
                break

            raw_artifact = write_raw_artifact(
                root,
                source=ETOPO_SOURCE,
                payload=payload,
                retrieved_at=retrieved,
                media_type=_response_media_type(response.headers),
                provenance={
                    "source_url": source_uri,
                    "request_parameters": parameters,
                    "response_headers": dict(response.headers),
                    "response_status_code": response.status_code,
                    "provider": "NOAA NCEI",
                    "dataset": ETOPO_DATASET,
                    "dataset_version": ETOPO_DATASET_VERSION,
                    "dataset_doi": ETOPO_DOI,
                    "asset_role": "terrain_elevation_source_subset",
                    "block_id": block.block_id,
                },
            )
            if not 200 <= response.status_code < 300:
                coverage = _record_coverage(
                    ledger,
                    block=block,
                    expected_id=expected_id,
                    status=CoverageStatus.FAILED,
                    retrieved_at=retrieved,
                    artifact_sha256s=[raw_artifact.raw_artifact_id],
                    error=f"ETOPO returned HTTP {response.status_code}",
                    detail={"stage": "response", "source_uri": source_uri},
                )
                results.append(EtopoBlockCollection(block, raw_artifact, None, None, coverage))
                continue

            try:
                decoded = decode_etopo_tiff(payload, block=block)
                features = derive_terrain_features(
                    decoded.elevation_m,
                    north_latitude=decoded.north,
                    pixel_width_degrees=decoded.pixel_width_degrees,
                    pixel_height_degrees=decoded.pixel_height_degrees,
                )
                static_path, static_sha256, static_bytes = write_terrain_features(
                    root,
                    block=block,
                    features=features,
                    grid_west=decoded.west,
                    grid_north=decoded.north,
                    pixel_width_degrees=decoded.pixel_width_degrees,
                    pixel_height_degrees=decoded.pixel_height_degrees,
                )
                normalized_artifact = write_normalized_jsonl(
                    root,
                    entity="static_cell_features",
                    records=[
                        _normalized_record(
                            root,
                            block=block,
                            raw_artifact=raw_artifact,
                            normalized_path=static_path,
                            static_sha256=static_sha256,
                            static_bytes=static_bytes,
                            decoded=decoded,
                        )
                    ],
                    partitions={"dataset": "etopo-2022-15s", "block": block.block_id},
                    raw_artifact_ids=[raw_artifact.raw_artifact_id],
                    transformation_version=STATIC_TERRAIN_NORMALIZATION_VERSION,
                    generated_at=retrieved,
                )
            except (EtopoTerrainError, OSError, ValueError) as exc:
                coverage = _record_coverage(
                    ledger,
                    block=block,
                    expected_id=expected_id,
                    status=CoverageStatus.FAILED,
                    retrieved_at=retrieved,
                    artifact_sha256s=[raw_artifact.raw_artifact_id],
                    error=str(exc),
                    detail={"stage": "decode_or_compact"},
                )
                results.append(EtopoBlockCollection(block, raw_artifact, None, None, coverage))
                continue

            coverage = _record_coverage(
                ledger,
                block=block,
                expected_id=expected_id,
                status=CoverageStatus.COMPLETE,
                retrieved_at=retrieved,
                artifact_sha256s=[raw_artifact.raw_artifact_id],
                detail={
                    "static_artifact": static_path.relative_to(root).as_posix(),
                    "static_artifact_sha256": static_sha256,
                    "static_artifact_bytes": static_bytes,
                    "normalized_artifact_id": normalized_artifact.normalized_artifact_id,
                    "source_grid": {
                        "west": decoded.west,
                        "north": decoded.north,
                        "pixel_width_degrees": decoded.pixel_width_degrees,
                        "pixel_height_degrees": decoded.pixel_height_degrees,
                        "width": block.width,
                        "height": block.height,
                    },
                },
            )
            results.append(
                EtopoBlockCollection(
                    block,
                    raw_artifact,
                    normalized_artifact,
                    static_path,
                    coverage,
                )
            )
    finally:
        if owns_session:
            active_session.close()
    return EtopoTerrainCollection(tuple(results), len(tile_ids))


def decode_etopo_tiff(payload: bytes, *, block: EtopoSourceBlock) -> DecodedEtopoRaster:
    """Decode and validate an exact NOAA S16 GeoTIFF source response."""
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            array = np.asarray(image)
            pixel_scale = image.tag_v2.get(33550)
            tiepoint = image.tag_v2.get(33922)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise EtopoTerrainError("ETOPO response is not a readable GeoTIFF") from exc
    if array.ndim != 2 or array.shape != (block.height, block.width):
        raise EtopoTerrainError(
            f"ETOPO raster shape {array.shape!r} does not match requested {(block.height, block.width)!r}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise EtopoTerrainError("ETOPO raster must contain finite numeric elevation values")
    rounded = np.rint(array)
    if np.max(np.abs(array - rounded)) > 0.01:
        raise EtopoTerrainError("ETOPO response ignored the requested one-metre S16 pixel type")
    if rounded.min() < np.iinfo(np.int16).min or rounded.max() > np.iinfo(np.int16).max:
        raise EtopoTerrainError("ETOPO elevation values cannot fit the compact int16 representation")
    scale_values = _numeric_tag(pixel_scale, label="ModelPixelScaleTag", minimum_length=2)
    tiepoint_values = _numeric_tag(tiepoint, label="ModelTiepointTag", minimum_length=5)
    pixel_width = abs(scale_values[0])
    pixel_height = abs(scale_values[1])
    if not _approximately_etopo_resolution(pixel_width) or not _approximately_etopo_resolution(pixel_height):
        raise EtopoTerrainError("ETOPO response did not use the expected 15 arc-second grid")
    west = tiepoint_values[3]
    north = tiepoint_values[4]
    if not -180.1 <= west <= 180.1 or not -90.1 <= north <= 90.1:
        raise EtopoTerrainError("ETOPO GeoTIFF has invalid WGS84 grid coordinates")
    return DecodedEtopoRaster(
        elevation_m=np.ascontiguousarray(rounded.astype(np.int16)),
        west=west,
        north=north,
        pixel_width_degrees=pixel_width,
        pixel_height_degrees=pixel_height,
    )


def derive_terrain_features(
    elevation_m: np.ndarray,
    *,
    north_latitude: float,
    pixel_width_degrees: float,
    pixel_height_degrees: float,
) -> TerrainFeatures:
    """Derive quantized slope and downhill aspect using geographic distances.

    The input grid is north-up WGS84.  Slope is stored at 0.5-degree
    resolution; aspect is the downhill direction clockwise from north at
    2-degree resolution, with 255 for flat or unavailable cells.
    """
    if elevation_m.ndim != 2 or min(elevation_m.shape) < 2:
        raise ValueError("elevation_m must be a two-dimensional grid at least two cells wide and high")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (pixel_width_degrees, pixel_height_degrees)
    ):
        raise ValueError("pixel dimensions must be finite positive degree values")
    if not math.isfinite(north_latitude) or not -90.0 <= north_latitude <= 90.0:
        raise ValueError("north_latitude must be a finite WGS84 latitude")

    values = elevation_m.astype(np.float32, copy=False)
    valid = values != float(NO_DATA_ELEVATION_METRES)
    working = np.where(valid, values, np.nan)
    earth_radius_metres = 6_371_008.8
    row_indices = np.arange(values.shape[0], dtype=np.float32)
    latitude = north_latitude - (row_indices + 0.5) * pixel_height_degrees
    metres_per_x = (
        earth_radius_metres
        * np.cos(np.deg2rad(latitude))
        * math.radians(pixel_width_degrees)
    )
    metres_per_y = earth_radius_metres * math.radians(pixel_height_degrees)
    if np.any(metres_per_x <= 0.0):
        raise ValueError("terrain grid must not include a pole")
    with np.errstate(invalid="ignore", divide="ignore"):
        rise_east = np.gradient(working, axis=1) / metres_per_x[:, np.newaxis]
        rise_north = -np.gradient(working, axis=0) / metres_per_y
        slope_degrees = np.degrees(np.arctan(np.hypot(rise_east, rise_north)))
        downhill_aspect = np.mod(np.degrees(np.arctan2(-rise_east, -rise_north)), 360.0)

    calculable = valid & np.isfinite(slope_degrees) & np.isfinite(downhill_aspect)
    slope_degrees_x2 = np.zeros(values.shape, dtype=np.uint8)
    slope_degrees_x2[calculable] = np.clip(
        np.rint(slope_degrees[calculable] * 2.0), 0, 180
    ).astype(np.uint8)
    aspect_degrees_x2 = np.full(values.shape, NO_DATA_ASPECT_DEGREES_X2, dtype=np.uint8)
    non_flat = calculable & (slope_degrees >= 0.05)
    aspect_degrees_x2[non_flat] = np.mod(
        np.rint(downhill_aspect[non_flat] / 2.0), 180
    ).astype(np.uint8)
    return TerrainFeatures(
        elevation_m=np.ascontiguousarray(elevation_m.astype(np.int16, copy=False)),
        slope_degrees_x2=slope_degrees_x2,
        aspect_degrees_x2=aspect_degrees_x2,
    )


def write_terrain_features(
    archive_root: str | Path,
    *,
    block: EtopoSourceBlock,
    features: TerrainFeatures,
    grid_west: float,
    grid_north: float,
    pixel_width_degrees: float,
    pixel_height_degrees: float,
) -> tuple[Path, str, int]:
    """Write an immutable compressed feature cube and return its identity."""
    root = Path(archive_root)
    target = _static_artifact_path(root, block)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = root / "staging" / "etopo-terrain"
    staging.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{block.block_id}.", suffix=".npz", dir=staging
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez_compressed(
                output,
                elevation_m=features.elevation_m,
                slope_degrees_x2=features.slope_degrees_x2,
                aspect_degrees_x2=features.aspect_degrees_x2,
                grid_west=np.float64(grid_west),
                grid_north=np.float64(grid_north),
                pixel_width_degrees=np.float64(pixel_width_degrees),
                pixel_height_degrees=np.float64(pixel_height_degrees),
            )
            output.flush()
            os.fsync(output.fileno())
        content_sha256, byte_count = _file_digest(temporary_path)
        try:
            os.link(temporary_path, target)
        except FileExistsError:
            existing_sha256, existing_size = _file_digest(target)
            if existing_sha256 != content_sha256:
                raise EtopoTerrainError(f"Refusing to overwrite static terrain block: {target}")
            return target, existing_sha256, existing_size
        return target, content_sha256, byte_count
    finally:
        temporary_path.unlink(missing_ok=True)


def _normalized_record(
    root: Path,
    *,
    block: EtopoSourceBlock,
    raw_artifact: RawArtifact,
    normalized_path: Path,
    static_sha256: str,
    static_bytes: int,
    decoded: DecodedEtopoRaster,
) -> dict[str, object]:
    return {
        "record_type": "static_terrain_block",
        "source": "NOAA NCEI ETOPO 2022",
        "dataset": ETOPO_DATASET,
        "dataset_version": ETOPO_DATASET_VERSION,
        "dataset_doi": ETOPO_DOI,
        "source_product_date": ETOPO_SOURCE_DATE,
        "vertical_datum": "EGM2008 geoid",
        "grid_crs": "EPSG:4326",
        "grid_resolution_arc_seconds": 15,
        "grid_west": decoded.west,
        "grid_north": decoded.north,
        "pixel_width_degrees": decoded.pixel_width_degrees,
        "pixel_height_degrees": decoded.pixel_height_degrees,
        "grid_width": block.width,
        "grid_height": block.height,
        "source_block_id": block.block_id,
        "source_block_bounds_wgs84": {
            "west": block.west,
            "south": block.south,
            "east": block.east,
            "north": block.north,
        },
        "request_bounds_wgs84": {
            "west": block.request_west,
            "south": block.request_south,
            "east": block.request_east,
            "north": block.request_north,
        },
        "halo_pixels": block.halo_pixels,
        "context_tile_ids": list(block.context_tile_ids),
        "raw_artifact_id": raw_artifact.raw_artifact_id,
        "static_artifact": {
            "relative_path": normalized_path.relative_to(root).as_posix(),
            "content_sha256": static_sha256,
            "bytes": static_bytes,
            "format": "npz",
            "arrays": {
                "elevation_m": "int16 metres; -32768 is unavailable",
                "slope_degrees_x2": "uint8; degrees times two",
                "aspect_degrees_x2": "uint8; downhill clockwise-from-north degrees divided by two; 255 is undefined",
            },
        },
        "slope_aspect_algorithm": "latitude-aware finite differences on the ETOPO north-up 15 arc-second grid",
        "static_source_quality_score": STATIC_SOURCE_QUALITY_SCORE,
        "retention_priority_score": STATIC_RETENTION_PRIORITY_SCORE,
    }


def _record_coverage(
    ledger: CoverageLedger,
    *,
    block: EtopoSourceBlock,
    expected_id: str,
    status: CoverageStatus,
    retrieved_at: datetime,
    artifact_sha256s: Iterable[str] = (),
    error: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> CoverageRecord:
    return ledger.record(
        source="NOAA NCEI",
        product="ETOPO_2022_15s_surface_terrain",
        coverage_start=ETOPO_SOURCE_DATE,
        coverage_end=ETOPO_SOURCE_DATE,
        region="United States and Canada",
        tile=block.block_id,
        expected_coverage_id=expected_id,
        status=status,
        artifact_sha256s=artifact_sha256s,
        error=error,
        detail={
            "dataset": ETOPO_DATASET,
            "context_tile_count": len(block.context_tile_ids),
            "request_shape": [block.height, block.width],
            **dict(detail or {}),
        },
        recorded_at=retrieved_at,
    )


def _expected_coverage_id(block: EtopoSourceBlock) -> str:
    return ":".join(
        (
            "etopo",
            "2022-v1",
            "15arcsec",
            "surface",
            block.block_id,
            f"halo{block.halo_pixels}",
        )
    )


def _static_artifact_path(root: Path, block: EtopoSourceBlock) -> Path:
    return root / "static" / "etopo-2022-15s" / f"{block.block_id}.npz"


def _conservative_block_bytes(block: EtopoSourceBlock) -> int:
    # S16 GeoTIFF source response plus int16 elevation and two uint8 derived
    # arrays.  The temporary feature cube is held under the staging headroom.
    return block.cell_count * 6 + 1_048_576


def _derived_block_bytes(block: EtopoSourceBlock) -> int:
    return block.cell_count * 4 + 65_536


def _retrying_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def _utc_now_or_value(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return resolved.astimezone(timezone.utc)


def _response_media_type(headers: Mapping[str, Any]) -> str:
    value = headers.get("Content-Type") or headers.get("content-type")
    if isinstance(value, str) and value.strip():
        return value.split(";", 1)[0].strip()
    return "image/tiff"


def _numeric_tag(value: object, *, label: str, minimum_length: int) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) < minimum_length:
        raise EtopoTerrainError(f"ETOPO GeoTIFF is missing {label}")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise EtopoTerrainError(f"ETOPO GeoTIFF has invalid {label}") from exc
    if not all(math.isfinite(item) for item in values):
        raise EtopoTerrainError(f"ETOPO GeoTIFF has non-finite {label}")
    return values


def _approximately_etopo_resolution(value: float) -> bool:
    return abs(value - ETOPO_RESOLUTION_DEGREES) <= 2.0e-5


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _validate_block_degrees(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("block_degrees must be finite and positive")
    horizontal = 360.0 / value
    vertical = 180.0 / value
    if not math.isclose(horizontal, round(horizontal), abs_tol=1.0e-9) or not math.isclose(
        vertical, round(vertical), abs_tol=1.0e-9
    ):
        raise ValueError("block_degrees must divide both 360 and 180 exactly")


def _block_index(value: float, *, origin: float, block_degrees: float) -> int:
    return int(math.floor((value - origin) / block_degrees))


def _grid_dimension(lower: float, upper: float) -> int:
    dimension = round((upper - lower) / ETOPO_RESOLUTION_DEGREES)
    if dimension <= 0:
        raise ValueError("ETOPO request bounds must have a positive dimension")
    return dimension
