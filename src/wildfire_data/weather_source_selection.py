"""Select input locations that cover nearby points."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree


WGS84_SEMI_MAJOR_AXIS_M = 6_378_137.0
WGS84_FLATTENING = 1 / 298.257223563


@dataclass(frozen=True)
class SpatialCover:
    """A cover using source points selected from the input array."""

    source_indices: np.ndarray
    assigned_source_indices: np.ndarray
    distances_m: np.ndarray


def lat_lon_to_ecef(points: np.ndarray, altitude_m: float = 0.0) -> np.ndarray:
    """Convert ``(longitude, latitude)`` surface points to WGS84 ECEF metres."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2) as (longitude, latitude)")

    longitude = np.deg2rad(points[:, 0])
    latitude = np.deg2rad(points[:, 1])
    eccentricity_squared = WGS84_FLATTENING * (2 - WGS84_FLATTENING)
    prime_vertical_radius = WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
        1 - eccentricity_squared * np.sin(latitude) ** 2
    )

    x = (prime_vertical_radius + altitude_m) * np.cos(latitude) * np.cos(longitude)
    y = (prime_vertical_radius + altitude_m) * np.cos(latitude) * np.sin(longitude)
    z = (prime_vertical_radius * (1 - eccentricity_squared) + altitude_m) * np.sin(latitude)
    return np.column_stack((x, y, z))


def minimum_covering_sources(points: np.ndarray, radius_m: float = 1_000.0) -> SpatialCover:
    """Return a deterministic subset of ``points`` covering every input point.

    Candidate sources are restricted to the supplied ``(longitude, latitude)``
    points.  Every input point is assigned to a selected source no farther than
    ``radius_m`` in WGS84 ECEF chord distance.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2) as (longitude, latitude)")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite longitude and latitude values")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if len(points) == 0:
        empty = np.array([], dtype=int)
        return SpatialCover(empty, empty, np.array([], dtype=float))

    unique_points, first_indices, point_to_unique = np.unique(
        points, axis=0, return_index=True, return_inverse=True
    )
    ecef_points = lat_lon_to_ecef(unique_points)
    tree = KDTree(ecef_points)
    covered = np.zeros(len(unique_points), dtype=bool)
    selected_unique_indices = []
    # ``np.unique`` gives this sweep a stable coordinate order without materializing
    # every pair of nearby points or solving an unbounded optimization problem.
    for unique_index in range(len(unique_points)):
        if covered[unique_index]:
            continue
        selected_unique_indices.append(unique_index)
        covered[tree.query_ball_point(ecef_points[unique_index], r=radius_m)] = True

    selected_unique_indices = np.asarray(selected_unique_indices, dtype=int)
    source_tree = KDTree(ecef_points[selected_unique_indices])
    unique_distances_m, assigned_selected_positions = source_tree.query(
        ecef_points, k=1, workers=-1
    )
    if np.any(unique_distances_m > radius_m + 1e-6):
        raise RuntimeError("The selected weather sources do not cover every input point.")

    source_indices = first_indices[selected_unique_indices]
    assigned_unique_source_indices = source_indices[assigned_selected_positions]
    return SpatialCover(
        source_indices=source_indices,
        assigned_source_indices=assigned_unique_source_indices[point_to_unique],
        distances_m=unique_distances_m[point_to_unique],
    )
