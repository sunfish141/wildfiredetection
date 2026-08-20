"""Versioned collection targets for the U.S. and Canadian fire dataset.

The catalog intentionally describes sources without embedding credentials or
executing network calls.  A scheduler can use it to create coverage-ledger
entries and snapshot each source into the immutable archive.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollectionTarget:
    """A source to snapshot with its intended cadence and semantics."""

    key: str
    provider: str
    entity: str
    region: str
    cadence_minutes: int
    label_tier: str | None
    description: str


COLLECTION_TARGETS = (
    CollectionTarget(
        "firms_viirs_hotspots",
        "NASA FIRMS",
        "fire_detection",
        "United States and Canada",
        60,
        None,
        "Unfiltered raw hotspot responses from every configured VIIRS platform/product.",
    ),
    CollectionTarget(
        "viirs_l2_observability",
        "NASA LP DAAC",
        "satellite_observation_mask",
        "United States and Canada",
        60,
        None,
        "VIIRS fire-mask, geolocation, and quality granules for clear/unknown labels.",
    ),
    CollectionTarget(
        "wfigs_current_perimeters",
        "NIFC WFIGS",
        "operational_perimeter",
        "United States",
        15,
        "operational",
        "Versioned operational perimeter geometries and their source timestamps.",
    ),
    CollectionTarget(
        "irwin_incidents",
        "NIFC IRWIN",
        "incident_snapshot",
        "United States",
        15,
        "operational",
        "Incident identifiers, status, containment, and source update times.",
    ),
    CollectionTarget(
        "feds_nrt",
        "NASA FEDS",
        "fire_progression",
        "United States and Canada",
        720,
        "weak_satellite",
        "NRT perimeter, active-front, and new-fire-pixel snapshots.",
    ),
    CollectionTarget(
        "cwfis_active_fires",
        "CWFIS",
        "incident_snapshot",
        "Canada",
        60,
        "operational",
        "Agency-reported active-fire snapshots.",
    ),
    CollectionTarget(
        "cwfis_fire_m3",
        "CWFIS",
        "fire_progression",
        "Canada",
        60,
        "weak_satellite",
        "Modeled Canadian perimeter estimates retained as a distinct label tier.",
    ),
    CollectionTarget(
        "hrrr_forecasts",
        "NOAA",
        "forecast_weather",
        "United States",
        60,
        None,
        "Issued HRRR forecast runs, including wind vectors and forecast horizons.",
    ),
    CollectionTarget(
        "hrdps_forecasts",
        "ECCC",
        "forecast_weather",
        "Canada",
        360,
        None,
        "Issued HRDPS forecast runs, including wind vectors and forecast horizons.",
    ),
    CollectionTarget(
        "static_spatial_assets",
        "Versioned public spatial sources",
        "static_spatial_asset",
        "United States and Canada",
        525_600,
        None,
        "Terrain, fuel, vegetation, water, access, and administrative source releases.",
    ),
)


def targets_for_entity(entity: str) -> tuple[CollectionTarget, ...]:
    """Return catalog targets for a canonical collection entity."""
    return tuple(target for target in COLLECTION_TARGETS if target.entity == entity)


def target_by_key(key: str) -> CollectionTarget:
    """Return one configured source target by its stable catalog key."""
    matches = [target for target in COLLECTION_TARGETS if target.key == key]
    if len(matches) != 1:
        raise ValueError(f"Unknown collection target: {key!r}")
    return matches[0]


def validate_collection_catalog(
    targets: tuple[CollectionTarget, ...] = COLLECTION_TARGETS,
) -> None:
    """Fail early if a catalog edit removes a required collection capability."""
    keys = [target.key for target in targets]
    if len(keys) != len(set(keys)):
        raise ValueError("collection target keys must be unique")
    required_entities = {
        "fire_detection",
        "satellite_observation_mask",
        "operational_perimeter",
        "forecast_weather",
        "static_spatial_asset",
    }
    present_entities = {target.entity for target in targets}
    missing = required_entities.difference(present_entities)
    if missing:
        raise ValueError(f"collection catalog is missing required entities: {sorted(missing)}")
    if any(target.cadence_minutes <= 0 for target in targets):
        raise ValueError("collection target cadence_minutes must be positive")
