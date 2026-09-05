"""A small recursive transition wrapper for the first spread classifier.

The retained tabular baseline predicts whether an unburned candidate cell will
newly burn in the next 12 hours.  It does not itself maintain the fire state
needed by an interactive simulation.  This module supplies that deliberately
simple state transition:

* active cells are rendered as synthetic FIRMS-compatible observations;
* nearby, unburned cells are scored by the fitted tabular classifier;
* cells at or above a fixed probability threshold ignite; and
* active cells decay for a fixed number of 12-hour steps before becoming
  burned cells that cannot ignite again.

Brightness, detection count, intensity decay, and active duration are explicit
heuristics, not learned future-FIRMS targets.  This makes the first recursive
demo reproducible without overstating it as a validated multi-step forecast.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .candidate_dataset import DEFAULT_CANDIDATE_RADIUS_CELLS, DEFAULT_MODEL_FEATURE_COLUMNS
from .training_grid import GridCell, cell_from_id, cell_from_wgs84, cells_in_square_radius


RECURSIVE_TRANSITION_VERSION = "recursive-firms-compatible-transition/v2"
RECURSIVE_MODEL_FEATURE_COLUMNS = tuple(
    name for name in DEFAULT_MODEL_FEATURE_COLUMNS if not name.startswith("firms_center_")
)
# The frontier holdout has a 6.15% positive rate. At 0.05 this first model is
# deliberately spread-sensitive (about 20.3% precision / 90.4% recall) so a
# deterministic interactive rollout does not require a misleading 0.50 cutoff
# on a heavily imbalanced task. Scenarios must retain this configurable value.
DEFAULT_IGNITION_THRESHOLD = 0.05
DEFAULT_ACTIVE_DURATION_STEPS = 2
DEFAULT_INTENSITY_RETENTION = 0.85
SYNTHETIC_BRIGHTNESS_MIN_K = 305.0
SYNTHETIC_BRIGHTNESS_MAX_K = 367.0
OBSERVATION_LOOKBACK_HOURS = 24.0
OBSERVATION_AVAILABILITY_LAG_HOURS = 3.0
# A new ignition represents evidence acquired halfway through the eligible
# portion of the preceding 12-hour step: ages [3, 12], midpoint 7.5 hours.
DEFAULT_NEW_IGNITION_AGE_HOURS = 7.5


class RecursiveTransitionError(ValueError):
    """Raised when a recursive state or model contract is invalid."""


@dataclass(frozen=True)
class SyntheticObservationCalibration:
    """Training-only means in five equal-width intensity bins.

    Platform counts are a proxy for stream diversity, not satellite identities.
    Neighbouring cell proxies combine by maximum; no platform identities or
    new overpasses are inferred.
    """

    detection_count_by_bin: tuple[float, ...]
    platform_count_by_bin: tuple[float, ...]
    training_row_count: int
    training_snapshot_times: tuple[str, ...]
    source_release_manifest_sha256: str
    calibration_version: str = "training-center-intensity-bins/v1"

    def __post_init__(self) -> None:
        if self.calibration_version != "training-center-intensity-bins/v1":
            raise RecursiveTransitionError("unsupported observation calibration version")
        for name, upper in (("detection_count_by_bin", math.inf), ("platform_count_by_bin", 3)):
            values = tuple(_finite(value, name) for value in getattr(self, name))
            if len(values) != 5 or any(value < 1 or value > upper for value in values):
                raise RecursiveTransitionError(f"{name} must contain five valid positive means")
            object.__setattr__(self, name, values)
        if (not isinstance(self.training_row_count, int) or isinstance(self.training_row_count, bool)
                or self.training_row_count < 1 or not self.training_snapshot_times):
            raise RecursiveTransitionError("calibration requires training provenance")
        snapshots = tuple(self.training_snapshot_times)
        if snapshots != tuple(sorted(set(snapshots))):
            raise RecursiveTransitionError("calibration snapshots must be sorted and unique")
        for value in snapshots:
            if datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
                raise RecursiveTransitionError("calibration snapshots must have timezones")
        object.__setattr__(self, "training_snapshot_times", snapshots)
        digest = self.source_release_manifest_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RecursiveTransitionError("calibration requires a release SHA-256")
        if any(p > c for p, c in zip(self.platform_count_by_bin, self.detection_count_by_bin)):
            raise RecursiveTransitionError("platform count cannot exceed detection count")

    def counts(self, intensity: float) -> tuple[int, int]:
        index = min(4, int(_unit_interval(intensity, "intensity") * 5))
        detections = max(1, math.floor(self.detection_count_by_bin[index] + 0.5))
        platforms = min(detections, max(1, math.floor(self.platform_count_by_bin[index] + 0.5)))
        return detections, platforms


@dataclass(frozen=True)
class ActiveFireCell:
    """One active 1 km cell in the recursive simulation state."""

    cell_id: str
    intensity: float
    remaining_active_steps: int
    observation_age_hours: float = DEFAULT_NEW_IGNITION_AGE_HOURS

    def __post_init__(self) -> None:
        cell_from_id(self.cell_id)
        intensity = _unit_interval(self.intensity, "intensity")
        if (
            not isinstance(self.remaining_active_steps, int)
            or isinstance(self.remaining_active_steps, bool)
            or self.remaining_active_steps < 1
        ):
            raise RecursiveTransitionError("remaining_active_steps must be a positive integer")
        object.__setattr__(self, "intensity", intensity)
        age = _finite(self.observation_age_hours, "observation_age_hours")
        if age < 0:
            raise RecursiveTransitionError("observation_age_hours must be non-negative")
        object.__setattr__(self, "observation_age_hours", age)


@dataclass(frozen=True)
class RecursiveFireState:
    """Active and already-burned cells at one 12-hour simulation step."""

    step_index: int
    active_cells: tuple[ActiveFireCell, ...]
    burned_cell_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool) or self.step_index < 0:
            raise RecursiveTransitionError("step_index must be a non-negative integer")
        active_ids = [cell.cell_id for cell in self.active_cells]
        if len(active_ids) != len(set(active_ids)):
            raise RecursiveTransitionError("active_cells must have unique cell IDs")
        for cell_id in self.burned_cell_ids:
            cell_from_id(cell_id)
        if len(self.burned_cell_ids) != len(set(self.burned_cell_ids)):
            raise RecursiveTransitionError("burned_cell_ids must be unique")
        overlap = set(active_ids).intersection(self.burned_cell_ids)
        if overlap:
            raise RecursiveTransitionError("a cell cannot be both active and burned")


@dataclass(frozen=True)
class CellTransitionPrediction:
    """One candidate cell score and the deterministic transition decision."""

    cell_id: str
    latitude: float
    longitude: float
    ignition_probability: float
    will_ignite: bool
    next_intensity: float


@dataclass(frozen=True)
class RecursiveStepResult:
    """The next state plus every candidate probability used to produce it."""

    transition_version: str
    previous_step_index: int
    state: RecursiveFireState
    predictions: tuple[CellTransitionPrediction, ...]


TerrainProvider = Callable[[str], Mapping[str, Any]]


class RecursiveTransitionModel:
    """Render active cells as FIRMS-like features and advance one time step."""

    def __init__(
        self,
        estimator: Any,
        *,
        feature_columns: Sequence[str],
        ignition_threshold: float = DEFAULT_IGNITION_THRESHOLD,
        candidate_radius_cells: int = DEFAULT_CANDIDATE_RADIUS_CELLS,
        active_duration_steps: int = DEFAULT_ACTIVE_DURATION_STEPS,
        intensity_retention: float = DEFAULT_INTENSITY_RETENTION,
        observation_calibration: SyntheticObservationCalibration | None = None,
        new_ignition_age_hours: float = DEFAULT_NEW_IGNITION_AGE_HOURS,
    ) -> None:
        if not callable(getattr(estimator, "predict_proba", None)):
            raise RecursiveTransitionError("estimator must provide predict_proba")
        columns = tuple(feature_columns)
        if columns != RECURSIVE_MODEL_FEATURE_COLUMNS:
            raise RecursiveTransitionError(
                "model feature contract does not match the recursive frontier features"
            )
        if (
            not isinstance(candidate_radius_cells, int)
            or isinstance(candidate_radius_cells, bool)
            or candidate_radius_cells < 1
        ):
            raise RecursiveTransitionError("candidate_radius_cells must be a positive integer")
        if (
            not isinstance(active_duration_steps, int)
            or isinstance(active_duration_steps, bool)
            or active_duration_steps < 1
        ):
            raise RecursiveTransitionError("active_duration_steps must be a positive integer")

        self.estimator = estimator
        self.feature_columns = columns
        self.ignition_threshold = _unit_interval(ignition_threshold, "ignition_threshold")
        self.candidate_radius_cells = candidate_radius_cells
        self.active_duration_steps = active_duration_steps
        self.intensity_retention = _unit_interval(intensity_retention, "intensity_retention")
        if observation_calibration is not None and not isinstance(
            observation_calibration, SyntheticObservationCalibration
        ):
            raise RecursiveTransitionError("observation_calibration must be a calibration contract")
        self.observation_calibration = observation_calibration
        self.new_ignition_age_hours = _finite(new_ignition_age_hours, "new_ignition_age_hours")
        if not OBSERVATION_AVAILABILITY_LAG_HOURS <= self.new_ignition_age_hours <= 12:
            raise RecursiveTransitionError("new_ignition_age_hours must be between 3 and 12")

    def transition_contract(self) -> dict[str, Any]:
        """Serializable renderer and scenario parameters for reproducible runs."""
        return {
            "transition_version": RECURSIVE_TRANSITION_VERSION,
            "time_step_hours": 12,
            "ignition_threshold": self.ignition_threshold,
            "candidate_radius_cells": self.candidate_radius_cells,
            "active_duration_steps": self.active_duration_steps,
            "intensity_retention": self.intensity_retention,
            "intensity_and_persistence_are_learned": False,
            "new_ignition_age_hours": self.new_ignition_age_hours,
            "observation_lookback_hours": OBSERVATION_LOOKBACK_HOURS,
            "observation_availability_lag_hours": OBSERVATION_AVAILABILITY_LAG_HOURS,
            "platform_aggregation": "maximum-cell-count-proxy; identities unavailable",
            "observation_calibration": (
                asdict(self.observation_calibration) if self.observation_calibration else None
            ),
        }

    @classmethod
    def from_model_bundle(
        cls,
        path: str | Path,
        *,
        renderer_contract: Mapping[str, Any] | None = None,
        **parameters: Any,
    ) -> "RecursiveTransitionModel":
        """Load a trusted local tabular-baseline bundle.

        Joblib uses pickle internally.  Callers must never pass an untrusted
        or user-uploaded model file to this method.
        """
        bundle = joblib.load(Path(path))
        if not isinstance(bundle, Mapping):
            raise RecursiveTransitionError("model bundle must contain a mapping")
        contract = bundle.get("feature_contract")
        if not isinstance(contract, Mapping):
            raise RecursiveTransitionError("model bundle has no feature contract")
        feature_columns = contract.get("feature_columns")
        if isinstance(feature_columns, str) or not isinstance(feature_columns, Sequence):
            raise RecursiveTransitionError("model bundle has no ordered feature list")
        if renderer_contract is not None:
            if parameters:
                raise RecursiveTransitionError("cannot override a persisted renderer contract")
            if (renderer_contract.get("transition_version") != RECURSIVE_TRANSITION_VERSION
                    or renderer_contract.get("time_step_hours") != 12
                    or renderer_contract.get("observation_lookback_hours") != OBSERVATION_LOOKBACK_HOURS
                    or renderer_contract.get("observation_availability_lag_hours") != OBSERVATION_AVAILABILITY_LAG_HOURS):
                raise RecursiveTransitionError("unsupported renderer contract")
            parameters = {name: renderer_contract[name] for name in (
                "ignition_threshold", "candidate_radius_cells", "active_duration_steps",
                "intensity_retention", "new_ignition_age_hours",
            )}
            calibration = renderer_contract.get("observation_calibration")
            parameters["observation_calibration"] = (
                SyntheticObservationCalibration(**calibration) if calibration else None
            )
        calibration = parameters.get("observation_calibration")
        if calibration is not None:
            cutoff = contract.get("chronological_split_cutoff_at")
            if (contract.get("split_group_column") != "source_snapshot_time"
                    or not isinstance(cutoff, str)
                    or datetime.fromisoformat(cutoff.replace("Z", "+00:00")) !=
                    datetime.fromisoformat(calibration.training_snapshot_times[-1].replace("Z", "+00:00"))):
                raise RecursiveTransitionError("model and calibration training snapshot boundary differ")
        return cls(bundle.get("model"), feature_columns=feature_columns, **parameters)

    def initial_state(
        self, ignitions: Mapping[str, float], *, observation_ages: Mapping[str, float] | None = None
    ) -> RecursiveFireState:
        """Create step zero from canonical cell IDs and slider intensities."""
        if not isinstance(ignitions, Mapping) or not ignitions:
            raise RecursiveTransitionError("ignitions must be a non-empty cell-to-intensity mapping")
        if observation_ages is not None and set(observation_ages) != set(ignitions):
            raise RecursiveTransitionError("observation ages must match ignition cell IDs")
        active = tuple(
            ActiveFireCell(
                cell_id=cell_from_id(cell_id).cell_id,
                intensity=intensity,
                remaining_active_steps=self.active_duration_steps,
                observation_age_hours=(
                    observation_ages[cell_id] if observation_ages is not None
                    else self.new_ignition_age_hours
                ),
            )
            for cell_id, intensity in sorted(
                ignitions.items(), key=lambda item: _cell_sort_key(cell_from_id(item[0]))
            )
        )
        return RecursiveFireState(step_index=0, active_cells=active)

    def initial_state_from_points(
        self, ignitions: Iterable[tuple[float, float, float]]
    ) -> RecursiveFireState:
        """Create step zero from ``(latitude, longitude, intensity)`` values."""
        by_cell: dict[str, float] = {}
        for latitude, longitude, intensity in ignitions:
            cell_id = cell_from_wgs84(latitude=latitude, longitude=longitude).cell_id
            by_cell[cell_id] = max(by_cell.get(cell_id, 0.0), _unit_interval(intensity, "intensity"))
        if not by_cell:
            raise RecursiveTransitionError("ignitions must contain at least one point")
        return self.initial_state(by_cell)

    def step(
        self,
        state: RecursiveFireState,
        *,
        terrain_provider: TerrainProvider,
    ) -> RecursiveStepResult:
        """Score the active frontier and advance exactly one 12-hour step."""
        if not isinstance(state, RecursiveFireState):
            raise TypeError("state must be a RecursiveFireState")
        if not callable(terrain_provider):
            raise TypeError("terrain_provider must be callable")

        active_by_id = {cell.cell_id: cell for cell in state.active_cells}
        candidate_features = self.candidate_feature_rows(
            state, terrain_provider=terrain_provider
        )
        ordered_candidates = tuple(item[0] for item in candidate_features)
        feature_rows = [item[1] for item in candidate_features]
        probabilities = self._probabilities(feature_rows)
        predictions = []
        new_active = []
        for cell, probability in zip(ordered_candidates, probabilities, strict=True):
            will_ignite = bool(probability >= self.ignition_threshold)
            next_intensity = (
                self._new_cell_intensity(cell, active_by_id=active_by_id) if will_ignite else 0.0
            )
            latitude, longitude = cell.center_wgs84
            predictions.append(
                CellTransitionPrediction(
                    cell_id=cell.cell_id,
                    latitude=latitude,
                    longitude=longitude,
                    ignition_probability=float(probability),
                    will_ignite=will_ignite,
                    next_intensity=next_intensity,
                )
            )
            if will_ignite:
                new_active.append(
                    ActiveFireCell(
                        cell_id=cell.cell_id,
                        intensity=next_intensity,
                        remaining_active_steps=self.active_duration_steps,
                        observation_age_hours=self.new_ignition_age_hours,
                    )
                )

        burned = set(state.burned_cell_ids)
        surviving_active = []
        for active in state.active_cells:
            if active.remaining_active_steps == 1:
                burned.add(active.cell_id)
            else:
                surviving_active.append(
                    ActiveFireCell(
                        cell_id=active.cell_id,
                        intensity=active.intensity * self.intensity_retention,
                        remaining_active_steps=active.remaining_active_steps - 1,
                        observation_age_hours=active.observation_age_hours + 12.0,
                    )
                )
        next_active = tuple(
            sorted((*surviving_active, *new_active), key=lambda item: _cell_sort_key(cell_from_id(item.cell_id)))
        )
        next_state = RecursiveFireState(
            step_index=state.step_index + 1,
            active_cells=next_active,
            burned_cell_ids=tuple(sorted(burned, key=lambda item: _cell_sort_key(cell_from_id(item)))),
        )
        return RecursiveStepResult(
            transition_version=RECURSIVE_TRANSITION_VERSION,
            previous_step_index=state.step_index,
            state=next_state,
            predictions=tuple(predictions),
        )

    def candidate_cells(self, state: RecursiveFireState) -> tuple[GridCell, ...]:
        """Return the deterministic unburned frontier scored at ``state``."""
        if not isinstance(state, RecursiveFireState):
            raise TypeError("state must be a RecursiveFireState")
        active_by_id = {cell.cell_id: cell for cell in state.active_cells}
        excluded = set(active_by_id).union(state.burned_cell_ids)
        candidates: dict[str, GridCell] = {}
        for active in state.active_cells:
            for cell in cells_in_square_radius(
                cell_from_id(active.cell_id), radius_cells=self.candidate_radius_cells
            ):
                if cell.cell_id not in excluded:
                    candidates[cell.cell_id] = cell
        return tuple(sorted(candidates.values(), key=_cell_sort_key))

    def candidate_feature_rows(
        self,
        state: RecursiveFireState,
        *,
        terrain_provider: TerrainProvider,
        include_cell_ids: Iterable[str] | None = None,
    ) -> tuple[tuple[GridCell, dict[str, float | None]], ...]:
        """Build model features for the inference frontier or a subset of it.

        ``include_cell_ids`` filters the already-derived frontier. IDs outside
        the frontier are ignored rather than turned into synthetic examples;
        this lets dataset aggregation intersect predictions with an existing
        historical label domain without changing inference behavior.
        """
        if not isinstance(state, RecursiveFireState):
            raise TypeError("state must be a RecursiveFireState")
        if not callable(terrain_provider):
            raise TypeError("terrain_provider must be callable")
        included = None
        if include_cell_ids is not None:
            included = {cell_from_id(cell_id).cell_id for cell_id in include_cell_ids}
        active_by_id = {cell.cell_id: cell for cell in state.active_cells}
        return tuple(
            (
                cell,
                self._feature_row(
                    cell, active_by_id=active_by_id, terrain_provider=terrain_provider
                ),
            )
            for cell in self.candidate_cells(state)
            if included is None or cell.cell_id in included
        )

    def _feature_row(
        self,
        cell: GridCell,
        *,
        active_by_id: Mapping[str, ActiveFireCell],
        terrain_provider: TerrainProvider,
    ) -> dict[str, float | None]:
        nearby = [
            active_by_id[neighbour.cell_id]
            for neighbour in cells_in_square_radius(cell, radius_cells=1)
            if neighbour.cell_id in active_by_id
            and OBSERVATION_AVAILABILITY_LAG_HOURS
            <= active_by_id[neighbour.cell_id].observation_age_hours
            <= OBSERVATION_LOOKBACK_HOURS
        ]
        local_counts = [
            self.observation_calibration.counts(active.intensity)[0]
            if self.observation_calibration else _synthetic_detection_count(active.intensity)
            for active in nearby
        ]
        local_brightness = [_synthetic_brightness(active.intensity) for active in nearby]
        total_count = sum(local_counts)
        if total_count:
            weighted_brightness = math.fsum(
                brightness * count for brightness, count in zip(local_brightness, local_counts, strict=True)
            ) / total_count
        else:
            weighted_brightness = None

        features: dict[str, float | None] = {
            "firms_local_3x3_has_detection": float(bool(nearby)),
            "firms_local_3x3_detection_count": float(total_count),
            "firms_local_3x3_bright_ti4_max": max(local_brightness) if nearby else None,
            "firms_local_3x3_bright_ti4_mean": weighted_brightness,
            "firms_local_3x3_platform_count": float(max(
                (self.observation_calibration.counts(active.intensity)[1]
                 if self.observation_calibration else 1 for active in nearby), default=0
            )),
            "firms_local_3x3_hours_since_last_detection": (
                min(active.observation_age_hours for active in nearby) if nearby else None
            ),
            "firms_local_3x3_active_cell_count": float(len(nearby)),
            **_terrain_features(terrain_provider(cell.cell_id)),
        }
        return {name: features[name] for name in self.feature_columns}

    def _probabilities(self, feature_rows: Sequence[Mapping[str, float | None]]) -> np.ndarray:
        if not feature_rows:
            return np.empty(0, dtype=np.float64)
        values = np.asarray(
            [[row[name] for name in self.feature_columns] for row in feature_rows],
            dtype=np.float64,
        )
        probabilities = np.asarray(self.estimator.predict_proba(values), dtype=np.float64)
        if probabilities.shape != (len(feature_rows), 2):
            raise RecursiveTransitionError("estimator returned an unexpected probability shape")
        positive = probabilities[:, 1]
        if not np.isfinite(positive).all() or (positive < 0).any() or (positive > 1).any():
            raise RecursiveTransitionError("estimator returned invalid ignition probabilities")
        return positive

    def _new_cell_intensity(
        self, cell: GridCell, *, active_by_id: Mapping[str, ActiveFireCell]
    ) -> float:
        neighbouring_intensities = [
            active_by_id[neighbour.cell_id].intensity
            for neighbour in cells_in_square_radius(cell, radius_cells=1)
            if neighbour.cell_id in active_by_id
        ]
        if not neighbouring_intensities:
            # The two-cell candidate radius can produce a threshold crossing
            # without an immediate active neighbour.  Keep that edge case
            # explicit and give it the weakest representable active state.
            return 0.0
        return max(neighbouring_intensities) * self.intensity_retention


def _terrain_features(values: Mapping[str, Any]) -> dict[str, float | None]:
    if not isinstance(values, Mapping):
        raise RecursiveTransitionError("terrain provider must return a mapping")
    valid = bool(values.get("terrain_valid", False))
    if not valid:
        return {
            "terrain_valid": 0.0,
            "terrain_elevation_m": None,
            "terrain_slope_degrees": None,
            "terrain_aspect_defined": 0.0,
            "terrain_aspect_sin": None,
            "terrain_aspect_cos": None,
        }
    return {
        "terrain_valid": 1.0,
        "terrain_elevation_m": _finite(values.get("terrain_elevation_m"), "terrain_elevation_m"),
        "terrain_slope_degrees": _finite(values.get("terrain_slope_degrees"), "terrain_slope_degrees"),
        "terrain_aspect_defined": float(bool(values.get("terrain_aspect_defined", False))),
        "terrain_aspect_sin": _finite(values.get("terrain_aspect_sin"), "terrain_aspect_sin"),
        "terrain_aspect_cos": _finite(values.get("terrain_aspect_cos"), "terrain_aspect_cos"),
    }


def _synthetic_detection_count(intensity: float) -> int:
    return max(1, math.ceil(3.0 * intensity))


def _synthetic_brightness(intensity: float) -> float:
    return SYNTHETIC_BRIGHTNESS_MIN_K + intensity * (
        SYNTHETIC_BRIGHTNESS_MAX_K - SYNTHETIC_BRIGHTNESS_MIN_K
    )


def _cell_sort_key(cell: GridCell) -> tuple[int, int]:
    return (cell.y_index, cell.x_index)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RecursiveTransitionError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise RecursiveTransitionError(f"{label} must be finite")
    return result


def _unit_interval(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise RecursiveTransitionError(f"{label} must be between zero and one")
    return result
