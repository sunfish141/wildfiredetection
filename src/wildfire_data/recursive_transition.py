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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .candidate_dataset import DEFAULT_CANDIDATE_RADIUS_CELLS, DEFAULT_MODEL_FEATURE_COLUMNS
from .training_grid import GridCell, cell_from_id, cell_from_wgs84, cells_in_square_radius


RECURSIVE_TRANSITION_VERSION = "recursive-firms-compatible-transition/v1"
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


class RecursiveTransitionError(ValueError):
    """Raised when a recursive state or model contract is invalid."""


@dataclass(frozen=True)
class ActiveFireCell:
    """One active 1 km cell in the recursive simulation state."""

    cell_id: str
    intensity: float
    remaining_active_steps: int

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

    @classmethod
    def from_model_bundle(
        cls,
        path: str | Path,
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
        return cls(bundle.get("model"), feature_columns=feature_columns, **parameters)

    def initial_state(self, ignitions: Mapping[str, float]) -> RecursiveFireState:
        """Create step zero from canonical cell IDs and slider intensities."""
        if not isinstance(ignitions, Mapping) or not ignitions:
            raise RecursiveTransitionError("ignitions must be a non-empty cell-to-intensity mapping")
        active = tuple(
            ActiveFireCell(
                cell_id=cell_from_id(cell_id).cell_id,
                intensity=intensity,
                remaining_active_steps=self.active_duration_steps,
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
        excluded = set(active_by_id).union(state.burned_cell_ids)
        candidates: dict[str, GridCell] = {}
        for active in state.active_cells:
            for cell in cells_in_square_radius(
                cell_from_id(active.cell_id), radius_cells=self.candidate_radius_cells
            ):
                if cell.cell_id not in excluded:
                    candidates[cell.cell_id] = cell
        ordered_candidates = tuple(sorted(candidates.values(), key=_cell_sort_key))

        feature_rows = [
            self._feature_row(cell, active_by_id=active_by_id, terrain_provider=terrain_provider)
            for cell in ordered_candidates
        ]
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
        ]
        local_counts = [_synthetic_detection_count(active.intensity) for active in nearby]
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
            # Synthetic evidence is one provider-neutral observation stream,
            # rather than a fabricated count of satellite platforms.
            "firms_local_3x3_platform_count": float(bool(nearby)),
            "firms_local_3x3_hours_since_last_detection": 0.0 if nearby else None,
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
