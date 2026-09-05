"""Bounded recursive state for the incident scheduled-sampling experiment.

Observed centre aggregates survive state initialization without brightness
clipping or invented new observations. Future intensity/persistence remain
explicit heuristics. No FEDS target or future geometry enters inference.
"""

from dataclasses import dataclass, replace
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from .incident_sequences import stable_fraction
from .recursive_transition import (
    ActiveFireCell, RecursiveFireState, RecursiveTransitionModel, _terrain_features,
    _synthetic_brightness, _synthetic_detection_count,
)
from .training_grid import cell_from_id, cells_in_square_radius


INCIDENT_TRANSITION_VERSION = "bounded-incident-observation-state/v1"
OBSERVATION_COLUMNS = (
    "firms_center_has_detection", "firms_center_detection_count",
    "firms_center_bright_ti4_max", "firms_center_bright_ti4_mean",
    "firms_center_platform_count", "firms_center_hours_since_last_detection",
)


def probability_logit(probabilities):
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class CalibratedSpreadEstimator:
    """Importable bundle type, including when training runs through ``-m``."""

    model: Any
    calibrator: Any

    def predict_proba(self, values):
        probability = self.model.predict_proba(values)[:, 1]
        return self.calibrator.predict_proba(probability_logit(probability))


@dataclass(frozen=True)
class EvidenceCell(ActiveFireCell):
    detection_count: int = 1
    bright_ti4_max: float = 305.
    bright_ti4_mean: float = 305.
    platform_count: int = 1

    def __post_init__(self):
        super().__post_init__()
        if (self.detection_count < 1 or int(self.detection_count) != self.detection_count
                or not 1 <= self.platform_count <= min(3, self.detection_count)
                or int(self.platform_count) != self.platform_count
                or not math.isfinite(self.bright_ti4_max) or not math.isfinite(self.bright_ti4_mean)
                or self.bright_ti4_mean > self.bright_ti4_max):
            raise ValueError("invalid observed FIRMS aggregates")


def observed_incident_state(model, frame: pd.DataFrame, *, step_index=0):
    if not set(OBSERVATION_COLUMNS).issubset(frame):
        raise ValueError("observed state requires all centre FIRMS aggregates")
    if not frame.firms_center_has_detection.isin([0, 1]).all():
        raise ValueError("centre detection flags must be binary")
    detected = frame.loc[frame.firms_center_has_detection.eq(1)]
    if not detected.cell_id.is_unique:
        raise ValueError("observed cell IDs must be unique")
    active = []
    for row in detected.to_dict("records"):
        age = float(row["firms_center_hours_since_last_detection"])
        if not 3 <= age <= 24:
            raise ValueError("observed age must satisfy the 3--24 hour policy")
        brightness = float(row["firms_center_bright_ti4_max"])
        active.append(EvidenceCell(
            cell_id=row["cell_id"], intensity=float(np.clip((brightness - 305) / 62, 0, 1)),
            remaining_active_steps=model.active_duration_steps, observation_age_hours=age,
            detection_count=float(row["firms_center_detection_count"]), bright_ti4_max=brightness,
            bright_ti4_mean=float(row["firms_center_bright_ti4_mean"]),
            platform_count=float(row["firms_center_platform_count"]),
        ))
    return RecursiveFireState(step_index, tuple(sorted(active, key=lambda c: c.cell_id)))


def mix_observed_and_predicted(observed, predicted, *, predicted_fraction: float, key: str):
    """Deterministic cell-level scheduled sampling, including absence/burn masks.

    Choosing the observed branch replaces that cell's simulated state, even
    when it corrects a false positive. At fraction=1 no observed correction
    can enter; at fraction=0 the result is exactly the observed state.
    """
    if not 0 <= predicted_fraction <= 1 or observed.step_index != predicted.step_index:
        raise ValueError("invalid scheduled-sampling fraction or state time")
    actual = {c.cell_id: c for c in observed.active_cells}
    synthetic = {c.cell_id: c for c in predicted.active_cells}
    ids = set(actual) | set(synthetic) | set(observed.burned_cell_ids) | set(predicted.burned_cell_ids)
    active, burned = [], []
    for cell_id in sorted(ids):
        use_prediction = stable_fraction(key + ":" + cell_id) < predicted_fraction
        selected = synthetic if use_prediction else actual
        source = predicted if use_prediction else observed
        if cell_id in selected:
            active.append(selected[cell_id])
        elif cell_id in source.burned_cell_ids:
            burned.append(cell_id)
    return RecursiveFireState(predicted.step_index, tuple(active), tuple(burned))


class IncidentTransitionModel(RecursiveTransitionModel):
    def __init__(self, *args, max_new_cells_per_step=128, max_candidates=5000,
                 growth_fraction=.5, allowed_cell: Callable[[str], bool] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if (not isinstance(max_new_cells_per_step, int) or max_new_cells_per_step < 1
                or not isinstance(max_candidates, int) or max_candidates < 1
                or not math.isfinite(growth_fraction) or not 0 < growth_fraction <= 1):
            raise ValueError("invalid candidate/growth bound")
        self.max_new_cells_per_step = max_new_cells_per_step
        self.max_candidates = max_candidates
        self.growth_fraction = growth_fraction
        self.allowed_cell = allowed_cell

    def transition_contract(self):
        return {**super().transition_contract(), "transition_version": INCIDENT_TRANSITION_VERSION,
                "maximum_step_distance_m": math.sqrt(2) * 1000,
                "max_new_cells_per_step": self.max_new_cells_per_step,
                "max_candidates": self.max_candidates, "growth_fraction": self.growth_fraction,
                "observed_aggregates_preserved": True,
                "extinction_policy": "two-active-steps; no spontaneous reactivation of burned cells",
                "candidate_pruning": "highest-neighbor-intensity then cell ID; bounded eight-neighbor frontier"}

    def candidate_cells(self, state):
        excluded = {c.cell_id for c in state.active_cells} | set(state.burned_cell_ids)
        candidates = {}
        for active in state.active_cells:
            for cell in cells_in_square_radius(cell_from_id(active.cell_id), radius_cells=1):
                if cell.cell_id in excluded or (self.allowed_cell and not self.allowed_cell(cell.cell_id)):
                    continue
                old = candidates.get(cell.cell_id)
                candidates[cell.cell_id] = (cell, max(active.intensity, old[1] if old else 0.))
        ordered = sorted(candidates.values(), key=lambda pair: (-pair[1], pair[0].cell_id))
        return tuple(cell for cell, _intensity in ordered[:self.max_candidates])

    def _feature_row(self, cell, *, active_by_id, terrain_provider):
        nearby = [active_by_id[c.cell_id] for c in cells_in_square_radius(cell, radius_cells=1)
                  if c.cell_id in active_by_id and 3 <= active_by_id[c.cell_id].observation_age_hours <= 24]
        counts, maxima, means, platforms = [], [], [], []
        for active in nearby:
            if isinstance(active, EvidenceCell):
                count, platform = active.detection_count, active.platform_count
                maximum, mean = active.bright_ti4_max, active.bright_ti4_mean
            else:
                count, platform = (self.observation_calibration.counts(active.intensity)
                                   if self.observation_calibration else (_synthetic_detection_count(active.intensity), 1))
                maximum = mean = _synthetic_brightness(active.intensity)
            counts.append(count); platforms.append(platform); maxima.append(maximum); means.append(mean)
        count = sum(counts)
        features = {
            "firms_local_3x3_has_detection": float(bool(nearby)),
            "firms_local_3x3_detection_count": float(count),
            "firms_local_3x3_bright_ti4_max": max(maxima) if nearby else None,
            "firms_local_3x3_bright_ti4_mean": math.fsum(c * m for c, m in zip(counts, means)) / count if count else None,
            "firms_local_3x3_platform_count": float(max(platforms, default=0)),
            "firms_local_3x3_hours_since_last_detection": min((c.observation_age_hours for c in nearby), default=None),
            "firms_local_3x3_active_cell_count": float(len(nearby)),
            **_terrain_features(terrain_provider(cell.cell_id)),
        }
        return {name: features[name] for name in self.feature_columns}

    def step(self, state, *, terrain_provider):
        result = super().step(state, terrain_provider=terrain_provider)
        limit = min(self.max_new_cells_per_step, math.ceil(len(state.active_cells) * self.growth_fraction))
        chosen = sorted((p for p in result.predictions if p.will_ignite),
                        key=lambda p: (-p.ignition_probability, p.cell_id))[:limit]
        admitted = {p.cell_id for p in chosen}
        old = {c.cell_id: c for c in state.active_cells}
        active = []
        for cell in result.state.active_cells:
            if cell.cell_id in old:
                # Preserve observation aggregates; decay simulated intensity,
                # but never rewrite the brightness of a historical detection.
                cell = replace(old[cell.cell_id], intensity=cell.intensity,
                               remaining_active_steps=cell.remaining_active_steps,
                               observation_age_hours=cell.observation_age_hours)
                active.append(cell)
            elif cell.cell_id in admitted:
                active.append(cell)
        predictions = tuple(replace(p, will_ignite=p.cell_id in admitted,
                                    next_intensity=p.next_intensity if p.cell_id in admitted else 0.)
                            for p in result.predictions)
        return replace(result, transition_version=INCIDENT_TRANSITION_VERSION,
                       state=replace(result.state, active_cells=tuple(active)), predictions=predictions)
