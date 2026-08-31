"""Open-loop evaluation for the experimental recursive transition model.

The evaluator initializes one validation run from observed FIRMS centre
detections, then advances only with model-generated state.  At each requested
horizon it compares predictions with the no-centre candidate rows belonging to
the corresponding historical source snapshot. Predictions outside that
released label domain remain visible as diagnostics and are never converted to
false-positive weak negatives.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .recursive_transition import (
    SYNTHETIC_BRIGHTNESS_MAX_K,
    SYNTHETIC_BRIGHTNESS_MIN_K,
    RecursiveFireState,
    RecursiveTransitionModel,
)
from .rollout_sequences import RolloutSequence, build_rollout_sequences, snapshot_frame
from .terrain_features import TerrainFeatureSampler


ROLLOUT_EVALUATION_VERSION = "recursive-open-loop-evaluation/v1"
DEFAULT_HORIZONS = (1, 2, 4, 8)
REQUIRED_COLUMNS = (
    "source_snapshot_time",
    "target_snapshot_time",
    "anchor_at",
    "target_end_at",
    "cell_id",
    "dataset_split",
    "target_newly_burned_12h",
    "firms_center_has_detection",
    "firms_center_bright_ti4_max",
)


class RolloutEvaluationError(ValueError):
    """Raised when a release cannot support an honest open-loop evaluation."""


@dataclass(frozen=True)
class RolloutHorizonMetrics:
    """Weak-label metrics for one open-loop horizon from a fixed origin."""

    horizon_steps: int
    horizon_hours: int
    source_snapshot_time: str
    evaluation_row_count: int
    evaluation_positive_count: int
    model_candidate_count: int
    candidates_inside_evaluation_domain: int
    candidates_outside_evaluation_domain: int
    evaluation_domain_coverage: float
    predicted_positive_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    true_negative_count: int
    precision: float
    recall: float
    f1: float
    brier_score: float
    roc_auc: float
    pr_auc: float


@dataclass(frozen=True)
class OpenLoopEvaluation:
    """One fully open-loop run initialized from an observed validation state."""

    evaluation_version: str
    transition_version: str
    split_name: str
    origin_source_snapshot_time: str
    initial_active_cell_count: int
    horizons: tuple[RolloutHorizonMetrics, ...]

    def as_dict(self) -> dict[str, Any]:
        document = asdict(self)
        document["horizons"] = [asdict(item) for item in self.horizons]
        return document


def evaluate_open_loop(
    model: RecursiveTransitionModel,
    examples: pd.DataFrame,
    sequence: RolloutSequence,
    *,
    start_snapshot_index: int,
    terrain_provider: Any,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    split_name: str = "validation",
) -> OpenLoopEvaluation:
    """Evaluate fixed-origin recursive predictions at selected 12-hour steps."""
    if not isinstance(model, RecursiveTransitionModel):
        raise TypeError("model must be a RecursiveTransitionModel")
    if not isinstance(examples, pd.DataFrame):
        raise TypeError("examples must be a pandas DataFrame")
    if not callable(terrain_provider):
        raise TypeError("terrain_provider must be callable")
    missing = sorted(set(REQUIRED_COLUMNS) - set(examples.columns))
    if missing:
        raise RolloutEvaluationError("examples is missing required columns: " + ", ".join(missing))
    requested_horizons = _horizons(horizons)
    if (
        not isinstance(start_snapshot_index, int)
        or isinstance(start_snapshot_index, bool)
        or start_snapshot_index < 0
    ):
        raise RolloutEvaluationError("start_snapshot_index must be a non-negative integer")
    last_required = start_snapshot_index + max(requested_horizons) - 1
    if start_snapshot_index >= len(sequence.snapshots) or last_required >= len(sequence.snapshots):
        raise RolloutEvaluationError("sequence does not contain every requested rollout horizon")

    selected_snapshots = sequence.snapshots[start_snapshot_index : last_required + 1]
    for snapshot in selected_snapshots:
        frame = snapshot_frame(examples, snapshot)
        split_values = set(frame["dataset_split"].dropna().astype(str))
        if split_values != {split_name}:
            raise RolloutEvaluationError(
                "every rollout snapshot must belong entirely to the requested split"
            )

    origin_frame = snapshot_frame(examples, selected_snapshots[0])
    state = initial_state_from_observed_firms(model, origin_frame)
    initial_active_count = len(state.active_cells)
    metrics = []
    requested = set(requested_horizons)
    for step_number, snapshot in enumerate(selected_snapshots, start=1):
        result = model.step(state, terrain_provider=terrain_provider)
        if step_number in requested:
            metrics.append(
                _score_horizon(
                    snapshot_frame(examples, snapshot),
                    predictions=result.predictions,
                    horizon_steps=step_number,
                    source_snapshot_time=snapshot.source_snapshot_time,
                )
            )
        state = result.state

    return OpenLoopEvaluation(
        evaluation_version=ROLLOUT_EVALUATION_VERSION,
        transition_version=result.transition_version,
        split_name=split_name,
        origin_source_snapshot_time=_format_timestamp(selected_snapshots[0].source_snapshot_time),
        initial_active_cell_count=initial_active_count,
        horizons=tuple(metrics),
    )


def initial_state_from_observed_firms(
    model: RecursiveTransitionModel, snapshot_examples: pd.DataFrame
) -> RecursiveFireState:
    """Convert observed centre detections into the recursive slider state."""
    required = {"cell_id", "firms_center_has_detection", "firms_center_bright_ti4_max"}
    missing = sorted(required - set(snapshot_examples.columns))
    if missing:
        raise RolloutEvaluationError(
            "snapshot examples is missing FIRMS initialization columns: " + ", ".join(missing)
        )
    center = pd.to_numeric(snapshot_examples["firms_center_has_detection"], errors="raise")
    if center.isna().any() or not center.isin([0, 1]).all():
        raise RolloutEvaluationError("firms_center_has_detection must contain binary 0/1 values")
    detected = snapshot_examples.loc[center == 1]
    if detected.empty:
        raise RolloutEvaluationError("rollout origin contains no observed FIRMS centre detections")
    brightness = pd.to_numeric(detected["firms_center_bright_ti4_max"], errors="raise")
    if brightness.isna().any() or not np.isfinite(brightness.to_numpy(dtype=float)).all():
        raise RolloutEvaluationError("observed FIRMS centre brightness must be finite")
    ignitions = {
        str(cell_id): _intensity_from_brightness(value)
        for cell_id, value in zip(detected["cell_id"], brightness, strict=True)
    }
    return model.initial_state(ignitions)


def first_split_origin(
    examples: pd.DataFrame,
    sequences: Sequence[RolloutSequence],
    *,
    split_name: str,
    required_snapshot_count: int,
) -> tuple[RolloutSequence, int]:
    """Return the first contiguous run wholly inside a requested data split."""
    if required_snapshot_count < 1:
        raise RolloutEvaluationError("required_snapshot_count must be positive")
    for sequence in sequences:
        run_start = None
        run_length = 0
        for index, snapshot in enumerate(sequence.snapshots):
            frame = snapshot_frame(examples, snapshot)
            values = set(frame["dataset_split"].dropna().astype(str))
            if values == {split_name}:
                if run_start is None:
                    run_start = index
                run_length += 1
                if run_length >= required_snapshot_count:
                    return sequence, int(run_start)
            else:
                run_start = None
                run_length = 0
    raise RolloutEvaluationError(
        f"no {split_name!r} sequence contains {required_snapshot_count} consecutive snapshots"
    )


def _score_horizon(
    snapshot_examples: pd.DataFrame,
    *,
    predictions: Sequence[Any],
    horizon_steps: int,
    source_snapshot_time: pd.Timestamp,
) -> RolloutHorizonMetrics:
    center = pd.to_numeric(snapshot_examples["firms_center_has_detection"], errors="raise")
    frontier = snapshot_examples.loc[center == 0]
    if frontier.empty:
        raise RolloutEvaluationError("evaluation snapshot has no no-centre frontier rows")
    targets = pd.to_numeric(frontier["target_newly_burned_12h"], errors="raise")
    if targets.isna().any() or not targets.isin([0, 1]).all():
        raise RolloutEvaluationError("target_newly_burned_12h must contain binary 0/1 values")
    if not frontier["cell_id"].astype(str).is_unique:
        raise RolloutEvaluationError("evaluation snapshot contains duplicate frontier cell IDs")

    probability_by_cell = {item.cell_id: float(item.ignition_probability) for item in predictions}
    decision_by_cell = {item.cell_id: bool(item.will_ignite) for item in predictions}
    domain_cells = frontier["cell_id"].astype(str).tolist()
    domain_set = set(domain_cells)
    probabilities = np.asarray([probability_by_cell.get(cell_id, 0.0) for cell_id in domain_cells])
    decisions = np.asarray([decision_by_cell.get(cell_id, False) for cell_id in domain_cells])
    actual = targets.to_numpy(dtype=np.int8)
    covered = sum(cell_id in probability_by_cell for cell_id in domain_cells)
    true_positive = int(np.sum(decisions & (actual == 1)))
    false_positive = int(np.sum(decisions & (actual == 0)))
    false_negative = int(np.sum(~decisions & (actual == 1)))
    true_negative = int(np.sum(~decisions & (actual == 0)))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    if len(np.unique(actual)) != 2:
        raise RolloutEvaluationError("evaluation frontier must contain both target classes")
    return RolloutHorizonMetrics(
        horizon_steps=horizon_steps,
        horizon_hours=12 * horizon_steps,
        source_snapshot_time=_format_timestamp(source_snapshot_time),
        evaluation_row_count=len(frontier),
        evaluation_positive_count=int(np.sum(actual)),
        model_candidate_count=len(predictions),
        candidates_inside_evaluation_domain=covered,
        candidates_outside_evaluation_domain=sum(
            item.cell_id not in domain_set for item in predictions
        ),
        evaluation_domain_coverage=covered / len(frontier),
        predicted_positive_count=int(np.sum(decisions)),
        true_positive_count=true_positive,
        false_positive_count=false_positive,
        false_negative_count=false_negative,
        true_negative_count=true_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        brier_score=float(brier_score_loss(actual, probabilities)),
        roc_auc=float(roc_auc_score(actual, probabilities)),
        pr_auc=float(average_precision_score(actual, probabilities)),
    )


def _intensity_from_brightness(brightness: float) -> float:
    value = (float(brightness) - SYNTHETIC_BRIGHTNESS_MIN_K) / (
        SYNTHETIC_BRIGHTNESS_MAX_K - SYNTHETIC_BRIGHTNESS_MIN_K
    )
    return min(1.0, max(0.0, value))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _horizons(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise RolloutEvaluationError("horizons must be positive integer step counts")
    result = tuple(values)
    if (
        not result
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise RolloutEvaluationError("horizons must be unique increasing positive integers")
    return result


def _format_timestamp(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(document, destination, indent=2, sort_keys=True, allow_nan=False)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--model-bundle", required=True, type=Path)
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidate_path = args.release / "candidate_examples.csv.gz"
    examples = pd.read_csv(candidate_path, compression="gzip", usecols=list(REQUIRED_COLUMNS))
    sequences = build_rollout_sequences(examples)
    sequence, start_index = first_split_origin(
        examples,
        sequences,
        split_name="validation",
        required_snapshot_count=max(DEFAULT_HORIZONS),
    )
    model = RecursiveTransitionModel.from_model_bundle(args.model_bundle)
    terrain = TerrainFeatureSampler(args.data_root, max_cached_blocks=8)
    evaluation = evaluate_open_loop(
        model,
        examples,
        sequence,
        start_snapshot_index=start_index,
        terrain_provider=terrain.sample_cell,
    )
    document = evaluation.as_dict()
    _atomic_json(args.output, document)
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
