"""Validated temporal sequences for supervised recursive spread rollouts.

This module does not train a model or create synthetic fire states.  It gives
later rollout training a small, explicit contract over an already-loaded
candidate table: rows from one FEDS source snapshot stay together, consecutive
snapshots are exactly 12 hours apart, and missing snapshots split rather than
silently stretching a transition.

Row positions refer to the caller's dataframe by integer position (``iloc``),
so the sequence metadata preserves each original cell-specific cutoff without
copying the large candidate table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


ROLLOUT_SEQUENCE_VERSION = "feds-snapshot-rollout-sequences/v1"
ROLLOUT_STEP = timedelta(hours=12)


class RolloutSequenceError(ValueError):
    """Raised when candidate rows cannot form unambiguous 12-hour sequences."""


@dataclass(frozen=True)
class RolloutSnapshot:
    """All candidate rows belonging to one FEDS source snapshot."""

    source_snapshot_time: pd.Timestamp
    target_snapshot_time: pd.Timestamp
    anchor_start_at: pd.Timestamp
    anchor_end_at: pd.Timestamp
    row_positions: tuple[int, ...]

    @property
    def row_count(self) -> int:
        return len(self.row_positions)


@dataclass(frozen=True)
class RolloutSequence:
    """A maximal run of consecutive 12-hour source snapshots."""

    sequence_index: int
    snapshots: tuple[RolloutSnapshot, ...]

    @property
    def source_start_at(self) -> pd.Timestamp:
        return self.snapshots[0].source_snapshot_time

    @property
    def source_end_at(self) -> pd.Timestamp:
        return self.snapshots[-1].source_snapshot_time

    @property
    def transition_count(self) -> int:
        return len(self.snapshots)

    @property
    def row_count(self) -> int:
        return sum(snapshot.row_count for snapshot in self.snapshots)


def build_rollout_sequences(examples: pd.DataFrame) -> tuple[RolloutSequence, ...]:
    """Group candidate rows into maximal, validated 12-hour snapshot runs.

    Each input row already represents the transition from its
    ``source_snapshot_time`` to ``target_snapshot_time``.  Distinct source
    snapshots must therefore be separated by a positive whole multiple of 12
    hours. A one-step interval continues the current sequence; a larger
    interval records missing source evidence by starting a new sequence.

    ``anchor_at`` and ``target_end_at`` remain cell-specific because FEDS
    local-solar alignment varies with longitude. They must still describe an
    exact 12-hour prediction horizon for every row.
    """
    _validate_frame(examples)
    source_times = _utc_column(examples, "source_snapshot_time")
    target_times = _utc_column(examples, "target_snapshot_time", allow_missing=True)
    anchors = _utc_column(examples, "anchor_at")
    target_ends = _utc_column(examples, "target_end_at")

    expected_target_times = source_times + ROLLOUT_STEP
    invalid_snapshot_targets = target_times.notna() & (target_times != expected_target_times)
    if invalid_snapshot_targets.any():
        position = int(np.flatnonzero(invalid_snapshot_targets.to_numpy())[0])
        raise RolloutSequenceError(
            "target_snapshot_time must be exactly 12 hours after source_snapshot_time "
            f"at row position {position}"
        )
    invalid_target_ends = target_ends != anchors + ROLLOUT_STEP
    if invalid_target_ends.any():
        position = int(np.flatnonzero(invalid_target_ends.to_numpy())[0])
        raise RolloutSequenceError(
            "target_end_at must be exactly 12 hours after anchor_at "
            f"at row position {position}"
        )

    snapshots = _snapshots(examples, source_times, anchors)
    sequence_groups: list[list[RolloutSnapshot]] = []
    for snapshot in snapshots:
        if not sequence_groups:
            sequence_groups.append([snapshot])
            continue
        previous = sequence_groups[-1][-1]
        difference = snapshot.source_snapshot_time - previous.source_snapshot_time
        step_count, remainder = divmod(difference, ROLLOUT_STEP)
        if remainder != timedelta(0) or step_count < 1:
            raise RolloutSequenceError(
                "distinct source snapshots must be separated by a positive whole "
                "multiple of 12 hours"
            )
        if step_count == 1:
            sequence_groups[-1].append(snapshot)
        else:
            sequence_groups.append([snapshot])

    return tuple(
        RolloutSequence(sequence_index=index, snapshots=tuple(group))
        for index, group in enumerate(sequence_groups)
    )


def snapshot_frame(examples: pd.DataFrame, snapshot: RolloutSnapshot) -> pd.DataFrame:
    """Return one snapshot's original rows in their original relative order."""
    if not isinstance(examples, pd.DataFrame):
        raise TypeError("examples must be a pandas DataFrame")
    if not isinstance(snapshot, RolloutSnapshot):
        raise TypeError("snapshot must be a RolloutSnapshot")
    return examples.iloc[list(snapshot.row_positions)]


def _snapshots(
    examples: pd.DataFrame,
    source_times: pd.Series,
    anchors: pd.Series,
) -> tuple[RolloutSnapshot, ...]:
    result = []
    for source_time in pd.DatetimeIndex(source_times.unique()).sort_values():
        positions_array = np.flatnonzero((source_times == source_time).to_numpy())
        positions = tuple(int(position) for position in positions_array)
        cells = examples.iloc[list(positions)]["cell_id"]
        if cells.isna().any() or cells.astype(str).str.strip().eq("").any():
            raise RolloutSequenceError(
                f"snapshot {source_time.isoformat()} contains a missing or blank cell_id"
            )
        duplicated = cells.astype(str).duplicated(keep=False)
        if duplicated.any():
            duplicate = cells.astype(str)[duplicated].iloc[0]
            raise RolloutSequenceError(
                f"snapshot {source_time.isoformat()} contains duplicate cell_id {duplicate!r}"
            )
        snapshot_anchors = anchors.iloc[list(positions)]
        result.append(
            RolloutSnapshot(
                source_snapshot_time=source_time,
                target_snapshot_time=source_time + ROLLOUT_STEP,
                anchor_start_at=snapshot_anchors.min(),
                anchor_end_at=snapshot_anchors.max(),
                row_positions=positions,
            )
        )
    return tuple(result)


def _validate_frame(examples: pd.DataFrame) -> None:
    if not isinstance(examples, pd.DataFrame):
        raise TypeError("examples must be a pandas DataFrame")
    if examples.empty:
        raise RolloutSequenceError("examples must not be empty")
    if not examples.columns.is_unique:
        raise RolloutSequenceError("examples must have unique columns")
    required = {
        "source_snapshot_time",
        "target_snapshot_time",
        "anchor_at",
        "target_end_at",
        "cell_id",
    }
    missing = sorted(required - set(examples.columns))
    if missing:
        raise RolloutSequenceError("examples is missing required columns: " + ", ".join(missing))


def _utc_column(
    examples: pd.DataFrame, name: str, *, allow_missing: bool = False
) -> pd.Series:
    try:
        values = pd.to_datetime(examples[name], utc=True, errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise RolloutSequenceError(f"{name} must contain UTC-parseable timestamps") from exc
    if not allow_missing and values.isna().any():
        raise RolloutSequenceError(f"{name} must not contain missing timestamps")
    return pd.Series(values, index=examples.index)
