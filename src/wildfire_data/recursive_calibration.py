"""Calibrate synthetic observation counts without reading holdout features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .recursive_transition import (
    OBSERVATION_AVAILABILITY_LAG_HOURS,
    OBSERVATION_LOOKBACK_HOURS,
    SYNTHETIC_BRIGHTNESS_MAX_K,
    SYNTHETIC_BRIGHTNESS_MIN_K,
    SyntheticObservationCalibration,
)


class RecursiveCalibrationError(ValueError):
    """Raised when calibration data violates the training-only contract."""


def validate_training_split(examples: pd.DataFrame) -> None:
    """Reject missing/mixed snapshot splits and nonchronological holdout."""
    if not {"dataset_split", "source_snapshot_time"}.issubset(examples.columns):
        raise RecursiveCalibrationError("examples requires dataset_split and source_snapshot_time")
    if examples.empty or not examples["dataset_split"].isin(["train", "validation"]).all():
        raise RecursiveCalibrationError("dataset splits must be train or validation without missing values")
    times = pd.to_datetime(examples["source_snapshot_time"], utc=True, errors="raise", format="mixed")
    if times.isna().any():
        raise RecursiveCalibrationError("source snapshot times must not be missing")
    groups = pd.DataFrame({"time": times, "split": examples["dataset_split"]})
    if (groups.groupby("time")["split"].nunique() != 1).any():
        raise RecursiveCalibrationError("one source snapshot contains mixed dataset splits")
    train = times[examples["dataset_split"] == "train"]
    validation = times[examples["dataset_split"] == "validation"]
    if train.empty:
        raise RecursiveCalibrationError("no training snapshots")
    if not validation.empty and train.max() >= validation.min():
        raise RecursiveCalibrationError("training snapshots must precede validation")


def fit_observation_calibration(
    examples: pd.DataFrame, *, release_manifest_sha256: str
) -> SyntheticObservationCalibration:
    """Fit mean per-cell counts in five fixed intensity bins on train only.

    Brightness is converted with the same clipped 305--367 K slider mapping
    used by observed initialization. Empty bins use the global training mean.
    Means are rounded half-up by the renderer. No targets are used.
    """
    validate_training_split(examples)
    columns = (
        "firms_center_has_detection", "firms_center_bright_ti4_max",
        "firms_center_detection_count", "firms_center_platform_count",
        "firms_center_hours_since_last_detection",
    )
    missing = set(columns) - set(examples.columns)
    if missing:
        raise RecursiveCalibrationError("missing calibration columns: " + ", ".join(sorted(missing)))
    train = examples.loc[examples["dataset_split"] == "train"]
    center = pd.to_numeric(train[columns[0]], errors="raise")
    if center.isna().any() or not center.isin([0, 1]).all():
        raise RecursiveCalibrationError("training centre detection must be binary")
    observed = train.loc[center == 1]
    values = observed[list(columns[1:])].apply(pd.to_numeric, errors="raise")
    if values.empty or not np.isfinite(values.to_numpy()).all():
        raise RecursiveCalibrationError("training centre observations must be nonempty and finite")
    counts = values["firms_center_detection_count"]
    platforms = values["firms_center_platform_count"]
    ages = values["firms_center_hours_since_last_detection"]
    if (counts < 1).any() or (counts % 1 != 0).any():
        raise RecursiveCalibrationError("observed detection counts must be positive integers")
    if (platforms < 1).any() or (platforms > 3).any() or (platforms > counts).any() or (platforms % 1 != 0).any():
        raise RecursiveCalibrationError("observed platform counts must be valid integers")
    if not ages.between(OBSERVATION_AVAILABILITY_LAG_HOURS, OBSERVATION_LOOKBACK_HOURS).all():
        raise RecursiveCalibrationError("observed ages violate the FIRMS availability/lookback policy")
    intensity = ((values["firms_center_bright_ti4_max"] - SYNTHETIC_BRIGHTNESS_MIN_K) /
                 (SYNTHETIC_BRIGHTNESS_MAX_K - SYNTHETIC_BRIGHTNESS_MIN_K)).clip(0, 1)
    bins = np.minimum(4, (intensity * 5).astype(int))

    def bin_means(series: pd.Series) -> tuple[float, ...]:
        means = series.groupby(bins).mean()
        return tuple(float(means.get(index, series.mean())) for index in range(5))

    return SyntheticObservationCalibration(
        detection_count_by_bin=bin_means(counts),
        platform_count_by_bin=bin_means(platforms),
        training_row_count=len(observed),
        training_snapshot_times=tuple(
            value.isoformat().replace("+00:00", "Z")
            for value in pd.DatetimeIndex(pd.to_datetime(
                train["source_snapshot_time"], utc=True, format="mixed"
            ).unique()).sort_values()
        ),
        source_release_manifest_sha256=release_manifest_sha256,
    )
