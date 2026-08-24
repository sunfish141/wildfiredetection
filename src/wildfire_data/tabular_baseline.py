"""A small, leakage-aware tabular baseline for wildfire training examples.

This module deliberately accepts an already-built model-ready table.  It does
not decide how FEDS labels, FIRMS features, weather, or static rasters are
constructed.  Its narrow responsibilities are to make the first baseline
repeatable:

* require an explicit target and explicit numeric feature list;
* hold out a later, disjoint set of anchor times for evaluation;
* reject label, source, geometry, identifier, and future-time metadata from
  the feature list; and
* persist the fitted estimator with the exact feature contract used to fit it.

``HistGradientBoostingClassifier`` natively handles missing numeric values.
Missingness is retained in the feature contract rather than silently imputed.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# The persisted feature contract gained ``split_group_column`` in v2.  A
# model bundle made under the older contract must not be mistaken for one
# whose holdout keeps all rows from a logical source snapshot together.
TABULAR_BASELINE_SCHEMA_VERSION = 2
TABULAR_BASELINE_VERSION = "tabular-hist-gradient-boosting/v2"
DEFAULT_SPLIT_FRACTION = 0.80
DEFAULT_CALIBRATION_BINS = 10


class TabularBaselineError(ValueError):
    """Raised when a table cannot be used for a leakage-safe baseline."""


# These fields occur in the current FEDS label and training-grid contracts.
# The policy is intentionally conservative: an identifier or source/label
# artifact should be used for lineage and grouping, never as a model input.
_LEAKAGE_EXACT_COLUMNS = frozenset(
    {
        "example_id",
        "cell_id",
        "cell_center_latitude",
        "cell_center_longitude",
        "anchor_at",
        "feature_cutoff_at",
        "target_end_at",
        "target_newly_burned_12h",
        "label_status",
        "label_observability",
        "label_tier",
        "label_source",
        "label_quality_score",
        "label_build_version",
        "positive_overlap_fraction",
        "source_snapshot_time",
        "target_snapshot_time",
        "source_time_semantics",
        "time_alignment_mode",
        "contributing_fire_count",
        "contributing_fires",
        "raw_artifact_id",
        "raw_artifact_ids",
        "normalized_artifact_id",
        "split",
        "fold",
    }
)
_LEAKAGE_PREFIXES = (
    "label_",
    "target_",
    "feds_",
    "raw_",
    "provenance_",
)
_LEAKAGE_SUBSTRINGS = (
    "artifact",
    "geometry",
    "perimeter",
    "snapshot",
    "provenance",
)
_LEAKAGE_SUFFIXES = ("_id", "_at", "_timestamp")


@dataclass(frozen=True)
class CalibrationBin:
    """Observed and predicted rates for one fixed probability bin."""

    lower_bound: float
    upper_bound: float
    count: int
    mean_predicted_probability: float | None
    observed_positive_rate: float | None


@dataclass(frozen=True)
class TabularBaselineMetrics:
    """Metrics calculated only on the chronologically later holdout set."""

    roc_auc: float
    pr_auc: float
    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    evaluation_positive_rate: float
    evaluation_row_count: int
    calibration_bins: tuple[CalibrationBin, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe metrics document."""
        document = asdict(self)
        document["calibration_bins"] = [asdict(item) for item in self.calibration_bins]
        return document


@dataclass(frozen=True)
class FeatureContract:
    """The input and chronological-split contract of a fitted model."""

    schema_version: int
    baseline_version: str
    target_column: str
    feature_columns: tuple[str, ...]
    anchor_column: str
    split_group_column: str
    split_fraction: float
    chronological_split_cutoff_at: str
    training_anchor_start_at: str
    training_anchor_end_at: str
    evaluation_anchor_start_at: str
    evaluation_anchor_end_at: str
    training_row_count: int
    evaluation_row_count: int
    training_positive_count: int
    evaluation_positive_count: int
    feature_dtypes: Mapping[str, str]
    training_missing_value_counts: Mapping[str, int]
    evaluation_missing_value_counts: Mapping[str, int]
    leakage_policy: str = "reject-label-source-geometry-id-and-future-metadata/v1"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe feature contract document."""
        document = asdict(self)
        document["feature_columns"] = list(self.feature_columns)
        document["feature_dtypes"] = dict(self.feature_dtypes)
        document["training_missing_value_counts"] = dict(self.training_missing_value_counts)
        document["evaluation_missing_value_counts"] = dict(self.evaluation_missing_value_counts)
        return document


@dataclass(frozen=True)
class TabularBaselineResult:
    """The fitted estimator, its input contract, and honest holdout metrics."""

    model: HistGradientBoostingClassifier
    feature_contract: FeatureContract
    metrics: TabularBaselineMetrics

    def model_bundle(self) -> dict[str, Any]:
        """Return the portable object stored by :func:`persist_tabular_baseline`."""
        return {
            "schema_version": TABULAR_BASELINE_SCHEMA_VERSION,
            "baseline_version": TABULAR_BASELINE_VERSION,
            "model": self.model,
            "feature_contract": self.feature_contract.as_dict(),
            "evaluation_metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class PersistedTabularBaseline:
    """Paths written by :func:`persist_tabular_baseline`."""

    model_bundle_path: Path
    feature_contract_path: Path
    metrics_path: Path


def leakage_metadata_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    """Return selected feature names that are forbidden metadata or leakage.

    This checks the *selected* feature list rather than trying to infer a
    schema from every dataframe column.  A training table should retain label
    and lineage fields, but callers must explicitly choose the eligible input
    variables.
    """
    offending = []
    for name in feature_columns:
        if not isinstance(name, str) or not name.strip():
            offending.append(str(name))
            continue
        normalized = name.strip().lower()
        if (
            normalized in _LEAKAGE_EXACT_COLUMNS
            or normalized.startswith(_LEAKAGE_PREFIXES)
            or normalized.endswith(_LEAKAGE_SUFFIXES)
            or any(token in normalized for token in _LEAKAGE_SUBSTRINGS)
        ):
            offending.append(name)
    return tuple(offending)


def train_tabular_baseline(
    examples: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Sequence[str],
    anchor_column: str = "anchor_at",
    split_group_column: str | None = None,
    split_fraction: float = DEFAULT_SPLIT_FRACTION,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    random_state: int = 0,
    model_parameters: Mapping[str, Any] | None = None,
) -> TabularBaselineResult:
    """Fit and evaluate a chronological HistGradientBoosting baseline.

    The split is made on distinct values of ``split_group_column`` (or
    ``anchor_column`` when it is omitted). All rows in one logical prediction
    instant stay together, so neighbouring cells cannot appear on both sides
    of the holdout. FEDS cells have locally estimated physical cutoffs, so its
    first baseline should pass ``source_snapshot_time`` as the split group.
    This is a temporal safeguard, not a substitute for the pipeline's future
    incident/region-held-out evaluation.

    ``feature_columns`` must be an explicit, ordered list of numeric inputs.
    Missing numeric values are accepted because the selected estimator handles
    them natively; infinity, labels, source snapshots, geometries, identifiers
    and raw timestamps are rejected.
    """
    _validate_dataframe(examples)
    target_name = _required_column_name(target_column, "target_column")
    anchor_name = _required_column_name(anchor_column, "anchor_column")
    split_group_name = _required_column_name(
        split_group_column if split_group_column is not None else anchor_name,
        "split_group_column",
    )
    selected_features = _validate_feature_columns(
        examples,
        target_column=target_name,
        anchor_column=anchor_name,
        split_group_column=split_group_name,
        feature_columns=feature_columns,
    )
    resolved_split_fraction = _validate_split_fraction(split_fraction)
    _validate_calibration_bins(calibration_bins)

    targets = _binary_target(examples[target_name], target_column=target_name)
    anchors = _anchors(examples[anchor_name], anchor_column=anchor_name)
    split_groups = _anchors(examples[split_group_name], anchor_column=split_group_name)
    feature_frame = _numeric_features(examples, selected_features)
    train_mask, evaluation_mask, cutoff_at = _chronological_masks(
        split_groups,
        split_fraction=resolved_split_fraction,
    )
    _require_both_classes(targets[train_mask], split_name="training")
    _require_both_classes(targets[evaluation_mask], split_name="evaluation")

    parameters: dict[str, Any] = {
        "learning_rate": 0.08,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": random_state,
    }
    if model_parameters is not None:
        if not isinstance(model_parameters, Mapping):
            raise TabularBaselineError("model_parameters must be a mapping when supplied")
        parameters.update(dict(model_parameters))

    model = HistGradientBoostingClassifier(**parameters)
    values = feature_frame.to_numpy(dtype=np.float64, copy=False)
    model.fit(values[train_mask], targets[train_mask])
    probabilities = model.predict_proba(values[evaluation_mask])[:, 1]
    metrics = _evaluation_metrics(
        targets[evaluation_mask], probabilities, calibration_bins=calibration_bins
    )
    contract = _feature_contract(
        feature_frame=feature_frame,
        targets=targets,
        anchors=anchors,
        train_mask=train_mask,
        evaluation_mask=evaluation_mask,
        target_column=target_name,
        feature_columns=selected_features,
        anchor_column=anchor_name,
        split_group_column=split_group_name,
        split_fraction=resolved_split_fraction,
        cutoff_at=cutoff_at,
    )
    return TabularBaselineResult(model=model, feature_contract=contract, metrics=metrics)


def persist_tabular_baseline(
    result: TabularBaselineResult,
    output_directory: str | Path,
    *,
    basename: str = "tabular_baseline",
) -> PersistedTabularBaseline:
    """Atomically persist a compressed joblib bundle plus readable contracts.

    The JSON files make model input requirements and evaluation results easy to
    inspect without unpickling code.  The joblib bundle is intended only for a
    trusted local environment because joblib/pickle files must never be loaded
    from an untrusted source.
    """
    if not isinstance(result, TabularBaselineResult):
        raise TypeError("result must be a TabularBaselineResult")
    stem = _safe_basename(basename)
    destination = Path(output_directory)
    if destination.exists() and not destination.is_dir():
        raise TabularBaselineError("output_directory must be a directory")
    destination.mkdir(parents=True, exist_ok=True)

    bundle_path = destination / f"{stem}.joblib"
    contract_path = destination / f"{stem}.feature-contract.json"
    metrics_path = destination / f"{stem}.metrics.json"
    _atomic_joblib_dump(bundle_path, result.model_bundle())
    _atomic_json_dump(contract_path, result.feature_contract.as_dict())
    _atomic_json_dump(metrics_path, result.metrics.as_dict())
    return PersistedTabularBaseline(
        model_bundle_path=bundle_path,
        feature_contract_path=contract_path,
        metrics_path=metrics_path,
    )


def _validate_dataframe(examples: pd.DataFrame) -> None:
    if not isinstance(examples, pd.DataFrame):
        raise TypeError("examples must be a pandas DataFrame")
    if examples.empty:
        raise TabularBaselineError("examples must not be empty")
    if not examples.columns.is_unique:
        raise TabularBaselineError("examples must have unique column names")


def _required_column_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TabularBaselineError(f"{label} must be a non-empty column name")
    return value.strip()


def _validate_feature_columns(
    examples: pd.DataFrame,
    *,
    target_column: str,
    anchor_column: str,
    split_group_column: str,
    feature_columns: Sequence[str],
) -> tuple[str, ...]:
    if target_column not in examples.columns:
        raise TabularBaselineError(f"examples is missing target column: {target_column}")
    if anchor_column not in examples.columns:
        raise TabularBaselineError(f"examples is missing anchor column: {anchor_column}")
    if split_group_column not in examples.columns:
        raise TabularBaselineError(
            f"examples is missing split_group column: {split_group_column}"
        )
    if isinstance(feature_columns, str) or not isinstance(feature_columns, Sequence):
        raise TabularBaselineError("feature_columns must be a non-empty sequence of column names")
    selected = tuple(_required_column_name(name, "feature column") for name in feature_columns)
    if not selected:
        raise TabularBaselineError("feature_columns must not be empty")
    if len(set(selected)) != len(selected):
        raise TabularBaselineError("feature_columns must not contain duplicates")
    missing = [name for name in selected if name not in examples.columns]
    if missing:
        raise TabularBaselineError(f"examples is missing feature column(s): {missing}")
    if target_column in selected:
        raise TabularBaselineError("target_column must not be included in feature_columns")
    if anchor_column in selected:
        raise TabularBaselineError("anchor_column must not be included in feature_columns")
    if split_group_column in selected:
        raise TabularBaselineError("split_group_column must not be included in feature_columns")
    leakage = leakage_metadata_columns(selected)
    if leakage:
        raise TabularBaselineError(
            "feature_columns contain leakage or metadata field(s): " + ", ".join(leakage)
        )
    return selected


def _binary_target(values: pd.Series, *, target_column: str) -> np.ndarray:
    if values.isna().any():
        raise TabularBaselineError(f"target column {target_column!r} contains missing values")
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise TabularBaselineError(f"target column {target_column!r} must be binary 0/1") from exc
    result = numeric.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(result).all() or not np.isin(result, (0.0, 1.0)).all():
        raise TabularBaselineError(f"target column {target_column!r} must contain only binary 0/1 values")
    return result.astype(np.int8, copy=False)


def _anchors(values: pd.Series, *, anchor_column: str) -> pd.Series:
    try:
        anchors = pd.to_datetime(values, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise TabularBaselineError(
            f"anchor column {anchor_column!r} must contain UTC-parseable timestamps"
        ) from exc
    if anchors.isna().any():
        raise TabularBaselineError(f"anchor column {anchor_column!r} contains missing timestamps")
    return pd.Series(anchors, index=values.index)


def _numeric_features(examples: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    temporal_features = [
        name
        for name in feature_columns
        if pd.api.types.is_datetime64_any_dtype(examples[name])
        or pd.api.types.is_timedelta64_dtype(examples[name])
    ]
    if temporal_features:
        raise TabularBaselineError(
            "feature_columns must not contain raw datetime or timedelta fields: "
            + ", ".join(temporal_features)
        )
    try:
        numeric = examples.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise TabularBaselineError("feature_columns must contain numeric values or missing values") from exc
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    if np.isinf(values).any():
        raise TabularBaselineError("feature_columns must not contain positive or negative infinity")
    return numeric.astype(np.float64)


def _chronological_masks(
    anchors: pd.Series, *, split_fraction: float
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    distinct = pd.DatetimeIndex(anchors.unique()).sort_values()
    if len(distinct) < 2:
        raise TabularBaselineError("chronological split requires at least two distinct anchor timestamps")
    training_time_count = math.floor(len(distinct) * split_fraction)
    if training_time_count < 1 or training_time_count >= len(distinct):
        raise TabularBaselineError(
            "split_fraction leaves no distinct anchor timestamp for training or evaluation"
        )
    cutoff = distinct[training_time_count - 1]
    train_mask = (anchors <= cutoff).to_numpy(dtype=bool, copy=False)
    evaluation_mask = (anchors > cutoff).to_numpy(dtype=bool, copy=False)
    if not train_mask.any() or not evaluation_mask.any():
        raise TabularBaselineError("chronological split produced an empty training or evaluation set")
    return train_mask, evaluation_mask, cutoff


def _validate_split_fraction(value: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TabularBaselineError("split_fraction must be between zero and one") from exc
    if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise TabularBaselineError("split_fraction must be between zero and one")
    return resolved


def _validate_calibration_bins(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 2:
        raise TabularBaselineError("calibration_bins must be an integer of at least two")


def _require_both_classes(targets: np.ndarray, *, split_name: str) -> None:
    values = np.unique(targets)
    if not np.array_equal(values, np.array([0, 1], dtype=targets.dtype)):
        raise TabularBaselineError(
            f"{split_name} split must contain both target classes; found {values.tolist()}"
        )


def _evaluation_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    calibration_bins: int,
) -> TabularBaselineMetrics:
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise RuntimeError("classifier returned invalid probability values")
    bins, expected_error, maximum_error = _calibration_bins(
        targets,
        probabilities,
        bin_count=calibration_bins,
    )
    return TabularBaselineMetrics(
        roc_auc=float(roc_auc_score(targets, probabilities)),
        # Average precision is the conventional, threshold-independent PR-AUC
        # summary for imbalanced binary examples.
        pr_auc=float(average_precision_score(targets, probabilities)),
        brier_score=float(brier_score_loss(targets, probabilities)),
        expected_calibration_error=expected_error,
        maximum_calibration_error=maximum_error,
        evaluation_positive_rate=float(np.mean(targets)),
        evaluation_row_count=int(len(targets)),
        calibration_bins=bins,
    )


def _calibration_bins(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    bin_count: int,
) -> tuple[tuple[CalibrationBin, ...], float, float]:
    assignment = np.minimum((probabilities * bin_count).astype(np.int64), bin_count - 1)
    result = []
    weighted_error = 0.0
    maximum_error = 0.0
    for index in range(bin_count):
        in_bin = assignment == index
        count = int(np.sum(in_bin))
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if count == 0:
            result.append(CalibrationBin(lower, upper, 0, None, None))
            continue
        mean_probability = float(np.mean(probabilities[in_bin]))
        observed_rate = float(np.mean(targets[in_bin]))
        error = abs(mean_probability - observed_rate)
        weighted_error += (count / len(targets)) * error
        maximum_error = max(maximum_error, error)
        result.append(
            CalibrationBin(
                lower,
                upper,
                count,
                mean_probability,
                observed_rate,
            )
        )
    return tuple(result), float(weighted_error), float(maximum_error)


def _feature_contract(
    *,
    feature_frame: pd.DataFrame,
    targets: np.ndarray,
    anchors: pd.Series,
    train_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    target_column: str,
    feature_columns: tuple[str, ...],
    anchor_column: str,
    split_group_column: str,
    split_fraction: float,
    cutoff_at: pd.Timestamp,
) -> FeatureContract:
    training_anchors = anchors[train_mask]
    evaluation_anchors = anchors[evaluation_mask]
    return FeatureContract(
        schema_version=TABULAR_BASELINE_SCHEMA_VERSION,
        baseline_version=TABULAR_BASELINE_VERSION,
        target_column=target_column,
        feature_columns=feature_columns,
        anchor_column=anchor_column,
        split_group_column=split_group_column,
        split_fraction=float(split_fraction),
        chronological_split_cutoff_at=_format_timestamp(cutoff_at),
        training_anchor_start_at=_format_timestamp(training_anchors.min()),
        training_anchor_end_at=_format_timestamp(training_anchors.max()),
        evaluation_anchor_start_at=_format_timestamp(evaluation_anchors.min()),
        evaluation_anchor_end_at=_format_timestamp(evaluation_anchors.max()),
        training_row_count=int(np.sum(train_mask)),
        evaluation_row_count=int(np.sum(evaluation_mask)),
        training_positive_count=int(np.sum(targets[train_mask])),
        evaluation_positive_count=int(np.sum(targets[evaluation_mask])),
        feature_dtypes={name: str(feature_frame[name].dtype) for name in feature_columns},
        training_missing_value_counts={
            name: int(feature_frame.loc[train_mask, name].isna().sum()) for name in feature_columns
        },
        evaluation_missing_value_counts={
            name: int(feature_frame.loc[evaluation_mask, name].isna().sum()) for name in feature_columns
        },
    )


def _format_timestamp(value: datetime | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _safe_basename(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TabularBaselineError("basename must be a non-empty filename stem")
    candidate = value.strip()
    if candidate in {".", ".."} or any(character in candidate for character in "/\\"):
        raise TabularBaselineError("basename must not contain a path separator")
    return candidate


def _atomic_joblib_dump(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.close(descriptor)
        joblib.dump(value, temporary_path, compress=3)
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_dump(path: Path, document: Mapping[str, Any]) -> None:
    encoded = json.dumps(document, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
