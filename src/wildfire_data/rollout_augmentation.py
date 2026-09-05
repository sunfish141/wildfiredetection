"""Generate inspected one-step synthetic states without fitting a new model.

For each consecutive pair of training snapshots, the current recursive model
starts from observed FIRMS state at the first snapshot and advances once.  The
next predicted frontier is intersected with the next historical snapshot's
no-centre label domain. Only that intersection becomes an augmentation row;
unreached labels and predictions outside the historical domain remain explicit
coverage diagnostics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from .recursive_calibration import fit_observation_calibration, validate_training_split

from .recursive_transition import (
    RECURSIVE_MODEL_FEATURE_COLUMNS,
    RecursiveTransitionModel,
)
from .rollout_evaluation import initial_state_from_observed_firms
from .rollout_sequences import RolloutSequence, build_rollout_sequences, snapshot_frame
from .terrain_features import TerrainFeatureSampler
from .train_recursive_transition import _verify_candidate_checksum


ROLLOUT_AUGMENTATION_VERSION = "one-step-synthetic-frontier-augmentation/v2"
AUGMENTATION_FILENAME = "synthetic_frontier_examples.csv.gz"
MANIFEST_FILENAME = "manifest.json"
BASE_COLUMNS = (
    "source_snapshot_time",
    "target_snapshot_time",
    "anchor_at",
    "target_end_at",
    "cell_id",
    "example_id",
    "dataset_split",
    "target_newly_burned_12h",
    "firms_center_has_detection",
    "firms_center_bright_ti4_max",
    "firms_center_hours_since_last_detection",
    "firms_center_detection_count",
    "firms_center_platform_count",
)
AUGMENTATION_COLUMNS = (
    "augmentation_version",
    "synthetic_state_steps",
    "generation_origin_source_snapshot_time",
    "synthetic_feature_source_snapshot_time",
    "original_example_id",
    "dataset_split",
    "anchor_at",
    "target_end_at",
    "cell_id",
    "target_newly_burned_12h",
    *RECURSIVE_MODEL_FEATURE_COLUMNS,
)


class RolloutAugmentationError(ValueError):
    """Raised when training snapshots cannot produce safe augmentation rows."""


@dataclass(frozen=True)
class AugmentationPairReport:
    """Coverage diagnostics for one observed-to-synthetic snapshot pair."""

    origin_source_snapshot_time: str
    synthetic_feature_source_snapshot_time: str
    origin_observed_active_cell_count: int
    predicted_active_cell_count: int
    model_frontier_candidate_count: int
    historical_frontier_row_count: int
    historical_frontier_positive_count: int
    matched_row_count: int
    matched_positive_count: int
    unreached_row_count: int
    unreached_positive_count: int
    candidates_outside_historical_frontier: int


@dataclass(frozen=True)
class RolloutAugmentationResult:
    """Generated rows and their complete training-only coverage report."""

    examples: pd.DataFrame
    split_name: str
    pair_reports: tuple[AugmentationPairReport, ...]
    observed_feature_rows: pd.DataFrame
    transition_contract: dict[str, Any]

    @property
    def pair_count(self) -> int:
        return len(self.pair_reports)


def generate_one_step_augmentation(
    model: RecursiveTransitionModel,
    examples: pd.DataFrame,
    sequences: Sequence[RolloutSequence],
    *,
    terrain_provider: Any,
    split_name: str = "train",
) -> RolloutAugmentationResult:
    """Generate on-policy feature rows from consecutive training snapshots."""
    if not isinstance(model, RecursiveTransitionModel):
        raise TypeError("model must be a RecursiveTransitionModel")
    if not isinstance(examples, pd.DataFrame):
        raise TypeError("examples must be a pandas DataFrame")
    if not callable(terrain_provider):
        raise TypeError("terrain_provider must be callable")
    if split_name != "train":
        raise RolloutAugmentationError("augmentation is restricted to the training split")
    required = set(BASE_COLUMNS).union(RECURSIVE_MODEL_FEATURE_COLUMNS)
    missing = sorted(required - set(examples.columns))
    if missing:
        raise RolloutAugmentationError(
            "examples is missing required columns: " + ", ".join(missing)
        )
    validate_training_split(examples)
    if not examples["example_id"].is_unique or examples["example_id"].isna().any():
        raise RolloutAugmentationError("original example IDs must be unique and present")
    # Rebuild from the actual frame: externally supplied row positions must not
    # allow a skipped snapshot or a larger-than-12-hour jump to masquerade as a pair.
    if tuple(sequences) != build_rollout_sequences(examples):
        raise RolloutAugmentationError("sequences do not match the source examples")

    generated_rows: list[dict[str, Any]] = []
    observed_rows: list[dict[str, Any]] = []
    reports = []
    for sequence in sequences:
        for origin_snapshot, next_snapshot in zip(
            sequence.snapshots, sequence.snapshots[1:]
        ):
            origin_frame = snapshot_frame(examples, origin_snapshot)
            next_frame = snapshot_frame(examples, next_snapshot)
            origin_split = set(origin_frame["dataset_split"].dropna().astype(str))
            next_split = set(next_frame["dataset_split"].dropna().astype(str))
            if len(origin_split) != 1 or len(next_split) != 1:
                raise RolloutAugmentationError("one source snapshot contains mixed dataset splits")
            if origin_split != {split_name} or next_split != {split_name}:
                continue

            observed_state = initial_state_from_observed_firms(model, origin_frame)
            predicted_step = model.step(observed_state, terrain_provider=terrain_provider)
            predicted_state = predicted_step.state
            model_frontier = model.candidate_cells(predicted_state)
            model_frontier_ids = {cell.cell_id for cell in model_frontier}

            center = pd.to_numeric(next_frame["firms_center_has_detection"], errors="raise")
            if center.isna().any() or not center.isin([0, 1]).all():
                raise RolloutAugmentationError(
                    "firms_center_has_detection must contain binary 0/1 values"
                )
            historical_frontier = next_frame.loc[center == 0]
            if not historical_frontier["cell_id"].astype(str).is_unique:
                raise RolloutAugmentationError(
                    "historical frontier contains duplicate cell IDs in one snapshot"
                )
            historical_ids = set(historical_frontier["cell_id"].astype(str))
            matched_ids = model_frontier_ids.intersection(historical_ids)
            matched = historical_frontier.loc[
                historical_frontier["cell_id"].astype(str).isin(matched_ids)
            ]
            terrain_by_cell = {
                str(row["cell_id"]): {name: row[name] for name in _terrain_columns()}
                for row in matched.to_dict(orient="records")
            }

            def matched_terrain(cell_id: str) -> Mapping[str, Any]:
                return terrain_by_cell[cell_id]

            feature_rows = model.candidate_feature_rows(
                predicted_state,
                terrain_provider=matched_terrain,
                include_cell_ids=matched_ids,
            )
            historical_by_cell = {
                str(row["cell_id"]): row for row in matched.to_dict(orient="records")
            }
            for cell, features in feature_rows:
                historical = historical_by_cell[cell.cell_id]
                generated_rows.append(
                    {
                        "augmentation_version": ROLLOUT_AUGMENTATION_VERSION,
                        "synthetic_state_steps": 1,
                        "generation_origin_source_snapshot_time": _format_timestamp(
                            origin_snapshot.source_snapshot_time
                        ),
                        "synthetic_feature_source_snapshot_time": _format_timestamp(
                            next_snapshot.source_snapshot_time
                        ),
                        "original_example_id": str(historical["example_id"]),
                        "dataset_split": "train",
                        "anchor_at": historical["anchor_at"],
                        "target_end_at": historical["target_end_at"],
                        "cell_id": cell.cell_id,
                        "target_newly_burned_12h": int(
                            historical["target_newly_burned_12h"]
                        ),
                        **features,
                    }
                )
                observed_rows.append(
                    {name: historical[name] for name in RECURSIVE_MODEL_FEATURE_COLUMNS}
                )

            targets = pd.to_numeric(
                historical_frontier["target_newly_burned_12h"], errors="raise"
            )
            if targets.isna().any() or not targets.isin([0, 1]).all():
                raise RolloutAugmentationError(
                    "target_newly_burned_12h must contain binary 0/1 values"
                )
            matched_targets = pd.to_numeric(
                matched["target_newly_burned_12h"], errors="raise"
            )
            historical_positive_count = int(targets.sum())
            matched_positive_count = int(matched_targets.sum())
            reports.append(
                AugmentationPairReport(
                    origin_source_snapshot_time=_format_timestamp(
                        origin_snapshot.source_snapshot_time
                    ),
                    synthetic_feature_source_snapshot_time=_format_timestamp(
                        next_snapshot.source_snapshot_time
                    ),
                    origin_observed_active_cell_count=len(observed_state.active_cells),
                    predicted_active_cell_count=len(predicted_state.active_cells),
                    model_frontier_candidate_count=len(model_frontier_ids),
                    historical_frontier_row_count=len(historical_frontier),
                    historical_frontier_positive_count=historical_positive_count,
                    matched_row_count=len(matched_ids),
                    matched_positive_count=matched_positive_count,
                    unreached_row_count=len(historical_ids - model_frontier_ids),
                    unreached_positive_count=(
                        historical_positive_count - matched_positive_count
                    ),
                    candidates_outside_historical_frontier=len(
                        model_frontier_ids - historical_ids
                    ),
                )
            )

    if not reports:
        raise RolloutAugmentationError(
            f"no consecutive snapshot pairs belong entirely to split {split_name!r}"
        )
    generated = pd.DataFrame(generated_rows, columns=AUGMENTATION_COLUMNS)
    observed = pd.DataFrame(observed_rows, columns=RECURSIVE_MODEL_FEATURE_COLUMNS)
    return RolloutAugmentationResult(
        examples=generated,
        split_name=split_name,
        pair_reports=tuple(reports),
        observed_feature_rows=observed,
        transition_contract=model.transition_contract(),
    )


def persist_rollout_augmentation(
    result: RolloutAugmentationResult,
    output_directory: str | Path,
    *,
    release_manifest_sha256: str,
    model_bundle_path: str,
    reference_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically persist generated CSV rows and an inspection manifest."""
    if not isinstance(result, RolloutAugmentationResult):
        raise TypeError("result must be a RolloutAugmentationResult")
    if result.split_name != "train" or not result.examples["dataset_split"].eq("train").all():
        raise RolloutAugmentationError("only training augmentation may be persisted")
    calibration = result.transition_contract.get("observation_calibration")
    if calibration and calibration["source_release_manifest_sha256"] != release_manifest_sha256:
        raise RolloutAugmentationError("calibration and inspection must use the same source release")
    output = Path(output_directory)
    if output.exists():
        raise RolloutAugmentationError("output directory must be new; inspection artifacts are immutable")
    model_sha256 = _sha256_file(Path(model_bundle_path))
    if (reference_manifest is not None and
            reference_manifest.get("source_release_manifest_sha256") != release_manifest_sha256):
        raise RolloutAugmentationError("reference inspection uses a different source release")
    renderer_comparison = compare_renderer_features(result.examples, result.observed_feature_rows)
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / AUGMENTATION_FILENAME
    _atomic_csv_gzip(csv_path, result.examples)
    reports = result.pair_reports
    historical_rows = sum(report.historical_frontier_row_count for report in reports)
    historical_positives = sum(
        report.historical_frontier_positive_count for report in reports
    )
    matched_positives = sum(report.matched_positive_count for report in reports)
    manifest = {
        "schema_version": 2,
        "kind": "wildfire-one-step-rollout-augmentation",
        "augmentation_version": ROLLOUT_AUGMENTATION_VERSION,
        "split_name": result.split_name,
        "source_release_manifest_sha256": release_manifest_sha256,
        "source_model_bundle": model_bundle_path,
        "source_model_bundle_sha256": model_sha256,
        "transition_contract": result.transition_contract,
        "pair_count": result.pair_count,
        "generated_row_count": len(result.examples),
        "generated_positive_count": int(result.examples["target_newly_burned_12h"].sum()),
        "generated_positive_rate": _safe_ratio(
            int(result.examples["target_newly_burned_12h"].sum()), len(result.examples)
        ),
        "historical_frontier_row_count": historical_rows,
        "historical_frontier_positive_count": historical_positives,
        "historical_frontier_positive_rate": _safe_ratio(historical_positives, historical_rows),
        "frontier_coverage": _safe_ratio(len(result.examples), historical_rows),
        "positive_frontier_coverage": _safe_ratio(matched_positives, historical_positives),
        "model_frontier_candidate_count": sum(
            report.model_frontier_candidate_count for report in reports
        ),
        "candidates_outside_historical_frontier": sum(
            report.candidates_outside_historical_frontier for report in reports
        ),
        "columns": list(AUGMENTATION_COLUMNS),
        "feature_columns": list(RECURSIVE_MODEL_FEATURE_COLUMNS),
        "synthetic_feature_summary": _feature_summary(
            result.examples, RECURSIVE_MODEL_FEATURE_COLUMNS
        ),
        "observed_matched_feature_summary": _feature_summary(
            result.observed_feature_rows, RECURSIVE_MODEL_FEATURE_COLUMNS
        ),
        "pair_reports": [asdict(report) for report in reports],
        "artifact": {
            "path": csv_path.as_posix(),
            "file_sha256": _sha256_file(csv_path),
            "byte_count": csv_path.stat().st_size,
        },
        "training_admitted": False,
        "renderer_comparison": renderer_comparison,
    }
    if reference_manifest is not None:
        manifest["comparison_to_previous_inspection"] = {
            name: {"previous": reference_manifest[name], "current": manifest[name],
                   "delta": manifest[name] - reference_manifest[name]}
            for name in ("generated_row_count", "generated_positive_rate", "frontier_coverage",
                         "positive_frontier_coverage", "candidates_outside_historical_frontier")
        }
    _atomic_json(output / MANIFEST_FILENAME, manifest)
    return manifest


def compare_renderer_features(
    synthetic: pd.DataFrame, observed: pd.DataFrame
) -> dict[str, Any]:
    """Predeclared engineering screen, not a test of operational skill.

    Each FIRMS field must have <=10 percentage points of missingness drift
    and <=0.5 observed standard deviations of mean or quartile drift. All
    observed statistics are from matched TRAINING rows. An empty comparison
    fails closed. Passing this screen does not itself admit training data.
    """
    if len(synthetic) != len(observed):
        raise RolloutAugmentationError("renderer comparison requires paired rows")
    fields = {}
    for name in RECURSIVE_MODEL_FEATURE_COLUMNS:
        if not name.startswith("firms_"):
            continue
        left = pd.to_numeric(synthetic[name], errors="raise").astype(float)
        right = pd.to_numeric(observed[name], errors="raise").astype(float)
        if np.isinf(left.to_numpy(dtype=float)).any() or np.isinf(right.to_numpy(dtype=float)).any():
            raise RolloutAugmentationError("renderer comparison features must be finite or missing")
        missing_gap = abs(float(left.isna().mean()) - float(right.isna().mean())) if len(left) else None
        mean_gap = quantile_gap = None
        if left.notna().any() and right.notna().any():
            scale = max(float(right.std(ddof=0)), 1e-12)
            mean_gap = abs(float(left.mean()) - float(right.mean())) / scale
            quantile_gap = float((left.quantile([.25, .5, .75]) -
                                  right.quantile([.25, .5, .75])).abs().max()) / scale
        both_missing = len(left) > 0 and left.isna().all() and right.isna().all()
        passed = bool(both_missing or (
            mean_gap is not None and missing_gap <= .1 and mean_gap <= .5 and quantile_gap <= .5
        ))
        fields[name] = {"missing_fraction_gap": missing_gap, "standardized_mean_gap": mean_gap,
                        "maximum_standardized_quartile_gap": quantile_gap, "passed": passed}
    return {
        "policy_version": "matched-training-renderer-screen/v1",
        "maximum_missing_fraction_gap": .1,
        "maximum_standardized_mean_or_quartile_gap": .5,
        "fields": fields,
        "passed": all(field["passed"] for field in fields.values()),
        "training_admission": "separate controlled experiment required even if screen passes",
    }


def _feature_summary(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> dict[str, dict[str, Any]]:
    summary = {}
    for name in feature_columns:
        values = pd.to_numeric(frame[name], errors="raise").astype(float)
        present = values.dropna()
        summary[name] = {
            "row_count": len(values),
            "missing_count": int(values.isna().sum()),
            "minimum": float(present.min()) if not present.empty else None,
            "mean": float(present.mean()) if not present.empty else None,
            "maximum": float(present.max()) if not present.empty else None,
            "standard_deviation": float(present.std(ddof=0)) if not present.empty else None,
            "quartiles": [float(present.quantile(q)) for q in (.25, .5, .75)] if not present.empty else None,
        }
    return summary


def _terrain_columns() -> tuple[str, ...]:
    return tuple(name for name in RECURSIVE_MODEL_FEATURE_COLUMNS if name.startswith("terrain_"))


def _format_timestamp(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).isoformat().replace("+00:00", "Z")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    os.close(descriptor)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                frame.to_csv(compressed, index=False, lineterminator="\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
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
    parser.add_argument("--reference-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    release_manifest_path = args.release / "dataset_manifest.json"
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    if (release_manifest.get("kind") != "wildfire-spread-candidate-dataset-release"
            or release_manifest.get("schema_version") != 2
            or release_manifest.get("weather", {}).get("available") is not False):
        raise RolloutAugmentationError("requires a completed schema-v2 no-weather release")
    if args.output.exists():
        raise RolloutAugmentationError("output directory must be new; use a new experiment path")
    _verify_candidate_checksum(args.release, args.release / "candidate_examples.csv.gz")
    columns = list(dict.fromkeys((*BASE_COLUMNS, *RECURSIVE_MODEL_FEATURE_COLUMNS)))
    examples = pd.read_csv(
        args.release / "candidate_examples.csv.gz", compression="gzip", usecols=columns
    )
    if len(examples) != release_manifest.get("candidate_row_count"):
        raise RolloutAugmentationError(
            "candidate CSV row count differs from the release manifest"
        )
    sequences = build_rollout_sequences(examples)
    calibration = fit_observation_calibration(
        examples, release_manifest_sha256=_sha256_file(release_manifest_path)
    )
    model = RecursiveTransitionModel.from_model_bundle(
        args.model_bundle, observation_calibration=calibration
    )
    sampler = TerrainFeatureSampler(args.data_root, max_cached_blocks=8)

    @lru_cache(maxsize=200_000)
    def cached_terrain(cell_id: str) -> Mapping[str, Any]:
        return sampler.sample_cell(cell_id)

    result = generate_one_step_augmentation(
        model,
        examples,
        sequences,
        terrain_provider=cached_terrain,
        split_name="train",
    )
    manifest = persist_rollout_augmentation(
        result,
        args.output,
        release_manifest_sha256=_sha256_file(release_manifest_path),
        model_bundle_path=args.model_bundle.as_posix(),
        reference_manifest=(json.loads(args.reference_manifest.read_text())
                            if args.reference_manifest else None),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
