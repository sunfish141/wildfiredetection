"""Train and persist the no-centre-detection recursive frontier baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .recursive_transition import (
    DEFAULT_ACTIVE_DURATION_STEPS,
    DEFAULT_IGNITION_THRESHOLD,
    DEFAULT_INTENSITY_RETENTION,
    RECURSIVE_MODEL_FEATURE_COLUMNS,
    RECURSIVE_TRANSITION_VERSION,
)
from .tabular_baseline import (
    TabularBaselineError,
    TabularBaselineResult,
    persist_tabular_baseline,
    train_tabular_baseline,
)


TARGET_COLUMN = "target_newly_burned_12h"
ANCHOR_COLUMN = "anchor_at"
SPLIT_GROUP_COLUMN = "source_snapshot_time"
ROW_FILTER_COLUMN = "firms_center_has_detection"
ROW_FILTER_POLICY = "unburned-frontier-center-has-no-detection/v1"


@dataclass(frozen=True)
class RecursiveFrontierTrainingResult:
    """Filtered-row counts and the fitted chronological baseline."""

    source_row_count: int
    frontier_row_count: int
    excluded_center_detection_row_count: int
    baseline: TabularBaselineResult


def train_recursive_frontier_baseline(examples: pd.DataFrame) -> RecursiveFrontierTrainingResult:
    """Fit only rows that match an unburned cell at recursive inference time."""
    if not isinstance(examples, pd.DataFrame):
        raise TypeError("examples must be a pandas DataFrame")
    if ROW_FILTER_COLUMN not in examples.columns:
        raise TabularBaselineError(f"examples is missing row-filter column: {ROW_FILTER_COLUMN}")
    center_detection = pd.to_numeric(examples[ROW_FILTER_COLUMN], errors="raise")
    if center_detection.isna().any() or not center_detection.isin([0, 1]).all():
        raise TabularBaselineError(f"{ROW_FILTER_COLUMN} must contain only binary 0/1 values")
    frontier = examples.loc[center_detection == 0].reset_index(drop=True)
    if frontier.empty:
        raise TabularBaselineError("recursive frontier filter produced no rows")
    baseline = train_tabular_baseline(
        frontier,
        target_column=TARGET_COLUMN,
        feature_columns=RECURSIVE_MODEL_FEATURE_COLUMNS,
        anchor_column=ANCHOR_COLUMN,
        split_group_column=SPLIT_GROUP_COLUMN,
        random_state=0,
    )
    return RecursiveFrontierTrainingResult(
        source_row_count=len(examples),
        frontier_row_count=len(frontier),
        excluded_center_detection_row_count=int((center_detection == 1).sum()),
        baseline=baseline,
    )


def train_release(
    release_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Train from one completed schema-v2 release and persist its contracts."""
    release = Path(release_directory)
    manifest_path = release / "dataset_manifest.json"
    candidate_path = release / "candidate_examples.csv.gz"
    manifest = _json_object(manifest_path, "dataset manifest")
    if manifest.get("kind") != "wildfire-spread-candidate-dataset-release":
        raise TabularBaselineError("release has an unexpected manifest kind")
    if manifest.get("schema_version") != 2:
        raise TabularBaselineError("recursive baseline requires a schema-v2 release")
    if manifest.get("weather", {}).get("available") is not False:
        raise TabularBaselineError("recursive baseline expects the completed no-weather release")
    if not set(RECURSIVE_MODEL_FEATURE_COLUMNS).issubset(manifest.get("model_feature_columns", [])):
        raise TabularBaselineError("release is missing recursive frontier features")
    _verify_candidate_checksum(release, candidate_path)

    columns = [
        TARGET_COLUMN,
        ANCHOR_COLUMN,
        SPLIT_GROUP_COLUMN,
        ROW_FILTER_COLUMN,
        *RECURSIVE_MODEL_FEATURE_COLUMNS,
    ]
    examples = pd.read_csv(candidate_path, compression="gzip", usecols=columns)
    if len(examples) != manifest.get("candidate_row_count"):
        raise TabularBaselineError("candidate CSV row count differs from its release manifest")
    trained = train_recursive_frontier_baseline(examples)
    output = Path(output_directory)
    persisted = persist_tabular_baseline(
        trained.baseline,
        output,
        basename="recursive_frontier_baseline",
    )
    transition_contract = {
        "schema_version": 1,
        "transition_version": RECURSIVE_TRANSITION_VERSION,
        "time_step_hours": 12,
        "row_filter_column": ROW_FILTER_COLUMN,
        "row_filter_value": 0,
        "row_filter_policy": ROW_FILTER_POLICY,
        "feature_columns": list(RECURSIVE_MODEL_FEATURE_COLUMNS),
        "ignition_threshold": DEFAULT_IGNITION_THRESHOLD,
        "active_duration_steps": DEFAULT_ACTIVE_DURATION_STEPS,
        "intensity_retention": DEFAULT_INTENSITY_RETENTION,
        "intensity_and_persistence_are_learned": False,
    }
    transition_contract_path = output / "recursive_transition.contract.json"
    _atomic_json(transition_contract_path, transition_contract)
    run_manifest = {
        "schema_version": 1,
        "kind": "wildfire-recursive-frontier-baseline-run",
        "release": {
            "directory": release.as_posix(),
            "dataset_manifest_sha256": _sha256_file(manifest_path),
            "candidate_build_id": manifest.get("candidate_build_id"),
            "candidate_row_count": manifest.get("candidate_row_count"),
        },
        "training": {
            "source_row_count": trained.source_row_count,
            "frontier_row_count": trained.frontier_row_count,
            "excluded_center_detection_row_count": trained.excluded_center_detection_row_count,
            "row_filter_policy": ROW_FILTER_POLICY,
            "feature_columns": list(RECURSIVE_MODEL_FEATURE_COLUMNS),
        },
        "feature_contract": trained.baseline.feature_contract.as_dict(),
        "evaluation_metrics": trained.baseline.metrics.as_dict(),
        "transition_contract": transition_contract,
        "artifacts": {
            "model_bundle": persisted.model_bundle_path.as_posix(),
            "feature_contract": persisted.feature_contract_path.as_posix(),
            "metrics": persisted.metrics_path.as_posix(),
            "transition_contract": transition_contract_path.as_posix(),
        },
    }
    run_manifest_path = output / "run_manifest.json"
    _atomic_json(run_manifest_path, run_manifest)
    return run_manifest


def _verify_candidate_checksum(release: Path, candidate_path: Path) -> None:
    checksum_path = release / "SHA256SUMS"
    entries = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator:
            entries[name] = digest
    expected = entries.get(candidate_path.name)
    if expected is None or _sha256_file(candidate_path) != expected:
        raise TabularBaselineError("candidate CSV is missing from or disagrees with SHA256SUMS")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TabularBaselineError(f"could not read {label}: {path}") from exc
    if not isinstance(document, dict):
        raise TabularBaselineError(f"{label} must be a JSON object")
    return document


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
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
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = train_release(args.release, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
