"""Assemble 1 km / 12 hour FEDS-positive training rows from retained evidence."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from .storage_budget import DEFAULT_POLICY_PATH, StorageBudgetError, load_storage_budget
from .training_dataset import (
    DEFAULT_FIRMS_AVAILABILITY_LAG,
    DEFAULT_FIRMS_LOOKBACK,
    DEFAULT_FIRMS_PRODUCTS,
    DEFAULT_FIRMS_REGION,
    DEFAULT_LABEL_BATCH_SIZE,
    DEFAULT_TERRAIN_CACHE_BLOCKS,
    TrainingDatasetError,
    build_and_store_feds_weak_positive_training_dataset,
)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Build immutable positive-only training rows; never infer weather or negatives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="first FEDS source-snapshot date")
    parser.add_argument("--end", required=True, type=_parse_date, help="last FEDS source-snapshot date")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--firms-lookback-hours",
        type=_positive_int,
        default=int(DEFAULT_FIRMS_LOOKBACK.total_seconds() / 3_600),
        help="trailing FIRMS evidence window used at each prediction cutoff",
    )
    parser.add_argument(
        "--firms-availability-lag-minutes",
        type=_nonnegative_int,
        default=int(DEFAULT_FIRMS_AVAILABILITY_LAG.total_seconds() / 60),
        help="conservative FIRMS availability lag applied before a detection is usable",
    )
    parser.add_argument(
        "--firms-product",
        action="append",
        dest="firms_products",
        help=(
            "FIRMS product whose daily coverage is required; repeat to override the "
            "three-platform default"
        ),
    )
    parser.add_argument(
        "--firms-region",
        default=DEFAULT_FIRMS_REGION,
        help="coverage-ledger region used by the FIRMS collection command",
    )
    parser.add_argument(
        "--label-batch-size",
        type=_positive_int,
        default=DEFAULT_LABEL_BATCH_SIZE,
        help="maximum labels held in memory while building one derived view",
    )
    parser.add_argument(
        "--max-cached-terrain-blocks",
        type=_nonnegative_int,
        default=DEFAULT_TERRAIN_CACHE_BLOCKS,
        help="bounded in-memory terrain source-block cache",
    )
    arguments = parser.parse_args(argv)
    try:
        result = build_and_store_feds_weak_positive_training_dataset(
            arguments.data_root,
            storage_budget=load_storage_budget(arguments.policy),
            start_date=arguments.start,
            end_date=arguments.end,
            firms_lookback=timedelta(hours=arguments.firms_lookback_hours),
            firms_availability_lag=timedelta(minutes=arguments.firms_availability_lag_minutes),
            firms_products=arguments.firms_products or DEFAULT_FIRMS_PRODUCTS,
            firms_region=arguments.firms_region,
            label_batch_size=arguments.label_batch_size,
            max_cached_terrain_blocks=arguments.max_cached_terrain_blocks,
        )
    except (StorageBudgetError, TrainingDatasetError) as exc:
        parser.error(str(exc))
    print(
        f"Assembled {result.training_row_count:,} positive-only training rows from "
        f"{result.input_label_count:,} FEDS labels into "
        f"{result.normalized_artifact_count:,} immutable artifacts. "
        "Weather is explicitly unavailable; no negatives were fabricated."
    )
    if result.manifest_path is not None:
        print(f"Published completed training-view manifest: {result.manifest_path}")
    return 0 if result.training_row_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
