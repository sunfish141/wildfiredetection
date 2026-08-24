"""Build one manifest-selected, no-weather FIRMS/FEDS candidate dataset."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from .candidate_dataset import (
    DEFAULT_CANDIDATE_RADIUS_CELLS,
    DEFAULT_MAX_WEAK_NEGATIVE_PROXIES_PER_SNAPSHOT,
    DEFAULT_TERRAIN_CACHE_BLOCKS,
    CandidateDatasetError,
    build_and_store_firms_candidate_dataset,
)
from .storage_budget import DEFAULT_POLICY_PATH, StorageBudgetError, load_storage_budget


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Build a no-weather candidate view; do not infer clear/no-burn labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="first FEDS source-snapshot date")
    parser.add_argument("--end", required=True, type=_parse_date, help="last FEDS source-snapshot date")
    parser.add_argument(
        "--split-start",
        type=_parse_date,
        help="first source date used to calculate the global chronological split; defaults to --start",
    )
    parser.add_argument(
        "--split-end",
        type=_parse_date,
        help="last source date used to calculate the global chronological split; defaults to --end",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--positive-view-manifest",
        help="completed positive-only training manifest; defaults to the newest valid manifest",
    )
    parser.add_argument(
        "--candidate-radius-cells",
        type=_nonnegative_int,
        default=DEFAULT_CANDIDATE_RADIUS_CELLS,
        help="square FIRMS seed radius in 1 km canonical cells",
    )
    parser.add_argument(
        "--max-weak-negative-proxies-per-snapshot",
        type=_nonnegative_int,
        default=DEFAULT_MAX_WEAK_NEGATIVE_PROXIES_PER_SNAPSHOT,
        help="deterministic cap; positives within FIRMS support are never capped",
    )
    parser.add_argument(
        "--max-cached-terrain-blocks",
        type=_nonnegative_int,
        default=DEFAULT_TERRAIN_CACHE_BLOCKS,
    )
    arguments = parser.parse_args(argv)
    try:
        result = build_and_store_firms_candidate_dataset(
            arguments.data_root,
            storage_budget=load_storage_budget(arguments.policy),
            start_date=arguments.start,
            end_date=arguments.end,
            positive_view_manifest=arguments.positive_view_manifest,
            split_start_date=arguments.split_start,
            split_end_date=arguments.split_end,
            radius_cells=arguments.candidate_radius_cells,
            max_weak_negative_proxies_per_snapshot=(
                arguments.max_weak_negative_proxies_per_snapshot
            ),
            max_cached_terrain_blocks=arguments.max_cached_terrain_blocks,
        )
    except (CandidateDatasetError, StorageBudgetError) as exc:
        parser.error(str(exc))
    print(
        f"Built {result.candidate_row_count:,} no-weather candidate rows: "
        f"{result.supported_positive_count:,} supported weak positives, "
        f"{result.weak_negative_proxy_count:,} weak-negative proxies, and "
        f"{result.unscored_positive_count:,} unscored positives."
    )
    print(f"Published completed candidate-view manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
