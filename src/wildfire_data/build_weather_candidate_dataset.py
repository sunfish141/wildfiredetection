"""Join a complete historical-weather backfill to one candidate view."""

from __future__ import annotations

import argparse

from .storage_budget import DEFAULT_POLICY_PATH, StorageBudgetError, load_storage_budget
from .weather_candidate_dataset import (
    WeatherCandidateDatasetError,
    build_weather_candidate_dataset,
)


def main(argv: list[str] | None = None) -> int:
    """Publish a separate weather-bearing candidate manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--weather-backfill-manifest", required=True)
    parser.add_argument(
        "--candidate-manifest",
        required=True,
        help="the exact completed base candidate-view manifest used for weather backfill",
    )
    arguments = parser.parse_args(argv)
    try:
        result = build_weather_candidate_dataset(
            arguments.data_root,
            storage_budget=load_storage_budget(arguments.policy),
            weather_backfill_manifest=arguments.weather_backfill_manifest,
            candidate_manifest=arguments.candidate_manifest,
        )
    except (WeatherCandidateDatasetError, StorageBudgetError) as exc:
        parser.error(str(exc))
    print(
        f"Built {result.candidate_row_count:,} weather-bearing candidate rows across "
        f"{result.weather_date_count:,} UTC dates."
    )
    print(f"Completed weather candidate manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
