"""Backfill hourly Open-Meteo weather for one completed candidate view."""

from __future__ import annotations

import argparse
from datetime import date

from .open_meteo_historical import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TILE_DISTANCE_METRES,
    DEFAULT_REQUESTS_PER_MINUTE,
    OpenMeteoHistoricalWeatherError,
    backfill_open_meteo_historical_weather,
)
from .storage_budget import DEFAULT_POLICY_PATH, StorageBudgetError, load_storage_budget


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


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Collect weather at the hourly candidate anchor, honoring rate limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--candidate-manifest",
        required=True,
        help="one completed base candidate-view manifest; weather is bound to this exact file",
    )
    parser.add_argument("--start", type=_parse_date)
    parser.add_argument("--end", type=_parse_date)
    parser.add_argument(
        "--resume-manifest",
        help="partial historical-weather backfill manifest to continue without re-requesting completed dates",
    )
    parser.add_argument("--model", default="ecmwf_ifs")
    parser.add_argument(
        "--max-tile-distance-m",
        type=_positive_float,
        default=DEFAULT_MAX_TILE_DISTANCE_METRES,
    )
    parser.add_argument(
        "--requests-per-minute",
        type=_positive_int,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help="Open-Meteo location units per minute; batches are charged by coordinate count",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    arguments = parser.parse_args(argv)
    try:
        result = backfill_open_meteo_historical_weather(
            arguments.data_root,
            storage_policy=load_storage_budget(arguments.policy),
            candidate_manifest=arguments.candidate_manifest,
            start_date=arguments.start,
            end_date=arguments.end,
            resume_manifest=arguments.resume_manifest,
            model=arguments.model,
            max_tile_distance_m=arguments.max_tile_distance_m,
            requests_per_minute=arguments.requests_per_minute,
            batch_size=arguments.batch_size,
        )
    except (OpenMeteoHistoricalWeatherError, StorageBudgetError, ValueError) as exc:
        parser.error(str(exc))
    state = "complete" if result.complete else "partial"
    print(
        f"Historical weather backfill {state}: {result.weather_date_count:,} date reports, "
        f"{result.captured_tile_count:,} tiles, {result.measurement_count:,} measurements, "
        f"and {result.api_call_units:,} location units."
    )
    print(f"Backfill manifest: {result.manifest_path}")
    return 0 if result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
