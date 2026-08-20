"""Create a scored, leakage-safe compact weather-tile plan from FIRMS evidence."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from .forecast_tile_planning import (
    DEFAULT_ANCHOR_HOURS,
    DEFAULT_FIRMS_AVAILABILITY_LAG_MINUTES,
    DEFAULT_HRDPS_RUN_HOURS,
    DEFAULT_HRDPS_TILES_PER_RUN,
    DEFAULT_TILE_KILOMETRES,
    iter_normalized_firms_detections,
    plan_forecast_tiles,
    write_forecast_tile_plan,
)
from .storage_budget import DEFAULT_POLICY_PATH, load_storage_budget


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: list[str] | None = None) -> int:
    """Plan compact forecast tile candidates without downloading weather values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="inclusive model-run UTC date")
    parser.add_argument("--end", required=True, type=_parse_date, help="inclusive model-run UTC date")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model", default="hrdps")
    parser.add_argument("--run-hour", type=int, action="append", dest="run_hours")
    parser.add_argument("--max-tiles-per-run", type=int, default=DEFAULT_HRDPS_TILES_PER_RUN)
    parser.add_argument("--tile-kilometres", type=float, default=DEFAULT_TILE_KILOMETRES)
    parser.add_argument("--anchor-hours", type=int, default=DEFAULT_ANCHOR_HOURS)
    parser.add_argument(
        "--firms-availability-lag-minutes",
        type=int,
        default=DEFAULT_FIRMS_AVAILABILITY_LAG_MINUTES,
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    plan = plan_forecast_tiles(
        iter_normalized_firms_detections(
            arguments.data_root,
            start_date=arguments.start - timedelta(days=1),
            end_date=arguments.end,
        ),
        model=arguments.model,
        start_date=arguments.start,
        end_date=arguments.end,
        run_hours=arguments.run_hours or DEFAULT_HRDPS_RUN_HOURS,
        max_tiles_per_run=arguments.max_tiles_per_run,
        tile_kilometres=arguments.tile_kilometres,
        anchor_hours=arguments.anchor_hours,
        availability_lag_minutes=arguments.firms_availability_lag_minutes,
    )
    selected = sum(candidate.selected for candidate in plan)
    if arguments.dry_run:
        print(
            f"Planned {len(plan):,} weather-tile candidates; {selected:,} are selected. "
            "No plan file was written."
        )
        return 0
    default_output = (
        Path(arguments.data_root)
        / "weather"
        / "forecast-tile-plans"
        / f"{arguments.model}_{arguments.start:%Y%m%d}_{arguments.end:%Y%m%d}.csv.gz"
    )
    output = write_forecast_tile_plan(
        arguments.data_root,
        plan=plan,
        output_path=arguments.output or default_output,
        storage_budget=load_storage_budget(arguments.policy),
    )
    print(
        f"Wrote {len(plan):,} scored weather-tile candidates; {selected:,} are selected: {output}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
