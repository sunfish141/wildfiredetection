"""Inventory VIIRS L2 fire products; legacy fire-file downloads require an override."""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from .viirs_l2_observability import (
    DEFAULT_BBOX,
    DEFAULT_PLATFORMS,
    DEFAULT_REGION,
    collect_viirs_l2_range,
)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: list[str] | None = None) -> int:
    """Run a resumable CMR-discovered L2 archive collection."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path("config/.env"))
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="inclusive UTC date")
    parser.add_argument("--end", required=True, type=_parse_date, help="inclusive UTC date")
    parser.add_argument("--archive-root", default="data")
    parser.add_argument("--bbox", default=DEFAULT_BBOX)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        choices=DEFAULT_PLATFORMS,
        help="repeat for each satellite; defaults to SNPP, NOAA-20, and NOAA-21",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="archive only CMR fire-file inventory evidence; do not download protected NetCDF files",
    )
    parser.add_argument(
        "--legacy-fire-files-only",
        action="store_true",
        help=(
            "allow the legacy collector to download fire-mask/QA files without their required "
            "geolocation partners; incompatible with the compact 20 GB local policy"
        ),
    )
    arguments = parser.parse_args(argv)
    earthdata_token = (os.getenv("EARTHDATA_TOKEN") or "").strip()
    if not arguments.dry_run and not earthdata_token:
        parser.error(
            "Set EARTHDATA_TOKEN before downloading Level-2 granules, or use --dry-run first."
        )
    if not arguments.dry_run and not arguments.legacy_fire_files_only:
        parser.error(
            "The compact 20 GB policy forbids active-fire files without matching geolocation. "
            "Use a paired-cutout workflow, or explicitly pass --legacy-fire-files-only outside that policy."
        )

    result = collect_viirs_l2_range(
        arguments.archive_root,
        start_date=arguments.start,
        end_date=arguments.end,
        platforms=arguments.platforms or DEFAULT_PLATFORMS,
        bbox=arguments.bbox,
        region=arguments.region,
        earthdata_token=earthdata_token,
        dry_run=arguments.dry_run,
    )
    if arguments.dry_run:
        print(
            f"Archived {result.inventory_response_count:,} CMR inventory responses and discovered "
            f"{result.discovered_granule_count:,} Level-2 granules; "
            f"{result.incomplete_window_count:,} date/product windows remain pending download."
        )
        return 0
    print(
        f"Archived {result.archived_granule_count:,} Level-2 granules; "
        f"skipped {result.skipped_granule_count:,} already-complete granules; "
        f"recorded {result.inventory_response_count:,} CMR inventory responses; "
        f"{result.incomplete_window_count:,} date/product windows need retry."
    )
    return 1 if result.incomplete_window_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
