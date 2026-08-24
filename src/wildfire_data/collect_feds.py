"""Collect NASA FEDS 12-hour perimeter snapshots as satellite-weak labels."""

from __future__ import annotations

import argparse
from datetime import date

from .data_archive import CoverageStatus
from .feds_collection import (
    DEFAULT_PAGE_SIZE,
    DEFAULT_REGION_NAMES,
    DEFAULT_SNAPSHOT_WINDOWS_PER_REQUEST,
    collect_feds_perimeters,
)
from .storage_budget import DEFAULT_POLICY_PATH, load_storage_budget


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: list[str] | None = None) -> int:
    """Run a resumable, quota-admitted FEDS collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="inclusive UTC date")
    parser.add_argument("--end", required=True, type=_parse_date, help="inclusive UTC date")
    parser.add_argument("--archive-root", default="data")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument(
        "--snapshot-windows-per-request",
        type=int,
        default=DEFAULT_SNAPSHOT_WINDOWS_PER_REQUEST,
        help=(
            "number of consecutive 12-hour source windows in one API query; the default "
            "reduces slow FEDS request overhead while preserving per-window coverage"
        ),
    )
    parser.add_argument(
        "--include-alaska",
        action="store_true",
        help=(
            "also archive Alaska. Its FEDS local-solar-time timestamps are not eligible "
            "for the first UTC-aligned training dataset."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-query even completed source windows",
    )
    arguments = parser.parse_args(argv)
    regions = DEFAULT_REGION_NAMES + (("Alaska",) if arguments.include_alaska else ())
    region_label = "+".join(regions)
    result = collect_feds_perimeters(
        arguments.archive_root,
        start_date=arguments.start,
        end_date=arguments.end,
        storage_budget=load_storage_budget(arguments.policy),
        region_names=regions,
        region_label=region_label,
        page_size=arguments.page_size,
        snapshot_windows_per_request=arguments.snapshot_windows_per_request,
        refresh=arguments.refresh,
    )
    if result.skipped_terminal_coverage:
        print("FEDS range already has terminal coverage; no requests were made.")
    else:
        completed = sum(
            window.coverage.status in {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED}
            for window in result.windows
        )
        print(
            f"Recorded {result.feature_count:,} FEDS perimeter snapshots across "
            f"{completed:,}/{len(result.windows):,} 12-hour windows; range coverage is "
            f"{result.coverage.status.value}."
        )
    return 0 if result.coverage.status in {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED} else 1


if __name__ == "__main__":
    raise SystemExit(main())
