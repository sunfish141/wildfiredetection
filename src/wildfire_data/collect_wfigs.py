"""Collect compact WFIGS Year-to-Date reference perimeters under the 20 GB cap."""

from __future__ import annotations

import argparse
from datetime import date

from .data_archive import CoverageStatus
from .storage_budget import DEFAULT_POLICY_PATH, load_storage_budget
from .wfigs_collection import DEFAULT_REGION, collect_wfigs_year_to_date


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: list[str] | None = None) -> int:
    """Run the compact reference-perimeter backfill without retaining snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="inclusive UTC date")
    parser.add_argument("--end", required=True, type=_parse_date, help="inclusive UTC date")
    parser.add_argument("--archive-root", default="data")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--page-size", type=int, default=2_000)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="capture a new current/reference view even if this range is already terminal",
    )
    arguments = parser.parse_args(argv)
    result = collect_wfigs_year_to_date(
        arguments.archive_root,
        start_date=arguments.start,
        end_date=arguments.end,
        region=arguments.region,
        page_size=arguments.page_size,
        storage_budget=load_storage_budget(arguments.policy),
        refresh=arguments.refresh,
    )
    if result.skipped_terminal_coverage:
        print(
            "WFIGS range already has terminal coverage; no new current/reference "
            "snapshot was fetched. Pass --refresh to capture a new reference view."
        )
    else:
        print(
            f"Recorded {result.feature_count:,} WFIGS reference perimeters across "
            f"{len(result.pages):,} pages; coverage is {result.coverage.status.value}."
        )
    return 0 if result.coverage.status in {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED} else 1


if __name__ == "__main__":
    raise SystemExit(main())
