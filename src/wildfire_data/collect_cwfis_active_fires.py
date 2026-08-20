"""Collect CWFIS active-fire record history under the 20 GB policy."""

from __future__ import annotations

import argparse
from datetime import date

from .cwfis_active_fires import (
    DEFAULT_PAGE_SIZE,
    DEFAULT_REGION,
    collect_cwfis_active_fire_history,
)
from .data_archive import CoverageStatus
from .storage_budget import DEFAULT_POLICY_PATH, load_storage_budget


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: list[str] | None = None) -> int:
    """Archive CWFIS record versions without creating fire-spread labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="inclusive UTC record-start date")
    parser.add_argument("--end", required=True, type=_parse_date, help="inclusive UTC record-start date")
    parser.add_argument("--archive-root", default="data")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="capture a new provider view even if the requested history range is terminal",
    )
    arguments = parser.parse_args(argv)
    result = collect_cwfis_active_fire_history(
        arguments.archive_root,
        start_date=arguments.start,
        end_date=arguments.end,
        region=arguments.region,
        page_size=arguments.page_size,
        storage_budget=load_storage_budget(arguments.policy),
        refresh=arguments.refresh,
    )
    if result.skipped_terminal_coverage:
        print("CWFIS active-fire history range is already terminal; no new records were fetched.")
    else:
        print(
            f"Recorded {result.feature_count:,} CWFIS active-fire record versions across "
            f"{len(result.pages):,} pages; coverage is {result.coverage.status.value}."
        )
    return 0 if result.coverage.status in {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED} else 1


if __name__ == "__main__":
    raise SystemExit(main())
