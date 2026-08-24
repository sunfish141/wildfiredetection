"""Derive 1 km positive satellite-weak labels from retained FEDS perimeters."""

from __future__ import annotations

import argparse
from datetime import date

from .feds_labels import (
    DEFAULT_POSITIVE_OVERLAP_FRACTION,
    DEFAULT_TIME_ALIGNMENT_MODE,
    FEDS_TIME_ALIGNMENT_NOMINAL_UTC,
    build_and_store_feds_weak_labels,
)
from .storage_budget import DEFAULT_POLICY_PATH, load_storage_budget


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: list[str] | None = None) -> int:
    """Run the resumable FEDS weak-label builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="first source snapshot date")
    parser.add_argument("--end", required=True, type=_parse_date, help="last source snapshot date to attempt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--positive-overlap-fraction",
        type=float,
        default=DEFAULT_POSITIVE_OVERLAP_FRACTION,
        help="minimum 1 km cell area fraction newly inside the future FEDS perimeter",
    )
    parser.add_argument(
        "--nominal-source-time",
        action="store_true",
        help=(
            "use FEDS' assigned t as UTC instead of the default per-cell local-solar "
            "overpass estimate; this is simpler but time-misaligned."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rebuild labels even when a terminal label-coverage entry already exists",
    )
    arguments = parser.parse_args(argv)
    mode = FEDS_TIME_ALIGNMENT_NOMINAL_UTC if arguments.nominal_source_time else DEFAULT_TIME_ALIGNMENT_MODE
    reports = build_and_store_feds_weak_labels(
        arguments.data_root,
        start_date=arguments.start,
        end_date=arguments.end,
        storage_budget=load_storage_budget(arguments.policy),
        time_alignment_mode=mode,
        positive_overlap_fraction=arguments.positive_overlap_fraction,
        refresh=arguments.refresh,
    )
    complete = sum(report.status.value in {"complete", "empty-confirmed"} for report in reports)
    positives = sum(report.positive_cell_count for report in reports)
    print(
        f"Built {positives:,} positive FEDS weak-label cells across "
        f"{complete:,}/{len(reports):,} source windows."
    )
    return 0 if complete == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
