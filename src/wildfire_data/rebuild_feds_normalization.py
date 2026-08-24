"""Rebuild primary-key FEDS perimeter snapshots from retained raw responses."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

from .feds_collection import rebuild_feds_primarykey_normalization
from .storage_budget import DEFAULT_POLICY_PATH, load_storage_budget


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("capture timestamp must use ISO-8601") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("capture timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def main(argv: list[str] | None = None) -> int:
    """Replay one coherent archived FEDS capture without making API requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="first primary-key snapshot date")
    parser.add_argument("--end", required=True, type=_parse_date, help="last primary-key snapshot date")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument(
        "--captured-at",
        type=_parse_timestamp,
        help=(
            "replay this exact retained FEDS collection timestamp; by default the "
            "largest, newest coherent capture is selected"
        ),
    )
    arguments = parser.parse_args(argv)
    report = rebuild_feds_primarykey_normalization(
        arguments.data_root,
        storage_budget=load_storage_budget(arguments.policy),
        start_date=arguments.start,
        end_date=arguments.end,
        captured_at=arguments.captured_at,
    )
    print(
        f"Rebuilt {report.feature_count:,} FEDS primary-key perimeter records across "
        f"{report.snapshot_count:,} 12-hour times from capture {report.selected_capture_at}; "
        f"{report.duplicate_record_count:,} duplicate rows, "
        f"{report.conflicting_record_count:,} conflicts, and "
        f"{report.invalid_record_count:,} invalid rows; status is {report.status.value}."
    )
    return 0 if report.status.value == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
