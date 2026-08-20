"""Persistent backward date-window selection for FIRMS collection."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path


_STATE_START_KEY = "completed_range_start"
_STATE_END_KEY = "completed_range_end"


def _parse_date(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date string") from exc


def _load_completed_range(state_path: Path) -> tuple[date, date] | None:
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid FIRMS range state at {state_path}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"Invalid FIRMS range state at {state_path}")

    start_date = _parse_date(state.get(_STATE_START_KEY), _STATE_START_KEY)
    end_date = _parse_date(state.get(_STATE_END_KEY), _STATE_END_KEY)
    if start_date > end_date:
        raise ValueError(f"Invalid FIRMS range state at {state_path}: start is after end")
    return start_date, end_date


def _oldest_export_range(results_directory: Path, results_prefix: str) -> tuple[date, date] | None:
    pattern = re.compile(
        rf"^{re.escape(results_prefix)}_(\d{{4}}-\d{{2}}-\d{{2}})_to_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$"
    )
    ranges = []
    for result_path in results_directory.glob(f"{results_prefix}_*_to_*.csv"):
        match = pattern.match(result_path.name)
        if not match:
            continue
        start_date = _parse_date(match.group(1), result_path.name)
        end_date = _parse_date(match.group(2), result_path.name)
        if start_date <= end_date:
            ranges.append((start_date, end_date))
    return min(ranges, default=None)


def firms_range_filename(
    start_date: date,
    end_date: date,
    *,
    prefix: str = "fires_with_weather",
) -> str:
    """Return a CSV filename labelled with an inclusive FIRMS query range."""
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    return f"{prefix}_{start_date.isoformat()}_to_{end_date.isoformat()}.csv"


def next_firms_date_range(
    state_path: str | Path,
    *,
    window_days: int,
    results_directory: str | Path = "data/exports",
    results_prefix: str = "fires_with_weather",
    fallback_end_date: date | None = None,
) -> tuple[date, date]:
    """Return the next non-overlapping inclusive range while moving backward in time."""
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    completed_range = _load_completed_range(Path(state_path))
    if completed_range is None:
        completed_range = _oldest_export_range(Path(results_directory), results_prefix)

    if completed_range is None:
        end_date = fallback_end_date or date.today()
    else:
        end_date = completed_range[0] - timedelta(days=1)
    start_date = end_date - timedelta(days=window_days - 1)
    return start_date, end_date


def save_completed_firms_range(
    state_path: str | Path,
    start_date: date,
    end_date: date,
) -> None:
    """Atomically record a successfully collected inclusive FIRMS date range."""
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")

    resolved_state_path = Path(state_path)
    resolved_state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_state_path.with_name(f".{resolved_state_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            {
                _STATE_START_KEY: start_date.isoformat(),
                _STATE_END_KEY: end_date.isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    temporary_path.replace(resolved_state_path)
