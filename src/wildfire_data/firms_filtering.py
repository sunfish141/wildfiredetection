"""Filtering helpers for NASA FIRMS detections."""

from __future__ import annotations

import pandas as pd


def filter_firms_by_minimum_brightness(
    fires: pd.DataFrame, *, minimum_bright_ti4: float
) -> pd.DataFrame:
    """Return only detections whose TI4 brightness meets the inclusive threshold."""
    if "bright_ti4" not in fires.columns:
        raise ValueError("fires is missing required column: 'bright_ti4'")

    brightness = pd.to_numeric(fires["bright_ti4"], errors="coerce")
    return fires.loc[brightness >= minimum_bright_ti4].copy()
