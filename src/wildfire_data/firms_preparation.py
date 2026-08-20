"""Prepare a full FIRMS dataframe for the legacy weather-enrichment view."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .firms_filtering import filter_firms_by_minimum_brightness


REQUIRED_FIRMS_COLUMNS = ("latitude", "longitude", "bright_ti4", "acq_date", "acq_time")


@dataclass(frozen=True)
class FirmsPreparation:
    """Unfiltered/filtered counts and the feature-preserving working frame."""

    fires: pd.DataFrame
    input_detections: int
    detections_with_required_fields: int


def prepare_firms_for_weather(
    fires: pd.DataFrame, *, minimum_bright_ti4: float
) -> FirmsPreparation:
    """Create weather-ready detections without discarding extra FIRMS fields.

    Raw records are archived elsewhere before this function is called.  This
    view may apply the existing TI4 visualization threshold, but source fields
    such as FRP, confidence, scan/track, and future FIRMS columns stay intact.
    """
    missing = set(REQUIRED_FIRMS_COLUMNS).difference(fires.columns)
    if missing:
        raise ValueError(f"FIRMS response is missing columns: {sorted(missing)}")
    prepared = fires.copy().dropna(subset=REQUIRED_FIRMS_COLUMNS)
    filtered = filter_firms_by_minimum_brightness(
        prepared, minimum_bright_ti4=minimum_bright_ti4
    )
    filtered["acq_time"] = filtered["acq_time"].astype(int).astype(str).str.zfill(4)
    filtered["acq_datetime"] = pd.to_datetime(
        filtered["acq_date"].astype(str) + " " + filtered["acq_time"],
        format="%Y-%m-%d %H%M",
        utc=True,
    )
    filtered["weather_lat"] = filtered["latitude"]
    filtered["weather_lon"] = filtered["longitude"]
    filtered["weather_hour"] = filtered["acq_datetime"].dt.floor("h")
    return FirmsPreparation(
        fires=filtered.sort_values("bright_ti4", ascending=False).reset_index(drop=True),
        input_detections=len(fires),
        detections_with_required_fields=len(prepared),
    )
