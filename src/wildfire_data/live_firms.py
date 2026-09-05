"""Bounded, transient FIRMS previews for the interactive simulator.

Responses are kept in memory for the preview only; this is not a training
collector and never appends to or modifies the immutable source archive.
"""

import csv
from datetime import datetime, timezone
import math

import requests

from .firms_normalization import normalize_firms_detection
from .collect_firms import DEFAULT_BBOX
from .incident_transition import EvidenceCell
from .recursive_transition import RecursiveFireState
from .training_grid import cell_from_wgs84


PRODUCTS = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
DEFAULT_BOUNDS = tuple(float(value) for value in DEFAULT_BBOX.split(','))


class LiveFirmsError(ValueError):
    pass


def fetch_current_firms(api_key, bounds=DEFAULT_BOUNDS, *, now=None, session=None):
    """Read all three streams; fail the preview if any stream is unavailable."""
    if not api_key:
        raise LiveFirmsError("Set NASA_FIRMS_API_KEY or MAP_KEY in config/.env, then restart the server.")
    now = now or datetime.now(timezone.utc)
    owned = session is None
    session = session or requests.Session()
    def detections():
        for product in PRODUCTS:
            bbox = ",".join(str(v) for v in bounds)
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{product}/{bbox}/2"
            try:
                # Never expose the credential-bearing URL through exceptions.
                with session.get(url, timeout=(10, 120), stream=True) as response:
                    response.raise_for_status()
                    reader = csv.DictReader(line.decode("utf-8-sig") for line in response.iter_lines())
                    required = {"latitude", "longitude", "acq_date", "acq_time", "bright_ti4"}
                    if not required.issubset(reader.fieldnames or ()):
                        raise LiveFirmsError(f"FIRMS returned an unavailable or invalid {product} feed. Try again later.")
                    for row in reader:
                        yield normalize_firms_detection(row,
                            provenance={"provider": "NASA FIRMS", "product": product}, minimum_bright_ti4=None)
            except (requests.RequestException, UnicodeError, ValueError, csv.Error) as exc:
                if isinstance(exc, LiveFirmsError):
                    raise
                raise LiveFirmsError(f"Could not load {product} from FIRMS. Check connectivity and the server's API key.") from None
    try:
        return aggregate_current_firms(detections(), bounds, now=now)
    finally:
        if owned:
            session.close()


def aggregate_current_firms(detections, bounds, *, now):
    west, south, east, north = bounds
    grouped, recent, seen = {}, 0, set()
    for row in detections:
        if not (west <= row["longitude"] <= east and south <= row["latitude"] <= north):
            continue
        identity = (row["provenance"]["product"], row["latitude"], row["longitude"], row["acquired_at"])
        if identity in seen:
            continue
        seen.add(identity)
        age = (now - datetime.fromisoformat(row["acquired_at"].replace("Z", "+00:00"))).total_seconds() / 3600
        if 0 <= age < 3:
            recent += 1
        if not 3 <= age <= 24:
            continue
        cell = cell_from_wgs84(latitude=row["latitude"], longitude=row["longitude"])
        stats = grouped.setdefault(cell.cell_id, {"count": 0, "total": 0., "maximum": -math.inf,
                                                  "age": math.inf, "products": set()})
        stats["count"] += 1
        stats["total"] += row["bright_ti4"]
        stats["maximum"] = max(stats["maximum"], row["bright_ti4"])
        stats["age"] = min(stats["age"], age)
        stats["products"].add(row["provenance"]["product"])
    active = []
    for cell_id, stats in sorted(grouped.items()):
        active.append(EvidenceCell(cell_id, min(1., max(0., (stats["maximum"] - 305) / 62)), 2,
            stats["age"], stats["count"], stats["maximum"],
            min(stats["maximum"], stats["total"] / stats["count"]), len(stats["products"])))
    return RecursiveFireState(0, tuple(active)), {
        "source": "NASA FIRMS", "as_of": now.isoformat(), "products": list(PRODUCTS),
        "eligible_detection_count": sum(stats["count"] for stats in grouped.values()),
        "recent_detections_excluded": recent, "active_cell_count": len(active),
        "observation_window_hours": [3, 24],
        "bounds": list(bounds), "aggregation": "canonical-1km-cells",
    }
