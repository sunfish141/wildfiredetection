"""Efficient, rate-limited Open-Meteo lookups for FIRMS detections."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

from .weather_source_selection import minimum_covering_sources


WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
]
WEATHER_CACHE_VERSION = 1

_TARGET_KEYS = ["weather_source_key", "weather_hour"]
_WEATHER_RESULT_COLUMNS = ["weather_observed_at", *WEATHER_FIELDS]
_CACHE_COLUMNS = [
    "weather_cache_version",
    "weather_source_key",
    "weather_source_lat",
    "weather_source_lon",
    "weather_hour",
    "weather_observed_at",
    "weather_fetched_at",
    *WEATHER_FIELDS,
]


class WeatherRequestPacer:
    """Space location lookups so a run stays below a per-minute API-call cap."""

    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._interval_seconds = 60 / requests_per_minute
        self._next_request_at = 0.0
        self.request_count = 0
        self.api_call_units = 0
        self.rate_limit_count = 0

    def wait(self, api_call_units: int = 1) -> None:
        if api_call_units <= 0:
            raise ValueError("api_call_units must be positive")
        delay = self._next_request_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_request_at = max(self._next_request_at, time.monotonic()) + (
            api_call_units * self._interval_seconds
        )
        self.request_count += 1
        self.api_call_units += api_call_units

    def defer(self, seconds: float) -> None:
        self._next_request_at = max(self._next_request_at, time.monotonic() + seconds)


class WeatherRateLimitPause(RuntimeError):
    """Signal that a checkpointed weather fetch should be resumed later."""


def _coordinate_key(latitude: float, longitude: float) -> str:
    """Return a stable, lossless cache key for a pair of Python floats."""
    return f"{float(latitude).hex()}:{float(longitude).hex()}"


def _batched(frame: pd.DataFrame, size: int):
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


def _source_mapping(fires: pd.DataFrame, max_distance_m: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    locations = (
        fires[["weather_lat", "weather_lon"]]
        .drop_duplicates()
        .sort_values(["weather_lon", "weather_lat"])
        .reset_index(drop=True)
    )
    cover = minimum_covering_sources(
        locations[["weather_lon", "weather_lat"]].to_numpy(), radius_m=max_distance_m
    )

    locations["_location_index"] = locations.index
    sources = locations.iloc[cover.source_indices].copy().reset_index(drop=True)
    sources["weather_source_id"] = [f"source_{index:06d}" for index in range(len(sources))]
    sources["weather_source_key"] = [
        _coordinate_key(latitude, longitude)
        for latitude, longitude in zip(sources["weather_lat"], sources["weather_lon"])
    ]
    sources = sources.rename(
        columns={"weather_lat": "weather_source_lat", "weather_lon": "weather_source_lon"}
    )

    source_by_location_index = sources.set_index("_location_index")
    mapping = locations.drop(columns="_location_index").copy()
    mapping["weather_source_id"] = source_by_location_index.loc[
        cover.assigned_source_indices, "weather_source_id"
    ].to_numpy()
    mapping["weather_source_key"] = source_by_location_index.loc[
        cover.assigned_source_indices, "weather_source_key"
    ].to_numpy()
    mapping["weather_source_lat"] = source_by_location_index.loc[
        cover.assigned_source_indices, "weather_source_lat"
    ].to_numpy()
    mapping["weather_source_lon"] = source_by_location_index.loc[
        cover.assigned_source_indices, "weather_source_lon"
    ].to_numpy()
    mapping["weather_source_distance_km"] = cover.distances_m / 1_000
    return sources.drop(columns="_location_index"), mapping


def sort_fires_for_export(fires: pd.DataFrame) -> pd.DataFrame:
    """Order fire records by oldest acquisition time, then nearest weather source."""
    required_columns = {"weather_source_distance_km", "acq_datetime"}
    missing_columns = required_columns.difference(fires.columns)
    if missing_columns:
        raise ValueError(f"fires is missing required export columns: {sorted(missing_columns)}")

    return (
        fires.assign(
            _sort_distance=pd.to_numeric(fires["weather_source_distance_km"], errors="coerce"),
            _sort_datetime=pd.to_datetime(fires["acq_datetime"], utc=True, errors="coerce"),
        )
        .sort_values(["_sort_datetime", "_sort_distance"], kind="stable", na_position="last")
        .drop(columns=["_sort_distance", "_sort_datetime"])
        .reset_index(drop=True)
    )


def weather_results_filename(fires: pd.DataFrame, *, prefix: str = "fires_with_weather") -> str:
    """Return an export filename labelled with the inclusive acquisition-date range."""
    if "acq_datetime" not in fires.columns:
        raise ValueError("fires is missing required export column: 'acq_datetime'")

    acquisition_times = pd.to_datetime(fires["acq_datetime"], utc=True, errors="coerce").dropna()
    if acquisition_times.empty:
        raise ValueError("fires has no valid acquisition timestamps")

    first_date = acquisition_times.min().date().isoformat()
    last_date = acquisition_times.max().date().isoformat()
    return f"{prefix}_{first_date}_to_{last_date}.csv"


def _empty_cache() -> pd.DataFrame:
    return pd.DataFrame(columns=_CACHE_COLUMNS)


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return _empty_cache()

    try:
        cache = pd.read_csv(cache_path)
    except pd.errors.EmptyDataError:
        return _empty_cache()
    if not set(_CACHE_COLUMNS).issubset(cache.columns):
        print(f"Ignoring incompatible weather cache at {cache_path}.")
        return _empty_cache()

    cache = cache[_CACHE_COLUMNS].copy()
    cache["weather_hour"] = pd.to_datetime(cache["weather_hour"], utc=True, errors="coerce")
    cache["weather_observed_at"] = pd.to_datetime(
        cache["weather_observed_at"], utc=True, errors="coerce"
    )
    cache["weather_fetched_at"] = pd.to_datetime(
        cache["weather_fetched_at"], utc=True, errors="coerce"
    )
    cache = cache.dropna(subset=["weather_source_key", "weather_hour", "weather_fetched_at"])
    cache = cache[cache["weather_cache_version"] == WEATHER_CACHE_VERSION]
    return cache.sort_values("weather_fetched_at").drop_duplicates(_TARGET_KEYS, keep="last")


def _merge_cache_rows(cache: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Keep the newest result for each source/hour cache key."""
    updated_cache = pd.concat([cache, rows], ignore_index=True)
    return updated_cache.sort_values("weather_fetched_at").drop_duplicates(_TARGET_KEYS, keep="last")


def _write_cache(cache_path: Path, cache: pd.DataFrame) -> None:
    """Atomically checkpoint weather results so an interrupted run can resume."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.tmp")
    cache.to_csv(temporary_path, index=False)
    temporary_path.replace(cache_path)


def _scope_cache_to_targets(cache: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Discard cache entries that do not belong to the current collection range."""
    if cache.empty:
        return cache
    target_keys = targets[_TARGET_KEYS].drop_duplicates()
    return cache.merge(target_keys, on=_TARGET_KEYS, how="inner", validate="one_to_one")[_CACHE_COLUMNS]


def _valid_cache(cache: pd.DataFrame, now: pd.Timestamp, cache_ttl_seconds: int) -> pd.DataFrame:
    if cache.empty:
        return cache
    historical = cache["weather_hour"].dt.date < now.date()
    fresh = cache["weather_fetched_at"] >= now - pd.Timedelta(seconds=cache_ttl_seconds)
    return cache[historical | fresh]


def _source_days(targets: pd.DataFrame) -> pd.DataFrame:
    return (
        targets[["weather_source_key", "weather_source_lat", "weather_source_lon", "weather_hour"]]
        .assign(weather_date=lambda frame: frame["weather_hour"].dt.date)
        .drop(columns="weather_hour")
        .drop_duplicates()
    )


def _request_count(source_days: pd.DataFrame, batch_size: int) -> int:
    return sum(math.ceil(len(day) / batch_size) for _, day in source_days.groupby("weather_date"))


def _retry_delay_seconds(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    return float(2**attempt)


def _get_with_retries(
    session: requests.Session,
    params: dict[str, str],
    pacer: WeatherRequestPacer,
    timeout: int,
    max_attempts: int,
    rate_limit_cooldown_seconds: int,
    api_call_units: int,
    max_consecutive_rate_limits: int = 2,
) -> requests.Response:
    attempt = 0
    consecutive_rate_limits = 0
    while True:
        pacer.wait(api_call_units)
        try:
            response = session.get(WEATHER_URL, params=params, timeout=timeout)
        except requests.RequestException:
            consecutive_rate_limits = 0
            if attempt >= max_attempts - 1:
                raise
            pacer.defer(2**attempt)
            attempt += 1
            continue

        if response.status_code == 429:
            consecutive_rate_limits += 1
            pacer.rate_limit_count += 1
            if consecutive_rate_limits >= max_consecutive_rate_limits:
                print(
                    "Open-Meteo returned consecutive 429 responses; pausing with completed "
                    "weather batches saved to the cache."
                )
                raise WeatherRateLimitPause("weather fetch paused after consecutive 429 responses")
            cooldown = max(rate_limit_cooldown_seconds, _retry_delay_seconds(response, attempt))
            print(
                "Open-Meteo returned 429; waiting "
                f"{cooldown:.0f} seconds before retrying this batch."
            )
            pacer.defer(cooldown)
            continue

        consecutive_rate_limits = 0
        if response.status_code in {500, 502, 503, 504} and attempt < max_attempts - 1:
            pacer.defer(_retry_delay_seconds(response, attempt))
            attempt += 1
            continue
        response.raise_for_status()
        return response


def _fetch_weather_batch(
    batch: pd.DataFrame,
    target_hours: pd.DataFrame,
    session: requests.Session,
    pacer: WeatherRequestPacer,
    timeout: int,
    max_attempts: int,
    rate_limit_cooldown_seconds: int,
    max_consecutive_rate_limits: int,
) -> pd.DataFrame:
    target_date = batch["weather_date"].iloc[0].isoformat()
    params = {
        "latitude": ",".join(batch["weather_source_lat"].astype(str)),
        "longitude": ",".join(batch["weather_source_lon"].astype(str)),
        "hourly": ",".join(WEATHER_FIELDS),
        "start_date": target_date,
        "end_date": target_date,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }
    response = _get_with_retries(
        session,
        params,
        pacer,
        timeout,
        max_attempts,
        rate_limit_cooldown_seconds,
        api_call_units=len(batch),
        max_consecutive_rate_limits=max_consecutive_rate_limits,
    )
    payloads = response.json()
    if isinstance(payloads, dict):
        payloads = [payloads]
    if not isinstance(payloads, list) or len(payloads) != len(batch):
        raise ValueError(
            f"Open-Meteo returned {len(payloads) if isinstance(payloads, list) else 'an invalid number of'} "
            f"payloads for {len(batch)} requested source locations."
        )

    requested_hours = (
        target_hours[target_hours["weather_source_key"].isin(batch["weather_source_key"])]
        .groupby("weather_source_key")["weather_hour"]
        .agg(set)
    )
    records = []
    for source, payload in zip(batch.itertuples(index=False), payloads):
        hourly = payload.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []), utc=True)
        source_hours = requested_hours[source.weather_source_key]
        for index, observed_at in enumerate(times):
            if observed_at not in source_hours:
                continue
            values = {}
            for field in WEATHER_FIELDS:
                field_values = hourly.get(field) or []
                values[field] = field_values[index] if index < len(field_values) else None
            records.append(
                {
                    "weather_source_key": source.weather_source_key,
                    "weather_hour": observed_at,
                    "weather_observed_at": observed_at,
                    **values,
                }
            )
    return pd.DataFrame(records, columns=[*_TARGET_KEYS, *_WEATHER_RESULT_COLUMNS])


def prepare_weather_queries(
    fires: pd.DataFrame,
    *,
    max_distance_m: float = 1_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Return selected sources, mapped fires, and one deduplicated source/hour query per lookup."""
    required_columns = {"weather_lat", "weather_lon", "weather_hour"}
    missing_columns = required_columns.difference(fires.columns)
    if missing_columns:
        raise ValueError(f"fires is missing required weather columns: {sorted(missing_columns)}")

    sources, mapping = _source_mapping(fires, max_distance_m)
    mapped_fires = fires.merge(
        mapping,
        on=["weather_lat", "weather_lon"],
        how="left",
        validate="many_to_one",
    )
    queries = mapped_fires[
        [
            "weather_source_id",
            "weather_source_key",
            "weather_source_lat",
            "weather_source_lon",
            "weather_hour",
        ]
    ].drop_duplicates()
    source_days = _source_days(queries)
    stats = {
        "input_locations": len(mapping),
        "weather_sources": len(sources),
        "source_hours": len(queries),
        "source_days": len(source_days),
    }
    return sources, mapped_fires, queries, stats


def fetch_weather_for_queries(
    mapped_fires: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    cache_path: str | Path = "data/weather/open_meteo_weather_cache.csv",
    batch_size: int = 50,
    requests_per_minute: int = 600,
    cache_ttl_seconds: int = 3_600,
    timeout: int = 90,
    max_attempts: int = 4,
    rate_limit_cooldown_seconds: int = 90,
    max_consecutive_rate_limits: int = 2,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Fetch weather, checkpoint each batch to CSV, and attach it to mapped fires."""
    required_fire_columns = {"weather_source_id", "weather_hour"}
    missing_fire_columns = required_fire_columns.difference(mapped_fires.columns)
    if missing_fire_columns:
        raise ValueError(
            "mapped_fires is missing required weather columns: "
            f"{sorted(missing_fire_columns)}"
        )
    required_query_columns = {
        "weather_source_id",
        "weather_source_key",
        "weather_source_lat",
        "weather_source_lon",
        "weather_hour",
    }
    missing_columns = required_query_columns.difference(queries.columns)
    if missing_columns:
        raise ValueError(f"queries is missing required weather columns: {sorted(missing_columns)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if rate_limit_cooldown_seconds <= 0:
        raise ValueError("rate_limit_cooldown_seconds must be positive")
    if max_consecutive_rate_limits <= 0:
        raise ValueError("max_consecutive_rate_limits must be positive")

    targets = queries[list(required_query_columns)].drop_duplicates()

    cache_path = Path(cache_path)
    now = pd.Timestamp.now(tz="UTC")
    cache = _scope_cache_to_targets(_load_cache(cache_path), targets)
    _write_cache(cache_path, cache)
    valid_cache = _valid_cache(cache, now, cache_ttl_seconds)
    cached_targets = targets.merge(
        valid_cache[[*_TARGET_KEYS, *_WEATHER_RESULT_COLUMNS]],
        on=_TARGET_KEYS,
        how="inner",
        validate="one_to_one",
    )
    missing_targets = targets.merge(
        cached_targets[_TARGET_KEYS], on=_TARGET_KEYS, how="left", indicator=True
    )
    missing_targets = missing_targets[missing_targets["_merge"] == "left_only"].drop(columns="_merge")

    all_source_days = _source_days(targets)
    missing_source_days = _source_days(missing_targets)
    pacer = WeatherRequestPacer(requests_per_minute)
    fetched_target_batches = []
    paused_for_rate_limit = False
    if len(missing_source_days):
        with requests.Session() as session:
            for _, day_sources in missing_source_days.groupby("weather_date"):
                for batch in _batched(day_sources.reset_index(drop=True), batch_size):
                    batch_date = batch["weather_date"].iloc[0]
                    batch_targets = missing_targets[
                        missing_targets["weather_source_key"].isin(batch["weather_source_key"])
                        & (missing_targets["weather_hour"].dt.date == batch_date)
                    ]
                    try:
                        fetched_weather = _fetch_weather_batch(
                            batch,
                            batch_targets,
                            session,
                            pacer,
                            timeout,
                            max_attempts,
                            rate_limit_cooldown_seconds,
                            max_consecutive_rate_limits,
                        )
                    except WeatherRateLimitPause:
                        paused_for_rate_limit = True
                        break
                    fetched_targets_batch = batch_targets.merge(
                        fetched_weather,
                        on=_TARGET_KEYS,
                        how="left",
                        validate="one_to_one",
                    )
                    fetched_targets_batch["weather_fetched_at"] = pd.Timestamp.now(tz="UTC")
                    new_cache_rows = fetched_targets_batch.assign(
                        weather_cache_version=WEATHER_CACHE_VERSION
                    )[_CACHE_COLUMNS]
                    cache = _merge_cache_rows(cache, new_cache_rows)
                    _write_cache(cache_path, cache)
                    fetched_target_batches.append(fetched_targets_batch)
                if paused_for_rate_limit:
                    break

    if missing_targets.empty:
        fetched_targets = missing_targets.copy()
        for column in _WEATHER_RESULT_COLUMNS:
            fetched_targets[column] = pd.Series(index=fetched_targets.index, dtype="object")
    elif fetched_target_batches:
        fetched_targets = pd.concat(fetched_target_batches, ignore_index=True)
    else:
        fetched_targets = missing_targets.iloc[0:0].copy()
        for column in _WEATHER_RESULT_COLUMNS:
            fetched_targets[column] = pd.Series(index=fetched_targets.index, dtype="object")

    weather_results = pd.concat([cached_targets, fetched_targets], ignore_index=True)
    completed_target_keys = weather_results[_TARGET_KEYS].drop_duplicates()
    remaining_targets = targets.merge(
        completed_target_keys, on=_TARGET_KEYS, how="left", indicator=True
    )
    remaining_targets = remaining_targets[remaining_targets["_merge"] == "left_only"].drop(
        columns="_merge"
    )
    remaining_source_days = _source_days(remaining_targets)
    enriched_fires = mapped_fires.merge(
        weather_results[["weather_source_id", "weather_hour", *_WEATHER_RESULT_COLUMNS]],
        on=["weather_source_id", "weather_hour"],
        how="left",
        validate="many_to_one",
    )
    stats = {
        "source_days": len(all_source_days),
        "cache_entries_retained": len(cache),
        "cache_hits": len(cached_targets),
        "projected_requests_without_cache": _request_count(all_source_days, batch_size),
        "projected_requests": _request_count(missing_source_days, batch_size),
        "projected_api_calls_without_cache": len(all_source_days),
        "projected_api_calls": len(missing_source_days),
        "http_attempts": pacer.request_count,
        "api_call_units": pacer.api_call_units,
        "rate_limit_retries": pacer.rate_limit_count,
        "paused_for_rate_limit": paused_for_rate_limit,
        "complete": not paused_for_rate_limit,
        "remaining_source_hours": len(remaining_targets),
        "remaining_api_calls": len(remaining_source_days),
        "remaining_requests": _request_count(remaining_source_days, batch_size),
    }
    if paused_for_rate_limit:
        print(
            "Weather fetch paused after consecutive 429 responses; "
            f"{stats['remaining_api_calls']:,} source/day calls remain. "
            "Rerun the weather fetch to resume from the saved cache."
        )
    return enriched_fires, stats


def enrich_fires_with_weather(
    fires: pd.DataFrame,
    *,
    cache_path: str | Path = "data/weather/open_meteo_weather_cache.csv",
    max_distance_m: float = 1_000.0,
    batch_size: int = 50,
    requests_per_minute: int = 600,
    cache_ttl_seconds: int = 3_600,
    timeout: int = 90,
    max_attempts: int = 4,
    rate_limit_cooldown_seconds: int = 90,
    max_consecutive_rate_limits: int = 2,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Convenience wrapper that prepares queries then fetches their weather."""
    _, mapped_fires, queries, preparation_stats = prepare_weather_queries(
        fires, max_distance_m=max_distance_m
    )
    enriched_fires, fetch_stats = fetch_weather_for_queries(
        mapped_fires,
        queries,
        cache_path=cache_path,
        batch_size=batch_size,
        requests_per_minute=requests_per_minute,
        cache_ttl_seconds=cache_ttl_seconds,
        timeout=timeout,
        max_attempts=max_attempts,
        rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
        max_consecutive_rate_limits=max_consecutive_rate_limits,
    )
    return enriched_fires, {**preparation_stats, **fetch_stats}
