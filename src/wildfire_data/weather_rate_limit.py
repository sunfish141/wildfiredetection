"""Conservative pacing and retry handling for weather-provider requests.

The collectors use location-counted pacing because one Open-Meteo request may
contain many coordinates.  A 429 is deliberately not hidden behind an
unbounded retry loop: after two consecutive responses, the caller stops with
all previously archived batches intact.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping

import requests


class WeatherRequestPacer:
    """Space request units so a collection stays below its configured rate."""

    def __init__(self, requests_per_minute: int):
        if not isinstance(requests_per_minute, int) or requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be a positive integer")
        self._interval_seconds = 60 / requests_per_minute
        self._next_request_at = 0.0
        self.request_count = 0
        self.api_call_units = 0
        self.rate_limit_count = 0

    def wait(self, api_call_units: int = 1) -> None:
        """Wait for an admitted request slot, then account for its location units."""
        if not isinstance(api_call_units, int) or api_call_units <= 0:
            raise ValueError("api_call_units must be a positive integer")
        delay = self._next_request_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_request_at = max(self._next_request_at, time.monotonic()) + (
            api_call_units * self._interval_seconds
        )
        self.request_count += 1
        self.api_call_units += api_call_units

    def defer(self, seconds: float) -> None:
        """Push the next request out after a provider-directed cooldown."""
        if seconds < 0:
            raise ValueError("seconds must not be negative")
        self._next_request_at = max(self._next_request_at, time.monotonic() + seconds)


class WeatherRateLimitPause(RuntimeError):
    """Signal that a weather collection paused after retaining prior batches."""

    def __init__(self, message: str, *, response: Any | None = None) -> None:
        super().__init__(message)
        self.response = response


def retry_delay_seconds(response: Any, attempt: int) -> float:
    """Use ``Retry-After`` when present, otherwise an exponential backoff."""
    if attempt < 0:
        raise ValueError("attempt must not be negative")
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    return float(2**attempt)


def get_with_retries(
    session: requests.Session,
    *,
    url: str,
    params: Mapping[str, object],
    pacer: WeatherRequestPacer,
    timeout: int | tuple[int, int],
    max_attempts: int,
    rate_limit_cooldown_seconds: int,
    api_call_units: int,
    max_consecutive_rate_limits: int = 2,
):
    """Issue one GET while retaining the legacy conservative 429 policy.

    ``max_attempts`` applies to transport and transient server failures.  A
    second consecutive HTTP 429 raises :class:`WeatherRateLimitPause` instead
    of continuing to consume the provider's quota.
    """
    if not url.strip():
        raise ValueError("url must be non-empty")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if rate_limit_cooldown_seconds <= 0:
        raise ValueError("rate_limit_cooldown_seconds must be positive")
    if max_consecutive_rate_limits <= 0:
        raise ValueError("max_consecutive_rate_limits must be positive")

    attempt = 0
    consecutive_rate_limits = 0
    while True:
        pacer.wait(api_call_units)
        try:
            response = session.get(url, params=dict(params), timeout=timeout)
        except requests.RequestException:
            consecutive_rate_limits = 0
            if attempt >= max_attempts - 1:
                raise
            pacer.defer(float(2**attempt))
            attempt += 1
            continue

        if response.status_code == 429:
            consecutive_rate_limits += 1
            pacer.rate_limit_count += 1
            if consecutive_rate_limits >= max_consecutive_rate_limits:
                print(
                    "Open-Meteo returned consecutive 429 responses; pausing after "
                    "retaining completed weather batches."
                )
                raise WeatherRateLimitPause(
                    "weather fetch paused after consecutive 429 responses", response=response
                )
            cooldown = max(
                float(rate_limit_cooldown_seconds), retry_delay_seconds(response, attempt)
            )
            print(
                "Open-Meteo returned 429; waiting "
                f"{cooldown:.0f} seconds before retrying this batch."
            )
            pacer.defer(cooldown)
            continue

        consecutive_rate_limits = 0
        if response.status_code in {500, 502, 503, 504} and attempt < max_attempts - 1:
            pacer.defer(retry_delay_seconds(response, attempt))
            attempt += 1
            continue
        response.raise_for_status()
        return response
