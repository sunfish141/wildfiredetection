import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import pandas as pd
import requests

import wildfire_data.weather_lookup as weather_lookup


class _FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, payloads):
        self._payloads = payloads

    def json(self):
        return self._payloads

    def raise_for_status(self):
        return None


class _FakeSession:
    request_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get(self, _url, *, params, timeout):
        type(self).request_count += 1
        hours = pd.date_range(
            params["start_date"], periods=24, freq="h", tz="UTC"
        ).strftime("%Y-%m-%dT%H:%M").tolist()
        payload = {
            "hourly": {
                "time": hours,
                "temperature_2m": list(range(24)),
                "relative_humidity_2m": [50] * 24,
                "precipitation": [0] * 24,
                "weather_code": [1] * 24,
                "wind_speed_10m": [10] * 24,
            }
        }
        return _FakeResponse([payload] * len(params["latitude"].split(",")))


class _InterruptingSession(_FakeSession):
    fail_on_request = 2

    def get(self, _url, *, params, timeout):
        type(self).request_count += 1
        if type(self).request_count == type(self).fail_on_request:
            raise requests.ConnectionError("simulated interruption")
        hours = pd.date_range(
            params["start_date"], periods=24, freq="h", tz="UTC"
        ).strftime("%Y-%m-%dT%H:%M").tolist()
        payload = {
            "hourly": {
                "time": hours,
                "temperature_2m": list(range(24)),
                "relative_humidity_2m": [50] * 24,
                "precipitation": [0] * 24,
                "weather_code": [1] * 24,
                "wind_speed_10m": [10] * 24,
            }
        }
        return _FakeResponse([payload] * len(params["latitude"].split(",")))


class _RateLimitedAfterFirstBatchSession(_FakeSession):
    def get(self, _url, *, params, timeout):
        type(self).request_count += 1
        if type(self).request_count > 1:
            response = _FakeResponse([])
            response.status_code = 429
            return response
        hours = pd.date_range(
            params["start_date"], periods=24, freq="h", tz="UTC"
        ).strftime("%Y-%m-%dT%H:%M").tolist()
        payload = {
            "hourly": {
                "time": hours,
                "temperature_2m": list(range(24)),
                "relative_humidity_2m": [50] * 24,
                "precipitation": [0] * 24,
                "weather_code": [1] * 24,
                "wind_speed_10m": [10] * 24,
            }
        }
        return _FakeResponse([payload] * len(params["latitude"].split(",")))


class WeatherLookupTests(unittest.TestCase):
    def test_defaults_to_the_organized_weather_cache_path(self):
        fetch_default = inspect.signature(weather_lookup.fetch_weather_for_queries).parameters[
            "cache_path"
        ].default
        enrich_default = inspect.signature(weather_lookup.enrich_fires_with_weather).parameters[
            "cache_path"
        ].default

        self.assertEqual(fetch_default, "data/weather/open_meteo_weather_cache.csv")
        self.assertEqual(enrich_default, "data/weather/open_meteo_weather_cache.csv")

    def test_request_pacer_spaces_http_attempts(self):
        pacer = weather_lookup.WeatherRequestPacer(requests_per_minute=60)

        with patch("wildfire_data.weather_lookup.time.monotonic", side_effect=[0.0, 0.0, 0.25, 1.0]), patch(
            "wildfire_data.weather_lookup.time.sleep"
        ) as sleep:
            pacer.wait()
            pacer.wait()

        sleep.assert_called_once_with(0.75)
        self.assertEqual(pacer.request_count, 2)

    def test_request_pacer_charges_each_location_in_a_batch(self):
        pacer = weather_lookup.WeatherRequestPacer(requests_per_minute=60)

        with patch("wildfire_data.weather_lookup.time.monotonic", side_effect=[0.0, 0.0, 0.0, 5.0]), patch(
            "wildfire_data.weather_lookup.time.sleep"
        ) as sleep:
            pacer.wait(api_call_units=5)
            pacer.wait(api_call_units=5)

        sleep.assert_called_once_with(5.0)
        self.assertEqual(pacer.request_count, 2)
        self.assertEqual(pacer.api_call_units, 10)

    def test_rate_limit_waits_90_seconds_then_retries_the_same_batch(self):
        rate_limited_response = _FakeResponse([])
        rate_limited_response.status_code = 429
        successful_response = _FakeResponse([])
        session = Mock()
        session.get.side_effect = [rate_limited_response, successful_response]
        pacer = Mock()
        pacer.rate_limit_count = 0

        with patch("builtins.print"):
            response = weather_lookup._get_with_retries(
                session,
                {},
                pacer,
                timeout=90,
                max_attempts=1,
                rate_limit_cooldown_seconds=90,
                api_call_units=50,
            )

        self.assertIs(response, successful_response)
        self.assertEqual(pacer.wait.call_args_list, [call(50), call(50)])
        pacer.defer.assert_called_once_with(90)
        self.assertEqual(pacer.rate_limit_count, 1)

    def test_two_consecutive_rate_limits_pause_the_fetch(self):
        rate_limited_response = _FakeResponse([])
        rate_limited_response.status_code = 429
        session = Mock()
        session.get.side_effect = [rate_limited_response, rate_limited_response]
        pacer = Mock()
        pacer.rate_limit_count = 0

        with patch("builtins.print"), self.assertRaises(weather_lookup.WeatherRateLimitPause):
            weather_lookup._get_with_retries(
                session,
                {},
                pacer,
                timeout=90,
                max_attempts=1,
                rate_limit_cooldown_seconds=90,
                api_call_units=50,
            )

        self.assertEqual(pacer.wait.call_args_list, [call(50), call(50)])
        pacer.defer.assert_called_once_with(90)
        self.assertEqual(pacer.rate_limit_count, 2)

    def test_sorts_export_by_oldest_acquisition_then_nearest_source(self):
        fires = pd.DataFrame(
            {
                "fire_id": ["later_far", "same_time_far", "same_time_near", "earliest_unknown"],
                "weather_source_distance_km": [1.2, 0.5, 0.1, None],
                "acq_datetime": [
                    "2026-07-22T01:00Z",
                    "2026-07-20T01:00Z",
                    "2026-07-20T01:00Z",
                    "2026-07-19T01:00Z",
                ],
            }
        )

        sorted_fires = weather_lookup.sort_fires_for_export(fires)

        self.assertEqual(
            sorted_fires["fire_id"].tolist(),
            ["earliest_unknown", "same_time_near", "same_time_far", "later_far"],
        )
        self.assertEqual(
            fires["fire_id"].tolist(),
            ["later_far", "same_time_far", "same_time_near", "earliest_unknown"],
        )

    def test_names_export_for_its_inclusive_date_range(self):
        fires = pd.DataFrame(
            {
                "acq_datetime": ["2026-07-29T01:00Z", "2026-07-26T23:59Z"],
            }
        )

        filename = weather_lookup.weather_results_filename(fires)

        self.assertEqual(filename, "fires_with_weather_2026-07-26_to_2026-07-29.csv")

    def test_checkpoints_completed_batches_before_an_interruption(self):
        fires = pd.DataFrame(
            {
                "weather_lat": [54.0, 54.0],
                "weather_lon": [-106.0, -106.0],
                "weather_hour": pd.to_datetime(["2026-07-20T01:00Z", "2026-07-21T01:00Z"]),
            }
        )
        _, mapped_fires, queries, _ = weather_lookup.prepare_weather_queries(fires)

        original_session = weather_lookup.requests.Session
        try:
            with tempfile.TemporaryDirectory() as directory:
                cache_path = Path(directory) / "weather.csv"
                _InterruptingSession.request_count = 0
                weather_lookup.requests.Session = _InterruptingSession
                with self.assertRaises(requests.ConnectionError):
                    weather_lookup.fetch_weather_for_queries(
                        mapped_fires,
                        queries,
                        cache_path=cache_path,
                        batch_size=1,
                        requests_per_minute=100_000,
                        max_attempts=1,
                    )

                self.assertEqual(len(pd.read_csv(cache_path)), 1)

                _FakeSession.request_count = 0
                weather_lookup.requests.Session = _FakeSession
                enriched, stats = weather_lookup.fetch_weather_for_queries(
                    mapped_fires,
                    queries,
                    cache_path=cache_path,
                    batch_size=1,
                    requests_per_minute=100_000,
                )
        finally:
            weather_lookup.requests.Session = original_session

        self.assertEqual(_FakeSession.request_count, 1)
        self.assertEqual(stats["cache_hits"], 1)
        self.assertTrue(enriched["temperature_2m"].notna().all())

    def test_discards_cache_rows_from_a_prior_collection_range(self):
        fires = pd.DataFrame(
            {
                "weather_lat": [54.0],
                "weather_lon": [-106.0],
                "weather_hour": pd.to_datetime(["2026-07-20T01:00Z"]),
            }
        )
        _, mapped_fires, queries, _ = weather_lookup.prepare_weather_queries(fires)
        query = queries.iloc[0]
        fetched_at = pd.Timestamp.now(tz="UTC")
        matching_row = {
            "weather_cache_version": weather_lookup.WEATHER_CACHE_VERSION,
            "weather_source_key": query.weather_source_key,
            "weather_source_lat": query.weather_source_lat,
            "weather_source_lon": query.weather_source_lon,
            "weather_hour": query.weather_hour,
            "weather_observed_at": query.weather_hour,
            "weather_fetched_at": fetched_at,
            "temperature_2m": 20.0,
            "relative_humidity_2m": 50,
            "precipitation": 0.0,
            "weather_code": 1,
            "wind_speed_10m": 10.0,
        }
        prior_range_row = {
            **matching_row,
            "weather_source_key": "prior-range-source",
            "weather_source_lat": 0.0,
            "weather_source_lon": 0.0,
            "weather_hour": pd.Timestamp("2020-01-01T01:00Z"),
            "weather_observed_at": pd.Timestamp("2020-01-01T01:00Z"),
        }

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "weather.csv"
            pd.DataFrame([matching_row, prior_range_row]).to_csv(cache_path, index=False)
            enriched, stats = weather_lookup.fetch_weather_for_queries(
                mapped_fires,
                queries,
                cache_path=cache_path,
                requests_per_minute=100_000,
            )
            scoped_cache = pd.read_csv(cache_path)

        self.assertEqual(stats["cache_entries_retained"], 1)
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(len(scoped_cache), 1)
        self.assertEqual(scoped_cache.loc[0, "weather_source_key"], query.weather_source_key)
        self.assertTrue(enriched["temperature_2m"].notna().all())

    def test_paused_fetch_resumes_from_checkpointed_batches(self):
        fires = pd.DataFrame(
            {
                "weather_lat": [54.0, 54.0],
                "weather_lon": [-106.0, -106.0],
                "weather_hour": pd.to_datetime(["2026-07-20T01:00Z", "2026-07-21T01:00Z"]),
            }
        )
        _, mapped_fires, queries, _ = weather_lookup.prepare_weather_queries(fires)

        original_session = weather_lookup.requests.Session
        try:
            with tempfile.TemporaryDirectory() as directory:
                cache_path = Path(directory) / "weather.csv"
                _RateLimitedAfterFirstBatchSession.request_count = 0
                weather_lookup.requests.Session = _RateLimitedAfterFirstBatchSession
                with patch("wildfire_data.weather_lookup.time.sleep"), patch("builtins.print"):
                    partial, paused_stats = weather_lookup.fetch_weather_for_queries(
                        mapped_fires,
                        queries,
                        cache_path=cache_path,
                        batch_size=1,
                        requests_per_minute=100_000,
                    )

                self.assertTrue(paused_stats["paused_for_rate_limit"])
                self.assertFalse(paused_stats["complete"])
                self.assertEqual(paused_stats["remaining_api_calls"], 1)
                self.assertEqual(len(pd.read_csv(cache_path)), 1)
                self.assertEqual(partial["temperature_2m"].notna().sum(), 1)

                _FakeSession.request_count = 0
                weather_lookup.requests.Session = _FakeSession
                resumed, resumed_stats = weather_lookup.fetch_weather_for_queries(
                    mapped_fires,
                    queries,
                    cache_path=cache_path,
                    batch_size=1,
                    requests_per_minute=100_000,
                )
        finally:
            weather_lookup.requests.Session = original_session

        self.assertTrue(resumed_stats["complete"])
        self.assertFalse(resumed_stats["paused_for_rate_limit"])
        self.assertEqual(_FakeSession.request_count, 1)
        self.assertTrue(resumed["temperature_2m"].notna().all())

    def test_reuses_cached_source_hourly_weather(self):
        fires = pd.DataFrame(
            {
                "weather_lat": [54.0000, 54.0000, 54.0050],
                "weather_lon": [-106.0000, -106.0000, -106.0050],
                "weather_hour": pd.to_datetime(
                    ["2026-07-20T01:00Z", "2026-07-20T02:00Z", "2026-07-20T01:00Z"],
                    utc=True,
                ),
            }
        )
        original_session = weather_lookup.requests.Session
        weather_lookup.requests.Session = _FakeSession
        _FakeSession.request_count = 0
        try:
            with tempfile.TemporaryDirectory() as directory:
                cache_path = Path(directory) / "weather.csv"
                sources, mapped_fires, queries, preparation_stats = (
                    weather_lookup.prepare_weather_queries(fires)
                )
                self.assertEqual(_FakeSession.request_count, 0)
                self.assertEqual(len(sources), 1)
                self.assertEqual(len(mapped_fires), len(fires))
                self.assertEqual(preparation_stats["source_hours"], 2)
                self.assertEqual(len(queries), 2)

                enriched, first_stats = weather_lookup.fetch_weather_for_queries(
                    mapped_fires,
                    queries,
                    cache_path=cache_path,
                    requests_per_minute=100_000,
                )
                _, second_stats = weather_lookup.fetch_weather_for_queries(
                    mapped_fires,
                    queries,
                    cache_path=cache_path,
                    requests_per_minute=100_000,
                )
                _, convenience_stats = weather_lookup.enrich_fires_with_weather(
                    fires,
                    cache_path=cache_path,
                    requests_per_minute=100_000,
                )
        finally:
            weather_lookup.requests.Session = original_session

        self.assertEqual(preparation_stats["weather_sources"], 1)
        self.assertEqual(first_stats["http_attempts"], 1)
        self.assertEqual(second_stats["http_attempts"], 0)
        self.assertEqual(convenience_stats["weather_sources"], 1)
        self.assertEqual(convenience_stats["http_attempts"], 0)
        self.assertEqual(_FakeSession.request_count, 1)
        self.assertTrue(enriched["temperature_2m"].notna().all())
        self.assertTrue((enriched["weather_source_distance_km"] <= 1).all())


if __name__ == "__main__":
    unittest.main()
