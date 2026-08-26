import unittest
from unittest.mock import Mock, call, patch

from wildfire_data.weather_rate_limit import (
    WeatherRateLimitPause,
    WeatherRequestPacer,
    get_with_retries,
)


class _FakeResponse:
    def __init__(self, status_code, *, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if not 200 <= self.status_code < 300:
            raise RuntimeError(f"HTTP {self.status_code}")


class WeatherRateLimitTests(unittest.TestCase):
    def test_pacer_spaces_batched_request_units_and_counts_them(self):
        pacer = WeatherRequestPacer(requests_per_minute=60)

        with patch(
            "wildfire_data.weather_rate_limit.time.monotonic",
            side_effect=[0.0, 0.0, 0.25, 1.0],
        ), patch("wildfire_data.weather_rate_limit.time.sleep") as sleep:
            pacer.wait(api_call_units=1)
            pacer.wait(api_call_units=1)

        sleep.assert_called_once_with(0.75)
        self.assertEqual(pacer.request_count, 2)
        self.assertEqual(pacer.api_call_units, 2)

    def test_429_defers_at_least_the_configured_cooldown_then_retries(self):
        rate_limited = _FakeResponse(429, headers={"Retry-After": "120"})
        successful = _FakeResponse(200)
        session = Mock()
        session.get.side_effect = [rate_limited, successful]
        pacer = Mock()
        pacer.rate_limit_count = 0

        response = get_with_retries(
            session,
            url="https://single-runs-api.open-meteo.com/v1/forecast",
            params={"latitude": "54", "longitude": "-106"},
            pacer=pacer,
            timeout=90,
            max_attempts=1,
            rate_limit_cooldown_seconds=90,
            api_call_units=50,
        )

        self.assertIs(response, successful)
        self.assertEqual(pacer.wait.call_args_list, [call(50), call(50)])
        pacer.defer.assert_called_once_with(120.0)
        self.assertEqual(pacer.rate_limit_count, 1)

    def test_two_consecutive_429s_pause_the_checkpointable_collection(self):
        rate_limited = _FakeResponse(429)
        session = Mock()
        session.get.side_effect = [rate_limited, rate_limited]
        pacer = Mock()
        pacer.rate_limit_count = 0

        with self.assertRaises(WeatherRateLimitPause) as raised:
            get_with_retries(
                session,
                url="https://single-runs-api.open-meteo.com/v1/forecast",
                params={"latitude": "54", "longitude": "-106"},
                pacer=pacer,
                timeout=90,
                max_attempts=1,
                rate_limit_cooldown_seconds=90,
                api_call_units=50,
                max_consecutive_rate_limits=2,
            )

        self.assertEqual(pacer.wait.call_args_list, [call(50), call(50)])
        pacer.defer.assert_called_once_with(90)
        self.assertEqual(pacer.rate_limit_count, 2)
        self.assertIs(raised.exception.response, rate_limited)


if __name__ == "__main__":
    unittest.main()
