import unittest

from wildfire_data.forecast_weather import (
    ForecastRecordError,
    forecasts_available_at,
    latest_forecasts_as_of,
    normalize_forecast_measurement,
)


class ForecastWeatherTests(unittest.TestCase):
    def _measurement(self, **changes):
        measurement = {
            "valid_at": "2026-07-20T06:00:00Z",
            "source_grid_id": "hrrr:123:456",
            "latitude_wgs84": 54.0,
            "longitude_wgs84": -106.0,
            "variable": "wind_u_10m",
            "level": "10m",
            "member": "deterministic",
            "value": 3.5,
            "unit": "m/s",
        }
        measurement.update(changes)
        return measurement

    def _record(self, measurement=None, **changes):
        arguments = {
            "provider": "noaa",
            "model": "hrrr",
            "model_run_at": "2026-07-20T00:00:00Z",
            "published_at": "2026-07-20T00:45:00Z",
            "retrieved_at": "2026-07-20T00:50:00Z",
            "raw_artifact_id": "a" * 64,
            "ingestion_id": "ingest-001",
            "source_uri": "s3://example/run",
        }
        arguments.update(changes)
        return normalize_forecast_measurement(measurement or self._measurement(), **arguments)

    def test_normalizes_long_form_measurement_with_full_provenance(self):
        record = self._record()

        self.assertEqual(record["product_kind"], "forecast")
        self.assertEqual(record["lead_hours"], 6.0)
        self.assertEqual(record["variable"], "wind_u_10m")
        self.assertEqual(record["raw_fields"]["value"], 3.5)
        self.assertEqual(len(record["weather_snapshot_id"]), 64)

    def test_rejects_a_publication_time_before_the_model_run(self):
        with self.assertRaisesRegex(ForecastRecordError, "must not precede"):
            self._record(published_at="2026-07-19T23:59:00Z")

    def test_filters_forecasts_to_information_available_at_the_cutoff(self):
        old = self._record(published_at="2026-07-20T00:45:00Z")
        future = self._record(published_at="2026-07-20T03:45:00Z")

        available = forecasts_available_at(
            [old, future], anchor_at="2026-07-20T01:00:00Z"
        )

        self.assertEqual(available, [old])

    def test_uses_captured_availability_when_a_provider_publication_time_is_unknown(self):
        captured = self._record(
            published_at=None,
            availability_at="2026-07-20T00:50:00Z",
            availability_basis="collector-captured-single-run-response/v1",
        )

        available = forecasts_available_at(
            [captured], anchor_at="2026-07-20T01:00:00Z"
        )

        self.assertEqual(available, [captured])
        self.assertNotIn("published_at", captured)
        self.assertEqual(
            captured["availability_basis"], "collector-captured-single-run-response/v1"
        )

    def test_excludes_a_forecast_value_that_is_not_after_the_prediction_cutoff(self):
        current = self._record(measurement=self._measurement(valid_at="2026-07-20T01:00:00Z"))

        available = forecasts_available_at(
            [current], anchor_at="2026-07-20T01:00:00Z"
        )

        self.assertEqual(available, [])

    def test_selects_the_latest_known_forecast_for_each_value(self):
        older = self._record(
            model_run_at="2026-07-20T00:00:00Z",
            published_at="2026-07-20T00:45:00Z",
        )
        newer = self._record(
            model_run_at="2026-07-20T01:00:00Z",
            published_at="2026-07-20T01:45:00Z",
        )

        selected = latest_forecasts_as_of(
            [older, newer], anchor_at="2026-07-20T02:00:00Z"
        )

        self.assertEqual(selected, [newer])
