import gzip
import json
import tempfile
import unittest
from datetime import datetime, timezone

from wildfire_data.data_archive import CoverageStatus
from wildfire_data.forecast_collection import archive_forecast_response


class ForecastCollectionTests(unittest.TestCase):
    def _measurement(self):
        return {
            "valid_at": "2026-07-26T06:00:00Z",
            "source_grid_id": "hrrr:123:456",
            "latitude_wgs84": 54.0,
            "longitude_wgs84": -106.0,
            "variable": "wind_v_10m",
            "value": -2.5,
            "unit": "m/s",
        }

    def test_archives_issued_forecast_metadata_and_normalized_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            result = archive_forecast_response(
                directory,
                payload=b'{"hourly": "provider response"}',
                measurements=[self._measurement()],
                provider="NOAA",
                product="HRRR",
                model="hrrr",
                model_run_at="2026-07-26T00:00:00Z",
                published_at="2026-07-26T00:45:00Z",
                source_uri="https://example.test/hrrr?token=private",
                coverage_start="2026-07-26T00:00:00Z",
                coverage_end="2026-07-26T06:00:00Z",
                region="United States",
                response_status_code=200,
                retrieved_at=datetime(2026, 7, 26, 0, 50, tzinfo=timezone.utc),
            )
            with gzip.open(result.normalized_artifacts[0].artifact_path, "rt", encoding="utf-8") as file:
                normalized = json.loads(file.readline())
            raw_manifest = result.raw_artifact.manifest_path.read_text(encoding="utf-8")

        self.assertEqual(result.coverage.status, CoverageStatus.COMPLETE)
        self.assertEqual(result.measurement_count, 1)
        self.assertEqual(normalized["model_run_at"], "2026-07-26T00:00:00Z")
        self.assertEqual(normalized["raw_artifact_id"], result.raw_artifact.raw_artifact_id)
        self.assertNotIn("private", raw_manifest)
        self.assertNotIn("private", normalized["source_uri"])

    def test_records_non_successful_forecasts_without_normalizing_them(self):
        with tempfile.TemporaryDirectory() as directory:
            result = archive_forecast_response(
                directory,
                payload=b"unavailable",
                measurements=[],
                provider="NOAA",
                product="HRRR",
                model="hrrr",
                model_run_at="2026-07-26T00:00:00Z",
                published_at="2026-07-26T00:45:00Z",
                source_uri="https://example.test/hrrr",
                coverage_start="2026-07-26T00:00:00Z",
                coverage_end="2026-07-26T06:00:00Z",
                region="United States",
                response_status_code=503,
            )

        self.assertEqual(result.coverage.status, CoverageStatus.FAILED)
        self.assertEqual(result.normalized_artifacts, ())
