import csv
import gzip
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from wildfire_data.forecast_tile_planning import (
    iter_normalized_firms_detections,
    plan_forecast_tiles,
    write_forecast_tile_plan,
)
from wildfire_data.storage_budget import StorageBudgetError, load_storage_budget


def _detection(identifier, *, acquired_at, latitude, longitude, brightness):
    return {
        "detection_id": identifier,
        "acquired_at": acquired_at,
        "latitude": latitude,
        "longitude": longitude,
        "bright_ti4": brightness,
    }


def _policy(path, *, total=10_000_000, weather_cap=10_000_000):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "whole_data_cap_bytes": total,
                "whole_data_cap_label": "test",
                "scope": "test data",
                "categories": [
                    {
                        "key": "issued_weather_tiles",
                        "cap_bytes": weather_cap,
                        "priority_score": 85,
                        "pinned": True,
                        "retention": "weather",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return load_storage_budget(path)


class ForecastTilePlanningTests(unittest.TestCase):
    def test_applies_the_firms_availability_lag_before_a_run_can_use_evidence(self):
        plan = plan_forecast_tiles(
            [_detection("late", acquired_at="2026-08-10T05:00:00Z", latitude=54, longitude=-106, brightness=330)],
            model="hrdps",
            start_date=date(2026, 8, 10),
            end_date=date(2026, 8, 10),
        )

        self.assertEqual([candidate.model_run_at.hour for candidate in plan], [12, 18])
        self.assertTrue(all(candidate.selected for candidate in plan))
        self.assertTrue(all("180 minute" in candidate.availability_policy for candidate in plan))

    def test_records_capped_candidates_instead_of_silently_dropping_them(self):
        plan = plan_forecast_tiles(
            [
                _detection("high", acquired_at="2026-08-10T01:00:00Z", latitude=54, longitude=-106, brightness=380),
                _detection("low", acquired_at="2026-08-10T01:00:00Z", latitude=56, longitude=-112, brightness=310),
            ],
            model="hrdps",
            start_date=date(2026, 8, 10),
            end_date=date(2026, 8, 10),
            max_tiles_per_run=1,
        )
        six_utc = [candidate for candidate in plan if candidate.model_run_at.hour == 6]

        self.assertEqual(len(six_utc), 2)
        self.assertEqual(six_utc[0].representative_detection_id, "high")
        self.assertTrue(six_utc[0].selected)
        self.assertFalse(six_utc[1].selected)
        self.assertEqual(six_utc[1].non_admission_reason, "per-run-tile-cap")

    def test_writes_scores_to_a_quota_admitted_compressed_csv(self):
        plan = plan_forecast_tiles(
            [_detection("one", acquired_at="2026-08-10T01:00:00Z", latitude=54, longitude=-106, brightness=330)],
            model="hrdps",
            start_date=date(2026, 8, 10),
            end_date=date(2026, 8, 10),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            output = root / "weather" / "forecast-tile-plans" / "plan.csv.gz"
            written = write_forecast_tile_plan(
                root,
                plan=plan,
                output_path=output,
                storage_budget=_policy(Path(directory) / "budget.json"),
            )
            with gzip.open(written, "rt", encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(io.StringIO(source.read())))

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["retention_priority_score"], "85")
        self.assertEqual(rows[0]["forecast_availability_score"], "0.600000")
        self.assertEqual(rows[0]["selected"], "true")

    def test_refuses_to_write_a_plan_that_exceeds_the_weather_cap(self):
        plan = plan_forecast_tiles(
            [_detection("one", acquired_at="2026-08-10T01:00:00Z", latitude=54, longitude=-106, brightness=330)],
            model="hrdps",
            start_date=date(2026, 8, 10),
            end_date=date(2026, 8, 10),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            with self.assertRaises(StorageBudgetError):
                write_forecast_tile_plan(
                    root,
                    plan=plan,
                    output_path=root / "weather" / "forecast-tile-plans" / "plan.csv.gz",
                    storage_budget=_policy(Path(directory) / "budget.json", total=100, weather_cap=100),
                )

    def test_reads_only_the_required_firms_date_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data" / "normalized" / "fire-detections"
            for acquisition_date, identifier in (("2026-08-09", "older"), ("2026-08-10", "target")):
                path = root / f"acq-date={acquisition_date}" / "records.jsonl.gz"
                path.parent.mkdir(parents=True)
                with gzip.open(path, "wt", encoding="utf-8") as destination:
                    json.dump({"record_type": "firms_detection", "detection_id": identifier}, destination)
                    destination.write("\n")
            records = list(
                iter_normalized_firms_detections(
                    Path(directory) / "data",
                    start_date=date(2026, 8, 10),
                    end_date=date(2026, 8, 10),
                )
            )

        self.assertEqual([record["detection_id"] for record in records], ["target"])


if __name__ == "__main__":
    unittest.main()
