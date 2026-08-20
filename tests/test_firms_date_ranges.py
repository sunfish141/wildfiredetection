import inspect
import tempfile
import unittest
from datetime import date
from pathlib import Path

from wildfire_data.firms_date_ranges import (
    firms_range_filename,
    next_firms_date_range,
    save_completed_firms_range,
)


class FirmsDateRangeTests(unittest.TestCase):
    def test_defaults_to_the_organized_exports_directory(self):
        default_directory = inspect.signature(next_firms_date_range).parameters[
            "results_directory"
        ].default

        self.assertEqual(default_directory, "data/exports")

    def test_uses_existing_export_as_the_initial_completed_range(self):
        with tempfile.TemporaryDirectory() as directory:
            results_directory = Path(directory)
            (results_directory / "fires_with_weather_2026-07-26_to_2026-07-29.csv").touch()

            start_date, end_date = next_firms_date_range(
                results_directory / "firms_range_state.json",
                window_days=4,
                results_directory=results_directory,
            )

        self.assertEqual((start_date, end_date), (date(2026, 7, 22), date(2026, 7, 25)))

    def test_saved_range_moves_backward_across_a_year_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "firms_range_state.json"
            save_completed_firms_range(state_path, date(2026, 1, 1), date(2026, 1, 4))

            start_date, end_date = next_firms_date_range(state_path, window_days=4)

        self.assertEqual((start_date, end_date), (date(2025, 12, 28), date(2025, 12, 31)))

    def test_defaults_to_a_window_ending_on_the_supplied_fallback_date(self):
        with tempfile.TemporaryDirectory() as directory:
            start_date, end_date = next_firms_date_range(
                Path(directory) / "firms_range_state.json",
                window_days=4,
                results_directory=Path(directory),
                fallback_end_date=date(2026, 7, 29),
            )

        self.assertEqual((start_date, end_date), (date(2026, 7, 26), date(2026, 7, 29)))

    def test_names_export_from_the_queried_inclusive_range(self):
        filename = firms_range_filename(date(2026, 7, 22), date(2026, 7, 25))

        self.assertEqual(filename, "fires_with_weather_2026-07-22_to_2026-07-25.csv")
