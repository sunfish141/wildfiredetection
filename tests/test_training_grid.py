import unittest
from datetime import datetime, timedelta, timezone

from wildfire_data.training_grid import (
    DEFAULT_CELL_SIZE_METRES,
    TrainingExampleKey,
    TrainingGridError,
    anchor_time_bin,
    cell_from_id,
    cell_from_wgs84,
    cells_in_square_radius,
)


class TrainingGridTests(unittest.TestCase):
    def test_wgs84_point_has_a_stable_equal_area_cell_and_round_trips(self):
        cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)
        self.assertEqual(cell_from_id(cell.cell_id), cell)
        xmin, ymin, xmax, ymax = cell.bounds_projected
        self.assertEqual(xmax - xmin, DEFAULT_CELL_SIZE_METRES)
        self.assertEqual(ymax - ymin, DEFAULT_CELL_SIZE_METRES)
        latitude, longitude = cell.center_wgs84
        self.assertAlmostEqual(latitude, 53.5461, places=2)
        self.assertAlmostEqual(longitude, -113.4938, places=2)

    def test_cell_ids_are_not_collection_tile_ids(self):
        with self.assertRaises(TrainingGridError):
            cell_from_id("webmercator-96km-12-7")

    def test_neighbourhood_is_deterministic_and_includes_the_centre(self):
        centre = cell_from_wgs84(latitude=50.0, longitude=-110.0)
        cells = cells_in_square_radius(centre, radius_cells=1)
        self.assertEqual(len(cells), 9)
        self.assertIn(centre, cells)
        self.assertEqual(cells[0].x_index, centre.x_index - 1)
        self.assertEqual(cells[0].y_index, centre.y_index - 1)

    def test_anchor_is_floored_to_12_hour_utc_bin(self):
        value = datetime(2026, 7, 1, 18, 59, tzinfo=timezone.utc)
        self.assertEqual(
            anchor_time_bin(value),
            datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
        )

    def test_example_key_allows_a_cell_specific_12_hour_phase(self):
        cell = cell_from_wgs84(latitude=50.0, longitude=-110.0)
        key = TrainingExampleKey(
            cell_id=cell.cell_id,
            anchor_at=datetime(2026, 7, 1, 12, tzinfo=timezone.utc),
        )
        self.assertEqual(key.target_end_at, datetime(2026, 7, 2, tzinfo=timezone.utc))
        self.assertEqual(len(key.example_id), 64)
        cell_local_phase_key = TrainingExampleKey(
            cell_id=cell.cell_id,
            anchor_at=datetime(2026, 7, 1, 19, 34, tzinfo=timezone.utc),
        )
        self.assertEqual(
            cell_local_phase_key.target_end_at,
            datetime(2026, 7, 2, 7, 34, tzinfo=timezone.utc),
        )
        with self.assertRaises(TrainingGridError):
            anchor_time_bin(
                datetime(2026, 7, 1, 12, tzinfo=timezone.utc),
                horizon=timedelta(hours=6),
            )


if __name__ == "__main__":
    unittest.main()
