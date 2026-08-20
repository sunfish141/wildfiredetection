import unittest
from unittest.mock import patch

import numpy as np

import wildfire_data.weather_source_selection as weather_source_selection
from wildfire_data.weather_source_selection import minimum_covering_sources


class MinimumCoveringSourcesTests(unittest.TestCase):
    def test_selects_one_source_for_nearby_points(self):
        points = np.array([
            [-106.0000, 54.0000],
            [-106.0050, 54.0000],
            [-106.0000, 54.0050],
        ])

        cover = minimum_covering_sources(points, radius_m=1_000)

        self.assertEqual(len(cover.source_indices), 1)
        self.assertTrue(np.all(cover.distances_m <= 1_000))

    def test_selects_separate_sources_beyond_the_radius(self):
        points = np.array([
            [-106.0000, 54.0000],
            [-106.0300, 54.0000],
        ])

        cover = minimum_covering_sources(points, radius_m=1_000)

        self.assertEqual(len(cover.source_indices), 2)
        self.assertTrue(np.array_equal(cover.assigned_source_indices, np.array([0, 1])))

    def test_preserves_assignments_for_duplicate_points(self):
        points = np.array([
            [-106.0000, 54.0000],
            [-106.0000, 54.0000],
            [-106.0050, 54.0000],
        ])

        cover = minimum_covering_sources(points, radius_m=1_000)

        self.assertEqual(len(cover.source_indices), 1)
        self.assertTrue(np.all(cover.distances_m <= 1_000))

    def test_covers_a_nontrivial_component(self):
        # Consecutive points are about 800 m apart: this is a four-point path
        # with no source capable of covering all four points by itself.
        points = np.array([
            [-106.0000, 54.0000],
            [-106.0000, 54.0072],
            [-106.0000, 54.0144],
            [-106.0000, 54.0216],
        ])

        cover = minimum_covering_sources(points, radius_m=1_000)

        self.assertEqual(len(cover.source_indices), 2)
        self.assertTrue(np.all(cover.distances_m <= 1_000))

    def test_selects_deterministic_sources_after_input_is_shuffled(self):
        points = np.array([
            [-106.0000, 54.0000],
            [-106.0000, 54.0060],
            [-106.0000, 54.0140],
            [-106.0000, 54.0200],
            [-105.9500, 54.0000],
        ])
        shuffled = points[[3, 0, 4, 1, 2]]

        cover = minimum_covering_sources(points, radius_m=1_000)
        shuffled_cover = minimum_covering_sources(shuffled, radius_m=1_000)

        np.testing.assert_array_equal(
            points[cover.source_indices], shuffled[shuffled_cover.source_indices]
        )
        assigned_sources = {
            tuple(point): tuple(points[source_index])
            for point, source_index in zip(points, cover.assigned_source_indices)
        }
        shuffled_assigned_sources = {
            tuple(point): tuple(shuffled[source_index])
            for point, source_index in zip(shuffled, shuffled_cover.assigned_source_indices)
        }
        self.assertEqual(assigned_sources, shuffled_assigned_sources)

    def test_scales_to_49115_locations_without_an_optimizer(self):
        location_count = 49_115
        chain = np.array([
            [-170.0, 20.0000],
            [-170.0, 20.0072],
            [-170.0, 20.0144],
            [-170.0, 20.0216],
        ])
        positions = np.arange(location_count - len(chain))
        isolated_points = np.column_stack((
            -160.0 + (positions % 701) * 0.1,
            20.0 + (positions // 701) * 0.1,
        ))
        points = np.vstack((chain, isolated_points))

        with patch.object(
            weather_source_selection,
            "milp",
            side_effect=AssertionError("source selection must not invoke an optimizer"),
            create=True,
        ) as milp:
            cover = minimum_covering_sources(points, radius_m=1_000)

        milp.assert_not_called()
        self.assertEqual(len(cover.source_indices), location_count - 2)
        self.assertTrue(np.all(cover.distances_m <= 1_000))


if __name__ == "__main__":
    unittest.main()
