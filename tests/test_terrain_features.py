import tempfile
import unittest
from pathlib import Path

import numpy as np

from wildfire_data.etopo_terrain import NO_DATA_ASPECT_DEGREES_X2, NO_DATA_ELEVATION_METRES
from wildfire_data.terrain_features import (
    TERRAIN_SAMPLING_METHOD,
    TerrainFeatureError,
    TerrainFeatureSampler,
    sample_terrain_features,
)
from wildfire_data.training_grid import cell_from_wgs84


class TerrainFeatureSamplerTests(unittest.TestCase):
    def test_samples_quantized_features_at_a_canonical_cell_centre(self):
        cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)
        latitude, longitude = cell.center_wgs84
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _write_block(
                root,
                "main",
                latitude=latitude,
                longitude=longitude,
                column=1,
                row=1,
                elevation=1234,
                slope_x2=17,
                aspect_x2=45,
            )
            features = TerrainFeatureSampler(root).sample_cell(cell)

        self.assertTrue(features["terrain_valid"])
        self.assertEqual(features["terrain_elevation_m"], 1234.0)
        self.assertEqual(features["terrain_slope_degrees"], 8.5)
        self.assertTrue(features["terrain_aspect_defined"])
        self.assertAlmostEqual(features["terrain_aspect_sin"], 1.0)
        self.assertAlmostEqual(features["terrain_aspect_cos"], 0.0, places=12)
        self.assertEqual(features["terrain_sampling_method"], TERRAIN_SAMPLING_METHOD)
        self.assertEqual(features["terrain_source_block_id"], "main")
        self.assertEqual(features["terrain_source_pixel_row"], 1)
        self.assertEqual(features["terrain_source_pixel_column"], 1)

    def test_prefers_an_interior_pixel_over_an_overlapping_halo(self):
        cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)
        latitude, longitude = cell.center_wgs84
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _write_block(
                root,
                "halo",
                latitude=latitude,
                longitude=longitude,
                column=0,
                row=0,
                elevation=111,
            )
            _write_block(
                root,
                "interior",
                latitude=latitude,
                longitude=longitude,
                column=1,
                row=1,
                elevation=222,
            )
            features = TerrainFeatureSampler(root).sample_cell(cell)

        self.assertEqual(features["terrain_elevation_m"], 222.0)
        self.assertEqual(features["terrain_source_block_id"], "interior")

    def test_uses_a_stable_path_tiebreak_for_equally_interior_overlap(self):
        cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)
        latitude, longitude = cell.center_wgs84
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _write_block(root, "zeta", latitude=latitude, longitude=longitude, column=1, row=1, elevation=999)
            _write_block(root, "alpha", latitude=latitude, longitude=longitude, column=1, row=1, elevation=111)
            features = TerrainFeatureSampler(root).sample_cell(cell)

        self.assertEqual(features["terrain_elevation_m"], 111.0)
        self.assertEqual(features["terrain_source_block_id"], "alpha")

    def test_uses_documented_half_open_source_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _write_block(
                root,
                "bounds",
                latitude=60.0,
                longitude=-120.0,
                column=0,
                row=0,
                elevation=321,
            )
            sampler = TerrainFeatureSampler(root)
            northwest_edge = sampler.sample_wgs84(latitude=60.0, longitude=-120.0)
            southeast_outer_edge = sampler.sample_wgs84(latitude=59.97, longitude=-119.97)

        self.assertTrue(northwest_edge["terrain_valid"])
        self.assertEqual(northwest_edge["terrain_source_pixel_row"], 0)
        self.assertEqual(northwest_edge["terrain_source_pixel_column"], 0)
        self.assertFalse(southeast_outer_edge["terrain_valid"])
        self.assertEqual(
            southeast_outer_edge["terrain_coverage_status"],
            "outside-retained-terrain-coverage",
        )

    def test_emits_neutral_aspect_for_flat_or_undefined_direction(self):
        cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)
        latitude, longitude = cell.center_wgs84
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _write_block(
                root,
                "flat",
                latitude=latitude,
                longitude=longitude,
                column=1,
                row=1,
                elevation=100,
                slope_x2=0,
                aspect_x2=int(NO_DATA_ASPECT_DEGREES_X2),
            )
            features = TerrainFeatureSampler(root).sample_cell(cell)

        self.assertTrue(features["terrain_valid"])
        self.assertFalse(features["terrain_aspect_defined"])
        self.assertEqual(features["terrain_aspect_sin"], 0.0)
        self.assertEqual(features["terrain_aspect_cos"], 0.0)

    def test_distinguishes_missing_coverage_and_source_no_data(self):
        cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)
        latitude, longitude = cell.center_wgs84
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            missing = TerrainFeatureSampler(root).sample_cell(cell)
            _write_block(
                root,
                "no-data",
                latitude=latitude,
                longitude=longitude,
                column=1,
                row=1,
                elevation=int(NO_DATA_ELEVATION_METRES),
            )
            no_data = TerrainFeatureSampler(root).sample_cell(cell)

        self.assertFalse(missing["terrain_valid"])
        self.assertEqual(missing["terrain_coverage_status"], "no-terrain-artifacts")
        self.assertFalse(no_data["terrain_valid"])
        self.assertEqual(no_data["terrain_coverage_status"], "source-no-data")
        self.assertEqual(no_data["terrain_source_block_id"], "no-data")

    def test_outside_retained_blocks_is_not_silently_interpolated(self):
        cell = cell_from_wgs84(latitude=53.5461, longitude=-113.4938)
        latitude, longitude = cell.center_wgs84
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _write_block(
                root,
                "far-away",
                latitude=latitude + 10.0,
                longitude=longitude,
                column=1,
                row=1,
            )
            features = sample_terrain_features(root, cell=cell, max_cached_blocks=0)

        self.assertFalse(features["terrain_valid"])
        self.assertEqual(features["terrain_coverage_status"], "outside-retained-terrain-coverage")
        self.assertIsNone(features["terrain_elevation_m"])

    def test_rejects_malformed_compact_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            path = root / "static" / "etopo-2022-15s" / "bad.npz"
            path.parent.mkdir(parents=True)
            np.savez_compressed(
                path,
                elevation_m=np.zeros((2, 2), dtype=np.int16),
                slope_degrees_x2=np.zeros((2, 2), dtype=np.uint8),
                aspect_degrees_x2=np.zeros((2, 2), dtype=np.uint8),
                grid_west=np.float64(-120.0),
                grid_north=np.float64(60.0),
                pixel_width_degrees=np.float64(-0.01),
                pixel_height_degrees=np.float64(0.01),
            )
            with self.assertRaises(TerrainFeatureError):
                TerrainFeatureSampler(root)


def _write_block(
    root: Path,
    name: str,
    *,
    latitude: float,
    longitude: float,
    column: int,
    row: int,
    elevation: int = 100,
    slope_x2: int = 0,
    aspect_x2: int = int(NO_DATA_ASPECT_DEGREES_X2),
) -> None:
    """Write a tiny north-up source block whose chosen pixel contains a point."""
    width = height = 3
    resolution = 0.01
    west = longitude - column * resolution
    north = latitude + row * resolution
    elevations = np.full((height, width), elevation, dtype=np.int16)
    slopes = np.full((height, width), slope_x2, dtype=np.uint8)
    aspects = np.full((height, width), aspect_x2, dtype=np.uint8)
    path = root / "static" / "etopo-2022-15s" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        elevation_m=elevations,
        slope_degrees_x2=slopes,
        aspect_degrees_x2=aspects,
        grid_west=np.float64(west),
        grid_north=np.float64(north),
        pixel_width_degrees=np.float64(resolution),
        pixel_height_degrees=np.float64(resolution),
    )


if __name__ == "__main__":
    unittest.main()
