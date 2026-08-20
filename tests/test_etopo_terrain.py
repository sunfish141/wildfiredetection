import gzip
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

from wildfire_data.data_archive import CoverageLedger, CoverageStatus
from wildfire_data.etopo_terrain import (
    ETOPO_RESOLUTION_DEGREES,
    collect_etopo_terrain,
    context_tile_ids_from_detections,
    derive_terrain_features,
    etopo_source_blocks,
    webmercator_context_tile_id,
    webmercator_tile_bounds,
)
from wildfire_data.storage_budget import load_storage_budget


class _FakeResponse:
    def __init__(self, content, *, status_code=200):
        self.content = content
        self.status_code = status_code
        self.headers = {"Content-Type": "image/tiff"}


class _TiffSession:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        width, height = (int(value) for value in params["size"].split(","))
        west, _south, _east, north = (float(value) for value in params["bbox"].split(","))
        values = np.add.outer(np.arange(height, dtype=np.int16), np.arange(width, dtype=np.int16))
        return _FakeResponse(_geotiff(values, west=west, north=north))


def _geotiff(values, *, west, north):
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[33550] = (ETOPO_RESOLUTION_DEGREES, ETOPO_RESOLUTION_DEGREES, 0.0)
    tags[33922] = (0.0, 0.0, 0.0, west, north, 0.0)
    output = io.BytesIO()
    Image.fromarray(values.astype(np.int16)).save(
        output,
        format="TIFF",
        compression="tiff_lzw",
        tiffinfo=tags,
    )
    return output.getvalue()


def _policy(path, *, total=100_000_000, static_cap=100_000_000):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "whole_data_cap_bytes": total,
                "whole_data_cap_label": "test",
                "scope": "test data",
                "categories": [
                    {
                        "key": "static_cell_features",
                        "cap_bytes": static_cap,
                        "priority_score": 70,
                        "pinned": True,
                        "retention": "terrain",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return load_storage_budget(path)


class EtopoTerrainTests(unittest.TestCase):
    def test_context_tiles_are_deterministic_and_have_wgs84_bounds(self):
        tile_id = webmercator_context_tile_id(65.9, -114.2)
        west, south, east, north = webmercator_tile_bounds(tile_id)

        self.assertTrue(tile_id.startswith("webmercator-96km-x"))
        self.assertLessEqual(west, -114.2)
        self.assertLessEqual(south, 65.9)
        self.assertGreater(east, -114.2)
        self.assertGreater(north, 65.9)

    def test_blocks_add_a_one_pixel_halo_and_preserve_context_ids(self):
        tile_id = webmercator_context_tile_id(52.0, -116.0)
        blocks = etopo_source_blocks([tile_id], block_degrees=10.0)

        self.assertGreaterEqual(len(blocks), 1)
        self.assertTrue(all(tile_id in block.context_tile_ids for block in blocks))
        self.assertTrue(all(block.width >= 2_400 and block.height >= 2_400 for block in blocks))
        self.assertTrue(all(block.request_west <= block.west for block in blocks))
        self.assertTrue(all(block.request_north >= block.north for block in blocks))

    def test_derives_downhill_aspect_and_quantized_slope(self):
        elevation = np.tile(np.arange(5, dtype=np.int16) * 100, (5, 1))
        features = derive_terrain_features(
            elevation,
            north_latitude=55.0,
            pixel_width_degrees=0.01,
            pixel_height_degrees=0.01,
        )

        self.assertEqual(features.elevation_m.dtype, np.int16)
        self.assertEqual(features.slope_degrees_x2.dtype, np.uint8)
        self.assertEqual(features.aspect_degrees_x2.dtype, np.uint8)
        self.assertGreater(features.slope_degrees_x2[2, 2], 0)
        self.assertEqual(features.aspect_degrees_x2[2, 2], 135)  # west, 270 / 2

    def test_collects_raw_and_compact_features_and_is_idempotent(self):
        tile_ids = context_tile_ids_from_detections(
            [{"latitude": 52.0, "longitude": -116.0}]
        )
        session = _TiffSession()
        captured_at = datetime(2026, 8, 18, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            policy = _policy(Path(directory) / "budget.json")
            first = collect_etopo_terrain(
                root,
                context_tile_ids=tile_ids,
                storage_budget=policy,
                session=session,
                block_degrees=1.0,
                retrieved_at=captured_at,
            )
            static_paths = [block.static_artifact_path for block in first.blocks]
            sample_path = next(path for path in static_paths if path is not None)
            with np.load(sample_path) as compact:
                self.assertEqual(compact["elevation_m"].dtype, np.int16)
                self.assertEqual(compact["slope_degrees_x2"].dtype, np.uint8)
                self.assertEqual(compact["aspect_degrees_x2"].dtype, np.uint8)
            normalized = next(root.glob("normalized/static-cell-features/**/*.jsonl.gz"))
            with gzip.open(normalized, "rt", encoding="utf-8") as source:
                record = json.loads(source.readline())
            repeat_session = _TiffSession()
            second = collect_etopo_terrain(
                root,
                context_tile_ids=tile_ids,
                storage_budget=policy,
                session=repeat_session,
                block_degrees=1.0,
                retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )
            coverage = CoverageLedger(root).entries()

        self.assertEqual(first.complete_count, len(first.blocks))
        self.assertGreater(len(session.calls), 0)
        self.assertEqual(record["dataset"], "ETOPO_2022_v1_15s_surface_elev")
        self.assertEqual(record["retention_priority_score"], 70)
        self.assertTrue(all(block.skipped_terminal_coverage for block in second.blocks))
        self.assertEqual(repeat_session.calls, [])
        self.assertTrue(all(entry.status == CoverageStatus.COMPLETE for entry in coverage))

    def test_records_partial_before_any_request_when_static_quota_is_too_small(self):
        tile_ids = (webmercator_context_tile_id(52.0, -116.0),)
        session = _TiffSession()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            result = collect_etopo_terrain(
                root,
                context_tile_ids=tile_ids,
                storage_budget=_policy(Path(directory) / "budget.json", total=1_000, static_cap=1_000),
                session=session,
                block_degrees=1.0,
                retrieved_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
            )

        self.assertEqual(result.blocks[0].coverage.status, CoverageStatus.PARTIAL)
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()
