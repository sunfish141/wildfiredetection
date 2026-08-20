import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from wildfire_data.collect_viirs_l2 import main


class CollectViirsL2CommandTests(unittest.TestCase):
    def test_dry_run_succeeds_even_though_it_intentionally_leaves_downloads_pending(self):
        result = SimpleNamespace(
            inventory_response_count=3,
            discovered_granule_count=12,
            incomplete_window_count=3,
        )
        with (
            patch("dotenv.load_dotenv"),
            patch.dict(os.environ, {"EARTHDATA_TOKEN": ""}, clear=False),
            patch("wildfire_data.collect_viirs_l2.collect_viirs_l2_range", return_value=result) as collect,
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(
                [
                    "--start",
                    "2026-05-31",
                    "--end",
                    "2026-08-10",
                    "--platform",
                    "snpp",
                    "--dry-run",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(collect.call_args.kwargs["dry_run"])
        self.assertEqual(collect.call_args.kwargs["earthdata_token"], "")

    def test_rejects_a_fire_file_download_without_the_legacy_override(self):
        with (
            patch("dotenv.load_dotenv"),
            patch.dict(os.environ, {"EARTHDATA_TOKEN": "test-token"}, clear=False),
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                main(["--start", "2026-05-31", "--end", "2026-08-10"])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
