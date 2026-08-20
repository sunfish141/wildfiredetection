import csv
import json
import tempfile
import unittest
from pathlib import Path

from wildfire_data.storage_budget import (
    StorageBudgetError,
    assess_admission,
    category_for_relative_path,
    load_storage_budget,
    measure_storage_usage,
    require_admission,
    write_storage_inventory,
)


class StorageBudgetTests(unittest.TestCase):
    def _write_policy(self, directory, *, total=100, firms_cap=50, labels_cap=40):
        policy_path = Path(directory) / "budget.json"
        policy_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "whole_data_cap_bytes": total,
                    "whole_data_cap_label": "test",
                    "scope": "test data root",
                    "categories": [
                        {
                            "key": "firms_and_detection_evidence",
                            "cap_bytes": firms_cap,
                            "priority_score": 100,
                            "pinned": True,
                            "retention": "FIRMS",
                        },
                        {
                            "key": "operational_labels_and_progression",
                            "cap_bytes": labels_cap,
                            "priority_score": 95,
                            "pinned": True,
                            "retention": "labels",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return policy_path

    def test_measures_every_existing_file_and_assigns_known_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            firms_path = root / "raw" / "nasa-firms" / "rows.gz"
            labels_path = root / "raw" / "nifc-wfigs" / "revisions.gz"
            firms_path.parent.mkdir(parents=True)
            labels_path.parent.mkdir(parents=True)
            firms_path.write_bytes(b"a" * 10)
            labels_path.write_bytes(b"b" * 7)

            usage = measure_storage_usage(root)

        self.assertEqual(usage.total_bytes, 17)
        self.assertEqual(usage.category_bytes["firms_and_detection_evidence"], 10)
        self.assertEqual(usage.category_bytes["operational_labels_and_progression"], 7)

    def test_rejects_a_write_that_exceeds_a_category_cap_before_the_whole_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            path = root / "raw" / "nasa-firms" / "rows.gz"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"a" * 45)
            policy = load_storage_budget(self._write_policy(directory))

            admission = assess_admission(
                policy,
                root,
                category="firms_and_detection_evidence",
                requested_bytes=6,
            )
            with self.assertRaises(StorageBudgetError):
                require_admission(
                    policy,
                    root,
                    category="firms_and_detection_evidence",
                    requested_bytes=6,
                )

        self.assertFalse(admission.allowed)
        self.assertIn("category cap", admission.reason)

    def test_rejects_a_write_that_exceeds_the_whole_data_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            path = root / "raw" / "nifc-wfigs" / "revisions.gz"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"a" * 95)
            policy = load_storage_budget(self._write_policy(directory, firms_cap=1, labels_cap=99))

            admission = assess_admission(
                policy,
                root,
                category="operational_labels_and_progression",
                requested_bytes=6,
            )

        self.assertFalse(admission.allowed)
        self.assertEqual(admission.reason, "whole-data cap would be exceeded")

    def test_writes_a_scored_csv_without_mutating_provider_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            source_path = root / "raw" / "nasa-firms" / "rows.gz"
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"source")
            policy = load_storage_budget(self._write_policy(directory))

            output = write_storage_inventory(policy, root)
            with output.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            source_contents = source_path.read_bytes()
            final_usage = measure_storage_usage(root)

        firms_row = next(row for row in rows if row["category"] == "firms_and_detection_evidence")
        whole_row = next(row for row in rows if row["category"] == "__whole_data__")
        self.assertEqual(firms_row["retention_priority_score"], "100")
        self.assertEqual(firms_row["used_bytes"], "6")
        self.assertEqual(int(whole_row["used_bytes"]), final_usage.total_bytes)
        self.assertEqual(source_contents, b"source")

    def test_maps_l2_fire_and_geolocation_paths_to_the_same_paired_cutout_category(self):
        self.assertEqual(
            category_for_relative_path("raw/nasa-lp-daac-viirs-l2-observability/fire.nc.gz"),
            "viirs_l2_paired_cutouts",
        )
        self.assertEqual(
            category_for_relative_path("raw/nasa-laads-viirs-geolocation/geo.nc.gz"),
            "viirs_l2_paired_cutouts",
        )

    def test_maps_cwfis_incident_history_to_the_operational_context_category(self):
        self.assertEqual(
            category_for_relative_path("raw/cwfis-cwfis-active-fires/page.json.gz"),
            "operational_labels_and_progression",
        )
        self.assertEqual(
            category_for_relative_path("normalized/incident-snapshots/page.jsonl.gz"),
            "operational_labels_and_progression",
        )

    def test_maps_etopo_raw_subsets_and_compact_blocks_to_static_features(self):
        self.assertEqual(
            category_for_relative_path("raw/noaa-ncei-etopo-terrain/response.tiff.gz"),
            "static_cell_features",
        )

    def test_maps_nalcms_raw_source_archives_to_static_features(self):
        self.assertEqual(
            category_for_relative_path("raw/cec-nalcms-land-cover/source.zip.gz"),
            "static_cell_features",
        )
        self.assertEqual(
            category_for_relative_path("static/etopo-2022-15s/block.npz"),
            "static_cell_features",
        )


if __name__ == "__main__":
    unittest.main()
