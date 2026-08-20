import gzip
import json
import tempfile
import unittest
from datetime import datetime, timezone

from wildfire_data.normalized_storage import (
    NormalizedStorageIntegrityError,
    write_normalized_jsonl,
)


class NormalizedStorageTests(unittest.TestCase):
    def test_writes_lossless_records_to_an_immutable_partitioned_artifact(self):
        records = [
            {
                "detection_id": "detection-1",
                "raw_source_fields": {"new_provider_field": ["kept", 1]},
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            artifact = write_normalized_jsonl(
                directory,
                entity="fire_detections",
                records=records,
                partitions={"acq_date": "2026-07-26"},
                raw_artifact_ids=["a" * 64],
                transformation_version="firms-normalized/v1",
                generated_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
            )
            with gzip.open(artifact.artifact_path, "rt", encoding="utf-8") as artifact_file:
                stored = [json.loads(line) for line in artifact_file]
            manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(stored, records)
        self.assertEqual(artifact.record_count, 1)
        self.assertTrue(artifact.created)
        self.assertIn("normalized/fire-detections/acq-date=2026-07-26", str(artifact.artifact_path))
        self.assertEqual(manifest["raw_artifact_ids"], ["a" * 64])

    def test_reuses_equivalent_content_without_overwriting_the_artifact(self):
        arguments = {
            "entity": "fire_detections",
            "records": [{"detection_id": "detection-1"}],
            "partitions": {"acq_date": "2026-07-26"},
            "raw_artifact_ids": ["a" * 64],
            "transformation_version": "firms-normalized/v1",
            "generated_at": datetime(2026, 7, 26, tzinfo=timezone.utc),
        }
        with tempfile.TemporaryDirectory() as directory:
            first = write_normalized_jsonl(directory, **arguments)
            second = write_normalized_jsonl(directory, **arguments)

        self.assertEqual(first.artifact_path, second.artifact_path)
        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertNotEqual(first.manifest_path, second.manifest_path)

    def test_rejects_an_existing_artifact_with_a_mismatched_content_hash(self):
        arguments = {
            "entity": "fire_detections",
            "records": [{"detection_id": "detection-1"}],
            "partitions": {"acq_date": "2026-07-26"},
            "raw_artifact_ids": ["a" * 64],
            "transformation_version": "firms-normalized/v1",
        }
        with tempfile.TemporaryDirectory() as directory:
            first = write_normalized_jsonl(directory, **arguments)
            first.artifact_path.write_bytes(gzip.compress(b'{"corrupted":true}\n', mtime=0))

            with self.assertRaises(NormalizedStorageIntegrityError):
                write_normalized_jsonl(directory, **arguments)

    def test_requires_at_least_one_source_record_and_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "raw_artifact_ids"):
                write_normalized_jsonl(
                    directory,
                    entity="fire_detections",
                    records=[{"detection_id": "detection-1"}],
                    partitions={"acq_date": "2026-07-26"},
                    raw_artifact_ids=[],
                    transformation_version="firms-normalized/v1",
                )
