import json
import unittest
from datetime import datetime, timezone

from wildfire_data.firms_normalization import (
    FirmsRecordValidationError,
    normalize_firms_detection,
)


class FirmsNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.source_fields = {
            "latitude": "54.1234",
            "longitude": -106.4321,
            "acq_date": "2026-07-26",
            "acq_time": "0040",
            "bright_ti4": "304.9",
            "confidence": "nominal",
            "future_firms_field": {"quality": ["clear", 1]},
        }
        self.provenance = {
            "provider": "NASA FIRMS",
            "product": "VIIRS_SNPP_NRT",
            "schema_version": "firms-normalized/v1",
            "raw_artifact_id": "sha256:raw-file",
            "raw_record_offset": 17,
            "ingestion_id": "ingestion-2026-07-26-01",
            "ingested_at": datetime(2026, 7, 26, 1, 0, tzinfo=timezone.utc),
            "collector_extension": {"request_attempt": 2},
        }

    def test_preserves_all_source_fields_and_provenance_as_json_safe_values(self):
        normalized = normalize_firms_detection(
            self.source_fields,
            provenance=self.provenance,
        )

        self.assertEqual(
            normalized["raw_source_fields"],
            {
                "latitude": "54.1234",
                "longitude": -106.4321,
                "acq_date": "2026-07-26",
                "acq_time": "0040",
                "bright_ti4": "304.9",
                "confidence": "nominal",
                "future_firms_field": {"quality": ["clear", 1]},
            },
        )
        self.assertEqual(normalized["provenance"]["raw_artifact_id"], "sha256:raw-file")
        self.assertEqual(normalized["provenance"]["raw_record_offset"], 17)
        self.assertEqual(normalized["provenance"]["ingestion_id"], "ingestion-2026-07-26-01")
        self.assertEqual(normalized["provenance"]["provider"], "NASA FIRMS")
        self.assertEqual(normalized["provenance"]["product"], "VIIRS_SNPP_NRT")
        self.assertEqual(normalized["provenance"]["schema_version"], "firms-normalized/v1")
        self.assertEqual(normalized["provenance"]["ingested_at"], "2026-07-26T01:00:00Z")
        self.assertEqual(normalized["provenance"]["collector_extension"], {"request_attempt": 2})
        json.dumps(normalized, allow_nan=False)

    def test_derives_stable_ids_without_using_ingestion_metadata(self):
        first = normalize_firms_detection(self.source_fields, provenance=self.provenance)
        later_provenance = {
            **self.provenance,
            "ingestion_id": "ingestion-2026-07-26-02",
            "ingested_at": "2026-07-26T02:00:00Z",
            "raw_record_offset": 99,
        }
        second = normalize_firms_detection(self.source_fields, provenance=later_provenance)

        self.assertEqual(first["source_id"], second["source_id"])
        self.assertEqual(first["detection_id"], second["detection_id"])
        self.assertTrue(first["source_id"].startswith("firms-source:"))
        self.assertTrue(first["detection_id"].startswith("firms-detection:"))

    def test_uses_native_source_detection_id_when_supplied(self):
        first = normalize_firms_detection(
            self.source_fields,
            provenance={**self.provenance, "source_native_id": "native-123"},
        )
        changed_fields = {**self.source_fields, "future_firms_field": "updated"}
        second = normalize_firms_detection(
            changed_fields,
            provenance={**self.provenance, "source_native_id": "native-123"},
        )

        self.assertEqual(first["detection_id"], second["detection_id"])

    def test_derives_utc_acquisition_time_from_date_and_zero_padded_time(self):
        normalized = normalize_firms_detection(self.source_fields, provenance=self.provenance)

        self.assertEqual(normalized["acquired_at"], "2026-07-26T00:40:00Z")
        self.assertEqual(normalized["latitude"], 54.1234)
        self.assertEqual(normalized["longitude"], -106.4321)

    def test_accepts_integer_acquisition_time_after_csv_leading_zero_loss(self):
        normalized = normalize_firms_detection(
            {**self.source_fields, "acq_time": 40},
            provenance=self.provenance,
        )

        self.assertEqual(normalized["acquired_at"], "2026-07-26T00:40:00Z")

    def test_keeps_rows_below_ti4_threshold_and_marks_the_derived_decision(self):
        normalized = normalize_firms_detection(
            self.source_fields,
            provenance=self.provenance,
            minimum_bright_ti4=305,
        )

        self.assertEqual(normalized["bright_ti4"], 304.9)
        self.assertEqual(
            normalized["derived"]["ti4_threshold"],
            {"minimum_bright_ti4": 305.0, "passes": False},
        )

    def test_reports_no_threshold_decision_when_no_threshold_is_configured(self):
        normalized = normalize_firms_detection(
            self.source_fields,
            provenance=self.provenance,
            minimum_bright_ti4=None,
        )

        self.assertEqual(
            normalized["derived"]["ti4_threshold"],
            {"minimum_bright_ti4": None, "passes": None},
        )

    def test_rejects_missing_or_invalid_core_fields(self):
        with self.assertRaisesRegex(FirmsRecordValidationError, "bright_ti4"):
            normalize_firms_detection(
                {key: value for key, value in self.source_fields.items() if key != "bright_ti4"},
                provenance=self.provenance,
            )
        with self.assertRaisesRegex(FirmsRecordValidationError, "latitude"):
            normalize_firms_detection(
                {**self.source_fields, "latitude": 91},
                provenance=self.provenance,
            )
        with self.assertRaisesRegex(FirmsRecordValidationError, "acq_time"):
            normalize_firms_detection(
                {**self.source_fields, "acq_time": "2460"},
                provenance=self.provenance,
            )

    def test_requires_provider_and_product_for_a_stable_source_id(self):
        with self.assertRaisesRegex(FirmsRecordValidationError, "provider"):
            normalize_firms_detection(
                self.source_fields,
                provenance={"product": "VIIRS_SNPP_NRT"},
            )
        with self.assertRaisesRegex(FirmsRecordValidationError, "product"):
            normalize_firms_detection(
                self.source_fields,
                provenance={"provider": "NASA FIRMS"},
            )


if __name__ == "__main__":
    unittest.main()
