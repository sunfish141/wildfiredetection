import unittest

import pandas as pd

from wildfire_data.firms_preparation import prepare_firms_for_weather


class FirmsPreparationTests(unittest.TestCase):
    def test_keeps_extra_firms_fields_in_the_weather_ready_view(self):
        fires = pd.DataFrame(
            {
                "latitude": [54.0, 55.0],
                "longitude": [-106.0, -107.0],
                "bright_ti4": [304.9, 306.0],
                "acq_date": ["2026-07-26", "2026-07-26"],
                "acq_time": [40, 55],
                "frp": [1.2, 2.3],
                "confidence": ["n", "h"],
                "future_firms_field": ["below", "kept"],
            }
        )

        prepared = prepare_firms_for_weather(fires, minimum_bright_ti4=305)

        self.assertEqual(prepared.input_detections, 2)
        self.assertEqual(prepared.detections_with_required_fields, 2)
        self.assertEqual(prepared.fires["future_firms_field"].tolist(), ["kept"])
        self.assertEqual(prepared.fires["frp"].tolist(), [2.3])
        self.assertEqual(prepared.fires["acq_time"].tolist(), ["0055"])
        self.assertEqual(str(prepared.fires["weather_hour"].iloc[0]), "2026-07-26 00:00:00+00:00")

    def test_requires_all_core_firms_columns(self):
        with self.assertRaisesRegex(ValueError, "bright_ti4"):
            prepare_firms_for_weather(
                pd.DataFrame({"latitude": [54], "longitude": [-106]}),
                minimum_bright_ti4=305,
            )
