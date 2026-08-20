import unittest

import pandas as pd

from wildfire_data.firms_filtering import filter_firms_by_minimum_brightness


class FirmsBrightnessFilteringTests(unittest.TestCase):
    def test_keeps_only_detections_at_or_above_the_minimum_brightness(self):
        fires = pd.DataFrame(
            {
                "bright_ti4": [304.9, 305.0, 305.1],
                "detection_id": ["below", "at-threshold", "above"],
            }
        )

        filtered = filter_firms_by_minimum_brightness(fires, minimum_bright_ti4=305)

        self.assertEqual(filtered["detection_id"].tolist(), ["at-threshold", "above"])
