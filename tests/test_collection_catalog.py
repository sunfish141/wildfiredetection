import unittest

from wildfire_data.collection_catalog import (
    COLLECTION_TARGETS,
    target_by_key,
    targets_for_entity,
    validate_collection_catalog,
)


class CollectionCatalogTests(unittest.TestCase):
    def test_catalog_has_the_required_collect_once_capabilities(self):
        validate_collection_catalog()

        entities = {target.entity for target in COLLECTION_TARGETS}
        self.assertTrue(
            {
                "fire_detection",
                "satellite_observation_mask",
                "operational_perimeter",
                "forecast_weather",
                "static_spatial_asset",
            }.issubset(entities)
        )

    def test_forecast_targets_cover_both_operational_regions(self):
        forecasts = targets_for_entity("forecast_weather")

        self.assertEqual({target.region for target in forecasts}, {"United States", "Canada"})

    def test_finds_targets_by_stable_key(self):
        self.assertEqual(target_by_key("feds_nrt").provider, "NASA FEDS")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            target_by_key("not-configured")
