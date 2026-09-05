import unittest
from unittest.mock import patch
from datetime import datetime, timezone

import numpy as np
from fastapi.testclient import TestClient

from wildfire_data.incident_transition import IncidentTransitionModel
from wildfire_data.live_firms import LiveFirmsError, aggregate_current_firms
from wildfire_data.recursive_transition import RECURSIVE_MODEL_FEATURE_COLUMNS
from wildfire_data.web_app import create_app
from wildfire_data.training_grid import GridCell, cell_from_id


class SpreadEstimator:
    def predict_proba(self, values):
        return np.tile([.1, .9], (len(values), 1))


def terrain(_cell):
    return {"terrain_coverage_status": "sampled", "terrain_elevation_m": 1000.,
        "terrain_slope_degrees": 10., "terrain_aspect_sin": 0., "terrain_aspect_cos": 1.,
        "terrain_valid": True, "terrain_aspect_defined": True}


class WebAppTests(unittest.TestCase):
    def setUp(self):
        model = IncidentTransitionModel(SpreadEstimator(), feature_columns=RECURSIVE_MODEL_FEATURE_COLUMNS,
                                        ignition_threshold=.2)
        self.client = TestClient(create_app(model=model, terrain_provider=terrain))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def seed(self):
        response = self.client.post('/api/seed', json={"ignitions": [
            {"latitude": 53.02, "longitude": -117.31, "intensity": .8}]})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_home_assets_and_config(self):
        self.assertIn('Wildfire Atlas', self.client.get('/').text)
        self.assertEqual(self.client.get('/static/app.js').status_code, 200)
        config = self.client.get('/api/config').json()
        self.assertTrue(config['model_ready'])
        self.assertEqual(config['default_speed_seconds'], 3)
        self.assertIsNone(config['max_steps'])
        self.assertNotIn('MAP_KEY', config)

    def test_seed_step_and_stateless_replay(self):
        initial = self.seed()
        self.assertEqual(initial['state']['step_index'], 0)
        self.assertEqual(initial['points'][0]['intensity'], .8)
        request = {'state': initial['state'], 'origin_at': initial['origin_at']}
        response = self.client.post('/api/step', json=request)
        self.assertEqual(response.status_code, 200, response.text)
        advanced = response.json()
        self.assertEqual(advanced['elapsed_hours'], 12)
        self.assertEqual(advanced['new_ignition_count'], 1)
        self.assertEqual(advanced['terrain_missing_count'], 0)
        self.assertTrue(any(p['ignition_probability'] == .9 for p in advanced['points']))
        self.assertEqual(self.client.post('/api/step', json=request).json(), advanced)
        self.assertEqual(self.seed()['state']['step_index'], 0)

    def test_masking_and_continuation_beyond_96_hours(self):
        frame = self.seed()
        initial_cell = frame['state']['active_cells'][0]['cell_id']
        for _ in range(32):
            response = self.client.post('/api/step', json={'state': frame['state'], 'origin_at': frame['origin_at']})
            self.assertEqual(response.status_code, 200, response.text)
            frame = response.json()
        self.assertFalse(frame['finished'])
        self.assertEqual(frame['elapsed_hours'], 384)
        self.assertIn(initial_cell, frame['state']['burned_cell_ids'])
        self.assertNotIn(initial_cell, [c['cell_id'] for c in frame['state']['active_cells']])
        self.assertEqual(self.client.post('/api/step', json={'state': frame['state'], 'origin_at': frame['origin_at']}).status_code, 200)

    def test_empty_fire_can_continue_and_large_burned_mask_is_preserved(self):
        frame = self.seed()
        cell = cell_from_id(frame['state']['active_cells'][0]['cell_id'])
        burned = [GridCell(cell.x_index + i, cell.y_index).cell_id for i in range(1800)]
        state = {'step_index': 500, 'active_cells': [], 'burned_cell_ids': burned}
        response = self.client.post('/api/step', json={'state': state, 'origin_at': frame['origin_at']})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['elapsed_hours'], 6012)
        self.assertFalse(result['finished'])
        self.assertTrue(result['extinct'])
        self.assertEqual(set(result['state']['burned_cell_ids']), set(burned))

    def test_large_active_state_is_accepted(self):
        frame = self.seed()
        original = frame['state']['active_cells'][0]
        cell = cell_from_id(original['cell_id'])
        frame['state']['active_cells'] = [{**original, 'cell_id': GridCell(cell.x_index + i, cell.y_index).cell_id}
                                        for i in range(1800)]
        response = self.client.post('/api/step', json={'state': frame['state'], 'origin_at': frame['origin_at']})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreaterEqual(response.json()['active_count'], 1800)

    def test_invalid_inputs(self):
        for point in [dict(latitude=0, longitude=-117, intensity=.7), dict(latitude=53, longitude=-117, intensity=2)]:
            self.assertEqual(self.client.post('/api/seed', json={'ignitions': [point]}).status_code, 422)
        self.assertEqual(self.client.post('/api/firms', json=dict(west=-100, south=50, east=-130, north=60)).status_code, 422)
        initial = self.seed()
        initial['state']['active_cells'] *= 2
        self.assertEqual(self.client.post('/api/step', json={'state': initial['state'], 'origin_at': initial['origin_at']}).status_code, 422)

    def test_firms_aggregation_roundtrips_evidence_and_cache(self):
        calls = []
        def loader(_key, bounds, *, now):
            calls.append(bounds)
            now = datetime(2026, 9, 5, 12, tzinfo=timezone.utc)
            rows = [{"latitude": 53.02, "longitude": -117.31, "acquired_at": '2026-09-05T06:00:00Z',
                     "bright_ti4": 290., "provenance": {"product": "VIIRS_SNPP_NRT"}}]
            return aggregate_current_firms(rows, bounds, now=now)
        model = IncidentTransitionModel(SpreadEstimator(), feature_columns=RECURSIVE_MODEL_FEATURE_COLUMNS)
        with TestClient(create_app(model=model, terrain_provider=terrain, firms_loader=loader)) as client:
            bounds = dict(west=-179, south=24, east=-52, north=84)
            response = client.post('/api/firms', json=bounds)
            self.assertEqual(response.status_code, 200)
            frame = response.json()
            self.assertEqual(frame['points'][0]['source'], 'FIRMS observation')
            self.assertEqual(frame['state']['active_cells'][0]['bright_ti4_max'], 290.)
            self.assertEqual(client.post('/api/firms', json=bounds).json(), frame)
            self.assertEqual(len(calls), 1)
            self.assertEqual(client.post('/api/firms').json(), frame)
            self.assertEqual(len(calls), 1)
            step = client.post('/api/step', json={'state': frame['state'], 'origin_at': frame['origin_at']})
            self.assertEqual(step.status_code, 200, step.text)
            survived = next(c for c in step.json()['state']['active_cells'] if c['cell_id'] == frame['points'][0]['cell_id'])
            self.assertEqual(survived['bright_ti4_max'], 290.)
            self.assertEqual(survived['observation_age_hours'], 18.)

    def test_model_failure_serves_actionable_page(self):
        with patch('wildfire_data.web_app.load_pass_model', side_effect=FileNotFoundError('missing')):
            with self.assertLogs('wildfire_data.web_app', level='ERROR'):
                with TestClient(create_app()) as client:
                    self.assertEqual(client.get('/').status_code, 200)
                    self.assertFalse(client.get('/api/config').json()['model_ready'])
                    self.assertEqual(client.post('/api/seed', json={'ignitions': [dict(latitude=53, longitude=-117, intensity=.7)]}).status_code, 503)


if __name__ == '__main__':
    unittest.main()
