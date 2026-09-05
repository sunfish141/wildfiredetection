from datetime import datetime, timezone
import unittest
from unittest.mock import MagicMock

import requests

from wildfire_data.live_firms import aggregate_current_firms, fetch_current_firms, LiveFirmsError
from wildfire_data.training_grid import GridCell, cell_from_wgs84


NOW = datetime(2026, 9, 5, 12, tzinfo=timezone.utc)
BOUNDS = (-118, 52, -117, 54)


def observation(hour, *, product='VIIRS_SNPP_NRT', brightness=320.):
    return {'latitude': 53.02, 'longitude': -117.31, 'acquired_at': f'2026-09-05T{hour:02}:00:00Z',
            'bright_ti4': brightness, 'provenance': {'product': product}}


class LiveFirmsTests(unittest.TestCase):
    def test_deduplicated_three_to_twenty_four_hour_observation_window(self):
        rows = [observation(6), observation(6), observation(8, product='VIIRS_NOAA20_NRT', brightness=300),
                observation(10), observation(13)]
        old = observation(6); old['acquired_at'] = '2026-09-04T06:00:00Z'; rows.append(old)
        outside = observation(5); outside['longitude'] = -120; rows.append(outside)
        state, metadata = aggregate_current_firms(rows, BOUNDS, now=NOW)
        self.assertEqual(len(state.active_cells), 1)
        cell = state.active_cells[0]
        self.assertEqual(cell.detection_count, 2)
        self.assertEqual(cell.platform_count, 2)
        self.assertEqual(cell.bright_ti4_mean, 310.)
        self.assertEqual(cell.observation_age_hours, 4.)
        self.assertEqual(metadata['recent_detections_excluded'], 1)

    def test_credentials_and_upstream_errors_are_not_exposed(self):
        with self.assertRaisesRegex(LiveFirmsError, 'Set NASA_FIRMS_API_KEY'):
            fetch_current_firms('', BOUNDS)
        session = MagicMock()
        session.get.side_effect = requests.ConnectionError('URL contains SECRET')
        with self.assertRaises(LiveFirmsError) as caught:
            fetch_current_firms('SECRET', BOUNDS, session=session)
        self.assertNotIn('SECRET', str(caught.exception))

    def test_all_three_feeds_required_and_empty_csv_is_valid(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [b'latitude,longitude,acq_date,acq_time,bright_ti4']
        session = MagicMock()
        session.get.return_value = response
        state, metadata = fetch_current_firms('SECRET', BOUNDS, now=NOW, session=session)
        self.assertEqual(session.get.call_count, 3)
        self.assertFalse(state.active_cells)
        self.assertEqual(metadata['eligible_detection_count'], 0)
        response.iter_lines.return_value = [b'Invalid MAP_KEY']
        with self.assertRaisesRegex(LiveFirmsError, 'unavailable or invalid'):
            fetch_current_firms('SECRET', BOUNDS, now=NOW, session=session)

    def test_full_region_stream_exceeds_previous_cell_cap_without_dropping_cells(self):
        cell = cell_from_wgs84(latitude=53, longitude=-117)
        rows = [b'latitude,longitude,acq_date,acq_time,bright_ti4']
        for i in range(2000):
            lat, lon = GridCell(cell.x_index + i % 100, cell.y_index + i // 100).center_wgs84
            rows.append(f'{lat},{lon},2026-09-05,0600,320'.encode())
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = rows
        session = MagicMock()
        session.get.return_value = response
        state, metadata = fetch_current_firms('SECRET', now=NOW, session=session)
        self.assertEqual(len(state.active_cells), 2000)
        self.assertEqual(metadata['eligible_detection_count'], 6000)
        self.assertEqual(state.active_cells[0].platform_count, 3)
        self.assertEqual(state.active_cells[0].detection_count, 3)
        self.assertIn('/-179.0,24.0,-52.0,84.0/2', session.get.call_args.args[0])
        response.iter_content.assert_not_called()


if __name__ == '__main__':
    unittest.main()
