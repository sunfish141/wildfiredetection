"""Optional real-browser checks against a running local server.

Install Playwright and Chromium, start the app, then run this file. Provider
responses are mocked only for the FIRMS UI check; spread calls use the server.
"""

import asyncio
import json
from pathlib import Path
import sys

from playwright.async_api import async_playwright, expect


async def main(base_url="http://127.0.0.1:8000"):
    output = Path('artifacts/web-preview')
    output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.goto(base_url)
        await expect(page.locator('#place')).to_be_enabled(timeout=60000)
        await page.locator('#intensity').fill('85')
        await expect(page.locator('#intensity-value')).to_have_text('85%')
        await page.locator('#place').click()
        await page.locator('#map').click(position={"x": 510, "y": 340})
        await expect(page.locator('#active-count')).to_have_text('1')
        await page.locator('#place').click()
        # Click the actual canvas fire marker and inspect the placed intensity.
        await page.locator('#map').click(position={"x": 510, "y": 340})
        await expect(page.locator('#inspector')).to_be_visible()
        await expect(page.locator('#point-details')).to_contain_text('85%')
        await page.locator('#close-inspector').click()
        await page.locator('#step').click()
        await expect(page.locator('#elapsed')).to_have_text('+12 hours', timeout=60000)
        await page.locator('#show-candidates').check()
        await page.locator('#fit').click()
        await page.wait_for_timeout(400)
        await page.evaluate("Object.values(layers.active._layers)[0].fire('click')")
        await page.locator('.sidebar').evaluate('(sidebar) => sidebar.scrollTop = 0')
        await page.screenshot(path=str(output / 'desktop.png'), full_page=True)
        await page.locator('#close-inspector').click()

        # Revisit the origin, then test pause during a deliberately slow request.
        await page.locator('#timeline').fill('0')
        await expect(page.locator('#elapsed')).to_have_text('+0 hours')
        await page.locator('#reset').click()
        await page.locator('.coordinates summary').click()
        await page.locator('#coordinate-place').click()
        await expect(page.locator('#active-count')).to_have_text('1')
        request_started = asyncio.Event()
        async def delayed_step(route):
            request_started.set()
            await asyncio.sleep(1.5)
            try:
                await route.continue_()
            except Exception:
                pass  # Abort is the behavior under test.
        await page.route('**/api/step', delayed_step)
        await page.locator('#speed').select_option('1')
        await page.locator('#play').click()
        await asyncio.wait_for(request_started.wait(), timeout=10)
        await page.locator('#play').click()
        await expect(page.locator('#playback-state')).to_have_text('PAUSED')
        await page.wait_for_timeout(2200)
        await expect(page.locator('#elapsed')).to_have_text('+0 hours')
        await page.unroute('**/api/step', delayed_step)
        # Resume really advances; pausing again leaves a stable frame.
        await page.locator('#play').click()
        await expect(page.locator('#elapsed')).to_have_text('+12 hours', timeout=60000)
        await page.locator('#play').click()
        await page.wait_for_timeout(1200)
        await expect(page.locator('#elapsed')).to_have_text('+12 hours')

        # Actual inference continues past the former stop, including extinction.
        await page.evaluate('async () => { for (let i = 0; i < 11; i++) await advance(); }')
        await expect(page.locator('#elapsed')).to_have_text('+144 hours')
        await expect(page.locator('#play')).to_be_enabled()
        await expect(page.locator('#timeline')).to_have_attribute('max', '12')
        # Exercise the rolling buffer independently of model propagation.
        await page.evaluate('''() => {
          const base = current();
          for (let i = 13; i <= 150; i++) appendFrame({...base,
            state: {...base.state, step_index: i}, elapsed_hours: i * 12});
          draw();
        }''')
        assert await page.evaluate('history.length') == 128
        await expect(page.locator('#timeline')).to_have_attribute('min', '23')
        await page.locator('#timeline').fill('30')
        await expect(page.locator('#elapsed')).to_have_text('+360 hours')
        await page.locator('#step').click()
        await expect(page.locator('#elapsed')).to_have_text('+372 hours')

        # Feed a valid real server seed through the FIRMS UI response boundary.
        response = await page.request.post(base_url + '/api/seed', data={'ignitions': [
            {'latitude': 53.02, 'longitude': -117.31, 'intensity': .8}]})
        fixture = await response.json()
        fixture['metadata'] = {'eligible_detection_count': 3, 'recent_detections_excluded': 1,
                               'as_of': fixture['origin_at']}
        fixture['points'][0]['source'] = 'FIRMS observation'
        requested_bounds = []
        async def firms_result(route):
            requested_bounds.append(route.request.post_data_json)
            await route.fulfill(json=fixture)
        await page.route('**/api/firms', firms_result)
        await page.locator('#firms-tab').click()
        # The production config may have no credential in a clean checkout.
        await page.locator('#load-firms').evaluate('(button) => button.disabled = false')
        await page.locator('#load-firms').click()
        await expect(page.locator('#status')).to_contain_text('Loaded 3 observations')
        await expect(page.locator('#elapsed')).to_have_text('+0 hours')
        assert requested_bounds == [dict(west=-179, south=24, east=-52, north=84)], requested_bounds
        await page.unroute('**/api/firms', firms_result)

        # Large maps cluster only their display; the complete point set survives.
        await page.evaluate('''() => {
          const base = current().points[0];
          current().points = Array.from({length: 2000}, (_, i) => ({...base,
            cell_id: `display-fixture-${i}`, latitude: 53 + (i % 50) * .01,
            longitude: -117 + Math.floor(i / 50) * .01}));
          map.setView([53.25, -116.8], 8); draw();
        }''')
        assert await page.evaluate('current().points.length') == 2000
        assert 0 < await page.evaluate('layers.active.getLayers().length') < 2000

        await page.set_viewport_size({'width': 390, 'height': 844})
        await page.screenshot(path=str(output / 'mobile.png'), full_page=True)
        assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Horizontal overflow'
        assert not errors, errors
        (output / 'browser-verification.json').write_text(json.dumps({
            'status': 'passed', 'javascript_errors': errors,
            'checks': ['map placement', 'intensity inspection', 'real model step', 'timeline history',
                       'pause during request', 'resume', 'real inference past 96 hours',
                       'rolling history', 'full-region FIRMS request', 'display clustering', 'mobile layout'],
        }, indent=2) + '\n')
        await browser.close()
        print('Browser checks passed; screenshots in artifacts/web-preview/')


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000'))
