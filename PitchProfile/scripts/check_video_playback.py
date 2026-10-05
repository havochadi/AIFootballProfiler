"""Read-only browser checks for the live footage player, including failed loads.

Run with the website running. Does not change annotations or start analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--browser', default='chrome', choices=['chrome', 'msedge'])
    args = parser.parse_args()
    artifacts = Path(__file__).resolve().parents[1] / '.runtime'
    checks, errors = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.browser, headless=True)
        page = browser.new_page(viewport={'width': 1366, 'height': 768})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(args.url, wait_until='domcontentloaded')
        page.wait_for_function('typeof reviewDesk!=="undefined" && reviewDesk.ready', timeout=45000)
        page.locator('#mp-watch').click()
        page.wait_for_function('''() => {
            const v=document.querySelector('#mp-video');
            return !v.paused && v.currentTime>1 && v.getVideoPlaybackQuality().totalVideoFrames>0;
        }''')
        assert page.locator('#mp-video').evaluate('(v)=>v.getBoundingClientRect().top>=0 && v.getBoundingClientRect().bottom<innerHeight')
        expect(page.locator('#mp-video-note')).to_contain_text('Playing footage')
        checks.append('Watch footage starts decoded video inside the viewport')

        page.locator('[data-evidence-view="map"]').click()
        expect(page.locator('#mp-video')).to_be_hidden()
        page.locator('#mp-watch').click()
        expect(page.locator('#mp-video')).to_be_visible()
        checks.append('Watch footage returns from event maps to the video')

        # Regression: Chrome stalled on an 8.56s seek inside the original MKV's
        # initial buffered range. Confirm decoded playback, not just paused=False.
        page.evaluate('seekVideo(8.56, true)')
        page.wait_for_function("() => {\n            const v=document.querySelector('#mp-video');\n            return v.readyState>=2 && !v.seeking && v.currentTime>9.1;\n        }", timeout=15000)
        checks.append('early buffered seek at 8.56 seconds resumes decoded playback')


        # Delay media metadata, then seek before it arrives. The target must survive loading.
        def slow_media(route):
            page.wait_for_timeout(250)
            route.continue_()
        page.route('**/media/**', slow_media)
        page.evaluate('''() => {
            const v=document.querySelector('#mp-video');v.pause();v.load();seekVideo(120);
        }''')
        page.wait_for_function('''() => {
            const v=document.querySelector('#mp-video');
            return v.readyState>=2 && !v.seeking && Math.abs(v.currentTime-120)<.2;
        }''', timeout=30000)
        page.unroute('**/media/**', slow_media)
        checks.append('Seeking before metadata loads preserves the selected moment')

        page.locator('#mp-video-tools').evaluate('(d)=>d.open=true')
        # The browser alone sees this artificial 404. Files and the server remain unchanged.
        def missing_media(route):
            route.fulfill(status=404, content_type='text/plain', body='Simulated missing footage')
        page.route('**/media/**', missing_media)
        page.locator('#mp-video-reload').click()
        page.wait_for_function('document.querySelector("#mp-video").error !== null')
        expect(page.locator('#mp-video-note')).to_contain_text('Try Reload footage or Open video')
        assert page.locator('#mp-video-open').get_attribute('href') == page.evaluate('document.querySelector("#mp-video").dataset.src')
        page.unroute('**/media/**', missing_media)
        page.locator('#mp-video-reload').click()
        page.wait_for_function('''() => {
            const v=document.querySelector('#mp-video');
            return !v.error && v.readyState>=2 && !v.seeking && Math.abs(v.currentTime-120)<.2;
        }''', timeout=30000)
        expect(page.locator('#mp-video-note')).to_contain_text('Footage ready')
        checks.append('Failed media loads have an inline explanation and reload restores playback at the same time')

        for width in (1366, 390):
            page.set_viewport_size({'width': width, 'height': 844})
            page.locator('#mp-watch').click()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            expect(page.locator('#mp-video')).to_be_in_viewport()
        checks.append('Footage controls work on desktop and mobile without horizontal overflow')
        page.locator('#mp-video').evaluate('(v)=>v.pause()')
        page.screenshot(path=str(artifacts / f'video-playback-{args.browser}.png'), full_page=True)
        assert not errors, errors
        browser.close()
    report = {'passed': True, 'browser': args.browser, 'checks': checks, 'page_errors': errors}
    (artifacts / f'video-playback-{args.browser}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
