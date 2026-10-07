r"""Drive the real web interface in a browser against the real data folder, read-only.

Checks that the review desk loads a player, that the tabs and menus show only what exists, that
evidence views switch, that the match library opens a half, and that nothing throws, returns an
error or tries to change data (any POST, PUT or DELETE fails the run).

Usage (from PitchProfile/): .\.venv\Scripts\python.exe scripts\check_ui_smoke.py
Needs Playwright and an installed Chrome or Edge. Screenshots and the report go to .runtime/ui-smoke-*.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    artifacts = ROOT / '.runtime' / ('ui-smoke-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    artifacts.mkdir(parents=True)
    report = {'passed': False, 'checks': [], 'page_errors': [], 'console_errors': [], 'failed_requests': [], 'writes': []}
    process = None
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        with (artifacts / 'server.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, 'run.py', '--port', str(port)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            for _ in range(150):
                if process.poll() is not None:
                    raise RuntimeError('Test server failed; see server.log')
                try:
                    with urlopen(url + '/api/status', timeout=1):
                        break
                except OSError:
                    time.sleep(.2)
            with sync_playwright() as pw:
                try:
                    browser = pw.chromium.launch(channel='chrome', headless=True)
                except Exception:
                    browser = pw.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(base_url=url, viewport={'width': 1440, 'height': 1000})
                page.set_default_timeout(60000)
                page.on('pageerror', lambda e: report['page_errors'].append(str(e)))
                page.on('console', lambda m: report['console_errors'].append(m.text) if m.type == 'error' and not m.text.startswith('Failed to load resource') else None)   # failed loads are reported by the response check
                page.on('response', lambda r: report['failed_requests'].append(f'{r.status} {r.url}') if r.status >= 400 and 'favicon' not in r.url else None)
                page.on('request', lambda r: report['writes'].append(f'{r.method} {r.url}') if r.method in ('POST', 'PUT', 'DELETE', 'PATCH') else None)
                try:
                    run_checks(page, report, artifacts)
                    assert not report['page_errors'], report['page_errors']
                    assert not report['console_errors'], report['console_errors']
                    assert not report['failed_requests'], report['failed_requests']
                    assert not report['writes'], report['writes']
                    report['passed'] = True
                except Exception:
                    page.screenshot(path=str(artifacts / 'failure.png'), full_page=True)
                    raise
                finally:
                    browser.close()
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    finally:
        (artifacts / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Report: ' + str(artifacts / 'report.json'), flush=True)
        if process and process.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, timeout=15)
            else:
                process.terminate()
    return 0 if report['passed'] else 1


def run_checks(page, report, artifacts):
    def passed(name):
        report['checks'].append(name)
        print('PASS ' + name, flush=True)

    def ready(pid=None):
        page.wait_for_function('(pid) => reviewDesk.ready && (!pid || state.pid === pid) && !document.querySelector("#label-workbench").inert', arg=pid)

    page.goto('/', wait_until='domcontentloaded')
    expect(page.locator('nav button[data-tab]')).to_have_text(['Review players', 'Matches'])
    expect(page.locator('nav button[aria-current="page"]')).to_have_text('Review players')
    page.locator('.tools-menu summary').click()
    expect(page.locator('.tools-menu button')).to_have_text(['Movement & tactics', 'Source profile & corrections', 'Upload a short clip'])
    page.locator('.tools-menu summary').click()
    passed('two main tabs and a tools menu with no rating, interval or model entries')

    ready()
    first = page.evaluate('state.pid')
    expect(page.locator('#mp-name')).not_to_be_empty()
    assert page.locator('#players button').count() > 1
    for gone in ('#mp-label', '#mp-roles', '#mp-bookmarks', '#queue-status', '#queue-progress', '#tab-learning', '#tab-intervals', '#tab-catalogue', '#tab-experiments'):
        assert page.locator(gone).count() == 0, gone
    page.screenshot(path=str(artifacts / 'review-desk.png'), full_page=True)
    passed('a player loads with footage controls and no rating form, bookmarks or status filter')

    page.locator('#mp-timeline button').first.wait_for()
    page.locator('#mp-forward').click()
    page.locator('[data-evidence-view="map"]').click()
    page.locator('#event-map-content').wait_for()
    page.wait_for_function('document.querySelector("#event-map-note").textContent !== "Loading pitch events…"')
    page.screenshot(path=str(artifacts / 'stats-and-maps.png'), full_page=True)
    page.locator('[data-evidence-view="video"]').click()
    passed('footage controls and the stats-and-maps view both work')

    page.locator('#mp-next').click()
    page.wait_for_function('(pid) => state.pid !== pid', arg=first)
    ready()
    passed('next player loads another player')

    page.locator('nav button[data-tab="data"]').click()
    page.locator('#learn-library table').wait_for()
    assert page.locator('#learn-library [data-open]').count() > 0
    page.locator('#learn-library [data-open]').first.click()
    page.wait_for_function('state.tab === "match"')
    ready()
    passed('the match library opens a half in the review desk')

    for tab in ('analysis', 'overview'):
        page.evaluate(f'changeTab("{tab}")')
        page.wait_for_function(f'!document.querySelector("#tab-{tab}").hidden')
    expect(page.locator('#tab-overview')).to_be_visible()
    assert 'Archetype' not in page.locator('#tab-overview').inner_text()
    page.screenshot(path=str(artifacts / 'overview.png'), full_page=True)
    passed('movement and source-profile screens open without errors or archetype wording')


if __name__ == '__main__':
    sys.exit(main())
