"""Exercise the review desk on copied match evidence; never write real labels.

Requires Playwright and installed Chrome. No detector/training jobs are started.
Artifacts and isolated annotation data are kept in .runtime/labelling-*.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_profiler import storage as S
from football_profiler import soccernet as SN
from football_profiler import match_context as MC


def run_checks(page, report, artifacts):
    def passed(name):
        report['checks'].append(name)
        print('PASS ' + name, flush=True)

    def ready(pid=None):
        page.wait_for_function('(pid) => reviewDesk.ready && (!pid || state.pid === pid) && !document.querySelector("#label-workbench").inert', arg=pid)

    ready()
    expect(page.locator('nav button')).to_have_count(3)
    expect(page.locator('nav button[aria-current="page"]')).to_have_text('Label players')
    page.wait_for_function('() => document.querySelector("#mp-video").readyState >= 2')
    first = page.evaluate('state.pid')
    dataset = page.evaluate('state.manifest.id')
    report.update(dataset=dataset, player=first)
    assert page.locator('#mp-timeline button').count() > 0
    page.screenshot(path=str(artifacts / 'labelling-desktop.png'), full_page=True)
    passed('three primary screens; real match, queue, video and appearance timeline load')

    page.locator('#mp-group').select_option('centre_forward')
    expect(page.locator('#mp-roles .role-card')).to_have_count(8)
    role = page.locator('#mp-roles .role-card').first
    role_id = role.get_attribute('data-role')
    expect(role.locator('details')).to_have_attribute('open', '')
    expect(role.locator('.role-examples li')).to_have_count(2)
    expect(role.locator('details p').first).to_be_visible()
    role.locator('button[data-value="50"]').click()
    expect(role.locator('output')).to_have_text('50%')
    page.locator('#mp-labeler').fill('Disposable UI reviewer')
    page.locator('#mp-notes').fill('Artificial review to verify the interface; not a real archetype label.')
    page.locator('#mp-bookmark').click()
    page.locator('#mp-bookmarks textarea').fill('Artificial bookmark note for the UI test')
    bookmark_time = page.evaluate('reviewDesk.evidence[0].time_s')
    expect(page.locator('#mp-draft-status')).to_contain_text('Draft saved')
    page.locator('#mp-next').click()
    page.wait_for_function('(pid) => reviewDesk.ready && state.pid !== pid', arg=first)
    second = page.evaluate('state.pid')
    page.evaluate('(pid) => selectPlayer(pid)', first)
    ready(first)
    expect(page.locator('#mp-draft-status')).to_contain_text('Restored')
    expect(page.locator('#mp-bookmarks textarea')).to_have_value('Artificial bookmark note for the UI test')
    page.reload(wait_until='domcontentloaded')
    ready()
    if page.evaluate('state.pid') != first:
        page.evaluate('(pid) => selectPlayer(pid)', first)
        ready(first)
    expect(page.locator(f'[data-role="{role_id}"] output')).to_have_text('50%')
    assert 'Artificial review' in page.locator('#mp-notes').input_value()
    passed('role definitions, quick ratings, bookmarks and drafts survive navigation and reload')

    with page.expect_response(lambda r: r.url.endswith('/archetype') and r.request.method == 'POST') as saved:
        page.locator('#mp-save-next').click()
    result = saved.value.json()
    assert result['labels'][role_id] == 50
    assert sum(v is None for v in result['labels'].values()) == 7
    assert result['evidence'] == [{'time_s': bookmark_time, 'note': 'Artificial bookmark note for the UI test'}]
    page.wait_for_function('(pid) => reviewDesk.ready && state.pid !== pid', arg=first)
    expect(page.locator('#queue-progress')).to_contain_text('1 /')
    page.locator('#queue-status').select_option('labelled')
    expect(page.locator('#players button')).to_have_count(1)
    page.locator('#players button').click()
    ready(first)
    expect(page.locator('#mp-notes')).to_have_value('Artificial review to verify the interface; not a real archetype label.')
    expect(page.locator('#mp-estimate-panel')).not_to_have_attribute('open', '')
    passed('save & next persists ratings, unknowns and bookmarks; labelled filter and counts update')

    page.locator('#queue-status').select_option('all')
    page.locator(f'[data-role="{role_id}"] button[data-value="0"]').click()
    page.locator('#mp-save').click()
    expect(page.locator('#mp-draft-status')).to_have_text('Saved assessment')
    stored = page.request.get(f'/api/datasets/{dataset}/players/{first}/archetype').json()['label']
    assert stored['labels'][role_id] == 0
    exported = page.request.get('/api/archetypes/export').json()
    assert exported['labels'][0]['evidence'][0]['time_s'] == bookmark_time
    passed('explicit zero remains distinct from unknown; export includes evidence')

    page.locator('#mp-notes').fill('Draft must survive a failed save')
    route = '**/players/*/archetype'
    def fail_save(request):
        if request.request.method == 'POST':
            request.fulfill(status=503, json={'detail': 'Simulated save failure'})
        else:
            request.continue_()
    page.route(route, fail_save)
    page.locator('#mp-save').click()
    expect(page.locator('#mp-save-state')).to_contain_text('Save failed')
    assert page.evaluate('readDraft(reviewDesk.key).payload.notes') == 'Draft must survive a failed save'
    page.unroute(route, fail_save)
    page.evaluate('(ids) => Promise.all(ids.map(pid => selectPlayer(pid)))', [first, second])
    ready(second)
    assert page.locator('#mp-name').inner_text() == page.evaluate('state.player.name')
    expect(page.locator('#mp-bookmarks textarea')).to_have_count(0)
    page.evaluate('(pid) => selectPlayer(pid)', first)
    ready(first)
    expect(page.locator('#mp-notes')).to_have_value('Draft must survive a failed save')
    passed('failed save retains draft; rapid selection cannot attach evidence to the wrong player')

    # A slower POST must not erase edits typed while the request is in flight.
    page.evaluate('''() => { window.testOriginalFetch = window.fetch; window.fetch = async (...args) => {
      const result = await window.testOriginalFetch(...args);
      if (String(args[0]).endsWith('/archetype') && args[1]?.method === 'POST') await new Promise(r=>setTimeout(r,1000));
      return result;
    }; }''')
    page.locator('#mp-save').click()
    page.locator('#mp-notes').fill('Newer edit while save was pending')
    expect(page.locator('#notice')).to_contain_text('newer edits')
    assert page.evaluate('readDraft(reviewDesk.key).payload.notes') == 'Newer edit while save was pending'
    page.evaluate('window.fetch = window.testOriginalFetch')
    passed('edits made during a save remain a separate recoverable draft')

    page.locator('#mp-appearance').click()
    page.wait_for_function('!document.querySelector("#mp-video").paused')
    page.locator('#mp-video').evaluate('(v)=>v.pause()')
    page.locator('#mp-speed').select_option('0.5')
    assert page.evaluate('document.querySelector("#mp-video").playbackRate') == .5
    page.locator('#mp-bookmark').click()
    expect(page.locator('#mp-bookmarks textarea')).to_have_count(2)
    page.locator('#mp-bookmarks textarea').last.fill('Typing b and j must not trigger playback shortcuts')
    page.locator('#mp-bookmarks textarea').last.press('b')
    expect(page.locator('#mp-bookmarks textarea')).to_have_count(2)
    page.locator('#mp-video').focus()
    page.keyboard.press('b')
    expect(page.locator('#mp-bookmarks textarea')).to_have_count(3)
    page.screenshot(path=str(artifacts / 'labelling-with-evidence.png'), full_page=True)
    passed('appearance playback, speed controls and keyboard bookmarks; typing does not trigger shortcuts')

    page.locator('nav [data-tab="data"]').click()
    expect(page.locator('#learn-library')).to_contain_text('Label players')
    page.locator('#match-search').fill('This match does not exist')
    expect(page.locator('#learn-library')).to_contain_text('No matching')
    page.locator('nav [data-tab="learning"]').click()
    expect(page.locator('#learn-summary')).to_contain_text('1 labelled')
    page.locator('.tools-menu summary').click()
    page.locator('.tools-menu [data-goto="catalogue"]').click()
    expect(page.locator('#catalogue-roles .catalogue-role')).to_have_count(42)
    page.locator('.tools-menu summary').click()
    page.locator('.tools-menu [data-goto="intervals"]').click()
    expect(page.locator('#case-create-button')).to_be_enabled()
    passed('searchable match library, model summary and all 42 roles; independent reviews remain accessible')

    page.locator('nav [data-tab="match"]').click()
    ready(first)
    for width in (1440, 1024, 768, 390):
        page.set_viewport_size({'width': width, 'height': 900})
        overflow = page.evaluate('document.documentElement.scrollWidth > innerWidth + 1')
        assert not overflow, f'Horizontal page overflow at {width}px'
    page.screenshot(path=str(artifacts / 'labelling-mobile.png'), full_page=True)
    passed('responsive layout at 1440, 1024, 768 and 390 pixels without page overflow')


def run_usability_checks(page, report, artifacts):
    page.set_viewport_size({'width': 1366, 'height': 768})
    page.locator('nav [data-tab="match"]').click()
    page.wait_for_function('reviewDesk.ready')
    first = page.evaluate('state.pid')
    # All definitions and examples appear within the same card, including the rare roles.
    seen = set()
    groups = page.evaluate('intervalState.catalogue.position_groups.map(g=>g.id)')
    for group in groups:
        page.locator('#mp-group').select_option(group)
        cards = page.locator('#mp-roles .role-card')
        for i in range(cards.count()):
            card = cards.nth(i)
            seen.add(card.get_attribute('data-role'))
            if not card.locator('details').evaluate('(d)=>d.open'):
                card.locator('summary').click()
            expect(page.locator('#mp-roles details[open]')).to_have_count(1)
            expect(card.locator('.role-examples li')).to_have_count(2)
            expect(card.locator('.role-examples')).to_be_visible()
            assert card.locator('.role-examples a').first.get_attribute('href').startswith('https://')
    assert len(seen) == 42
    report['checks'].append('all 42 roles have two examples beneath the definition; one role expands at a time')
    print('PASS ' + report['checks'][-1], flush=True)

    page.locator('#mp-group').select_option('centre_forward')
    expect(page.locator('#mp-play, #mp-events')).to_have_count(0)
    expect(page.locator('#identify-dialog #mp-crops')).to_have_count(1)
    expect(page.locator('#identify-dialog #mp-merge')).to_have_count(0)
    assert page.locator('#mp-label').evaluate('(e)=>getComputedStyle(e).overflowY') == 'visible'
    assert page.locator('#media-container video').count() == 0
    assert page.locator('#tab-match #match-players').count() == 0
    page.locator('#mp-watch').click()
    page.wait_for_function('!document.querySelector("#mp-video").paused')
    page.locator('nav [data-tab="data"]').click()
    assert page.locator('#mp-video').evaluate('(v)=>v.paused')
    page.locator('nav [data-tab="match"]').click()
    page.wait_for_function('reviewDesk.ready')
    report['checks'].append('duplicate event/playback/identity panels removed; hidden videos pause; ratings have no nested scroll')
    print('PASS ' + report['checks'][-1], flush=True)

    page.locator('#mp-notes').fill('Preserve this draft while evidence loading fails')
    def unavailable(route):
        route.fulfill(status=503, content_type='application/json', body='{"detail":"Artificial evidence outage"}')
    page.route('**/match/crops/*', unavailable)
    page.locator('#mp-next').click()
    expect(page.locator('#mp-retry')).to_be_visible()
    assert page.locator('#label-workbench').evaluate('(el)=>el.inert')
    page.unroute('**/match/crops/*', unavailable)
    page.locator('#mp-retry').click()
    page.wait_for_function('reviewDesk.ready')
    expect(page.locator('#mp-retry')).to_be_hidden()
    page.evaluate('(pid)=>selectPlayer(pid)', first)
    page.wait_for_function('(pid)=>reviewDesk.ready&&state.pid===pid', arg=first)
    expect(page.locator('#mp-notes')).to_have_value('Preserve this draft while evidence loading fails')
    page.locator('#mp-watch').click()
    page.wait_for_function('document.querySelector("#mp-video").readyState>=2&&!document.querySelector("#mp-video").seeking')
    page.screenshot(path=str(artifacts / 'simplified-review-with-examples.png'), full_page=True)
    report['checks'].append('failed evidence loads offer an accessible retry; recovery preserves the previous player draft')
    print('PASS ' + report['checks'][-1], flush=True)


def run_context_checks(page, report, artifacts):
    def passed(name):
        report['checks'].append(name); print('PASS ' + name, flush=True)

    page.set_viewport_size({'width': 1440, 'height': 1000})
    page.locator('nav [data-tab="match"]').click()
    identifier = 'sn-20160207-chelsea-manchester-united-h1'
    page.locator('#dataset-select').select_option(identifier)
    page.wait_for_function('(id)=>state.manifest.id===id && reviewDesk.ready && currentMatch()?.match_context?.context', arg=identifier)
    expect(page.locator('#fixture-banner')).to_contain_text('Stamford Bridge')
    assert page.evaluate('currentMatch().match_context.context.teams.map(t=>t.score)') == ['1', '1']
    # The direct picker resolves a person without first naming the kit clusters.
    page.evaluate('selectPlayer("A-10")')
    page.wait_for_function('reviewDesk.ready && state.pid === "A-10"')
    assert not page.evaluate('currentMatch().match_context.binding.teams')
    page.locator('#mp-correct-player').click()
    expect(page.locator('#identify-confirm')).to_be_disabled()
    page.locator('[data-identity-team="360"]').click()
    page.locator('#identify-search').fill('Rooney')
    expect(page.locator('#identify-candidates [data-athlete]')).to_have_count(1)
    page.locator('[data-athlete="21046"]').click()
    expect(page.locator('#identify-selection')).to_contain_text('#10 · Wayne Rooney')
    page.wait_for_function("document.querySelector('[data-athlete] img').naturalWidth>0")
    page.screenshot(path=str(artifacts / 'guided-player-picker.png'))
    page.locator('#identify-confirm').click()
    expect(page.locator('#identify-dialog')).not_to_be_visible()
    expect(page.locator('#mp-name')).to_have_text('Wayne Rooney')
    assert not page.evaluate('currentMatch().match_context.binding.teams')
    page.locator('#mp-correct-player').click()
    page.locator('#identify-search').fill('999')
    expect(page.locator('#identify-candidates [data-athlete]')).to_have_count(0)
    expect(page.locator('#identify-list-note')).to_contain_text('No match')
    page.locator('#identify-manual').click()
    expect(page.locator('#identify-quick')).to_be_visible()
    page.locator('#identify-cancel').click()
    expect(page.locator('#mp-name')).to_have_text('Wayne Rooney')
    page.locator('#mp-correct-player').click()
    page.locator('#correction-reset').click()
    expect(page.locator('#identify-dialog')).not_to_be_visible()
    passed('team and searchable photo cards identify a player without kit setup; empty search, cancel and undo work')
    page.locator('#match-context-panel > summary').click()
    expect(page.locator('#official-lineups .lineup-row')).to_have_count(36)
    page.locator('#binding-a').select_option('360')  # Disposable fixture only: A assigned to United.
    page.locator('#binding-b').select_option('363')
    page.locator('#binding-reviewer').fill('Disposable identity reviewer')
    page.locator('#kit-binding button').click()
    page.wait_for_function('reviewDesk.ready && currentMatch().match_context.binding.teams?.A === "360"')
    page.evaluate('selectPlayer("A-10")')
    page.wait_for_function('reviewDesk.ready && state.pid === "A-10"')
    expect(page.locator('#mp-name')).to_have_text('Wayne Rooney')
    expect(page.locator('#mp-portrait img')).to_have_attribute('src', __import__('re').compile(r'^/api/match-assets/[a-f0-9]{32}$'))
    page.wait_for_function('document.querySelector("#mp-portrait img").naturalWidth > 0')
    page.locator('#mp-portrait img').evaluate('(image)=>image.decode()')
    page.locator('#match-context-panel').evaluate('(el)=>el.open=false')
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path=str(artifacts / 'rooney-portrait-and-score.png'))
    expect(page.locator('#mp-identity-status')).to_contain_text('verify')
    page.locator('#mp-correct-player').click()
    page.locator('#identify-search').fill('35')
    expect(page.locator('#identify-candidates [data-athlete]')).to_have_count(1)
    page.locator('#identify-candidates [data-athlete="169136"]').click()
    page.locator('#identify-confirm').click()
    page.wait_for_function('reviewDesk.ready && currentMatch().players.find(p=>p.identity===state.pid)?.identity_status === "confirmed"')
    expect(page.locator('#mp-name')).to_have_text('Jesse Lingard')
    expect(page.locator('#mp-sub')).to_contain_text('model read #10')
    assert page.evaluate('state.pid') == 'A-10'
    assert page.evaluate('state.player.manifest_player.jersey') == 10
    passed('historical 1–1 score, 36 match-day squad members, locally rendered Rooney #10 portrait and reversible identity correction')

    # Select a player with actual detected passes and compare screen coordinates to source data.
    pid = page.evaluate('currentMatch().players.find(p=>p.on_ball?.passes>0).identity')
    page.evaluate('(pid)=>selectPlayer(pid)', pid)
    page.wait_for_function('(pid)=>reviewDesk.ready && state.pid===pid', arg=pid)
    page.locator('[data-evidence-view="map"]').click()
    page.wait_for_function('eventMap.data && document.querySelectorAll("#event-pitch [data-map-event]").length>0')
    event = page.evaluate('eventMap.data.events.find(e=>e.type==="pass" && e.x!==null && e.attack_sign)')
    actual = page.locator(f'#event-pitch [data-map-event="{event["id"]}"] circle').get_attribute('cx')
    expected = 40 + 10 * (105 - event['x'] if event['attack_sign'] < 0 else event['x'])
    assert abs(float(actual) - expected) < .001
    page.locator(f'#event-pitch [data-map-event="{event["id"]}"]').click()
    expect(page.locator('#event-inspector')).to_be_visible()
    page.locator('#selected-event-outcome').select_option('intercepted')
    page.locator('#selected-event-reviewer').fill('Disposable event reviewer')
    page.locator('#selected-event-note').fill('Artificial outcome for UI verification only')
    page.locator('#event-outcome-form button').click()
    expect(page.locator('#event-outcome-status')).to_have_text('Outcome saved')
    page.locator('#event-map-legend [data-outcome="intercepted"]').click()
    assert page.evaluate('filteredMapEvents().every(e=>e.outcome==="intercepted")')
    page.locator('#event-map-mode').select_option('heatmap')
    expect(page.locator('#event-pitch [data-map-event]')).not_to_have_count(0)
    page.screenshot(path=str(artifacts / 'event-heatmap-desktop.png'), full_page=True)
    passed('event origins and pass routes, correct attack-direction transform, outcome filters, heatmap and saved reviews')

    shot_pid = page.evaluate('currentMatch().players.find(p=>p.on_ball?.shots>0).identity')
    page.evaluate('(pid)=>selectPlayer(pid)', shot_pid)
    page.wait_for_function('(pid)=>reviewDesk.ready && state.pid===pid && eventMap.key===reviewDesk.key && eventMap.data', arg=shot_pid)
    page.locator('#event-kind-buttons [data-kind="shot"]').click()
    assert page.evaluate('filteredMapEvents().every(e=>e.outcome === "unknown")')
    page.locator('#map-event-list [data-map-event]').first.click()
    page.locator('#selected-event-outcome').select_option('blocked')
    page.locator('#selected-event-reviewer').fill('Disposable event reviewer')
    page.locator('#event-outcome-form button').click()
    expect(page.locator('#event-outcome-status')).to_have_text('Outcome saved')
    page.locator('#selected-event-watch').click()
    expect(page.locator('#video-evidence')).to_be_visible()
    page.wait_for_function('!document.querySelector("#mp-video").paused')
    page.locator('#mp-video').evaluate('(v)=>v.pause()')
    page.locator('[data-evidence-view="map"]').click()
    page.locator('#event-map-mode').select_option('events')
    page.locator('#match-context-panel').evaluate('(el)=>el.open=false')
    page.evaluate('window.scrollTo(0,0)')
    page.screenshot(path=str(artifacts / 'scouting-room-desktop.png'))
    for width in (1440, 1024, 768, 390):
        page.set_viewport_size({'width': width, 'height': 900})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), width
    page.screenshot(path=str(artifacts / 'event-map-mobile.png'), full_page=True)
    passed('shot outcomes stay unknown until reviewed; blocked shot can be saved and replayed; redesigned event maps fit mobile')


def run_correction_checks(page, report, artifacts):
    page.set_viewport_size({'width': 1440, 'height': 1000})
    page.locator('[data-evidence-view="map"]').click()
    page.wait_for_function('eventMap.data && eventMap.key===reviewDesk.key')
    page.locator('#event-kind-buttons [data-kind="movement"]').click()
    expect(page.locator('#movement-evidence')).to_be_visible()
    expect(page.locator('#event-map-content')).to_be_hidden()
    expect(page.locator('#mp-heatmap')).to_have_count(1)
    expect(page.locator('#mp-stats')).to_contain_text('Physical')
    expect(page.locator('#mp-stats')).not_to_contain_text('On the ball')
    page.locator('#event-kind-buttons [data-kind="pass"]').click()
    expect(page.locator('#stat-summary')).to_contain_text('Completed')
    expect(page.locator('#movement-evidence')).to_be_hidden()
    page.locator('#event-kind-buttons [data-kind="shot"]').click()
    expect(page.locator('#stat-summary')).to_contain_text('Mean distance')
    page.locator('#event-kind-buttons [data-kind="movement"]').click()
    expect(page.locator('#event-inspector')).to_be_hidden()
    page.screenshot(path=str(artifacts / 'integrated-movement-desktop.png'), full_page=True)
    page.locator('#event-kind-buttons [data-kind="profile"]').click()
    expect(page.locator('#profile-evidence')).to_be_visible()
    expect(page.locator('#event-map-content')).to_be_hidden()
    profile = page.evaluate('currentMatch().players.find(p=>p.identity===state.pid)?.profile')
    if profile:
        expect(page.locator('#mp-profile')).to_contain_text('Percentile against')
        expect(page.locator('#mp-profile .profile-row')).not_to_have_count(0)
    page.screenshot(path=str(artifacts / 'style-profile-desktop.png'), full_page=True)
    report['checks'].append('movement, physical statistics, style profile percentiles, passes and shots share one '
                            'evidence panel without duplicate statistics')
    print('PASS ' + report['checks'][-1], flush=True)

    pid = page.evaluate('currentMatch().players.find(p=>p.role==="goalkeeper").identity')
    page.evaluate('(pid)=>selectPlayer(pid)', pid)
    page.wait_for_function('(pid)=>reviewDesk.ready&&state.pid===pid', arg=pid)
    page.locator('[data-evidence-view="video"]').click()
    page.locator('#mp-notes').fill('Keep this draft through identity correction')
    page.locator('#mp-not-goalkeeper').click()
    expect(page.locator('#correction-role')).to_have_value('player')
    expect(page.locator('#correction-shirt')).to_have_value('')
    page.locator('#identify-reviewer-details').evaluate('(d)=>d.open=true')
    page.locator('#correction-reviewer').fill('Disposable correction reviewer')
    page.locator('#identify-extra').evaluate('(d)=>d.open=true')
    page.locator('#correction-name').fill('Corrected test midfielder')
    page.locator('#correction-shirt').fill('10')
    before = page.locator('#mp-video').evaluate('(v)=>v.currentTime')
    pattern = '**/track-correction'
    page.route(pattern, lambda r: r.fulfill(status=503, content_type='application/json', body='{"detail":"Artificial save failure"}'))
    page.locator('#identify-confirm').click()
    expect(page.locator('#correction-status')).to_contain_text('Could not save')
    expect(page.locator('#correction-name')).to_have_value('Corrected test midfielder')
    page.unroute(pattern)
    page.locator('#identify-confirm').click()
    expect(page.locator('#identify-dialog')).not_to_be_visible()
    expect(page.locator('#identity-save-status')).to_contain_text('Correction saved')
    expect(page.locator('#mp-name')).to_have_text('Corrected test midfielder')
    expect(page.locator('#mp-not-goalkeeper')).to_be_hidden()
    expect(page.locator('#mp-notes')).to_have_value('Keep this draft through identity correction')
    assert abs(page.locator('#mp-video').evaluate('(v)=>v.currentTime') - before) < .1
    expect(page.locator('#mp-group-warning')).to_be_visible()
    page.locator('#mp-group').select_option('central_midfield')
    expect(page.locator('#mp-group-warning')).to_be_hidden()
    page.locator('#mp-correct-player').click()
    for width in (1440, 1024, 768, 390):
        page.set_viewport_size({'width': width, 'height': 1000})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'), width
    page.screenshot(path=str(artifacts / 'player-correction-mobile.png'), full_page=True)
    page.reload(wait_until='domcontentloaded')
    page.wait_for_function('(pid)=>reviewDesk.ready&&state.pid===pid', arg=pid)
    expect(page.locator('#mp-name')).to_have_text('Corrected test midfielder')
    expect(page.locator('#mp-notes')).to_have_value('Keep this draft through identity correction')
    page.locator('#mp-correct-player').click()
    page.locator('#correction-reset').click()
    expect(page.locator('#identity-save-status')).to_have_text('Model identity restored.')
    expect(page.locator('#mp-not-goalkeeper')).to_be_visible()
    assert page.evaluate('currentMatch().players.find(p=>p.identity===state.pid).identity_correction') is None
    report['checks'].append('wrong goalkeeper corrected beside footage; failed save retains form; draft/time survive; correction persists and can be undone')
    print('PASS ' + report['checks'][-1], flush=True)


def main():
    artifacts = ROOT / '.runtime' / ('labelling-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    artifacts.mkdir(parents=True)
    data = artifacts / 'data'
    report = {'passed': False, 'checks': [], 'page_errors': [], 'server_errors': []}
    process = None
    try:
        manifest = next(m for m in S.datasets() if str(m.get('analysis', '')).startswith('full-match') and m.get('video'))
        examples = [manifest]
        chelsea = next((m for m in S.datasets() if m['id'] == 'sn-20160207-chelsea-manchester-united-h1'), None)
        if chelsea and chelsea['id'] != manifest['id']:
            examples.append(chelsea)
        for example in examples:
            target = data / 'datasets' / example['id']
            target.mkdir(parents=True)
            for name in ('manifest.json', 'profiles.json', 'tracks.csv.gz', 'crops.zip', 'match_stats.json', 'events.json', 'identities.json', 'analysis.json'):
                path = S.dataset_dir(example['id']) / name
                if path.exists():
                    shutil.copy2(path, target / name)
            cached = MC.cache_path(example)
            if cached.exists():
                (data / 'match_context').mkdir(exist_ok=True)
                shutil.copy2(cached, data / 'match_context' / cached.name)
                context = S.read_json(cached)
                urls = [t.get('logo') for t in context['teams']]
                urls.extend(p.get('headshot') for t in context['teams'] for p in t['players'])
                for url in filter(None, urls):
                    image = MC.media_path(url)
                    if image.is_file():
                        (data / 'match_assets').mkdir(exist_ok=True)
                        shutil.copy2(image, data / 'match_assets' / image.name)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        env = dict(os.environ, PITCHPROFILE_DATA=str(data), PITCHPROFILE_SOCCERNET=str(SN.root()), PITCHPROFILE_DEVICE='cuda:0', PYTHONUNBUFFERED='1')
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
                browser = pw.chromium.launch(channel='chrome', headless=True)
                page = browser.new_page(base_url=url, viewport={'width': 1440, 'height': 1000})
                page.set_default_timeout(20000)
                page.on('pageerror', lambda e: report['page_errors'].append(str(e)))
                page.on('response', lambda r: report['server_errors'].append(r.url) if r.status >= 500 and r.status != 503 else None)
                try:
                    page.goto(url, wait_until='domcontentloaded')
                    run_checks(page, report, artifacts)
                    run_usability_checks(page, report, artifacts)
                    run_correction_checks(page, report, artifacts)
                    run_context_checks(page, report, artifacts)
                    assert not report['page_errors'], report['page_errors']
                    assert not report['server_errors'], report['server_errors']
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
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
