"""Check the integrated features against the running local app and real D: data.

Uses the first analysed full-match half. Every change it makes to real data
(team names, an archetype label) is reverted before it finishes.
"""
import json
import sys
from pathlib import Path
from urllib.request import urlopen
from playwright.sync_api import sync_playwright, expect
from check_browser import open_view

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from football_profiler import storage as S

BASE='http://127.0.0.1:8000'


def main():
    report={'checks':[],'page_errors':[],'server_errors':[]}
    target=S.EVIDENCE/'repository-integration';target.mkdir(parents=True,exist_ok=True)
    demo='soccernet-chelsea-burnley-h1-demo'
    with urlopen(BASE+'/api/matches/library') as response:halves=json.load(response)['halves']
    analysed=[h['dataset_id'] for h in halves if h['analysed']]
    assert analysed,'Analyse at least one full half first (scripts/analyse_matches.py)'
    match=analysed[0]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(channel='chrome',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.set_default_timeout(30000)
        page.on('pageerror',lambda e:report['page_errors'].append(str(e)))
        page.on('response',lambda r:report['server_errors'].append(r.url) if r.status>=500 else None)
        page.goto(BASE,wait_until='networkidle')
        page.locator('nav button[data-tab="data"]').click()
        expect(page.locator('#library-video option')).to_have_count(len(halves))
        expect(page.locator('#library-info')).to_contain_text('official split:')
        expect(page.locator('#library-events tbody tr')).not_to_have_count(0)
        page.locator('#library-video').evaluate('(el)=>el.closest("details").open=true')
        page.locator('#library-events').evaluate('(el)=>el.parentElement.open=true')
        page.locator('#library-events button').nth(1).click()
        assert float(page.locator('#library-start').input_value())>0
        report['checks'].append(f'{len(halves)} external halves, official splits, event-to-start selection')
        page.screenshot(path=str(target/'library.png'),full_page=True)
        open_view(page, 'overview')
        page.locator('#dataset-select').select_option(demo)
        page.wait_for_function('(id)=>state.manifest?.id===id && state.player?.player_id===state.pid',arg=demo)
        open_view(page, "analysis")
        expect(page.locator('#motion-note')).to_contain_text('Observed segments only')
        expect(page.locator('#kit-suggestion')).to_contain_text('Kit suggestion')
        expect(page.locator('#tactics-note')).to_contain_text('unavailable')
        report['checks'].append('interval detections, kit suggestions, uncalibrated movement abstention')
        response=page.request.get(f'{BASE}/api/datasets/{demo}/mot')
        assert response.ok and response.body().startswith(b'PK')
        report['checks'].append('MOT archive download')
        # Full-match analysis: calibrated movement, tactics and occupancy from video only.
        page.locator('#dataset-select').select_option(match)
        page.wait_for_function('(id)=>state.manifest?.id===id && state.player?.player_id===state.pid',arg=match)
        open_view(page, "analysis")
        expect(page.locator('#motion-note')).to_contain_text('Observed segments')
        # A moment of play where both teams have at least three identified outfield players in view.
        busy=page.evaluate('''async id=>{const t=await (await fetch('/api/datasets/'+id+'/match/events?kind=touch')).json();
          const times=t.events.map(e=>e.time_s);
          for(let i=0;i<80&&times.length;i++){const s=times[Math.floor((i+.5)*times.length/80)];
            const r=await (await fetch('/api/datasets/'+id+'/tactics?seconds='+s)).json();
            if((r.teams||[]).length===2)return s;}
          return null}''',match)
        assert busy is not None,'No moment with both teams identified in view'
        page.locator('#tactics-time').fill(str(round(busy,1)))
        page.locator('#tactics-form button').click()
        expect(page.locator('#tactics-teams tbody tr')).to_have_count(2)
        page.locator('#load-occupancy').click()
        expect(page.locator('#occupancy-maps canvas')).to_have_count(3)
        report['checks'].append('full-match movement, team radar and GPU occupancy difference from video')
        page.screenshot(path=str(target/'movement-and-tactics.png'),full_page=True)
        page.locator('nav button[data-tab="match"]').click()
        expect(page.locator('#match-content')).to_be_visible()
        assert page.locator('#match-players tbody tr').count()>=20
        page.locator('#team-names').evaluate('(el)=>{for(let d=el.closest("details");d;d=d.parentElement.closest("details"))d.open=true;}')
        original={k:page.locator(f'#team-{k.lower()}-name').input_value() for k in 'AB'}
        page.locator('#team-a-name').fill('Check A');page.locator('#team-b-name').fill('Check B')
        page.locator('#team-names button').click()
        page.wait_for_function('()=>state.manifest?.teams?.A?.name==="Check A"')
        page.request.post(f'{BASE}/api/datasets/{match}/match/team-names',data=original,headers={'Origin':BASE})
        report['checks'].append('team naming saved and restored')
        page.locator('nav button[data-tab="match"]').click()
        page.locator('nav button[data-tab="data"]').click()
        page.locator('#match-players').evaluate('(el)=>el.closest("details").open=true')
        page.locator('#match-players tbody tr').first.click()
        page.locator('nav button[data-tab="match"]').click()
        page.wait_for_function("()=>document.querySelectorAll('#mp-roles input[type=range]').length>0 && matchState.boxes?.length>0")
        page.locator('#mp-correct-player').click()
        page.locator('#mp-crops').scroll_into_view_if_needed()
        page.wait_for_function("()=>[...document.querySelectorAll('#mp-crops img')].some(i=>i.complete&&i.naturalWidth>0)")
        page.locator('#identify-cancel').click()
        expect(page.locator('#mp-stats')).to_contain_text('Physical')
        page.locator('[data-evidence-view="map"]').click()
        page.wait_for_function('eventMap.data !== null')
        expect(page.locator('#stat-summary')).to_contain_text('Passes')
        events=page.evaluate('eventMap.data.events')
        if events:
            page.evaluate('(id)=>selectMappedEvent(id)',events[0]['id'])
            page.locator('#selected-event-watch').click()
            page.wait_for_function('!document.querySelector("#mp-video").paused')
            page.locator('#mp-video').evaluate('(v)=>v.pause()')
        report['checks'].append('player statistics, thumbnails, events and video seeking with identity box')
        pid=page.evaluate('()=>state.pid')
        before=page.evaluate('async a=>(await (await fetch(a)).json()).label',f'/api/datasets/{match}/players/{pid}/archetype')
        page.locator('#mp-labeler').fill('ui-check')
        page.locator('#mp-roles .role-card button[data-value="50"]').first.click()
        page.locator('#mp-roles input[type=range]').first.evaluate("(el)=>{el.value=70;el.dispatchEvent(new Event('input',{bubbles:true}));}")
        page.locator('#mp-mixture').evaluate('(el)=>el.closest("details").open=true')
        expect(page.locator('#mp-mixture')).to_contain_text('100.0%')
        page.locator('#mp-save').click()
        page.wait_for_function('()=>document.querySelector("#mp-label-state").textContent.includes("ui-check")')
        if before is None:
            page.request.delete(f'{BASE}/api/datasets/{match}/players/{pid}/archetype',headers={'Origin':BASE})
        else:
            page.request.post(f'{BASE}/api/datasets/{match}/players/{pid}/archetype',headers={'Origin':BASE},
                              data={k:before[k] for k in ('labeler','position_group','labels','notes','evidence')})
        report['checks'].append('archetype percentage label saved from the match view and reverted')
        page.screenshot(path=str(target/'match-statistics.png'),full_page=True)
        # Name players: unnamed players listed with pictures; one gets an unused number, then it is undone.
        page.wait_for_function("()=>matchState.data && !document.querySelector('#naming-panel').hidden")
        unnamed=page.evaluate("()=>matchState.data.players.filter(p=>p.unnamed).length")
        if unnamed:
            page.locator('#naming-panel > summary').click()
            expect(page.locator('#naming-list .naming-row')).to_have_count(unnamed)
            page.wait_for_function("()=>[...document.querySelectorAll('#naming-list img')].some(i=>i.complete&&i.naturalWidth>0)")
            page.screenshot(path=str(target/'naming-panel.png'),full_page=False)
            row=page.locator('#naming-list .naming-row').first
            source=row.get_attribute('data-pid');team=source.split('-')[0]
            used=page.evaluate("(t)=>matchState.data.players.filter(p=>p.team===t&&p.jersey!=null).map(p=>p.jersey)",team)
            number=next(n for n in range(99,0,-1) if n not in used)
            row.locator('input').fill(str(number))
            expect(row.locator('.naming-join')).to_contain_text('New player')
            expect(page.locator('#naming-apply')).to_have_text('Apply 1 change')
            page.locator('#naming-apply').click()
            named=f'{team}-{number}'
            page.wait_for_function("(t)=>matchState.data?.players.some(p=>p.identity===t&&p.identity_confirmed&&p.jersey!==null)",arg=named,timeout=300000)
            page.locator('#naming-given').evaluate('(el)=>el.open=true')
            page.locator(f'[data-undo="{named}"]').click()
            page.wait_for_function("(t)=>matchState.data&&!matchState.data.players.some(p=>p.identity===t)",arg=named,timeout=300000)
            # The same player marked as not a player leaves every player's statistics, then is restored.
            row=page.locator('#naming-list .naming-row').first
            row.locator('[data-notplayer]').click()
            expect(row.locator('.naming-join')).to_contain_text('Removed from player statistics')
            expect(page.locator('#naming-apply')).to_have_text('Apply 1 change')
            page.locator('#naming-apply').click()
            page.wait_for_function("()=>matchState.data?.not_player_segments>0",timeout=300000)
            page.locator('#naming-given').evaluate('(el)=>el.open=true')
            page.locator('[data-undo="NONE"]').click()
            page.wait_for_function("()=>matchState.data&&!matchState.data.not_player_segments",timeout=300000)
            report['checks'].append(f'name players: {unnamed} unnamed players with pictures; a number and a not-a-player mark applied, then undone')
        page.locator('nav button[data-tab="learning"]').click()
        expect(page.locator('#learn-library tbody tr')).to_have_count(len(halves))
        expect(page.locator('#learn-summary')).to_contain_text('labelled player appearances')
        report['checks'].append('archetype learning workflow: library, label summary, fit controls')
        page.screenshot(path=str(target/'archetype-learning.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        for tab in ('match','learning','analysis','data'):
            open_view(page, tab)
            page.wait_for_timeout(300)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'),tab
        report['checks'].append('mobile match, learning, movement and library layouts')
        assert not report['page_errors'],report['page_errors']
        assert not report['server_errors'],report['server_errors']
        browser.close()
    with urlopen(BASE+'/api/status') as response:report['status']=json.load(response)
    S.write_json(target/'browser-report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
