"""The live model view's data: every shown event says which model made it and how its player was picked."""
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from football_profiler import event_live as EL
from football_profiler import event_review as ER
from football_profiler import storage as S

EVENTS = [
    {'type': 'pass', 'source': 'rules', 'segment': 's1', 'team': 'A', 'identity': 'A-7', 'time_s': 10.0,
     'x': 50.0, 'y': 30.0, 'outcome': 'complete', 'receiver_identity': 'A-9', 'length_m': 14.2},
    {'type': 'touch', 'source': 'rules', 'segment': 's1', 'team': 'A', 'identity': 'A-7', 'time_s': 9.9},
    {'type': 'pressure', 'source': 'rules', 'segment': 's3', 'team': 'B', 'identity': 'B-4', 'time_s': 10.1},
    {'type': 'shot', 'source': 'action_spotter', 'label': 'SHOT', 'score': .91, 'confident': True, 'team': 'A',
     'segment': 's2', 'identity': 'A-9', 'time_s': 12.4, 'credited_by': 'picture', 'uncredited': None,
     'x': 90.0, 'y': 34.0, 'ball_distance_m': float('nan')},
    {'type': 'tackle', 'source': 'action_spotter', 'label': 'PLAYER SUCCESSFUL TACKLE', 'score': .2,
     'confident': False, 'team': 'B', 'segment': None, 'identity': None, 'time_s': 11.0,
     'credited_by': None, 'uncredited': 'no ball', 'x': None, 'y': None},
]


def full_match(identifier='sn-test-h1', events=EVENTS, ball=True, spotter=True):
    d = S.dataset_dir(identifier, create=True)
    S.write_json(d / 'manifest.json', {
        'id': identifier, 'title': 'Alpha v Beta · H1', 'analysis': 'full-match-v9', 'half': 1,
        'teams': {'A': {'name': 'Alpha'}, 'B': {'name': 'Beta'}},
        'coverage': {'pitch_view_share': .8, 'ball_observed_share': .55, 'live_seconds': 100},
        'players': [{'player_id': 'A-7', 'name': 'Alpha #7', 'team_key': 'A', 'jersey': 7},
                    {'player_id': 'A-9', 'name': 'Alpha #9', 'team_key': 'A', 'jersey': 9},
                    {'player_id': 'B-4', 'name': 'Beta #4', 'team_key': 'B', 'jersey': 4}]})
    S.write_json(d / 'events.json', {'events': events})
    post = {'named_players': 3}
    if spotter:
        post['action_spotter'] = {'model': 'tdeed', 'events': {'shot': 1, 'pass': 4}}
    S.write_json(d / 'analysis.json', {'postprocess': post})
    if ball:
        pd.DataFrame({'time_s': [9.8, 9.88, 9.96, 12.0, 12.08, 20.0], 'cx': [1, 1, 1, 1, None, 1],
                      'interpolated': [False, False, False, False, False, True]}) \
            .to_csv(d / 'ball.csv.gz', index=False, compression='gzip')
    return identifier


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(EL.router)

    @app.exception_handler(ValueError)                 # as football_profiler/app.py
    async def value_error(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=400)
    return TestClient(app)


def test_live_events_keep_the_model_score_and_credit_and_hide_touches(isolated_data, api):
    identifier = full_match()
    r = api.get(f'/api/datasets/{identifier}/match/live-events')
    assert r.status_code == 200
    events = r.json()['events']
    assert [e['type'] for e in events] == ['pass', 'tackle', 'shot']           # sorted, no touch or pressure
    p, t, s = events
    assert p['model'] == 'rules' and p['credited_by'] == 'possession' and p['receiver_identity'] == 'A-9'
    assert 'score' not in p
    assert s['model'] == 'video' and s['score'] == .91 and s['confident'] and s['credited_by'] == 'picture'
    assert 'ball_distance_m' not in s                                            # NaN is left out
    assert 'identity' not in t                                                    # no player
    assert t['uncredited'] == 'no ball' and t['confident'] is False
    # The same ids as the event map's reviews, so a reviewed event can be found in both views.
    assert s['id'] == ER.event_id(EVENTS[3])


def test_ball_seen_intervals_skip_filled_in_and_missing_ball(isolated_data):
    identifier = full_match()
    spans = EL.ball_seen(S.dataset_dir(identifier))
    assert spans == [[9.8, 10.04], [12.0, 12.08]]                  # 12.08 has no picture position; 20.0 filled in
    assert EL.ball_seen(S.dataset_dir(full_match('sn-noball-h1', ball=False))) is None


def test_models_strip_says_whether_the_spotter_ran(isolated_data, api):
    on = {m['id']: m for m in api.get(f'/api/datasets/{full_match()}/match/live-events').json()['models']}
    assert on['action_spotter']['ran'] and on['action_spotter']['detail'] == '5 confident actions'
    assert on['pitch']['ran'] and on['detection']['detail'] == 'ball seen in 55% of live frames'
    off = {m['id']: m for m in api.get(f"/api/datasets/{full_match('sn-off-h1', spotter=False)}/match/live-events").json()['models']}
    assert not off['action_spotter']['ran'] and off['event_rules']['ran']


def test_whole_match_reads_the_spotter_from_its_halves(isolated_data):
    full_match('sn-x-h1')
    d = S.dataset_dir('sn-x-whole', create=True)
    S.write_json(d / 'manifest.json', {'id': 'sn-x-whole', 'analysis': 'full-match-v9', 'halves': ['sn-x-h1', 'sn-x-h2'],
                                       'players': [], 'teams': {}})
    S.write_json(d / 'events.json', {'events': []})
    models = {m['id']: m for m in EL.live_events('sn-x-whole')['models']}
    assert models['action_spotter']['ran']                        # the missing second half is skipped


def test_only_full_match_analyses_have_a_live_view(isolated_data, api):
    d = S.dataset_dir('clip-1', create=True)
    S.write_json(d / 'manifest.json', {'id': 'clip-1', 'analysis': 'clip'})
    assert api.get('/api/datasets/clip-1/match/live-events').status_code == 400
    with pytest.raises(ValueError):
        EL.live_events('clip-1')
