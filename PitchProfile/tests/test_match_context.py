import json

import pytest

from football_profiler import storage as S
from football_profiler import match_context as MC
from football_profiler import event_review as ER


def fixture():
    m = {'id': 'context-test', 'match_id': 'soccernet:england_epl/2015-2016/2016-02-07 - Chelsea 1 - 1 Manchester United',
         'date': '2016-02-07', 'analysis': 'full-match-1', 'duration_seconds': 2700,
         'fixture_teams': {'home': 'Chelsea', 'away': 'Manchester United'},
         'players': [{'player_id': 'A-10', 'team_key': 'A', 'jersey': 10, 'name': 'Team A #10'},
                     {'player_id': 'A-99', 'team_key': 'A', 'jersey': 99, 'name': 'Team A #99'}]}
    S.write_json(S.dataset_dir(m['id'], create=True) / 'manifest.json', m)
    context = {'provider': 'ESPN', 'event_id': '422419', 'date': '2016-02-07T16:00Z', 'source_url': 'https://www.espn.com/soccer/match/_/gameId/422419',
               'teams': [{'id': '363', 'side': 'home', 'name': 'Chelsea', 'logo': None, 'players': []},
                         {'id': '360', 'side': 'away', 'name': 'Manchester United', 'logo': None, 'players': [
                             {'id': '21046', 'shirt': 10, 'name': 'Wayne Rooney', 'played': True, 'starter': True,
                              'headshot': 'https://a.espncdn.com/i/headshots/soccer/players/full/21046.png'},
                             {'id': '169136', 'shirt': 35, 'name': 'Jesse Lingard', 'played': True, 'starter': True, 'headshot': None},
                             {'id': '999', 'shirt': 44, 'name': 'Unused test substitute', 'played': False, 'starter': False, 'headshot': None}]}]}
    S.write_json(MC.cache_path(m), context)
    return m, context


def test_match_day_number_lookup_requires_kit_mapping_and_keeps_original_track(client):
    m, context = fixture()
    p = m['players'][0]
    assert MC.decorate(m, p)['identity_status'] == 'kit_unmapped'
    MC.bind_teams(m, {'A': '360', 'B': '363'}, 'reviewer')
    resolved = MC.decorate(m, p)
    assert resolved['name'] == 'Wayne Rooney' and resolved['jersey'] == 10
    assert resolved['headshot'].endswith('21046.png')
    assert resolved['identity_status'] == 'lineup_match'  # Lookup is not proof of visual identity.
    unknown = MC.decorate(m, m['players'][1])
    assert unknown['identity_status'] == 'not_in_lineup'
    url = '/api/datasets/context-test/players/A-99/lineup-identity'
    result = client.post(url, json={'athlete_id': '169136', 'reviewer': 'human', 'note': 'Read #35 in footage'})
    assert result.status_code == 200
    assert result.json()['name'] == 'Jesse Lingard'
    assert result.json()['jersey'] == 35 and result.json()['automatic_jersey'] == 99
    assert result.json()['player_id'] == 'A-99' and result.json()['identity_status'] == 'confirmed'
    assert S.read_json(S.dataset_dir(m['id']) / 'manifest.json')['players'][1]['jersey'] == 99
    # Rebuilding raw manifests cannot erase the separate reviewer correction.
    S.write_json(S.dataset_dir(m['id']) / 'manifest.json', m)
    assert MC.decorate(m, m['players'][1])['name'] == 'Jesse Lingard'
    with S.db() as db:
        assert db.execute('SELECT COUNT(*) FROM player_identity_link_history').fetchone()[0] == 1
    assert client.post(url, json={'athlete_id': '999', 'reviewer': 'human'}).status_code == 400
    MC.bind_teams(m, {'A': '363', 'B': '360'}, 'reviewer')
    assert MC.decorate(m, m['players'][1])['identity_status'] == 'not_in_lineup'


def test_kit_validation_historical_cache_and_url_boundaries(client):
    m, _ = fixture()
    with pytest.raises(ValueError):
        MC.bind_teams(m, {'A': '360', 'B': '360'}, 'reviewer')
    with pytest.raises(ValueError):
        MC.bind_teams(m, {'A': '360', 'B': '363'}, ' ')
    assert MC.cache_path({**m, 'id': 'other-half'}) == MC.cache_path(m)
    for bad in ('http://127.0.0.1/gameId/422419', 'https://www.espn.com.evil.test/gameId/422419', 'javascript:alert(1)'):
        with pytest.raises(ValueError):
            MC.event_id_from(bad)
    assert MC.event_id_from('https://www.espn.co.uk/football/match/_/gameId/422419/manchester-united-chelsea') == '422419'
    assert MC.safe_media('https://evil.test/player.png') is None


def test_direct_player_selection_does_not_require_or_change_team_mapping(client):
    m, _ = fixture()
    url = '/api/datasets/context-test/players/A-99/lineup-identity'
    result = client.post(url, json={'athlete_id': '21046', 'team_id': '360', 'reviewer': 'Reviewer'})
    assert result.status_code == 200, result.text
    assert result.json()['name'] == 'Wayne Rooney' and result.json()['identity_status'] == 'confirmed'
    assert result.json()['team_id'] == '360'
    assert MC.state_for(m)['binding'] == {}
    assert MC.decorate(m, m['players'][0])['identity_status'] == 'kit_unmapped'
    # An explicitly selected individual is independent of later global kit mapping.
    MC.bind_teams(m, {'A': '363', 'B': '360'}, 'Reviewer')
    assert MC.decorate(m, m['players'][1])['name'] == 'Wayne Rooney'
    assert client.post(url, json={'athlete_id': '21046', 'team_id': '363', 'reviewer': 'Reviewer'}).status_code == 400
    assert client.post(url, json={'athlete_id': '999', 'team_id': '360', 'reviewer': 'Reviewer'}).status_code == 400
    assert MC.decorate(m, m['players'][1])['name'] == 'Wayne Rooney'
    reset = client.post('/api/datasets/context-test/players/A-99/track-correction', json={'role': 'player', 'reviewer': 'Reviewer', 'reset': True})
    assert reset.status_code == 200 and reset.json()['identity_status'] != 'confirmed'
    with S.db() as db:
        assert db.execute('SELECT COUNT(*) FROM player_identity_link_history').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM player_track_correction_history').fetchone()[0] == 1


def test_parser_uses_historical_jersey_and_deduplicates_reported_shots():
    player = {'jersey': '10', 'starter': True, 'athlete': {'id': '21046', 'jersey': '99', 'fullName': 'Wayne Rooney',
              'headshot': {'href': 'https://a.espncdn.com/i/headshots/soccer/players/full/21046.png'}}}
    shot = {'play': {'id': 'test-shot', 'type': {'type': 'shot-blocked'}, 'period': {'number': 1},
                    'clock': {'displayValue': "12'", 'value': 720}, 'participants': [{'athlete': {'displayName': 'Wayne Rooney'}}]}}
    raw = {'header': {'id': '422419', 'competitions': [{'date': '2016-02-07T16:00Z', 'competitors': [
        {'team': {'id': '360', 'displayName': 'Manchester United'}, 'homeAway': 'away', 'score': '1'}]}]},
        'rosters': [{'team': {'id': '360'}, 'roster': [player]}], 'commentary': [shot, shot]}
    parsed = MC.parse_summary(raw, 'eng.1')
    assert parsed['teams'][0]['players'][0]['shirt'] == 10
    assert len(parsed['reported_shots']) == 1 and parsed['reported_shots'][0]['outcome'] == 'blocked'
    assert 'x' not in parsed['reported_shots'][0]  # No unverified conversion of provider coordinates.


def test_refresh_failure_does_not_replace_cached_match(monkeypatch):
    m, original = fixture()
    monkeypatch.setattr(MC, 'fetch_json', lambda *args: {'header': {'id': '1', 'competitions': []}})
    with pytest.raises(ValueError):
        MC.sync(m, '422419', refresh=True)
    assert S.read_json(MC.cache_path(m)) == original


def test_event_maps_and_outcome_reviews_are_separate_from_detections(client):
    m, _ = fixture()
    events = [
        {'type': 'pass', 'segment': 'seg1', 'team': 'A', 'identity': 'A-10', 'time_s': 10, 'x': 20, 'y': 30,
         'end_x': 50, 'end_y': 40, 'attack_sign': -1, 'outcome': 'complete'},
        {'type': 'shot', 'segment': 'seg1', 'team': 'A', 'identity': 'A-10', 'time_s': 15, 'x': 15, 'y': 34,
         'attack_sign': -1, 'evidence': 'model', 'probability': .8},
        {'type': 'tackle', 'segment': 'seg1', 'team': 'A', 'identity': 'A-10', 'time_s': 20, 'x': 150, 'y': 40},
    ]
    path = S.dataset_dir(m['id']) / 'events.json'; S.write_json(path, {'events': events})
    root = '/api/datasets/context-test/match'
    data = client.get(root + '/event-map?player=A-10').json()
    assert [e['outcome'] for e in data['events']] == ['complete', 'unknown', 'won']
    assert data['events'][0]['attack_sign'] == -1
    assert data['events'][2]['x'] is None
    key = data['events'][1]['id']
    response = client.post(root + f'/events/{key}/review', json={'outcome': 'goal', 'reviewer': 'human', 'notes': 'Watched the goal'})
    assert response.status_code == 200
    reviewed = client.get(root + '/event-map?player=A-10').json()['events'][1]
    assert reviewed['outcome'] == 'goal' and reviewed['outcome_source'] == 'reviewed'
    assert S.read_json(path) == {'events': events}
    assert client.post(root + f'/events/{key}/review', json={'outcome': 'complete', 'reviewer': 'human'}).status_code == 400
    assert client.get(root + '/event-map?player=missing').status_code == 404
    # Changing the source event's identity-bearing segment invalidates the old review key.
    events[1]['segment'] = 'new-segment'; S.write_json(path, {'events': events})
    assert client.get(root + '/event-map?player=A-10').json()['events'][1]['outcome'] == 'unknown'
    assert client.post(root + f'/events/{key}/review', json={'outcome': 'goal', 'reviewer': 'human'}).status_code == 404
