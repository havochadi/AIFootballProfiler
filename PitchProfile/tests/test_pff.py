import bz2
import json
import zipfile

import pytest

from football_profiler import pff as P, storage as S


def _frame(period, elapsed, frame_num, home, away):
    return {'period': period, 'periodElapsedTime': elapsed, 'frameNum': frame_num,
            'homePlayers': home, 'awayPlayers': away}


def _sample(jersey, x, y, visible=True):
    return {'jerseyNum': jersey, 'confidence': 'HIGH', 'x': x, 'y': y,
            'visibility': 'VISIBLE' if visible else 'ESTIMATED'}


@pytest.fixture
def sample_archive(tmp_path):
    """Two synthetic frames per period for a 2v2 fixture, matching the real
    PFF release format verified against actual World Cup 2022 data."""
    meta = [{'id': '9001', 'homeTeam': {'id': '1', 'name': 'Home FC'}, 'awayTeam': {'id': '2', 'name': 'Away FC'},
            'homeTeamStartLeft': True, 'fps': 10.0, 'date': '2022-11-21T13:00:00'}]
    roster = [
        {'player': {'id': '501', 'nickname': 'Home Keeper'}, 'positionGroupType': 'GK', 'shirtNumber': '1', 'team': {'id': '1'}},
        {'player': {'id': '502', 'nickname': 'Home Striker'}, 'positionGroupType': 'CF', 'shirtNumber': '9', 'team': {'id': '1'}},
        {'player': {'id': '601', 'nickname': 'Away Keeper'}, 'positionGroupType': 'GK', 'shirtNumber': '1', 'team': {'id': '2'}},
        {'player': {'id': '602', 'nickname': 'Away Winger'}, 'positionGroupType': 'LW', 'shirtNumber': '11', 'team': {'id': '2'}},
    ]
    # Home GK defends the left goal (raw x very negative) throughout period 1;
    # ends swap at half-time so it defends the right goal (raw x very positive) in period 2.
    frames = [
        _frame(1, 0.0, 100, [_sample('1', -50.0, 0.0), _sample('9', 10.0, 0.0)], [_sample('1', 50.0, 0.0), _sample('11', -10.0, 0.0, visible=False)]),
        _frame(1, 0.1, 101, [_sample('1', -49.0, 1.0), _sample('9', 11.0, 1.0)], [_sample('1', 49.0, 0.0), _sample('11', -9.0, 0.0)]),
        _frame(2, 0.0, 200, [_sample('1', 50.0, 0.0), _sample('9', -10.0, 0.0)], [_sample('1', -50.0, 0.0), _sample('11', 10.0, 0.0)]),
        _frame(2, 0.1, 201, [_sample('1', 49.0, 1.0), _sample('9', -11.0, 1.0)], [_sample('1', -49.0, 0.0), _sample('11', 9.0, 0.0)]),
    ]
    jsonl = '\n'.join(json.dumps(f) for f in frames).encode('utf-8')
    path = tmp_path / 'archive.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('Metadata/9001.json', json.dumps(meta))
        z.writestr('Rosters/9001.json', json.dumps(roster))
        z.writestr('Tracking Data/9001.jsonl.bz2', bz2.compress(jsonl))
    return [path]


def test_list_games_finds_metadata_entries(sample_archive):
    assert P.list_games(sample_archive) == ['9001']


def test_direction_normalises_consistently_across_the_half_time_swap(sample_archive):
    meta, by_period = P.parse_game(sample_archive, '9001', sampling_hz=10)
    for period in (1, 2):
        tracks, players, info = by_period[period]
        home_gk = tracks[tracks.player_id == 'pff-501']
        # After normalisation the home goalkeeper stays near the defensive (low-x) side
        # in both periods, even though the raw coordinates flip sign at half-time.
        assert (home_gk.x < 10).all()
        assert info['half'] == period


def test_position_group_mapped_from_roster_and_visibility_maps_to_detected(sample_archive):
    _, by_period = P.parse_game(sample_archive, '9001', sampling_hz=10)
    tracks, players, info = by_period[1]
    by_id = {p['player_id']: p for p in players}
    assert by_id['pff-501']['position_group'] == 'goalkeeper'
    assert by_id['pff-502']['position_group'] == 'centre_forward'
    assert by_id['pff-602']['position_group'] == 'wide_attacker'
    away_winger = tracks[tracks.player_id == 'pff-602']
    assert away_winger.detected.tolist() == [0, 1]  # first sample was ESTIMATED, second VISIBLE


def test_total_sampled_frames_does_not_double_count_across_periods(sample_archive):
    """frameNum is a whole-match counter (200/201 for period 2, not 0/1);
    total_sampled_frames must reflect this half's own distinct frames only."""
    _, by_period = P.parse_game(sample_archive, '9001', sampling_hz=10)
    for period in (1, 2):
        tracks, players, info = by_period[period]
        assert info['total_sampled_frames'] == tracks.frame.nunique() == 2


def test_import_match_creates_both_halves_and_rejects_reimport(sample_archive):
    manifests = P.import_match(sample_archive, '9001')
    assert set(manifests) == {1, 2}
    assert manifests[1]['id'] == 'pff-wc2022-9001-h1'
    assert manifests[1]['match_id'] == 'pff:9001'
    assert manifests[1]['clock'] == 'period_relative'
    assert manifests[1]['source_kind'] == 'provider_tracking'
    profiles = S.read_json(S.dataset_dir('pff-wc2022-9001-h1') / 'profiles.json')
    assert {p['player_id'] for p in profiles} == {'pff-501', 'pff-502', 'pff-601', 'pff-602'}
    with pytest.raises(ValueError, match='already imported'):
        P.import_match(sample_archive, '9001')


def test_missing_orientation_is_rejected(sample_archive, tmp_path):
    with zipfile.ZipFile(sample_archive[0]) as z:
        meta = json.loads(z.read('Metadata/9001.json'))
    meta[0]['homeTeamStartLeft'] = None
    path = tmp_path / 'no_orientation.zip'
    with zipfile.ZipFile(sample_archive[0]) as src, zipfile.ZipFile(path, 'w') as dst:
        for name in src.namelist():
            dst.writestr(name, json.dumps(meta) if name == 'Metadata/9001.json' else src.read(name))
    with pytest.raises(ValueError, match='orientation'):
        P.parse_game([path], '9001', sampling_hz=10)
