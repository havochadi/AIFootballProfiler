import pytest

from football_profiler import storage as S, match_context as MC


def keeper_fixture():
    person = {'player_id': 'A-GK', 'team_key': 'A', 'team': 'Team A', 'name': 'Team A · goalkeeper',
              'role': 'Goalkeeper', 'jersey': None, 'position_group': 'goalkeeper'}
    manifest = {'id': 'correction-test', 'match_id': 'artificial-correction-test', 'analysis': 'full-match-1',
                'title': 'Artificial correction fixture', 'duration_seconds': 300, 'players': [person],
                'teams': {'A': {'name': 'Team A'}, 'B': {'name': 'Team B'}}}
    directory = S.dataset_dir(manifest['id'], create=True)
    S.write_json(directory / 'manifest.json', manifest)
    S.write_json(directory / 'profiles.json', [person])
    S.write_json(directory / 'match_stats.json', {'players': [{'identity': 'A-GK', 'team': 'A', 'role': 'goalkeeper',
                 'visible_seconds': 200, 'on_ball': {'passes': 5}, 'positional': {}, 'physical': {}}]})
    return manifest, person, directory


def payload(**extra):
    return {'role': 'player', 'jersey': 10, 'name': 'Alex Example', 'reviewer': 'Reviewer',
            'time_s': 30, 'note': 'Not the goalkeeper', **extra}


URL = '/api/datasets/correction-test/players/A-GK/track-correction'


def test_correction_without_online_context_survives_reload_and_preserves_evidence(client):
    m, person, directory = keeper_fixture()
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    result = client.post(URL, json=payload())
    assert result.status_code == 200, result.text
    corrected = result.json()
    assert corrected['role'] == 'player' and corrected['position_group'] is None
    assert corrected['name'] == 'Alex Example' and corrected['jersey'] == 10
    assert corrected['automatic_role'] == 'Goalkeeper' and corrected['identity_status'] == 'corrected'
    assert corrected['identity_correction']['time_s'] == 30
    assert corrected['identity_correction']['scope'] == 'track_in_half'
    assert client.get('/api/datasets/correction-test/players/A-GK').json()['role'] == 'player'
    assert client.get('/api/datasets/correction-test').json()['profiles'][0]['name'] == 'Alex Example'
    summary = client.get('/api/datasets/correction-test/match').json()['players'][0]
    assert summary['role'] == 'player' and summary['on_ball']['passes'] == 5
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert MC.decorate(m, person)['name'] == 'Alex Example'


def test_not_goalkeeper_requires_no_name_and_can_restore_model_identity(client):
    m, person, _ = keeper_fixture()
    correction = payload(name='', jersey=None)
    assert client.post(URL, json=correction).status_code == 200
    assert MC.decorate(m, person)['name'] == 'Team A · outfield player'
    result = client.post(URL, json={**correction, 'reset': True})
    assert result.status_code == 200
    assert result.json()['role'] == 'Goalkeeper'
    assert result.json()['identity_correction'] is None
    with S.db() as db:
        assert db.execute('SELECT COUNT(*) FROM player_track_correction_history').fetchone()[0] == 2


def test_lineup_confirmation_also_corrects_role_and_supersedes_manual_name(client):
    m, person, _ = keeper_fixture()
    context = {'event_id': '123', 'source_url': 'https://www.espn.com/soccer/match/_/gameId/123', 'teams': [
        {'id': '1', 'name': 'Home', 'logo': None, 'players': [
            {'id': '10', 'name': 'Actual midfielder', 'shirt': 10, 'position': 'Midfielder', 'played': True, 'headshot': None},
            {'id': '1', 'name': 'Actual keeper', 'shirt': 1, 'position': 'Goalkeeper', 'played': True, 'headshot': None}]},
        {'id': '2', 'name': 'Away', 'logo': None, 'players': []}]}
    S.write_json(MC.cache_path(m), context)
    MC.bind_teams(m, {'A': '1', 'B': '2'}, 'Reviewer')
    # Rejecting the goalkeeper must not redisplay an automatic keeper portrait/name.
    assert client.post(URL, json=payload(name='', jersey=1)).status_code == 200
    assert MC.decorate(m, person)['lineup_player'] is None
    result = client.post(URL.replace('track-correction', 'lineup-identity'), json={'athlete_id': '10', 'reviewer': 'Reviewer'})
    assert result.status_code == 200
    corrected = result.json()
    assert corrected['role'] == 'player' and corrected['name'] == 'Actual midfielder'
    assert corrected['position_group'] is None and corrected['identity_correction'] is None


@pytest.mark.parametrize('changes,status', [({'reviewer': ' '}, 400), ({'jersey': 0}, 422),
    ({'jersey': 10.5}, 422), ({'role': 'referee'}, 422), ({'time_s': 300}, 400)])
def test_invalid_correction_does_not_write(client, changes, status):
    m, person, _ = keeper_fixture()
    assert client.post(URL, json=payload(**changes)).status_code == status
    assert MC.decorate(m, person)['identity_correction'] is None


def test_unknown_track_is_rejected(client):
    keeper_fixture()
    assert client.post(URL.replace('A-GK', 'missing'), json=payload()).status_code == 404
