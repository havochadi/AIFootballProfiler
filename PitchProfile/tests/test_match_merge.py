import numpy as np
import pandas as pd

from football_profiler import match_merge as MM
from football_profiler import storage as S


def test_kit_groups_are_matched_by_colour():
    first = {'centres': {'A': {'lab': [200, 128, 128]}, 'B': {'lab': [60, 150, 170]}}}
    swapped = {'centres': {'A': {'lab': [62, 148, 168]}, 'B': {'lab': [198, 129, 127]}}}
    assert MM.kit_mapping(first, swapped) == {'A': 'B', 'B': 'A'}
    assert MM.kit_mapping(first, first) == {'A': 'A', 'B': 'B'}
    assert MM.kit_mapping(first, {}) == {'A': 'A', 'B': 'B'}


def test_player_ids_across_halves():
    kits = {'A': 'B', 'B': 'A'}
    assert MM.match_player_id('A-16', 1, kits) == 'A-16'
    assert MM.match_player_id('B-16', 2, kits) == 'A-16'          # same player, kit groups swapped
    assert MM.match_player_id('B-GK', 2, kits) == 'A-GK'
    assert MM.match_player_id('A-X3', 2, kits) == 'B-Y3'          # unnamed second-half player
    assert MM.match_player_id('B-P2', 2, kits) == 'A-P2'          # a reviewer's person keeps his id
    assert MM.half_player_id('A-16', 2, kits) == 'B-16'
    assert MM.half_player_id('B-Y3', 2, kits) == 'A-X3'
    assert MM.half_name('NONE', 2, kits) == 'NONE'


def test_combine_shifts_the_second_half_and_mirrors_it():
    def half(team_b_attacks_right, colours):
        frames = pd.DataFrame({'sample': [0, 1], 'time_s': [0.0, .08], 'view_shot': [0, 0], 'source_frame': [0, 2],
                               'pitch_view': [True, True]})
        people = pd.DataFrame({'sample': [0, 1], 'time_s': [0.0, .08], 'view_shot': [0, 0], 'track': ['t1', 't1'],
                               'tracklet': ['t1', 't1'], 'segment': ['s1', 's1'], 'team': ['A', 'A'], 'x': [10.0, 11.0],
                               'y': [20.0, 20.0]})
        events = pd.DataFrame({'type': ['pass'], 'segment': ['s1'], 'team': ['A'], 'time_s': [0.0], 'view_shot': [0],
                               'x': [10.0], 'y': [20.0], 'end_x': [30.0], 'end_y': [20.0], 'attack_sign': [1],
                               'receiver': [np.nan]})
        spells = pd.DataFrame({'segment': ['s1'], 'team': ['A'], 'start_s': [0.0], 'end_s': [.08], 'start_sample': [0],
                               'end_sample': [1], 'view_shot': [0], 'x0': [10.0], 'y0': [20.0], 'x1': [11.0], 'y1': [20.0]})
        ball = pd.DataFrame({'sample': [0], 'time_s': [0.0], 'view_shot': [0], 'ball_segment': ['b1'], 'x': [10.0], 'y': [20.0]})
        rows = pd.DataFrame([{'identity': 'A-7', 'team': 'A', 'label': 'x', 'named': True, 'number': 7, 'segments': 1,
                              'observed_s': 1.0, 'confirmed': False}])
        return {'frames': frames, 'people': people, 'events': events, 'spells': spells, 'ball': ball,
                'identity': {'s1': 'A-7'}, 'identities': rows, 'numbers': {}, 'live_seconds': 1.0, 'hz': 12.5,
                'meta': {'detection': {'source_fps': 25.0}},
                'directions': {'status': 'inferred', 'attacks_right': 'B' if team_b_attacks_right else 'A',
                               'defends_left': 'A' if team_b_attacks_right else 'B'},
                'team_info': {'centres': {'A': {'lab': colours[0]}, 'B': {'lab': colours[1]}}}}
    first = half(False, ([200, 128, 128], [60, 150, 170]))       # A attacks right in the first half
    second = half(False, ([200, 128, 128], [60, 150, 170]))      # ...and, same kits, left after half-time?
    second['directions'] = {'status': 'inferred', 'attacks_right': 'B', 'defends_left': 'A'}
    c = MM.combine(first, second, 2700.0)
    assert c['mirrored'] and c['kits'] == {'A': 'A', 'B': 'B'}
    p = c['people']
    assert list(p.segment) == ['h1:s1', 'h1:s1', 'h2:s1', 'h2:s1'] and list(p['sample']) == [0, 1, 2, 3]
    assert p.time_s.iloc[2] == 2700.0 and p.x.iloc[2] == 105 - 10.0 and p.y.iloc[2] == 68 - 20.0
    assert c['identity'] == {'h1:s1': 'A-7', 'h2:s1': 'A-7'}
    assert list(c['identities'].identity) == ['A-7'] and c['identities'].segments.iloc[0] == 2
    assert list(c['events'].attack_sign) == [1, 1]
    assert list(c['frames'].source_frame) == [0, 2, 67500, 67502]


def test_names_on_a_whole_match_go_to_the_half_they_belong_to(client, monkeypatch):
    from football_profiler import app as APP
    monkeypatch.setattr(APP, 'start_job', lambda kind, func, **kw: {'job_id': 'j', 'halves': sorted(kw['halves'])})
    for h in ('m-h1', 'm-h2'):
        S.write_json(S.dataset_dir(h, create=True) / 'manifest.json', {'id': h, 'analysis': 'full-match-1', 'players': []})
    d = S.dataset_dir('m', create=True)
    S.write_json(d / 'manifest.json', {'id': 'm', 'analysis': 'full-match-1', 'players': [], 'whole_match': True,
                                       'halves': ['m-h1', 'm-h2'], 'kit_mapping_second_half': {'A': 'B', 'B': 'A'}})
    S.write_json(d / 'identities.json', {'segments': {'h1:s1': {'identity': 'A-X1'}, 'h2:s9': {'identity': 'A-X1'},
                                                      'h2:s5': {'identity': 'A-Y4'}, 'h1:s2': {'identity': 'A-16'}}})
    base = '/api/datasets/m/match/identities'
    r = client.post(base + '/names', json={'names': [{'source': 'A-X1', 'number': 8}]})
    assert r.status_code == 200 and r.json()['halves'] == ['m-h1', 'm-h2']
    assert S.read_json(S.dataset_dir('m-h1') / 'confirmed_identities.json') == {'s1': 'A-8'}
    assert S.read_json(S.dataset_dir('m-h2') / 'confirmed_identities.json') == {'s9': 'B-8'}   # kit group B there
    # Two unnamed players confirmed as one person share a new reviewer id in both halves.
    assert client.post(base + '/names', json={'names': [{'source': 'A-Y4', 'same_as': 'A-16'}]}).status_code == 200
    assert S.read_json(S.dataset_dir('m-h2') / 'confirmed_identities.json')['s5'] == 'B-16'
    S.write_json(d / 'identities.json', {'segments': {'h1:s1': {'identity': 'A-X1'}, 'h2:s5': {'identity': 'A-Y4'}}})
    assert client.post(base + '/names', json={'names': [{'source': 'A-Y4', 'same_as': 'A-X1'}]}).status_code == 200
    assert S.read_json(S.dataset_dir('m-h1') / 'confirmed_identities.json')['s1'] == 'A-P1'
    assert S.read_json(S.dataset_dir('m-h2') / 'confirmed_identities.json')['s5'] == 'B-P1'
    assert client.post(base + '/names', json={'names': [{'source': 'A-Y4', 'same_as': 'B-9'}]}).status_code == 400
    assert client.post(base + '/undo', json={'identity': 'A-P1'}).status_code == 200
    assert 's1' not in S.read_json(S.dataset_dir('m-h1') / 'confirmed_identities.json')
    assert 's5' not in S.read_json(S.dataset_dir('m-h2') / 'confirmed_identities.json')


def test_corrections_from_both_halves_fill_the_whole_match_without_replacing_one():
    with S.db() as c:
        c.execute('INSERT INTO player_identity_links(dataset_id,player_id,payload) VALUES(?,?,?)', ('x-h1', 'A-16', 'first'))
        c.execute('INSERT INTO player_identity_links(dataset_id,player_id,payload) VALUES(?,?,?)', ('x-h2', 'B-16', 'second'))
    assert S.copy_player_records('x-h1', 'x', {'A-16': 'A-16'}) == 1
    assert S.copy_player_records('x-h2', 'x', {'B-16': 'A-16'}) == 0           # the whole match already has one: it stays
    with S.db() as c:
        assert c.execute('SELECT payload FROM player_identity_links WHERE dataset_id=? AND player_id=?', ('x', 'A-16')).fetchone()['payload'] == 'first'


def test_unnamed_players_take_a_name_seen_in_the_other_half():
    def ev(v):
        v = np.array(v, float)
        return {'emb': v, 'crops': 1, 'keeper': 0, 'tens': np.zeros(10), 'units': np.zeros(10), 'weight': 1.0}
    identity = {'h1:a': 'A-16', 'h2:b': 'A-Y1', 'h1:c': 'A-8', 'h2:d': 'A-Y2', 'h2:e': 'A-16', 'h1:f': 'A-X1'}
    evidence = {'h1:a': ev([1, 0, 0]), 'h2:b': ev([.99, .1, 0]), 'h1:c': ev([0, 1, 0]), 'h2:d': ev([.7, .7, 0]),
                'h2:e': ev([1, 0, 0]), 'h1:f': ev([0, 0, 1])}
    spans = {'h1:a': (0, 10), 'h2:b': (2800, 2900), 'h1:c': (0, 10), 'h2:d': (2800, 2810), 'h2:e': (2850, 2860),
             'h1:f': (20, 30)}
    moves = MM.carry_names(identity, evidence, spans)
    assert moves == {'A-Y1': 'A-16'}     # A-Y2 is between two players; A-X1 looks like nobody named
    # A player on screen at the same time as the named player's own segments in that half cannot be him.
    spans['h2:e'] = (2800, 2900)
    assert MM.carry_names(identity, evidence, spans) == {}
