import numpy as np
import pandas as pd

from football_profiler import match_identity as MI


def _players(rng, k=3, per=6, dim=32):
    centres = rng.normal(size=(k, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    emb, starts, ends, who = [], [], [], []
    for p in range(k):
        for j in range(per):
            v = centres[p] + rng.normal(scale=.08, size=dim)
            emb.append(v / np.linalg.norm(v))
            # All players are on screen together in each 10-s window; one player never overlaps himself.
            starts.append(j * 10.0 + p * .1)
            ends.append(j * 10.0 + 8.0 + p * .1)
            who.append(p)
    return np.array(emb), np.array(starts), np.array(ends), np.array(who)


def test_group_segments_recovers_players_and_never_joins_simultaneous_segments():
    rng = np.random.default_rng(0)
    emb, starts, ends, who = _players(rng)
    groups = MI.group_segments(emb, np.ones(len(emb)), starts, ends, threshold=.7)
    assert sorted(sorted(who[g].tolist()) for g in groups) == [[0] * 6, [1] * 6, [2] * 6]
    for g in groups:
        for i in g:
            for j in g:
                if i != j:
                    assert not (starts[i] <= ends[j] and starts[j] <= ends[i])


def test_group_segments_respects_the_threshold():
    rng = np.random.default_rng(1)
    emb, starts, ends, who = _players(rng)
    assert len(MI.group_segments(emb, np.ones(len(emb)), starts, ends, threshold=.999)) == len(emb)


def test_naming_rules():
    legacy = {'s1': {'number': 7, 'confidence': .9, 'votes': 2.0}, 's2': {'number': 7, 'confidence': .8, 'votes': 1.0}}
    p = {'number_confidence': .5, 'number_evidence': 1.0, 'legacy_share': .6, 'legacy_votes': 1.5, 'model_alone': .85}
    confident_17, weak_7 = (17, .9, 3.0), (7, .4, 3.0)
    assert MI.name_group(['s1', 's2'], confident_17, legacy, {**p, 'naming': 'model'}) == 17
    assert MI.name_group(['s1', 's2'], confident_17, legacy, {**p, 'naming': 'legacy'}) == 7
    assert MI.name_group(['s1', 's2'], confident_17, legacy, {**p, 'naming': 'agree'}) is None
    assert MI.name_group(['s1', 's2'], weak_7, legacy, {**p, 'naming': 'agree'}) == 7
    assert MI.name_group(['s1', 's2'], confident_17, legacy, {**p, 'naming': 'either'}) == 7
    assert MI.name_group(['s3'], confident_17, legacy, {**p, 'naming': 'either'}) == 17
    assert MI.name_group(['s3'], (17, .7, 3.0), legacy, {**p, 'naming': 'either'}) is None


def test_settle_numbers_merges_non_overlapping_and_unnames_overlapping():
    ev = {'tens': np.zeros(10), 'units': np.zeros(10), 'weight': 1.0, 'emb': np.ones(4), 'crops': 1, 'keeper': 0.0}
    a = {'segments': ['a'], 'ev': ev, 'number': 9, 'confidence': .9, 'spans': [(0, 10)], 'duration': 10}
    b = {'segments': ['b'], 'ev': ev, 'number': 9, 'confidence': .7, 'spans': [(20, 30)], 'duration': 10}
    c = {'segments': ['c'], 'ev': ev, 'number': 9, 'confidence': .6, 'spans': [(5, 8)], 'duration': 3}
    out = MI._settle_numbers([a, b, c])
    nine = [q for q in out if q['number'] == 9]
    assert len(nine) == 1 and sorted(nine[0]['segments']) == ['a', 'b']
    assert [q['number'] for q in out if 'c' in q['segments']] == [None]


def test_group_segments_tolerates_brief_shared_time():
    # One player's two segments share 1 s (a tracker handover); a teammate overlaps both for long.
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    emb = np.stack([e1, e1, e2])
    starts, ends = np.array([0.0, 9.0, 0.0]), np.array([10.0, 20.0, 20.0])
    strict = MI.group_segments(emb, np.ones(3), starts, ends, threshold=.9)
    assert sorted(len(g) for g in strict) == [1, 1, 1]
    tolerant = MI.group_segments(emb, np.ones(3), starts, ends, threshold=.9, tolerance=2.0)
    assert sorted(sorted(g) for g in tolerant) == [[0, 1], [2]]


def test_motion_links_follow_the_player_across_a_gap():
    ev = lambda v: {'emb': np.array(v, float), 'crops': 1}
    ends = pd.DataFrame({'t0': [0.0, 11.0, 11.5], 't1': [10.0, 20.0, 20.0],
                         'x0': [0.0, 42.0, 60.0], 'y0': [0.0, 30.0, 30.0],
                         'x1': [40.0, 50.0, 70.0], 'y1': [30.0, 30.0, 30.0],
                         'vx': [2.0, 0.0, 0.0], 'vy': [0.0, 0.0, 0.0]}, index=['a', 'b', 'c'])
    p = {'link_gap': 5.0, 'link_radius': 3.0, 'link_speed': 2.0, 'link_ratio': 2.5, 'link_min': .5}
    evidence = {'a': ev([1, 0]), 'b': ev([1, 0]), 'c': ev([1, 0])}
    assert MI.motion_links(ends, ['a', 'b', 'c'], evidence, p) == [('a', 'b')]
    # An appearance contradiction vetoes the link.
    evidence['b'] = ev([0, 1])
    assert MI.motion_links(ends, ['a', 'b', 'c'], evidence, p) == []
    assert MI.link_chains([('a', 'b'), ('b', 'c')], ['c', 'b', 'a']) == [[2, 1, 0]]


def test_consensus_keeps_a_legacy_number_only_when_the_model_does_not_prefer_another():
    from football_profiler import identity_model as IM
    legacy = {'s1': {'number': 8, 'confidence': .9, 'votes': 3.0}}
    p = {'number_confidence': .5, 'number_evidence': 1.0, 'legacy_share': .6, 'legacy_votes': 1.5, 'model_alone': .85,
         'naming': 'consensus', 'legacy_ratio': .5}

    def ev(tens, units):
        t, u = np.full(10, -8.0), np.full(10, -8.0)
        for k, v in tens.items():
            t[k] = np.log(v)
        for k, v in units.items():
            u[k] = np.log(v)
        return {'tens': t, 'units': u, 'weight': 1.0}
    reads_18 = ev({1: .35, 0: .3}, {8: .8, 3: .2})           # model unsure, prefers 18 but gives 8 (tens 'none') .86 of that
    assert IM.probability(reads_18, 8) / IM.reading(reads_18)[1] > .5
    assert MI.name_group(['s1'], IM.reading(reads_18), legacy, p, reads_18) == 8
    rejects_8 = ev({1: .45, 0: .05, 2: .4}, {8: .8, 3: .2})  # model unsure but gives 8 a ninth of its 18
    assert MI.name_group(['s1'], IM.reading(rejects_8), legacy, p, rejects_8) is None
    assert MI.name_group(['s1'], IM.reading(rejects_8), legacy, p) == 8     # without evidence: unchanged rule


def test_jersey_aggregate_votes_legible_readings_only():
    from football_profiler import jersey as J
    scores = np.array([.9, .8, .2, .95, .7])
    text = np.array(['7', '7', '1', '17', 'x'], dtype=object)
    conf = np.array([.9, .8, .9, .5, .6], np.float32)
    r = J.aggregate(scores, text, conf)
    assert r['number'] == 7 and r['legible_crops'] == 4 and r['crops'] == 5
    assert abs(r['votes'] - 1.7) < 1e-6 and abs(r['confidence'] - 1.7 / 2.8) < 1e-6
    assert J.aggregate(np.array([.1]), np.array(['5'], dtype=object), np.array([.9], np.float32))['number'] is None


def test_vote_reading_and_votes_naming():
    from football_profiler import identity_model as IM
    votes = np.zeros(100)
    votes[[18, 8]] = [8.0, 1.0]
    ev = {'votes': votes, 'tens': np.zeros(10), 'units': np.zeros(10), 'weight': 1.0}
    assert IM.vote_reading(ev)[0] == 18 and abs(IM.vote_reading(ev)[1] - 8 / 9) < 1e-9
    assert IM.vote_reading({'votes': np.zeros(100)}) == (None, 0.0, 0.0)
    p = {'naming': 'votes', 'vote_share': .8, 'vote_agreed': .3, 'legacy_agree': .5, 'number_confidence': .5,
         'number_evidence': 1.0, 'legacy_share': .6, 'legacy_votes': 1.5, 'model_alone': .85}
    assert MI.name_group(['s'], (8, .9, 3.0), {}, p, ev) == 18            # the legible votes decide
    split = dict(ev, votes=np.where(np.arange(100) == 8, 6.0, votes))     # 18: 8 votes, 8: 6 votes
    assert MI.name_group(['s'], (8, .9, 3.0), {}, p, split) is None
    legacy = {'s': {'number': 18, 'confidence': .9, 'votes': 3.0}}
    assert MI.name_group(['s'], (8, .9, 3.0), legacy, p, split) == 18     # plurality the legacy reader shares


def test_keeper_calls_need_a_goal_and_a_plausible_share():
    from football_profiler import match_pipeline as MP
    t = pd.DataFrame({'track': ['k', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'r'],
                      'team': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'referee'],
                      'role': ['player'] * 7 + ['referee'],
                      'mean_x': [4.0, 50.0, 60.0, 40.0, 100.0, 50.0, 95.0, 50.0],
                      'observations': [10, 100, 100, 100, 100, 100, 100, 50]})
    # Team A: its keeper near a goal is accepted, a midfielder called goalkeeper is not.
    out = MP.override_keepers(t, keep={'k', 'a1', 'r'}, outfield=set())
    assert out.set_index('track').role.to_dict()['k'] == 'goalkeeper'
    assert out.set_index('track').role.to_dict()['a1'] == 'player'
    assert out.set_index('track').role.to_dict()['r'] == 'referee'
    # Team B: calls covering two thirds of its detections (a kit mistaken for a keeper's) are ignored.
    out = MP.override_keepers(t, keep={'b1', 'b3'}, outfield=set())
    assert (out[out.team.eq('B')].role == 'player').all()


def test_keeper_calls_ignored_for_a_kit_the_model_cannot_tell_apart():
    from football_profiler import match_pipeline as MP
    t = pd.DataFrame({'track': ['ka', 'a1', 'a2', 'a3', 'a4', 'kb', 'b1', 'b2', 'b3'],
                      'team': ['A'] * 5 + ['B'] * 4, 'role': ['player'] * 9,
                      'mean_x': [3.0, 50.0, 101.0, 30.0, 70.0, 102.0, 40.0, 60.0, 70.0],
                      'observations': [10, 100, 100, 100, 100, 10, 100, 100, 100]})
    out = MP.override_keepers(t, keep={'ka', 'kb'}, outfield=set(),
                              probability={'ka': .9, 'a1': 0.0, 'a2': 0.1, 'a3': 0.0, 'a4': .1, 'kb': .9, 'b1': .8,
                                           'b2': .9, 'b3': .7})
    roles = out.set_index('track').role.to_dict()
    assert roles['ka'] == 'goalkeeper'
    assert roles['kb'] == 'player'          # team B's typical player scores 0.8: its calls are ignored
    assert len(MP.keeper_tracks({'x': {'keeper': 1.8, 'crops': 2}, 'y': {'keeper': 0.0, 'crops': 1}})) == 3


def test_reviewer_names_override_the_resolver_and_get_their_own_row():
    seg = pd.DataFrame({'team': ['A', 'A', 'A'], 'n': [10, 20, 5], 'yn': [10.0, 12.0, 60.0], 'depth': [.2, .3, .9],
                        'duration_s': [4.0, 8.0, 2.0]}, index=['s1', 's2', 's3'])
    identity = {'s1': 'A-X1', 's2': 'A-X1', 's3': 'A-7'}
    rows = pd.DataFrame([{'identity': 'A-X1', 'team': 'A', 'number': None, 'named': False},
                         {'identity': 'A-7', 'team': 'A', 'number': 7, 'named': True}])
    out, rows2 = MI.apply_confirmed(identity, rows, {'s1': 'A-8', 's2': 'A-8', 'gone': 'A-9'}, seg)
    assert out == {'s1': 'A-8', 's2': 'A-8', 's3': 'A-7'}
    new = rows2.set_index('identity').loc['A-8']
    assert new.number == 8 and bool(new.named) and bool(new.confirmed) and new.observed_s == 12.0
    assert not rows2.set_index('identity').loc['A-7'].confirmed
    # Naming a player with a number already in use joins that player's row.
    out, rows3 = MI.apply_confirmed(identity, rows, {'s1': 'A-7', 's2': 'A-7'}, seg)
    assert set(out.values()) == {'A-7'} and len(rows3) == 2 and bool(rows3.set_index('identity').loc['A-7'].confirmed)


def test_goalkeeper_is_one_person_in_one_place():
    def ev(v, keeper=1):
        v = np.array(v, float)
        return {'emb': v, 'crops': 1, 'keeper': keeper, 'tens': np.zeros(10), 'units': np.zeros(10), 'weight': 1.0}
    keepers = pd.DataFrame({'start_s': [0.0, 20.0, 5.0, 40.0, 60.0, 70.0, 22.0],
                            'end_s': [15.0, 30.0, 12.0, 50.0, 65.0, 72.0, 28.0],
                            'xn': [3.0, 4.0, 3.5, 30.0, 3.0, 5.0, 60.0], 'yn': [34.0] * 7},
                           index=['k1', 'k2', 'dup', 'ref', 'nopic', 'k3', 'far'])
    keepers['duration_s'] = keepers.end_s - keepers.start_s
    evidence = {'k1': ev([1, 0, 0]), 'k2': ev([.98, .2, 0]), 'dup': ev([.9, .3, 0]),
                'ref': ev([0, 1, 0]), 'k3': ev([.8, .3, .2], 0), 'far': ev([.9, .2, .1])}
    kept, back = MI.select_keeper(keepers, evidence)
    assert sorted(kept) == ['dup', 'k1', 'k2', 'k3']   # 'dup' is a second box on him, in the same place
    assert sorted(back) == ['far', 'ref']               # someone else, or somewhere else at the same moment
    # 'nopic' (no thumbnails) joins no player.


def test_lineup_corrects_misreadings_and_names_clear_groups():
    def q(number, votes, start):
        v = np.zeros(100)
        for n, w in votes.items():
            v[n] = w
        return {'segments': [f's{start}'], 'number': number, 'confidence': .9, 'duration': 10.0,
                'spans': [(start, start + 10)], 'ev': {'votes': v, 'tens': np.zeros(10), 'units': np.zeros(10),
                                                       'weight': 1.0, 'emb': np.ones(3), 'crops': 1, 'keeper': 0}}
    players = [q(68, {68: 6, 28: 4}, 0),          # read 68, but nobody wore 68: re-read against the line-up
               q(None, {7: 9, 1: 1}, 20),         # unnamed, clearly #7 among line-up numbers
               q(None, {9: 5, 11: 5}, 40),        # unnamed, split between two numbers: stays unnamed
               q(10, {10: 9}, 60)]                # read 10, in the line-up: kept
    out = MI._apply_roster(players, {7, 9, 10, 11, 28}, {'tolerance': 0.0})
    numbers = sorted(p['number'] for p in out if p['number'] is not None)
    assert numbers == [7, 10, 28]
    # A line-up number with only a sliver of the votes is not enough.
    weak = MI._apply_roster([q(68, {68: 9, 28: 1}, 0)], {28}, {'tolerance': 0.0})
    assert weak[0]['number'] is None
