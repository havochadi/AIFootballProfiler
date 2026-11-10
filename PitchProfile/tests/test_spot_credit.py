"""Crediting spotted actions to players (action_spotting.attribute) on small synthetic scenes."""
import numpy as np
import pandas as pd

from football_profiler import action_spotting as AS
from scripts.credit_reasons import table, tally

HZ = 12.5
SAMPLE = 100                                   # the spot's sample; attribute() looks 2 samples either side
DIRECTIONS = {'status': 'inferred', 'defends_left': 'A', 'attacks_right': 'A'}     # spotter side 'left' = team A


def scene(players, ball_xy=None, ball_px=None):
    """people and ball tables over samples 95-105.

    players: {segment: (team, (x, y), box or None)}, box = (x, y, w, h) in the picture.
    ball_xy: the ball on the pitch (metres), ball_px: in the picture (pixels); None for not seen.
    """
    samples = range(SAMPLE - 5, SAMPLE + 6)
    people = pd.DataFrame([
        {'sample': s, 'time_s': s / HZ, 'segment': seg, 'team': team, 'role': 'player', 'view_shot': 0,
         'x': float(x), 'y': float(y), **dict(zip(('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'), box or (np.nan,) * 4))}
        for s in samples for seg, (team, (x, y), box) in players.items()])
    cols = ['sample', 'time_s', 'x', 'y', 'cx', 'cy', 'interpolated']
    if ball_xy is None and ball_px is None:
        return people, pd.DataFrame(columns=cols)
    bx, by = ball_xy or (np.nan, np.nan)
    cx, cy = ball_px or (np.nan, np.nan)
    ball = pd.DataFrame([{'sample': s, 'time_s': s / HZ, 'x': bx, 'y': by, 'cx': cx, 'cy': cy, 'interpolated': False}
                         for s in samples], columns=cols)
    return people, ball


def credit(label, people, ball, directions=DIRECTIONS, side='left'):
    spots = [{'label': label, 'side': side, 'frame': SAMPLE, 'time_s': SAMPLE / HZ, 'score': .9}]
    rows = AS.attribute(spots, people, ball, directions, HZ)
    assert len(rows) == 1
    return rows[0]


def test_the_ball_in_a_players_picture_box_beats_a_nearer_player_on_the_pitch():
    people, ball = scene({'A1': ('A', (50, 30), (100, 100, 40, 100)), 'A2': ('A', (50.4, 30), (400, 100, 40, 100))},
                         ball_xy=(50.5, 30), ball_px=(120, 190))
    r = credit('PASS', people, ball)
    assert (r['segment'], r['credited_by'], r['uncredited']) == ('A1', 'picture', None)


def test_without_a_picture_ball_the_nearest_player_of_the_team_within_3_m():
    people, ball = scene({'A1': ('A', (50, 30), None), 'A2': ('A', (60, 30), None), 'B1': ('B', (51, 30), None)},
                         ball_xy=(51.5, 30))
    r = credit('PASS', people, ball)
    assert (r['segment'], r['credited_by'], r['team']) == ('A1', 'pitch', 'A')       # B1 is nearer but on the other team


def test_uncredited_spots_say_why():
    team_a = {'A1': ('A', (50, 30), None)}
    far, _ = scene(team_a)
    _, far_ball = scene(team_a, ball_xy=(60, 30))
    assert credit('PASS', far, far_ball)['uncredited'] == 'ball away from players'      # 10 m from A1
    assert credit('PASS', *scene(team_a))['uncredited'] == 'no ball'
    assert credit('PASS', *scene({'B1': ('B', (50, 30), None)}, ball_xy=(50, 30)))['uncredited'] == 'team not on screen'
    unknown = {'status': 'unknown'}
    r = credit('PASS', *scene(team_a, ball_xy=(50, 30)), directions=unknown)
    assert (r['segment'], r['team'], r['uncredited']) == (None, None, 'no team')
    assert set(AS.UNCREDITED) == {'no team', 'team not on screen', 'no ball', 'ball away from players'}


def test_credit_reasons_counts_confident_spots_by_way_and_reason():
    rows = [{'type': 'pass', 'confident': True, 'credited_by': 'picture', 'uncredited': None},
            {'type': 'pass', 'confident': True, 'credited_by': None, 'uncredited': 'no ball'},
            {'type': 'pass', 'confident': False, 'credited_by': None, 'uncredited': 'no ball'},
            {'type': 'header', 'confident': True, 'credited_by': None, 'uncredited': 'team not on screen'}]
    t = table(tally(rows), AS.UNCREDITED)
    assert t['pass']['spots'] == 2 and t['pass']['credited'] == .5 and t['pass']['no ball'] == .5
    assert t['header'] == {'spots': 1, 'credited': 0.0, 'picture': 0.0, 'pitch': 0.0, 'no team': 0.0,
                           'team not on screen': 1.0, 'no ball': 0.0, 'ball away from players': 0.0}
