"""Crediting spotted actions to players (action_spotting.attribute) on small synthetic scenes."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from football_profiler import action_spotting as AS
from scripts.credit_reasons import table, tally

ROOT = Path(__file__).resolve().parents[1]
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


def test_throw_in_goes_to_the_one_player_of_the_team_at_the_touchline(monkeypatch):
    # The ball held overhead is not on the ball path: nobody can be credited from it.
    players = {'A1': ('A', (40, -0.6), None), 'A2': ('A', (45, 8), None), 'B1': ('B', (42, 1), None)}
    people, ball = scene(players)
    monkeypatch.setattr(AS, 'THROW_IN_LINE', False)
    assert credit('THROW IN', people, ball)['uncredited'] == 'no ball'                  # the pipeline as it is
    monkeypatch.setattr(AS, 'THROW_IN_LINE', True)
    r = credit('THROW IN', people, ball)
    assert (r['segment'], r['credited_by'], r['uncredited']) == ('A1', 'touchline', None)
    assert (r['x'], r['y']) == (40.0, -0.6)                                           # the thrower's position
    assert credit('PASS', people, ball)['segment'] is None                            # throw-ins only


def test_throw_in_rule_needs_exactly_one_player_of_the_team_at_a_touchline(monkeypatch):
    monkeypatch.setattr(AS, 'THROW_IN_LINE', True)
    two = {'A1': ('A', (40, 67.5), None), 'A2': ('A', (20, 0.4), None)}                 # one at each touchline
    assert credit('THROW IN', *scene(two))['uncredited'] == 'no ball'
    inside = {'A1': ('A', (40, 2.0), None)}                                             # 2 m in: not taking a throw
    assert credit('THROW IN', *scene(inside))['segment'] is None
    far_out = {'A1': ('A', (40, -3.5), None)}                                           # off the pitch (bench, warm-up)
    assert credit('THROW IN', *scene(far_out))['segment'] is None


def test_throw_in_rule_does_not_replace_a_credit_from_the_ball(monkeypatch):
    monkeypatch.setattr(AS, 'THROW_IN_LINE', True)
    people, ball = scene({'A1': ('A', (40, -0.5), None), 'A2': ('A', (44, 3), None)}, ball_xy=(44.5, 3))
    r = credit('THROW IN', people, ball)
    assert (r['segment'], r['credited_by']) == ('A2', 'pitch')


def test_throw_in_rule_is_off_unless_the_variable_switches_it_on():
    code = 'from football_profiler import action_spotting as AS; print(AS.THROW_IN_LINE)'
    env = {k: v for k, v in os.environ.items() if k != 'PITCHPROFILE_THROWIN_LINE'}
    read = lambda e: subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=e, capture_output=True,
                                    text=True, check=True).stdout.strip()
    assert read(env) == 'False'
    assert read({**env, 'PITCHPROFILE_THROWIN_LINE': '1'}) == 'True'


def test_credit_reasons_counts_confident_spots_by_way_and_reason():
    rows = [{'type': 'pass', 'confident': True, 'credited_by': 'picture', 'uncredited': None},
            {'type': 'pass', 'confident': True, 'credited_by': None, 'uncredited': 'no ball'},
            {'type': 'pass', 'confident': False, 'credited_by': None, 'uncredited': 'no ball'},
            {'type': 'throw_in', 'confident': True, 'credited_by': 'touchline', 'uncredited': None}]
    t = table(tally(rows), AS.UNCREDITED)
    assert t['pass']['spots'] == 2 and t['pass']['credited'] == .5 and t['pass']['no ball'] == .5
    assert t['throw_in']['touchline'] == 1.0


def test_throw_in_team_from_the_last_touch_is_off_unless_switched_on(monkeypatch):
    # Team A (left side) last touched the ball 12 s before; the spotter wrongly gives A the throw-in.
    spots = [{'label': 'PASS', 'side': 'left', 'frame': 100, 'time_s': 8.0, 'score': .9},
             {'label': 'THROW IN', 'side': 'left', 'frame': 250, 'time_s': 20.0, 'score': .9}]
    people, ball = scene({'A1': ('A', (50, 30), None), 'B1': ('B', (52, 30), None)}, ball_xy=(50.5, 30))
    team = lambda: next(r['team'] for r in AS.attribute(spots, people, ball, DIRECTIONS, HZ) if r['type'] == 'throw_in')
    monkeypatch.setattr(AS, 'THROW_IN_TEAM', False)
    assert team() == 'A'                                                   # the spotter's own side
    monkeypatch.setattr(AS, 'THROW_IN_TEAM', True)
    assert team() == 'B'                                                   # the team that did not touch it last
    monkeypatch.setattr(AS, 'THROW_IN_TOUCH_S', 10.0)
    assert team() == 'A'                                                   # no touch within 10 s: the spotter's side
    assert AS.won_ball_side(spots, spots[1]) is None                       # tackles and blocks keep their 4 s


def test_throw_in_team_rule_is_off_unless_the_variable_switches_it_on():
    code = 'from football_profiler import action_spotting as AS; print(AS.THROW_IN_TEAM)'
    env = {k: v for k, v in os.environ.items() if k != 'PITCHPROFILE_THROWIN_TEAM'}
    read = lambda e: subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=e, capture_output=True,
                                    text=True, check=True).stdout.strip()
    assert read(env) == 'False'
    assert read({**env, 'PITCHPROFILE_THROWIN_TEAM': '1'}) == 'True'
