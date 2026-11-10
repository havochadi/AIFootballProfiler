"""Event rules (match_events) on small synthetic possessions."""
import re

import numpy as np
import pandas as pd

from football_profiler import match_events as ME

HZ = 12.5
who = lambda s: re.sub(r'^seg-|-v\d+$', '', s) if isinstance(s, str) else s


def duel():
    """B1 takes the ball off A1, who stands 1.5 m away: a quick, close change of possession."""
    players = {'A1': ('A', (40, 34)), 'B1': ('B', (41.5, 34))}
    ball = [(s, 40.4, 34.0) for s in range(10)] + [(s, 41.9, 34.0) for s in range(10, 24)]
    people = [{'tracklet': f'{name}-v0', 'track': name, 'sample': s, 'time_s': s / HZ, 'view_shot': 0, 'team': team,
               'role': 'player', 'x': float(x), 'y': float(y), 'bbox_x': np.nan, 'bbox_y': np.nan,
               'bbox_w': np.nan, 'bbox_h': np.nan, 'conf': .9}
              for s in range(24) for name, (team, (x, y)) in players.items()]
    b = pd.DataFrame(ball, columns=['sample', 'x', 'y'])
    b['time_s'], b['view_shot'] = b['sample'] / HZ, 0
    b['cx'], b['cy'], b['interpolated'] = np.nan, np.nan, False
    directions = {'status': 'inferred', 'attacks_right': 'A', 'defends_left': 'A'}
    return pd.DataFrame(people), b, directions


def test_duels_give_no_rule_tackles_unless_measuring_the_rule():
    people, ball, directions = duel()
    ev = ME.events(people, ball, directions, HZ)[0]
    # Tackles come from the video spotter only, also when it did not run.
    assert not ev.type.isin(['tackle', 'dispossessed']).any()
    assert not ev.type.eq('pass').any()                          # a duel is still not an intercepted pass
    assert [who(s) for s in ev[ev.type == 'recovery'].segment] == ['B1']
    measured = ME.events(people, ball, directions, HZ, tackle_rule=True)[0]     # evaluate_events_soccertrack.py
    assert [who(s) for s in measured[measured.type == 'tackle'].segment] == ['B1']
    assert [who(s) for s in measured[measured.type == 'dispossessed'].segment] == ['A1']
