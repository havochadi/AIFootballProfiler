"""Evaluate the touch/pass/carry/shot logic of football_profiler.match_events on
SoccerTrack v2 ground truth: annotated player positions (GSR), the released ball
track and the Ball Action Spotting (BAS) annotations of match 117092.

This isolates the event logic from vision errors. It needs the GSR halves and
BAS file (scripts/download_soccertrack.py) and the ball tracks ball/117092_*_ball.npz
in the same raw folder. The report goes to evidence/event_logic_soccertrack.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_events_soccertrack.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import match_events as ME
from football_profiler import soccertrack as ST
from football_profiler import storage as S

HZ = 12.5
# BAS positions are milliseconds from the start of the match in the released file:
# second-half events align with the half-2 video clock after subtracting 2699 s
# (verified by passer-to-ball distance at annotated passes; see LOCAL_VERIFICATION.md).
BAS_OFFSET_S = {1: 0.0, 2: 2699.0}
PASS_LABELS = ['PASS', 'HIGH PASS', 'CROSS']
ON_BALL = ['PASS', 'DRIVE', 'HIGH PASS', 'CROSS', 'SHOT', 'HEADER', 'THROW IN', 'FREE KICK',
           'PLAYER SUCCESSFUL TACKLE', 'BALL PLAYER BLOCK']


def load(raw: Path, half: int):
    suffix = '1st' if half == 1 else '2nd'
    tracks, info, _ = ST.parse_half(raw / 'gsr' / '117092' / f'117092_{suffix}.json', '117092', half, sampling_hz=HZ)
    info = {p['player_id']: p for p in info}
    people = tracks.rename(columns={'player_id': 'tracklet', 'frame': 'sample'}).copy()
    people['team'] = people.tracklet.map(lambda p: 'A' if info[p]['provider_team_side'] == 'left' else 'B')
    people['role'] = people.tracklet.map(lambda p: 'goalkeeper' if info[p]['role'] == 'Goalkeeper' else 'player')
    people['view_shot'] = 0
    for c in ('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'):
        people[c] = np.nan
    z = np.load(raw / 'ball' / f'117092_{suffix}_ball.npz')
    ball = pd.DataFrame({'src': z['frame'].astype(int) - 1, 'x': z['x'] + 52.5, 'y': z['y'] + 34, 'status': z['status']})
    # The released ball track holds quantised positions between updates; rebuild a
    # continuous path from the update keyframes and drop dead-ball periods (status 0).
    moved = (ball.x.diff().abs() > 1e-4) | (ball.y.diff().abs() > 1e-4)
    key = ball[moved | (ball.index == 0)]
    ball['x'] = np.interp(ball.src, key.src, key.x)
    ball['y'] = np.interp(ball.src, key.src, key.y)
    ball[['x', 'y']] = ball[['x', 'y']].rolling(5, center=True, min_periods=1).mean()
    stride = int(round(25 / HZ))
    ball = ball[(ball.status > 0) & (ball.src % stride == 0)].copy()
    ball['sample'] = ball.src // stride
    ball['time_s'] = ball.src / 25
    ball['view_shot'], ball['cx'], ball['cy'], ball['interpolated'] = 0, np.nan, np.nan, False
    mean_x = people.groupby('team').x.mean()
    directions = {'status': 'inferred', 'attacks_right': mean_x.idxmin(), 'defends_left': mean_x.idxmin()}
    bas = json.loads((raw / 'bas' / '117092' / '117092_12_class_events.json').read_text(encoding='utf-8'))['actions']
    bas = pd.DataFrame([a for a in bas if a['gameTime'].startswith(f'{half} ')])
    bas['time_s'] = bas.position.astype(float) / 1000 - BAS_OFFSET_S[half]
    key_map = {(p['provider_player_id'], p['provider_team_side']): p['player_id'] for p in info.values()}
    bas['pid'] = [key_map.get((str(a.player_id), a.team)) for a in bas.itertuples()]
    return people, ball, directions, bas


def match(pred, truth, tol=1.0, by_player=True):
    used, tp = set(), 0
    for p in pred.sort_values('time_s').itertuples():
        cand = truth[(np.abs(truth.time_s - p.time_s) <= tol) & ~truth.index.isin(used)]
        if by_player:
            cand = cand[cand.pid == p.pid]
        if len(cand):
            used.add((cand.time_s - p.time_s).abs().idxmin())
            tp += 1
    return {'predicted': int(len(pred)), 'annotated': int(len(truth)), 'matched': tp,
            'precision': tp / max(len(pred), 1), 'recall': tp / max(len(truth), 1)}


def ball_wins(bas):
    """Annotated possession wins: an on-ball action by the team that did not make the previous one.

    'tackle' when the previous action was the opponent's drive (the ball was taken off the carrier),
    'after_pass' when it was the opponent's pass or cross (interception or loose-ball recovery).
    """
    seq = bas[bas.label.isin(ON_BALL + ['OUT'])].sort_values('time_s').reset_index(drop=True)
    rows = []
    for i in range(1, len(seq)):
        a, b = seq.iloc[i - 1], seq.iloc[i]
        if b.label in ('THROW IN', 'FREE KICK', 'OUT') or a.label in ('OUT',) or b.team == a.team:
            continue
        if b.time_s - a.time_s > 6:
            continue
        kind = 'after_pass' if a.label in PASS_LABELS else 'from_carrier' if a.label == 'DRIVE' else 'other'
        rows.append({'time_s': b.time_s, 'pid': b.pid, 'team': b.team, 'kind': kind})
    return pd.DataFrame(rows, columns=['time_s', 'pid', 'team', 'kind'])


def evaluate(people, ball, directions, bas):
    ev, sp, pos, pp, _ = ME.events(people, ball, directions, HZ)
    ev['pid'] = ev.segment.map(pp.groupby('segment').tracklet.first())
    if 'opponent' in ev:
        ev['opponent_pid'] = ev.opponent.map(pp.groupby('segment').tracklet.first())
    wins = ball_wins(bas)
    tackles_gt = bas[bas.label.eq('PLAYER SUCCESSFUL TACKLE')]
    blocks_gt = bas[bas.label.eq('BALL PLAYER BLOCK')]
    our_wins = ev[ev.type.isin(['tackle', 'interception', 'recovery'])]
    our_wins = our_wins.sort_values('time_s').drop_duplicates(['pid', 'time_s'])
    passes = ev[ev.type.isin(['pass', 'clearance'])]
    truth_passes = bas[bas.label.isin(PASS_LABELS)]
    ours = passes.groupby('pid').size()
    truth = truth_passes.groupby('pid').size()
    joined = pd.concat([ours, truth], axis=1).fillna(0)
    on_ball = bas[bas.label.isin(ON_BALL)].sort_values('time_s')
    return {
        'passes_same_player_1s': match(passes, truth_passes),
        'passes_any_player_1s': match(passes, truth_passes, by_player=False),
        'carries_vs_drive': match(ev[ev.type.isin(['carry', 'dribble'])], bas[bas.label.eq('DRIVE')]),
        'shots': match(ev[ev.type.eq('shot')], bas[bas.label.eq('SHOT')]),
        'crosses': match(ev[(ev.type == 'pass') & (ev.get('cross') == True)], bas[bas.label.eq('CROSS')]),
        'tackles_same_player_2s': match(ev[ev.type.eq('tackle')], tackles_gt, tol=2.0),
        'tackles_any_player_2s': match(ev[ev.type.eq('tackle')], tackles_gt, tol=2.0, by_player=False),
        'blocks_same_player_1s': match(ev[ev.type.isin(['block', 'deflection'])], blocks_gt),
        'blocks_any_player_1s': match(ev[ev.type.isin(['block', 'deflection'])], blocks_gt, by_player=False),
        'ball_wins_same_player_2s': match(our_wins, wins, tol=2.0),
        'ball_wins_after_pass_same_player_2s': match(ev[ev.type.isin(['interception', 'recovery'])],
                                                     wins[wins.kind == 'after_pass'], tol=2.0),
        'annotated_ball_wins_by_kind': wins.kind.value_counts().to_dict(),
        'per_player_pass_counts': {'pearson': float(joined.corr().iat[0, 1]),
                                   'spearman': float(joined.corr(method='spearman').iat[0, 1]), 'players': int(len(joined))},
        'possession_changes': {'ours': int((sp.team != sp.team.shift()).sum() - 1),
                               'annotated_action_sequence': int((on_ball.team != on_ball.team.shift()).sum() - 1)},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw', type=Path, default=S.DATA / 'raw' / 'soccertrack-v2')
    args = p.parse_args()
    report = {'match': 'SoccerTrack v2 117092', 'sampling_hz': HZ,
              'note': 'Event logic on annotated positions and ball track, not vision output. Tolerance 1 s. '
                      'SoccerTrack Pass = intended ground pass; our passes include clearances and misplaced passes.',
              'halves': {}}
    for half in (1, 2):
        people, ball, directions, bas = load(args.raw, half)
        report['halves'][str(half)] = evaluate(people, ball, directions, bas)
        print(f'half {half}:', json.dumps(report['halves'][str(half)], indent=1), flush=True)
    S.write_json(S.EVIDENCE / 'event_logic_soccertrack.json', report)


if __name__ == '__main__':
    main()
