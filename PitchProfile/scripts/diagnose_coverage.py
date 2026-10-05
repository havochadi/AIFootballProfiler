"""Measure how much of each analysed half the statistics actually cover.

For every analysed SoccerNet half in the working set this reports: calibrated
live time, ball visibility, how many players are detected per frame, how much
player time is attributed to an identity, tracklet fragmentation, and event
counts per team. As a reference for a fully observed match, the same event
types are counted in the SoccerTrack v2 117092 ground-truth action annotations
(all 22 players tracked from a panoramic camera, every action labelled).

Output: evidence/coverage_diagnostics.json and a printed summary.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\diagnose_coverage.py [--label baseline]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import match_pipeline as MP
from football_profiler import soccernet as SN
from football_profiler import storage as S

EVENTS = ['touch', 'pass', 'carry', 'take_on', 'dribble', 'shot', 'tackle', 'interception', 'recovery', 'clearance',
          'pressure', 'dispossessed', 'deflection', 'block', 'header', 'cross', 'high_pass', 'throw_in']
REFERENCE_CLASSES = ['PASS', 'HIGH PASS', 'CROSS', 'DRIVE', 'SHOT', 'HEADER', 'PLAYER SUCCESSFUL TACKLE',
                     'BALL PLAYER BLOCK']


def half_report(identifier):
    d = S.dataset_dir(identifier)
    frames = pd.read_csv(d / 'raw_frames.csv.gz', usecols=['sample', 'time_s'])
    ball = pd.read_csv(d / 'ball.csv.gz')
    tracks = S.load_tracks(identifier)
    raw = pd.read_csv(d / 'raw_people.csv.gz', usecols=['sample', 'track', 'cls'], dtype={'track': str})
    stats = S.read_json(d / 'match_stats.json', {})
    events = pd.DataFrame(S.read_json(d / 'events.json', {'events': []})['events'])
    hz = S.read_json(d / 'analysis.json')['detection']['sample_hz']
    live = float(stats.get('live_seconds') or 0)
    live_samples = live * hz
    observed_ball = int((~ball.interpolated.astype(bool)).sum()) if len(ball) else 0
    players = raw[raw.cls == 2]
    per_frame = players.groupby('sample').size()
    track_len = players.groupby('track').size() / hz
    identified = tracks[np.isfinite(tracks.x)]
    report = {
        'half': identifier, 'live_minutes': live / 60, 'samples': int(len(frames)),
        'ball_observed_share_of_live': observed_ball / max(live_samples, 1),
        'outfield_detections_per_frame_median': float(per_frame.median()) if len(per_frame) else 0.0,
        'tracklets': int(players.track.nunique()),
        'tracklet_median_s': float(track_len.median()) if len(track_len) else 0.0,
        'tracklet_share_over_10s': float((track_len >= 10).mean()) if len(track_len) else 0.0,
        'identified_share_of_outfield_detections': float(len(identified) / max(len(players), 1)),
        'identities': int(len(stats.get('players', []))),
        'events_per_team': {},
    }
    if len(events):
        # Video-spotter events count as their scores (expected counts), like the player statistics.
        spotted = events.get('source', pd.Series(index=events.index, dtype=object)).eq('action_spotter')
        weight = np.where(spotted & events.type.ne('shot'), events.get('score', pd.Series(1.0, index=events.index)).fillna(0), 1.0)
        events = events.assign(weight=weight)
        for team in ('A', 'B'):
            mine = events[events.team == team]
            report['events_per_team'][team] = {k: round(float(mine.weight[mine.type == k].sum()), 1)
                                               for k in EVENTS if (events.type == k).any()}
        attributed = events[events.get('identity').notna()] if 'identity' in events else events.iloc[:0]
        report['events_attributed_to_a_player_share'] = float(len(attributed) / len(events))
    rows = []
    for p in stats.get('players', []):
        if p.get('visible_seconds', 0) >= 600 and p.get('role') != 'goalkeeper':
            rows.append({k: (p.get('per90_visible') or {}).get(k) for k in
                         ('passes', 'carries', 'take_ons', 'dribbles', 'tackles', 'interceptions', 'recoveries',
                          'pressures', 'shots', 'blocks', 'headers', 'receptions', 'runs_in_behind', 'pressing_runs')})
    if rows:
        frame = pd.DataFrame(rows)
        report['players_visible_10min'] = int(len(frame))
        report['per90_on_screen_median'] = {k: float(frame[k].median()) for k in frame}
        report['share_of_players_with_zero'] = {k: float((frame[k].fillna(0) == 0).mean()) for k in frame}
    return report


def reference_counts():
    """Actions per team per half in the fully annotated SoccerTrack v2 match."""
    path = S.DATA / 'raw' / 'soccertrack-v2' / 'bas' / '117092' / '117092_12_class_events.json'
    if not path.is_file():
        return None
    actions = pd.DataFrame(json.loads(path.read_text(encoding='utf-8'))['actions'])
    actions['half'] = actions.gameTime.str[0]
    counts = actions[actions.label.isin(REFERENCE_CLASSES)].groupby(['half', 'team', 'label']).size()
    per_team_half = counts.groupby('label').mean()
    return {'source': 'SoccerTrack v2 117092 ground-truth ball actions (every action of all 22 players)',
            'mean_per_team_per_half': {k: float(v) for k, v in per_team_half.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--label', default='baseline', help='name for this snapshot in the evidence file')
    args = p.parse_args()
    halves = []
    for item in SN.library():
        identifier = MP.soccernet_identifier(item['game'], item['half'])
        if (S.dataset_dir(identifier) / 'match_stats.json').is_file():
            halves.append(half_report(identifier))
            print(identifier, 'done', flush=True)
    df = pd.DataFrame(halves)
    summary = {k: float(df[k].median()) for k in
               ('live_minutes', 'ball_observed_share_of_live', 'outfield_detections_per_frame_median',
                'tracklet_median_s', 'tracklet_share_over_10s', 'identified_share_of_outfield_detections')}
    team_events = pd.DataFrame([dict(v) for h in halves for v in h['events_per_team'].values()]).fillna(0)
    summary['events_per_team_per_half_median'] = {k: float(team_events[k].median()) for k in team_events}
    zero = pd.DataFrame([h['share_of_players_with_zero'] for h in halves if 'share_of_players_with_zero' in h])
    summary['share_of_players_with_zero_mean'] = {k: float(zero[k].mean()) for k in zero}
    per90 = pd.DataFrame([h['per90_on_screen_median'] for h in halves if 'per90_on_screen_median' in h])
    summary['per90_on_screen_median'] = {k: float(per90[k].median()) for k in per90}
    out = S.EVIDENCE / 'coverage_diagnostics.json'
    saved = S.read_json(out, {}) or {}
    saved[args.label] = {'created': S.now(), 'halves': halves, 'summary': summary, 'reference': reference_counts()}
    S.write_json(out, saved)
    print(json.dumps({'summary': summary, 'reference': saved[args.label]['reference']}, indent=1))


if __name__ == '__main__':
    main()
