"""Group D in the app: what the event models said at each moment of a match, and how well they score.

Read-only endpoints for the web page:
  GET /api/datasets/{id}/match/live-events   every event of an analysed half or whole match, with the
      model that made it (the video action spotter or the event rules), how sure it was, and how its
      player was picked or why none was; plus when the ball was seen and which models ran.
      static/event_live.js shows the events near the video's playhead.
  GET /api/model-performance                 the measured scores, gathered from the evidence folder
      (model_performance.gather). static/model_performance.js draws them.

Nothing here changes data, statistics or the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter

from . import event_review as ER
from . import storage as S

router = APIRouter()

# One touch event per ball contact and one pressure event per pressing player would bury the actions.
HIDDEN = ('touch', 'pressure')
# Event fields the live view shows, where present (rule events have no score; spotted ones no receiver).
FIELDS = ('type', 'label', 'team', 'identity', 'receiver_identity', 'outcome', 'score', 'confident',
          'credited_by', 'uncredited', 'filled', 'cross', 'length_m', 'ball_distance_m')
BALL_GAP_S = .5                     # ball sightings closer than this join into one interval


def _full_match(identifier):
    d = S.dataset_dir(identifier)
    m = S.read_json(d / 'manifest.json')
    if not str(m.get('analysis', '')).startswith('full-match'):
        raise ValueError('This source is not a full-match video analysis')
    return d, m


def _value(v):
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def live_event(raw):
    """One events.json row as the live view needs it."""
    spotted = raw.get('source') == 'action_spotter'
    out = {'id': ER.event_id(raw), 'time_s': float(raw['time_s']), 'model': 'video' if spotted else 'rules'}
    for k in FIELDS:
        v = _value(raw.get(k))
        if v is not None:
            out[k] = v
    if not spotted and out.get('identity'):
        out['credited_by'] = 'possession'           # the rules start from who had the ball
    return out


def ball_seen(d):
    """[[start, end], ...] seconds in which the ball detector saw the ball (not filled in between)."""
    path = d / 'ball.csv.gz'
    if not path.is_file():
        return None
    b = pd.read_csv(path, usecols=lambda c: c in ('time_s', 'cx', 'interpolated'))
    if 'time_s' not in b:
        return None
    seen = b.dropna(subset=['time_s'])
    if 'interpolated' in seen:
        seen = seen[~seen.interpolated.fillna(False).astype(bool)]
    if 'cx' in seen:
        seen = seen.dropna(subset=['cx'])
    times = np.sort(seen.time_s.to_numpy(float))
    spans = []
    for t in times:
        if spans and t - spans[-1][1] <= BALL_GAP_S:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    return [[round(a, 2), round(b + .08, 2)] for a, b in spans]


def models(d, m):
    """Which models ran on this match, for the strip above the live events."""
    halves = m.get('halves') or [m['id']]
    spotter = None
    for half in halves:
        try:
            info = (S.read_json(S.dataset_dir(half) / 'analysis.json', {}) or {}).get('postprocess') or {}
        except FileNotFoundError:
            continue
        if info.get('action_spotter'):
            spotter = info['action_spotter']
            break
    post = (S.read_json(d / 'analysis.json', {}) or {}).get('postprocess') or {}
    coverage = m.get('coverage') or {}
    return [
        {'id': 'detection', 'group': 'A', 'name': 'Player and ball detector', 'ran': True,
         'detail': f"ball seen in {coverage['ball_observed_share']:.0%} of live frames"
         if coverage.get('ball_observed_share') is not None else None},
        {'id': 'pitch', 'group': 'B', 'name': 'Pitch mapping', 'ran': bool(coverage.get('pitch_view_share')),
         'detail': f"pitch found in {coverage['pitch_view_share']:.0%} of the half"
         if coverage.get('pitch_view_share') is not None else None},
        {'id': 'identity', 'group': 'C', 'name': 'Player identity', 'ran': True,
         'detail': f"{post['named_players']} players named" if post.get('named_players') is not None else None},
        {'id': 'event_rules', 'group': 'D', 'name': 'Event rules', 'ran': True,
         'detail': 'passes, carries, interceptions from who had the ball'},
        {'id': 'action_spotter', 'group': 'D', 'name': 'Action spotter (T-DEED)', 'ran': spotter is not None,
         'detail': (f"{sum((spotter.get('events') or {}).values())} confident actions"
                    if isinstance(spotter.get('events'), dict) else None) if spotter else 'did not run on this match'},
    ]


def live_events(identifier):
    d, m = _full_match(identifier)
    events = [live_event(e) for e in ER.raw_events(identifier)
              if e.get('type') not in HIDDEN and e.get('time_s') is not None]
    events.sort(key=lambda e: e['time_s'])
    return {'events': events, 'ball_seen': ball_seen(d), 'models': models(d, m),
            'note': 'Video model events carry the model\'s score (0-1); low-score ones count as part of an event '
                    'in the statistics. Rule events come from who had the ball and where it went next.'}


@router.get('/api/datasets/{identifier}/match/live-events')
def live_events_api(identifier):
    return S.clean_json(live_events(identifier))


@router.get('/api/model-performance')
def model_performance_api():
    from . import model_performance as MPF
    return S.clean_json(MPF.gather(S.EVIDENCE))
