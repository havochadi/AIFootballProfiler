"""Pitch event maps and reviewed outcomes; raw model events are never rewritten."""
from __future__ import annotations

import hashlib
import json
import math
from . import storage as S

OUTCOMES = {
    'pass': ['complete', 'intercepted', 'incomplete', 'out_of_play', 'unknown'],
    'shot': ['goal', 'on_target', 'off_target', 'blocked', 'woodwork', 'unknown'],
    'tackle': ['won', 'lost', 'unknown'], 'interception': ['successful', 'unsuccessful', 'unknown'],
    'recovery': ['won', 'unknown'], 'carry': ['complete', 'incomplete', 'unknown'],
    'dribble': ['complete', 'incomplete', 'unknown'], 'clearance': ['cleared', 'unsuccessful', 'unknown'],
    'pressure': ['effective', 'ineffective', 'unknown'], 'touch': ['observed', 'unknown'],
    'take_on': ['complete', 'incomplete', 'unknown'], 'block': ['successful', 'unknown'],
    'header': ['won', 'lost', 'unknown'], 'high_pass': ['complete', 'intercepted', 'unknown'],
    'cross': ['complete', 'intercepted', 'unknown'],
}


def event_id(event):
    # A changed detection must not inherit a review of an earlier event by row index.
    signature = [event.get(k) for k in ('type', 'segment', 'team', 'time_s', 'x', 'y')]
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]


def raw_events(identifier):
    return S.read_json(S.dataset_dir(identifier) / 'events.json', {'events': []})['events']


def reviews(identifier):
    with S.db() as db:
        return {r['event_id']: json.loads(r['payload']) for r in db.execute(
            'SELECT event_id,payload FROM event_outcome_reviews WHERE dataset_id=?', (identifier,))}


def valid_point(x, y):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y)) and 0 <= x <= 105 and 0 <= y <= 68


def map_events(identifier, player=None):
    saved = reviews(identifier); out = []
    for raw in raw_events(identifier):
        kind = raw.get('type')
        if kind not in OUTCOMES or (player is not None and raw.get('identity') != player):
            continue
        if raw.get('source') == 'action_spotter' and not raw.get('confident'):
            continue                       # low-scoring spots count towards expected totals only
        key = event_id(raw); reviewed = saved.get(key)
        outcome = raw.get('outcome') or {'tackle': 'won', 'interception': 'successful', 'recovery': 'won', 'touch': 'observed',
                                         'block': 'successful'}.get(kind, 'unknown')
        if outcome not in OUTCOMES[kind]:
            outcome = 'unknown'
        # The detector currently recognises shot attempts, not goals or saves.
        source = 'unclassified' if outcome == 'unknown' else 'video_model'
        if reviewed:
            outcome = reviewed['outcome']; source = 'reviewed'
        x, y = raw.get('x'), raw.get('y'); ex, ey = raw.get('end_x'), raw.get('end_y')
        sign = raw.get('attack_sign', 0)
        mapped = valid_point(x, y)
        end_mapped = valid_point(ex, ey)
        out.append({'id': key, 'type': kind, 'time_s': raw.get('time_s'), 'identity': raw.get('identity'),
                    'outcome': outcome, 'outcome_source': source, 'review': reviewed,
                    'x': x if mapped else None, 'y': y if mapped else None,
                    'end_x': ex if end_mapped else None, 'end_y': ey if end_mapped else None,
                    'attack_sign': sign, 'progressive': bool(raw.get('progressive')),
                    'cross': bool(raw.get('cross')), 'length_m': raw.get('length_m'),
                    'probability': raw.get('probability'), 'detection_evidence': raw.get('evidence')})
    return {'events': out, 'outcomes': OUTCOMES, 'pitch': {'length': 105, 'width': 68},
            'note': 'Origins use calibrated pitch positions. Reviewed outcomes are stored separately from model statistics. '
                    'Shots are unknown until reviewed; detected tackles and interceptions describe successful possession changes only.'}


def save_review(identifier, key, outcome, reviewer, notes=''):
    event = next((e for e in raw_events(identifier) if event_id(e) == key), None)
    if event is None:
        raise FileNotFoundError('This event has changed or no longer exists. Refresh the player before reviewing it.')
    if outcome not in OUTCOMES.get(event.get('type'), []):
        raise ValueError('Choose an outcome compatible with this event type')
    if not reviewer.strip():
        raise ValueError('Enter your name to confirm the outcome')
    payload = {'outcome': outcome, 'reviewer': reviewer.strip(), 'notes': notes, 'updated': S.now()}
    with S.db() as db:
        db.execute('INSERT OR REPLACE INTO event_outcome_reviews VALUES(?,?,?)', (identifier, key, json.dumps(payload)))
        db.execute('INSERT INTO event_outcome_history(dataset_id,event_id,payload) VALUES(?,?,?)',
                   (identifier, key, json.dumps(payload)))
    return payload
