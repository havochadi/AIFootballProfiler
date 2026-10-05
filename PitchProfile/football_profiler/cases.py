"""Immutable, interval-aligned review cases for the proposal's all-position rubric."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

from . import features as F, storage as S, taxonomy as T


@lru_cache(maxsize=64)
def _digest(path, size, modified):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(dataset_id):
    directory = S.dataset_dir(dataset_id)
    pieces = []
    for name in ('manifest.json', 'tracks.csv.gz', 'aligned_events.json'):
        path = directory / name
        if path.exists():
            stat = path.stat()
            pieces.append(_digest(str(path), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256('|'.join(pieces).encode()).hexdigest()


def interval_rows(rows, definition):
    return rows[rows.player_id.eq(definition['player_id']) & rows.period.eq(definition['period'])
                & rows.time_s.ge(definition['start_s']) & rows.time_s.lt(definition['end_s'])].copy()


def interval_profile(rows, player, manifest, definition):
    start, end, period = definition['start_s'], definition['end_s'], definition['period']
    fps = float(manifest['sampling_hz'])
    # Provider presence intervals are half-open frame intervals on the source clock.
    presence = player.get('intervals', {}).get(str(period))
    if presence:
        eligible = max(0, min(math.ceil(end * fps), int(presence[1]))
                       - max(math.ceil(start * fps), int(presence[0])))
    else:
        eligible = max(0, math.ceil(end * fps) - math.ceil(start * fps))
    metadata = {**player, 'eligible_frames': eligible, 'playing_seconds': eligible / fps,
                'coverage_note': 'Observed positions in the selected period and [start, end) interval only.'}
    selected = interval_rows(rows, definition)
    # Whole-match provider rates must never leak into a shorter reviewed interval.
    result = F.profile(selected, metadata, manifest, events=None)
    return {**result, **definition, 'match_id': manifest.get('match_id', manifest['id']),
            'date': manifest.get('date'), 'taxonomy_version': T.VERSION,
            'benchmark_split': manifest.get('benchmark_split'),
            'reviewed_seconds': end - start, 'feature_events_note': 'Whole-match event aggregates excluded from interval features.'}


def create(dataset_id, player_id, period, start_s, end_s, position_group, context=''):
    T.compatible(position_group)
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise ValueError('Choose a positive period number')
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (start_s, end_s)) or not 0 <= start_s < end_s:
        raise ValueError('Use finite interval times with 0 <= start < end')
    directory = S.dataset_dir(dataset_id)
    manifest = S.read_json(directory / 'manifest.json')
    player = next((p for p in manifest['players'] if str(p['player_id']) == str(player_id)), None)
    if player is None:
        raise ValueError('Player is unavailable')
    if end_s > manifest['duration_seconds']:
        raise ValueError('The interval extends beyond the dataset timeline')
    if start_s < manifest.get('video_start_s', 0):
        raise ValueError(f"The available video starts at {manifest['video_start_s']:g} seconds; choose a later interval start")
    rows = S.load_tracks(dataset_id)
    if period not in set(rows.period):
        raise ValueError('This period is not present in the dataset')
    definition = {'dataset_id': dataset_id, 'player_id': str(player_id), 'period': period,
                  'start_s': float(start_s), 'end_s': float(end_s), 'position_group': position_group,
                  'context': context.strip()[:3000], 'rubric_version': T.VERSION}
    if manifest.get('clock') != 'period_relative' and any(
            interval_rows(rows, {**definition, 'period': other}).shape[0]
            for other in set(rows.period) - {period}):
        raise ValueError('The interval spans other periods; create separate cases per half')
    profile = interval_profile(rows, player, manifest, definition)
    if not profile['eligible_frames'] or not profile['valid_position_frames']:
        raise ValueError('No usable observed positions occur in this player interval')
    source = fingerprint(dataset_id)
    with S.db() as c:
        for old in c.execute('SELECT id,definition,fingerprint FROM interval_cases WHERE dataset_id=? AND player_id=?',
                             (dataset_id, str(player_id))):
            d = json.loads(old['definition'])
            if (old['fingerprint'] == source and d['period'] == period
                    and max(start_s, d['start_s']) < min(end_s, d['end_s'])):
                if all(d[k] == definition[k] for k in ('period', 'start_s', 'end_s', 'position_group', 'context')):
                    return get(old['id'])
                raise ValueError('This player already has an overlapping case. Use its exact interval or create a non-overlapping interval.')
        case_id = 'case-' + uuid.uuid4().hex[:16]
        profile['case_id'] = case_id
        c.execute('INSERT INTO interval_cases VALUES(?,?,?,?,?,?,?)',
                  (case_id, dataset_id, str(player_id), json.dumps(definition), json.dumps(profile), source, S.now()))
    return get(case_id)


def get(case_id):
    with S.db() as c:
        row = c.execute('SELECT * FROM interval_cases WHERE id=?', (case_id,)).fetchone()
    if row is None:
        raise FileNotFoundError('Review case does not exist')
    profile = json.loads(row['profile'])
    try:
        stale = row['fingerprint'] != fingerprint(row['dataset_id'])
    except FileNotFoundError:
        stale = True
    return {'id': case_id, **json.loads(row['definition']), 'profile': profile, 'stale': stale,
            'created': row['created'], 'labels': list(T.compatible(profile['position_group'])),
            'pilot_duration_met': profile['reviewed_seconds'] >= T.CATALOGUE['pilot']['reviewed_seconds']}


def listing(dataset_id=None, player_id=None):
    query, args = 'SELECT id FROM interval_cases WHERE 1=1', []
    for key, value in (('dataset_id', dataset_id), ('player_id', player_id)):
        if value is not None:
            query += f' AND {key}=?'
            args.append(str(value))
    with S.db() as c:
        ids = [r['id'] for r in c.execute(query + ' ORDER BY created', args)]
    return [get(identifier) for identifier in ids]


def reviews(case_id=None):
    with S.db() as c:
        rows = c.execute('SELECT * FROM interval_reviews' + (' WHERE case_id=?' if case_id else ''),
                         (case_id,) if case_id else ()).fetchall()
    return [{'case_id': r['case_id'], 'reviewer': r['reviewer'], 'updated': r['updated'],
             **json.loads(r['payload'])} for r in rows]


def save_review(case_id, reviewer, labels, evidence, sequences,
                confidence='uncertain', abstain_reason='', notes=''):
    case = get(case_id)
    if case['stale']:
        raise ValueError('Source data changed. Create a new case and review the revised evidence.')
    T.validate_labels(case['position_group'], labels)
    reviewer = reviewer.strip().casefold()
    if not reviewer or len(reviewer) > 80 or not evidence.strip():
        raise ValueError('Enter an individual reviewer ID and an evidence source')
    if confidence not in ('uncertain', 'low', 'medium', 'high'):
        raise ValueError('Choose uncertain, low, medium or high confidence')
    if all(value is None for value in labels.values()) and not abstain_reason.strip():
        raise ValueError('Explain why the evidence is insufficient')
    if not isinstance(sequences, list) or len(sequences) > 100:
        raise ValueError('Use at most 100 timestamped evidence sequences')
    cleaned, seen = [], set()
    for sequence in sequences:
        start, end = sequence.get('start_s'), sequence.get('end_s')
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in (start, end)) or not case['start_s'] <= start < end <= case['end_s']:
            raise ValueError('Each supporting sequence must be inside the reviewed interval')
        if not str(sequence.get('note', '')).strip():
            raise ValueError('Describe the behaviour or relevant opportunity in each evidence sequence')
        if (start, end) in seen:
            raise ValueError('Supporting sequences must be distinct')
        seen.add((start, end))
        cleaned.append({'start_s': start, 'end_s': end, 'note': str(sequence['note'])[:2000]})
    payload = {'labels': labels, 'evidence': evidence.strip()[:3000], 'sequences': cleaned,
               'confidence': confidence, 'abstain_reason': abstain_reason[:3000],
               'notes': notes[:3000], 'rubric_version': T.VERSION}
    stamp, raw = S.now(), json.dumps(payload)
    with S.db() as c:
        c.execute('INSERT INTO interval_review_history(case_id,reviewer,payload,created) VALUES(?,?,?,?)',
                  (case_id, reviewer, raw, stamp))
        c.execute('INSERT INTO interval_reviews VALUES(?,?,?,?) ON CONFLICT(case_id,reviewer) DO UPDATE SET payload=excluded.payload,updated=excluded.updated',
                  (case_id, reviewer, raw, stamp))
        c.execute('DELETE FROM interval_adjudications WHERE case_id=?', (case_id,))
    return consensus(case_id)


def consensus(case_id):
    case, rows = get(case_id), reviews(case_id)
    with S.db() as c:
        decisions = {r['label']: dict(r) for r in c.execute('SELECT * FROM interval_adjudications WHERE case_id=?', (case_id,))}
    result = {}
    for label in case['labels']:
        known = [r['labels'][label] for r in rows if r['labels'][label] is not None]
        value, spread, status = None, None, 'needs independent review'
        if case['stale']:
            status = 'source changed'
        elif label in decisions:
            value, status = decisions[label]['value'], 'adjudicated'
        elif len(known) >= 2:
            spread = max(known) - min(known)
            status = 'agreed' if spread <= T.AGREEMENT_TOLERANCE else 'disagreement'
            if status == 'agreed':
                value = sum(known) / len(known)
        result[label] = {'value': value, 'spread': spread, 'status': status, 'reviewer_count': len(known)}
    # Only agreed/adjudicated ratings settle a role; a visible spread on a disagreement
    # helps a human adjudicate but must never leak into primary/secondary or training.
    settled = {key: v['value'] for key, v in result.items() if v['value'] is not None}
    primary_role = max(settled, key=settled.get) if settled else None
    if primary_role is not None and settled[primary_role] < T.SECONDARY_FLOOR:
        primary_role = None
    secondary_traits = sorted((key for key, value in settled.items() if value >= T.SECONDARY_FLOOR and key != primary_role),
                              key=settled.get, reverse=True)
    return {'reviewers': len(rows), 'labels': result, 'primary_role': primary_role, 'secondary_traits': secondary_traits}


def adjudicate(case_id, label, value, reviewer, reason):
    status = consensus(case_id)
    rows = reviews(case_id)
    reviewer = reviewer.strip().casefold()
    if label not in status['labels'] or status['labels'][label]['status'] != 'disagreement':
        raise ValueError('Only an unresolved disagreement can be adjudicated')
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100
            or not reviewer or len(reviewer) > 80 or not reason.strip()):
        raise ValueError('Enter a percentage decision from 0 to 100, a third reviewer, and a supporting reason')
    if reviewer in {r['reviewer'] for r in rows}:
        raise ValueError('A different third reviewer must adjudicate')
    with S.db() as c:
        c.execute('INSERT OR REPLACE INTO interval_adjudications VALUES(?,?,?,?,?,?)',
                  (case_id, label, float(value), reviewer, reason[:3000], S.now()))
    return consensus(case_id)


def training_profiles():
    result = []
    for case in listing():
        p = case['profile']
        if case['stale'] or not case['pilot_duration_met'] or not p['direction_known'] or (p['position_coverage'] or 0) < .2 or p['observed_seconds'] < 30:
            continue
        rows = reviews(case['id'])
        if len(rows) < 2 or any(len(r['sequences']) < 3 for r in rows):
            continue
        status = consensus(case['id'])
        y = [status['labels'].get(label, {}).get('value') if T.ROLES[label]['trainable'] else None for label in T.LABELS]
        if all(value is None for value in y):
            continue
        result.append({**p, 'y': [np.nan if value is None else value / 100 for value in y]})
    return result


def summary():
    items, ready = listing(), training_profiles()
    support = []
    for label in T.LABELS:
        cases = [p for p in ready if np.isfinite(p['y'][T.LABELS.index(label)])]
        support.append({'label': label, 'known_cases': len(cases),
                        'positive_cases': sum(p['y'][T.LABELS.index(label)] == 1 for p in cases),
                        'matches': len({p['match_id'] for p in cases}),
                        'verified_players': len({p['global_id'] for p in cases if p.get('identity_verified') and p.get('global_id')})})
    return S.clean_json({'rubric_version': T.VERSION, 'cases': len(items), 'review_rows': len(reviews()),
                        'usable_training_cases': len(ready), 'match_groups': len({p['match_id'] for p in ready}),
                        'support': support, 'stale_cases': sum(p['stale'] for p in items)})


def export():
    with S.db() as c:
        history = [dict(r) for r in c.execute('SELECT * FROM interval_review_history ORDER BY id')]
        adjudications = [dict(r) for r in c.execute('SELECT * FROM interval_adjudications')]
    for row in history:
        row['payload'] = json.loads(row['payload'])
    return {'rubric': T.CATALOGUE, 'cases': listing(), 'reviews': reviews(),
            'review_history': history, 'adjudications': adjudications, 'exported': S.now()}
