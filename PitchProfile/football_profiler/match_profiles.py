"""Percentile profiles: each player's statistics ranked against players in the same position group.

Rates are per 90 minutes of the player's own identified screen time, so players identified for
more of the half are not favoured. Peers are every outfield appearance in the analysed full-match
datasets with at least MIN_VISIBLE_S identified, in the same position group (suggested from average
position, or the one a reviewer set); groups with fewer than MIN_PEERS appearances are
compared with all outfield players instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import storage as S

MIN_VISIBLE_S = 600
MIN_PEERS = 8
# (section, key, label, source): source 'rate' = per90_visible, 'ball' = on_ball, 'physical',
# 'movement'. lower_better marks statistics where a low value is the better one.
PROFILE = [
    ('Attacking', 'shots', 'Shots', 'rate'),
    ('Attacking', 'touches_in_box', 'Touches in the box', 'rate'),
    ('Attacking', 'receptions_in_box', 'Passes received in the box', 'rate'),
    ('Attacking', 'box_runs', 'Runs into the box', 'rate'),
    ('Attacking', 'runs_in_behind', 'Runs in behind the defence', 'rate'),
    ('Attacking', 'receptions_behind_line', 'Passes received in behind', 'rate'),
    ('Passing', 'passes', 'Passes', 'rate'),
    ('Passing', 'pass_completion', 'Pass completion', 'ball'),
    ('Passing', 'progressive_passes', 'Progressive passes', 'rate'),
    ('Passing', 'passes_into_final_third', 'Passes into the final third', 'rate'),
    ('Passing', 'passes_into_box', 'Passes into the box', 'rate'),
    ('Passing', 'passes_behind_line', 'Balls played in behind', 'rate'),
    ('Passing', 'key_passes', 'Key passes', 'rate'),
    ('Passing', 'crosses', 'Crosses', 'rate'),
    ('Passing', 'switches', 'Switches of play', 'rate'),
    ('Passing', 'lofted_passes', 'Lofted passes', 'rate'),
    ('Carrying', 'receptions', 'Passes received', 'rate'),
    ('Carrying', 'carries', 'Carries', 'rate'),
    ('Carrying', 'progressive_carries', 'Progressive carries', 'rate'),
    ('Carrying', 'carries_into_final_third', 'Carries into the final third', 'rate'),
    ('Carrying', 'take_ons', 'Take-ons attempted', 'rate'),
    ('Carrying', 'take_on_success', 'Take-on success', 'ball'),
    ('Carrying', 'turnovers', 'Turnovers', 'rate', True),
    ('Defending', 'tackles', 'Tackles won', 'rate'),
    ('Defending', 'interceptions', 'Interceptions', 'rate'),
    ('Defending', 'recoveries', 'Ball recoveries', 'rate'),
    ('Defending', 'blocks', 'Blocks', 'rate'),
    ('Defending', 'headers', 'Headers', 'rate'),
    ('Defending', 'pressures', 'Pressures', 'rate'),
    ('Defending', 'pressing_runs', 'Pressing runs', 'rate'),
    ('Defending', 'recovery_runs', 'Recovery runs', 'rate'),
    ('Movement', 'high_intensity_runs', 'High-intensity runs', 'rate'),
    ('Movement', 'distance_per_min_m', 'Distance per minute', 'physical'),
    ('Movement', 'top_speed_kmh', 'Top speed', 'physical'),
    ('Movement', 'mean_x_vs_team_m', 'Height vs team-mates (m)', 'movement'),
    ('Movement', 'mean_width_m', 'Width from centre line (m)', 'movement'),
    ('Movement', 'between_lines_share', 'Time between the lines', 'movement'),
]


def value(player, key, source):
    part = {'rate': player.get('per90_visible'), 'ball': player.get('on_ball'), 'physical': player.get('physical'),
            'movement': player.get('movement')}[source] or {}
    v = part.get(key)
    return float(v) if v is not None and np.isfinite(v) else None


_cache = {'key': None, 'table': None}


def peer_table():
    """One row per eligible outfield appearance: dataset, identity, group and every PROFILE value."""
    datasets = [m for m in S.datasets() if str(m.get('analysis', '')).startswith('full-match')]
    stamps = []
    for m in datasets:
        path = S.dataset_dir(m['id']) / 'match_stats.json'
        stamps.append((m['id'], path.stat().st_mtime_ns if path.is_file() else 0))
    key = tuple(stamps)
    if _cache['key'] == key:
        return _cache['table']
    rows = []
    for m in datasets:
        groups = {p['player_id']: p.get('position_group') for p in m['players']}
        for p in S.read_json(S.dataset_dir(m['id']) / 'match_stats.json', {}).get('players', []):
            if p.get('role') == 'goalkeeper' or p.get('visible_seconds', 0) < MIN_VISIBLE_S:
                continue
            group = groups.get(p['identity'])
            rows.append({'dataset_id': m['id'], 'identity': p['identity'], 'group': group,
                         **{spec[1]: value(p, spec[1], spec[3]) for spec in PROFILE}})
    table = pd.DataFrame(rows)
    _cache.update(key=key, table=table)
    return table


def profile(dataset_id, player, group):
    """Sections of {label, value, percentile} for one player, ranked against peers."""
    table = peer_table()
    if table.empty or player.get('role') == 'goalkeeper':
        return None
    peers = table[table.group == group] if group else table.iloc[:0]
    basis = 'same position group'
    if len(peers) < MIN_PEERS:
        peers, basis = table, 'all outfield players'
    sections = {}
    for spec in PROFILE:
        section, key, label, source = spec[:4]
        lower_better = len(spec) > 4 and spec[4]
        v = value(player, key, source)
        col = peers[key].dropna() if key in peers else pd.Series(dtype=float)
        pct = None
        if v is not None and len(col):
            # Share of peers strictly below: a player with none of an action most peers also lack
            # ranks at 0, not in the middle, so rare actions are not flattered.
            pct = float((col < v).sum() / len(col) * 100)
        sections.setdefault(section, []).append({'key': key, 'label': label, 'value': v, 'percentile': pct,
                                                 'lower_better': bool(lower_better)})
    return {'peer_basis': basis, 'peer_group': group if basis == 'same position group' else None,
            'peers': int(len(peers)), 'eligible': player.get('visible_seconds', 0) >= MIN_VISIBLE_S,
            'sections': [{'name': k, 'stats': v} for k, v in sections.items()],
            'note': 'Percentile among players with at least 10 identified minutes in a half; rates per 90 '
                    'minutes of identified screen time.'}
