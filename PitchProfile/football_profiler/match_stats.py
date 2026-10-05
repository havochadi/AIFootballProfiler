"""Full-match analysis, stage 5: per-player and per-team statistics from video evidence.

Counts cover what the broadcast showed of each identified player. Three rate
bases are reported:

- per90: per 90 minutes of the team's observed live play (calibrated pitch
  view), one denominator for every player of a team;
- per90_visible: per 90 minutes of the player's own identified screen time,
  which does not depend on how much of the player could be identified;
- per100_touches: on-ball actions per 100 touches, the composition of a
  player's play (style ratios), independent of time and coverage.

Physical measures use only the player's own visible time. Off-ball runs and
positional context come from match_movement; tackles, blocks, headers, lofted
passes and set pieces from the video action spotter (action_spotting) when its
spots exist for the half.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fieldcal as FC

L, W = FC.PITCH_LENGTH, FC.PITCH_WIDTH
HEATMAP = (20, 32)                 # rows across the width, columns along the length
SPEED_ZONES = [('walk', 0, 7.2), ('jog', 7.2, 14.4), ('run', 14.4, 19.8), ('high_speed', 19.8, 25.2),
               ('sprint', 25.2, np.inf)]
SPRINT_KMH, SPRINT_MIN_S = 25.2, 1.0
MAX_SPEED_MS = 11.0                # faster steps are tracking or calibration errors
SMOOTH_S = 1.0                     # seconds of centred position smoothing
STEP_S = .4                        # speed baseline
KEY_PASS_S = 5.0

FINAL_THIRD_X = 2 * L / 3
BOX_X, BOX_Y = L - 16.5, (13.84, 54.16)
SHORT_PASS_M, LONG_PASS_M = 15.0, 30.0
SWITCH_LATERAL_M = 30.0
BEHIND_LINE_M = 1.0                # receiver this far beyond the last opposing outfielder at reception

COUNTED = ['touches', 'passes', 'passes_completed', 'progressive_passes', 'long_passes', 'crosses', 'key_passes',
           'carries', 'progressive_carries', 'dribbles', 'shots', 'tackles', 'interceptions', 'recoveries',
           'clearances', 'pressures', 'dispossessed', 'take_ons',
           'receptions', 'progressive_receptions', 'receptions_final_third', 'receptions_in_box', 'receptions_behind_line',
           'touches_final_third', 'touches_in_box', 'passes_into_final_third', 'passes_into_box', 'switches',
           'passes_behind_line', 'carries_into_final_third', 'carries_into_box', 'turnovers',
           'blocks', 'headers', 'lofted_passes', 'throw_ins',
           'high_intensity_runs', 'runs_in_behind', 'box_runs', 'forward_runs', 'overlaps', 'underlaps',
           'pressing_runs', 'recovery_runs']
COMPOSITION = ['passes', 'progressive_passes', 'long_passes', 'crosses', 'key_passes', 'carries',
               'progressive_carries', 'dribbles', 'shots', 'dispossessed', 'passes_into_final_third',
               'passes_into_box', 'switches', 'passes_behind_line', 'carries_into_final_third', 'turnovers',
               'lofted_passes', 'headers']
SPOTTED = {'tackle': 'tackles', 'block': 'blocks', 'header': 'headers', 'high_pass': 'lofted_passes',
           'throw_in': 'throw_ins', 'cross': 'crosses'}
MIN_RATE_VISIBLE_S = 60.0          # own-time rates need at least a minute on screen


def attack_normalised(q, sign):
    x, y = q.x.to_numpy(float), q.y.to_numpy(float)
    return (x, y) if sign >= 0 else (L - x, W - y)


def physical(rows, sample_hz):
    """Distance, speed zones, sprints and top speed from smoothed visible positions.

    Box and calibration jitter of a few decimetres per frame would read as
    running at 12.5 samples/s, so positions are averaged over about a second
    and speed is measured over STEP_S baselines.
    """
    zones = {name: 0.0 for name, *_ in SPEED_ZONES}
    distance = seconds = 0.0
    speeds_all, sprints = [], 0
    window = max(3, int(round(SMOOTH_S * sample_hz)) | 1)
    stride = max(1, int(round(STEP_S * sample_hz)))
    for _, seg in rows.groupby('segment'):
        s = seg.sort_values('sample')
        s = s[np.isfinite(s.x)]
        if len(s) < window // 2 + stride:
            continue
        xy = s[['x', 'y']].rolling(window, center=True, min_periods=window // 2).mean()
        keep = xy.notna().all(axis=1).to_numpy()
        samples = s['sample'].to_numpy()[keep][::stride]
        xy = xy.to_numpy()[keep][::stride]
        if len(xy) < 2:
            continue
        gap = np.diff(samples)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dt = gap / sample_hz
        speed = step / np.maximum(dt, 1e-9)
        ok = (gap == stride) & (speed <= MAX_SPEED_MS)
        distance += float(step[ok].sum())
        seconds += float(dt[ok].sum())
        kmh = speed * 3.6
        for name, lo, hi in SPEED_ZONES:
            zones[name] += float(dt[ok & (kmh >= lo) & (kmh < hi)].sum())
        speeds_all.append(kmh[ok])
        fast = ok & (kmh >= SPRINT_KMH)
        run = 0
        for f in fast:
            run = run + 1 if f else 0
            if run == int(np.ceil(SPRINT_MIN_S / STEP_S)):
                sprints += 1
    speeds = np.concatenate(speeds_all) if speeds_all else np.array([])
    return {'distance_m': distance, 'moving_seconds': seconds,
            'distance_per_min_m': distance / seconds * 60 if seconds else None,
            # 98th percentile resists single-step spikes that survive smoothing.
            'top_speed_kmh': float(np.percentile(speeds, 98)) if len(speeds) >= 10 else None,
            'sprints': sprints, 'zone_seconds': zones}


def positional(rows, sign):
    q = rows[np.isfinite(rows.x) & rows.x.between(-2, L + 2) & rows.y.between(-2, W + 2)]
    if q.empty:
        return {'heatmap': np.zeros(HEATMAP).tolist(), 'mean_x': None, 'mean_y': None}
    x, y = attack_normalised(q, sign)
    h = np.histogram2d(y, x, bins=HEATMAP, range=((0, W), (0, L)))[0]
    h = (h / h.sum()) if h.sum() else h
    box = (x >= L - 16.5) & (y >= 13.84) & (y <= 54.16)
    return {'heatmap': h.round(5).tolist(), 'mean_x': float(x.mean()), 'mean_y': float(y.mean()),
            'spread_x': float(x.std()), 'spread_y': float(y.std()),
            'defensive_third': float(np.mean(x < L / 3)), 'middle_third': float(np.mean((x >= L / 3) & (x < 2 * L / 3))),
            'attacking_third': float(np.mean(x >= 2 * L / 3)), 'left_lane': float(np.mean(y < W / 3)),
            'central_lane': float(np.mean((y >= W / 3) & (y <= 2 * W / 3))), 'right_lane': float(np.mean(y > 2 * W / 3)),
            'box_share': float(np.mean(box))}


def _flag(frame, column):
    return int(frame[column].eq(True).sum()) if column in frame else 0


def _norm(frame, xcol='x', ycol='y'):
    """Attack-normalised coordinates of event rows (their own attack_sign)."""
    if not len(frame) or xcol not in frame:
        return np.array([]), np.array([])
    sign = frame.attack_sign.fillna(0).to_numpy()
    x, y = frame[xcol].astype(float).to_numpy(), frame[ycol].astype(float).to_numpy()
    return np.where(sign < 0, L - x, x), np.where(sign < 0, W - y, y)


def _in_box(x, y):
    return (x >= BOX_X) & (y >= BOX_Y[0]) & (y <= BOX_Y[1])


def on_ball(ev, spells, spotted=False):
    """Counts, time on ball and pass qualities for one identity's events.

    With spotted=True, tackles, crosses and the other SPOTTED statistics come from the video action
    spotter instead of the tracking rules: the sum of its scores for the player's spots (expected
    counts, see action_spotting.attribute).
    """
    t = ev.type
    source = ev.source if 'source' in ev else pd.Series('rules', index=ev.index)
    video = source.eq('action_spotter')
    score = ev.score.fillna(0).astype(float) if 'score' in ev else pd.Series(0.0, index=ev.index)
    expected = lambda kind: round(float(score[t.eq(kind) & video].sum()), 2)
    rules = ~video
    passes = ev[t.eq('pass') & rules]
    # Passes seen only by the video spotter (match_pipeline.fill_from_spotter) count as passes, but
    # their outcome, length and direction are unknown, so every pass quality uses tracked passes.
    video_passes = int((t.eq('pass') & video).sum())
    completed = passes[passes.outcome.eq('complete')] if 'outcome' in passes else passes.iloc[:0]
    carries = ev[t.eq('carry')]
    # Spotted shots replace the shot model's when the spotter ran (match_pipeline.merge_spotted).
    shots = ev[t.eq('shot')]
    touches = ev[t.eq('touch')]
    tx, ty = _norm(touches)
    px, py = _norm(completed)
    pex, pey = _norm(completed, 'end_x', 'end_y')
    cx, cy = _norm(carries)
    cex, cey = _norm(carries, 'end_x', 'end_y')
    lateral = np.abs(pey - py) if len(completed) else np.array([])
    length = passes.length_m.astype(float).to_numpy() if len(passes) else np.array([])
    forward = passes.forward_m.astype(float).to_numpy() if len(passes) else np.array([])
    out = {
        'touches': int(t.eq('touch').sum()),
        'time_on_ball_s': float(spells.duration_s.sum()) if len(spells) else 0.0,
        'passes': int(len(passes)) + video_passes, 'passes_completed': int(len(completed)),
        'passes_seen_by_video_only': video_passes,
        'pass_completion': float(len(completed) / len(passes)) if len(passes) else None,
        'progressive_passes': _flag(passes, 'progressive'),
        'long_passes': _flag(passes, 'long'),
        'crosses': _flag(passes, 'cross'),
        'key_passes': _flag(ev, 'key_pass'),
        'mean_pass_length_m': float(passes.length_m.mean()) if len(passes) else None,
        'forward_pass_share': float((passes.forward_m > 0).mean()) if len(passes) else None,
        'carries': int(len(carries)),
        'progressive_carries': _flag(carries, 'progressive'),
        'carry_distance_m': float(carries.length_m.sum()) if len(carries) else 0.0,
        'dribbles': int(t.eq('dribble').sum()),
        'take_ons': int(t.eq('take_on').sum()),
        'take_on_success': float(t.eq('dribble').sum() / t.eq('take_on').sum()) if t.eq('take_on').any() else None,
        'shots': int(len(shots)),
        'mean_shot_distance_m': float(shots.distance_to_goal_m.mean()) if len(shots) else None,
        'tackles': expected('tackle') if spotted else int((t.eq('tackle') & rules).sum()),
        'interceptions': int(t.eq('interception').sum()),
        'recoveries': int(t.eq('recovery').sum()), 'clearances': int(t.eq('clearance').sum()),
        'pressures': int(t.eq('pressure').sum()), 'dispossessed': int((t.eq('dispossessed') & rules).sum()),
        'touches_final_third': int((tx >= FINAL_THIRD_X).sum()), 'touches_in_box': int(_in_box(tx, ty).sum()),
        'passes_into_final_third': int(((px < FINAL_THIRD_X) & (pex >= FINAL_THIRD_X)).sum()),
        'passes_into_box': int((~_in_box(px, py) & _in_box(pex, pey)).sum()),
        'switches': int((lateral >= SWITCH_LATERAL_M).sum()),
        'passes_behind_line': _flag(completed, 'behind_line'),
        'carries_into_final_third': int(((cx < FINAL_THIRD_X) & (cex >= FINAL_THIRD_X)).sum()),
        'carries_into_box': int((~_in_box(cx, cy) & _in_box(cex, cey)).sum()),
        'turnovers': int((t.eq('dispossessed') & rules).sum()) + int(len(passes) - len(completed)),
        'short_pass_share': float((length < SHORT_PASS_M).mean()) if len(length) else None,
        'long_pass_share': float((length >= LONG_PASS_M).mean()) if len(length) else None,
        'backward_pass_share': float((forward <= -2).mean()) if len(forward) else None,
        'sideways_pass_share': float((np.abs(forward) < 2).mean()) if len(forward) else None,
    }
    for kind, key in SPOTTED.items():
        if key == 'tackles':
            continue
        if key == 'crosses':
            if spotted:
                out['crosses'] = expected('cross')
            continue
        out[key] = expected(kind)
    return out


def receptions(ev, identity):
    """Completed passes received, per identity, with where they were received."""
    out = {}
    if not len(ev) or 'receiver' not in ev:
        return out
    got = ev[ev.type.eq('pass') & ev.outcome.eq('complete') & ev.receiver.notna()]
    if 'source' in got:
        got = got[got.source.ne('action_spotter')]
    ex, ey = _norm(got, 'end_x', 'end_y')
    for (_, r), x, y in zip(got.iterrows(), ex, ey):
        ident = identity.get(r.receiver)
        if ident is None:
            continue
        acc = out.setdefault(ident, {'receptions': 0, 'progressive_receptions': 0, 'receptions_final_third': 0,
                                     'receptions_in_box': 0, 'receptions_behind_line': 0})
        acc['receptions'] += 1
        acc['progressive_receptions'] += int(bool(r.get('progressive')))
        acc['receptions_final_third'] += int(x >= FINAL_THIRD_X)
        acc['receptions_in_box'] += int(bool(_in_box(np.array([x]), np.array([y]))[0]))
        acc['receptions_behind_line'] += int(bool(r.get('behind_line')))
    return out


def mark_behind_line(ev, ctx, sample_hz):
    """Flag completed passes received beyond the last opposing outfielder (balls played in behind)."""
    ev = ev.copy()
    ev['behind_line'] = False
    if not len(ev) or ctx is None or not len(ctx):
        return ev
    lines = ctx.set_index(['sample', 'team']).last_line
    mask = ev.type.eq('pass') & ev.get('outcome', pd.Series(index=ev.index, dtype=object)).eq('complete')
    for i, r in ev[mask].iterrows():
        sample = int(round((r.time_s + (r.get('flight_s') or 0)) * sample_hz))
        line = lines.get((sample, r.team), np.nan)
        if not np.isfinite(line) or not np.isfinite(r.end_x):
            continue
        end_xn = r.end_x if r.attack_sign >= 0 else L - r.end_x
        ev.at[i, 'behind_line'] = bool(end_xn >= line + BEHIND_LINE_M and (r.forward_m or 0) > 0)
    return ev


def mark_key_passes(ev):
    """A completed pass whose receiver shoots within KEY_PASS_S."""
    ev = ev.copy()
    ev['key_pass'] = False
    shots = ev[ev.type.eq('shot')]
    for i, p in ev[ev.type.eq('pass') & ev.get('outcome', pd.Series(index=ev.index, dtype=object)).eq('complete')].iterrows():
        follow = shots[(shots.segment == p.receiver) & (shots.time_s >= p.time_s) & (shots.time_s <= p.time_s + KEY_PASS_S)]
        if len(follow):
            ev.at[i, 'key_pass'] = True
    return ev


def build(people, events, spells, identity, directions, sample_hz, live_seconds):
    """Statistics for every identity and both teams. identity maps segment -> identity id."""
    from . import match_movement as MV
    from .match_events import attack_sign
    ctx = MV.frame_context(people, directions)
    people = people.assign(identity=people.segment.map(identity))
    ev = mark_key_passes(events) if len(events) else events
    ev = mark_behind_line(ev, ctx, sample_hz) if len(ev) else ev
    ev = ev.assign(identity=ev.segment.map(identity)) if len(ev) else ev
    spotted = bool(len(ev)) and 'source' in ev and ev.source.eq('action_spotter').any()
    sp = spells.assign(identity=spells.segment.map(identity)) if len(spells) else spells
    received = receptions(ev, identity)
    movement = MV.movement_stats(people, spells, identity, directions, sample_hz, ctx=ctx)
    per90 = 90 * 60 / live_seconds if live_seconds else None
    players = []
    for ident, rows in people[people.identity.notna()].groupby('identity'):
        team = rows.team.mode().iat[0]
        role = rows.role.mode().iat[0]
        sign = attack_sign(team, directions)
        mine = ev[ev.identity.eq(ident)] if len(ev) else ev
        stats = on_ball(mine, sp[sp.identity.eq(ident)] if len(sp) else sp, spotted) if len(mine) else on_ball(
            pd.DataFrame(columns=['type', 'outcome', 'length_m', 'forward_m', 'distance_to_goal_m']), sp.iloc[:0],
            spotted)
        stats.update(received.get(ident, {k: 0 for k in ('receptions', 'progressive_receptions', 'receptions_final_third',
                                                         'receptions_in_box', 'receptions_behind_line')}))
        moves = movement.get(ident, {})
        stats.update({k: moves.get(k, 0) for k in ('high_intensity_runs', 'runs_in_behind', 'box_runs',
                                                   'forward_runs', 'overlaps', 'underlaps', 'pressing_runs',
                                                   'recovery_runs')})
        visible = float(rows.groupby('segment').time_s.agg(lambda s: s.max() - s.min()).sum())
        own90 = 90 * 60 / visible if visible >= MIN_RATE_VISIBLE_S else None
        touches = stats.get('touches') or 0
        players.append({
            'identity': ident, 'team': team, 'role': role, 'segments': int(rows.segment.nunique()),
            'visible_seconds': visible, 'physical': physical(rows, sample_hz), 'positional': positional(rows, sign),
            'on_ball': stats, 'movement': moves, 'spotter': spotted,
            'per90': {k: stats[k] * per90 for k in COUNTED if per90 is not None and stats.get(k) is not None},
            'per90_visible': {k: stats[k] * own90 for k in COUNTED if own90 is not None and stats.get(k) is not None},
            'per100_touches': {k: stats[k] * 100 / touches for k in COMPOSITION if touches and stats.get(k) is not None},
        })
    teams = []
    for team in ('A', 'B'):
        mine = ev[ev.team.eq(team)] if len(ev) else ev
        sp_team = sp[sp.team.eq(team)] if len(sp) else sp
        teams.append({'team': team, 'on_ball': on_ball(mine, sp_team, spotted) if len(mine) else None,
                      'possession_share': float(sp_team.duration_s.sum() / sp.duration_s.sum()) if len(sp) and sp.duration_s.sum() else None})
    return {'players': players, 'teams': teams, 'live_seconds': live_seconds, 'per90_factor': per90,
            'note': 'Video-derived counts for what the broadcast showed of each identified player. per90 uses the '
                    'team\'s observed live play, per90_visible the player\'s own identified screen time, and '
                    'per100_touches describes the mix of on-ball actions. Physical measures use visible time only.'}
