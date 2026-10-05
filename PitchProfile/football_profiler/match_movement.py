"""Full-match analysis: off-ball movement and positional context per player.

These statistics use tracked player positions, the most complete evidence in
broadcast analysis. They do not need the ball to be detected; the possession
timeline (which team had the last touch) only labels whether a run happened
in or out of possession.

Runs are high-intensity movements (at least RUN_MIN_SPEED for RUN_MIN_S) of
one stitched segment. Coordinates are attack-normalised: every team attacks
towards x = 105, and a team's defensive line is the deepest outfield players
of the opposition visible in the same frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fieldcal as FC

L, W = FC.PITCH_LENGTH, FC.PITCH_WIDTH
SMOOTH_S = 1.0
RUN_MIN_SPEED = 5.5             # m/s (19.8 km/h), the usual high-intensity threshold
RUN_MIN_S = 1.0
RUN_JOIN_S = .4                 # dips below the threshold this short do not end a run
POSSESSION_MEMORY_S = 4.0       # a team keeps possession this long after its last touch
FORWARD_SHARE = .7              # displacement share along the attack axis for a forward/backward run
RUN_FORWARD_M = 5.0
LINE_MIN_OPPONENTS = 3          # opponents in view needed to place the last defensive line
LINES_MIN_OPPONENTS = 6         # ... and to separate their defensive and midfield lines
LINE_PLAYERS = 4
BEHIND_MARGIN_M = 1.0
PRESS_END_M = 3.0
PRESS_AFTER_S = 1.0
OVERLAP_LATERAL_M = 2.0
OVERLAP_MAX_GAP_M = 15.0
BOX_X, BOX_Y = L - 16.5, (13.84, 54.16)


def _smoothed(seg, sample_hz):
    """Smoothed positions and speed of one segment (attack-normalised columns xn, yn)."""
    s = seg.sort_values('sample')
    s = s[np.isfinite(s.xn)]
    window = max(3, int(round(SMOOTH_S * sample_hz)) | 1)
    if len(s) < window:
        return None
    xy = s[['xn', 'yn']].rolling(window, center=True, min_periods=window // 2).mean()
    s = s.assign(sx=xy.xn.to_numpy(), sy=xy.yn.to_numpy())
    dt = np.diff(s['sample'].to_numpy()) / sample_hz
    step = np.hypot(np.diff(s.sx.to_numpy()), np.diff(s.sy.to_numpy()))
    speed = np.r_[np.nan, np.where(dt > 0, step / np.maximum(dt, 1e-9), np.nan)]
    speed[np.r_[False, dt > 2.5 / sample_hz]] = np.nan          # no speed across gaps
    speed[speed > 11] = np.nan                                     # tracking or calibration jumps
    return s.assign(speed=speed)


def _runs(s, sample_hz):
    """(start index, end index) of high-intensity runs in one smoothed segment."""
    fast = np.nan_to_num(s.speed.to_numpy(), nan=0) >= RUN_MIN_SPEED
    samples = s['sample'].to_numpy()
    out, start, last = [], None, None
    join = max(1, int(round(RUN_JOIN_S * sample_hz)))
    for i, f in enumerate(fast):
        if f:
            if start is not None and samples[i] - samples[last] > join:
                out.append((start, last))
                start = None
            start = i if start is None else start
            last = i
    if start is not None:
        out.append((start, last))
    return [(a, b) for a, b in out if (samples[b] - samples[a]) / sample_hz >= RUN_MIN_S]


def possession_timeline(spells, samples, sample_hz):
    """Team in possession at each sample (None when nobody touched the ball recently)."""
    owner = pd.Series(None, index=pd.Index(samples, name='sample'), dtype=object)
    if not len(spells):
        return owner
    memory = int(round(POSSESSION_MEMORY_S * sample_hz))
    sp = spells.sort_values('start_sample')
    for r in sp.itertuples():
        a = int(r.start_sample)
        b = int(r.end_sample) + memory
        owner.loc[(owner.index >= a) & (owner.index <= b)] = r.team
    return owner


def _lines(opponents):
    """Deepest line (mean of the deepest outfield opponents) and the midfield line behind it, in xn."""
    x = np.sort(opponents)[::-1]
    if len(x) < LINE_MIN_OPPONENTS:
        return np.nan, np.nan
    last = float(x[0])                       # the last defender, for runs in behind
    if len(x) < LINES_MIN_OPPONENTS:
        return last, np.nan
    mid = float(np.mean(x[LINE_PLAYERS:LINE_PLAYERS + 3]))
    return last, mid


def frame_context(people, directions):
    """Per (sample, team) opposition lines and team centroid, in that team's attack-normalised frame."""
    from .match_events import attack_sign
    q = people[people.team.isin(['A', 'B']) & people.role.eq('player') & np.isfinite(people.x)]
    rows = []
    for (sample, team), g in q.groupby(['sample', 'team']):
        rows.append((sample, team, g.x.to_numpy(), g.y.to_numpy()))
    by = {(s, t): (x, y) for s, t, x, y in rows}
    out = []
    for (sample, team), (x, y) in by.items():
        sign = attack_sign(team, directions)
        if not sign:
            continue
        other = 'B' if team == 'A' else 'A'
        ox = by.get((sample, other), (np.array([]), None))[0]
        oxn = ox if sign > 0 else L - ox
        last, mid = _lines(oxn)
        xn = x if sign > 0 else L - x
        out.append({'sample': sample, 'team': team, 'last_line': last, 'mid_line': mid,
                    'team_mean_xn': float(xn.mean()), 'teammates': int(len(x)), 'opponents': int(len(ox))})
    return pd.DataFrame(out, columns=['sample', 'team', 'last_line', 'mid_line', 'team_mean_xn', 'teammates',
                                      'opponents'])


def movement_stats(people, spells, identity, directions, sample_hz, ctx=None):
    """Off-ball runs and positional context for every identity.

    Returns identity -> dict of counts, distances and shares. Counts cover the
    identified screen time of that player, like the on-ball statistics. ctx is
    frame_context(people, directions), computed here when not given.
    """
    from .match_events import attack_sign
    p = people[people.team.isin(['A', 'B']) & people.role.eq('player') & np.isfinite(people.x) &
               people.segment.isin(identity)].copy()
    if p.empty:
        return {}
    sign = p.team.map(lambda t: attack_sign(t, directions)).to_numpy()
    p = p[sign != 0]
    sign = sign[sign != 0]
    p['xn'] = np.where(sign > 0, p.x, L - p.x)
    p['yn'] = np.where(sign > 0, p.y, W - p.y)
    ctx = (frame_context(people, directions) if ctx is None else ctx).set_index(['sample', 'team'])
    owner = possession_timeline(spells, np.sort(people['sample'].unique()), sample_hz)
    holder_by_sample = _holders(spells)
    # Every tracked outfield player in stadium coordinates (identified or not): ball carriers
    # being pressed or overlapped need not be identified themselves.
    everyone = people[people.team.isin(['A', 'B']) & people.role.isin(['player', 'goalkeeper']) & np.isfinite(people.x)]
    positions = {s: g[['segment', 'team', 'x', 'y']] for s, g in everyone.groupby('sample')}
    out = {}
    for seg, g in p.groupby('segment'):
        s = _smoothed(g, sample_hz)
        if s is None:
            continue
        ident = identity[seg]
        team = s.team.iat[0]
        acc = out.setdefault(ident, _empty())
        _position_context(acc, s, team, ctx, owner)
        for a, b in _runs(s, sample_hz):
            after = s.iloc[b + 1:b + 1 + int(round(PRESS_AFTER_S * sample_hz))]
            _classify_run(acc, s.iloc[a:b + 1], after, team, attack_sign(team, directions), ctx, owner,
                          holder_by_sample, positions)
    for acc in out.values():
        _finish(acc)
    return out


def _empty():
    return {'high_intensity_runs': 0, 'runs_in_possession': 0, 'runs_out_of_possession': 0,
            'forward_runs': 0, 'runs_in_behind': 0, 'box_runs': 0, 'overlaps': 0, 'underlaps': 0,
            'pressing_runs': 0, 'recovery_runs': 0,
            'hi_distance_in_possession_m': 0.0, 'hi_distance_out_of_possession_m': 0.0,
            '_n': 0, '_rel_x': 0.0, '_width': 0.0, '_between': 0, '_between_n': 0, '_final_third': 0,
            '_in_poss_n': 0, '_rel_x_poss': 0.0, '_out_poss_n': 0, '_rel_x_out': 0.0}


def _position_context(acc, s, team, ctx, owner):
    keys = list(zip(s['sample'], [team] * len(s)))
    c = ctx.reindex(keys)
    rel = s.sx.to_numpy() - c.team_mean_xn.to_numpy()
    valid = np.isfinite(rel) & (c.teammates.to_numpy() >= 4)
    acc['_n'] += int(valid.sum())
    acc['_rel_x'] += float(np.nansum(rel[valid]))
    acc['_width'] += float(np.nansum(np.abs(s.sy.to_numpy()[valid] - W / 2)))
    poss = owner.reindex(s['sample']).to_numpy()
    mine, theirs = valid & (poss == team), valid & (poss != team) & pd.notna(poss)
    acc['_in_poss_n'] += int(mine.sum()); acc['_rel_x_poss'] += float(np.nansum(rel[mine]))
    acc['_out_poss_n'] += int(theirs.sum()); acc['_rel_x_out'] += float(np.nansum(rel[theirs]))
    last, mid = c.last_line.to_numpy(), c.mid_line.to_numpy()
    lines = mine & np.isfinite(mid) & np.isfinite(last)
    between = lines & (s.sx.to_numpy() < last - BEHIND_MARGIN_M) & (s.sx.to_numpy() > mid + BEHIND_MARGIN_M) & \
        (np.abs(s.sy.to_numpy() - W / 2) < 20)
    acc['_between_n'] += int(lines.sum())
    acc['_between'] += int(between.sum())
    acc['_final_third'] += int((valid & (s.sx.to_numpy() >= 2 * L / 3)).sum())


def _holders(spells):
    """sample -> (segment, team) of the player on the ball."""
    out = {}
    for r in spells.itertuples():
        for k in range(int(r.start_sample), int(r.end_sample) + 1):
            out[k] = (r.segment, r.team)
    return out


def _classify_run(acc, run, after, team, sign, ctx, owner, holders, positions):
    dx = float(run.sx.iat[-1] - run.sx.iat[0])
    dy = float(run.sy.iat[-1] - run.sy.iat[0])
    length = float(np.hypot(np.diff(run.sx), np.diff(run.sy)).sum())
    mid_sample = int(run['sample'].iat[len(run) // 2])
    poss = owner.get(mid_sample)
    acc['high_intensity_runs'] += 1
    in_poss = poss == team
    out_poss = poss is not None and poss == ('B' if team == 'A' else 'A')
    if in_poss:
        acc['runs_in_possession'] += 1
        acc['hi_distance_in_possession_m'] += length
    elif out_poss:
        acc['runs_out_of_possession'] += 1
        acc['hi_distance_out_of_possession_m'] += length
    displacement = np.hypot(dx, dy)
    forward = displacement > 0 and dx >= RUN_FORWARD_M and dx / displacement >= FORWARD_SHARE
    backward = displacement > 0 and -dx >= RUN_FORWARD_M and -dx / displacement >= FORWARD_SHARE
    end_x, end_y = float(run.sx.iat[-1]), float(run.sy.iat[-1])
    if in_poss and forward:
        acc['forward_runs'] += 1
        c0 = ctx.reindex([(int(run['sample'].iat[0]), team)]).iloc[0]
        c1 = ctx.reindex([(int(run['sample'].iat[-1]), team)]).iloc[0]
        if np.isfinite(c0.last_line) and np.isfinite(c1.last_line) and \
                run.sx.iat[0] <= c0.last_line and end_x >= c1.last_line - BEHIND_MARGIN_M:
            acc['runs_in_behind'] += 1
        if end_x >= BOX_X and BOX_Y[0] <= end_y <= BOX_Y[1] and run.sx.iat[0] < BOX_X:
            acc['box_runs'] += 1
        kind = _overlap(run, team, sign, holders, positions)
        if kind:
            acc[kind] += 1
    if out_poss:
        if backward:
            acc['recovery_runs'] += 1
        if _presses(run, after, team, holders, positions):
            acc['pressing_runs'] += 1


def _presses(run, after, team, holders, positions):
    """The runner closes an opposing ball carrier to PRESS_END_M, by the end of the sprint or just after.

    Speed is smoothed over a second, so a sprint 'ends' about half a second before the runner
    arrives; the check continues through the following PRESS_AFTER_S.
    """
    tail = pd.concat([run.tail(3), after])
    for r in tail.itertuples():
        holder = holders.get(int(r.sample))
        if holder is None or holder[1] == team:
            continue
        g = positions.get(int(r.sample))
        if g is None:
            continue
        h = g[g.segment == holder[0]]
        # Stadium coordinates for both: the carrier is normalised for the other team.
        if len(h) and np.hypot(h.x.iat[0] - r.x, h.y.iat[0] - r.y) <= PRESS_END_M:
            return True
    return False


def _overlap(run, team, sign, holders, positions):
    """'overlaps' / 'underlaps' when the runner passes a team-mate on the ball on the outside / inside."""
    a, b = int(run['sample'].iat[0]), int(run['sample'].iat[-1])
    ha, hb = holders.get(a), holders.get(b)
    if ha is None or hb is None or ha[0] != hb[0] or ha[1] != team:
        return None
    ga, gb = positions.get(a), positions.get(b)
    if ga is None or gb is None:
        return None
    ca, cb = ga[ga.segment == ha[0]], gb[gb.segment == ha[0]]
    if ca.empty or cb.empty:
        return None
    r0, r1 = (run.sx.iat[0], run.sy.iat[0]), (run.sx.iat[-1], run.sy.iat[-1])
    norm = (lambda x, y: (x, y)) if sign > 0 else (lambda x, y: (L - x, W - y))
    c0, c1 = norm(ca.x.iat[0], ca.y.iat[0]), norm(cb.x.iat[0], cb.y.iat[0])
    if not (r0[0] < c0[0] and r1[0] > c1[0]):                      # from behind the carrier to ahead of him
        return None
    if abs(r1[1] - c1[1]) > OVERLAP_MAX_GAP_M or np.sign(r1[1] - W / 2) != np.sign(c1[1] - W / 2):
        return None
    wider = abs(r1[1] - W / 2) - abs(c1[1] - W / 2)
    if wider >= OVERLAP_LATERAL_M:
        return 'overlaps'
    if wider <= -OVERLAP_LATERAL_M:
        return 'underlaps'
    return None


def _finish(acc):
    tmp = {k: acc.pop(k) for k in [k for k in acc if k.startswith('_')]}
    ratio = lambda num, den: tmp[num] / tmp[den] if tmp[den] else None
    acc['mean_x_vs_team_m'] = ratio('_rel_x', '_n')
    acc['mean_width_m'] = ratio('_width', '_n')
    acc['mean_x_vs_team_in_possession_m'] = ratio('_rel_x_poss', '_in_poss_n')
    acc['mean_x_vs_team_out_of_possession_m'] = ratio('_rel_x_out', '_out_poss_n')
    acc['between_lines_share'] = ratio('_between', '_between_n')
    acc['final_third_share_context'] = ratio('_final_third', '_n')
