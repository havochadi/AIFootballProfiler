"""Full-match analysis, stage 3: possession and on-ball events from video evidence.

Inputs are per-sample person positions (with team and role) and the ball path
produced by match_post. Everything is expressed per stitched player segment;
match-level identities are resolved afterwards.

Heuristics are explicit and conservative. They describe what the camera saw:
events outside the broadcast view, or while the ball is unobserved, are missed
rather than invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fieldcal as FC

L, W = FC.PITCH_LENGTH, FC.PITCH_WIDTH
# Tracklet stitching inside one continuous view.
STITCH_MAX_GAP_S = 1.2
STITCH_BASE_M = 1.8
STITCH_SPEED = 7.5            # m/s a player can plausibly cover while unobserved
# Possession: touches are velocity-change kicks at a player, or sustained close control.
TOUCH_DV = 4.0                # m/s change between incoming and outgoing ball velocity
TOUCH_RADIUS_M = 2.0          # player feet to ball at a kick
CONTROL_RADIUS_M = 1.2        # ball kept this close while moving with the player
CONTROL_MAX_SPEED = 9.0       # m/s; faster balls are passing by, not being dribbled
CONTROL_MIN_SAMPLES = 3
CONTROL_MARGIN_M = .5         # current holder within CONTROL_RADIUS_M + this keeps possession against markers
CONTEST_RADIUS_M = 2.0
CONTACT_PAD = .35             # box widths either side of a player that still count as image contact
SPELL_MAX_GAP_S = 2.5         # touches by one player further apart start a new spell
DEFLECTION_MAX_SAMPLES = 2    # an opposing spell this short, between two of the same team, is a deflection
DEFLECTION_MAX_GAP_S = 2.0
# Events.
MAX_PASS_FLIGHT_S = 4.0
MAX_BALL_SPEED = 35.0         # m/s implied between release and next touch; faster means a broken ball path
TACKLE_RADIUS_M = 2.2
INTERCEPT_MIN_SPEED = 6.0     # m/s ball speed just before the opponent's first touch
INTERCEPT_MIN_FLIGHT_M = 3.0
TACKLE_MAX_GAP_S = .6
PROGRESSIVE_M = 10.0
LONG_PASS_M = 30.0
CARRY_MIN_M = 2.0             # ball moved with one player (SoccerTrack 'Drive' median is 2.9 m)
PRESSURE_RADIUS_M = 3.0
DRIBBLE_FRONT_M = 2.0        # opponent this close in front of a carrier...
TAKE_ON_LATERAL_M = 1.2       # ...and in his path
TAKE_ON_DRIVE_MS = 2.0        # carrier running at him (m/s towards goal); 7-9 attempts per team-half
TAKE_ON_MIN_S = .4
# Ball contacts (contacts()): looser than possession touches, so a stab or deflection counts. Used to
# credit video-spotted actions to the player at the ball. Classifying tackles and blocks from contact
# kinematics alone was tried on SoccerTrack ground truth and rejected: annotated tackles and blocks
# (about 1% of contacts) have the same distances, velocity changes and timings as other contacts.
DEF_CONTACT_DV = 2.5          # m/s velocity change at a contact
DEF_CONTACT_RADIUS_M = 3.5    # player's feet to the ball


def attack_sign(team, directions):
    """+1 if the team attacks towards x=105 in this half, -1 towards x=0, 0 if unknown."""
    if directions.get('status') not in ('inferred', 'uncertain') or team not in ('A', 'B'):
        return 0
    return 1 if directions['attacks_right'] == team else -1


def stitch(people, max_gap_s=STITCH_MAX_GAP_S):
    """Join consecutive tracklets of one person within a view by motion continuity.

    Returns a mapping tracklet -> segment id. Only same-team, same-role,
    non-overlapping tracklets whose end and start positions are reachable are
    joined, greedily by distance.
    """
    q = people[np.isfinite(people.x)]
    if q.empty:
        return {}
    ends = q.sort_values('sample').groupby('tracklet').agg(
        view=('view_shot', 'first'), team=('team', 'first'), role=('role', 'first'),
        t0=('time_s', 'min'), t1=('time_s', 'max'),
        x0=('x', 'first'), y0=('y', 'first'), x1=('x', 'last'), y1=('y', 'last'))
    parent = {k: k for k in ends.index}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    candidates = []
    for view, group in ends.groupby('view'):
        g = group.sort_values('t0')
        for a in g.itertuples():
            later = g[(g.t0 > a.t1) & (g.t0 - a.t1 <= max_gap_s) & (g.team.fillna('') == (a.team or '')) &
                      (g.role == a.role)]
            for b in later.itertuples():
                gap = b.t0 - a.t1
                dist = float(np.hypot(b.x0 - a.x1, b.y0 - a.y1))
                if dist <= STITCH_BASE_M + STITCH_SPEED * gap:
                    candidates.append((dist + 2 * gap, a.Index, b.Index))
    used_end, used_start = set(), set()
    for _, a, b in sorted(candidates):
        if a in used_end or b in used_start or root(a) == root(b):
            continue
        used_end.add(a); used_start.add(b)
        parent[root(b)] = root(a)
    return {k: 'seg-' + root(k) for k in ends.index}


def ball_kinematics(ball, sample_hz, k=2):
    """Incoming/outgoing ball velocity around each sample and the change between them.

    A touch (kick, trap, deflection) shows up as an abrupt velocity change. Only
    observed samples within one view contribute; interpolated samples are
    straight lines and carry no touch evidence of their own.
    """
    b = ball.sort_values('sample').reset_index(drop=True).copy()
    b['dv'] = np.nan
    b['speed'] = np.nan
    if 'ball_segment' not in b:
        b['ball_segment'] = b.view_shot
    # Velocity is only measured along one continuous piece of the ball path.
    for _, g in b.groupby('ball_segment'):
        idx = g.index.to_numpy()
        s = g['sample'].to_numpy()
        xy = g[['x', 'y']].to_numpy(float)
        for pos, i in enumerate(idx):
            a, c = pos - k, pos + k
            if a < 0 or c >= len(idx) or s[c] - s[a] != 2 * k:
                continue
            v_in = (xy[pos] - xy[a]) * sample_hz / k
            v_out = (xy[c] - xy[pos]) * sample_hz / k
            b.at[i, 'dv'] = float(np.hypot(*(v_out - v_in)))
            b.at[i, 'speed'] = float(np.hypot(*((xy[c] - xy[a]) * sample_hz / (2 * k))))
    return b


def possession(people, ball, sample_hz):
    """Per-sample touches: velocity-change kicks and sustained close control.

    holder is set only on samples with touch evidence: a local velocity-change
    peak with a player at the ball (kick), or a player keeping the ball within
    CONTROL_RADIUS_M while moving with it (control). A ball rolling past a
    player without changing course is not a touch.
    """
    players = people[people.team.isin(['A', 'B']) & people.role.isin(['player', 'goalkeeper'])]
    by_sample = {s: g for s, g in players.groupby('sample')}
    b = ball_kinematics(ball, sample_hz)
    nearest, dist, contested, around = [], [], [], []
    has_boxes = 'bbox_x' in players and players.bbox_x.notna().any()
    for r in b.itertuples():
        g = by_sample.get(r.sample)
        if g is None or not len(g) or not np.isfinite(r.x):
            nearest.append(None); dist.append(np.nan); contested.append(False); around.append({})
            continue
        d = np.nan_to_num(np.hypot(g.x.to_numpy() - r.x, g.y.to_numpy() - r.y), nan=np.inf)
        if has_boxes and np.isfinite(getattr(r, 'cx', np.nan)):
            # A touch needs the ball on the player in the image too; lofted balls
            # project to the pitch far from where they are.
            bw, bh = g.bbox_w.to_numpy(), g.bbox_h.to_numpy()
            contact = ((r.cx >= g.bbox_x.to_numpy() - CONTACT_PAD * bw) & (r.cx <= g.bbox_x.to_numpy() + (1 + CONTACT_PAD) * bw) &
                       (r.cy >= g.bbox_y.to_numpy() - .1 * bh) & (r.cy <= g.bbox_y.to_numpy() + 1.15 * bh))
            d = np.where(contact | ~np.isfinite(bw), d, np.inf)
        k = int(np.argmin(d))
        nearest.append(g.segment.iat[k] if np.isfinite(d[k]) else None)
        dist.append(float(d[k]))
        contested.append(bool(((d <= CONTEST_RADIUS_M) & (g.team.to_numpy() != g.team.iat[k])).any()))
        around.append({seg: float(v) for seg, v in zip(g.segment, d) if v <= TOUCH_RADIUS_M})
    b['nearest'], b['distance_m'], b['contested'] = nearest, dist, contested
    observed = ~b.interpolated.to_numpy(bool)
    dv = b.dv.to_numpy()
    peak = np.zeros(len(b), bool)
    for i in range(len(b)):
        if np.isfinite(dv[i]) and dv[i] >= TOUCH_DV:
            peak[i] = dv[i] >= np.nanmax(dv[max(0, i - 2): i + 3])
    close = (b.distance_m.to_numpy() <= CONTROL_RADIUS_M) & (np.nan_to_num(b.speed.to_numpy(), nan=0) <= CONTROL_MAX_SPEED) & observed
    samples, views, nearest_seg = b['sample'].to_numpy(), b.view_shot.to_numpy(), b.nearest.to_numpy(object)
    holder, touch = np.full(len(b), None, object), np.full(len(b), None, object)
    current, run_seg, run_start = {}, None, None
    for i in range(len(b)):
        view = views[i]
        cur = current.get(view)
        if peak[i] and observed[i]:
            # The kicker is the player at the ball just before it changes course.
            best = {}
            for j in range(max(0, i - 2), i + 1):
                if views[j] == view and samples[i] - samples[j] <= 2:
                    for seg, d in around[j].items():
                        best[seg] = min(d, best.get(seg, np.inf))
            if best:
                pick = min(best, key=best.get)
                if cur in best and best[cur] <= best[pick] + .5:
                    pick = cur
                holder[i], touch[i], current[view] = pick, 'kick', pick
                run_seg = None
                continue
        if not close[i]:
            run_seg = None
            continue
        cand = nearest_seg[i]
        if cand == cur:
            holder[i], touch[i] = cand, 'control'
            continue
        # While the current holder is still at the ball, a marker standing a
        # little closer does not take possession without touching it.
        if cur is not None and around[i].get(cur, np.inf) <= CONTROL_RADIUS_M + CONTROL_MARGIN_M:
            run_seg = None
            continue
        consecutive = i > 0 and samples[i] - samples[i - 1] == 1 and views[i - 1] == view
        if run_seg != cand or not consecutive:
            run_seg, run_start = cand, i
        if i - run_start + 1 >= CONTROL_MIN_SAMPLES:
            holder[run_start:i + 1], touch[run_start:i + 1] = cand, 'control'
            current[view] = cand
    b['holder'], b['touch'] = holder, touch
    return b.rename(columns={'x': 'ball_x', 'y': 'ball_y', 'interpolated': 'ball_interpolated'})[
        ['sample', 'time_s', 'view_shot', 'ball_x', 'ball_y', 'ball_interpolated', 'speed', 'dv',
         'nearest', 'distance_m', 'contested', 'holder', 'touch']]


def spells(pos, segments, sample_hz):
    """Consecutive touches by one player (no other touch between) form a spell."""
    info = segments.set_index('segment')
    touches = pos[pos.holder.notna()].sort_values('sample')
    out = []
    for view, g in touches.groupby('view_shot'):
        current, start, last, pts = None, None, None, []
        for r in g.itertuples():
            if current is not None and (r.holder != current or (r.sample - last) / sample_hz > SPELL_MAX_GAP_S):
                out.append(_spell(view, current, start, last, pts, info, sample_hz))
                current = None
            if current is None:
                current, start, pts = r.holder, r.sample, []
            last = r.sample
            pts.append((r.sample, r.time_s, r.ball_x, r.ball_y, r.contested))
        if current is not None:
            out.append(_spell(view, current, start, last, pts, info, sample_hz))
    cols = ['view_shot', 'segment', 'team', 'role', 'start_sample', 'end_sample', 'start_s', 'end_s', 'duration_s',
            'x0', 'y0', 'x1', 'y1', 'contested_share', 'samples']
    return pd.DataFrame(out, columns=cols).sort_values('start_sample').reset_index(drop=True)


def remove_deflections(sp, max_samples=DEFLECTION_MAX_SAMPLES, max_gap_s=DEFLECTION_MAX_GAP_S):
    """Split brief opposing contacts off the possession sequence.

    A spell of at most max_samples by one team, between spells of the other team
    that resume within max_gap_s in the same view, is a deflection or contested
    touch, not a change of possession. On SoccerTrack ground truth this halves
    the excess of possession changes over the annotated action sequence.
    """
    if len(sp) < 3:
        return sp, sp.iloc[:0]
    drop = np.zeros(len(sp), bool)
    for i in range(1, len(sp) - 1):
        a, s, b = sp.iloc[i - 1], sp.iloc[i], sp.iloc[i + 1]
        if (s.team != a.team and b.team == a.team and s.samples <= max_samples and a.view_shot == s.view_shot == b.view_shot
                and b.start_s - a.end_s <= max_gap_s):
            drop[i] = True
    return sp[~drop].reset_index(drop=True), sp[drop].reset_index(drop=True)


def _spell(view, holder, start, last, pts, info, sample_hz):
    p = np.array([(x, y) for _, _, x, y, _ in pts], float)
    seg = info.loc[holder]
    return (view, holder, seg.team, seg.role, start, last, pts[0][1], pts[-1][1], (last - start + 1) / sample_hz,
            p[0, 0], p[0, 1], p[-1, 0], p[-1, 1], float(np.mean([c for *_, c in pts])), len(pts))


def _ball_after(pos, view, sample, until):
    q = pos[(pos.view_shot == view) & (pos['sample'] > sample) & (pos['sample'] <= until)]
    return q.sort_values('sample')


def events(people, ball, directions, sample_hz):
    """Possession spells and derived events for one analysed half.

    Returns (events, spells, possession, people, segment table). Coordinates
    are in stadium orientation; 'forward_m' is measured towards the goal the
    acting team attacks. Shots are not decided here: they come from the video action spotter
    (match_pipeline.merge_spotted).
    """
    seg_map = stitch(people)
    people = people.assign(segment=people.tracklet.map(seg_map))
    people = people[people.segment.notna()]
    segments = people.groupby('segment').agg(team=('team', 'first'), role=('role', 'first'),
                                            view_shot=('view_shot', 'first'), start_s=('time_s', 'min'),
                                            end_s=('time_s', 'max')).reset_index()
    pos = possession(people, ball, sample_hz)
    sp = spells(pos, segments, sample_hz)
    sp, deflections = remove_deflections(sp)
    ev = [{'type': 'deflection', 'segment': d.segment, 'team': d.team, 'view_shot': int(d.view_shot),
           'time_s': float(d.start_s), 'x': float(d.x0), 'y': float(d.y0), 'attack_sign': attack_sign(d.team, directions)}
          for d in deflections.itertuples()]

    def add(kind, s, **extra):
        sign = attack_sign(s.team, directions)
        ev.append({'type': kind, 'segment': s.segment, 'team': s.team, 'view_shot': int(s.view_shot),
                   'time_s': float(s.end_s), 'x': float(s.x1), 'y': float(s.y1), **extra,
                   'attack_sign': sign})

    positions = {s: g for s, g in people.groupby('sample')}
    for i, s in enumerate(sp.itertuples()):
        sign = attack_sign(s.team, directions)
        carry = np.array([s.x1 - s.x0, s.y1 - s.y0])
        carried = float(np.hypot(*carry))
        ev.append({'type': 'touch', 'segment': s.segment, 'team': s.team, 'view_shot': int(s.view_shot),
                   'time_s': float(s.start_s), 'x': float(s.x0), 'y': float(s.y0), 'duration_s': float(s.duration_s),
                   'attack_sign': sign})
        if carried >= CARRY_MIN_M:
            ev.append({'type': 'carry', 'segment': s.segment, 'team': s.team, 'view_shot': int(s.view_shot),
                       'time_s': float(s.start_s), 'x': float(s.x0), 'y': float(s.y0), 'end_x': float(s.x1),
                       'end_y': float(s.y1), 'length_m': carried, 'forward_m': float(carry[0] * sign),
                       'progressive': bool(sign and carry[0] * sign >= PROGRESSIVE_M), 'attack_sign': sign})
        nxt = sp.iloc[i + 1] if i + 1 < len(sp) else None
        ev += _dribbles(s, positions, sign, sample_hz, nxt)
        same_view = nxt is not None and nxt.view_shot == s.view_shot and nxt.start_s - s.end_s <= MAX_PASS_FLIGHT_S
        if same_view:
            implied = np.hypot(nxt.x0 - s.x1, nxt.y0 - s.y1) / max(nxt.start_s - s.end_s, 1 / sample_hz)
            same_view = implied <= MAX_BALL_SPEED
        if not same_view:
            continue
        if nxt.segment == s.segment:
            continue
        length = float(np.hypot(nxt.x0 - s.x1, nxt.y0 - s.y1))
        forward = float((nxt.x0 - s.x1) * sign)
        gap = float(nxt.start_s - s.end_s)
        if nxt.team == s.team:
            add('pass', s, receiver=nxt.segment, outcome='complete', end_x=float(nxt.x0), end_y=float(nxt.y0),
                length_m=length, forward_m=forward, flight_s=gap, progressive=bool(sign and forward >= PROGRESSIVE_M),
                long=length >= LONG_PASS_M, cross=_cross(s, nxt, sign, _ball_after(pos, s.view_shot, s.end_sample, nxt.start_sample)))
            continue
        # Possession changed team. Close, quick transfers are duels; otherwise interceptions.
        contact = _distance_at(positions, s.segment, nxt.segment, nxt.start_sample)
        if gap <= TACKLE_MAX_GAP_S and contact is not None and contact <= TACKLE_RADIUS_M:
            ev.append({'type': 'tackle', 'segment': nxt.segment, 'team': nxt.team, 'view_shot': int(nxt.view_shot),
                       'time_s': float(nxt.start_s), 'x': float(nxt.x0), 'y': float(nxt.y0), 'opponent': s.segment,
                       'attack_sign': attack_sign(nxt.team, directions)})
            add('dispossessed', s, opponent=nxt.segment)
        else:
            kind = 'clearance' if _clearance(s, length, sign) else 'pass'
            add(kind, s, receiver=None, outcome='intercepted' if kind == 'pass' else 'cleared', end_x=float(nxt.x0),
                end_y=float(nxt.y0), length_m=length, forward_m=forward, flight_s=gap,
                progressive=bool(sign and forward >= PROGRESSIVE_M), long=length >= LONG_PASS_M,
                cross=_cross(s, nxt, sign, _ball_after(pos, s.view_shot, s.end_sample, nxt.start_sample)))
            # Cutting out a moving pass is an interception; collecting a slow or loose ball is only a recovery.
            before = pos[(pos.view_shot == s.view_shot) & (pos['sample'] < nxt.start_sample) &
                         (pos['sample'] >= nxt.start_sample - 3)].speed
            if length >= INTERCEPT_MIN_FLIGHT_M and before.notna().any() and before.median() >= INTERCEPT_MIN_SPEED:
                ev.append({'type': 'interception', 'segment': nxt.segment, 'team': nxt.team, 'view_shot': int(nxt.view_shot),
                           'time_s': float(nxt.start_s), 'x': float(nxt.x0), 'y': float(nxt.y0), 'opponent': s.segment,
                           'attack_sign': attack_sign(nxt.team, directions)})
        ev.append({'type': 'recovery', 'segment': nxt.segment, 'team': nxt.team, 'view_shot': int(nxt.view_shot),
                   'time_s': float(nxt.start_s), 'x': float(nxt.x0), 'y': float(nxt.y0), 'opponent': s.segment,
                   'attack_sign': attack_sign(nxt.team, directions)})
    ev += _pressures(sp, positions, directions)
    frame = pd.DataFrame(ev)
    if not frame.empty:
        frame = frame.sort_values(['time_s', 'type']).reset_index(drop=True)
    return frame, sp, pos, people, segments


def _in_box(x, y, sign):
    box_x = (x >= L - 16.5) if sign > 0 else (x <= 16.5)
    return box_x & (y >= 13.84) & (y <= 54.16)


def _cross(s, nxt, sign, path=None):
    """A delivery from the wide attacking zone whose ball path or reception enters the penalty area."""
    if not sign:
        return False
    attacking_third = s.x1 >= L * 2 / 3 if sign > 0 else s.x1 <= L / 3
    wide = abs(s.y1 - W / 2) >= 18.0
    if not (attacking_third and wide):
        return False
    if path is not None and len(path) and bool(_in_box(path.ball_x.to_numpy(), path.ball_y.to_numpy(), sign).any()):
        return True
    return bool(_in_box(np.array([nxt.x0]), np.array([nxt.y0]), sign)[0])


def _clearance(s, length, sign):
    if not sign:
        return False
    own_third = s.x1 <= L / 3 if sign > 0 else s.x1 >= 2 * L / 3
    return bool(own_third and length >= 20)


def _distance_at(positions, a, b, sample):
    g = positions.get(sample)
    if g is None:
        return None
    pa, pb = g[g.segment == a], g[g.segment == b]
    if pa.empty or pb.empty:
        return None
    return float(np.hypot(pa.x.iat[0] - pb.x.iat[0], pa.y.iat[0] - pb.y.iat[0]))


def _dribbles(s, positions, sign, sample_hz, nxt=None):
    """Take-ons: the carrier drives forward at an opponent close in front of him.

    An attempt needs the opponent within DRIBBLE_FRONT_M ahead and TAKE_ON_LATERAL_M across, with
    the carrier moving towards goal at TAKE_ON_DRIVE_MS or faster. As in data-provider definitions
    it succeeds ('dribble') when his team keeps the ball afterwards. Following the opponent until
    he is behind the carrier was tried first; monocular position noise of about a metre made it
    flip too often (about 5% success on 3 analysed halves), so retention decides the outcome.
    """
    if not sign or s.duration_s < TAKE_ON_MIN_S:
        return []
    track = []
    for sample in range(int(s.start_sample), int(s.end_sample) + 1):
        g = positions.get(sample)
        if g is None:
            continue
        me = g[g.segment == s.segment]
        if len(me):
            track.append((sample, me.x.iat[0], me.y.iat[0], g))
    for i, (sample, mx, my, g) in enumerate(track):
        opp = g[g.team.isin(['A', 'B']) & (g.team != s.team) & np.isfinite(g.x)]
        if opp.empty:
            continue
        ahead, lateral = (opp.x.to_numpy() - mx) * sign, np.abs(opp.y.to_numpy() - my)
        front = np.flatnonzero((ahead > 0) & (ahead <= DRIBBLE_FRONT_M) & (lateral <= TAKE_ON_LATERAL_M))
        if not len(front):
            continue
        j0, j1 = max(0, i - 3), min(len(track) - 1, i + 3)
        dt = (track[j1][0] - track[j0][0]) / sample_hz
        if dt <= 0 or (track[j1][1] - track[j0][1]) * sign / dt < TAKE_ON_DRIVE_MS:
            continue
        kept = nxt is not None and nxt.team == s.team and nxt.view_shot == s.view_shot
        base = {'segment': s.segment, 'team': s.team, 'view_shot': int(s.view_shot), 'time_s': float(s.start_s),
                'x': float(s.x0), 'y': float(s.y0), 'opponent': opp.segment.iat[int(front[np.argmin(ahead[front])])],
                'attack_sign': sign}
        out = [{'type': 'take_on', **base, 'outcome': 'complete' if kept else 'incomplete'}]
        if kept:
            out.append({'type': 'dribble', **base, 'outcome': 'complete'})
        return out
    return []


def _pressures(sp, positions, directions):
    """Defenders who close the ball carrier to within PRESSURE_RADIUS_M during a spell."""
    out = []
    for s in sp.itertuples():
        seen = set()
        for sample in range(int(s.start_sample), int(s.end_sample) + 1):
            g = positions.get(sample)
            if g is None:
                continue
            me = g[g.segment == s.segment]
            if me.empty:
                continue
            opp = g[g.team.isin(['A', 'B']) & (g.team != s.team)]
            d = np.hypot(opp.x - me.x.iat[0], opp.y - me.y.iat[0])
            for seg in opp.segment[d <= PRESSURE_RADIUS_M]:
                if seg not in seen:
                    seen.add(seg)
                    row = opp[opp.segment == seg].iloc[0]
                    out.append({'type': 'pressure', 'segment': seg, 'team': row.team, 'view_shot': int(s.view_shot),
                                'time_s': float(g.time_s.iat[0]), 'x': float(row.x), 'y': float(row.y),
                                'opponent': s.segment, 'attack_sign': attack_sign(row.team, directions)})
    return out


def contacts(pos, people, ball=None):
    """Ball contacts: local velocity-change peaks with every player then within reach of the ball.

    Looser than possession touches: a stab, block or deflection counts. Where player boxes and the
    ball's image position exist, the ball must also be on the player in the image (lofted balls
    project onto the pitch far from where they are).
    """
    p = pos.sort_values('sample').reset_index(drop=True)
    dv = p.dv.to_numpy(float)
    observed = ~p.ball_interpolated.to_numpy(bool)
    samples, views = p['sample'].to_numpy(), p.view_shot.to_numpy()
    bx, by_ = p.ball_x.to_numpy(float), p.ball_y.to_numpy(float)
    q = people[people.team.isin(['A', 'B']) & people.role.isin(['player', 'goalkeeper']) & np.isfinite(people.x)]
    by_sample = {s: g for s, g in q.groupby('sample')}
    image = {}
    if ball is not None and 'cx' in ball:
        image = {int(r.sample): (r.cx, r.cy) for r in ball.itertuples() if np.isfinite(r.cx)}
    out = []
    for i in range(len(p)):
        if not (observed[i] and np.isfinite(dv[i]) and dv[i] >= DEF_CONTACT_DV):
            continue
        if dv[i] < np.nanmax(dv[max(0, i - 2): i + 3]):
            continue
        near = {}
        for j in range(max(0, i - 2), i + 1):
            if views[j] != views[i] or samples[i] - samples[j] > 2 or not np.isfinite(bx[j]):
                continue
            g = by_sample.get(samples[j])
            if g is None:
                continue
            d = np.hypot(g.x.to_numpy() - bx[j], g.y.to_numpy() - by_[j])
            img = image.get(int(samples[j]))
            if img is not None and 'bbox_x' in g and g.bbox_w.notna().any():
                bw, bh = g.bbox_w.to_numpy(), g.bbox_h.to_numpy()
                on = ((img[0] >= g.bbox_x.to_numpy() - CONTACT_PAD * bw) & (img[0] <= g.bbox_x.to_numpy() + (1 + CONTACT_PAD) * bw) &
                      (img[1] >= g.bbox_y.to_numpy() - .1 * bh) & (img[1] <= g.bbox_y.to_numpy() + 1.15 * bh))
                d = np.where(on | ~np.isfinite(bw), d, np.inf)
            for seg, team, dd in zip(g.segment, g.team, d):
                if dd <= DEF_CONTACT_RADIUS_M and dd < near.get(seg, (None, np.inf))[1]:
                    near[seg] = (team, float(dd))
        if near:
            out.append({'sample': int(samples[i]), 'time_s': float(p.time_s.iat[i]), 'view_shot': views[i],
                        'x': float(bx[i]), 'y': float(by_[i]), 'dv': float(dv[i]), 'near': near})
    return out

