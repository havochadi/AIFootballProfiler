"""Full-match analysis, stage 2: turn raw detections into pitch-space evidence.

Everything here is CPU-only and deterministic, so it can be re-run on the raw
files written by match_analysis.detection_pass without touching the GPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fieldcal as FC
from . import football_models as FM
from . import storage as S

H_COLUMNS = [f'h{k}' for k in range(9)]
PITCH_VIEW_MIN_SHARE = .5     # share of calibration attempts that must succeed in a shot
MAX_INTERPOLATION_S = 1.2     # calibration gaps bridged inside a shot
MAX_HOLD_S = .1               # one-sided reuse: a single neighbouring sample (fast zooms break longer holds)
SMOOTH_WINDOW = 5             # samples; centred moving average of where fixed image points land
JUMP_M = 12.0                 # projected image-centre jump that reveals an undetected cut
PITCH_MARGIN = 3.0            # metres outside the lines still accepted (throw-ins, keepers)
BALL_MAX_SPEED = 38.0         # m/s, above a hard-struck shot
BALL_GAP_SAMPLES = 8          # detection gaps bridged in the ball path search
BALL_INTERPOLATE_S = .8


OUTLIER_WINDOW = 7            # fitted frames in the running median used to reject wrong fits
OUTLIER_FLOOR_M = 1.0         # a fit this close to its neighbours' median is always kept


def reject_outlier_fits(f, frame_shape):
    """Drop fitted calibrations that disagree with their neighbours in the same shot.

    The broadcast camera moves smoothly, so where fixed image points land on the
    pitch changes smoothly too. A fit whose landing points leave the running
    median of its neighbours is a wrong keypoint solution, not camera motion.
    """
    valid = f.calibration_valid.astype(bool).to_numpy().copy()
    Hs = f[H_COLUMNS].to_numpy(float).reshape(-1, 3, 3)
    for _, rows in f.groupby('shot').groups.items():
        idx = np.asarray(sorted(rows))
        fitted = idx[valid[idx]]
        if len(fitted) < 3:
            continue
        pts, ok = zip(*(FC.grid_on_pitch(Hs[i], frame_shape) for i in fitted))
        pts = np.where(np.stack(ok)[..., None], np.stack(pts), np.nan).reshape(len(fitted), -1)
        median = pd.DataFrame(pts).rolling(OUTLIER_WINDOW, center=True, min_periods=1).median().to_numpy()
        diff = np.abs(pts - median)
        # A fit with no probe point on the ground is unusable: infinite deviation.
        deviation = np.where(np.isnan(diff).all(1), np.inf, np.nanmax(np.where(np.isnan(diff), -np.inf, diff), axis=1))
        mad = np.median(deviation[np.isfinite(deviation)]) * 1.4826 if np.isfinite(deviation).any() else 0.0
        bad = ~(deviation <= max(OUTLIER_FLOOR_M, 4 * mad))
        valid[fitted[bad]] = False
    return valid


def fill_calibration(frames, frame_shape=(720, 1280)):
    """Reject, interpolate, smooth and validate per-frame homographies inside each shot.

    Returns a copy of frames with h0..h8 filled where usable, plus columns
    pitch_view (bool), calibration_source ('fit', 'interpolated', 'held') and
    view_shot, a shot id that also splits at calibration discontinuities.
    """
    f = frames.sort_values('sample').reset_index(drop=True).copy()
    kept = reject_outlier_fits(f, frame_shape)
    f['calibration_rejected'] = f.calibration_valid.astype(bool) & ~kept
    f['calibration_source'] = np.where(kept, 'fit', None)
    f['pitch_view'] = False
    Hs = f[H_COLUMNS].to_numpy(float).reshape(-1, 3, 3)
    times = f.time_s.to_numpy()
    for _, rows in f.groupby('shot').groups.items():
        idx = np.asarray(sorted(rows))
        attempted = f.loc[idx, 'calibration_attempted'].astype(bool)
        fitted = idx[kept[idx]]
        if attempted.sum() == 0 or len(fitted) < 2 or len(fitted) / attempted.sum() < PITCH_VIEW_MIN_SHARE:
            continue
        for i in idx:
            if kept[i]:
                continue
            before, after = fitted[fitted < i], fitted[fitted > i]
            a = before[-1] if len(before) else None
            b = after[0] if len(after) else None
            if a is not None and b is not None and times[b] - times[a] <= MAX_INTERPOLATION_S:
                H = FC.interpolate(Hs[a], Hs[b], (times[i] - times[a]) / (times[b] - times[a]), frame_shape)
                if H is not None:
                    Hs[i] = H
                    f.at[i, 'calibration_source'] = 'interpolated'
                    continue
            near = [j for j in (a, b) if j is not None and abs(times[j] - times[i]) <= MAX_HOLD_S]
            if near:
                Hs[i] = Hs[min(near, key=lambda j: abs(times[j] - times[i]))]
                f.at[i, 'calibration_source'] = 'held'
        usable = idx[f.loc[idx, 'calibration_source'].notna().to_numpy()]
        if len(usable) < 2:
            continue
        # Smooth where fixed image points land on the pitch, then refit each frame.
        pts, ok = zip(*(FC.grid_on_pitch(Hs[i], frame_shape) for i in usable))
        pts, ok = np.stack(pts), np.stack(ok)
        flat = np.where(ok[..., None], pts, np.nan).reshape(len(usable), -1)
        smooth = pd.DataFrame(flat).rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().to_numpy()
        smooth = smooth.reshape(pts.shape)
        for k, i in enumerate(usable):
            H = FC.refit(smooth[k], ok[k] & np.isfinite(smooth[k]).all(1), frame_shape)
            if H is not None:
                Hs[i] = H
        f.loc[usable, 'pitch_view'] = True
    f[H_COLUMNS] = Hs.reshape(-1, 9)
    f.loc[~f.pitch_view, H_COLUMNS] = np.nan
    f.loc[~f.pitch_view, 'calibration_source'] = None
    f['view_shot'] = _split_on_jumps(f, frame_shape)
    return f


def _split_on_jumps(f, frame_shape):
    """Shot ids that also break where the calibrated view jumps (undetected cuts)."""
    h, w = frame_shape[:2]
    centre = np.full((len(f), 2), np.nan)
    ok = f.pitch_view.to_numpy()
    Hs = f[H_COLUMNS].to_numpy(float).reshape(-1, 3, 3)
    for i in np.flatnonzero(ok):
        centre[i] = FC.to_pitch([[w / 2, h * .65]], Hs[i])[0]
    ids, current, prev = [], 0, None
    for i, shot in enumerate(f.shot.to_numpy()):
        if prev is not None and (shot != f.shot.iat[prev] or
                                 (ok[i] and ok[prev] and np.linalg.norm(centre[i] - centre[prev]) > JUMP_M)):
            current += 1
        ids.append(current)
        prev = i
    return ids


def project_people(people, frames):
    """Pitch coordinates (stadium orientation) of each tracked person's feet."""
    p = people.merge(frames[['sample', 'time_s', 'view_shot', 'pitch_view', *H_COLUMNS]], on='sample', how='left')
    p['x'], p['y'] = np.nan, np.nan
    view = p.pitch_view.fillna(False).astype(bool).to_numpy()
    if view.any():
        feet = np.c_[p.bbox_x + p.bbox_w / 2, p.bbox_y + p.bbox_h][view]
        Hs = p.loc[view, H_COLUMNS].to_numpy(float).reshape(-1, 3, 3)
        xy = np.array([FC.to_pitch([pt], H)[0] for pt, H in zip(feet, Hs)])
        p.loc[view, ['x', 'y']] = xy
    p['on_pitch'] = p.x.between(-PITCH_MARGIN, FC.PITCH_LENGTH + PITCH_MARGIN) & \
        p.y.between(-PITCH_MARGIN, FC.PITCH_WIDTH + PITCH_MARGIN)
    # A tracklet split by an undetected cut gets separate ids per view segment.
    p['tracklet'] = p.track + '-v' + p.view_shot.fillna(-1).astype(int).astype(str)
    return p.drop(columns=H_COLUMNS)


def tracklets(p, kits):
    """One row per tracklet with role vote, kit colour and pitch summary."""
    g = p.groupby('tracklet')
    t = pd.DataFrame({
        'track': g.track.first(), 'view_shot': g.view_shot.first(),
        'first_sample': g['sample'].min(), 'last_sample': g['sample'].max(), 'observations': g.size(),
        'start_s': g.time_s.min(), 'end_s': g.time_s.max(), 'confidence': g.conf.mean(),
        'box_height': g.bbox_h.median(), 'on_pitch_share': g.on_pitch.mean(),
        'pitch_share': g.x.apply(lambda s: float(np.isfinite(s).mean())),
        'mean_x': g.x.mean(), 'mean_y': g.y.mean()})
    votes = p.groupby(['tracklet', 'cls']).size().unstack(fill_value=0)
    for cls in FM.PEOPLE:
        if cls not in votes:
            votes[cls] = 0
    t['role'] = votes[list(FM.PEOPLE)].idxmax(axis=1).map(FM.ROLE_NAMES)
    t['role_share'] = votes[list(FM.PEOPLE)].max(axis=1) / votes[list(FM.PEOPLE)].sum(axis=1)
    colours = {k: np.asarray(v, float) for k, v in kits.items()}
    lab = t.track.map(lambda k: np.median(colours[k], axis=0) if k in colours and len(colours[k]) else None)
    t['kit_L'] = lab.map(lambda v: v[0] if v is not None else np.nan)
    t['kit_a'] = lab.map(lambda v: v[1] if v is not None else np.nan)
    t['kit_b'] = lab.map(lambda v: v[2] if v is not None else np.nan)
    t['kit_samples'] = t.track.map(lambda k: len(colours.get(k, [])))
    return t


def assign_teams(t, seed=0):
    """Two kit clusters for outfield players; referees and keepers keep their role."""
    t = t.copy()
    t['team'] = None
    t['team_confidence'] = np.nan
    t.loc[t.role.eq('referee'), 'team'] = 'referee'
    # Clusters are fitted on well-sampled outfield tracklets, then every player
    # tracklet with at least one colour sample takes the nearest cluster.
    outfield = t.role.eq('player') & t.kit_samples.ge(3) & t.on_pitch_share.ge(.5)
    X = t.loc[outfield, ['kit_L', 'kit_a', 'kit_b']].to_numpy(float) * [.5, 1, 1]
    weight = np.sqrt(t.loc[outfield, 'observations'].to_numpy(float))
    if len(X) < 6:
        return t, {'status': 'insufficient_colour_observations', 'centres': []}
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(12):
        centres = X[rng.choice(len(X), 2, replace=False, p=weight / weight.sum())]
        for _ in range(50):
            d = np.linalg.norm(X[:, None] - centres[None], axis=2)
            lab = d.argmin(1)
            new = np.array([np.average(X[lab == k], axis=0, weights=weight[lab == k]) if (lab == k).any() else centres[k]
                            for k in range(2)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = float((weight * d.min(1) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, centres, lab, d)
    _, centres, lab, d = best
    # Stable naming: team A is the cluster with the larger colour share.
    order = np.argsort(-np.array([weight[lab == k].sum() for k in range(2)]))
    names = {int(order[0]): 'A', int(order[1]): 'B'}
    coloured = t.role.eq('player') & t.kit_samples.ge(1)
    Y = t.loc[coloured, ['kit_L', 'kit_a', 'kit_b']].to_numpy(float) * [.5, 1, 1]
    d = np.linalg.norm(Y[:, None] - centres[None], axis=2)
    idx = t.index[coloured]
    t.loc[idx, 'team'] = [names[int(k)] for k in d.argmin(1)]
    t.loc[idx, 'team_confidence'] = np.abs(d[:, 0] - d[:, 1]) / (d.sum(1) + 1e-6)
    import cv2
    rgb = cv2.cvtColor(np.clip(centres / [.5, 1, 1], 0, 255).astype(np.uint8)[None], cv2.COLOR_LAB2RGB)[0]
    info = {'status': 'suggestions', 'separation': float(np.linalg.norm(centres[0] - centres[1])),
            'centres': {names[k]: {'lab': (centres[k] / [.5, 1, 1]).round(1).tolist(),
                                   'colour': '#' + ''.join(f'{int(c):02x}' for c in rgb[k])} for k in range(2)}}
    return t, info


def attack_directions(t):
    """Which goal each team defends, from goal-side positioning of its outfield players.

    Defenders stay between the ball and their own goal, so over a half the team
    defending the left goal has the lower mean x even in one-sided matches.
    """
    q = t[t.team.isin(['A', 'B']) & t.mean_x.notna()]
    if q.team.nunique() < 2:
        return {'status': 'unknown'}
    mean = {team: float(np.average(g.mean_x, weights=g.observations)) for team, g in q.groupby('team')}
    left = min(mean, key=mean.get)
    gap = abs(mean['A'] - mean['B'])
    return {'status': 'inferred' if gap >= 1.5 else 'uncertain', 'defends_left': left,
            'attacks_right': left, 'mean_x': mean, 'separation_m': gap}


def assign_keepers(t, directions):
    """A keeper belongs to the team defending the goal he stays near."""
    t = t.copy()
    keepers = t.role.eq('goalkeeper') & t.mean_x.notna()
    if directions.get('status') not in ('inferred', 'uncertain') or not keepers.any():
        return t
    left = directions['defends_left']
    right = 'B' if left == 'A' else 'A'
    t.loc[keepers, 'team'] = np.where(t.loc[keepers, 'mean_x'] < FC.PITCH_LENGTH / 2, left, right)
    t.loc[keepers, 'team_confidence'] = (t.loc[keepers, 'mean_x'] - FC.PITCH_LENGTH / 2).abs() / (FC.PITCH_LENGTH / 2)
    return t


BALL_COLUMNS = ['sample', 'time_s', 'view_shot', 'ball_segment', 'cx', 'cy', 'x', 'y', 'conf', 'source', 'interpolated']
BALL_PITCH_MARGIN = 1.5  # metres beyond the lines; balls on advertising boards project further out
MISSING_COST = 1.0      # per sampled frame without a chosen ball
RESTART_COST = 4.0      # beginning a new trajectory piece inside a shot
EMIT_SCALE = .5         # confidence weight; consistent low-confidence balls still survive
MAX_STEP_PX = 90.0      # image motion per sampled frame (ball plus camera pan), plus slack below


def select_ball(samples, image_xy, conf):
    """Globally cheapest one-ball-per-frame path through candidates of one shot.

    Frames may be skipped (MISSING_COST each). Consecutive picks must be
    motion-consistent in image space; otherwise a restart is paid. Returns the
    indices of the chosen candidates in time order.
    """
    order = np.lexsort((-conf, samples))
    s, xy, c = samples[order], image_xy[order], conf[order]
    n = len(s)
    if n == 0:
        return np.array([], int)
    emit = -EMIT_SCALE * np.log(np.clip(c, 1e-3, 1))
    first, last = s.min(), s.max()
    cost = np.empty(n)
    back = np.full(n, -1)
    # Running minimum of cost[i] - MISSING_COST * s[i] over strictly earlier frames, for restarts.
    best_restart, best_restart_i = np.inf, -1
    pending = []  # candidates of the current frame, released once the frame advances
    for j in range(n):
        if pending and s[pending[0]] < s[j]:
            for i in pending:
                value = cost[i] - MISSING_COST * s[i]
                if value < best_restart:
                    best_restart, best_restart_i = value, i
            pending = []
        start = MISSING_COST * (s[j] - first)
        cost[j], back[j] = start, -1
        if best_restart_i >= 0:
            restart = best_restart + MISSING_COST * (s[j] - 1) + RESTART_COST
            if restart < cost[j]:
                cost[j], back[j] = restart, best_restart_i
        lo = np.searchsorted(s, s[j] - BALL_GAP_SAMPLES)
        hi = np.searchsorted(s, s[j])
        if hi > lo:
            prev = np.arange(lo, hi)
            gap = s[j] - s[prev]
            dist = np.linalg.norm(xy[prev] - xy[j], axis=1)
            ok = dist <= MAX_STEP_PX * gap + 20
            if ok.any():
                total = cost[prev[ok]] + MISSING_COST * (gap[ok] - 1) + .01 * dist[ok] / gap[ok]
                k = int(np.argmin(total))
                if total[k] < cost[j]:
                    cost[j], back[j] = total[k], prev[ok][k]
        cost[j] += emit[j]
        pending.append(j)
    end = int(np.argmin(cost + MISSING_COST * (last - s)))
    if cost[end] + MISSING_COST * (last - s[end]) >= MISSING_COST * (last - first + 1):
        return np.array([], int)  # choosing nothing is cheaper than any path
    path = []
    while end != -1:
        path.append(end)
        end = back[end]
    return order[np.array(path[::-1])]


def ball_path(ball, frames, sample_hz):
    """Ball trajectory per view segment in image space, projected to the pitch, gaps interpolated.

    ball_segment increases wherever the chosen path restarts (a jump the motion
    limit does not allow) or leaves a long gap: kinematics and events must not
    treat the two sides of such a break as one continuous flight.
    """
    cand = ball.merge(frames[['sample', 'view_shot', 'pitch_view', *H_COLUMNS]], on='sample')
    cand = cand[cand.pitch_view.astype(bool)].reset_index(drop=True)
    if cand.empty:
        return pd.DataFrame(columns=BALL_COLUMNS)
    xy = np.array([FC.to_pitch([[cx, cy]], np.asarray(h, float).reshape(3, 3))[0]
                   for cx, cy, h in zip(cand.cx, cand.cy, cand[H_COLUMNS].to_numpy(float))])
    cand = cand.assign(x=xy[:, 0], y=xy[:, 1])
    m = BALL_PITCH_MARGIN
    cand = cand[cand.x.between(-m, FC.PITCH_LENGTH + m) & cand.y.between(-m, FC.PITCH_WIDTH + m)]
    chosen = []
    for _, c in cand.groupby('view_shot'):
        picked = select_ball(c['sample'].to_numpy(), c[['cx', 'cy']].to_numpy(float), c.conf.to_numpy(float))
        chosen.append(c.iloc[picked])
    path = pd.concat(chosen) if chosen else pd.DataFrame()
    if path.empty:
        return pd.DataFrame(columns=BALL_COLUMNS)
    path = path.sort_values('sample')
    step = np.r_[np.inf, np.hypot(np.diff(path.cx), np.diff(path.cy))]
    gap = np.r_[np.inf, np.diff(path['sample'])]
    new_view = np.r_[True, path.view_shot.to_numpy()[1:] != path.view_shot.to_numpy()[:-1]]
    brk = new_view | (gap > BALL_GAP_SAMPLES) | (step > MAX_STEP_PX * gap + 20)
    path['ball_segment'] = np.cumsum(brk)
    path['time_s'] = path['sample'].map(frames.set_index('sample').time_s)
    path = path[BALL_COLUMNS[:-1]]
    return _interpolate_ball(path, frames, sample_hz)


def _interpolate_ball(path, frames, sample_hz):
    rows = [path.assign(interpolated=False)]
    for segment, q in path.groupby('ball_segment'):
        s = q['sample'].to_numpy()
        for a, b in zip(range(len(s) - 1), range(1, len(s))):
            gap = s[b] - s[a]
            if 1 < gap <= int(round(BALL_INTERPOLATE_S * sample_hz)):
                missing = np.arange(s[a] + 1, s[b])
                w = (missing - s[a]) / gap
                pa, pb = q.iloc[a], q.iloc[b]
                rows.append(pd.DataFrame({
                    'sample': missing, 'view_shot': pa.view_shot, 'ball_segment': segment,
                    'cx': pa.cx + w * (pb.cx - pa.cx), 'cy': pa.cy + w * (pb.cy - pa.cy),
                    'x': pa.x + w * (pb.x - pa.x), 'y': pa.y + w * (pb.y - pa.y),
                    'conf': np.nan, 'source': 'interpolated', 'interpolated': True}))
    out = pd.concat(rows, ignore_index=True).sort_values('sample')
    out['time_s'] = out['sample'].map(frames.set_index('sample').time_s)
    return out.reset_index(drop=True)
