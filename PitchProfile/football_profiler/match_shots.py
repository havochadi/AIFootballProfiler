"""Shot recognition for ball releases, learned from SoccerNet match labels.

Candidates are releases near the goal a team attacks: the end of a possession
spell, or a fast ball flight towards goal that no detected touch explains (the
ball often blurs at the strike, so the kick itself is missed). Each gets
geometric, ball-flight and broadcast-editing features, and a gradient-boosted
classifier trained on SoccerNet 'Shots on target', 'Shots off target', 'Goal'
and 'Penalty' labels (scripts/train_shot_model.py) decides whether it was a
shot. The labels are training data only: analysing new footage uses nothing
but the video.
"""
from __future__ import annotations

import math
import pickle
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import fieldcal as FC

L, W = FC.PITCH_LENGTH, FC.PITCH_WIDTH
MODEL_FILE = 'shot_classifier.pkl'
FEATURES_VERSION = 2
MAX_RELEASE_DISTANCE_M = 45.0     # releases further from the attacked goal are never shots
FLIGHT_S = .5                     # ball flight used for release speed and heading
AFTER_S = 1.5                     # window for how the released ball travels while in view
CUT_HORIZON_S = 10.0
POST_Y = (W / 2 - 3.66, W / 2 + 3.66)
# Strikes whose kick is not detected as a touch (the ball blurs at contact) but whose flight is seen.
FLIGHT_MIN_SPEED = 10.0           # m/s between consecutive observed samples
FLIGHT_MAX_SPEED = 40.0           # faster steps are ball-path jumps, not flight
FLIGHT_GOAL_M = 35.0
FLIGHT_COVERED_S = 1.5            # a release by the attacking team this recently already explains the flight
SHOOTER_RADIUS_M = 6.0
SHOOTER_LOOKBACK = 3              # samples before the flight searched for the shooter
FEATURES = [
    'from_flight', 'dist_goal', 'angle_goal', 'abs_y',
    'speed_after', 'toward_goal', 'cross_offset', 'seen_after', 'closest_to_goal_line',
    'contact_height', 'blockers', 'keeper_distance', 'spell_duration_s', 'spell_samples',
    'next_gap_s', 'next_same_team', 'next_is_keeper', 'next_dist_goal',
    'following_gap_s', 'following_same_team', 'following_is_keeper', 'following_dist_goal',
    'cut_after_s', 'cuts_after', 'pitch_view_2s', 'pitch_view_later',
]


class Context:
    """Per-half lookups shared by all releases: image cuts, pitch view, ball image position, players."""

    def __init__(self, frames, ball, positions):
        f = frames.sort_values('sample')
        self.times = f.time_s.to_numpy(float)
        self.cut_times = self.times[1:][np.diff(f['shot'].to_numpy()) != 0]
        self.view_cum = np.r_[0, np.cumsum(f.pitch_view.fillna(False).astype(bool).to_numpy())]
        b = ball.drop_duplicates('sample').set_index('sample')
        self.ball_cy = b.cy.to_dict() if 'cy' in b else {}
        self.positions = positions

    def cuts(self, t):
        later = self.cut_times[(self.cut_times > t) & (self.cut_times <= t + CUT_HORIZON_S)]
        return (float(later[0] - t) if len(later) else CUT_HORIZON_S), float(len(later))

    def view_share(self, t0, t1):
        a, b = np.searchsorted(self.times, [t0, t1], side='right')
        return float((self.view_cum[b] - self.view_cum[a]) / (b - a)) if b > a else 0.0


def _in_triangle(px, py, a, b, c):
    def side(p1x, p1y, q, r):
        return (p1x - r[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p1y - r[1])
    d1, d2, d3 = side(px, py, a, b), side(px, py, b, c), side(px, py, c, a)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def _next_features(prefix, s, n, goal_x):
    if n is None:
        return {f'{prefix}_gap_s': np.nan, f'{prefix}_same_team': np.nan, f'{prefix}_is_keeper': 0.0,
                f'{prefix}_dist_goal': np.nan}
    return {f'{prefix}_gap_s': float(n.start_s - s.end_s), f'{prefix}_same_team': float(n.team == s.team),
            f'{prefix}_is_keeper': float(n.role == 'goalkeeper' and n.team != s.team),
            f'{prefix}_dist_goal': float(math.hypot(goal_x - n.x0, W / 2 - n.y0))}


def release_features(s, nxt, following, pos, ctx, sign):
    """Features of the release at the end of spell s.

    nxt is the next spell when it continues the same view within the pass-flight
    limit, following the next spell in any view (either may be None).
    """
    goal_x = L if sign > 0 else 0.0
    x, y = float(s.x1), float(s.y1)
    a, b = np.array([goal_x - x, POST_Y[0] - y]), np.array([goal_x - x, POST_Y[1] - y])
    after = pos[(pos.view_shot == s.view_shot) & (pos['sample'] > s.end_sample) & (pos.time_s <= s.end_s + AFTER_S)]
    seen = after[~after.ball_interpolated.astype(bool)]
    flight = seen[seen.time_s <= s.end_s + FLIGHT_S]
    out = {'dist_goal': math.hypot(goal_x - x, W / 2 - y), 'angle_goal': abs(math.atan2(a[0] * b[1] - a[1] * b[0], a @ b)),
           'abs_y': abs(y - W / 2), 'speed_after': np.nan, 'toward_goal': np.nan, 'cross_offset': np.nan,
           'seen_after': float(len(seen)),
           'closest_to_goal_line': float(np.abs(goal_x - seen.ball_x).min()) if len(seen) else np.nan,
           'contact_height': np.nan, 'blockers': np.nan, 'keeper_distance': np.nan,
           'spell_duration_s': float(s.duration_s), 'spell_samples': float(s.samples)}
    if len(flight):
        d = np.array([flight.ball_x.iat[-1] - x, flight.ball_y.iat[-1] - y])
        out['speed_after'] = float(np.hypot(*d) / max(flight.time_s.iat[-1] - s.end_s, 1e-6))
        to_goal = np.array([goal_x - x, W / 2 - y])
        out['toward_goal'] = float(d @ to_goal / (np.hypot(*d) * np.hypot(*to_goal) + 1e-9))
        out['cross_offset'] = float(min(abs(y + (goal_x - x) / d[0] * d[1] - W / 2), 99.0)) if d[0] * sign > 0 else 99.0
    g = ctx.positions.get(int(s.end_sample))
    if g is not None:
        me = g[g.segment == s.segment]
        cy = ctx.ball_cy.get(int(s.end_sample), np.nan)
        if len(me) and np.isfinite(cy) and me.bbox_h.iat[0] > 0:
            # Ball height on the shooter's box: 0 at the head, 1 at the feet (headers vs kicks).
            out['contact_height'] = float((cy - me.bbox_y.iat[0]) / me.bbox_h.iat[0])
        opp = g[g.team.isin(['A', 'B']) & (g.team != s.team) & np.isfinite(g.x)]
        out['blockers'] = float(_in_triangle(opp.x.to_numpy(), opp.y.to_numpy(), (x, y), (goal_x, POST_Y[0]),
                                             (goal_x, POST_Y[1])).sum())
        keeper = opp[opp.role == 'goalkeeper']
        if len(keeper):
            out['keeper_distance'] = float(np.hypot(keeper.x.iat[0] - x, keeper.y.iat[0] - y))
    out.update(_next_features('next', s, nxt, goal_x))
    out.update(_next_features('following', s, following, goal_x))
    # Broadcast editing: shots are followed by cuts to close-ups and replays.
    out['cut_after_s'], out['cuts_after'] = ctx.cuts(s.end_s)
    out['pitch_view_2s'] = ctx.view_share(s.end_s, s.end_s + 2)
    out['pitch_view_later'] = ctx.view_share(s.end_s + 2, s.end_s + CUT_HORIZON_S)
    return out


def flight_releases(pos, sp, positions, team_signs, sample_hz):
    """Pseudo-spells for fast balls towards a goal that no detected touch explains.

    The shooter is the attacking player nearest the ball when its flight starts.
    """
    obs = pos[~pos.ball_interpolated.astype(bool) & np.isfinite(pos.ball_x)].sort_values('sample')
    s, v = obs['sample'].to_numpy(), obs.view_shot.to_numpy()
    x, y, t = obs.ball_x.to_numpy(float), obs.ball_y.to_numpy(float), obs.time_s.to_numpy(float)
    fast = np.zeros(len(obs), bool)
    if len(obs) > 1:
        gap = np.diff(s)
        step = np.hypot(np.diff(x), np.diff(y)) * sample_hz / np.maximum(gap, 1)
        fast[1:] = (gap <= 2) & (v[1:] == v[:-1]) & (step >= FLIGHT_MIN_SPEED) & (step <= FLIGHT_MAX_SPEED)
    out, i = [], 1
    while i < len(obs):
        if not fast[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(obs) and fast[j + 1]:
            j += 1
        start = i - 1                                   # last observed position before the flight
        heading_x = x[j] - x[start]
        for team, sign in team_signs.items():
            goal_x = L if sign > 0 else 0.0
            if not sign or heading_x * sign <= 0 or math.hypot(goal_x - x[start], W / 2 - y[start]) > FLIGHT_GOAL_M:
                continue
            recent = sp[(sp.team == team) & (sp.view_shot == v[start]) & (sp.end_s >= t[start] - FLIGHT_COVERED_S) &
                        (sp.end_s <= t[start] + .2)]
            if len(recent):
                continue
            best = None
            for sample in range(int(s[start]) - SHOOTER_LOOKBACK, int(s[start]) + 1):
                g = positions.get(sample)
                if g is None:
                    continue
                mine = g[(g.team == team) & np.isfinite(g.x)]
                if len(mine):
                    d = np.hypot(mine.x.to_numpy() - x[start], mine.y.to_numpy() - y[start])
                    k = int(np.argmin(d))
                    if d[k] <= SHOOTER_RADIUS_M and (best is None or d[k] < best[0]):
                        best = (d[k], mine.iloc[k])
            if best is not None:
                p = best[1]
                out.append(SimpleNamespace(segment=p.segment, team=team, role=p.role, view_shot=v[start],
                                           start_sample=int(s[start]), end_sample=int(s[start]), start_s=t[start],
                                           end_s=t[start], duration_s=0.0, x0=x[start], y0=y[start], x1=x[start],
                                           y1=y[start], samples=0))
        i = j + 1
    return out


def release_table(sp, pos, ctx, signs, max_flight_s, team_signs=None, sample_hz=None):
    """One feature row per candidate release.

    'spell' is the row position in sp for spell ends, -1 for unexplained flights
    (team_signs and sample_hz enable those).
    """
    spells = list(sp.itertuples())
    starts = sp.start_s.to_numpy(float)
    items = [(i, s, signs[i]) for i, s in enumerate(spells)]
    if team_signs is not None:
        items += [(-1, f, team_signs[f.team]) for f in flight_releases(pos, sp, ctx.positions, team_signs, sample_hz)]
    rows = []
    for i, s, sign in items:
        if not sign or math.hypot((L if sign > 0 else 0.0) - s.x1, W / 2 - s.y1) > MAX_RELEASE_DISTANCE_M:
            continue
        k = i + 1 if i >= 0 else int(np.searchsorted(starts, s.end_s, side='right'))
        following = spells[k] if k < len(spells) else None
        nxt = following if (following is not None and following.view_shot == s.view_shot and
                            following.start_s - s.end_s <= max_flight_s) else None
        rows.append({'spell': i, 'time_s': float(s.end_s), 'segment': s.segment, 'team': s.team, 'sign': int(sign),
                     'x': float(s.x1), 'y': float(s.y1), 'view_shot': int(s.view_shot), 'from_flight': float(i < 0),
                     **release_features(s, nxt, following, pos, ctx, sign)})
    return pd.DataFrame(rows)


def model():
    """{'model', 'threshold', ...} from the weights folder, or None when no usable shot model is trained."""
    from .football_models import weights_dir
    path = weights_dir() / MODEL_FILE
    if not path.is_file():
        return None
    return _load(str(path), path.stat().st_mtime_ns)


def model_for(match_id):
    """The shot model to analyse one match with.

    A match used in training gets the cross-validation model that never saw its
    labels, so its statistics are as honest as those of new footage.
    """
    saved = model()
    if saved is None:
        return None
    fold = saved.get('fold_models', {}).get(str(match_id or '').removeprefix('soccernet:'))
    return {**saved, 'model': fold, 'cross_fitted': True} if fold is not None else saved


@lru_cache(maxsize=2)
def _load(path, modified):
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    if saved.get('features') != FEATURES or saved.get('features_version') != FEATURES_VERSION:
        return None
    return saved


def predict(saved, table):
    """Shot probability per candidate row."""
    if not len(table):
        return np.zeros(0)
    return saved['model'].predict_proba(table[FEATURES].to_numpy(float))[:, 1]


def select_shots(table, threshold, merge_s=1.0):
    """Candidates called shots; of same-team calls within merge_s only the most probable is kept."""
    if not len(table):
        return table
    hits = table[table.probability >= threshold].sort_values('probability', ascending=False)
    kept = []
    for r in hits.itertuples():
        if not any(k.team == r.team and abs(k.time_s - r.time_s) <= merge_s for k in kept):
            kept.append(r)
    return hits.loc[[k.Index for k in kept]].sort_values('time_s')


def shot_fields(r):
    """Event fields of a candidate the model called a shot."""
    out = {'distance_to_goal_m': float(r['dist_goal']), 'probability': round(float(r['probability']), 3),
           'evidence': 'model_flight' if r['from_flight'] else 'model'}
    if np.isfinite(r['speed_after']):
        out['speed_ms'] = float(r['speed_after'])
    return out
