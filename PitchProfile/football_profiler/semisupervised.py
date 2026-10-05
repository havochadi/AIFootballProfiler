"""Semi-supervised archetype profiling over video-derived player statistics.

Each analysed player appearance (one player in one analysed half) is a node.
Features are on-ball and defensive rates per 90 minutes of the player's own
screen time, the mix of on-ball actions per 100 touches, positional shares, a
coarse heatmap and physical measures, standardised across all appearances.
Rates on the player's own time and per touch do not depend on how much of the
player the shirt-number reader could identify. Human
labels are archetype percentages (0-100 per compatible role, several roles per
player). Within each position group, labels spread over a k-nearest-neighbour
graph (label spreading, Zhou et al. 2004), so every unlabelled appearance gets
a similarity-weighted estimate plus a confidence that reflects how strongly it
connects to labelled examples. Unlabelled appearances shape the graph; that is
the semi-supervised part. Grouped-by-match cross-validation compares the result
with a labelled-only nearest-neighbour baseline.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import storage as S
from . import taxonomy as T

ALPHA = .85            # label spreading smoothness
K = 8                  # graph neighbours
MIN_VISIBLE_S = 120    # appearances observed less than this are too noisy to profile
RATE_FEATURES = ['touches', 'passes', 'progressive_passes', 'long_passes', 'crosses', 'key_passes', 'carries',
                 'progressive_carries', 'dribbles', 'shots', 'tackles', 'interceptions', 'recoveries',
                 'clearances', 'pressures', 'dispossessed',
                 'take_ons', 'receptions', 'progressive_receptions', 'receptions_in_box', 'receptions_behind_line',
                 'touches_in_box', 'passes_into_final_third', 'passes_into_box', 'switches', 'passes_behind_line',
                 'carries_into_final_third', 'turnovers', 'blocks', 'headers', 'lofted_passes',
                 'high_intensity_runs', 'runs_in_behind', 'box_runs', 'forward_runs', 'overlaps', 'underlaps',
                 'pressing_runs', 'recovery_runs']
POSITION_FEATURES = ['mean_x', 'mean_y_abs', 'spread_x', 'spread_y', 'defensive_third', 'middle_third',
                     'attacking_third', 'central_lane', 'box_share']
OTHER_FEATURES = ['pass_completion', 'mean_pass_length_m', 'forward_pass_share', 'time_on_ball_per90',
                  'distance_per_min_m', 'top_speed_kmh', 'high_intensity_share', 'short_pass_share',
                  'backward_pass_share', 'take_on_success', 'mean_x_vs_team_m', 'mean_width_m', 'between_lines_share']
COMPOSITION_FEATURES = ['passes', 'progressive_passes', 'long_passes', 'crosses', 'key_passes', 'carries',
                        'progressive_carries', 'dribbles', 'shots', 'dispossessed']
HEAT_BINS = (4, 4)       # lateral x longitudinal cells of the 20 x 32 heatmap


def save_label(dataset_id, player_id, labeler, position_group, labels, notes='', evidence=None):
    """One labeller's archetype percentages for a player appearance (replaces the previous label)."""
    labeler = labeler.strip()
    if not labeler or len(labeler) > 80:
        raise ValueError('Enter your name or labeller ID (up to 80 characters)')
    T.validate_labels(position_group, labels)
    if not any(v is not None for v in labels.values()):
        raise ValueError('Rate at least one role, or leave the player unlabelled')
    stamp = S.now()
    args = (dataset_id, str(player_id), labeler, position_group, json.dumps(labels), notes[:3000], stamp)
    with S.db() as c:
        previous = c.execute('SELECT evidence FROM player_label_evidence WHERE dataset_id=? AND player_id=?',
                             (dataset_id, str(player_id))).fetchone()
        # Older clients do not send evidence; editing their ratings must preserve bookmarks.
        saved_evidence = json.dumps(evidence) if evidence is not None else (previous['evidence'] if previous else '[]')
        history = c.execute('INSERT INTO player_label_history(dataset_id,player_id,labeler,position_group,labels,notes,created) '
                            'VALUES(?,?,?,?,?,?,?)', args)
        c.execute('INSERT INTO player_label_evidence_history VALUES(?,?)', (history.lastrowid, saved_evidence))
        c.execute('INSERT OR REPLACE INTO player_label_evidence VALUES(?,?,?)',
                  (dataset_id, str(player_id), saved_evidence))
        c.execute('INSERT INTO player_labels VALUES(?,?,?,?,?,?,?) ON CONFLICT(dataset_id,player_id) DO UPDATE SET '
                  'labeler=excluded.labeler, position_group=excluded.position_group, labels=excluded.labels, '
                  'notes=excluded.notes, updated=excluded.updated', args)
    return label(dataset_id, player_id)


def delete_label(dataset_id, player_id):
    with S.db() as c:
        c.execute('DELETE FROM player_labels WHERE dataset_id=? AND player_id=?', (dataset_id, str(player_id)))
        c.execute('DELETE FROM player_label_evidence WHERE dataset_id=? AND player_id=?', (dataset_id, str(player_id)))


def labels():
    with S.db() as c:
        rows = [dict(r) for r in c.execute('SELECT p.*, e.evidence FROM player_labels p LEFT JOIN player_label_evidence e '
                                         'ON p.dataset_id=e.dataset_id AND p.player_id=e.player_id')]
    for r in rows:
        r['labels'] = json.loads(r['labels'])
        r['evidence'] = json.loads(r['evidence'] or '[]')
    return rows


def label(dataset_id, player_id):
    return next((r for r in labels() if r['dataset_id'] == dataset_id and r['player_id'] == str(player_id)), None)


def appearances():
    """One feature row per analysed player appearance across all full-match datasets."""
    from . import match_context as MC
    rows = []
    manifests = S.datasets()
    merged = {h for m in manifests for h in (m.get('halves') or [])}
    for m in manifests:
        if not str(m.get('analysis', '')).startswith('full-match') or m['id'] in merged:
            continue
        d = S.dataset_dir(m['id'])
        stats = S.read_json(d / 'match_stats.json', {})
        manifest_players = {p['player_id']: p for p in m['players']}
        context = MC.state_for(m)
        for p in stats.get('players', []):
            info = manifest_players.get(p['identity'])
            if info is None or p.get('visible_seconds', 0) < MIN_VISIBLE_S or p['role'] == 'referee':
                continue
            info = MC.decorate(m, {**info, 'role': p['role']}, context)
            ob, pos, phy = p['on_ball'], p['positional'], p['physical']
            own, mix = p.get('per90_visible') or {}, p.get('per100_touches') or {}
            f = {k: own.get(k, 0.0) for k in RATE_FEATURES}
            f.update({f'share_{k}': mix.get(k, 0.0) for k in COMPOSITION_FEATURES})
            f.update(mean_x=pos.get('mean_x'), mean_y_abs=abs((pos.get('mean_y') or 34) - 34),
                     spread_x=pos.get('spread_x'), spread_y=pos.get('spread_y'),
                     defensive_third=pos.get('defensive_third'), middle_third=pos.get('middle_third'),
                     attacking_third=pos.get('attacking_third'), central_lane=pos.get('central_lane'),
                     box_share=pos.get('box_share'), pass_completion=ob.get('pass_completion'),
                     mean_pass_length_m=ob.get('mean_pass_length_m'), forward_pass_share=ob.get('forward_pass_share'),
                     time_on_ball_per90=ob.get('time_on_ball_s', 0) * 5400 / p['visible_seconds'],
                     distance_per_min_m=phy.get('distance_per_min_m'), top_speed_kmh=phy.get('top_speed_kmh'),
                     high_intensity_share=_high_intensity(phy), short_pass_share=ob.get('short_pass_share'),
                     backward_pass_share=ob.get('backward_pass_share'), take_on_success=ob.get('take_on_success'),
                     mean_x_vs_team_m=(p.get('movement') or {}).get('mean_x_vs_team_m'),
                     mean_width_m=(p.get('movement') or {}).get('mean_width_m'),
                     between_lines_share=(p.get('movement') or {}).get('between_lines_share'))
            heat = np.asarray(pos.get('heatmap') or np.zeros((20, 32)), float)
            coarse = heat.reshape(HEAT_BINS[0], 20 // HEAT_BINS[0], HEAT_BINS[1], 32 // HEAT_BINS[1]).sum((1, 3))
            for i, v in enumerate(coarse.ravel()):
                f[f'heat_{i}'] = float(v)
            rows.append({'dataset_id': m['id'], 'player_id': p['identity'], 'match_id': m.get('match_id', m['id']),
                         'name': info.get('name'), 'team': info.get('team'), 'role': info['role'],
                         'identity_reviewed': bool(info.get('identity_correction') or info.get('identity_status') == 'confirmed'),
                         'suggested_group': info.get('position_group'), 'visible_seconds': p['visible_seconds'], **f})
    return pd.DataFrame(rows)


def _high_intensity(phy):
    z = phy.get('zone_seconds') or {}
    moving = phy.get('moving_seconds') or 0
    return (z.get('high_speed', 0) + z.get('sprint', 0)) / moving if moving else None


def feature_matrix(frame):
    cols = [c for c in frame.columns if c in RATE_FEATURES or c in POSITION_FEATURES or c in OTHER_FEATURES
            or c.startswith('heat_') or c.startswith('share_')]
    X = frame[cols].astype(float).copy()
    for c in RATE_FEATURES + [f'share_{k}' for k in COMPOSITION_FEATURES]:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    X = X.fillna(X.median()).fillna(0)
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
    return ((X - mu) / sd).to_numpy(float), cols


def spread(X, Y, known, alpha=ALPHA, k=K):
    """Label spreading of soft targets Y (n x r, values 0..1) from rows where known is True.

    Returns (prediction n x r, support n): prediction is the graph-weighted
    average of labelled values reaching each node; support is the share of the
    node's propagated mass that came from labels (0 = no labelled neighbours).
    """
    n = len(X)
    if n < 2 or not known.any():
        return np.full(Y.shape, np.nan), np.zeros(n)
    k = min(k, n - 1)
    d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    sigma = np.median(d[np.arange(n)[:, None], nn]) or 1.0
    Wm = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    Wm[rows, nn.ravel()] = np.exp(-(d[rows, nn.ravel()] / sigma) ** 2)
    Wm = np.maximum(Wm, Wm.T)
    deg = Wm.sum(1)
    deg[deg == 0] = 1
    Sm = Wm / np.sqrt(deg[:, None] * deg[None])
    # A role left unrated (NaN) is unknown, not zero: each role spreads its own mass.
    rated = known[:, None] & np.isfinite(Y)
    Y0 = np.where(rated, Y, 0.0)
    M0 = rated.astype(float)
    # Closed form of the iteration F <- aSF + (1-a)Y0, solved for values and mass together.
    A = np.eye(n) - alpha * Sm
    solved = np.linalg.solve(A, (1 - alpha) * np.hstack([Y0, M0]))
    F, mass = solved[:, :Y.shape[1]], solved[:, Y.shape[1]:]
    pred = np.where(mass > 1e-9, F / np.maximum(mass, 1e-9), np.nan)
    support = mass.max(1)
    return pred, support / (support.max() or 1)


def _supervised_knn(X, Y, known, k=5):
    """Labelled-only baseline: mean of the k nearest labelled neighbours."""
    out = np.full(Y.shape, np.nan)
    lab = np.flatnonzero(known)
    if not len(lab):
        return out
    for i in range(len(X)):
        cand = lab[lab != i]
        if not len(cand):
            continue
        near = cand[np.argsort(np.linalg.norm(X[cand] - X[i], axis=1))[:k]]
        out[i] = np.nanmean(Y[near], axis=0)
    return out


def _group_data(frame, group, X, label_rows):
    roles = T.compatible(group)
    idx = np.flatnonzero(frame.group.eq(group).to_numpy())
    Y = np.full((len(idx), len(roles)), np.nan)
    known = np.zeros(len(idx), bool)
    for j, i in enumerate(idx):
        lab = label_rows.get((frame.dataset_id.iat[i], frame.player_id.iat[i]))
        if lab and lab['position_group'] == group:
            known[j] = True
            Y[j] = [np.nan if lab['labels'].get(r) is None else lab['labels'][r] / 100 for r in roles]
    return idx, roles, Y, known


def _assign_groups(frame, X, label_rows):
    """Confirmed group for labelled players; otherwise spread group labels, falling back to the suggestion."""
    groups = list(T.GROUPS)
    Y = np.zeros((len(frame), len(groups)))
    known = np.zeros(len(frame), bool)
    for i, r in enumerate(frame.itertuples()):
        lab = label_rows.get((r.dataset_id, r.player_id))
        if lab:
            Y[i, groups.index(lab['position_group'])] = 1
            known[i] = True
    source = np.array(['suggested'] * len(frame), dtype=object)
    group = frame.suggested_group.fillna('central_midfield').to_numpy(object)
    group[frame.role.eq('goalkeeper').to_numpy()] = 'goalkeeper'
    if known.sum() >= 8:
        pred, support = spread(X, Y, known)
        for i in range(len(frame)):
            if known[i]:
                group[i], source[i] = groups[int(Y[i].argmax())], 'labelled'
            elif support[i] > .15 and frame.role.iat[i] != 'goalkeeper':
                group[i], source[i] = groups[int(np.nanargmax(pred[i]))], 'propagated'
    else:
        for i in range(len(frame)):
            if known[i]:
                group[i], source[i] = groups[int(Y[i].argmax())], 'labelled'
    for i, r in enumerate(frame.itertuples()):
        if getattr(r, 'identity_reviewed', False) and (r.role == 'goalkeeper') != (group[i] == 'goalkeeper'):
            suggested = r.suggested_group if r.suggested_group in T.GROUPS and r.suggested_group != 'goalkeeper' else 'central_midfield'
            group[i] = 'goalkeeper' if r.role == 'goalkeeper' else suggested
            source[i] = 'corrected role'
    return group, source


def fit(evaluate=True):
    """Predict archetype percentages for every appearance; optionally cross-validate."""
    frame = appearances()
    if frame.empty:
        return {'status': 'no_appearances', 'message': 'Analyse full matches first.'}
    X, cols = feature_matrix(frame)
    label_rows = {(r['dataset_id'], r['player_id']): r for r in labels()}
    for r in frame.itertuples():
        key = (r.dataset_id, r.player_id)
        lab = label_rows.get(key)
        if r.identity_reviewed and lab and (r.role == 'goalkeeper') != (lab['position_group'] == 'goalkeeper'):
            del label_rows[key]  # Retain the saved assessment, but exclude conflicting ratings until reviewed.
    frame['group'], frame['group_source'] = _assign_groups(frame, X, label_rows)
    predictions, report = [], {'groups': {}}
    for group in T.GROUPS:
        idx, roles, Y, known = _group_data(frame, group, X, label_rows)
        if not len(idx):
            continue
        Xg = X[idx]
        pred, support = spread(Xg, Y, known) if known.sum() >= 2 else (np.full(Y.shape, np.nan), np.zeros(len(idx)))
        for j, i in enumerate(idx):
            values = pred[j] if not known[j] else Y[j]
            predictions.append({'dataset_id': frame.dataset_id.iat[i], 'player_id': frame.player_id.iat[i],
                                'position_group': group, 'group_source': frame.group_source.iat[i],
                                'labelled': bool(known[j]), 'support': float(support[j]),
                                'roles': {r: (None if not np.isfinite(v) else float(v * 100)) for r, v in zip(roles, values)}})
        info = {'appearances': int(len(idx)), 'labelled': int(known.sum()), 'roles': roles}
        if evaluate and known.sum() >= 6:
            info['cross_validation'] = _cross_validate(frame.iloc[idx], Xg, Y, known)
        report['groups'][group] = info
    result = {'status': 'fitted', 'created': S.now(), 'appearances': int(len(frame)),
              'labelled': int(sum(1 for p in predictions if p['labelled'])), 'features': cols,
              'method': 'Label spreading over a k-nearest-neighbour graph of all appearances in each position group '
                        f'(alpha={ALPHA}, k={K}); groups for unlabelled players propagated from labelled ones.',
              'report': report, 'predictions': predictions}
    S.write_json(S.DATA / 'models' / 'semisupervised.json', result)
    return {k: v for k, v in result.items() if k != 'predictions'}


def _cross_validate(frame, X, Y, known, folds=5):
    """Hide labels match by match; compare label spreading with a labelled-only kNN baseline."""
    matches = frame.match_id.to_numpy()
    labelled_matches = np.unique(matches[known])
    rng = np.random.default_rng(0)
    order = rng.permutation(labelled_matches)
    splits = np.array_split(order, min(folds, len(order)))
    err_ss, err_sup, top_ss, top_sup, n = [], [], [], [], 0
    for held in splits:
        test = known & np.isin(matches, held)
        train = known & ~test
        if not test.any() or train.sum() < 2:
            continue
        pred, _ = spread(X, Y, train)
        base = _supervised_knn(X, Y, train)
        for i in np.flatnonzero(test):
            m = np.isfinite(Y[i])
            if not m.any():
                continue
            n += 1
            err_ss.append(np.nanmean(np.abs(pred[i][m] - Y[i][m])) * 100)
            err_sup.append(np.nanmean(np.abs(base[i][m] - Y[i][m])) * 100)
            if m.sum() > 1:
                top_ss.append(int(np.nanargmax(np.where(m, pred[i], -1)) == np.nanargmax(np.where(m, Y[i], -1))))
                top_sup.append(int(np.nanargmax(np.where(m, base[i], -1)) == np.nanargmax(np.where(m, Y[i], -1))))
    if not n:
        return None
    return {'held_out_appearances': n, 'folds': len(splits),
            'semi_supervised_mae_points': float(np.mean(err_ss)), 'labelled_only_knn_mae_points': float(np.mean(err_sup)),
            'semi_supervised_top_role_agreement': float(np.mean(top_ss)) if top_ss else None,
            'labelled_only_top_role_agreement': float(np.mean(top_sup)) if top_sup else None,
            'note': 'Labels of whole matches are hidden in turn; lower MAE (percentage points) is better.'}


def prediction(dataset_id, player_id):
    fitted = S.read_json(S.DATA / 'models' / 'semisupervised.json')
    if not fitted:
        return {'status': 'not_fitted', 'message': 'Label some players, then fit the semi-supervised model.'}
    with S.db() as db:
        corrections = [json.loads(r['payload']) for r in db.execute(
            'SELECT payload FROM player_track_correction_history WHERE dataset_id=? AND player_id=? '
            'UNION ALL SELECT payload FROM player_identity_link_history WHERE dataset_id=? AND player_id=?',
            (dataset_id, player_id, dataset_id, player_id))]
    if any(r.get('updated', '') > fitted['created'] for r in corrections):
        return {'status': 'identity_changed', 'message': 'Identity corrected since the last model fit. Review this player’s position group and refit to refresh the estimate.'}
    p = next((x for x in fitted['predictions'] if x['dataset_id'] == dataset_id and x['player_id'] == player_id), None)
    if p is None:
        return {'status': 'not_in_model', 'message': 'This appearance was not in the last fit (too little visible time, or analysed afterwards). Refit to include it.'}
    return {'status': 'fitted', 'created': fitted['created'], **p}
