"""Choose and measure the identity-model resolver's settings on FOOTPASS ground truth.

For each FOOTPASS validation half analysed by analyse_footpass.py: the identity model
(football_profiler.identity_model) reads every tracklet's thumbnails once (cached as
identity_evidence.npz), evidence is summed per segment, and match_identity.resolve_clusters runs
over a grid of settings. Each setting is scored on the ground-truth player boxes matched to our
segments (scripts/evaluate_identity_footpass.py):
  named_correct - share of visible player time given the right team and shirt number;
  named_wrong   - share given a wrong number (or wrong team);
  grouped       - share given any player identity (named or a consistent unnamed player);
  unnamed_purity - share of unnamed players' time that belongs to each one's main true player.
Settings are chosen on two games and scored on the third, for each game in turn, so the reported
figures are on games the choice never saw. Report: evidence/identity_clustering.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\tune_identity_clustering.py
"""
from __future__ import annotations

import itertools
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_identity_footpass as E  # noqa: E402  (sets the evaluation data folder)
from football_profiler import identity_model as IM  # noqa: E402
from football_profiler import match_identity as MI  # noqa: E402
from football_profiler import storage as S  # noqa: E402

GRID = {'similarity': [.7, .8, .85, .9], 'number_confidence': [.5, .7],
        'naming': ['model', 'legacy', 'agree', 'either'], 'model_alone': [.8, .9], 'min_player_s': [10.0]}
CACHE = {}                     # grouping per half, team and similarity


def track_evidence(d):
    """{raw track id: evidence}, cached next to the half's thumbnails (rebuilt for new weights)."""
    from football_profiler import match_pipeline as MP
    return MP.track_evidence(d)


def _unused_track_evidence(d):
    import cv2
    cache = d / 'identity_evidence.npz'
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        return z['evidence'].item()
    groups = {}
    with zipfile.ZipFile(d / 'crops.zip') as archive:
        for name in archive.namelist():
            img = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                groups.setdefault(name.split('/')[0], []).append(img)
    ev = {}
    keys = list(groups)
    for i in range(0, len(keys), 400):                       # bounded memory
        ev.update(IM.evidence({k: groups[k] for k in keys[i:i + 400]}))
    np.savez_compressed(cache, evidence=np.array(ev, dtype=object))
    return ev


def half_inputs(d):
    pairs, team_map, _ = E.evaluate_half(d)
    z = np.load(d / 'appearance.npz', allow_pickle=True)
    a = {k: z[k] for k in z.files}
    tev = track_evidence(d)
    seg_ev = {}
    for i, tracks in enumerate(a['tracks']):
        ev = IM.combine([tev.get(t) for t in str(tracks).split('|')])
        if ev is not None:
            seg_ev[i] = ev
    seg = pd.DataFrame({'team': a['team'].astype(str), 'role': a['role'].astype(str), 'start_s': a['start_s'],
                        'end_s': a['end_s'], 'n': a['samples'], 'duration_s': a['end_s'] - a['start_s'],
                        'yn': 34.0, 'depth': .5})
    # The model's goalkeeper output decides the role (as the pipeline now does before events).
    keeper = np.array([seg_ev[i]['keeper'] / max(seg_ev[i]['crops'], 1) if i in seg_ev else 0.0 for i in range(len(seg))])
    seg.loc[keeper >= MI.KEEPER_PROBABILITY, 'role'] = 'goalkeeper'
    seg.loc[(keeper < MI.KEEPER_PROBABILITY) & seg.role.eq('goalkeeper') & (keeper > 0), 'role'] = 'player'
    # Legacy readings (legibility classifier and PARSeq) saved by the baseline post-processing.
    legacy = {i: {'number': int(a['number'][i]), 'confidence': float(a['number_confidence'][i]),
                  'votes': float(a['number_votes'][i])} for i in range(len(seg)) if a['number'][i] >= 0}
    return pairs, team_map, seg, seg_ev, legacy


def score(pairs, team_map, identity):
    det = pairs[pairs.detected].copy()
    names = [identity.get(int(s)) if pd.notna(s) else None for s in det.segment]
    parsed = [E.parse_identity(n) for n in names]
    det['team'] = [t for t, _, _ in parsed]
    det['number'] = [n for _, n, _ in parsed]
    det['name'] = names
    det['kind'] = ['cluster' if isinstance(n, str) and '-X' in n else k for n, (_, _, k) in zip(names, parsed)]
    team_ok = np.array([team_map.get(t) == g for t, g in zip(det.team, det.gt_team)])
    named = det.kind.isin(['number']).to_numpy()
    keeper = det.kind.eq('keeper').to_numpy()
    right = named & team_ok & (det.number.to_numpy() == det.gt_shirt.to_numpy())
    right |= keeper & team_ok & det.gt_keeper.to_numpy()
    n = len(pairs)
    grouped = named | keeper | det.kind.eq('cluster').to_numpy()
    unnamed = det[det.kind.eq('cluster')]
    purity = (unnamed.groupby(['half', 'name']).gt_player.agg(lambda s: s.value_counts().iloc[0]).sum() /
              max(len(unnamed), 1)) if len(unnamed) else None
    return {'named_correct': int(right.sum()) / n, 'named_wrong': int(((named | keeper) & ~right).sum()) / n,
            'grouped': int(grouped.sum()) / n, 'unnamed_purity': purity, 'detected': len(det) / n}


def evaluate(halves, params):
    out = []
    for name, (pairs, team_map, seg, seg_ev, legacy) in halves.items():
        identity, _ = MI.resolve_clusters(seg, seg_ev, params, cache=CACHE.setdefault(name, {}), legacy=legacy)
        pairs = pairs.assign(half=name)
        out.append((len(pairs), score(pairs, team_map, identity)))
    total = sum(n for n, _ in out)
    keys = ['named_correct', 'named_wrong', 'grouped', 'detected']
    res = {k: sum(n * r[k] for n, r in out) / total for k in keys}
    pur = [(n, r['unnamed_purity']) for n, r in out if r['unnamed_purity'] is not None]
    res['unnamed_purity'] = sum(n * v for n, v in pur) / max(sum(n for n, _ in pur), 1) if pur else None
    return res


def choose(halves):
    """Most correctly named time among settings whose named identities are at least 85% right."""
    best = None
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        r = evaluate(halves, params)
        named = r['named_correct'] + r['named_wrong']
        precision = r['named_correct'] / named if named else 0
        key = (precision >= .85, r['named_correct'] + .25 * r['grouped'] * (r['unnamed_purity'] or 0))
        if best is None or key > best[0]:
            best = (key, params, r)
    return best[1], best[2]


def main():
    dirs = [d for d in sorted((S.DATA / 'datasets').glob('fp-val-*'))
            if (d / 'appearance.npz').is_file() and not (S.read_json(d / 'analysis.json', {}).get('footpass') or {}).get('variant')]
    halves = {}
    for d in dirs:
        halves[d.name] = half_inputs(d)
        print(d.name, 'ready', flush=True)
    games = sorted({n.split('-')[2] for n in halves})
    folds = []
    for g in games:
        train = {k: v for k, v in halves.items() if k.split('-')[2] != g}
        test = {k: v for k, v in halves.items() if k.split('-')[2] == g}
        params, fit = choose(train)
        result = evaluate(test, params)
        folds.append({'held_out_game': g, 'params': params, 'result_on_held_out': result, 'fit_on_others': fit})
        print(g, params, {k: round(v, 3) if v is not None else None for k, v in result.items()}, flush=True)
    params, fit = choose(halves)
    report = {'created': S.now(), 'halves': list(halves), 'grid': GRID, 'folds': folds,
              'chosen_on_all': params, 'result_on_all': fit}
    S.write_json(S.EVIDENCE / 'identity_clustering.json', report)
    print('chosen on all games:', params, {k: round(v, 3) if v is not None else None for k, v in fit.items()})


if __name__ == '__main__':
    main()
