"""Run the grouping-and-naming identity step on SoccerNet tracking clips and score it (in domain).

Pieces of the true tracks (PIECE_S seconds) from every clip of one game half stand in for the
pipeline's segments (clips are separate moments of the half, like segments between camera cuts).
Each piece gets the identity model's evidence and the legacy reader's reading; match_identity.
resolve_clusters then groups and names them per side, over a grid of settings. Scored against the
true shirt numbers, weighted by time: named right, named wrong, grouped into any player, and the
purity of unnamed players. This uses true tracks, so it measures grouping and naming, not tracking.
Per-piece readings are cached in SoccerNet/sn-tracking-2023/pieces_cache.npz (with --weights, a
trial identity-model file beside the adopted one, in pieces_cache_<name>.npz, reusing the legacy
readings).
Report: evidence/identity_sn_simulation.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\simulate_identity_sn_tracking.py [--weights model_trial.pt]
"""
from __future__ import annotations

import itertools
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_identity_sn_tracking as T  # noqa: E402
from football_profiler import identity_model as IM  # noqa: E402
from football_profiler import match_identity as MI  # noqa: E402
from football_profiler import soccernet as SN  # noqa: E402
from football_profiler import storage as S  # noqa: E402

GRID = {'strategy': ['group'], 'naming': ['votes'], 'vote_share': [.8, .85, .9], 'vote_agreed': [.3],
        'name_after_merge': [False, True], 'min_player_s': [10.0]}


def pieces_cache(trial=None):
    main = SN.root() / 'sn-tracking-2023' / 'pieces_cache.npz'
    cache = main if not trial else main.with_name(f'pieces_cache_{Path(trial).stem}.npz')
    from football_profiler import jersey as J
    kept = None
    if trial and main.is_file():
        kept = np.load(main, allow_pickle=True)['legacy'].item()     # legacy readings do not depend on the weights
    if cache.is_file() and cache.stat().st_mtime >= IM.weights_path().stat().st_mtime:
        z = np.load(cache, allow_pickle=True)
        meta, evidence, legacy = z['meta'].item(), z['evidence'].item(), z['legacy'].item()
        if not evidence or 'votes' in next(iter(evidence.values())) or not J.available():
            return meta, evidence, legacy
        kept = legacy                       # evidence from before legible-thumbnail votes: re-read it only
    z = zipfile.ZipFile(SN.root() / 'sn-tracking-2023' / 'test.zip')
    meta, imgs = {}, {}
    for k, clip in enumerate(T.clips(z), 1):
        info, crops, game, half, start_s = T.clip_data(z, clip)
        for c in crops:
            m = info[c['id']]
            piece = int(c['frame'] / (T.PIECE_S * T.FPS))
            key = f"{clip}:{c['id']}:{piece}"
            if key not in meta:
                meta[key] = {'game': game, 'half': half, 'side': m['side'], 'number': m['number'], 'clip': clip,
                             't0': start_s + piece * T.PIECE_S, 't1': start_s + (piece + 1) * T.PIECE_S}
                imgs[key] = []
            imgs[key].append(c['img'])
        print(f'[{k}] {clip}', flush=True)
    evidence = IM.evidence(imgs)
    legacy = kept if kept is not None else J.read_groups(imgs) if J.available() else {}
    np.savez_compressed(cache, meta=np.array(meta, dtype=object), evidence=np.array(evidence, dtype=object),
                        legacy=np.array(legacy, dtype=object))
    return meta, evidence, legacy


def run(meta, evidence, legacy, params, cache):
    rows = []
    for (game, half), keys in pd.Series(list(meta)).groupby([pd.Series([meta[k]['game'] for k in meta]),
                                                              pd.Series([meta[k]['half'] for k in meta])]):
        keys = [k for k in keys if k in evidence and meta[k]['side'] in ('left', 'right')]
        seg = pd.DataFrame({'team': ['A' if meta[k]['side'] == 'left' else 'B' for k in keys], 'role': 'player',
                            'start_s': [meta[k]['t0'] for k in keys], 'end_s': [meta[k]['t1'] for k in keys],
                            'n': [evidence[k]['crops'] for k in keys], 'yn': 34.0, 'depth': .5}, index=keys)
        seg['duration_s'] = seg.end_s - seg.start_s
        identity, _ = MI.resolve_model(seg, {k: evidence[k] for k in keys}, params,
                                       cache=cache.setdefault((game, half), {}), legacy=legacy)
        for k in keys:
            m, ident = meta[k], identity.get(k)
            team, number, kind = (None, None, 'none')
            if ident and '-X' in ident:
                kind = 'unnamed'
            elif ident:
                team, n = ident.split('-', 1)
                number, kind = (int(n), 'named') if n.isdigit() else (None, 'other')
            rows.append({'player': (game, half, m['side'], m['number'] if m['number'] is not None else k),
                         'known': m['number'] is not None, 'truth': m['number'], 'identity': ident, 'kind': kind,
                         'number': number, 'duration': m['t1'] - m['t0'], 'group': (game, half, ident)})
    r = pd.DataFrame(rows)
    groups_per_player = r[r.known & r.identity.notna()].groupby('player').identity.nunique().mean()
    k = r[r.known]
    w = k.duration.sum()
    named = k[k.kind.eq('named')]
    unnamed = r[r.kind.eq('unnamed')]
    purity = (unnamed.groupby('group').apply(lambda g: g.groupby('player').duration.sum().max()).sum() /
              max(unnamed.duration.sum(), 1e-9)) if len(unnamed) else None
    return {'named_correct': float(named[named.number == named.truth].duration.sum() / w),
            'named_wrong': float(named[named.number != named.truth].duration.sum() / w),
            'grouped': float(r[r.kind.isin(['named', 'unnamed'])].duration.sum() / r.duration.sum()),
            'unnamed_purity': float(purity) if purity is not None else None,
            'groups_per_player': float(groups_per_player)}


def main():
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument('--weights', default=None, help='trial identity-model weights file beside the adopted one')
    args = a.parse_args()
    if args.weights:
        IM.WEIGHTS = str(Path(IM.WEIGHTS).parent / args.weights)
        IM.model.cache_clear()
    meta, evidence, legacy = pieces_cache(args.weights)
    print(f'{len(meta)} pieces', flush=True)
    cache, results = {}, []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        res = run(meta, evidence, legacy, params, cache)
        results.append({'params': params, **{k: round(v, 3) if v is not None else None for k, v in res.items()}})
    table = pd.DataFrame(results)
    table['precision'] = (table.named_correct / (table.named_correct + table.named_wrong)).round(3)
    best = table.sort_values('named_correct', ascending=False).drop_duplicates(subset=['precision'])
    print(table.sort_values(['precision', 'named_correct'], ascending=False).head(20).to_string())
    S.write_json(S.EVIDENCE / 'identity_sn_simulation.json', {'created': S.now(), 'pieces': len(meta),
                                                             'piece_s': T.PIECE_S, 'results': results})


if __name__ == '__main__':
    main()
