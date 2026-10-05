"""Measure shirt-number reading and appearance matching on SoccerNet tracking clips (in domain).

SoccerNet's tracking test set (SN-Tracking-2023): 30-second broadcast clips with every player's
true track, side and shirt number, several clips per game. Frames are scaled to 720p (this
project's footage) before thumbnails are cut, every STEP frames, padded like the pipeline's.

Numbers: each true track is read by the legacy reader (football_profiler.jersey, match_identity
thresholds) and the identity model (football_profiler.identity_model): share named, and right.
Appearance: tracks are cut into PIECE_S-second pieces; each piece is hidden and matched to its
side's players in the same game half, each player represented by his other pieces that are not
on screen at the same time (the constraint the pipeline uses), with the legacy appearance head
(football_profiler.reid) and with the identity model. Report: evidence/identity_sn_tracking_<tag>.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_identity_sn_tracking.py [--tag current] [--clips 49]
"""
from __future__ import annotations

import argparse
import configparser
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import soccernet as SN  # noqa: E402
from football_profiler import storage as S  # noqa: E402

STEP = 5
PIECE_S = 3.0
FPS = 25.0
SCALE = 720 / 1080
PAD = .08
MIN_HEIGHT = 30


def clips(z):
    out = {}
    for n in z.namelist():
        parts = n.split('/')
        if len(parts) >= 3 and parts[1].startswith('SNMOT-'):
            out.setdefault(parts[1], set()).add(n)
    return sorted(out)


def clip_data(z, clip):
    import cv2
    ini = configparser.ConfigParser()
    ini.read_string(z.read(f'test/{clip}/gameinfo.ini').decode())
    sec = ini['Sequence']
    info = {}
    for k, v in sec.items():
        if k.startswith('trackletid_'):
            parts = [x.strip() for x in v.split(';')]
            role = parts[0]
            number = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            info[int(k.split('_')[1])] = {'side': 'left' if 'left' in role else 'right' if 'right' in role else None,
                                          'player': role.startswith('player') or role.startswith('goalkeeper'),
                                          'number': number}
    gt = pd.read_csv(z.open(f'test/{clip}/gt/gt.txt'), header=None).iloc[:, :6]
    gt.columns = ['frame', 'id', 'x', 'y', 'w', 'h']
    gt = gt[gt.id.map(lambda i: info.get(i, {}).get('player', False)) & (gt.frame % STEP == 0)]
    crops = []
    for f, g in gt.groupby('frame'):
        name = f'test/{clip}/img1/{int(f):06d}.jpg'
        try:
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
        except KeyError:
            continue
        img = cv2.resize(img, (int(img.shape[1] * SCALE), int(img.shape[0] * SCALE)), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]
        for r in g.itertuples():
            x, y, w, h = r.x * SCALE, r.y * SCALE, r.w * SCALE, r.h * SCALE
            if h < MIN_HEIGHT:
                continue
            pad = PAD * h
            a, b = max(0, int(x - pad)), min(W, int(x + w + pad))
            c, e = max(0, int(y - pad)), min(H, int(y + h + pad))
            if b - a >= 4 and e - c >= 8:
                crops.append({'clip': clip, 'id': int(r.id), 'frame': int(r.frame), 'img': img[c:e, a:b].copy()})
    game = sec.get('gameid')
    half = (sec.get('gametimestart') or '1').split('-')[0].strip()
    start_s = int(sec.get('clipstart', 0)) / 1000
    return info, crops, game, half, start_s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', default='current')
    p.add_argument('--clips', type=int, default=None)
    args = p.parse_args()
    z = zipfile.ZipFile(SN.root() / 'sn-tracking-2023' / 'test.zip')
    names = clips(z)[:args.clips] if args.clips else clips(z)
    tracks, pieces = {}, {}
    for k, clip in enumerate(names, 1):
        info, crops, game, half, start_s = clip_data(z, clip)
        for c in crops:
            meta = info[c['id']]
            key = (clip, c['id'])
            tracks.setdefault(key, {'number': meta['number'], 'side': meta['side'], 'game': game, 'half': half,
                                    'imgs': []})['imgs'].append(c['img'])
            piece = int(c['frame'] / (PIECE_S * FPS))
            pk = (clip, c['id'], piece)
            pieces.setdefault(pk, {'player': (game, half, meta['side'], meta['number'] if meta['number'] is not None
                                              else f"{clip}:{c['id']}"),
                                   'side': (game, half, meta['side']), 'clip': clip,
                                   't0': start_s + piece * PIECE_S, 't1': start_s + (piece + 1) * PIECE_S, 'imgs': []})
            pieces[pk]['imgs'].append(c['img'])
        print(f'[{k}/{len(names)}] {clip}: {len(crops)} thumbnails', flush=True)
    report = {'created': S.now(), 'clips': len(names), 'tracks': len(tracks), 'pieces': len(pieces)}
    keys = list(tracks)
    truth = np.array([tracks[k]['number'] if tracks[k]['number'] is not None else -1 for k in keys])
    groups = {i: tracks[k]['imgs'] for i, k in enumerate(keys)}
    from football_profiler import jersey as J
    from football_profiler import match_identity as MI
    from football_profiler import identity_model as IM
    from football_profiler import reid as R

    def number_result(pred):
        visible = truth >= 0
        named = pred >= 0
        return {'named_share': round(float(named[visible].mean()), 3),
                'accuracy_when_named': round(float((pred[visible & named] == truth[visible & named]).mean()), 3)
                if (visible & named).any() else None}
    if J.available():
        old = J.read_groups(groups)
        pred = np.array([o['number'] if o and o.get('number') is not None and o['confidence'] >= MI.MIN_NUMBER_SHARE
                         and o['votes'] >= MI.MIN_NUMBER_VOTES else -1 for o in (old.get(i) for i in range(len(keys)))])
        report['numbers_legacy'] = number_result(pred)
    if IM.available():
        ev = IM.evidence(groups)
        read = [IM.reading(ev[i]) if i in ev else (-1, 0, 0) for i in range(len(keys))]
        for thr in (.5, .7, .85):
            pred = np.array([n if c >= thr else -1 for n, c, _ in read])
            report[f'numbers_identity_model_{thr}'] = number_result(pred)
    print({k: v for k, v in report.items() if k.startswith('numbers')}, flush=True)
    pk = [k for k in pieces if isinstance(pieces[k]['player'][3], int)]
    pgroups = {i: pieces[k]['imgs'] for i, k in enumerate(pk)}
    meta = pd.DataFrame([{k2: pieces[k][k2] for k2 in ('player', 'side', 'clip', 't0', 't1')} for k in pk])
    for name, embed in (('legacy_head', lambda g: R.embed_groups(g) if R.available() else {}),
                        ('identity_model', lambda g: {i: IM.embedding(e) for i, e in IM.evidence(g).items()})):
        emb = embed(pgroups)
        if not emb:
            continue
        report[f'appearance_{name}'] = hidden_match(meta, emb)
        print(name, report[f'appearance_{name}'], flush=True)
    S.write_json(S.EVIDENCE / f'identity_sn_tracking_{args.tag}.json', report)


def hidden_match(meta, emb):
    res = []
    for side, g in meta.groupby('side'):
        idx = [i for i in g.index if i in emb]
        for i in idx:
            r = g.loc[i]
            scores = {}
            for player, q in g[g.index != i].groupby('player'):
                q = q[q.index.isin(idx)]
                q = q[~((q.clip == r.clip) & (q.t0 < r.t1) & (r.t0 < q.t1))]
                if len(q):
                    proto = np.mean([emb[j] for j in q.index], 0)
                    scores[player] = float(emb[i] @ (proto / (np.linalg.norm(proto) + 1e-9)))
            if len(scores) >= 2 and r.player in scores:
                ranked = sorted(scores.items(), key=lambda kv: -kv[1])
                res.append((ranked[0][0] == r.player, ranked[0][1] - ranked[1][1]))
    r = pd.DataFrame(res, columns=['correct', 'margin'])
    out = {'pieces': int(len(r)), 'accuracy_all': round(float(r.correct.mean()), 3), 'by_margin': []}
    for m in (.02, .05, .1):
        sel = r[r.margin >= m]
        out['by_margin'].append({'min_margin': m, 'attached_share': round(len(sel) / max(len(r), 1), 3),
                                 'accuracy': round(float(sel.correct.mean()), 3) if len(sel) else None})
    return out


if __name__ == '__main__':
    main()
