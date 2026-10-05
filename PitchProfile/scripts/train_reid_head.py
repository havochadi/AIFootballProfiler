"""Train the player re-identification head (football_profiler.reid) on shirt-number pseudo-labels.

Labels come from this project's own analyses: thumbnails of segments whose shirt number was read
reliably (and match the player's identity) are grouped per (half, team, number). Frozen CLIP
features pass through a small head trained with a supervised-contrastive loss, one half per batch
so team-mates in the same kit are the negatives.

Evaluation hides each numbered segment of the official SoccerNet validation and test matches in
turn and matches it to the other segments' players (never overlapping it in time). The margin
threshold for attaching unnumbered segments is the smallest at which the held-out accuracy of
the attached segments reaches TARGET_ACCURACY. Report: evidence/reid_head.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\train_reid_head.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import match_pipeline as MP
from football_profiler import reid as R
from football_profiler import soccernet as SN
from football_profiler import storage as S

CACHE = S.DATA / 'models' / 'reid_features'
TARGET_ACCURACY = .90
STEPS, IDS_PER_BATCH, CROPS_PER_ID = 3000, 24, 8


def halves():
    out = []
    for item in SN.library():
        ident = MP.soccernet_identifier(item['game'], item['half'])
        if (S.dataset_dir(ident) / 'identities.json').is_file():
            out.append((ident, item['game'], item['benchmark_split']))
    return out


def labelled_segments(ident):
    d = S.dataset_dir(ident)
    info = S.read_json(d / 'identities.json')
    tracks = S.load_tracks(ident)
    span = tracks.groupby('track_id').time_s.agg(['min', 'max'])
    rows = []
    for seg, v in info['segments'].items():
        j = v.get('jersey') or {}
        team = v['identity'].split('-')[0]
        # Only segments whose own reading is the player's number (not ones attached to him).
        if j.get('number') is None or v['identity'] != f"{team}-{int(j['number'])}" or seg not in span.index:
            continue
        rows.append({'segment': seg, 'team': team, 'number': int(j['number']), 't0': span.loc[seg, 'min'],
                     't1': span.loc[seg, 'max'], 'tracks': info['segment_tracks'].get(seg, [])})
    return pd.DataFrame(rows)


def features(ident):
    path = CACHE / f'{ident}.npz'
    if path.is_file():
        z = np.load(path, allow_pickle=True)
        return pd.DataFrame(z['meta'].tolist()), z['feat']
    seg = labelled_segments(ident)
    wanted = {t: r for r in seg.itertuples() for t in r.tracks}
    meta, imgs = [], []
    with zipfile.ZipFile(S.dataset_dir(ident) / 'crops.zip') as z:
        for name in z.namelist():
            r = wanted.get(name.split('/')[0])
            if r is None:
                continue
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)
                meta.append({'segment': r.segment, 'team': r.team, 'number': r.number, 't0': r.t0, 't1': r.t1})
    feat = R.clip_features(imgs)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(meta, dtype=object), feat=feat)
    return pd.DataFrame(meta), feat


def supcon(z, labels, tau=.1):
    sim = z @ z.T / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim.masked_fill(eye, -1e9)
    logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
    has = pos.any(1)
    return -(logp * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


def leave_one_out(meta, emb):
    """(correct, margin, similarity, duration, crops) per numbered segment with its number hidden."""
    seg_emb = {}
    for s, g in meta.groupby('segment'):
        m = emb[g.index].mean(0)
        seg_emb[s] = m / (np.linalg.norm(m) + 1e-9)
    crops = meta.groupby('segment').size()
    seg = meta.drop_duplicates('segment').set_index('segment')
    res = []
    for _, t in seg.groupby('team'):
        for s, r in t.iterrows():
            scores = {}
            for number, g in t.drop(index=s).groupby('number'):
                if ((g.t0 <= r.t1) & (r.t0 <= g.t1)).any():
                    continue
                proto = np.mean([seg_emb[k] for k in g.index], axis=0)
                scores[number] = float(seg_emb[s] @ (proto / (np.linalg.norm(proto) + 1e-9)))
            if len(scores) < 2 or r.number not in scores:
                continue
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            res.append((ranked[0][0] == r.number, ranked[0][1] - ranked[1][1], ranked[0][1], r.t1 - r.t0, int(crops[s])))
    return pd.DataFrame(res, columns=['correct', 'margin', 'similarity', 'duration', 'crops'])


def main():
    data = {}
    for ident, game, split in halves():
        meta, feat = features(ident)
        if len(meta):
            data[ident] = (meta.reset_index(drop=True), feat, split)
            print(ident, split, len(meta), 'crops', meta.segment.nunique(), 'segments', flush=True)
    train = [k for k, v in data.items() if v[2] == 'train']
    held = [k for k, v in data.items() if v[2] != 'train']
    torch.manual_seed(0)
    _, _, _, device = R.backbone()
    head = R.head_module().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for step in range(STEPS):
        meta, feat, _ = data[train[rng.integers(len(train))]]
        labels = (meta.team + '-' + meta.number.astype(str)).to_numpy()
        names = np.unique(labels)
        ids = np.sort(rng.choice(names, size=min(IDS_PER_BATCH, len(names)), replace=False))
        idx = np.concatenate([rng.choice(np.flatnonzero(labels == i), size=CROPS_PER_ID, replace=True) for i in ids])
        z = head(torch.from_numpy(feat[idx]).to(device))
        loss = supcon(z, torch.from_numpy(np.searchsorted(ids, labels[idx])).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f'step {step} loss {loss.item():.3f}', flush=True)
    head.eval()
    project = lambda f: head(torch.from_numpy(f).to(device)).detach().cpu().numpy()
    results = {}
    for name, fn in (('clip_frozen', lambda f: f), ('clip_with_head', project)):
        r = pd.concat([leave_one_out(data[k][0], fn(data[k][1])) for k in held], ignore_index=True)
        results[name] = r
    r = results['clip_with_head']
    table, chosen = [], None
    for thr in np.round(np.arange(0, .2, .005), 3):
        sel = r[r.margin >= thr]
        if not len(sel):
            break
        acc = float(np.average(sel.correct, weights=sel.duration))
        table.append({'min_margin': float(thr), 'attached_share': float(sel.duration.sum() / r.duration.sum()),
                      'accuracy': acc})
        if chosen is None and acc >= TARGET_ACCURACY:
            chosen = float(thr)
    few = r[r.crops <= 3]
    report = {
        'created': S.now(), 'backbone': R.BACKBONE, 'train_halves': train, 'held_out_halves': held,
        'labels': 'shirt-number pseudo-labels from this project (segments whose own reading is the player number)',
        'held_out_segments': int(len(r)),
        'accuracy_all': {k: float(np.average(v.correct, weights=v.duration)) for k, v in results.items()},
        'target_accuracy': TARGET_ACCURACY, 'min_margin': chosen, 'threshold_table': table,
        'segments_with_3_or_fewer_crops': {
            'n': int(len(few)),
            'accuracy_all': float(np.average(few.correct, weights=few.duration)) if len(few) else None,
            'accuracy_at_min_margin': float(np.average(few[few.margin >= chosen].correct, weights=few[few.margin >= chosen].duration))
            if chosen is not None and (few.margin >= chosen).any() else None},
    }
    S.write_json(S.EVIDENCE / 'reid_head.json', report)
    torch.save(head.state_dict(), R.weights_path())
    print({k: v for k, v in report.items() if k not in ('threshold_table', 'train_halves', 'held_out_halves')})


if __name__ == '__main__':
    main()
