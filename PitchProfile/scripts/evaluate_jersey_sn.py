"""Compare shirt-number readers on SoccerNet's jersey-number test set (SN-Jersey-2023).

The set's tracklets are player thumbnails cut from SoccerNet broadcasts (the leagues, seasons and
720p quality of this project's matches), each labelled with the shirt number, or -1 when no
number can be seen. Read straight from the downloaded zip. For each reader, over tracklets with a
visible number: the share it names at each confidence and how often it is right; over tracklets
without one: how often it names a number anyway. Up to MAX_CROPS evenly spaced thumbnails per
tracklet, like the pipeline's thumbnails.

Readers: 'legacy' (football_profiler.jersey: legibility classifier and PARSeq, thresholds of
match_identity), 'identity_model' (football_profiler.identity_model), and their agreement.
Report: evidence/jersey_sn_<tag>.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_jersey_sn.py [--tag current] [--max-tracklets 1211]
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import soccernet as SN  # noqa: E402
from football_profiler import storage as S  # noqa: E402

MAX_CROPS = 30


def load(split='test', max_tracklets=None):
    import cv2
    z = zipfile.ZipFile(SN.root() / 'sn-jersey-2023' / f'{split}.zip')
    gt = json.loads(z.read(f'{split}/{split}_gt.json'))
    names = {}
    for n in z.namelist():
        parts = n.split('/')
        if len(parts) == 4 and parts[1] == 'images' and n.endswith('.jpg'):
            names.setdefault(parts[2], []).append(n)
    keys = sorted(names, key=int)
    if max_tracklets:
        keys = keys[:max_tracklets]
    groups = {}
    for k in keys:
        files = sorted(names[k], key=lambda n: int(n.rsplit('_', 1)[1].split('.')[0]))
        pick = files if len(files) <= MAX_CROPS else [files[i] for i in np.linspace(0, len(files) - 1, MAX_CROPS).astype(int)]
        groups[k] = [img for img in (cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR) for n in pick)
                     if img is not None]
    return groups, {k: int(gt[k]) for k in keys}


def curve(truth, number, confidence, thresholds):
    visible = truth >= 0
    out = []
    for thr in thresholds:
        named = (confidence >= thr) & (number >= 0)
        out.append({'min_confidence': thr,
                    'named_share_of_visible': round(float(named[visible].mean()), 3),
                    'accuracy_when_named': round(float((number[visible & named] == truth[visible & named]).mean()), 3)
                    if (visible & named).any() else None,
                    'named_when_no_number': round(float(named[~visible].mean()), 3) if (~visible).any() else None})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', default='current')
    p.add_argument('--max-tracklets', type=int, default=None)
    args = p.parse_args()
    groups, gt = load('test', args.max_tracklets)
    keys = list(groups)
    truth = np.array([gt[k] for k in keys])
    print(f'{len(keys)} tracklets, {(truth >= 0).mean():.0%} with a visible number', flush=True)
    report = {'created': S.now(), 'tracklets': len(keys), 'visible_share': round(float((truth >= 0).mean()), 3)}
    rows = pd.DataFrame({'tracklet': keys, 'truth': truth})
    from football_profiler import jersey as J
    from football_profiler import match_identity as MI
    if J.available():
        old = J.read_groups(groups)
        num = np.array([(old.get(k) or {}).get('number') if (old.get(k) or {}).get('number') is not None else -1 for k in keys])
        share = np.array([(old.get(k) or {}).get('confidence') or 0.0 for k in keys])
        votes = np.array([(old.get(k) or {}).get('votes') or 0.0 for k in keys])
        trusted = (share >= MI.MIN_NUMBER_SHARE) & (votes >= MI.MIN_NUMBER_VOTES)
        rows['legacy'] = np.where(trusted, num, -1)
        rows['legacy_share'] = share
        report['legacy'] = curve(truth, rows.legacy.to_numpy(), np.where(trusted, 1.0, 0.0), [.5])
        print('legacy', report['legacy'], flush=True)
    from football_profiler import identity_model as IM
    if IM.available():
        ev = IM.evidence(groups)
        read = [IM.reading(ev[k]) if k in ev else (-1, 0.0, 0.0) for k in keys]
        rows['model'] = [r[0] for r in read]
        rows['model_conf'] = [r[1] for r in read]
        report['identity_model'] = curve(truth, rows.model.to_numpy(), rows.model_conf.to_numpy(),
                                         [0, .3, .5, .7, .8, .9])
        print('identity model', report['identity_model'], flush=True)
        if 'legacy' in rows:
            agree = (rows.legacy == rows.model) & (rows.legacy >= 0)
            report['agreement'] = curve(truth, np.where(agree, rows.model, -1), agree.astype(float).to_numpy(), [.5])
            print('both agree', report['agreement'], flush=True)
    rows.to_csv(S.EVIDENCE / f'jersey_sn_{args.tag}.csv', index=False)
    S.write_json(S.EVIDENCE / f'jersey_sn_{args.tag}.json', report)


if __name__ == '__main__':
    main()
