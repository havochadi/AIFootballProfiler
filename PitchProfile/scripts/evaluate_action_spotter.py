"""Evaluate the ball-action spotter on this project's footage and choose score thresholds.

Five of its classes have SoccerNet Labels-v2 counterparts on every local game: THROW IN (Throw-in),
OUT (Ball out of play), SHOT (Shots on/off target), GOAL (Goal) and FREE KICK (Direct/Indirect
free-kick). For those, precision and recall within TOLERANCE_S are measured at a grid of score
thresholds; the threshold maximising F1 on train+validation halves is reported on the held-out
test halves. The spotter's ball-action head never saw these games (it was trained on English
Football League games); its backbone was jointly trained on SoccerNet match events, which include
the official training games, so held-out test halves are the clean check.

Classes without local labels (tackle, block, header, cross, high pass) use the median of the
labelled classes' chosen thresholds; their counts per team and half are reported next to a fully
annotated real match with the same class definitions (SoccerTrack v2 117092) for plausibility.
Report: evidence/action_spotter.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_action_spotter.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import action_spotting as AS
from football_profiler import match_pipeline as MP
from football_profiler import soccernet as SN
from football_profiler import storage as S

TOLERANCE_S = 2.0
LABELS = {'THROW IN': ['Throw-in'], 'OUT': ['Ball out of play'], 'SHOT': ['Shots on target', 'Shots off target'],
          'GOAL': ['Goal'], 'FREE KICK': ['Direct free-kick', 'Indirect free-kick']}
GRID = np.round(np.arange(.05, .81, .025), 3)
REFERENCE = ['PLAYER SUCCESSFUL TACKLE', 'BALL PLAYER BLOCK', 'HEADER', 'CROSS', 'HIGH PASS', 'SHOT', 'PASS', 'DRIVE']


def match(pred_t, true_t, tol=TOLERANCE_S):
    used, tp = set(), 0
    for t in sorted(pred_t):
        cand = [i for i, u in enumerate(true_t) if i not in used and abs(u - t) <= tol]
        if cand:
            used.add(min(cand, key=lambda i: abs(true_t[i] - t)))
            tp += 1
    return tp


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spots-file', default='action_spots.json',
                   help='e.g. action_spots_published.json to score spots kept by spot_actions.py')
    p.add_argument('--report', default='action_spotter.json',
                   help='evidence file; action_spotter.json holds the thresholds the pipeline uses')
    args = p.parse_args()
    halves = []
    for item in SN.library():
        ident = MP.soccernet_identifier(item['game'], item['half'])
        saved = S.read_json(S.dataset_dir(ident) / args.spots_file)
        if saved:
            labels = SN.annotations(item['game'], item['half'])
            halves.append({'half': ident, 'split': item['benchmark_split'], 'spots': saved['spots'], 'labels': labels})
    rows = []
    for h in halves:
        for cls, names in LABELS.items():
            truth = [x['time_s'] for x in h['labels'] if x['label'] in names]
            spots = [s for s in h['spots'] if s['label'] == cls]
            for thr in GRID:
                pred = [s['time_s'] for s in spots if s['score'] >= thr]
                rows.append({'half': h['half'], 'split': h['split'], 'class': cls, 'threshold': float(thr),
                             'predicted': len(pred), 'labelled': len(truth), 'matched': match(pred, truth)})
            rows.append({'half': h['half'], 'split': h['split'], 'class': cls, 'threshold': -1,
                         'predicted': float(sum(s['score'] for s in spots)), 'labelled': len(truth), 'matched': 0})
    d = pd.DataFrame(rows)
    report = {'created': S.now(), 'tolerance_s': TOLERANCE_S, 'halves': len(halves),
              'test_halves': sorted({h['half'] for h in halves if h['split'] == 'test'}), 'classes': {}}
    chosen = {}
    for cls in LABELS:
        c = d[(d['class'] == cls) & (d.threshold >= 0)]
        dev = c[c.split != 'test'].groupby('threshold')[['predicted', 'labelled', 'matched']].sum()
        f1 = 2 * dev.matched / (dev.predicted + dev.labelled).clip(lower=1)
        thr = float(f1.idxmax())
        chosen[cls] = thr
        test = c[(c.split == 'test') & (c.threshold == thr)][['predicted', 'labelled', 'matched']].sum()
        calib = d[(d['class'] == cls) & (d.threshold == -1)][['predicted', 'labelled']].sum()
        report['classes'][cls] = {
            'threshold': thr,
            'train_validation': {'predicted': int(dev.loc[thr, 'predicted']), 'labelled': int(dev.loc[thr, 'labelled']),
                                 'precision': float(dev.loc[thr, 'matched'] / max(dev.loc[thr, 'predicted'], 1)),
                                 'recall': float(dev.loc[thr, 'matched'] / max(dev.loc[thr, 'labelled'], 1)),
                                 'f1': float(f1.max())},
            'held_out_test': {'predicted': int(test.predicted), 'labelled': int(test.labelled),
                              'precision': float(test.matched / max(test.predicted, 1)),
                              'recall': float(test.matched / max(test.labelled, 1))},
            'sum_of_scores_vs_labels': {'sum_of_scores': float(calib.predicted), 'labels': int(calib.labelled)}}
    generic = float(np.median(list(chosen.values())))
    report['threshold_for_unlabelled_classes'] = generic
    per_half = {}
    for cls in REFERENCE:
        thr = chosen.get(cls, generic)
        counts = [sum(1 for s in h['spots'] if s['label'] == cls and s['score'] >= thr) / 2 for h in halves]
        per_half[cls] = {'threshold': thr, 'per_team_per_half_median': float(np.median(counts)) if counts else None}
    ref_path = S.DATA / 'raw' / 'soccertrack-v2' / 'bas' / '117092' / '117092_12_class_events.json'
    if ref_path.is_file():
        a = pd.DataFrame(json.loads(ref_path.read_text(encoding='utf-8'))['actions'])
        a = a[a.gameTime.str[0].isin(['1', '2'])]
        ref = a.groupby(['label']).size() / 4
        for cls in per_half:
            per_half[cls]['soccertrack_reference_per_team_per_half'] = float(ref.get(cls, 0))
    report['counts'] = per_half
    report['thresholds'] = {**{c: generic for c in AS.CLASSES}, **chosen}
    report['spots_file'] = args.spots_file
    S.write_json(S.EVIDENCE / args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'test_halves'}, indent=1))


if __name__ == '__main__':
    main()
