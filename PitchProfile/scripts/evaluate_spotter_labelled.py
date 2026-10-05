"""Measure an action-spotter checkpoint on labelled games it was not trained on.

Datasets (packed by pack_spotter_frames.py): SoccerNet Ball Action Spotting test games (bas/test;
the published checkpoint trained on BAS train and was selected on BAS valid) and FOOTPASS
validation games (footpass/val; FOOTPASS classes are measured against the spotter classes in
spotter_data.FOOTPASS_MEASURED). For each class: average precision, precision and recall within
1 s at several score thresholds, whether the team side was right, and the sum of scores against
the number of labels (statistics add scores up, so this is their calibration). For FOOTPASS it
also reports team agreement under both readings of left_to_right, to check spotter_data's.

Spots are cached per video as spots_<tag>.json next to the packed frames.
Report: evidence/action_spotter_<dataset>_<split>_<tag>.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_spotter_labelled.py --dataset bas --split test
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_spotter_labelled.py --dataset footpass --split val --checkpoint published
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import action_spotting as AS  # noqa: E402
from football_profiler import spotter_data as SD  # noqa: E402
from football_profiler import storage as S  # noqa: E402

TOLERANCE_S = 1.0
GRID = [.05, .1, .15, .2, .3, .4, .5, .6]
FPS = 25 / AS.FRAME_STRIDE


def spots_for(video, tag):
    cache = video.folder / f'spots_{tag}.json'
    if cache.is_file():
        return json.loads(cache.read_text(encoding='utf-8'))['spots']
    scores = AS.frame_scores(None, frames=video.iter_frames())
    spots = AS.peaks(scores, FPS)
    cache.write_text(json.dumps({'spots': spots}), encoding='utf-8')
    return spots


def truth_for(video):
    """{measured class: [{'time_s', 'side', 'ltr'}]} for this video's labels."""
    out = {}
    for e in video.events:
        name = e['label']
        if video.dataset == 'bas' and name not in AS.CLASSES:
            continue
        out.setdefault(name, []).append({'time_s': e['frame'] / FPS, 'side': SD.side(e, video.dataset),
                                         'ltr': e.get('left_to_right')})
    return out


def greedy(pred, truth):
    """Match predictions (highest score first) to the nearest unused label within the tolerance.

    Returns per prediction (in score order) the matched label index or -1.
    """
    used, out = set(), []
    times = np.array([t['time_s'] for t in truth]) if truth else np.zeros(0)
    for p in pred:
        best, gap = -1, TOLERANCE_S
        if len(times):
            for i in np.flatnonzero(np.abs(times - p['time_s']) <= TOLERANCE_S):
                if i not in used and abs(times[i] - p['time_s']) <= gap:
                    best, gap = int(i), abs(times[i] - p['time_s'])
        if best >= 0:
            used.add(best)
        out.append(best)
    return out


def measure(videos, tag):
    classes = list(AS.CLASSES) if videos[0].dataset == 'bas' else list(SD.FOOTPASS_MEASURED)
    acc = {c: {'scores': [], 'hits': [], 'team_ok': [], 'team_flip_ok': [], 'team_rule_ok': [], 'offsets': [],
               'labels': 0, 'sum_scores': 0.0} for c in classes}
    for v in videos:
        spots = spots_for(v, tag)
        truth = truth_for(v)
        print(f'{v.name}: {len(spots)} spots, {sum(len(t) for t in truth.values())} labels', flush=True)
        for c in classes:
            measured = [c] if v.dataset == 'bas' else SD.FOOTPASS_MEASURED[c]
            pred = sorted([s for s in spots if s['label'] in measured], key=lambda s: -s['score'])
            t = truth.get(c, [])
            match = greedy(pred, t)
            a = acc[c]
            a['labels'] += len(t)
            a['sum_scores'] += sum(s['score'] for s in pred)
            for s, m in zip(pred, match):
                a['scores'].append(s['score'])
                a['hits'].append(m >= 0)
                a['team_ok'].append(m >= 0 and t[m]['side'] == s['side'])
                flipped = {'left': 'right', 'right': 'left'}.get(t[m]['side']) if m >= 0 else None
                a['team_flip_ok'].append(m >= 0 and flipped == s['side'])
                # Won-ball actions: the other team from the last touch (action_spotting.WON_BALL).
                rule = AS.won_ball_side(spots, s) if m >= 0 else None
                a['team_rule_ok'].append(m >= 0 and (rule or s['side']) == t[m]['side'])
                if m >= 0 and s['score'] >= .3:
                    a['offsets'].append(s['time_s'] - t[m]['time_s'])
    report = {}
    for c, a in acc.items():
        order = np.argsort(-np.array(a['scores'])) if a['scores'] else np.zeros(0, int)
        hits = np.array(a['hits'], bool)[order]
        scores = np.array(a['scores'])[order]
        n = max(a['labels'], 1)
        cum = np.cumsum(hits)
        ap = float(np.sum((cum / np.arange(1, len(hits) + 1))[hits]) / n) if len(hits) else 0.0
        rows = {}
        for thr in GRID:
            keep = scores >= thr
            tp = int(hits[keep].sum())
            team = np.array(a['team_ok'], bool)[order][keep]
            flip = np.array(a['team_flip_ok'], bool)[order][keep]
            rule = np.array(a['team_rule_ok'], bool)[order][keep]
            rows[str(thr)] = {'predicted': int(keep.sum()), 'matched': tp,
                              'precision': round(tp / max(int(keep.sum()), 1), 3), 'recall': round(tp / n, 3),
                              'team_correct': round(int(team.sum()) / max(tp, 1), 3),
                              'team_correct_if_flipped': round(int(flip.sum()) / max(tp, 1), 3),
                              'team_correct_previous_touch_rule': round(int(rule.sum()) / max(tp, 1), 3)}
        f1 = {k: 2 * r['precision'] * r['recall'] / max(r['precision'] + r['recall'], 1e-9) for k, r in rows.items()}
        best = max(f1, key=f1.get)
        report[c] = {'labels': a['labels'], 'average_precision': round(ap, 3), 'sum_of_scores': round(a['sum_scores'], 1),
                     # spot minus label time for matches scoring >= .3: near 0 when labels and video line up
                     'median_offset_s': round(float(np.median(a['offsets'])), 3) if a['offsets'] else None,
                     'best_f1_threshold': float(best), 'best_f1': round(f1[best], 3), 'by_threshold': rows}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=['bas', 'footpass'], default='bas')
    p.add_argument('--split', default='test')
    p.add_argument('--checkpoint', type=Path, default=None,
                   help="'published' or a checkpoint path (default: the pipeline's checkpoint)")
    p.add_argument('--tag', default=None, help='cache and report name (needed with a checkpoint path)')
    args = p.parse_args()
    if str(args.checkpoint) == 'published':
        AS.checkpoint_path = AS.published_checkpoint_path
        args.tag = args.tag or 'published'
    elif args.checkpoint is not None:
        if not args.tag:
            raise SystemExit('--tag is required with a checkpoint path (spots are cached under it)')
        AS.checkpoint_path = lambda: args.checkpoint
    else:
        args.tag = args.tag or ('published' if AS.checkpoint_path() == AS.published_checkpoint_path() else 'finetuned')
    videos = SD.videos(args.dataset, args.split)
    if not videos:
        raise SystemExit(f'No packed videos in {SD.root() / args.dataset / args.split}: run pack_spotter_frames.py')
    classes = measure(videos, args.tag)
    report = {'created': S.now(), 'dataset': args.dataset, 'split': args.split, 'videos': [v.name for v in videos],
              'checkpoint': args.tag, 'checkpoint_file': str(AS.checkpoint_path()), 'tolerance_s': TOLERANCE_S,
              'mean_average_precision': round(float(np.mean([c['average_precision'] for c in classes.values()
                                                             if c['labels']])), 3),
              'classes': classes}
    S.write_json(S.EVIDENCE / f'action_spotter_{args.dataset}_{args.split}_{args.tag}.json', report)
    print(f"mAP {report['mean_average_precision']:.3f}")
    for c, v in classes.items():
        b = v['by_threshold'][str(v['best_f1_threshold'])]
        print(f"{c:26} labels {v['labels']:5} | AP {v['average_precision']:.2f} | best thr {v['best_f1_threshold']:.2f}: "
              f"P {b['precision']:.2f} R {b['recall']:.2f} team {b['team_correct']:.2f} "
              f"(flipped {b['team_correct_if_flipped']:.2f}, previous-touch rule "
              f"{b['team_correct_previous_touch_rule']:.2f}) | sum of scores {v['sum_of_scores']} | "
              f"offset {v['median_offset_s']}")


if __name__ == '__main__':
    main()
