"""Score detection, player identity and per-player actions against FOOTPASS ground truth.

For halves analysed by analyse_footpass.py. FOOTPASS gives every visible player's box in the
full-HD picture with shirt number and team, and every ball action with the acting player.

Identity: on each sampled frame, our player and goalkeeper boxes are matched one-to-one to the
ground-truth boxes (IoU >= 0.5). Each ground-truth player-sample is then one of: not detected;
detected but not identified; identified with the right team and shirt number; identified wrongly.
Our kit groups (A/B) are mapped to the FOOTPASS teams by majority. Identified-by-number (the
segment's own reading) and identified-by-appearance are reported separately, and coverage is
broken down by box height.

Actions: each labelled action is looked for among our events of the matching kind within 1 s
(rule events, and spotted events above their confidence threshold). It is found, credited to a
named player, and credited to the right one (team and shirt number) or not.

Report: evidence/identity_footpass_<tag>.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_identity_footpass.py [--split VAL] [--tag baseline]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_main = Path((ROOT / '.data-location').read_text(encoding='utf-8').strip()) if (ROOT / '.data-location').is_file() \
    else ROOT / 'data'
os.environ.setdefault('PITCHPROFILE_DATA', str(_main.parent.parent / 'PitchProfile-eval' / 'data'))
os.environ.setdefault('PITCHPROFILE_WEIGHTS', str(_main.parent / 'weights'))
sys.path.insert(0, str(ROOT))

from football_profiler import football_models as FM  # noqa: E402
from football_profiler import soccernet as SN  # noqa: E402
from football_profiler import storage as S  # noqa: E402

FPS = 25.0
IOU_MIN = .5
ACTION_WINDOW_S = 1.0
WITHOUT_FILL = False           # --without-fill: leave out passes and carries filled from the spotter
CLASSES = {1: 'Drive', 2: 'Pass', 3: 'Cross', 4: 'Throw-in', 5: 'Shot', 6: 'Header', 7: 'Tackle', 8: 'Block'}
OURS = {'Drive': ['carry'], 'Pass': ['pass'], 'Cross': ['cross'], 'Throw-in': ['throw_in'], 'Shot': ['shot'],
        'Header': ['header'], 'Tackle': ['tackle'], 'Block': ['block']}
HEIGHTS = [(0, 60, 'under 60 px'), (60, 100, '60-100 px'), (100, 150, '100-150 px'), (150, 10_000, '150 px and over')]


def parse_identity(identity):
    """'A-10' -> ('A', 10, 'number'); 'A-GK' -> ('A', None, 'keeper'); others -> (team, None, kind)."""
    if not identity:
        return None, None, 'none'
    m = re.fullmatch(r'([AB])-(\d+)', identity)
    if m:
        return m.group(1), int(m.group(2)), 'number'
    if identity.endswith('-GK'):
        return identity[0], None, 'keeper'
    if re.fullmatch(r'[AB]-X\d+', identity):
        return identity[0], None, 'unnamed'
    return identity[0], None, 'slot' if re.fullmatch(r'[AB]-P\d+', identity) else 'none'


def iou_matrix(a, b):
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    return inter / ((a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter + 1e-9)


def half_data(d):
    meta = S.read_json(d / 'analysis.json')
    fp = meta['footpass']
    frames = pd.read_csv(d / 'raw_frames.csv.gz', usecols=['sample', 'source_frame', 'time_s'])
    people = pd.read_csv(d / 'raw_people.csv.gz', dtype={'track': str})
    people = people[people.cls.isin([FM.PLAYER, FM.GOALKEEPER])]
    z = np.load(d / 'appearance.npz', allow_pickle=True)
    a = {k: z[k] for k in z.files}
    track_seg = {}
    for i, s in enumerate(a['segment'].astype(str)):
        for t in str(a['tracks'][i]).split('|'):
            track_seg[t] = i
    info = S.read_json(d / 'identities.json')
    ident = {s: v['identity'] for s, v in info['segments'].items()}
    return meta, fp, frames, people, a, track_seg, ident


def ground_truth(fp):
    import h5py
    with h5py.File(SN.root() / 'sn-pcbas-2026' / f"{fp['split'].lower()}_tactical_data.h5", 'r') as f:
        g = pd.DataFrame(f[f"{fp['game']}_H{fp['half']}"][:],
                         columns=['frame', 'player_id', 'ltr', 'shirt', 'role', 'x', 'y', 'vx', 'vy',
                                  'roi_x', 'roi_y', 'roi_w', 'roi_h', 'cls'])
    g['team'] = (g.player_id // 100).astype(int)
    return g


def evaluate_half(d):
    from scipy.optimize import linear_sum_assignment
    meta, fp, frames, people, a, track_seg, ident = half_data(d)
    gt = ground_truth(fp)
    start = fp['first_frame']
    by_frame_gt = {f: g for f, g in gt[gt.roi_w.notna()].groupby('frame')}
    people = people.merge(frames[['sample', 'source_frame']], on='sample')
    pairs = []
    for f, ours in people.groupby('source_frame'):
        truth = by_frame_gt.get(f)
        if truth is None:
            continue
        iou = iou_matrix(truth[['roi_x', 'roi_y', 'roi_w', 'roi_h']].to_numpy(float),
                         ours[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']].to_numpy(float))
        r, c = linear_sum_assignment(-iou)
        matched = {i: j for i, j in zip(r, c) if iou[i, j] >= IOU_MIN}
        for i, row in enumerate(truth.itertuples()):
            j = matched.get(i)
            seg = track_seg.get(str(ours.iloc[j].track)) if j is not None else None
            pairs.append({'frame': f, 'gt_player': int(row.player_id), 'gt_team': row.team, 'gt_shirt': int(row.shirt),
                          'gt_keeper': row.role == 1,
                          'height': row.roi_h, 'detected': j is not None, 'segment': seg})
    # Truth frames never sampled by us (outside the analysed range) are not counted.
    p = pd.DataFrame(pairs)
    seg_names = a['segment'].astype(str)
    p['identity'] = [ident.get(seg_names[int(s)]) if s is not None and not pd.isna(s) else None for s in p.segment]
    parsed = [parse_identity(i) for i in p.identity]
    p['our_team'] = [t for t, _, _ in parsed]
    p['our_number'] = [n for _, n, _ in parsed]
    p['kind'] = [k for _, _, k in parsed]
    p['own_read'] = [int(a['number'][int(s)]) if s is not None and not pd.isna(s) else -1 for s in p.segment]
    votes = Counter((t, g) for t, g in zip(p.our_team, p.gt_team) if t in ('A', 'B'))
    straight = votes[('A', 1)] + votes[('B', 2)]
    crossed = votes[('A', 2)] + votes[('B', 1)]
    team_map = {'A': 1, 'B': 2} if straight >= crossed else {'A': 2, 'B': 1}
    p['team_ok'] = [team_map.get(t) == g for t, g in zip(p.our_team, p.gt_team)]
    p['number_ok'] = p.team_ok & (p.our_number == p.gt_shirt)
    p['keeper_ok'] = p.team_ok & p.gt_keeper & p.kind.eq('keeper')
    # Keepers are identified as "the A goalkeeper", not by number; that counts when team and role are right.
    p['identified'] = p.kind.isin(['number', 'keeper'])
    p['correct'] = p.number_ok | p.keeper_ok
    # Each segment's main true player, to score crediting apart from naming.
    owner = p[p.detected & p.segment.notna()].groupby('segment').gt_player.agg(lambda s: s.value_counts().index[0])
    seg_owner = {seg_names[int(i)]: int(v) for i, v in owner.items()}
    return p, team_map, actions(d, fp, gt, team_map, start, seg_owner)


def actions(d, fp, gt, team_map, start, seg_owner=None):
    from football_profiler import action_spotting as AS
    ev = pd.DataFrame(S.read_json(d / 'events.json', {}).get('events', []))
    labels = gt[gt.cls > 0].copy()
    labels['label'] = labels.cls.astype(int).map(CLASSES)
    labels['t'] = (labels.frame - start) / FPS
    rows = []
    if ev.empty:
        return rows
    ev['source'] = ev.get('source', pd.Series('rules', index=ev.index)).fillna('rules')
    if 'confident' not in ev:
        ev['confident'] = True
    usable = ev[ev.source.eq('rules') | ev.confident.fillna(False).astype(bool)]
    if WITHOUT_FILL and 'filled' in usable:
        usable = usable[~usable.filled.fillna(False).astype(bool)]
    inverse = {v: k for k, v in team_map.items()}
    for r in labels.itertuples():
        cand = usable[usable.type.isin(OURS[r.label]) & ((usable.time_s - r.t).abs() <= ACTION_WINDOW_S)]
        row = {'label': r.label, 'found': len(cand) > 0, 'team_ok': False, 'credited': False, 'correct': False,
               'right_track': False}
        if len(cand):
            e = cand.iloc[int((cand.time_s - r.t).abs().argmin())]
            team, number, kind = parse_identity(e.get('identity') if isinstance(e.get('identity'), str) else None)
            row['team_ok'] = e.get('team') == inverse.get(int(r.team))
            row['credited'] = kind == 'number'
            row['correct'] = kind == 'number' and team == inverse.get(int(r.team)) and number == int(r.shirt)
            row['right_track'] = (seg_owner or {}).get(str(e.get('segment'))) == int(r.player_id)
        rows.append(row)
    return rows


def unnamed_purity(det):
    """Share of unnamed players' time that belongs to each one's main true player."""
    u = det[det.kind.eq('unnamed')]
    if u.empty or 'half' not in u:
        return None
    per = u.groupby(['half', 'identity']).gt_player.agg(lambda s: s.value_counts().iloc[0]).sum()
    return round(float(per / len(u)), 3)


def tracking(det):
    """Segment purity (share of a segment's time on its main true player) and fragmentation."""
    if det.empty or 'half' not in det:
        return {}
    unsegmented = float(det.segment.isna().mean())
    det = det[det.segment.notna()]
    key = det.half + ':' + det.segment.astype(int).astype(str)
    counts = det.groupby([key, det.gt_player]).size()
    purity = counts.groupby(level=0).max().sum() / max(counts.sum(), 1)
    per_player = det.groupby([det.half, det.gt_player]).agg(segments=('segment', 'nunique'), samples=('segment', 'size'))
    minutes = per_player.samples.sum() / 12.5 / 60
    seg_len = det.groupby(key).size() / 12.5
    return {'detected_but_in_no_segment': round(unsegmented, 3), 'segment_purity': round(float(purity), 3),
            'segments_per_player_minute': round(float(per_player.segments.sum() / max(minutes, 1e-9)), 2),
            'visible_time_in_segments_10s_or_longer': round(float(seg_len[seg_len >= 10].sum() / max(seg_len.sum(), 1e-9)), 3)}


def summarise(p, action_rows):
    n = len(p)
    det = p[p.detected]
    identified = det[det.identified]
    numbered = det[det.kind.eq('number')]
    by_read = numbered[numbered.own_read == numbered.our_number]
    by_appearance = numbered[numbered.own_read != numbered.our_number]
    out = {
        'ground_truth_player_samples': int(n),
        'detected_share': round(len(det) / max(n, 1), 3),
        'identified_share_of_visible': round(len(identified) / max(n, 1), 3),
        'identified_share_of_detected': round(len(identified) / max(len(det), 1), 3),
        'correct_share_of_visible': round(int(identified.correct.sum()) / max(n, 1), 3),
        'identity_accuracy': round(float(identified.correct.mean()) if len(identified) else 0.0, 3),
        'accuracy_by_number_read': round(float(by_read.number_ok.mean()) if len(by_read) else 0.0, 3),
        'accuracy_by_appearance': round(float(by_appearance.number_ok.mean()) if len(by_appearance) else 0.0, 3),
        'share_by_number_read': round(len(by_read) / max(n, 1), 3),
        'share_by_appearance': round(len(by_appearance) / max(n, 1), 3),
        'team_accuracy_of_detected': round(float(det[det.our_team.isin(['A', 'B'])].team_ok.mean()), 3) if len(det) else 0.0,
        'grouped_share_of_visible': round(float(det.kind.isin(['number', 'keeper', 'unnamed']).sum()) / max(n, 1), 3),
        'unnamed_share_of_visible': round(float(det.kind.eq('unnamed').sum()) / max(n, 1), 3),
        'unnamed_purity': unnamed_purity(det),
        'keepers_identified': round(float(det[det.gt_keeper].keeper_ok.mean()), 3) if det.gt_keeper.any() else None,
        # Of the time labelled "goalkeeper", the share that is that team's true goalkeeper.
        'keeper_purity': round(float(det[det.kind.eq('keeper')].keeper_ok.mean()), 3) if det.kind.eq('keeper').any() else None,
        'tracking': tracking(det), 'by_box_height': {}}
    for lo, hi, name in HEIGHTS:
        q = p[(p.height >= lo) & (p.height < hi)]
        qd = q[q.detected]
        out['by_box_height'][name] = {
            'share_of_samples': round(len(q) / max(n, 1), 3), 'detected': round(len(qd) / max(len(q), 1), 3),
            'identified_correctly': round(int(qd.correct.sum()) / max(len(q), 1), 3),
            'identified_wrongly': round(int((qd.identified & ~qd.correct).sum()) / max(len(q), 1), 3)}
    acts = pd.DataFrame(action_rows)
    out['actions'] = {}
    if len(acts):
        for label, g in acts.groupby('label'):
            out['actions'][label] = {'labels': int(len(g)), 'found': round(float(g.found.mean()), 3),
                                     'team_right_when_found': round(float(g[g.found].team_ok.mean()), 3) if g.found.any() else None,
                                     'credited_to_named_player': round(float(g.credited.mean()), 3),
                                     'credited_correctly': round(float(g.correct.mean()), 3),
                                     'credited_to_right_track': round(float(g.right_track.mean()), 3),
                                     'accuracy_when_credited': round(float(g[g.credited].correct.mean()), 3) if g.credited.any() else None}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='VAL')
    p.add_argument('--tag', default='current')
    p.add_argument('--variant', default='', help='the experiment analysed with analyse_footpass.py --variant')
    p.add_argument('--without-fill', action='store_true', help='score actions without the spotter-filled passes and carries')
    args = p.parse_args()
    global WITHOUT_FILL
    WITHOUT_FILL = args.without_fill
    dirs = sorted((S.DATA / 'datasets').glob(f'fp-{args.split.lower()}-*'))
    dirs = [d for d in dirs if (d / 'appearance.npz').is_file() and (d / 'identities.json').is_file()
            and (S.read_json(d / 'analysis.json', {}).get('footpass') or {}).get('variant', '') == args.variant]
    if not dirs:
        raise SystemExit('No analysed FOOTPASS halves: run scripts/analyse_footpass.py first')
    all_pairs, all_actions, halves = [], [], {}
    for d in dirs:
        pairs, team_map, acts = evaluate_half(d)
        pairs['half'] = d.name
        halves[d.name] = summarise(pairs, acts)
        all_pairs.append(pairs)
        all_actions += acts
        h = halves[d.name]
        print(f"{d.name}: detected {h['detected_share']:.0%}, identified {h['identified_share_of_visible']:.0%} of visible "
              f"time, accuracy {h['identity_accuracy']:.0%}", flush=True)
    report = {'created': S.now(), 'tag': args.tag, 'split': args.split, 'iou_min': IOU_MIN,
              'overall': summarise(pd.concat(all_pairs, ignore_index=True), all_actions), 'halves': halves}
    S.write_json(S.EVIDENCE / f'identity_footpass_{args.tag}.json', report)
    o = report['overall']
    print(f"\nOVERALL: visible player time detected {o['detected_share']:.1%}; identified {o['identified_share_of_visible']:.1%} "
          f"(by number read {o['share_by_number_read']:.1%}, by appearance {o['share_by_appearance']:.1%}); "
          f"identity accuracy {o['identity_accuracy']:.1%} (read {o['accuracy_by_number_read']:.1%}, appearance "
          f"{o['accuracy_by_appearance']:.1%}); correct identity on {o['correct_share_of_visible']:.1%} of visible time")
    for k, v in o['by_box_height'].items():
        print(f'  boxes {k:16} {v}')
    for k, v in o['actions'].items():
        print(f'  {k:9} {v}')


if __name__ == '__main__':
    main()
