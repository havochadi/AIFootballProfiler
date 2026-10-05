"""Measure identity linking of unnumbered player segments on hidden shirt numbers.

Reads each half's appearance.npz (written by post-processing: every team segment with its time
span, thumbnails, shirt-number reading and appearance embedding). For each team, the segments
of its reliably read numbers are split into folds; one fold's numbers are hidden in turn, and
the hidden segments are linked together with the team's genuinely unnumbered segments:

  direct   - today's rule (match_identity.resolve_numbers): each segment with enough thumbnails
             joins the numbered player it clearly resembles (REID_MARGIN), never overlapping him.
  cluster  - GTA-style connector first (Sun et al. 2024, "GTA: Global Tracklet Association"):
             unnumbered segments that never overlap in time are merged greedily by appearance
             similarity (>= tau), pooling their thumbnails; each cluster then joins a player by
             the same margin rule on its pooled thumbnails, and all its segments take his number.
  +rounds  - either method repeated with player prototypes updated by what joined them.

Accuracy (share of attached hidden time given the right number) and coverage (share of hidden
time attached) are reported per method, with the time of genuinely unnumbered segments each
method would attach. Held-out halves (SoccerNet validation and test) are used by default, as
the appearance head was trained on the training halves' numbers.

Report: evidence/identity_linking.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_linking.py [--all-splits]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import match_identity as MI  # noqa: E402
from football_profiler import match_pipeline as MP  # noqa: E402
from football_profiler import soccernet as SN  # noqa: E402
from football_profiler import storage as S  # noqa: E402

FOLDS = 5


def load(ident):
    path = S.dataset_dir(ident) / 'appearance.npz'
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=True)          # written by this project's post-processing
    out = {k: z[k] for k in z.files}
    for k in ('segment', 'team', 'role'):
        out[k] = out[k].astype('U')
    return out


def anchors_of(a, team):
    """{number: [segment indices]} for the team's strong numbers, as match_identity.resolve_numbers."""
    idx = np.flatnonzero((a['team'] == team) & (a['role'] == 'player'))
    reads = [i for i in idx if a['number'][i] >= 0 and a['number_confidence'][i] >= MI.MIN_NUMBER_SHARE
             and a['number_votes'][i] >= MI.MIN_NUMBER_VOTES]
    by = {}
    for i in reads:
        by.setdefault(int(a['number'][i]), []).append(i)
    evidence = {n: sum(a['number_votes'][i] for i in s) for n, s in by.items()}
    strong = [n for n in sorted(evidence, key=evidence.get, reverse=True)
              if len(by[n]) >= MI.ANCHOR_MIN_SEGMENTS and evidence[n] >= MI.ANCHOR_MIN_VOTES][:MI.MAX_OUTFIELD_NUMBERS]
    out = {}
    for n in strong:
        kept = []
        for i in sorted(by[n], key=lambda i: -a['number_votes'][i]):
            if not any(a['start_s'][k] <= a['end_s'][i] and a['start_s'][i] <= a['end_s'][k] for k in kept):
                kept.append(i)
        out[n] = kept
    return out, idx


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def cluster(a, pool, tau):
    """Greedy constrained merging of pool segments; returns lists of segment indices.

    Two clusters may merge only if no member of one overlaps a member of the other in time; the
    overlap matrix is kept per cluster (OR of its members' rows).
    """
    if tau is None or len(pool) < 2:
        return [[i] for i in pool]
    pool = np.asarray(pool)
    emb = a['embedding'].astype(np.float32)
    w = np.maximum(a['crops'][pool], 1).astype(np.float32)
    sums = emb[pool] * w[:, None]
    proto = unit(sums.copy())
    s, e = a['start_s'][pool], a['end_s'][pool]
    overlap = (s[:, None] <= e[None, :]) & (s[None, :] <= e[:, None])
    members = [[int(i)] for i in pool]
    alive = np.ones(len(pool), bool)
    sim = proto @ proto.T
    sim[overlap] = -np.inf
    np.fill_diagonal(sim, -np.inf)
    while True:
        k = int(np.argmax(sim))
        i, j = divmod(k, len(pool))
        if not np.isfinite(sim[i, j]) or sim[i, j] < tau:
            break
        members[i] += members[j]
        sums[i] += sums[j]
        overlap[i] |= overlap[j]
        overlap[:, i] = overlap[i]
        alive[j] = False
        sim[j, :] = sim[:, j] = -np.inf
        proto[i] = unit(sums[i])
        row = proto @ proto[i]
        row[~alive | overlap[i]] = -np.inf
        row[i] = -np.inf
        sim[i, :] = sim[:, i] = row
    return [members[i] for i in np.flatnonzero(alive)]


def attach(a, groups, anchor_members, rounds, extra=0.0):
    """Margin rule of resolve_numbers on groups (pooled thumbnails); {segment index: number}.

    Rounds after the first use prototypes that include what joined, with `extra` added to the margin.
    """
    emb = a['embedding'].astype(np.float32)
    busy = {n: [(a['start_s'][i], a['end_s'][i]) for i in s] for n, s in anchor_members.items()}
    joined = {n: list(s) for n, s in anchor_members.items()}
    out = {}
    todo = sorted(groups, key=lambda g: -a['samples'][g].sum())
    for r in range(rounds):
        protos = {n: unit(emb[s].mean(0)) for n, s in joined.items() if len(s)}
        left = []
        for g in todo:
            crops = int(a['crops'][g].sum())
            need = MI._reid_margin(crops)
            if need is None:
                left.append(g)
                continue
            need += extra if r else 0.0
            v = unit((emb[g] * np.maximum(a['crops'][g], 1)[:, None]).sum(0))
            spans = [(a['start_s'][i], a['end_s'][i]) for i in g]
            scores = sorted(((float(v @ p), n) for n, p in protos.items()
                             if not any(s0 <= e and s <= e0 for s, e in spans for s0, e0 in busy[n])), reverse=True)
            if len(scores) < 2 or scores[0][0] - scores[1][0] < need:
                left.append(g)
                continue
            n = scores[0][1]
            for i in g:
                out[i] = n
            busy[n] += spans
            joined[n] += list(g)
        if len(left) == len(todo):
            break
        todo = left
    return out


# (name, clustering threshold or None, rounds, extra margin in later rounds)
METHODS = [('direct', None, 1, 0.0), ('direct+rounds', None, 3, 0.0)] + \
          [(f'direct+rounds margin+{m}', None, 3, m) for m in (.01, .02, .03, .05)] + \
          [(f'cluster tau={t}', t, 1, 0.0) for t in (.9, .8)]


def evaluate(a, rng):
    dur = a['end_s'] - a['start_s'] + 1 / 12.5
    res = {m[0]: {'hidden_s': 0.0, 'attached_s': 0.0, 'correct_s': 0.0, 'unnumbered_attached_s': 0.0,
                  'unnumbered_s': 0.0} for m in METHODS}
    for team in ('A', 'B'):
        anchors, idx = anchors_of(a, team)
        if len(anchors) < 3:
            continue
        numbered = [i for s in anchors.values() for i in s]
        truth = {i: n for n, s in anchors.items() for i in s}
        # Everything that is not an anchor may join a player, as in resolve_numbers.
        unnumbered = [i for i in idx if i not in truth and a['crops'][i] > 0]
        order = rng.permutation(numbered)
        for f in range(FOLDS):
            hidden = set(order[f::FOLDS].tolist())
            kept = {n: [i for i in s if i not in hidden] for n, s in anchors.items()}
            kept = {n: s for n, s in kept.items() if s}
            pool = sorted(hidden) + unnumbered
            pool = [i for i in pool if a['crops'][i] > 0]
            for name, tau, rounds, extra in METHODS:
                got = attach(a, cluster(a, pool, tau), kept, rounds, extra)
                r = res[name]
                for i in hidden:
                    r['hidden_s'] += dur[i]
                    if i in got:
                        r['attached_s'] += dur[i]
                        r['correct_s'] += dur[i] * (got[i] == truth[i])
                if f == 0:
                    r['unnumbered_s'] += sum(dur[i] for i in unnumbered)
                    r['unnumbered_attached_s'] += sum(dur[i] for i in unnumbered if i in got)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--all-splits', action='store_true')
    args = p.parse_args()
    rng = np.random.default_rng(0)
    total = {m[0]: {} for m in METHODS}
    used = []
    for item in SN.library():
        if not args.all_splits and item.get('benchmark_split') == 'train':
            continue
        ident = MP.soccernet_identifier(item['game'], item['half'])
        a = load(ident)
        if a is None or not len(a['segment']):
            continue
        used.append(ident)
        for name, r in evaluate(a, rng).items():
            for k, v in r.items():
                total[name][k] = total[name].get(k, 0.0) + float(v)
        print(ident, 'done', flush=True)
    if not used:
        raise SystemExit('No appearance.npz yet: run scripts/analyse_matches.py --postprocess-only')
    report = {'created': S.now(), 'halves': used, 'folds': FOLDS, 'methods': {}}
    for name, r in total.items():
        report['methods'][name] = {
            'accuracy': round(r['correct_s'] / max(r['attached_s'], 1e-9), 3),
            'coverage_of_hidden': round(r['attached_s'] / max(r['hidden_s'], 1e-9), 3),
            'unnumbered_time_attached': round(r['unnumbered_attached_s'] / max(r['unnumbered_s'], 1e-9), 3),
            'unnumbered_hours_attached': round(r['unnumbered_attached_s'] / 3600, 2)}
        m = report['methods'][name]
        print(f"{name:26} accuracy {m['accuracy']:.3f}  hidden coverage {m['coverage_of_hidden']:.3f}  "
              f"unnumbered attached {m['unnumbered_time_attached']:.3f} ({m['unnumbered_hours_attached']} h)")
    S.write_json(S.EVIDENCE / 'identity_linking.json', report)


if __name__ == '__main__':
    main()
