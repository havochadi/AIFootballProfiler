"""Full-match analysis, stage 4: link player segments across camera cuts.

Shirt numbers read from each segment (a stitched tracklet inside one
continuous view) define the players of a team. Only segments with a reliable
reading are attributed. When too few numbers can be read, a fallback assigns
segments to tactical slots per team by role signature (lateral lane and depth
relative to visible teammates), never placing two simultaneously visible
segments in one slot; a slot is a position, not necessarily one person.
Confirmed identities from the review UI always take precedence.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import fieldcal as FC

L, W = FC.PITCH_LENGTH, FC.PITCH_WIDTH
OUTFIELD_SLOTS = 10
MIN_SEGMENT_S = .8        # shorter segments carry too little role evidence to anchor a slot
ITERATIONS = 6
DEPTH_WEIGHT = 30.0       # metres-equivalent weight of the 0..1 depth rank


def normalise(x, y, sign):
    """Attack-normalised coordinates: the team attacks towards +x; small y is its left flank."""
    if sign >= 0:
        return x, y
    return L - x, W - y


def signatures(people, directions):
    """Per-segment role signature: lateral lane (y'), depth rank among visible teammates, x'."""
    from .match_events import attack_sign
    q = people[people.team.isin(['A', 'B']) & np.isfinite(people.x) & people.segment.notna()].copy()
    sign = q.team.map(lambda t: attack_sign(t, directions)).to_numpy()
    q['xn'] = np.where(sign >= 0, q.x, L - q.x)
    q['yn'] = np.where(sign >= 0, q.y, W - q.y)
    outfield = q.role.eq('player')
    q['depth'], q['rel_x'], q['company'] = np.nan, np.nan, np.nan
    # Rank of each outfield player's x' among the visible outfield teammates in the same frame.
    g = q[outfield].groupby(['sample', 'team']).xn
    rank = g.rank(method='average')
    count = g.transform('size')
    q.loc[outfield, 'depth'] = np.where(count > 1, (rank - 1) / (count - 1).clip(lower=1), .5)
    q.loc[outfield, 'rel_x'] = q.loc[outfield, 'xn'] - g.transform('mean')
    q.loc[outfield, 'company'] = count
    seg = q.groupby('segment').agg(team=('team', 'first'), role=('role', 'first'), start_s=('time_s', 'min'),
                                   end_s=('time_s', 'max'), n=('sample', 'size'), yn=('yn', 'mean'),
                                   xn=('xn', 'mean'), depth=('depth', 'mean'), rel_x=('rel_x', 'mean'),
                                   company=('company', 'mean'))
    seg['duration_s'] = seg.end_s - seg.start_s
    return seg


def _overlaps(intervals, start, end):
    return any(s <= end and start <= e for s, e in intervals)


def assign_slots(seg, slots=OUTFIELD_SLOTS, seed=0):
    """Constrained k-means: slot centres in (y', depth) space; no slot holds two overlapping segments."""
    seg = seg.copy()
    seg['slot'] = -1
    feats = np.c_[seg.yn.to_numpy(), np.nan_to_num(seg.depth.to_numpy(), nan=.5) * DEPTH_WEIGHT]
    weight = np.sqrt(seg.n.to_numpy(float))
    anchor = seg.duration_s.to_numpy() >= MIN_SEGMENT_S
    if anchor.sum() < slots:
        anchor = np.ones(len(seg), bool)
    rng = np.random.default_rng(seed)
    X = feats[anchor]
    # k-means++ style initialisation on the anchoring segments.
    centres = [X[rng.choice(len(X), p=weight[anchor] / weight[anchor].sum())]]
    while len(centres) < min(slots, len(X)):
        d = np.min(np.linalg.norm(X[:, None] - np.array(centres)[None], axis=2), axis=1) ** 2
        centres.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    centres = np.array(centres)
    order = np.argsort(-seg.duration_s.to_numpy())
    starts, ends = seg.start_s.to_numpy(), seg.end_s.to_numpy()
    cost = None
    for _ in range(ITERATIONS):
        cost = np.linalg.norm(feats[:, None] - centres[None], axis=2)
        taken = [[] for _ in centres]
        labels = np.full(len(seg), -1)
        for i in order:
            for k in np.argsort(cost[i]):
                if not _overlaps(taken[k], starts[i], ends[i]):
                    taken[k].append((starts[i], ends[i]))
                    labels[i] = k
                    break
        updated = centres.copy()
        for k in range(len(centres)):
            m = (labels == k) & anchor
            if m.any():
                updated[k] = np.average(feats[m], axis=0, weights=weight[m])
        if np.allclose(updated, centres, atol=.05):
            break
        centres = updated
    seg['slot'] = labels
    seg['slot_cost'] = [cost[i, k] if k >= 0 else np.nan for i, k in enumerate(labels)]
    return seg, centres


def slot_name(yn, depth):
    lane = 'left' if yn < W / 3 else 'right' if yn > 2 * W / 3 else 'central'
    line = 'defender' if depth < .34 else 'forward' if depth > .66 else 'midfielder'
    return f'{lane} {line}'


MIN_NUMBER_SHARE = .5          # winning number's share of a segment's legible reading confidence
MIN_NUMBER_VOTES = .3          # summed recogniser confidence behind a segment's number
ANCHOR_MIN_SEGMENTS = 3        # a shirt number becomes a player only with this many readings...
ANCHOR_MIN_VOTES = 1.5         # ...and this much summed confidence
MAX_OUTFIELD_NUMBERS = 14      # starting outfielders plus substitutes in one half
# Appearance re-identification attaches unnumbered segments (embeddings from the identity model
# on the normal path). Thresholds from held-out matches (thumbnails re-cut by scripts/extract_crops.py):
# a margin over the runner-up player of 0.02 with 17+ thumbnails picked the right player 91% of
# the time and 0.025 with 9-16 thumbnails 90%; segments with fewer thumbnails did not reach 90%
# at any margin (5-8: 86%, 1-2: under 50%), so they stay unassigned.
REID_MIN_CROPS = 9
REID_MARGIN = {9: .025, 17: .02}         # minimum crops -> minimum similarity margin


def _reid_margin(crops):
    need = [m for k, m in sorted(REID_MARGIN.items()) if crops >= k]
    return need[-1] if need else None


def resolve_numbers(seg, numbers, team, embeddings=None):
    """Jersey-anchored identities for one team's outfield segments; None if too few readings.

    embeddings maps segment -> (unit appearance embedding, thumbnail count); unnumbered segments
    join a numbered player when their embedding clearly prefers him (REID_MARGIN).
    """
    t = seg[(seg.team == team) & (seg.role == 'player')]
    reads = {s: numbers[s] for s in t.index if s in numbers and numbers[s].get('number') is not None
             and numbers[s]['confidence'] >= MIN_NUMBER_SHARE and numbers[s]['votes'] >= MIN_NUMBER_VOTES}
    by_number = {}
    for s, r in reads.items():
        by_number.setdefault(r['number'], []).append(s)
    # At 720p the reader confuses some digits consistently (26 read as 36, 22 as 23),
    # so only the best-supported numbers of a team become players; weaker readings
    # are treated as unread and may join a player only on a clear role match.
    evidence = {n: sum(reads[s]['votes'] for s in segs) for n, segs in by_number.items()}
    strong = [n for n in sorted(evidence, key=evidence.get, reverse=True)
              if len(by_number[n]) >= ANCHOR_MIN_SEGMENTS and evidence[n] >= ANCHOR_MIN_VOTES][:MAX_OUTFIELD_NUMBERS]
    anchors = {}
    for number in strong:
        segs = by_number[number]
        # Two simultaneous segments cannot share a shirt: keep the surer reading.
        kept = []
        for s in sorted(segs, key=lambda s: -reads[s]['votes']):
            if not _overlaps([(t.start_s[k], t.end_s[k]) for k in kept], t.start_s[s], t.end_s[s]):
                kept.append(s)
        anchors[number] = kept
    if len(anchors) < 3:
        return None
    # Matching unread segments by tactical role was tested on segments with known numbers
    # (hidden): it picked the right player less than half the time. Appearance embeddings do
    # far better, so only they attach unread segments, and only when clearly decided.
    identity = {s: f'{team}-{n}' for n, segs in anchors.items() for s in segs}
    attached = {n: [] for n in anchors}
    if embeddings:
        busy = {n: [(t.start_s[s], t.end_s[s]) for s in segs] for n, segs in anchors.items()}
        protos = {}
        for n, segs in anchors.items():
            vecs = [embeddings[s][0] for s in segs if s in embeddings]
            if vecs:
                v = np.mean(vecs, axis=0)
                protos[n] = v / (np.linalg.norm(v) + 1e-9)
        rest = t.drop(index=list(identity)).sort_values('n', ascending=False)
        for s, r in rest.iterrows():
            if s not in embeddings:
                continue
            vec, crops = embeddings[s]
            need = _reid_margin(crops)
            if need is None:
                continue
            scores = sorted(((float(vec @ protos[n]), n) for n in protos
                             if not _overlaps(busy[n], r.start_s, r.end_s)), reverse=True)
            if len(scores) < 2 or scores[0][0] - scores[1][0] < need:
                continue
            n = scores[0][1]
            identity[s] = f'{team}-{n}'
            busy[n].append((r.start_s, r.end_s))
            attached[n].append(s)
    rows = []
    for n, segs in anchors.items():
        m = t.loc[segs + attached[n]]
        w = m.n.to_numpy(float)
        lane, depth = float(np.average(m.yn, weights=w)), float(np.average(np.nan_to_num(m.depth, nan=.5), weights=w))
        rows.append({'identity': f'{team}-{n}', 'team': team, 'role': 'player', 'number': int(n),
                     'label': slot_name(lane, depth), 'lane_y': lane, 'depth': depth,
                     'segments': int(len(segs) + len(attached[n])), 'numbered_segments': int(len(segs)),
                     'appearance_segments': int(len(attached[n])),
                     'observed_s': float(m.duration_s.sum()),
                     'mean_share': float(np.mean([reads[s]['confidence'] for s in segs])),
                     'votes': float(sum(reads[s]['votes'] for s in segs))})
    return identity, rows


def resolve(people, directions, confirmed=None, numbers=None, embeddings=None):
    """Segment -> identity for both teams.

    numbers maps segment -> jersey reading (football_profiler.jersey). With enough
    readings, identities are shirt numbers; segments without a reliable reading join
    a player only on a clear appearance match (embeddings from the identity model)
    and otherwise stay unassigned (their events still count for the team). With too
    few readings tactical slots are used. confirmed maps segment -> identity from
    human review and always wins.
    """
    seg = signatures(people, directions)
    identity, rows = {}, []
    for team in ('A', 'B'):
        numbered = resolve_numbers(seg, numbers, team, embeddings) if numbers else None
        if numbered is not None:
            ids, team_rows = numbered
            identity.update(ids)
            rows += team_rows
            t = seg[seg.team == team]
            keepers = t[t.role == 'goalkeeper']
            for s in keepers.index:
                identity[s] = f'{team}-GK'
            if len(keepers):
                rows.append({'identity': f'{team}-GK', 'team': team, 'role': 'goalkeeper', 'label': 'goalkeeper',
                             'lane_y': float(keepers.yn.mean()), 'depth': 0.0, 'segments': int(len(keepers)),
                             'observed_s': float(keepers.duration_s.sum())})
            continue
        t = seg[seg.team == team]
        keepers = t[t.role == 'goalkeeper']
        for s in keepers.index:
            identity[s] = f'{team}-GK'
        outfield = t[t.role == 'player']
        if len(outfield) >= OUTFIELD_SLOTS:
            assigned, centres = assign_slots(outfield)
            for s, k in assigned.slot.items():
                identity[s] = f'{team}-P{int(k) + 1:02d}' if k >= 0 else f'{team}-U-{s}'
            for k, c in enumerate(centres):
                members = assigned[assigned.slot == k]
                rows.append({'identity': f'{team}-P{k + 1:02d}', 'team': team, 'role': 'player',
                             'label': slot_name(c[0], c[1] / DEPTH_WEIGHT), 'lane_y': float(c[0]),
                             'depth': float(c[1] / DEPTH_WEIGHT), 'segments': int(len(members)),
                             'observed_s': float(members.duration_s.sum())})
        else:
            for s in outfield.index:
                identity[s] = f'{team}-U-{s}'
        if len(keepers):
            rows.append({'identity': f'{team}-GK', 'team': team, 'role': 'goalkeeper', 'label': 'goalkeeper',
                         'lane_y': float(keepers.yn.mean()), 'depth': 0.0, 'segments': int(len(keepers)),
                         'observed_s': float(keepers.duration_s.sum())})
    for s, ident in (confirmed or {}).items():
        if s in identity:
            identity[s] = ident
    return identity, pd.DataFrame(rows), seg


# ---------------------------------------------------------------------------------------------
# Identity from the identity model (football_profiler.identity_model), trained on FOOTPASS true
# identities. Its appearance embedding separates team-mates (held-out FOOTPASS games: a hidden
# tracklet matched its true player 97.6% of the time), so a team's segments are first grouped into
# players by appearance (never joining segments seen at the same moment), and each group's shirt
# number is then read from all its thumbnails together. Groups with a confident number are named;
# the others stay consistent unnamed players, whose statistics still accumulate. Thresholds are
# chosen on FOOTPASS validation games (scripts/tune_identity_clustering.py).

CLUSTER_SIMILARITY = .85       # cosine similarity for joining two groups of one team
NUMBER_CONFIDENCE = .50        # combined digit probability that names a group
NUMBER_EVIDENCE = 1.0          # summed thumbnail certainty behind a name
MIN_PLAYER_S = 20.0            # an unnamed group shorter than this joins a player or stays team-only
ATTACH_MARGIN = .05            # ... and joins only a clearly most similar player
ATTACH_SLACK = .15             # ... whose similarity is at least CLUSTER_SIMILARITY - this
KEEPER_PROBABILITY = .5
# A team fields ten outfield players and makes a few substitutions, so a half has at most about
# MAX_OUTFIELD distinct outfield players; while there are more groups, the most similar two that
# are never on screen together merge (two named players never do), down to MERGE_FLOOR.
MAX_OUTFIELD = 14
MERGE_FLOOR = .7
MERGE_OVERLAP = .03            # tolerated shared screen time (share of the smaller group): groups are ~95% pure
MERGE_UNNAMED_ONLY = True      # merging into named players added wrong names on FOOTPASS (6% -> 11% of time)
# The match line-up, when the reviewer connected it and matched it to the kit groups in the app
# (params['rosters']: kit group -> outfield shirt numbers that played): a number nobody in the
# line-up wore is a misreading, and an unnamed group whose legible votes, restricted to line-up
# numbers not yet taken, give one number ROSTER_SHARE of them is named with it.
ROSTER_MIN_OF_ALL = .2          # ... and at least this share of all its legible votes
ROSTER_SHARE = .9                # FOOTPASS validation, true line-ups: right names 53.0% -> 64.2%, wrong 3.0% -> 2.7%
NAME_AFTER_MERGE = True        # name groups after small groups attach and the team-size merge, so a name
                               # is read from all the thumbnails a merged player has (else before)
TIMELINE_BIN_S = .5
# Grouping treats two groups as simultaneous only beyond this much shared screen time. A segment
# the tracker carried onto another player for a while overlaps that player's own segments, and on
# FOOTPASS validation halves a strict rule kept 94% of a player's split-off screen time apart.
# Appearance already separates teammates (similarity of the closest teammates' segments ~.86 at the
# 99th percentile), so up to 15 s of shared time raised the share of a player's time in his main
# identity from 63% to 89% with unnamed-group purity unchanged (94%); with no limit it fell to 84%.
OVERLAP_TOLERANCE_S = 15.0
NUMBER_CONSTRAINTS = False     # trusted readings must-link / cannot-link segments while grouping
# How a group is named: 'model' (the identity model's combined reading), 'legacy' (the legibility
# classifier and PARSeq readings of its segments), 'agree' (both give the same number), 'either'
# (a well-supported legacy reading, otherwise a very confident model reading), 'consensus'
# (agreement; else a well-supported legacy reading the model does not confidently contradict;
# else a very confident model reading where the legacy reader has none) or 'votes' (the identity
# model's readings of the group's legible thumbnails: a clear majority, or a plurality the legacy
# reader agrees with; groups without vote evidence fall back to 'consensus').
# 'votes' against 'consensus' (the time share of true players named right / wrong): FOOTPASS
# validation halves 45.8% / 3.4% against 22.9% / 1.4%; SoccerNet tracking clips 65.8% / 6.2%
# against 49.0% / 3.7%. With the identity model retrained on legible thumbnails only, groups
# named after merging and VOTE_SHARE .85: FOOTPASS 50.8% / 2.8%, SoccerNet clips 71.8% / 6.8%.
NAMING = 'votes'
VOTE_SHARE = .85               # legible votes' share that names a group on its own
VOTE_SHARE_AGREED = .3         # ... or with the legacy reading agreeing at LEGACY_AGREE_SHARE
LEGACY_AGREE_SHARE = .5
LEGACY_GROUP_VOTES = 1.5       # summed legacy evidence behind a group's number
LEGACY_GROUP_SHARE = .6        # ... and its share of the group's legacy evidence
MODEL_ALONE_CONFIDENCE = .85   # a model reading used without legacy support
# A legacy reading the model disagrees with is kept only when the model gives that number at least
# this share of its own top reading's probability. On FOOTPASS validation groups where the two
# disagreed the legacy number was right 44% of the time overall and 95% above this ratio (the
# legacy reader drops or swaps a digit on some kits: 18 -> 8, 81 -> 8, 11 -> 10).
LEGACY_MODEL_RATIO = .5
# Motion links: pitch coordinates carry across camera cuts, so a segment that starts shortly after
# a teammate's segment ends, where that player would be by then, continues him. On FOOTPASS
# validation halves such links join the same true player 91-95% of the time (with the loose
# appearance check) and a true player's consecutive segments 2-6 m apart were otherwise joined
# only half of the time. Linked segments enter grouping as one.
LINK = True
LINK_GAP_S = 5.0
LINK_RADIUS_M = 3.0            # allowed distance from the predicted position at a zero gap...
LINK_SPEED = 2.0               # ... growing by this many metres per second of gap
LINK_RATIO = 2.5               # a runner-up this much worse (relative) leaves the link unambiguous
LINK_MIN_SIMILARITY = .5       # appearance must not contradict the link
LINK_EDGE_S = .5               # seconds at each end of a segment giving its position and velocity


def _overlap_matrix(starts, ends):
    return (starts[:, None] <= ends[None, :]) & (starts[None, :] <= ends[:, None])


def _overlap_seconds(starts, ends):
    return np.clip(np.minimum(ends[:, None], ends[None, :]) - np.maximum(starts[:, None], starts[None, :]), 0, None)


def group_segments(emb, weights, starts, ends, threshold, labels=None, chains=None, tolerance=0.0):
    """Greedy constrained agglomeration; returns lists of row indices.

    Two groups merge only when no member of one overlaps a member of the other in time and their
    weighted mean embeddings are at least `threshold` similar (most similar pair first). labels
    (a trusted shirt number per row, -1 for none) first join rows of the same number that never
    overlap (must-link) and forbid joining groups carrying different numbers (cannot-link).
    chains (lists of row indices, e.g. motion-linked segments) start as groups. tolerance (seconds)
    lets groups sharing at most that much screen time merge.
    """
    n = len(emb)
    if n == 0:
        return []
    sums = emb * weights[:, None]
    touch = _overlap_matrix(starts, ends)
    shared = _overlap_seconds(starts, ends) if tolerance > 0 else None
    busy = touch.copy() if shared is None else touch & (shared > tolerance)

    def join(h, i):
        touch[h] |= touch[i]
        touch[:, h] = touch[h]
        if shared is None:
            busy[h] = touch[h]
        else:
            shared[h] += shared[i]
            shared[:, h] = shared[h]
            busy[h] = touch[h] & (shared[h] > tolerance)
        busy[:, h] = busy[h]
    members = [[i] for i in range(n)]
    alive = np.ones(n, bool)
    label = np.full(n, -1) if labels is None else np.asarray(labels).copy()
    for chain in chains or []:
        h = chain[0]
        for i in chain[1:]:
            if alive[i] and not busy[h, i]:
                members[h] += members[i]
                sums[h] += sums[i]
                join(h, i)
                alive[i] = False
    # Must-link: rows reading the same number join, heaviest first, when they never overlap.
    for value in [v for v in np.unique(label) if v >= 0]:
        rows = sorted(np.flatnonzero(label == value), key=lambda i: -weights[i])
        heads = []
        for i in rows:
            for h in heads:
                if not busy[h, i]:
                    members[h] += members[i]
                    sums[h] += sums[i]
                    join(h, i)
                    alive[i] = False
                    break
            else:
                heads.append(i)
    proto = sums / (np.linalg.norm(sums, axis=1, keepdims=True) + 1e-9)
    sim = proto @ proto.T
    sim[busy] = -np.inf
    conflict = (label[:, None] >= 0) & (label[None, :] >= 0) & (label[:, None] != label[None, :])
    sim[conflict] = -np.inf
    sim[~alive, :] = -np.inf
    sim[:, ~alive] = -np.inf
    np.fill_diagonal(sim, -np.inf)
    while n > 1:
        k = int(np.argmax(sim))
        i, j = divmod(k, n)
        if not np.isfinite(sim[i, j]) or sim[i, j] < threshold:
            break
        members[i] += members[j]
        sums[i] += sums[j]
        join(i, j)
        alive[j] = False
        if label[i] < 0:
            label[i] = label[j]
        sim[j, :] = -np.inf
        sim[:, j] = -np.inf
        proto[i] = sums[i] / (np.linalg.norm(sums[i]) + 1e-9)
        row = proto @ proto[i]
        row[~alive | busy[i]] = -np.inf
        if label[i] >= 0:
            row[(label >= 0) & (label != label[i])] = -np.inf
        row[i] = -np.inf
        sim[i, :] = row
        sim[:, i] = row
    return [members[i] for i in np.flatnonzero(alive)]


def segment_ends(people):
    """Per segment: first and last time, median pitch position over LINK_EDGE_S at each end and the
    velocity at its end (m/s, clipped to a sprint), from calibrated samples."""
    q = people[people.segment.notna() & np.isfinite(people.x) & np.isfinite(people.y)]
    q = q[['segment', 'time_s', 'x', 'y']].sort_values('time_s')
    rows = {}
    for s, g in q.groupby('segment', sort=False):
        t = g.time_s.to_numpy(float)
        xy = g[['x', 'y']].to_numpy(float)
        head, tail = t <= t[0] + LINK_EDGE_S, t >= t[-1] - LINK_EDGE_S
        v = np.zeros(2)
        if tail.sum() >= 3 and t[tail][-1] - t[tail][0] >= .2:
            v = np.array([np.polyfit(t[tail], xy[tail, k], 1)[0] for k in (0, 1)]).clip(-8, 8)
        x0, y0 = np.median(xy[head], axis=0)
        x1, y1 = np.median(xy[tail], axis=0)
        rows[s] = {'t0': t[0], 't1': t[-1], 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'vx': v[0], 'vy': v[1]}
    return pd.DataFrame.from_dict(rows, orient='index')


def motion_links(ends, segments, evidence, p):
    """(earlier, later) segment pairs of one team that continue each other on the pitch.

    A later segment qualifies when it starts within link_gap seconds of the earlier one's end, no
    further than link_radius + link_speed * gap metres from where the earlier one was heading
    (velocity applied for at most a second), and its appearance is at least link_min similar.
    Pairs are taken best first, one successor and one predecessor per segment, skipping any whose
    runner-up (for either segment) is not clearly worse.
    """
    from . import identity_model as IM
    segs = [s for s in segments if s in ends.index]
    if len(segs) < 2:
        return []
    e = ends.loc[segs]
    t0, t1 = e.t0.to_numpy(float), e.t1.to_numpy(float)
    gap = t0[None, :] - t1[:, None]                                  # [earlier, later]
    horizon = np.minimum(np.clip(gap, 0, None), 1.0)
    px = e.x1.to_numpy(float)[:, None] + e.vx.to_numpy(float)[:, None] * horizon
    py = e.y1.to_numpy(float)[:, None] + e.vy.to_numpy(float)[:, None] * horizon
    dist = np.hypot(e.x0.to_numpy(float)[None, :] - px, e.y0.to_numpy(float)[None, :] - py)
    reach = p['link_radius'] + p['link_speed'] * np.clip(gap, 0, None)
    ok = (gap >= 0) & (gap <= p['link_gap']) & (dist <= reach)
    cost = np.where(ok, dist / reach, np.inf)
    cand = []
    for i, j in zip(*np.nonzero(ok)):
        a, b = segs[i], segs[j]
        if a in evidence and b in evidence and \
                float(IM.embedding(evidence[a]) @ IM.embedding(evidence[b])) < p['link_min']:
            cost[i, j] = np.inf
            continue
        cand.append((float(cost[i, j]), int(i), int(j)))
    links, used_a, used_b = [], set(), set()
    for c, i, j in sorted(cand):
        if i in used_a or j in used_b:
            continue
        rivals = [x for x in np.concatenate([np.delete(cost[i], j), np.delete(cost[:, j], i)]) if np.isfinite(x)]
        if rivals and min(rivals) < p['link_ratio'] * c and min(rivals) - c < .3:
            continue
        used_a.add(i)
        used_b.add(j)
        links.append((segs[i], segs[j]))
    return links


def link_chains(links, ids):
    """Motion links -> chains of row indices into ids (each segment has at most one successor)."""
    pos = {s: k for k, s in enumerate(ids)}
    succ = {a: b for a, b in links if a in pos and b in pos}
    has_pred = set(succ.values())
    chains = []
    for s in ids:
        if s in has_pred or s not in succ:
            continue
        chain = [pos[s]]
        while s in succ:
            s = succ[s]
            chain.append(pos[s])
        chains.append(chain)
    return chains


def _free(spans, others, tolerance=0.0):
    if tolerance > 0:
        shared = sum(max(0.0, min(b, e) - max(a, s)) for a, b in spans for s, e in others)
        return shared <= tolerance
    return not any(_overlaps(others, a, b) for a, b in spans)


def legacy_group_reading(segments, legacy):
    """(number, share, votes) from the legacy readings of a group's segments, or (None, 0, 0)."""
    votes = {}
    for s in segments:
        r = legacy.get(s) or {}
        if r.get('number') is not None and r.get('votes'):
            votes[r['number']] = votes.get(r['number'], 0.0) + float(r['votes']) * float(r.get('confidence') or 1)
    if not votes:
        return None, 0.0, 0.0
    total = sum(votes.values())
    number = max(votes, key=votes.get)
    return number, votes[number] / total, votes[number]


def name_group(segments, model_reading, legacy, p, ev=None):
    """Shirt number for a group under the naming rule, or None (ev: the group's combined evidence,
    which lets 'consensus' check a legacy reading against the model's probabilities)."""
    number, confidence, weight = model_reading
    model_ok = confidence >= p['number_confidence'] and weight >= p['number_evidence']
    old, share, votes = legacy_group_reading(segments, legacy or {})
    legacy_ok = old is not None and share >= p['legacy_share'] and votes >= p['legacy_votes']
    rule = p['naming']
    if rule == 'votes':
        if ev is not None and 'votes' in ev:
            from . import identity_model as IM
            voted, vote_share, _ = IM.vote_reading(ev)
            if voted is None:
                return None
            if vote_share >= p['vote_share']:
                return voted
            agreed = old == voted and share >= p['legacy_agree']
            return voted if agreed and vote_share >= p['vote_agreed'] else None
        rule = 'consensus'
    if rule == 'model':
        return number if model_ok else None
    if rule == 'legacy':
        return old if legacy_ok else None
    if rule == 'agree':
        return number if old is not None and old == number and (model_ok or legacy_ok) else None
    if rule == 'consensus':
        if old is not None and old == number and (model_ok or legacy_ok):
            return number
        if legacy_ok and not model_ok:                              # the model is unsure, not contradicting
            if ev is None or old == number:
                return old
            from . import identity_model as IM
            return old if IM.probability(ev, old) >= p['legacy_ratio'] * confidence else None
        if old is None and confidence >= p['model_alone'] and weight >= p['number_evidence']:
            return number
        return None                                                 # confident disagreement: leave unnamed
    if legacy_ok:                                                   # 'either'
        return old
    return number if confidence >= p['model_alone'] and weight >= p['number_evidence'] else None


def segment_label(ev, legacy_reading, p):
    """A segment's trusted shirt number for grouping constraints (-1 when not trusted).

    A legacy reading passing the usual per-segment thresholds, or a very confident model reading;
    when both are trusted and disagree there is no label.
    """
    from . import identity_model as IM
    old = None
    r = legacy_reading or {}
    if r.get('number') is not None and (r.get('confidence') or 0) >= MIN_NUMBER_SHARE and \
            (r.get('votes') or 0) >= MIN_NUMBER_VOTES:
        old = int(r['number'])
    number, confidence, weight = IM.reading(ev)
    new = number if confidence >= p['model_alone'] and weight >= p['number_evidence'] else None
    if old is not None and new is not None and old != new:
        return -1
    value = old if old is not None else new
    return -1 if value is None else int(value)


# A team's goalkeeper is one person, in one place at a time. On this project's halves the
# goalkeeper role alone had put referees, other players and background detections into the
# goalkeeper (up to a fifth of his time; a second "goalkeeper" box elsewhere on the pitch for
# minutes of some halves, so the highlighted box jumped). Of the goalkeeper-role segments, one
# that does not look like the keeper (similarity to his most confident, longest segments), or that
# is on screen together with him somewhere else on the pitch, returns to outfield grouping;
# segments without thumbnails belong to no player. A second box on the keeper himself (same place,
# same moment: the detector sometimes reports him as goalkeeper and as player) stays his.
KEEPER_SIMILARITY = .5          # below this a goalkeeper-role segment is someone else
KEEPER_OVERLAP_S = 1.0          # shared screen time with the keeper elsewhere that is still allowed
KEEPER_SAME_PLACE_M = 5.0       # mean positions this close while overlapping: a second box on him
KEEPER_SEED = 8                 # confident segments that define his appearance


def select_keeper(keepers, evidence):
    """(goalkeeper segments, segments that return to outfield grouping) of one team.

    keepers: signatures() rows with the goalkeeper role (mean positions xn, yn when available).
    Segments without thumbnails are in neither list (they belong to no player).
    """
    from . import identity_model as IM
    seen = [s for s in keepers.index if evidence.get(s) and evidence[s]['crops'] > 0]
    if not seen:
        return [], []
    prob = {s: evidence[s]['keeper'] / evidence[s]['crops'] for s in seen}
    dur = keepers.duration_s
    confident = [s for s in seen if prob[s] >= KEEPER_PROBABILITY] or seen
    seed = sorted(confident, key=lambda s: -dur[s])[:KEEPER_SEED]
    proto = IM.embedding(IM.combine([evidence[s] for s in seed]))
    where = {s: (float(keepers.xn[s]), float(keepers.yn[s])) for s in seen} if {'xn', 'yn'} <= set(keepers) else {}
    kept, back = [], []
    for s in sorted(seen, key=lambda s: -dur[s]):
        if float(IM.embedding(evidence[s]) @ proto) < KEEPER_SIMILARITY:
            back.append(s)
            continue
        a, b = float(keepers.start_s[s]), float(keepers.end_s[s])
        elsewhere = 0.0
        for k in kept:
            shared = max(0.0, min(b, float(keepers.end_s[k])) - max(a, float(keepers.start_s[k])))
            if shared and not (where and np.hypot(where[s][0] - where[k][0], where[s][1] - where[k][1]) <= KEEPER_SAME_PLACE_M):
                elsewhere += shared
        (back if elsewhere > KEEPER_OVERLAP_S else kept).append(s)
    return kept, back


def resolve_clusters(seg, evidence, params=None, cache=None, legacy=None, ends=None):
    """Segment -> identity for both teams from identity-model evidence.

    seg: signatures() table (team, role, start_s, end_s, n, duration_s, yn, depth); evidence:
    segment -> identity_model evidence. Returns (identity, rows). Named players are
    '<team>-<number>', unnamed ones '<team>-X<k>' (longest first), goalkeepers '<team>-GK'.
    cache (a dict) keeps the grouping per team and similarity when settings are compared; legacy
    maps segment -> legacy reading (football_profiler.jersey) for the NAMING rule; ends
    (segment_ends) enables motion links.
    """
    from . import identity_model as IM
    p = {'similarity': CLUSTER_SIMILARITY, 'number_confidence': NUMBER_CONFIDENCE,
         'number_evidence': NUMBER_EVIDENCE, 'min_player_s': MIN_PLAYER_S, 'attach_margin': ATTACH_MARGIN,
         'attach_slack': ATTACH_SLACK, 'naming': NAMING, 'legacy_votes': LEGACY_GROUP_VOTES,
         'legacy_share': LEGACY_GROUP_SHARE, 'model_alone': MODEL_ALONE_CONFIDENCE,
         'legacy_ratio': LEGACY_MODEL_RATIO, 'vote_share': VOTE_SHARE, 'vote_agreed': VOTE_SHARE_AGREED,
         'legacy_agree': LEGACY_AGREE_SHARE,
         'constraints': NUMBER_CONSTRAINTS, 'max_outfield': MAX_OUTFIELD, 'merge_floor': MERGE_FLOOR,
         'link': LINK, 'link_gap': LINK_GAP_S, 'link_radius': LINK_RADIUS_M, 'link_speed': LINK_SPEED,
         'link_ratio': LINK_RATIO, 'link_min': LINK_MIN_SIMILARITY, 'tolerance': OVERLAP_TOLERANCE_S,
         'name_after_merge': NAME_AFTER_MERGE, **(params or {})}
    identity, rows = {}, []
    for team in ('A', 'B'):
        t = seg[seg.team == team]
        kept, back = select_keeper(t[t.role == 'goalkeeper'], evidence)
        keepers = t.loc[kept]
        for s in kept:
            identity[s] = f'{team}-GK'
        if len(keepers):
            rows.append({'identity': f'{team}-GK', 'team': team, 'role': 'goalkeeper', 'label': 'goalkeeper',
                         'lane_y': float(keepers.yn.mean()), 'depth': 0.0, 'segments': int(len(keepers)),
                         'observed_s': float(keepers.duration_s.sum()), 'named': True})
        out = t[(t.role.eq('player') | t.index.isin(back)) & t.index.isin(list(evidence))]
        if out.empty:
            continue
        ids = list(out.index)
        linked = p['link'] and ends is not None
        key = (team, p['similarity'], p['constraints'], p['tolerance']) + ((p['link_gap'], p['link_radius'], p['link_speed'],
                                                            p['link_ratio'], p['link_min']) if linked else ())
        if cache is not None and key in cache:
            groups = cache[key]
        else:
            emb = np.stack([IM.embedding(evidence[s]) for s in ids])
            w = np.array([max(evidence[s]['crops'], 1) for s in ids], float)
            labels = [segment_label(evidence[s], (legacy or {}).get(s), p) for s in ids] if p['constraints'] else None
            chains = link_chains(motion_links(ends, ids, evidence, p), ids) if linked else None
            groups = group_segments(emb, w, out.start_s.to_numpy(float), out.end_s.to_numpy(float), p['similarity'],
                                    labels, chains, p['tolerance'])
            if cache is not None:
                cache[key] = groups
        players = [{'segments': segs, 'ev': IM.combine([evidence[s] for s in segs]), 'number': None, 'confidence': 0.0,
                    'spans': [(float(out.start_s[s]), float(out.end_s[s])) for s in segs],
                    'duration': float(out.duration_s[segs].sum())}
                   for segs in ([ids[i] for i in g] for g in groups)]
        late = p['name_after_merge']
        if not late:
            _name_players(players, legacy, p)
            players = _settle_numbers(players, p['tolerance'])
        big = [q for q in players if q['number'] is not None or q['duration'] >= p['min_player_s']]
        small = [q for q in players if q['number'] is None and q['duration'] < p['min_player_s']]
        for q in sorted(small, key=lambda q: -q['duration']):
            v = IM.embedding(q['ev'])
            scores = sorted(((float(v @ IM.embedding(big[k]['ev'])), k) for k in range(len(big))
                             if _free(q['spans'], big[k]['spans'], p['tolerance'])), reverse=True)
            if not scores or scores[0][0] < p['similarity'] - p['attach_slack']:
                continue
            if len(scores) > 1 and scores[0][0] - scores[1][0] < p['attach_margin']:
                continue
            k = scores[0][1]
            big[k]['segments'] += q['segments']
            big[k]['spans'] += q['spans']
            big[k]['duration'] += q['duration']
            big[k]['ev'] = IM.combine([big[k]['ev'], q['ev']])
        big = _consolidate(big, p['max_outfield'], p['merge_floor'], p.get('merge_overlap'), p.get('unnamed_only'))
        if late:
            _name_players(big, legacy, p)
            big = _settle_numbers(big, p['tolerance'])
        if (p.get('rosters') or {}).get(team):
            big = _apply_roster(big, p['rosters'][team], p)
        unnamed = sorted([q for q in big if q['number'] is None], key=lambda q: -q['duration'])
        for k, q in enumerate(unnamed, 1):
            q['name'] = f'{team}-X{k}'
        for q in big:
            ident = q.get('name') or f"{team}-{q['number']}"
            for s in q['segments']:
                identity[s] = ident
            m = out.loc[q['segments']]
            wts = m.n.to_numpy(float)
            lane = float(np.average(m.yn, weights=wts))
            depth = float(np.average(np.nan_to_num(m.depth, nan=.5), weights=wts))
            label = slot_name(lane, depth)
            if q['number'] is None:
                label += ' · unnamed ' + ident.split('-X')[-1]
            guess, guess_share, _ = IM.vote_reading(q['ev'])
            rows.append({'identity': ident, 'team': team, 'role': 'player',
                         'number': int(q['number']) if q['number'] is not None else None,
                         'label': label, 'lane_y': lane, 'depth': depth, 'segments': int(len(q['segments'])),
                         'appearance_segments': int(len(q['segments']) - 1), 'observed_s': float(m.duration_s.sum()),
                         'number_confidence': round(float(q['confidence']), 3), 'named': q['number'] is not None,
                         'number_guess': guess, 'guess_share': round(float(guess_share), 3),
                         'readable_views': int(q['ev'].get('legible', 0))})
    return identity, pd.DataFrame(rows)


def _name_players(players, legacy, p):
    """Give each group its shirt number under the naming rule (None when unnamed) and its confidence."""
    from . import identity_model as IM
    for q in players:
        reading = IM.reading(q['ev'])
        number = name_group(q['segments'], reading, legacy, p, q['ev'])
        voted, vote_share, _ = IM.vote_reading(q['ev'])
        if p['naming'] == 'votes' and number is not None and number == voted:
            conf = vote_share
        elif number == reading[0]:
            conf = reading[1]
        else:
            conf = legacy_group_reading(q['segments'], legacy or {})[1]
        q['number'], q['confidence'] = number, conf


CONFIRMED_NAME = re.compile(r'^([AB])-(\d{1,2}|GK|P\d+)$')    # number, goalkeeper, or a reviewer's person
NOT_A_PLAYER = 'NONE'           # a reviewer's mark for a referee, coach or other non-player


def apply_confirmed(identity, rows, confirmed, seg):
    """Reviewer names (segment -> identity) over the resolver's, and a row for each new name.

    A segment keeps its reviewer name through re-analysis. A name like 'A-8' that the resolver did
    not produce gets its own row (number 8, a lane label from its segments); rows of names a
    reviewer gave are marked confirmed. Segments marked NOT_A_PLAYER belong to no player.
    """
    identity = dict(identity)
    for s, ident in confirmed.items():
        if s in identity:
            if ident == NOT_A_PLAYER:
                del identity[s]
            else:
                identity[s] = ident
    given = {confirmed[s] for s in confirmed if s in identity}
    rows = rows.copy()
    rows['confirmed'] = rows.identity.isin(given) if len(rows) else pd.Series(dtype=bool)
    new = []
    for ident in sorted(given - set(rows.identity if len(rows) else [])):
        segs = [s for s, i in identity.items() if i == ident and s in seg.index]
        if not segs:
            continue
        m = seg.loc[segs]
        w = m.n.to_numpy(float)
        lane = float(np.average(m.yn, weights=w))
        depth = float(np.average(np.nan_to_num(m.depth.to_numpy(float), nan=.5), weights=w))
        match = CONFIRMED_NAME.match(ident)
        keeper = bool(match) and match.group(2) == 'GK'
        person = bool(match) and match.group(2).startswith('P')
        number = int(match.group(2)) if match and not keeper and not person else None
        label = 'goalkeeper' if keeper else slot_name(lane, depth) + (f' · confirmed player {match.group(2)[1:]}' if person else '')
        new.append({'identity': ident, 'team': match.group(1) if match else m.team.mode().iat[0],
                    'role': 'goalkeeper' if keeper else 'player', 'number': number,
                    'label': label, 'lane_y': lane, 'depth': depth,
                    'segments': len(segs), 'observed_s': float(m.duration_s.sum()), 'named': number is not None or keeper,
                    'confirmed': True})
    if new:
        rows = pd.concat([rows, pd.DataFrame(new)], ignore_index=True)
    return identity, rows


def _apply_roster(players, roster, p):
    """Names checked against the line-up (see ROSTER_SHARE); returns the settled players."""
    allowed_all = {int(n) for n in roster if 0 <= int(n) < 100}
    taken = {q['number'] for q in players if q['number'] in allowed_all}
    for q in sorted(players, key=lambda q: -q['duration']):
        if q['number'] is not None and q['number'] in allowed_all:
            continue
        votes = (q['ev'] or {}).get('votes')
        q['number'] = None                                    # outside the line-up: a misreading
        if votes is None:
            continue
        allowed = sorted(allowed_all - taken)
        v = np.zeros(100)
        v[allowed] = np.asarray(votes)[allowed]
        total = float(np.asarray(votes).sum())
        if v.sum() > 0 and v.max() / v.sum() >= ROSTER_SHARE and v.max() >= p.get('roster_min_of_all', ROSTER_MIN_OF_ALL) * total:
            q['number'], q['confidence'], q['from_lineup'] = int(v.argmax()), float(v.max() / v.sum()), True
            taken.add(q['number'])
    return _settle_numbers(players, p['tolerance'])


def _timeline(spans, bins):
    t = np.zeros(bins, bool)
    for a, b in spans:
        t[int(a / TIMELINE_BIN_S):int(b / TIMELINE_BIN_S) + 1] = True
    return t


def _consolidate(players, cap, floor, overlap=None, unnamed_only=None):
    """Merge the most similar (almost) never-simultaneous groups while a team has more than cap players."""
    overlap = MERGE_OVERLAP if overlap is None else overlap
    unnamed_only = MERGE_UNNAMED_ONLY if unnamed_only is None else unnamed_only
    from . import identity_model as IM
    if not cap or len(players) <= cap:
        return players
    bins = int(max(b for q in players for _, b in q['spans']) / TIMELINE_BIN_S) + 2
    lines = [_timeline(q['spans'], bins) for q in players]
    emb = [IM.embedding(q['ev']) for q in players]
    alive = [True] * len(players)
    while sum(alive) > cap:
        best = None
        for i in range(len(players)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(players)):
                if not alive[j] or (players[i]['number'] is not None and players[j]['number'] is not None):
                    continue
                if unnamed_only and (players[i]['number'] is not None or players[j]['number'] is not None):
                    continue
                shared = int((lines[i] & lines[j]).sum())
                if shared and shared > overlap * min(int(lines[i].sum()), int(lines[j].sum())):
                    continue
                sim = float(emb[i] @ emb[j])
                if best is None or sim > best[0]:
                    best = (sim, i, j)
        if best is None or best[0] < floor:
            break
        _, i, j = best
        if players[i]['number'] is None and players[j]['number'] is not None:
            i, j = j, i
        a, b = players[i], players[j]
        a['segments'] += b['segments']
        a['spans'] += b['spans']
        a['duration'] += b['duration']
        a['ev'] = IM.combine([a['ev'], b['ev']])
        lines[i] |= lines[j]
        emb[i] = IM.embedding(a['ev'])
        alive[j] = False
    return [q for q, keep in zip(players, alive) if keep]


def _settle_numbers(players, tolerance=0.0):
    """One player per shirt number.

    Groups with the same number that never overlap are one player; of overlapping ones the surer
    keeps the number and the others become unnamed.
    """
    from . import identity_model as IM
    by_number = {}
    for q in players:
        if q['number'] is not None:
            by_number.setdefault(q['number'], []).append(q)
    for qs in by_number.values():
        qs.sort(key=lambda q: -q['confidence'])
        keep = qs[0]
        for q in qs[1:]:
            if _free(q['spans'], keep['spans'], tolerance):
                keep['segments'] += q['segments']
                keep['spans'] += q['spans']
                keep['duration'] += q['duration']
                keep['ev'] = IM.combine([keep['ev'], q['ev']])
                q['merged'] = True
            else:
                q['number'] = None
    return [q for q in players if not q.get('merged')]


# Anchor-first variant: players named from trusted readings first, then every other segment joins
# the named player it clearly resembles (one decision per segment, no chains of merges), and only
# what is left is grouped into unnamed players. On SoccerNet tracking clips, grouping by
# appearance alone split each player into about four groups across camera cuts, so names reached
# few of them; anchors carry a name across cuts through the readings themselves.
ANCHOR_SEGMENTS = 2            # trusted readings behind a named player
MAX_PLAYERS = 16               # named outfield players per team and half
ATTACH_MIN_SIMILARITY = .5
ATTACH_ROUNDS = 2
STRATEGY = 'group'             # 'group' (resolve_clusters) or 'anchor' (resolve_anchor_first)


def resolve_anchor_first(seg, evidence, params=None, legacy=None, cache=None):
    """Segment -> identity for both teams: named anchors, appearance attachment, unnamed groups."""
    from . import identity_model as IM
    p = {'similarity': CLUSTER_SIMILARITY, 'number_confidence': NUMBER_CONFIDENCE,
         'number_evidence': NUMBER_EVIDENCE, 'min_player_s': MIN_PLAYER_S, 'attach_margin': ATTACH_MARGIN,
         'attach_slack': ATTACH_SLACK, 'naming': NAMING, 'legacy_votes': LEGACY_GROUP_VOTES,
         'legacy_share': LEGACY_GROUP_SHARE, 'model_alone': MODEL_ALONE_CONFIDENCE, 'constraints': False,
         'anchor_segments': ANCHOR_SEGMENTS, 'max_players': MAX_PLAYERS, 'attach_min': ATTACH_MIN_SIMILARITY,
         'rounds': ATTACH_ROUNDS, **(params or {})}
    identity, rows = {}, []
    for team in ('A', 'B'):
        t = seg[seg.team == team]
        keepers = t[t.role == 'goalkeeper']
        for s in keepers.index:
            identity[s] = f'{team}-GK'
        if len(keepers):
            rows.append({'identity': f'{team}-GK', 'team': team, 'role': 'goalkeeper', 'label': 'goalkeeper',
                         'lane_y': float(keepers.yn.mean()), 'depth': 0.0, 'segments': int(len(keepers)),
                         'observed_s': float(keepers.duration_s.sum()), 'named': True})
        out = t[(t.role == 'player') & t.index.isin(list(evidence))]
        if out.empty:
            continue
        span = {s: (float(out.start_s[s]), float(out.end_s[s])) for s in out.index}
        label = {s: segment_label(evidence[s], (legacy or {}).get(s), p) for s in out.index}
        by_number = {}
        for s, v in label.items():
            if v >= 0:
                by_number.setdefault(v, []).append(s)
        weight = {n: sum(evidence[s]['weight'] for s in ss) for n, ss in by_number.items()}
        strong = [n for n in sorted(by_number, key=lambda n: -weight[n])
                  if len(by_number[n]) >= p['anchor_segments']][:p['max_players']]
        players = {}
        for n in strong:
            kept = []
            for s in sorted(by_number[n], key=lambda s: -evidence[s]['weight']):
                if _free([span[s]], [span[k] for k in kept]):
                    kept.append(s)
            players[n] = kept
        rest = [s for s in out.index if not any(s in ss for ss in players.values())]
        for _ in range(p['rounds']):
            protos = {n: IM.embedding(IM.combine([evidence[s] for s in ss])) for n, ss in players.items()}
            busy = {n: [span[s] for s in ss] for n, ss in players.items()}
            left = []
            for s in sorted(rest, key=lambda s: -evidence[s]['crops']):
                v = IM.embedding(evidence[s])
                scores = sorted(((float(v @ protos[n]), n) for n in protos if _free([span[s]], busy[n])),
                                reverse=True)
                if not scores or scores[0][0] < p['attach_min'] or \
                        (len(scores) > 1 and scores[0][0] - scores[1][0] < p['attach_margin']):
                    left.append(s)
                    continue
                n = scores[0][1]
                players[n].append(s)
                busy[n].append(span[s])
            if len(left) == len(rest):
                break
            rest = left
        # What is left: unnamed players by appearance (a group may still earn a free number).
        groups = []
        if rest:
            emb = np.stack([IM.embedding(evidence[s]) for s in rest])
            w = np.array([max(evidence[s]['crops'], 1) for s in rest], float)
            idx = group_segments(emb, w, np.array([span[s][0] for s in rest]), np.array([span[s][1] for s in rest]),
                                 p['similarity'])
            groups = [[rest[i] for i in g] for g in idx]
        unnamed = []
        for g in groups:
            ev = IM.combine([evidence[s] for s in g])
            number = name_group(g, IM.reading(ev), legacy, p)
            if number is not None and number not in players:
                players[number] = g
            elif number is not None and _free([span[s] for s in g], [span[s] for s in players[number]]):
                players[number] += g
            else:
                unnamed.append(g)
        dur = lambda g: sum(span[s][1] - span[s][0] for s in g)
        keep = sorted([g for g in unnamed if dur(g) >= p['min_player_s']], key=lambda g: -dur(g))
        final = [(f'{team}-{n}', ss, n) for n, ss in players.items()]
        final += [(f'{team}-X{k}', g, None) for k, g in enumerate(keep, 1)]
        for ident, ss, n in final:
            for s in ss:
                identity[s] = ident
            m = out.loc[ss]
            wts = m.n.to_numpy(float)
            lane = float(np.average(m.yn, weights=wts))
            depth = float(np.average(np.nan_to_num(m.depth, nan=.5), weights=wts))
            rows.append({'identity': ident, 'team': team, 'role': 'player', 'number': int(n) if n is not None else None,
                         'label': slot_name(lane, depth) + ('' if n is not None else ' · unnamed ' + ident.split('-X')[-1]),
                         'lane_y': lane, 'depth': depth, 'segments': int(len(ss)),
                         'appearance_segments': int(sum(1 for s in ss if label.get(s, -1) != n)),
                         'observed_s': float(m.duration_s.sum()), 'named': n is not None})
    return identity, pd.DataFrame(rows)


def resolve_model(seg, evidence, params=None, legacy=None, cache=None, ends=None):
    """The identity-model resolver chosen by STRATEGY (params['strategy'] overrides)."""
    strategy = (params or {}).get('strategy', STRATEGY)
    if strategy == 'anchor':
        return resolve_anchor_first(seg, evidence, params, legacy=legacy, cache=cache)
    return resolve_clusters(seg, evidence, params, cache=cache, legacy=legacy, ends=ends)
