"""Full-match analysis orchestration: video half -> app dataset with per-player statistics.

Stage 1 (GPU) is match_analysis.detection_pass. postprocess() then runs every
CPU stage from the raw evidence and writes an app-compatible dataset, so the
analysis can be refined and re-run without repeating detection.
"""
from __future__ import annotations

import os
import re
import time
import zipfile

import numpy as np
import pandas as pd

from . import match_events as ME
from . import match_identity as MI
from . import match_post as MP
from . import match_shots as SH
from . import match_stats as MS
from . import storage as S
from . import taxonomy as T

ANALYSIS_VERSION = 'full-match-1'


def fixture_teams(title):
    """'2015-02-21 - 18-00 Chelsea 1 - 1 Burnley' -> ('Chelsea', 'Burnley')."""
    m = re.search(r'\d{4}-\d{2}-\d{2} - \d{2}-\d{2} (.+?) \d+ - \d+ (.+)$', title or '')
    return (m.group(1), m.group(2)) if m else (None, None)


def suggest_position_group(role, positional):
    """Archetype position group from where the player actually stood (attack-normalised)."""
    if role == 'goalkeeper':
        return 'goalkeeper'
    x, y = positional.get('mean_x'), positional.get('mean_y')
    if x is None or y is None:
        return None
    wide = abs(y - 34) > 14
    if x < 38:
        return 'full_back' if wide else 'centre_back'
    if x < 50:
        return 'full_back' if wide else 'defensive_midfield'
    if x < 60:
        return 'wide_attacker' if wide else 'central_midfield'
    if x < 68:
        return 'wide_attacker' if wide else 'attacking_midfield'
    return 'wide_attacker' if wide else 'centre_forward'


def _crops_by_segment(directory, segment_tracks):
    """BGR thumbnails per segment from crops.zip (keys are raw tracklet ids)."""
    import cv2
    path = directory / 'crops.zip'
    if not path.is_file():
        return {}
    wanted = {}
    for seg, tracks in segment_tracks.items():
        for t in tracks:
            wanted[t] = seg
    out = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            track = name.split('/')[0]
            seg = wanted.get(track)
            if seg is None:
                continue
            img = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                out.setdefault(seg, []).append(img)
    return out


# Which identity reader post-processing uses: 'identity_model' (football_profiler.identity_model,
# trained on FOOTPASS true identities; used when its weights are present) or 'legacy' (the
# legibility classifier and PARSeq with the pseudo-label appearance head). On held-out FOOTPASS
# games the identity model gave the right player for 47-51% of visible player time at 93-95%
# accuracy, against 26% at 49% for the legacy path (DETECTION_RESEARCH.md section 6).
IDENTITY_READER = os.environ.get('PITCHPROFILE_IDENTITY_READER', 'identity_model')


def use_identity_model():
    from . import identity_model as IM
    return IDENTITY_READER == 'identity_model' and IM.available()


def track_evidence(directory):
    """{raw tracklet id: identity-model evidence} for the half's thumbnails (cached).

    The cache (identity_evidence.npz) is rebuilt when crops.zip or the model weights are newer.
    """
    import cv2
    from . import identity_model as IM
    path, cache = directory / 'crops.zip', directory / 'identity_evidence.npz'
    if not path.is_file():
        return {}
    if cache.is_file() and cache.stat().st_mtime >= max(path.stat().st_mtime, IM.weights_path().stat().st_mtime):
        ev = np.load(cache, allow_pickle=True)['evidence'].item()
        from . import jersey as J
        # Evidence cached before legible-thumbnail votes existed is rebuilt when they can be computed.
        if not ev or 'votes' in next(iter(ev.values())) or not J.available():
            return ev
    groups = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            img = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                groups.setdefault(name.split('/')[0], []).append(img)
    ev, keys = {}, list(groups)
    for i in range(0, len(keys), 400):
        ev.update(IM.evidence({k: groups[k] for k in keys[i:i + 400]}))
    np.savez_compressed(cache, evidence=np.array(ev, dtype=object))
    return ev


# The identity model's goalkeeper output mistakes some outfield kits for a goalkeeper's (on the
# yellow kits of Villarreal and Barcelona the median player of that team scored 0.9, while on
# other halves the median is about 0 and the calls cover 1-5% of a team's detections). Its calls
# are ignored for a team whose median player scores above KEEPER_TEAM_MEDIAN or whose calls
# cover more than KEEPER_MAX_SHARE of its detections (the detector's goalkeeper class then
# decides); a called goalkeeper must also stay within the penalty area's depth of a goal line
# (mistaken calls stood 16-20 m out on Ajax v Barcelona, where the kit clusters mix the teams;
# on FOOTPASS validation this costs 3 points of goalkeepers found, 91.5% -> 88.5%). Not
# necessarily the goal of his kit cluster's team: a goalkeeper's kit often lands in the other
# team's cluster, which is why match_post.assign_keepers gives keepers the team whose goal
# they stay near.
KEEPER_MAX_SHARE = .15
KEEPER_TEAM_MEDIAN = .3
KEEPER_GOAL_DISTANCE_M = 16.5         # the penalty area


def override_keepers(t, keep, outfield, probability=None):
    """Tracklet roles with the identity model's goalkeeper calls applied (see KEEPER_MAX_SHARE).

    probability: tracklet -> the model's mean goalkeeper probability.
    """
    t = t.copy()
    near_goal = (t.mean_x <= KEEPER_GOAL_DISTANCE_M) | (t.mean_x >= MI.L - KEEPER_GOAL_DISTANCE_M)
    called = t.track.isin(keep) & t.role.ne('referee') & near_goal.fillna(False).astype(bool)
    for team in ('A', 'B'):
        mine = t.team.eq(team)
        share = t.loc[mine & called, 'observations'].sum() / max(t.loc[mine, 'observations'].sum(), 1)
        probs = t.loc[mine, 'track'].map(probability or {}).dropna()
        typical = float(np.median(probs)) if len(probs) else 0.0
        if share > KEEPER_MAX_SHARE or typical > KEEPER_TEAM_MEDIAN:
            called &= ~mine
    t.loc[called, 'role'] = 'goalkeeper'
    t.loc[t.track.isin(outfield) & t.role.eq('goalkeeper'), 'role'] = 'player'
    return t


def keeper_tracks(track_ev, min_crops=2):
    """(goalkeeper tracklets, outfield tracklets, tracklet -> goalkeeper probability) as the
    identity model reads them; tracklets with fewer than min_crops thumbnails are in neither set
    and keep the detector's role."""
    keep, outfield, probability = set(), set(), {}
    for t, e in track_ev.items():
        if e['crops'] >= min_crops:
            probability[t] = e['keeper'] / e['crops']
            (keep if probability[t] >= MI.KEEPER_PROBABILITY else outfield).add(t)
    return keep, outfield, probability


def segment_evidence(people, track_ev):
    from . import identity_model as IM
    seg_tracks = people[people.segment.notna()].groupby('segment').track.agg(lambda t: sorted(set(t))).to_dict()
    out = {}
    for s, tracks in seg_tracks.items():
        ev = IM.combine([track_ev.get(t) for t in tracks])
        if ev is not None:
            out[s] = ev
    return out


def model_readings(evidence):
    """(numbers, embeddings) per segment, in the shapes read_appearance returns."""
    from . import identity_model as IM
    numbers, embeddings = {}, {}
    for s, ev in evidence.items():
        number, confidence, weight = IM.reading(ev)
        numbers[s] = {'number': number, 'confidence': confidence, 'votes': weight, 'crops': ev['crops'],
                      'keeper_probability': ev['keeper'] / max(ev['crops'], 1), 'reader': 'identity_model'}
        embeddings[s] = (IM.embedding(ev), ev['crops'])
    return numbers, embeddings


def legacy_crop_readings(directory):
    """Per thumbnail of crops.zip: (raw tracklet, legibility, PARSeq text, PARSeq confidence).

    Cached in legacy_readings.npz (rebuilt when crops.zip is newer), so re-running
    post-processing does not read every thumbnail again.
    """
    import cv2
    from . import jersey as J
    path, cache = directory / 'crops.zip', directory / 'legacy_readings.npz'
    if cache.is_file() and cache.stat().st_mtime >= path.stat().st_mtime:
        z = np.load(cache)
        return z['track'], z['legibility'], z['text'], z['confidence']
    tracks, crops = [], []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            img = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                tracks.append(name.split('/')[0])
                crops.append(img)
    scores, text, conf = J.read_crops(crops)
    track, text = np.array(tracks, dtype='U'), text.astype('U')
    np.savez_compressed(cache, track=track, legibility=scores.astype(np.float32), text=text, confidence=conf)
    return track, scores, text, conf


def legacy_numbers(directory, people):
    """Legibility-classifier and PARSeq readings per segment ({} without the recogniser)."""
    from . import jersey as J
    if not J.available() or not (directory / 'crops.zip').is_file():
        return {}
    seg_tracks = people[people.segment.notna()].groupby('segment').track.agg(lambda s: sorted(set(s))).to_dict()
    track, scores, text, conf = legacy_crop_readings(directory)
    order = np.argsort(track, kind='stable')
    names, starts = np.unique(track[order], return_index=True)
    rows = {t: order[a:b] for t, a, b in zip(names, starts, list(starts[1:]) + [len(order)])}
    out = {}
    for seg, tracks in seg_tracks.items():
        idx = np.concatenate([rows[t] for t in tracks if t in rows] or [np.zeros(0, int)])
        if len(idx):
            out[seg] = J.aggregate(scores[idx], text[idx], conf[idx])
    return out


def read_numbers(directory, people):
    """Jersey reading per segment, or {} when the recogniser is unavailable."""
    return read_appearance(directory, people)[0]


def read_appearance(directory, people):
    """(jersey reading per segment, appearance embedding and thumbnail count per segment).

    Used only when the identity model's weights are missing: shirt numbers are read with the
    legibility classifier and PARSeq, and there are no appearance embeddings (the second value
    is always {}); the identity model supplies those on the normal path.
    """
    from . import jersey as J
    if not J.available():
        return {}, {}
    seg_tracks = people.groupby('segment').track.agg(lambda s: sorted(set(s))).to_dict()
    crops = _crops_by_segment(directory, seg_tracks)
    if not crops:
        return {}, {}
    return J.read_groups(crops), {}


def save_appearance(directory, people, numbers, embeddings):
    """Per-segment identity evidence (appearance.npz): every team segment, identified or not.

    Team, role, time span, samples, thumbnails, shirt-number reading and appearance embedding;
    used to measure identity linking offline (scripts/evaluate_linking.py).
    """
    q = people[people.segment.notna() & people.team.isin(['A', 'B'])]
    if q.empty:
        return
    seg = q.groupby('segment').agg(team=('team', 'first'), role=('role', 'first'), start_s=('time_s', 'min'),
                                   end_s=('time_s', 'max'), samples=('sample', 'size'))
    dim = len(next(iter(embeddings.values()))[0]) if embeddings else 0
    emb = np.zeros((len(seg), dim), np.float16)
    crops = np.zeros(len(seg), np.int32)
    for i, s in enumerate(seg.index):
        if s in embeddings:
            emb[i], crops[i] = embeddings[s][0], embeddings[s][1]
    read = [numbers.get(s) or {} for s in seg.index]
    tracks = q.groupby('segment').track.agg(lambda t: '|'.join(sorted(set(map(str, t)))))
    np.savez_compressed(directory / 'appearance.npz', segment=np.array(seg.index.astype(str), dtype='U'),
                        tracks=np.array([tracks[s] for s in seg.index], dtype='U'),
                        team=np.array(seg.team.astype(str), dtype='U'), role=np.array(seg.role.astype(str), dtype='U'),
                        start_s=seg.start_s.to_numpy(float), end_s=seg.end_s.to_numpy(float),
                        samples=seg.samples.to_numpy(int), crops=crops, embedding=emb,
                        number=np.array([r.get('number') if r.get('number') is not None else -1 for r in read], int),
                        number_confidence=np.array([r.get('confidence') or 0.0 for r in read], float),
                        number_votes=np.array([r.get('votes') or 0.0 for r in read], float))


def stages(identifier, progress=lambda *a: None, keepers=None):
    """CPU stages from the raw detection evidence up to the ball path.

    Returns (analysis meta, sample rate, frames, people with team/role, ball path,
    attack directions, team info). keepers = (goalkeeper tracklets, outfield tracklets) from the
    identity model overrides the detector's role vote; goalkeepers then take the team whose goal
    they stay near (match_post.assign_keepers).
    """
    d = S.dataset_dir(identifier)
    meta = S.read_json(d / 'analysis.json')
    hz = meta['detection']['sample_hz']
    raw_frames = pd.read_csv(d / 'raw_frames.csv.gz')
    raw_people = pd.read_csv(d / 'raw_people.csv.gz', dtype={'track': str})
    raw_ball = pd.read_csv(d / 'raw_ball.csv.gz')
    kits = S.read_json(d / 'raw_kits.json', {})
    shape = (meta['detection']['height'], meta['detection']['width'])
    progress(.05, 'Filling and smoothing pitch calibration')
    frames = MP.fill_calibration(raw_frames, shape)
    people = MP.project_people(raw_people, frames)
    progress(.2, 'Grouping kits into teams and inferring attack directions')
    t = MP.tracklets(people, kits)
    t, team_info = MP.assign_teams(t)
    directions = MP.attack_directions(t)
    if keepers:
        t = override_keepers(t, *keepers)
    t = MP.assign_keepers(t, directions)
    people = people.merge(t[['team', 'role']], left_on='tracklet', right_index=True, how='left')
    progress(.35, 'Reconstructing the ball path')
    ball = MP.ball_path(raw_ball, frames, hz)
    return meta, hz, frames, people, ball, directions, team_info


def player_moves(old_segments, identity, weight):
    """{old player id: new player id} for players whose footage now carries another id.

    old_segments: previous identity -> its segments; identity: segment -> new identity; weight:
    segment -> screen time (samples). A player moves to the new id holding most of his previous
    footage, when that is at least half of it.
    """
    moves = {}
    for old, segs in old_segments.items():
        time_by = {}
        for s in segs:
            if identity.get(s) is not None:
                time_by[identity[s]] = time_by.get(identity[s], 0) + weight.get(s, 0)
        total = sum(weight.get(s, 0) for s in segs)
        if time_by and total:
            new, t = max(time_by.items(), key=lambda kv: kv[1])
            if new != old and t / total >= .5:
                moves[old] = new
    return moves


def lineup_numbers(manifest):
    """{kit group: outfield shirt numbers that played} from the match line-up, only when the
    reviewer connected the match and confirmed which kit group is which team in the app;
    {} otherwise. A half without its own confirmation uses its whole match's (kit groups mapped)."""
    from . import match_context as MC

    def from_binding(m):
        info = MC.state_for(m)
        context, binding = info['context'], info['binding']
        if not context or not binding.get('teams'):
            return {}
        teams = {t['id']: t for t in context['teams']}
        return {kit: {int(q['shirt']) for q in teams[tid]['players']
                      if q.get('played') and q.get('shirt') is not None and 'goalkeeper' not in str(q.get('position', '')).lower()}
                for kit, tid in binding['teams'].items() if tid in teams}
    own = from_binding(manifest)
    if own:
        return own
    from . import match_merge as MM
    whole_id = MM.whole_match_of(manifest.get('id', ''))
    whole = S.read_json(S.DATA / 'datasets' / whole_id / 'manifest.json', {}) if whole_id else {}
    if not whole:
        return {}
    rosters = from_binding(whole)
    kits = whole.get('kit_mapping_second_half') or {'A': 'A', 'B': 'B'}
    return rosters if manifest.get('half') != 2 else {k: rosters[v] for k, v in kits.items() if v in rosters}


def half_state(identifier, progress=lambda *a: None, confirmed=None):
    """Everything post-processing derives for one half before statistics are written: frames,
    people (with team, role and segment), ball path, events, possession spells, attack
    directions, identities and their rows. Whole matches (match_merge) combine two of these."""
    d = S.dataset_dir(identifier)
    track_ev = track_evidence(d) if use_identity_model() else {}
    meta, hz, frames, people, ball, directions, team_info = stages(
        identifier, progress, keepers=keeper_tracks(track_ev) if track_ev else None)
    progress(.5, 'Detecting touches, passes, shots and duels')
    shot_model = None if os.environ.get('PITCHPROFILE_NO_SHOT_MODEL') else SH.model_for(meta.get('match_id'))    # the variable is for measuring without it
    events, spells, pos, people, segments = ME.events(people, ball, directions, hz, frames=frames, shot_model=shot_model)
    events, spotted = merge_spotted(d, events, spells, people, ball, directions, hz)
    progress(.65, 'Reading jersey numbers and linking players across cuts')
    confirmed = {**S.read_json(d / 'confirmed_identities.json', {}), **(confirmed or {})}
    if track_ev:
        evidence = segment_evidence(people, track_ev)
        numbers, embeddings = model_readings(evidence)
        legacy = {} if os.environ.get('PITCHPROFILE_NO_LEGACY_READER') else legacy_numbers(d, people)    # the variable is for measuring without it
        save_appearance(d, people, legacy or numbers, embeddings)
        signatures = MI.signatures(people, directions)
        rosters = lineup_numbers(S.read_json(d / 'manifest.json', {}))
        identity, identities = MI.resolve_model(signatures, evidence, params={'rosters': rosters} if rosters else None,
                                                legacy=legacy, ends=MI.segment_ends(people))
        identity, identities = MI.apply_confirmed(identity, identities, confirmed, signatures)
        meta['lineup_used'] = bool(rosters)
    else:
        evidence = {}
        numbers, embeddings = read_appearance(d, people)
        save_appearance(d, people, numbers, embeddings)
        identity, identities, signatures = MI.resolve(people, directions, confirmed=confirmed, numbers=numbers,
                                                      embeddings=embeddings)
    return {'directory': d, 'meta': meta, 'hz': hz, 'frames': frames, 'people': people, 'ball': ball, 'events': events,
            'spells': spells, 'directions': directions, 'team_info': team_info, 'identity': identity,
            'identities': identities, 'numbers': numbers, 'signatures': signatures, 'evidence': evidence,
            'live_seconds': float(frames.pitch_view.sum() / hz), 'spotted': spotted, 'shot_model': shot_model,
            'reader': 'identity_model' if track_ev else 'legacy'}


def postprocess(identifier, progress=lambda *a: None, confirmed=None):
    d = S.dataset_dir(identifier)
    started = time.perf_counter()
    old_segments = {}
    for s, v in (S.read_json(d / 'identities.json', {}) or {}).get('segments', {}).items():
        old_segments.setdefault(v.get('identity'), []).append(s)
    h = half_state(identifier, progress, confirmed)
    meta, people, identity, identities, numbers = h['meta'], h['people'], h['identity'], h['identities'], h['numbers']
    progress(.8, 'Computing per-player statistics')
    stats = MS.build(people, h['events'], h['spells'], identity, h['directions'], h['hz'], h['live_seconds'])
    progress(.9, 'Writing the dataset')
    write_dataset(identifier, meta, h['frames'], people, h['ball'], h['events'], identity, identities, stats,
                  h['team_info'], h['directions'], numbers, h['hz'], h['live_seconds'])
    # Ratings, bookmarks and corrections follow each player to his new id (renames, re-analysis).
    moves = player_moves(old_segments, identity, people.groupby('segment').size().to_dict())
    moved = S.move_player_records(identifier, moves)
    shot_model = h['shot_model']
    meta['postprocess'] = {'player_moves': moves, 'records_moved': moved, 'version': ANALYSIS_VERSION,
                           'wall_seconds': time.perf_counter() - started,
                           'created': S.now(), 'identities': int(len(set(identity.values()))),
                           'identity_reader': h['reader'], 'lineup_used': bool(meta.get('lineup_used')),
                           'jersey_numbers_read': int(sum(1 for v in numbers.values() if v.get('number') is not None
                                                          and v.get('confidence', 1) >= MI.NUMBER_CONFIDENCE)),
                           'named_players': int(identities['named'].sum()) if 'named' in identities else None,
                           'action_spotter': h['spotted'],
                           'appearance_attached_segments': int(identities['appearance_segments'].sum())
                           if 'appearance_segments' in identities else 0,
                           'shots': 'heuristic' if shot_model is None else
                           f"{'cross-fitted ' if shot_model.get('cross_fitted') else ''}model {shot_model['created']}"}
    S.write_json(d / 'analysis.json', meta)
    progress(1, 'Match analysis complete')
    return meta


# Free kicks are not used: against SoccerNet's free-kick labels on these halves the published
# spotter found 13-20% of them and the fine-tune sums to 2.8 times their number (FOOTPASS training
# labels let a pass read as a free kick), so neither gives a usable count.
SPOTTER_TYPES = ('tackle', 'block', 'header', 'high_pass', 'throw_in', 'cross', 'shot')
SHOT_MATCH_S = 1.5
SHOT_OVERRIDE = os.environ.get('PITCHPROFILE_SHOT_OVERRIDE', 'on') != 'off'    # 'off' credits spotted shots with the picture rule only (for measuring)


def merge_spotted(directory, events, spells, people, ball, directions, hz):
    """Add the video action spotter's events (action_spots.json) to the tracking-rule events.

    Spotted tackles replace the rule-based tackles, which found under a fifth of annotated tackles
    on SoccerTrack ground truth; for confident tackles the opposing player on the ball is marked
    dispossessed. Blocks, headers, lofted passes, set pieces and crosses are added. Every spot
    keeps its score, and statistics count scores (expected counts). Returns (events, summary).
    """
    from . import action_spotting as AS
    saved = S.read_json(directory / 'action_spots.json')
    if not saved:
        return events, None
    found = pd.DataFrame(AS.attribute(saved['spots'], people, ball, directions, hz))
    summary = {'model': saved.get('model'), 'created': saved.get('created'), 'thresholds': AS.thresholds()}
    if found.empty:
        return events, {**summary, 'events': 0}
    ev = events.copy()
    ev['source'] = 'rules'
    filled = fill_from_spotter(found, ev) if FILL_FROM_SPOTTER else pd.DataFrame()
    found = found[found.type.isin(SPOTTER_TYPES)].copy()
    # Shots: on SoccerNet labels the spotter found 0.71/0.70 (precision/recall) against the shot
    # model's 0.36/0.47 on train+validation halves (0.78/0.63 vs 0.35/0.56 on held-out test
    # halves), so confident spotted shots replace the model's. The shooter is taken from the
    # model's release by that team within SHOT_MATCH_S when there is one (tracking knows who
    # struck it), otherwise the nearest player to the ball at the spot.
    found = found[~(found.type.eq('shot') & ~found.confident)]
    rule_shots = ev[ev.type.eq('shot')]
    for i, r in found[found.type.eq('shot')].iterrows():
        near = rule_shots[(rule_shots.team == r.team) & ((rule_shots.time_s - r.time_s).abs() <= SHOT_MATCH_S)]
        if len(near) and SHOT_OVERRIDE:
            m = near.iloc[int((near.time_s - r.time_s).abs().argmin())]
            found.loc[i, ['segment', 'view_shot', 'x', 'y']] = [m.segment, m.view_shot, m.x, m.y]
    if found.type.eq('shot').any() and 'x' in found:
        s = found.type.eq('shot')
        goal_x = np.where(found.loc[s, 'attack_sign'] >= 0, 105.0, 0.0)
        found.loc[s, 'distance_to_goal_m'] = np.hypot(goal_x - found.loc[s, 'x'].astype(float),
                                                      34.0 - found.loc[s, 'y'].astype(float))
    ev = ev[~(ev.type.eq('tackle') | ev.type.eq('dispossessed') | ev.type.eq('shot'))]
    losers = []
    sp = spells.sort_values('start_s')
    for r in found[found.type.eq('tackle') & found.confident].itertuples():
        # The carrier who lost the ball: the other team's spell running up to the tackle.
        other = sp[(sp.team != r.team) & (sp.start_s <= r.time_s + .3) & (sp.end_s >= r.time_s - 1.0)]
        if len(other):
            c = other.iloc[-1]
            losers.append({'type': 'dispossessed', 'source': 'action_spotter', 'segment': c.segment, 'team': c.team,
                           'view_shot': int(c.view_shot), 'time_s': float(r.time_s), 'x': r.x, 'y': r.y,
                           'opponent': r.segment, 'attack_sign': ME.attack_sign(c.team, directions)})
    found['opponent'] = None
    merged = pd.concat([ev, found, pd.DataFrame(losers), filled], ignore_index=True)
    merged = merged.sort_values(['time_s', 'type']).reset_index(drop=True)
    confident = found[found.confident].type.value_counts().to_dict()
    expected = found.groupby('type').score.sum().to_dict()
    return merged, {**summary, 'events': {k: int(v) for k, v in confident.items()},
                    'expected': {k: round(float(v), 1) for k, v in expected.items()},
                    'attributed_share': float(found.segment.notna().mean()),
                    'filled_from_video': filled.type.value_counts().to_dict() if len(filled) else {}}


# Passes and carries the tracking rules miss (a player or the ball not detected, a camera cut) but
# the video spotter sees: a confident spotted pass or drive with no rule pass or carry by that
# team within FILL_MATCH_S becomes an event credited to the player at the ball (action_spotting.
# attribute), or to the team only. Their receiver, length and direction are unknown, so pass
# completion and pass geometry still come from tracked passes only (match_stats.on_ball).
FILL_FROM_SPOTTER = not os.environ.get('PITCHPROFILE_NO_SPOTTER_FILL')    # the variable is for measuring without it
FILL_MATCH_S = 1.0
FILL_KINDS = {'pass': ('pass', 'high_pass'), 'carry': ('drive',)}


def fill_from_spotter(found, ev):
    rows = []
    for kind, spotted in FILL_KINDS.items():
        spots = found[found.type.isin(spotted) & found.confident & found.team.notna()]
        rule = ev[ev.type.eq(kind)]
        for r in spots.sort_values('score', ascending=False).itertuples():
            if ((rule.team == r.team) & ((rule.time_s - r.time_s).abs() <= FILL_MATCH_S)).any():
                continue
            if any(x['type'] == kind and x['team'] == r.team and abs(x['time_s'] - r.time_s) <= FILL_MATCH_S for x in rows):
                continue
            rows.append({'type': kind, 'source': 'action_spotter', 'filled': True, 'label': r.label,
                         'segment': r.segment, 'team': r.team, 'view_shot': r.view_shot, 'time_s': float(r.time_s),
                         'x': r.x, 'y': r.y, 'score': float(r.score), 'confident': True,
                         'outcome': 'unknown' if kind == 'pass' else None, 'attack_sign': r.attack_sign})
    return pd.DataFrame(rows)


def write_dataset(identifier, meta, frames, people, ball, events, identity, identities, stats, team_info,
                  directions, numbers, hz, live_seconds):
    from .features import build_profiles
    d = S.dataset_dir(identifier)
    home, away = fixture_teams(str(meta.get('match_id', '')).rsplit('/', 1)[-1])
    names = S.read_json(d / 'team_names.json', {}) or {}
    team_label = {k: names.get(k) or f'Team {k}' for k in ('A', 'B')}
    by_ident = {p['identity']: p for p in stats['players']}
    p = people[people.segment.isin(identity)].copy()
    p['player_id'] = p.segment.map(identity)
    sign = p.team.map(lambda t: ME.attack_sign(t, directions)).to_numpy()
    # Stored coordinates attack to the right for every player (direction_known contract).
    p['x'] = np.where(sign < 0, 105 - p.x, p.x)
    p['y'] = np.where(sign < 0, 68 - p.y, p.y)
    tracks = pd.DataFrame({'frame': p['sample'].astype(int), 'time_s': p.time_s, 'player_id': p.player_id,
                           'track_id': p.segment, 'period': meta.get('half', 1), 'x': p.x, 'y': p.y,
                           'detected': 1, 'calibration_valid': np.isfinite(p.x).astype(int),
                           'bbox_x': p.bbox_x, 'bbox_y': p.bbox_y, 'bbox_w': p.bbox_w, 'bbox_h': p.bbox_h,
                           'confidence': p.conf, 'source_frame': p['sample'].map(frames.set_index('sample').source_frame)})
    S.save_tracks(identifier, tracks)
    players = []
    for ident, g in tracks.groupby('player_id'):
        st = by_ident.get(ident, {})
        team = st.get('team') or people.loc[people.segment.isin([s for s, i in identity.items() if i == ident]), 'team'].mode().iat[0]
        role = st.get('role', 'player')
        row = identities[identities.identity.eq(ident)]
        label = row.label.iat[0] if len(row) else ident
        number = row.number.iat[0] if len(row) and 'number' in row and pd.notna(row.number.iat[0]) else None
        cell = lambda k: row[k].iat[0] if len(row) and k in row and pd.notna(row[k].iat[0]) else None
        name = f"{team_label.get(team, team)} #{int(number)}" if number is not None else f"{team_label.get(team, team)} · {label}"
        direction = 'right' if ME.attack_sign(team, directions) > 0 else 'left' if ME.attack_sign(team, directions) < 0 else 'unknown'
        players.append({'player_id': ident, 'name': name, 'team': team_label.get(team, team), 'team_key': team,
                        'role': 'Goalkeeper' if role == 'goalkeeper' else 'Outfield player',
                        'position_group': suggest_position_group(role, st.get('positional', {})),
                        'position_group_source': 'suggested from observed positions',
                        'jersey': int(number) if number is not None else None,
                        'identity_verified': False, 'global_id': None, 'direction': direction,
                        'direction_known': direction != 'unknown',
                        'eligible_frames': int(frames.pitch_view.sum()), 'playing_seconds': live_seconds,
                        'visible_seconds': st.get('visible_seconds'),
                        'unnamed': role != 'goalkeeper' and number is None,
                        'number_guess': int(cell('number_guess')) if cell('number_guess') is not None else None,
                        'number_guess_share': float(cell('guess_share')) if cell('guess_share') is not None else None,
                        'readable_views': int(cell('readable_views')) if cell('readable_views') is not None else None,
                        'identity_confirmed': bool(cell('confirmed')),
                        'coverage_note': 'Visible time inside calibrated live broadcast views. Identity is automatic '
                                         '(jersey number where legible, otherwise tactical role); review before labelling.'})
    manifest = {
        'id': identifier, 'title': meta.get('title', identifier), 'match_id': meta.get('match_id') or identifier,
        'date': meta.get('date'), 'half': meta.get('half', 1), 'source': meta.get('source', 'Video analysis'),
        'source_kind': 'model_predictions', 'analysis': ANALYSIS_VERSION, 'sampling_hz': hz,
        'duration_seconds': float(frames.time_s.max() - frames.time_s.min() + 1 / hz),
        'total_sampled_frames': int(len(frames)), 'players': players, 'video': meta.get('video_name'),
        'soccernet_library_id': meta.get('library_id'), 'source_offset_s': float(frames.time_s.min()),
        'source_fps': meta['detection']['source_fps'], 'clock': 'analysed_interval_relative',
        'benchmark_split': meta.get('benchmark_split'), 'width': meta['detection']['width'],
        'height': meta['detection']['height'], 'preview': None, 'overlay': None,
        'teams': {k: {'name': team_label[k], 'colour': team_info.get('centres', {}).get(k, {}).get('colour'),
                      'attacks': ('right' if directions.get('attacks_right') == k else 'left')
                      if directions.get('status') in ('inferred', 'uncertain') else 'unknown'} for k in ('A', 'B')},
        'fixture_teams': {'home': home, 'away': away},
        'coverage': {'pitch_view_share': float(frames.pitch_view.mean()), 'live_seconds': live_seconds,
                     'ball_observed_share': float((~ball.interpolated.astype(bool)).sum() / max(frames.pitch_view.sum(), 1)) if len(ball) else 0.0},
        'note': 'Automatic full-match analysis from video: football detector, NBJW pitch calibration, BoT-SORT, '
                'kit clustering, ball path, touch-based events. Review identities and teams before labelling.',
        'created': S.now(), 'processing': meta.get('detection'),
    }
    # Tracks are stored relative to the analysed interval start.
    manifest['duration_seconds'] = float(frames.time_s.max() - frames.time_s.min() + 1 / hz)
    S.write_json(d / 'manifest.json', manifest)
    b = ball.copy()
    b.to_csv(d / 'ball.csv.gz', index=False, compression='gzip')
    ev = events.assign(identity=events.segment.map(identity)) if len(events) else events
    for col in ('receiver', 'opponent'):
        if col in ev:
            ev[col + '_identity'] = ev[col].map(identity)
    S.write_json(d / 'events.json', {'events': ev.replace({np.nan: None}).to_dict('records') if len(ev) else [],
                                     'note': 'Touch-based events from video. Times are half-relative seconds.'})
    S.write_json(d / 'match_stats.json', stats)
    S.write_json(d / 'teams.json', {'teams': team_info, 'directions': directions, 'names': team_label})
    seg_tracks = people.groupby('segment').track.agg(lambda s: sorted(set(s))).to_dict()
    S.write_json(d / 'identities.json', {
        'segments': {s: {'identity': i, 'jersey': numbers.get(s)} for s, i in identity.items()},
        'segment_tracks': {s: seg_tracks.get(s, []) for s in identity},
        'identities': identities.replace({np.nan: None}).to_dict('records')})
    build_profiles(identifier)


def rename_teams(identifier, names):
    """Human team names for kit groups A and B, applied to the manifest and player names."""
    from .features import build_profiles
    d = S.dataset_dir(identifier)
    S.write_json(d / 'team_names.json', names)
    m = S.read_json(d / 'manifest.json')
    old = {k: v['name'] for k, v in m.get('teams', {}).items()}
    for k, v in m.get('teams', {}).items():
        v['name'] = names[k]
    for p in m['players']:
        key = p.get('team_key')
        if key in names:
            p['team'] = names[key]
            if key in old and p['name'].startswith(old[key]):
                p['name'] = names[key] + p['name'][len(old[key]):]
    S.write_json(d / 'manifest.json', m)
    build_profiles(identifier)
    return {'teams': m['teams']}


def soccernet_identifier(game, half):
    """'england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley', 1 -> 'sn-20150221-chelsea-burnley-h1'."""
    name = game.rsplit('/', 1)[-1]
    date = name[:10].replace('-', '')
    home, away = fixture_teams(name)
    slug = '-'.join(re.sub(r'[^a-z0-9]+', '-', (t or 'team').lower()).strip('-') for t in (home, away))
    return f'sn-{date}-{slug}-h{int(half)}'[:90]


def soccernet_meta(item):
    """Analysis metadata for a SoccerNet library entry (football_profiler.soccernet.library)."""
    name = item['game'].rsplit('/', 1)[-1]
    home, away = fixture_teams(name)
    return {'title': f"{home} v {away} · {name[:10]} · H{item['half']}" if home else f"{name} · H{item['half']}",
            'match_id': 'soccernet:' + item['game'], 'date': name[:10], 'half': item['half'],
            'library_id': item['id'], 'benchmark_split': item.get('benchmark_split'),
            'source': 'SoccerNet 720p broadcast video; PitchProfile automatic analysis (no provider metadata)'}


def analyse_soccernet(library_id, progress=lambda *a: None, **detection_options):
    from . import soccernet as SN
    item = SN.entry(library_id)
    meta = soccernet_meta(item)
    identifier = soccernet_identifier(item['game'], item['half'])
    postprocess_meta = analyse(identifier, SN.source_path(library_id), title=meta['title'], meta=meta,
                               progress=progress, **detection_options)
    return {'dataset_id': identifier, **postprocess_meta.get('postprocess', {})}


def analyse(identifier, video_path, *, title, meta=None, progress=lambda *a: None, **detection_options):
    """Detection pass followed by post-processing, with progress mapped onto one job."""
    from .match_analysis import detection_pass
    S.dataset_dir(identifier, create=True)
    detection = detection_pass(identifier, video_path, progress=lambda f, m: progress(f * .85, m), **detection_options)
    info = {**(meta or {}), 'title': title, 'detection': detection, 'video_name': str(video_path).split('\\')[-1].split('/')[-1],
            'created': S.now()}
    S.write_json(S.dataset_dir(identifier) / 'analysis.json', info)
    return postprocess(identifier, progress=lambda f, m: progress(.85 + f * .15, m))
