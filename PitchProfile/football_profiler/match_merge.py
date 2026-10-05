"""Whole matches: both analysed halves of a match as one dataset.

One video (the two half files joined without re-encoding), one player list and statistics over
the whole match, so each player is rated once. The halves stay the unit of analysis: a whole
match is rebuilt from them (after re-analysis or reviewer names) and names given on it are
written back to the half they belong to.

Joining the halves:
- the second half's samples, times and view shots follow the first half's, its times shifted by
  the first video file's duration;
- kit groups A and B are clustered per half, so the second half's groups are matched to the
  first half's by kit colour;
- teams change ends at half-time: the second half's pitch coordinates are mirrored so each team
  attacks the same way throughout, and the first half's attack directions apply;
- players: named players and goalkeepers are one player when team and shirt number agree, as are
  reviewer-confirmed people ('<team>-P<k>'); an unnamed second-half player is '<team>-Y<k>'
  until a reviewer names him or confirms him as someone from the first half.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import match_events as ME
from . import match_identity as MI
from . import match_pipeline as MP
from . import match_stats as MS
from . import storage as S

L, W = MI.L, MI.W
SUFFIX = re.compile(r'-h1$')
# An unnamed player takes the name of a player named in the other half when their appearance is
# clearly the same: on FOOTPASS validation games this named 7.6% more of visible player time
# right and 1.2% wrong (similarity .85: +8.4% / +2.5%; .8: +8.5% / +6.3%).
CROSS_HALF_SIMILARITY = .9
CROSS_HALF_MARGIN = .05
UNNAMED = re.compile(r'^[AB]-[XY]\d+$')


def halves_by_match():
    """{whole-match id: (first half id, second half id)} for matches with both halves analysed."""
    found = {}
    for m in S.datasets():
        if not str(m.get('analysis', '')).startswith('full-match') or m.get('halves') or m.get('half') not in (1, 2):
            continue
        found.setdefault(m.get('match_id') or m['id'], {})[m['half']] = m['id']
    return {SUFFIX.sub('', h[1]): (h[1], h[2]) for h in found.values() if 1 in h and 2 in h and SUFFIX.search(h[1])}


def whole_match_of(half_id):
    """The whole-match id a half belongs to, or None."""
    return next((w for w, pair in halves_by_match().items() if half_id in pair), None)


def kit_mapping(first_info, second_info):
    """Second-half kit group -> first-half kit group, by kit colour (Lab)."""
    a, b = first_info.get('centres') or {}, second_info.get('centres') or {}
    if not all(k in a and k in b and a[k].get('lab') for k in 'AB'):
        return {'A': 'A', 'B': 'B'}
    d = lambda x, y: float(np.linalg.norm(np.subtract(a[x]['lab'], b[y]['lab'])))
    return {'A': 'A', 'B': 'B'} if d('A', 'A') + d('B', 'B') <= d('A', 'B') + d('B', 'A') else {'A': 'B', 'B': 'A'}


def video_duration(path):
    """Seconds of a video file, from ffmpeg's container header."""
    import imageio_ffmpeg
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-i', str(path)],
                         capture_output=True, text=True).stderr
    h, m, s = re.search(r'Duration: (\d+):(\d+):([\d.]+)', out).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def join_videos(first, second, target):
    """Both half files as one, streams copied (minutes, no re-encoding)."""
    import imageio_ffmpeg
    if target.is_file() and target.stat().st_mtime >= max(first.stat().st_mtime, second.stat().st_mtime):
        return
    listing = target.with_suffix('.txt')
    listing.write_text(f"file '{first.as_posix()}'\nfile '{second.as_posix()}'\n", encoding='utf-8')
    tmp = target.with_name(target.stem + '.part' + target.suffix)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
                    '-safe', '0', '-i', str(listing), '-c', 'copy', str(tmp)], check=True)
    tmp.replace(target)
    listing.unlink(missing_ok=True)


def match_player_id(ident, half, kits):
    """A half's player id as the whole match's: second-half team keys follow the first half's;
    an unnamed second-half player is '<team>-Y<k>'."""
    m = re.match(r'^([AB])-(.+)$', str(ident))
    if not m:
        return f'h{half}:{ident}'
    team, rest = m.groups()
    if half == 2:
        team = kits[team]
        if rest.startswith('X'):
            rest = 'Y' + rest[1:]
    return f'{team}-{rest}'                     # numbers, GK and reviewer people 'P<k>' are shared


def half_player_id(match_id, half, kits):
    """The inverse of match_player_id for named players: the id this player has in one half."""
    team, rest = match_id.split('-', 1)
    if half == 2:
        team = {v: k for k, v in kits.items()}[team]
        if rest.startswith('Y'):
            rest = 'X' + rest[1:]
    return f'{team}-{rest}'


def _prefix(series, half):
    return series.map(lambda v: f'h{half}:{v}' if isinstance(v, str) and v else v)


def _mirror(df, pairs):
    for x, y in pairs:
        if x in df:
            df[x] = L - df[x]
        if y in df:
            df[y] = W - df[y]


def carry_names(identity, evidence, spans):
    """{unnamed player: named player} for unnamed players who look like a player named in the
    other half (similarity at least CROSS_HALF_SIMILARITY, clear of the runner-up by
    CROSS_HALF_MARGIN, and never on screen with that player's own segments of this half).

    identity: prefixed segment ('h1:...') -> whole-match id; evidence and spans (start, end) per
    prefixed segment.
    """
    from . import identity_model as IM
    players = {}
    for s, i in identity.items():
        players.setdefault(i, []).append(s)
    numbered = lambda i: '-' in i and i.split('-', 1)[1].isdigit()
    combined = lambda segs: IM.combine([evidence[s] for s in segs if s in evidence])
    moves = {}
    for u, segs in players.items():
        if not UNNAMED.match(u):
            continue
        half = segs[0][:2]
        e = combined(segs)
        if e is None:
            continue
        v, mine = IM.embedding(e), [spans[s] for s in segs if s in spans]
        cands = []
        for n, nsegs in players.items():
            if not numbered(n) or n[0] != u[0]:
                continue
            other = [s for s in nsegs if s[:2] != half]
            same = [spans[s] for s in nsegs if s[:2] == half and s in spans]
            ne = combined(other)
            if ne is None or not MI._free(mine, same, MI.OVERLAP_TOLERANCE_S):
                continue
            cands.append((float(v @ IM.embedding(ne)), n))
        cands.sort(reverse=True)
        if cands and cands[0][0] >= CROSS_HALF_SIMILARITY and \
                (len(cands) == 1 or cands[0][0] - cands[1][0] >= CROSS_HALF_MARGIN):
            moves[u] = cands[0][1]
    return moves


def combine(first, second, offset_s):
    """Two half_state() dicts -> the whole match's tables (see the module docstring)."""
    kits = kit_mapping(first['team_info'], second['team_info'])
    d1, d2 = first['directions'], second['directions']
    known = d1.get('status') in ('inferred', 'uncertain') and d2.get('status') in ('inferred', 'uncertain')
    # Teams change ends at half-time; mirror unless both halves say the (mapped) team kept its end.
    mirror = not (known and kits.get(d2['attacks_right']) == d1['attacks_right'])
    directions = d1 if d1.get('status') in ('inferred', 'uncertain') else \
        ({**d2, 'defends_left': kits[d2['defends_left']], 'attacks_right': kits[d2['attacks_right']]} if known or
         d2.get('status') in ('inferred', 'uncertain') else d1)
    s1 = int(first['frames']['sample'].max()) + 1
    v1 = int(np.nanmax(first['frames']['view_shot'])) + 1
    f1 = int(round(offset_s * float(first['meta']['detection']['source_fps'])))
    team = lambda s: s.map(lambda t: kits.get(t, t))

    out = {}
    for name in ('frames', 'people', 'ball', 'events', 'spells'):
        a, b = first[name].copy(), second[name].copy()
        for df, half in ((a, 1), (b, 2)):
            for col in ('segment', 'track', 'tracklet', 'ball_segment', 'receiver', 'opponent'):
                if col in df:
                    df[col] = _prefix(df[col].astype(object).where(df[col].notna()), half)
        for col in ('sample', 'start_sample', 'end_sample'):
            if col in b:
                b[col] = b[col] + s1
        for col in ('time_s', 'start_s', 'end_s'):
            if col in b:
                b[col] = b[col] + offset_s
        if 'view_shot' in b:
            b['view_shot'] = b['view_shot'] + v1
        if 'source_frame' in b:
            b['source_frame'] = b['source_frame'] + f1
        if 'team' in b:
            b['team'] = team(b['team'])
        if mirror:
            _mirror(b, [('x', 'y'), ('end_x', 'end_y'), ('x0', 'y0'), ('x1', 'y1')])
        out[name] = pd.concat([a, b], ignore_index=True)
    if 'attack_sign' in out['events']:
        out['events']['attack_sign'] = out['events'].team.map(lambda t: ME.attack_sign(t, directions))
    identity = {f'h1:{s}': match_player_id(i, 1, kits) for s, i in first['identity'].items()}
    identity.update({f'h2:{s}': match_player_id(i, 2, kits) for s, i in second['identity'].items()})
    evidence, spans = {}, {}
    for h, half in ((first, 1), (second, 2)):
        evidence.update({f'h{half}:{s}': e for s, e in (h.get('evidence') or {}).items()})
        sig = h.get('signatures')
        if sig is not None and len(sig):
            spans.update({f'h{half}:{s}': (float(a), float(b)) for s, a, b in zip(sig.index, sig.start_s, sig.end_s)})
    carried = carry_names(identity, evidence, spans) if evidence else {}
    identity = {s: carried.get(i, i) for s, i in identity.items()}
    rows = []
    for rows_half, half in ((first['identities'], 1), (second['identities'], 2)):
        r = rows_half.copy()
        r['identity'] = r.identity.map(lambda i: match_player_id(i, half, kits)).map(lambda i: carried.get(i, i))
        r['team'] = r.team.map(lambda t: kits.get(t, t) if half == 2 else t)
        if half == 2:
            r['label'] = r.label.astype(str).str.replace(r' · unnamed (\d+)', r' · unnamed \1 (2nd half)', regex=True)
        rows.append(r)
    rows = pd.concat(rows, ignore_index=True)
    # A player's row keeps the details of his named part (an unnamed part may have joined him).
    rows['named'] = rows['named'].fillna(False).astype(bool)
    rows = rows.sort_values(['named', 'observed_s'], ascending=False).groupby('identity', as_index=False).agg(
        {**{c: 'first' for c in rows.columns if c not in ('identity', 'segments', 'observed_s', 'confirmed', 'named')},
         'segments': 'sum', 'observed_s': 'sum', 'named': 'max', **({'confirmed': 'max'} if 'confirmed' in rows else {})})
    numbers = {f'h1:{s}': v for s, v in first['numbers'].items()}
    numbers.update({f'h2:{s}': v for s, v in second['numbers'].items()})
    return {**out, 'identity': identity, 'identities': rows, 'numbers': numbers, 'directions': directions,
            'kits': kits, 'mirrored': mirror, 'named_from_other_half': carried, 'live_seconds': first['live_seconds'] + second['live_seconds'],
            'hz': first['hz'], 'team_info': first['team_info']}


def build(identifier, progress=lambda *a: None):
    """(Re)build one whole match from its two analysed halves."""
    import time
    started = time.perf_counter()
    first_id, second_id = halves_by_match()[identifier]
    d = S.dataset_dir(identifier, create=True)
    m1 = S.read_json(S.dataset_dir(first_id) / 'manifest.json')
    m2 = S.read_json(S.dataset_dir(second_id) / 'manifest.json')
    v1, v2 = S.video_path(S.dataset_dir(first_id), m1), S.video_path(S.dataset_dir(second_id), m2)
    progress(.05, 'Joining the two half videos')
    video = d / f'{identifier}{v1.suffix}'
    join_videos(v1, v2, video)
    offset = video_duration(v1)
    old_segments = {}
    for s, v in (S.read_json(d / 'identities.json', {}) or {}).get('segments', {}).items():
        old_segments.setdefault(v.get('identity'), []).append(s)
    progress(.15, 'Reading the first half')
    first = MP.half_state(first_id)
    progress(.45, 'Reading the second half')
    second = MP.half_state(second_id)
    progress(.75, 'Combining players and statistics over the whole match')
    c = combine(first, second, offset)
    names = S.read_json(S.dataset_dir(first_id) / 'team_names.json', {}) or {}
    if names:
        S.write_json(d / 'team_names.json', names)
    title = re.sub(r'\s*·\s*H1$', '', m1.get('title', identifier)) + ' · whole match'
    meta = {**first['meta'], 'title': title, 'half': 1, 'library_id': None, 'video_name': video.name}
    stats = MS.build(c['people'], c['events'], c['spells'], c['identity'], c['directions'], c['hz'], c['live_seconds'])
    MP.write_dataset(identifier, meta, c['frames'], c['people'], c['ball'], c['events'], c['identity'], c['identities'],
                     stats, c['team_info'], c['directions'], c['numbers'], c['hz'], c['live_seconds'])
    tracks = S.load_tracks(identifier)
    tracks['period'] = np.where(tracks.time_s >= offset, 2, 1)
    S.save_tracks(identifier, tracks)
    manifest = S.read_json(d / 'manifest.json')
    manifest.update({'whole_match': True, 'halves': [first_id, second_id], 'second_half_offset_s': offset,
                     'kit_mapping_second_half': c['kits'], 'second_half_mirrored': c['mirrored'], 'half': None})
    S.write_json(d / 'manifest.json', manifest)
    for half_id in (first_id, second_id):                       # halves point at their whole match
        hm = S.read_json(S.dataset_dir(half_id) / 'manifest.json')
        if hm.get('whole_match_id') != identifier:
            hm['whole_match_id'] = identifier
            S.write_json(S.dataset_dir(half_id) / 'manifest.json', hm)
    # Ratings and corrections: follow re-numbered players, then bring in newer ones from the halves.
    moves = MP.player_moves(old_segments, c['identity'], c['people'].groupby('segment').size().to_dict())
    moved = S.move_player_records(identifier, moves)
    for half_id, half in ((first_id, 1), (second_id, 2)):
        ids = {p['player_id'] for p in S.read_json(S.dataset_dir(half_id) / 'manifest.json')['players']}
        S.copy_player_records(half_id, identifier, {i: match_player_id(i, half, c['kits']) for i in ids})
    _copy_binding(first_id, identifier)
    analysis = {'title': title, 'match_id': meta.get('match_id'), 'date': meta.get('date'), 'whole_match': True,
                'halves': [first_id, second_id], 'detection': meta.get('detection'),
                'postprocess': {'version': MP.ANALYSIS_VERSION, 'created': S.now(), 'player_moves': moves,
                                'records_moved': moved, 'wall_seconds': time.perf_counter() - started,
                                'identities': int(len(set(c['identity'].values()))),
                                'named_players': int(c['identities']['named'].sum()) if 'named' in c['identities'] else None,
                                'kit_mapping_second_half': c['kits'], 'second_half_offset_s': offset,
                                'named_from_other_half': c['named_from_other_half']}}
    S.write_json(d / 'analysis.json', analysis)
    progress(1, 'Whole match ready')
    return analysis


def _copy_binding(source, target):
    with S.db() as c:
        row = c.execute('SELECT payload FROM match_context_bindings WHERE dataset_id=?', (source,)).fetchone()
        if row and not c.execute('SELECT 1 FROM match_context_bindings WHERE dataset_id=?', (target,)).fetchone():
            c.execute('INSERT INTO match_context_bindings VALUES(?,?)', (target, row['payload']))


def build_all(progress=print):
    for i, identifier in enumerate(sorted(halves_by_match()), 1):
        progress(f'[{i}] {identifier}')
        build(identifier)


def sync_ratings(identifier):
    """Copy ratings saved on a half after its whole match was built (newer ones win)."""
    d = S.dataset_dir(identifier)
    m = S.read_json(d / 'manifest.json')
    kits = m.get('kit_mapping_second_half') or {'A': 'A', 'B': 'B'}
    copied = 0
    for half, half_id in enumerate(m.get('halves') or [], 1):
        ids = {p['player_id'] for p in S.read_json(S.dataset_dir(half_id) / 'manifest.json')['players']}
        copied += S.copy_player_records(half_id, identifier, {i: match_player_id(i, half, kits) for i in ids})
    return copied


def half_segments(identifier, match_identity):
    """{half id: (segments of this whole-match player in that half, the player's id there)}."""
    d = S.dataset_dir(identifier)
    m = S.read_json(d / 'manifest.json')
    kits = m.get('kit_mapping_second_half') or {'A': 'A', 'B': 'B'}
    out = {}
    for s, v in S.read_json(d / 'identities.json', {}).get('segments', {}).items():
        if v.get('identity') != match_identity:
            continue
        half = int(s[1])
        seg = s.split(':', 1)[1]
        out.setdefault(m['halves'][half - 1], ([], half))[0].append(seg)
    return {h: (segs, half) for h, (segs, half) in out.items()}, kits


def half_name(name, half, kits):
    """A reviewer name for a whole-match player as that half writes it ('NONE' stays)."""
    return name if name == MI.NOT_A_PLAYER else half_player_id(name, half, kits)
