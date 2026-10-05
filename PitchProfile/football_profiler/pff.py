"""Direct importer for PFF FC's released 2022 World Cup broadcast tracking data.

Reads straight out of the provider's delivered zip archives (no bulk extraction
to disk): each game's Metadata/Rosters/Tracking Data entry can live in either
archive, so callers pass every archive path and this module finds each entry.

Each game's clock resets at half-time (periodElapsedTime starts near zero in
both periods), exactly like SoccerTrack's released format, so each half is
imported as its own dataset — see football_profiler/soccertrack.py for the
same one-half-per-identifier precedent and why it matters for interval reviews.
"""
from __future__ import annotations

import bz2
import json
import math
import zipfile

import pandas as pd

from . import features as F, storage as S, taxonomy as T

SOURCE = 'PFF FC 2022 World Cup release'
SOURCE_URL = 'https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data'
# PFF's fine-grained roster position codes, mapped onto the proposal's 8 archetype groups.
POSITION_GROUP_MAP = {
    'GK': 'goalkeeper',
    'LCB': 'centre_back', 'RCB': 'centre_back', 'MCB': 'centre_back',
    'LB': 'full_back', 'RB': 'full_back', 'LWB': 'full_back', 'RWB': 'full_back',
    'DM': 'defensive_midfield', 'CM': 'central_midfield', 'AM': 'attacking_midfield',
    'LW': 'wide_attacker', 'RW': 'wide_attacker', 'CF': 'centre_forward',
}


def _find(zip_paths, *candidates):
    """Return (zip_path, entry_name) for the first candidate found in any archive."""
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as z:
            names = set(z.namelist())
            for candidate in candidates:
                if candidate in names:
                    return zip_path, candidate
    raise ValueError(f'None of {candidates} found in the supplied archives')


def _read_json(zip_paths, *candidates):
    zip_path, name = _find(zip_paths, *candidates)
    with zipfile.ZipFile(zip_path) as z, z.open(name) as f:
        return json.loads(f.read())


def list_games(zip_paths):
    ids = set()
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.startswith('Metadata/') and name.endswith('.json'):
                    ids.add(name[len('Metadata/'):-len('.json')])
    return sorted(ids, key=int)


def parse_game(zip_paths, game_id, sampling_hz=10):
    """Stream both periods in a single pass over the tracking file (it's the
    expensive step); returns (meta, {1: (tracks, players, info), 2: (...)})."""
    game_id = str(int(game_id))
    meta = _read_json(zip_paths, f'Metadata/{game_id}.json')
    if not isinstance(meta, list) or len(meta) != 1:
        raise ValueError('Expected exactly one metadata record per game')
    meta = meta[0]
    if str(meta.get('id')) != game_id:
        raise ValueError('Metadata game id does not match the requested game')
    roster = _read_json(zip_paths, f'Rosters/{game_id}.json')
    fps = float(meta['fps'])
    if not math.isfinite(fps) or not 0 < sampling_hz <= fps <= 120:
        raise ValueError('Choose positive sampling_hz no greater than the finite source frame rate')
    stride = max(1, round(fps / sampling_hz))
    actual_hz = fps / stride
    home_id, away_id = str(meta['homeTeam']['id']), str(meta['awayTeam']['id'])
    home_start_left = meta.get('homeTeamStartLeft')
    if not isinstance(home_start_left, bool):
        raise ValueError('This game has no confirmed kickoff-end orientation; review before importing')

    people, roster_by_side = {}, {}
    for entry in roster:
        side = 'home' if str(entry['team']['id']) == home_id else 'away' if str(entry['team']['id']) == away_id else None
        if side is None:
            raise ValueError('Roster entry references a team not in this fixture')
        jersey = str(entry['shirtNumber'])
        roster_by_side.setdefault(side, {})[jersey] = entry
        pid = f'pff-{entry["player"]["id"]}'
        group = POSITION_GROUP_MAP.get(entry['positionGroupType'])
        people[pid] = {'player_id': pid, 'name': entry['player']['nickname'],
                       'team': meta['homeTeam']['name'] if side == 'home' else meta['awayTeam']['name'],
                       'jersey': jersey, 'role': T.GROUPS.get(group, entry['positionGroupType']),
                       'position_group': group, 'global_id': f'pff:{entry["player"]["id"]}',
                       'identity_verified': True, 'direction_known': True,
                       'coverage_note': 'Broadcast-derived tracking; confidence and visibility per '
                       'sample are not an independent correctness guarantee.'}

    def right_facing(side, period):
        # homeTeamStartLeft=True means home defends the left goal (attacks right) in period 1;
        # ends swap at half-time. Verified empirically against a real game: with
        # homeTeamStartLeft=True, the home goalkeeper's raw x sits near -52.5 (left) in period 1.
        home_right = home_start_left != (period == 2)
        return home_right if side == 'home' else not home_right

    zip_path, tracking_name = _find(zip_paths, f'Tracking Data/{game_id}.jsonl.bz2')
    rows = {1: [], 2: []}
    frame_counts = {1: {pid: 0 for pid in people}, 2: {pid: 0 for pid in people}}
    with zipfile.ZipFile(zip_path) as z, z.open(tracking_name) as raw, \
            bz2.open(raw, 'rt', encoding='utf-8') as stream:
        for line in stream:
            frame = json.loads(line)
            period = frame.get('period')
            if period not in (1, 2) or frame['frameNum'] % stride:
                continue
            time_s = frame.get('periodElapsedTime')
            if time_s is None or not math.isfinite(time_s) or time_s < 0:
                continue
            for side, key in (('home', 'homePlayers'), ('away', 'awayPlayers')):
                sign = 1 if right_facing(side, period) else -1
                by_jersey = roster_by_side.get(side, {})
                for sample in frame.get(key) or []:
                    entry = by_jersey.get(str(sample.get('jerseyNum', '')))
                    if entry is None:
                        continue
                    x, y = sample.get('x'), sample.get('y')
                    if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
                        continue
                    pid = f'pff-{entry["player"]["id"]}'
                    detected = 1 if sample.get('visibility') == 'VISIBLE' else 0
                    rows[period].append((frame['frameNum'] // stride, time_s, pid, pid, period,
                                         round(x * sign + 52.5, 5), round(y * sign + 34, 5), detected, 1))
                    frame_counts[period][pid] += 1
    by_period = {}
    for period in (1, 2):
        if not rows[period]:
            raise ValueError(f'No sampled tracking rows found for period {period}')
        tracks = pd.DataFrame(rows[period], columns=['frame', 'time_s', 'player_id', 'track_id',
                                                      'period', 'x', 'y', 'detected', 'calibration_valid'])
        players = []
        for pid, person in people.items():
            frames = frame_counts[period][pid]
            if not frames:
                continue
            side = 'home' if person['team'] == meta['homeTeam']['name'] else 'away'
            players.append({**person, 'eligible_frames': frames, 'playing_seconds': frames / actual_hz,
                            'stadium_direction': 'right' if right_facing(side, period) else 'left'})
        info = {'sampling_hz': actual_hz, 'source_fps': fps,
                'duration_seconds': float(tracks.time_s.max()) + 1 / actual_hz,
                # frameNum is a whole-match counter that does not reset at half-time, so a
                # distinct-value count is used rather than max()+1 (which would double-count).
                'total_sampled_frames': int(tracks.frame.nunique()), 'half': period,
                'home_team': meta['homeTeam']['name'], 'away_team': meta['awayTeam']['name']}
        by_period[period] = (tracks, players, info)
    return meta, by_period


def import_match(zip_paths, game_id, sampling_hz=10, periods=(1, 2)):
    game_id = str(int(game_id))
    for period in periods:
        identifier = f'pff-wc2022-{game_id}-h{period}'
        if (S.DATA / 'datasets' / identifier).exists():
            raise ValueError(f'{identifier} is already imported; remove the existing dataset directory first')
    meta, by_period = parse_game(zip_paths, game_id, sampling_hz)
    manifests = {}
    for period in periods:
        identifier = f'pff-wc2022-{game_id}-h{period}'
        tracks, players, info = by_period[period]
        manifest = {'id': identifier, 'title': f"{meta['homeTeam']['name']} vs {meta['awayTeam']['name']} — half {period}",
                    'match_id': f'pff:{game_id}', 'date': meta.get('date'), 'source': SOURCE,
                    'source_kind': 'provider_tracking', 'source_url': SOURCE_URL, 'clock': 'period_relative',
                    'coordinate_orientation': 'attack_right',
                    'players': players, **info, 'created': S.now(), 'video': None,
                    'note': 'Real PFF FC broadcast-derived tracking. Corresponding match footage is not '
                            'included. Position groups are mapped from the provider roster, not reviewed.',
                    'coverage_note': 'Sampled broadcast tracking; a player only appears while the provider '
                                      'tracker reports their position, which understates true playing time '
                                      'for unused substitutes and may include estimated (interpolated) samples.'}
        directory = S.dataset_dir(identifier, True)
        try:
            S.save_tracks(identifier, tracks)
            S.write_json(directory / 'manifest.json', manifest)
            F.build_profiles(identifier)
        except Exception:
            import shutil
            target = directory.resolve()
            if target.parent != (S.DATA / 'datasets').resolve() or target.name != identifier:
                raise RuntimeError('Import cleanup target is outside the dataset directory')
            shutil.rmtree(target)
            raise
        manifests[period] = manifest
    return manifests
