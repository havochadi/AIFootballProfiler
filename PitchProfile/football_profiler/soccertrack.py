"""Streaming importer for SoccerTrack v2's released SoccerNet-COCO GSR objects.

The provider's 2026-08-11 divergence notice takes precedence over its old flat
example. No protected files are bundled. Import one half at a time; preserve the
same match_id across halves for grouped evaluation.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

import ijson
import pandas as pd

from . import features as F, storage as S

SOURCE = 'https://huggingface.co/datasets/atomscott/soccertrack-v2'
FORMATS = 'https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md'
ACTIONS = ('Pass','Drive','Header','High Pass','Out','Cross','Throw In','Shot',
           'Ball Player Block','Player Successful Tackle','Free Kick','Goal')
BENCHMARK_SPLITS = {**dict.fromkeys(('117092','118575','118576','118577','118578','128058'), 'train'),
                    **dict.fromkeys(('117093','132877'), 'validation'),
                    **dict.fromkeys(('128057','132831'), 'test')}


def items(path, prefix):
    with Path(path).open('rb') as stream:
        yield from ijson.items(stream, prefix, use_float=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def video_start_time(path):
    """Read the first video PTS; stream-copy trims can retain a nonzero start."""
    import subprocess
    from .vision import ffmpeg_executable
    result = subprocess.run([ffmpeg_executable(), '-hide_banner', '-nostdin', '-copyts',
                             '-i', str(path), '-map', '0:v:0', '-vf', 'showinfo',
                             '-frames:v', '1', '-f', 'null', '-'],
                            capture_output=True, text=True, timeout=30)
    match = re.search(r'\bn:\s*0\s+pts:\s*\S+\s+pts_time:([\d.eE+-]+)', result.stderr)
    if result.returncode or not match:
        raise ValueError('Unable to verify the first video presentation timestamp')
    start = float(match.group(1))
    if not math.isfinite(start) or start < 0:
        raise ValueError('Unsupported video presentation timeline')
    return start


def parse_half(path, match_id, half, sampling_hz=5, left_attacks='unknown'):
    if not re.fullmatch(r'\d{6}', str(match_id)) or half not in (1, 2):
        raise ValueError('Use a six-digit match ID and half 1 or 2')
    if left_attacks not in ('unknown', 'left', 'right'):
        raise ValueError('left_attacks must be unknown, left or right after visual confirmation')
    info = next(items(path, 'info'), None)
    if not isinstance(info, dict):
        raise ValueError('Expected released GSR object with info, images and annotations; flat example files are unsupported')
    declared = re.search(r'CLPD-(\d{6})', str(info.get('name', '')))
    if declared and declared.group(1) != str(match_id):
        raise ValueError('The GSR fixture identity does not match match_id')
    fps = float(info.get('frame_rate', 0))
    if not math.isfinite(fps) or not 0 < sampling_hz <= fps <= 120:
        raise ValueError('Choose positive sampling_hz no greater than the finite source frame rate')
    stride = max(1, round(fps / sampling_hz))
    actual_hz = fps / stride
    frames, prefixes, seen_frames = {}, set(), set()
    for image in items(path, 'images.item'):
        image_id = str(image.get('image_id', ''))
        if not image_id.isdigit() or len(image_id) < 7 or int(image_id[-6:]) < 1:
            raise ValueError('Expected sequence-prefixed, one-based six-digit GSR frame IDs')
        frame = int(image_id[-6:]) - 1
        if image_id in frames or frame in seen_frames:
            raise ValueError('Duplicate GSR image/frame ID')
        prefixes.add(image_id[:-6]); seen_frames.add(frame)
        frames[image_id] = (frame, image.get('is_labeled', True))
    if not frames or len(prefixes) != 1:
        raise ValueError('A GSR file must describe a single nonempty half sequence')
    nframes = max(seen_frames) + 1
    sampled = math.ceil(nframes / stride)
    people, rows, seen = {}, [], set()
    skipped = {'non_player': 0, 'missing_pitch_position': 0, 'unlabelled_image': 0}
    for annotation in items(path, 'annotations.item'):
        attrs = annotation.get('attributes') or {}
        role = attrs.get('role')
        if role not in ('player', 'goalkeeper'):
            skipped['non_player'] += 1
            continue
        image_id = str(annotation.get('image_id', ''))
        if image_id not in frames:
            raise ValueError('Player annotation references an unknown GSR image')
        frame, labelled = frames[image_id]
        if frame % stride:
            continue
        if not labelled:
            skipped['unlabelled_image'] += 1
            continue
        raw_track = str(annotation.get('track_id', ''))
        if not re.fullmatch(r'\d+', raw_track):
            raise ValueError('Expected a numeric GSR track_id')
        pid = f'h{half}-t{raw_track}'
        key = (frame, pid)
        if key in seen:
            raise ValueError('Duplicate player/frame annotation')
        seen.add(key)
        side = attrs.get('team')
        if side not in ('left', 'right'):
            raise ValueError('Player team must be left or right within this half')
        provider_id = str(attrs['player_id']) if attrs.get('player_id') is not None else None
        jersey = str(attrs['jersey']) if attrs.get('jersey') is not None else None
        direction = left_attacks if side == 'left' else {'left': 'right', 'right': 'left', 'unknown': 'unknown'}[left_attacks]
        if pid not in people:
            people[pid] = {'player_id': pid, 'name': f'{side.title()} team #{jersey or "?"} (track {raw_track})',
                           'team': f'SoccerTrack {match_id} half {half} {side}',
                           'role': 'Goalkeeper' if role == 'goalkeeper' else 'Unconfirmed',
                           'jersey': jersey, 'provider_player_id': provider_id, 'provider_team_side': side,
                           'identity_verified': False, 'global_id': None, 'direction': direction,
                           'direction_known': direction != 'unknown', 'eligible_frames': sampled,
                           'playing_seconds': sampled / actual_hz,
                           'coverage_note': 'Sampled annotation coverage over the entire half; playing-time intervals and persistent identities need review.'}
        elif (people[pid]['provider_player_id'], people[pid]['provider_team_side']) != (provider_id, side):
            raise ValueError('Track identity/team changes inside this half; review the source before importing')
        pitch = annotation.get('bbox_pitch') or {}
        x, y = pitch.get('x_bottom_middle'), pitch.get('y_bottom_middle')
        if x is None or y is None:
            skipped['missing_pitch_position'] += 1
            continue
        x, y = float(x), float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('GSR pitch coordinates must be finite')
        x, y = x + 52.5, y + 34
        if direction == 'left':
            x, y = 105 - x, 68 - y
        rows.append((frame // stride, frame / fps, pid, pid, half, x, y, 1, 1, frame))
    if not rows:
        raise ValueError('No sampled player pitch annotations found')
    tracks = pd.DataFrame(rows, columns=['frame','time_s','player_id','track_id','period','x','y','detected','calibration_valid','source_frame'])
    tracked_ids = set(tracks.player_id)
    people = [p for p in people.values() if p['player_id'] in tracked_ids]
    return tracks, people, {'sampling_hz': actual_hz, 'source_fps': fps, 'source_frames': nframes,
                           'total_sampled_frames': sampled, 'duration_seconds': nframes / fps,
                           'skipped_annotations': skipped, 'half': half}


def parse_events(path, match_id, half, players, duration, offset_ms):
    if not math.isfinite(offset_ms) or offset_ms < 0:
        raise ValueError('Supply a verified nonnegative event-clock offset in milliseconds')
    declared_match = next(items(path, 'match_id'), None)
    if declared_match is None:
        declared_match = next(items(path, 'UrlLocal'), None)
    if declared_match is not None and str(declared_match) != str(match_id):
        raise ValueError('BAS match identity does not match the selected GSR fixture')
    actor_map = {}
    for p in players:
        if p['provider_player_id']:
            actor_map.setdefault((p['provider_player_id'], p['provider_team_side']), []).append(p['player_id'])
    canonical = {label.casefold(): label for label in ACTIONS}
    # Released files use actions; the older example uses annotations.
    with Path(path).open('rb') as stream:
        prefix = next((key for key, event, value in ijson.parse(stream)
                       if event == 'start_array' and key in ('actions', 'annotations')), None)
    if prefix is None:
        raise ValueError('BAS file requires an actions or annotations array')
    accepted, excluded = [], {'other_period': 0, 'ambiguous_period': 0, 'outside_half': 0, 'unmatched_actor': 0}
    for event in items(path, prefix + '.item'):
        clock = str(event.get('gameTime', ''))
        period = re.match(r'^([12])\s*-', clock)
        if not period:
            excluded['ambiguous_period'] += 1
            continue
        if int(period.group(1)) != half:
            excluded['other_period'] += 1
            continue
        label = canonical.get(str(event.get('label', '')).casefold())
        if not label:
            raise ValueError('Unknown BAS action label')
        seconds = (float(event['position']) - offset_ms) / 1000
        if not math.isfinite(seconds):
            raise ValueError('BAS timestamp must be finite')
        if not 0 <= seconds < duration:
            excluded['outside_half'] += 1
            continue
        matches = actor_map.get((str(event.get('player_id')), event.get('team')), [])
        if len(matches) != 1:
            excluded['unmatched_actor'] += 1
            continue
        accepted.append({'player_id': matches[0], 'period': half, 'time_s': seconds, 'kind': label,
                         'visibility': event.get('visibility'), 'source': 'SoccerTrack v2 BAS reference',
                         'provider_position_ms': event['position']})
    return {'source': SOURCE, 'match_id': f'soccertrack:{match_id}', 'alignment_offset_ms': offset_ms,
            'events': accepted, 'excluded': excluded}


def import_half(gsr, match_id, half, video=None, bas=None, event_offset_ms=None,
                alignment_note='', sampling_hz=5, left_attacks='unknown'):
    if str(match_id) not in BENCHMARK_SPLITS:
        raise ValueError('Unknown SoccerTrack benchmark match; verify the provider split before extending the importer')
    identifier = f'soccertrack-{match_id}-h{half}'
    if (S.DATA / 'datasets' / identifier).exists():
        raise ValueError('This half is already present; use a separate data directory for a different import')
    tracks, players, info = parse_half(gsr, match_id, half, sampling_hz, left_attacks)
    source_video, geometry = None, {}
    if video:
        from .vision import video_info
        source_video = Path(video).resolve(strict=True)
        geometry = video_info(source_video)
        video_start = video_start_time(source_video)
        offset_frames = video_start * info['source_fps']
        if (abs(geometry['fps'] - info['source_fps']) > .01
                or abs(offset_frames - round(offset_frames)) > .01
                or abs(geometry['frames'] + round(offset_frames) - info['source_frames']) > 2):
            raise ValueError('Video frame rate/count does not align with the GSR half; verify the exact paired file')
        video_end = video_start + geometry['frames'] / geometry['fps']
        end = min(video_end, info['duration_seconds'])
        original_rows = len(tracks)
        tracks = tracks[tracks.time_s.ge(video_start) & tracks.time_s.lt(end)].copy()
        if tracks.empty:
            raise ValueError('No reference positions overlap the video presentation timeline')
        first_sample = math.ceil(video_start * info['sampling_hz'])
        last_sample = math.ceil(end * info['sampling_hz'])
        present = set(tracks.player_id)
        players = [p for p in players if p['player_id'] in present]
        for player in players:
            player.update(eligible_frames=last_sample-first_sample,
                          playing_seconds=(last_sample-first_sample)/info['sampling_hz'],
                          intervals={str(half): [first_sample, last_sample]})
        info.update(video_start_s=video_start, video_end_s=video_end,
                    video_frames=geometry['frames'], reference_duration_seconds=info['duration_seconds'],
                    duration_seconds=end, total_sampled_frames=last_sample,
                    excluded_reference_rows_outside_video=original_rows-len(tracks),
                    video_alignment_note='Reference clock retained; first video presentation timestamp and frame span checked. Only reference samples within the available video timeline are included.')
    events = None
    if bas and event_offset_ms is not None:
        if not alignment_note.strip():
            raise ValueError('Record how the event/video clock offset was verified')
        events = parse_events(bas, match_id, half, players, info['duration_seconds'], event_offset_ms)
        events['alignment_note'] = alignment_note
    manifest = {'id': identifier, 'title': f'SoccerTrack {match_id} — half {half}',
                'match_id': f'soccertrack:{match_id}', 'date': None, 'source': 'SoccerTrack v2 GSR reference annotations',
                'clock': 'period_relative',
                'source_url': SOURCE, 'format_reference': FORMATS, 'source_kind': 'soccertrack_reference',
                'players': players, **info, 'created': S.now(),
                'video': source_video.name if source_video else None,
                'source_video_path': str(source_video) if source_video else None,
                'width': geometry.get('width'), 'height': geometry.get('height'),
                'source_files': {'gsr_sha256': digest(gsr), 'bas_sha256': digest(bas) if bas else None},
                'benchmark_split': BENCHMARK_SPLITS[str(match_id)],
                'benchmark_split_source': SOURCE + '/blob/eae5179377d3b189d9f15c6b4c3a6a6c41a63eaa/README.md',
                'note': 'Reference pitch annotations, not detector predictions. Half-relative video clock. Jersey/track identities are not verified across matches; confirm position and attack direction before role review.',
                'event_status': 'aligned reference events' if events else 'not imported; verified event-clock alignment required'}
    directory = S.dataset_dir(identifier, True)
    try:
        S.save_tracks(identifier, tracks)
        if events:
            S.write_json(directory / 'aligned_events.json', events)
        S.write_json(directory / 'manifest.json', manifest)
        F.build_profiles(identifier)
    except Exception:
        # Only this newly created import directory is owned by this operation.
        import shutil
        target = directory.resolve()
        if target.parent != (S.DATA / 'datasets').resolve() or target.name != identifier:
            raise RuntimeError('Import cleanup target is outside the dataset directory')
        shutil.rmtree(target)
        raise
    return manifest
