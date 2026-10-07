"""Local SoccerNet library. Videos stay on the configured data drive."""
from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from functools import lru_cache
from pathlib import Path
from . import storage as S

SOURCE_REVISION = '26e8e46f8258e306fdbf019540e1eda4221a863d'
# SoccerNet's action-spotting label release for all 500 games (same events as
# SN-Labels Labels-v2.json; team is given as the screen side, left/right).
LABEL_ARCHIVES = ('train_labels.zip', 'valid_labels.zip', 'test_labels.zip')


def root():
    return Path(os.environ.get('PITCHPROFILE_SOCCERNET') or S.DATA.parent.parent / 'SoccerNet').resolve()


def _archived_label_index():
    return _label_index(root())


@lru_cache(maxsize=4)
def _label_index(base):
    base = base / 'bas-extra-labels' / 'ExtraLabelsActionSpotting500games'
    index = {}
    for name in LABEL_ARCHIVES:
        if (base / name).is_file():
            with zipfile.ZipFile(base / name) as archive:
                for member in archive.namelist():
                    if member.endswith('/Labels-v2.json'):
                        index[member.rsplit('/', 1)[0]] = (base / name, member)
    return index


def _labels_file(game, filename, archive_first=False):
    """Parsed label file for a game: the synced SN-Labels copy or the 500-game archive (Labels-v2 only)."""
    path = (root() / 'labels' / game / filename).resolve()
    if not path.is_relative_to((root() / 'labels').resolve()):
        raise ValueError('Invalid SoccerNet label path')
    archived = _archived_label_index().get(game) if filename == 'Labels-v2.json' else None
    if path.is_file() and not (archive_first and archived):
        return S.read_json(path, {})
    if archived is None:
        return {}
    with zipfile.ZipFile(archived[0]) as archive:
        return json.loads(archive.read(archived[1]))


def video_root():
    return root() / 'videos-720p'


@lru_cache(maxsize=128)
def _metadata(path, size, modified):
    from .vision import video_info
    return video_info(path)


def official_splits():
    metadata = S.read_json(root() / 'official_splits.json', {})
    # SoccerNet calls the file "Valid"; the app's holdout contract uses
    # "validation". Accept older local sync files without reassigning matches.
    metadata['games'] = {game: ('validation' if split == 'val' else split)
                         for game, split in metadata.get('games', {}).items()}
    return metadata


def working_set():
    """Games in the active working set (root/working_set.json), or None to use every local video.

    Analysis, shot-model training and the app's match list use only these games;
    the other downloaded videos stay on disk. PITCHPROFILE_ALL_VIDEOS=1 disables it.
    """
    if os.environ.get('PITCHPROFILE_ALL_VIDEOS'):
        return None
    games = S.read_json(root() / 'working_set.json', {}).get('games')
    return set(games) if games else None


def library(all_videos=False):
    splits = official_splits().get('games', {})
    active = None if all_videos else working_set()
    entries = []
    base = video_root().resolve()
    for path in sorted(base.glob('*/*/*/[12]_720p.mkv')):
        if not path.resolve().is_relative_to(base):
            continue
        relative = path.relative_to(base).as_posix()
        game = path.parent.relative_to(base).as_posix()
        if active is not None and game not in active:
            continue
        stat = path.stat()
        try:
            info = _metadata(str(path), stat.st_size, stat.st_mtime_ns)
        except ValueError:
            continue
        labels = root() / 'labels' / game
        entries.append({'id': hashlib.sha256(relative.encode()).hexdigest()[:20],
                        'game': game, 'title': path.parent.name, 'half': int(path.name[0]),
                        'relative_path': relative, 'bytes': stat.st_size, **info,
                        'benchmark_split': splits.get(game),
                        'actions_available': (labels / 'Labels-v2.json').is_file() or game in _archived_label_index(),
                        'camera_labels_available': (labels / 'Labels-cameras.json').is_file()})
    return entries


def entry(identifier):
    item = next((x for x in library(all_videos=True) if x['id'] == identifier), None)
    if item is None:
        raise FileNotFoundError('SoccerNet video is unavailable in the configured local library')
    return item


def source_path(identifier):
    item = entry(identifier)
    return video_root() / item['relative_path']


def hq_source(item):
    """(path, start_s) of this half in the 1080p broadcast (videos-HQ), or None when not downloaded.

    The 720p halves (and every SoccerNet label) start `start_time_second` into the HQ file, as
    given by the game's video.ini (scripts/download_soccernet_hq.py).
    """
    import configparser
    folder = root() / 'videos-HQ' / item['game']
    path, ini = folder / f"{item['half']}_HQ.mkv", folder / 'video.ini'
    if not path.is_file() or not ini.is_file():
        return None
    config = configparser.ConfigParser()
    config.read(ini, encoding='utf-8')
    section = f"{item['half']}_HQ.mkv"
    if not config.has_option(section, 'start_time_second'):
        return None
    return path, float(config.get(section, 'start_time_second'))


def annotations(game, half, filename='Labels-v2.json', screen_sides=False):
    """Events of one half. screen_sides=True prefers the archive copy, whose team is 'left'/'right'."""
    # game comes only from the scanned library, never a user-supplied path.
    result = []
    for item in _labels_file(game, filename, archive_first=screen_sides).get('annotations', []):
        try:
            period = int(item['gameTime'].split(' - ')[0])
            seconds = float(item['position']) / 1000
        except (KeyError, ValueError, TypeError):
            continue
        if period != half or not math.isfinite(seconds) or seconds < 0:
            continue
        result.append({**item, 'time_s': seconds, 'half': half})
    return sorted(result, key=lambda x: x['time_s'])
