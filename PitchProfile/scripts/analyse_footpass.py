"""Run the full PitchProfile pipeline on FOOTPASS games, for measurement against their ground truth.

FOOTPASS (SoccerNet player-centric ball action spotting, NDA data) gives, for every frame of its
broadcast games, all 22 players' shirt numbers and teams, their boxes in the full-HD picture, and
every ball action with the acting player. This script analyses its full-HD videos exactly as a
SoccerNet half is analysed (detection and tracking, action spotting, thumbnails, post-processing)
so scripts/evaluate_identity_footpass.py can score detection, identity and per-player actions.

Each FOOTPASS video holds a whole match; each half is analysed separately with its own time
origin (the first labelled frame of the half), as SoccerNet halves are. Results go to a separate
data folder (PitchProfile-eval next to PitchProfile), so these games never appear in the app.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\analyse_footpass.py [--split VAL] [--game game_18] [--force]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_main = Path((ROOT / '.data-location').read_text(encoding='utf-8').strip()) if (ROOT / '.data-location').is_file() \
    else ROOT / 'data'
os.environ.setdefault('PITCHPROFILE_DATA', str(_main.parent.parent / 'PitchProfile-eval' / 'data'))
os.environ.setdefault('PITCHPROFILE_WEIGHTS', str(_main.parent / 'weights'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from football_profiler import action_spotting as AS  # noqa: E402
from football_profiler import match_analysis as MA  # noqa: E402
from football_profiler import match_pipeline as MP  # noqa: E402
from football_profiler import soccernet as SN  # noqa: E402
from football_profiler import storage as S  # noqa: E402

FPS = 25.0
MARGIN_FRAMES = 25


def base():
    return SN.root() / 'sn-pcbas-2026'


def videos(split):
    return sorted(p for p in (base() / 'videos_fullHD' / split).rglob('game_*') if p.suffix.lower() in ('.mp4', '.mkv'))


def halves(split, game):
    """{half: (first frame, last frame)} of the labelled frames of each half."""
    import h5py
    out = {}
    with h5py.File(base() / f'{split.lower()}_tactical_data.h5', 'r') as f:
        for h in (1, 2):
            key = f'{game}_H{h}'
            if key in f:
                frames = f[key][:, 0]
                out[h] = (int(frames.min()), int(frames.max()))
    return out


def identifier(split, game, half, variant=''):
    return f"fp-{split.lower()}-{game.replace('_', '')}-h{half}" + (f'-{variant}' if variant else '')


def game_spots(video):
    """Action spots for the whole video (cached in the evaluation data folder)."""
    cache = S.DATA / 'footpass_spots' / f'{video.stem}.json'
    saved = S.read_json(cache, None)
    if saved:
        return saved
    result = AS.spot_video(video)
    result['created'] = S.now()
    S.write_json(cache, result)
    return result


def analyse_half(split, video, half, first, last, force=False, minutes=None, variant='', tracker=None):
    import extract_crops as EC
    ident = identifier(split, video.stem, half, variant)
    d = S.dataset_dir(ident, create=True)
    meta = S.read_json(d / 'analysis.json', {}) or {}
    if meta.get('postprocess') and not force:
        print(f'  {ident}: already analysed', flush=True)
        return ident
    start, stop = max(0, first - MARGIN_FRAMES), last + MARGIN_FRAMES
    if minutes:
        stop = min(stop, start + int(minutes * 60 * FPS))
    origin = start / FPS
    started, last_print = time.time(), [0.0]

    def progress(fraction, message):
        if time.time() - last_print[0] > 120:
            print(f'    {fraction:.0%} {message}', flush=True)
            last_print[0] = time.time()

    detection = MA.detection_pass(ident, video, start_s=origin, max_seconds=(stop - start) / FPS,
                                  time_origin_s=origin, tracker_overrides=tracker, progress=progress)
    meta = {'title': f'FOOTPASS {split} {video.stem} H{half}', 'match_id': f'footpass:{split}:{video.stem}',
            'half': half, 'source': 'FOOTPASS full-HD broadcast (evaluation only; never shown in the app)',
            'footpass': {'split': split, 'game': video.stem, 'half': half, 'first_frame': start, 'last_frame': stop,
                         'variant': variant, 'tracker_overrides': tracker},
            'detection': detection, 'video_name': video.name, 'created': S.now()}
    S.write_json(d / 'analysis.json', meta)
    spots = game_spots(video)
    duration = (stop - start) / FPS
    # Spot frames count at 12.5 frames/s (every second video frame).
    shift_frames = int(round(origin * FPS / AS.FRAME_STRIDE))
    shifted = [{**s, 'time_s': s['time_s'] - origin, 'frame': s['frame'] - shift_frames}
               for s in spots['spots'] if 0 <= s['time_s'] - origin <= duration]
    S.write_json(d / 'action_spots.json', {**{k: v for k, v in spots.items() if k != 'spots'}, 'spots': shifted,
                                           'time_origin_s': origin})
    EC.extract(ident, video)
    MP.postprocess(ident)
    print(f'  {ident}: done in {(time.time() - started) / 60:.1f} min', flush=True)
    return ident


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='VAL')
    p.add_argument('--game', default='')
    p.add_argument('--force', action='store_true')
    p.add_argument('--minutes', type=float, default=None, help='analyse only the first minutes of each half')
    p.add_argument('--variant', default='', help='suffix for the dataset names (an experiment)')
    p.add_argument('--tracker', nargs='*', default=[], help='tracker overrides, e.g. track_buffer=90 with_reid=true')
    args = p.parse_args()
    tracker = {}
    for kv in args.tracker:
        k, v = kv.split('=', 1)
        tracker[k] = {'true': True, 'false': False}.get(v.lower(), float(v) if v.replace('.', '', 1).isdigit() else v)
        if isinstance(tracker[k], float) and tracker[k].is_integer():
            tracker[k] = int(tracker[k])
    found = [v for v in videos(args.split) if args.game in v.stem]
    if not found:
        raise SystemExit(f'No full-HD FOOTPASS videos in {base() / "videos_fullHD" / args.split}: download and unpack '
                         'them first (download, then scripts/extract_bas_training.py)')
    print(f'{len(found)} games; results in {S.DATA}', flush=True)
    for video in found:
        for half, (first, last) in halves(args.split, video.stem).items():
            analyse_half(args.split, video, half, first, last, args.force, args.minutes, args.variant, tracker or None)


if __name__ == '__main__':
    main()
