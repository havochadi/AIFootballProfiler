"""Pack labelled training videos into clip-ready frame files for fine-tuning the action spotter.

Sources (unpacked by extract_bas_training.py): SoccerNet Ball Action Spotting games (sn-bas-2025,
720p, 12 classes with team side) and FOOTPASS games (sn-pcbas-2026, 640x352, 8 player-centric
classes with the acting player's direction of play). Every second frame (12.5 frames/s, the
spotter's rate) is stored as a JPEG in one file per video with an offset index, so a training
clip is a single contiguous read. BAS frames are resized to the spotter's 796x448 exactly as in
action_spotting._frames; FOOTPASS frames keep their size and are upscaled on the GPU.

Output: SoccerNet/spotter_frames/<dataset>/<split>/<video>/{frames.bin, index.npy, labels.json,
meta.json}. meta.json is written last, so a video with meta.json is complete and is skipped on
rerun. These games are never analysed as matches.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\pack_spotter_frames.py [--workers 6] [--only bas|footpass]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import action_spotting as AS  # noqa: E402
from football_profiler import soccernet as SN  # noqa: E402

JPEG_QUALITY = 90
SOURCE_FPS = 25
# FOOTPASS class ids in the tactical data. The dataset page lists Shot, Header, Throw-in as 4-6,
# but on the validation games class 4 coincides with the published spotter's throw-ins (66%),
# 5 with its shots (70%) and 6 with its headers (31%; 41% unmatched): the ids are in this order.
FOOTPASS_CLASSES = {1: 'Drive', 2: 'Pass', 3: 'Cross', 4: 'Throw-in', 5: 'Shot', 6: 'Header', 7: 'Tackle', 8: 'Block'}
BAS_SPLITS = {'train': 'train', 'valid': 'valid', 'test': 'test'}
FOOTPASS_SPLITS = {'TRAIN': 'train', 'VAL': 'val'}


def out_root():
    return SN.root() / 'spotter_frames'


def bas_jobs():
    base = SN.root() / 'sn-bas-2025'
    for folder, split in BAS_SPLITS.items():
        for video in sorted((base / folder).glob('*/*/*/720p.mp4')):
            game = video.parent
            events = [{'frame': int(int(a['position']) / 1000 * SOURCE_FPS) // AS.FRAME_STRIDE, 'label': a['label'],
                       'side': a.get('team')}
                      for a in json.loads((game / 'Labels-ball.json').read_text(encoding='utf-8'))['annotations']]
            yield {'dataset': 'bas', 'split': split, 'name': game.name, 'video': str(video), 'resize': True,
                   'events': events}


def footpass_events(f, game):
    events = []
    for half in ('H1', 'H2'):
        key = f'{game}_{half}'
        if key not in f:
            continue
        d = f[key][:]
        for row in d[d[:, 13] > 0]:
            # columns: frame, player_id, left_to_right, shirt_number, role_id, ..., class
            events.append({'frame': int(row[0]) // AS.FRAME_STRIDE, 'source_frame': int(row[0]),
                           'label': FOOTPASS_CLASSES[int(row[13])], 'class_id': int(row[13]),
                           'left_to_right': int(row[2]), 'player_id': int(row[1]), 'shirt': int(row[3]),
                           'half': half})
    return sorted(events, key=lambda e: e['frame'])


def footpass_jobs():
    import h5py
    base = SN.root() / 'sn-pcbas-2026'
    for folder, split in FOOTPASS_SPLITS.items():
        h5 = base / f'{split}_tactical_data.h5'
        videos = sorted((base / 'videos_352x640' / folder).glob('game_*.mp4'))
        if not h5.is_file() or not videos:
            continue
        with h5py.File(h5, 'r') as f:
            for video in videos:
                yield {'dataset': 'footpass', 'split': split, 'name': video.stem, 'video': str(video), 'resize': False,
                       'events': footpass_events(f, video.stem)}


def relabel_footpass():
    """Rewrite labels.json of packed FOOTPASS videos from the tactical data (frames untouched)."""
    import h5py
    base = SN.root() / 'sn-pcbas-2026'
    n = 0
    for split in FOOTPASS_SPLITS.values():
        packed = sorted((out_root() / 'footpass' / split).glob('*/meta.json'))
        if not packed:
            continue
        with h5py.File(base / f'{split}_tactical_data.h5', 'r') as f:
            for meta_path in packed:
                folder = meta_path.parent
                frames = json.loads(meta_path.read_text(encoding='utf-8'))['frames']
                events = [e for e in footpass_events(f, folder.name) if e['frame'] < frames]
                (folder / 'labels.json').write_text(json.dumps(events), encoding='utf-8')
                n += 1
    print(f'relabelled {n} packed FOOTPASS videos')


def pack(job):
    import cv2
    cv2.setNumThreads(1)
    out = out_root() / job['dataset'] / job['split'] / job['name']
    if (out / 'meta.json').is_file():
        return job['name'], 'already packed'
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(job['video'])
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    offsets, pos, src, size = [0], 0, 0, None
    with (out / 'frames.bin').open('wb') as f:
        while True:
            if src % AS.FRAME_STRIDE:
                if not cap.grab():
                    break
                src += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            src += 1
            if job['resize']:
                frame = cv2.resize(frame, (AS.WIDTH, AS.HEIGHT), interpolation=cv2.INTER_AREA)
            size = frame.shape[1], frame.shape[0]
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            f.write(buf.tobytes())
            pos += len(buf)
            offsets.append(pos)
    cap.release()
    np.save(out / 'index.npy', np.array(offsets, np.int64))
    n = len(offsets) - 1
    (out / 'labels.json').write_text(json.dumps([e for e in job['events'] if e['frame'] < n]), encoding='utf-8')
    (out / 'meta.json').write_text(json.dumps({'source': job['video'], 'dataset': job['dataset'], 'split': job['split'],
                                               'frames': n, 'source_frames_read': src,
                                               'source_frames_reported': source_frames, 'stride': AS.FRAME_STRIDE,
                                               'frames_per_second': SOURCE_FPS / AS.FRAME_STRIDE,
                                               'width': size[0] if size else None, 'height': size[1] if size else None,
                                               'jpeg_quality': JPEG_QUALITY, 'events': len(job['events'])}),
                                   encoding='utf-8')
    return job['name'], f'{n} frames, {len(job["events"])} events'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--only', choices=['bas', 'footpass'])
    p.add_argument('--relabel', action='store_true', help='only rewrite FOOTPASS labels of packed videos')
    args = p.parse_args()
    if args.relabel:
        return relabel_footpass()
    jobs = []
    if args.only in (None, 'bas'):
        jobs += list(bas_jobs())
    if args.only in (None, 'footpass'):
        jobs += list(footpass_jobs())
    todo = [j for j in jobs if not (out_root() / j['dataset'] / j['split'] / j['name'] / 'meta.json').is_file()]
    print(f'{len(jobs)} videos, {len(todo)} to pack into {out_root()}', flush=True)
    with Pool(args.workers) as pool:
        for k, (name, status) in enumerate(pool.imap_unordered(pack, todo), 1):
            print(f'[{k}/{len(todo)}] {name}: {status}', flush=True)


if __name__ == '__main__':
    main()
