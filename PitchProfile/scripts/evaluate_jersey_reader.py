"""Evaluate the jersey-number reader (football_profiler.jersey) on SoccerNet Jersey 2023.

Download test.zip of the Hugging Face dataset SoccerNet/SN-Jersey-2023 into --data.
Evenly spaced crops per tracklet are cached next to the zip. The report is
tracklet-level accuracy including "not visible" (-1) tracklets, written to evidence.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_jersey_reader.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import jersey as J
from football_profiler import storage as S

CROPS = 32
CACHE_SIZE = (64, 128)


def cache(data: Path, split: str):
    """Up to CROPS evenly spaced crops per tracklet, resized once for fast evaluation."""
    out = data / f'cache_{split}_{CROPS}.npz'
    if out.is_file():
        z = np.load(out)
        return z['images'], z['owner'], z['labels'], z['ids']
    archive = zipfile.ZipFile(data / f'{split}.zip')
    names = archive.namelist()
    labels = json.loads(archive.read(next(n for n in names if n.endswith('_gt.json'))))
    by_track = {}
    for n in names:
        if n.endswith('.jpg'):
            by_track.setdefault(n.split('/')[-2], []).append(n)
    ids = sorted(by_track, key=int)
    images, owner = [], []
    for t, track in enumerate(ids):
        files = sorted(by_track[track], key=lambda n: int(n.rsplit('_', 1)[1].split('.')[0]))
        for i in np.linspace(0, len(files) - 1, min(CROPS, len(files))).round().astype(int):
            img = cv2.imdecode(np.frombuffer(archive.read(files[i]), np.uint8), cv2.IMREAD_COLOR)
            images.append(cv2.resize(img, CACHE_SIZE, interpolation=cv2.INTER_AREA))
            owner.append(t)
    images, owner = np.stack(images), np.asarray(owner, np.int32)
    lab = np.asarray([int(labels[i]) for i in ids], np.int32)
    np.savez(out, images=images, owner=owner, labels=lab, ids=np.asarray(ids))
    return images, owner, lab, np.asarray(ids)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path(r'D:\CVDL Football Data\SoccerNet\jersey-2023'))
    p.add_argument('--split', default='test')
    args = p.parse_args()
    images, owner, labels, _ = cache(args.data, args.split)
    started = time.perf_counter()
    groups = {t: [images[i] for i in np.flatnonzero(owner == t)] for t in range(len(labels))}
    readings = J.read_groups(groups)
    pred = np.array([readings[t]['number'] if readings[t]['number'] is not None else -1 for t in range(len(labels))])
    legible = labels >= 0
    report = {
        'dataset': f'SoccerNet Jersey 2023, official {args.split} split',
        'reader': 'SoccerNet legibility ResNet-34 + SoccerNet fine-tuned PARSeq (Koshkina & Elder), '
                  f'torso band {J.TORSO} of the player box instead of pose keypoints',
        'tracklets': int(len(labels)), 'crops_per_tracklet': CROPS,
        'accuracy': float(np.mean(pred == labels)),
        'legible_tracklets_correct': float(np.mean(pred[legible] == labels[legible])),
        'illegible_tracklets_correct': float(np.mean(pred[~legible] == -1)),
        'seconds': time.perf_counter() - started,
        'note': 'Tracklet-level accuracy including "not visible" (-1) tracklets, on evenly spaced crops '
                'rather than every frame. These are 1080p SoccerNet tracking crops; 720p broadcast players are smaller.',
    }
    print(json.dumps(report, indent=2))
    S.write_json(S.EVIDENCE / 'jersey_number_reader.json', report)


if __name__ == '__main__':
    main()
