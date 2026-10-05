"""Measure ball detection on SoccerNet tracking clips (true ball box in every frame it is visible).

SN-Tracking-2023 clips are 30-second 1080p broadcast excerpts; each has one ball tracklet
(gameinfo.ini: "ball;1"). Frames are taken every STEP frames and scaled to 720p by default, the
resolution of most of this project's footage. A detector is right on a frame when its most
confident ball lies within MATCH_PX (at 1080p) of the true ball centre, or inside the true box
grown by half its size. Reported per confidence threshold: the share of frames with a visible
ball where the top detection is right (recall), and the share of reported balls that are right
(precision); also recall by ball size.

Detectors: 'current' (the pipeline: the player detector's ball class, the dedicated ball model
only on frames where that found none), 'player' (player detector only), 'ball' (ball model
only), or a path to other YOLO weights (class 0 = ball, or the only class). --tile runs the
detector on overlapping tiles of the frame as well (SAHI-style) and keeps the best ball.

Report: evidence/ball_detection_<tag>.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_ball_detection.py [--detector current] [--scale 0.6667]
"""
from __future__ import annotations

import argparse
import configparser
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import football_models as FM  # noqa: E402
from football_profiler import soccernet as SN  # noqa: E402
from football_profiler import storage as S  # noqa: E402

STEP = 5
MATCH_PX = 15.0                    # at 1080p
THRESHOLDS = (.05, .1, .2, .3, .4, .5, .6, .7)
SIZE_BINS = [(0, 10, 'under 10 px'), (10, 16, '10-16 px'), (16, 25, '16-25 px'), (25, 1e9, '25 px and over')]


def clips(z, split):
    return sorted({n.split('/')[1] for n in z.namelist() if n.count('/') >= 2 and n.split('/')[1].startswith('SNMOT-')})


def ball_truth(z, split, clip):
    """Frame -> true ball box (x, y, w, h at 1080p) for frames where the ball is annotated."""
    ini = configparser.ConfigParser()
    ini.read_string(z.read(f'{split}/{clip}/gameinfo.ini').decode())
    ids = [int(k.split('_')[1]) for k, v in ini['Sequence'].items()
           if k.startswith('trackletid_') and v.strip().lower().startswith('ball')]
    gt = pd.read_csv(z.open(f'{split}/{clip}/gt/gt.txt'), header=None).iloc[:, :6]
    gt.columns = ['frame', 'id', 'x', 'y', 'w', 'h']
    gt = gt[gt.id.isin(ids) & (gt.w > 0) & (gt.h > 0)]
    return {int(r.frame): (r.x, r.y, r.w, r.h) for r in gt.itertuples()}


def detectors(name, imgsz):
    """A function images -> per image an array of ball candidates [cx, cy, conf] in pixels."""
    def boxes(result, keep):
        b = result.boxes.cpu().numpy()
        m = keep(b.cls.astype(int))
        xywh, conf = b.xywh[m], b.conf[m]
        return np.column_stack([xywh[:, 0], xywh[:, 1], conf]) if len(conf) else np.zeros((0, 3))

    if name in ('current', 'player', 'ball'):
        player = FM.player_detector() if name != 'ball' else None
        ball = FM.ball_detector() if name != 'player' else None

        def run(images):
            out = [np.zeros((0, 3))] * len(images)
            if player is not None:
                res = player.predict(images, imgsz=imgsz, conf=.05, iou=.5, half=True, verbose=False)
                out = [boxes(r, lambda c: c == FM.BALL) for r in res]
            need = [i for i, o in enumerate(out) if not len(o)] if name == 'current' else list(range(len(images)))
            if ball is not None and need:
                res = ball.predict([images[i] for i in need], imgsz=imgsz, conf=.05, half=True, verbose=False)
                for i, r in zip(need, res):
                    out[i] = boxes(r, lambda c: np.ones(len(c), bool))
            return out
        return run
    model = FM._yolo(Path(name))
    names = getattr(model, 'names', {0: 'ball'})
    ball_ids = [k for k, v in names.items() if 'ball' in str(v).lower()] or [0]

    def run(images):
        res = model.predict(images, imgsz=imgsz, conf=.05, half=True, verbose=False)
        return [boxes(r, lambda c: np.isin(c, ball_ids)) for r in res]
    return run


def tiled(run, images, tile=640, overlap=.2):
    """Candidates from the whole frame and from overlapping tiles, in frame pixels."""
    out = [list(c) for c in run(images)]
    H, W = images[0].shape[:2]
    stride = int(tile * (1 - overlap))
    xs = list(range(0, max(W - tile, 0) + 1, stride)) + ([W - tile] if W > tile and (W - tile) % stride else [])
    ys = list(range(0, max(H - tile, 0) + 1, stride)) + ([H - tile] if H > tile and (H - tile) % stride else [])
    for x0 in xs:
        for y0 in ys:
            cand = run([im[y0:y0 + tile, x0:x0 + tile] for im in images])
            for i, c in enumerate(cand):
                out[i] += [[cx + x0, cy + y0, cf] for cx, cy, cf in c]
    return [np.array(o) if o else np.zeros((0, 3)) for o in out]


def evaluate(args):
    import cv2
    z = zipfile.ZipFile(SN.root() / 'sn-tracking-2023' / f'{args.split}.zip')
    run = detectors(args.detector, args.imgsz)
    rows = []
    names = clips(z, args.split)[:args.clips]
    for k, clip in enumerate(names, 1):
        truth = ball_truth(z, args.split, clip)
        frames = [f for f in sorted({int(n.split('/')[-1][:6]) for n in z.namelist()
                                     if n.startswith(f'{args.split}/{clip}/img1/') and n.endswith('.jpg')}) if f % STEP == 0]
        for i in range(0, len(frames), args.batch):
            chunk = frames[i:i + args.batch]
            images = []
            for f in chunk:
                img = cv2.imdecode(np.frombuffer(z.read(f'{args.split}/{clip}/img1/{f:06d}.jpg'), np.uint8), cv2.IMREAD_COLOR)
                if args.scale != 1:
                    img = cv2.resize(img, (round(img.shape[1] * args.scale), round(img.shape[0] * args.scale)),
                                     interpolation=cv2.INTER_AREA)
                images.append(img)
            cands = tiled(run, images, args.tile) if args.tile else run(images)
            for f, c in zip(chunk, cands):
                t = truth.get(f)
                top = c[np.argmax(c[:, 2])] if len(c) else None
                row = {'clip': clip, 'frame': f, 'visible': t is not None, 'size': max(t[2], t[3]) if t else None,
                       'conf': float(top[2]) if top is not None else 0.0, 'right': False}
                if t is not None and top is not None:
                    x, y = top[0] / args.scale, top[1] / args.scale
                    cx, cy = t[0] + t[2] / 2, t[1] + t[3] / 2
                    inside = abs(x - cx) <= t[2] and abs(y - cy) <= t[3]
                    row['right'] = bool(np.hypot(x - cx, y - cy) <= MATCH_PX or inside)
                rows.append(row)
        print(f'[{k}/{len(names)}] {clip}', flush=True)
    return pd.DataFrame(rows)


def summarise(r):
    vis = r[r.visible]
    out = {'frames': int(len(r)), 'frames_with_visible_ball': int(len(vis)), 'by_threshold': []}
    for th in THRESHOLDS:
        reported = r[r.conf >= th]
        out['by_threshold'].append({
            'min_confidence': th,
            'recall': round(float((vis.right & (vis.conf >= th)).mean()), 3) if len(vis) else None,
            'precision': round(float(reported.right.mean()), 3) if len(reported) else None,
            'reported_share': round(float(len(reported) / max(len(r), 1)), 3)})
    th = .1
    out['recall_by_size_at_0.1'] = {name: round(float(((g.conf >= th) & g.right).mean()), 3) if len(g) else None
                                    for lo, hi, name in SIZE_BINS
                                    for g in [vis[(vis['size'] >= lo) & (vis['size'] < hi)]]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--detector', default='current', help="current, player, ball or a path to YOLO weights")
    p.add_argument('--split', default='test')
    p.add_argument('--scale', type=float, default=720 / 1080, help='frame scale before detection (1 = 1080p)')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--tile', type=int, default=0, help='also detect on tiles of this size (0 = off)')
    p.add_argument('--clips', type=int, default=None)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--tag', default=None)
    args = p.parse_args()
    r = evaluate(args)
    tag = args.tag or (Path(args.detector).stem if args.detector not in ('current', 'player', 'ball') else args.detector)
    report = {'created': S.now(), 'detector': args.detector, 'scale': args.scale, 'imgsz': args.imgsz, 'tile': args.tile,
              'step': STEP, 'match_px_1080p': MATCH_PX, **summarise(r)}
    S.write_json(S.EVIDENCE / f'ball_detection_{tag}.json', report)
    print({k: v for k, v in report.items() if k != 'by_threshold'})
    for row in report['by_threshold']:
        print(row)


if __name__ == '__main__':
    main()
