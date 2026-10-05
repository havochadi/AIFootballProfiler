"""Ball-detector training images from SoccerNet tracking clips (true ball box in every frame).

SN-Tracking-2023 train clips (1080p) are scaled to 720p, the resolution of most of this
project's footage, where the ball is about 8 pixels across. Every STEP-th frame is written with a
YOLO label (class 0 = ball); frames without an annotated ball are kept as negatives
(NEGATIVE_SHARE of them). VAL_CLIPS clips are held out for choosing the checkpoint; the test
clips stay untouched for scripts/evaluate_ball_detection.py.

Output: SoccerNet/sn-tracking-2023/yolo_ball/{images,labels}/{train,val} and data.yaml.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\prepare_ball_training.py
"""
from __future__ import annotations

import random
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_ball_detection as B  # noqa: E402
from football_profiler import soccernet as SN  # noqa: E402

STEP = 3
SCALE = 720 / 1080
VAL_CLIPS = 5
NEGATIVE_SHARE = .3


def main():
    import cv2
    rng = random.Random(0)
    root = SN.root() / 'sn-tracking-2023'
    out = root / 'yolo_ball'
    z = zipfile.ZipFile(root / 'train.zip')
    clips = B.clips(z, 'train')
    val = set(rng.sample(clips, VAL_CLIPS))
    counts = {'train': 0, 'val': 0, 'negatives': 0}
    for k, clip in enumerate(clips, 1):
        split = 'val' if clip in val else 'train'
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
        truth = B.ball_truth(z, 'train', clip)
        frames = sorted({int(n.split('/')[-1][:6]) for n in z.namelist()
                         if n.startswith(f'train/{clip}/img1/') and n.endswith('.jpg')})
        for f in frames[::STEP]:
            t = truth.get(f)
            if t is None and rng.random() > NEGATIVE_SHARE:
                continue
            stem = f'{clip}_{f:06d}'
            image = out / 'images' / split / f'{stem}.jpg'
            if not image.is_file():
                img = cv2.imdecode(np.frombuffer(z.read(f'train/{clip}/img1/{f:06d}.jpg'), np.uint8), cv2.IMREAD_COLOR)
                H, W = img.shape[:2]
                img = cv2.resize(img, (round(W * SCALE), round(H * SCALE)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(image), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                H, W = 1080, 1920
            label = ''
            if t is not None:
                x, y, w, h = t
                label = f'0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n'
            else:
                counts['negatives'] += 1
            (out / 'labels' / split / f'{stem}.txt').write_text(label)
            counts[split] += 1
        print(f'[{k}/{len(clips)}] {clip} ({split})', flush=True)
    (out / 'data.yaml').write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    print(counts)


if __name__ == '__main__':
    main()
