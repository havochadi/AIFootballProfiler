"""Fine-tune the ball detector on SoccerNet tracking clips at this project's resolution.

Starts from the football ball model (football-ball-detection.pt, YOLOv8x) and trains on the
720p frames written by scripts/prepare_ball_training.py (the ball about 8 pixels across), with
the early backbone layers frozen. Weights and training curves go to
<weights>/ball_v2/<name>/ on the data drive. Measure the result with
scripts/evaluate_ball_detection.py --detector <weights>/ball_v2/<name>/weights/best.pt.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\train_ball_detector.py [--epochs 8] [--name sn_tracking]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import football_models as FM  # noqa: E402
from football_profiler import soccernet as SN  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--freeze', type=int, default=10, help='backbone layers kept fixed')
    p.add_argument('--name', default='sn_tracking')
    args = p.parse_args()
    data = SN.root() / 'sn-tracking-2023' / 'yolo_ball' / 'data.yaml'
    model = FM._yolo(FM.weights_dir() / FM.BALL_WEIGHTS)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, freeze=args.freeze,
                project=str(FM.weights_dir() / 'ball_v2'), name=args.name, exist_ok=True, workers=4, patience=4,
                cos_lr=True, close_mosaic=2, plots=False, verbose=False, single_cls=True)


if __name__ == '__main__':
    main()
