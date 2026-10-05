"""Cut labelled player crops from FOOTPASS full-HD videos (shirt number, team, role known).

FOOTPASS gives every visible player's box in the full-HD picture. This cuts those boxes every
STEP frames (2.5 per second), padded like the pipeline's thumbnails (scripts/extract_crops.py)
and stored at most MAX_HEIGHT pixels tall, one zip per game with a CSV of labels. They train and
measure the shirt-number reader and the appearance model on true identities
(scripts/train_identity_models.py). Training data only; never analysed as matches.

Output: SoccerNet/sn-pcbas-2026/crops/<split>/<game>.zip and <game>.csv (columns: crop, game,
half, frame, player_id, shirt, team, role, box_h).

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\footpass_crops.py --split TRAIN [--workers 2]
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import soccernet as SN  # noqa: E402

STEP = 10
MIN_HEIGHT = 40
MAX_HEIGHT = 160
PAD = .08


def base():
    return SN.root() / 'sn-pcbas-2026'


def jobs(split):
    folder = base() / 'videos_fullHD' / split
    out = base() / 'crops' / split
    return [(split, str(v), str(out)) for v in sorted(folder.rglob('game_*')) if v.suffix.lower() in ('.mp4', '.mkv')
            and not (out / f'{v.stem}.csv').is_file()]


def boxes(split, game):
    """{frame: [(player_id, shirt, team, role, half, x, y, w, h)]} for every STEP-th frame."""
    import h5py
    out = {}
    with h5py.File(base() / f'{split.lower()}_tactical_data.h5', 'r') as f:
        for half in (1, 2):
            key = f'{game}_H{half}'
            if key not in f:
                continue
            d = f[key][:]
            d = d[(d[:, 0] % STEP == 0) & np.isfinite(d[:, 11]) & (d[:, 12] >= MIN_HEIGHT)]
            for r in d:
                out.setdefault(int(r[0]), []).append((int(r[1]), int(r[3]), int(r[1]) // 100, int(r[4]), half,
                                                      r[9], r[10], r[11], r[12]))
    return out


def cut(job):
    import cv2
    cv2.setNumThreads(1)
    split, video, out = job
    video, out = Path(video), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = boxes(split, video.stem)
    last = max(wanted) if wanted else -1
    rows, n = [], 0
    cap = cv2.VideoCapture(str(video))
    tmp = out / f'{video.stem}.zip.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as archive:
        index = 0
        while index <= last:
            if index not in wanted:
                if not cap.grab():
                    break
                index += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            for pid, shirt, team, role, half, x, y, w, h in wanted[index]:
                pad = PAD * h
                a, b = max(0, int(x - pad)), min(W, int(x + w + pad))
                c, e = max(0, int(y - pad)), min(H, int(y + h + pad))
                if b - a < 8 or e - c < 16:
                    continue
                crop = frame[c:e, a:b]
                if crop.shape[0] > MAX_HEIGHT:
                    s = MAX_HEIGHT / crop.shape[0]
                    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), MAX_HEIGHT), interpolation=cv2.INTER_AREA)
                ok, data = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    name = f'{pid}/{index}.jpg'
                    archive.writestr(name, data.tobytes())
                    rows.append({'crop': name, 'game': video.stem, 'half': half, 'frame': index, 'player_id': pid,
                                 'shirt': shirt, 'team': team, 'role': role, 'box_h': round(float(h), 1)})
                    n += 1
            index += 1
    cap.release()
    tmp.replace(out / f'{video.stem}.zip')
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=['crop', 'game', 'half', 'frame', 'player_id', 'shirt', 'team', 'role', 'box_h'])
    w.writeheader()
    w.writerows(rows)
    (out / f'{video.stem}.csv').write_text(buf.getvalue(), encoding='utf-8')
    return video.stem, n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='TRAIN')
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    todo = jobs(args.split)
    print(f'{len(todo)} games to cut', flush=True)
    with Pool(args.workers) as pool:
        for k, (game, n) in enumerate(pool.imap_unordered(cut, todo), 1):
            print(f'[{k}/{len(todo)}] {game}: {n} crops', flush=True)


if __name__ == '__main__':
    main()
