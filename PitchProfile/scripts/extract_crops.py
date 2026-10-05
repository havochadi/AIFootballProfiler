"""Re-cut player thumbnails from the video using the boxes stored by the detection pass.

The detection pass keeps at most 8 thumbnails per tracklet (about 2.6 on average). The shirt-number
reader and the re-identification embeddings both improve with more views, so this pass keeps up to
PER_TRACK of the tallest boxes of every tracklet, at least SPACING samples apart, without running
detection again. The previous archive is kept as crops_v1.zip.

--hq cuts the same thumbnails from the 1080p broadcast (videos-HQ) where it is sharper than the
720p file the half was analysed from: boxes are scaled to the HQ frame and frames are shifted by
the half's start inside the HQ file (soccernet.hq_source). Shirt numbers and appearance are then
read from the sharper crops by the post-processing pass. The 720p archive is kept as
crops_720p.zip. Halves whose HQ file is not taller than 720p are left as they are.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\extract_crops.py [--match Chelsea] [--force] [--hq]
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import football_models as FM
from football_profiler import match_pipeline as MP
from football_profiler import soccernet as SN
from football_profiler import storage as S

PER_TRACK = 24
MIN_HEIGHT = 40
SPACING = 3                    # samples (0.24 s at 12.5 samples/s)
PAD = .08
VERSION = 2


def select(people):
    """{sample: [(track, box)]} of the thumbnails to cut."""
    q = people[(people.cls != FM.REFEREE) & (people.bbox_h >= MIN_HEIGHT)]
    wanted = {}
    for track, g in q.sort_values('bbox_h', ascending=False).groupby('track', sort=False):
        kept = []
        for r in g.itertuples():
            if all(abs(r.sample - k) >= SPACING for k in kept):
                kept.append(r.sample)
                wanted.setdefault(int(r.sample), []).append((track, (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h)))
                if len(kept) >= PER_TRACK:
                    break
    return wanted


def extract(identifier, video_path, hq=None):
    """Re-cut thumbnails; hq = (HQ path, frame shift, box scale) cuts them from the HQ broadcast."""
    d = S.dataset_dir(identifier)
    people = pd.read_csv(d / 'raw_people.csv.gz', dtype={'track': str})
    frames = pd.read_csv(d / 'raw_frames.csv.gz', usecols=['sample', 'source_frame']).set_index('sample').source_frame
    wanted = select(people)
    shift, scale = (hq[1], hq[2]) if hq else (0, 1.0)
    if hq:
        video_path = hq[0]
        wanted = {s: [(t, tuple(v * scale for v in box)) for t, box in items] for s, items in wanted.items()}
    by_frame = {int(frames[s]) + shift: items for s, items in wanted.items()}
    sample_of = {int(frames[s]) + shift: s for s in wanted}
    tmp = d / 'crops.v2.tmp'
    n = 0
    cap = cv2.VideoCapture(str(video_path))
    try:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as archive:
            last = max(by_frame) if by_frame else -1
            # Frames are counted from the start of the file, never sought: OpenCV's seek follows
            # timestamps, which in one HQ file start 7 frames away from the counted frames.
            index = 0
            while index <= last:
                if index not in by_frame:
                    if not cap.grab():
                        break
                    index += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                sample = sample_of[index]
                for track, (x, y, bw, bh) in by_frame[index]:
                    pad = PAD * bh
                    a, b = max(0, int(x - pad)), min(w, int(x + bw + pad))
                    c, e = max(0, int(y - pad)), min(h, int(y + bh + pad))
                    if b - a < 4 or e - c < 4:
                        continue
                    ok, data = cv2.imencode('.jpg', frame[c:e, a:b], [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if ok:
                        archive.writestr(f'{track}/{sample}.jpg', data.tobytes())
                        n += 1
                index += 1
    finally:
        cap.release()
    old = d / 'crops.zip'
    meta = S.read_json(d / 'analysis.json')
    if hq:
        reference = d / 'crops_720p.zip' if (meta.get('crops') or {}).get('source') == 'HQ' else old
        agreement = verify(tmp, reference)
        if agreement < VERIFY_MIN_CORR:
            tmp.unlink()
            print(f'  {identifier}: HQ thumbnails do not match the 720p ones (median correlation '
                  f'{agreement:.2f}); kept the 720p thumbnails', flush=True)
            return None
        print(f'  {identifier}: HQ thumbnails match the 720p ones (median correlation {agreement:.2f})', flush=True)
    if old.is_file() and not (d / 'crops_v1.zip').is_file():
        old.rename(d / 'crops_v1.zip')
    elif old.is_file() and hq and (meta.get('crops') or {}).get('source', '720p') == '720p':
        old.replace(d / 'crops_720p.zip')
    tmp.replace(old)
    meta['crops'] = {'version': VERSION, 'per_track': PER_TRACK, 'min_height': MIN_HEIGHT, 'spacing': SPACING,
                     'crops': n, 'tracks': int(len({t for items in wanted.values() for t, _ in items})),
                     'source': 'HQ' if hq else '720p', 'video': str(video_path), 'frame_shift': shift,
                     'box_scale': scale, 'created': S.now()}
    S.write_json(d / 'analysis.json', meta)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--match', default='')
    p.add_argument('--force', action='store_true')
    p.add_argument('--hq', action='store_true', help='cut from the 1080p broadcast where it is sharper than 720p')
    args = p.parse_args()
    items = [v for v in SN.library() if args.match.lower() in v['game'].lower()]
    for k, item in enumerate(items, 1):
        identifier = MP.soccernet_identifier(item['game'], item['half'])
        d = S.dataset_dir(identifier)
        meta = S.read_json(d / 'analysis.json', {}) or {}
        crops = meta.get('crops') or {}
        if args.hq and crops.get('hq_rejected') and not args.force:
            print(f'[{k}/{len(items)}] {identifier}: HQ thumbnails were tried and rejected ({crops.get("note")})', flush=True)
            continue
        hq = hq_plan(item) if args.hq else None
        if args.hq and hq is None:
            print(f'[{k}/{len(items)}] {identifier}: no HQ file sharper than 720p, left as is', flush=True)
            continue
        want = 'HQ' if hq else '720p'
        if crops.get('version') == VERSION and crops.get('source', '720p') == want and not args.force:
            print(f'[{k}/{len(items)}] {identifier}: already re-cut ({want})', flush=True)
            continue
        started = time.time()
        n = extract(identifier, SN.source_path(item['id']), hq)
        if n is None:
            continue
        print(f'[{k}/{len(items)}] {identifier}: {n} {want} thumbnails in {(time.time() - started) / 60:.1f} min',
              flush=True)


ALIGN_WINDOW_S = 3.0
ALIGN_MIN_CORR = .95
VERIFY_MIN_CORR = .8
VERIFY_SAMPLE = 200


def _thumb(frame, size=(96, 54)):
    x = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(float).ravel()
    return (x - x.mean()) / (x.std() + 1e-9)


def align(lq_path, hq_path, start_s, fps, duration_s):
    """Shift, in frames counted from the start of each file, of the 720p half inside the HQ file.

    video.ini gives whole seconds, and OpenCV's seek follows timestamps that in one HQ file start 7
    frames away from the counted frames, so the shift is measured the way extract() reads: three
    720p frames (a quarter, half and three quarters into the half; seeks in the 720p files match
    their counted frames) are compared with every HQ frame, read in order from the start, within
    ALIGN_WINDOW_S of the ini position. All three must agree to the frame with correlation of at
    least ALIGN_MIN_CORR; otherwise None.
    """
    probes = [duration_s * f for f in (.25, .5, .75)]
    cap = cv2.VideoCapture(str(lq_path))
    refs = {}
    for t in probes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            return None
        refs[t] = _thumb(f)
    cap.release()
    window = int(ALIGN_WINDOW_S * fps)
    target = {t: int(round((t + start_s) * fps)) for t in probes}
    best = {t: (-1.0, None) for t in probes}
    cap = cv2.VideoCapture(str(hq_path))
    index, last = 0, max(target.values()) + window
    while index <= last:
        near = [t for t in probes if abs(index - target[t]) <= window]
        if not near:
            if not cap.grab():
                break
            index += 1
            continue
        ok, f = cap.read()
        if not ok:
            break
        th = _thumb(f)
        for t in near:
            c = float(refs[t] @ th / len(th))
            if c > best[t][0]:
                best[t] = (c, index - int(t * fps))
        index += 1
    cap.release()
    if any(c < ALIGN_MIN_CORR for c, _ in best.values()):
        return None
    shifts = [s for _, s in best.values()]
    return shifts[0] if max(shifts) - min(shifts) <= 1 else None


def verify(hq_zip, ref_zip):
    """Median picture correlation of up to VERIFY_SAMPLE HQ thumbnails with their 720p versions."""
    import random
    import numpy as np
    with zipfile.ZipFile(hq_zip) as a, zipfile.ZipFile(ref_zip) as b:
        names = sorted(set(a.namelist()) & set(b.namelist()))
        random.Random(0).shuffle(names)
        scores = []
        for n in names[:VERIFY_SAMPLE]:
            hq = cv2.imdecode(np.frombuffer(a.read(n), np.uint8), cv2.IMREAD_COLOR)
            lq = cv2.imdecode(np.frombuffer(b.read(n), np.uint8), cv2.IMREAD_COLOR)
            if hq is None or lq is None:
                continue
            x, y = _thumb(lq, (24, 48)), _thumb(hq, (24, 48))
            scores.append(float(x @ y / len(x)))
    return float(np.median(scores)) if scores else 0.0


def hq_plan(item):
    """(HQ path, frame shift, box scale) when this half's HQ broadcast is taller than its 720p file
    and its frames are verified to line up (align)."""
    from football_profiler.vision import video_info
    src = SN.hq_source(item)
    if src is None:
        return None
    path, start_s = src
    lq_path = SN.source_path(item['id'])
    lq, hq = video_info(lq_path), video_info(path)
    if hq['height'] <= lq['height'] or abs(hq['fps'] - lq['fps']) > .01:
        return None
    shift = align(lq_path, path, start_s, lq['fps'], lq['duration'])
    if shift is None:
        print(f"  {item['game']} H{item['half']}: HQ frames do not line up with the 720p half; left as is", flush=True)
        return None
    return path, shift, hq['width'] / lq['width']


if __name__ == '__main__':
    main()
