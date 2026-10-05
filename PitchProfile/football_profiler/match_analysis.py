"""Full-match automatic analysis, stage 1: GPU detection pass over one video half.

Writes raw evidence that later CPU stages (calibration filling, teams, ball,
possession, events, identities, statistics) can recompute without the GPU:

- raw_frames.csv.gz   one row per sampled frame (time, shot, calibration)
- raw_people.csv.gz   tracked person boxes per sampled frame
- raw_ball.csv.gz     ball candidates per sampled frame
- raw_kits.json       torso colour samples per tracklet
- crops.zip           a few thumbnails per tracklet for review and jersey reading
"""
from __future__ import annotations

import queue
import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import football_models as FM
from . import fieldcal as FC
from . import storage as S
from .kits import torso_colour

TRACKER_CONFIG = 'botsort_match.yaml'
# HSV hue/saturation histogram distance between consecutive samples. On 42 min of
# SoccerNet 720p at 12.5 samples/s this found 115/133 labelled abrupt cuts, versus
# 19/133 for the earlier mean-grey-difference > 55 rule.
CUT_THRESHOLD = 0.35
KIT_SAMPLES_PER_TRACK = 40
CROPS_PER_TRACK = 8   # only ~18% of player crops show a legible number
CROP_MIN_HEIGHT = 48
CROP_SPACING = 5     # samples between kept thumbnails of one tracklet


class CutDetector:
    """Abrupt shot changes between consecutive sampled frames."""

    def __init__(self, threshold=CUT_THRESHOLD):
        self.threshold, self.previous = threshold, None

    def __call__(self, frame):
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        previous, self.previous = self.previous, hist
        if previous is None:
            return False
        return cv2.compareHist(hist, previous, cv2.HISTCMP_BHATTACHARYYA) > self.threshold


def _tracker(sample_hz, overrides=None):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import YAML, IterableSimpleNamespace
    cfg = IterableSimpleNamespace(**{**YAML.load(S.ROOT / 'config' / TRACKER_CONFIG), **(overrides or {})})
    # frame_rate scales the lost-track buffer: keep about one second of memory.
    return BOTSORT(args=cfg, frame_rate=max(1, int(round(sample_hz))))


def _reader(path, start_frame, stop_frame, step, out, stop):
    cap = cv2.VideoCapture(str(path))
    try:
        # Frames are counted from the start of the file, never sought: OpenCV's seek follows
        # timestamps, which in some files start several frames away from the counted frames.
        index = 0
        while index < start_frame and not stop.is_set():
            if not cap.grab():
                break
            index += 1
        while index < stop_frame and not stop.is_set():
            if (index - start_frame) % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                out.put((index, frame))
            elif not cap.grab():
                break
            index += 1
    finally:
        cap.release()
        out.put(None)


class _Crops:
    """Keep the tallest few, time-spread thumbnails per tracklet."""

    def __init__(self):
        self.best = {}

    def offer(self, key, sample, frame, box):
        x1, y1, x2, y2 = box
        height = y2 - y1
        if height < CROP_MIN_HEIGHT:
            return
        kept = self.best.setdefault(key, [])
        if any(abs(sample - s) < CROP_SPACING for _, s, _ in kept):
            return
        if len(kept) >= CROPS_PER_TRACK and height <= min(h for h, _, _ in kept):
            return
        h, w = frame.shape[:2]
        pad = .08 * height
        a, b = max(0, int(x1 - pad)), min(w, int(x2 + pad))
        c, d = max(0, int(y1 - pad)), min(h, int(y2 + pad))
        ok, data = cv2.imencode('.jpg', frame[c:d, a:b], [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return
        kept.append((height, sample, data.tobytes()))
        if len(kept) > CROPS_PER_TRACK:
            kept.remove(min(kept, key=lambda item: item[0]))

    def write(self, path):
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as archive:
            for key, kept in self.best.items():
                for _, sample, data in sorted(kept, key=lambda item: item[1]):
                    archive.writestr(f'{key}/{sample}.jpg', data)


class _TrackWorker(threading.Thread):
    """CPU tracking, colour sampling and thumbnails, overlapped with GPU inference."""

    def __init__(self, sample_hz, tracker_overrides=None):
        super().__init__(daemon=True)
        self.inbox = queue.Queue(maxsize=4)
        self.tracker = _tracker(sample_hz, tracker_overrides)
        self.people, self.kits, self.crops = [], {}, _Crops()
        self.error = None

    def run(self):
        try:
            while (item := self.inbox.get()) is not None:
                self._process(*item)
        except BaseException as exc:  # surfaced by the producer after each batch
            self.error = exc
            while self.inbox.get() is not None:
                pass

    def _process(self, base, chunk, new_shot, shot_ids, boxes):
        for i, (_, frame) in enumerate(chunk):
            index = base + i
            if new_shot[i]:
                self.tracker.reset()
            b = boxes[i]
            people = b[np.isin(b.cls.astype(int), FM.PEOPLE)]
            tracks = self.tracker.update(people, frame) if len(people) else np.zeros((0, 8), np.float32)
            for x1, y1, x2, y2, tid, score, c, _ in tracks:
                key = f's{shot_ids[i]}-t{int(tid)}'
                self.people.append((index, key, int(c), float(score), float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
                samples = self.kits.setdefault(key, [])
                # Every early observation, then every third: short tracklets still get a kit colour.
                if len(samples) < KIT_SAMPLES_PER_TRACK and (len(samples) < 6 or index % 3 == 0) and int(c) == FM.PLAYER:
                    colour = torso_colour(frame, (x1, y1, x2, y2))
                    if colour is not None:
                        samples.append(colour)
                if int(c) != FM.REFEREE:
                    self.crops.offer(key, index, frame, (x1, y1, x2, y2))


def redetect_ball(identifier, weights, video=None, imgsz=1280, conf=.05, batch_size=16, progress=lambda *a: None):
    """Replace a half's ball candidates (raw_ball.csv.gz) using another ball detector.

    Only the ball is detected again: player tracks, segments, identities, reviewer names and
    ratings stay as they are. The sampled frames are the ones in raw_frames.csv.gz, read by
    counting from the start of the video (as detection_pass). The previous candidates are kept
    once as raw_ball_first.csv.gz. Post-processing afterwards rebuilds the ball path and events.
    video: the analysed file when it is not the dataset's own (FOOTPASS evaluation halves).
    """
    d = S.dataset_dir(identifier)
    manifest = S.read_json(d / 'manifest.json')
    video = Path(video) if video else S.video_path(d, manifest)
    if not video.is_file():
        raise FileNotFoundError(f'Video not found for {identifier}: {video}')
    frames = pd.read_csv(d / 'raw_frames.csv.gz', usecols=['sample', 'source_frame'])
    by_frame = dict(zip(frames.source_frame.astype(int), frames['sample'].astype(int)))
    first, last = min(by_frame), max(by_frame)
    steps = np.diff(sorted(by_frame))
    step = int(pd.Series(steps).mode().iat[0]) if len(steps) else 1
    model = FM._yolo(Path(weights))
    ball_ids = [k for k, v in getattr(model, 'names', {0: 'ball'}).items() if 'ball' in str(v).lower()] or [0]
    frames_in, stop = queue.Queue(maxsize=64), threading.Event()
    reader = threading.Thread(target=_reader, args=(video, first, last + 1, step, frames_in, stop), daemon=True)
    reader.start()
    rows, batch, done = [], [], 0

    def flush():
        res = model.predict([f for _, f in batch], imgsz=imgsz, conf=conf, half=True, verbose=False)
        for (sample, _), r in zip(batch, res):
            b = r.boxes.cpu().numpy()
            keep = np.isin(b.cls.astype(int), ball_ids)
            for j in np.argsort(-b.conf[keep])[:4]:
                x, y, w, h = b.xywh[keep][j]
                rows.append((sample, float(b.conf[keep][j]), float(x), float(y), float(w), float(h), 'ball_model_v2'))
        batch.clear()
    try:
        while True:
            item = frames_in.get()
            if item is None:
                break
            index, frame = item
            if index in by_frame:
                batch.append((by_frame[index], frame))
                done += 1
            if len(batch) >= batch_size:
                flush()
                progress(done / max(len(by_frame), 1), f'Ball detection {done}/{len(by_frame)} frames')
        if batch:
            flush()
    finally:
        stop.set()
    if done < .9 * len(by_frame):                 # never replace the candidates with a partial read
        raise ValueError(f'Only {done} of {len(by_frame)} sampled frames were read from {video}')
    backup = d / 'raw_ball_first.csv.gz'
    if not backup.is_file() and (d / 'raw_ball.csv.gz').is_file():
        (d / 'raw_ball.csv.gz').replace(backup)
    pd.DataFrame(rows, columns=['sample', 'conf', 'cx', 'cy', 'w', 'h', 'source']) \
        .to_csv(d / 'raw_ball.csv.gz', index=False, compression='gzip')
    return {'frames': done, 'candidates': len(rows), 'weights': str(weights)}


def detection_pass(identifier, video_path, *, start_s=0.0, max_seconds=None, target_hz=12.5,
                   calibrate_every=2, known_cuts=(), use_ball_model=True, batch_size=8,
                   imgsz=1280, time_origin_s=0.0, tracker_overrides=None, progress=lambda *a: None):
    """Detect, track and calibrate every sampled frame of one continuous video.

    time_s is the video time minus time_origin_s (the start of a half inside a longer file);
    source_frame stays the frame index in `video_path`.
    """
    import torch
    from .runtime import torch_device
    from .vision import video_info
    FM.require()
    device = str(torch_device())
    half = device.startswith('cuda')
    info = video_info(video_path)
    fps = info['fps']
    step = max(1, int(round(fps / target_hz)))
    sample_hz = fps / step
    start_frame = int(round(max(0.0, start_s) * fps))
    stop_frame = info['frames'] if max_seconds is None else min(info['frames'], start_frame + int(max_seconds * fps))
    if stop_frame - start_frame < step:
        raise ValueError('The selected interval is shorter than one sampled frame')
    expected = (stop_frame - start_frame + step - 1) // step
    directory = S.dataset_dir(identifier, create=True)
    player_model, ball_model = FM.player_detector(), (FM.ball_detector() if use_ball_model else None)
    cuts = np.asarray(sorted(float(t) for t in known_cuts), dtype=float)

    frames_rows, ball_rows = [], []
    shot, sample, last_time = 0, 0, None
    is_cut = CutDetector()
    started = time.perf_counter()
    frames_in = queue.Queue(maxsize=48)
    stop = threading.Event()
    reader = threading.Thread(target=_reader, args=(video_path, start_frame, stop_frame, step, frames_in, stop), daemon=True)
    worker = _TrackWorker(sample_hz, tracker_overrides)
    reader.start()
    worker.start()
    try:
        finished = False
        while not finished:
            chunk = []
            while len(chunk) < batch_size:
                item = frames_in.get()
                if item is None:
                    finished = True
                    break
                chunk.append(item)
            if not chunk:
                break
            images = [frame for _, frame in chunk]
            # Shot boundaries first: the tracker and calibration must not cross a cut.
            new_shot, shot_ids = [], []
            for source_index, frame in chunk:
                t = source_index / fps - time_origin_s
                first = last_time is None
                cut = is_cut(frame) or (not first and bool(((cuts > last_time) & (cuts <= t)).any()))
                if cut:
                    shot += 1
                new_shot.append(cut or first)
                shot_ids.append(shot)
                last_time = t
                frames_rows.append({'sample': sample + len(shot_ids) - 1, 'source_frame': source_index,
                                    'time_s': t, 'shot': shot})
            want = [i for i in range(len(chunk)) if new_shot[i] or (sample + i) % calibrate_every == 0]
            calibrations = dict(zip(want, FC.calibrate([images[i] for i in want]))) if want else {}
            detections = player_model.predict(images, imgsz=imgsz, conf=.1, iou=.5, half=half,
                                              device=device, verbose=False)
            boxes = [r.boxes.cpu().numpy() for r in detections]
            need_ball = [i for i, b in enumerate(boxes) if not (b.cls.astype(int) == FM.BALL).any()]
            extra = {}
            if ball_model is not None and need_ball:
                found = ball_model.predict([images[i] for i in need_ball], imgsz=imgsz, conf=.1, half=half,
                                           device=device, verbose=False)
                extra = {i: r.boxes.cpu().numpy() for i, r in zip(need_ball, found)}
            for i in range(len(chunk)):
                index = sample + i
                row = frames_rows[index]
                cal = calibrations.get(i)
                row.update(calibration_attempted=i in calibrations, calibration_valid=bool(cal and cal['valid']),
                           inliers=cal['inliers'] if cal else 0, rmse_px=cal['rmse_px'] if cal else np.nan,
                           line_support=cal.get('line_support') if cal else np.nan)
                if cal and cal['valid']:
                    row.update({f'h{k}': float(v) for k, v in enumerate(cal['H'].ravel())})
                cls = boxes[i].cls.astype(int)
                for source, candidates in (('player_model', boxes[i][cls == FM.BALL]), ('ball_model', extra.get(i))):
                    if candidates is None or not len(candidates):
                        continue
                    for j in np.argsort(-candidates.conf)[:4]:
                        x, y, w, h = candidates.xywh[j]
                        ball_rows.append((index, float(candidates.conf[j]), float(x), float(y), float(w), float(h), source))
            worker.inbox.put((sample, chunk, new_shot, shot_ids, boxes))
            if worker.error is not None:
                raise worker.error
            sample += len(chunk)
            elapsed = time.perf_counter() - started
            progress(min(.97, sample / max(expected, 1)),
                     f'Analysed {sample}/{expected} frames ({sample / max(elapsed, 1e-6):.1f} frames/s)')
    finally:
        stop.set()
        worker.inbox.put(None)
        worker.join()
        while reader.is_alive():
            try:
                frames_in.get_nowait()
            except queue.Empty:
                reader.join(timeout=.1)
        if half:
            torch.cuda.empty_cache()
    if worker.error is not None:
        raise worker.error
    if not frames_rows:
        raise ValueError('No frames were read from the video')
    frames = pd.DataFrame(frames_rows)
    for k in range(9):
        if f'h{k}' not in frames:
            frames[f'h{k}'] = np.nan
    frames.to_csv(directory / 'raw_frames.csv.gz', index=False, compression='gzip')
    pd.DataFrame(worker.people, columns=['sample', 'track', 'cls', 'conf', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']) \
        .to_csv(directory / 'raw_people.csv.gz', index=False, compression='gzip')
    pd.DataFrame(ball_rows, columns=['sample', 'conf', 'cx', 'cy', 'w', 'h', 'source']) \
        .to_csv(directory / 'raw_ball.csv.gz', index=False, compression='gzip')
    S.write_json(directory / 'raw_kits.json', {k: v for k, v in worker.kits.items() if v})
    worker.crops.write(directory / 'crops.zip')
    wall = time.perf_counter() - started
    return {'source_fps': fps, 'sample_step': step, 'sample_hz': sample_hz, 'start_frame': start_frame,
            'stop_frame': stop_frame, 'time_origin_s': float(time_origin_s), 'tracker_overrides': tracker_overrides,
            'samples': len(frames), 'shots': int(frames.shot.max()) + 1,
            'width': info['width'], 'height': info['height'], 'wall_seconds': wall,
            'frames_per_second': len(frames) / max(wall, 1e-6), 'device': device, 'image_size': imgsz,
            'calibration_every': calibrate_every, 'ball_model': bool(use_ball_model)}
