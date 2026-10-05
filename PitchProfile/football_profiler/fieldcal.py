"""Automatic broadcast pitch calibration from NBJW keypoint heatmaps.

Coordinates use a 105 x 68 m pitch with the origin at the far-left corner as
seen from the main camera: x runs towards the right goal, y towards the near
touchline. Only ground-plane keypoints enter the homography.
"""
from __future__ import annotations

import cv2
import numpy as np

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
# NBJW keypoint world positions (57, one-based order), in this pitch frame.
KEYPOINTS = np.array([
    [0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84], [105., 13.84],
    [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84], [0., 30.34], [0., 30.34],
    [105., 30.34], [105., 30.34], [0., 37.66], [0., 37.66], [105., 37.66], [105., 37.66],
    [0., 43.16], [5.5, 43.16], [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16],
    [88.5, 54.16], [105., 54.16], [0., 68.], [52.5, 68.], [105., 68.], [16.5, 26.68],
    [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15], [88.5, 41.31], [19.99, 32.29],
    [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7], [43.68, 36.46], [61.31, 36.46],
    [85., 35.7], [11., 34.], [16.5, 34.], [20.15, 34.], [46.03, 27.53], [58.97, 27.53],
    [43.35, 34.], [52.5, 34.], [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.],
    [88.5, 34.], [94., 34.]])
# One-based ids of crossbar points, which are 2.44 m above the ground plane.
ELEVATED = (12, 15, 16, 19)
GROUND = np.array([i not in ELEVATED for i in range(1, 58)])
NET_SIZE = (960, 540)
KEYPOINT_THRESHOLD = 0.1486  # NBJW's published operating point
BORDER = 15  # heatmap pixels; NBJW never accepts peaks this close to the border
MIN_INLIERS = 5
MAX_RMSE_PX = 4.0
RANSAC_PX = 6.0


def _prepare(frames, device, half):
    import torch
    batch = np.stack([cv2.cvtColor(cv2.resize(f, NET_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                      for f in frames])
    x = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2)
    return (x.half() if half else x.float()) / 255


def decode(heatmaps, threshold=KEYPOINT_THRESHOLD):
    """Top peak per channel with log-parabola sub-pixel refinement; returns (B,57,3) x,y,score in net pixels."""
    import torch
    hm = heatmaps[:, :57].float()
    b, c, h, w = hm.shape
    hm = hm.clone()
    hm[..., :BORDER, :] = 0; hm[..., -BORDER:, :] = 0
    hm[..., :, :BORDER] = 0; hm[..., :, -BORDER:] = 0
    flat = hm.view(b, c, -1)
    scores, index = flat.max(-1)
    ys, xs = torch.div(index, w, rounding_mode='floor'), index % w

    def value(y, x):
        return flat.gather(2, (y.clamp(0, h - 1) * w + x.clamp(0, w - 1))[..., None])[..., 0].clamp_min(1e-6).log()

    def offset(minus, centre, plus):
        # A parabola through the log values is exact for Gaussian-shaped peaks.
        curvature = minus - 2 * centre + plus
        d = torch.where(curvature < -1e-6, .5 * (minus - plus) / curvature, torch.zeros_like(curvature))
        return d.clamp(-.5, .5)

    centre = value(ys, xs)
    dx = offset(value(ys, xs - 1), centre, value(ys, xs + 1))
    dy = offset(value(ys - 1, xs), centre, value(ys + 1, xs))
    # The network output is at half the input resolution.
    out = torch.stack([(xs + dx) * 2, (ys + dy) * 2, scores], -1)
    out[scores < threshold] = float('nan')
    return out.cpu().numpy()


def fit(image_points, world_points, frame_shape):
    """Robust world->image homography, returned as image->pitch plus quality."""
    if len(image_points) < MIN_INLIERS:
        return None
    world = np.asarray(world_points, np.float64)
    image = np.asarray(image_points, np.float64)
    H, mask = cv2.findHomography(world, image, cv2.RANSAC, RANSAC_PX, maxIters=2000, confidence=.999)
    if H is None or mask is None or abs(np.linalg.det(H)) < 1e-12:
        return None
    inl = mask.ravel().astype(bool)
    if inl.sum() < MIN_INLIERS:
        return None
    # Refit on inliers only, then measure the pixel error on those points.
    H, _ = cv2.findHomography(world[inl], image[inl], 0)
    if H is None or abs(np.linalg.det(H)) < 1e-12:
        return None
    proj = cv2.perspectiveTransform(world[inl][None], H)[0]
    rmse = float(np.sqrt(np.mean(np.sum((proj - image[inl]) ** 2, 1))))
    spread = np.linalg.svd(world[inl] - world[inl].mean(0), compute_uv=False) / np.sqrt(inl.sum())
    valid = rmse <= MAX_RMSE_PX and spread.min() >= 2.0
    try:
        to_pitch = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    to_pitch /= to_pitch[2, 2]
    # The camera must see the pitch from above: image points below the horizon map forwards.
    h, w = frame_shape[:2]
    probe = cv2.perspectiveTransform(np.array([[[w / 2, h * .95], [w / 2, h * .5]]], np.float64), to_pitch)[0]
    if not np.isfinite(probe).all() or np.abs(probe).max() > 1000:
        valid = False
    return {'H': to_pitch, 'inliers': int(inl.sum()), 'points': int(len(world)), 'rmse_px': rmse,
            'spread_m': float(spread.min()), 'valid': bool(valid)}


SUPPORT_SIZE = (640, 360)
SUPPORT_TOLERANCE_PX = 3.0   # at SUPPORT_SIZE
MIN_SUPPORT = .45            # share of visible projected markings lying on white lines
MIN_SUPPORT_SAMPLES = 40     # fewer visible marking samples cannot verify a fit
RELIABLE_INLIERS = 8         # fits this well supported by keypoints need no line check


def line_evidence(frame):
    """Distance (at SUPPORT_SIZE) to the nearest painted-line pixel, and the grass region."""
    small = cv2.resize(frame, SUPPORT_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    field = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    lines = (tophat > 22) & (field > 0)
    distance = cv2.distanceTransform(np.where(lines, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    return distance, field > 0


_MARKINGS = None


def _marking_samples():
    global _MARKINGS
    if _MARKINGS is None:
        pts = []
        for line in pitch_polylines():
            for a, b in zip(line[:-1], line[1:]):
                n = max(2, int(np.ceil(np.linalg.norm(b - a) / .5)))
                pts.append(np.linspace(a, b, n, endpoint=False))
        _MARKINGS = np.concatenate(pts)
    return _MARKINGS


def line_support(H, evidence, frame_shape):
    """Share of visible projected pitch markings that coincide with painted lines."""
    distance, field = evidence
    h, w = frame_shape[:2]
    scale = np.array([SUPPORT_SIZE[0] / w, SUPPORT_SIZE[1] / h])
    img = to_pitch(_marking_samples(), np.linalg.inv(H)) * scale
    ok = np.isfinite(img).all(1)
    img = img[ok]
    inside = (img[:, 0] >= 0) & (img[:, 0] < SUPPORT_SIZE[0] - 1) & (img[:, 1] >= 0) & (img[:, 1] < SUPPORT_SIZE[1] - 1)
    px = np.round(img[inside]).astype(int)
    on_field = field[px[:, 1], px[:, 0]]
    px = px[on_field]
    if len(px) < MIN_SUPPORT_SAMPLES:
        return None, int(len(px))
    return float(np.mean(distance[px[:, 1], px[:, 0]] <= SUPPORT_TOLERANCE_PX)), int(len(px))


def calibrate(frames, batch_size=8):
    """Calibrate same-sized BGR frames; returns one result dict (or None) per frame."""
    import torch
    from .football_models import pitch_keypoint_model
    net, device = pitch_keypoint_model()
    half = device.type == 'cuda'
    results = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        with torch.inference_mode():
            peaks = decode(net(_prepare(chunk, device, half)))
        for frame, kp in zip(chunk, peaks):
            h, w = frame.shape[:2]
            sx, sy = w / NET_SIZE[0], h / NET_SIZE[1]
            ok = GROUND & np.isfinite(kp[:, 0])
            image = kp[ok, :2] * [sx, sy]
            result = fit(image, KEYPOINTS[ok], frame.shape)
            if result is not None:
                result['keypoints'] = {int(i + 1): [float(x * sx), float(y * sy)]
                                       for i, (x, y) in enumerate(kp[:, :2]) if np.isfinite(x)}
                support, samples = line_support(result['H'], line_evidence(frame), frame.shape)
                result['line_support'], result['line_samples'] = support, samples
                # Few keypoints can agree on a wrong correspondence; painted lines must confirm it.
                if result['inliers'] < RELIABLE_INLIERS:
                    result['valid'] = result['valid'] and support is not None and support >= MIN_SUPPORT
                elif support is not None and support < MIN_SUPPORT / 2:
                    result['valid'] = False
            results.append(result)
    return results


def to_pitch(points, H):
    p = np.asarray(points, np.float64).reshape(-1, 1, 2)
    if not len(p):
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64)).reshape(-1, 2)


def image_grid(frame_shape):
    """Fixed image probe points on the lower part of the frame, where the pitch is."""
    h, w = frame_shape[:2]
    return np.array([[x * w, y * h] for x in (.05, .3, .5, .7, .95) for y in (.4, .6, .8, .98)], np.float64)


def grid_on_pitch(H, frame_shape):
    """Pitch positions of the image probe grid, and which of them are on the ground plane.

    Image points above the horizon have homogeneous scale of the opposite sign
    to points on the pitch and must not enter any fit.
    """
    pts = image_grid(frame_shape)
    hom = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, np.float64).T
    ref = hom[-3, 2]   # bottom centre of the frame is always on the ground
    ok = (np.sign(hom[:, 2]) == np.sign(ref)) & (np.abs(hom[:, 2]) > 1e-9)
    xy = np.full((len(pts), 2), np.nan)
    xy[ok] = hom[ok, :2] / hom[ok, 2:]
    ok &= np.isfinite(xy).all(1) & (np.abs(xy) < 400).all(1)
    return xy, ok


def refit(pitch_xy, ok, frame_shape):
    """Image->pitch homography through the probe grid and target pitch positions."""
    if ok.sum() < 4:
        return None
    H, _ = cv2.findHomography(image_grid(frame_shape)[ok], pitch_xy[ok], 0)
    return None if H is None or abs(np.linalg.det(H)) < 1e-12 else H / H[2, 2]


def interpolate(Ha, Hb, fraction, frame_shape):
    """Blend two image->pitch homographies by interpolating where fixed image points land."""
    a, ok_a = grid_on_pitch(Ha, frame_shape)
    b, ok_b = grid_on_pitch(Hb, frame_shape)
    ok = ok_a & ok_b
    return refit(np.where(ok[:, None], a * (1 - fraction) + b * fraction, np.nan), ok, frame_shape)


def pitch_polylines(step=1.0):
    """Painted pitch markings as world polylines, for drawing calibration overlays."""
    L, W = PITCH_LENGTH, PITCH_WIDTH
    lines = [[(0, 0), (L, 0), (L, W), (0, W), (0, 0)], [(L / 2, 0), (L / 2, W)]]
    # The penalty arc is drawn only outside the penalty area.
    arc = np.linspace(-np.arccos(5.5 / 9.15), np.arccos(5.5 / 9.15), 24)
    for x0, s in ((0, 1), (L, -1)):
        lines.append([(x0, 13.84), (x0 + s * 16.5, 13.84), (x0 + s * 16.5, 54.16), (x0, 54.16)])
        lines.append([(x0, 24.84), (x0 + s * 5.5, 24.84), (x0 + s * 5.5, 43.16), (x0, 43.16)])
        lines.append([(x0 + s * (11 + 9.15 * np.cos(a)), 34 + 9.15 * np.sin(a)) for a in arc])
    t = np.linspace(0, 2 * np.pi, 72)
    lines.append([(L / 2 + 9.15 * np.cos(a), W / 2 + 9.15 * np.sin(a)) for a in t])
    return [np.array(line, np.float64) for line in lines if len(line) > 1]


def draw(frame, H, colour=(255, 200, 0), thickness=2):
    """Draw projected pitch markings onto a frame with an image->pitch homography."""
    inv = np.linalg.inv(np.asarray(H, np.float64))
    h, w = frame.shape[:2]
    for line in pitch_polylines():
        dense = np.concatenate([np.linspace(a, b, 12, endpoint=False) for a, b in zip(line[:-1], line[1:])] + [line[-1:]])
        img = to_pitch(dense, inv)
        ok = np.isfinite(img).all(1) & (np.abs(img[:, 0]) < 4 * w) & (np.abs(img[:, 1]) < 4 * h)
        for a, b, good in zip(img[:-1], img[1:], ok[:-1] & ok[1:]):
            if good:
                cv2.line(frame, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), colour, thickness)
    return frame
