"""Ball-action spotting on broadcast video with the SoccerNet 2025 team ball-action baseline.

The model (T-DEED, Xarles et al. 2024; team head from SoccerNet's sn-teamspotting devkit, GPL-3.0)
finds the moment of 12 ball actions (pass, drive, header, high pass, out, cross, throw-in, shot,
block, successful tackle, free kick, goal) in 8-second clips of 796x448 frames at 12.5 frames/s,
and says whether the team on the left or right of the pitch performed each. Tackles, blocks and
headers are what tracking-based rules cannot see (see match_events), so these spots are credited
to the player of that team who is at the ball at that moment (attribute()).

The devkit is an external checkout on the data drive (third_party/sn-teamspotting), imported at
runtime and not copied into PitchProfile; the checkpoint is the devkit's published baseline.
"""
from __future__ import annotations

import os
import sys
import types
from argparse import Namespace
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import storage as S

CLASSES = ['PASS', 'DRIVE', 'HEADER', 'HIGH PASS', 'OUT', 'CROSS', 'THROW IN', 'SHOT', 'BALL PLAYER BLOCK',
           'PLAYER SUCCESSFUL TACKLE', 'FREE KICK', 'GOAL']
DEVKIT_COMMIT = '091fed2fc35c33f7489f3596958a2fe385e37d65'
PUBLISHED_CHECKPOINT = 'tdeed_team_bas/checkpoint_best.pt'           # devkit baseline (fetch_football_models.py)
PUBLISHED_SHA256 = 'ed7c558d575eba1918be8f67813571723962eeff807dba68a12a2d794dd1c72a'
# Fine-tuned from it on SoccerNet Ball Action Spotting + FOOTPASS (scripts/finetune_action_spotter.py).
# On held-out games it raised mean average precision from 0.61 to 0.63 (BAS test) and 0.48 to 0.60
# (FOOTPASS validation), blocks from 0.18 to 0.34 and headers from 0.64 to 0.76; on this project's
# halves it counts shots, passes and carries far closer to the labelled numbers
# (DETECTION_RESEARCH.md, section 5). Used when present, otherwise the published checkpoint.
CHECKPOINT = 'tdeed_team_bas_ft/checkpoint_best.pt'
CHECKPOINT_SHA256 = '7d228ba697668ccdac711d8d2b52ac4d62a75f08e953e01231973e4ac1b31944'
WIDTH, HEIGHT = 796, 448
CLIP_LEN = 100
CLIP_STEP = 25                 # the devkit's evaluation overlap (three quarters of a clip)
FRAME_STRIDE = 2               # 25 fps source -> 12.5 frames/s, the stride the model was trained with
NMS_WINDOW = 12                # frames (about 1 s), the devkit's soft-NMS window for ball actions
KEEP_SCORE = .05               # spots stored and counted (score-weighted) in statistics
SCORE_THRESHOLD = .2           # confident-spot threshold for classes without a measured one
BATCH = 4


def devkit_dir() -> Path:
    return Path(os.environ.get('PITCHPROFILE_TEAMSPOTTING') or S.DATA.parent.parent / 'third_party' / 'sn-teamspotting')


def published_checkpoint_path() -> Path:
    from .football_models import weights_dir
    return weights_dir() / PUBLISHED_CHECKPOINT


def checkpoint_path() -> Path:
    from .football_models import weights_dir
    tuned = weights_dir() / CHECKPOINT
    return tuned if tuned.is_file() else published_checkpoint_path()


def available():
    return checkpoint_path().is_file() and (devkit_dir() / 'model' / 'model.py').is_file()


def _import_devkit():
    """Import the devkit's `model` package under a private name (NBJW also ships a `model` package)."""
    saved = {k: v for k, v in sys.modules.items() if k == 'model' or k.startswith('model.')}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(devkit_dir()))
    try:
        import timm
        create = timm.create_model
        # The backbone's ImageNet weights are replaced by the checkpoint; do not download them.
        timm.create_model = lambda *a, **kw: create(*a, **{**kw, 'pretrained': False})
        try:
            from model.model import TDEEDModel
            from model.modules import process_double_head, process_predictionTeam
        finally:
            timm.create_model = create
        loaded = {k: v for k, v in sys.modules.items() if k == 'model' or k.startswith('model.')}
    finally:
        sys.path.remove(str(devkit_dir()))
        for k in [k for k in sys.modules if k == 'model' or k.startswith('model.')]:
            del sys.modules[k]
        sys.modules.update(saved)
    for k, v in loaded.items():
        sys.modules['sn_teamspotting_' + k] = v
    return types.SimpleNamespace(TDEEDModel=TDEEDModel, process_double_head=process_double_head,
                                 process_predictionTeam=process_predictionTeam)


@lru_cache(maxsize=1)
def model():
    import torch
    from .runtime import torch_device
    if not available():
        raise ValueError('Action spotter unavailable: run scripts/fetch_football_models.py (checkpoint and '
                         'sn-teamspotting devkit).')
    kit = _import_devkit()
    args = Namespace(modality='rgb', temporal_arch='ed_sgp_mixer', radi_displacement=4, feature_arch='rny002_gsf',
                     event_team=True, clip_len=CLIP_LEN, n_layers=2, sgp_ks=9, sgp_r=4, num_classes=len(CLASSES),
                     crop_dim=None)
    net = kit.TDEEDModel.Impl(args=args)
    net.update_pred_head([len(CLASSES) + 1, 18])          # joint-training head of the published checkpoint
    state = torch.load(checkpoint_path(), map_location='cpu', weights_only=True)
    net.load_state_dict(state)
    device = torch_device()
    return net.to(device).eval(), kit, device


def _frames(video_path, progress, max_frames=None):
    """(index, RGB 448x796 frame) at 12.5 frames/s, read sequentially."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // FRAME_STRIDE
    i = src = 0
    try:
        while True:
            if src % FRAME_STRIDE:
                if not cap.grab():
                    break
                src += 1
                continue
            ok, frame = cap.read()
            if not ok or (max_frames is not None and i >= max_frames):
                break
            src += 1
            yield i, cv2.cvtColor(cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            i += 1
            if i % 2500 == 0:
                progress(min(.95, i / max(total, 1)), f'Action spotting {i}/{total} frames')
    finally:
        cap.release()


def double_head(pred, displ, num_classes):
    """The devkit's process_double_head without its per-frame loop (same result, one scatter).

    Softmax over the ball-action head, then each frame's probabilities move to the frame its
    displacement points at (t - round(displacement), clipped to the clip), keeping the maximum.
    """
    import torch
    probs = torch.softmax(pred[:, :, :num_classes], dim=2)
    t = torch.arange(probs.shape[1], device=probs.device)
    target = (t - displ.round().int()).clamp(0, probs.shape[1] - 1).long()
    out = torch.zeros_like(probs)
    return out.scatter_reduce(1, target.unsqueeze(-1).expand_as(probs), probs, reduce='amax', include_self=True)


def frame_scores(video_path, progress=lambda *a: None, max_frames=None, frames=None):
    """Per-frame scores (frames x 25): background, then each class for the left and right team.

    `frames` replaces reading `video_path`: an iterable of (index, RGB 448x796 frame) at 12.5/s.
    """
    import torch
    net, kit, device = model()
    buf = {}
    scores, support = [], []

    def run(batch_starts):
        clips = np.stack([np.stack([buf[s + k] for k in range(CLIP_LEN)]) for s in batch_starts])
        x = torch.from_numpy(clips).to(device).permute(0, 1, 4, 2, 3).float()
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == 'cuda'):
            out, _ = net(x, inference=True)
        pred = double_head(out['im_feat'].float(), out['displ_feat'].float(), num_classes=len(CLASSES) + 1)
        pred = kit.process_predictionTeam(pred.cpu(), out['team_feat'].float().cpu()).numpy()
        for s, p in zip(batch_starts, pred):
            need = s + CLIP_LEN
            while len(scores) < need:
                scores.append(np.zeros(pred.shape[-1], np.float32))
                support.append(0)
            for k in range(CLIP_LEN):
                scores[s + k] += p[k]
                support[s + k] += 1

    pending, next_start, n = [], 0, 0
    for i, frame in (frames if frames is not None else _frames(video_path, progress, max_frames)):
        buf[i] = frame
        n = i + 1
        while next_start + CLIP_LEN <= n:
            pending.append(next_start)
            next_start += CLIP_STEP
            if len(pending) == BATCH:
                run(pending)
                pending = []
                # Keep one clip of margin: the final clip is aligned to the end of the video.
                for k in [k for k in buf if k < next_start - CLIP_LEN]:
                    del buf[k]
    last_start = next_start - CLIP_STEP
    if n >= CLIP_LEN and last_start + CLIP_LEN < n:
        pending.append(n - CLIP_LEN)
    if pending:
        run(pending)
    s = np.array(scores[:n]) / np.maximum(np.array(support[:n]), 1)[:, None]
    return s


def peaks(scores, fps, threshold=KEEP_SCORE):
    """Soft-NMS per class (both teams together), as in the devkit's evaluation."""
    out = []
    for c, label in enumerate(CLASSES):
        cand = []
        for side, col in (('left', 1 + 2 * c), ('right', 2 + 2 * c)):
            idx = np.flatnonzero(scores[:, col] >= .01)
            cand += [[int(i), float(scores[i, col]), side] for i in idx]
        while cand:
            best = max(range(len(cand)), key=lambda k: cand[k][1])
            frame, score, side = cand.pop(best)
            if score < threshold:
                break
            out.append({'label': label, 'side': side, 'frame': frame, 'time_s': frame / fps, 'score': round(score, 4)})
            for e in cand:
                d = abs(e[0] - frame)
                if d <= NMS_WINDOW:
                    e[1] *= d ** 2 / NMS_WINDOW ** 2
    return sorted(out, key=lambda e: e['frame'])


def spot_video(video_path, progress=lambda *a: None, max_frames=None):
    from .vision import video_info
    fps = video_info(video_path)['fps'] / FRAME_STRIDE
    scores = frame_scores(video_path, progress, max_frames)
    tuned = checkpoint_path().parent.name == Path(CHECKPOINT).parent.name
    return {'model': 'T-DEED team ball-action spotter (SoccerNet sn-teamspotting)'
                     + (', fine-tuned on BAS + FOOTPASS' if tuned else ', published baseline'),
            'checkpoint': checkpoint_path().parent.name + '/' + checkpoint_path().name, 'devkit_commit': DEVKIT_COMMIT,
            'frames_per_second': fps, 'frames': int(len(scores)), 'score_threshold_stored': KEEP_SCORE,
            'spots': peaks(scores, fps)}


# Actions credited to a player of the spotted team, and the statistic they count toward.
PLAYER_ACTIONS = {'PLAYER SUCCESSFUL TACKLE': 'tackle', 'BALL PLAYER BLOCK': 'block', 'HEADER': 'header',
                  'CROSS': 'cross', 'SHOT': 'shot', 'DRIVE': 'drive', 'PASS': 'pass', 'HIGH PASS': 'high_pass',
                  'THROW IN': 'throw_in', 'FREE KICK': 'free_kick'}
ATTRIBUTE_RADIUS_M = 3.0
ATTRIBUTE_WINDOW = 2           # samples either side of the spot
# Tackles and blocks win the ball from the other team: in both labelled datasets (SoccerNet BAS,
# FOOTPASS) the tackler's or blocker's team differs from the team of the last pass or drive before
# it in 99-100% of cases, whereas the published spotter's own team output for them is near chance.
# With WON_BALL_SIDE = 'previous_touch' their side is the opposite of the last confident on-ball
# spot within TOUCH_WINDOW_S before (the spotter's side when there is none). On held-out games
# (scripts/evaluate_spotter_labelled.py) this gave blocks the right team 81% (BAS test) and 91%
# (FOOTPASS validation) of the time, against 48% and 31% from the spotter's team output.
WON_BALL = ('PLAYER SUCCESSFUL TACKLE', 'BALL PLAYER BLOCK')
WON_BALL_SIDE = 'previous_touch'
TOUCH_LABELS = ('PASS', 'DRIVE', 'HIGH PASS', 'CROSS', 'SHOT', 'HEADER', 'THROW IN', 'FREE KICK')
TOUCH_SCORE = .3
TOUCH_WINDOW_S = 4.0


def won_ball_side(spots, spot):
    """Side of a tackle or block from the last confident touch before it (None if there is none)."""
    before = [s for s in spots if s['label'] in TOUCH_LABELS and s['score'] >= TOUCH_SCORE
              and 0 < spot['time_s'] - s['time_s'] <= TOUCH_WINDOW_S]
    if not before:
        return None
    return {'left': 'right', 'right': 'left'}[max(before, key=lambda s: s['time_s'])['side']]


def thresholds():
    """Per-class score above which a spot is shown as a confident event (scripts/evaluate_action_spotter.py)."""
    measured = (S.read_json(S.EVIDENCE / 'action_spotter.json', {}) or {}).get('thresholds', {})
    return {c: float(measured.get(c, SCORE_THRESHOLD)) for c in CLASSES}


# Crediting in the picture first. A header or throw-in is played with the ball in the air, so its
# position projected onto the pitch can be metres from the player; the ball's position in the
# picture, inside the player's box (for aerial actions up to above his head), is what shows who
# played it. The nearest player on the pitch within ATTRIBUTE_RADIUS_M is the fallback.
AERIAL = ('HEADER', 'THROW IN')
IMAGE_PAD_X = .5                  # of the box width, either side
IMAGE_ABOVE = {True: .6, False: .1}   # of the box height above its top (aerial / on the ground)
IMAGE_BELOW = .15


def _image_pick(players, cx, cy, aerial):
    """(distance in box heights, player row) of the player whose box holds the ball, or (inf, None)."""
    x, y = players.bbox_x.to_numpy(float), players.bbox_y.to_numpy(float)
    w, h = players.bbox_w.to_numpy(float), players.bbox_h.to_numpy(float)
    inside = ((cx >= x - IMAGE_PAD_X * w) & (cx <= x + w + IMAGE_PAD_X * w) &
              (cy >= y - IMAGE_ABOVE[aerial] * h) & (cy <= y + h + IMAGE_BELOW * h))
    if not inside.any():
        return np.inf, None
    ax, ay = x + w / 2, (y if aerial else y + h)                      # head or feet
    d = np.where(inside, np.hypot(cx - ax, cy - ay) / np.maximum(h, 1), np.inf)
    j = int(np.argmin(d))
    return float(d[j]), players.iloc[j]


def attribute(spots, people, ball, directions, sample_hz, threshold=KEEP_SCORE):
    """Credit spotted actions to the player of the spotted team who played the ball.

    'left' is the team defending the left goal. The player is the one of that team whose picture
    box holds the ball's picture position within ATTRIBUTE_WINDOW samples of the spot (head or
    feet nearest, see _image_pick), otherwise the one nearest the ball on the pitch within
    ATTRIBUTE_RADIUS_M; with neither the spot stays team-level (segment None). Each event keeps its
    score; 'confident' marks scores above the class threshold (thresholds()). Statistics add up
    scores (expected counts): on this footage the spotter's scores behaved as probabilities
    (summed throw-in scores 166 against 171 labelled throw-ins).
    """
    confident = thresholds()
    import pandas as pd
    from .match_events import attack_sign
    if directions.get('status') not in ('inferred', 'uncertain'):
        side_team = {}
    else:
        left = directions['defends_left']
        side_team = {'left': left, 'right': 'B' if left == 'A' else 'A'}
    q = people[people.team.isin(['A', 'B']) & people.role.isin(['player', 'goalkeeper'])]
    by_sample = {s: g for s, g in q.groupby('sample')}
    b = ball.dropna(subset=['x', 'y']).set_index('sample') if len(ball) else pd.DataFrame(columns=['x', 'y'])
    seen = ball[~ball.interpolated.astype(bool)].dropna(subset=['cx', 'cy']).set_index('sample') \
        if len(ball) and 'cx' in ball else pd.DataFrame(columns=['cx', 'cy'])
    out = []
    for s in spots:
        if s['score'] < threshold or s['label'] not in PLAYER_ACTIONS:
            continue
        side = s['side']
        if WON_BALL_SIDE == 'previous_touch' and s['label'] in WON_BALL:
            side = won_ball_side(spots, s) or side
        team = side_team.get(side)
        sample = int(round(s['time_s'] * sample_hz))
        aerial = s['label'] in AERIAL
        best, picked, how = (np.inf, None, None), (np.inf, None), None
        for k in range(sample - ATTRIBUTE_WINDOW, sample + ATTRIBUTE_WINDOW + 1):
            g = by_sample.get(k)
            if g is None or team is None:
                continue
            mine = g[g.team == team]
            if mine.empty:
                continue
            if k in seen.index:
                r = seen.loc[k]
                r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
                d, pl = _image_pick(mine, float(r.cx), float(r.cy), aerial)
                if d < picked[0]:
                    picked = (d, pl)
            row = b.loc[k] if k in b.index else None
            if row is None:
                continue
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            on_pitch = mine[np.isfinite(mine.x)]
            if on_pitch.empty:
                continue
            d = np.hypot(on_pitch.x.to_numpy() - row.x, on_pitch.y.to_numpy() - row.y)
            j = int(np.argmin(d))
            if d[j] < best[0]:
                best = (float(d[j]), on_pitch.iloc[j], row)
        dist, player, row = best
        if picked[1] is not None:
            player, how = picked[1], 'picture'
        elif player is not None and dist <= ATTRIBUTE_RADIUS_M:
            how = 'pitch'
        else:
            player = None
        x = float(row.x) if row is not None else (float(player.x) if player is not None and np.isfinite(player.x) else None)
        y = float(row.y) if row is not None else (float(player.y) if player is not None and np.isfinite(player.y) else None)
        out.append({'type': PLAYER_ACTIONS[s['label']], 'source': 'action_spotter', 'label': s['label'],
                    'score': s['score'], 'confident': bool(s['score'] >= confident[s['label']]),
                    'team': team, 'time_s': float(s['time_s']),
                    'segment': player.segment if player is not None else None,
                    'view_shot': int(player.view_shot) if player is not None else None,
                    'credited_by': how, 'x': x, 'y': y,
                    'ball_distance_m': dist if np.isfinite(dist) else None,
                    'attack_sign': attack_sign(team, directions) if team else 0})
    return out
