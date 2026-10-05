"""Locate and lazily load the football-specific detection and calibration models.

Weights live on the data drive, next to the prepared data, not inside the
project. NBJW (GPL-2.0) is used as an external checkout imported at runtime;
its code is not copied into PitchProfile.
"""
from __future__ import annotations

import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

from . import storage as S

PLAYER_WEIGHTS = 'football-player-detection.pt'
BALL_WEIGHTS = 'football-ball-detection.pt'
NBJW_KEYPOINT_WEIGHTS = 'SV_kp'
# Class ids of the football player detector (Roboflow sports, YOLOv8x).
BALL, GOALKEEPER, PLAYER, REFEREE = 0, 1, 2, 3
PEOPLE = (GOALKEEPER, PLAYER, REFEREE)
ROLE_NAMES = {GOALKEEPER: 'goalkeeper', PLAYER: 'player', REFEREE: 'referee'}
_lock = threading.Lock()


def weights_dir() -> Path:
    return Path(os.environ.get('PITCHPROFILE_WEIGHTS') or S.DATA.parent / 'weights')


def nbjw_dir() -> Path:
    return Path(os.environ.get('PITCHPROFILE_NBJW') or
                S.DATA.parent.parent / 'third_party' / 'No-Bells-Just-Whistles')


def availability():
    w, nb = weights_dir(), nbjw_dir()
    files = {'player_detector': w / PLAYER_WEIGHTS, 'ball_detector': w / BALL_WEIGHTS,
             'pitch_keypoints': nb / 'weights' / NBJW_KEYPOINT_WEIGHTS,
             'pitch_keypoint_code': nb / 'model' / 'cls_hrnet.py'}
    return {name: path.is_file() for name, path in files.items()}


def require():
    missing = [name for name, ok in availability().items() if not ok]
    if missing:
        raise ValueError('Football models are missing (' + ', '.join(missing) + '). Run '
                         'scripts/fetch_football_models.py to download them to the data drive.')


def _yolo(path: Path):
    os.environ.setdefault('YOLO_CONFIG_DIR', str(S.DATA / 'ultralytics_config'))
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    from ultralytics import YOLO
    return YOLO(str(path))


@lru_cache(maxsize=1)
def player_detector():
    require()
    with _lock:
        return _yolo(weights_dir() / PLAYER_WEIGHTS)


@lru_cache(maxsize=1)
def ball_detector():
    require()
    with _lock:
        return _yolo(weights_dir() / BALL_WEIGHTS)


@lru_cache(maxsize=1)
def pitch_keypoint_model():
    """HRNetV2-W48 keypoint network from NBJW, in half precision on the GPU."""
    import torch
    import yaml
    from .runtime import torch_device
    require()
    root = nbjw_dir()
    with _lock:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from model.cls_hrnet import get_cls_net
        device = torch_device()
        cfg = yaml.safe_load((root / 'config' / 'hrnetv2_w48.yaml').read_text(encoding='utf-8'))
        net = get_cls_net(cfg)
        state = torch.load(root / 'weights' / NBJW_KEYPOINT_WEIGHTS, map_location=device, weights_only=True)
        net.load_state_dict(state)
        net.to(device).eval()
        if device.type == 'cuda':
            net.half()
        return net, device
