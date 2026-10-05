"""Jersey-number reading for player tracklets.

Two published SoccerNet models from Koshkina & Elder's jersey-number pipeline
(CC BY-NC 3.0, see THIRD_PARTY_NOTICES.md), loaded from the data-drive weights:
a ResNet-34 legibility classifier picks crops whose number is readable, then a
SoccerNet fine-tuned PARSeq scene-text recogniser reads the torso. Readings of
all legible crops of a tracklet are combined by confidence-weighted vote. The
pose model of the original pipeline is replaced by a fixed torso band of the
player box (scripts/evaluate_jersey_reader.py measures the effect).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

LEGIBILITY_WEIGHTS = 'legibility_resnet34_soccer.pth'
PARSEQ_WEIGHTS = 'parseq_soccernet_jersey.ckpt'
LEGIBILITY_SIZE = (256, 256)
LEGIBLE = .5
TORSO = (.15, .55)          # vertical band of the player box read by PARSeq
TORSO_X = (.12, .88)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def weights_paths():
    from .football_models import weights_dir
    return weights_dir() / LEGIBILITY_WEIGHTS, weights_dir() / PARSEQ_WEIGHTS


def available():
    try:
        import strhub  # noqa: F401
    except ImportError:
        return False
    return all(p.is_file() for p in weights_paths())


@lru_cache(maxsize=1)
def models():
    import torch
    from torchvision import models as tv
    from strhub.models.parseq.system import PARSeq
    from .runtime import torch_device
    if not available():
        raise ValueError('Jersey reader unavailable: install requirements-jersey.txt and run '
                         'scripts/fetch_football_models.py to download its weights.')
    device = torch_device()
    legibility_path, parseq_path = weights_paths()
    legibility = tv.resnet34()
    legibility.fc = torch.nn.Linear(512, 1)
    state = torch.load(legibility_path, map_location='cpu', weights_only=True)
    legibility.load_state_dict({k.replace('model_ft.', ''): v for k, v in state.items()})
    ckpt = torch.load(parseq_path, map_location='cpu', weights_only=False)
    reader = PARSeq(**ckpt['hyper_parameters'])
    # The checkpoint predates strhub's model wrapper; its keys lack the 'model.' prefix.
    reader.load_state_dict({(k if k.startswith('model.') else 'model.' + k): v for k, v in ckpt['state_dict'].items()})
    legibility = legibility.to(device).eval()
    reader = reader.to(device).eval()
    if device.type == 'cuda':
        legibility.half()
    return legibility, reader, device


def _legibility(crops, net, device):
    import cv2
    import torch
    x = np.stack([cv2.resize(c[:, :, ::-1], LEGIBILITY_SIZE, interpolation=cv2.INTER_AREA) for c in crops])
    x = torch.from_numpy(((x.astype(np.float32) / 255 - MEAN) / STD)).permute(0, 3, 1, 2).to(device)
    with torch.inference_mode():
        return torch.sigmoid(net(x.half() if device.type == 'cuda' else x).float()).squeeze(-1).cpu().numpy()


def torso(crop):
    h, w = crop.shape[:2]
    return crop[int(TORSO[0] * h):max(int(TORSO[1] * h), int(TORSO[0] * h) + 2),
                int(TORSO_X[0] * w):max(int(TORSO_X[1] * w), int(TORSO_X[0] * w) + 2)]


def _read(crops, reader, device):
    import cv2
    import torch
    h, w = reader.hparams.img_size
    x = np.stack([cv2.resize(torso(c)[:, :, ::-1], (w, h), interpolation=cv2.INTER_CUBIC) for c in crops])
    x = torch.from_numpy((x.astype(np.float32) / 255 - .5) / .5).permute(0, 3, 1, 2).to(device)
    with torch.inference_mode():
        probs = reader(x).softmax(-1)
    text, conf = reader.tokenizer.decode(probs)
    return text, [float(c.prod()) for c in conf]


def valid_number(text):
    return text.isdigit() and 1 <= len(text) <= 2 and not (len(text) == 2 and text[0] == '0')


def read_crops(crops):
    """Per thumbnail: (legibility, PARSeq text, PARSeq confidence); only legible ones are read."""
    legibility, reader, device = models()
    scores = np.concatenate([_legibility(crops[i:i + 256], legibility, device) for i in range(0, len(crops), 256)]) \
        if crops else np.zeros(0, np.float32)
    text = np.full(len(crops), '', dtype=object)
    conf = np.zeros(len(crops), np.float32)
    legible = np.flatnonzero(scores >= LEGIBLE)
    for i in range(0, len(legible), 256):
        t, c = _read([crops[j] for j in legible[i:i + 256]], reader, device)
        text[legible[i:i + 256]] = t
        conf[legible[i:i + 256]] = c
    return scores, text, conf


def aggregate(scores, text, conf):
    """One group's reading from its thumbnails' read_crops values (as read_groups)."""
    legible = scores >= LEGIBLE
    total, votes = float(conf[legible].sum()), {}
    for t, c in zip(text[legible], conf[legible]):
        if valid_number(t):
            votes[int(t)] = votes.get(int(t), 0.0) + float(c)
    out = {'legible_crops': int(legible.sum()), 'crops': int(len(scores))}
    if not votes:
        return {'number': None, 'confidence': 0.0, 'votes': 0.0, **out}
    number = max(votes, key=votes.get)
    return {'number': number, 'confidence': votes[number] / max(total, 1e-9), 'votes': votes[number], **out}


def read_groups(groups):
    """Number per group of BGR player crops (e.g. all crops of one tracklet).

    Returns key -> {'number', 'confidence', 'votes', 'legible_crops', 'crops'}:
    number is None without a valid legible reading; confidence is the winning
    number's share of all legible reading confidence.
    """
    legibility, reader, device = models()
    keys = [k for k, crops in groups.items() if crops]
    flat, owner = [], []
    for k in keys:
        flat += groups[k]
        owner += [k] * len(groups[k])
    scores = np.concatenate([_legibility(flat[i:i + 256], legibility, device) for i in range(0, len(flat), 256)]) \
        if flat else np.array([])
    legible = np.flatnonzero(scores >= LEGIBLE)
    texts, confs = [], []
    for i in range(0, len(legible), 256):
        t, c = _read([flat[j] for j in legible[i:i + 256]], reader, device)
        texts += t
        confs += c
    votes, totals = {}, {}
    for j, text, conf in zip(legible, texts, confs):
        k = owner[j]
        totals[k] = totals.get(k, 0.0) + conf
        if valid_number(text):
            votes.setdefault(k, {}).setdefault(int(text), 0.0)
            votes[k][int(text)] += conf
    out = {}
    counts = {k: 0 for k in keys}
    for j in legible:
        counts[owner[j]] += 1
    for k in keys:
        v = votes.get(k, {})
        if v:
            number = max(v, key=v.get)
            out[k] = {'number': number, 'confidence': v[number] / max(totals[k], 1e-9), 'votes': v[number],
                      'legible_crops': counts[k], 'crops': len(groups[k])}
        else:
            out[k] = {'number': None, 'confidence': 0.0, 'votes': 0.0, 'legible_crops': counts[k], 'crops': len(groups[k])}
    return out
