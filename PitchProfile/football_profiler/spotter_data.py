"""Labelled clips for fine-tuning and measuring the action spotter (see action_spotting).

Reads the frame files written by scripts/pack_spotter_frames.py: SoccerNet Ball Action Spotting
games (the spotter's own 12 classes and team side) and FOOTPASS games (8 player-centric classes).
FOOTPASS labels map onto the spotter's classes as sets of allowed classes, because FOOTPASS does
not separate high passes or free kicks and does not annotate outs or goals: a FOOTPASS pass may be
a BAS pass, high pass or free kick, and an unlabelled FOOTPASS frame may be background, an out or
a goal. Training uses a partial-label loss over these sets (scripts/finetune_action_spotter.py).

Team side: 'left' is the team defending the left goal (as in action_spotting.attribute; on this
project's analyses the spotter's side matched the team in possession for 96% of confident passes
and carries). In FOOTPASS that team has left_to_right == 0: its goalkeeper stands at x of about
0.1 (0 = left goal line), and the published spotter's side agreed with this reading for 83-96% of
matched passes, carries and crosses (scripts/evaluate_spotter_labelled.py).

These games are training and measurement data only; they are never analysed as matches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import action_spotting as AS
from . import soccernet as SN

NUM_CLASSES = len(AS.CLASSES) + 1                     # background + the spotter's 12 classes
INDEX = {c: i + 1 for i, c in enumerate(AS.CLASSES)}
RADIUS = 4                                            # label radius in frames at 12.5/s (devkit radi_displacement)
FOOTPASS_ALLOWED = {
    'Drive': ['DRIVE'],
    'Pass': ['PASS', 'HIGH PASS', 'FREE KICK'],
    'Cross': ['CROSS', 'FREE KICK'],
    'Shot': ['SHOT', 'FREE KICK'],
    'Header': ['HEADER', 'SHOT'],
    'Throw-in': ['THROW IN'],
    'Tackle': ['PLAYER SUCCESSFUL TACKLE'],
    'Block': ['BALL PLAYER BLOCK'],
}
FOOTPASS_UNANNOTATED = ['OUT', 'GOAL']
FOOTPASS_LTR_IS_LEFT = False           # left_to_right == 0 is the team defending the left goal
# The spotter classes each FOOTPASS class is measured against.
FOOTPASS_MEASURED = {'Drive': ['DRIVE'], 'Pass': ['PASS', 'HIGH PASS'], 'Cross': ['CROSS'], 'Shot': ['SHOT'],
                     'Header': ['HEADER'], 'Throw-in': ['THROW IN'], 'Tackle': ['PLAYER SUCCESSFUL TACKLE'],
                     'Block': ['BALL PLAYER BLOCK']}


def root() -> Path:
    return SN.root() / 'spotter_frames'


def mask(names):
    m = 0
    for n in names:
        m |= 1 << (0 if n == 'background' else INDEX[n])
    return m


def allowed(event, dataset):
    """Bit mask of spotter classes (bit 0 = background) this labelled event may be."""
    if dataset == 'bas':
        return mask([event['label']])
    return mask(FOOTPASS_ALLOWED[event['label']])


def side(event, dataset):
    if dataset == 'bas':
        return event.get('side')
    if 'left_to_right' not in event:
        return None
    return 'left' if bool(event['left_to_right']) == FOOTPASS_LTR_IS_LEFT else 'right'


def targets(events, n, dataset, radius=RADIUS):
    """Per-frame targets: allowed-class bit mask, displacement to the event, team (-1 none, 0 left, 1 right).

    As in the devkit, frames within `radius` of an event carry its label and their offset from it;
    where windows overlap the later event wins.
    """
    background = mask(['background'] + (FOOTPASS_UNANNOTATED if dataset == 'footpass' else []))
    allow = np.full(n, background, np.int32)
    displ = np.zeros(n, np.float32)
    team = np.full(n, -1, np.int8)
    for e in sorted(events, key=lambda e: e['frame']):
        a, s = allowed(e, dataset), side(e, dataset)
        for i in range(max(0, e['frame'] - radius), min(n, e['frame'] + radius + 1)):
            allow[i] = a
            displ[i] = i - e['frame']
            team[i] = -1 if s is None else int(s == 'right')
    return allow, displ, team


def expand(allow):
    """Bit masks (any shape) -> boolean array with a trailing class axis."""
    bits = 1 << np.arange(NUM_CLASSES)
    return (np.asarray(allow)[..., None] & bits) > 0


def partial_label_loss(logits, allow_bits, weights):
    """Per-frame -log(sum of the allowed classes' probabilities), weighted by the heaviest allowed class.

    With one allowed class this is the devkit's weighted cross-entropy; frames whose allowed set
    includes background (unlabelled frames) keep weight 1.
    """
    import torch
    bits = torch.as_tensor(1 << np.arange(NUM_CLASSES), device=logits.device)
    allowed = (allow_bits.unsqueeze(-1) & bits) > 0
    logp = torch.log_softmax(logits.float(), dim=-1)
    nll = -torch.logsumexp(logp.masked_fill(~allowed, float('-inf')), dim=-1)
    w = (weights * allowed).amax(dim=-1)
    w = torch.where(allowed[..., 0], torch.ones_like(w), w)
    return w * nll


class PackedVideo:
    """One packed video: JPEG frames at 12.5/s with an offset index, its labels and targets."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / 'meta.json').read_text(encoding='utf-8'))
        self.dataset = self.meta['dataset']
        self.name = self.folder.name
        self.n = int(self.meta['frames'])
        self.index = np.load(self.folder / 'index.npy')
        self.events = json.loads((self.folder / 'labels.json').read_text(encoding='utf-8'))
        self.allow, self.displ, self.team = targets(self.events, self.n, self.dataset)

    def frames(self, start, length, pool=None):
        """RGB uint8 frames (length x 448 x 796 x 3); frames past either end are black.

        `pool` (a thread pool) decodes the JPEGs in parallel; OpenCV releases the GIL.
        """
        import cv2
        out = np.zeros((length, AS.HEIGHT, AS.WIDTH, 3), np.uint8)
        lo, hi = max(0, start), min(self.n, start + length)
        if hi <= lo:
            return out
        with (self.folder / 'frames.bin').open('rb') as f:
            f.seek(int(self.index[lo]))
            blob = f.read(int(self.index[hi] - self.index[lo]))
        base = int(self.index[lo])

        def decode(k):
            buf = np.frombuffer(blob, np.uint8, int(self.index[k + 1] - self.index[k]), int(self.index[k]) - base)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img.shape[0] != AS.HEIGHT or img.shape[1] != AS.WIDTH:
                img = cv2.resize(img, (AS.WIDTH, AS.HEIGHT), interpolation=cv2.INTER_LINEAR)
            out[k - start] = img[:, :, ::-1]

        list((pool.map if pool is not None else map)(decode, range(lo, hi)))
        return out

    def iter_frames(self, chunk=250, threads=8):
        """(index, RGB frame) for the whole video, as action_spotting.frame_scores expects."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as pool:
            yield from self._iter(chunk, pool)

    def _iter(self, chunk, pool):
        for s in range(0, self.n, chunk):
            block = self.frames(s, min(chunk, self.n - s), pool)
            for k in range(len(block)):
                yield s + k, block[k]

    def clip_targets(self, start, length):
        lo, hi = max(0, start), min(self.n, start + length)
        allow = np.full(length, mask(['background']), np.int32)
        displ = np.zeros(length, np.float32)
        team = np.full(length, -1, np.int8)
        if hi > lo:
            allow[lo - start:hi - start] = self.allow[lo:hi]
            displ[lo - start:hi - start] = self.displ[lo:hi]
            team[lo - start:hi - start] = self.team[lo:hi]
        return allow, displ, team


def videos(dataset, split):
    base = root() / dataset / split
    return [PackedVideo(p.parent) for p in sorted(base.glob('*/meta.json'))]
