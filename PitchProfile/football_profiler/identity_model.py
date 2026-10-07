"""Player identity model: shirt number and appearance from one fine-tuned CLIP ViT-B/16.

Trained by scripts/train_identity_models.py on player crops from FOOTPASS broadcast games, each
labelled with the player's true shirt number and team (SoccerNet NDA data, used for training and
measurement only). For a thumbnail it gives:
  - the shirt number, read as a tens digit (none, 1-9) and a units digit (0-9), so numbers that
    never appeared in training can still be read;
  - a 256-d appearance embedding trained to separate every player from the other players of the
    same game, team-mates included;
  - the probability that the player is a goalkeeper (their kit fits neither team's colours, so
    kit clustering alone often leaves them unassigned).
A segment's number combines its thumbnails' digit log-probabilities, each thumbnail weighted by
how sure it is, so back-turned or blurred views count for little (read_tracklet).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

BACKBONE = 'vit_base_patch16_clip_224.openai'
WEIGHTS = 'identity_vitb16/model.pt'
SIZE = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def weights_path() -> Path:
    from .football_models import weights_dir
    return weights_dir() / WEIGHTS


def available():
    return weights_path().is_file()


def digits(number):
    return (number // 10 if number >= 10 else 0), number % 10


def letterbox(img, size=SIZE):
    """Scale a BGR crop to fit size x size, padded with grey (players are tall and narrow)."""
    import cv2
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 114, np.uint8)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


def to_tensor(batch):
    """Letterboxed BGR uint8 crops -> normalised RGB float tensor (N, 3, SIZE, SIZE)."""
    import torch
    x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255.0
    x = x[:, [2, 1, 0]]
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def build(pretrained=True):
    import timm
    import torch
    from .football_models import weights_dir
    os.environ.setdefault('HF_HOME', str(weights_dir() / 'hf'))            # where the CLIP backbone weights are cached

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0)
            d = self.backbone.num_features
            self.tens = torch.nn.Linear(d, 10)
            self.units = torch.nn.Linear(d, 10)
            self.embed = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.GELU(), torch.nn.Linear(512, 256))
            self.keeper = torch.nn.Linear(d, 1)

        def forward(self, x):
            f = self.backbone(x)
            return (self.tens(f), self.units(f), torch.nn.functional.normalize(self.embed(f), dim=-1),
                    self.keeper(f).squeeze(-1))
    return Net()


@lru_cache(maxsize=1)
def model():
    import torch
    from .runtime import torch_device
    net = build(pretrained=False)
    net.load_state_dict(torch.load(weights_path(), map_location='cpu', weights_only=True))
    device = torch_device()
    return net.to(device).eval(), device


def predict(crops, batch=128):
    """Per crop: (tens log-probabilities, units log-probabilities, embedding, goalkeeper probability)."""
    import torch
    net, device = model()
    tens, units, embs, keeper = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = to_tensor([letterbox(c) for c in crops[i:i + batch]]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                lt, lu, z, lk = net(x)
            tens.append(torch.log_softmax(lt.float(), -1).cpu().numpy())
            units.append(torch.log_softmax(lu.float(), -1).cpu().numpy())
            embs.append(z.float().cpu().numpy())
            keeper.append(torch.sigmoid(lk.float()).cpu().numpy())
    if not tens:
        return np.zeros((0, 10)), np.zeros((0, 10)), np.zeros((0, 256)), np.zeros(0)
    return np.concatenate(tens), np.concatenate(units), np.concatenate(embs), np.concatenate(keeper)


def read_tracklet(lt, lu):
    """(number, confidence, evidence) for one segment's thumbnails.

    Each thumbnail's digit log-probabilities are weighted by its own certainty (the product of
    its best tens and units probabilities) and summed; confidence is the product of the combined
    tens and units probabilities of the chosen number; evidence is the summed certainty.
    """
    conf = np.exp(lt.max(1)) * np.exp(lu.max(1))
    w = conf / (conf.sum() + 1e-9)
    t = (lt * w[:, None]).sum(0)
    u = (lu * w[:, None]).sum(0)
    tp, up = np.exp(t - t.max()), np.exp(u - u.max())
    tp, up = tp / tp.sum(), up / up.sum()
    number = int(tp.argmax()) * 10 + int(up.argmax()) if tp.argmax() else int(up.argmax())
    return number, float(tp.max() * up.max()), float(conf.sum())


# A turned or blurred thumbnail still gives a (guessed) number, so only thumbnails the legibility
# classifier (football_profiler.jersey) finds readable vote. On FOOTPASS validation groups the
# legible thumbnails' vote picked the right number for 76% of grouped time against 54% for all
# thumbnails' summed digits (weights trained on every thumbnail); retrained with number labels on
# legible thumbnails only (train_identity_models.py --legible-only) the vote is right for 88%.
LEGIBLE = .5


def legibility(crops):
    """Per-thumbnail legibility probability (None when the jersey reader is not installed)."""
    from . import jersey as J
    if not crops or not J.available():
        return None
    net, _, device = J.models()
    return np.concatenate([J._legibility(crops[i:i + 256], net, device) for i in range(0, len(crops), 256)])


def top_numbers(lt, lu):
    """Each thumbnail's most likely shirt number (tens class 0 means a one-digit number)."""
    t, u = lt.argmax(1), lu.argmax(1)
    return np.where(t > 0, t * 10 + u, u)


def evidence(groups):
    """{key: [BGR thumbnails]} -> {key: evidence} to be summed over any set of keys (combine).

    Evidence holds each digit's log-probabilities weighted by every thumbnail's certainty, the
    total certainty, the summed embedding, the thumbnail count, the summed goalkeeper probability
    and the legible thumbnails' certainty-weighted votes for their numbers (100 bins; absent
    without the legibility classifier), so a segment or a whole player is read from all its
    thumbnails at once.
    """
    keys, crops, owner = list(groups), [], []
    for k in keys:
        crops += groups[k]
        owner += [k] * len(groups[k])
    lt, lu, emb, keeper = predict(crops)
    conf = np.exp(lt.max(1)) * np.exp(lu.max(1)) if len(lt) else np.zeros(0)
    legible = legibility(crops)
    top = top_numbers(lt, lu) if len(lt) else np.zeros(0, int)
    owner = np.array(owner, dtype=object)
    out = {}
    for k in keys:
        idx = np.flatnonzero(owner == k)
        if len(idx):
            out[k] = {'tens': (lt[idx] * conf[idx, None]).sum(0), 'units': (lu[idx] * conf[idx, None]).sum(0),
                      'weight': float(conf[idx].sum()), 'emb': emb[idx].sum(0), 'crops': int(len(idx)),
                      'keeper': float(keeper[idx].sum())}
            if legible is not None:
                m = idx[legible[idx] >= LEGIBLE]
                out[k]['votes'] = np.bincount(top[m], weights=conf[m], minlength=100)
                out[k]['legible'] = int(len(m))
    return out


def combine(items):
    """Sum several evidence dicts (None when there are none)."""
    items = [e for e in items if e]
    if not items:
        return None
    out = {'tens': sum(e['tens'] for e in items), 'units': sum(e['units'] for e in items),
           'weight': sum(e['weight'] for e in items), 'emb': sum(e['emb'] for e in items),
           'crops': sum(e['crops'] for e in items), 'keeper': sum(e['keeper'] for e in items)}
    if all('votes' in e for e in items):
        out['votes'] = sum(e['votes'] for e in items)
        out['legible'] = sum(e['legible'] for e in items)
    return out


def vote_reading(ev):
    """(number, share of the legible votes, summed votes) or (None, 0, 0) without legible thumbnails."""
    v = ev.get('votes') if ev else None
    if v is None or v.sum() <= 0:
        return None, 0.0, 0.0
    number = int(v.argmax())
    return number, float(v[number] / v.sum()), float(v[number])


def reading(ev):
    """(number, confidence, evidence weight) from combined evidence, as read_tracklet."""
    w = max(ev['weight'], 1e-9)
    t, u = ev['tens'] / w, ev['units'] / w
    tp, up = np.exp(t - t.max()), np.exp(u - u.max())
    tp, up = tp / tp.sum(), up / up.sum()
    number = int(tp.argmax()) * 10 + int(up.argmax()) if tp.argmax() else int(up.argmax())
    return number, float(tp.max() * up.max()), float(ev['weight'])


def probability(ev, number):
    """The combined evidence's probability of one shirt number (0-99)."""
    if number is None or not 0 <= number <= 99:
        return 0.0
    w = max(ev['weight'], 1e-9)
    t, u = ev['tens'] / w, ev['units'] / w
    tp, up = np.exp(t - t.max()), np.exp(u - u.max())
    return float(tp[number // 10] / tp.sum() * up[number % 10] / up.sum())


def embedding(ev):
    v = ev['emb'] / max(ev['crops'], 1)
    return v / (np.linalg.norm(v) + 1e-9)


def read_groups(groups):
    """{key: [BGR thumbnails]} -> ({key: reading}, {key: (unit embedding, thumbnails)})."""
    keys, crops, owner = list(groups), [], []
    for k in keys:
        crops += groups[k]
        owner += [k] * len(groups[k])
    lt, lu, emb, keeper = predict(crops)
    owner = np.array(owner, dtype=object)
    numbers, embeddings = {}, {}
    for k in keys:
        idx = np.flatnonzero(owner == k)
        if not len(idx):
            continue
        number, confidence, evidence = read_tracklet(lt[idx], lu[idx])
        numbers[k] = {'number': number, 'confidence': confidence, 'votes': evidence, 'crops': int(len(idx)),
                      'keeper_probability': float(keeper[idx].mean()), 'reader': 'identity_model'}
        v = emb[idx].mean(0)
        embeddings[k] = (v / (np.linalg.norm(v) + 1e-9), int(len(idx)))
    return numbers, embeddings
