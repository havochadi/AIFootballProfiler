"""Player re-identification embeddings for linking segments whose shirt number cannot be read.

Frozen CLIP ViT-B/16 image features (OpenAI weights via timm) pass through a small projection
head trained by scripts/train_reid_head.py on this project's own analyses: thumbnails of
segments whose shirt number was read reliably are the labels (same half, team and number =
same player), and team-mates in the same kit are the negatives. No external identity labels are
used. match_identity attaches an unnumbered segment to a numbered player only when its
embedding clearly prefers that player (MIN_MARGIN), which on held-out matches picked the right
player for about nine in ten such segments.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

HEAD_FILE = 'reid_head.pt'
BACKBONE = 'vit_base_patch16_clip_224.openai'
SIZE = 224
DIM_IN, DIM_OUT = 768, 256


def weights_path():
    from .football_models import weights_dir
    return weights_dir() / HEAD_FILE


def available():
    return weights_path().is_file()


def head_module():
    import torch
    import torch.nn.functional as F

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(DIM_IN, 512), torch.nn.GELU(), torch.nn.Dropout(.1),
                                           torch.nn.Linear(512, DIM_OUT))

        def forward(self, x):
            return F.normalize(self.net(x), dim=1)
    return Head()


@lru_cache(maxsize=1)
def backbone():
    import timm
    import torch
    from .football_models import weights_dir
    from .runtime import torch_device
    os.environ.setdefault('HF_HOME', str(weights_dir() / 'hf'))
    device = torch_device()
    net = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(device).eval()
    if device.type == 'cuda':
        net = net.half()
    cfg = net.pretrained_cfg
    return net, np.array(cfg['mean'], np.float32), np.array(cfg['std'], np.float32), device


@lru_cache(maxsize=1)
def head():
    import torch
    net, _, _, device = backbone()
    h = head_module()
    h.load_state_dict(torch.load(weights_path(), map_location='cpu', weights_only=True))
    return h.to(device).eval()


def clip_features(crops):
    """L2-normalised CLIP features (n x 768) of BGR crops."""
    import cv2
    import torch
    import torch.nn.functional as F
    net, mean, std, device = backbone()
    out = []
    for i in range(0, len(crops), 256):
        x = np.stack([cv2.resize(c[:, :, ::-1], (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
                      for c in crops[i:i + 256]]).astype(np.float32) / 255
        x = torch.from_numpy((x - mean) / std).permute(0, 3, 1, 2).to(device)
        with torch.inference_mode():
            f = net(x.half() if device.type == 'cuda' else x).float()
        out.append(F.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, DIM_IN), np.float32)


def project(features):
    import torch
    _, _, _, device = backbone()
    with torch.inference_mode():
        return head()(torch.from_numpy(np.asarray(features, np.float32)).to(device)).cpu().numpy()


def embed_groups(groups):
    """key -> unit embedding (mean of its crops' embeddings) for groups of BGR crops."""
    keys = [k for k, crops in groups.items() if crops]
    flat, owner = [], []
    for k in keys:
        flat += groups[k]
        owner += [k] * len(groups[k])
    if not flat:
        return {}
    emb = project(clip_features(flat))
    owner = np.array(owner, dtype=object)
    out = {}
    for k in keys:
        m = emb[owner == k].mean(0)
        out[k] = m / (np.linalg.norm(m) + 1e-9)
    return out
