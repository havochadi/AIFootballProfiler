"""Torso-colour observations and GPU clustering; never a team/identity classifier."""
from __future__ import annotations
import cv2
import numpy as np
from .runtime import torch_device


def torso_colour(frame, box):
    x1, y1, x2, y2 = map(float, box)
    h, w = frame.shape[:2]
    a, b = max(0, int(x1 + .18 * (x2-x1))), min(w, int(x1 + .82 * (x2-x1)))
    c, d = max(0, int(y1 + .15 * (y2-y1))), min(h, int(y1 + .50 * (y2-y1)))
    if b-a < 4 or d-c < 4:
        return None
    crop = frame[c:d, a:b]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, np.array([30,35,30]), np.array([88,255,255])) > 0
    valid = (~grass) & (hsv[:,:,2] > 25)
    if valid.sum() < 12:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[valid]
    return np.median(lab, axis=0).astype(float).tolist()


def group_colours(observations, clusters=3):
    """Deterministic farthest-point K-means on GPU; track-level temporal votes."""
    import torch
    device = torch_device()
    items = [(pid, colour) for pid, colours in observations.items() for colour in colours]
    report = {'method': 'Torso CIELAB; grass mask; GPU K-means; temporal vote',
              'device': str(device), 'groups': [], 'tracks': {},
              'note': 'Anonymous colour groups only. Lighting, green kits and similar uniforms can confuse these suggestions; no referee or goalkeeper class is inferred.'}
    if len(items) < 12 or len(observations) < 3:
        report['status'] = 'insufficient_colour_observations'
        return report
    with torch.inference_mode():
        x = torch.tensor([v for _, v in items], dtype=torch.float32, device=device) / 255
        centres = [x[0]]
        for _ in range(min(clusters, len(items))-1):
            distances = torch.cdist(x, torch.stack(centres)).square().min(1).values
            if float(distances.max()) < 1e-4:
                break
            centres.append(x[distances.argmax()])
        centres = torch.stack(centres)
        for _ in range(30):
            assignment = torch.cdist(x, centres).argmin(1)
            updated = torch.stack([x[assignment.eq(i)].mean(0) if assignment.eq(i).any() else c
                                   for i,c in enumerate(centres)])
            if float((updated-centres).abs().max()) < 1e-4:
                centres = updated
                break
            centres = updated
        assignment = torch.cdist(x, centres).argmin(1).cpu().numpy()
        lab = (centres * 255).clamp(0,255).byte().cpu().numpy()
    rgb = cv2.cvtColor(lab[None], cv2.COLOR_LAB2RGB)[0]
    report['status'] = 'suggestions'
    report['groups'] = [{'id': f'kit-{i+1}', 'colour': '#'+''.join(f'{int(c):02x}' for c in row)}
                        for i,row in enumerate(rgb)]
    for pid in observations:
        labels = assignment[[p == pid for p,_ in items]]
        if len(labels) < 3:
            continue
        votes = np.bincount(labels, minlength=len(centres))
        winner = int(votes.argmax())
        report['tracks'][pid] = {'group': f'kit-{winner+1}', 'vote_share': float(votes[winner]/len(labels)),
                                 'samples': len(labels), 'colour': report['groups'][winner]['colour']}
    return report
