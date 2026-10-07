"""Train the player-identity model on FOOTPASS true identities and compare it with the current one.

One network does both identity jobs on a player thumbnail:
  - shirt number: two heads read the tens digit (none, 1-9) and the units digit (0-9), so numbers
    missing from the training games can still be read;
  - appearance: a 256-d embedding trained so crops of the same player are closer than crops of any
    other player of the same game, team-mates included (supervised contrastive loss).
It is CLIP ViT-B/16 (the same backbone as the retired re-identification head) fine-tuned end to end on crops cut by
scripts/footpass_crops.py from FOOTPASS training games (SoccerNet NDA data; each crop labelled with
the player's true shirt number and team). Crops are scaled down at random so 720p broadcasts are
covered, and never mirrored (mirrored digits read differently).

Measured on the held-out FOOTPASS validation games, per true tracklet (a player's crops with no
gap over 2 s), against the current reader (football_profiler.jersey: legibility classifier and
PARSeq) and the retired appearance head (football_profiler.reid, removed). Report:
evidence/identity_model.json; weights: <weights>/identity_vitb16/model.pt.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\train_identity_models.py [--steps 6000] [--eval-only]
"""
from __future__ import annotations

import argparse
import io
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_profiler import soccernet as SN  # noqa: E402
from football_profiler import identity_model as IM  # noqa: E402
from football_profiler import storage as S  # noqa: E402

IDS_PER_BATCH, CROPS_PER_ID = 16, 4
RUN_GAP_FRAMES = 50          # a player's crops with gaps up to 2 s form one true tracklet
# With --legible-only the number heads learn only from crops the legibility classifier
# (football_profiler.jersey) finds readable; every FOOTPASS crop is labelled with the player's
# number, shirt visible or not, and learning those makes the heads guess from appearance.
LEGIBLE_TRAIN = .5


def crops_dir(split):
    return SN.root() / 'sn-pcbas-2026' / 'crops' / split


def legibility_scores(csv):
    """Per-row legibility of one game's crops, cached next to its csv (<game>.legibility.npy)."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    from football_profiler import jersey as J
    cache = csv.with_suffix('.legibility.npy')
    rows = pd.read_csv(csv)
    if cache.is_file():
        scores = np.load(cache)
        if len(scores) == len(rows):
            return scores
    net, _, device = J.models()
    scores = np.zeros(len(rows), np.float32)
    with zipfile.ZipFile(csv.with_suffix('.zip')) as z:
        names = list(rows.crop)
        decode = lambda n: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
        with ThreadPoolExecutor(8) as pool:
            for i in range(0, len(names), 1024):
                imgs = list(pool.map(decode, names[i:i + 1024]))
                ok = [k for k, im in enumerate(imgs) if im is not None]
                if ok:
                    scores[[i + k for k in ok]] = np.concatenate(
                        [J._legibility([imgs[k] for k in ok[j:j + 256]], net, device) for j in range(0, len(ok), 256)])
    np.save(cache, scores)
    print(f'{csv.stem}: {len(scores)} crops, legible share {float((scores >= LEGIBLE_TRAIN).mean()):.3f}', flush=True)
    return scores


def load_split(split, legibility=False):
    frames = []
    for csv in sorted(crops_dir(split).glob('*.csv')):
        if (csv.with_suffix('.zip')).is_file():
            f = pd.read_csv(csv)
            if legibility:
                f['legibility'] = legibility_scores(csv)
            frames.append(f)
    if not frames:
        raise SystemExit(f'No crops in {crops_dir(split)}: run scripts/footpass_crops.py --split {split}')
    d = pd.concat(frames, ignore_index=True)
    d['identity'] = d.game + ':' + d.player_id.astype(str)
    return d


class Crops:
    """Reads crops from the per-game zips (one open handle per process)."""

    def __init__(self, split):
        self.split, self.handles = split, {}

    def image(self, game, name):
        import cv2
        z = self.handles.get(game)
        if z is None:
            z = self.handles[game] = zipfile.ZipFile(crops_dir(self.split) / f'{game}.zip')
        return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)


def augment(img, rng):
    import cv2
    # Down-scale to between 45% and 100% of the full-HD size and back: 720p is two thirds.
    s = rng.uniform(.45, 1.0)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(4, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    if rng.random() < .5:
        ok, enc = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(35, 90))])
        small = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    img = small
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    if rng.random() < .5:                        # another kit colour: digits, not colours, must decide
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(0, 180)) % 180
    hsv[..., 1] *= rng.uniform(.7, 1.3)
    hsv[..., 2] *= rng.uniform(.7, 1.3)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < .15:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    # Small crop jitter, as detector boxes vary.
    h, w = img.shape[:2]
    dx, dy = int(w * rng.uniform(0, .08)), int(h * rng.uniform(0, .06))
    img = img[dy:h - int(h * rng.uniform(0, .06)) or h, dx:w - int(w * rng.uniform(0, .08)) or w]
    return img


class Batches:
    """P identities x K crops, all from one game, so team-mates are each other's negatives."""

    def __init__(self, data, seed):
        self.by_game = {g: {i: q.index.to_numpy() for i, q in d.groupby('identity')} for g, d in data.groupby('game')}
        self.games = list(self.by_game)
        self.data, self.rng = data, random.Random(seed)
        self.crops = Crops('TRAIN')

    def __iter__(self):
        return self

    def __next__(self):
        game = self.rng.choice(self.games)
        ids = self.rng.sample(list(self.by_game[game]), min(IDS_PER_BATCH, len(self.by_game[game])))
        imgs, tens, units, who, keeper = [], [], [], [], []
        for k, identity in enumerate(ids):
            rows = self.by_game[game][identity]
            for j in self.rng.choices(list(rows), k=CROPS_PER_ID):
                r = self.data.loc[j]
                img = self.crops.image(r.game, r.crop)
                if img is None:
                    continue
                imgs.append(IM.letterbox(augment(img, self.rng)))
                t, u = IM.digits(int(r.shirt))
                if 'legibility' in self.data and r.legibility < LEGIBLE_TRAIN:
                    t = u = -100                     # ignored by the number loss
                tens.append(t)
                units.append(u)
                who.append(k)
                keeper.append(float(r.role == 1))
        return imgs, tens, units, who, keeper


JERSEY_PER_BATCH = 48


def jersey_index(split='train'):
    """{tracklet: (number, [image names])} of SN-Jersey-2023 tracklets with a visible number."""
    z = zipfile.ZipFile(SN.root() / 'sn-jersey-2023' / f'{split}.zip')
    import json
    gt = json.loads(z.read(f'{split}/{split}_gt.json'))
    names = {}
    for n in z.namelist():
        parts = n.split('/')
        if len(parts) == 4 and parts[1] == 'images' and n.endswith('.jpg'):
            names.setdefault(parts[2], []).append(n)
    return {k: (int(gt[k]), v) for k, v in names.items() if int(gt.get(k, -1)) >= 0}


class JerseyBatches:
    """One thumbnail from each of JERSEY_PER_BATCH random SN-Jersey tracklets (number labels only)."""

    def __init__(self, index, seed, split='train'):
        self.items, self.rng = list(index.values()), random.Random(seed)
        self.path, self.zip = SN.root() / 'sn-jersey-2023' / f'{split}.zip', None

    def __next__(self):
        import cv2
        if self.zip is None:
            self.zip = zipfile.ZipFile(self.path)
        imgs, tens, units = [], [], []
        for number, names in self.rng.sample(self.items, JERSEY_PER_BATCH):
            img = cv2.imdecode(np.frombuffer(self.zip.read(self.rng.choice(names)), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            imgs.append(IM.letterbox(augment(img, self.rng)))
            t, u = IM.digits(number)
            tens.append(t)
            units.append(u)
        return imgs, tens, units


class BatchStream(torch.utils.data.IterableDataset):
    """Batches built in loader worker processes (each with its own zip handles and random stream)."""

    def __init__(self, data, seed, jersey=None):
        self.data, self.seed, self.jersey = data, seed, jersey

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        seed = self.seed + 1000 * (info.id if info else 0)
        batches = Batches(self.data, seed)
        jersey = JerseyBatches(self.jersey, seed + 7) if self.jersey else None
        while True:
            imgs, tens, units, who, keeper = next(batches)
            out = [IM.to_tensor(imgs), torch.tensor(tens), torch.tensor(units), torch.tensor(who), torch.tensor(keeper)]
            if jersey is not None:
                ji, jt, ju = next(jersey)
                out += [IM.to_tensor(ji), torch.tensor(jt), torch.tensor(ju)]
            yield tuple(out)


def batch_stream(data, seed, workers, jersey=None):
    return iter(torch.utils.data.DataLoader(BatchStream(data, seed, jersey), batch_size=None, num_workers=workers,
                                            prefetch_factor=4 if workers else None, persistent_workers=workers > 0,
                                            pin_memory=True))


def supcon(z, labels, tau=.1):
    import torch
    sim = z @ z.T / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim.masked_fill(eye, -1e9)
    logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
    has = pos.any(1)
    return -(logp * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


def train(args, device):
    import torch
    data = load_split('TRAIN', legibility=args.legible_only)
    print(f'training crops: {len(data)} from {data.game.nunique()} games, {data.identity.nunique()} players', flush=True)
    if args.legible_only:
        print(f'number labels kept on legible crops: {float((data.legibility >= LEGIBLE_TRAIN).mean()):.3f}', flush=True)
    net = IM.build().to(device)
    heads = [p for n, p in net.named_parameters() if not n.startswith('backbone.')]
    opt = torch.optim.AdamW([{'params': net.backbone.parameters(), 'lr': args.lr},
                             {'params': heads, 'lr': args.lr * 20}], weight_decay=.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 300) * .5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    jersey = jersey_index('train') if args.jersey else None
    if jersey:
        print(f'SN-Jersey training tracklets with a number: {len(jersey)}', flush=True)
    batches = batch_stream(data, args.seed, args.workers, jersey)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=.05)
    net.train()
    t0, running = time.time(), np.zeros(3)
    for step in range(1, args.steps + 1):
        batch = next(batches)
        x, tens, units, who, keeper = batch[:5]
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lt, lu, z, lk = net(x)
        lt, lu, z, lk = lt.float(), lu.float(), z.float(), lk.float()
        tens, units = tens.to(device), units.to(device)
        l_num = ce(lt, tens) + ce(lu, units) if (tens >= 0).any() else lt.sum() * 0
        l_id = supcon(z, who.to(device))
        l_keep = torch.nn.functional.binary_cross_entropy_with_logits(lk, keeper.to(device))
        loss = l_num + args.id_weight * l_id + .5 * l_keep
        if len(batch) > 5:                       # SN-Jersey: shirt numbers from other leagues and seasons
            jx = batch[5].to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                jt, ju, _, _ = net(jx)
            l_jersey = ce(jt.float(), batch[6].to(device)) + ce(ju.float(), batch[7].to(device))
            loss = loss + args.jersey_weight * l_jersey
            l_num = (l_num + l_jersey) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        running += [float(l_num), float(l_id), 1]
        if step % 100 == 0:
            rate = step / (time.time() - t0)
            print(f'step {step}/{args.steps} number loss {running[0] / running[2]:.3f} identity loss '
                  f'{running[1] / running[2]:.3f} | {rate:.1f} steps/s, {(args.steps - step) / rate / 60:.0f} min left',
                  flush=True)
            running[:] = 0
    from football_profiler.football_models import weights_dir
    path = output_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)
    print('weights saved to', path, flush=True)
    return net


def output_path(args):
    """The adopted weights, or a trial file beside them (--out) that replaces nothing."""
    return IM.weights_path() if not args.out else IM.weights_path().parent / args.out


def tracklets(data):
    """True tracklets: a player's crops in one half with no gap over RUN_GAP_FRAMES."""
    data = data.sort_values(['game', 'half', 'player_id', 'frame']).reset_index(drop=True)
    gap = (data.frame.diff() > RUN_GAP_FRAMES) | data.player_id.ne(data.player_id.shift()) | \
        data.game.ne(data.game.shift()) | data.half.ne(data.half.shift())
    data['tracklet'] = gap.cumsum()
    return data


def predict(net, data, device, crops, batch=128):
    """Per crop: log-probabilities of the tens and units digits and the embedding."""
    import torch
    net.eval()
    tens, units, embs, keeper = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(data), batch):
            rows = data.iloc[i:i + batch]
            x = IM.to_tensor([IM.letterbox(crops.image(r.game, r.crop)) for r in rows.itertuples()]).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lt, lu, z, lk = net(x)
            tens.append(torch.log_softmax(lt.float(), -1).cpu().numpy())
            units.append(torch.log_softmax(lu.float(), -1).cpu().numpy())
            embs.append(z.float().cpu().numpy())
            keeper.append(torch.sigmoid(lk.float()).cpu().numpy())
    return np.concatenate(tens), np.concatenate(units), np.concatenate(embs), np.concatenate(keeper)


def coverage_curve(correct, confidence):
    """Share of tracklets read, and accuracy of those, at confidence thresholds."""
    out = []
    for thr in (0, .3, .5, .7, .8, .9, .95):
        sel = confidence >= thr
        out.append({'min_confidence': thr, 'read_share': round(float(sel.mean()), 3),
                    'accuracy': round(float(correct[sel].mean()), 3) if sel.any() else None})
    return out


def evaluate(net, device, max_tracklets=None):
    from football_profiler import jersey as J
    data = tracklets(load_split('VAL'))
    crops = Crops('VAL')
    if max_tracklets:
        keep = pd.Series(data.tracklet.unique()).sample(min(max_tracklets, data.tracklet.nunique()), random_state=0)
        data = data[data.tracklet.isin(keep)].reset_index(drop=True)
    print(f'validation: {len(data)} crops, {data.tracklet.nunique()} true tracklets, {data.game.nunique()} games', flush=True)
    report = {'validation_games': sorted(data.game.unique()), 'tracklets': int(data.tracklet.nunique()),
              'crops': int(len(data))}
    lt, lu, emb, keeper = predict(net, data, device, crops)
    rows = []
    for k, g in data.groupby('tracklet'):
        idx = g.index.to_numpy()
        number, p, best = IM.read_tracklet(lt[idx], lu[idx])
        rows.append({'tracklet': k, 'truth': int(g.shirt.iat[0]), 'new': number, 'new_conf': p, 'crops': len(idx),
                     'box_h': float(g.box_h.median()), 'keeper_truth': bool(g.role.iat[0] == 1),
                     'keeper_prob': float(keeper[idx].mean())})
    t = pd.DataFrame(rows)
    report['new_reader'] = coverage_curve((t.new == t.truth).to_numpy(), t.new_conf.to_numpy())
    kt = t.keeper_truth.to_numpy()
    kp = t.keeper_prob.to_numpy() >= .5
    report['goalkeeper'] = {'tracklets': int(kt.sum()), 'recall': round(float(kp[kt].mean()), 3) if kt.any() else None,
                            'precision': round(float(kt[kp].mean()), 3) if kp.any() else None}
    # The current reader on the same tracklets.
    if J.available():
        groups = {k: [crops.image(r.game, r.crop) for r in g.itertuples()] for k, g in data.groupby('tracklet')}
        old = J.read_groups(groups)
        t['old'] = [(old.get(k) or {}).get('number') for k in t.tracklet]
        t['old_conf'] = [(old.get(k) or {}).get('confidence') or 0.0 for k in t.tracklet]
        t['old_votes'] = [(old.get(k) or {}).get('votes') or 0.0 for k in t.tracklet]
        from football_profiler import match_identity as MI
        trusted = (t.old_conf >= MI.MIN_NUMBER_SHARE) & (t.old_votes >= MI.MIN_NUMBER_VOTES) & t.old.notna()
        report['current_reader'] = {'read_share': round(float(trusted.mean()), 3),
                                    'accuracy': round(float((t.old[trusted] == t.truth[trusted]).mean()), 3) if trusted.any() else None,
                                    'any_reading_share': round(float(t.old.notna().mean()), 3)}
    # Appearance: hide each tracklet and match it to its team's players built from the other
    # tracklets that do not overlap it in time, as match_identity does with shirt-number anchors.
    report['appearance'] = appearance_eval(data, emb)
    t.to_csv(S.EVIDENCE / 'identity_model_tracklets.csv', index=False)
    return report


def appearance_eval(data, emb):
    from football_profiler import match_identity as MI
    res = []
    for (game, half), g in data.groupby(['game', 'half']):
        seg = g.groupby('tracklet').agg(player=('player_id', 'first'), team=('team', 'first'),
                                        f0=('frame', 'min'), f1=('frame', 'max'), n=('frame', 'size'))
        vec = {k: emb[q.index].mean(0) for k, q in g.groupby('tracklet')}
        vec = {k: v / (np.linalg.norm(v) + 1e-9) for k, v in vec.items()}
        for k, r in seg.iterrows():
            same = seg[(seg.team == r.team) & (seg.index != k)]
            scores = {}
            for pl, q in same.groupby('player'):
                q = q[~((q.f0 <= r.f1) & (r.f0 <= q.f1))]
                if len(q):
                    proto = np.mean([vec[j] for j in q.index], 0)
                    scores[pl] = float(vec[k] @ (proto / (np.linalg.norm(proto) + 1e-9)))
            if len(scores) < 2 or r.player not in scores:
                continue
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            res.append((ranked[0][0] == r.player, ranked[0][1] - ranked[1][1], int(r.n)))
    r = pd.DataFrame(res, columns=['correct', 'margin', 'crops'])
    out = {'tracklets': int(len(r)), 'accuracy_all': round(float(r.correct.mean()), 3), 'by_margin': []}
    for m in (0, .02, .05, .1, .15, .2):
        sel = r[r.margin >= m]
        out['by_margin'].append({'min_margin': m, 'attached_share': round(len(sel) / max(len(r), 1), 3),
                                 'accuracy': round(float(sel.correct.mean()), 3) if len(sel) else None})
    return out


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--id-weight', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--jersey', action='store_true', help='also learn numbers from SN-Jersey-2023 train tracklets')
    p.add_argument('--jersey-weight', type=float, default=1.0)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--max-val-tracklets', type=int, default=None)
    p.add_argument('--legible-only', action='store_true',
                   help='learn shirt numbers only from crops the legibility classifier finds readable')
    p.add_argument('--out', default=None, help='weights file name beside the adopted weights (a trial run)')
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    if args.eval_only:
        from football_profiler.football_models import weights_dir
        net = IM.build().to(device)
        path = output_path(args)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    else:
        net = train(args, device)
    report = {'created': S.now(), 'backbone': IM.BACKBONE, 'steps': args.steps, **evaluate(net, device, args.max_val_tracklets)}
    S.write_json(S.EVIDENCE / ('identity_model.json' if not args.out else f'identity_model_{Path(args.out).stem}.json'),
                 report)
    print({k: v for k, v in report.items() if k not in ('validation_games',)})


if __name__ == '__main__':
    main()
