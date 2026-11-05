"""Fine-tune the action spotter (T-DEED team ball-action baseline) on BAS + FOOTPASS labelled games.

Starts from the published checkpoint (action_spotting.PUBLISHED_CHECKPOINT) and continues training its
ball-action head, displacement head and team head on packed frames (pack_spotter_frames.py):
SoccerNet Ball Action Spotting train + valid games (12 classes, exact labels) and FOOTPASS train
games (8 classes as partial labels, spotter_data.FOOTPASS_ALLOWED). BAS test games and FOOTPASS
validation games are held out (evaluate_spotter_labelled.py).

Training follows the devkit (train_tdeed_bas.py, model.TDEEDModel.epoch): 100-frame clips at
12.5 frames/s, mixup (Beta(0.2, 0.2)), the model's colour/blur augmentation, foreground classes
weighted 5x, losses class + displacement + 2 x team. Differences: a partial-label class loss
(-log of the summed probability of the allowed classes; equal to cross-entropy for exact labels),
a lower learning rate for fine-tuning, extra weight on tackles and blocks, and a share of clips
centred on rare actions. The unused SoccerNet-v2 joint-training head is left untouched.

Checkpoints: <weights>/tdeed_team_bas_ft/checkpoint_best.pt (lowest held-out class loss on
FOOTPASS validation clips) and checkpoint_last.pt; log: evidence/action_spotter_finetune.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\finetune_action_spotter.py [--steps 8000] [--batch 4]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import action_spotting as AS  # noqa: E402
from football_profiler import spotter_data as SD  # noqa: E402
from football_profiler import storage as S  # noqa: E402

OUT_DIR = 'tdeed_team_bas_ft'
FG_WEIGHT = 5.0
LOSS_WEIGHTS = (1.0, 1.0, 2.0)                  # class, displacement, team (devkit)
RARE = ['PLAYER SUCCESSFUL TACKLE', 'BALL PLAYER BLOCK', 'SHOT', 'CROSS', 'HEADER', 'THROW IN']


class Clips:
    """Random training clips: a share centred on rare actions, the rest anywhere in a video."""

    def __init__(self, groups, rare_share, seed, rare=None):
        self.groups = groups                         # [(weight, [PackedVideo])]
        self.rare_share = rare_share
        self.rng = random.Random(seed)
        rare_bits = SD.mask(rare or RARE)
        # Indexed by position: worker processes receive copies of the videos.
        self.rare = [[[e['frame'] for e in v.events if SD.allowed(e, v.dataset) & rare_bits] for v in vs]
                     for _, vs in groups]

    def sample(self):
        r, acc = self.rng.random() * sum(w for w, _ in self.groups), 0.0
        for g, (w, vs) in enumerate(self.groups):
            acc += w
            if r <= acc:
                break
        k = self.rng.randrange(len(vs))
        v, frames = vs[k], self.rare[g][k]
        if frames and self.rng.random() < self.rare_share:
            start = self.rng.choice(frames) - self.rng.randrange(AS.CLIP_LEN)
        else:
            start = self.rng.randrange(-5, max(1, v.n - AS.CLIP_LEN + 5))
        return v, start


class ClipDataset:
    """torch Dataset of clips; each item decodes one clip (index i draws from a seeded stream)."""

    def __init__(self, clips, length):
        self.clips, self.length = clips, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        v, start = self.clips.sample()
        allow, displ, team = v.clip_targets(start, AS.CLIP_LEN)
        return {'frame': v.frames(start, AS.CLIP_LEN), 'allow': allow, 'displ': displ, 'team': team.astype(np.float32)}


class FixedClips:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        v, start = self.items[i]
        allow, displ, team = v.clip_targets(start, AS.CLIP_LEN)
        return {'frame': v.frames(start, AS.CLIP_LEN), 'allow': allow, 'displ': displ, 'team': team.astype(np.float32)}


def worker_init(k):
    import torch
    info = torch.utils.data.get_worker_info()
    info.dataset.clips.rng.seed(info.seed)


def class_weights(device, rare_weight):
    import torch
    w = torch.full((SD.NUM_CLASSES,), FG_WEIGHT, device=device)
    w[0] = 1.0
    for c in ('PLAYER SUCCESSFUL TACKLE', 'BALL PLAYER BLOCK'):
        w[SD.INDEX[c]] *= rare_weight
    return w


def losses(net, batch, mix, weights, device, train):
    """Devkit losses with mixup; `mix` is a second batch (or None) blended into the first."""
    import torch
    import torch.nn.functional as F
    x = batch['frame'].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float()
    allow1 = batch['allow'].to(device).long()
    d1, t1 = batch['displ'].to(device), batch['team'].to(device)
    if mix is not None:
        lam = torch.tensor(np.random.beta(.2, .2, size=x.shape[0]), device=device, dtype=torch.float32)
        x2 = mix['frame'].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float()
        x = lam.view(-1, 1, 1, 1, 1) * x + (1 - lam.view(-1, 1, 1, 1, 1)) * x2
        allow2 = mix['allow'].to(device).long()
        d2, t2 = mix['displ'].to(device), mix['team'].to(device)
    with torch.autocast('cuda', enabled=device.type == 'cuda'):
        out, _ = net(x, inference=not train)
    logits = out['im_feat'][..., :SD.NUM_CLASSES]
    predD, predT = out['displ_feat'].float(), out['team_feat'].float()
    if mix is None:
        lossC = SD.partial_label_loss(logits, allow1, weights).mean()
        target_d, target_t = d1, t1
    else:
        l = lam.view(-1, 1)
        lossC = (l * SD.partial_label_loss(logits, allow1, weights) +
                 (1 - l) * SD.partial_label_loss(logits, allow2, weights)).mean()
        target_d = l * d1 + (1 - l) * d2
        both = (t1 >= 0) & (t2 >= 0)
        target_t = torch.where(both, l * t1 + (1 - l) * t2, torch.where(t1 >= 0, t1, t2))
    lossD = F.mse_loss(predD, target_d)
    has = target_t >= 0
    lossT = F.binary_cross_entropy_with_logits(predT[has], target_t[has]) if has.any() else predT.sum() * 0
    total = LOSS_WEIGHTS[0] * lossC + LOSS_WEIGHTS[1] * lossD + LOSS_WEIGHTS[2] * lossT
    return total, {'class': float(lossC.detach()), 'displacement': float(lossD.detach()), 'team': float(lossT.detach())}


def freeze(net, names):
    """Freeze early backbone stages (parameters and batch-norm statistics).

    Nothing trainable comes before them, so autograd keeps none of their activations: at 796x448
    these full-resolution stages hold most of the training memory but only 16k of the backbone's
    3.2M parameters (RegNetY-002: stem, s1, s2 without temporal shift; s3 and s4 carry GSF).
    """
    frozen = [getattr(net._features, n) for n in names]
    for m in frozen:
        for q in m.parameters():
            q.requires_grad_(False)

    def set_mode():
        for m in frozen:
            m.eval()
    return set_mode


def validation_items(videos, count, seed):
    rng = random.Random(seed)
    rare_bits = SD.mask(RARE)
    items = []
    for k in range(count):
        v = videos[k % len(videos)]
        rare = [e['frame'] for e in v.events if SD.allowed(e, v.dataset) & rare_bits]
        if rare and k % 2:
            items.append((v, rng.choice(rare) - rng.randrange(AS.CLIP_LEN)))
        else:
            items.append((v, rng.randrange(0, max(1, v.n - AS.CLIP_LEN))))
    return items


def main():
    import torch
    from torch.utils.data import DataLoader
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=8000)
    # The devkit's 4 clips per step do not fit in 24 GB: 2 clips, gradients accumulated over 2.
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accum', type=int, default=2, help='micro-batches accumulated per optimiser step')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--rare-share', type=float, default=.3)
    p.add_argument('--rare-weight', type=float, default=2.0, help='extra class weight for tackles and blocks')
    p.add_argument('--rare', default='', help='comma-separated classes that clips are centred on (default: RARE)')
    p.add_argument('--bas-share', type=float, default=.35, help='share of clips from BAS (rest FOOTPASS)')
    p.add_argument('--val-every', type=int, default=500)
    p.add_argument('--val-clips', type=int, default=240)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--no-mixup', action='store_true')
    p.add_argument('--freeze', default='stem,s1,s2', help='backbone stages kept fixed (comma-separated)')
    p.add_argument('--out-dir', default=OUT_DIR, help='checkpoint folder under the weights folder (a trial run '
                   'goes elsewhere so it never replaces the adopted fine-tune)')
    p.add_argument('--start', choices=['published', 'finetuned'], default='published',
                   help='start from the published checkpoint or from the adopted fine-tune')
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bas = SD.videos('bas', 'train') + SD.videos('bas', 'valid')
    footpass = SD.videos('footpass', 'train')
    held_out = SD.videos('footpass', 'val')
    if not bas or not footpass or not held_out:
        raise SystemExit(f'Packed videos missing (BAS train+valid {len(bas)}, FOOTPASS train {len(footpass)}, '
                         f'FOOTPASS val {len(held_out)}): run pack_spotter_frames.py')
    print(f'train: {len(bas)} BAS + {len(footpass)} FOOTPASS videos; held out: {len(held_out)} FOOTPASS', flush=True)

    start = AS.published_checkpoint_path() if args.start == 'published' else AS.checkpoint_path()
    AS.checkpoint_path = lambda: start
    net, _, device = AS.model()
    frozen_eval = freeze(net, [n for n in args.freeze.split(',') if n])
    net.train()
    frozen_eval()
    weights = class_weights(device, args.rare_weight)
    per_step = args.batch * args.accum * (1 if args.no_mixup else 2)

    def make_loader(footpass, seed):
        groups = [(args.bas_share, bas), (1 - args.bas_share, footpass)]
        rare = [c.strip() for c in args.rare.split(',')] if args.rare else None
        train = ClipDataset(Clips(groups, args.rare_share, seed, rare), args.steps * per_step)
        # A clip is about 107 MB decoded; 2 batches per worker in flight keeps this near 5 GB of RAM.
        loader = DataLoader(train, batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                            worker_init_fn=worker_init, prefetch_factor=2, persistent_workers=True)
        return loader, iter(loader)

    loader, it = make_loader(footpass, args.seed)
    val_loader = DataLoader(FixedClips(validation_items(held_out, args.val_clips, 7)), batch_size=args.batch,
                            num_workers=min(args.workers, 6), pin_memory=True)
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * .5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    def validate():
        net.eval()
        tot = {'class': 0.0, 'displacement': 0.0, 'team': 0.0}
        with torch.no_grad():
            for b in val_loader:
                _, parts = losses(net, b, None, weights, device, train=False)
                for k in tot:
                    tot[k] += parts[k] * len(b['frame'])
        net.train()
        frozen_eval()
        return {k: round(v / len(val_loader.dataset), 5) for k, v in tot.items()}

    out_dir = AS.published_checkpoint_path().parent.parent / args.out_dir
    log_path = S.EVIDENCE / ('action_spotter_finetune.json' if args.out_dir == OUT_DIR
                             else f'action_spotter_finetune_{args.out_dir}.json')
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {'created': S.now(), 'args': vars(args), 'start_checkpoint': str(start),
           'train_videos': {'bas': [v.name for v in bas], 'footpass': [v.name for v in footpass]},
           'held_out': [v.name for v in held_out], 'history': []}
    base = validate()
    log['history'].append({'step': 0, 'validation': base})
    print('step 0 validation', base, flush=True)
    best = base['class']
    torch.save(net.state_dict(), out_dir / 'checkpoint_best.pt')
    running, t0, step = {}, time.time(), 0
    report_every = 10 if args.steps <= 200 else 50
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            batch = next(it)
            mix = None if args.no_mixup else next(it)
            total, parts = losses(net, batch, mix, weights, device, train=True)
            scaler.scale(total / args.accum).backward()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v / args.accum
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % report_every == 0:
            rate = step / (time.time() - t0)
            print(f'step {step}/{args.steps} ' + ' '.join(f'{k} {v / report_every:.4f}' for k, v in running.items()) +
                  f' lr {sched.get_last_lr()[0]:.2e} | {rate:.2f} steps/s, {(args.steps - step) / rate / 60:.0f} min left',
                  flush=True)
            running = {}
        if step % args.val_every == 0 or step == args.steps:
            val = validate()
            improved = val['class'] < best
            log['history'].append({'step': step, 'validation': val, 'best': improved})
            print(f'step {step} validation {val}{" (best)" if improved else ""}', flush=True)
            if improved:
                best = val['class']
                torch.save(net.state_dict(), out_dir / 'checkpoint_best.pt')
            torch.save(net.state_dict(), out_dir / 'checkpoint_last.pt')
            # Games packed since the start (pack_spotter_frames.py still running) join the training set.
            latest = SD.videos('footpass', 'train')
            if len(latest) > len(footpass) and step < args.steps:
                footpass = latest
                del it, loader
                loader, it = make_loader(footpass, args.seed + step)
                log['train_videos']['footpass'] = [v.name for v in footpass]
                log['history'][-1]['footpass_videos'] = len(footpass)
                print(f'step {step}: training on {len(footpass)} FOOTPASS videos', flush=True)
            S.write_json(log_path, log)
    log['finished'] = S.now()
    log['best_validation_class_loss'] = best
    S.write_json(log_path, log)
    print('done; best validation class loss', best, 'checkpoints in', out_dir)


if __name__ == '__main__':
    main()
