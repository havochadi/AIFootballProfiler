"""Train and score the partial-label ablation of the action spotter in one go (on the GPU machine).

Two fine-tunes from the published checkpoint with the same settings, seed and data, differing only in
how FOOTPASS labels are used:
  partial  the adopted recipe: label sets (spotter_data.FOOTPASS_ALLOWED) and the partial-label loss
  hard     exact labels (spotter_data.FOOTPASS_HARD) and plain cross-entropy (--hard-labels)
Each is scored with evaluate_spotter_labelled.py on the held-out games (FOOTPASS validation, BAS test)
and the average precision per class is tabulated next to the published checkpoint and the adopted
fine-tune, where their reports exist. The settings are copied from the adopted run's log
(evidence/action_spotter_finetune.json) so the partial arm repeats that recipe on today's data.
One run per arm, so small differences may be noise.

Report: evidence/loss_ablation.json.

Usage (from PitchProfile/; both arms take about 7-8 hours on one RTX 3090):
  .\\.venv\\Scripts\\python.exe scripts\\run_loss_ablation.py [--arms partial hard] [--dry-run]
  .\\.venv\\Scripts\\python.exe scripts\\run_loss_ablation.py --summary-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_profiler import storage as S  # noqa: E402

ARMS = {'partial': 'tdeed_ablation_partial', 'hard': 'tdeed_ablation_hard'}
HELD_OUT = [('footpass', 'val'), ('bas', 'test')]
# Report tags of earlier evaluations (evaluate_spotter_labelled.py) and their column names.
COLUMNS = {'published': 'published', 'finetuned': 'adopted', ARMS['partial']: 'partial', ARMS['hard']: 'hard'}
# Training settings carried over from the adopted run; the rest of its log (out_dir, start, workers) is not.
COPIED = ['steps', 'batch', 'accum', 'lr', 'warmup', 'rare_share', 'rare_weight', 'rare', 'bas_share',
          'val_every', 'val_clips', 'seed', 'no_mixup', 'freeze']
DEFAULTS = {'steps': 3000}         # the adopted run (model documentation, section 5.3) if its log is missing


def recipe(log):
    """finetune_action_spotter.py flags for the adopted recipe, from its log's args (else DEFAULTS)."""
    args = {k: v for k, v in ((log or {}).get('args') or {}).items() if k in COPIED} or dict(DEFAULTS)
    flags = []
    for k, v in args.items():
        name = '--' + k.replace('_', '-')
        if isinstance(v, bool):
            flags += [name] if v else []
        elif v not in (None, ''):
            flags += [name, str(v)]
    return flags


def commands(flags, arms, weights):
    """{arm: (training command, [scoring commands])}; each arm trains into its own weights folder."""
    py = sys.executable
    plan = {}
    for arm in arms:
        out_dir = ARMS[arm]
        train = [py, 'scripts/finetune_action_spotter.py', *flags, '--out-dir', out_dir]
        train += ['--hard-labels'] if arm == 'hard' else []
        checkpoint = Path(weights) / out_dir / 'checkpoint_best.pt'
        scores = [[py, 'scripts/evaluate_spotter_labelled.py', '--dataset', dataset, '--split', split,
                   '--checkpoint', str(checkpoint), '--tag', out_dir] for dataset, split in HELD_OUT]
        plan[arm] = (train, scores)
    return plan


def summarise(reports):
    """Mean and per-class average precision per model, from {held-out set: {tag: evaluation report}}."""
    table = {}
    for key, by_tag in reports.items():
        classes = list(dict.fromkeys(c for r in by_tag.values() for c in r['classes']))
        table[key] = {
            'mean_average_precision': {COLUMNS.get(t, t): r['mean_average_precision'] for t, r in by_tag.items()},
            'average_precision': {c: {COLUMNS.get(t, t): r['classes'][c]['average_precision']
                                      for t, r in by_tag.items() if c in r['classes']} for c in classes}}
    return table


def show(table):
    lines = []
    for key, t in table.items():
        names = list(t['mean_average_precision'])
        lines.append(f'\n{key:26}' + ''.join(f'{n:>10}' for n in names))
        lines.append(f'{"mean AP":26}' + ''.join(f'{t["mean_average_precision"][n]:>10.3f}' for n in names))
        for c, row in t['average_precision'].items():
            lines.append(f'{c:26}' + ''.join(f'{row[n]:>10.3f}' if n in row else f'{"-":>10}' for n in names))
    return '\n'.join(lines)


def load_reports(evidence):
    out = {}
    for dataset, split in HELD_OUT:
        for tag in COLUMNS:
            report = S.read_json(Path(evidence) / f'action_spotter_{dataset}_{split}_{tag}.json')
            if report:
                out.setdefault(f'{dataset}_{split}', {})[tag] = report
    return out


def run(cmd):
    print(time.strftime('%H:%M:%S') + ' $ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    p.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    p.add_argument('--summary-only', action='store_true', help='only tabulate the reports that already exist')
    args = p.parse_args()
    from football_profiler import spotter_data as SD
    from football_profiler.football_models import weights_dir
    log = S.read_json(S.EVIDENCE / 'action_spotter_finetune.json', {})
    flags = recipe(log)
    print('recipe: ' + ' '.join(flags) + ('' if log else '   (adopted log not found: documented settings)'))
    for arm, (train, scores) in ({} if args.summary_only else commands(flags, args.arms, weights_dir())).items():
        if args.dry_run:
            print('\n'.join('$ ' + ' '.join(c) for c in [train, *scores]))
            continue
        run(train)
        # Spots cached by an earlier run of this arm came from an older checkpoint.
        for dataset, split in HELD_OUT:
            for cached in (SD.root() / dataset / split).glob(f'*/spots_{ARMS[arm]}.json'):
                cached.unlink()
        for cmd in scores:
            run(cmd)
    if args.dry_run:
        return
    table = summarise(load_reports(S.EVIDENCE))
    S.write_json(S.EVIDENCE / 'loss_ablation.json', {'created': S.now(), 'recipe': flags, 'arms': args.arms,
                                                      'table': table})
    print(show(table))


if __name__ == '__main__':
    main()
