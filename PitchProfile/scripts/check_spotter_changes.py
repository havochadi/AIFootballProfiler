"""Check the action spotter's speed changes and the throw-in team rule on the GPU machine.

1. Speed, on the first --frames frames (12.5 per second) of one analysed half, three ways: as it was
   (frames read in turn with the model), with read-ahead (action_spotting.READ_AHEAD, now the default;
   no score may change), and with read-ahead plus half overlap (PITCHPROFILE_SPOTTER_HALF_OVERLAP:
   a clip every 50 frames instead of 25). Seconds, frames per second, minutes for a 45-minute half,
   speed-up, and the largest change in any score from the first way.
2. Accuracy of half overlap on the held-out games (FOOTPASS validation, BAS test):
   evaluate_spotter_labelled.py with the switch on (tag finetuned_half_overlap) next to the current
   setting (tag finetuned, whose cached spots are reused, so that run needs no GPU when they exist).
3. Throw-in team (action_spotting.THROW_IN_TEAM): from the same reports, the spotter's own team for
   throw-ins against the team opposite the last touch within 4, 10, 20 or 30 s.

Report: evidence/spotter_changes.json.

Usage (from PitchProfile/; about half an hour on one RTX 3090 with nothing else on it):
  .\\.venv\\Scripts\\python.exe scripts\\check_spotter_changes.py [--match Chelsea] [--frames 7500]
  .\\.venv\\Scripts\\python.exe scripts\\check_spotter_changes.py --summary-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_profiler import action_spotting as AS  # noqa: E402
from football_profiler import storage as S  # noqa: E402

HELD_OUT = [('footpass', 'val'), ('bas', 'test')]
SWITCH = 'PITCHPROFILE_SPOTTER_HALF_OVERLAP'
SETTINGS = {'before': (0, 25), 'read_ahead': (AS.READ_AHEAD, 25), 'read_ahead_half_overlap': (AS.READ_AHEAD, 50)}
HALF_FRAMES = 45 * 60 * 12.5                       # frames the spotter sees in a 45-minute half
THROW_IN = {'footpass': 'Throw-in', 'bas': 'THROW IN'}


def pick_video(match=''):
    """(name, path) of the first working-set half whose game contains `match` and whose video is here."""
    from football_profiler import soccernet as SN
    for item in SN.library():
        if match.lower() in item['game'].lower():
            path = Path(SN.source_path(item['id']))
            if path.is_file():
                return f"{item['game']} half {item['half']}", path
    raise SystemExit(f'No working-set video found{" for " + match if match else ""}')


def warm_file(path):
    """Read the file once so that every timed way finds it in the disk cache alike."""
    with open(path, 'rb') as f:
        while f.read(1 << 24):
            pass


def time_settings(video, frames, settings=None):
    """{way: frames, seconds, frames per second, minutes per half, speed-up, largest score change, spots}.

    The model is loaded and warmed up first, so that no way pays for it.
    """
    settings = settings or SETTINGS
    saved = AS.READ_AHEAD, AS.CLIP_STEP
    out, first = {}, None
    try:
        AS.frame_scores(video, max_frames=min(frames, 300))
        for name, (ahead, step) in settings.items():
            AS.READ_AHEAD, AS.CLIP_STEP = ahead, step
            started = time.perf_counter()
            scores = AS.frame_scores(video, max_frames=frames)
            seconds = time.perf_counter() - started
            first = scores if first is None else first
            rate = len(scores) / max(seconds, 1e-9)
            out[name] = {'read_ahead': ahead, 'clip_step': step, 'frames': int(len(scores)),
                         'seconds': round(seconds, 2), 'frames_per_second': round(rate, 1),
                         'minutes_per_half': round(HALF_FRAMES / rate / 60, 2),
                         'largest_score_change': float(np.abs(scores - first).max()) if len(scores) else None,
                         'spots': len(AS.peaks(scores, 25 / AS.FRAME_STRIDE))}
    finally:
        AS.READ_AHEAD, AS.CLIP_STEP = saved
    first_seconds = next(iter(out.values()))['seconds'] if out else None
    for row in out.values():
        row['speed_up'] = round(first_seconds / row['seconds'], 2) if first_seconds and row['seconds'] else None
    return out


def evaluate(dataset, split, tag, half):
    """evaluate_spotter_labelled.py on one held-out set, with half overlap switched on or off."""
    env = {k: v for k, v in os.environ.items() if k != SWITCH}
    if half:
        env[SWITCH] = '1'
    cmd = [sys.executable, 'scripts/evaluate_spotter_labelled.py', '--dataset', dataset, '--split', split,
           '--tag', tag]
    print(time.strftime('%H:%M:%S') + (f' [{SWITCH}=1]' if half else '') + ' $ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def load_reports(tags):
    out = {}
    for dataset, split in HELD_OUT:
        for tag in tags:
            report = S.read_json(S.EVIDENCE / f'action_spotter_{dataset}_{split}_{tag}.json')
            if report:
                out.setdefault(f'{dataset}_{split}', {})[tag] = report
    return out


def compare(reports, current, half):
    """Per held-out set: mean and per-class AP with half overlap off and on, and the throw-in team."""
    out = {}
    for dataset, split in HELD_OUT:
        by_tag = reports.get(f'{dataset}_{split}', {})
        base, alt = by_tag.get(current), by_tag.get(half)
        row = {}
        if base and alt:
            m0, m1 = base['mean_average_precision'], alt['mean_average_precision']
            row['mean_average_precision'] = {'current': m0, 'half_overlap': m1, 'change': round(m1 - m0, 3)}
            row['average_precision_change'] = {
                c: round(alt['classes'][c]['average_precision'] - v['average_precision'], 3)
                for c, v in base['classes'].items() if v['labels'] and c in alt['classes']}
        throw = (base or {}).get('classes', {}).get(THROW_IN[dataset])
        if throw and throw['labels']:
            b = throw['by_threshold'][str(throw['best_f1_threshold'])]
            row['throw_in_team'] = {'spotter': b['team_correct'], 'last_touch_4s': b['team_correct_previous_touch_rule'],
                                    **{f'last_touch_{w}s': v
                                       for w, v in b.get('team_correct_previous_touch_within_s', {}).items()}}
        out[f'{dataset}_{split}'] = row
    return out


def show(report):
    lines = []
    timing = report.get('timing')
    if timing:
        lines.append(f"\nSpeed on {timing['video']} ({timing['frames']} frames each):")
        for name, r in timing['settings'].items():
            change = r['largest_score_change']
            lines.append(f"  {name:24} {r['seconds']:8.1f} s  {r['frames_per_second']:7.1f} frames/s  "
                         f"{r['minutes_per_half']:5.1f} min per half  x{r['speed_up']}  "
                         f"largest score change {'-' if change is None else f'{change:.2g}'}  spots {r['spots']}")
    for key, row in report.get('held_out', {}).items():
        lines.append(f'\n{key}:')
        if 'mean_average_precision' in row:
            m = row['mean_average_precision']
            lines.append(f"  mean AP {m['current']:.3f} -> {m['half_overlap']:.3f} with half overlap "
                         f"(change {m['change']:+.3f})")
        else:
            lines.append('  half overlap: no reports yet')
        if 'throw_in_team' in row:
            lines.append('  throw-in team right: ' + ', '.join(f'{k} {v:.2f}' for k, v in row['throw_in_team'].items()))
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--match', default='', help='time the first working-set half whose game contains this')
    p.add_argument('--frames', type=int, default=7500, help='frames timed per way (7,500 = 10 minutes of video)')
    p.add_argument('--summary-only', action='store_true', help='only tabulate the reports that already exist')
    args = p.parse_args()
    current = 'published' if AS.checkpoint_path() == AS.published_checkpoint_path() else 'finetuned'
    half = current + '_half_overlap'
    report = S.read_json(S.EVIDENCE / 'spotter_changes.json', {}) or {}
    if not args.summary_only:
        name, video = pick_video(args.match)
        warm_file(video)
        print(f'Timing on {name}, {args.frames} frames per way', flush=True)
        report['timing'] = {'video': name, 'frames': args.frames, 'settings': time_settings(video, args.frames)}
        print(show({'timing': report['timing']}), flush=True)
        try:
            import torch
            torch.cuda.empty_cache()                    # leave the GPU memory to the evaluation runs
        except Exception:
            pass
        from football_profiler import spotter_data as SD
        for dataset, split in HELD_OUT:
            # Spots cached by an earlier half-overlap run may come from older code.
            for cached in (SD.root() / dataset / split).glob(f'*/spots_{half}.json'):
                cached.unlink()
            evaluate(dataset, split, current, half=False)
            evaluate(dataset, split, half, half=True)
    report.update({'created': S.now(), 'checkpoint': current,
                   'held_out': compare(load_reports([current, half]), current, half)})
    S.write_json(S.EVIDENCE / 'spotter_changes.json', report)
    print(show(report))


if __name__ == '__main__':
    main()
