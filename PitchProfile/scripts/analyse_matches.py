"""Analyse local SoccerNet 720p halves end to end (video only, no provider metadata).

Each half becomes an app dataset sn-<date>-<home>-<away>-h<half> with per-player
statistics. Finished halves are skipped, so the command is safe to rerun. Only
games in the working set (SoccerNet/working_set.json) are used unless --all-videos
is given.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\analyse_matches.py                 # every local half
  .\\.venv\\Scripts\\python.exe scripts\\analyse_matches.py --match Chelsea  # titles containing "Chelsea"
  .\\.venv\\Scripts\\python.exe scripts\\analyse_matches.py --postprocess-only   # re-run CPU stages only
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import match_pipeline as MP
from football_profiler import soccernet as SN
from football_profiler import storage as S


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--match', default='', help='Only halves whose game path contains this text')
    p.add_argument('--half', type=int, choices=[1, 2])
    p.add_argument('--postprocess-only', action='store_true', help='Recompute CPU stages from existing raw evidence')
    p.add_argument('--force', action='store_true', help='Re-run halves that are already complete')
    p.add_argument('--target-hz', type=float, default=12.5)
    p.add_argument('--all-videos', action='store_true',
                   help='Ignore the working set (SoccerNet/working_set.json) and use every local video')
    args = p.parse_args()
    items = [v for v in SN.library(all_videos=args.all_videos)
             if args.match.lower() in v['game'].lower() and (args.half in (None, v['half']))]
    print(f'{len(items)} halves selected', flush=True)
    done, skipped, failed = [], [], []
    for n, item in enumerate(items, 1):
        identifier = MP.soccernet_identifier(item['game'], item['half'])
        directory = S.DATA / 'datasets' / identifier
        meta = S.read_json(directory / 'analysis.json', {}) if directory.is_dir() else {}
        has_raw = (directory / 'raw_frames.csv.gz').is_file() and 'detection' in meta
        if args.postprocess_only:
            if not has_raw:
                skipped.append(identifier)
                print(f'[{n}/{len(items)}] {identifier}: no raw evidence, skipping', flush=True)
                continue
        elif meta.get('postprocess') and not args.force:
            skipped.append(identifier)
            print(f'[{n}/{len(items)}] {identifier}: already analysed', flush=True)
            continue
        started, last = time.time(), [0.0]

        def progress(fraction, message):
            if time.time() - last[0] > 60:
                print(f'    {fraction:.0%} {message}', flush=True)
                last[0] = time.time()
        try:
            if args.postprocess_only or (has_raw and not args.force):
                MP.postprocess(identifier, progress=progress)
            else:
                MP.analyse_soccernet(item['id'], progress=progress, target_hz=args.target_hz)
            done.append(identifier)
            print(f'[{n}/{len(items)}] {identifier}: done in {(time.time() - started) / 60:.1f} min', flush=True)
        except Exception as exc:
            failed.append(identifier)
            print(f'[{n}/{len(items)}] {identifier}: FAILED - {type(exc).__name__}: {exc}', flush=True)
            traceback.print_exc()
    print(f'\nDone: {len(done)} analysed, {len(skipped)} skipped, {len(failed)} failed', flush=True)
    if failed:
        print('Failed:', failed)


if __name__ == '__main__':
    main()
