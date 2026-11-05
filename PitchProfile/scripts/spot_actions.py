"""Run the ball-action spotter (football_profiler.action_spotting) on every working-set half.

Writes action_spots.json (time, action, team side, score) into each analysed dataset. Halves that
already have spots are skipped unless --force. The CPU post-processing (scripts/analyse_matches.py
--postprocess-only) then credits the actions to players.

--checkpoint runs another checkpoint (e.g. the fine-tune from finetune_action_spotter.py); the
spots it replaces are kept once as action_spots_<--keep-as>.json, so the two can be compared
(evaluate_action_spotter.py --spots-file) and restored.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\spot_actions.py [--match Chelsea] [--force] [--checkpoint PATH]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import action_spotting as AS
from football_profiler import match_pipeline as MP
from football_profiler import soccernet as SN
from football_profiler import storage as S


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--match', default='')
    p.add_argument('--force', action='store_true')
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--keep-as', default='published', help='suffix for the replaced spots file')
    args = p.parse_args()
    if args.checkpoint is not None:
        AS.checkpoint_path = lambda: args.checkpoint
    items = [v for v in SN.library() if args.match.lower() in v['game'].lower()]
    for n, item in enumerate(items, 1):
        identifier = MP.soccernet_identifier(item['game'], item['half'])
        d = S.DATA / 'datasets' / identifier
        if not d.is_dir():
            print(f'[{n}/{len(items)}] {identifier}: not analysed, skipping', flush=True)
            continue
        if (d / 'action_spots.json').is_file() and not args.force:
            print(f'[{n}/{len(items)}] {identifier}: already spotted', flush=True)
            continue
        started = time.time()
        result = AS.spot_video(SN.source_path(item['id']))
        result['created'] = S.now()
        result['checkpoint'] = str(AS.checkpoint_path())
        kept = d / f'action_spots_{args.keep_as}.json'
        if (d / 'action_spots.json').is_file() and not kept.is_file():
            (d / 'action_spots.json').replace(kept)
        S.write_json(d / 'action_spots.json', result)
        strong = sum(s['score'] >= AS.SCORE_THRESHOLD for s in result['spots'])
        print(f'[{n}/{len(items)}] {identifier}: {len(result["spots"])} spots ({strong} >= {AS.SCORE_THRESHOLD}) '
              f'in {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
