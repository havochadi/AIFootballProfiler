"""Build (or rebuild) whole matches from their two analysed halves (football_profiler.match_merge).

With --reanalyse-halves each half is post-processed first, so its players, reviewer names and
ratings reflect the current code before the halves are joined. Ratings given on a half are
carried into the whole match (the latest rating wins when a player was rated in both halves).

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\build_whole_matches.py [--reanalyse-halves] [--match chelsea]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import match_merge as MM  # noqa: E402
from football_profiler import match_pipeline as MP  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reanalyse-halves', action='store_true')
    p.add_argument('--match', default='', help='only whole matches whose id contains this text')
    args = p.parse_args()
    pairs = {k: v for k, v in sorted(MM.halves_by_match().items()) if args.match.lower() in k.lower()}
    print(f'{len(pairs)} whole matches', flush=True)
    for i, (identifier, halves) in enumerate(pairs.items(), 1):
        t0 = time.time()
        if args.reanalyse_halves:
            for h in halves:
                MP.postprocess(h)
        a = MM.build(identifier)
        pp = a['postprocess']
        print(f"[{i}/{len(pairs)}] {identifier}: {pp['identities']} players, {pp['named_players']} named, "
              f"kits {pp['kit_mapping_second_half']}, {time.time() - t0:.0f} s", flush=True)
    # Ratings saved on a half while the batch ran reach their whole match too.
    copied = sum(MM.sync_ratings(identifier) for identifier in pairs)
    print(f'ratings copied from halves after building: {copied}', flush=True)


if __name__ == '__main__':
    main()
