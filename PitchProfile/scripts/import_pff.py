"""Bulk-import PFF FC 2022 World Cup matches directly from their delivered zip archives.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\import_pff.py --archive-dir ..\\WorldCup2022
  .\\.venv\\Scripts\\python.exe scripts\\import_pff.py --archive-dir ..\\WorldCup2022 --games 3813 3814
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import pff, storage as S


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-dir', type=Path, required=True, help='Folder containing the provider zip archives')
    p.add_argument('--games', nargs='*', default=None, help='Specific game IDs; defaults to every game found')
    p.add_argument('--sampling-hz', type=float, default=10)
    args = p.parse_args()
    zips = sorted(args.archive_dir.glob('*.zip'))
    if not zips:
        p.error(f'No .zip files found in {args.archive_dir}')
    games = args.games or pff.list_games(zips)
    print(f'{len(zips)} archives, {len(games)} games to process')
    done, skipped, failed = [], [], []
    for i, game_id in enumerate(games, 1):
        existing = (S.DATA / 'datasets' / f'pff-wc2022-{game_id}-h1').exists()
        if existing:
            skipped.append(game_id)
            print(f'[{i}/{len(games)}] {game_id}: already imported, skipping')
            continue
        start = time.perf_counter()
        try:
            manifests = pff.import_match(zips, game_id, sampling_hz=args.sampling_hz)
            elapsed = time.perf_counter() - start
            title = manifests[1]['title'].rsplit(' — ', 1)[0]
            print(f'[{i}/{len(games)}] {game_id}: {title} ({elapsed:.1f}s)')
            done.append(game_id)
        except Exception as e:
            print(f'[{i}/{len(games)}] {game_id}: FAILED — {type(e).__name__}: {e}')
            failed.append(game_id)
    print(f'\nDone: {len(done)} imported, {len(skipped)} already present, {len(failed)} failed')
    if failed:
        print('Failed games:', failed)


if __name__ == '__main__':
    main()
