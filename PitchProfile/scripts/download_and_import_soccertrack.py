"""Bulk-download and import every SoccerTrack v2 match from Hugging Face.

Wraps the existing single-match download_soccertrack.py / import_soccertrack.py
scripts unchanged, looping over every match in the provider's benchmark split.
Point --output at wherever you want the ~140 GB of raw video/annotations to
land (e.g. an external drive) - never defaults to a location under this repo.
Already-imported halves are skipped, and Hugging Face downloads resume/reuse
completed files, so this is safe to interrupt and rerun.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\download_and_import_soccertrack.py --output E:\\SoccerTrack
  .\\.venv\\Scripts\\python.exe scripts\\download_and_import_soccertrack.py --output E:\\SoccerTrack --matches 117093 118575
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import storage as S
from football_profiler.soccertrack import BENCHMARK_SPLITS

ROOT = Path(__file__).resolve().parents[1]
ALL_MATCHES = sorted(BENCHMARK_SPLITS)


def run(args):
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f'Command failed: {" ".join(str(a) for a in args)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='Root folder for raw downloads (point this at your external drive)')
    p.add_argument('--matches', nargs='*', default=None, help='Specific match IDs; defaults to all 10 (already-imported ones are skipped)')
    p.add_argument('--sampling-hz', type=float, default=5)
    args = p.parse_args()
    matches = args.matches or ALL_MATCHES
    print(f'{len(matches)} matches to process, downloading to {args.output}')
    done, skipped, failed = [], [], []
    for i, match_id in enumerate(matches, 1):
        already = all((S.DATA / 'datasets' / f'soccertrack-{match_id}-h{h}').exists() for h in (1, 2))
        if already:
            skipped.append(match_id)
            print(f'[{i}/{len(matches)}] {match_id}: already imported, skipping')
            continue
        start = time.perf_counter()
        try:
            run(['scripts/download_soccertrack.py', '--match', match_id, '--output', str(args.output), '--download'])
            for half, suffix in ((1, '1st'), (2, '2nd')):
                identifier = f'soccertrack-{match_id}-h{half}'
                if (S.DATA / 'datasets' / identifier).exists():
                    continue
                gsr = args.output / 'gsr' / match_id / f'{match_id}_{suffix}.json'
                video = args.output / 'videos' / match_id / f'{match_id}_panorama_{suffix}_half.mp4'
                bas = args.output / 'bas' / match_id / f'{match_id}_12_class_events.json'
                run(['scripts/import_soccertrack.py', '--match-id', match_id, '--half', str(half),
                     '--gsr', str(gsr), '--video', str(video), '--bas', str(bas),
                     '--sampling-hz', str(args.sampling_hz)])
            elapsed = time.perf_counter() - start
            print(f'[{i}/{len(matches)}] {match_id}: done ({elapsed:.1f}s)')
            done.append(match_id)
        except Exception as e:
            print(f'[{i}/{len(matches)}] {match_id}: FAILED - {type(e).__name__}: {e}')
            failed.append(match_id)
    print(f'\nDone: {len(done)} imported, {len(skipped)} already present, {len(failed)} failed')
    if failed:
        print('Failed matches:', failed)


if __name__ == '__main__':
    main()
