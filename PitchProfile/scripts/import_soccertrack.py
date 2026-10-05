"""Import one authorised SoccerTrack v2 GSR half without loading its multi-GB JSON at once."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler.soccertrack import import_half


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gsr', type=Path, required=True)
    parser.add_argument('--match-id', required=True)
    parser.add_argument('--half', type=int, choices=[1,2], required=True)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--bas', type=Path)
    parser.add_argument('--event-offset-ms', type=float)
    parser.add_argument('--alignment-note', default='')
    parser.add_argument('--sampling-hz', type=float, default=5)
    parser.add_argument('--left-attacks', choices=['unknown','left','right'], default='unknown')
    args = vars(parser.parse_args())
    result = import_half(**args)
    print(f"Imported {result['id']}: {len(result['players'])} tracks, {result['duration_seconds']:.1f} seconds")
