"""Gather Group D's measured scores into one file and print them (CPU only, seconds).

Reads the evidence reports listed in football_profiler/model_performance.py, prints the before and
after of each change with its verdict, and saves everything to evidence/model_performance.json:
the one file to send back after the runs. The app's Model performance page reads the same reports
directly, so this script is not needed for the page.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\gather_model_performance.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_profiler import model_performance as MPF  # noqa: E402
from football_profiler import storage as S  # noqa: E402


def show(g):
    lines = []
    for key, s in g['spotter']['sets'].items():
        lines.append(f"\nSpotter, {s['name']} (mean AP):")
        for v in g['spotter']['versions']:
            if v['tag'] in s['mean_average_precision']:
                lines.append(f"  {v['name']:44} {s['mean_average_precision'][v['tag']]:.3f}")
    current = (g['crediting'].get('current') or {}).get('actions', {})
    if current:
        lines.append('\nCrediting on FOOTPASS (found / right player / right track):')
        for label, a in sorted(current.items()):
            lines.append(f"  {label:10} {a.get('found', 0):6.1%} {a.get('credited_correctly', 0):8.1%} "
                         f"{a.get('credited_to_right_track', 0):8.1%}   ({a.get('labels')} labels)")
    own = g.get('own_labels')
    if own:
        lines.append(f"\nOwn labels ({own.get('labels_file')}): found {own['overall']['found']:.1%} of "
                     f"{own['overall']['labels']} labelled actions")
    lines.append('\nChanges:')
    for c in g['changes']:
        lines.append(f"  {c['id']:4} {c['name']:36} {c['state']:9} {c['result']}" +
                     (f"  -> {c['verdict']}" if c.get('verdict') else ''))
    if g['missing']:
        lines.append('\nNot found yet: ' + ', '.join(g['missing']))
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evidence', type=Path, default=S.EVIDENCE, help='folder of the reports (default: evidence/)')
    args = p.parse_args()
    g = MPF.gather(args.evidence)
    S.write_json(args.evidence / 'model_performance.json', g)
    print(show(g))
    print(f"\nSaved {args.evidence / 'model_performance.json'}")


if __name__ == '__main__':
    main()
