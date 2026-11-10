"""How spotted actions are credited to players, and why the rest are not, on analysed halves (CPU only).

For each analysed half with action_spots.json, the CPU stages are rebuilt (pitch positions, teams,
ball path: match_pipeline.stages; player segments: match_events.events) and the spots are credited
as the pipeline does it (action_spotting.attribute). Every confident spot (score at or above its
class threshold) is counted by how it was credited (picture, pitch) or why it was not
(action_spotting.UNCREDITED). Whether a credit names the right player needs ground truth: scripts/measure_shot_credit.py.

Report: evidence/credit_reasons.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\credit_reasons.py                 (this project's analysed halves)
  .\\.venv\\Scripts\\python.exe scripts\\credit_reasons.py --footpass      (the FOOTPASS validation halves)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAYS = ('picture', 'pitch')


def tally(rows):
    """{action type: Counter of how each confident spot was credited, or why not} for attribute() rows."""
    out = {}
    for r in rows:
        if not r.get('confident'):
            continue
        out.setdefault(r['type'], Counter())[r.get('credited_by') or r.get('uncredited') or 'unknown'] += 1
    return out


def table(by_type, reasons):
    """Rows of shares per action type: credited (and how), then each reason for not."""
    rows = {}
    for kind, c in sorted(by_type.items()):
        n = sum(c.values())
        rows[kind] = {'spots': n, 'credited': round(sum(c[w] for w in WAYS) / n, 3),
                      **{w: round(c[w] / n, 3) for w in WAYS}, **{r: round(c[r] / n, 3) for r in reasons}}
    return rows


def merge(total, part):
    for kind, c in part.items():
        total.setdefault(kind, Counter()).update(c)
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--footpass', action='store_true', help='the FOOTPASS validation halves (analyse_footpass.py)')
    p.add_argument('--match', default='', help='only datasets whose name contains this')
    args = p.parse_args()
    if args.footpass:                    # where analyse_footpass.py keeps them (as evaluate_identity_footpass.py)
        data = Path((ROOT / '.data-location').read_text(encoding='utf-8').strip()) \
            if (ROOT / '.data-location').is_file() else ROOT / 'data'
        os.environ.setdefault('PITCHPROFILE_DATA', str(data.parent.parent / 'PitchProfile-eval' / 'data'))
    sys.path.insert(0, str(ROOT))
    from football_profiler import action_spotting as AS
    from football_profiler import match_events as ME
    from football_profiler import match_pipeline as MP
    from football_profiler import storage as S

    pattern = 'fp-val-*' if args.footpass else '*'
    dirs = [d for d in sorted((S.DATA / 'datasets').glob(pattern))
            if args.match.lower() in d.name.lower() and (d / 'action_spots.json').is_file()
            and (d / 'raw_people.csv.gz').is_file()]
    if not dirs:
        raise SystemExit(f'No analysed halves with action_spots.json in {S.DATA / "datasets"}')
    totals, halves = {}, {}
    for d in dirs:
        spots = S.read_json(d / 'action_spots.json')['spots']
        _, hz, _, people, ball, directions, _ = MP.stages(d.name)
        people = ME.events(people, ball, directions, hz)[3]           # players with their segments
        part = tally(AS.attribute(spots, people, ball, directions, hz))
        merge(totals, part)
        halves[d.name] = table(part, AS.UNCREDITED)
        print(f'{d.name}: {sum(sum(c.values()) for c in part.values())} confident spots', flush=True)
    report = {'created': S.now(), 'datasets': [d.name for d in dirs], 'reasons': list(AS.UNCREDITED),
              'overall': table(totals, AS.UNCREDITED), 'halves': halves}
    S.write_json(S.EVIDENCE / 'credit_reasons.json', report)
    cols = ['spots', 'credited', *WAYS, *AS.UNCREDITED]
    print(f'\n{"":12}' + ''.join(f'{c:>24}' for c in cols))
    for kind, row in report['overall'].items():
        print(f'{kind:12}' + ''.join(f'{row[c]:>24}' for c in cols))


if __name__ == '__main__':
    main()
