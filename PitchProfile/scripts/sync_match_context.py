"""Cache historical ESPN results and lineups for local SoccerNet games on the data drive.

No model inference or kit/identity assignment is performed. Human kit confirmation
is done in the review desk. Successful lookups are reused by both match halves.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import storage as S
from football_profiler import soccernet as SN
from football_profiler import match_pipeline as MP
from football_profiler import match_context as MC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='', help='Filter fixture names')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args(); seen = set(); failures = []
    for item in SN.library():
        game = item['game']
        if game in seen or args.match.casefold() not in game.casefold():
            continue
        seen.add(game); meta = MP.soccernet_meta(item)
        home, away = MP.fixture_teams(game.rsplit('/', 1)[-1])
        manifest = {'id': MP.soccernet_identifier(game, item['half']), 'match_id': meta['match_id'],
                    'date': meta['date'], 'fixture_teams': {'home': home, 'away': away}}
        try:
            context = MC.sync(manifest, refresh=args.refresh)
            media = MC.cache_media(context)
            print(f"OK {meta['title']} | {context['event_id']} | {sum(len(t['players']) for t in context['teams'])} squad members | {media['cached']}/{media['available']} portraits and badges", flush=True)
        except ValueError as error:
            failures.append(game)
            print(f"UNAVAILABLE {meta['title']} | {error}", flush=True)
    print(f'{len(seen)-len(failures)}/{len(seen)} fixtures cached in {S.DATA / "match_context"}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
