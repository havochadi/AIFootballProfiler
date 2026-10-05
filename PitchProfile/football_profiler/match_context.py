"""Historical match-day identities, kept separate from video-derived features.

ESPN's public website feed is undocumented and may change. Cache successful
responses locally; a failed refresh never destroys a previously fetched lineup.
Kit groups are assigned by the reviewer, never assumed to mean home/away.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from . import storage as S

LEAGUES = {'england_epl': 'eng.1', 'spain_laliga': 'esp.1', 'italy_serie-a': 'ita.1',
           'germany_bundesliga': 'ger.1', 'france_ligue-1': 'fra.1',
           'europe_uefa-champions-league': 'uefa.champions'}
API = 'https://site.api.espn.com/apis/site/v2/sports/soccer/'
ALIASES = {'manutd': 'manchesterunited', 'manunited': 'manchesterunited', 'manchestercityfc': 'manchestercity',
           'parissg': 'parissaintgermain', 'psg': 'parissaintgermain', 'bayernmunchen': 'bayernmunich',
           'bayer04leverkusen': 'bayerleverkusen', 'internazionale': 'intermilan', 'inter': 'intermilan',
           'atleticomadrid': 'atleticomadrid', 'athleticbilbao': 'athleticclub', 'athbilbao': 'athleticclub',
           'mgladbach': 'borussiamonchengladbach', 'dortmund': 'borussiadortmund', 'westham': 'westhamunited',
           'tottenham': 'tottenhamhotspur', 'ajaxamsterdam': 'ajax', 'chievo': 'chievoverona', 'sscnapoli': 'napoli'}


def normal_name(name):
    name = ''.join(c for c in unicodedata.normalize('NFKD', name or '') if not unicodedata.combining(c))
    name = re.sub(r'\b(fc|afc|cf)\b', '', name.lower())
    name = re.sub('[^a-z0-9]', '', name)
    return ALIASES.get(name, name)


def similarity(a, b):
    return SequenceMatcher(None, normal_name(a), normal_name(b)).ratio()


def league_for(manifest):
    match = str(manifest.get('match_id', '')).removeprefix('soccernet:')
    return LEAGUES.get(match.split('/')[0])


def cache_path(manifest):
    fixture = manifest.get('match_id') or manifest['id']
    key = hashlib.sha256(fixture.encode()).hexdigest()[:24]
    return S.DATA / 'match_context' / (key + '.json')


def fetch_json(league, resource, params):
    if league not in LEAGUES.values() or resource not in ('scoreboard', 'summary'):
        raise ValueError('Unsupported match data endpoint')
    url = API + league + '/' + resource + '?' + urlencode(params)
    try:
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}), timeout=25) as response:
            content = response.read(8_000_001)
        if len(content) > 8_000_000:
            raise ValueError('The match-data response is too large')
        return json.loads(content)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ValueError('The match provider is unavailable. Cached data is still usable; try connecting again later.') from exc


def safe_media(url):
    parsed = urlparse(url or '')
    return url if parsed.scheme == 'https' and parsed.hostname in ('a.espncdn.com', 'a1.espncdn.com', 'a2.espncdn.com') else None


def media_path(url):
    return S.DATA / 'match_assets' / (hashlib.sha256(url.encode()).hexdigest()[:32] + '.img')


def media_url(url):
    if not url:
        return None
    path = media_path(url)
    return '/api/match-assets/' + path.stem if path.is_file() else url


def cache_media(context):
    """Download only source-provided, allowlisted portraits and badges; never fabricate faces."""
    urls = {t.get('logo') for t in context['teams']}
    urls.update(p.get('headshot') for t in context['teams'] for p in t['players'])

    def download(url):
        if not safe_media(url):
            return False
        path = media_path(url)
        if path.is_file():
            return True
        try:
            with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=15) as response:
                if not safe_media(response.geturl()):
                    return False
                content = response.read(3_000_001)
            if len(content)>3_000_000 or not (content.startswith(b'\x89PNG\r\n') or content.startswith(b'\xff\xd8\xff') or (content.startswith(b'RIFF') and content[8:12]==b'WEBP')):
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replacement also permits simultaneous connection of two halves.
            import os
            import tempfile
            fd, temporary = tempfile.mkstemp(dir=path.parent)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(content)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return True
        except (OSError, HTTPError, URLError, TimeoutError):
            return False
    valid = [u for u in urls if u]
    with ThreadPoolExecutor(max_workers=4) as pool:
        count = sum(pool.map(download, valid))
    return {'cached': count, 'available': len(valid)}


def event_id_from(value):
    if re.fullmatch(r'\d{1,12}', str(value)):
        return str(value)
    parsed = urlparse(value or '')
    if parsed.scheme != 'https' or parsed.hostname not in ('www.espn.com', 'www.espn.co.uk', 'www.espn.com.sg'):
        raise ValueError('Use an ESPN match link or its numeric event ID')
    match = re.search(r'/gameId/(\d{1,12})(?:/|$)', parsed.path)
    if not match:
        raise ValueError('This is not an ESPN match link')
    return match.group(1)


def candidates(manifest, league):
    date = str(manifest.get('date', ''))[:10]
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        raise ValueError('This source needs a match date before searching for a lineup')
    data = fetch_json(league, 'scoreboard', {'dates': date.replace('-', ''), 'limit': 100})
    expected = manifest.get('fixture_teams') or {}
    out = []
    for event in data.get('events', []):
        competitions = event.get('competitions') or []
        if not competitions:
            continue
        teams = {p['homeAway']: p['team']['displayName'] for p in competitions[0].get('competitors', [])}
        scores = [similarity(expected.get(side, ''), teams.get(side, '')) for side in ('home', 'away')]
        if min(scores) < .6:
            continue
        out.append({'event_id': str(event['id']), 'date': event.get('date'), 'home': teams.get('home'),
                    'away': teams.get('away'), 'match_score': round(min(scores), 3)})
    return sorted(out, key=lambda x: x['match_score'], reverse=True)


def parse_summary(raw, league):
    if not isinstance(raw, dict):
        raise ValueError('The provider returned an unsupported match format')
    header = raw.get('header') or {}
    competitions = header.get('competitions') or []
    if not competitions:
        raise ValueError('The provider did not return a match')
    game = competitions[0]
    roster_by_team = {str(r['team']['id']): r for r in raw.get('rosters', [])}
    teams = []
    for competitor in game.get('competitors', []):
        team = competitor['team']; roster = []
        for row in roster_by_team.get(str(team['id']), {}).get('roster', []):
            athlete = row['athlete']; shirt = str(row.get('jersey', ''))
            stats = {v['name']: v.get('value') for v in row.get('stats', [])}
            # This is the match roster's jersey, never the athlete's current jersey.
            roster.append({'id': str(athlete['id']), 'name': athlete.get('fullName') or athlete.get('displayName'),
                           'shirt': int(shirt) if shirt.isdigit() else None, 'starter': bool(row.get('starter')),
                           'played': bool(row.get('starter') or row.get('subbedIn') or stats.get('appearances')),
                           'position': row.get('position', {}).get('displayName'),
                           'headshot': safe_media((athlete.get('headshot') or {}).get('href')),
                           'profile_url': 'https://www.espn.com/soccer/player/_/id/' + str(athlete['id']),
                           'subbed_in': bool(row.get('subbedIn')), 'subbed_out': bool(row.get('subbedOut'))})
        teams.append({'id': str(team['id']), 'name': team['displayName'], 'side': competitor['homeAway'],
                      'score': competitor.get('score'), 'abbreviation': team.get('abbreviation'),
                      'logo': safe_media((team.get('logos') or [{}])[0].get('href')), 'players': roster})
    outcomes = {'shot-on-target': 'on_target', 'shot-off-target': 'off_target', 'shot-blocked': 'blocked',
                'goal': 'goal', 'own-goal': 'own_goal', 'shot-hit-woodwork': 'woodwork'}
    reported = []; seen = set()
    for row in raw.get('commentary', []):
        play = row.get('play') or {}; kind = play.get('type', {}).get('type')
        if kind not in outcomes or not play.get('id') or play['id'] in seen:
            continue
        seen.add(play['id'])
        people = play.get('participants') or []
        reported.append({'id': str(play['id']), 'outcome': outcomes[kind], 'period': play.get('period', {}).get('number'),
                         'clock': play.get('clock', {}).get('displayValue'), 'match_seconds': play.get('clock', {}).get('value'),
                         'team': play.get('team', {}).get('displayName'),
                         'player': people[0].get('athlete', {}).get('displayName') if people else None,
                         'description': play.get('text') or row.get('text', '')})
    return {'provider': 'ESPN', 'league': league, 'event_id': str(header['id']), 'date': game.get('date'),
            'competition': header.get('season', {}).get('name'), 'status': game.get('status', {}).get('type', {}).get('description'),
            'venue': raw.get('gameInfo', {}).get('venue', {}).get('fullName'), 'attendance': raw.get('gameInfo', {}).get('attendance'),
            'teams': sorted(teams, key=lambda t: t['side'] != 'home'), 'reported_shots': reported,
            'source_url': 'https://www.espn.com/soccer/match/_/gameId/' + str(header['id']), 'fetched': S.now(),
            'note': 'Historical match roster; portraits may be current. Match-clock events are separate from video-clock detections.'}


def sync(manifest, event_id=None, league=None, refresh=False):
    path = cache_path(manifest); existing = S.read_json(path)
    if existing and not refresh and not event_id:
        return existing
    league = league or league_for(manifest)
    if league not in LEAGUES.values():
        raise ValueError('Choose a supported competition for the match lookup')
    if not event_id:
        options = candidates(manifest, league)
        if not options or options[0]['match_score'] < .82 or (len(options)>1 and options[0]['match_score']-options[1]['match_score']<.12):
            raise ValueError('No unambiguous fixture found. Paste the ESPN match link to connect the correct game.')
        event_id = options[0]['event_id']
    event_id = event_id_from(event_id)
    raw = fetch_json(league, 'summary', {'event': event_id})
    try:
        context = parse_summary(raw, league)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError('The provider match format changed. Existing cached data was kept.') from exc
    if context['event_id'] != event_id or (manifest.get('date') and context['date'][:10] != manifest['date'][:10]):
        raise ValueError('The provider match date does not match this video')
    expected = manifest.get('fixture_teams') or {}
    for team in context['teams']:
        if expected.get(team['side']) and similarity(expected[team['side']], team['name']) < .6:
            raise ValueError('The provider teams do not match this video')
    if len(context['teams']) != 2 or any(len(t['players']) < 11 for t in context['teams']):
        raise ValueError('The provider did not return complete starting lineups. Existing cached data was kept.')
    S.write_json(path, context)
    return context


def state_for(manifest):
    context = S.read_json(cache_path(manifest))
    if context:
        for team in context['teams']:
            team['logo_source'] = team.get('logo')
            team['logo'] = media_url(team.get('logo'))
            for person in team['players']:
                person['headshot_source'] = person.get('headshot')
                person['headshot'] = media_url(person.get('headshot'))
    with S.db() as db:
        row = db.execute('SELECT payload FROM match_context_bindings WHERE dataset_id=?', (manifest['id'],)).fetchone()
        corrections = {r['player_id']: json.loads(r['payload']) for r in db.execute(
            'SELECT player_id,payload FROM player_identity_links WHERE dataset_id=?', (manifest['id'],))}
        track_corrections = {r['player_id']: json.loads(r['payload']) for r in db.execute(
            'SELECT player_id,payload FROM player_track_corrections WHERE dataset_id=?', (manifest['id'],))}
    binding = json.loads(row['payload']) if row else {}
    if not context or binding.get('event_id') != context['event_id']:
        binding = {}
    return {'context': context, 'binding': binding, 'corrections': corrections, 'track_corrections': track_corrections}


def bind_teams(manifest, mapping, reviewer):
    context = S.read_json(cache_path(manifest))
    if not context:
        raise ValueError('Connect the historical match first')
    if set(mapping) != {'A', 'B'} or set(mapping.values()) != {t['id'] for t in context['teams']}:
        raise ValueError('Assign each kit group to a different team from this match')
    if not reviewer.strip():
        raise ValueError('Enter your name to confirm the kit groups')
    payload = {'event_id': context['event_id'], 'teams': mapping, 'reviewer': reviewer.strip(), 'updated': S.now()}
    with S.db() as db:
        db.execute('INSERT OR REPLACE INTO match_context_bindings VALUES(?,?)', (manifest['id'], json.dumps(payload)))
    return state_for(manifest)


def decorate(manifest, player, info=None):
    info = info or state_for(manifest); context = info['context']; pid = player.get('player_id') or player.get('identity')
    original = next((p for p in manifest['players'] if p['player_id'] == pid), player)
    automatic = original.get('jersey'); kit = original.get('team_key') or player.get('team')
    manual = info.get('track_corrections', {}).get(pid)
    out = {'automatic_jersey': automatic, 'automatic_role': original.get('role'),
           'identity_status': 'no_context', 'headshot': None, 'lineup_player': None, 'identity_correction': manual}
    if manual:
        group = player.get('position_group', original.get('position_group'))
        group = 'goalkeeper' if manual['role'] == 'goalkeeper' else None if group == 'goalkeeper' else group
        label = 'goalkeeper' if manual['role'] == 'goalkeeper' else 'outfield player'
        team_name = original.get('team') or manifest.get('teams', {}).get(kit, {}).get('name') or f'Team {kit}'
        player = {**player, 'role': manual['role'], 'position_group': group, 'jersey': manual['jersey'],
                  'name': manual['name'] or (f"{team_name} #{manual['jersey']}" if manual['jersey'] is not None else f'{team_name} · {label}')}
        out.update(identity_status='corrected', identity_reviewer=manual['reviewer'])
    if not context:
        return {**player, **out}
    correction = info['corrections'].get(pid, {})
    explicit = correction.get('team_explicit') and correction.get('event_id') == context['event_id']
    team_id = correction.get('team_id') if explicit else info['binding'].get('teams', {}).get(kit)
    team = next((t for t in context['teams'] if t['id'] == team_id), None)
    if not team:
        return {**player, **out, 'identity_status': 'corrected' if manual else 'kit_unmapped'}
    confirmed = correction.get('event_id') == context['event_id'] and correction.get('team_id') == team_id
    number = manual['jersey'] if manual else automatic
    matches = [p for p in team['players'] if (p['id'] == correction.get('athlete_id') if confirmed else p['shirt'] == number and number is not None)]
    out.update(team_name=team['name'], team_logo=team['logo'], team_id=team_id)
    if manual and (manual['name'] or len(matches) != 1 or
                   (matches[0].get('position') and (matches[0]['position'].lower() == 'goalkeeper') != (manual['role'] == 'goalkeeper'))):
        return {**player, **out}
    if len(matches) != 1:
        return {**player, **out, 'identity_status': 'not_in_lineup', 'name': f"{team['name']} #{automatic}" if automatic is not None else player['name']}
    person = matches[0]
    status = 'confirmed' if confirmed else 'corrected' if manual else 'lineup_match' if person['played'] else 'unused_substitute'
    out.update(name=person['name'], jersey=person['shirt'], headshot=person['headshot'], lineup_player=person,
               identity_status=status, identity_reviewer=correction.get('reviewer') if confirmed else manual['reviewer'] if manual else None,
               identity_source=context['source_url'])
    if confirmed and person.get('position'):
        role = 'goalkeeper' if person['position'].lower() == 'goalkeeper' else 'player'
        out.update(role=role, position_group='goalkeeper' if role == 'goalkeeper' else
                   None if player.get('position_group') == 'goalkeeper' else player.get('position_group'))
    return {**player, **out}


def correct_track(manifest, pid, role, jersey, name, reviewer, note='', time_s=None, reset=False):
    """Review metadata for this half's track; raw detections and evidence remain intact."""
    original = next((p for p in manifest['players'] if p['player_id'] == pid), None)
    if original is None:
        raise FileNotFoundError('Unknown player')
    if not reviewer.strip():
        raise ValueError('Enter your name to save the correction')
    if time_s is not None:
        start = float(manifest.get('source_offset_s', 0))
        if not start <= time_s < start + float(manifest['duration_seconds']):
            raise ValueError('Correction timestamp is outside this analysed half')
    payload = {'role': role, 'jersey': jersey, 'name': name.strip(), 'reviewer': reviewer.strip(),
               'note': note.strip(), 'time_s': time_s, 'scope': 'track_in_half', 'updated': S.now(), 'reset': reset}
    with S.db() as db:
        if reset:
            db.execute('DELETE FROM player_track_corrections WHERE dataset_id=? AND player_id=?', (manifest['id'], pid))
            db.execute('DELETE FROM player_identity_links WHERE dataset_id=? AND player_id=?', (manifest['id'], pid))
        else:
            db.execute('INSERT OR REPLACE INTO player_track_corrections VALUES(?,?,?)', (manifest['id'], pid, json.dumps(payload)))
            # A newly reviewed identity supersedes a previous line-up assignment.
            db.execute('DELETE FROM player_identity_links WHERE dataset_id=? AND player_id=?', (manifest['id'], pid))
        db.execute('INSERT INTO player_track_correction_history(dataset_id,player_id,payload) VALUES(?,?,?)',
                   (manifest['id'], pid, json.dumps(payload)))
    return decorate(manifest, original)


def link_player(manifest, pid, athlete_id, reviewer, note='', team_id=None):
    original = next((p for p in manifest['players'] if p['player_id'] == pid), None)
    if original is None:
        raise FileNotFoundError('Unknown player')
    info = state_for(manifest); context = info['context']
    explicit = team_id is not None
    team_id = team_id or info['binding'].get('teams', {}).get(original.get('team_key'))
    team = next((t for t in (context or {}).get('teams', []) if t['id'] == team_id), None)
    person = next((p for p in (team or {}).get('players', []) if p['id'] == athlete_id), None)
    if not person:
        raise ValueError('Choose a player from the selected team’s match-day lineup')
    if not person['played']:
        raise ValueError('This substitute did not play according to the match source; review the kit mapping and identity')
    if not reviewer.strip():
        raise ValueError('Enter your name to confirm the identity')
    payload = {'event_id': context['event_id'], 'team_id': team_id, 'athlete_id': athlete_id,
               'team_explicit': explicit, 'shirt': person['shirt'], 'reviewer': reviewer.strip(), 'note': note, 'updated': S.now()}
    with S.db() as db:
        db.execute('DELETE FROM player_track_corrections WHERE dataset_id=? AND player_id=?', (manifest['id'], pid))
        db.execute('INSERT OR REPLACE INTO player_identity_links VALUES(?,?,?)', (manifest['id'], pid, json.dumps(payload)))
        db.execute('INSERT INTO player_identity_link_history(dataset_id,player_id,payload) VALUES(?,?,?)',
                   (manifest['id'], pid, json.dumps(payload)))
    return decorate(manifest, original)
