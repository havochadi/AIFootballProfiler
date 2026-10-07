"""API for full-match video analysis, identity review, archetype labels and semi-supervised profiles."""
from __future__ import annotations

import zipfile
import re
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import Response, JSONResponse, FileResponse
from pydantic import BaseModel, Field, model_validator

from . import storage as S
from . import taxonomy as T
from . import match_context as MC
from . import event_review as ER
from . import match_identity as MI

router = APIRouter()


def _analysed(identifier):
    d = S.dataset_dir(identifier)
    m = S.read_json(d / 'manifest.json')
    if not str(m.get('analysis', '')).startswith('full-match'):
        raise ValueError('This source is not a full-match video analysis')
    return d, m


@router.get('/api/matches/library')
def match_library():
    from . import match_pipeline as MP
    from . import soccernet as SN
    out = []
    for item in SN.library():
        identifier = MP.soccernet_identifier(item['game'], item['half'])
        meta = S.read_json(S.DATA / 'datasets' / identifier / 'analysis.json', {}) if (S.DATA / 'datasets' / identifier).is_dir() else {}
        out.append({'library_id': item['id'], 'title': MP.soccernet_meta(item)['title'], 'half': item['half'],
                    'duration': item['duration'], 'benchmark_split': item.get('benchmark_split'),
                    'dataset_id': identifier, 'analysed': bool(meta.get('postprocess')),
                    'detected': bool(meta.get('detection'))})
    return {'halves': sorted(out, key=lambda x: x['title'])}


class MatchAnalysis(BaseModel):
    library_id: str
    target_hz: float = Field(default=12.5, ge=5, le=25)


@router.post('/api/matches/analyse')
def analyse_match(payload: MatchAnalysis):
    from .app import start_job
    from . import match_pipeline as MP
    from . import soccernet as SN
    item = SN.entry(payload.library_id)
    identifier = MP.soccernet_identifier(item['game'], item['half'])
    return {**start_job('match-analysis', MP.analyse_soccernet, library_id=payload.library_id,
                        target_hz=payload.target_hz), 'dataset_id': identifier}


@router.get('/api/datasets/{identifier}/match')
def match_summary(identifier):
    d, m = _analysed(identifier)
    stats = S.read_json(d / 'match_stats.json', {})
    people = {p['player_id']: p for p in m['players']}
    context = MC.state_for(m)
    teams = {k: dict(v) for k, v in (m.get('teams') or {}).items()}
    for kit, team_id in context['binding'].get('teams', {}).items():
        actual = next((t for t in context['context']['teams'] if t['id'] == team_id), None)
        if actual and kit in teams:
            teams[kit]['name'] = actual['name']
    players = []
    from . import match_profiles as PF
    for p in stats.get('players', []):
        info = people.get(p['identity'])
        if info is None:
            continue
        group = info.get('position_group')
        players.append(MC.decorate(m, {**{k: v for k, v in p.items() if k != 'positional'},
                        'positional': {k: v for k, v in p['positional'].items() if k != 'heatmap'},
                        'name': info['name'], 'jersey': info.get('jersey'), 'position_group': info.get('position_group'),
                        **{k: info.get(k) for k in ('unnamed', 'number_guess', 'number_guess_share', 'readable_views',
                                                    'identity_confirmed')},
                        'profile': PF.profile(identifier, p, group)}, context))
    removed = sum(1 for half in (m.get('halves') or [identifier])
                  for v in S.read_json(S.dataset_dir(half) / 'confirmed_identities.json', {}).values() if v == MI.NOT_A_PLAYER)
    return S.clean_json({'id': identifier, 'title': m['title'], 'teams': teams, 'coverage': m.get('coverage'),
                         'not_player_segments': removed,
                         'fixture_teams': m.get('fixture_teams'), 'team_stats': stats.get('teams'),
                         'live_seconds': stats.get('live_seconds'), 'players': players, 'note': stats.get('note'),
                         'processing': S.read_json(d / 'analysis.json', {}).get('postprocess'),
                         'match_context': context})


@router.get('/api/datasets/{identifier}/match/context')
def match_context(identifier):
    _, m = _analysed(identifier)
    return MC.state_for(m)


class ContextConnection(BaseModel):
    event: str | None = Field(default=None, max_length=500)
    league: str | None = Field(default=None, max_length=30)
    refresh: bool = False


@router.post('/api/datasets/{identifier}/match/context')
def connect_context(identifier, payload: ContextConnection):
    _, m = _analysed(identifier)
    context = MC.sync(m, payload.event, payload.league, payload.refresh)
    MC.cache_media(context)
    return MC.state_for(m)


@router.get('/api/match-assets/{asset_id}')
def match_asset(asset_id):
    if not re.fullmatch(r'[0-9a-f]{32}', asset_id):
        raise FileNotFoundError('Portrait or badge unavailable')
    path = S.DATA / 'match_assets' / (asset_id + '.img')
    if not path.is_file():
        raise FileNotFoundError('Portrait or badge unavailable')
    with path.open('rb') as stream:
        prefix = stream.read(12)
    kind = 'image/png' if prefix.startswith(b'\x89PNG') else 'image/jpeg' if prefix.startswith(b'\xff\xd8') else 'image/webp'
    return FileResponse(path, media_type=kind, headers={'Cache-Control': 'public, max-age=86400'})


class KitBinding(BaseModel):
    teams: dict[str, str]
    reviewer: str = Field(min_length=1, max_length=80)


@router.post('/api/datasets/{identifier}/match/kit-binding')
def bind_kits(identifier, payload: KitBinding):
    """Confirm which team wears each kit group; the match's line-up then checks shirt-number
    naming, so the match is re-analysed (a whole match is rebuilt from its halves)."""
    from .app import start_job
    from . import match_merge as MM
    from . import match_pipeline as MP
    _, m = _analysed(identifier)
    state = MC.bind_teams(m, payload.teams, payload.reviewer)
    job = start_job('lineup-names', MM.build if m.get('halves') else MP.postprocess, identifier=identifier)
    return {**state, **job}


class LineupIdentity(BaseModel):
    athlete_id: str = Field(min_length=1, max_length=20, pattern=r'^\d+$')
    team_id: str | None = Field(default=None, min_length=1, max_length=20, pattern=r'^\d+$')
    reviewer: str = Field(min_length=1, max_length=80)
    note: str = Field(default='', max_length=1000)


@router.post('/api/datasets/{identifier}/players/{pid}/lineup-identity')
def correct_lineup_identity(identifier, pid, payload: LineupIdentity):
    _, m = _analysed(identifier)
    return MC.link_player(m, pid, payload.athlete_id, payload.reviewer, payload.note, payload.team_id)


class TrackCorrection(BaseModel):
    role: Literal['goalkeeper', 'player']
    jersey: int | None = Field(default=None, ge=1, le=99, strict=True)
    name: str = Field(default='', max_length=100)
    reviewer: str = Field(min_length=1, max_length=80)
    note: str = Field(default='', max_length=1000)
    time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    reset: bool = False


@router.post('/api/datasets/{identifier}/players/{pid}/track-correction')
def correct_player_track(identifier, pid, payload: TrackCorrection):
    _, m = _analysed(identifier)
    return MC.correct_track(m, pid, **payload.model_dump())


@router.get('/api/datasets/{identifier}/match/event-map')
def event_map(identifier, player: str):
    _, m = _analysed(identifier)
    if not any(p['player_id'] == player for p in m['players']):
        raise FileNotFoundError('Unknown player')
    return S.clean_json(ER.map_events(identifier, player))


class EventOutcome(BaseModel):
    outcome: str = Field(max_length=40)
    reviewer: str = Field(min_length=1, max_length=80)
    notes: str = Field(default='', max_length=1000)


@router.post('/api/datasets/{identifier}/match/events/{event_id}/review')
def review_event_outcome(identifier, event_id, payload: EventOutcome):
    _analysed(identifier)
    return ER.save_review(identifier, event_id, payload.outcome, payload.reviewer, payload.notes)


@router.get('/api/datasets/{identifier}/match/events')
def match_events(identifier, player: str = '', kind: str = ''):
    d, _ = _analysed(identifier)
    events = S.read_json(d / 'events.json', {'events': []})['events']
    if player:
        events = [e for e in events if e.get('identity') == player]
    if kind:
        events = [e for e in events if e.get('type') == kind]
    else:
        events = [e for e in events if e.get('type') not in ('touch', 'pressure')]
    return {'events': events[:2000], 'note': 'Touch and pressure events are omitted unless requested by kind.'}


@router.get('/api/datasets/{identifier}/match/boxes/{player_id}')
def player_boxes(identifier, player_id: str):
    """Image boxes of one identity over time, for drawing it on the source video."""
    d, m = _analysed(identifier)
    tracks = S.load_tracks(identifier)
    columns = ['time_s', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
    if set(columns).issubset(tracks.columns):
        q = tracks[tracks.player_id.eq(player_id)].dropna(subset=columns).sort_values('time_s')
        rows = q[columns].round(2).to_numpy().tolist()
    else:
        rows = []
    return {'width': m.get('width'), 'height': m.get('height'), 'source_offset_s': m.get('source_offset_s', 0),
            'boxes': rows}


def _tracks_for(d, player_id):
    info = S.read_json(d / 'identities.json', {})
    return [t for seg, v in info.get('segments', {}).items() if v.get('identity') == player_id
            for t in info.get('segment_tracks', {}).get(seg, [])]


def _crop_archive(d, m, track):
    """(crops.zip holding this track's thumbnails, the track's name inside it). A whole match's
    tracks are 'h1:<track>' / 'h2:<track>' and live in that half's archive."""
    if m.get('halves') and len(track) > 3 and track[0] == 'h' and track[2] == ':':
        return S.dataset_dir(m['halves'][int(track[1]) - 1]) / 'crops.zip', track[3:]
    return d / 'crops.zip', track


@router.get('/api/datasets/{identifier}/match/crops/{player_id}')
def player_crops(identifier, player_id: str, limit: int = 12):
    d, m = _analysed(identifier)
    by_archive = {}
    for t in _tracks_for(d, player_id):
        path, inner = _crop_archive(d, m, t)
        by_archive.setdefault(path, {})[inner] = t
    names = []
    for path, tracks in by_archive.items():
        if path.is_file():
            with zipfile.ZipFile(path) as archive:
                names += [(tracks[n.split('/')[0]], n.split('/', 1)[1]) for n in archive.namelist() if n.split('/')[0] in tracks]
    step = max(1, len(names) // max(1, limit))
    return {'crops': [f'/api/datasets/{identifier}/match/crop/{t}/{n}' for t, n in names[::step][:limit]]}


@router.get('/api/datasets/{identifier}/match/crop/{track}/{name}')
def crop(identifier, track: str, name: str):
    d, m = _analysed(identifier)
    path, inner = _crop_archive(d, m, track)
    with zipfile.ZipFile(path) as archive:
        try:
            data = archive.read(f'{inner}/{name}')
        except KeyError:
            raise FileNotFoundError('Crop unavailable')
    return Response(data, media_type='image/jpeg')


class TeamNames(BaseModel):
    A: str = Field(min_length=1, max_length=60)
    B: str = Field(min_length=1, max_length=60)


@router.post('/api/datasets/{identifier}/match/team-names')
def team_names(identifier, payload: TeamNames):
    from . import match_pipeline as MP
    _analysed(identifier)
    return MP.rename_teams(identifier, payload.model_dump())


class IdentityEdit(BaseModel):
    source: str
    target: str = Field(min_length=1, max_length=40)


@router.post('/api/datasets/{identifier}/match/identities')
def edit_identity(identifier, payload: IdentityEdit):
    """Merge one identity into another (or rename it); re-runs the CPU analysis stages as a job."""
    from .app import start_job
    from . import match_pipeline as MP
    d, _ = _analysed(identifier)
    info = S.read_json(d / 'identities.json', {})
    segments = [s for s, v in info.get('segments', {}).items() if v.get('identity') == payload.source]
    if not segments:
        raise ValueError('Unknown identity')
    confirmed = S.read_json(d / 'confirmed_identities.json', {})
    confirmed.update({s: payload.target for s in segments})
    S.write_json(d / 'confirmed_identities.json', confirmed)
    return start_job('identity-update', MP.postprocess, identifier=identifier)


class IdentityName(BaseModel):
    source: str = Field(min_length=1, max_length=40)
    number: int | None = Field(default=None, ge=1, le=99, strict=True)
    not_player: bool = False            # a referee, coach or other non-player: no player statistics
    same_as: str | None = Field(default=None, min_length=1, max_length=40)   # the same person as this player

    @model_validator(mode='after')
    def one_choice(self):
        if sum([self.number is not None, self.not_player, self.same_as is not None]) != 1:
            raise ValueError('Give a shirt number, mark the player as not a player, or choose who he is')
        return self


class IdentityNames(BaseModel):
    names: list[IdentityName] = Field(min_length=1, max_length=80)


def _identity_segments(d):
    info = S.read_json(d / 'identities.json', {})
    out = {}
    for s, v in info.get('segments', {}).items():
        out.setdefault(v.get('identity'), []).append(s)
    return out


def _person_name(d, m, team, segments):
    """A reviewer name for someone without a shirt number: '<team>-P<k>', unused in this match."""
    used = set()
    for half in (m.get('halves') or [m['id']]):
        used |= set(S.read_json(S.dataset_dir(half) / 'confirmed_identities.json', {}).values())
    used |= set(segments)
    k = 1
    while any(u.endswith(f'-P{k}') for u in used):
        k += 1
    return f'{team}-P{k}'


def _write_names(identifier, names):
    """Store {player id: reviewer name} per footage segment; returns the halves to re-analyse.

    On a whole match the names go to the half each segment comes from (in that half's kit
    group), so they survive re-analysis of the half and rebuilding of the match."""
    from . import match_merge as MM
    d, m = _analysed(identifier)
    segments = _identity_segments(d)
    changed = set()
    for source, name in names.items():
        if source not in segments:
            raise ValueError(f'Unknown player {source}')
        if m.get('halves'):
            parts, kits = MM.half_segments(identifier, source)
            for half_id, (segs, half) in parts.items():
                path = S.dataset_dir(half_id) / 'confirmed_identities.json'
                confirmed = S.read_json(path, {})
                confirmed.update({s: MM.half_name(name, half, kits) for s in segs})
                S.write_json(path, confirmed)
                changed.add(half_id)
        else:
            confirmed = S.read_json(d / 'confirmed_identities.json', {})
            confirmed.update({s: name for s in segments[source]})
            S.write_json(d / 'confirmed_identities.json', confirmed)
            changed.add(identifier)
    return changed


def _reanalyse(identifier, halves, progress=lambda *a: None):
    """Re-run post-processing of the changed halves, then rebuild the whole match."""
    from . import match_merge as MM
    from . import match_pipeline as MP
    m = S.read_json(S.dataset_dir(identifier) / 'manifest.json')
    if not m.get('halves'):
        return MP.postprocess(identifier, progress)
    for i, half in enumerate(sorted(halves)):
        progress(.05 + .4 * i / max(len(halves), 1), f'Re-analysing {half}')
        MP.postprocess(half)
    return MM.build(identifier, lambda f, msg: progress(.5 + f / 2, msg))


@router.post('/api/datasets/{identifier}/match/identities/names')
def name_identities(identifier, payload: IdentityNames):
    """Give players shirt numbers (each keeps its team), mark them as not players, or say who
    they are (another player of the team); one re-analysis applies them all.

    The names are stored per footage segment (confirmed_identities.json of the half), so they
    survive re-analysis; a number already in use on that team joins that player.
    """
    from .app import start_job
    d, m = _analysed(identifier)
    segments = _identity_segments(d)
    names = {}
    for n in payload.names:
        team = n.source.split('-', 1)[0]
        if n.source not in segments or team not in ('A', 'B'):
            raise ValueError(f'Unknown player {n.source}')
        if n.not_player:
            names[n.source] = MI.NOT_A_PLAYER
        elif n.number is not None:
            names[n.source] = f'{team}-{n.number}'
        else:
            target = n.same_as
            if target not in segments or target.split('-', 1)[0] != team or target == n.source:
                raise ValueError(f'Choose another player of the same team for {n.source}')
            suffix = target.split('-', 1)[1]
            if suffix.isdigit() or suffix == 'GK' or suffix.startswith('P'):
                names[n.source] = target                      # he joins that player's name
            else:
                person = names.get(target) or _person_name(d, m, team, names.values())
                names[n.source] = names[target] = person      # two unnamed: one new person
    changed = _write_names(identifier, names)
    return start_job('identity-update', _reanalyse, identifier=identifier, halves=changed)


class IdentityUndo(BaseModel):
    identity: str = Field(min_length=1, max_length=40)


@router.post('/api/datasets/{identifier}/match/identities/undo')
def undo_identity_name(identifier, payload: IdentityUndo):
    """Remove the names a reviewer gave to this player's segments (identity 'NONE': every segment
    marked as not a player); the model's naming returns."""
    from .app import start_job
    from . import match_merge as MM
    d, m = _analysed(identifier)
    halves = [(identifier, 1)] if not m.get('halves') else [(h, i + 1) for i, h in enumerate(m['halves'])]
    kits = m.get('kit_mapping_second_half') or {'A': 'A', 'B': 'B'}
    changed = set()
    for half_id, half in halves:
        name = payload.identity if not m.get('halves') else MM.half_name(payload.identity, half, kits)
        path = S.dataset_dir(half_id) / 'confirmed_identities.json'
        confirmed = S.read_json(path, {})
        kept = {s: i for s, i in confirmed.items() if i != name}
        if len(kept) != len(confirmed):
            S.write_json(path, kept)
            changed.add(half_id)
    if not changed:
        raise ValueError('No reviewer name to undo for this player')
    return start_job('identity-update', _reanalyse, identifier=identifier, halves=changed)
