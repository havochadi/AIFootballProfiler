"""Build data.js for the PitchProfile demo site from one really analysed whole match.

Reads the app's output for the match (both halves merged) (match_stats.json, manifest.json, events.json,
tracks.csv.gz, ball.csv.gz, analysis.json) from the data drive and writes a compact
numbers-only script, so the demo contains no footage and no frames.

    python build_demo_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r'D:\CVDL Football Data\PitchProfile\data\datasets')
IDENTIFIER = 'sn-20160207-chelsea-manchester-united'      # both halves merged by the app (second half mirrored)
# Kit groups are anonymous in the pipeline. These names were confirmed by eye on a frame with the
# broadcast scoreboard ("CHE 0-0 MU"): red kit = group A = Manchester United, blue = group B = Chelsea.
TEAM_NAMES = {'A': ('Manchester United', 'Man United'), 'B': ('Chelsea', 'Chelsea')}
MATCH = {'title': 'Chelsea v Manchester United', 'competition': 'Premier League', 'date': '7 Feb 2016', 'scope': 'full match'}
# the app's joined video of both halves (95:20); video time equals the analysis clock
VIDEO = {'path': 'D:/CVDL Football Data/PitchProfile/data/datasets/sn-20160207-chelsea-manchester-united/'
                 'sn-20160207-chelsea-manchester-united.mkv', 'name': 'sn-20160207-chelsea-manchester-united.mkv'}
CLIPS = [(0, 'Kick-off'), (781, 'Open play'), (1256, 'Model-detected shot'),
         (2700, 'Second-half kick-off'), (4001, 'Model-detected shot'), (5034, 'Late in the match')]
CLIP_SECONDS = 75
LENGTHS = (2, 5, 10, 15, 20, 30, 45)
OUT = Path(__file__).with_name('data.js')
MIN_VISIBLE_S = 120          # the app's own threshold for profiling an appearance
MAIN_VISIBLE_S = 600         # shown by default: on screen for 10+ minutes (the app's threshold for style percentiles)
EVENT_TYPES = ['pass', 'carry', 'shot', 'tackle', 'interception', 'recovery', 'pressure', 'take_on',
               'clearance', 'cross', 'header', 'block']
OUTCOMES = {None: 0, 'complete': 1, 'intercepted': 2, 'unknown': 0}

POSITION = {'goalkeeper': 'Goalkeeper', 'centre_back': 'Centre-back', 'full_back': 'Full-back',
            'defensive_midfield': 'Defensive mid', 'central_midfield': 'Central mid',
            'attacking_midfield': 'Attacking mid', 'winger': 'Winger', 'centre_forward': 'Forward'}


def r(v, n=1):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), n)


def main():
    d = DATA / IDENTIFIER
    stats = json.load(open(d / 'match_stats.json'))
    man = json.load(open(d / 'manifest.json'))
    analysis = json.load(open(d / 'analysis.json'))
    events = json.load(open(d / 'events.json'))['events']
    tracks = pd.read_csv(d / 'tracks.csv.gz')
    ball = pd.read_csv(d / 'ball.csv.gz')
    info = {p['player_id']: p for p in man['players']}

    # ---- players -----------------------------------------------------------------------------
    players, index = [], {}
    unnumbered = {'A': 0, 'B': 0}
    for p in sorted(stats['players'], key=lambda p: (p['team'], -p['visible_seconds'])):
        if p['visible_seconds'] < MIN_VISIBLE_S:
            continue
        meta = info[p['identity']]
        short_team = TEAM_NAMES[p['team']][1]
        if meta.get('jersey') is not None:
            label = f"#{meta['jersey']}"
        elif p['role'] == 'goalkeeper':
            label = 'Goalkeeper'
        else:
            unnumbered[p['team']] += 1
            label = f"Unnumbered {unnumbered[p['team']]}"
        o, ph, pos, mv = p['on_ball'], p['physical'], p['positional'], p['movement']
        keys = ['touches', 'passes', 'passes_completed', 'pass_completion', 'progressive_passes', 'long_passes',
                'key_passes', 'crosses', 'carries', 'progressive_carries', 'dribbles', 'take_ons', 'shots',
                'receptions', 'touches_final_third', 'touches_in_box', 'interceptions', 'recoveries',
                'clearances', 'pressures', 'tackles', 'blocks', 'headers', 'turnovers', 'time_on_ball_s']
        index[p['identity']] = len(players)
        players.append({
            'id': p['identity'], 'team': p['team'], 'label': label,
            'name': f"{short_team} {label}" if label.startswith('#') else f"{short_team} {label.lower()}",
            'gk': p['role'] == 'goalkeeper',
            'position': POSITION.get(meta.get('position_group'), 'Outfield'),
            'visible_s': r(p['visible_seconds'], 0), 'short': p['visible_seconds'] < MAIN_VISIBLE_S,
            'distance_m': r(ph['distance_m'], 0), 'top_kmh': r(ph['top_speed_kmh']), 'sprints': ph['sprints'],
            'zones_s': {k: r(v, 0) for k, v in ph['zone_seconds'].items()},
            'mean': [r(pos['mean_x']), r(pos['mean_y'])],
            'thirds': [r(pos['defensive_third'], 3), r(pos['middle_third'], 3), r(pos['attacking_third'], 3)],
            'lanes': [r(pos['left_lane'], 3), r(pos['central_lane'], 3), r(pos['right_lane'], 3)],
            'box_share': r(pos['box_share'], 3),
            'heat': [[round(v, 4) for v in row] for row in pos['heatmap']],
            'on': {k: r(o.get(k), 2) for k in keys},
            'per90': {k: r(v, 1) for k, v in p['per90_visible'].items() if k in keys},
            'runs': {k: mv.get(k) for k in ('high_intensity_runs', 'runs_in_behind', 'box_runs', 'overlaps',
                                           'pressing_runs', 'recovery_runs')},
        })

    # ---- events (normalised so each team attacks to the right) ---------------------------------
    rows, unattributed = [], 0
    for e in events:
        kind = 'take_on' if e['type'] in ('take_on', 'dribble') else e['type']
        if kind not in EVENT_TYPES:
            continue
        if kind == 'take_on' and e['type'] == 'dribble':
            continue          # a dribble is a take-on that kept the ball; count the take-on once
        sign = e.get('attack_sign') or 1
        x, y, ex, ey = e['x'], e['y'], e.get('end_x'), e.get('end_y')
        if x is None or y is None:
            continue          # no calibrated pitch position, so it cannot be mapped
        if sign < 0:
            x, y = 105 - x, 68 - y
            ex = None if ex is None else 105 - ex
            ey = None if ey is None else 68 - ey
        who = index.get(e.get('identity'), -1)
        unattributed += who < 0
        rows.append([EVENT_TYPES.index(kind), r(e['time_s'], 1), 0 if e['team'] == 'A' else 1, who,
                     r(x), r(y), r(ex), r(ey), OUTCOMES.get(e.get('outcome'), 0)])
    rows.sort(key=lambda row: row[1])

    # ---- timeline: touches per team per 5-minute bucket -------------------------------------------
    n_buckets = max(1, round(man['duration_seconds'] / 300))      # the last few seconds fold into the final block
    buckets = [[0, 0] for _ in range(n_buckets)]
    for e in events:
        if e['type'] == 'touch':
            buckets[min(n_buckets - 1, int(e['time_s'] // 300))][0 if e['team'] == 'A' else 1] += 1

    # ---- tracking clips: boxes in the video image plus pitch positions, for the overlay ---------
    ball_by_sample = {int(f): (cx, cy, x, y) for f, cx, cy, x, y in zip(ball['sample'], ball.cx, ball.cy, ball.x, ball.y)}
    clips = []
    for start, label in CLIPS:
        win = tracks[(tracks.time_s >= start) & (tracks.time_s < start + CLIP_SECONDS)]
        frames, tids = [], {}
        for f, g in win.groupby('frame'):
            rows_ = []
            for pid, tid, bx, by, bw, bh, x, y, ok in zip(g.player_id, g.track_id, g.bbox_x, g.bbox_y, g.bbox_w,
                                                          g.bbox_h, g.x, g.y, g.calibration_valid):
                mapped = bool(ok) and not np.isnan(x)
                rows_.append([index.get(pid, -1), 0 if pid[0] == 'A' else 1, int(bx), int(by), int(bw), int(bh),
                              r(x) if mapped else None, r(y) if mapped else None, tids.setdefault(tid, len(tids))])
            bl = ball_by_sample.get(int(f))
            frames.append({'t': r(g.time_s.iloc[0], 2), 'r': rows_,
                           'b': None if bl is None else [int(bl[0]), int(bl[1]), r(bl[2]), r(bl[3])]})
        clips.append({'start_s': start, 'seconds': CLIP_SECONDS, 'label': label, 'frames': frames})

    # ---- how much footage is needed: identified player-time in the first N minutes of every analysed half ----
    halves = [p for p in sorted(DATA.glob('sn-*-h[12]')) if (p / 'tracks.csv.gz').exists()]
    per_half = {n: {'ge120': [], 'ge600': []} for n in LENGTHS}
    for h in halves:
        cal = pd.read_csv(h / 'tracks.csv.gz', usecols=['time_s', 'player_id', 'calibration_valid'])
        cal = cal[cal.calibration_valid == 1]
        for n in LENGTHS:
            seen = cal[cal.time_s < n * 60].groupby('player_id').size() * 0.08
            per_half[n]['ge120'].append(int((seen >= MIN_VISIBLE_S).sum()))
            per_half[n]['ge600'].append(int((seen >= 600).sum()))
    length = [{'minutes': n, 'median': float(np.median(v['ge120'])), 'min': min(v['ge120']), 'max': max(v['ge120']),
               'median600': float(np.median(v['ge600'])), 'halves': len(halves)} for n, v in per_half.items()]

    parts = [json.load(open(DATA / h / 'analysis.json')) for h in man['halves']]
    post = analysis.get('postprocess', {})
    team_stats = {t['team']: t for t in stats['teams']}

    def team(k):
        t, on = team_stats[k], team_stats[k]['on_ball']
        mine = [p for p in players if p['team'] == k]
        return {'name': TEAM_NAMES[k][0], 'short': TEAM_NAMES[k][1], 'possession': r(t['possession_share'], 3),
                'touches': on['touches'], 'passes': on['passes'], 'completed': on['passes_completed'],
                'completion': r(on['pass_completion'], 3), 'progressive': on['progressive_passes'],
                'shots': on['shots'], 'key_passes': on['key_passes'], 'interceptions': on['interceptions'],
                'recoveries': on['recoveries'], 'pressures': on['pressures'], 'clearances': on['clearances'],
                'tackles': r(on['tackles'], 1), 'blocks': r(on['blocks'], 1), 'dribbles': on['dribbles'],
                'carries': on['carries'], 'final_third': on['touches_final_third'], 'box': on['touches_in_box'],
                'distance_km': r(sum(p['distance_m'] for p in mine) / 1000, 1),
                'players': len(mine)}

    out = {
        'match': {**MATCH, 'duration_s': man['duration_seconds'], 'video': VIDEO,
                  'live_s': r(stats['live_seconds'], 0), 'pitch_view': r(man['coverage']['pitch_view_share'], 3),
                  'ball_seen': r(man['coverage']['ball_observed_share'], 3), 'identities': len(info),
                  'profiled': sum(not p['short'] for p in players), 'profiled_all': len(players), 'events': len(rows), 'unattributed_events': int(unattributed),
                  'attack': {'A': 'right', 'B': 'left'}},
        'run': {'halves': len(parts), 'samples': sum(a['detection']['samples'] for a in parts),
                'hz': parts[0]['detection']['sample_hz'], 'cuts': sum(a['detection']['shots'] for a in parts),
                'detect_wall_s': r(sum(a['detection']['wall_seconds'] for a in parts), 0), 'device': 'NVIDIA RTX 3090',
                'numbers_read': sum(a['postprocess'].get('jersey_numbers_read') or 0 for a in parts),
                'crops': sum((a.get('crops') or {}).get('crops') or 0 for a in parts),
                'cpu_wall_s': r(sum(a['postprocess']['wall_seconds'] for a in parts), 0)},
        'teams': {'A': team('A'), 'B': team('B')},
        'players': players,
        'events': {'types': EVENT_TYPES, 'rows': rows},
        'timeline': buckets,
        'clips': clips,
        'length': length,
    }
    OUT.write_text('window.DEMO = ' + json.dumps(out, separators=(',', ':')) + ';\n', encoding='utf-8')
    print(f'{OUT} {OUT.stat().st_size / 1024:.0f} KB; {len(players)} players, {len(rows)} events, '
          f'{len(clips)} clips, {sum(len(c["frames"]) for c in clips)} frames')


if __name__ == '__main__':
    main()
