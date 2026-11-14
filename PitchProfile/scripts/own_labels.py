"""Score an analysed match's events against our own labels of one clip (CPU only, seconds).

The labels are a CSV sheet, one row per action seen in the clip:
  time     from the start of the clip, as mm:ss, mm:ss.s or seconds
  half     1 or 2
  action   pass, cross, shot, header, tackle, interception, throw-in, carry, clearance or block
  team     the team's name as the app shows it, or the kit group A or B
  jersey   shirt number, or blank / unknown when it cannot be read
  name     the player's name (for reading the sheet; not scored)
  notes    anything else
Each label is matched to the nearest unused event of the same action within --tolerance seconds
(rule events, and video-model events above their confidence threshold, as the app's statistics use).
Reported per action and overall: found (recall), precision against our events inside the labelled
clip, the team right when found, credited to a player when found, and the right player (team and
shirt number) among the labels with a readable number.

Report: evidence/own_labels_<tag>.json (the app's Model performance page shows the newest).

Usage (from PitchProfile/):
  python scripts/own_labels.py clip --dataset <dataset id> --start 12:00 --length 10:00
      cuts the clip to label from the dataset's own video (720p at most, no sound, about 150 MB)
      into .runtime/ and prints the matching check command
  python scripts/own_labels.py template labels/own_labels.csv
  python scripts/own_labels.py check --dataset <dataset id> --labels labels/own_labels.csv
      --clip-start 12:00 [--clip-length 10:00] [--tolerance 1.0] [--tag clip1]
--clip-start is where the clip starts on the dataset's video, which is the app's video clock; `clip`
cuts from that same video at that same time, so a label's time in the clip plus the start is the
time the app gives the event.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_profiler import storage as S  # noqa: E402

COLUMNS = ['time', 'half', 'action', 'team', 'jersey', 'name', 'notes']
# Our event types for each labelled action; a rule pass marked as a cross also counts as a cross.
ACTIONS = {'pass': ('pass', 'high_pass'), 'cross': ('cross',), 'shot': ('shot',), 'header': ('header',),
           'tackle': ('tackle',), 'interception': ('interception',), 'throw-in': ('throw_in',),
           'carry': ('carry', 'dribble'), 'clearance': ('clearance',), 'block': ('block',)}
ALIASES = {'throw in': 'throw-in', 'throwin': 'throw-in', 'throw_in': 'throw-in', 'drive': 'carry',
           'dribble': 'carry', 'high pass': 'pass', 'long pass': 'pass', 'lofted pass': 'pass'}
UNKNOWN = {'', '?', 'unknown', 'n/a', 'na', '-'}
TOLERANCE_S = 1.0                   # as the FOOTPASS crediting check (evaluate_identity_footpass.py)


def parse_time(text):
    """'12:34', '12:34.5', '1:02:03' or '754.5' -> seconds."""
    text = str(text).strip()
    if not re.fullmatch(r'\d+(:\d{1,2})*(\.\d+)?', text):
        raise ValueError(f'Unreadable time {text!r}: use mm:ss or seconds')
    seconds = 0.0
    for part in text.split(':'):
        seconds = seconds * 60 + float(part)
    return seconds


def action_of(text):
    a = ALIASES.get(str(text).strip().lower(), str(text).strip().lower())
    if a not in ACTIONS:
        raise ValueError(f'Unknown action {text!r}: use one of {", ".join(ACTIONS)}')
    return a


def team_of(text, teams):
    """A label's team -> kit group 'A' / 'B', or None when blank or not recognised."""
    t = str(text or '').strip()
    if t.lower() in UNKNOWN:
        return None
    if t.upper() in ('A', 'B'):
        return t.upper()
    for key, info in (teams or {}).items():
        if t.lower() == str(info.get('name', '')).strip().lower():
            return key
    return None


def jersey_of(text):
    t = str(text or '').strip().lstrip('#')
    return int(t) if t.isdigit() else None


def read_labels(path, manifest, clip_start_s=0.0):
    """The sheet's rows with their time on the dataset's video clock (clip start + time in the clip),
    kit group and shirt number. 'half' is checked against a half's dataset; it does not move times."""
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            r = {k.strip().lower(): (v or '').strip() for k, v in r.items() if k}
            if not any(r.values()):
                continue
            try:
                half = int(r.get('half') or manifest.get('half') or 1)
                time_s = clip_start_s + parse_time(r['time'])
                action = action_of(r['action'])
            except (KeyError, ValueError) as e:
                raise ValueError(f'{path}, row {i}: {e}') from None
            if not manifest.get('halves') and manifest.get('half') and half != manifest['half']:
                raise ValueError(f'{path}, row {i}: half {half}, but this dataset is half {manifest["half"]}')
            rows.append({'row': i, 'time_s': round(time_s, 2), 'half': half, 'action': action,
                         'team_text': r.get('team', ''), 'team': team_of(r.get('team'), manifest.get('teams')),
                         'jersey': jersey_of(r.get('jersey')), 'name': r.get('name', ''), 'notes': r.get('notes', '')})
    return sorted(rows, key=lambda r: r['time_s'])


def counted(events):
    """Events the statistics count: rule events, and video-model events above their threshold."""
    return [e for e in events if e.get('time_s') is not None
            and (e.get('source') != 'action_spotter' or e.get('confident'))]


def is_action(event, action):
    return event.get('type') in ACTIONS[action] or (action == 'cross' and event.get('type') == 'pass'
                                                    and bool(event.get('cross')))


def match(labels, events, tolerance=TOLERANCE_S):
    """Each label (in time order) takes the nearest unused event of its action within the tolerance."""
    used, out = {}, []                  # used: event index -> the action of the label it matched
    for lab in labels:
        best = None
        for j, e in enumerate(events):
            if j in used or not is_action(e, lab['action']):
                continue
            dt = abs(float(e['time_s']) - lab['time_s'])
            if dt <= tolerance and (best is None or dt < best[0]):
                best = (dt, j)
        if best is not None:
            used[best[1]] = lab['action']
        out.append((lab, None if best is None else events[best[1]]))
    return out, used


def _share(n, d):
    return round(n / d, 3) if d else None


def score(labels, events, players, window, tolerance=TOLERANCE_S):
    """Per action and overall shares, and one row per label for checking by hand."""
    events = counted(events)
    pairs, used = match(labels, events, tolerance)
    jersey = {p['player_id']: (p.get('jersey'), p.get('team_key') or p.get('team')) for p in players}
    rows = []
    for lab, e in pairs:
        ident = e.get('identity') if e else None
        number, _ = jersey.get(ident, (None, None))
        team_ok = None if not e or lab['team'] is None else e.get('team') == lab['team']
        rows.append({**{k: lab[k] for k in ('row', 'time_s', 'action', 'team', 'jersey')},
                     'found': e is not None, 'event_time_s': e.get('time_s') if e else None,
                     'event_type': e.get('type') if e else None,
                     'event_model': ('video' if e.get('source') == 'action_spotter' else 'rules') if e else None,
                     'event_team': e.get('team') if e else None, 'event_player': ident, 'event_jersey': number,
                     'team_ok': team_ok, 'credited': bool(ident),
                     'right_player': None if lab['jersey'] is None else bool(
                         e and ident and number == lab['jersey'] and team_ok is not False)})
    lo, hi = window

    def summary(action=None):
        mine = [r for r in rows if action is None or r['action'] == action]
        kinds = [action] if action else sorted({r['action'] for r in rows})
        ours = [j for j, e in enumerate(events) if lo <= float(e['time_s']) <= hi and any(is_action(e, k) for k in kinds)]
        hit = [j for j in ours if j in used and (action is None or used[j] == action)]
        found = [r for r in mine if r['found']]
        teamed = [r for r in found if r['team_ok'] is not None]
        numbered = [r for r in mine if r['right_player'] is not None]
        return {'labels': len(mine), 'found': _share(len(found), len(mine)),
                'events_in_clip': len(ours), 'precision': _share(len(hit), len(ours)),
                'team_right_when_found': _share(sum(r['team_ok'] for r in teamed), len(teamed)),
                'credited_when_found': _share(sum(r['credited'] for r in found), len(found)),
                'labels_with_jersey': len(numbered),
                'right_player': _share(sum(r['right_player'] for r in numbered), len(numbered))}

    return {'overall': summary(), 'actions': {a: summary(a) for a in sorted({r['action'] for r in rows})},
            'rows': rows}


def check(args):
    d = S.dataset_dir(args.dataset)
    m = S.read_json(d / 'manifest.json')
    start = parse_time(args.clip_start)
    labels = read_labels(args.labels, m, start)
    if not labels:
        raise SystemExit(f'No labels in {args.labels}')
    unknown_team = sorted({r['team_text'] for r in labels if r['team'] is None and r['team_text'].lower() not in UNKNOWN})
    if unknown_team:
        names = ', '.join(f"{k} = {v.get('name')}" for k, v in (m.get('teams') or {}).items())
        print(f'Team names not recognised (left unscored): {unknown_team}. This match: {names}')
    if args.clip_length:
        window = (start, start + parse_time(args.clip_length))
    else:
        window = (labels[0]['time_s'] - args.tolerance, labels[-1]['time_s'] + args.tolerance)
    events = S.read_json(d / 'events.json', {'events': []})['events']
    result = score(labels, events, m.get('players', []), window, args.tolerance)
    report = {'created': S.now(), 'dataset': args.dataset, 'title': m.get('title'),
              'labels_file': Path(args.labels).name, 'clip_start_s': start, 'window_s': [round(x, 2) for x in window],
              'tolerance_s': args.tolerance, **result}
    out = S.EVIDENCE / f'own_labels_{args.tag}.json'
    S.write_json(out, report)
    print(f"{'action':13}{'labels':>7}{'found':>8}{'precision':>11}{'team ok':>9}{'player':>8}")
    for name, a in [*result['actions'].items(), ('ALL', result['overall'])]:
        cell = lambda x: '-' if x is None else f'{x:.0%}'
        print(f"{name:13}{a['labels']:>7}{cell(a['found']):>8}{cell(a['precision']):>11}"
              f"{cell(a['team_right_when_found']):>9}{cell(a['right_player']):>8}")
    print(f'\nSaved {out}')


def clip(args):
    """Cut the clip to label from the dataset's video, frame-accurately (re-encoded, so the start is exact)."""
    import subprocess
    import imageio_ffmpeg
    d = S.dataset_dir(args.dataset)
    m = S.read_json(d / 'manifest.json')
    if not m.get('video'):
        raise SystemExit(f'{args.dataset} has no video attached')
    video = S.video_path(d, m)
    start, length = parse_time(args.start), parse_time(args.length)
    out = Path(args.out) if args.out else ROOT / '.runtime' / f'{args.dataset}_clip_{int(start)}s.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{start:.3f}',
                    '-i', str(video), '-t', f'{length:.3f}', '-an', '-vf', "scale=-2:'min(720,ih)'",
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(out)], check=True)
    print(f'Wrote {out} ({out.stat().st_size / 1e6:.0f} MB): {args.length} from {args.start} of {m.get("title", args.dataset)}')
    print(f'Score it once labelled: python scripts/own_labels.py check --dataset {args.dataset} '
          f'--labels labels/own_labels.csv --clip-start {args.start} --clip-length {args.length}')


def template(args):
    path = Path(args.path)
    if path.exists():
        raise SystemExit(f'{path} exists; not overwriting it')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(COLUMNS)
    print(f'Wrote {path}: columns {", ".join(COLUMNS)}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    k = sub.add_parser('clip', help="cut the clip to label from a dataset's video")
    k.add_argument('--dataset', required=True)
    k.add_argument('--start', required=True, help='mm:ss on the video clock')
    k.add_argument('--length', default='10:00')
    k.add_argument('--out', default=None)
    k.set_defaults(run=clip)
    t = sub.add_parser('template', help='write an empty labelling sheet')
    t.add_argument('path')
    t.set_defaults(run=template)
    c = sub.add_parser('check', help='score a dataset against a labelled sheet')
    c.add_argument('--dataset', required=True)
    c.add_argument('--labels', required=True)
    c.add_argument('--clip-start', default='0', help="the clip's start on the video clock (mm:ss or seconds)")
    c.add_argument('--clip-length', default=None, help='length of the clip, for precision (default: first to last label)')
    c.add_argument('--tolerance', type=float, default=TOLERANCE_S)
    c.add_argument('--tag', default='clip')
    c.set_defaults(run=check)
    args = p.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
