"""Group D's measured scores in one place, read from the evidence reports of the machine that ran them.

gather(evidence) reads whichever of these reports exist and lists the rest under 'missing':
  action_spotter_<set>_<tag>.json            evaluate_spotter_labelled.py: the spotter on held-out games;
                                             tags published, finetuned and the two ablation arms
  identity_footpass_variant_current.json     evaluate_identity_footpass.py through measure_shot_credit.py:
  identity_footpass_variant_throwinline.json actions found and credited to the right player (FOOTPASS)
  credit_reasons.json                        credit_reasons.py: how spotted actions were credited, or why not
  spotter_changes.json                       check_spotter_changes.py: speed, half overlap, throw-in team
  event_logic_soccertrack.json               evaluate_events_soccertrack.py: the event rules on true positions
  own_labels_<tag>.json                      scripts/own_labels.py: our events against our own labelled clip
The result feeds the app page (event_live.py, static/model_performance.js) and
scripts/gather_model_performance.py, which saves it as evidence/model_performance.json.
"""
from __future__ import annotations

from pathlib import Path

from . import storage as S

HELD_OUT = {'footpass_val': 'FOOTPASS, held-out games', 'bas_test': 'SoccerNet BAS, held-out games'}
VERSIONS = [('published', 'Public model'), ('finetuned', 'Ours (in the app)'),
            ('tdeed_ablation_partial', 'Retrained, with the trick'),
            ('tdeed_ablation_hard', 'Retrained, without the trick')]
CREDIT = {'current': ('identity_footpass_variant_current.json', 'identity_footpass_current.json'),
          'throwinline': ('identity_footpass_variant_throwinline.json',)}
REASONS = ('no ball', 'ball away from players', 'team not on screen', 'no team')
# Decision rules of README section 8, applied to the numbers once they exist.
SCORE_CHANGE_MAX = .001             # read-ahead must leave every score as it was
HALF_OVERLAP_MAP_MAX = .005         # half overlap may move mean AP by less than this on both sets


def _read(path):
    try:
        return S.read_json(path)
    except (ValueError, OSError):
        return None


def spotter(evidence):
    """Mean and per-class average precision of each spotter version on each held-out set."""
    sets, used = {}, []
    for key, name in HELD_OUT.items():
        means, classes = {}, {}
        for tag, _ in VERSIONS:
            file = f'action_spotter_{key}_{tag}.json'
            r = _read(evidence / file)
            if not r:
                continue
            used.append(file)
            means[tag] = r.get('mean_average_precision')
            for c, v in (r.get('classes') or {}).items():
                if v.get('labels'):
                    classes.setdefault(c, {'labels': v['labels']})[tag] = v.get('average_precision')
        if means:
            sets[key] = {'name': name, 'mean_average_precision': means, 'average_precision': classes}
    return {'versions': [{'tag': t, 'name': n} for t, n in VERSIONS], 'sets': sets}, used


def crediting(evidence):
    """Per FOOTPASS action: found, team right, credited to the right named player, to the right track."""
    out, used = {}, []
    for key, files in CREDIT.items():
        for file in files:
            r = _read(evidence / file)
            if r and (r.get('overall') or {}).get('actions'):
                out[key] = {'file': file, 'actions': r['overall']['actions']}
                used.append(file)
                break
    return out, used


def reasons(evidence):
    r = _read(evidence / 'credit_reasons.json')
    if not r:
        return None, []
    overall = r.get('overall') or {}
    return {'reasons': r.get('reasons') or list(REASONS), 'datasets': len(r.get('datasets') or []),
            'rule_off': overall.get('rule_off') or {}, 'rule_on': overall.get('rule_on') or {}}, ['credit_reasons.json']


def speed(evidence):
    r = _read(evidence / 'spotter_changes.json')
    if not r:
        return None, []
    return {'timing': r.get('timing'), 'held_out': r.get('held_out') or {}}, ['spotter_changes.json']


def rules(evidence):
    """The event rules on SoccerTrack's true positions, both halves pooled."""
    r = _read(evidence / 'event_logic_soccertrack.json')
    halves = list(((r or {}).get('halves') or {}).values())
    if not halves:
        return None, []
    out = {}
    for key in ('passes_any_player_1s', 'passes_same_player_1s', 'carries_vs_drive'):
        rows = [h[key] for h in halves if key in h]
        if rows:
            matched, predicted, annotated = (sum(x[k] for x in rows) for k in ('matched', 'predicted', 'annotated'))
            out[key] = {'matched': matched, 'predicted': predicted, 'annotated': annotated,
                        'precision': round(matched / max(predicted, 1), 3), 'recall': round(matched / max(annotated, 1), 3)}
    corr = [h['per_player_pass_counts']['pearson'] for h in halves if 'per_player_pass_counts' in h]
    if corr:
        out['pass_count_correlation'] = round(sum(corr) / len(corr), 3)
    return out, ['event_logic_soccertrack.json']


def own_labels(evidence):
    """The newest own_labels_<tag>.json report (scripts/own_labels.py check)."""
    reports = [(r.get('created') or '', p.name, r) for p in sorted(Path(evidence).glob('own_labels_*.json'))
               if (r := _read(p))]
    if not reports:
        return None, []
    _, name, r = max(reports)
    return r, [name]


def _pct(x):
    return f'{x:.0%}' if isinstance(x, (int, float)) else '?'


def changes(g):
    """The six Group D changes with their measured effect and, where the rule allows, a verdict."""
    sets = g['spotter']['sets']
    rows = [
        {'id': 'C1', 'name': 'Tackle leak', 'kind': 'bug fix',
         'before': 'If the spotter did not run, a tackle rule only 6-14% right fed the stats',
         'after': 'That rule never feeds the stats', 'result': 'Covered by a test', 'state': 'done'},
        {'id': 'C2', 'name': 'Tidy-ups', 'kind': 'clean-up',
         'before': 'One function written twice; one function nobody used',
         'after': 'Both removed; nothing behaves differently', 'result': 'All tests pass', 'state': 'done'},
    ]
    have = [s for s in sets.values() if {'tdeed_ablation_partial', 'tdeed_ablation_hard'} <= set(s['mean_average_precision'])]
    c3 = {'id': 'C3', 'name': 'Training trick (partial-label loss)', 'kind': 'experiment',
          'before': 'No proof the trick itself helped', 'after': 'Trained with and without it, same settings'}
    if have:
        parts = [f"{s['name'].split(',')[0]}: {s['mean_average_precision']['tdeed_ablation_partial']:.3f} with, "
                 f"{s['mean_average_precision']['tdeed_ablation_hard']:.3f} without" for s in have]
        gaps = [s['mean_average_precision']['tdeed_ablation_partial'] - s['mean_average_precision']['tdeed_ablation_hard']
                for s in have]
        verdict = ('the trick helps' if all(x >= .01 for x in gaps) else
                   'too close to call with one run each' if all(abs(x) < .01 for x in gaps) else
                   'no gain from the trick: plain labels did as well or better' if all(x < .01 for x in gaps) else
                   'mixed: see the table')
        c3.update(result='; '.join(parts), state='measured', verdict=verdict)
    else:
        c3.update(result='Waiting for the overnight training run', state='waiting')
    rows.append(c3)

    rs = g.get('reasons')
    c4a = {'id': 'C4a', 'name': 'Why no player', 'kind': 'diagnostic',
           'before': 'About 1 in 4 spotted passes and carries got no player, with no record of why',
           'after': 'Every uncredited action records why'}
    pas = (rs or {}).get('rule_off', {}).get('pass')
    if pas and pas.get('credited') is not None and pas['credited'] < 1:
        missing = 1 - pas['credited']
        top = sorted(((pas.get(r, 0) / missing, r) for r in REASONS), reverse=True)[:3]
        c4a.update(result='Uncredited passes: ' + ', '.join(f'{r} {_pct(s)}' for s, r in top), state='measured')
    else:
        c4a.update(result='Waiting for credit_reasons.py', state='done')
    rows.append(c4a)

    cr = g['crediting']
    c4b = {'id': 'C4b', 'name': 'Throw-ins by the touchline', 'kind': 'switch, off',
           'before': 'Throw-ins credited to the right player worst of all',
           'after': 'The one teammate on the touchline gets it'}
    a = (cr.get('current') or {}).get('actions', {}).get('Throw-in')
    b = (cr.get('throwinline') or {}).get('actions', {}).get('Throw-in')
    if a and b:
        up = (b.get('credited_correctly') or 0) > (a.get('credited_correctly') or 0)
        kept = (b.get('accuracy_when_credited') or 0) >= (a.get('accuracy_when_credited') or 0) - 1e-9
        c4b.update(result=f"Right player {_pct(a.get('credited_correctly'))} -> {_pct(b.get('credited_correctly'))}; "
                          f"right when credited {_pct(a.get('accuracy_when_credited'))} -> {_pct(b.get('accuracy_when_credited'))}",
                   state='measured', verdict='switch on' if up and kept else 'leave off')
    else:
        c4b.update(result='Waiting for measure_shot_credit.py throwinline and current', state='waiting')
    rows.append(c4b)

    timing = ((g.get('speed') or {}).get('timing') or {}).get('settings') or {}
    held = (g.get('speed') or {}).get('held_out') or {}
    ra, half = timing.get('read_ahead'), timing.get('read_ahead_half_overlap')
    c5a = {'id': 'C5a', 'name': 'Read the video ahead', 'kind': 'speed-up, on',
           'before': 'Video reading and the model took turns', 'after': 'They run at the same time'}
    if ra:
        change = ra.get('largest_score_change')
        same = change is None or change < SCORE_CHANGE_MAX
        c5a.update(result=f"x{ra.get('speed_up')} faster, {ra.get('minutes_per_half')} min per half; "
                          f"largest score change {change if change is not None else 0}",
                   state='measured', verdict='keep on' if same else 'scores changed: check before the demo')
    else:
        c5a.update(result='Same results checked on the Mac; speed waiting for check_spotter_changes.py', state='done')
    rows.append(c5a)

    c5b = {'id': 'C5b', 'name': 'Half the clip overlap', 'kind': 'switch, off',
           'before': 'Each frame looked at 4 times', 'after': 'Each frame looked at 2 times'}
    moves = {k: v['mean_average_precision']['change'] for k, v in held.items() if 'mean_average_precision' in v}
    if half and moves:
        small = len(moves) == len(HELD_OUT) and all(abs(x) < HALF_OVERLAP_MAP_MAX for x in moves.values())
        c5b.update(result=f"x{half.get('speed_up')} faster; mean AP change " +
                          ', '.join(f"{HELD_OUT.get(k, k).split(',')[0]} {v:+.3f}" for k, v in moves.items()),
                   state='measured', verdict='switch on' if small else 'team decides: speed against accuracy')
    else:
        c5b.update(result='Waiting for check_spotter_changes.py', state='waiting')
    rows.append(c5b)

    c6 = {'id': 'C6', 'name': 'Throw-in team from the last touch', 'kind': 'switch, off',
          'before': "Throw-in team from the spotter's own guess", 'after': 'The team that did not touch it last'}
    teams = {k: v['throw_in_team'] for k, v in held.items() if v.get('throw_in_team')}
    if teams:
        best = {}
        for k, t in teams.items():
            look = {w: v for w, v in t.items() if w.startswith('last_touch') and v is not None}
            w = max(look, key=look.get) if look else None
            best[k] = (t.get('spotter'), w, look.get(w))
        wins = len(best) == len(HELD_OUT) and all(v is not None and s is not None and v > s for s, _, v in best.values())
        c6.update(result='; '.join(f"{HELD_OUT.get(k, k).split(',')[0]}: spotter {_pct(s)}, "
                                   f"last touch ({(w or '-').replace('last_touch_', '').replace('s', ' s')}) {_pct(v)}"
                                   for k, (s, w, v) in best.items()),
                  state='measured', verdict='switch on' if wins else 'leave off')
    else:
        c6.update(result='Waiting for check_spotter_changes.py', state='waiting')
    rows.append(c6)
    return rows


def headline(g):
    """The few numbers for the top of the page."""
    out = {}
    fp = g['spotter']['sets'].get('footpass_val', {}).get('mean_average_precision', {})
    if 'finetuned' in fp:
        out['spotter_map'] = {'value': fp['finetuned'], 'from': fp.get('published')}
    p = (g['crediting'].get('current') or {}).get('actions', {}).get('Pass')
    if p:
        out['pass_found'] = p.get('found')
        out['pass_right_player'] = p.get('credited_correctly')
        out['pass_right_track'] = p.get('credited_to_right_track')
    if g.get('rules') and 'pass_count_correlation' in g['rules']:
        out['pass_count_correlation'] = g['rules']['pass_count_correlation']
    own = (g.get('own_labels') or {}).get('overall')
    if own:
        out['own_labels_found'] = own.get('found')
    return out


def gather(evidence=None):
    evidence = Path(evidence or S.EVIDENCE)
    g, used = {'created': S.now()}, []
    for key, fn in (('spotter', spotter), ('crediting', crediting), ('reasons', reasons), ('speed', speed),
                    ('rules', rules), ('own_labels', own_labels)):
        g[key], files = fn(evidence)
        used += files
    g['changes'] = changes(g)
    g['headline'] = headline(g)
    expected = [f'action_spotter_{k}_{t}.json' for k in HELD_OUT for t, _ in VERSIONS] + \
        [CREDIT['current'][0], CREDIT['throwinline'][0], 'credit_reasons.json', 'spotter_changes.json',
         'event_logic_soccertrack.json']
    g['sources'] = used
    g['missing'] = [f for f in expected if f not in used and not (f == CREDIT['current'][0] and 'current' in g['crediting'])]
    if g['own_labels'] is None:
        g['missing'].append('own_labels_<tag>.json')
    return g
