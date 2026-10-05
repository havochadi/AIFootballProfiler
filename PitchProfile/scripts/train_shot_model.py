"""Train the shot classifier used by football_profiler.match_events.

Every analysed SoccerNet half contributes its candidate releases (ends of
possession spells and unexplained ball flights near the attacked goal,
football_profiler.match_shots) with features computed from the video analysis only. SoccerNet Labels-v2 'Shots on
target', 'Shots off target', 'Goal' and 'Penalty' events supply the targets:

- a shot label claims the release by the labelled team closest in time within
  [label - 2.5 s, label + 1.5 s]; a goal label (stamped when the ball crosses
  the line) claims that team's last release in [label - 6 s, label + 0.5 s];
- the team is the label's screen side ('left' defends the left goal, so it
  attacks towards x = 105) matched against our inferred attack directions;
- other releases inside a claimed window within 1 s of the claimed one are
  ambiguous and left out; all remaining candidates are negatives.

Official SoccerNet test games are held out. Leave-one-match-out
cross-validation on train+validation games sets the threshold (called shots
equal labelled shots) and measures accuracy; the final model is fitted on those
games and scored once on the test halves. Each fold model is saved too and
analyses the match it did not see, so no half's statistics come from a model
trained on its own labels. The model goes to the weights folder
(football_models.weights_dir) and the report to evidence/shot_model.json.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\train_shot_model.py [--refresh]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import match_events as ME
from football_profiler import match_pipeline as MPL
from football_profiler import match_shots as SH
from football_profiler import soccernet as SN
from football_profiler import storage as S
from football_profiler.football_models import weights_dir

SHOT_LABELS = {'Shots on target', 'Shots off target', 'Penalty'}
GOAL_LABEL = 'Goal'
SHOT_WINDOW = (-2.5, 1.5)
GOAL_WINDOW = (-6.0, 0.5)
AMBIGUOUS_S = 1.0
CACHE = S.DATA / 'models' / 'shot_candidates'
MIN_POSITIVES = 20


def analysed_halves():
    lib = {x['id']: x for x in SN.library()}
    out = []
    for d in sorted((S.DATA / 'datasets').glob('sn-*')):
        meta = S.read_json(d / 'analysis.json', {})
        item = lib.get(meta.get('library_id'))
        if meta.get('detection') and item and item['actions_available']:
            out.append((d.name, item, meta))
    return out


def candidates(identifier, meta, refresh=False):
    """Candidate releases of one half, with whether the heuristic called each a shot (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path, key_path = CACHE / f'{identifier}.csv.gz', CACHE / f'{identifier}.json'
    key = {'features_version': SH.FEATURES_VERSION, 'features': SH.FEATURES,
           'detection': meta['detection'].get('created') or meta.get('created')}
    if not refresh and path.is_file() and S.read_json(key_path, {}) == key:
        table = pd.read_csv(path)
        directions = S.read_json(key_path.with_suffix('.directions.json'), {})
        return table, directions
    _, hz, frames, people, ball, directions, _ = MPL.stages(identifier)
    ev, sp, pos, people, _ = ME.events(people, ball, directions, hz, shot_model=None)
    positions = {s: g for s, g in people.groupby('sample')}
    table = ME.shot_candidates(sp, pos, positions, ball, frames, directions, hz)
    heuristic = set(zip(ev[ev.type == 'shot'].segment, ev[ev.type == 'shot'].time_s.round(3))) if len(ev) else set()
    table['heuristic'] = [(s, round(t, 3)) in heuristic for s, t in zip(table.segment, table.time_s)]
    table.to_csv(path, index=False, compression='gzip')
    S.write_json(key_path.with_suffix('.directions.json'), directions)
    S.write_json(key_path, key)
    return table, directions


def attach_labels(table, labels):
    """Target per candidate (1 shot, 0 not, NaN ambiguous) and per-label match records."""
    table = table.assign(target=0.0, label=None)
    records = []
    for lab in labels:
        kind = lab['label']
        window = GOAL_WINDOW if kind == GOAL_LABEL else SHOT_WINDOW
        t = lab['time_s']
        side = {'left': 1, 'right': -1}.get(lab.get('team'))
        inside = table[(table.time_s >= t + window[0]) & (table.time_s <= t + window[1])]
        own = inside[inside.sign == side] if side else inside
        free = own[own.label.isna()]
        pick = None
        if len(free):
            pick = free.time_s.idxmax() if kind == GOAL_LABEL else (free.time_s - t).abs().idxmin()
            table.loc[pick, ['target', 'label']] = [1.0, kind]
            near = inside.index[(inside.time_s - table.at[pick, 'time_s']).abs() <= AMBIGUOUS_S]
            unclaimed = [i for i in near if i != pick and pd.isna(table.at[i, 'label'])]
            table.loc[unclaimed, 'target'] = np.nan
        records.append({'time_s': t, 'label': kind, 'visibility': lab.get('visibility'), 'side': side,
                        'candidates_in_window': int(len(inside)), 'own_side_candidates': int(len(own)),
                        'matched': pick is not None, 'candidate': None if pick is None else int(pick)})
    return table, records


def model_factory():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=250, min_samples_leaf=15,
                                          l2_regularization=1.0, class_weight='balanced', random_state=0)


def counts(frame, probability, threshold, labels):
    """Shots called per half as at analysis time (same-team calls within 1 s merged), against the labels."""
    predicted = found = 0
    for _, half in frame.assign(probability=probability).groupby('half_id'):
        called = SH.select_shots(half, threshold)
        predicted += len(called)
        found += int((half.loc[called.index, 'target'] == 1).sum())
    return {'predicted': predicted, 'found': found, 'labels': int(labels),
            'precision': found / max(predicted, 1), 'recall': found / max(labels, 1)}


def rule_counts(frame, labels):
    rule = frame.heuristic.astype(bool)
    found = int((rule & (frame.target == 1)).sum())
    return {'predicted': int(rule.sum()), 'found': found, 'labels': int(labels),
            'precision': found / max(int(rule.sum()), 1), 'recall': found / max(labels, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--refresh', action='store_true', help='recompute candidate features from the raw evidence')
    args = p.parse_args()
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import LeaveOneGroupOut

    tables, label_rows, halves_report = [], [], []
    for identifier, item, meta in analysed_halves():
        table, directions = candidates(identifier, meta, args.refresh)
        labels = [x for x in SN.annotations(item['game'], item['half'], screen_sides=True)
                  if x['label'] in SHOT_LABELS | {GOAL_LABEL}]
        table, records = attach_labels(table, labels)
        table['half_id'], table['game'], table['split'] = identifier, item['game'], item['benchmark_split']
        tables.append(table)
        label_rows += [{**r, 'half_id': identifier, 'split': item['benchmark_split']} for r in records]
        halves_report.append({'half': identifier, 'split': item['benchmark_split'], 'candidates': int(len(table)),
                              'labels': len(records), 'matched': sum(r['matched'] for r in records),
                              'direction_status': directions.get('status'),
                              'heuristic_shots': int(table.heuristic.sum())})
        print(json.dumps(halves_report[-1]), flush=True)
    # Ambiguous candidates (target NaN) are not trained on but are still called at analysis time.
    data = pd.concat(tables, ignore_index=True)
    labels = pd.DataFrame(label_rows)
    develop = data[data.split.isin(['train', 'validation'])].reset_index(drop=True)
    test = data[data.split.eq('test')].reset_index(drop=True)
    known = develop.target.notna().to_numpy()
    X, groups = develop[SH.FEATURES].to_numpy(float), develop.game.to_numpy()
    y = develop.target.fillna(0).astype(int).to_numpy()
    if y[known].sum() < MIN_POSITIVES:
        raise SystemExit(f'Only {int(y[known].sum())} labelled shots in train/validation halves; analyse more halves first.')
    n_dev = int(labels.split.isin(['train', 'validation']).sum())

    # Leave one match out. Each fold model is kept: it analyses its own match
    # (cross-fitting), so no half's statistics come from a model trained on its labels.
    oof, fold_models = np.zeros(len(develop)), {}
    for fit_idx, held_idx in LeaveOneGroupOut().split(X, y, groups):
        fit_idx = fit_idx[known[fit_idx]]
        clf = model_factory().fit(X[fit_idx], y[fit_idx])
        oof[held_idx] = clf.predict_proba(X[held_idx])[:, 1]
        fold_models[groups[held_idx[0]]] = clf
    # Shot counts are statistics: pick the threshold at which the called shots
    # match the number of labelled shots, rather than the F1 optimum (which over-counts).
    grid = np.round(np.arange(.05, .951, .01), 2)
    called = [counts(develop, oof, t, n_dev)['predicted'] for t in grid]
    threshold = float(grid[int(np.argmin(np.abs(np.array(called) - n_dev)))])
    report = {
        'created': S.now(), 'features': SH.FEATURES, 'features_version': SH.FEATURES_VERSION,
        'label_matching': {'shot_window_s': SHOT_WINDOW, 'goal_window_s': GOAL_WINDOW, 'ambiguous_s': AMBIGUOUS_S,
                           'labels': int(len(labels)), 'matched_to_a_release': int(labels.matched.sum())},
        'halves': halves_report,
        'threshold': threshold,
        'threshold_rule': 'called shots equal labelled shots in cross-validation',
        'cross_validation': {
            'scheme': 'leave-one-match-out on official train+validation games',
            'games': int(len(fold_models)), 'halves': int(develop.half_id.nunique()),
            'candidates': int(len(develop)), 'labels_with_a_candidate': int(y[known].sum()),
            'average_precision': float(average_precision_score(y[known], oof[known])),
            'model': counts(develop, oof, threshold, n_dev), 'rule': rule_counts(develop, n_dev)},
        'note': 'found = labelled shots whose release the model called a shot; recall is over all labelled shots, '
                'including those without any candidate release.',
    }
    final = model_factory().fit(X[known], y[known])
    if len(test):
        n_test = int(labels.split.eq('test').sum())
        pt = final.predict_proba(test[SH.FEATURES].to_numpy(float))[:, 1]
        tk = test.target.notna().to_numpy()
        report['held_out_test'] = {
            'games': int(test.game.nunique()), 'halves': int(test.half_id.nunique()), 'candidates': int(len(test)),
            'average_precision': float(average_precision_score(test.target[tk].astype(int), pt[tk]))
            if test.target[tk].sum() else None,
            'model': counts(test, pt, threshold, n_test), 'rule': rule_counts(test, n_test)}
    weights_dir().mkdir(parents=True, exist_ok=True)
    with (weights_dir() / SH.MODEL_FILE).open('wb') as f:
        pickle.dump({'model': final, 'threshold': threshold, 'features': SH.FEATURES,
                     'features_version': SH.FEATURES_VERSION, 'created': report['created'],
                     'fold_models': fold_models, 'trained_on': sorted(develop.half_id.unique().tolist())}, f)
    S.write_json(S.EVIDENCE / 'shot_model.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'halves'}, indent=1))


if __name__ == '__main__':
    main()
