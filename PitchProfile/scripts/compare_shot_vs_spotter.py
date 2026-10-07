"""Score the shot classifier and the fine-tuned T-DEED spotter on the same four held-out SoccerNet halves, with the same labels and matching.
Usage (from PitchProfile/): .\.venv\Scripts\python.exe scripts\compare_shot_vs_spotter.py
Reads only cached evidence (shot candidates, action_spots.json, SoccerNet Labels-v2); writes evidence/shot_vs_tdeed.json."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import soccernet as SN, match_shots as SH, storage as S
from football_profiler.football_models import weights_dir

HALVES = {'sn-20150310-real-madrid-schalke-h1': '5efc5ed7fc616794a78b', 'sn-20150310-real-madrid-schalke-h2': '3c2ded0293b11ba7782a',
          'sn-20160827-tottenham-liverpool-h1': '8b476927c0f33719011f', 'sn-20160827-tottenham-liverpool-h2': 'c33c69f1fd303e8d3bea'}
CACHE = S.DATA / 'models' / 'shot_candidates'
SHOT_L = {'Shots on target', 'Shots off target', 'Penalty'}
bundle = pickle.load((weights_dir() / SH.MODEL_FILE).open('rb'))
model, THR = bundle['model'], bundle['threshold']
lib = {x['id']: x for x in SN.library()}

labels, clf, tdeed = [], [], []
for hid, lid in HALVES.items():
    item = lib[lid]
    for l in SN.annotations(item['game'], item['half'], screen_sides=True):
        if l['label'] in SHOT_L | {'Goal'}:
            labels.append(dict(half=hid, t=l['time_s'], kind='goal' if l['label'] == 'Goal' else 'shot', side=l.get('team')))
    tab = pd.read_csv(CACHE / f'{hid}.csv.gz')
    tab['probability'] = model.predict_proba(tab[SH.FEATURES].to_numpy(float))[:, 1]
    for r in SH.select_shots(tab, THR).itertuples():
        clf.append(dict(half=hid, t=float(r.time_s), side='left' if r.sign == 1 else 'right', score=float(r.probability)))
    spots = json.load(open(S.DATA / 'datasets' / hid / 'action_spots.json'))['spots']
    for s in spots:
        if s['label'] in ('SHOT', 'GOAL'):
            tdeed.append(dict(half=hid, t=s['time_s'], side=s['side'], score=s['score'], cls=s['label']))
print('labels', pd.Series([l['kind'] for l in labels]).value_counts().to_dict(), 'classifier calls', len(clf), 'T-DEED SHOT/GOAL spots >=0.05', len(tdeed))


def merge(preds, gap=1.0):
    """Same-side calls within gap seconds: keep the highest score (what the classifier does at analysis time)."""
    kept = []
    for p in sorted(preds, key=lambda x: -x['score']):
        if not any(k['half'] == p['half'] and k['side'] == p['side'] and abs(k['t'] - p['t']) <= gap for k in kept):
            kept.append(p)
    return kept


def score(preds, labs, window, need_side):
    """Greedy one-to-one matching, highest score first, to the nearest unused label whose window contains the prediction."""
    used, tp, tp_side = set(), 0, 0
    for p in sorted(preds, key=lambda x: -x['score']):
        best = None
        for i, l in enumerate(labs):
            if i in used or l['half'] != p['half']:
                continue
            lo, hi = window(l)
            if not (l['t'] + lo <= p['t'] <= l['t'] + hi):
                continue
            if need_side and l['side'] != p['side']:
                continue
            if best is None or abs(p['t'] - l['t']) < abs(p['t'] - labs[best]['t']):
                best = i
        if best is not None:
            used.add(best); tp += 1
    n, m = len(preds), len(labs)
    pr, rc = tp / max(n, 1), tp / max(m, 1)
    return dict(pred=n, labels=m, tp=tp, precision=round(pr, 3), recall=round(rc, 3), f1=round(2 * pr * rc / max(pr + rc, 1e-9), 3))


shots = [l for l in labels if l['kind'] == 'shot']
T = lambda thr, classes=('SHOT',): merge([p for p in tdeed if p['cls'] in classes and p['score'] >= thr])
out = {}
tol2 = lambda l: (-2.0, 2.0)
own = lambda l: (-6.0, 0.5) if l['kind'] == 'goal' else (-2.5, 1.5)
for name, labs, win in (('A shots only (57), +-2 s', shots, tol2), ('B shots and goals (70), classifier windows', labels, own)):
    for side in (False, True):
        key = f"{name} | team side {'required' if side else 'ignored'}"
        out[key] = {'classifier 0.67': score(clf, labs, win, side),
                    'T-DEED SHOT>=0.3': score(T(0.3), labs, win, side),
                    'T-DEED SHOT>=0.2 (best F1 on BAS)': score(T(0.2), labs, win, side),
                    'T-DEED SHOT+GOAL>=0.3': score(T(0.3, ('SHOT', 'GOAL')), labs, win, side)}
for k, v in out.items():
    print(k)
    for m, r in v.items():
        print('   ', m.ljust(36), r)
# the stored classifier numbers for comparison
print('stored held_out_test:', json.load(open(S.EVIDENCE / 'shot_model.json'))['held_out_test']['model'])
json.dump(out, open(S.EVIDENCE / 'shot_vs_tdeed.json', 'w'), indent=1)

# Oracle check: best F1 over all thresholds on these halves (flattering to both), shots only, +-2 s
allclf = []
for hid in HALVES:
    tab = pd.read_csv(CACHE / f'{hid}.csv.gz'); tab['probability'] = model.predict_proba(tab[SH.FEATURES].to_numpy(float))[:, 1]
    for r in SH.select_shots(tab, 0.02).itertuples():
        allclf.append(dict(half=hid, t=float(r.time_s), side='left' if r.sign == 1 else 'right', score=float(r.probability)))
best = {}
for name, pool in (('classifier', allclf), ('T-DEED SHOT', [p for p in tdeed])):
    rows = []
    for thr in np.arange(0.05, 0.95, 0.05):
        pr = merge([p for p in pool if p['score'] >= thr and p.get('cls', 'SHOT') == 'SHOT'])
        rows.append((thr, score(pr, shots, tol2, False)))
    b = max(rows, key=lambda x: x[1]['f1']); best[name] = (round(float(b[0]), 2), b[1]); print('oracle', name, best[name])
out['oracle best-F1 shots only +-2 s'] = {k: dict(threshold=v[0], **v[1]) for k, v in best.items()}
json.dump(out, open(S.EVIDENCE / 'shot_vs_tdeed.json', 'w'), indent=1)
