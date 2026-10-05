from __future__ import annotations
import json
import os
import re
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_LOCATION = ROOT / ".data-location"
LOCAL_DATA = Path(DATA_LOCATION.read_text(encoding="utf-8").strip()) if DATA_LOCATION.is_file() else ROOT / "data"
DATA = Path(os.environ.get("PITCHPROFILE_DATA") or LOCAL_DATA).resolve()
EVIDENCE = Path(os.environ.get("PITCHPROFILE_EVIDENCE") or
                (LOCAL_DATA.parent / "evidence" if DATA_LOCATION.is_file() else ROOT / "evidence")).resolve()
LABELS = ("target_forward", "runner_behind", "link_forward")

def now():
    return datetime.now(timezone.utc).isoformat()

def clean_json(value):
    if isinstance(value, dict): return {str(k): clean_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray): return clean_json(value.tolist())
    if isinstance(value, np.generic): return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def _shared(fn, attempts=40, wait=.025):
    """Run fn, retrying Windows sharing violations: a file being replaced by another thread
    (a re-analysis job) cannot be opened or replaced for a moment."""
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i==attempts-1:raise
            time.sleep(wait)

def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix=".tmp")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(clean_json(data),f,indent=2,allow_nan=False)
        _shared(lambda:os.replace(tmp,path))
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def read_json(path, default=None):
    p=Path(path)
    return json.loads(_shared(lambda:p.read_text(encoding="utf-8"))) if p.exists() else default

def dataset_dir(identifier, create=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,90}",identifier):raise ValueError("Invalid dataset identifier")
    p=DATA/"datasets"/identifier
    if create:p.mkdir(parents=True,exist_ok=True)
    if not p.is_dir():raise FileNotFoundError("Dataset does not exist")
    return p

# Per-player records of a half that belong to the person, not to the label the analysis gave him:
# (table, columns that with dataset_id and player_id form its key).
PLAYER_RECORDS = (('player_labels', ()), ('player_label_evidence', ()), ('player_identity_links', ()),
                  ('player_track_corrections', ()), ('reviews', ('reviewer',)), ('manual_events', None))


def move_player_records(dataset_id, moves):
    """Re-key a half's per-player records after re-analysis renamed players.

    moves: {old player id: new player id}. Ratings, bookmarks, identity corrections, legacy
    reviews and manual events follow the player; a record never replaces one the new player
    already has (that record stays under its old id). Returns the number of records moved.
    """
    moves = {str(k): str(v) for k, v in moves.items() if k != v}
    if not moves:
        return 0
    moved = 0
    with db() as c:
        for table, key in PLAYER_RECORDS:
            if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                continue
            if key is None:                                      # no key on the player: move every row
                for old, new in moves.items():
                    moved += c.execute(f'UPDATE {table} SET player_id=? WHERE dataset_id=? AND player_id=?',
                                       (new, dataset_id, old)).rowcount
                continue
            rows = [dict(r) for r in c.execute(f'SELECT * FROM {table} WHERE dataset_id=?', (dataset_id,))]
            if not any(r['player_id'] in moves for r in rows):
                continue
            ident = lambda r: (r['player_id'],) + tuple(r[k] for k in key)
            groups = {}
            for r in rows:
                groups.setdefault((moves.get(r['player_id'], r['player_id']),) + tuple(r[k] for k in key), []).append(r)
            final, losers = [], []
            for (dest, *_), group in groups.items():
                stay = [r for r in group if r['player_id'] == dest]
                winner = stay[0] if stay else group[0]
                final.append({**winner, 'player_id': dest})
                moved += winner['player_id'] != dest
                losers += [r for r in group if r is not winner]
            taken = {ident(r) for r in final}
            for r in losers:                                       # a conflict keeps its record under a free id
                r = dict(r)
                while ident(r) in taken:
                    r['player_id'] += '~kept'
                taken.add(ident(r))
                final.append(r)
            c.execute(f'DELETE FROM {table} WHERE dataset_id=?', (dataset_id,))
            for r in final:
                cols = list(r)
                c.execute(f'INSERT INTO {table}({",".join(cols)}) VALUES({",".join("?" * len(cols))})',
                          [r[k] for k in cols])
    return moved


def copy_player_records(source, target, mapping):
    """Bring a half's ratings, bookmarks and identity corrections into its whole match.

    mapping: {player id in source: player id in target}. A rating replaces the target's only
    when it is newer (a player rated in both halves keeps the latest rating); identity
    corrections are copied only where the target has none. Returns the number of records copied.
    """
    copied = 0
    with db() as c:
        for r in [dict(x) for x in c.execute('SELECT * FROM player_labels WHERE dataset_id=?', (source,))]:
            dest = mapping.get(r['player_id'])
            if dest is None:
                continue
            have = c.execute('SELECT updated FROM player_labels WHERE dataset_id=? AND player_id=?', (target, dest)).fetchone()
            if have and str(have['updated']) >= str(r['updated']):
                continue
            c.execute('INSERT OR REPLACE INTO player_labels(dataset_id,player_id,labeler,position_group,labels,notes,updated) '
                      'VALUES(?,?,?,?,?,?,?)', (target, dest, r['labeler'], r['position_group'], r['labels'], r['notes'],
                                                r['updated']))
            ev = c.execute('SELECT evidence FROM player_label_evidence WHERE dataset_id=? AND player_id=?',
                           (source, r['player_id'])).fetchone()
            c.execute('INSERT OR REPLACE INTO player_label_evidence VALUES(?,?,?)',
                      (target, dest, ev['evidence'] if ev else '[]'))
            copied += 1
        for table in ('player_identity_links', 'player_track_corrections'):
            if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                continue
            for r in [dict(x) for x in c.execute(f'SELECT * FROM {table} WHERE dataset_id=?', (source,))]:
                dest = mapping.get(r['player_id'])
                if dest is not None:
                    copied += c.execute(f'INSERT OR IGNORE INTO {table} VALUES(?,?,?)', (target, dest, r['payload'])).rowcount
    return copied


def datasets():
    return [read_json(p) for p in sorted((DATA/"datasets").glob("*/manifest.json"))]


def video_path(directory, manifest):
    if manifest.get('source_kind') == 'model_predictions' and manifest.get('soccernet_library_id'):
        from .soccernet import source_path
        return source_path(manifest['soccernet_library_id'])
    # Only the local SoccerTrack CLI creates external video links. Canonical web
    # imports are always source_kind=imported_tracking and cannot use this branch.
    if manifest.get('source_kind') == 'soccertrack_reference' and manifest.get('source_video_path'):
        return Path(manifest['source_video_path'])
    return Path(directory) / manifest['video']

def load_tracks(identifier):
    return pd.read_csv(dataset_dir(identifier)/"tracks.csv.gz",dtype={"player_id":str,"track_id":str})

def save_tracks(identifier, tracks):
    tracks.to_csv(dataset_dir(identifier,True)/"tracks.csv.gz",index=False,compression="gzip")

@contextmanager
def db():
    DATA.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(DATA/"annotations.sqlite",timeout=15)
    c.row_factory=sqlite3.Row
    try:
        c.executescript("""
    CREATE TABLE IF NOT EXISTS reviews (
      dataset_id TEXT NOT NULL, player_id TEXT NOT NULL, reviewer TEXT NOT NULL,
      labels TEXT NOT NULL, evidence TEXT NOT NULL, notes TEXT NOT NULL,
      rubric_version TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL,
      PRIMARY KEY(dataset_id,player_id,reviewer));
    CREATE TABLE IF NOT EXISTS review_history (
      id INTEGER PRIMARY KEY, dataset_id TEXT,player_id TEXT,reviewer TEXT,
      labels TEXT,evidence TEXT,notes TEXT,rubric_version TEXT,created TEXT);
    CREATE TABLE IF NOT EXISTS adjudications (
      dataset_id TEXT,player_id TEXT,label TEXT,value INTEGER,reviewer TEXT,
      reason TEXT,updated TEXT,PRIMARY KEY(dataset_id,player_id,label));
    CREATE TABLE IF NOT EXISTS manual_events (
      id INTEGER PRIMARY KEY,dataset_id TEXT,player_id TEXT,time_s REAL,
      kind TEXT,reviewer TEXT,notes TEXT,created TEXT);
    CREATE TABLE IF NOT EXISTS interval_cases (
      id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, player_id TEXT NOT NULL,
      definition TEXT NOT NULL, profile TEXT NOT NULL, fingerprint TEXT NOT NULL, created TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS interval_reviews (
      case_id TEXT NOT NULL, reviewer TEXT NOT NULL, payload TEXT NOT NULL, updated TEXT NOT NULL,
      PRIMARY KEY(case_id,reviewer));
    CREATE TABLE IF NOT EXISTS interval_review_history (
      id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, reviewer TEXT NOT NULL,
      payload TEXT NOT NULL, created TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS interval_adjudications (
      case_id TEXT NOT NULL, label TEXT NOT NULL, value INTEGER NOT NULL,
      reviewer TEXT NOT NULL, reason TEXT NOT NULL, updated TEXT NOT NULL,
      PRIMARY KEY(case_id,label));
    CREATE TABLE IF NOT EXISTS player_labels (
      dataset_id TEXT NOT NULL, player_id TEXT NOT NULL, labeler TEXT NOT NULL,
      position_group TEXT NOT NULL, labels TEXT NOT NULL, notes TEXT NOT NULL, updated TEXT NOT NULL,
      PRIMARY KEY(dataset_id,player_id));
    CREATE TABLE IF NOT EXISTS player_label_history (
      id INTEGER PRIMARY KEY, dataset_id TEXT, player_id TEXT, labeler TEXT, position_group TEXT,
      labels TEXT, notes TEXT, created TEXT);
    CREATE TABLE IF NOT EXISTS player_label_evidence (
      dataset_id TEXT NOT NULL, player_id TEXT NOT NULL, evidence TEXT NOT NULL,
      PRIMARY KEY(dataset_id,player_id));
    CREATE TABLE IF NOT EXISTS player_label_evidence_history (
      label_history_id INTEGER PRIMARY KEY, evidence TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS match_context_bindings (
      dataset_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS player_identity_links (
      dataset_id TEXT NOT NULL, player_id TEXT NOT NULL, payload TEXT NOT NULL,
      PRIMARY KEY(dataset_id,player_id));
    CREATE TABLE IF NOT EXISTS player_identity_link_history (
      id INTEGER PRIMARY KEY, dataset_id TEXT, player_id TEXT, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS player_track_corrections (
      dataset_id TEXT NOT NULL, player_id TEXT NOT NULL, payload TEXT NOT NULL,
      PRIMARY KEY(dataset_id,player_id));
    CREATE TABLE IF NOT EXISTS player_track_correction_history (
      id INTEGER PRIMARY KEY, dataset_id TEXT, player_id TEXT, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS event_outcome_reviews (
      dataset_id TEXT NOT NULL, event_id TEXT NOT NULL, payload TEXT NOT NULL,
      PRIMARY KEY(dataset_id,event_id));
    CREATE TABLE IF NOT EXISTS event_outcome_history (
      id INTEGER PRIMARY KEY, dataset_id TEXT, event_id TEXT, payload TEXT NOT NULL);
        """)
        with c:
            yield c
    finally:
        c.close()

def reviews(dataset_id=None, player_id=None):
    q="SELECT * FROM reviews WHERE 1=1";args=[]
    for key,v in [("dataset_id",dataset_id),("player_id",player_id)]:
        if v is not None:q+=f" AND {key}=?";args.append(str(v))
    with db() as c:rows=[dict(x) for x in c.execute(q,args)]
    for r in rows:r["labels"]=json.loads(r["labels"])
    return rows

def save_review(dataset_id,player_id,reviewer,labels,evidence,notes=""):
    reviewer=reviewer.strip().casefold()
    if not reviewer or len(reviewer)>80:raise ValueError("Enter a reviewer name or ID (up to 80 characters)")
    if not evidence.strip():raise ValueError("Record footage timestamps or another independently reviewed source")
    if set(labels)!=set(LABELS) or any(x not in (0,1,None) or isinstance(x,bool) for x in labels.values()):
        raise ValueError("Each archetype must be 0, 1 or null (insufficient evidence)")
    stamp=now();args=(dataset_id,str(player_id),reviewer,json.dumps(labels),evidence[:2000],notes[:3000],"1.0",stamp,stamp)
    with db() as c:
        c.execute("INSERT INTO review_history(dataset_id,player_id,reviewer,labels,evidence,notes,rubric_version,created) VALUES(?,?,?,?,?,?,?,?)",args[:-1])
        c.execute("""INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(dataset_id,player_id,reviewer)
        DO UPDATE SET labels=excluded.labels,evidence=excluded.evidence,notes=excluded.notes,
        rubric_version=excluded.rubric_version,updated=excluded.updated""",args)
        # A changed original review invalidates any previous adjudication for this case.
        c.execute("DELETE FROM adjudications WHERE dataset_id=? AND player_id=?",(dataset_id,str(player_id)))

def consensus(dataset_id, player_id):
    rows=reviews(dataset_id,player_id)
    with db() as c:
        adj={r["label"]:dict(r) for r in c.execute("SELECT * FROM adjudications WHERE dataset_id=? AND player_id=?",(dataset_id,str(player_id)))}
    out={}
    for name in LABELS:
        known=[r["labels"][name] for r in rows if r["labels"][name] is not None]
        if name in adj:out[name]={"value":adj[name]["value"],"status":"adjudicated"}
        elif len(known)>=2 and len(set(known))==1:out[name]={"value":known[0],"status":"agreed"}
        elif len(set(known))>1:out[name]={"value":None,"status":"disagreement"}
        else:out[name]={"value":None,"status":"needs independent review"}
    return {"reviewers":len(rows),"labels":out}

def adjudicate(dataset_id,player_id,label,value,reviewer,reason):
    rows=reviews(dataset_id,player_id);reviewer=reviewer.strip().casefold()
    if label not in LABELS or value not in (0,1) or isinstance(value,bool):raise ValueError("Invalid adjudication")
    if len(rows)<2:raise ValueError("Two independent reviews are required first")
    if not reviewer or reviewer in {x["reviewer"] for x in rows}:raise ValueError("A different third reviewer must adjudicate")
    if not reason.strip():raise ValueError("An evidence-based reason is required")
    with db() as c:
        c.execute("INSERT OR REPLACE INTO adjudications VALUES(?,?,?,?,?,?,?)",(dataset_id,str(player_id),label,value,reviewer,reason[:3000],now()))

def manual_events(dataset_id,player_id=None):
    q="SELECT * FROM manual_events WHERE dataset_id=?";args=[dataset_id]
    if player_id is not None:q+=" AND player_id=?";args.append(str(player_id))
    with db() as c:return [dict(x) for x in c.execute(q+" ORDER BY time_s",args)]
