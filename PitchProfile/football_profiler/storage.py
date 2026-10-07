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
PLAYER_RECORDS = (('player_identity_links', ()), ('player_track_corrections', ()), ('manual_events', None))


def move_player_records(dataset_id, moves):
    """Re-key a half's per-player records after re-analysis renamed players.

    moves: {old player id: new player id}. Identity corrections, track corrections and manual
    events follow the player; a record never replaces one the new player
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
    """Bring a half's identity and track corrections into its whole match.

    mapping: {player id in source: player id in target}. A correction is copied only where the
    target has none. Returns the number of records copied.
    """
    copied = 0
    with db() as c:
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
    CREATE TABLE IF NOT EXISTS manual_events (
      id INTEGER PRIMARY KEY,dataset_id TEXT,player_id TEXT,time_s REAL,
      kind TEXT,reviewer TEXT,notes TEXT,created TEXT);
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

def manual_events(dataset_id,player_id=None):
    q="SELECT * FROM manual_events WHERE dataset_id=?";args=[dataset_id]
    if player_id is not None:q+=" AND player_id=?";args.append(str(player_id))
    with db() as c:return [dict(x) for x in c.execute(q+" ORDER BY time_s",args)]
