r"""Re-run post-processing on the six analysed FOOTPASS validation halves and compare their outputs with a saved copy.

Used after code changes that should not alter results: any difference in events.json, identities.json or match_stats.json
is listed. Usage (from PitchProfile/):
  .\.venv\Scripts\python.exe scripts\check_outputs_unchanged.py BACKUP_DIR
BACKUP_DIR holds one folder per half (fp-val-game18-h1 ...) with the files copied before the change.
"""
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_main = Path((ROOT / '.data-location').read_text(encoding='utf-8').strip()) if (ROOT / '.data-location').is_file() else ROOT / 'data'
os.environ.setdefault('PITCHPROFILE_DATA', str(_main.parent.parent / 'PitchProfile-eval' / 'data'))
os.environ.setdefault('PITCHPROFILE_WEIGHTS', str(_main.parent / 'weights'))
sys.path.insert(0, str(ROOT))
from football_profiler import match_pipeline as MP, storage as S  # noqa: E402

backup = Path(sys.argv[1])
halves = [f'fp-val-{g}-h{h}' for g in ('game18', 'game24', 'game47') for h in (1, 2)]
for ident in halves:
    MP.postprocess(ident)
digest = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
bad = []
for ident in halves:
    for f in ('events.json', 'identities.json', 'match_stats.json', 'profiles.json'):
        if digest(backup / ident / f) != digest(S.dataset_dir(ident) / f):
            bad.append((ident, f))
print('files compared', len(halves) * 4, 'different', len(bad))
for b in bad:
    print('  differs:', *b)
sys.exit(1 if bad else 0)
