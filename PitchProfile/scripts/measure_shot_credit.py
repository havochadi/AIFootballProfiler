r"""Measure end-to-end crediting on the FOOTPASS validation halves under a variant of the pipeline.

Re-runs the post-processing of the six analysed FOOTPASS halves under one variant, then scores per-player
actions with evaluate_identity_footpass.py (report evidence/identity_footpass_<tag>.json).
  current     the pipeline as it is
  ballfirst   the ball candidates before redetect_ball (raw_ball_first.csv.gz), kept as raw_ball_current.csv.gz for restoring
  nolegacy    PITCHPROFILE_NO_LEGACY_READER=1: players are named without the ResNet-34 + PARSeq readings

Usage (from PitchProfile/): .\.venv\Scripts\python.exe scripts\measure_shot_credit.py current|nolegacy|ballfirst
Leaves the datasets analysed with that variant: finish by restoring the saved files, or running it once more with `current`.
"""
import os, subprocess, sys
from pathlib import Path

variant = sys.argv[1]
env = dict(os.environ)
env.pop('PITCHPROFILE_NO_LEGACY_READER', None)
if variant == 'ballfirst':
    pass                                            # raw_ball.csv.gz is swapped for raw_ball_first.csv.gz below (the pre-redetection candidates)
elif variant == 'nolegacy':
    env['PITCHPROFILE_NO_LEGACY_READER'] = '1'      # naming without the ResNet-34 + PARSeq readings
elif variant != 'current':
    raise SystemExit(__doc__)
root = Path(__file__).resolve().parents[1]
if variant == 'ballfirst':
    import shutil
    _main = Path((root / '.data-location').read_text(encoding='utf-8').strip()) if (root / '.data-location').is_file() else root / 'data'
    base = Path(env.get('PITCHPROFILE_DATA') or _main.parent.parent / 'PitchProfile-eval' / 'data') / 'datasets'
    for g in ('game18', 'game24', 'game47'):
        for h in (1, 2):
            d = base / f'fp-val-{g}-h{h}'
            if not (d / 'raw_ball_current.csv.gz').is_file():
                shutil.copy2(d / 'raw_ball.csv.gz', d / 'raw_ball_current.csv.gz')
            shutil.copy2(d / 'raw_ball_first.csv.gz', d / 'raw_ball.csv.gz')
code = ("import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')\n"
        "import analyse_footpass as AF\n"
        "from football_profiler import match_pipeline as MP\n"
        "for g in ('game18', 'game24', 'game47'):\n"
        "    for h in (1, 2):\n"
        "        MP.postprocess(f'fp-val-{g}-h{h}')\n"
        "print('postprocess done')\n") % (root, root / 'scripts')
subprocess.run([sys.executable, '-c', code], env=env, check=True, cwd=root)
subprocess.run([sys.executable, 'scripts/evaluate_identity_footpass.py', '--split', 'VAL', '--tag', f'variant_{variant}'], env=env, check=True, cwd=root)
