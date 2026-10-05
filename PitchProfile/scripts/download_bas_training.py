"""Download SoccerNet Ball Action Spotting training data (SoccerNet/SN-BAS-2025) to the data drive.

Used only to fine-tune the action spotter (football_profiler.action_spotting); these games are
never analysed as matches. The video zips are password-protected: the password comes from the
SoccerNet NDA (https://www.soccer-net.org/data). This script downloads; extract_bas_training.py
asks for the password at the terminal and unpacks. Labelled splits only (train, valid, test,
about 15 GB); the challenge split has no public labels. Reruns reuse completed files.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\download_bas_training.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.runtime' / 'huggingface'))
sys.path.insert(0, str(ROOT))
from football_profiler import soccernet as SN  # noqa: E402

REPO = 'SoccerNet/SN-BAS-2025'
FILES = ['train.zip', 'valid.zip', 'test.zip', 'README.md']


def target():
    return SN.root() / 'sn-bas-2025'


def main():
    from huggingface_hub import HfApi, hf_hub_download
    revision = HfApi().dataset_info(REPO).sha
    out = target()
    out.mkdir(parents=True, exist_ok=True)
    record = []
    for name in FILES:
        print(f'downloading {name} ...', flush=True)
        path = Path(hf_hub_download(REPO, name, repo_type='dataset', revision=revision, local_dir=out))
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1 << 22), b''):
                h.update(block)
        record.append({'file': name, 'bytes': path.stat().st_size, 'sha256': h.hexdigest()})
        print(f'  done: {path.stat().st_size / 1e9:.2f} GB', flush=True)
    (out / 'download.json').write_text(json.dumps({'repository': REPO, 'revision': revision, 'files': record},
                                                  indent=1), encoding='utf-8')
    print('All files present in', out)


if __name__ == '__main__':
    main()
