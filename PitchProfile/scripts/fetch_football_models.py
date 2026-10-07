"""Download the pretrained models used by full-match video analysis, verifying SHA-256.

Files go to the data-drive weights folder (football_profiler.football_models.weights_dir)
and the NBJW calibration and SoccerNet team-spotting code to its third_party folder; nothing
is written into the project. Already-present files with the right hash are kept. The CLIP
backbone of the identity model is fetched by timm on first use into
weights/hf; the identity model and the shot classifier are trained locally
(scripts/train_identity_models.py, scripts/train_shot_model.py).

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\fetch_football_models.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import football_models as FM

NBJW_REPO = 'https://github.com/mguti97/No-Bells-Just-Whistles.git'
NBJW_COMMIT = 'bd993b31c2917096c23bb8aadf148314d17f8345'
# name: (source, url or Google Drive id, sha256)
WEIGHTS = {
    'football-player-detection.pt': ('gdrive', '17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q',
                                     '75b09c377fbf9d0791d23f6cfb689f5aed6eaa43a6818bd1fb884cf7507fffaf'),
    'football-ball-detection.pt': ('gdrive', '1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V',
                                   '678fbad05134f19c5094cb8d273812ec9c6691228180d46832551ecf99ed2912'),
    'legibility_resnet34_soccer.pth': ('gdrive', '18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw',
                                       'b9c61dabaea4a6ec99528c5ae394f5875aecb8207de38484eccb0f977a373e41'),
    'parseq_soccernet_jersey.ckpt': ('gdrive', '1uRln22tlhneVt3P6MePmVxBWSLMsL3bm',
                                     '14aeb3b13876500e04c93674716a3dae54c2e2d4e06b1abe04758d260d314879'),
}
NBJW_KEYPOINTS = ('https://github.com/mguti97/No-Bells-Just-Whistles/releases/download/v1.0.0/SV_kp',
                  '7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113')
# Ball-action spotter (football_profiler.action_spotting): SoccerNet sn-teamspotting devkit and
# its published T-DEED baseline checkpoint.
SPOTTING_REPO = 'https://github.com/SoccerNet/sn-teamspotting.git'
SPOTTING_CHECKPOINT = ('gdrive', '1Dp2tqDIrNwLmfuUJPPhpz1db7DZjxIzd')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(path: Path, source: str, where: str, digest: str):
    if path.is_file() and sha256(path) == digest:
        print(f'ok        {path.name}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'download  {path.name}', flush=True)
    if source == 'gdrive':
        import gdown
        gdown.download(id=where, output=str(path), quiet=True)
    else:
        urllib.request.urlretrieve(where, path)
    actual = sha256(path)
    if actual != digest:
        raise SystemExit(f'{path.name}: SHA-256 {actual} does not match the recorded {digest}')
    print(f'verified  {path.name}')


def main():
    weights = FM.weights_dir()
    for name, (source, where, digest) in WEIGHTS.items():
        fetch(weights / name, source, where, digest)
    nbjw = FM.nbjw_dir()
    if not (nbjw / '.git').is_dir():
        nbjw.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--quiet', NBJW_REPO, str(nbjw)], check=True)
    subprocess.run(['git', '-C', str(nbjw), 'checkout', '--quiet', NBJW_COMMIT], check=True)
    fetch(nbjw / 'weights' / FM.NBJW_KEYPOINT_WEIGHTS, 'url', *NBJW_KEYPOINTS)
    from football_profiler import action_spotting as AS
    kit = AS.devkit_dir()
    if not (kit / '.git').is_dir():
        kit.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--quiet', SPOTTING_REPO, str(kit)], check=True)
    subprocess.run(['git', '-C', str(kit), 'checkout', '--quiet', AS.DEVKIT_COMMIT], check=True)
    # The published baseline; the fine-tune (action_spotting.CHECKPOINT) is made locally by
    # scripts/finetune_action_spotter.py from SoccerNet NDA data and used when present.
    fetch(AS.published_checkpoint_path(), *SPOTTING_CHECKPOINT, AS.PUBLISHED_SHA256)
    print('All football models are present:', FM.availability(), '| action spotter:', AS.available())


if __name__ == '__main__':
    main()
