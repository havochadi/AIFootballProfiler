"""Download the high-quality (1080p) SoccerNet halves of the working-set games to the data drive.

Same games as the 720p library (SoccerNet/working_set.json), from the videos-HQ branch of
SoccerNet/SoccerNet_raw_HQ, into SoccerNet/videos-HQ/<league>/<season>/<game>/[12]_HQ.mkv.
Rerunning reuses completed files. Uses the locally authenticated Hugging Face account.

Usage (from PitchProfile/):
  .\\.venv\\Scripts\\python.exe scripts\\download_soccernet_hq.py [--all-videos]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.runtime' / 'huggingface'))
sys.path.insert(0, str(ROOT))
from football_profiler import soccernet as SN  # noqa: E402

REPO, BRANCH = 'SoccerNet/SoccerNet_raw_HQ', 'videos-HQ'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--all-videos', action='store_true', help='every local 720p game, not only the working set')
    p.add_argument('--min-gb', type=float, default=0.0,
                   help='skip halves smaller than this; many "HQ" files are 1280x720 (about 1.1-1.7 GB a half), '
                        'the 1920x1080 ones are about 2.8 GB or more, so 2.2 fetches only true 1080p')
    args = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    revision = api.dataset_info(REPO, revision=BRANCH).sha
    games = sorted({x['game'] for x in SN.library(all_videos=args.all_videos)})
    out = SN.root() / 'videos-HQ'
    done = []
    for k, game in enumerate(games, 1):
        # video.ini gives where each 720p half starts inside the HQ broadcast (labels follow the 720p halves).
        hf_hub_download(REPO, f'{game}/video.ini', repo_type='dataset', revision=revision, local_dir=out)
        sizes = {f.path.rsplit('/', 1)[-1]: getattr(f, 'size', 0) or 0
                 for f in api.list_repo_tree(REPO, path_in_repo=game, repo_type='dataset', revision=revision)}
        for half in (1, 2):
            if sizes.get(f'{half}_HQ.mkv', 0) < args.min_gb * 1e9:
                print(f'[{k}/{len(games)}] {game}/{half}_HQ.mkv: {sizes.get(f"{half}_HQ.mkv", 0) / 1e9:.2f} GB, skipped',
                      flush=True)
                continue
            name = f'{game}/{half}_HQ.mkv'
            path = Path(hf_hub_download(REPO, name, repo_type='dataset', revision=revision, local_dir=out))
            done.append({'file': name, 'bytes': path.stat().st_size})
            print(f'[{k}/{len(games)}] {name}: {path.stat().st_size / 1e9:.2f} GB', flush=True)
    (out / 'download.json').write_text(json.dumps({'repository': REPO, 'branch': BRANCH, 'revision': revision,
                                                   'files': done}, indent=1), encoding='utf-8')
    print('All HQ halves present in', out)


if __name__ == '__main__':
    main()
