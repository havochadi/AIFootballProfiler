"""Authenticate locally and selectively download an approved SoccerTrack match."""
import argparse
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_profiler import storage as S
os.environ.setdefault('HF_HOME', str(ROOT / '.runtime/huggingface'))
REPO = 'atomscott/soccertrack-v2'


def main():
    from huggingface_hub import HfApi, get_token, hf_hub_download, login
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--login', action='store_true', help='Prompt privately for an HF read token; never paste it into chat')
    p.add_argument('--match', default='117092')
    p.add_argument('--download', action='store_true', help='Download selected match files; otherwise list their sizes')
    p.add_argument('--output', type=Path, default=S.DATA / 'raw/soccertrack-v2')
    args = p.parse_args()
    if args.login:
        login(add_to_git_credential=False)
        print('Local Hugging Face login saved. You can now list or download an approved match.')
        return
    if not re.fullmatch(r'\d{6}', args.match):
        p.error('Use a six-digit SoccerTrack match ID')
    token = get_token()
    if not token:
        p.error('No local login. Run this script with --login using the account with approved access.')
    api = HfApi(token=token)
    info = api.dataset_info(REPO, files_metadata=True)
    selected = []
    for f in info.siblings:
        parts = Path(f.rfilename).parts
        if parts[0] not in {'gsr', 'bas', 'videos', 'raw'}:
            continue
        if args.match not in parts and not Path(f.rfilename).name.startswith(args.match + '_'):
            continue
        # Calibration maps can be large; only the exact timing metadata is needed here.
        if parts[0] == 'raw' and not (f.rfilename.endswith('.xml') or f.rfilename.endswith('_padding_info.csv')):
            continue
        selected.append(f)
    if not selected:
        raise ValueError('No matching files found; inspect the current provider layout before downloading')
    total = sum(f.size or 0 for f in selected)
    print(f'{REPO} revision {info.sha}: {len(selected)} files, {total / 1e9:.2f} GB')
    for f in selected:
        print(f'{(f.size or 0) / 1e6:10.1f} MB  {f.rfilename}')
    if not args.download:
        print('Add --download to acquire exactly these files. Access must already be approved.')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for f in selected:
        print(f'Acquiring {f.rfilename} (completed local files are reused)...', flush=True)
        hf_hub_download(REPO, f.rfilename, repo_type='dataset', revision=info.sha,
                        token=token, local_dir=args.output)
    import json
    (args.output / f'download-{args.match}.json').write_text(json.dumps({
        'repository': REPO, 'revision': info.sha, 'match_id': args.match,
        'files': [{'path': f.rfilename, 'bytes': f.size} for f in selected],
    }, indent=2), encoding='utf-8')
    print(f'Download complete: {args.output}')


if __name__ == '__main__':
    main()
