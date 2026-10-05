"""Download SoccerNet manual labels and pinned official splits for local D: videos."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HOME',str(ROOT/'.runtime/huggingface'))
from football_profiler import soccernet as SN, storage as S


def main():
    from huggingface_hub import HfApi, hf_hub_download
    games=sorted({v['game'] for v in SN.library()})
    if not games:
        raise SystemExit('No local 720p halves found in '+str(SN.video_root()))
    splits={}
    for split,name in [('train','Train'),('validation','Valid'),('test','Test')]:
        url=f'https://raw.githubusercontent.com/SoccerNet/SoccerNet/{SN.SOURCE_REVISION}/SoccerNet/data/SoccerNetGames{name}.json'
        with urlopen(url,timeout=60) as response:
            data=json.load(response)
        for league,seasons in data.items():
            for season,matches in seasons.items():
                for match in matches:
                    splits[f'{league}/{season}/{match}']=split
    S.write_json(SN.root()/'official_splits.json',{'revision':SN.SOURCE_REVISION,'games':splits})
    revision=HfApi().dataset_info('SoccerNet/SN-Labels').sha
    downloaded=[]
    for game in games:
        for filename in ('Labels-v2.json','Labels-cameras.json'):
            path=Path(hf_hub_download('SoccerNet/SN-Labels',f'{game}/{filename}',repo_type='dataset',
                                      revision=revision,local_dir=SN.root()/'labels'))
            downloaded.append({'path':path.relative_to(SN.root()).as_posix(),'bytes':path.stat().st_size,
                               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f'Labels ready: {game} ({splits.get(game,"unknown split")})',flush=True)
    S.write_json(SN.root()/'labels_download.json',{'repository':'SoccerNet/SN-Labels','revision':revision,
                                               'files':downloaded,'created':S.now()})


if __name__=='__main__':
    main()
