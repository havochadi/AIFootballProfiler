"""Audit or remove runtime data that is byte-identical to a retained D: file.

The original project ZIP is archived on the data drive before its workspace
copy is removed. Source code, environments, credentials and unique outputs
are outside the duplicate scan. Use --apply to perform the verified cleanup.
"""
from __future__ import annotations
import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from football_profiler import storage as S


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true')
    parser.add_argument('--report',type=Path,help='Optional audit-report destination')
    args=parser.parse_args()
    workspace=ROOT.parent.resolve();runtime=(ROOT/'.runtime').resolve();data=S.DATA.parent.parent.resolve()
    if data==workspace or data.is_relative_to(workspace):raise ValueError('External data drive is required')
    suffixes={'.csv','.gz','.mp4','.mkv','.jpg','.png','.npy','.npz','.json','.zip','.sqlite'}
    candidates=[p for folder in runtime.iterdir() if folder.is_dir() and folder.name.startswith(('pytest-','browser-'))
                for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in suffixes]
    by_size=defaultdict(list)
    for p in data.rglob('*'):
        if p.is_file() and '.cache' not in p.parts:by_size[p.stat().st_size].append(p)
    cache={};matches=[]
    for p in candidates:
        resolved=p.resolve()
        if not resolved.is_relative_to(runtime):raise ValueError('Runtime link escapes workspace: '+str(p))
        targets=by_size.get(p.stat().st_size,[])
        if not targets:continue
        sha=digest(p)
        for target in targets:
            retained=target.resolve()
            if not retained.is_relative_to(data):continue
            if retained not in cache:cache[retained]=digest(retained)
            if cache[retained]==sha:
                record={'source':str(resolved),'retained':str(retained),'bytes':p.stat().st_size,'sha256':sha}
                if args.apply:
                    # Revalidate source immediately before deletion; canonical
                    # data files are not edited by this operation.
                    if digest(resolved)!=sha:raise ValueError('Candidate changed during cleanup')
                    resolved.unlink()
                matches.append(record)
                break
    archive=None
    source=(workspace/'PitchProfile_Project.zip').resolve()
    if source.is_file():
        if source.parent!=workspace:raise ValueError('Unexpected archive location')
        destination=(data/'Archives'/source.name).resolve()
        if not destination.is_relative_to(data):raise ValueError('Archive target escapes data drive')
        sha=digest(source)
        archive={'source':str(source),'retained':str(destination),'bytes':source.stat().st_size,'sha256':sha}
        if args.apply:
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():shutil.copy2(source,destination)
            if digest(destination)!=sha or digest(source)!=sha:raise ValueError('Archive verification failed; source retained')
            source.unlink()
    report={'applied':args.apply,'created':S.now(),'verified_duplicates':len(matches),
            'duplicate_bytes':sum(x['bytes'] for x in matches),'files':matches,'archived_project_zip':archive}
    S.write_json(args.report or (S.EVIDENCE/'duplicate_cleanup.json' if args.apply else runtime/'duplicate_cleanup_plan.json'),report)
    print({k:v for k,v in report.items() if k!='files'})


if __name__=='__main__':main()
