"""Recover PFF stadium direction metadata from the original source archives.

Coordinates are already normalised; this only records how to reverse that
normalisation for a common-pitch team radar. No trajectory or review is changed.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from football_profiler import storage as S, pff


def main():
    archives=list((S.DATA.parent.parent/'WorldCup2022').glob('*.zip'))
    count=0
    for manifest in S.datasets():
        if not manifest['id'].startswith('pff-wc2022-'):continue
        game_id=manifest['match_id'].split(':')[1]
        meta=pff._read_json(archives,f'Metadata/{game_id}.json')[0]
        if not isinstance(meta.get('homeTeamStartLeft'),bool):raise ValueError('Unknown kickoff direction')
        home_right=meta['homeTeamStartLeft']!=(manifest['half']==2)
        for player in manifest['players']:
            if player['team'] not in (meta['homeTeam']['name'],meta['awayTeam']['name']):
                raise ValueError('Team name does not match provider metadata')
            right=home_right if player['team']==meta['homeTeam']['name'] else not home_right
            player['stadium_direction']='right' if right else 'left'
        manifest['coordinate_orientation']='attack_right'
        S.write_json(S.dataset_dir(manifest['id'])/'manifest.json',manifest)
        count+=1
    print(f'Recorded source-backed stadium directions for {count} PFF half datasets.')


if __name__=='__main__':main()
