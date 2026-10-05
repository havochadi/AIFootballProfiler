"""Convert person MOT annotations to YOLO without mixing train/validation sequences.

Use the provider's separate train and validation directories. Each contains
sequence/seqinfo.ini, sequence/img1 and sequence/gt/gt.txt. Explicitly select
person class IDs, or confirm that every box is a person for six-column files.
Images are hard-linked on the data drive, with a copy fallback across volumes.
"""
from __future__ import annotations
import argparse
import configparser
import os
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from football_profiler import storage as S


def convert_sequence(sequence, output, split, person_classes=None, all_people=False, box_origin=1):
    config=configparser.ConfigParser();config.read(sequence/'seqinfo.ini')
    width=int(config['Sequence']['imWidth']);height=int(config['Sequence']['imHeight'])
    if min(width,height)<=0:raise ValueError('Invalid sequence dimensions')
    annotations={}
    for line in (sequence/'gt/gt.txt').read_text().splitlines():
        if not line.strip():continue
        values=[float(v) for v in line.split(',')]
        if len(values)<6 or not np.isfinite(values).all():raise ValueError('Malformed MOT row')
        if len(values)>=7 and values[6]<=0:continue
        if not all_people:
            if len(values)<8:raise ValueError('No class column: explicitly confirm all boxes are people')
            if int(values[7]) not in person_classes:continue
        frame=int(values[0]);x,y,w,h=values[2:6];x-=box_origin;y-=box_origin
        left,top=max(0,x),max(0,y);right,bottom=min(width,x+w),min(height,y+h)
        if right<=left or bottom<=top:continue
        box=((left+right)/(2*width),(top+bottom)/(2*height),(right-left)/width,(bottom-top)/height)
        annotations.setdefault(frame,[]).append('0 '+' '.join(f'{v:.7f}' for v in box))
    image_out=output/'images'/split;label_out=output/'labels'/split
    image_out.mkdir(parents=True,exist_ok=True);label_out.mkdir(parents=True,exist_ok=True)
    images=sorted((sequence/config['Sequence'].get('imDir','img1')).glob('*'))
    count=0
    for image in images:
        if image.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
        frame=int(image.stem);name=sequence.name+'_'+image.name
        destination=image_out/name
        try:os.link(image,destination)
        except OSError:shutil.copy2(image,destination)
        (label_out/(Path(name).stem+'.txt')).write_text('\n'.join(annotations.get(frame,[])),encoding='utf-8')
        count+=1
    if not count:raise ValueError('No sequence images found: '+str(sequence))
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root',required=True,type=Path)
    parser.add_argument('--val-root',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=S.DATA/'detector-training')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--person-class-ids',nargs='+',type=int)
    group.add_argument('--all-boxes-are-people',action='store_true')
    parser.add_argument('--box-origin',type=int,choices=[0,1],default=1)
    args=parser.parse_args()
    sequences={split:sorted(p.parent for p in folder.glob('*/seqinfo.ini'))
               for split,folder in [('train',args.train_root),('val',args.val_root)]}
    if not all(sequences.values()):parser.error('Both official split directories must contain sequences')
    if {p.name for p in sequences['train']} & {p.name for p in sequences['val']}:
        parser.error('Train and validation sequence names overlap')
    if args.output.exists():parser.error('Choose a new output directory; existing training data is preserved')
    args.output.mkdir(parents=True)
    counts={}
    for split,folders in sequences.items():
        counts[split]=sum(convert_sequence(p,args.output,split,set(args.person_class_ids or []),
                                         args.all_boxes_are_people,args.box_origin) for p in folders)
    import yaml
    (args.output/'data.yaml').write_text(yaml.safe_dump({'path':str(args.output.resolve()),'train':'images/train',
                                                       'val':'images/val','names':{0:'person'}}),encoding='utf-8')
    S.write_json(args.output/'provenance.json',{'sequences':{k:[str(p.resolve()) for p in v] for k,v in sequences.items()},
                                              'image_counts':counts,'person_class_ids':args.person_class_ids,
                                              'all_boxes_are_people':args.all_boxes_are_people,'created':S.now()})
    print(args.output/'data.yaml',counts)


if __name__=='__main__':main()
