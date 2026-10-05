"""Fine-tune a person detector on a reviewed YOLO dataset using CUDA.

Raw broadcast videos and match-event labels are not detection ground truth.
Supply a YOLO data.yaml with person boxes and disjoint match-level train/val sets.
Outputs go to the configured data drive; this never replaces active weights.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from football_profiler import storage as S
from football_profiler.runtime import torch_device


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--weights',type=Path,default=ROOT/'models/yolo11n.pt')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--imgsz',type=int,default=960)
    args=parser.parse_args()
    if not args.data.is_file() or not args.weights.is_file():parser.error('Data YAML and initial weights must exist locally')
    if not 1<=args.epochs<=300 or args.batch<1:parser.error('Use 1–300 epochs and a positive batch size')
    device=torch_device()
    if device.type!='cuda':parser.error('Detector training requires your NVIDIA GPU')
    import yaml
    config=yaml.safe_load(args.data.read_text(encoding='utf-8'))
    names=config.get('names')
    names=list(names.values()) if isinstance(names,dict) else names
    if names!=['person']:parser.error('This app tracks class 0=person. Supply a one-class person detection dataset.')
    if not config.get('train') or not config.get('val') or config['train']==config['val']:
        parser.error('Provide separate match-level train and validation inputs')
    os.environ.setdefault('YOLO_CONFIG_DIR',str(S.DATA/'ultralytics_config'))
    from ultralytics import YOLO
    model=YOLO(str(args.weights))
    model.train(data=str(args.data.resolve()),epochs=args.epochs,batch=args.batch,imgsz=args.imgsz,
                device=str(device),workers=0,seed=42,project=str(S.DATA/'models/detector'),
                name='soccernet-person',exist_ok=False,pretrained=True)
    print('Training finished. Evaluate the saved weights on separate matches before setting PITCHPROFILE_DETECTOR_WEIGHTS.')


if __name__=='__main__':main()
