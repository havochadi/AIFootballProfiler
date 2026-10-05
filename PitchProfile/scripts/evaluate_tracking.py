"""Run official TrackEval HOTA, CLEAR and Identity metrics on local MOT files.

The GT folder contains sequence/seqinfo.ini and sequence/gt/gt.txt. Predictions
are one sequence.txt per sequence, using the exact same frame numbering and
evaluation sampling as the ground truth. This does not download ground truth.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from contextlib import contextmanager
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from football_profiler import storage as S


@contextmanager
def legacy_numpy_aliases():
    # Upstream still refers to removed aliases. Scope compatibility to its
    # modules, leaving PitchProfile's NumPy and numerical algorithms untouched.
    class NumpyProxy:
        float=float
        int=int
        bool=bool
        def __getattr__(self,name):return getattr(np,name)
    modules=[m for name,m in list(sys.modules.items()) if name.startswith('trackeval.') and getattr(m,'np',None) is np]
    for module in modules:module.np=NumpyProxy()
    try:yield
    finally:
        for module in modules:module.np=np


def evaluate(gt_folder, prediction_folder, output, sequences=None):
    import trackeval
    sequences=sequences or sorted(p.parent.name for p in gt_folder.glob('*/seqinfo.ini'))
    if not sequences:raise ValueError('No MOT sequences found')
    for name in sequences:
        if '/' in name or '\\' in name or name in ('.','..'):raise ValueError('Invalid sequence name')
        if not (gt_folder/name/'gt/gt.txt').is_file() or not (prediction_folder/(name+'.txt')).is_file():
            raise ValueError('Missing paired ground truth or predictions for '+name)
    output.mkdir(parents=True,exist_ok=True)
    evaluator=trackeval.Evaluator({'USE_PARALLEL':False,'PRINT_RESULTS':False,'PRINT_CONFIG':False,
                                   'OUTPUT_SUMMARY':True,'OUTPUT_DETAILED':True,'PLOT_CURVES':False,
                                   'LOG_ON_ERROR':str(output/'error.log')})
    dataset=trackeval.datasets.MotChallenge2DBox({
        'GT_FOLDER':str(gt_folder.resolve()),'TRACKERS_FOLDER':str(prediction_folder.resolve().parent),
        'TRACKERS_TO_EVAL':[prediction_folder.name],'TRACKER_SUB_FOLDER':'',
        'OUTPUT_FOLDER':str(output.resolve()),'SKIP_SPLIT_FOL':True,'DO_PREPROC':False,
        'SEQ_INFO':{name:None for name in sequences},'PRINT_CONFIG':False})
    metrics=[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),
             trackeval.metrics.Identity({'PRINT_CONFIG':False})]
    with legacy_numpy_aliases():
        results,messages=evaluator.evaluate([dataset],metrics)
    report={'results':results,'messages':messages,'sequences':sequences,
            'note':'Official TrackEval metrics for supplied aligned person boxes only; no archetype evaluation.'}
    S.write_json(output/'tracking_metrics.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-folder',type=Path,required=True)
    parser.add_argument('--prediction-folder',type=Path,required=True)
    parser.add_argument('--sequences',nargs='+')
    parser.add_argument('--output',type=Path,default=S.EVIDENCE/'tracking-evaluation')
    args=parser.parse_args()
    evaluate(args.gt_folder,args.prediction_folder,args.output,args.sequences)
    print('Saved evaluation to '+str(args.output))


if __name__=='__main__':main()
