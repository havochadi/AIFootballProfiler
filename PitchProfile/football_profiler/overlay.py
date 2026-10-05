"""Render coloured tracks, short image trails and calibrated speeds for review."""
from collections import defaultdict, deque
import os
import cv2
import numpy as np
from . import storage as S
from .features import observed_points
from .runtime import torch_device
from .vision import video_info, encode_overlay


def render(identifier, progress=lambda *a:None):
    directory=S.dataset_dir(identifier)
    manifest=S.read_json(directory/'manifest.json')
    if manifest.get('source_kind')!='model_predictions':raise ValueError('Choose a processed video dataset')
    source=S.video_path(directory,manifest)
    info=video_info(source);device=str(torch_device())
    tracks=S.load_tracks(identifier)
    kits=S.read_json(directory/'kit_groups.json',{}).get('tracks',{})
    people={str(p['player_id']):p for p in manifest['players']}
    valid=observed_points(tracks)
    speed={}
    for pid,q in valid.groupby('player_id'):
        q=q.sort_values('frame')
        dt=q.time_s.diff();dist=np.hypot(q.x.diff(),q.y.diff());v=dist/dt
        good=dt.gt(0)&dt.le(1.5/manifest['sampling_hz']+1e-6)&v.le(12)&q.period.diff().eq(0)
        speed.update({(int(frame),pid):float(value*3.6) for frame,value in zip(q.loc[good,'frame'],v[good])})
    groups={int(frame):rows for frame,rows in tracks.groupby('frame')}
    trails=defaultdict(lambda:deque(maxlen=12));last={}
    capture=cv2.VideoCapture(str(source));writer=None
    avi=directory/'enhanced-overlay.avi';encoded=directory/'enhanced-overlay.mp4'
    fps=manifest['sampling_hz'];count=manifest['total_sampled_frames']
    try:
        for index in range(count):
            source_index=int(round((manifest.get('source_offset_s',0)+index/fps)*info['fps']))
            capture.set(cv2.CAP_PROP_POS_FRAMES,source_index)
            ok,frame=capture.read()
            if not ok:raise ValueError('A source frame could not be read while rendering the overlay')
            rows=groups.get(index)
            if rows is not None:
                for row in rows.itertuples():
                    pid=row.player_id;person=people[pid];kit=kits.get(pid,{})
                    hex_colour=kit.get('colour','#37d2a5').lstrip('#')
                    colour=tuple(int(hex_colour[i:i+2],16) for i in (4,2,0))
                    x,y,w,h=map(int,(row.bbox_x,row.bbox_y,row.bbox_w,row.bbox_h))
                    feet=(x+w//2,y+h)
                    if index-last.get(pid,-2)>1:trails[pid].clear()
                    trails[pid].append(feet);last[pid]=index
                    if len(trails[pid])>1:cv2.polylines(frame,[np.array(trails[pid],np.int32)],False,colour,2)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),colour,2)
                    name=person.get('name',pid)
                    label=name if person.get('identity_verified') else pid
                    if (index,pid) in speed:label+=f' {speed[index,pid]:.1f} km/h'
                    elif kit:label+=' '+kit['group']
                    cv2.putText(frame,label,(x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,colour,1,cv2.LINE_AA)
            cv2.rectangle(frame,(0,0),(min(frame.shape[1],1020),32),(25,34,39),-1)
            cv2.putText(frame,'Kit colours are suggestions | pixel trails include camera motion | speeds need valid pitch mapping',
                        (8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            if writer is None:
                writer=cv2.VideoWriter(str(avi),cv2.VideoWriter_fourcc(*'MJPG'),fps,(info['width'],info['height']))
                if not writer.isOpened():raise ValueError('Unable to create enhanced overlay')
            writer.write(frame)
            if index%20==0:progress(.9*index/max(count,1),f'Rendering overlay {index+1}/{count}')
    finally:
        capture.release()
        if writer:writer.release()
    progress(.95,'Encoding annotated overlay on GPU')
    encoder=encode_overlay(avi,encoded,device)
    os.replace(encoded,directory/'overlay.mp4');avi.unlink()
    manifest['overlay']='overlay.mp4'
    manifest['processing'].update(overlay_style='kit colours, image trails, calibrated speeds when available',overlay_encoder=encoder)
    S.write_json(directory/'manifest.json',manifest)
    progress(1,'Annotated overlay saved')
    return {'dataset_id':identifier,'encoder':encoder}
