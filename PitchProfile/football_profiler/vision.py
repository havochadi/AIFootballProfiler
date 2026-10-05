from __future__ import annotations
import os
import shutil
import time
import subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from . import storage as S
from .features import build_profiles
from .runtime import torch_device, runtime_info

def detector():
    os.environ.setdefault("YOLO_CONFIG_DIR",str(S.DATA/"ultralytics_config"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("YOLO_AUTOINSTALL","false")
    from ultralytics import YOLO
    weights=Path(os.environ.get('PITCHPROFILE_DETECTOR_WEIGHTS') or S.ROOT/"models/yolo11n.pt")
    if not weights.is_file():raise ValueError("YOLO11n weights are missing. Restore the bundled models/yolo11n.pt file.")
    return YOLO(str(weights))

def ffmpeg_executable():
    executable=shutil.which("ffmpeg")
    if executable:return executable
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError,RuntimeError,OSError):
        return None

def encode_overlay(source,destination,device):
    executable=ffmpeg_executable()
    if not executable:raise ValueError("FFmpeg is unavailable. Install imageio-ffmpeg or put FFmpeg on PATH to create the browser overlay.")
    gpu=str(device).startswith("cuda")
    encoder="h264_nvenc" if gpu else "libx264"
    options=["-c:v",encoder]
    if gpu:
        index=str(device).split(":",1)[1] if ":" in str(device) else "0"
        options += ["-gpu",index,"-preset","p4","-cq","23","-b:v","0"]
    else:options += ["-preset","veryfast","-crf","23"]
    command=[executable,"-hide_banner","-loglevel","error","-y","-i",str(source),*options,"-pix_fmt","yuv420p","-movflags","+faststart",str(destination)]
    try:
        subprocess.run(command,check=True,timeout=120,capture_output=True)
    except (OSError,subprocess.SubprocessError) as exc:
        detail=getattr(exc,"stderr",b"") or b""
        if isinstance(detail,bytes):detail=detail.decode("utf-8",errors="replace")
        message=("GPU overlay encoding failed. Check that FFmpeg supports h264_nvenc and the NVIDIA driver supports NVENC."
                 if gpu else "CPU overlay encoding failed. Check FFmpeg codec support and available disk space.")
        raise ValueError(message+(" "+detail.strip()[-600:] if detail else "")) from exc
    return encoder

def video_info(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError("Video could not be opened. Use MP4, MOV, AVI or WebM.")
    info={"fps":float(cap.get(cv2.CAP_PROP_FPS)),"frames":int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),"width":int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),"height":int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    if info["fps"]<=0 or info["frames"]<=0 or min(info["width"],info["height"])<=0:raise ValueError("Video metadata is invalid")
    info["duration"]=info["frames"]/info["fps"]
    return info

def read_frame(path, seconds=0):
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_MSEC,max(0,seconds)*1000)
    ok,frame=cap.read();cap.release()
    if not ok:raise ValueError("Frame is unavailable at that timestamp")
    return frame

def process_video(identifier,input_path,title,sampling_hz=5,max_seconds=60,imgsz=960,progress=lambda *a:None,source_note=None,
                  start_s=0,tracker='botsort',library_id=None,camera_cuts=None,match_id=None,benchmark_split=None):
    from .kits import torso_colour, group_colours
    from .match_analysis import CutDetector
    if not .5<=sampling_hz<=15 or not 1<=max_seconds<=3600:raise ValueError("Use 0.5–15 samples/s and 1–3600 seconds per job")
    if tracker not in ('botsort','bytetrack'):raise ValueError('Choose botsort or bytetrack')
    tracker_config=str(S.ROOT/'config/botsort_pitchprofile.yaml') if tracker=='botsort' else 'bytetrack.yaml'
    device=str(torch_device())
    d=S.dataset_dir(identifier,True);info=video_info(input_path)
    if not np.isfinite(start_s) or not 0<=start_s<info['duration']:raise ValueError('Start must be inside the source video')
    start_frame=int(round(start_s*info['fps']));source_offset=start_frame/info['fps']
    interval=max(1,round(info["fps"]/sampling_hz));actual_hz=info["fps"]/interval
    nframes=min(info["frames"]-start_frame,int(max_seconds*info["fps"]))
    model=detector();cap=cv2.VideoCapture(str(input_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    rows=[];sampled=0;scene=0;seen_frame=False;started=time.perf_counter()
    output=d/"overlay.avi";writer=None;cuts=[];colours={}
    known_cuts=np.asarray(sorted(camera_cuts or []))-source_offset
    is_cut=CutDetector()
    try:
        for index in range(nframes):
            ok,frame=cap.read()
            if not ok:break
            if index%interval:continue
            if sampled==0:cv2.imwrite(str(d/"preview.jpg"),frame)
            # No track identity is carried across a detected cut.
            cut=is_cut(frame) or (seen_frame and
                bool(((known_cuts>(index-interval)/info['fps'])&(known_cuts<=index/info['fps'])).any()))
            if cut:
                scene+=1;cuts.append(index/info["fps"])
                for active_tracker in getattr(getattr(model,"predictor",None),"trackers",[]):active_tracker.reset()
            seen_frame=True
            result=model.track(frame,persist=True,classes=[0],conf=.12,iou=.5,imgsz=imgsz,tracker=tracker_config,device=device,verbose=False)[0]
            boxes=result.boxes
            raw_frame=frame.copy()
            if boxes.id is not None:
                for b,tid,score in zip(boxes.xyxy.cpu().numpy(),boxes.id.cpu().numpy().astype(int),boxes.conf.cpu().numpy()):
                    pid=f"s{scene}-t{tid}"
                    rows.append((sampled,index/info["fps"],pid,pid,scene+1,np.nan,np.nan,1,0,float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1]),float(score),start_frame+index))
                    samples=colours.setdefault(pid,[])
                    if len(samples)<120:
                        colour=torso_colour(raw_frame,b)
                        if colour is not None:samples.append(colour)
                    cv2.rectangle(frame,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(55,210,165),2)
                    cv2.putText(frame,pid,(int(b[0]),max(18,int(b[1])-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,(55,210,165),2)
            cv2.rectangle(frame,(0,0),(min(frame.shape[1],800),34),(25,34,39),-1)
            cv2.putText(frame,"PitchProfile predictions | person IDs require team confirmation",(10,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            if writer is None:
                writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"MJPG"),actual_hz,(info["width"],info["height"]))
                if not writer.isOpened():raise ValueError("The tracking overlay could not be written. Check available disk space and video codec support.")
            writer.write(frame);sampled+=1
            if sampled%5==0:progress(min(.9,index/max(nframes,1)),f"Tracked {sampled} sampled frames")
    finally:
        cap.release()
        if writer:writer.release()
    if sampled==0:raise ValueError("No frames were processed")
    cols=["frame","time_s","player_id","track_id","period","x","y","detected","calibration_valid","bbox_x","bbox_y","bbox_w","bbox_h","confidence","source_frame"]
    tracks=pd.DataFrame(rows,columns=cols)
    S.save_tracks(identifier,tracks)
    players=[]
    for pid,q in tracks.groupby("player_id"):
        players.append({"player_id":pid,"name":pid,"team":"Unconfirmed","role":"Unconfirmed","identity_verified":False,"global_id":None,"direction_known":False,"eligible_frames":sampled,"playing_seconds":sampled/actual_hz,"first_seen":float(q.time_s.min()),"last_seen":float(q.time_s.max()),"coverage_note":"Detection share of the entire analysed clip, not confirmed playing time. Spectators or staff can be detected; confirm the player identity."})
    detector_name=('YOLO11n pretrained COCO' if not os.environ.get('PITCHPROFILE_DETECTOR_WEIGHTS')
                   else 'User-supplied detector: '+Path(os.environ['PITCHPROFILE_DETECTOR_WEIGHTS']).name)
    m={"id":identifier,"title":title,"match_id":match_id or identifier,"date":None,"source":source_note or "Uploaded video processed with a YOLO detector","source_kind":"model_predictions","sampling_hz":actual_hz,"duration_seconds":min(sampled/actual_hz,nframes/info['fps']),"total_sampled_frames":sampled,"width":info["width"],"height":info["height"],"players":players,"preview":"preview.jpg","video":Path(input_path).name,"overlay":None,"note":"Model detections. Pitch coordinates remain unavailable until calibration succeeds; identities and teams require confirmation.","cut_times":cuts,"created":S.now(),"processing":{"wall_seconds":time.perf_counter()-started,"sampled_frames":sampled,"device":device,"device_name":runtime_info().get("gpu_name") if device.startswith("cuda") else "CPU","detector":detector_name,"tracker":"BoT-SORT + ORB" if tracker=='botsort' else "ByteTrack","image_size":imgsz}}
    m.update(source_fps=info['fps'],source_offset_s=source_offset,clock='analysed_interval_relative',
             soccernet_library_id=library_id,benchmark_split=benchmark_split,
             benchmark_split_source='SoccerNet official match split' if benchmark_split else None,
             camera_cut_source='SoccerNet manual cuts + image change detector' if camera_cuts else 'image change detector')
    progress(.92,'Grouping observed kit colours on GPU')
    S.write_json(d/'kit_groups.json',group_colours(colours))
    progress(.95,"Encoding the browser overlay")
    m["processing"]["overlay_encoder"]=encode_overlay(output,d/"overlay.mp4",device)
    m["overlay"]="overlay.mp4";output.unlink()
    S.write_json(d/"manifest.json",m);build_profiles(identifier)
    progress(1,"Video analysis complete")
    return m

def fit_homography(image_points,pitch_points):
    a=np.asarray(image_points,dtype=np.float64);b=np.asarray(pitch_points,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2 or len(a)<4:raise ValueError("Choose at least four corresponding image and pitch points")
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("Calibration points must be finite")
    if (b<0).any() or (b[:,0]>105).any() or (b[:,1]>68).any():raise ValueError("Pitch coordinates must lie within 105 by 68 metres")
    if np.linalg.matrix_rank(np.c_[a,np.ones(len(a))])<3 or np.linalg.matrix_rank(np.c_[b,np.ones(len(b))])<3:raise ValueError("Calibration points must not be collinear")
    h,mask=cv2.findHomography(a,b,cv2.RANSAC,1.5)
    if h is None or abs(np.linalg.det(h))<1e-14 or mask.sum()<4:raise ValueError("These points do not produce a valid pitch mapping")
    projected=cv2.perspectiveTransform(a[None].astype(np.float32),h)[0]
    error=np.linalg.norm(projected-b,axis=1)
    return h,{"control_point_rmse_m":float(np.sqrt(np.mean(error[mask.ravel().astype(bool)]**2))),"inliers":int(mask.sum()),"points":len(a),"note":"Fit error on supplied control points, not independent localisation accuracy."}

def map_points(points,h):
    p=np.asarray(points,dtype=np.float32)
    return cv2.perspectiveTransform(p[None],np.asarray(h,dtype=np.float64))[0]

def calibrate(identifier,image_points,pitch_points,reference_s=0,start_s=0,end_s=None,static_camera=False,progress=lambda *a:None):
    d=S.dataset_dir(identifier);m=S.read_json(d/"manifest.json")
    if m.get("source_kind")!="model_predictions" or not m.get("video"):raise ValueError("Calibrate a processed video dataset")
    end_s=float(end_s if end_s is not None else m["duration_seconds"])
    if not 0<=start_s<end_s<=m["duration_seconds"]+.05:raise ValueError("Calibration interval must be inside the analysed clip")
    if not start_s<=reference_s<end_s:raise ValueError("Reference frame must lie inside the calibration interval")
    if any(start_s<float(cut)<end_s for cut in m.get("cut_times",[])):
        raise ValueError("The calibration interval crosses a detected camera cut. Calibrate each continuous camera segment separately.")
    h,quality=fit_homography(image_points,pitch_points)
    source_offset=m.get('source_offset_s',0)
    src=S.video_path(d,m);anchor=read_frame(src,reference_s+source_offset)
    orb=cv2.ORB_create(nfeatures=2400)
    # User-selected field landmarks delimit the feature region; exclude outside stands.
    anchor_mask=np.zeros(anchor.shape[:2],np.uint8)
    hull=cv2.convexHull(np.asarray(image_points,np.int32));cv2.fillConvexPoly(anchor_mask,hull,255)
    ak,ad=orb.detectAndCompute(cv2.cvtColor(anchor,cv2.COLOR_BGR2GRAY),anchor_mask)
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    tracks=S.load_tracks(identifier);frames=tracks.loc[tracks.time_s.between(start_s,end_s,inclusive="left"),["frame","time_s"]].drop_duplicates("frame")
    left_players={str(p["player_id"]) for p in m["players"] if p.get("direction")=="left"}
    tracks.loc[tracks.time_s.between(start_s,end_s,inclusive="left"),["x","y"]]=np.nan
    tracks.loc[tracks.time_s.between(start_s,end_s,inclusive="left"),"calibration_valid"]=0
    valid=0;details=[]
    cap=cv2.VideoCapture(str(src))
    try:
        for i,(_,record) in enumerate(frames.iterrows()):
            current_h=None;inliers=None
            if static_camera:current_h=h
            elif abs(record.time_s-reference_s)<=.5/m["sampling_hz"]:current_h=h
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC,(float(record.time_s)+source_offset)*1000);ok,img=cap.read()
                if ok and ad is not None:
                    ck,cd=orb.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)
                    if cd is not None and len(cd)>=2:
                        pairs=matcher.knnMatch(ad,cd,k=2)
                        good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
                        if len(good)>=12:
                            ref=np.float32([ak[x.queryIdx].pt for x in good]);cur=np.float32([ck[x.trainIdx].pt for x in good])
                            transform,mask=cv2.findHomography(cur,ref,cv2.RANSAC,3.)
                            if transform is not None and mask is not None and mask.sum()>=10 and mask.mean()>=.6:
                                current_h=h@transform;inliers=int(mask.sum())
            selected=tracks.frame.eq(record.frame)
            if current_h is not None:
                q=tracks.loc[selected];feet=np.c_[q.bbox_x+q.bbox_w/2,q.bbox_y+q.bbox_h]
                xy=map_points(feet,current_h)
                left=q.player_id.astype(str).isin(left_players).to_numpy()
                xy[left]=np.array([105,68])-xy[left]
                sane=np.isfinite(xy).all(axis=1)&(np.abs(xy).max(axis=1)<10000)
                tracks.loc[q.index[sane],["x","y"]]=xy[sane]
                tracks.loc[q.index[sane],"calibration_valid"]=1
                valid+=1
            details.append({"frame":int(record.frame),"valid":current_h is not None,"inliers":inliers})
            if i%10==0:progress(i/max(len(frames),1),f"Calibrated {i+1}/{len(frames)} sampled frames")
    finally:cap.release()
    # The homography uses stadium orientation; stored positions preserve each player's
    # confirmed attack direction, including when recalibrating an existing interval.
    S.save_tracks(identifier,tracks)
    m["calibration"]={"reference_s":reference_s,"start_s":start_s,"end_s":end_s,"static_camera_confirmed":static_camera,"valid_sampled_frames":valid,"attempted_sampled_frames":len(frames),"quality":quality,"image_points":image_points,"pitch_points":pitch_points,"H":h.tolist(),"frames":details}
    S.write_json(d/"manifest.json",m);build_profiles(identifier);progress(1,"Calibration finished")
    return m["calibration"]

def detection_sanity(image_path,annotations_path,outdir):
    import json
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    device=str(torch_device())
    model=detector();start=time.perf_counter();result=model.predict(str(image_path),classes=[0],conf=.25,imgsz=1280,device=device,verbose=False)[0]
    data=json.loads(Path(annotations_path).read_text());image_id=str(data["images"][0]["image_id"])
    gt=[]
    for a in data["annotations"]:
        if str(a.get("image_id"))!=image_id or a.get("attributes",{}).get("role") not in ("player","goalkeeper","referee"):continue
        b=a.get("bbox_image")
        if b:gt.append([b["x"],b["y"],b["x"]+b["w"],b["y"]+b["h"]])
    pred=result.boxes.xyxy.cpu().numpy();scores=result.boxes.conf.cpu().numpy();used=set();tp=0
    for i in np.argsort(-scores):
        p=pred[i];best=-1;best_iou=0
        for j,g in enumerate(gt):
            if j in used:continue
            inter=np.prod(np.maximum(0,np.minimum(p[2:],g[2:])-np.maximum(p[:2],g[:2])))
            union=np.prod(p[2:]-p[:2])+np.prod(np.asarray(g[2:])-g[:2])-inter
            iou=inter/union if union else 0
            if iou>best_iou:best=j;best_iou=iou
        if best_iou>=.5:used.add(best);tp+=1
    cv2.imwrite(str(outdir/"single_frame_detection.jpg"),result.plot())
    report={"task":"Single-image person detection sanity check","dataset":"SoccerNet SNGS-060 first image","reference_people":len(gt),"predicted_people":len(pred),"true_positive_iou_05":tp,"precision":tp/len(pred) if len(pred) else None,"recall":tp/len(gt) if gt else None,"confidence_threshold":.25,"image_size":1280,"wall_seconds":time.perf_counter()-start,"limitation":"One image only; no tracking, archetype or generalisation accuracy claim. Pretrained COCO detector; not fine-tuned on these data."}
    report.update(device=device,device_name=runtime_info().get("gpu_name") if device.startswith("cuda") else "CPU")
    S.write_json(outdir/"detection_sanity.json",report)
    return report
