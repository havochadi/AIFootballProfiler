from __future__ import annotations
import io
import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from . import storage as S
from . import features as F
from . import taxonomy as T
from .match_api import router as match_router

app=FastAPI(title="PitchProfile",version="0.1.0")
app.include_router(match_router)
jobs={};jobs_lock=threading.Lock();pool=ThreadPoolExecutor(max_workers=1)
STATIC=S.ROOT/"static"

@app.exception_handler(ValueError)
async def value_error(request,exc):return JSONResponse({"detail":str(exc)},status_code=400)
@app.exception_handler(FileNotFoundError)
async def missing(request,exc):return JSONResponse({"detail":str(exc)},status_code=404)

@app.middleware("http")
async def local_origin(request:Request,call_next):
    if request.method in {"POST","PUT","DELETE","PATCH"}:
        origin=request.headers.get("origin")
        if origin and origin.rstrip("/")!=str(request.base_url).rstrip("/"):
            return JSONResponse({"detail":"Use the application's own page for changes"},status_code=403)
        if int(request.headers.get("content-length","0"))>512*1024*1024:
            return JSONResponse({"detail":"Upload limit is 512 MB"},status_code=413)
    return await call_next(request)

def get_player(dataset_id,player_id):
    d=S.dataset_dir(dataset_id);m=S.read_json(d/"manifest.json")
    p=next((p for p in m["players"] if str(p["player_id"])==str(player_id)),None)
    if p is None:raise HTTPException(404,"Player is unavailable")
    return d,m,p

def start_job(kind,func,**kwargs):
    jid=uuid.uuid4().hex[:12]
    with jobs_lock:
        if sum(j["status"] in ("queued","running") for j in jobs.values())>=3:raise HTTPException(429,"Three jobs are already queued; wait for the current work to finish")
        jobs[jid]={"id":jid,"kind":kind,"status":"queued","progress":0,"message":"Queued","created":S.now()}
    def progress(fraction,message):
        with jobs_lock:jobs[jid].update(progress=float(fraction),message=message)
    def run():
        with jobs_lock:jobs[jid]["status"]="running"
        try:
            result=func(progress=progress,**kwargs)
            with jobs_lock:jobs[jid].update(status="complete",progress=1,result=S.clean_json(result),message="Complete")
        except Exception as e:
            traceback.print_exc()
            with jobs_lock:jobs[jid].update(status="failed",message=str(e),error=type(e).__name__)
        S.write_json(S.DATA/"jobs"/(jid+".json"),jobs[jid])
    pool.submit(run)
    return {"job_id":jid}

@app.get("/api/status")
def status():
    from .runtime import runtime_info
    return {"version":"0.1.0","datasets":len(S.datasets()),"position_groups":T.GROUPS,"runtime":runtime_info()}

@app.get("/api/datasets")
def datasets():return S.datasets()

@app.get("/api/datasets/{identifier}")
def dataset(identifier):
    d=S.dataset_dir(identifier);m=S.read_json(d/"manifest.json")
    profiles=S.read_json(d/"profiles.json",[])
    if str(m.get('analysis','')).startswith('full-match'):
        from . import match_context as MC
        context=MC.state_for(m)
        profiles=[MC.decorate(m,p,context) for p in profiles]
    return {"manifest":m,"profiles":profiles}

@app.get("/api/datasets/{identifier}/players/{pid}")
def player(identifier,pid):
    d,m,p=get_player(identifier,pid)
    result=next(x for x in S.read_json(d/"profiles.json") if x["player_id"]==pid)
    if str(m.get('analysis','')).startswith('full-match'):
        from . import match_context as MC
        result=MC.decorate(m,result)
    return {**result,"manual_events":S.manual_events(identifier,pid),"manifest_player":p}

@app.get("/api/datasets/{identifier}/players/{pid}/path")
def trajectory(identifier,pid):
    get_player(identifier,pid);t=S.load_tracks(identifier);q=F.observed_points(t[t.player_id.eq(pid)])
    stride=max(1,len(q)//1200)
    return S.clean_json(q.iloc[::stride][["time_s","x","y","frame","period"]].to_dict("records"))

@app.get("/api/datasets/{identifier}/players/{pid}/history")
def history(identifier,pid):
    _,m,p=get_player(identifier,pid)
    path=S.DATA/"player_history.csv"
    if not p.get("identity_verified") or not str(p.get("global_id","")).startswith("skillcorner:") or not m.get("date") or not path.exists():
        return {"matches":[],"note":"Historical comparison requires verified records for this exact player."}
    data=pd.read_csv(path,dtype={"player_id":str,"match_id":str})
    q=data[data.player_id.eq(p["global_id"].split(":",1)[1])&(pd.to_datetime(data.date)<pd.to_datetime(m["date"]))]
    cols=["match_id","date","position_group","minutes_played","possession_end_pass_per90","possession_end_shot_per90","run_behind_per90","on_ball_engagement_per90"]
    return S.clean_json({"matches":q.sort_values("date")[cols].to_dict("records"),"note":"SkillCorner provider event rates from earlier fixtures only. Minimum 60 playing minutes per appearance; these are not video-derived counts."})

@app.get("/media/{identifier}/{filename}")
def media(identifier,filename):
    d=S.dataset_dir(identifier);m=S.read_json(d/"manifest.json",{})
    allowed={m.get("video"),m.get("overlay"),m.get("preview"),"preview.jpg"}
    if filename not in allowed or filename is None:raise HTTPException(404,"Media file unavailable")
    path=S.video_path(d,m) if filename==m.get('video') else d/filename
    if not path.is_file():raise HTTPException(404,"Media file unavailable")
    return FileResponse(path)

@app.get("/api/datasets/{identifier}/frame")
def frame(identifier,seconds:float=0):
    from .vision import read_frame
    d=S.dataset_dir(identifier);m=S.read_json(d/"manifest.json")
    if not m.get("video"):
        if m.get("preview") and seconds==0:return FileResponse(d/m["preview"])
        raise ValueError("This reference dataset does not include the source video")
    video_start=m.get('video_start_s',0)
    if not video_start<=seconds<m["duration_seconds"]:raise ValueError("Timestamp is outside the available video interval")
    img=read_frame(S.video_path(d,m),seconds-video_start+m.get('source_offset_s',0));ok,encoded=cv2.imencode(".jpg",img)
    if not ok:raise ValueError("Unable to encode frame")
    return Response(encoded.tobytes(),media_type="image/jpeg")

@app.get('/api/soccernet/library')
def soccernet_library():
    from . import soccernet as SN
    return {'root':str(SN.video_root()),'videos':SN.library()}

@app.get('/api/datasets/{identifier}/players/{pid}/analytics')
def player_analytics(identifier,pid):
    from .analytics import player_analysis
    return player_analysis(identifier,pid)

@app.get('/api/datasets/{identifier}/tactics')
def tactics(identifier,seconds:float=0):
    from .analytics import snapshot
    manifest=S.read_json(S.dataset_dir(identifier)/'manifest.json')
    if not np.isfinite(seconds) or not 0<=seconds<manifest['duration_seconds']:raise ValueError('Timestamp is outside the analysed interval')
    return S.clean_json(snapshot(S.load_tracks(identifier),manifest,seconds))

@app.get('/api/datasets/{identifier}/match-events')
def match_events(identifier):
    return S.read_json(S.dataset_dir(identifier)/'match_events.json',{'events':[],'note':'No aligned SoccerNet match labels for this source.'})

@app.get('/api/datasets/{identifier}/occupancy')
def team_occupancy(identifier):
    from .analytics import occupancy
    return occupancy(identifier)

@app.get('/api/datasets/{identifier}/mot')
def export_mot(identifier):
    from .analytics import mot_export
    return Response(mot_export(identifier),media_type='application/zip',
                    headers={'Content-Disposition':f'attachment; filename={identifier}_mot.zip'})

@app.post('/api/datasets/{identifier}/render-overlay')
def render_enhanced_overlay(identifier):
    from .overlay import render
    S.dataset_dir(identifier)
    return start_job('overlay',render,identifier=identifier)

@app.get('/api/soccernet/library/{library_id}/events')
def soccernet_events(library_id):
    from . import soccernet as SN
    item=SN.entry(library_id)
    return {'events':SN.annotations(item['game'],item['half']),
            'note':'SoccerNet manual match events; no player attribution. Times are half-relative seconds.'}

@app.get("/api/jobs/{jid}")
def job(jid):
    if jid not in jobs:raise HTTPException(404,"Job unavailable in this session")
    return jobs[jid]

class Identity(BaseModel):
    name:str=Field(min_length=1,max_length=120)
    team:str=Field(default="Unconfirmed",max_length=120)
    role:str=Field(default="Unconfirmed",max_length=60)
    position_group:str|None=None
    global_id:str|None=None
    identity_verified:bool=False
    direction:str="unknown"

@app.post("/api/datasets/{identifier}/players/{pid}/identity")
def identity(identifier,pid,payload:Identity):
    d,m,p=get_player(identifier,pid)
    if payload.direction not in ("unknown","right","left"):raise ValueError("Choose unknown, right or left attacking direction")
    if payload.position_group is not None and payload.position_group not in T.GROUPS:raise ValueError("Choose a position group from the list, or leave it unset")
    if payload.identity_verified and not payload.global_id:raise ValueError("A persistent identity key is required for verified history")
    # SkillCorner/reference imports retain their documented orientation. Uploaded
    # clips and SoccerTrack halves can be confirmed against their source video.
    can_orient = m['source_kind'] in ('model_predictions', 'soccertrack_reference')
    if can_orient:
        t=S.load_tracks(identifier);sel=t.player_id.eq(pid)
        old=p.get("direction","unknown")
        if (old=="left")!=(payload.direction=="left"):
            t.loc[sel,"x"]=105-t.loc[sel,"x"];t.loc[sel,"y"]=68-t.loc[sel,"y"]
            S.save_tracks(identifier,t)
    elif payload.direction!=p.get("direction","unknown") and payload.direction!="unknown":
        raise ValueError("Reference coordinates use their documented orientation. Import a reviewed normalised dataset to change it.")
    p.update(payload.model_dump(exclude={"direction"}));p["direction"]=payload.direction
    if can_orient:p["direction_known"]=payload.direction!="unknown"
    S.write_json(d/"manifest.json",m);F.build_profiles(identifier)
    return {"saved":True}

class Event(BaseModel):
    time_s:float
    kind:str
    reviewer:str
    notes:str=""

@app.post("/api/datasets/{identifier}/players/{pid}/events")
def add_event(identifier,pid,payload:Event):
    _,m,_=get_player(identifier,pid)
    if payload.kind not in ("pass","shot","tackle","interception","run_behind","other"):raise ValueError("Unknown event type")
    if not 0<=payload.time_s<m["duration_seconds"] or not payload.reviewer.strip():raise ValueError("Enter a valid timestamp and reviewer")
    with S.db() as c:
        c.execute("INSERT INTO manual_events(dataset_id,player_id,time_s,kind,reviewer,notes,created) VALUES(?,?,?,?,?,?,?)",(identifier,pid,payload.time_s,payload.kind,payload.reviewer.strip()[:80],payload.notes[:1000],S.now()))
    return {"saved":True,"events":S.manual_events(identifier,pid)}

@app.get("/api/datasets/{identifier}/export")
def export_dataset(identifier):return FileResponse(S.dataset_dir(identifier)/"tracks.csv.gz",filename=identifier+"_tracks.csv.gz",media_type="application/gzip")

@app.post("/api/import-tracks")
async def import_tracks(tracks:UploadFile=File(...),manifest:UploadFile=File(...)):
    from .importers import import_canonical
    tb=await tracks.read(100*1024*1024+1);mb=await manifest.read(2*1024*1024+1)
    if len(tb)>100*1024*1024 or len(mb)>2*1024*1024:raise ValueError("Tracking import is limited to 100 MB plus a 2 MB manifest")
    identifier="import-"+uuid.uuid4().hex[:10]
    return import_canonical(identifier,pd.read_csv(io.BytesIO(tb),dtype={"player_id":str}),json.loads(mb))

@app.get("/")
def index():return FileResponse(STATIC/"index.html")

app.mount("/static",StaticFiles(directory=STATIC),name="static")
