from __future__ import annotations
import numpy as np
import pandas as pd
from . import storage as S
from . import taxonomy as T

GRID=(20,32)  # rows across width, columns along length
FEATURE_NAMES=("coverage","mean_x","mean_y","std_x","std_y","defensive_third","middle_third","attacking_third","central_channel","wide_channels","observed_minutes","coordinate_coverage","direction_known","events_available","pass_rate","shot_rate","run_behind_rate","engagement_rate")

def observed_points(rows):
    if rows.empty:return rows.copy()
    ok=rows.detected.eq(1)&np.isfinite(rows.x)&np.isfinite(rows.y)&rows.x.between(0,105)&rows.y.between(0,68)
    if "calibration_valid" in rows:ok &= rows.calibration_valid.eq(1)
    return rows.loc[ok].sort_values("frame").drop_duplicates(["frame","player_id"])

def heatmap(rows):
    q=observed_points(rows)
    h=np.histogram2d(q.y,q.x,bins=GRID,range=((0,68),(0,105)))[0].astype(np.float32)
    if h.sum():h/=h.sum()
    return h

def drop_observations(rows, fraction, rng):
    """Mask a contiguous part of the observed timeline; never invent coordinates."""
    if not 0<=fraction<1:raise ValueError("Gap fraction must be in [0,1)")
    result=rows.copy();q=observed_points(rows)
    n=int(round(len(q)*fraction))
    if n:
        start=int(rng.integers(0,len(q)-n+1))
        result.loc[q.index[start:start+n],"detected"]=0
    return result

def profile(rows, player, manifest, events=None):
    q=observed_points(rows)
    eligible=int(player.get("eligible_frames",manifest.get("total_sampled_frames",0)))
    fps=float(manifest.get("sampling_hz",10))
    finite=np.isfinite(rows.x)&np.isfinite(rows.y) if len(rows) else np.array([],bool)
    detected_count=int(rows.loc[rows.detected.eq(1),"frame"].nunique()) if len(rows) else 0
    coord_count=int(rows.loc[finite,"frame"].nunique()) if len(rows) else 0
    duration=float(player.get("playing_seconds",eligible/fps if fps else 0))
    coverage=min(detected_count/eligible,1) if eligible else None
    poscoverage=min(len(q)/eligible,1) if eligible else None
    direction=bool(player.get("direction_known",False))
    h=heatmap(rows)
    meanx=float(q.x.mean()/105) if len(q) else 0
    meany=float(q.y.mean()/68) if len(q) else 0
    thirds=[float(np.mean((q.x>=a)&(q.x<b))) if len(q) else None for a,b in [(0,35),(35,70),(70,105.000001)]]
    distance=0.;valid_steps=0
    if len(q)>1:
        dt=np.diff(q.time_s.to_numpy());dx=np.diff(q.x.to_numpy());dy=np.diff(q.y.to_numpy())
        d=np.hypot(dx,dy)
        good=(dt>0)&(dt<=max(2.1/fps,.25))&(d/np.maximum(dt,1e-9)<=12)
        if "period" in q:good&=np.diff(q.period.to_numpy())==0
        distance=float(d[good].sum());valid_steps=int(good.sum())
    rates=(events or {}).get("rates_per90",{})
    vector=np.array([poscoverage or 0,meanx,meany,float(q.x.std(ddof=0)/105) if len(q) else 0,float(q.y.std(ddof=0)/68) if len(q) else 0,*[x or 0 for x in thirds],float(q.y.between(68/3,136/3).mean()) if len(q) else 0,float((~q.y.between(68/3,136/3)).mean()) if len(q) else 0,len(q)/fps/60,coord_count/eligible if eligible else 0,float(direction),float(bool(events)),*[np.log1p(max(0,float(rates.get(k,0)))) for k in ("possession_end_pass","possession_end_shot","run_behind","on_ball_engagement")]],dtype=np.float32)
    position_group=player.get("position_group")
    return S.clean_json({"player_id":str(player["player_id"]),"name":player.get("name",str(player["player_id"])),"team":player.get("team","Unconfirmed"),"role":player.get("role","Unconfirmed"),"identity_verified":bool(player.get("identity_verified",False)),"global_id":player.get("global_id"),"direction_known":direction,"eligible_frames":eligible,"detected_frames":detected_count,"valid_position_frames":len(q),"coordinate_frames":coord_count,"detected_coverage":coverage,"position_coverage":poscoverage,"coordinate_coverage":coord_count/eligible if eligible else None,"playing_seconds":duration,"observed_seconds":len(q)/fps,"heatmap":h,"feature_vector":vector,"features_available":bool(len(q)),"zone_shares":thirds,"central_share":float(q.y.between(68/3,136/3).mean()) if len(q) else None,"observed_path_m":distance,"path_valid_steps":valid_steps,"events":events,"source":manifest["source"],"source_kind":manifest.get("source_kind","unknown"),"heatmap_note":"Observed, in-pitch coordinates only. Estimated and invalid points excluded.","coverage_note":player.get("coverage_note",manifest.get("coverage_note","Coverage is measured over the selected clip; confirm presence intervals for full-match interpretation.")),"position_group":position_group,"taxonomy_version":T.VERSION if position_group in T.GROUPS else None})

def build_profiles(dataset_id):
    d=S.dataset_dir(dataset_id);m=S.read_json(d/"manifest.json");t=S.load_tracks(dataset_id)
    events=S.read_json(d/"provider_events.json",{})
    out=[]
    for p in m["players"]:
        row=t[t.player_id.eq(str(p["player_id"]))]
        entry=profile(row,p,m,events.get(str(p["player_id"])))
        out.append(entry)
    S.write_json(d/"profiles.json",out)
    return out

def case_profiles():
    result=[]
    for m in S.datasets():
        for p in S.read_json(S.dataset_dir(m["id"])/"profiles.json",[]):
            result.append({**p,"dataset_id":m["id"],"match_id":m.get("match_id",m["id"]),"date":m.get("date"),"case_id":m["id"]+":"+p["player_id"]})
    return result
