from __future__ import annotations
import json
import math
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from . import storage as S
from .features import build_profiles

def prepare_demo(audit_dir):
    audit=Path(audit_dir)
    mid="2017461";identifier="skillcorner-2017461"
    source=json.loads((audit/f"cache/{mid}/{mid}_match.json").read_text())
    players={str(p["id"]):p for p in source["players"] if ((p.get("playing_time") or {}).get("total") or {}).get("minutes_played",0)>0}
    intervals={pid:{int(a["name"].split("_")[-1]):(a["start_frame"],a["end_frame"]) for a in p["playing_time"]["by_period"]} for pid,p in players.items()}
    L,W=source["pitch_length"],source["pitch_width"]
    rows=[]
    for line in (audit/f"tracking_cache/{mid}_tracking_extrapolated.jsonl").open():
        r=json.loads(line);period=r["period"];frame=r["frame"]
        if period not in (1,2):continue
        for point in r["player_data"]:
            pid=str(point["player_id"])
            interval=intervals.get(pid,{}).get(period)
            if not interval or not interval[0]<=frame<interval[1]:continue
            if any(point.get(k) is None or not math.isfinite(point[k]) for k in ("x","y")):continue
            home=players[pid]["team_id"]==source["home_team"]["id"]
            right=source["home_team_side"][period-1]=="left_to_right"
            sign=1 if right==home else -1
            x=(point["x"]*sign/L+.5)*105;y=(point["y"]*sign/W+.5)*68
            flag=1 if point["is_detected"] is True else (0 if point["is_detected"] is False else -1)
            rows.append((frame,frame/10,pid,pid,period,round(x,5),round(y,5),flag,1))
    d=S.dataset_dir(identifier,True)
    meta=[]
    for pid,p in players.items():
        team=source["home_team"] if p["team_id"]==source["home_team"]["id"] else source["away_team"]
        eligible=sum(b-a for a,b in intervals[pid].values())
        meta.append({"player_id":pid,"name":(p["first_name"]+" "+p["last_name"]).strip(),"team":team["name"],"jersey":p["number"],"role":p["player_role"]["position_group"],"global_id":"skillcorner:"+pid,"identity_verified":True,"direction_known":True,"eligible_frames":eligible,"playing_seconds":p["playing_time"]["total"]["minutes_played"]*60,"intervals":intervals[pid],"coverage_note":"10 Hz samples within each provider playing-period interval [start,end). Half-time excluded. Provider detections are not an independent correctness guarantee."})
    manifest={"id":identifier,"title":"Melbourne Victory vs Auckland FC","match_id":mid,"date":source["date_time"],"source":"SkillCorner Open Data","source_kind":"provider_tracking","source_url":"https://github.com/SkillCorner/opendata","source_revision":"4340d274572876239c154c90bc507a9b3250a656","sampling_hz":10,"duration_seconds":7042,"total_sampled_frames":58450,"players":meta,"video":None,"note":"Real provider tracking. Corresponding match footage is not included. Heatmaps use detected positions only.","created":S.now()}
    S.write_json(d/"manifest.json",manifest)
    S.save_tracks(identifier,pd.DataFrame(rows,columns=["frame","time_s","player_id","track_id","period","x","y","detected","calibration_valid"]))
    historical=pd.read_csv(audit/"results/player_match_features.csv",dtype={"player_id":str,"match_id":str})
    events={}
    for _,r in historical[historical.match_id.eq(mid)].iterrows():
        rates={k.removesuffix("_per90"):float(r[k]) for k in historical if k.endswith("_per90")}
        counts={k:int(round(v*float(r.minutes_played)/90)) for k,v in rates.items()}
        events[str(r.player_id)]={"source":"SkillCorner dynamic event records","scope":"Full provider playing interval","minutes_denominator":float(r.minutes_played),"rates_per90":rates,"counts":counts,"note":"Provider event types; passing options are not completed passes and engagements are not necessarily tackles."}
    S.write_json(d/"provider_events.json",events)
    shutil.copy2(audit/"results/player_match_features.csv",S.DATA/"player_history.csv")
    build_profiles(identifier)
    import_soccernet(audit/"access_checks/sample_Labels-GameState.json",audit/"access_checks/sample_000001.jpg")
    evidence=S.EVIDENCE;evidence.mkdir(parents=True,exist_ok=True)
    for name in ["diagnostic_baseline.json","data_summary.json","tracking_coverage.json"]:
        shutil.copy2(audit/"results"/name,evidence/name)
    return {"skillcorner_players":len(meta),"skillcorner_coordinate_records":len(rows)}

def import_soccernet(labels_path,image_path=None):
    source=json.loads(Path(labels_path).read_text())
    info=source["info"];identifier="soccernet-"+info["name"].lower()
    d=S.dataset_dir(identifier,True)
    image_order={str(x["image_id"]):i for i,x in enumerate(source["images"])}
    fps=float(info["frame_rate"]);rows=[];players={}
    for a in source["annotations"]:
        attrs=a.get("attributes",{})
        if attrs.get("role") not in ("player","goalkeeper"):continue
        b=a.get("bbox_pitch") or {};bb=a.get("bbox_image") or {}
        x=b.get("x_bottom_middle");y=b.get("y_bottom_middle")
        if x is None or y is None or not all(math.isfinite(float(v)) for v in (x,y)):continue
        pid=str(a["track_id"]);frame=image_order[str(a["image_id"])]
        rows.append((frame,frame/fps,pid,pid,1,x+52.5,y+34,1,1,bb.get("x",0),bb.get("y",0),bb.get("w",0),bb.get("h",0),1.))
        players[pid]={"player_id":pid,"name":f"{attrs.get('team','Unknown').title()} team · #{attrs.get('jersey','?')} · track {pid}","team":attrs.get("team","unknown"),"jersey":attrs.get("jersey"),"role":attrs.get("role"),"global_id":None,"identity_verified":False,"direction_known":False,"eligible_frames":len(source["images"]),"playing_seconds":len(source["images"])/fps}
    m={"id":identifier,"title":"SoccerNet SNGS-060 reference clip","match_id":"soccernet-game-"+str(info.get("game_id",identifier)),"date":None,"source":"SoccerNet GSR v1.3 annotations","source_kind":"reference_annotations","source_url":"https://github.com/SoccerNet/sn-gamestate","sampling_hz":fps,"duration_seconds":len(source["images"])/fps,"total_sampled_frames":len(source["images"]),"players":list(players.values()),"video":None,"preview":"preview.jpg" if image_path else None,"note":"Reference coordinates, not model predictions. Only the first source image is included. Track IDs are not named player identities; attacking direction is unconfirmed.","coverage_note":"Valid supplied reference positions divided by the 750-frame clip. This does not imply every player was present throughout the clip.","created":S.now()}
    S.write_json(d/"manifest.json",m)
    S.save_tracks(identifier,pd.DataFrame(rows,columns=["frame","time_s","player_id","track_id","period","x","y","detected","calibration_valid","bbox_x","bbox_y","bbox_w","bbox_h","confidence"]))
    if image_path:shutil.copy2(image_path,d/"preview.jpg")
    build_profiles(identifier)
    return m

def import_canonical(identifier,tracks,manifest):
    """Validate a canonical CSV and metadata before creating any dataset files."""
    if not isinstance(manifest,dict):raise ValueError("Manifest must be a JSON object")
    if not isinstance(tracks,pd.DataFrame):raise ValueError("Tracks must be a CSV table")
    required={"frame","time_s","player_id","x","y","detected"}
    if not required.issubset(tracks.columns):raise ValueError("CSV requires: "+", ".join(sorted(required)))
    if not tracks.columns.is_unique:raise ValueError("CSV column names must be unique")
    if tracks.empty:raise ValueError("CSV must contain at least one tracking row")
    if len(tracks)>2_000_000:raise ValueError("Use at most two million rows per import")

    def player_id(value):
        if not isinstance(value,(str,int,float,np.integer,np.floating)) or isinstance(value,bool):
            raise ValueError("Each tracking row and player requires a nonempty player_id")
        if isinstance(value,(float,np.floating)) and not math.isfinite(value):
            raise ValueError("Each tracking row and player requires a nonempty player_id")
        value=str(value).strip()
        if not value:raise ValueError("Each tracking row and player requires a nonempty player_id")
        return value

    def positive_number(value,name):
        try:
            if isinstance(value,bool):raise ValueError()
            number=float(value)
        except (TypeError,ValueError,OverflowError):
            raise ValueError(f"{name} must be a positive finite number") from None
        if not math.isfinite(number) or number<=0:
            raise ValueError(f"{name} must be a positive finite number")
        return number

    def positive_integer(value,name):
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or not 0<value<=2**53-1:
            raise ValueError(f"{name} must be a positive integer up to 2^53 - 1")
        return int(value)

    def display_text(value,default,name):
        if value is None:return default
        if not isinstance(value,(str,int,float)) or isinstance(value,bool):
            raise ValueError(f"{name} must be text")
        if isinstance(value,float) and not math.isfinite(value):raise ValueError(f"{name} must be text")
        return str(value).strip() or default

    tracks=tracks.copy()
    tracks["player_id"]=tracks.player_id.map(player_id)
    for column,default in [("period",1),("calibration_valid",1)]:
        if column not in tracks:tracks[column]=default
    for column in ["frame","time_s","x","y","detected","period","calibration_valid"]:
        try:tracks[column]=pd.to_numeric(tracks[column],errors="raise")
        except (TypeError,ValueError,OverflowError):
            raise ValueError(f"CSV column {column} must contain numeric values") from None
        if column in ("x","y"):
            if np.isinf(tracks[column]).any():raise ValueError(f"{column} cannot contain infinite coordinates")
        elif tracks[column].isna().any() or not np.isfinite(tracks[column]).all():
            raise ValueError(f"{column} must contain finite values")
    if (tracks.frame<0).any() or (tracks.frame%1!=0).any() or (tracks.frame>2**53-1).any():
        raise ValueError("frame must contain nonnegative whole numbers up to 2^53 - 1")
    if (tracks.time_s<0).any():raise ValueError("time_s cannot be negative")
    if (tracks.period<1).any() or (tracks.period%1!=0).any():raise ValueError("period must contain positive whole numbers")
    if not tracks.calibration_valid.isin([0,1]).all():raise ValueError("calibration_valid must be 0 or 1")
    if not tracks.detected.isin([-1,0,1]).all():raise ValueError("detected must be -1, 0 or 1")
    tracks["frame"]=tracks.frame.astype("int64")
    # Check after numeric conversion: CSV frames "1" and "1.0" are the same sample.
    if tracks.duplicated(["frame","player_id"]).any():raise ValueError("Duplicate player/frame rows")
    fps=positive_number(manifest.get("sampling_hz"),"sampling_hz")
    if fps>120:raise ValueError("Set sampling_hz between 0 and 120")
    if "track_id" not in tracks:tracks["track_id"]=tracks.player_id
    manifest={**manifest,"id":identifier,"source_kind":"imported_tracking","created":S.now(),"video":None}
    if not isinstance(manifest.get("source"),str) or not manifest["source"].strip():
        raise ValueError("A source description is required")
    manifest["source"]=manifest["source"].strip()
    manifest["sampling_hz"]=fps
    manifest["title"]=display_text(manifest.get("title"),identifier,"title")
    manifest["match_id"]=display_text(manifest.get("match_id"),identifier,"match_id")
    orientation=manifest.get("coordinate_orientation")
    if "coordinate_orientation" in manifest and orientation not in ("stadium","attack_right"):
        raise ValueError("coordinate_orientation must be stadium or attack_right")
    people=manifest.get("players")
    if not isinstance(people,list) or not people or not all(isinstance(p,dict) for p in people):
        raise ValueError("Manifest players must be a nonempty list of player objects")
    people=[dict(p) for p in people]
    counts=tracks.groupby("player_id").size()
    seen=set()
    for p in people:
        pid=p["player_id"]=player_id(p.get("player_id"))
        if pid in seen:raise ValueError("Manifest contains duplicate player_id entries")
        seen.add(pid)
        p["eligible_frames"]=positive_integer(p.get("eligible_frames"),"Each player's eligible_frames")
        if counts.get(pid,0)>p["eligible_frames"]:raise ValueError("eligible_frames is smaller than supplied observations")
        p["name"]=display_text(p.get("name"),pid,"Player name")
        p["team"]=display_text(p.get("team"),"Unconfirmed","Player team")
        for flag in ("direction_known","identity_verified"):
            if not isinstance(p.get(flag,False),bool):raise ValueError(f"{flag} must be a JSON boolean")
            p.setdefault(flag,False)
        if orientation=="stadium" and p["direction_known"]:
            raise ValueError("direction_known requires coordinates normalised to attack right")
        p["playing_seconds"]=positive_number(p.get("playing_seconds",p["eligible_frames"]/fps),"playing_seconds")
    if seen!=set(tracks.player_id):raise ValueError("Manifest must describe every player_id exactly")
    manifest["players"]=people
    minimum_frames=max(int(tracks.frame.nunique()),max(p["eligible_frames"] for p in people))
    manifest["total_sampled_frames"]=positive_integer(manifest.get("total_sampled_frames",minimum_frames),"total_sampled_frames")
    if manifest["total_sampled_frames"]<minimum_frames:
        raise ValueError("total_sampled_frames cannot be smaller than supplied frames or eligible_frames")
    inferred_duration=max(float(tracks.time_s.max())+1/fps,manifest["total_sampled_frames"]/fps,max(p["playing_seconds"] for p in people))
    manifest["duration_seconds"]=positive_number(manifest.get("duration_seconds",inferred_duration),"duration_seconds")
    if manifest["duration_seconds"]<float(tracks.time_s.max()):
        raise ValueError("duration_seconds cannot be smaller than the last tracking timestamp")
    d=S.dataset_dir(identifier,True);S.write_json(d/"manifest.json",manifest);S.save_tracks(identifier,tracks);build_profiles(identifier)
    return manifest
