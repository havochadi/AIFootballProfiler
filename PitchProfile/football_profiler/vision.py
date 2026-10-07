"""Video helpers shared by the full-match pipeline: ffmpeg lookup, overlay encoding, video metadata and frame reads."""
from __future__ import annotations
import shutil
import subprocess
import cv2


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
