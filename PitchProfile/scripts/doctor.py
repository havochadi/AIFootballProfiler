"""Local readiness check: python scripts/doctor.py."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
location = ROOT / ".data-location"
default_data = Path(location.read_text(encoding="utf-8").strip()) if location.is_file() else ROOT / "data"
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(os.environ.get("PITCHPROFILE_DATA") or default_data) / "ultralytics_config"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main():
    failures = []
    print(f"Python {sys.version.split()[0]} | {sys.executable}")
    if not (3, 12) <= sys.version_info[:2] <= (3, 13):
        failures.append("Use Python 3.12 or 3.13 with these pinned dependencies.")
    for module in ("fastapi", "uvicorn", "multipart", "numpy", "pandas", "scipy",
                   "sklearn", "cv2", "torch", "torchvision", "ultralytics", "lap", "imageio_ffmpeg"):
        try:
            loaded = importlib.import_module(module)
            print(f"OK {module}: {getattr(loaded, '__version__', 'installed')}")
        except Exception as exc:
            failures.append(f"{module}: {exc}")
    for name in ("static/index.html", "static/app.js", "models/yolo11n.pt"):
        if not (ROOT / name).is_file():
            failures.append(f"Missing {name}")
    try:
        from football_profiler import storage, vision, runtime
        device = runtime.torch_device()
        import torch
        probe = torch.ones((16, 16), device=device)
        assert (probe @ probe).sum().item() == 4096
        print(f"Compute: {runtime.runtime_info()}")
        manifests = storage.datasets()
        print(f"Data directory: {storage.DATA}")
        print(f"Prepared sources: {len(manifests)}")
        for manifest in manifests:
            print(f"  {manifest['id']}: {len(manifest['players'])} players/tracks")
            for name in ("tracks.csv.gz", "profiles.json"):
                if not (storage.dataset_dir(manifest['id']) / name).is_file():
                    failures.append(f"{manifest['id']} is missing {name}")
        if not manifests:
            failures.append("No prepared datasets found. Restore the bundled data folder.")
        from football_profiler import football_models, jersey
        models = football_models.availability()
        print(f"Full-match analysis models: {models} | jersey reader: {'available' if jersey.available() else 'unavailable'}")
        if not all(models.values()) or not jersey.available():
            print("  (optional) Run scripts/fetch_football_models.py to enable full-match video analysis.")
        from football_profiler import action_spotting
        spotter = action_spotting.available()
        print(f"Action spotter (shots, tackles, blocks, headers): {'available' if spotter else 'unavailable'}")
        if not spotter:
            print("  (optional) Run scripts/fetch_football_models.py; without it a match has no shots, tackles, blocks or headers.")
        ffmpeg = vision.ffmpeg_executable()
        print(f"FFmpeg: {ffmpeg or 'unavailable'}")
        if not ffmpeg:
            failures.append("Install imageio-ffmpeg or put FFmpeg on PATH for browser overlays.")
    except Exception as exc:
        failures.append(f"Application data/runtime: {exc}")
    if failures:
        for message in failures:
            print(f"FAIL {message}")
        print("Install requirements.txt in the project virtual environment and retry.")
        return 1
    print("Ready. Launch with python run.py and open http://127.0.0.1:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
