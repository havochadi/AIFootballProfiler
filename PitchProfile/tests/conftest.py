"""All test writes, including reviews and trained models, stay in pytest temp data."""

import importlib
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from football_profiler import features as F
from football_profiler import storage as S
SOURCE_DATA = S.DATA


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(S, "DATA", data)
    return data


@pytest.fixture
def client(isolated_data, monkeypatch):
    from fastapi.testclient import TestClient

    application = importlib.import_module("football_profiler.app")
    monkeypatch.setattr(application, "jobs", {})
    with TestClient(application.app) as instance:
        yield instance


@pytest.fixture
def bundled_data(isolated_data):
    """Copy only read-only inputs needed by profile API smoke tests."""
    # These API tests exercise just these two sources. Avoid copying every
    # World Cup profile into each disposable test directory.
    for identifier in ('skillcorner-2017461','soccernet-sngs-060'):
        manifest=SOURCE_DATA/'datasets'/identifier/'manifest.json'
        destination = isolated_data / "datasets" / manifest.parent.name
        destination.mkdir(parents=True)
        for filename in ("manifest.json", "profiles.json"):
            shutil.copy2(manifest.parent / filename, destination / filename)
    shutil.copy2(SOURCE_DATA / "player_history.csv", isolated_data / "player_history.csv")
    return isolated_data


@pytest.fixture
def sample_dataset(isolated_data):
    identifier = "artificial-test-case"
    player = {
        "player_id": "p1", "name": "Artificial test player", "team": "Test team",
        "role": "Centre Forward", "eligible_frames": 10, "direction_known": True,
        "identity_verified": True, "global_id": "test:p1",
    }
    manifest = {
        "id": identifier, "title": "Artificial test fixture", "match_id": "test-match",
        "date": "2025-05-17T09:35:00Z", "source": "Artificial test data",
        "source_kind": "imported_tracking", "sampling_hz": 10,
        "duration_seconds": 1, "total_sampled_frames": 10, "players": [player],
        "video": None,
    }
    tracks = pd.DataFrame({
        "frame": range(10), "time_s": [i / 10 for i in range(10)],
        "player_id": "p1", "track_id": "p1", "period": 1,
        "x": 30.0, "y": 34.0, "detected": 1, "calibration_valid": 1,
    })
    destination = S.dataset_dir(identifier, create=True)
    S.write_json(destination / "manifest.json", manifest)
    S.save_tracks(identifier, tracks)
    F.build_profiles(identifier)
    return identifier, player, manifest, tracks
