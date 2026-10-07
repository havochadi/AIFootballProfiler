from datetime import datetime

import numpy as np
import pytest

from football_profiler import storage as S


def test_identity_position_group_is_saved_and_validated(client, isolated_data):
    """A position group set via /identity is stored on the player and checked against the known groups."""
    import pandas as pd
    identifier = "artificial-identity-case"
    player = {"player_id": "p1", "name": "Artificial player", "eligible_frames": 40,
              "direction_known": True, "identity_verified": True, "global_id": "test:p1"}
    manifest = {"id": identifier, "match_id": identifier, "source": "Artificial test data",
                "source_kind": "imported_tracking", "sampling_hz": 1, "players": [player]}
    S.write_json(S.dataset_dir(identifier, create=True) / "manifest.json", manifest)
    S.save_tracks(identifier, pd.DataFrame({
        "frame": range(40), "time_s": range(40), "player_id": "p1", "period": 1,
        "x": 30.0, "y": 34.0, "detected": 1, "calibration_valid": 1,
    }))
    from football_profiler import features as F
    F.build_profiles(identifier)
    url = f"/api/datasets/{identifier}/players/p1"
    before = client.get(url).json()
    assert before["position_group"] is None
    identity = {"name": before["name"], "team": before["team"], "role": before["role"],
                "position_group": "wide_attacker"}
    assert client.post(url + "/identity", json=identity).status_code == 200
    after = client.get(url).json()
    assert after["position_group"] == "wide_attacker"
    rejected = client.post(url + "/identity", json={**identity, "position_group": "not-a-real-group"})
    assert rejected.status_code == 400 and "position group" in rejected.json()["detail"]


def test_bundled_profile_and_history_api(client, bundled_data):
    listing = client.get("/api/datasets")
    assert listing.status_code == 200
    assert {"skillcorner-2017461", "soccernet-sngs-060"} <= {item["id"] for item in listing.json()}
    dataset = client.get("/api/datasets/skillcorner-2017461").json()
    player = next(item for item in dataset["profiles"] if item["global_id"] == "skillcorner:38673")
    url = f'/api/datasets/skillcorner-2017461/players/{player["player_id"]}'
    response = client.get(url)
    assert response.status_code == 200
    profile = response.json()
    assert profile["valid_position_frames"] == 22557
    assert profile["eligible_frames"] == 58449
    assert profile["position_coverage"] == pytest.approx(22557 / 58449)
    assert np.asarray(profile["heatmap"]).sum() == pytest.approx(1)
    assert "consensus" not in profile
    history = client.get(url + "/history").json()["matches"]
    assert history
    dates = [datetime.fromisoformat(item["date"].replace("Z", "+00:00")) for item in history]
    current = datetime.fromisoformat(dataset["manifest"]["date"].replace("Z", "+00:00"))
    assert dates == sorted(dates)
    assert all(date < current for date in dates)


def test_startup_and_status(client, bundled_data):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    status = client.get("/api/status").json()
    assert status["datasets"] >= 2 and len(status["position_groups"]) == 8
    assert "labels" not in status and "review_rows" not in status


def test_unknown_player_and_cross_origin_write_are_rejected(client, sample_dataset):
    identifier = sample_dataset[0]
    assert client.get(f"/api/datasets/{identifier}/players/missing").status_code == 404
    response = client.post(f"/api/datasets/{identifier}/players/p1/events", headers={
        "Origin": "https://another-site.invalid",
    }, json={"time_s": 1, "kind": "pass", "reviewer": "test"})
    assert response.status_code == 403
    assert S.manual_events(identifier) == []


def test_invalid_import_does_not_create_dataset(client, isolated_data):
    response = client.post("/api/import-tracks", files={
        "tracks": ("tracks.csv", b"frame,time_s,player_id\n0,0,p1\n", "text/csv"),
        "manifest": ("manifest.json", b'{"source":"Artificial test"}', "application/json"),
    })
    assert response.status_code == 400
    assert "CSV requires" in response.json()["detail"]
    assert S.datasets() == []


def test_observed_path_and_track_download(client, sample_dataset):
    identifier, _, _, tracks = sample_dataset
    tracks.loc[1, "detected"] = 0
    tracks.loc[2, "calibration_valid"] = 0
    S.save_tracks(identifier, tracks)
    path = client.get(f"/api/datasets/{identifier}/players/p1/path")
    assert path.status_code == 200
    assert {point["frame"] for point in path.json()} == {0, *range(3, 10)}
    exported = client.get(f"/api/datasets/{identifier}/export")
    assert exported.status_code == 200
    assert exported.content[:2] == b"\x1f\x8b"
