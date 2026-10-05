import csv
import io
from datetime import datetime

import numpy as np
import pytest

from football_profiler import storage as S
from football_profiler import taxonomy as T


def test_identity_position_group_enables_v2_prediction_without_a_case(client, isolated_data):
    """Setting a position group via /identity (no interval case involved) must let a
    whole-player profile reach the v2 predictor instead of being permanently blocked by
    the 20-minute pilot gate, which only applies to interval-case profiles."""
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
    assert before["position_group"] is None and before["taxonomy_version"] is None
    assert before["prediction"]["status"] == "position_unconfirmed"
    identity = {"name": before["name"], "team": before["team"], "role": before["role"],
                "position_group": "wide_attacker"}
    assert client.post(url + "/identity", json=identity).status_code == 200
    after = client.get(url).json()
    assert after["position_group"] == "wide_attacker"
    assert after["taxonomy_version"] == T.VERSION
    # No v2 model exists yet in this isolated test environment; it must fail on that,
    # not resurface the interval-only "reviewed interval of at least 20 minutes" message.
    assert after["prediction"]["status"] == "not_trained"
    rejected = client.post(url + "/identity", json={**identity, "position_group": "not-a-real-group"})
    assert rejected.status_code == 400 and "archetype catalogue" in rejected.json()["detail"]


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
    assert profile["prediction"]["status"] == "position_unconfirmed"
    assert "consensus" not in profile
    history = client.get(url + "/history").json()["matches"]
    assert history
    dates = [datetime.fromisoformat(item["date"].replace("Z", "+00:00")) for item in history]
    current = datetime.fromisoformat(dataset["manifest"]["date"].replace("Z", "+00:00"))
    assert dates == sorted(dates)
    assert all(date < current for date in dates)


def test_startup_and_training_gate_do_not_create_a_model(client, bundled_data, isolated_data):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    status = client.get("/api/status").json()
    assert status["review_rows"] == 0
    assert status["interval_cases"] == 0
    assert status["taxonomy_version"] == T.VERSION
    assert len(status["labels"]) == 42
    assert len(status["position_groups"]) == 8
    assert status["archetype_model"] is None
    summary = client.get("/api/annotations/summary").json()
    assert summary["usable_training_cases"] == 0
    response = client.post("/api/train", json={"epochs": 1})
    assert response.status_code == 400
    assert "independently reviewed" in response.json()["detail"]
    assert not (isolated_data / "models").exists()
    assert not (isolated_data / "jobs").exists()
    experiments = client.get("/api/experiments").json()
    assert experiments["archetype"] is None
    assert experiments["reconstruction"] is None


def test_independent_reviews_adjudication_invalidation_and_csv_export(client, sample_dataset):
    identifier = sample_dataset[0]
    url = f"/api/datasets/{identifier}/players/p1"
    labels = dict(zip(S.LABELS, [1, None, 0]))
    first = {"reviewer": " Alice ", "labels": labels,
             "evidence": "Artificial test evidence at 00:01", "notes": "=TEST()"}
    response = client.post(url + "/review", json=first)
    assert response.status_code == 200
    assert response.json()["labels"]["target_forward"]["value"] is None
    # The same name with different casing must not supply a second independent vote.
    first["reviewer"] = "ALICE"
    assert client.post(url + "/review", json=first).json()["reviewers"] == 1
    hidden = client.get(url + "/reviews", params={"reviewer": "bob"}).json()
    assert hidden == {"own_review": None, "reviewer_count": 1}
    own = client.get(url + "/reviews", params={"reviewer": " ALICE "}).json()
    assert own["own_review"]["reviewer"] == "alice"
    second = {**first, "reviewer": "bob", "labels": {**labels, "target_forward": 0}}
    consensus = client.post(url + "/review", json=second).json()
    assert consensus["reviewers"] == 2
    assert consensus["labels"]["target_forward"]["status"] == "disagreement"
    assert consensus["labels"]["link_forward"]["status"] == "agreed"
    assert consensus["labels"]["runner_behind"]["value"] is None
    adjudication = {"label": "target_forward", "value": 1, "reviewer": " ALICE ",
                    "reason": "Artificial independent test evidence"}
    assert client.post(url + "/adjudicate", json=adjudication).status_code == 400
    adjudication["reviewer"] = "carol"
    result = client.post(url + "/adjudicate", json=adjudication)
    assert result.status_code == 200
    assert result.json()["labels"]["target_forward"]["status"] == "adjudicated"
    result = client.post(url + "/review", json=first).json()
    assert result["labels"]["target_forward"]["status"] == "disagreement"
    with S.db() as connection:
        assert connection.execute("SELECT count(*) FROM review_history").fetchone()[0] == 4
    exported = client.get("/api/annotations/export")
    assert exported.status_code == 200
    rows = list(csv.DictReader(io.StringIO(exported.text)))
    assert {row["reviewer"] for row in rows} == {"alice", "bob"}
    assert all(row["runner_behind"] == "unknown" for row in rows)
    assert all(row["notes"] == "'=TEST()" for row in rows)


@pytest.mark.parametrize("invalid", [True, "1", 2, -1])
def test_review_api_rejects_non_integer_or_out_of_range_labels(client, sample_dataset, invalid):
    identifier = sample_dataset[0]
    response = client.post(f"/api/datasets/{identifier}/players/p1/review", json={
        "reviewer": "test-reviewer", "labels": dict(zip(S.LABELS, [invalid, None, 0])),
        "evidence": "Artificial test evidence",
    })
    assert response.status_code in {400, 422}
    assert S.reviews() == []


def test_unknown_player_and_cross_origin_write_are_rejected(client, sample_dataset):
    identifier = sample_dataset[0]
    assert client.get(f"/api/datasets/{identifier}/players/missing").status_code == 404
    response = client.post(f"/api/datasets/{identifier}/players/p1/review", headers={
        "Origin": "https://another-site.invalid",
    }, json={"reviewer": "test", "labels": dict.fromkeys(S.LABELS), "evidence": "test evidence"})
    assert response.status_code == 403
    assert S.reviews() == []


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
