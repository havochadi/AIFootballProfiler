import numpy as np
import pandas as pd
import pytest

from football_profiler import features as F
from football_profiler import learning as L
from football_profiler import storage as S


def test_group_splits_are_disjoint_complete_and_reproducible():
    groups = np.repeat(["match-a", "match-b", "match-c", "match-d", "match-e"], 6)
    splits = L.split_groups(groups, seed=42)
    assert sorted(np.concatenate(splits).tolist()) == list(range(len(groups)))
    group_sets = [set(groups[indices]) for indices in splits]
    assert all(group_sets)
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    for actual, repeated in zip(splits, L.split_groups(groups, seed=42)):
        np.testing.assert_array_equal(actual, repeated)


def test_group_split_requires_three_independent_groups():
    with pytest.raises(ValueError, match="three independent groups"):
        L.split_groups(["match-a"] * 10 + ["match-b"] * 10)


def test_unknown_targets_have_zero_gradient():
    import torch
    from football_profiler.runtime import torch_device

    device = torch_device()
    logits = torch.zeros((2, 3), requires_grad=True, device=device)
    labels = torch.tensor([[1., float("nan"), 0.], [float("nan"), float("nan"), 1.]], device=device)
    loss = L.masked_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~torch.isfinite(labels)]).item() == 0
    assert torch.count_nonzero(logits.grad[torch.isfinite(labels)]).item() == 3
    blank_logits = torch.zeros((2, 3), requires_grad=True, device=device)
    blank_loss = L.masked_loss(blank_logits, torch.full((2, 3), float("nan"), device=device))
    blank_loss.backward()
    assert blank_loss.item() == 0
    assert torch.count_nonzero(blank_logits.grad).item() == 0


def test_training_function_rejects_unreviewed_cases(sample_dataset, isolated_data):
    with pytest.raises(ValueError, match="Only 0 usable reviewed cases"):
        L.train_models(epochs=1)
    assert not (isolated_data / "models").exists()


def test_training_quality_gate_requires_coverage_duration_direction_and_independent_reviews(monkeypatch):
    cases = []
    scenarios = {
        "exact-boundary": {},
        "low-coverage": {"position_coverage": .1999},
        "short-observation": {"observed_seconds": 29.99},
        "unknown-direction": {"direction_known": False},
        "missing-features": {"features_available": False},
        "unknown-coverage": {"position_coverage": None},
        "single-review": {},
        "unknown-labels": {},
    }
    for pid, changes in scenarios.items():
        cases.append({"dataset_id": "artificial-quality-cases", "player_id": pid,
                      "features_available": True, "direction_known": True,
                      "position_coverage": .2, "observed_seconds": 30, **changes})
        labels = dict.fromkeys(S.LABELS, None)
        if pid != "unknown-labels":
            labels["target_forward"] = 1
        for reviewer in (["reviewer-a"] if pid == "single-review" else ["reviewer-a", "reviewer-b"]):
            S.save_review("artificial-quality-cases", pid, reviewer, labels, "Artificial test evidence")
    monkeypatch.setattr(L, "case_profiles", lambda: cases)
    eligible = L.labelled_cases()
    assert [case["player_id"] for case in eligible] == ["exact-boundary"]
    assert eligible[0]["y"][0] == 1
    assert np.isnan(eligible[0]["y"][1:]).all()


def test_artificial_training_roundtrip_keeps_unknown_heads_unavailable(isolated_data, monkeypatch):
    """Disposable artificial labels verify execution, never real model performance."""
    import torch
    from football_profiler.runtime import torch_device

    selected_device = torch_device()
    observed_devices = []
    build_network = L.build_network

    def instrumented_network():
        net = build_network()

        def record_devices(module, inputs):
            observed_devices.append((next(module.parameters()).device, *(value.device for value in inputs)))

        net.register_forward_pre_hook(record_devices)
        return net

    monkeypatch.setattr(L, "build_network", instrumented_network)
    for group in range(3):
        identifier = f"artificial-training-{group}"
        players, rows = [], []
        for index in range(6):
            pid = f"p{group}-{index}"
            players.append({"player_id": pid, "name": "Artificial training fixture",
                            "eligible_frames": 400, "direction_known": True,
                            "identity_verified": True, "global_id": f"test:{pid}"})
            label = index % 2
            for frame in range(400):
                rows.append((frame, frame / 10, pid, 1, 20 + label * 60 + frame % 5,
                             30 + index, 1, 1))
            for reviewer in ("artificial-reviewer-a", "artificial-reviewer-b"):
                S.save_review(identifier, pid, reviewer, dict(zip(S.LABELS, [label, None, 1 - label])),
                              "Artificial test data only; not a review of any real player")
        manifest = {"id": identifier, "match_id": identifier, "source": "Artificial test data",
                    "source_kind": "imported_tracking", "sampling_hz": 10, "players": players}
        S.write_json(S.dataset_dir(identifier, create=True) / "manifest.json", manifest)
        S.save_tracks(identifier, pd.DataFrame(rows, columns=[
            "frame", "time_s", "player_id", "period", "x", "y", "detected", "calibration_valid",
        ]))
        F.build_profiles(identifier)
    report = L.train_models(epochs=1, seed=42)
    assert report["cases"] == 18
    assert report["active_labels"] == [True, False, True]
    assert report["device"] == str(selected_device)
    groups = [{item["group"] for item in entries} for entries in report["splits"].values()]
    assert len(set.union(*groups)) == 3
    assert sum(map(len, groups)) == 3
    destination = isolated_data / "models" / report["run_id"]
    player = F.case_profiles()[0]
    for model in ("logistic", "cnn", "cnn_with_gaps"):
        # Exercise loading each saved model, independently of validation's selection.
        report["selected_model"] = model
        S.write_json(destination / "report.json", report)
        prediction = L.predict(player)
        assert prediction["status"] == "model_suggestion"
        assert prediction["model"] == model
        known, unknown, other = prediction["labels"]
        assert 0 <= known["percentage"] <= 100
        assert 0 <= other["percentage"] <= 100
        assert unknown["percentage"] is None
        assert unknown["suggested"] is None
        assert prediction == L.predict(player)
        if model != "logistic":
            state = torch.load(destination / (model + ".pt"), weights_only=True)
            assert all(tensor.device.type == "cpu" for tensor in state.values())
    assert observed_devices
    assert all(all(device == selected_device for device in devices) for devices in observed_devices)
