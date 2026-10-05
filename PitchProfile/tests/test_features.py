import numpy as np
import pandas as pd
import pytest

from football_profiler import features as F


def test_profiles_exclude_estimated_invalid_and_duplicate_positions():
    rows = pd.DataFrame({
        "frame": [0, 0, 1, 2, 3, 4, 5], "time_s": [0, 0, .1, .2, .3, .4, .5],
        "player_id": "test-player", "period": 1,
        "x": [10, 10, 100, 106, 30, np.nan, 80],
        "y": [34, 34, 34, 34, 34, 34, 34],
        "detected": [1, 1, 0, 1, 1, 1, 1],
        "calibration_valid": [1, 1, 1, 1, 0, 1, 1],
    })
    result = F.profile(rows, {"player_id": "test-player", "eligible_frames": 6}, {
        "sampling_hz": 10, "source": "Artificial test observations",
    })
    assert F.observed_points(rows).frame.tolist() == [0, 5]
    assert result["valid_position_frames"] == 2
    assert result["detected_frames"] == 5
    assert result["coordinate_frames"] == 5
    assert result["position_coverage"] == pytest.approx(2 / 6)
    assert result["coordinate_coverage"] == pytest.approx(5 / 6)
    assert result["zone_shares"] == [.5, 0, .5]
    assert result["observed_path_m"] == 0  # A gap is not counted as travelled distance.
    assert np.asarray(result["heatmap"]).sum() == pytest.approx(1)


def test_gap_masking_removes_only_contiguous_observations_without_mutating_coordinates():
    original = pd.DataFrame({
        "frame": range(20), "time_s": np.arange(20) / 10, "player_id": "p1",
        "x": np.arange(20), "y": 34, "detected": 1, "calibration_valid": 1,
    })
    original.loc[3, "detected"] = 0
    original.loc[8, "calibration_valid"] = 0
    before = original.copy(deep=True)
    result = F.drop_observations(original, .5, np.random.default_rng(42))
    observed = F.observed_points(original)
    removed = observed.index[result.loc[observed.index, "detected"].eq(0)]
    positions = np.flatnonzero(observed.index.isin(removed))
    assert len(removed) == 9
    assert np.all(np.diff(positions) == 1)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(result.drop(columns="detected"), original.drop(columns="detected"))
    assert result.loc[3, "detected"] == 0
    assert result.loc[8, "calibration_valid"] == 0


@pytest.mark.parametrize("fraction", [-.01, 1, 1.1])
def test_gap_masking_rejects_invalid_fraction(sample_dataset, fraction):
    with pytest.raises(ValueError, match="Gap fraction"):
        F.drop_observations(sample_dataset[3], fraction, np.random.default_rng(1))


def test_empty_observations_remain_unavailable(sample_dataset):
    _, player, manifest, rows = sample_dataset
    rows["detected"] = 0
    result = F.profile(rows, player, manifest)
    assert result["features_available"] is False
    assert result["observed_seconds"] == 0
    assert result["zone_shares"] == [None, None, None]
    assert np.asarray(result["heatmap"]).sum() == 0
    assert np.isfinite(result["feature_vector"]).all()


def test_path_distance_does_not_bridge_periods(sample_dataset):
    _, player, manifest, rows = sample_dataset
    rows = rows.iloc[:3].copy()
    rows["x"] = [30, 30.5, 31]
    rows["period"] = [1, 1, 2]
    result = F.profile(rows, player, manifest)
    assert result["path_valid_steps"] == 1
    assert result["observed_path_m"] == pytest.approx(.5)
