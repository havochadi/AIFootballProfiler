import numpy as np
import pytest

from football_profiler.vision import fit_homography, map_points


def test_homography_maps_control_points_and_independent_center():
    image = [[10, 20], [220, 20], [220, 156], [10, 156]]
    pitch = [[0, 0], [105, 0], [105, 68], [0, 68]]
    transform, quality = fit_homography(image, pitch)
    np.testing.assert_allclose(map_points(image, transform), pitch, atol=1e-5)
    np.testing.assert_allclose(map_points([[115, 88]], transform), [[52.5, 34]], atol=1e-5)
    assert quality["inliers"] == 4
    assert quality["control_point_rmse_m"] < 1e-5


@pytest.mark.parametrize("image,pitch,error", [
    ([[0, 0], [1, 0], [0, 1]], [[0, 0], [105, 0], [0, 68]], "at least four"),
    ([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [105, 0], [105, 68], [0, 68]], "collinear"),
    ([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 0], [106, 0], [105, 68], [0, 68]], "within"),
    ([[0, 0], [1, 0], [1, 1], [0, float("nan")]], [[0, 0], [105, 0], [105, 68], [0, 68]], "finite"),
])
def test_homography_rejects_invalid_geometry(image, pitch, error):
    with pytest.raises(ValueError, match=error):
        fit_homography(image, pitch)
