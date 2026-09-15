import numpy as np
from scripts.prepare_data import trace_natural
from scripts.prepare_zind_sample import camera_rotation
from src.panorama import perspective_rays


def test_zind_camera_rotations_are_proper():
    for angle in [-359.46, -325.50, -17.01, 0.43]:
        r = camera_rotation(angle)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-6)
        assert np.linalg.det(r) > 0.999


def test_natural_room_trace_has_finite_rgb_and_depth():
    rays, _ = perspective_rays(24, 90)
    rgb, depth = trace_natural(np.array([-0.3, -0.05, -0.1], np.float32), rays)
    assert rgb.shape == (24, 24, 3)
    assert depth.shape == (24, 24)
    assert np.isfinite(rgb).all() and np.isfinite(depth).all()
    assert ((rgb >= 0) & (rgb <= 1)).all()
    assert (depth > 0).all()
    assert np.std(rgb) > 0.05
