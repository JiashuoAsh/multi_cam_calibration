import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from mcca.core.step5_camera_to_base import build_B_T_T_from_config, mean_C_T_T_from_pnp


class TestStep5CameraToBaseCore(unittest.TestCase):
    def test_build_B_T_T_tag0_center_passthrough(self) -> None:
        board_cfg = {
            "tags_x": 4,
            "tags_y": 4,
            "tag_size": 55.0,
            "tag_spacing": 16.5,
        }
        transform_cfg = {
            "translation": [1.0, 2.0, 3.0],
            "rotation_euler_deg": [0.0, 0.0, 0.0],
            "translation_reference": "tag0_center",
        }

        B_T_T = build_B_T_T_from_config(transform_cfg=transform_cfg, board_cfg=board_cfg)

        np.testing.assert_allclose(B_T_T[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(B_T_T[:3, 3], np.array([1.0, 2.0, 3.0]), atol=1e-12)
        np.testing.assert_allclose(B_T_T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-12)

    def test_build_B_T_T_reference_point_conversion(self) -> None:
        board_cfg = {
            "tags_x": 4,
            "tags_y": 4,
            "tag_size": 55.0,
            "tag_spacing": 16.5,
        }
        transform_cfg = {
            "translation": [1.0, 0.0, 0.0],
            "rotation_euler_deg": [0.0, 0.0, 90.0],
            "translation_reference": "grid_center",
            "translation_reference_point_in_T_m": [0.1, 0.0, 0.0],
        }

        B_T_T = build_B_T_T_from_config(transform_cfg=transform_cfg, board_cfg=board_cfg)

        R_B_T = Rotation.from_euler("XYZ", [0.0, 0.0, 90.0], degrees=True).as_matrix()
        expected_t = np.array([1.0, 0.0, 0.0]) - (R_B_T @ np.array([0.1, 0.0, 0.0]))

        np.testing.assert_allclose(B_T_T[:3, :3], R_B_T, atol=1e-12)
        np.testing.assert_allclose(B_T_T[:3, 3], expected_t, atol=1e-12)
        np.testing.assert_allclose(B_T_T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-12)

    def test_mean_C_T_T_from_pnp_identity(self) -> None:
        rvecs = [np.zeros((3, 1)), np.zeros((3, 1))]
        tvecs = [np.array([[1.0], [2.0], [3.0]]), np.array([[1.0], [2.0], [5.0]])]

        C_T_T, stats = mean_C_T_T_from_pnp(rvecs=rvecs, tvecs=tvecs)

        np.testing.assert_allclose(C_T_T[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(C_T_T[:3, 3], np.array([1.0, 2.0, 4.0]), atol=1e-12)
        self.assertGreaterEqual(stats.t_std_norm_m, 0.0)
        self.assertGreaterEqual(stats.r_std_norm_rad, 0.0)


if __name__ == "__main__":
    unittest.main()
