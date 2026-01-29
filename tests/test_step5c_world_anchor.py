import tempfile
import unittest
from pathlib import Path

import numpy as np

from step5c_camera_to_base_from_world import solve_camera_to_base_from_world


class TestStep5cWorldAnchor(unittest.TestCase):
    def test_world_anchor_compute_B_T_C(self) -> None:
        # 构造一个最小 config：只测试 B_T_C = inv(W_T_B) @ W_T_C
        cfg = {
            "camera_to_base_calibration": {
                "mode": "world_anchor",
                "world_anchor": {
                    "reference_camera": "cam0",
                    "world_T_base": {
                        "translation": [5.0, 0.0, 0.0],
                        "rotation_euler_deg": [0.0, 0.0, 0.0],
                        "euler_order": "XYZ",
                    },
                    "world_T_reference_camera": {
                        "translation": [7.0, 0.0, 0.0],
                        "rotation_euler_deg": [0.0, 0.0, 0.0],
                        "euler_order": "XYZ",
                    },
                },
            }
        }

        with tempfile.TemporaryDirectory() as d:
            out = solve_camera_to_base_from_world(config=cfg, results_dir=Path(d))

        B_T_C = out["B_T_C"]
        self.assertIn("cam0", B_T_C)

        T = np.asarray(B_T_C["cam0"], dtype=np.float64)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], np.array([2.0, 0.0, 0.0]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
