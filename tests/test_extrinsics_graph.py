import unittest

import numpy as np

from libs.extrinsics_graph import ExtrinsicsGraph, invert_transform, propagate_B_T_C


def _T_from_t(tx: float, ty: float, tz: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


class TestExtrinsicsGraphPropagation(unittest.TestCase):
    def test_propagate_anchor_not_reference(self) -> None:
        # 构造一个简单的外参图：
        # - reference = ref
        # - camA <- ref : 平移 +1m (x)
        # - camB <- ref : 平移 +2m (y)
        graph = ExtrinsicsGraph(
            reference="ref",
            T_cam_from_ref={
                "ref": np.eye(4, dtype=np.float64),
                "camA": _T_from_t(1.0, 0.0, 0.0),
                "camB": _T_from_t(0.0, 2.0, 0.0),
            },
            source="<unit_test>",
        )

        # 已知 anchor 相机 camA 的 B_T_C（camA -> base）
        B_T_camA = _T_from_t(10.0, 0.0, 0.0)

        out = propagate_B_T_C(B_T_C_anchor=B_T_camA, anchor_cam="camA", graph=graph)

        # ref -> base 应该是 B_T_camA @ (camA <- ref)
        B_T_ref_expected = B_T_camA @ graph.T_cam_from_ref["camA"]
        np.testing.assert_allclose(out["ref"], B_T_ref_expected, atol=1e-12)

        # camA 传播结果应保持不变
        np.testing.assert_allclose(out["camA"], B_T_camA, atol=1e-12)

        # camB -> base: B_T_ref @ inv(camB <- ref)
        B_T_camB_expected = B_T_ref_expected @ invert_transform(
            graph.T_cam_from_ref["camB"], "inv_camB"
        )
        np.testing.assert_allclose(out["camB"], B_T_camB_expected, atol=1e-12)

    def test_propagate_anchor_is_reference(self) -> None:
        graph = ExtrinsicsGraph(
            reference="left",
            T_cam_from_ref={
                "left": np.eye(4, dtype=np.float64),
                "right": _T_from_t(0.5, 0.0, 0.0),
            },
            source="<unit_test>",
        )

        B_T_left = _T_from_t(1.0, 2.0, 3.0)
        out = propagate_B_T_C(B_T_C_anchor=B_T_left, anchor_cam="left", graph=graph)

        np.testing.assert_allclose(out["left"], B_T_left, atol=1e-12)

        B_T_right_expected = B_T_left @ invert_transform(graph.T_cam_from_ref["right"], "inv")
        np.testing.assert_allclose(out["right"], B_T_right_expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
