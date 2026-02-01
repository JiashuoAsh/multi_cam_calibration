import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mcca.core.extrinsics_graph import (
    ExtrinsicsGraph,
    invert_transform,
    load_extrinsics_graph,
    propagate_B_T_C,
)


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
            reference="cam0",
            T_cam_from_ref={
                "cam0": np.eye(4, dtype=np.float64),
                "cam1": _T_from_t(0.5, 0.0, 0.0),
            },
            source="<unit_test>",
        )

        B_T_cam0 = _T_from_t(1.0, 2.0, 3.0)
        out = propagate_B_T_C(B_T_C_anchor=B_T_cam0, anchor_cam="cam0", graph=graph)

        np.testing.assert_allclose(out["cam0"], B_T_cam0, atol=1e-12)

        B_T_cam1_expected = B_T_cam0 @ invert_transform(graph.T_cam_from_ref["cam1"], "inv")
        np.testing.assert_allclose(out["cam1"], B_T_cam1_expected, atol=1e-12)


class TestExtrinsicsGraphLoad(unittest.TestCase):
    def _write_multi_camera_extrinsics(
        self,
        results_dir: Path,
        *,
        translation_unit: str | None,
        T_cam_from_ref: dict[str, np.ndarray],
    ) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "reference": "ref",
            "T_cam_from_ref": {cam: {"T": T.tolist()} for cam, T in T_cam_from_ref.items()},
        }
        if translation_unit is not None:
            payload["translation_unit"] = translation_unit

        (results_dir / "multi_camera_extrinsics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def test_load_requires_translation_unit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            results_dir = Path(d)
            self._write_multi_camera_extrinsics(
                results_dir,
                translation_unit=None,
                T_cam_from_ref={"ref": np.eye(4, dtype=np.float64)},
            )

            with self.assertRaises(ValueError):
                load_extrinsics_graph(results_dir=results_dir)

    def test_load_translation_unit_m_passthrough(self) -> None:
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.array([1.2, 0.0, 0.0], dtype=np.float64)

        with tempfile.TemporaryDirectory() as d:
            results_dir = Path(d)
            self._write_multi_camera_extrinsics(
                results_dir,
                translation_unit="m",
                T_cam_from_ref={"ref": np.eye(4, dtype=np.float64), "camA": T},
            )

            graph = load_extrinsics_graph(results_dir=results_dir)
            self.assertIsNotNone(graph)
            assert graph is not None

            np.testing.assert_allclose(graph.T_cam_from_ref["camA"][:3, 3], T[:3, 3], atol=1e-12)

    def test_load_translation_unit_mm_scales_to_m(self) -> None:
        # 约定：当文件声明 mm 时，loader 会把平移统一转换为米。
        T_mm = np.eye(4, dtype=np.float64)
        T_mm[:3, 3] = np.array([1200.0, 0.0, 0.0], dtype=np.float64)

        with tempfile.TemporaryDirectory() as d:
            results_dir = Path(d)
            self._write_multi_camera_extrinsics(
                results_dir,
                translation_unit="mm",
                T_cam_from_ref={"ref": np.eye(4, dtype=np.float64), "camA": T_mm},
            )

            graph = load_extrinsics_graph(results_dir=results_dir)
            self.assertIsNotNone(graph)
            assert graph is not None

            np.testing.assert_allclose(
                graph.T_cam_from_ref["camA"][:3, 3],
                np.array([1.2, 0.0, 0.0], dtype=np.float64),
                atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
