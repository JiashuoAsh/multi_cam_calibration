import unittest

import numpy as np

import mcca.core.detection as detection


class TestDetectionMultiscaleWiring(unittest.TestCase):
    def test_use_multiscale_calls_module_level_function(self) -> None:
        """验证 use_multiscale 分支可被 monkeypatch。

        目的：
        - 证明 core.detection 对多尺度实现的依赖是显式的（模块级符号），
          而不是藏在函数体里的动态 import。
        - 这样单测/上层代码可以替换实现（例如注入缓存/统计），提升可测试性。
        """

        calls: list[dict[str, object]] = []

        def _stub_detect_apriltag_multiscale(
            image: np.ndarray,
            aruco_dict: object,
            detector_params: object,
            verbose: bool = False,
            opencv_refine: bool = False,
            board: object = None,
            camera_matrix: object = None,
            dist_coeffs: object = None,
        ):
            calls.append(
                {
                    "shape": tuple(int(x) for x in image.shape),
                    "verbose": bool(verbose),
                    "opencv_refine": bool(opencv_refine),
                }
            )

            corners = [np.zeros((1, 4, 2), dtype=np.float32)]
            ids = np.array([[0]], dtype=np.int32)
            return corners, ids

        old = detection.detect_apriltag_multiscale
        detection.detect_apriltag_multiscale = _stub_detect_apriltag_multiscale
        try:
            img = np.zeros((60, 80, 3), dtype=np.uint8)
            corners, ids = detection.detect_apriltag_corners(
                img,
                aruco_dict=object(),
                detector_params=None,
                use_multiscale=True,
                opencv_refine=False,
            )
        finally:
            detection.detect_apriltag_multiscale = old

        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(corners)
        self.assertIsNotNone(ids)
        self.assertEqual(tuple(ids.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
