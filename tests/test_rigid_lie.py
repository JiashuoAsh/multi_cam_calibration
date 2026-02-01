import unittest

import numpy as np

from mcca.core.lie import se3_exp, se3_log
from mcca.core.rigid import invert_T, make_T


class TestRigid(unittest.TestCase):
    def test_invert_T_translation_only(self) -> None:
        T = make_T(np.eye(3), np.array([1.0, 2.0, 3.0]), "T")
        T_inv = invert_T(T, "T_inv")

        np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(T_inv @ T, np.eye(4), atol=1e-12)


class TestLie(unittest.TestCase):
    def test_se3_exp_log_roundtrip(self) -> None:
        # 选择一个非零旋转+平移，覆盖一般情况。
        xi = np.array([0.1, -0.2, 0.3, 1.0, -2.0, 0.5], dtype=np.float64)
        T = se3_exp(xi)
        xi2 = se3_log(T)

        # 数值误差允许很小的容差。
        np.testing.assert_allclose(xi2, xi, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
