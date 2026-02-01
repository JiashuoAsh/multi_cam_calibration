"""刚体变换的基础工具（SE(3) 的矩阵层操作）。

注意：
- 这里只提供对 4x4 齐次矩阵的格式检查、构造与解析求逆。
- 不包含 OpenCV / AprilTag / 文件 IO。
"""

from __future__ import annotations

import numpy as np


def ensure_T(T: np.ndarray, name: str) -> None:
    """检查 4x4 齐次变换矩阵格式。"""

    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: 期望 (4,4)，得到 {T.shape}")

    bottom = T[3, :]
    bottom_target = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if float(np.linalg.norm(bottom - bottom_target)) > 1e-8:
        raise ValueError(f"{name}: 底行应为 [0 0 0 1]，当前 {bottom}")


def make_T(Rm: np.ndarray, t: np.ndarray, name: str) -> np.ndarray:
    """由 R(3x3) 与 t(3,) 构造 4x4 齐次矩阵。"""

    Rm = np.asarray(Rm, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t
    ensure_T(T, name)
    return T


def invert_T(T: np.ndarray, name: str = "inv") -> np.ndarray:
    """对刚体变换求逆（解析法）。"""

    T = np.asarray(T, dtype=np.float64)
    ensure_T(T, name + ".in")

    Rm = T[:3, :3]
    t = T[:3, 3]

    Rm_inv = Rm.T
    t_inv = -Rm_inv @ t
    return make_T(Rm_inv, t_inv, name)
