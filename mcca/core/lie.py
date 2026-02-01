"""Lie 群/代数的最小工具集（SO(3)/SE(3)）。

注意：
- 仅提供 Step4/优化中常用的 exp/log 映射。
- 不包含文件 IO/图像处理。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from mcca.core.rigid import make_T, ensure_T


def skew(w: np.ndarray) -> np.ndarray:
    """将 3D 向量转为反对称矩阵（hat 操作）。"""

    w = np.asarray(w, dtype=np.float64).reshape(3)
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=np.float64)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """so(3) -> SO(3)（rotvec 指数映射）。"""

    return Rotation.from_rotvec(np.asarray(w, dtype=np.float64).reshape(3)).as_matrix()


def so3_log(Rm: np.ndarray) -> np.ndarray:
    """SO(3) -> so(3)（rotvec 对数映射）。"""

    return Rotation.from_matrix(np.asarray(Rm, dtype=np.float64)).as_rotvec()


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """se(3) -> SE(3)。

    xi = [w(3), v(3)]
    """

    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    w = xi[:3]
    v = xi[3:]

    theta = float(np.linalg.norm(w))
    Rm = so3_exp(w)

    W = skew(w)
    I = np.eye(3, dtype=np.float64)

    if theta < 1e-8:
        V = I + 0.5 * W
    else:
        A = float(np.sin(theta) / theta)
        B = float((1.0 - np.cos(theta)) / (theta * theta))
        C = float((theta - np.sin(theta)) / (theta**3))
        V = I + B * W + C * (W @ W)

    t = V @ v
    return make_T(Rm, t, "se3_exp")


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) -> se(3) (6D)。

    返回 xi = [w(3), v(3)]
    """

    T = np.asarray(T, dtype=np.float64)
    ensure_T(T, "se3_log")

    Rm = T[:3, :3]
    t = T[:3, 3]

    w = so3_log(Rm)
    theta = float(np.linalg.norm(w))
    W = skew(w)
    I = np.eye(3, dtype=np.float64)

    if theta < 1e-8:
        V_inv = I - 0.5 * W
    else:
        A = float(np.sin(theta) / theta)
        B = float((1.0 - np.cos(theta)) / (theta * theta))
        # 经典公式：V^{-1} = I - 0.5 W + (1/theta^2) * (1 - A/(2B)) * W^2
        coef = float((1.0 / (theta * theta)) * (1.0 - A / (2.0 * B)))
        V_inv = I - 0.5 * W + coef * (W @ W)

    v = V_inv @ t
    xi = np.zeros(6, dtype=np.float64)
    xi[:3] = w
    xi[3:] = v
    return xi
