from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PinholeIntrinsics:
    """针孔相机内参（用于几何 FOV 计算）。

    说明：
    - 该数据结构只包含几何计算所需字段。
    - 不包含畸变参数，因为“理想针孔 FOV”与畸变无关。
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class FovResult:
    """FOV 结果（单位：度）。"""

    hfov_deg: float
    vfov_deg: float
    dfov_deg: float


def _extract_k_values(data: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """从 JSON dict 中提取 fx/fy/cx/cy。

    支持来源：
    - 顶层键：fx/fy/cx/cy
    - camera_matrix（3x3）：从中读取

    Raises:
        KeyError: 缺少必要字段。
        ValueError: 字段类型不合法。
    """

    if all(k in data for k in ("fx", "fy", "cx", "cy")):
        return float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"])

    cm = data.get("camera_matrix")
    if isinstance(cm, list) and len(cm) == 3 and all(isinstance(r, list) and len(r) == 3 for r in cm):
        fx = float(cm[0][0])
        fy = float(cm[1][1])
        cx = float(cm[0][2])
        cy = float(cm[1][2])
        return fx, fy, cx, cy

    raise KeyError("缺少 fx/fy/cx/cy 或 camera_matrix")


def _extract_image_size(data: Mapping[str, Any]) -> tuple[int, int]:
    """从 JSON dict 中提取 (width,height)。

    Raises:
        KeyError: 缺少 image_size。
        ValueError: image_size 不合法。
    """

    size = data.get("image_size")
    if isinstance(size, list) and len(size) == 2:
        w = int(size[0])
        h = int(size[1])
        if w <= 0 or h <= 0:
            raise ValueError("image_size 必须为正数")
        return w, h

    raise KeyError("缺少 image_size（需要 [width, height]）")


def pinhole_from_step3_intrinsics_dict(
    data: Mapping[str, Any],
    *,
    override_image_size: tuple[int, int] | None = None,
) -> PinholeIntrinsics:
    """从 Step3 产出的内参 JSON dict 中提取针孔模型参数。

    说明：
    - 本函数不做文件 IO，仅解释数据结构。

    Args:
        data: 内参 JSON 的 dict。
        override_image_size: 可选覆盖 (width,height)。

    Returns:
        PinholeIntrinsics

    Raises:
        KeyError/ValueError: 当字段缺失或不合法时。
    """

    fx, fy, cx, cy = _extract_k_values(data)

    if override_image_size is not None:
        width, height = int(override_image_size[0]), int(override_image_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("override_image_size 必须为正数")
    else:
        width, height = _extract_image_size(data)

    if fx <= 0 or fy <= 0:
        raise ValueError(f"fx/fy 必须为正数，但读到 fx={fx}, fy={fy}")

    return PinholeIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        width=int(width),
        height=int(height),
    )


def compute_fov_deg(intri: PinholeIntrinsics) -> FovResult:
    """基于针孔模型计算 HFOV/VFOV/DFOV（单位：度）。

    说明：
    - 主点不在中心时，左右/上下视场角不对称；这里使用加和形式：
      HFOV = atan(cx/fx) + atan((w-cx)/fx)
      VFOV = atan(cy/fy) + atan((h-cy)/fy)
    - 对角 DFOV：取四个角中离主点“归一化距离”最远的角作为半对角，再乘 2。

    Args:
        intri: 针孔内参。

    Returns:
        FovResult（度）。
    """

    w = float(intri.width)
    h = float(intri.height)
    fx = float(intri.fx)
    fy = float(intri.fy)
    cx = float(intri.cx)
    cy = float(intri.cy)

    left = math.atan(cx / fx)
    right = math.atan((w - cx) / fx)
    up = math.atan(cy / fy)
    down = math.atan((h - cy) / fy)

    hfov = left + right
    vfov = up + down

    corners = ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h))
    max_r = 0.0
    for u, v in corners:
        x = (u - cx) / fx
        y = (v - cy) / fy
        r = math.sqrt(x * x + y * y)
        if r > max_r:
            max_r = r
    dfov = 2.0 * math.atan(max_r)

    return FovResult(
        hfov_deg=math.degrees(hfov),
        vfov_deg=math.degrees(vfov),
        dfov_deg=math.degrees(dfov),
    )
