"""Step4 外参图加载与相机外参传播。

该模块的目标是把“相机间外参（Step4 产物）”统一成一套可复用接口，
供 Step5（相机->底盘）在不同求解方式下复用。

约定（与仓库 README 一致）：
- 齐次变换 `A_T_B` 表示 **B -> A**。
- Step4 pose graph 输出 `T_cam_from_ref[cam]` 表示：Cam_i <- Cam_ref，即 `C_cam_T_Cref`。
- Step4 stereo 输出 `R,t` 满足：X_right = R * X_left + t（左->右），即 `Cr_T_Cl`。

本模块只负责：
1) 从 results 目录加载 Step4 外参（multi 或 stereo）。
2) 给定某个相机的 `B_T_C`（Cam -> Base），把它通过 Step4 外参传播到其它相机。

注意：这里的传播只涉及 SE(3) 矩阵运算，不依赖 OpenCV。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ExtrinsicsGraph:
    """相机间外参图（全部表达在同一参考相机下）。"""

    reference: str
    T_cam_from_ref: Dict[str, np.ndarray]
    source: str


def _ensure_transform(T: np.ndarray, name: str) -> None:
    """检查 4x4 齐次变换矩阵格式。"""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: 期望 (4,4)，得到 {T.shape}")

    bottom = T[3, :]
    bottom_target = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if float(np.linalg.norm(bottom - bottom_target)) > 1e-8:
        raise ValueError(f"{name}: 底行应为 [0 0 0 1]，当前 {bottom}")


def _make_transform(R: np.ndarray, t: np.ndarray, name: str) -> np.ndarray:
    """由 R(3x3) 与 t(3,) 构造 4x4 齐次矩阵。"""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    _ensure_transform(T, name)
    return T


def invert_transform(T: np.ndarray, name: str = "inv") -> np.ndarray:
    """对刚体变换求逆（解析法）。"""
    T = np.asarray(T, dtype=np.float64)
    _ensure_transform(T, name + ".in")

    R = T[:3, :3]
    t = T[:3, 3]

    R_inv = R.T
    t_inv = -R_inv @ t
    return _make_transform(R_inv, t_inv, name)


def load_extrinsics_graph(*, results_dir: Path = Path("results")) -> Optional[ExtrinsicsGraph]:
    """加载 Step4 外参（multi 优先，其次 stereo）。

    Args:
        results_dir: 结果目录，默认 "results"。

    Returns:
        若存在 Step4 外参文件则返回 ExtrinsicsGraph，否则返回 None。
    """

    multi_path = results_dir / "multi_camera_extrinsics.json"
    if multi_path.exists():
        data = json.loads(multi_path.read_text(encoding="utf-8"))
        reference = str(data["reference"])

        out: Dict[str, np.ndarray] = {}
        for cam, entry in (data.get("T_cam_from_ref", {}) or {}).items():
            T = np.asarray(entry["T"], dtype=np.float64)
            _ensure_transform(T, f"multi.T_cam_from_ref[{cam}]")
            out[str(cam)] = T

        return ExtrinsicsGraph(reference=reference, T_cam_from_ref=out, source=str(multi_path))

    stereo_path = results_dir / "stereo_extrinsics.json"
    if stereo_path.exists():
        data = json.loads(stereo_path.read_text(encoding="utf-8"))
        R = np.asarray(data["R"], dtype=np.float64)
        t_mm = np.asarray(data["t"], dtype=np.float64).reshape(3)

        # stereoCalibrate 的 t 单位跟 objectPoints 一致：本仓库 Step4 使用 mm。
        t_m = t_mm / 1000.0

        # 约定：Cr_T_Cl（右 <- 左）
        Cr_T_Cl = _make_transform(R, t_m, "Cr_T_Cl")

        reference = "left"
        out = {
            "left": np.eye(4, dtype=np.float64),
            "right": Cr_T_Cl,
        }
        return ExtrinsicsGraph(reference=reference, T_cam_from_ref=out, source=str(stereo_path))

    return None


def propagate_B_T_C(
    *,
    B_T_C_anchor: np.ndarray,
    anchor_cam: str,
    graph: ExtrinsicsGraph,
) -> Dict[str, np.ndarray]:
    """从一个 anchor 相机的 B_T_C，传播得到其它相机的 B_T_C。

    推导（关键方向）：
    - 已知 Step4 给出：C_cam <- C_ref，即 `C_cam_T_Cref`。
    - 已知 anchor 相机的 `B_T_C_anchor`（anchor -> Base）。

    先得到 `B_T_Cref`：
      若 anchor==ref：B_T_Cref = B_T_C_anchor
      否则：p_anchor = C_anchor_T_Cref p_ref
            p_B = B_T_C_anchor p_anchor
          => B_T_Cref = B_T_C_anchor @ C_anchor_T_Cref

    再对任意 cam：
      p_cam = C_cam_T_Cref p_ref
      p_B   = B_T_C_cam p_cam
      又有  p_B = B_T_Cref p_ref
      => B_T_C_cam = B_T_Cref @ inv(C_cam_T_Cref)

    Args:
        B_T_C_anchor: anchor 相机到 Base 的外参（4x4，Cam->Base）。
        anchor_cam: anchor 相机名。
        graph: Step4 外参图。

    Returns:
        {cam: B_T_C_cam}（包含 reference 与 anchor）。
    """

    B_T_C_anchor = np.asarray(B_T_C_anchor, dtype=np.float64)
    _ensure_transform(B_T_C_anchor, "B_T_C_anchor")

    anchor_cam = str(anchor_cam)
    if not anchor_cam:
        raise ValueError("anchor_cam 不能为空")

    if anchor_cam == graph.reference:
        B_T_Cref = B_T_C_anchor
    else:
        if anchor_cam not in graph.T_cam_from_ref:
            raise ValueError(
                f"Step4 外参图缺少 anchor_cam={anchor_cam}。"
                f"已知相机: {sorted(graph.T_cam_from_ref.keys())}"
            )
        C_anchor_T_Cref = graph.T_cam_from_ref[anchor_cam]
        B_T_Cref = B_T_C_anchor @ C_anchor_T_Cref
        _ensure_transform(B_T_Cref, "B_T_Cref")

    out: Dict[str, np.ndarray] = {}
    for cam, C_cam_T_Cref in graph.T_cam_from_ref.items():
        B_T_C_cam = B_T_Cref @ invert_transform(C_cam_T_Cref, f"Cref_T_{cam}")
        _ensure_transform(B_T_C_cam, f"B_T_{cam}")
        out[str(cam)] = B_T_C_cam

    # 如果 graph 里没有显式包含 anchor（极少数情况），也要补上。
    if anchor_cam not in out:
        out[anchor_cam] = B_T_C_anchor

    return out
