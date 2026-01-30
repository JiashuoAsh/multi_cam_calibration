"""Step4 外参图加载与相机外参传播。

该模块的目标是把“相机间外参（Step4 产物）”统一成一套可复用接口，
供 Step5（相机->底盘）复用。

约定（与仓库 README 一致）：
- 齐次变换 `A_T_B` 表示 **B -> A**。
- Step4 pose graph 输出 `T_cam_from_ref[cam]` 表示：Cam_i <- Cam_ref，即 `C_cam_T_Cref`。
注：本仓库已统一使用 Step4 位姿图（multi_camera_extrinsics.json），不再维护 stereo 专用外参输出。

本模块只负责：
1) 从 results 目录加载 Step4 外参（multi 或 stereo）。
2) 给定某个相机的 `B_T_C`（Cam -> Base），把它通过 Step4 外参传播到其它相机。

注意：这里的传播只涉及 SE(3) 矩阵运算，不依赖 OpenCV。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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


def _transform_basic_payload(
    T: np.ndarray,
    *,
    parent_frame: str,
    child_frame: str,
) -> Dict[str, Any]:
    """将 4x4 齐次变换拆解为 JSON 友好格式。

    说明：
    - 本仓库约定 `A_T_B` 表示 B->A。
    - 因此这里的 `parent_frame/child_frame` 表示：把点从 child 变到 parent。
    - 也等价于：child 坐标系在 parent 中的“位姿表达”。
    """

    T = np.asarray(T, dtype=np.float64)
    _ensure_transform(T, "transform_payload")

    R = T[:3, :3]
    t = T[:3, 3]

    return {
        "parent_frame": str(parent_frame),
        "child_frame": str(child_frame),
        "T": T.tolist(),
        "R": R.tolist(),
        "t": t.tolist(),
    }


def transform_payload(
    T: np.ndarray,
    *,
    parent_frame: str,
    child_frame: str,
    name: str = "T",
    include_inverse: bool = True,
) -> Dict[str, Any]:
    """生成带“求逆/原点坐标”信息的变换 payload（用于写入结果 JSON）。

    Args:
        T: 4x4 齐次变换，表示 child->parent。
        parent_frame: 变换目标坐标系名。
        child_frame: 变换源坐标系名。
        name: 仅用于内部报错/调试标识。

    Args:
        include_inverse: 是否在 payload 中包含 inverse（以及由此得到的 origin_parent_in_child）。
            - True：适合调试时“一份文件里看正反两个方向”。
            - False：适合“位姿/外参分文件输出”场景，避免在一个文件里混入另一个方向。

    Returns:
        JSON 友好的 dict。
        - 总是包含：T/R/t、origin_child_in_parent。
        - 若 include_inverse=True：额外包含 inverse、origin_parent_in_child。
    """

    T = np.asarray(T, dtype=np.float64)
    _ensure_transform(T, name)

    payload = _transform_basic_payload(
        T,
        parent_frame=parent_frame,
        child_frame=child_frame,
    )

    T_inv = invert_transform(T, name=f"inv({name})")
    inv_payload = _transform_basic_payload(
        T_inv,
        parent_frame=child_frame,
        child_frame=parent_frame,
    )

    # 便于人读：child 原点在 parent 里的坐标，等于平移向量 t。
    payload["origin_child_in_parent"] = payload["t"]

    if not include_inverse:
        return payload

    payload["inverse"] = inv_payload
    # 便于人读：parent 原点在 child 里的坐标（来自 inverse.t）。
    payload["origin_parent_in_child"] = inv_payload["t"]
    return payload


def load_extrinsics_graph(*, results_dir: Path = Path("results")) -> Optional[ExtrinsicsGraph]:
    """加载 Step4 外参（多相机位姿图）。

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

            # 说明：历史上 Step4 的平移有两种常见单位来源：
            # - 若 3D 点用米（m），则平移通常在 0~2 量级；
            # - 若 3D 点用毫米（mm），则平移常在几百~几千量级。
            # Step5(b/c) 的输入（W_T_B 等）与本仓库其它模块都以“米”为主，
            # 因此这里做一个保守的启发式归一化：当平移范数明显过大时，按 mm->m 缩放。
            t_norm = float(np.linalg.norm(T[:3, 3]))
            if t_norm > 5.0:
                T = T.copy()
                T[:3, 3] = T[:3, 3] / 1000.0

            out[str(cam)] = T

        return ExtrinsicsGraph(reference=reference, T_cam_from_ref=out, source=str(multi_path))

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
