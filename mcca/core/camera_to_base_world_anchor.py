"""Step5c 核心逻辑：基于世界位姿锚点求解相机->底盘外参。

分层说明：
- 本模块属于 core：只包含数学/配置解析与求解，不做 CLI/文件 IO。
- CLI/落盘输出由 entry 层负责（见 `mcca.entry.step5c_world_anchor`）。

约定：
- 变换命名采用仓库统一的 A_T_B 表示 B->A。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from mcca.core.extrinsics_graph import (
    ExtrinsicsGraph,
    invert_transform,
    load_extrinsics_graph,
    propagate_B_T_C,
)
from mcca.core.rigid import ensure_T, make_T


def _parse_transform(spec: Any, *, name: str) -> np.ndarray:
    """从 config 片段解析 4x4 齐次变换。

    支持形式：
    1) {"T": [[...],[...],[...],[...]]}
    2) {"R": [[...],[...],[...]], "t": [x,y,z]}
    3) {"translation": [x,y,z], "rotation_euler_deg": [roll,pitch,yaw], "euler_order": "XYZ"}

    说明：
    - 旋转欧拉角使用 SciPy Rotation.from_euler(order, angles, degrees=True)
    - order 默认 "XYZ"
    """

    if not isinstance(spec, dict):
        raise ValueError(f"{name}: 期望 dict，得到 {type(spec).__name__}")

    if "T" in spec:
        T = np.asarray(spec["T"], dtype=np.float64)
        ensure_T(T, name)
        return T

    if "R" in spec and "t" in spec:
        Rm = np.asarray(spec["R"], dtype=np.float64)
        t = np.asarray(spec["t"], dtype=np.float64).reshape(3)
        return make_T(Rm, t, name)

    if "translation" in spec and "rotation_euler_deg" in spec:
        t = np.asarray(spec["translation"], dtype=np.float64).reshape(3)
        euler = np.asarray(spec["rotation_euler_deg"], dtype=np.float64).reshape(3)
        order = str(spec.get("euler_order", "XYZ")) or "XYZ"
        Rm = Rotation.from_euler(order, euler.tolist(), degrees=True).as_matrix()
        return make_T(Rm, t, name)

    raise ValueError(
        f"{name}: 未识别的变换格式。需要 T 或 (R,t) 或 (translation,rotation_euler_deg)。"
    )


def _get_world_anchor_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    calib_cfg = (config or {}).get("camera_to_base_calibration", {})
    if not isinstance(calib_cfg, dict):
        calib_cfg = {}

    wa = calib_cfg.get("world_anchor", {})
    if not isinstance(wa, dict):
        wa = {}

    ref_cam = str(wa.get("reference_camera", ""))
    if not ref_cam:
        raise ValueError(
            "camera_to_base_calibration.world_anchor.reference_camera 未配置"
        )

    W_T_B = _parse_transform(wa.get("world_T_base", None), name="world_T_base")
    W_T_Cref = _parse_transform(
        wa.get("world_T_reference_camera", None), name="world_T_reference_camera"
    )

    return {
        "reference_camera": ref_cam,
        "W_T_B": W_T_B,
        "W_T_Cref": W_T_Cref,
        "raw": wa,
    }


def solve_camera_to_base_from_world(
    *,
    config: Dict[str, Any],
    results_dir: Path = Path("results"),
) -> Dict[str, Any]:
    """从世界锚点求解 B_T_C（并通过 Step4 外参传播）。

    返回值（稳定 API）：
    - B_T_C: Dict[str, np.ndarray]
    - methods: Dict[str, str]
    - propagation: Dict[str, Any]
    - anchor: Dict[str, Any]
    """

    wa = _get_world_anchor_cfg(config)
    anchor_cam = str(wa["reference_camera"])

    W_T_B = np.asarray(wa["W_T_B"], dtype=np.float64)
    W_T_Canchor = np.asarray(wa["W_T_Cref"], dtype=np.float64)

    # B_T_Canchor = inv(W_T_B) @ W_T_Canchor
    B_T_W = invert_transform(W_T_B, "B_T_W")
    B_T_Canchor = B_T_W @ W_T_Canchor
    ensure_T(B_T_Canchor, "B_T_Canchor")

    graph: Optional[ExtrinsicsGraph] = load_extrinsics_graph(results_dir=results_dir)

    B_T_C: Dict[str, np.ndarray] = {anchor_cam: B_T_Canchor}
    methods: Dict[str, str] = {anchor_cam: "world_anchor"}

    propagated: list[str] = []
    if graph is not None and len(graph.T_cam_from_ref) > 0:
        try:
            propagated_all = propagate_B_T_C(
                B_T_C_anchor=B_T_Canchor,
                anchor_cam=anchor_cam,
                graph=graph,
            )
        except ValueError as e:
            # 通常是 anchor_cam 不在 Step4 外参图里
            propagated_all = {anchor_cam: B_T_Canchor}
            methods[anchor_cam] = "world_anchor"
            propagation_info = {
                "used": False,
                "source": graph.source,
                "reference": graph.reference,
                "propagated_cameras": [],
                "warning": str(e),
            }
        else:
            for cam, T in propagated_all.items():
                if cam in B_T_C:
                    continue
                B_T_C[cam] = T
                methods[cam] = "propagated_from_step4"
                propagated.append(cam)

            propagation_info = {
                "used": bool(len(propagated) > 0),
                "source": graph.source,
                "reference": graph.reference,
                "propagated_cameras": propagated,
            }
    else:
        propagation_info = {
            "used": False,
            "source": None,
            "reference": None,
            "propagated_cameras": [],
        }

    return {
        "B_T_C": B_T_C,
        "methods": methods,
        "propagation": propagation_info,
        "anchor": {
            "reference_camera": anchor_cam,
            "W_T_B": W_T_B,
            "W_T_reference_camera": W_T_Canchor,
        },
    }
