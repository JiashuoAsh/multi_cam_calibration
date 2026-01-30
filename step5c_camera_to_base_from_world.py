#!/usr/bin/env python3
"""Step 5c: 基于“世界位姿锚点”的相机->底盘外参求解（支持多相机）

适用场景：
- 你已知机器人底盘在世界坐标系中的位姿 W_T_B（例如 SLAM/GNSS/动捕）。
- 你已知某个参考相机在世界坐标系中的位姿 W_T_Cref（例如动捕、或由其它系统直接给出）。
- 你已做过 Step4（相机间外参）：
    - results/multi_camera_extrinsics.json（pose graph），或
    - results/stereo_extrinsics.json（双目）。

核心推导（本仓库统一约定 A_T_B 表示 B->A）：
- 先把参考相机的 Cam->Base 求出来：
    B_T_Cref = inv(W_T_B) @ W_T_Cref
- 再用 Step4 外参传播到其它相机：
    已知 Step4 输出 C_cam_T_Cref_graph（Cam <- Ref_graph），则
    B_T_C_cam = B_T_Cref_graph @ inv(C_cam_T_Cref_graph)

输出：
- results/camera_to_base.json（相机位姿：B_T_C，Cam->Base）
- results/base_to_camera_extrinsics.json（相机外参：C_T_B，Base->Cam，便于 OpenCV 使用）

注意：
- 本步骤不需要 AprilTag 图片，也不依赖 board_to_base_transform。
- 但它强依赖 “世界系定义” 与 “底盘系定义” 的一致性；若世界系来自不同系统，请确保轴向/单位统一。
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from libs.extrinsics_graph import (
    ExtrinsicsGraph,
    invert_transform,
    load_extrinsics_graph,
    propagate_B_T_C,
    transform_payload,
)
from utils import load_config


def _ensure_transform(T: np.ndarray, name: str) -> None:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: 期望 (4,4)，得到 {T.shape}")
    bottom = T[3, :]
    if float(np.linalg.norm(bottom - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-8:
        raise ValueError(f"{name}: 底行应为 [0 0 0 1]，当前 {bottom}")


def _make_transform(Rm: np.ndarray, t: np.ndarray, name: str) -> np.ndarray:
    Rm = np.asarray(Rm, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t
    _ensure_transform(T, name)
    return T


def _parse_transform(spec: Any, *, name: str) -> np.ndarray:
    """从 config 片段解析 4x4 齐次变换。

    支持形式：
    1) {"T": [[...],[...],[...],[...]]}
    2) {"R": [[...],[...],[...]], "t": [x,y,z]}
    3) {"translation": [x,y,z], "rotation_euler_deg": [roll,pitch,yaw], "euler_order": "XYZ"}

    说明：
    - 旋转欧拉角使用 SciPy Rotation.from_euler(order, angles, degrees=True)
    - order 默认 "XYZ"（与 Step5b 的 board_to_base_transform 一致）
    """

    if not isinstance(spec, dict):
        raise ValueError(f"{name}: 期望 dict，得到 {type(spec).__name__}")

    if "T" in spec:
        T = np.asarray(spec["T"], dtype=np.float64)
        _ensure_transform(T, name)
        return T

    if "R" in spec and "t" in spec:
        Rm = np.asarray(spec["R"], dtype=np.float64)
        t = np.asarray(spec["t"], dtype=np.float64).reshape(3)
        return _make_transform(Rm, t, name)

    if "translation" in spec and "rotation_euler_deg" in spec:
        t = np.asarray(spec["translation"], dtype=np.float64).reshape(3)
        euler = np.asarray(spec["rotation_euler_deg"], dtype=np.float64).reshape(3)
        order = str(spec.get("euler_order", "XYZ"))
        if not order:
            order = "XYZ"
        Rm = Rotation.from_euler(order, euler.tolist(), degrees=True).as_matrix()
        return _make_transform(Rm, t, name)

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
    """核心求解：从世界锚点求解 B_T_C（并通过 Step4 外参传播）。"""

    wa = _get_world_anchor_cfg(config)
    anchor_cam = str(wa["reference_camera"])

    W_T_B = np.asarray(wa["W_T_B"], dtype=np.float64)
    W_T_Canchor = np.asarray(wa["W_T_Cref"], dtype=np.float64)

    # B_T_Canchor = inv(W_T_B) @ W_T_Canchor
    B_T_W = invert_transform(W_T_B, "B_T_W")
    B_T_Canchor = B_T_W @ W_T_Canchor
    _ensure_transform(B_T_Canchor, "B_T_Canchor")

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 5c: 基于世界位姿锚点的相机->底盘外参求解（支持多相机）"
    )
    parser.add_argument(
        "--config",
        default="config/apriltag_config.json",
        help="配置文件路径（默认：config/apriltag_config.json）",
    )
    args = parser.parse_args()

    try:
        cfg_path = str(args.config)
        config = load_config(cfg_path)

        out = solve_camera_to_base_from_world(config=config, results_dir=Path("results"))

        B_T_C: Dict[str, np.ndarray] = out["B_T_C"]
        methods: Dict[str, str] = out["methods"]

        B_T_C_detail: Dict[str, Any] = {}
        for cam, T in B_T_C.items():
            B_T_C_detail[str(cam)] = transform_payload(
                T,
                parent_frame="base",
                child_frame=f"camera:{cam}",
                name=f"B_T_C[{cam}]",
                include_inverse=False,
            )

        # 保存
        os.makedirs("results", exist_ok=True)
        payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "method": "world_anchor",
            "convention": {
                "A_T_B": "B->A",
                "apply": "p_A = A_T_B @ p_B",
                "matrix": "A_T_B = [[R,t],[0,1]]",
                "units": {"translation": "m"},
            },
            "meaning": {
                "B_T_C": "将点从相机坐标系变到 base 坐标系（Cam->Base）。等价于：相机坐标系在 base 中的位姿表达。",
                "opencv_note": "如需 OpenCV 常用外参方向（Base->Cam），请查看 results/base_to_camera_extrinsics.json。",
            },
            "frames": {
                "world": "外部系统定义的世界坐标系（W）",
                "base": "由 world_T_base 定义的底盘坐标系（B）",
                "camera": "OpenCV 相机坐标系：x 右、y 下、z 前（常见约定）",
            },
            "B_T_C": {cam: T.tolist() for cam, T in B_T_C.items()},
            "B_T_C_detail": B_T_C_detail,
            "pose_stats": {
                cam: {"method": methods.get(cam)} for cam in sorted(B_T_C.keys())
            },
            "propagation": out.get("propagation", {}),
            "anchor": {
                "reference_camera": out.get("anchor", {}).get("reference_camera"),
                "W_T_B": np.asarray(out.get("anchor", {}).get("W_T_B")).tolist()
                if out.get("anchor", {}).get("W_T_B") is not None
                else None,
                "W_T_reference_camera": np.asarray(
                    out.get("anchor", {}).get("W_T_reference_camera")
                ).tolist()
                if out.get("anchor", {}).get("W_T_reference_camera") is not None
                else None,
            },
            "config_used": {
                "camera_to_base_calibration": (config or {}).get(
                    "camera_to_base_calibration", {}
                )
            },
        }

        with open("results/camera_to_base.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # 单独输出 OpenCV 常用方向的外参：C_T_B（Base->Cam）
        C_T_B_mats: Dict[str, np.ndarray] = {
            cam: invert_transform(T, name=f"C_T_B[{cam}]") for cam, T in B_T_C.items()
        }
        C_T_B_detail: Dict[str, Any] = {}
        for cam, T in C_T_B_mats.items():
            C_T_B_detail[str(cam)] = transform_payload(
                T,
                parent_frame=f"camera:{cam}",
                child_frame="base",
                name=f"C_T_B[{cam}]",
                include_inverse=False,
            )

        extrinsics_payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "method": "base_to_camera_extrinsics",
            "source_pose_file": "results/camera_to_base.json",
            "convention": payload["convention"],
            "meaning": {
                "C_T_B": "将点从 base 坐标系变到相机坐标系（Base->Cam）。若把 base 当作 OpenCV 的 world，则这就是 OpenCV 常用外参方向。"
            },
            "frames": payload["frames"],
            "C_T_B": {cam: T.tolist() for cam, T in C_T_B_mats.items()},
            "C_T_B_detail": C_T_B_detail,
            "propagation": out.get("propagation", {}),
            "anchor": payload.get("anchor", {}),
            "config_used": payload.get("config_used", {}),
        }
        with open("results/base_to_camera_extrinsics.json", "w", encoding="utf-8") as f:
            json.dump(extrinsics_payload, f, indent=2, ensure_ascii=False)

        print("=" * 60)
        print("Step 5c: world-anchor 相机->底盘外参求解完成")
        print("=" * 60)
        print("[OK] 已保存: results/camera_to_base.json")
        print("[OK] 已保存: results/base_to_camera_extrinsics.json")
        print(f"  相机数: {len(B_T_C)}")
        if out.get("propagation", {}).get("source"):
            print(f"  使用的 Step4 外参: {out['propagation']['source']}")

        return 0

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
