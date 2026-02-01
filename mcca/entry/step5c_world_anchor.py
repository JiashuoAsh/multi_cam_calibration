#!/usr/bin/env python3
"""Step 5c 入口：基于“世界位姿锚点”的相机->底盘外参求解（支持多相机）。

说明：
- 纯求解逻辑位于 `mcca.core.camera_to_base_world_anchor`。
- 本模块仅负责 CLI + 读配置 + 写 results/*.json。
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from mcca.core.camera_to_base_world_anchor import (
    solve_camera_to_base_from_world,
)
from mcca.core.config import load_config
from mcca.core.extrinsics_graph import (
    invert_transform,
    transform_payload,
)


# 说明：命名中显式包含矩阵方向，避免“外参/位姿”混淆。
POSE_OUT_PATH = "results/camera_poses_B_T_C.json"  # 位姿：B_T_C（Cam->Base）
EXTRINSICS_OUT_PATH = "results/camera_extrinsics_C_T_B.json"  # 外参：C_T_B（Base->Cam）


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 5c: 基于世界位姿锚点的相机->底盘外参求解（支持多相机）"
    )
    parser.add_argument(
        "--config",
        default="config/apriltag_config.json",
        help="配置文件路径（默认：config/apriltag_config.json）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

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
                "opencv_note": f"如需 OpenCV 常用外参方向（Base->Cam），请查看 {EXTRINSICS_OUT_PATH}。",
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

        with open(POSE_OUT_PATH, "w", encoding="utf-8") as f:
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
            "source_pose_file": POSE_OUT_PATH,
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
        with open(EXTRINSICS_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(extrinsics_payload, f, indent=2, ensure_ascii=False)

        print("=" * 60)
        print("Step 5c: world-anchor 相机->底盘外参求解完成")
        print("=" * 60)
        print(f"[OK] 已保存: {POSE_OUT_PATH}")
        print(f"[OK] 已保存: {EXTRINSICS_OUT_PATH}")
        print(f"  相机数: {len(B_T_C)}")
        if out.get("propagation", {}).get("source"):
            print(f"  使用的 Step4 外参: {out['propagation']['source']}")

        return 0

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        return 1


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
