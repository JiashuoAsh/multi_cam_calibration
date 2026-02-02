#!/usr/bin/env python3
"""Step 5b：相机 -> 底盘坐标系标定（支持多相机，CLI 入口）。

本模块是仓库根目录旧 Step5b 脚本的工程化落点：
- entry：负责 argparse、组装依赖、调用 adapters 扫描、调用 core 求解、结果落盘。
- core：负责纯数学计算（见 `mcca.core.step5_camera_to_base`）。

运行：
- `python -m mcca.entry.step5b_camera_to_base --config config/apriltag_config.json`

输出：
- results/camera_poses_B_T_C.json
- results/camera_extrinsics_C_T_B.json
- results/step5b_scan_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mcca.adapters.apriltag_perf.cache import CacheConfig
from mcca.adapters.apriltag_perf.prefilter import PrefilterConfig
from mcca.adapters.apriltag_perf.scan import ScanLimits, ScanOrder
from mcca.adapters.step5_apriltag_scan import (
    discover_step5_cameras,
    process_step5_images,
)
from mcca.core.board import create_apriltag_board, create_opencv_aruco_board, get_aruco_dict
from mcca.core.config import load_config
from mcca.core.datasets import get_step5_cameras, get_step5_dataset
from mcca.core.detection import get_detection_settings
from mcca.core.extrinsics_graph import transform_payload
from mcca.core.rigid import ensure_T, invert_T
from mcca.core.step5_camera_to_base import build_B_T_T_from_config, solve_camera_to_base


# 说明：命名中显式包含矩阵方向，避免“外参/位姿”混淆。
POSE_OUT_PATH = "results/camera_poses_B_T_C.json"  # 位姿：B_T_C（Cam->Base）
EXTRINSICS_OUT_PATH = "results/camera_extrinsics_C_T_B.json"  # 外参：C_T_B（Base->Cam）


# region 日志与格式化（verbose 控制）

# 默认尽量安静：只输出关键结果；需要更多过程信息用 --verbose。
VERBOSE: bool = False


def _vprint(*args, **kwargs) -> None:
    """仅在 VERBOSE=True 时打印。"""

    if VERBOSE:
        print(*args, **kwargs)


def _pretty_mat(name: str, T: np.ndarray, *, indent: str = "  ") -> str:
    """格式化 4x4 矩阵，便于 verbose 阅读。"""

    T = np.asarray(T, dtype=np.float64)
    s = np.array2string(
        T,
        formatter={"float_kind": lambda v: f"{float(v): .6f}"},
        suppress_small=False,
    )
    return f"{indent}{name} =\n{indent}{s.replace(chr(10), chr(10) + indent)}"


def _print_transform_sanity(*, name: str, T: np.ndarray, indent: str = "  ") -> None:
    """打印变换矩阵的自检信息（用于排查反射/不必要翻转）。

    说明：
        - 合法旋转应满足 det(R)=+1；若 det(R)=-1 则属于反射（镜像），不应出现在姿态链路中。
        - 严格旋转应满足 R^T R = I。
        - 轴向映射可帮助把欧拉角/矩阵转换为更直观的方向语义。
    """

    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]

    det = float(np.linalg.det(R))
    ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))

    x_T_in_parent = R @ np.array([1.0, 0.0, 0.0])
    y_T_in_parent = R @ np.array([0.0, 1.0, 0.0])
    z_T_in_parent = R @ np.array([0.0, 0.0, 1.0])

    _vprint(_pretty_mat(name, T, indent=indent))
    _vprint(f"{indent}{name}: det(R)={det:.6f} (期望 +1.0；若为 -1.0 则是反射/镜像)")
    _vprint(f"{indent}{name}: max|R^T R - I|={ortho_err:.3e} (越接近 0 越好)")
    _vprint(f"{indent}{name}: t(parent)={t.tolist()} (m)")
    _vprint(f"{indent}{name}: x_T_in_parent={x_T_in_parent.tolist()}")
    _vprint(f"{indent}{name}: y_T_in_parent={y_T_in_parent.tolist()}")
    _vprint(f"{indent}{name}: z_T_in_parent={z_T_in_parent.tolist()}")


def load_calibration_data(*, config_path: str) -> Dict[str, Any]:
    """加载标定数据（与相机数量无关）。"""

    _vprint("\n加载标定数据...")

    config = load_config(config_path)
    use_multiscale, opencv_refine = get_detection_settings(config)
    board_cfg = config["apriltag_board"]
    transform_cfg = config["board_to_base_transform"]

    # 这两行对使用者很关键：保留为非 verbose 输出
    print("\n标定板到底盘的变换:")
    print(f"  - 平移 (m): {transform_cfg['translation']}")
    print(f"  - 旋转 (度): {transform_cfg['rotation_euler_deg']}")

    # verbose：额外打印“程序实际使用的 B_T_T”，避免只盯着欧拉角导致误解。
    if VERBOSE:
        translation_ref = str(transform_cfg.get("translation_reference", "tag0_center"))
        ref_point = transform_cfg.get("translation_reference_point_in_T_m", None)

        _vprint("\n[verbose] B_T_T 构造与自检（用于排查反射/翻转/参考点定义）")
        _vprint(f"  translation_reference: {translation_ref!r}")
        _vprint(f"  translation_reference_point_in_T_m: {ref_point}")
        if translation_ref != "tag0_center" and ref_point is None:
            _vprint(
                "  注意：translation_reference != 'tag0_center' 且未提供 translation_reference_point_in_T_m。\n"
                "  程序将回退到默认网格中心（由 tag_size + tag_spacing + tags_x/tags_y 估算）。"
            )

        B_T_T = build_B_T_T_from_config(transform_cfg=transform_cfg, board_cfg=board_cfg)
        _print_transform_sanity(name="B_T_T (T->B)", T=B_T_T)

    # 创建 AprilTag 标定板
    obj_points_mm, tag_ids = create_apriltag_board(config)
    obj_points = obj_points_mm.astype(np.float64) / 1000.0
    aruco_dict = get_aruco_dict(board_cfg["family"])
    board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)

    return {
        "config": config,
        "board_cfg": board_cfg,
        "transform_cfg": transform_cfg,
        "obj_points": obj_points,
        "tag_ids": tag_ids,
        "aruco_dict": aruco_dict,
        "opencv_board": board,
        "use_multiscale": use_multiscale,
        "opencv_refine": opencv_refine,
    }


# endregion


def save_calibration_results(
    calibration_data: Dict[str, Any],
    pose_data: Dict[str, Any],
    solver_result,
    *,
    image_root: Path,
) -> None:
    """保存标定结果到 JSON 文件。"""

    B_T_C: Dict[str, np.ndarray] = solver_result.B_T_C
    methods: Dict[str, str] = solver_result.methods

    pose_stats: Dict[str, Dict[str, Any]] = pose_data.get("pose_stats", {})
    per_cam_stats: Dict[str, Dict[str, Any]] = {}
    for cam in sorted(set(list(pose_stats.keys()) + list(B_T_C.keys()))):
        s = dict(pose_stats.get(cam, {}))
        s["method"] = methods.get(cam)
        per_cam_stats[cam] = s

    # 说明：位姿/外参分开写文件：
    # - results/camera_poses_B_T_C.json：仅保存相机“位姿”（B_T_C, Cam->Base）
    # - results/camera_extrinsics_C_T_B.json：仅保存相机“外参”（C_T_B, Base->Cam）
    B_T_C_detail: Dict[str, Any] = {}
    for cam, T in B_T_C.items():
        B_T_C_detail[str(cam)] = transform_payload(
            T,
            parent_frame="base",
            child_frame=f"camera:{cam}",
            name=f"B_T_C[{cam}]",
            include_inverse=False,
        )

    result: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "image_root": str(image_root.as_posix()),
        "convention": {
            "A_T_B": "B->A",
            "apply": "p_A = A_T_B @ p_B",
            "matrix": "A_T_B = [[R,t],[0,1]]",
            "units": {"translation": "m"},
        },
        "meaning": {
            "B_T_C": "将点从相机坐标系变到 base 坐标系（Cam->Base）。等价于：相机坐标系在 base 中的位姿表达。"
        },
        "frames": {
            "base": "由 board_to_base_transform 定义的机器人底盘坐标系",
            "camera": "OpenCV 相机坐标系：x 右、y 下、z 前（常见约定）",
        },
        "B_T_C": {cam: T.tolist() for cam, T in B_T_C.items()},
        "B_T_C_detail": B_T_C_detail,
        "pose_stats": per_cam_stats,
        "propagation": solver_result.propagation,
        "config_used": {
            "board_to_base_transform": calibration_data["transform_cfg"],
            "apriltag_board": calibration_data["board_cfg"],
        },
    }

    os.makedirs("results", exist_ok=True)
    with open(POSE_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 单独输出 OpenCV 常用方向的外参：C_T_B（Base->Cam）
    C_T_B_mats: Dict[str, np.ndarray] = {}
    for cam, T in B_T_C.items():
        C_T_B = invert_T(np.asarray(T, dtype=np.float64), name=f"C_T_B[{cam}]")
        ensure_T(C_T_B, f"C_T_B[{cam}]")
        C_T_B_mats[str(cam)] = C_T_B

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
        "convention": result["convention"],
        "meaning": {
            "C_T_B": "将点从 base 坐标系变到相机坐标系（Base->Cam）。若把 base 当作 OpenCV 的 world，则这就是 OpenCV 常用外参方向。"
        },
        "frames": result["frames"],
        "C_T_B": {cam: T.tolist() for cam, T in C_T_B_mats.items()},
        "C_T_B_detail": C_T_B_detail,
        "propagation": solver_result.propagation,
        "config_used": result.get("config_used", {}),
    }
    with open(EXTRINSICS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(extrinsics_payload, f, indent=2, ensure_ascii=False)

    # 单独输出 scan/perf 报告，便于对比不同参数的检测耗时、缓存命中率等。
    scan_reports = pose_data.get("scan_reports") or {}
    try:
        with open("results/step5b_scan_report.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "image_root": str(image_root.as_posix()),
                    "scan_reports": scan_reports,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass

    print(f"\n位姿已保存到 {POSE_OUT_PATH}")
    print(f"外参已保存到 {EXTRINSICS_OUT_PATH}")
    print(f"  相机数: {len(B_T_C)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Step 5b: AprilTag 相机到机器人底盘外参标定（支持多相机）")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出更多中间过程信息（用于排查/对齐坐标系与单位）。",
    )
    parser.add_argument(
        "--config",
        default="config/apriltag_config.json",
        help="配置文件路径（默认：config/apriltag_config.json）",
    )
    parser.add_argument(
        "--image_root",
        default=None,
        help="Step5 图片根目录（默认 None：优先读 config.step5_dataset.image_root；否则 images/step5）",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="相机列表（空则自动扫描 image_root 下的子目录）",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="限制每个相机最多处理的图片数量（默认处理全部）",
    )

    parser.add_argument(
        "--min_tags",
        type=int,
        default=0,
        help="每张图最少 tag 数（0=使用 config.calibration_settings.min_tags_for_pose）。",
    )

    # 性能优先：流式扫描/早停/并行/缓存/预筛选
    parser.add_argument(
        "--max_valid_poses",
        type=int,
        default=30,
        help="每个相机最多使用多少个有效位姿进行平均（默认 30；0=不按此条件早停）。",
    )
    parser.add_argument(
        "--max_total_images",
        type=int,
        default=0,
        help="每个相机最多尝试检测多少张候选图像（0=自动=8*max_valid_poses）。",
    )
    parser.add_argument(
        "--max_detect_seconds",
        type=float,
        default=0.0,
        help="每个相机检测总耗时上限（秒，0=不限制）。",
    )
    parser.add_argument(
        "--scan_strategy",
        type=str,
        default="uniform",
        choices=["sequential", "random", "uniform"],
        help="候选图像扫描策略：sequential/random/uniform（默认 uniform）。",
    )
    parser.add_argument(
        "--scan_seed",
        type=int,
        default=0,
        help="random 策略的随机种子（保证可复现）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="多进程 worker 数（0=自动=CPU核数；1=禁用并行）。",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=0,
        help="并行时的预提交任务数（0=自动）。",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache/apriltag_detection",
        help="检测缓存目录（默认 cache/apriltag_detection）。",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="禁用检测缓存（会显著变慢）。",
    )
    parser.add_argument(
        "--force_redetect",
        action="store_true",
        help="忽略缓存强制重新检测（用于调参/排查）。",
    )
    parser.add_argument(
        "--prefilter",
        action="store_true",
        help="启用廉价预筛选（可能过滤掉明显无效帧，减少 detector 调用）。",
    )

    parser.add_argument(
        "--no_multiscale",
        action="store_true",
        help="强制关闭多尺度检测（更快，但对小标签/远距离可能更不稳）。",
    )
    parser.add_argument(
        "--no_opencv_refine",
        action="store_true",
        help="强制关闭 OpenCV refine（更快）。",
    )

    args = parser.parse_args(argv)

    global VERBOSE
    VERBOSE = bool(args.verbose)

    print("=" * 60)
    print("Step 5b: AprilTag 相机到底盘坐标系标定")
    if VERBOSE:
        print("(verbose: ON)")
    print("=" * 60)

    try:
        cfg_path = str(args.config)
        config = load_config(cfg_path)
        ds = get_step5_dataset(config)

        if args.image_root is not None:
            image_root = Path(str(args.image_root))
        else:
            image_root = ds.get("image_root", Path("images/step5"))
            if not isinstance(image_root, Path):
                image_root = Path(str(image_root))

        cameras = args.cameras
        if cameras is None:
            if bool(ds.get("enabled", False)):
                cameras = get_step5_cameras(config, allow_scan=True)
            else:
                cameras = discover_step5_cameras(image_root)

        if len(cameras) == 0:
            raise ValueError(
                f"未发现任何相机目录：{image_root} 下没有可用子目录。\n"
                "期望结构：images/step5/<cam>/*.png|jpg|jpeg|bmp"
            )

        # 1) 加载标定数据
        calibration_data = load_calibration_data(config_path=cfg_path)

        # detection overrides
        if bool(getattr(args, "no_multiscale", False)):
            calibration_data["use_multiscale"] = False
        if bool(getattr(args, "no_opencv_refine", False)):
            calibration_data["opencv_refine"] = False

        min_tags_cfg = int(config.get("calibration_settings", {}).get("min_tags_for_pose", 4))
        min_tags = int(args.min_tags) if int(args.min_tags) > 0 else min_tags_cfg

        cache_cfg = CacheConfig(
            enabled=not bool(args.no_cache),
            cache_dir=str(args.cache_dir),
            force_redetect=bool(args.force_redetect),
        )
        prefilter_cfg = PrefilterConfig(enabled=bool(args.prefilter))
        scan_limits = ScanLimits(
            target_valid=int(args.max_valid_poses),
            max_total=int(args.max_total_images),
            max_seconds=float(args.max_detect_seconds),
        )
        scan_order = ScanOrder(strategy=str(args.scan_strategy), seed=int(args.scan_seed))

        # 2) 处理图像并计算各相机位姿 C_T_T
        pose_data = process_step5_images(
            calibration_data,
            image_root=image_root,
            cameras=list(cameras),
            config_path=cfg_path,
            max_images=args.max_images,
            min_tags=int(min_tags),
            scan_limits=scan_limits,
            scan_order=scan_order,
            workers=int(args.workers),
            prefetch=int(args.prefetch),
            cache_cfg=cache_cfg,
            prefilter_cfg=prefilter_cfg,
            verbose=bool(VERBOSE),
        )

        # 3) 计算相机到底盘的变换矩阵 B_T_C（含 Step4 传播）
        solver_result = solve_camera_to_base(
            transform_cfg=calibration_data["transform_cfg"],
            board_cfg=calibration_data["board_cfg"],
            C_T_T_by_cam=pose_data["C_T_T_by_cam"],
            results_dir=Path("results"),
        )

        # 4) 保存标定结果
        save_calibration_results(
            calibration_data,
            pose_data,
            solver_result,
            image_root=image_root,
        )

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        return 1

    return 0


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
