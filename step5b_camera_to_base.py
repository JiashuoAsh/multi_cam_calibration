#!/usr/bin/env python3
"""Step 5b: 相机到底盘坐标系标定（支持多相机）

核心思路
1) 已知标定板在底盘坐标系中的固定安装位姿（config: board_to_base_transform），构造 B_T_T（T->B）。
2) 对每个相机，用 Step5 图像估计 C_T_T（T->C，PnP 结果），再得到：

     B_T_C = B_T_T @ inv(C_T_T)

3) 若某些相机缺少 Step5 图像，但已做过 Step4（相机间外参），可用 Step4 输出把 B_T_C 从已知相机传播到其它相机。

输入
- Step5 图像：默认 images/step5/<cam>/*.png|jpg|jpeg|bmp
- 每个相机内参：results/<cam>_intrinsics.json（由 Step3 生成）
- Step4 相机间外参（可选）：
    - results/multi_camera_extrinsics.json（多相机 pose graph 输出）
    - 或 results/stereo_extrinsics.json（双目输出，作为特例）

输出
- results/camera_to_base.json：
    - B_T_C: {cam: 4x4}，表示 Cam -> Base
"""

import argparse
import json
import os
import glob
import traceback
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation
from libs.extrinsics_graph import load_extrinsics_graph, propagate_B_T_C
from utils import (
    load_config,
    get_step5_dataset,
    get_step5_cameras,
    get_step5_camera_images,
    get_aruco_dict,
    create_apriltag_board,
    create_opencv_aruco_board,
    get_detection_settings,
    get_detection_profile,
    get_detection_roi,
    get_detection_auto_roi,
    create_detector_params,
    estimate_pose_apriltag,
)

from libs.apriltag_perf.cache import CacheConfig
from libs.apriltag_perf.prefilter import PrefilterConfig
from libs.apriltag_perf.scan import ScanLimits, ScanOrder, iter_scan_parallel_ordered, iter_scan_sequential
from libs.apriltag_perf.service import CachedAprilTagDetector


# region 日志与格式化（verbose 控制）


# 默认尽量安静：只输出关键结果；需要更多过程信息用 --verbose。
VERBOSE: bool = False


_G_DET: Optional[CachedAprilTagDetector] = None
_G_OBJ_POINTS: Optional[np.ndarray] = None
_G_TAG_IDS: Optional[List[int]] = None
_G_K: Optional[np.ndarray] = None
_G_DIST: Optional[np.ndarray] = None
_G_MIN_TAGS: int = 1


def _vprint(*args, **kwargs) -> None:
    """仅在 VERBOSE=True 时打印。"""
    if VERBOSE:
        print(*args, **kwargs)


def _fmt4(x) -> str:
    """将向量/矩阵展平后以4位小数格式化为字符串。"""
    arr = np.asarray(x, dtype=np.float64)
    return "[" + ", ".join(f"{v:.4f}" for v in arr.reshape(-1)) + "]"


def _pretty_mat(name: str, T: np.ndarray, indent: str = "  ") -> None:
    """用4位小数打印矩阵。"""
    T = np.asarray(T, dtype=np.float64)
    s = np.array2string(
        T,
        formatter={"float_kind": lambda v: f"{float(v): .4f}"},
        suppress_small=False,
    )
    _vprint(f"{indent}{name} =\n{indent}{s.replace(chr(10), chr(10) + indent)}")


# endregion


def _build_algo_key(config: Dict[str, Any], *, profile: str) -> Dict[str, Any]:
    """构建用于缓存的 algo_key（必须可 JSON 序列化）。"""

    det_cfg = (config or {}).get("calibration_settings", {}).get("detection", {})
    board_cfg = (config or {}).get("apriltag_board", {})
    corner_ref = (config or {}).get("calibration_settings", {}).get("corner_refinement")

    return {
        "opencv_version": str(getattr(cv2, "__version__", "unknown")),
        "apriltag_family": str(board_cfg.get("family")),
        "corner_refinement": str(corner_ref),
        "profile": str(profile),
        "detection": det_cfg,
    }


# region SE(3) 变换工具


def _ensure_transform(T: np.ndarray, name: str) -> None:
    """检查4x4齐次变换矩阵格式。"""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: 期望 (4,4)，得到 {T.shape}")

    bottom = T[3, :]
    bottom_target = np.array([0.0, 0.0, 0.0, 1.0])
    bottom_err = float(np.linalg.norm(bottom - bottom_target))
    if bottom_err > 1e-6:
        raise ValueError(f"{name}: 底行应为 [0 0 0 1]，当前 {bottom}")


def _make_transform(R: np.ndarray, t: np.ndarray, name: str) -> np.ndarray:
    """构造4x4齐次变换矩阵。"""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t
    _ensure_transform(T, name)
    return T


def _invert_transform(T: np.ndarray, name: str, *, strict: bool = True) -> np.ndarray:
    """对 4x4 齐次变换求逆（解析法）。

    说明：
        对“刚体变换”而言，逆变换可以解析地写成：
        R^{-1} = R^T，t^{-1} = -R^T t。

        为避免数值误差/实现问题，本函数在 verbose 下会对比一次 `np.linalg.inv` 的结果，
        并在 strict 模式下做一次 Frobenius 范数自检。
    """
    T = np.asarray(T, dtype=np.float64)
    _ensure_transform(T, name + ".input")

    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = _make_transform(R_inv, t_inv, name)

    # 对比数值求逆，主要用于排查“不是刚体变换/矩阵坏了”等异常。
    if VERBOSE:
        T_inv_num = np.linalg.inv(T)
        diff_r = float(np.linalg.norm(T_inv_num[:3, :3] - R_inv))
        diff_t = float(np.linalg.norm(T_inv_num[:3, 3] - t_inv))
        _vprint(f"  [对比] {name}: diff_R={diff_r:.3e}, diff_t={diff_t:.3e}")

    if strict:
        err = float(np.linalg.norm(T @ T_inv - np.eye(4), ord="fro"))
        _vprint(f"  [自检] {name}: inv_err_fro={err:.3e}")
        if err > 1e-6:
            raise ValueError(f"{name}: 求逆自检失败，误差过大 ({err})")

    return T_inv


def _euler_to_rotation_matrix(roll, pitch, yaw, degrees=True):
    """欧拉角转旋转矩阵（XYZ内旋）。"""
    R = Rotation.from_euler("XYZ", [roll, pitch, yaw], degrees=degrees).as_matrix()
    return np.asarray(R, dtype=np.float64)


def _tf_point(T_4x4: np.ndarray, p_3: np.ndarray) -> np.ndarray:
    """用4x4变换矩阵变换3D点。"""
    p_3 = np.asarray(p_3, dtype=np.float64).reshape(3, 1)
    p_h = np.vstack([p_3, [[1.0]]])
    result = T_4x4 @ p_h
    return result[:3, 0]


# endregion


def _get_prominent_tag_ids(
    tags_x: int, tags_y: int, tag_centers: Dict[int, np.ndarray]
) -> List[int]:
    """获取显眼的tag ID（四角+中心）。"""
    candidate_ids = [
        0,
        tags_x - 1,
        (tags_y - 1) * tags_x,
        tags_x * tags_y - 1,
        (tags_y // 2) * tags_x + (tags_x // 2),
    ]
    return [tid for tid in candidate_ids if tid in tag_centers]


def _init_step5_pose_worker(state: Dict[str, Any]) -> None:
    """多进程 worker 初始化：为单相机创建 CachedAprilTagDetector。"""

    global _G_DET, _G_OBJ_POINTS, _G_TAG_IDS, _G_K, _G_DIST, _G_MIN_TAGS

    config = state["config"]
    profile = str(state["profile"])

    _G_OBJ_POINTS = np.asarray(state["obj_points"], dtype=np.float64)
    _G_TAG_IDS = [int(x) for x in state["tag_ids"]]
    _G_K = np.asarray(state["K"], dtype=np.float64)
    _G_DIST = np.asarray(state["dist"], dtype=np.float64)
    _G_MIN_TAGS = int(state["min_tags"])

    aruco_dict = get_aruco_dict(str(state["family"]))
    board = state.get("opencv_board")

    detector_params = create_detector_params(config)
    algo_key = _build_algo_key(config, profile=profile)

    cache_cfg = CacheConfig(
        enabled=bool(state["cache"]["enabled"]),
        cache_dir=str(state["cache"]["cache_dir"]),
        force_redetect=bool(state["cache"]["force_redetect"]),
    )
    prefilter_cfg = PrefilterConfig(enabled=bool(state["prefilter"]["enabled"]))

    roi_raw = state.get("roi")
    roi_t: Optional[Tuple[int, int, int, int]] = None
    if isinstance(roi_raw, (list, tuple)) and len(roi_raw) == 4:
        roi_t = (int(roi_raw[0]), int(roi_raw[1]), int(roi_raw[2]), int(roi_raw[3]))

    auto_roi_cfg = state.get("auto_roi_cfg") or {}

    _G_DET = CachedAprilTagDetector(
        aruco_dict=aruco_dict,
        detector_params=detector_params,
        algo_key=algo_key,
        use_multiscale=bool(state["use_multiscale"]),
        opencv_refine=bool(state["opencv_refine"]),
        board=board,
        camera_matrix=_G_K,
        dist_coeffs=_G_DIST,
        roi=roi_t,
        auto_roi=bool(auto_roi_cfg.get("enabled", False)),
        auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
        auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
        auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
        cache_cfg=cache_cfg,
        prefilter_cfg=prefilter_cfg,
    )


def _step5_pose_worker(image_path: str) -> Dict[str, Any]:
    """多进程扫描的单任务：检测并尝试 PnP，返回 rvec/tvec（若成功）。"""

    global _G_DET, _G_OBJ_POINTS, _G_TAG_IDS, _G_K, _G_DIST, _G_MIN_TAGS

    if _G_DET is None or _G_OBJ_POINTS is None or _G_TAG_IDS is None or _G_K is None or _G_DIST is None:
        return {"path": str(image_path), "valid": False, "error": "worker 未初始化"}

    p = str(image_path)
    res = _G_DET.detect_path(Path(p))
    ids = np.asarray(res.ids) if res.ids is not None else np.zeros((0, 1), dtype=np.int32)
    n_tags = int(ids.shape[0])

    out: Dict[str, Any] = {
        "path": p,
        "valid": False,
        "n_tags": int(n_tags),
        "from_cache": bool(res.from_cache),
        "status": int(res.status),
        "elapsed_ms": float(res.elapsed_ms),
    }

    if n_tags < int(_G_MIN_TAGS):
        return out

    ok, rvec, tvec = estimate_pose_apriltag(
        res.corners,
        res.ids,
        _G_OBJ_POINTS,
        _G_TAG_IDS,
        _G_K,
        _G_DIST,
    )

    if not ok:
        out["pnp_ok"] = False
        return out

    out["pnp_ok"] = True
    out["valid"] = True
    out["rvec"] = np.asarray(rvec, dtype=np.float64).reshape(3).tolist()
    out["tvec"] = np.asarray(tvec, dtype=np.float64).reshape(3).tolist()
    return out


def _scan_step5_poses_for_camera(
    *,
    image_paths: List[str],
    config: Dict[str, Any],
    cam: str,
    K: np.ndarray,
    dist: np.ndarray,
    opencv_board,
    obj_points: np.ndarray,
    tag_ids: List[int],
    use_multiscale: bool,
    opencv_refine: bool,
    min_tags: int,
    scan_limits: ScanLimits,
    scan_order: ScanOrder,
    workers: int,
    prefetch: int,
    cache_cfg: CacheConfig,
    prefilter_cfg: PrefilterConfig,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    """对单个相机的 Step5 图像做流式扫描，收集一定数量的有效 PnP 位姿。"""

    profile = get_detection_profile(config)
    auto_roi_cfg = get_detection_auto_roi(config)
    roi = get_detection_roi(config, camera=cam)

    family = (config or {}).get("apriltag_board", {}).get("family")
    if family is None:
        raise ValueError("config.apriltag_board.family 缺失")

    if int(workers) <= 0:
        workers = int(os.cpu_count() or 1)

    # 若用户未给 max_total，则按 target_valid 做一个默认上限，避免无意扫全量导致耗时爆炸。
    if int(scan_limits.max_total) <= 0 and int(scan_limits.target_valid) > 0:
        scan_limits = ScanLimits(
            target_valid=int(scan_limits.target_valid),
            max_total=int(min(len(image_paths), max(8 * int(scan_limits.target_valid), int(scan_limits.target_valid)))),
            max_seconds=float(scan_limits.max_seconds),
        )

    state: Dict[str, Any] = {
        "config": config,
        "profile": str(profile),
        "family": str(family),
        "opencv_board": opencv_board,
        "obj_points": np.asarray(obj_points, dtype=np.float64).tolist(),
        "tag_ids": [int(x) for x in tag_ids],
        "K": np.asarray(K, dtype=np.float64).tolist(),
        "dist": np.asarray(dist, dtype=np.float64).tolist(),
        "roi": (list(roi) if roi is not None else None),
        "auto_roi_cfg": auto_roi_cfg,
        "use_multiscale": bool(use_multiscale),
        "opencv_refine": bool(opencv_refine),
        "min_tags": int(min_tags),
        "cache": {
            "enabled": bool(cache_cfg.enabled),
            "cache_dir": str(cache_cfg.cache_dir),
            "force_redetect": bool(cache_cfg.force_redetect),
        },
        "prefilter": {"enabled": bool(prefilter_cfg.enabled)},
    }

    def _is_valid(r: Dict[str, Any]) -> bool:
        return bool(r.get("valid", False))

    if int(workers) <= 1:
        _init_step5_pose_worker(state)
        results, counters = iter_scan_sequential(
            image_paths,
            worker_fn=_step5_pose_worker,
            is_valid_fn=_is_valid,
            limits=scan_limits,
            order=scan_order,
        )
    else:
        results, counters = iter_scan_parallel_ordered(
            image_paths,
            worker_fn=_step5_pose_worker,
            is_valid_fn=_is_valid,
            limits=scan_limits,
            order=scan_order,
            max_workers=int(workers),
            prefetch=int(prefetch),
            initializer=_init_step5_pose_worker,
            initargs=(state,),
        )

    # 聚合统计
    det_total = 0
    det_cache_hit = 0
    det_cache_miss = 0
    det_prefilter_skipped = 0
    det_error = 0
    det_ms_sum = 0.0
    pnp_ok = 0
    pnp_fail = 0

    valid_poses: List[Tuple[np.ndarray, np.ndarray]] = []
    for r in results:
        det_total += 1
        if bool(r.get("from_cache", False)):
            det_cache_hit += 1
        else:
            det_cache_miss += 1
        st = int(r.get("status", 0))
        if st == 2:
            det_prefilter_skipped += 1
        elif st == 1:
            det_error += 1
        det_ms_sum += float(r.get("elapsed_ms", 0.0))

        if "pnp_ok" in r:
            if bool(r.get("pnp_ok")):
                pnp_ok += 1
            else:
                pnp_fail += 1

        if bool(r.get("valid", False)):
            rv = np.asarray(r.get("rvec"), dtype=np.float64).reshape(3, 1)
            tv = np.asarray(r.get("tvec"), dtype=np.float64).reshape(3, 1)
            valid_poses.append((rv, tv))

    stop_reason = ""
    if int(scan_limits.target_valid) > 0 and int(counters.valid) >= int(scan_limits.target_valid):
        stop_reason = "target_valid_poses"
    elif int(scan_limits.max_total) > 0 and int(counters.completed) >= int(scan_limits.max_total):
        stop_reason = "max_total_images"
    elif float(scan_limits.max_seconds) > 0 and float(counters.elapsed_s) >= float(scan_limits.max_seconds):
        stop_reason = "max_detect_seconds"
    else:
        stop_reason = "exhausted_candidates"

    scan_report: Dict[str, Any] = {
        "cam": str(cam),
        "profile": str(profile),
        "min_tags": int(min_tags),
        "scan": {
            "strategy": str(scan_order.strategy),
            "seed": int(scan_order.seed),
            "limits": {
                "target_valid": int(scan_limits.target_valid),
                "max_total": int(scan_limits.max_total),
                "max_seconds": float(scan_limits.max_seconds),
            },
            "counters": {
                "total_candidates": int(counters.total_candidates),
                "submitted": int(counters.submitted),
                "completed": int(counters.completed),
                "valid": int(counters.valid),
                "elapsed_s": float(counters.elapsed_s),
            },
            "stop_reason": str(stop_reason),
        },
        "perf": {
            "workers": int(workers),
            "prefetch": int(prefetch),
            "cache": {
                "enabled": bool(cache_cfg.enabled),
                "cache_dir": str(cache_cfg.cache_dir),
                "force_redetect": bool(cache_cfg.force_redetect),
            },
            "prefilter": {"enabled": bool(prefilter_cfg.enabled)},
        },
        "detector_calls": {
            "total_images": int(det_total),
            "cache_hit": int(det_cache_hit),
            "cache_miss": int(det_cache_miss),
            "prefilter_skipped": int(det_prefilter_skipped),
            "error": int(det_error),
            "detect_ms_sum": float(det_ms_sum),
        },
        "pnp": {
            "ok": int(pnp_ok),
            "fail": int(pnp_fail),
        },
        "results": {
            "valid_poses": int(len(valid_poses)),
        },
    }

    return valid_poses, scan_report


def load_intrinsics(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载相机内参。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["camera_matrix"], dtype=np.float64), np.array(
        data["dist_coeffs"], dtype=np.float64
    )


def _intrinsics_path_for_camera(cam: str) -> str:
    """获取某相机的内参路径（统一为 results/<cam>_intrinsics.json）。"""
    return str(Path("results") / f"{cam}_intrinsics.json")


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


def _discover_cameras(image_root: Path) -> List[str]:
    """从 image_root 下的子目录自动发现相机名。"""
    cams: List[str] = []
    if not image_root.exists():
        return cams

    for p in sorted(image_root.iterdir()):
        if not p.is_dir():
            continue
        # 说明：image_root 下可能存在 _comment 等说明目录，不应当被当作相机。
        if p.name.startswith("_") or p.name.startswith(".") or p.name.startswith("__"):
            continue
        # 仅当目录里存在图片文件才认为是相机
        imgs = _glob_images(p)
        if len(imgs) > 0:
            cams.append(p.name)
    return cams


def _glob_images(cam_dir: Path) -> List[str]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(str(cam_dir / pat)))
    return sorted(files)


def compute_C_T_T_mean_pose(valid_rvecs, valid_tvecs, *, cam_name: str) -> np.ndarray:
    """由多帧PnP结果计算平均位姿，输出 {cam_name}_T_T (T -> cam)。"""
    _vprint(f"  计算 {cam_name}_T_T 平均位姿（{len(valid_rvecs)} 帧）")

    if len(valid_rvecs) == 0:
        raise ValueError(f"{cam_name}: 没有有效位姿")

    if len(valid_rvecs) > 1:
        t_std = np.std(valid_tvecs, axis=0)
        t_std_norm = np.linalg.norm(t_std)

        # 简单估算旋转的稳定性
        r_std = np.std(valid_rvecs, axis=0)
        r_std_norm = np.linalg.norm(r_std)

        _vprint(f"  [稳定性检查] 平移标准差: {t_std_norm*1000:.2f} mm (越小越好)")
        _vprint(f"  [稳定性检查] 旋转标准差: {r_std_norm:.4f} rad")

        if t_std_norm > 0.005: # 阈值 5mm
            _vprint("  警告: 位姿抖动较大 (>5mm)，建议增加光照或检查标定板是否晃动")
        else:
            _vprint("  位姿稳定，单位置标定可靠")

    # 对旋转向量取平均（转为旋转矩阵后平均）
    R_matrices = [cv2.Rodrigues(rvec)[0] for rvec in valid_rvecs]
    R_mean = np.mean(R_matrices, axis=0)

    # SVD正交化，确保是有效旋转
    U, _, Vt = np.linalg.svd(R_mean)
    R_mean = U @ Vt

    # 避免反射(det=-1)
    if float(np.linalg.det(R_mean)) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt

    t_mean = np.mean(valid_tvecs, axis=0)
    C_T_T = _make_transform(R_mean, t_mean, f"{cam_name}_T_T")
    _pretty_mat(f"{cam_name}_T_T", C_T_T)
    return C_T_T


def build_B_T_T_from_config(
    transform_cfg: Dict[str, Any], board_cfg: Dict[str, Any]
) -> np.ndarray:
    """由配置构造 B_T_T (T -> B)。"""
    _vprint("\n[求解] 构造标定板到底盘的变换 (B_T_T: T -> B)")

    rotation_euler = transform_cfg["rotation_euler_deg"]
    R_B_T = _euler_to_rotation_matrix(*rotation_euler, degrees=True)

    translation_ref = transform_cfg.get("translation_reference", "tag0_center")
    translation_input_B = np.array(
        transform_cfg["translation"], dtype=np.float64
    ).reshape(3, 1)

    # translation_reference_point_in_T_m: 参考点在 T 坐标系中的位置（米）
    # 其中 T 坐标系原点为 Tag0 中心（由 create_apriltag_board 的 3D 点定义决定）。
    ref_point_cfg = transform_cfg.get("translation_reference_point_in_T_m", None)

    # 默认网格中心（tag 中心点的几何中心），用于：
    # 1) 没配 ref_point 时兼容回退；2) 打印对照方便排查。
    tag_pitch_m = (board_cfg["tag_size"] + board_cfg["tag_spacing"]) / 1000.0
    default_grid_center_T = np.array(
        [
            (board_cfg["tags_x"] - 1) * tag_pitch_m / 2.0,
            (board_cfg["tags_y"] - 1) * tag_pitch_m / 2.0,
            0.0,
        ],
        dtype=np.float64,
    ).reshape(3, 1)

    if translation_ref == "tag0_center":
        translation_B = translation_input_B
        _vprint("  - 直接使用translation作为Tag0位置 (translation_reference=tag0_center)")
    else:
        if ref_point_cfg is None:
            ref_point_T = default_grid_center_T
            _vprint("  未配置 translation_reference_point_in_T_m，回退使用默认网格中心(ref=grid_center)")
        else:
            ref_point_T = np.array(ref_point_cfg, dtype=np.float64)
            if ref_point_T.size != 3:
                raise ValueError(
                    "translation_reference_point_in_T_m 期望长度为3的[x,y,z]（单位米）"
                )
            ref_point_T = ref_point_T.reshape(3, 1)

        # ref_B = R_B_T * ref_T + origin_B
        # origin_B(Tag0) = ref_B - R_B_T * ref_T
        translation_B = translation_input_B - (R_B_T @ ref_point_T)
        _vprint(
            f"  - translation_reference={translation_ref}: 使用translation_reference_point_in_T_m换算为Tag0位置"
        )
        _vprint(f"    reference_point_T(m)      = {_fmt4(ref_point_T)}")
        _vprint(f"    default_grid_center_T(m)  = {_fmt4(default_grid_center_T)}")

    B_T_T = _make_transform(R_B_T, translation_B, "B_T_T")
    _pretty_mat("B_T_T", B_T_T)
    return B_T_T


def compute_B_T_Cl(B_T_T: np.ndarray, Cl_T_T: np.ndarray) -> np.ndarray:
    """主链：B_T_Cl = B_T_T @ inv(Cl_T_T)。"""
    _vprint("\n[求解] 计算左相机到底盘变换 (B_T_Cl: Cl -> B)")
    T_T_Cl = _invert_transform(Cl_T_T, "T_T_Cl")
    _pretty_mat("Cl_T_T", Cl_T_T)
    _pretty_mat("T_T_Cl", T_T_Cl)
    _pretty_mat("B_T_T", B_T_T)
    B_T_Cl = B_T_T @ T_T_Cl
    _ensure_transform(B_T_Cl, "B_T_Cl")
    _pretty_mat("B_T_Cl", B_T_Cl)
    return B_T_Cl


def compute_B_T_Cr(B_T_Cl: np.ndarray, Cr_T_Cl: np.ndarray) -> np.ndarray:
    """右相机：B_T_Cr = B_T_Cl @ inv(Cr_T_Cl)。"""
    _vprint("\n[求解] 计算右相机到底盘变换 (B_T_Cr: Cr -> B)")
    Cl_T_Cr = _invert_transform(Cr_T_Cl, "Cl_T_Cr")
    B_T_Cr = B_T_Cl @ Cl_T_Cr
    _ensure_transform(B_T_Cr, "B_T_Cr")
    _pretty_mat("B_T_Cr", B_T_Cr)
    return B_T_Cr


def validate_stereo_consistency(
    *,
    Cl_T_T: np.ndarray,
    Cr_T_T: np.ndarray,
    Cr_T_Cl: np.ndarray,
    obj_points: np.ndarray,
    tag_ids: List[int],
    board_cfg: Dict[str, Any],
    K_l: Optional[np.ndarray] = None,
    dist_l: Optional[np.ndarray] = None,
    K_r: Optional[np.ndarray] = None,
    dist_r: Optional[np.ndarray] = None,
    strict: bool = True,
) -> bool:
    """校验双目外参与左右相机位姿的一致性。"""
    _vprint("\n" + "-" * 50)
    _vprint("[双目外参一致性校验]")

    # 计算预测的右相机位姿：Cr_T_T_pred = Cr_T_Cl @ Cl_T_T
    Cr_T_T_pred = Cr_T_Cl @ Cl_T_T
    Cl_T_Cr = _invert_transform(Cr_T_Cl, "Cl_T_Cr", strict=False)

    # 计算位姿差异
    Delta_T = Cr_T_T @ _invert_transform(Cr_T_T_pred, "Cr_T_T_pred_inv")

    # 旋转误差（度）
    R_error = Delta_T[:3, :3]
    rot_error_deg = float(Rotation.from_matrix(R_error).magnitude() * 180.0 / np.pi)
    # 平移误差（米）
    trans_error_m = float(np.linalg.norm(Delta_T[:3, 3]))

    # 非 verbose 时也给一个简短结论，方便用户快速判断质量
    print(f"双目一致性: 旋转误差 {rot_error_deg:.3f}°，平移误差 {trans_error_m:.4f}m")
    _vprint(f"位姿误差分析:")
    _vprint(f"  旋转误差: {rot_error_deg:.3f}°")
    _vprint(f"  平移误差: {trans_error_m:.4f}m")

    # 几何验证：选择几个显眼Tag进行3D点验证
    tag_centers_T = {
        int(tag_ids[i]): obj_points[i].mean(axis=0).reshape(3)
        for i in range(len(tag_ids))
    }
    picked_ids = _get_prominent_tag_ids(
        board_cfg["tags_x"], board_cfg["tags_y"], tag_centers_T
    )

    _vprint(f"\n几何一致性验证（{len(picked_ids)}个关键Tag）:")
    max_point_error = 0.0
    max_pixel_error_l = 0.0
    max_pixel_error_r = 0.0
    pixel_error_available = (
        K_l is not None
        and K_r is not None
        and dist_l is not None
        and dist_r is not None
    )

    for tid in picked_ids[:3]:  # 只检查前3个关键点
        p_T = tag_centers_T[tid]

        # 右相机中的3D点（两种方式计算）
        p_Cr_measured = _tf_point(Cr_T_T, p_T)
        p_Cr_predicted = _tf_point(Cr_T_T_pred, p_T)

        # 3D点误差
        point_error = np.linalg.norm(p_Cr_measured - p_Cr_predicted)
        max_point_error = max(max_point_error, point_error)

        _vprint(f"  Tag{tid}: 3D点误差 {point_error:.4f}m", end="")

        # 像素重投影误差（如果有相机内参）
        if pixel_error_available:
            # 左相机像素误差
            p_Cl_measured = _tf_point(Cl_T_T, p_T)
            p_Cl_predicted = _tf_point(
                Cl_T_Cr @ Cr_T_T_pred, p_T
            )  # 通过右相机预测的左相机位置

            if p_Cl_measured[2] > 0.1 and p_Cl_predicted[2] > 0.1:  # Z > 0.1m 才有效
                # 重投影到左相机图像平面
                if K_l is not None and dist_l is not None:
                    pixel_measured_l, _ = cv2.projectPoints(
                        p_Cl_measured.reshape(1, 1, 3),
                        np.zeros(3),
                        np.zeros(3),
                        K_l,
                        dist_l,
                    )
                    pixel_predicted_l, _ = cv2.projectPoints(
                        p_Cl_predicted.reshape(1, 1, 3),
                        np.zeros(3),
                        np.zeros(3),
                        K_l,
                        dist_l,
                    )
                else:
                    pixel_error_l = float("nan")
                    continue

                pixel_error_l = np.linalg.norm(
                    pixel_measured_l[0, 0] - pixel_predicted_l[0, 0]
                )
                max_pixel_error_l = max(max_pixel_error_l, pixel_error_l)
            else:
                pixel_error_l = float("nan")

            # 右相机像素误差
            if p_Cr_measured[2] > 0.1 and p_Cr_predicted[2] > 0.1:  # Z > 0.1m 才有效
                # 重投影到右相机图像平面
                if K_r is not None and dist_r is not None:
                    pixel_measured_r, _ = cv2.projectPoints(
                        p_Cr_measured.reshape(1, 1, 3),
                        np.zeros(3),
                        np.zeros(3),
                        K_r,
                        dist_r,
                    )
                    pixel_predicted_r, _ = cv2.projectPoints(
                        p_Cr_predicted.reshape(1, 1, 3),
                        np.zeros(3),
                        np.zeros(3),
                        K_r,
                        dist_r,
                    )
                else:
                    pixel_error_r = float("nan")
                    continue

                pixel_error_r = np.linalg.norm(
                    pixel_measured_r[0, 0] - pixel_predicted_r[0, 0]
                )
                max_pixel_error_r = max(max_pixel_error_r, pixel_error_r)
            else:
                pixel_error_r = float("nan")

            _vprint(f", 左像素误差 {pixel_error_l:.2f}px, 右像素误差 {pixel_error_r:.2f}px")
        else:
            _vprint()  # 换行

    _vprint(f"  最大3D点误差: {max_point_error:.4f}m")
    if pixel_error_available:
        _vprint(f"  最大左像素误差: {max_pixel_error_l:.2f}px")
        _vprint(f"  最大右像素误差: {max_pixel_error_r:.2f}px")

    # 判断是否通过校验
    rot_threshold = 3.0  # 度
    trans_threshold = 0.05  # 米
    point_threshold = 0.05  # 米
    pixel_threshold = 5.0  # 像素

    passed = bool(
        rot_error_deg < rot_threshold
        and trans_error_m < trans_threshold
        and max_point_error < point_threshold
    )

    # 如果有像素误差数据，也要满足像素阈值
    if (
        pixel_error_available
        and not np.isnan(max_pixel_error_l)
        and not np.isnan(max_pixel_error_r)
    ):
        pixel_passed = bool(
            max_pixel_error_l < pixel_threshold and max_pixel_error_r < pixel_threshold
        )
        passed = passed and pixel_passed
        _vprint(
            f"\n校验阈值: 旋转<{rot_threshold}°, 平移<{trans_threshold}m, 3D点<{point_threshold}m, 像素<{pixel_threshold}px"
        )
    else:
        _vprint(f"\n校验阈值: 旋转<{rot_threshold}°, 平移<{trans_threshold}m, 3D点<{point_threshold}m")

    if passed:
        print("双目外参一致性校验通过")
    else:
        print("警告: 双目外参一致性校验失败")
        _vprint("  未通过的指标:")
        if rot_error_deg >= rot_threshold:
            _vprint(f"    - 旋转误差: {rot_error_deg:.3f}° (阈值: {rot_threshold}°)")
        if trans_error_m >= trans_threshold:
            _vprint(f"    - 平移误差: {trans_error_m:.4f}m (阈值: {trans_threshold}m)")
        if max_point_error >= point_threshold:
            _vprint(f"    - 3D点误差: {max_point_error:.4f}m (阈值: {point_threshold}m)")
        if (
            pixel_error_available
            and not np.isnan(max_pixel_error_l)
            and not np.isnan(max_pixel_error_r)
        ):
            if max_pixel_error_l >= pixel_threshold:
                _vprint(f"    - 左像素误差: {max_pixel_error_l:.2f}px (阈值: {pixel_threshold}px)")
            if max_pixel_error_r >= pixel_threshold:
                _vprint(f"    - 右像素误差: {max_pixel_error_r:.2f}px (阈值: {pixel_threshold}px)")

    _vprint("-" * 50)
    return passed


def print_tag_coordinates(
    *,
    B_T_T: np.ndarray,
    Cl_T_T: np.ndarray,
    B_T_Cl: np.ndarray,
    obj_points: np.ndarray,
    tag_ids: List[int],
    board_cfg: Dict[str, Any],
) -> None:
    """打印显眼Tag中心点的链式坐标变换，每一步都可验证。"""
    if not VERBOSE:
        return
    print("\n" + "-" * 60)
    print("[显眼Tag中心点坐标 - 链式变换验证]")

    # 计算tag中心点
    tag_centers_T = {
        int(tag_ids[i]): obj_points[i].mean(axis=0).reshape(3)
        for i in range(len(tag_ids))
    }
    picked_ids = _get_prominent_tag_ids(
        board_cfg["tags_x"], board_cfg["tags_y"], tag_centers_T
    )

    print("变换链: T -> B -> Cl")
    print("验证: B_T_Cl @ Cl_T_T 应等于 B_T_T")
    print()

    for tid in picked_ids:
        p_T = tag_centers_T[tid]

        # 直接变换
        p_B_direct = _tf_point(B_T_T, p_T)
        p_Cl_direct = _tf_point(Cl_T_T, p_T)

        # 链式变换: T -> Cl -> B
        p_Cl_from_T = _tf_point(Cl_T_T, p_T)
        p_B_from_Cl = _tf_point(B_T_Cl, p_Cl_from_T)

        # 验证链式一致性
        chain_error = np.linalg.norm(p_B_direct - p_B_from_Cl)

        print(f"Tag {tid:>2d}:")
        print(f"  T坐标:     {_fmt4(p_T)}")
        print(f"  B坐标(直接): {_fmt4(p_B_direct)}")
        print(f"  B坐标(链式): {_fmt4(p_B_from_Cl)}")
        print(f"  Cl坐标:    {_fmt4(p_Cl_direct)}")
        print(f"  链式误差:   {chain_error:.6f}m")

        if chain_error > 1e-10:
            print("    警告: 链式验证失败")
        else:
            print("    链式验证通过")
        print()

    # 整体变换矩阵验证
    print("变换矩阵链式验证:")
    B_T_T_computed = B_T_Cl @ Cl_T_T
    matrix_diff = np.linalg.norm(B_T_T - B_T_T_computed, ord="fro")
    print(f"  ||B_T_T - (B_T_Cl @ Cl_T_T)||_F = {matrix_diff:.2e}")

    if matrix_diff < 1e-10:
        print("  变换矩阵链式验证通过")
    else:
        print("  警告: 变换矩阵链式验证失败")

    print("-" * 60)


def process_step5_images(
    calibration_data: Dict[str, Any],
    *,
    image_root: Path,
    cameras: List[str],
    config_path: str,
    max_images: Optional[int] = None,
    min_tags: int,
    scan_limits: ScanLimits,
    scan_order: ScanOrder,
    workers: int,
    prefetch: int,
    cache_cfg: CacheConfig,
    prefilter_cfg: PrefilterConfig,
) -> Dict[str, Any]:
    """处理 Step5 图像并估计每个相机的 C_T_T（T->C）。"""
    _vprint("\n处理 step5 图像...")

    config = load_config(config_path)
    ds = get_step5_dataset(config)
    use_ds = bool(ds.get("enabled", False))

    C_T_T_by_cam: Dict[str, np.ndarray] = {}
    pose_stats: Dict[str, Dict[str, Any]] = {}
    scan_reports: Dict[str, Any] = {}

    for cam in cameras:
        expected_desc = ""
        if use_ds:
            images_p = get_step5_camera_images(config, cam)
            images = [str(p) for p in images_p]
            expected_desc = "config.step5_dataset 指定的 raw_dir/raw_glob 或 image_root/<cam>"
        else:
            cam_dir = image_root / cam
            images = _glob_images(cam_dir)
            expected_desc = f"{cam_dir}/*.png|jpg|jpeg|bmp"
        if max_images is not None and max_images > 0:
            images = images[: int(max_images)]

        if len(images) == 0:
            print(f"  - {cam}: 未找到图像，跳过（期望 {expected_desc}）")
            continue

        intr_path = _intrinsics_path_for_camera(cam)
        if not os.path.exists(intr_path):
            print(f"  - {cam}: 缺少内参 {intr_path}，跳过")
            continue

        K, dist = load_intrinsics(intr_path)

        _vprint(f"\n处理 {cam}: {len(images)} 张图像...")
        poses, scan_report = _scan_step5_poses_for_camera(
            image_paths=images,
            config=config,
            cam=str(cam),
            K=K,
            dist=dist,
            opencv_board=calibration_data.get("opencv_board"),
            obj_points=calibration_data["obj_points"],
            tag_ids=calibration_data["tag_ids"],
            use_multiscale=bool(calibration_data.get("use_multiscale", True)),
            opencv_refine=bool(calibration_data.get("opencv_refine", False)),
            min_tags=int(min_tags),
            scan_limits=scan_limits,
            scan_order=scan_order,
            workers=int(workers),
            prefetch=int(prefetch),
            cache_cfg=cache_cfg,
            prefilter_cfg=prefilter_cfg,
        )

        scan_reports[str(cam)] = scan_report

        pose_stats[cam] = {
            "total_images": int(len(images)),
            "valid_poses": int(len(poses)),
            "scan": scan_report.get("scan"),
            "detector_calls": scan_report.get("detector_calls"),
            "pnp": scan_report.get("pnp"),
        }

        print(f"  - {cam}: 有效位姿: {len(poses)}/{len(images)}")

        if len(poses) < 1:
            print(f"  - {cam}: 有效位姿过少，跳过")
            continue

        rvecs = [p[0] for p in poses]
        tvecs = [p[1] for p in poses]
        C_T_T_by_cam[cam] = compute_C_T_T_mean_pose(rvecs, tvecs, cam_name=cam)

    if len(C_T_T_by_cam) == 0:
        raise ValueError(
            "没有任何相机得到有效位姿。\n"
            "请检查：Step5 图像路径/清晰度、内参 results/<cam>_intrinsics.json、以及 AprilTag 检测参数。"
        )

    return {
        "C_T_T_by_cam": C_T_T_by_cam,
        "pose_stats": pose_stats,
        "scan_reports": scan_reports,
    }


def compute_B_T_C(B_T_T: np.ndarray, C_T_T: np.ndarray, *, cam: str) -> np.ndarray:
    """主链：B_T_C = B_T_T @ inv(C_T_T)。"""
    T_T_C = _invert_transform(C_T_T, f"T_T_{cam}")
    B_T_C = B_T_T @ T_T_C
    _ensure_transform(B_T_C, f"B_T_{cam}")
    return B_T_C


def compute_camera_to_base_transforms(
    calibration_data: Dict[str, Any], pose_data: Dict[str, Any]
) -> Dict[str, Any]:
    """计算每个相机到 Base 的外参 B_T_C。"""
    _vprint("\n计算相机到底盘变换矩阵...")

    B_T_T = build_B_T_T_from_config(
        calibration_data["transform_cfg"], calibration_data["board_cfg"]
    )

    C_T_T_by_cam: Dict[str, np.ndarray] = pose_data["C_T_T_by_cam"]
    B_T_C: Dict[str, np.ndarray] = {}
    methods: Dict[str, str] = {}

    # 先对有 Step5 图像的相机做直接求解
    for cam, C_T_T in C_T_T_by_cam.items():
        B_T_C[cam] = compute_B_T_C(B_T_T, C_T_T, cam=cam)
        methods[cam] = "direct_pnp"

    # 再用 Step4 外参把 B_T_C 传播到其它相机（如果需要）
    graph = load_extrinsics_graph(results_dir=Path("results"))
    propagated: List[str] = []

    if graph is not None and len(graph.T_cam_from_ref) > 0 and len(B_T_C) > 0:
        anchor_cam = next(iter(B_T_C.keys()))
        try:
            propagated_all = propagate_B_T_C(
                B_T_C_anchor=B_T_C[anchor_cam],
                anchor_cam=anchor_cam,
                graph=graph,
            )
        except ValueError as e:
            _vprint(f"  [提示] Step4 外参传播跳过：{e}")
            propagated_all = {}

        for cam, T in propagated_all.items():
            if cam in B_T_C:
                continue
            B_T_C[cam] = T
            methods[cam] = "propagated_from_step4"
            propagated.append(cam)

    return {
        "B_T_C": B_T_C,
        "methods": methods,
        "propagation": {
            "used": bool(len(propagated) > 0),
            "source": (graph.source if graph is not None else None),
            "reference": (graph.reference if graph is not None else None),
            "propagated_cameras": propagated,
        },
    }


def save_calibration_results(
    calibration_data: Dict[str, Any],
    pose_data: Dict[str, Any],
    transform_data: Dict[str, Any],
    *,
    image_root: Path,
):
    """保存标定结果到 JSON 文件。"""
    B_T_C: Dict[str, np.ndarray] = transform_data["B_T_C"]
    methods: Dict[str, str] = transform_data["methods"]

    pose_stats: Dict[str, Dict[str, Any]] = pose_data.get("pose_stats", {})
    per_cam_stats: Dict[str, Dict[str, Any]] = {}
    for cam in sorted(set(list(pose_stats.keys()) + list(B_T_C.keys()))):
        s = dict(pose_stats.get(cam, {}))
        s["method"] = methods.get(cam)
        per_cam_stats[cam] = s

    result: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "image_root": str(image_root.as_posix()),
        "B_T_C": {cam: T.tolist() for cam, T in B_T_C.items()},
        "pose_stats": per_cam_stats,
        "propagation": transform_data.get("propagation", {}),
        "config_used": {
            "board_to_base_transform": calibration_data["transform_cfg"],
            "apriltag_board": calibration_data["board_cfg"],
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/camera_to_base.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

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

    print(f"\n结果已保存到 results/camera_to_base.json")
    print(f"  相机数: {len(B_T_C)}")


def main():
    """主函数 - 封装后的清晰流程。"""
    parser = argparse.ArgumentParser(
        description="Step 5b: AprilTag 相机到机器人底盘外参标定（支持多相机）"
    )
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
    args = parser.parse_args()

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
                cameras = _discover_cameras(image_root)

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
        )

        # 3) 计算相机到底盘的变换矩阵 B_T_C
        transform_data = compute_camera_to_base_transforms(calibration_data, pose_data)

        # 4) 保存标定结果
        save_calibration_results(
            calibration_data, pose_data, transform_data, image_root=image_root
        )

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
