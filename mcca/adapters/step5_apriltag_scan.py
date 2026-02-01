"""Step5b（AprilTag PnP）图像扫描与位姿聚合（adapters 层）。

本模块用于把 Step5b 入口脚本中“IO/并行/缓存/扫描预算/统计聚合”的实现下沉到 adapters，
让 `mcca.entry.step5b_camera_to_base` 更专注于：
- CLI 参数解析
- 组装依赖
- 调用 core 求解
- 结果落盘

设计约束：
- adapters -> core：允许依赖 `mcca.core.*` 的纯逻辑/数据结构。
- 不依赖 entry（不得 import `mcca.entry.*`）。
- 注释与报错信息使用中文，便于仓库使用者定位问题。

注意：
- 该模块包含多进程 worker 的全局状态（为减少进程间重复初始化开销）。
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from mcca.adapters.apriltag_perf.cache import CacheConfig
from mcca.adapters.apriltag_perf.prefilter import PrefilterConfig
from mcca.adapters.apriltag_perf.scan import (
    ScanLimits,
    ScanOrder,
    iter_scan_parallel_ordered,
    iter_scan_sequential,
)
from mcca.adapters.apriltag_perf.service import CachedAprilTagDetector
from mcca.core.config import load_config
from mcca.core.datasets import (
    get_step5_camera_images,
    get_step5_dataset,
)
from mcca.core.detection import (
    create_detector_params,
    get_detection_auto_roi,
    get_detection_profile,
    get_detection_roi,
    get_detection_settings,
)
from mcca.core.pose import estimate_pose_apriltag
from mcca.core.step5_camera_to_base import mean_C_T_T_from_pnp
from mcca.core.board import create_opencv_aruco_board, get_aruco_dict


def _pretty_mat(name: str, T: np.ndarray, *, indent: str = "  ") -> str:
    """用 4 位小数格式化矩阵，便于 verbose 输出阅读。"""

    T = np.asarray(T, dtype=np.float64)
    s = np.array2string(
        T,
        formatter={"float_kind": lambda v: f"{float(v): .4f}"},
        suppress_small=False,
    )
    return f"{indent}{name} =\n{indent}{s.replace(chr(10), chr(10) + indent)}"


# region Apriltag 检测 worker（并行扫描使用的全局状态）

_G_DET: Optional[CachedAprilTagDetector] = None
_G_OBJ_POINTS: Optional[np.ndarray] = None
_G_TAG_IDS: Optional[List[int]] = None
_G_K: Optional[np.ndarray] = None
_G_DIST: Optional[np.ndarray] = None
_G_MIN_TAGS: int = 1


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


def _init_step5_apriltag_pose_worker(state: Dict[str, Any]) -> None:
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

    # 注意（Windows 多进程）：cv2.aruco.Board 不是可 pickle 对象，
    # 不能作为 state 直接传给子进程。这里统一在 worker 进程内重建 Board。
    board = None
    if bool(state.get("opencv_refine", False)):
        try:
            obj_points_mm = np.asarray(state.get("board_obj_points_mm"), dtype=np.float32)
            board = create_opencv_aruco_board(obj_points_mm, [int(x) for x in _G_TAG_IDS], aruco_dict)
        except Exception:
            board = None

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


def _step5_apriltag_pose_worker(image_path: str) -> Dict[str, Any]:
    """多进程扫描的单任务：检测并尝试 PnP，返回 rvec/tvec（若成功）。"""

    global _G_DET, _G_OBJ_POINTS, _G_TAG_IDS, _G_K, _G_DIST, _G_MIN_TAGS

    if (
        _G_DET is None
        or _G_OBJ_POINTS is None
        or _G_TAG_IDS is None
        or _G_K is None
        or _G_DIST is None
    ):
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


# endregion


def _glob_images(cam_dir: Path) -> List[str]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(str(cam_dir / pat)))
    return sorted(files)


def discover_step5_cameras(image_root: Path) -> List[str]:
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


def load_intrinsics(json_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """加载相机内参（K/dist）。"""

    p = Path(str(json_path))
    data = json.loads(p.read_text(encoding="utf-8"))
    return np.array(data["camera_matrix"], dtype=np.float64), np.array(data["dist_coeffs"], dtype=np.float64)


def intrinsics_path_for_camera(cam: str, *, results_dir: Path = Path("results")) -> Path:
    """获取某相机的内参路径（统一为 results/<cam>_intrinsics.json）。"""

    return Path(results_dir) / f"{cam}_intrinsics.json"


def _scan_step5_apriltag_poses_for_camera(
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
        # 说明：不要在 state 里直接塞 OpenCV Board 对象（Windows 多进程无法 pickle）。
        # 改为传递可序列化的数据，让子进程在 init 时重建 Board。
        "board_obj_points_mm": (np.asarray(obj_points, dtype=np.float64) * 1000.0).astype(np.float32).tolist(),
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
        _init_step5_apriltag_pose_worker(state)
        results, counters = iter_scan_sequential(
            image_paths,
            worker_fn=_step5_apriltag_pose_worker,
            is_valid_fn=_is_valid,
            limits=scan_limits,
            order=scan_order,
        )
    else:
        results, counters = iter_scan_parallel_ordered(
            image_paths,
            worker_fn=_step5_apriltag_pose_worker,
            is_valid_fn=_is_valid,
            limits=scan_limits,
            order=scan_order,
            max_workers=int(workers),
            prefetch=int(prefetch),
            initializer=_init_step5_apriltag_pose_worker,
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
    verbose: bool = False,
) -> Dict[str, Any]:
    """处理 Step5 图像并估计每个相机的 C_T_T（T->C）。

    Returns:
        {
          "C_T_T_by_cam": {cam: 4x4},
          "pose_stats": {cam: {...}},
          "scan_reports": {cam: {...}},
        }
    """

    def _vprint(*args, **kwargs) -> None:
        if verbose:
            print(*args, **kwargs)

    _vprint("\n处理 step5 图像...")

    config = load_config(config_path)
    ds = get_step5_dataset(config)
    use_ds = bool(ds.get("enabled", False))

    # detection overrides（这些开关由 entry 统一写入 calibration_data）
    use_multiscale, opencv_refine = get_detection_settings(config)
    if "use_multiscale" in calibration_data:
        use_multiscale = bool(calibration_data["use_multiscale"])
    if "opencv_refine" in calibration_data:
        opencv_refine = bool(calibration_data["opencv_refine"])

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

        intr_path = intrinsics_path_for_camera(cam)
        if not intr_path.exists():
            print(f"  - {cam}: 缺少内参 {intr_path}，跳过")
            continue

        K, dist = load_intrinsics(intr_path)

        _vprint(f"\n处理 {cam}: {len(images)} 张图像...")
        poses, scan_report = _scan_step5_apriltag_poses_for_camera(
            image_paths=images,
            config=config,
            cam=str(cam),
            K=K,
            dist=dist,
            opencv_board=calibration_data.get("opencv_board"),
            obj_points=calibration_data["obj_points"],
            tag_ids=calibration_data["tag_ids"],
            use_multiscale=bool(use_multiscale),
            opencv_refine=bool(opencv_refine),
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
        C_T_T, stats = mean_C_T_T_from_pnp(rvecs=rvecs, tvecs=tvecs)

        if verbose and len(poses) > 1:
            _vprint(f"  [稳定性检查] 平移标准差: {stats.t_std_norm_m*1000:.2f} mm (越小越好)")
            _vprint(f"  [稳定性检查] 旋转标准差: {stats.r_std_norm_rad:.4f} rad")
            if stats.t_std_norm_m > 0.005:
                _vprint("  警告: 位姿抖动较大 (>5mm)，建议增加光照或检查标定板是否晃动")
            else:
                _vprint("  位姿稳定，单位置标定可靠")

        if verbose:
            _vprint(_pretty_mat(f"{cam}_T_T", C_T_T))

        C_T_T_by_cam[cam] = C_T_T

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
