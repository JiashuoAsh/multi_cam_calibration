#!/usr/bin/env python3
"""Step3 入口：AprilTag 内参标定（多相机）。

职责边界：
- 入口层负责：参数解析、读取配置、扫描图片、并行检测（含缓存/预筛选/早停）、落盘输出与可视化。
- 纯数学/统计逻辑在 `mcca.core.step3_intrinsic`：去重/标定/误差计算。

输入：
- 优先：images/filtered/<cam>/*.(png|jpg|jpeg|bmp)
- 回退：images/raw/<cam>/*.(png|jpg|jpeg|bmp)
- 配置：config/apriltag_config.json

输出：
- results/<cam>_intrinsics.json
- results/step3_intrinsic_report.json
- 可视化（可选）：results/visualization/step3_intrinsic_<cam>/*
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from mcca.core.board import (
    create_apriltag_board,
    create_opencv_aruco_board,
    get_aruco_dict,
)
from mcca.core.config import load_config
from mcca.core.datasets import (
    get_camera_filtered_dir,
    get_camera_filtered_images,
    get_camera_raw_images,
    get_dataset_cameras,
)
from mcca.core.detection import (
    create_detector_params,
    get_detection_auto_roi,
    get_detection_profile,
    get_detection_roi,
    get_detection_settings,
)
from mcca.core.step3_intrinsic import (
    Step3DetectResult,
    calibrate_camera_intrinsics,
    count_total_points,
    dedup_views,
    summarize_step3_results,
)


# 最大有效图像数量（用于内参标定）
# 说明：OpenCV 的 calibrateCamera 会为每张图像引入 6 个外参变量（rvec/tvec），
# 当 view 数过大（例如 300+）时，优化问题会变得非常大且可能“看起来卡死”。
# 实践中内参标定通常 30~120 张高质量图像就足够。
MAX_VALID_IMAGES = 120


# 默认尽量安静：只输出关键结果；需要逐张处理细节用 --verbose。
VERBOSE: bool = True


def _vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


def _safe_imwrite(path: str, img: np.ndarray) -> bool:
    """兼容中文路径的写图：优先 cv2.imwrite，失败则使用 imencode + Python 写文件。"""

    try:
        if cv2.imwrite(path, img):
            return True
    except Exception:
        pass

    ext = os.path.splitext(path)[1] or ".jpg"  # 需要带点，例如 ".jpg"
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            print(f"  警告: imencode 失败，无法写入: {path}")
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:
        print(f"  警告: 写入失败: {path} ({e})")
        return False


def _build_algo_key(config: Dict[str, Any], *, profile: str, camera: str) -> Dict[str, Any]:
    """构建用于缓存的 algo_key（必须可 JSON 序列化）。"""

    det_cfg = (config or {}).get("calibration_settings", {}).get("detection", {})
    board_cfg = (config or {}).get("apriltag_board", {})
    corner_ref = (config or {}).get("calibration_settings", {}).get("corner_refinement")

    return {
        "opencv_version": str(getattr(cv2, "__version__", "unknown")),
        "apriltag_family": str(board_cfg.get("family")),
        "corner_refinement": str(corner_ref),
        "profile": str(profile),
        "camera": str(camera),
        "detection": det_cfg,
        "stage": "step3_intrinsic",
    }


def _make_detector_for_camera(
    *,
    config: Dict[str, Any],
    profile: str,
    camera: str,
    roi: Optional[Tuple[int, int, int, int]],
    cache_cfg: CacheConfig,
    prefilter_cfg: PrefilterConfig,
) -> CachedAprilTagDetector:
    """在主进程创建 detector：用于从 cache 取 corners/ids 并生成可视化。"""

    board_cfg = config["apriltag_board"]
    aruco_dict = get_aruco_dict(board_cfg["family"])
    obj_points_mm, tag_ids = create_apriltag_board(config)
    board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)
    detector_params = create_detector_params(config)

    use_multiscale, opencv_refine = get_detection_settings(config)
    auto_roi_cfg = get_detection_auto_roi(config)

    return CachedAprilTagDetector(
        aruco_dict=aruco_dict,
        detector_params=detector_params,
        algo_key=_build_algo_key(config, profile=profile, camera=camera),
        use_multiscale=bool(use_multiscale),
        opencv_refine=bool(opencv_refine),
        board=board,
        roi=roi,
        auto_roi=bool(auto_roi_cfg.get("enabled", False)),
        auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
        auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
        auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
        cache_cfg=cache_cfg,
        prefilter_cfg=prefilter_cfg,
    )


_G_STEP3_DET: Optional[CachedAprilTagDetector] = None
_G_STEP3_MIN_TAGS: int = 0
_G_STEP3_TAG_SET: Optional[set[int]] = None


def _init_step3_worker(state: Dict[str, Any]) -> None:
    """多进程 worker 初始化：在子进程内创建检测器（Windows spawn 安全）。"""

    global _G_STEP3_DET, _G_STEP3_MIN_TAGS, _G_STEP3_TAG_SET

    config = state["config"]
    profile = str(state["profile"])
    camera = str(state["camera"])
    family = str(state["family"])

    aruco_dict = get_aruco_dict(family)
    obj_points_mm = np.asarray(state["obj_points_mm"], dtype=np.float32)
    tag_ids = [int(x) for x in state["tag_ids"]]
    board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)
    detector_params = create_detector_params(config)

    roi = tuple(state["roi"]) if state.get("roi") is not None else None
    auto_roi_cfg = state.get("auto_roi_cfg") or {}

    algo_key = _build_algo_key(config, profile=profile, camera=camera)
    cache_cfg = CacheConfig(
        enabled=bool(state["cache"]["enabled"]),
        cache_dir=str(state["cache"]["cache_dir"]),
        force_redetect=bool(state["cache"]["force_redetect"]),
    )
    prefilter_cfg = PrefilterConfig(enabled=bool(state["prefilter"]["enabled"]))

    _G_STEP3_DET = CachedAprilTagDetector(
        aruco_dict=aruco_dict,
        detector_params=detector_params,
        algo_key=algo_key,
        use_multiscale=bool(state["use_multiscale"]),
        opencv_refine=bool(state["opencv_refine"]),
        board=board,
        roi=roi,
        auto_roi=bool(auto_roi_cfg.get("enabled", False)),
        auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
        auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
        auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
        cache_cfg=cache_cfg,
        prefilter_cfg=prefilter_cfg,
    )
    _G_STEP3_MIN_TAGS = int(state["min_tags"])
    _G_STEP3_TAG_SET = set(int(x) for x in tag_ids)


def _step3_worker(image_path: str) -> Step3DetectResult:
    global _G_STEP3_DET, _G_STEP3_MIN_TAGS, _G_STEP3_TAG_SET
    if _G_STEP3_DET is None or _G_STEP3_TAG_SET is None:
        return Step3DetectResult(
            image_path=str(image_path),
            valid=False,
            n_in_board=0,
            status=1,
            from_cache=False,
            elapsed_ms=0.0,
        )

    res = _G_STEP3_DET.detect_path(Path(str(image_path)))
    ids = np.asarray(res.ids) if res.ids is not None else np.zeros((0, 1), dtype=np.int32)
    ids_flat = [int(x) for x in ids.reshape(-1).tolist()] if ids.size > 0 else []
    n_in_board = int(sum(1 for tid in ids_flat if int(tid) in _G_STEP3_TAG_SET))
    valid = bool(int(res.status) == 0 and n_in_board >= int(_G_STEP3_MIN_TAGS))

    return Step3DetectResult(
        image_path=str(image_path),
        valid=valid,
        n_in_board=n_in_board,
        status=int(res.status),
        from_cache=bool(res.from_cache),
        elapsed_ms=float(res.elapsed_ms),
    )


def clean_visualization_dirs() -> None:
    """清空 step3 的可视化输出目录。"""

    base = Path("results/visualization")
    if not base.exists():
        return

    # 兼容旧输出：step3_intrinsic_左 / step3_intrinsic_右
    for p in sorted(base.glob("step3_intrinsic_*")):
        if p.is_dir():
            shutil.rmtree(str(p))
            _vprint(f"  已清空: {p.as_posix()}")


@dataclass(frozen=True)
class _Step3CalibrationReport:
    """单相机的标定运行时报告（用于 pipeline 汇总）。"""

    limits: Dict[str, Any]
    order: Dict[str, Any]
    workers: int
    prefetch: int
    submitted: int
    completed: int
    valid: int
    elapsed_s: float
    perf: Dict[str, Any]


def calibrate_camera_apriltag(
    image_files: Sequence[str],
    obj_points_all: np.ndarray,
    tag_ids: Sequence[int],
    aruco_dict: cv2.aruco.Dictionary,
    camera_name: str,
    *,
    config: Dict[str, Any],
    profile: str,
    use_multiscale: bool,
    opencv_refine: bool,
    board: cv2.aruco.Board,
    roi: Optional[Tuple[int, int, int, int]] = None,
    auto_roi_cfg: Optional[Dict[str, Any]] = None,
    save_visualization: bool = True,
    max_valid_images: int = MAX_VALID_IMAGES,
    min_tags: int = 1,
    scan_limits: Optional[ScanLimits] = None,
    scan_order: Optional[ScanOrder] = None,
    workers: int = 1,
    prefetch: int = 0,
    cache_cfg: Optional[CacheConfig] = None,
    prefilter_cfg: Optional[PrefilterConfig] = None,
    dedup: bool = False,
    dedup_bins: int = 80,
) -> Tuple[
    bool,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
    Optional[float],
    Optional[Tuple[int, int]],
    Dict[str, Any],
]:
    """使用 AprilTag 标定单个相机的内参。"""

    print(f"\n{camera_name}相机标定:")
    print(f"  - 图像数量: {len(image_files)}")
    print(f"  - 最大有效图像: {max_valid_images}")
    if int(min_tags) != 5:
        print(f"  - min_tags: {min_tags}")

    limits = scan_limits or ScanLimits(target_valid=int(max_valid_images))
    order = scan_order or ScanOrder(strategy="uniform", seed=0)
    workers_i = int(workers) if int(workers) > 0 else 1
    if workers_i <= 0:
        workers_i = 1

    cache_cfg = cache_cfg or CacheConfig()
    prefilter_cfg = prefilter_cfg or PrefilterConfig()

    vis_base: Optional[str] = None
    detection_dir: Optional[str] = None
    reprojection_dir: Optional[str] = None
    undistorted_dir: Optional[str] = None
    if save_visualization:
        vis_base = f"results/visualization/step3_intrinsic_{camera_name.lower()}"
        detection_dir = f"{vis_base}/detection"
        reprojection_dir = f"{vis_base}/reprojection"
        undistorted_dir = f"{vis_base}/undistorted"
        os.makedirs(detection_dir, exist_ok=True)
        os.makedirs(reprojection_dir, exist_ok=True)
        os.makedirs(undistorted_dir, exist_ok=True)
        _vprint(f"  - 可视化目录: {vis_base}")

    auto_roi_cfg = auto_roi_cfg or {}

    init_state = {
        "config": config,
        "profile": str(profile),
        "camera": str(camera_name),
        "family": str(config["apriltag_board"]["family"]),
        "use_multiscale": bool(use_multiscale),
        "opencv_refine": bool(opencv_refine),
        "obj_points_mm": np.asarray(obj_points_all, dtype=np.float32),
        "tag_ids": [int(x) for x in tag_ids],
        "roi": list(roi) if roi is not None else None,
        "auto_roi_cfg": auto_roi_cfg,
        "min_tags": int(min_tags),
        "cache": {
            "enabled": bool(cache_cfg.enabled),
            "cache_dir": str(cache_cfg.cache_dir),
            "force_redetect": bool(cache_cfg.force_redetect),
        },
        "prefilter": {"enabled": bool(prefilter_cfg.enabled)},
    }

    items = [str(p) for p in image_files]

    if workers_i > 1:
        out, counters = iter_scan_parallel_ordered(
            items,
            worker_fn=_step3_worker,
            is_valid_fn=lambda r: bool(r.valid),
            limits=limits,
            order=order,
            max_workers=workers_i,
            prefetch=int(prefetch),
            initializer=_init_step3_worker,
            initargs=(init_state,),
        )
    else:
        _init_step3_worker(init_state)
        out, counters = iter_scan_sequential(
            items,
            worker_fn=_step3_worker,
            is_valid_fn=lambda r: bool(r.valid),
            limits=limits,
            order=order,
        )

    scan_results = list(out)
    scan_summary = summarize_step3_results(scan_results)

    print(
        f"  扫描: submitted={counters.submitted} completed={counters.completed} valid={counters.valid} "
        f"(elapsed={counters.elapsed_s:.2f}s, strategy={order.strategy}, workers={workers_i})"
    )
    print(
        "  统计: "
        f"cache_hit={scan_summary['cache_hit']} prefilter_skipped={scan_summary['prefilter_skipped']} error={scan_summary['error']} "
        f"mean_detect_ms={scan_summary['mean_detect_ms']:.1f} p95_detect_ms={scan_summary['p95_detect_ms']:.1f}"
    )

    valid_candidates = [r.image_path for r in scan_results if bool(r.valid)]

    det_main = _make_detector_for_camera(
        config=config,
        profile=str(profile),
        camera=str(camera_name),
        roi=roi,
        cache_cfg=cache_cfg,
        prefilter_cfg=prefilter_cfg,
    )

    id_to_idx = {int(t): int(i) for i, t in enumerate(tag_ids)}
    all_obj_pts: List[np.ndarray] = []
    all_img_pts: List[np.ndarray] = []
    valid_images: List[str] = []
    image_size: Optional[Tuple[int, int]] = None

    n_read_fail = 0
    n_low_tags = 0

    for img_path in valid_candidates:
        img = cv2.imread(str(img_path))
        if img is None:
            n_read_fail += 1
            _vprint(f"  警告: 无法读取 {img_path}")
            continue

        if image_size is None:
            image_size = (int(img.shape[1]), int(img.shape[0]))

        det_res = det_main.detect_path(Path(str(img_path)))
        corners, ids = det_res.corners, det_res.ids

        if ids is None or corners is None or int(len(ids)) < int(min_tags):
            n_low_tags += 1
            _vprint(
                f"  跳过 {os.path.basename(str(img_path))}: 标签不足 ({len(ids) if ids is not None else 0} < {min_tags})"
            )
            continue

        img_obj_pts: List[np.ndarray] = []
        img_img_pts: List[np.ndarray] = []

        ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
        for i, tag_id in enumerate(ids_flat.tolist()):
            idx = id_to_idx.get(int(tag_id))
            if idx is None:
                continue
            obj_pts = np.asarray(obj_points_all[idx], dtype=np.float32).reshape(-1, 3)
            img_pts = np.asarray(corners[i], dtype=np.float32).reshape(-1, 2)
            img_obj_pts.append(obj_pts)
            img_img_pts.append(img_pts)

        if len(img_obj_pts) == 0:
            continue

        obj_pts_img = np.vstack(img_obj_pts).astype(np.float32, copy=False)
        img_pts_img = np.vstack(img_img_pts).astype(np.float32, copy=False)

        all_obj_pts.append(obj_pts_img)
        all_img_pts.append(img_pts_img)
        valid_images.append(str(img_path))

        if save_visualization:
            assert detection_dir is not None
            vis_img = img.copy()
            cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
            info_text = f"Image {len(valid_images)}: {len(ids)} tags detected"
            cv2.putText(
                vis_img,
                info_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            detection_path = f"{detection_dir}/{len(valid_images):02d}_tags_detected.jpg"
            _safe_imwrite(detection_path, vis_img)

    print(f"  - 有效图像: {len(valid_images)}/{len(image_files)} (scan_valid={len(valid_candidates)})")
    if (n_read_fail or n_low_tags) and (not VERBOSE):
        parts: List[str] = []
        if n_read_fail:
            parts.append(f"读取失败 {n_read_fail}")
        if n_low_tags:
            parts.append(f"标签不足 {n_low_tags}")
        print(f"  - 跳过统计: {', '.join(parts)}")

    scan_report = _Step3CalibrationReport(
        limits={
            "target_valid": int(limits.target_valid),
            "max_total": int(limits.max_total),
            "max_seconds": float(limits.max_seconds),
        },
        order={"strategy": str(order.strategy), "seed": int(order.seed)},
        workers=int(workers_i),
        prefetch=int(prefetch),
        submitted=int(counters.submitted),
        completed=int(counters.completed),
        valid=int(counters.valid),
        elapsed_s=float(counters.elapsed_s),
        perf=scan_summary,
    )

    if len(valid_images) < 3 or image_size is None:
        print("  错误: 有效图像太少 (<3)")
        return (
            False,
            None,
            None,
            None,
            None,
            None,
            None,
            {
                "limits": scan_report.limits,
                "order": scan_report.order,
                "workers": scan_report.workers,
                "prefetch": scan_report.prefetch,
                "submitted": scan_report.submitted,
                "completed": scan_report.completed,
                "valid": scan_report.valid,
                "elapsed_s": scan_report.elapsed_s,
                "perf": scan_report.perf,
            },
        )

    if bool(dedup) and len(valid_images) > 0:
        all_obj_pts, all_img_pts, valid_images, dd = dedup_views(
            all_obj_pts=all_obj_pts,
            all_img_pts=all_img_pts,
            valid_images=valid_images,
            image_size=image_size,
            bins=int(dedup_bins),
        )
        print(
            f"  - 视图去重: before={dd['before']} after={dd['after']} removed={dd['removed']} (bins={dd['bins']})"
        )

    print("  - 正在标定...")

    n_views = int(len(all_obj_pts))
    total_points = count_total_points(all_img_pts)
    avg_points = float(total_points) / float(max(1, n_views))
    print(f"  - 标定输入: views={n_views}, total_points={total_points}, avg_points_per_view={avg_points:.1f}")
    if n_views >= 200:
        print(
            "  提示: 当前 view 数较大，OpenCV 内参优化可能非常慢。"
            "建议用 --max_valid_images 设为 60~150，或先用 --cameras 单独跑某个相机。"
        )

    ret_rms, K, dist, rvecs, tvecs, elapsed_s = calibrate_camera_intrinsics(
        all_obj_pts,
        all_img_pts,
        image_size,
    )
    print(f"  - 标定耗时: {elapsed_s:.2f} 秒")
    print(f"  - RMS(ret, OpenCV): {ret_rms:.4f} 像素")

    # 计算重投影误差并保存可视化
    mean_error = 0.0

    for i in range(len(all_obj_pts)):
        img_pts2, _ = cv2.projectPoints(all_obj_pts[i], rvecs[i], tvecs[i], K, dist)
        proj = img_pts2.reshape(-1, 2)
        det = np.asarray(all_img_pts[i], dtype=np.float64).reshape(-1, 2)

        per_pt = np.linalg.norm(det - proj, axis=1)
        error = float(np.mean(per_pt)) if per_pt.size > 0 else 0.0
        mean_error += error

        if save_visualization:
            assert reprojection_dir is not None
            vis_img = cv2.imread(valid_images[i])
            if vis_img is None:
                continue

            img_pts_orig = det
            img_pts_reproj = img_pts2.reshape(-1, 2)

            for j in range(len(img_pts_orig)):
                pt_orig = tuple(img_pts_orig[j].astype(int))
                pt_reproj = tuple(img_pts_reproj[j].astype(int))

                cv2.circle(vis_img, pt_orig, 6, (0, 255, 0), -1)  # 绿色：检测点
                cv2.circle(vis_img, pt_reproj, 4, (0, 0, 255), -1)  # 红色：重投影点
                cv2.line(vis_img, pt_orig, pt_reproj, (255, 0, 0), 1)  # 蓝线：误差

            error_text = f"Image {i + 1}: Mean Error = {error:.3f} px"
            cv2.putText(
                vis_img,
                error_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis_img,
                "Green: Detected | Red: Reprojected",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            reproj_path = f"{reprojection_dir}/{i + 1:02d}_error_map.jpg"
            _safe_imwrite(reproj_path, vis_img)

    mean_error = float(mean_error / float(max(1, len(all_obj_pts))))
    print(f"  - 平均误差(mean_error, per-image mean): {mean_error:.4f} 像素")

    if mean_error < 0.5:
        quality = "优秀"
    elif mean_error < 1.0:
        quality = "良好"
    else:
        quality = "需要改进"

    print(f"  - 标定质量: {quality}")

    if save_visualization:
        assert detection_dir is not None
        assert reprojection_dir is not None
        assert undistorted_dir is not None

        _vprint(f"  - 已保存 {len(valid_images)} 张检测图像到 {detection_dir}")
        _vprint(f"  - 已保存 {len(valid_images)} 张重投影图像到 {reprojection_dir}")

        print("  - 正在生成去畸变图像...")

        first_img = cv2.imread(valid_images[0])
        if first_img is None:
            print("  警告: 无法读取第一张有效图像，跳过去畸变输出")
            return (
                True,
                K,
                dist,
                rvecs,
                tvecs,
                mean_error,
                image_size,
                {
                    "limits": scan_report.limits,
                    "order": scan_report.order,
                    "workers": scan_report.workers,
                    "prefetch": scan_report.prefetch,
                    "submitted": scan_report.submitted,
                    "completed": scan_report.completed,
                    "valid": scan_report.valid,
                    "elapsed_s": scan_report.elapsed_s,
                    "perf": scan_report.perf,
                },
            )

        h, w = first_img.shape[:2]
        none_umat: Any = None
        mapx, mapy = cv2.initUndistortRectifyMap(
            K,
            dist,
            none_umat,
            K,
            (w, h),
            cv2.CV_32FC1,
        )

        for i, img_path in enumerate(valid_images):
            img = cv2.imread(img_path)
            if img is None:
                _vprint(f"  警告: 无法读取图像，跳过去畸变: {img_path}")
                continue

            img_undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            base_name = os.path.basename(img_path)
            undist_path = f"{undistorted_dir}/{i + 1:02d}_undistorted_{base_name}"
            _safe_imwrite(undist_path, img_undist)

        _vprint(f"  - 已保存 {len(valid_images)} 张去畸变图像到 {undistorted_dir}")

    return (
        True,
        K,
        dist,
        list(rvecs),
        list(tvecs),
        mean_error,
        image_size,
        {
            "limits": scan_report.limits,
            "order": scan_report.order,
            "workers": scan_report.workers,
            "prefetch": scan_report.prefetch,
            "submitted": scan_report.submitted,
            "completed": scan_report.completed,
            "valid": scan_report.valid,
            "elapsed_s": scan_report.elapsed_s,
            "perf": scan_report.perf,
        },
    )


def save_intrinsics(output_path: str, K: np.ndarray, dist: np.ndarray, reproj_error: float, image_size: Tuple[int, int]) -> None:
    """保存内参标定结果。"""

    result = {
        "camera_matrix": np.asarray(K).tolist(),
        "dist_coeffs": np.asarray(dist).flatten().tolist(),
        "reprojection_error": float(reproj_error),
        "image_size": list(image_size),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }

    Path(output_path).write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"  - 已保存: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step3：AprilTag 内参标定（默认安静，--verbose 可看更多过程信息）")

    parser.add_argument(
        "--config",
        type=str,
        default="config/apriltag_config.json",
        help="配置文件路径（默认 config/apriltag_config.json）",
    )
    parser.add_argument("--verbose", action="store_true", help="输出更多过程信息（逐张/跳过原因等）")
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="不保存可视化输出（detection/reprojection/undistorted），更快更省空间",
    )

    parser.add_argument(
        "--max_valid_images",
        type=int,
        default=MAX_VALID_IMAGES,
        help=f"最多使用多少张有效图像进行标定（默认 {MAX_VALID_IMAGES}）",
    )
    parser.add_argument(
        "--min_tags",
        type=int,
        default=5,
        help="每张图像至少需要检测到多少个标签才算有效（默认 5）",
    )

    parser.add_argument(
        "--dedup_views",
        action="store_true",
        help="对有效视图按(质心/尺度/主方向)做量化去重，删除重复帧以加速标定",
    )
    parser.add_argument(
        "--dedup_bins",
        type=int,
        default=80,
        help="视图去重量化桶数量（越小越容易合并，默认 80）",
    )

    # 性能优先：流式扫描/早停/并行/缓存/预筛选
    parser.add_argument(
        "--max_total_images",
        type=int,
        default=0,
        help="最多尝试检测多少张候选图像（0=自动=8*max_valid_images）。",
    )
    parser.add_argument(
        "--max_detect_seconds",
        type=float,
        default=0.0,
        help="检测总耗时上限（秒，0=不限制）。",
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
        help="启用廉价预筛选（可能过滤掉明显无效/过暗/过曝/模糊帧，减少 detector 调用）。",
    )

    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help=(
            "要标定的相机名称列表（对应 images/<source>/<cam>/...）。"
            "不传则优先用 config.image_dataset.cameras；若未配置则尝试扫描 images/filtered（为空再扫 images/raw）。"
        ),
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

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    global VERBOSE
    VERBOSE = bool(args.verbose)

    print("=" * 60)
    print("Step 3: AprilTag 内参标定")
    print("=" * 60)

    if not bool(args.no_vis):
        print("\n清理旧的可视化文件...")
        clean_visualization_dirs()

    config = load_config(str(args.config))

    cameras: List[str] = []
    if args.cameras is not None and len(args.cameras) > 0:
        cameras = [str(x) for x in args.cameras]
    else:
        cameras = get_dataset_cameras(config, allow_scan=True)
        if len(cameras) == 0:
            # 最后兜底：允许完全不写 image_dataset 时仍能跑起来。
            base = Path("images/filtered")
            if base.exists():
                cameras = [p.name for p in sorted(base.iterdir()) if p.is_dir() and not p.name.startswith("_")]

            if len(cameras) == 0:
                base = Path("images/raw")
                if base.exists():
                    cameras = [p.name for p in sorted(base.iterdir()) if p.is_dir() and not p.name.startswith("_")]

    if len(cameras) == 0:
        print("\n错误: 未找到任何相机图像目录 images/filtered/<cam>/ 或 images/raw/<cam>/")
        print("请先采集图像（或运行 python -m mcca.entry.step2_filter_images 生成 filtered 图像）")
        return 1

    use_multiscale, opencv_refine = get_detection_settings(config)
    if bool(getattr(args, "no_multiscale", False)):
        use_multiscale = False
    if bool(getattr(args, "no_opencv_refine", False)):
        opencv_refine = False

    profile = get_detection_profile(config)
    auto_roi_cfg = get_detection_auto_roi(config)

    _ = create_detector_params(config)

    obj_points, tag_ids = create_apriltag_board(config)
    aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])
    board = create_opencv_aruco_board(obj_points, tag_ids, aruco_dict)

    cam_to_images: Dict[str, List[str]] = {}
    cam_to_source: Dict[str, str] = {}

    for cam in cameras:
        filtered_imgs = get_camera_filtered_images(config, cam)
        if len(filtered_imgs) > 0:
            cam_to_images[cam] = [str(p) for p in filtered_imgs]
            cam_to_source[cam] = "filtered"
        else:
            raw_imgs = get_camera_raw_images(config, cam)
            cam_to_images[cam] = [str(p) for p in raw_imgs]
            cam_to_source[cam] = "raw"

    missing = [c for c in cameras if len(cam_to_images.get(c, [])) == 0]
    if len(missing) > 0:
        print("\n错误: 以下相机未找到任何图像：")
        for c in missing:
            print(f"  - {c} (期望 images/filtered/{c}/ 或 images/raw/{c}/)")
        return 1

    print("\n找到图像：")
    for cam in cameras:
        if cam_to_source[cam] == "filtered":
            _src_dir = get_camera_filtered_dir(config, cam)
        else:
            _src_dir = Path("<raw>")
        print(f"  - {cam}: {len(cam_to_images[cam])} 张 (source={cam_to_source[cam]})")

    print(f"  - detection profile: {profile}")
    if bool(auto_roi_cfg.get("enabled", False)):
        print(
            "  - auto_roi: enabled "
            f"(pre_scale={auto_roi_cfg.get('pre_scale')}, min_tags={auto_roi_cfg.get('min_tags')}, margin={auto_roi_cfg.get('margin')})"
        )

    os.makedirs("results", exist_ok=True)

    max_valid_images = int(args.max_valid_images)
    max_total_images = int(args.max_total_images)
    if max_total_images <= 0:
        max_total_images = int(max_valid_images) * 8
    max_total_images = int(max(max_total_images, max_valid_images))

    limits = ScanLimits(
        target_valid=int(max_valid_images),
        max_total=int(max_total_images),
        max_seconds=float(args.max_detect_seconds),
    )
    order = ScanOrder(strategy=str(args.scan_strategy), seed=int(args.scan_seed))

    workers = int(args.workers) if int(args.workers) > 0 else int(os.cpu_count() or 4)
    if workers <= 0:
        workers = 1

    cache_cfg = CacheConfig(
        enabled=(not bool(args.no_cache)),
        cache_dir=str(args.cache_dir),
        force_redetect=bool(args.force_redetect),
    )
    prefilter_cfg = PrefilterConfig(enabled=bool(args.prefilter))

    step3_report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(args.config),
        "profile": str(profile),
        "scan": {
            "limits": {
                "target_valid": int(limits.target_valid),
                "max_total": int(limits.max_total),
                "max_seconds": float(limits.max_seconds),
            },
            "order": {"strategy": str(order.strategy), "seed": int(order.seed)},
            "workers": int(workers),
            "prefetch": int(args.prefetch),
        },
        "cache": {
            "enabled": bool(cache_cfg.enabled),
            "cache_dir": str(cache_cfg.cache_dir),
            "force_redetect": bool(cache_cfg.force_redetect),
        },
        "prefilter": {"enabled": bool(prefilter_cfg.enabled)},
        "per_camera": {},
    }

    per_cam_summary: List[Tuple[str, np.ndarray, float]] = []

    for cam in cameras:
        images = cam_to_images[cam]
        roi = get_detection_roi(config, camera=cam)
        if roi is not None:
            print(f"\n{cam} ROI: {roi}")

        success, K, dist, rvecs, tvecs, err, img_size, scan_report = calibrate_camera_apriltag(
            images,
            obj_points,
            tag_ids,
            aruco_dict,
            cam,
            config=config,
            profile=str(profile),
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            roi=roi,
            auto_roi_cfg=auto_roi_cfg,
            save_visualization=(not bool(args.no_vis)),
            max_valid_images=int(args.max_valid_images),
            min_tags=int(args.min_tags),
            scan_limits=limits,
            scan_order=order,
            workers=int(workers),
            prefetch=int(args.prefetch),
            cache_cfg=cache_cfg,
            prefilter_cfg=prefilter_cfg,
            dedup=bool(args.dedup_views),
            dedup_bins=int(args.dedup_bins),
        )

        if not bool(success) or K is None or dist is None or err is None or img_size is None:
            print(f"\n错误: 相机 {cam} 标定失败！")
            return 1

        save_intrinsics(f"results/{cam}_intrinsics.json", K, dist, float(err), img_size)
        per_cam_summary.append((cam, K, float(err)))

        step3_report["per_camera"][cam] = {
            "n_images": int(len(images)),
            "source": str(cam_to_source.get(cam, "")),
            "roi": list(roi) if roi is not None else None,
            "scan": scan_report,
            "reprojection_error": float(err),
        }

    try:
        Path("results/step3_intrinsic_report.json").write_text(
            json.dumps(step3_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _vprint("\n已写入报告: results/step3_intrinsic_report.json")
    except Exception as e:
        print(f"\n警告: 写入 Step3 报告失败: {e}")

    print("\n" + "=" * 60)
    print("内参标定完成！")
    print("=" * 60)

    for cam, K, err in per_cam_summary:
        print(f"\n{cam}:")
        print(f"  - fx = {K[0, 0]:.2f}, fy = {K[1, 1]:.2f}")
        print(f"  - cx = {K[0, 2]:.2f}, cy = {K[1, 2]:.2f}")
        print(f"  - 平均误差(mean_error) = {err:.4f} 像素")
        print("  - RMS(ret) 见上方标定日志")

    print("\n下一步: 运行 python -m mcca.entry.step4_multi_extrinsic")

    return 0


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
