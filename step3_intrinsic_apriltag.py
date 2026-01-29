#!/usr/bin/env python3
"""
Step 3: 内参标定 - AprilTag 标定板

命名规范:
    A_T_B 表示 "B -> A 的变换" (X_A = A_T_B @ X_B)

功能:
    使用筛选后的合格图像进行相机内参标定。
    - 相机命名统一为 cam0/cam1/cam2...（由 config.image_dataset.cameras 指定）
    - 也支持 --cameras 显式指定相机名列表
    计算相机内参矩阵 K 和畸变系数 dist。

工作流程:
    1. 读取筛选后的图像
    2. 检测每张图像中的 AprilTag 标签
    3. 建立 2D-3D 点对应关系
    4. 使用 cv2.calibrateCamera 优化内参
    5. 生成去畸变效果对比图

标定方法:
    - 使用 OpenCV 的 cv2.calibrateCamera 函数
    - 自动计算焦距、主点和畸变系数

使用方法:
    python step3_intrinsic_apriltag.py

输入:
    - images/filtered/<cam>/*.(png|jpg|jpeg|bmp): 筛选后的相机图像（推荐）
    - images/raw/<cam>/*.(png|jpg|jpeg|bmp): 原始图像（filtered 缺失时回退）
    - config/apriltag_config.json: 标定板配置

输出:
        - results/<cam>_intrinsics.json: 相机内参
      - camera_matrix: 3x3 内参矩阵 K
      - dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3, ...]
      - reprojection_error: 重投影误差（像素）
        - （可选）results/visualization/step3_intrinsic_<cam>/*: 检测与重投影可视化

质量评估:
    - 重投影误差 < 0.5 像素: 优秀
    - 重投影误差 < 1.0 像素: 良好
    - 重投影误差 > 1.0 像素: 需要改进（检查标定板或图像质量）

下一步:
    - 运行 python step4_multi_extrinsic_pose_graph.py
"""

import argparse
import glob
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from utils import (
    load_config,
    get_aruco_dict,
    create_apriltag_board,
    create_opencv_aruco_board,
    get_detection_settings,
    get_detection_profile,
    get_detection_roi,
    get_detection_auto_roi,
    create_detector_params,
    get_image_dataset,
    get_dataset_cameras,
    get_camera_raw_images,
    get_camera_filtered_dir,
    get_camera_filtered_images,
)

from libs.apriltag_perf.cache import CacheConfig
from libs.apriltag_perf.prefilter import PrefilterConfig
from libs.apriltag_perf.scan import ScanLimits, ScanOrder, iter_scan_parallel_ordered, iter_scan_sequential
from libs.apriltag_perf.service import CachedAprilTagDetector

# 最大有效图像数量（用于内参标定）
# 说明：OpenCV 的 calibrateCamera 会为每张图像引入 6 个外参变量（rvec/tvec），
# 当 view 数过大（例如 300+）时，优化问题会变得非常大且可能“看起来卡死”。
# 实践中内参标定通常 30~120 张高质量图像就足够。
MAX_VALID_IMAGES = 120
MIN_TAGS = 1


# 默认尽量安静：只输出关键结果；需要逐张处理细节用 --verbose。
VERBOSE: bool = True

def _safe_imwrite(path: str, img) -> bool:
    """
    兼容中文路径的写图：优先 cv2.imwrite，失败则使用 imencode + Python 写文件。
    """
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


def _vprint(*args, **kwargs) -> None:
    """Verbose print (guarded by VERBOSE)."""
    if VERBOSE:
        print(*args, **kwargs)


@dataclass(frozen=True)
class _Step3DetectResult:
    """Step3 单张图像的检测摘要（用于早停与可观测性统计）。"""

    image_path: str
    valid: bool
    n_in_board: int
    status: int
    from_cache: bool
    elapsed_ms: float


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _summarize_step3_results(results: list[_Step3DetectResult]) -> dict[str, Any]:
    total = int(len(results))
    valid = int(sum(1 for r in results if bool(r.valid)))
    cache_hit = int(sum(1 for r in results if bool(r.from_cache)))
    prefilter_skipped = int(sum(1 for r in results if int(r.status) == 2))
    error = int(sum(1 for r in results if int(r.status) == 1))

    detect_ms = [float(r.elapsed_ms) for r in results if (not bool(r.from_cache)) and float(r.elapsed_ms) > 0]
    mean_ms = float(np.mean(detect_ms)) if len(detect_ms) > 0 else 0.0
    p95_ms = _percentile(detect_ms, 95.0) if len(detect_ms) > 0 else 0.0

    return {
        "total": total,
        "valid": valid,
        "cache_hit": cache_hit,
        "cache_miss": int(total - cache_hit),
        "prefilter_skipped": prefilter_skipped,
        "error": error,
        "mean_detect_ms": float(mean_ms),
        "p95_detect_ms": float(p95_ms),
    }


def _view_signature(
    img_pts: np.ndarray,
    image_size: tuple[int, int],
    *,
    bins: int,
) -> tuple[int, int, int, int]:
    """为单张视图生成一个“近似姿态签名”，用于去重。

    设计目标：
      - 把视频中大量相似/重复帧合并掉，避免 calibrateCamera 的 view 数爆炸。
      - 同时尽量保留“覆盖整个画面”的视图（质心分布更均匀）。

    签名由以下特征量化而来：
      1) 检测点云质心 (cx, cy)
      2) 点云尺度（平均半径）
      3) 点云主方向（2D PCA 主轴角度）

    Args:
        img_pts: (N,2) 2D 点（像素坐标）。
        image_size: (w, h)。
        bins: 量化桶数量：越小越容易把相似视图合并（去重更强）；越大越倾向保留差异。
    """
    pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    w, h = int(image_size[0]), int(image_size[1])
    if pts.size == 0 or w <= 0 or h <= 0:
        return (0, 0, 0, 0)

    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    r = np.sqrt(dx * dx + dy * dy)
    scale = float(np.mean(r)) if r.size > 0 else 0.0

    # 2D PCA 主方向（忽略符号，映射到 [-pi/2, pi/2)）
    cov = np.cov(np.stack([dx, dy], axis=0)) if pts.shape[0] >= 2 else np.eye(2)
    ang = 0.0
    try:
        # tan(2*theta) = 2*cov_xy / (cov_xx - cov_yy)
        ang = 0.5 * float(np.arctan2(2.0 * float(cov[0, 1]), float(cov[0, 0] - cov[1, 1])))
    except Exception:
        ang = 0.0

    # 归一化到 [0,1]，再量化
    cx_n = float(np.clip(cx / float(w), 0.0, 1.0))
    cy_n = float(np.clip(cy / float(h), 0.0, 1.0))
    sc_n = float(np.clip(scale / float(max(w, h)), 0.0, 1.0))
    # ang in [-pi/2, pi/2) -> [0,1)
    ang_n = float((ang + (np.pi / 2.0)) / np.pi)
    ang_n = float(np.clip(ang_n, 0.0, 0.999999))

    b = int(max(4, int(bins)))
    return (
        int(cx_n * b),
        int(cy_n * b),
        int(sc_n * b),
        int(ang_n * b),
    )


def _dedup_views(
    *,
    all_obj_pts: list[np.ndarray],
    all_img_pts: list[np.ndarray],
    valid_images: list[str],
    image_size: tuple[int, int],
    bins: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict[str, Any]]:
    """按视图签名去重（保留信息量更高的一张）。"""
    n0 = int(len(valid_images))
    if n0 == 0:
        return all_obj_pts, all_img_pts, valid_images, {"before": 0, "after": 0, "removed": 0, "bins": int(bins)}

    best: dict[tuple[int, int, int, int], int] = {}

    for i in range(n0):
        sig = _view_signature(all_img_pts[i], image_size, bins=int(bins))

        # 评分：点数越多，一般约束越强；用于同签名冲突时择优。
        score = int(np.asarray(all_img_pts[i]).reshape(-1, 2).shape[0])
        j = best.get(sig)
        if j is None:
            best[sig] = i
        else:
            score_j = int(np.asarray(all_img_pts[j]).reshape(-1, 2).shape[0])
            if score > score_j:
                best[sig] = i

    keep_idx = sorted(set(best.values()))
    all_obj_pts2 = [all_obj_pts[i] for i in keep_idx]
    all_img_pts2 = [all_img_pts[i] for i in keep_idx]
    valid_images2 = [valid_images[i] for i in keep_idx]

    n1 = int(len(valid_images2))
    return (
        all_obj_pts2,
        all_img_pts2,
        valid_images2,
        {"before": n0, "after": n1, "removed": int(n0 - n1), "bins": int(bins)},
    )


def _build_algo_key(config: dict[str, Any], *, profile: str, camera: str) -> dict[str, Any]:
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
    config: dict[str, Any],
    profile: str,
    camera: str,
    roi: tuple[int, int, int, int] | None,
    cache_cfg: CacheConfig,
    prefilter_cfg: PrefilterConfig,
) -> CachedAprilTagDetector:
    """主进程 detector：用于从 cache 取 corners/ids 并生成可视化。"""

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


_G_STEP3_DET: CachedAprilTagDetector | None = None
_G_STEP3_MIN_TAGS: int = 0
_G_STEP3_TAG_SET: set[int] | None = None


def _init_step3_worker(state: dict[str, Any]) -> None:
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
    _G_STEP3_TAG_SET = set(int(x) for x in state["tag_ids"])


def _step3_worker(image_path: str) -> _Step3DetectResult:
    global _G_STEP3_DET, _G_STEP3_MIN_TAGS, _G_STEP3_TAG_SET
    if _G_STEP3_DET is None or _G_STEP3_TAG_SET is None:
        return _Step3DetectResult(
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
    return _Step3DetectResult(
        image_path=str(image_path),
        valid=valid,
        n_in_board=n_in_board,
        status=int(res.status),
        from_cache=bool(res.from_cache),
        elapsed_ms=float(res.elapsed_ms),
    )


def clean_visualization_dirs():
    """清空 step3 的可视化输出目录"""
    base = Path("results/visualization")
    if not base.exists():
        return

    # 兼容旧输出：step3_intrinsic_左 / step3_intrinsic_右
    for p in sorted(base.glob("step3_intrinsic_*")):
        if p.is_dir():
            shutil.rmtree(str(p))
            _vprint(f"  已清空: {p.as_posix()}")


def _glob_images(dir_path: str) -> list[str]:
    out: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        out.extend(glob.glob(os.path.join(dir_path, ext)))
    return sorted(out)


def _scan_cameras_from_source(source: str) -> list[str]:
    base = Path("images") / source
    if not base.exists():
        return []
    cams: list[str] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        # 说明：images/<source>/ 下可能存在 _comment 等说明目录，不应当被当作相机。
        if p.name.startswith("_") or p.name.startswith(".") or p.name.startswith("__"):
            continue
        files = _glob_images(str(p))
        if len(files) > 0:
            cams.append(p.name)
    return cams


def _pick_images_for_camera(cam: str) -> tuple[list[str], str]:
    """为单个相机挑选图像列表，并返回 (files, source)。

    优先 filtered；若 filtered 为空则回退 raw。
    """
    filtered_dir = os.path.join("images", "filtered", cam)
    raw_dir = os.path.join("images", "raw", cam)

    files = _glob_images(filtered_dir) if os.path.isdir(filtered_dir) else []
    if len(files) > 0:
        return files, "filtered"

    files = _glob_images(raw_dir) if os.path.isdir(raw_dir) else []
    if len(files) > 0:
        return files, "raw"

    return [], ""


def calibrate_camera_apriltag(
    image_files,
    obj_points_all,
    tag_ids,
    aruco_dict,
    side_name,
    *,
    config: dict[str, Any],
    profile: str,
    detector_params,
    use_multiscale: bool,
    opencv_refine: bool,
    board,
    roi=None,
    auto_roi_cfg=None,
    save_visualization: bool = True,
    max_valid_images: int = MAX_VALID_IMAGES,
    min_tags: int = MIN_TAGS,
    scan_limits: ScanLimits | None = None,
    scan_order: ScanOrder | None = None,
    workers: int = 1,
    prefetch: int = 0,
    cache_cfg: CacheConfig | None = None,
    prefilter_cfg: PrefilterConfig | None = None,
    dedup_views: bool = False,
    dedup_bins: int = 80,
):
    """
    使用 AprilTag 标定单个相机的内参

    命名规范:
        cv2.calibrateCamera 返回的 rvecs/tvecs 表示 "T -> C" (标定板 -> 相机)
        即 Cl_T_T 或 Cr_T_T，但本函数不直接使用这些变换矩阵

    Args:
        image_files: 图像文件路径列表
        obj_points_all: 所有标签的 3D 角点 (num_tags, 4, 3)
        tag_ids: 标定板上所有标签的 ID 列表
        aruco_dict: ArUco 字典
        side_name: 相机名称（用于日志）
        save_visualization: 是否保存可视化图像（detection + reprojection）
        max_valid_images: 最大有效图像数量，默认50张

    Returns:
        success: 是否标定成功
        K: 相机内参矩阵 (3, 3)
        dist: 畸变系数
        rvecs: 每张图像的旋转向量列表 (T -> C)
        tvecs: 每张图像的平移向量列表 (T -> C)
        reproj_error: 平均重投影误差

    """
    print(f"\n{side_name}相机标定:")
    print(f"  - 图像数量: {len(image_files)}")
    print(f"  - 最大有效图像: {max_valid_images}")
    if min_tags != 5:
        print(f"  - min_tags: {min_tags}")

    limits = scan_limits or ScanLimits(target_valid=int(max_valid_images))
    order = scan_order or ScanOrder(strategy="uniform", seed=0)
    workers = int(workers) if int(workers) > 0 else 1
    if workers <= 0:
        workers = 1
    cache_cfg = cache_cfg or CacheConfig()
    prefilter_cfg = prefilter_cfg or PrefilterConfig()

    # 创建可视化输出目录
    vis_base = None
    detection_dir = None
    reprojection_dir = None
    undistorted_dir = None
    if save_visualization:
        vis_base = f"results/visualization/step3_intrinsic_{side_name.lower()}"
        detection_dir = f"{vis_base}/detection"
        reprojection_dir = f"{vis_base}/reprojection"
        undistorted_dir = f"{vis_base}/undistorted"
        os.makedirs(detection_dir, exist_ok=True)
        os.makedirs(reprojection_dir, exist_ok=True)
        os.makedirs(undistorted_dir, exist_ok=True)
        _vprint(f"  - 可视化目录: {vis_base}")

    # detector_params 由外部统一创建（便于 profile/配置调参）

    # === 性能优先：先做流式扫描/早停（可并行+缓存+预筛选） ===
    auto_roi_cfg = auto_roi_cfg or {}
    init_state = {
        "config": config,
        "profile": str(profile),
        "camera": str(side_name),
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
    if int(workers) > 1:
        out, counters = iter_scan_parallel_ordered(
            items,
            worker_fn=_step3_worker,
            is_valid_fn=lambda r: bool(r.valid),
            limits=limits,
            order=order,
            max_workers=int(workers),
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
    scan_summary = _summarize_step3_results(scan_results)
    print(
        f"  扫描: submitted={counters.submitted} completed={counters.completed} valid={counters.valid} "
        f"(elapsed={counters.elapsed_s:.2f}s, strategy={order.strategy}, workers={workers})"
    )
    print(
        "  统计: "
        f"cache_hit={scan_summary['cache_hit']} prefilter_skipped={scan_summary['prefilter_skipped']} error={scan_summary['error']} "
        f"mean_detect_ms={scan_summary['mean_detect_ms']:.1f} p95_detect_ms={scan_summary['p95_detect_ms']:.1f}"
    )

    valid_candidates = [r.image_path for r in scan_results if bool(r.valid)]

    # === 第二阶段：在主进程读取 corners/ids（优先命中缓存）并构造标定输入 ===
    det_main = _make_detector_for_camera(
        config=config,
        profile=str(profile),
        camera=str(side_name),
        roi=roi,
        cache_cfg=cache_cfg,
        prefilter_cfg=prefilter_cfg,
    )

    id_to_idx = {int(t): int(i) for i, t in enumerate(tag_ids)}
    all_obj_pts: list[np.ndarray] = []
    all_img_pts: list[np.ndarray] = []
    valid_images: list[str] = []
    image_size = None
    n_read_fail = 0
    n_low_tags = 0

    for img_path in valid_candidates:
        img = cv2.imread(str(img_path))
        if img is None:
            n_read_fail += 1
            _vprint(f"  警告: 无法读取 {img_path}")
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])

        det_res = det_main.detect_path(Path(str(img_path)))
        corners, ids = det_res.corners, det_res.ids
        if ids is None or corners is None or int(len(ids)) < int(min_tags):
            n_low_tags += 1
            _vprint(
                f"  跳过 {os.path.basename(str(img_path))}: 标签不足 ({len(ids) if ids is not None else 0} < {min_tags})"
            )
            continue

        img_obj_pts: list[np.ndarray] = []
        img_img_pts: list[np.ndarray] = []

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
        parts = []
        if n_read_fail:
            parts.append(f"读取失败 {n_read_fail}")
        if n_low_tags:
            parts.append(f"标签不足 {n_low_tags}")
        print(f"  - 跳过统计: {', '.join(parts)}")

    if len(valid_images) < 3:
        print(f"  错误: 有效图像太少 (<3)")
        return False, None, None, None, None, None, None, {
            "limits": {
                "target_valid": int(limits.target_valid),
                "max_total": int(limits.max_total),
                "max_seconds": float(limits.max_seconds),
            },
            "order": {"strategy": str(order.strategy), "seed": int(order.seed)},
            "workers": int(workers),
            "prefetch": int(prefetch),
            "submitted": int(counters.submitted),
            "completed": int(counters.completed),
            "valid": int(counters.valid),
            "elapsed_s": float(counters.elapsed_s),
            "perf": scan_summary,
        }

    assert image_size is not None

    # 可选：去掉重复视图（常见于视频逐帧抽取/板子停在同一位置）。
    # 说明：去重的目的不是“减少图片”，而是减少重复约束，让优化问题更小、收敛更快。
    if bool(dedup_views) and len(valid_images) > 0:
        all_obj_pts, all_img_pts, valid_images, dd = _dedup_views(
            all_obj_pts=all_obj_pts,
            all_img_pts=all_img_pts,
            valid_images=valid_images,
            image_size=image_size,
            bins=int(dedup_bins),
        )
        print(
            f"  - 视图去重: before={dd['before']} after={dd['after']} removed={dd['removed']} (bins={dd['bins']})"
        )

    # 执行相机标定
    print("  - 正在标定...")

    # 可观测性：打印输入规模，便于判断“是真的卡死”还是在做一个很大的优化问题。
    n_views = int(len(all_obj_pts))
    total_points = int(sum(int(pts.shape[0]) for pts in all_img_pts))
    avg_points = float(total_points) / float(max(1, n_views))
    print(f"  - 标定输入: views={n_views}, total_points={total_points}, avg_points_per_view={avg_points:.1f}")
    if n_views >= 200:
        print(
            "  提示: 当前 view 数较大，OpenCV 内参优化可能非常慢。"
            "建议用 --max_valid_images 设为 60~150，或先用 --cameras 单独跑某个相机。"
        )

    # OpenCV 允许用 None 让其自动初始化；但类型存根对 None 不友好。
    none_umat: Any = None
    t0 = time.perf_counter()
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_pts,
        all_img_pts,
        image_size,
        none_umat,
        none_umat,
    )
    t1 = time.perf_counter()
    print(f"  - 标定耗时: {t1 - t0:.2f} 秒")

    # OpenCV 返回的 ret 是全局 RMS 重投影误差（单位：像素）
    print(f"  - RMS(ret, OpenCV): {ret:.4f} 像素")

    # 计算重投影误差并保存可视化
    mean_error = 0
    per_image_errors = []

    for i in range(len(all_obj_pts)):
        img_pts2, _ = cv2.projectPoints(all_obj_pts[i], rvecs[i], tvecs[i], K, dist)
        # mean reprojection error（像素）: 每个点的欧氏误差取平均
        proj = img_pts2.reshape(-1, 2)
        det = np.asarray(all_img_pts[i], dtype=np.float64).reshape(-1, 2)
        per_pt = np.linalg.norm(det - proj, axis=1)
        error = float(np.mean(per_pt))
        mean_error += error
        per_image_errors.append(error)

        # 保存重投影误差可视化
        if save_visualization:
            assert reprojection_dir is not None
            # 注意：不要把所有图像缓存到内存里（数据集一大很容易爆内存/触发换页，表现为卡死）。
            # 这里按需从磁盘读取用于可视化。
            vis_img = cv2.imread(valid_images[i])
            if vis_img is None:
                continue
            img_pts_orig = det
            img_pts_reproj = img_pts2.reshape(-1, 2)

            # 绘制原始检测点（绿色）和重投影点（红色）
            for j in range(len(img_pts_orig)):
                pt_orig = tuple(img_pts_orig[j].astype(int))
                pt_reproj = tuple(img_pts_reproj[j].astype(int))

                cv2.circle(vis_img, pt_orig, 6, (0, 255, 0), -1)  # 绿色：检测点
                cv2.circle(vis_img, pt_reproj, 4, (0, 0, 255), -1)  # 红色：重投影点
                cv2.line(vis_img, pt_orig, pt_reproj, (255, 0, 0), 1)  # 蓝线：误差

            # 显示误差信息
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

    mean_error /= len(all_obj_pts)

    print(f"  - 平均误差(mean_error, per-image mean): {mean_error:.4f} 像素")

    # 评估标定质量
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

        # 保存所有图像的去畸变版本
        print(f"  - 正在生成去畸变图像...")

        # 使用第一张图像的尺寸创建去畸变映射（所有图像尺寸相同）
        first_img = cv2.imread(valid_images[0])
        if first_img is None:
            print("  警告: 无法读取第一张有效图像，跳过去畸变输出")
            return True, K, dist, rvecs, tvecs, mean_error, image_size, {
                "limits": {
                    "target_valid": int(limits.target_valid),
                    "max_total": int(limits.max_total),
                    "max_seconds": float(limits.max_seconds),
                },
                "order": {"strategy": str(order.strategy), "seed": int(order.seed)},
                "workers": int(workers),
                "prefetch": int(prefetch),
                "submitted": int(counters.submitted),
                "completed": int(counters.completed),
                "valid": int(counters.valid),
                "elapsed_s": float(counters.elapsed_s),
                "perf": scan_summary,
            }
        h, w = first_img.shape[:2]

        # 使用 remap 方法进行去畸变（推荐方法）
        # 保持原始内参矩阵K，不改变图像尺寸，只矫正畸变
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

            # 使用remap进行去畸变（比undistort更快且效果更好）
            img_undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            # 保存去畸变图像（保持原始尺寸720x1280，不缩放）
            base_name = os.path.basename(img_path)
            undist_path = f"{undistorted_dir}/{i + 1:02d}_undistorted_{base_name}"
            _safe_imwrite(undist_path, img_undist)

        _vprint(f"  - 已保存 {len(valid_images)} 张去畸变图像到 {undistorted_dir}")

    return True, K, dist, rvecs, tvecs, mean_error, image_size, {
        "limits": {
            "target_valid": int(limits.target_valid),
            "max_total": int(limits.max_total),
            "max_seconds": float(limits.max_seconds),
        },
        "order": {"strategy": str(order.strategy), "seed": int(order.seed)},
        "workers": int(workers),
        "prefetch": int(prefetch),
        "submitted": int(counters.submitted),
        "completed": int(counters.completed),
        "valid": int(counters.valid),
        "elapsed_s": float(counters.elapsed_s),
        "perf": scan_summary,
    }


def save_intrinsics(output_path, K, dist, reproj_error, image_size):
    """保存内参标定结果"""
    result = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "reprojection_error": float(reproj_error),
        "image_size": list(image_size),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  - 已保存: {output_path}")


def main():
    """主函数"""
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
            "不传则自动扫描 images/filtered（为空再扫 images/raw）。默认双目工程推荐 cam0 cam1。"
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

    args = parser.parse_args()

    global VERBOSE
    VERBOSE = bool(args.verbose)

    print("=" * 60)
    print("Step 3: AprilTag 内参标定")
    print("=" * 60)

    # 清空之前的可视化输出（若禁用可视化则不清理，避免误删旧结果）
    if not args.no_vis:
        print("\n清理旧的可视化文件...")
        clean_visualization_dirs()

    # 加载配置
    config = load_config(str(args.config))
    # 统一数据集入口：本仓库不再支持固定的 left/right 目录约定。
    # 相机列表来自 config.image_dataset.cameras；若未配置则尝试扫描 images/*/<cam>/。
    use_dataset = True

    # 确定要标定的相机列表（优先 CLI，其次 config.image_dataset，再次扫描目录）
    cameras = list(args.cameras) if args.cameras is not None and len(args.cameras) > 0 else []
    if len(cameras) == 0 and use_dataset:
        cameras = get_dataset_cameras(config, allow_scan=True)
    if len(cameras) == 0:
        cameras = _scan_cameras_from_source("filtered")
        if len(cameras) == 0:
            cameras = _scan_cameras_from_source("raw")
    if len(cameras) == 0:
        print("\n错误: 未找到任何相机图像目录 images/filtered/<cam>/ 或 images/raw/<cam>/")
        print("请先采集图像（或运行 python step2_filter_images.py 生成 filtered 图像）")
        return

    use_multiscale, opencv_refine = get_detection_settings(config)
    if bool(getattr(args, "no_multiscale", False)):
        use_multiscale = False
    if bool(getattr(args, "no_opencv_refine", False)):
        opencv_refine = False
    profile = get_detection_profile(config)
    auto_roi_cfg = get_detection_auto_roi(config)
    detector_params = create_detector_params(config)

    # 创建 AprilTag 标定板
    obj_points, tag_ids = create_apriltag_board(config)
    aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])
    board = create_opencv_aruco_board(obj_points, tag_ids, aruco_dict)

    # 收集每个相机的图像文件（每个相机独立：优先 filtered，若为空则回退 raw）
    cam_to_images: dict[str, list[str]] = {}
    cam_to_source: dict[str, str] = {}
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
        return

    print("\n找到图像：")
    for cam in cameras:
        if cam_to_source[cam] == "filtered":
            _src_dir = get_camera_filtered_dir(config, cam)
        else:
            # raw 可能来自 glob，不一定在 raw_root/<cam>
            _src_dir = Path("<raw>")
        print(f"  - {cam}: {len(cam_to_images[cam])} 张 (source={cam_to_source[cam]})")
    print(f"  - detection profile: {profile}")
    if bool(auto_roi_cfg.get("enabled", False)):
        print(
            "  - auto_roi: enabled "
            f"(pre_scale={auto_roi_cfg.get('pre_scale')}, min_tags={auto_roi_cfg.get('min_tags')}, margin={auto_roi_cfg.get('margin')})"
        )

    # 确保输出目录存在
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

    step3_report: dict[str, Any] = {
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

    # 逐相机标定
    per_cam_summary: list[tuple[str, np.ndarray, float]] = []
    for cam in cameras:
        images = cam_to_images[cam]
        roi = get_detection_roi(config, camera=cam)
        if roi is not None:
            print(f"\n{cam} ROI: {roi}")

        display_name = cam
        success, K, dist, rvecs, tvecs, err, img_size, scan_report = calibrate_camera_apriltag(
            list(images),
            obj_points,
            tag_ids,
            aruco_dict,
            display_name,
            config=config,
            profile=str(profile),
            detector_params=detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            roi=roi,
            auto_roi_cfg=auto_roi_cfg,
            save_visualization=(not args.no_vis),
            max_valid_images=int(args.max_valid_images),
            min_tags=int(args.min_tags),
            scan_limits=limits,
            scan_order=order,
            workers=int(workers),
            prefetch=int(args.prefetch),
            cache_cfg=cache_cfg,
            prefilter_cfg=prefilter_cfg,
            dedup_views=bool(args.dedup_views),
            dedup_bins=int(args.dedup_bins),
        )

        if not success:
            print(f"\n错误: 相机 {cam} 标定失败！")
            return

        assert K is not None and dist is not None and img_size is not None
        assert err is not None
        save_intrinsics(f"results/{cam}_intrinsics.json", K, dist, err, img_size)
        per_cam_summary.append((cam, K, float(err)))
        step3_report["per_camera"][cam] = {
            "n_images": int(len(images)),
            "source": str(cam_to_source.get(cam, "")),
            "roi": list(roi) if roi is not None else None,
            "scan": scan_report,
            "reprojection_error": float(err),
        }

    try:
        with open("results/step3_intrinsic_report.json", "w", encoding="utf-8") as f:
            json.dump(step3_report, f, ensure_ascii=False, indent=2)
        _vprint("\n已写入报告: results/step3_intrinsic_report.json")
    except Exception as e:
        print(f"\n警告: 写入 Step3 报告失败: {e}")

    # 显示总结
    print("\n" + "=" * 60)
    print("内参标定完成！")
    print("=" * 60)
    for cam, K, err in per_cam_summary:
        print(f"\n{cam}:")
        print(f"  - fx = {K[0, 0]:.2f}, fy = {K[1, 1]:.2f}")
        print(f"  - cx = {K[0, 2]:.2f}, cy = {K[1, 2]:.2f}")
        print(f"  - 平均误差(mean_error) = {err:.4f} 像素")
        print("  - RMS(ret) 见上方标定日志")

    print("\n下一步: 运行 python step4_multi_extrinsic_pose_graph.py")


if __name__ == "__main__":
    main()
