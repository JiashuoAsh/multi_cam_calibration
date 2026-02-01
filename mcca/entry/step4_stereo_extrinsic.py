#!/usr/bin/env python3
"""Step 4: 双目外参标定 - AprilTag 标定板（临时标记：待清理）

说明：
    本仓库已统一使用 config 驱动的相机命名（cam0/cam1/cam2...），不再使用 left/right。

输入（推荐数据集结构）：
    - 图像：images/<source>/<cam_name>/*.png|*.jpg|*.jpeg|*.bmp
      - <source> 通常为 images/filtered（优先）或 images/raw
      - “同步”的含义：不同相机同一时刻的帧文件名 stem 相同（或形如 <cam>_ 前缀会被自动去掉）。
    - 内参：results/<cam_name>_intrinsics.json（由 Step3 生成）

输出：
    - results/stereo_extrinsics.json（R,t,E,F,baseline + camera_a/camera_b 等元信息）
    - results/stereo_rectification.json（R1,R2,P1,P2,Q + roi_a/roi_b）
    - results/stereo_rectification_demo.jpg

OpenCV 约定：
    stereoCalibrate 返回的 R,t 满足：
        X_cam_b = R * X_cam_a + t
"""

import cv2
import numpy as np
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from mcca.core.board import (
    create_apriltag_board,
    create_opencv_aruco_board,
    get_aruco_dict,
)
from mcca.core.config import load_config
from mcca.core.datasets import (
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

from mcca.adapters.apriltag_perf.cache import CacheConfig
from mcca.adapters.apriltag_perf.prefilter import PrefilterConfig
from mcca.adapters.apriltag_perf.scan import (
    ScanLimits,
    ScanOrder,
    iter_scan_parallel_ordered,
    iter_scan_sequential,
)
from mcca.adapters.apriltag_perf.service import CachedAprilTagDetector

# 最大有效图像对数量（用于双目外参标定）
MAX_VALID_IMAGES = 50

# 默认质量过滤阈值：每对图像至少需要的共同标签数（与 --min_common_tags 默认一致）
MIN_COMMON_TAGS_DEFAULT = 1
# 每张图像最少需要检测到的标签数（用于 collect_stereo_points 的单目门槛）。
# - None: 跟随 MIN_COMMON_TAGS（默认，保持当前行为）
# - int : 使用固定值（例如 4），便于更严格地剔除“单目检测很少标签”的图像。
MIN_TAGS_PER_IMAGE_DEFAULT = 1

# 用于“PnP/重投影误差统计”的最小标签数（注意：仅影响误差统计时是否跳过某一视角，不影响 stereoCalibrate）。
# 原先代码里写死了 "len(obj_pts) < 10" —— 这是一个经验阈值，但确实属于隐藏超参数。
# 这里改成显式常量，默认取 1（即 1 个 tag = 4 个角点，就足够 solvePnP 跑起来；想更稳可改大）。
MIN_PNP_TAGS_PER_VIEW = 1
# 默认尽量安静：保留关键指标输出；矩阵/几何细节用 --verbose。
VERBOSE: bool = False


_G_DET_A: Optional[CachedAprilTagDetector] = None
_G_DET_B: Optional[CachedAprilTagDetector] = None
_G_OBJ_POINTS_ALL: Optional[np.ndarray] = None
_G_TAG_ID_TO_IDX: Optional[Dict[int, int]] = None
_G_MIN_COMMON_TAGS: int = 1
_G_MIN_TAGS_PER_IMAGE: int = 1


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _normalize_frame_key(stem: str, cam: str) -> str:
    """从文件名 stem 生成用于“同帧匹配”的 frame_key。

    说明：
        很多数据集会把相机名写进文件名前缀，例如：
            cam1_seq000015.png
            cam3_seq000015.png
        这两张其实是同一帧，但 stem 不同。
        这里做一个保守归一化：若 stem 以 "{cam}_" 或 "{cam}-" 开头，则去掉该前缀。
    """

    stem = str(stem)
    cam = str(cam)
    if stem.startswith(cam + "_"):
        return stem[len(cam) + 1 :]
    if stem.startswith(cam + "-"):
        return stem[len(cam) + 1 :]
    return stem


def _build_frame_map_from_paths(paths: List[Path], cam: str) -> Dict[str, Path]:
    """构建 frame_key -> Path 的映射（去重，保留最先出现的文件）。"""

    out: Dict[str, Path] = {}
    dup = 0
    for p in sorted(paths):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        key = _normalize_frame_key(p.stem, cam)
        if key in out:
            dup += 1
            continue
        out[str(key)] = p
    if dup > 0:
        _vprint(f"警告: {cam} 存在 {dup} 个 frame_key 冲突（归一化后重名），已保留最先出现的文件。")
    return out


def _collect_synced_pairs(
    *,
    config: Dict[str, Any],
    cam_a: str,
    cam_b: str,
    image_source: str,
) -> Tuple[List[Tuple[str, str]], str, Dict[str, Any]]:
    """收集并按 frame_key(stem) 对齐的图像对。

    Returns:
        pairs: [(path_a, path_b), ...]
        image_source_used: "filtered" | "raw"
        diag: 诊断信息
    """

    image_source = str(image_source).strip().lower() or "auto"
    if image_source not in {"auto", "filtered", "raw"}:
        raise ValueError(f"image_source 只能是 auto/filtered/raw，当前={image_source}")

    def _get_paths(source: str, cam: str) -> List[Path]:
        if source == "filtered":
            return list(get_camera_filtered_images(config, cam))
        if source == "raw":
            return list(get_camera_raw_images(config, cam))
        raise ValueError(f"unknown source: {source}")

    source_used = image_source
    if image_source == "auto":
        pa_f = _get_paths("filtered", cam_a)
        pb_f = _get_paths("filtered", cam_b)
        if len(pa_f) > 0 and len(pb_f) > 0:
            source_used = "filtered"
        else:
            source_used = "raw"

    paths_a = _get_paths(source_used, cam_a)
    paths_b = _get_paths(source_used, cam_b)

    if len(paths_a) == 0 or len(paths_b) == 0:
        raise FileNotFoundError(
            "未找到可用图像。"
            f"\n  - source={source_used}"
            f"\n  - {cam_a} images: {len(paths_a)}"
            f"\n  - {cam_b} images: {len(paths_b)}"
        )

    map_a = _build_frame_map_from_paths(paths_a, cam_a)
    map_b = _build_frame_map_from_paths(paths_b, cam_b)

    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    pairs = [(str(map_a[k]), str(map_b[k])) for k in keys]

    diag = {
        "image_source": source_used,
        "per_cam_images": {cam_a: int(len(paths_a)), cam_b: int(len(paths_b))},
        "per_cam_frame_keys": {cam_a: int(len(map_a)), cam_b: int(len(map_b))},
        "intersection_pairs": int(len(pairs)),
        "sample_frame_keys": keys[:10],
        "note": "frame_key 来自文件名 stem（会自动去掉形如 <cam>_ / <cam>- 的前缀）。",
    }
    return pairs, source_used, diag

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


def load_intrinsics(json_path):
    """加载内参标定结果"""
    with open(json_path, "r") as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)

    return K, dist


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


def _init_stereo_pair_worker(state: Dict[str, Any]) -> None:
    """多进程 worker 初始化：在子进程内创建 OpenCV 检测器与缓存服务。

    说明：
    - Windows 下 multiprocessing 采用 spawn，OpenCV 对象不可 pickle，必须在子进程初始化。
    - cache 目录可被多个进程同时写；同一 key 的竞争写入属于幂等覆盖。
    """

    global _G_DET_A, _G_DET_B, _G_OBJ_POINTS_ALL, _G_TAG_ID_TO_IDX
    global _G_MIN_COMMON_TAGS, _G_MIN_TAGS_PER_IMAGE

    config = state["config"]
    profile = str(state["profile"])

    obj_points = np.asarray(state["obj_points"], dtype=np.float32)
    tag_ids = [int(x) for x in state["tag_ids"]]
    _G_OBJ_POINTS_ALL = obj_points
    _G_TAG_ID_TO_IDX = {int(t): i for i, t in enumerate(tag_ids)}

    aruco_dict = get_aruco_dict(str(state["family"]))
    board = create_opencv_aruco_board(obj_points, tag_ids, aruco_dict)

    detector_params = create_detector_params(config)

    algo_key = _build_algo_key(config, profile=profile)
    cache_cfg = CacheConfig(
        enabled=bool(state["cache"]["enabled"]),
        cache_dir=str(state["cache"]["cache_dir"]),
        force_redetect=bool(state["cache"]["force_redetect"]),
    )
    prefilter_cfg = PrefilterConfig(enabled=bool(state["prefilter"]["enabled"]))

    auto_roi_cfg = state.get("auto_roi_cfg") or {}

    _G_DET_A = CachedAprilTagDetector(
        aruco_dict=aruco_dict,
        detector_params=detector_params,
        algo_key=algo_key,
        use_multiscale=bool(state["use_multiscale"]),
        opencv_refine=bool(state["opencv_refine"]),
        board=board,
        camera_matrix=np.asarray(state["K_a"], dtype=np.float64),
        dist_coeffs=np.asarray(state["dist_a"], dtype=np.float64),
        roi=tuple(state["roi_a"]) if state.get("roi_a") is not None else None,
        auto_roi=bool(auto_roi_cfg.get("enabled", False)),
        auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
        auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
        auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
        cache_cfg=cache_cfg,
        prefilter_cfg=prefilter_cfg,
    )
    _G_DET_B = CachedAprilTagDetector(
        aruco_dict=aruco_dict,
        detector_params=detector_params,
        algo_key=algo_key,
        use_multiscale=bool(state["use_multiscale"]),
        opencv_refine=bool(state["opencv_refine"]),
        board=board,
        camera_matrix=np.asarray(state["K_b"], dtype=np.float64),
        dist_coeffs=np.asarray(state["dist_b"], dtype=np.float64),
        roi=tuple(state["roi_b"]) if state.get("roi_b") is not None else None,
        auto_roi=bool(auto_roi_cfg.get("enabled", False)),
        auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
        auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
        auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
        cache_cfg=cache_cfg,
        prefilter_cfg=prefilter_cfg,
    )

    _G_MIN_COMMON_TAGS = int(state["min_common_tags"])
    _G_MIN_TAGS_PER_IMAGE = int(state["min_tags_per_image"])


def _stereo_pair_worker(pair: Tuple[str, str]) -> Dict[str, Any]:
    """多进程扫描的单任务：检测一对图像并在满足阈值时输出 stereoCalibrate 所需点。"""

    global _G_DET_A, _G_DET_B, _G_OBJ_POINTS_ALL, _G_TAG_ID_TO_IDX
    global _G_MIN_COMMON_TAGS, _G_MIN_TAGS_PER_IMAGE

    a_path, b_path = pair

    if _G_DET_A is None or _G_DET_B is None or _G_OBJ_POINTS_ALL is None or _G_TAG_ID_TO_IDX is None:
        return {
            "valid": False,
            "a": a_path,
            "b": b_path,
            "error": "worker 未初始化",
        }

    res_a = _G_DET_A.detect_path(Path(a_path))
    res_b = _G_DET_B.detect_path(Path(b_path))

    ids_a = np.asarray(res_a.ids) if res_a.ids is not None else np.zeros((0, 1), dtype=np.int32)
    ids_b = np.asarray(res_b.ids) if res_b.ids is not None else np.zeros((0, 1), dtype=np.int32)
    corners_a = res_a.corners or []
    corners_b = res_b.corners or []

    n_a = int(ids_a.shape[0])
    n_b = int(ids_b.shape[0])

    if n_a < int(_G_MIN_TAGS_PER_IMAGE) or n_b < int(_G_MIN_TAGS_PER_IMAGE):
        return {
            "valid": False,
            "a": a_path,
            "b": b_path,
            "n_a": n_a,
            "n_b": n_b,
            "common": 0,
            "a_from_cache": bool(res_a.from_cache),
            "b_from_cache": bool(res_b.from_cache),
            "a_status": int(res_a.status),
            "b_status": int(res_b.status),
            "a_ms": float(res_a.elapsed_ms),
            "b_ms": float(res_b.elapsed_ms),
        }

    ids_a_flat = ids_a.reshape(-1).astype(np.int32, copy=False)
    ids_b_flat = ids_b.reshape(-1).astype(np.int32, copy=False)
    common_ids = set(int(x) for x in ids_a_flat.tolist()) & set(int(x) for x in ids_b_flat.tolist())
    common_n = int(len(common_ids))

    if common_n < int(_G_MIN_COMMON_TAGS):
        return {
            "valid": False,
            "a": a_path,
            "b": b_path,
            "n_a": n_a,
            "n_b": n_b,
            "common": common_n,
            "a_from_cache": bool(res_a.from_cache),
            "b_from_cache": bool(res_b.from_cache),
            "a_status": int(res_a.status),
            "b_status": int(res_b.status),
            "a_ms": float(res_a.elapsed_ms),
            "b_ms": float(res_b.elapsed_ms),
        }

    # 收集共同标签的对应点
    id_to_a_idx = {int(t): i for i, t in enumerate(ids_a_flat.tolist())}
    id_to_b_idx = {int(t): i for i, t in enumerate(ids_b_flat.tolist())}

    obj_pts_pair: List[np.ndarray] = []
    img_pts_a_pair: List[np.ndarray] = []
    img_pts_b_pair: List[np.ndarray] = []

    for tag_id in common_ids:
        bidx = _G_TAG_ID_TO_IDX.get(int(tag_id))
        if bidx is None:
            continue

        ai = id_to_a_idx.get(int(tag_id))
        bi = id_to_b_idx.get(int(tag_id))
        if ai is None or bi is None:
            continue

        obj_pts = _G_OBJ_POINTS_ALL[bidx]
        img_a = np.asarray(corners_a[ai], dtype=np.float32).reshape(-1, 2)
        img_b = np.asarray(corners_b[bi], dtype=np.float32).reshape(-1, 2)

        obj_pts_pair.append(np.asarray(obj_pts, dtype=np.float32))
        img_pts_a_pair.append(img_a.astype(np.float32, copy=False))
        img_pts_b_pair.append(img_b.astype(np.float32, copy=False))

    if len(obj_pts_pair) == 0:
        return {
            "valid": False,
            "a": a_path,
            "b": b_path,
            "n_a": n_a,
            "n_b": n_b,
            "common": common_n,
            "a_from_cache": bool(res_a.from_cache),
            "b_from_cache": bool(res_b.from_cache),
            "a_status": int(res_a.status),
            "b_status": int(res_b.status),
            "a_ms": float(res_a.elapsed_ms),
            "b_ms": float(res_b.elapsed_ms),
        }

    obj_pts_combined = np.vstack(obj_pts_pair).astype(np.float32)
    img_pts_a_combined = np.vstack(img_pts_a_pair).astype(np.float32)
    img_pts_b_combined = np.vstack(img_pts_b_pair).astype(np.float32)

    return {
        "valid": True,
        "a": a_path,
        "b": b_path,
        "n_a": n_a,
        "n_b": n_b,
        "common": common_n,
        "obj": obj_pts_combined,
        "img_a": img_pts_a_combined,
        "img_b": img_pts_b_combined,
        "a_from_cache": bool(res_a.from_cache),
        "b_from_cache": bool(res_b.from_cache),
        "a_status": int(res_a.status),
        "b_status": int(res_b.status),
        "a_ms": float(res_a.elapsed_ms),
        "b_ms": float(res_b.elapsed_ms),
    }

def compute_mean_reproj_error_pnp(all_obj_pts, all_img_pts, K, dist):
    """
    仿照 step3 的口径计算 mean_error（每张图像的 mean reprojection error 再取平均）
    error = mean(||x_detect - x_proj||_2)
    """
    per_view_errors = []
    used = 0

    for obj_pts, img_pts in zip(all_obj_pts, all_img_pts):
        obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
        img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)

        # 1 个 tag = 4 个角点；少于该阈值时 solvePnP 容易不稳定，统计时跳过该视角。
        if len(obj_pts) < int(max(1, int(MIN_PNP_TAGS_PER_VIEW))) * 4:
            continue

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)

        per_pt = np.linalg.norm(img_pts - proj, axis=1)
        err = float(np.mean(per_pt))
        per_view_errors.append(float(err))
        used += 1

    mean_error = float(np.mean(per_view_errors)) if used > 0 else float("nan")
    return mean_error, per_view_errors, used


def compute_cam_b_mean_reproj_error_using_rt(
    all_obj_pts,
    all_img_pts_a,
    all_img_pts_b,
    K_a,
    dist_a,
    K_b,
    dist_b,
    R_ba,
    t_ba,
):
    """CamB 视角的重投影误差（使用 stereo 的 R,t）。

    步骤：
        1) 在 CamA 用 solvePnP 得到 (Board -> CamA) 位姿
        2) 用 stereo 外参把位姿变到 CamB (Board -> CamB_pred)
        3) 在 CamB 上 projectPoints，与 CamB 的检测点算误差
    """
    per_view_errors = []
    used = 0

    R_ba = np.asarray(R_ba, dtype=np.float64)
    t_ba = np.asarray(t_ba, dtype=np.float64).reshape(3, 1)

    for obj_pts, img_a, img_b in zip(all_obj_pts, all_img_pts_a, all_img_pts_b):
        obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
        img_a = np.asarray(img_a, dtype=np.float64).reshape(-1, 2)
        img_b = np.asarray(img_b, dtype=np.float64).reshape(-1, 2)

        if len(obj_pts) < int(max(1, int(MIN_PNP_TAGS_PER_VIEW))) * 4:
            continue

        ok, rvec_a, tvec_a = cv2.solvePnP(
            obj_pts, img_a, K_a, dist_a, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue

        R_a, _ = cv2.Rodrigues(rvec_a)          # Board -> CamA
        R_b = R_ba @ R_a                         # Board -> CamB_pred
        t_b = R_ba @ tvec_a + t_ba               # Board -> CamB_pred
        rvec_b, _ = cv2.Rodrigues(R_b)

        proj_b, _ = cv2.projectPoints(obj_pts, rvec_b, t_b, K_b, dist_b)
        proj_b = proj_b.reshape(-1, 2)

        per_pt = np.linalg.norm(img_b - proj_b, axis=1)
        err_b = float(np.mean(per_pt))
        per_view_errors.append(float(err_b))
        used += 1

    mean_error = float(np.mean(per_view_errors)) if used > 0 else float("nan")
    return mean_error, per_view_errors, used


def compute_stereo_mean_reproj_error_using_rt(
    all_obj_pts,
    all_img_pts_a,
    all_img_pts_b,
    K_a,
    dist_a,
    K_b,
    dist_b,
    R_ba,
    t_ba,
):
    """计算双目 mean reprojection error（像素，mean 口径）。

    步骤：
        1) 对每对图像，在 CamA 用 solvePnP 解 (Board -> CamA)
        2) 用 stereo 外参推导 (Board -> CamB_pred)
        3) 将 3D 点投影回 CamA/CamB 图像，与检测点计算每点欧氏误差
        4) 汇总所有点（CamA+CamB 合计）的 mean

    Returns:
        mean_all: 所有点（CamA+CamB 合计）的 mean reprojection error
        mean_a: CamA mean
        mean_b: CamB mean
        used_pairs: 实际用于统计的图像对数量
    """
    R_ba = np.asarray(R_ba, dtype=np.float64)
    t_ba = np.asarray(t_ba, dtype=np.float64).reshape(3, 1)

    all_err = []
    all_err_a = []
    all_err_b = []
    used_pairs = 0

    for obj_pts, img_a, img_b in zip(all_obj_pts, all_img_pts_a, all_img_pts_b):
        obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
        img_a = np.asarray(img_a, dtype=np.float64).reshape(-1, 2)
        img_b = np.asarray(img_b, dtype=np.float64).reshape(-1, 2)

        if len(obj_pts) < int(max(1, int(MIN_PNP_TAGS_PER_VIEW))) * 4:
            continue

        ok, rvec_a, tvec_a = cv2.solvePnP(
            obj_pts, img_a, K_a, dist_a, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue

        # CamA reprojection
        proj_a, _ = cv2.projectPoints(obj_pts, rvec_a, tvec_a, K_a, dist_a)
        proj_a = proj_a.reshape(-1, 2)
        err_a = np.linalg.norm(img_a - proj_a, axis=1)
        all_err.extend(err_a.tolist())
        all_err_a.extend(err_a.tolist())

        # CamB reprojection using stereo constraint
        R_a, _ = cv2.Rodrigues(rvec_a)
        R_b = R_ba @ R_a
        t_b = R_ba @ tvec_a + t_ba
        rvec_b, _ = cv2.Rodrigues(R_b)

        proj_b, _ = cv2.projectPoints(obj_pts, rvec_b, t_b, K_b, dist_b)
        proj_b = proj_b.reshape(-1, 2)
        err_b = np.linalg.norm(img_b - proj_b, axis=1)
        all_err.extend(err_b.tolist())
        all_err_b.extend(err_b.tolist())

        used_pairs += 1

    if used_pairs == 0:
        return float("nan"), float("nan"), float("nan"), 0

    mean_all = float(np.mean(np.asarray(all_err, dtype=np.float64)))
    mean_a = float(np.mean(np.asarray(all_err_a, dtype=np.float64)))
    mean_b = float(np.mean(np.asarray(all_err_b, dtype=np.float64)))
    return mean_all, mean_a, mean_b, used_pairs


def scan_and_collect_stereo_points(
    pairs: List[Tuple[str, str]],
    *,
    obj_points_all: np.ndarray,
    tag_ids: List[int],
    min_common_tags: int,
    min_tags_per_image: int,
    limits: ScanLimits,
    order: ScanOrder,
    max_workers: int,
    prefetch: int,
    det_a: Optional[CachedAprilTagDetector] = None,
    det_b: Optional[CachedAprilTagDetector] = None,
    parallel_state: Optional[Dict[str, Any]] = None,
):
    """流式扫描并收集 stereoCalibrate 所需点。

    关键点：
    - 每对图像只“检测一次”，同时完成质量评估与点收集。
    - 支持：target_valid / max_total / max_seconds 的早停。
    - 可选多进程并行：仅在 worker 内初始化 OpenCV 检测器。
    """

    min_common_tags = int(max(1, int(min_common_tags)))
    min_tags_per_image = int(max(1, int(min_tags_per_image)))

    tag_id_to_idx = {int(t): i for i, t in enumerate(tag_ids)}

    def _is_valid(r: Dict[str, Any]) -> bool:
        return bool(r.get("valid", False))

    def _worker_local(pair: Tuple[str, str]) -> Dict[str, Any]:
        if det_a is None or det_b is None:
            return {"valid": False, "a": pair[0], "b": pair[1], "error": "detector 未初始化"}

        ap, bp = pair
        res_a = det_a.detect_path(Path(ap))
        res_b = det_b.detect_path(Path(bp))

        ids_a = np.asarray(res_a.ids) if res_a.ids is not None else np.zeros((0, 1), dtype=np.int32)
        ids_b = np.asarray(res_b.ids) if res_b.ids is not None else np.zeros((0, 1), dtype=np.int32)
        corners_a = res_a.corners or []
        corners_b = res_b.corners or []

        n_a = int(ids_a.shape[0])
        n_b = int(ids_b.shape[0])

        if n_a < min_tags_per_image or n_b < min_tags_per_image:
            return {
                "valid": False,
                "a": ap,
                "b": bp,
                "n_a": n_a,
                "n_b": n_b,
                "common": 0,
                "a_from_cache": bool(res_a.from_cache),
                "b_from_cache": bool(res_b.from_cache),
                "a_status": int(res_a.status),
                "b_status": int(res_b.status),
                "a_ms": float(res_a.elapsed_ms),
                "b_ms": float(res_b.elapsed_ms),
            }

        ids_a_flat = ids_a.reshape(-1).astype(np.int32, copy=False)
        ids_b_flat = ids_b.reshape(-1).astype(np.int32, copy=False)
        common_ids = set(int(x) for x in ids_a_flat.tolist()) & set(int(x) for x in ids_b_flat.tolist())
        common_n = int(len(common_ids))
        if common_n < min_common_tags:
            return {
                "valid": False,
                "a": ap,
                "b": bp,
                "n_a": n_a,
                "n_b": n_b,
                "common": common_n,
                "a_from_cache": bool(res_a.from_cache),
                "b_from_cache": bool(res_b.from_cache),
                "a_status": int(res_a.status),
                "b_status": int(res_b.status),
                "a_ms": float(res_a.elapsed_ms),
                "b_ms": float(res_b.elapsed_ms),
            }

        id_to_a_idx = {int(t): i for i, t in enumerate(ids_a_flat.tolist())}
        id_to_b_idx = {int(t): i for i, t in enumerate(ids_b_flat.tolist())}

        obj_pts_pair: List[np.ndarray] = []
        img_pts_a_pair: List[np.ndarray] = []
        img_pts_b_pair: List[np.ndarray] = []

        for tag_id in common_ids:
            bidx = tag_id_to_idx.get(int(tag_id))
            if bidx is None:
                continue
            ai = id_to_a_idx.get(int(tag_id))
            bi = id_to_b_idx.get(int(tag_id))
            if ai is None or bi is None:
                continue
            obj_pts = obj_points_all[bidx]
            img_a = np.asarray(corners_a[ai], dtype=np.float32).reshape(-1, 2)
            img_b = np.asarray(corners_b[bi], dtype=np.float32).reshape(-1, 2)
            obj_pts_pair.append(np.asarray(obj_pts, dtype=np.float32))
            img_pts_a_pair.append(img_a.astype(np.float32, copy=False))
            img_pts_b_pair.append(img_b.astype(np.float32, copy=False))

        if len(obj_pts_pair) == 0:
            return {
                "valid": False,
                "a": ap,
                "b": bp,
                "n_a": n_a,
                "n_b": n_b,
                "common": common_n,
                "a_from_cache": bool(res_a.from_cache),
                "b_from_cache": bool(res_b.from_cache),
                "a_status": int(res_a.status),
                "b_status": int(res_b.status),
                "a_ms": float(res_a.elapsed_ms),
                "b_ms": float(res_b.elapsed_ms),
            }

        return {
            "valid": True,
            "a": ap,
            "b": bp,
            "n_a": n_a,
            "n_b": n_b,
            "common": common_n,
            "obj": np.vstack(obj_pts_pair).astype(np.float32),
            "img_a": np.vstack(img_pts_a_pair).astype(np.float32),
            "img_b": np.vstack(img_pts_b_pair).astype(np.float32),
            "a_from_cache": bool(res_a.from_cache),
            "b_from_cache": bool(res_b.from_cache),
            "a_status": int(res_a.status),
            "b_status": int(res_b.status),
            "a_ms": float(res_a.elapsed_ms),
            "b_ms": float(res_b.elapsed_ms),
        }

    if int(max_workers) > 0:
        if parallel_state is None:
            raise ValueError("parallel_state 不能为空（并行模式需要传入初始化参数）")

        results, counters = iter_scan_parallel_ordered(
            pairs,
            worker_fn=_stereo_pair_worker,
            is_valid_fn=_is_valid,
            limits=limits,
            order=order,
            max_workers=int(max_workers),
            prefetch=int(prefetch),
            initializer=_init_stereo_pair_worker,
            initargs=(parallel_state,),
        )
    else:
        results, counters = iter_scan_sequential(
            pairs,
            worker_fn=_worker_local,
            is_valid_fn=_is_valid,
            limits=limits,
            order=order,
        )

    all_obj_pts: List[np.ndarray] = []
    all_img_pts_a: List[np.ndarray] = []
    all_img_pts_b: List[np.ndarray] = []
    valid_pairs: List[Tuple[str, str]] = []
    valid_common_counts: List[int] = []

    perf = {
        "pairs_scanned": int(counters.completed),
        "pairs_valid": int(counters.valid),
        "elapsed_s": float(counters.elapsed_s),
        "a_cache_hit": 0,
        "b_cache_hit": 0,
        "a_prefilter_skipped": 0,
        "b_prefilter_skipped": 0,
        "a_error": 0,
        "b_error": 0,
        "detect_ms_sum": 0.0,
    }

    for r in results:
        perf["a_cache_hit"] += 1 if bool(r.get("a_from_cache", False)) else 0
        perf["b_cache_hit"] += 1 if bool(r.get("b_from_cache", False)) else 0
        perf["a_prefilter_skipped"] += 1 if int(r.get("a_status", 0)) == 2 else 0
        perf["b_prefilter_skipped"] += 1 if int(r.get("b_status", 0)) == 2 else 0
        perf["a_error"] += 1 if int(r.get("a_status", 0)) == 1 else 0
        perf["b_error"] += 1 if int(r.get("b_status", 0)) == 1 else 0
        perf["detect_ms_sum"] += float(r.get("a_ms", 0.0)) + float(r.get("b_ms", 0.0))

        if not bool(r.get("valid", False)):
            continue
        all_obj_pts.append(r["obj"])
        all_img_pts_a.append(r["img_a"])
        all_img_pts_b.append(r["img_b"])
        valid_pairs.append((r["a"], r["b"]))
        valid_common_counts.append(int(r.get("common", 0)))

    return all_obj_pts, all_img_pts_a, all_img_pts_b, valid_pairs, valid_common_counts, perf


def main(argv: Optional[Sequence[str]] = None) -> int:
    """主函数"""
    parser = argparse.ArgumentParser(
        prog="python -m mcca.entry.step4_stereo_extrinsic",
        description="Step4：AprilTag 双目外参标定（默认仅关键指标，--verbose 打印矩阵细节）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/apriltag_config.json",
        help="配置文件路径（默认 config/apriltag_config.json）",
    )
    parser.add_argument(
        "--cameras",
        nargs=2,
        default=None,
        metavar=("CAM_A", "CAM_B"),
        help="参与双目标定的两个相机名（默认取 config.image_dataset.cameras 的前两个；若>2则必须显式指定）",
    )
    parser.add_argument(
        "--image_source",
        type=str,
        default="auto",
        choices=["auto", "filtered", "raw"],
        help="图像来源：auto(优先filtered)/filtered/raw（默认 auto）",
    )
    parser.add_argument(
        "--intrinsics_dir",
        type=str,
        default="results",
        help="内参目录（默认 results；内参文件名约定为 <cam>_intrinsics.json）",
    )
    parser.add_argument("--verbose", action="store_true", help="打印 R/t 矩阵、会聚几何等额外信息")
    parser.add_argument(
        "--min_common_tags",
        type=int,
        default=MIN_COMMON_TAGS_DEFAULT,
        help=f"质量过滤：每对图像至少需要的共同标签数（默认 {MIN_COMMON_TAGS_DEFAULT}）",
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
    parser.add_argument(
        "--max_valid_pairs",
        type=int,
        default=MAX_VALID_IMAGES,
        help=f"最多使用多少对有效图像进行标定（默认 {MAX_VALID_IMAGES}）",
    )
    parser.add_argument(
        "--min_tags_per_image",
        type=int,
        default=0,
        help=(
            "每张图像最少需要检测到的标签数（0=跟随 --min_common_tags）。"
            "用于避免单目检测很少标签导致的弱约束视角进入 stereoCalibrate。"
        ),
    )
    parser.add_argument(
        "--max_total_pairs",
        type=int,
        default=0,
        help="最多扫描多少对图像（0=不限制）。",
    )
    parser.add_argument(
        "--max_detect_seconds",
        type=float,
        default=0.0,
        help="扫描+检测阶段的总耗时上限（秒，0=不限制）。",
    )
    parser.add_argument(
        "--scan_strategy",
        type=str,
        default="sequential",
        choices=["sequential", "random", "uniform"],
        help="扫描顺序策略：sequential/random/uniform。",
    )
    parser.add_argument(
        "--scan_seed",
        type=int,
        default=0,
        help="random 扫描的随机种子（用于可复现）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行进程数（0=单进程；>0 启用多进程并行检测）。",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=0,
        help="并行模式预提交任务数（0=自动）。",
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
        help="关闭检测缓存（不推荐）。",
    )
    parser.add_argument(
        "--force_redetect",
        action="store_true",
        help="忽略缓存并强制重检（仍会写入新缓存）。",
    )
    parser.add_argument(
        "--prefilter",
        action="store_true",
        help="启用廉价预筛选（默认关闭，阈值偏保守）。",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    global VERBOSE
    VERBOSE = bool(args.verbose)

    print("=" * 60)
    print("Step 4: AprilTag 双目外参标定")
    print("=" * 60)

    # 加载配置
    config = load_config(str(args.config))

    cams_cfg = get_dataset_cameras(config, allow_scan=True)
    if args.cameras is not None:
        cam_a, cam_b = str(args.cameras[0]), str(args.cameras[1])
    else:
        if len(cams_cfg) < 2:
            print("\n错误: 未在 config.image_dataset 或 images/raw 下找到至少 2 个相机。")
            print("请检查 config.image_dataset.cameras，或准备 images/raw/<cam> 数据集目录。")
            return 2
        if len(cams_cfg) > 2:
            print("\n错误: 检测到超过 2 个相机，双目 Step4 无法自动决定使用哪两路。")
            print(f"请显式指定：--cameras <camA> <camB>。已检测相机: {cams_cfg}")
            return 2
        cam_a, cam_b = str(cams_cfg[0]), str(cams_cfg[1])
    use_multiscale, opencv_refine = get_detection_settings(config)
    if bool(getattr(args, "no_multiscale", False)):
        use_multiscale = False
    if bool(getattr(args, "no_opencv_refine", False)):
        opencv_refine = False

    # Step4 以 Step2 为准：复用同一套 detection profile / ROI / auto_roi / detector 参数
    profile = get_detection_profile(config)
    roi_a = get_detection_roi(config, camera=cam_a)
    roi_b = get_detection_roi(config, camera=cam_b)
    auto_roi_cfg = get_detection_auto_roi(config)
    detector_params = create_detector_params(config)

    # 加载内参
    intr_dir = Path(str(args.intrinsics_dir))
    intr_a_path = intr_dir / f"{cam_a}_intrinsics.json"
    intr_b_path = intr_dir / f"{cam_b}_intrinsics.json"
    if not intr_a_path.exists() or not intr_b_path.exists():
        print("\n错误: 未找到相机内参文件。")
        print(f"  - {cam_a}: {intr_a_path} ({'存在' if intr_a_path.exists() else '缺失'})")
        print(f"  - {cam_b}: {intr_b_path} ({'存在' if intr_b_path.exists() else '缺失'})")
        print("请先运行 Step3 生成 results/<cam>_intrinsics.json。")
        return 2

    print("\n加载内参...")
    K_a, dist_a = load_intrinsics(str(intr_a_path))
    K_b, dist_b = load_intrinsics(str(intr_b_path))
    print(f"  [OK] 相机内参已加载: {cam_a}, {cam_b}")

    # 创建 AprilTag 标定板
    obj_points, tag_ids = create_apriltag_board(config)
    aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])
    board = create_opencv_aruco_board(obj_points, tag_ids, aruco_dict)

    # 获取图像对：按 frame_key(stem) 同步匹配，避免仅靠排序 zip 导致“错配”。
    try:
        pairs, image_source, sync_diag = _collect_synced_pairs(
            config=config,
            cam_a=cam_a,
            cam_b=cam_b,
            image_source=str(args.image_source),
        )
    except Exception as e:
        print("\n错误: 读取/配对图像失败")
        print(f"  - {e}")
        return 2

    print(f"\n相机: {cam_a} + {cam_b}")
    print(f"图像来源: {image_source}")
    print(f"每相机图片数: {sync_diag.get('per_cam_images')}")
    print(f"可配对图像对数(frame_key 交集): {sync_diag.get('intersection_pairs')}")
    if int(sync_diag.get("intersection_pairs", 0)) == 0:
        print("\n错误: 无法找到任何可配对的同帧图像（frame_key 交集为 0）。")
        print("这通常是文件命名/同步规则不一致导致：脚本以文件名 stem 作为‘同一时刻’判据。")
        print(f"sample frame_key: {sync_diag.get('sample_frame_keys')}")
        return 2

    print("\n扫描并收集双目对应点...")
    if VERBOSE:
        print(f"  - detection profile: {profile}")
        if roi_a is not None or roi_b is not None:
            print(f"  - ROI({cam_a}): {roi_a}")
            print(f"  - ROI({cam_b}): {roi_b}")
        if bool((auto_roi_cfg or {}).get("enabled", False)):
            print(
                "  - auto_roi: enabled "
                f"(pre_scale={auto_roi_cfg.get('pre_scale')}, min_tags={auto_roi_cfg.get('min_tags')}, margin={auto_roi_cfg.get('margin')})"
            )

    min_common_tags = int(max(1, int(args.min_common_tags)))
    if int(args.min_tags_per_image) <= 0:
        min_tags_per_image = int(min_common_tags)
    else:
        min_tags_per_image = int(max(1, int(args.min_tags_per_image)))

    # 候选 pairs 使用“已同步匹配”的列表

    limits = ScanLimits(
        target_valid=int(args.max_valid_pairs),
        max_total=int(args.max_total_pairs),
        max_seconds=float(args.max_detect_seconds),
    )
    order = ScanOrder(strategy=str(args.scan_strategy), seed=int(args.scan_seed))

    cache_cfg = CacheConfig(
        enabled=not bool(args.no_cache),
        cache_dir=str(args.cache_dir),
        force_redetect=bool(args.force_redetect),
    )
    prefilter_cfg = PrefilterConfig(enabled=bool(args.prefilter))

    algo_key = _build_algo_key(config, profile=str(profile))

    if int(args.workers) > 0:
        parallel_state = {
            "config": config,
            "profile": str(profile),
            "family": str(config["apriltag_board"]["family"]),
            "obj_points": np.asarray(obj_points, dtype=np.float32),
            "tag_ids": [int(x) for x in tag_ids],
            "K_a": np.asarray(K_a, dtype=np.float64),
            "dist_a": np.asarray(dist_a, dtype=np.float64),
            "K_b": np.asarray(K_b, dtype=np.float64),
            "dist_b": np.asarray(dist_b, dtype=np.float64),
            "roi_a": list(roi_a) if roi_a is not None else None,
            "roi_b": list(roi_b) if roi_b is not None else None,
            "auto_roi_cfg": auto_roi_cfg or {},
            "use_multiscale": bool(use_multiscale),
            "opencv_refine": bool(opencv_refine),
            "min_common_tags": int(min_common_tags),
            "min_tags_per_image": int(min_tags_per_image),
            "cache": {
                "enabled": bool(cache_cfg.enabled),
                "cache_dir": str(cache_cfg.cache_dir),
                "force_redetect": bool(cache_cfg.force_redetect),
            },
            "prefilter": {
                "enabled": bool(prefilter_cfg.enabled),
            },
        }

        all_obj_pts, all_img_pts_a, all_img_pts_b, valid_pairs, valid_common_counts, perf = (
            scan_and_collect_stereo_points(
                pairs,
                obj_points_all=np.asarray(obj_points, dtype=np.float32),
                tag_ids=[int(x) for x in tag_ids],
                min_common_tags=int(min_common_tags),
                min_tags_per_image=int(min_tags_per_image),
                limits=limits,
                order=order,
                max_workers=int(args.workers),
                prefetch=int(args.prefetch),
                det_a=None,
                det_b=None,
                parallel_state=parallel_state,
            )
        )
    else:
        det_a = CachedAprilTagDetector(
            aruco_dict=aruco_dict,
            detector_params=detector_params,
            algo_key=algo_key,
            use_multiscale=bool(use_multiscale),
            opencv_refine=bool(opencv_refine),
            board=board,
            camera_matrix=K_a,
            dist_coeffs=dist_a,
            roi=roi_a,
            auto_roi=bool((auto_roi_cfg or {}).get("enabled", False)),
            auto_roi_pre_scale=float((auto_roi_cfg or {}).get("pre_scale", 0.5)),
            auto_roi_min_tags=int((auto_roi_cfg or {}).get("min_tags", 1)),
            auto_roi_margin=float((auto_roi_cfg or {}).get("margin", 0.25)),
            cache_cfg=cache_cfg,
            prefilter_cfg=prefilter_cfg,
        )
        det_b = CachedAprilTagDetector(
            aruco_dict=aruco_dict,
            detector_params=detector_params,
            algo_key=algo_key,
            use_multiscale=bool(use_multiscale),
            opencv_refine=bool(opencv_refine),
            board=board,
            camera_matrix=K_b,
            dist_coeffs=dist_b,
            roi=roi_b,
            auto_roi=bool((auto_roi_cfg or {}).get("enabled", False)),
            auto_roi_pre_scale=float((auto_roi_cfg or {}).get("pre_scale", 0.5)),
            auto_roi_min_tags=int((auto_roi_cfg or {}).get("min_tags", 1)),
            auto_roi_margin=float((auto_roi_cfg or {}).get("margin", 0.25)),
            cache_cfg=cache_cfg,
            prefilter_cfg=prefilter_cfg,
        )

        all_obj_pts, all_img_pts_a, all_img_pts_b, valid_pairs, valid_common_counts, perf = (
            scan_and_collect_stereo_points(
                pairs,
                obj_points_all=np.asarray(obj_points, dtype=np.float32),
                tag_ids=[int(x) for x in tag_ids],
                min_common_tags=int(min_common_tags),
                min_tags_per_image=int(min_tags_per_image),
                limits=limits,
                order=order,
                max_workers=0,
                prefetch=0,
                det_a=det_a,
                det_b=det_b,
                parallel_state=None,
            )
        )

    print(f"  - 扫描: {perf['pairs_scanned']}/{len(pairs)} 对, valid: {len(valid_pairs)}")
    print(
        "  - 缓存命中(A/B): "
        f"{perf['a_cache_hit']}/{perf['pairs_scanned']} / {perf['b_cache_hit']}/{perf['pairs_scanned']}"
    )
    if bool(args.prefilter):
        print(
            "  - 预筛选跳过(A/B): "
            f"{perf['a_prefilter_skipped']} / {perf['b_prefilter_skipped']}"
        )
    if int(perf["a_error"]) > 0 or int(perf["b_error"]) > 0:
        print(f"  - 读取/检测异常(A/B): {perf['a_error']} / {perf['b_error']}")
    print(f"  - 检测阶段耗时: {perf['elapsed_s']:.2f} s (sum detect {perf['detect_ms_sum'] / 1000.0:.2f} s)")

    if len(valid_common_counts) > 0:
        print("\n有效图像对质量(共同标签数):")
        print(f"  平均: {np.mean(valid_common_counts):.1f} 个")
        print(f"  范围: {int(np.min(valid_common_counts))} - {int(np.max(valid_common_counts))} 个")
        print(f"  标准差: {np.std(valid_common_counts):.1f} 个")

    if len(valid_pairs) < 10:
        print(f"\n警告: 只有 {len(valid_pairs)} 对有效图像")
        print("   会聚式双目建议至少 20 对优质图像以获得最佳标定质量")

    if len(valid_pairs) < 5:
        print(f"\n错误: 有效双目图像对太少 ({len(valid_pairs)} < 5)")
        print("\n建议:")
        print("   1. 重新采集更多图像，确保标定板在两相机重叠视野中央")
        print(f"   2. 每对图像应至少检测到 {min_common_tags} 个共同标签")
        return 2

    # 获取图像尺寸
    img = cv2.imread(valid_pairs[0][0])
    if img is None:
        print("\n错误: 无法读取用于获取 image_size 的图像")
        return 2
    image_size = (img.shape[1], img.shape[0])

    # 执行双目标定
    print("\n执行双目标定...")

    ret, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(
        all_obj_pts,
        all_img_pts_a,
        all_img_pts_b,
        K_a,
        dist_a,
        K_b,
        dist_b,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,  # 固定内参，只优化外参
    )

    baseline = np.linalg.norm(t)

    # Step4 的主要误差口径：mean reprojection error（与 step3 保存的 reprojection_error 一致）
    mean_stereo_all, mean_stereo_a, mean_stereo_b, used_stereo = (
        compute_stereo_mean_reproj_error_using_rt(
            all_obj_pts,
            all_img_pts_a,
            all_img_pts_b,
            K_a,
            dist_a,
            K_b,
            dist_b,
            R,
            t,
        )
    )

    print(f"  - 重投影误差(mean, stereo 约束, 左右合计): {mean_stereo_all:.4f} 像素")
    print(f"    - {cam_a} mean: {mean_stereo_a:.4f} 像素")
    print(f"    - {cam_b} mean: {mean_stereo_b:.4f} 像素")
    print(f"    - used pairs: {used_stereo}/{len(all_obj_pts)}")
    print(f"  - OpenCV ret (RMS, stereoCalibrate 返回): {ret:.4f} 像素")
    print(f"  - 基线距离: {baseline:.2f} mm ({baseline / 10:.2f} cm)")

    mean_err_b_pnp, _, used_b_pnp = compute_mean_reproj_error_pnp(
        all_obj_pts, all_img_pts_b, K_b, dist_b
    )
    print(
        f"  - {cam_b} mean_error(PnP, per-view mean): {mean_err_b_pnp:.4f} px "
        f"(used {used_b_pnp}/{len(all_obj_pts)})"
    )

    # （可选）右目 mean_error：使用 stereo 的 R,t（左PnP + R,t 预测右目位姿）
    mean_err_b_rt, _, used_b_rt = compute_cam_b_mean_reproj_error_using_rt(
        all_obj_pts,
        all_img_pts_a,
        all_img_pts_b,
        K_a,
        dist_a,
        K_b,
        dist_b,
        R,
        t,
    )
    print(
        f"  - {cam_b} mean_error(CamA PnP + R,t 预测): {mean_err_b_rt:.4f} px "
        f"(used {used_b_rt}/{len(all_obj_pts)})"
    )

    # 质量评估
    if mean_stereo_all < 1.5:
        print(f"  标定质量: 优秀（mean < 1.5 px）")
    elif mean_stereo_all < 3.0:
        print(f"  标定质量: 良好（mean < 3.0 px）")
    elif mean_stereo_all < 5.0:
        print(f"  标定质量: 一般（mean < 5.0 px）")
    else:
        print(f"  标定质量: 较差（mean ≥ 5.0 px）")
        print(f"  建议: 采集更多高质量图像（每对 ≥20 共同标签）")

    # 保存双目外参
    used_pairs_payload = [
        {
            "cam_a": os.path.relpath(ap).replace("\\", "/"),
            "cam_b": os.path.relpath(bp).replace("\\", "/"),
        }
        for ap, bp in valid_pairs
    ]
    extrinsics = {
        # 元信息：用于 verify 精确复现 Step4 的数据集/筛选口径
        "camera_a": str(cam_a),
        "camera_b": str(cam_b),
        "cameras": [str(cam_a), str(cam_b)],
        "image_source": str(image_source),
        "quality_filter_min_common_tags": int(min_common_tags),
        "min_tags_per_image": int(min_tags_per_image),
        "max_valid_images": int(args.max_valid_pairs),
        "used_pairs": used_pairs_payload,
        "note": "R,t 满足: X_cam_b = R * X_cam_a + t",

        "R": R.tolist(),
        "t": t.flatten().tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "baseline": float(baseline),
        # 统一口径：reprojection_error 保存 mean reprojection error（便于与 step3 对比）
        "reprojection_error": float(mean_stereo_all),
        "reprojection_error_cam_a_mean": float(mean_stereo_a),
        "reprojection_error_cam_b_mean": float(mean_stereo_b),
        "reprojection_error_used_pairs": int(used_stereo),
        # 保留 OpenCV 原始返回值（RMS）以供需要时排查
        "opencv_ret_rms": float(ret),

        "mean_reprojection_error_cam_b_pnp": float(mean_err_b_pnp),
        "mean_reprojection_error_cam_b_pnp_used_pairs": int(used_b_pnp),

        "mean_reprojection_error_cam_b_pred_using_rt": float(mean_err_b_rt),
        "mean_reprojection_error_cam_b_pred_using_rt_used_pairs": int(used_b_rt),
    }

    with open("results/stereo_extrinsics.json", "w") as f:
        json.dump(extrinsics, f, indent=2)

    print("  [OK] 已保存: results/stereo_extrinsics.json")


    # 立体校正
    print("\n执行立体校正...")

    R1, R2, P1, P2, Q, roi_a, roi_b = cv2.stereoRectify(
        K_a,
        dist_a,
        K_b,
        dist_b,
        image_size,
        R,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,  # 保留所有像素
    )

    # 保存立体校正参数
    rectification = {
        "camera_a": str(cam_a),
        "camera_b": str(cam_b),
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
        "roi_a": list(roi_a),
        "roi_b": list(roi_b),
    }

    with open("results/stereo_rectification.json", "w") as f:
        json.dump(rectification, f, indent=2)

    print("  [OK] 已保存: results/stereo_rectification.json")

    # 生成立体校正效果图
    print("\n生成立体校正效果图...")

    # 使用第一对图像
    img_a = cv2.imread(valid_pairs[0][0])
    img_b = cv2.imread(valid_pairs[0][1])
    if img_a is None or img_b is None:
        print("\n错误: 无法读取用于立体校正 demo 的图像")
        return 2

    # 计算映射
    map1_a, map2_a = cv2.initUndistortRectifyMap(
        K_a, dist_a, R1, P1, image_size, cv2.CV_32FC1
    )
    map1_b, map2_b = cv2.initUndistortRectifyMap(
        K_b, dist_b, R2, P2, image_size, cv2.CV_32FC1
    )

    # 应用校正
    rect_a = cv2.remap(img_a, map1_a, map2_a, cv2.INTER_LINEAR)
    rect_b = cv2.remap(img_b, map1_b, map2_b, cv2.INTER_LINEAR)

    # 创建对比图，绘制水平线
    demo = np.hstack([rect_a, rect_b])

    # 绘制水平辅助线（每隔一定距离）
    for y in range(0, demo.shape[0], 50):
        cv2.line(demo, (0, y), (demo.shape[1], y), (0, 255, 0), 1)

    # 添加标题
    cv2.putText(
        demo,
        f"{cam_a} (Rectified)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        demo,
        f"{cam_b} (Rectified)",
        (image_size[0] + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        2,
    )

    _safe_imwrite("results/stereo_rectification_demo.jpg", demo)
    print("  [OK] 已保存: results/stereo_rectification_demo.jpg")

    # 显示总结
    print("\n" + "=" * 60)
    print("双目外参标定完成！")
    print("=" * 60)
    print(f"\n双目参数:")
    print(f"  - 基线距离: {baseline:.2f} mm ({baseline / 10:.2f} cm)")
    print(f"  - 重投影误差(mean): {mean_stereo_all:.4f} 像素")
    print(f"  - OpenCV ret (RMS): {ret:.4f} 像素")
    print(f"  - 使用图像对: {len(valid_pairs)}")

    if VERBOSE:
        print(f"\n旋转矩阵 R ({cam_a}->{cam_b}; X_{cam_b} = R * X_{cam_a} + t):")
        print(R)

        print(f"\n平移向量 t ({cam_a}->{cam_b}, 单位:mm):")
        print(t.flatten())
    else:
        print("\n(提示) 使用 --verbose 可打印 R/t 矩阵与会聚几何细节")

    # 会聚式双目特殊说明
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_matrix(R)
    euler = rot.as_euler("xyz", degrees=True)
    convergence_angle = abs(euler[1])

    if VERBOSE:
        print(f"\n会聚式双目几何:")
        print(f"  - 会聚角: {convergence_angle:.2f}°")
        print(f"  - 垂直偏移: {t.flatten()[1]:.2f} mm")
        print(f"  - 前后偏移: {t.flatten()[2]:.2f} mm")

    print("\n质量检查:")
    print("  - 打开 results/stereo_rectification_demo.jpg")
    print("  - 确认左右图像的水平线对齐")
    print("  - 如果对齐良好，说明标定成功")

    print("\n下一步: (可选) 进行 Step5 相机->底盘标定")
    print(f"  1) 准备 images/step5/{cam_a} 与 images/step5/{cam_b} 的图像对（可由视频抽帧获得）")
    print("  2) 运行: python -m mcca.entry.step5b_camera_to_base")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
