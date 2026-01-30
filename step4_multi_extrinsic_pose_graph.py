#!/usr/bin/env python3
"""Step 4 (Multi-Cam): 多相机外参标定（位姿图优化 / Pose Graph）

核心思想：
- 对每个时间戳(同名帧)的每个相机，用 AprilTag 标定板做 PnP，得到该帧的 C_T_B（Board->Camera）。
- 如果同一帧里相机 i 和 j 都成功 PnP，则得到一条相机间相对位姿观测：

    Z_ij = Ci_T_Cj = (Ci_T_B) * inv(Cj_T_B)

- 将所有 Z_ij 组成位姿图（节点=相机，边=相对位姿观测），对所有相机位姿 X_i 做最小二乘优化：

    Z_ij ≈ X_i * inv(X_j)

  其中 X_ref 固定为单位阵，ref 为参考相机。

输入约定（推荐）：
- 图像：images/<source>/<cam_name>/*.png|*.jpg
  - <source> 可用 images/filtered（默认）或 images/raw
  - 不要求所有相机都能看到板：只要存在一些帧产生边，且整体图连通即可。
  - “同步”的含义：不同相机同一时刻的帧文件名 stem 相同。
    例如：
      images/filtered/cam0/frame_000120.png
      images/filtered/cam1/frame_000120.png
      images/filtered/cam2/frame_000120.png

- 内参：results/<cam_name>_intrinsics.json
    - 相机名与 Step3 输出保持一致：results/<cam>_intrinsics.json。

输出：
- results/multi_camera_extrinsics.json
  - 每个相机相对 reference 的 4x4 齐次变换矩阵（Cam_i <- Cam_ref）
- results/multi_camera_pose_graph_report.json
  - 图连通性、每对相机边数量、优化收敛信息等报告

使用示例：
- 直接扫描 images/filtered 下的相机目录：
    python step4_multi_extrinsic_pose_graph.py

- 手动指定相机列表与参考相机：
    python step4_multi_extrinsic_pose_graph.py --cameras cam0 cam1 cam2 cam3 --reference cam0

常见失败：
- 图不连通：相机之间没有足够“同帧共同看到板”的边，无法把所有相机标到同一参考系。
- 内参缺失：没有对应的 results/<cam>_intrinsics.json。
- 标签太少：min_tags 设太大/光照太差导致 PnP 失败，边数量不足。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from utils import (
    create_apriltag_board,
    create_detector_params,
    create_opencv_aruco_board,
    get_aruco_dict,
    get_detection_auto_roi,
    get_detection_profile,
    get_detection_roi,
    get_detection_settings,
    load_config,
)

from libs.apriltag_perf.cache import CacheConfig
from libs.apriltag_perf.prefilter import PrefilterConfig
from libs.apriltag_perf.scan import ScanLimits, ScanOrder, iter_scan_parallel_ordered, iter_scan_sequential
from libs.apriltag_perf.service import CachedAprilTagDetector

VERBOSE: bool = True


def _to_rvec_tvec_from_T(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将 4x4 齐次变换转换为 OpenCV 可用的 (rvec, tvec)。

    说明：
    - OpenCV projectPoints/solvePnP 使用 rvec(3) + tvec(3,1)。
    - 这里约定 tvec 的单位与 obj_pts 一致（本工程通常为 mm）。
    """

    T = np.asarray(T, dtype=np.float64)
    _ensure_T(T, "to_rvec_tvec")
    Rm = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(Rm)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    return rvec, t


_G_DET_BY_CAM: Optional[Dict[str, CachedAprilTagDetector]] = None
_G_FRAME_MAPS: Optional[Dict[str, Dict[str, str]]] = None
_G_OBJ_POINTS_ALL: Optional[np.ndarray] = None
_G_TAG_ID_TO_IDX: Optional[Dict[int, int]] = None
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


def _vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


def _ensure_T(T: np.ndarray, name: str) -> None:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: 期望 (4,4)，得到 {T.shape}")
    if float(np.linalg.norm(T[3, :] - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-8:
        raise ValueError(f"{name}: 底行必须是 [0 0 0 1]，当前={T[3, :]}")


def _make_T(Rm: np.ndarray, t: np.ndarray, name: str) -> np.ndarray:
    Rm = np.asarray(Rm, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t
    _ensure_T(T, name)
    return T


def _inv_T(T: np.ndarray, name: str) -> np.ndarray:
    """解析求逆（比 np.linalg.inv 更稳更快），并保留自检。"""
    T = np.asarray(T, dtype=np.float64)
    _ensure_T(T, name + ".in")
    Rm = T[:3, :3]
    t = T[:3, 3]
    Rm_inv = Rm.T
    t_inv = -Rm_inv @ t
    T_inv = _make_T(Rm_inv, t_inv, name)
    return T_inv


def _skew(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array(
        [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=np.float64
    )


def _so3_exp(w: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(w, dtype=np.float64).reshape(3)).as_matrix()


def _so3_log(Rm: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(Rm, dtype=np.float64)).as_rotvec()


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    """se(3) -> SE(3)

    xi = [w(3), v(3)]
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    w = xi[:3]
    v = xi[3:]

    theta = float(np.linalg.norm(w))
    Rm = _so3_exp(w)

    W = _skew(w)
    I = np.eye(3, dtype=np.float64)

    if theta < 1e-8:
        V = I + 0.5 * W
    else:
        A = float(np.sin(theta) / theta)
        B = float((1.0 - np.cos(theta)) / (theta * theta))
        C = float((theta - np.sin(theta)) / (theta**3))
        V = I + B * W + C * (W @ W)

    t = V @ v
    return _make_T(Rm, t, "se3_exp")


def _se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) -> se(3) (6D)

    返回 xi = [w(3), v(3)]
    """
    T = np.asarray(T, dtype=np.float64)
    _ensure_T(T, "se3_log")

    Rm = T[:3, :3]
    t = T[:3, 3]

    w = _so3_log(Rm)
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    I = np.eye(3, dtype=np.float64)

    if theta < 1e-8:
        V_inv = I - 0.5 * W
    else:
        A = float(np.sin(theta) / theta)
        B = float((1.0 - np.cos(theta)) / (theta * theta))
        # 经典公式：V^{-1} = I - 0.5 W + (1/theta^2) * (1 - A/(2B)) * W^2
        coef = float((1.0 / (theta * theta)) * (1.0 - A / (2.0 * B)))
        V_inv = I - 0.5 * W + coef * (W @ W)

    v = V_inv @ t
    xi = np.zeros(6, dtype=np.float64)
    xi[:3] = w
    xi[3:] = v
    return xi


@dataclass(frozen=True)
class CameraIntrinsics:
    name: str
    K: np.ndarray
    dist: np.ndarray


@dataclass(frozen=True)
class PoseObs:
    """单帧单相机 PnP 观测。"""

    cam: str
    frame_key: str
    C_T_B: np.ndarray
    n_tags: int
    reproj_mean_px: float
    obj_pts: np.ndarray
    img_pts: np.ndarray


@dataclass(frozen=True)
class EdgeObs:
    """位姿图边观测：Ci <- Cj"""

    cam_i: str
    cam_j: str
    Ci_T_Cj: np.ndarray
    weight: float
    frame_key: str


def _fmt_path_short(p: Optional[Path]) -> str:
    """用于控制台诊断输出的短路径显示（避免一行太长）。"""

    if p is None:
        return ""
    try:
        parts = list(p.parts)
        if len(parts) >= 3:
            return str(Path(parts[-3]) / parts[-2] / parts[-1])
        if len(parts) == 2:
            return str(Path(parts[-2]) / parts[-1])
        return str(p)
    except Exception:
        return str(p)


def _load_intrinsics(cam: str, intrinsics_dir: Path) -> CameraIntrinsics:
    """加载单个相机内参。

    约定：results/<cam>_intrinsics.json。
    """

    path = intrinsics_dir / f"{cam}_intrinsics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到相机内参文件：{path}。\n"
            "请先为每个相机生成内参结果（schema 与 step3 输出一致：camera_matrix/dist_coeffs）。"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return CameraIntrinsics(name=cam, K=K, dist=dist)


def _list_cameras(image_root: Path, cameras_arg: Optional[Sequence[str]]) -> List[str]:
    if cameras_arg:
        return [str(c) for c in cameras_arg]

    # 自动扫描子目录作为相机名（排除 __pycache__ 等）
    cams: List[str] = []
    for p in sorted(image_root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        # 说明：images/filtered 下经常会放一些说明目录（如 _comment），不应当被当作相机。
        if name.startswith(".") or name.startswith("__") or name.startswith("_"):
            continue
        cams.append(name)

    if len(cams) == 0:
        raise FileNotFoundError(
            f"未在 {image_root} 下找到相机子目录。\n"
            "建议目录结构：images/filtered/cam0/*.png, images/filtered/cam1/*.png ..."
        )

    return cams


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _normalize_frame_key(stem: str, cam: str) -> str:
    """从文件名 stem 生成用于“同帧匹配”的 frame_key。

    现象与动机：
    - 本脚本默认用“不同相机文件名 stem 完全相同”来表示同一时刻的帧。
    - 但很多数据集会把相机名写进文件名前缀，例如：
        cam1_seq000015_f16.bmp
        cam3_seq000015_f16.bmp
      这两张其实是同一帧，但 stem 不同，导致候选同帧交集为 0，从而永远构不出边。

    这里做一个保守的自动归一化：
    - 若 stem 以 "{cam}_" 或 "{cam}-" 开头，则去掉该前缀。
    - 否则保持原 stem，不改变原有“同名帧”行为。
    """

    stem = str(stem)
    cam = str(cam)
    if stem.startswith(cam + "_"):
        return stem[len(cam) + 1 :]
    if stem.startswith(cam + "-"):
        return stem[len(cam) + 1 :]
    return stem


def _build_frame_map(cam_dir: Path) -> Dict[str, Path]:
    """构建 frame_key -> Path 的映射。

    注意：这里不用 Path.glob("*.JPG") 等大小写组合，而是统一用 suffix.lower()。
    这样能兼容 Windows 上常见的 .BMP/.JPG 大写后缀。
    """

    out: Dict[str, Path] = {}
    cam = cam_dir.name
    dup = 0
    for p in sorted(cam_dir.iterdir()):
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


def _collect_correspondences(
    *,
    corners: List[np.ndarray],
    ids: np.ndarray,
    obj_points: np.ndarray,
    tag_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """根据检测到的 tags，组装 solvePnP 需要的 (object_points, image_points)。"""
    ids_flat = ids.flatten()
    object_points = []
    image_points = []

    used_tags = 0
    for i, tag_id in enumerate(ids_flat):
        tag_id_int = int(tag_id)
        if tag_id_int not in tag_ids:
            continue
        idx = tag_ids.index(tag_id_int)
        object_points.append(obj_points[idx])
        image_points.append(corners[i].reshape(-1, 2))
        used_tags += 1

    if used_tags == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), 0

    obj = np.vstack(object_points).astype(np.float32)
    img = np.vstack(image_points).astype(np.float32)
    return obj, img, used_tags


def _collect_correspondences_fast(
    *,
    corners: List[np.ndarray],
    ids: np.ndarray,
    obj_points: np.ndarray,
    tag_id_to_idx: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """根据检测到的 tags，组装 solvePnP 需要的 (object_points, image_points)。

    说明：
    - 这里使用 tag_id_to_idx 加速映射，避免对 tag_ids 做 O(n) index 查找。
    """

    ids_flat = np.asarray(ids).reshape(-1)
    object_points: List[np.ndarray] = []
    image_points: List[np.ndarray] = []

    used_tags = 0
    for i, tag_id in enumerate(ids_flat.tolist()):
        idx = tag_id_to_idx.get(int(tag_id))
        if idx is None:
            continue
        object_points.append(np.asarray(obj_points[idx], dtype=np.float32))
        image_points.append(np.asarray(corners[i], dtype=np.float32).reshape(-1, 2))
        used_tags += 1

    if used_tags == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), 0

    obj = np.vstack(object_points).astype(np.float32, copy=False)
    img = np.vstack(image_points).astype(np.float32, copy=False)
    return obj, img, used_tags


def _init_multicam_frame_worker(state: Dict[str, Any]) -> None:
    """多进程 worker 初始化：在子进程内创建 OpenCV 检测器与缓存服务。

    说明：
    - Windows 下 multiprocessing 采用 spawn，OpenCV 对象不可 pickle，必须在子进程初始化。
    - 每个 worker 内部会创建“每相机一个 CachedAprilTagDetector”。
    """

    global _G_DET_BY_CAM, _G_FRAME_MAPS, _G_OBJ_POINTS_ALL, _G_TAG_ID_TO_IDX, _G_MIN_TAGS

    config = state["config"]
    cameras = [str(x) for x in state["cameras"]]

    _G_FRAME_MAPS = state["frame_maps"]

    obj_points = np.asarray(state["obj_points"], dtype=np.float32)
    tag_ids = [int(x) for x in state["tag_ids"]]
    _G_OBJ_POINTS_ALL = obj_points
    _G_TAG_ID_TO_IDX = {int(t): i for i, t in enumerate(tag_ids)}

    aruco_dict = get_aruco_dict(str(state["family"]))
    board = create_opencv_aruco_board(obj_points, tag_ids, aruco_dict)
    detector_params = create_detector_params(config)

    cache_cfg = CacheConfig(
        enabled=bool(state["cache"]["enabled"]),
        cache_dir=str(state["cache"]["cache_dir"]),
        force_redetect=bool(state["cache"]["force_redetect"]),
    )
    prefilter_cfg = PrefilterConfig(enabled=bool(state["prefilter"]["enabled"]))

    auto_roi_cfg = state.get("auto_roi_cfg") or {}
    use_multiscale = bool(state["use_multiscale"])
    opencv_refine = bool(state["opencv_refine"])
    algo_key = state["algo_key"]

    _G_DET_BY_CAM = {}
    roi_by_cam = state.get("roi_by_cam") or {}
    intr_by_cam = state.get("intr_by_cam") or {}

    for cam in cameras:
        intr = intr_by_cam[cam]
        _G_DET_BY_CAM[cam] = CachedAprilTagDetector(
            aruco_dict=aruco_dict,
            detector_params=detector_params,
            algo_key=algo_key,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            camera_matrix=np.asarray(intr["K"], dtype=np.float64),
            dist_coeffs=np.asarray(intr["dist"], dtype=np.float64),
            roi=tuple(roi_by_cam[cam]) if roi_by_cam.get(cam) is not None else None,
            auto_roi=bool(auto_roi_cfg.get("enabled", False)),
            auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
            auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
            auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
            cache_cfg=cache_cfg,
            prefilter_cfg=prefilter_cfg,
        )

    _G_MIN_TAGS = int(state["min_tags"])


def _multicam_frame_worker(frame_key: str) -> Dict[str, Any]:
    """多进程扫描的单任务：对某个 frame_key 的多相机图像做检测与 PnP。"""

    global _G_DET_BY_CAM, _G_FRAME_MAPS, _G_OBJ_POINTS_ALL, _G_TAG_ID_TO_IDX, _G_MIN_TAGS

    if _G_DET_BY_CAM is None or _G_FRAME_MAPS is None or _G_OBJ_POINTS_ALL is None or _G_TAG_ID_TO_IDX is None:
        return {
            "frame_key": str(frame_key),
            "poses_by_cam": {},
            "paths_by_cam": {},
            "det_by_cam": {},
            "valid": False,
            "error": "worker 未初始化",
        }

    poses_by_cam: Dict[str, Dict[str, Any]] = {}
    paths_by_cam: Dict[str, str] = {}
    det_by_cam: Dict[str, Dict[str, Any]] = {}

    for cam, det in _G_DET_BY_CAM.items():
        p = (_G_FRAME_MAPS.get(cam) or {}).get(str(frame_key))
        if p is None:
            continue

        paths_by_cam[cam] = str(p)
        res = det.detect_path(Path(p))

        ids = np.asarray(res.ids) if res.ids is not None else np.zeros((0, 1), dtype=np.int32)
        corners = res.corners or []
        n_tags = int(ids.shape[0])

        det_by_cam[cam] = {
            "from_cache": bool(res.from_cache),
            "status": int(res.status),
            "elapsed_ms": float(res.elapsed_ms),
            "n_tags": int(n_tags),
        }

        if n_tags < int(_G_MIN_TAGS):
            continue

        try:
            obj_pts, img_pts, used_tags = _collect_correspondences_fast(
                corners=list(corners),
                ids=ids,
                obj_points=_G_OBJ_POINTS_ALL,
                tag_id_to_idx=_G_TAG_ID_TO_IDX,
            )
            if used_tags < int(_G_MIN_TAGS):
                continue

            K = np.asarray(det.camera_matrix, dtype=np.float64)
            dist = np.asarray(det.dist_coeffs, dtype=np.float64)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                det_by_cam[cam]["pnp_ok"] = False
                continue
            det_by_cam[cam]["pnp_ok"] = True

            Rm, _ = cv2.Rodrigues(rvec)
            t = np.asarray(tvec, dtype=np.float64).reshape(3)
            C_T_B = _make_T(Rm, t, "C_T_B")

            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            proj = proj.reshape(-1, 2)
            det_pts = img_pts.reshape(-1, 2).astype(np.float64)
            per_pt = np.linalg.norm(det_pts - proj, axis=1)
            reproj_mean = float(np.mean(per_pt)) if per_pt.size > 0 else float("inf")

            poses_by_cam[cam] = {
                "C_T_B": np.asarray(C_T_B, dtype=np.float64).tolist(),
                "n_tags": int(used_tags),
                "reproj_mean_px": float(reproj_mean),
                # 说明：用于后续“外参优化完成后”的一致性重投影误差评估。
                # 这里存的是 solvePnP 参与拟合的点集（已按 ids/corners 展开到角点级）。
                "obj_pts": np.asarray(obj_pts, dtype=np.float32).tolist(),
                "img_pts": np.asarray(img_pts, dtype=np.float32).tolist(),
            }
        except Exception as e:
            det_by_cam[cam]["pnp_error"] = repr(e)
            continue

    valid = len(poses_by_cam) >= 2
    return {
        "frame_key": str(frame_key),
        "poses_by_cam": poses_by_cam,
        "paths_by_cam": paths_by_cam,
        "det_by_cam": det_by_cam,
        "valid": bool(valid),
    }


def _scan_pose_observations(
    *,
    image_root: Path,
    cameras: Sequence[str],
    intrinsics: Dict[str, CameraIntrinsics],
    config: dict,
    max_frames: int,
    min_tags: int,
    scan_limits: ScanLimits,
    scan_order: ScanOrder,
    workers: int,
    prefetch: int,
    cache_cfg: CacheConfig,
    prefilter_cfg: PrefilterConfig,
    use_multiscale: bool,
    opencv_refine: bool,
    stop_when_connected: bool,
    stop_min_edges: int,
    max_reproj_mean_px: float,
) -> Tuple[Dict[str, Dict[str, PoseObs]], Dict[str, Dict[str, Path]], List[EdgeObs], Dict[str, Any]]:
    """扫描并构建 pose/edges。

    返回：
    - poses_by_frame[frame_key][cam] = PoseObs
    - paths_by_frame[frame_key][cam] = Path
    - edges: 位姿图边观测列表
    - scan_report: 检测/扫描统计（用于落盘）
    """

    profile = get_detection_profile(config)
    auto_roi_cfg = get_detection_auto_roi(config)

    obj_points_mm, tag_ids = create_apriltag_board(config)
    family = config["apriltag_board"]["family"]

    # 每个相机：frame_key -> path
    frame_maps_p: Dict[str, Dict[str, Path]] = {}
    for cam in cameras:
        cam_dir = image_root / cam
        if not cam_dir.exists():
            raise FileNotFoundError(f"未找到相机目录：{cam_dir}")
        frame_maps_p[cam] = _build_frame_map(cam_dir)

    # 诊断：每对相机同名帧交集数量（不看检测，只看命名是否能对齐）。
    # 说明：若交集很小/为 0，则后续很难形成边；若交集很大但误差仍大，则更像“同名帧不同步”。
    pair_intersections: Dict[str, Any] = {}
    cams_list = list(cameras)
    for i in range(len(cams_list)):
        for j in range(i + 1, len(cams_list)):
            a = str(cams_list[i])
            b = str(cams_list[j])
            inter = set(frame_maps_p[a].keys()) & set(frame_maps_p[b].keys())
            pair_intersections[f"{a}__{b}"] = {
                "n": int(len(inter)),
                "sample": sorted(list(inter))[:10],
            }

    # ===== 帧同步 / 同帧匹配诊断 =====
    per_cam_frames = {str(cam): int(len(mp)) for cam, mp in frame_maps_p.items()}
    _vprint(f"每相机可用图片数量(frame_key 去重后): {per_cam_frames}")

    # 候选 frame_key：至少要有 2 个相机存在该帧，否则不可能形成边。

    # 候选 frame_key：至少要有 2 个相机存在该帧，否则不可能形成边。
    key_counts: Dict[str, int] = defaultdict(int)
    for cam in cameras:
        for k in frame_maps_p[cam].keys():
            key_counts[str(k)] += 1

    keys = sorted([k for k, n in key_counts.items() if int(n) >= 2])
    if int(max_frames) > 0:
        keys = keys[: int(max_frames)]

    # 当 keys=0 时，基本可以断定是“同帧匹配规则/文件命名”问题，而不是 AprilTag/PnP。
    # 这里输出最关键的定位信息：每对相机的 frame_key 交集大小，以及每个相机的样例 key。
    if len(keys) == 0:
        print("错误: 没有找到任何‘至少两相机同名帧’的候选 frame_key，因此不可能构建位姿图边。")
        print("  这通常是帧同步/命名规则不一致导致的：脚本用同名 frame_key 作为‘同一时刻’判据。")
        print(f"  每相机可用图片数: {per_cam_frames}")

        # 每相机抽样一些 key，便于肉眼对比
        for cam in cameras:
            sample = sorted(list(frame_maps_p[cam].keys()))[:10]
            print(f"  {cam} sample frame_key: {sample}")

        # 每对相机交集大小
        cams_list = list(cameras)
        for i in range(len(cams_list)):
            for j in range(i + 1, len(cams_list)):
                a = cams_list[i]
                b = cams_list[j]
                inter = set(frame_maps_p[a].keys()) & set(frame_maps_p[b].keys())
                inter_sample = sorted(list(inter))[:10]
                print(f"  intersection({a}, {b}) = {len(inter)} sample={inter_sample}")

    # 串行/并行共享：frame_maps 转为纯 str，避免 Path 在多进程间多次序列化。
    frame_maps: Dict[str, Dict[str, str]] = {
        cam: {k: str(p) for k, p in mp.items()}
        for cam, mp in frame_maps_p.items()
    }

    roi_by_cam: Dict[str, Optional[Tuple[int, int, int, int]]] = {
        cam: get_detection_roi(config, camera=cam) for cam in cameras
    }

    roi_by_cam_ser: Dict[str, Optional[List[int]]] = {}
    for cam in cameras:
        roi = roi_by_cam.get(cam)
        roi_by_cam_ser[str(cam)] = [int(x) for x in roi] if roi is not None else None

    algo_key = _build_algo_key(config, profile=str(profile))

    state: Dict[str, Any] = {
        "config": config,
        "cameras": list(cameras),
        "family": str(family),
        "obj_points": np.asarray(obj_points_mm, dtype=np.float32).tolist(),
        "tag_ids": [int(x) for x in tag_ids],
        "frame_maps": frame_maps,
        "roi_by_cam": roi_by_cam_ser,
        "intr_by_cam": {
            cam: {
                "K": np.asarray(intrinsics[cam].K, dtype=np.float64).tolist(),
                "dist": np.asarray(intrinsics[cam].dist, dtype=np.float64).tolist(),
            }
            for cam in cameras
        },
        "auto_roi_cfg": auto_roi_cfg,
        "use_multiscale": bool(use_multiscale),
        "opencv_refine": bool(opencv_refine),
        "algo_key": algo_key,
        "cache": {
            "enabled": bool(cache_cfg.enabled),
            "cache_dir": str(cache_cfg.cache_dir),
            "force_redetect": bool(cache_cfg.force_redetect),
        },
        "prefilter": {"enabled": bool(prefilter_cfg.enabled)},
        "min_tags": int(min_tags),
    }

    poses_by_frame: Dict[str, Dict[str, PoseObs]] = {}
    paths_by_frame: Dict[str, Dict[str, Path]] = {}
    edges: List[EdgeObs] = []

    # 可观测性：检测调用级统计
    det_total = 0
    det_cache_hit = 0
    det_cache_miss = 0
    det_prefilter_skipped = 0
    det_error = 0
    det_ms_sum = 0.0
    pnp_ok = 0
    pnp_fail = 0

    # 诊断：按 reproj_mean_px 过滤的 PnP 观测数量
    pose_drop_by_reproj = 0

    # union-find：用于连通性动态早停
    parent: Dict[str, str] = {c: c for c in cameras}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra = _find(a)
        rb = _find(b)
        if ra != rb:
            parent[rb] = ra

    stop_reason: str = ""

    def _stop_fn(r: Dict[str, Any], counters) -> bool:
        nonlocal det_total, det_cache_hit, det_cache_miss, det_prefilter_skipped, det_error, det_ms_sum
        nonlocal pnp_ok, pnp_fail, stop_reason
        nonlocal pose_drop_by_reproj

        # 统计 detector 调用层信息
        det_by_cam = r.get("det_by_cam") or {}
        for _cam, d in det_by_cam.items():
            det_total += 1
            if bool(d.get("from_cache", False)):
                det_cache_hit += 1
            else:
                det_cache_miss += 1
            st = int(d.get("status", 0))
            if st == 2:
                det_prefilter_skipped += 1
            elif st == 1:
                det_error += 1
            try:
                det_ms_sum += float(d.get("elapsed_ms", 0.0))
            except Exception:
                pass
            if "pnp_ok" in d:
                if bool(d.get("pnp_ok")):
                    pnp_ok += 1
                else:
                    pnp_fail += 1

        frame_key = str(r.get("frame_key"))
        poses_raw = r.get("poses_by_cam") or {}
        paths_raw = r.get("paths_by_cam") or {}

        if len(poses_raw) > 0:
            poses_this: Dict[str, PoseObs] = {}
            paths_this: Dict[str, Path] = {}
            for cam, pr in poses_raw.items():
                reproj = float(pr.get("reproj_mean_px", 0.0))
                if float(max_reproj_mean_px) > 0.0 and reproj > float(max_reproj_mean_px):
                    pose_drop_by_reproj += 1
                    continue

                T = np.asarray(pr["C_T_B"], dtype=np.float64)
                obj_pts = np.asarray(pr.get("obj_pts") or [], dtype=np.float32).reshape(-1, 3)
                img_pts = np.asarray(pr.get("img_pts") or [], dtype=np.float32).reshape(-1, 2)
                poses_this[str(cam)] = PoseObs(
                    cam=str(cam),
                    frame_key=str(frame_key),
                    C_T_B=T,
                    n_tags=int(pr.get("n_tags", 0)),
                    reproj_mean_px=reproj,
                    obj_pts=obj_pts,
                    img_pts=img_pts,
                )
                if cam in paths_raw:
                    paths_this[str(cam)] = Path(str(paths_raw[cam]))

            # 注意：只记录“至少两相机可形成边”的帧，避免把单相机帧也统计为有效。
            if len(poses_this) >= 2:
                poses_by_frame[frame_key] = poses_this
                paths_by_frame[frame_key] = paths_this

                cams = sorted(list(poses_this.keys()))
                for cam_i, cam_j in combinations(cams, 2):
                    pi = poses_this[cam_i]
                    pj = poses_this[cam_j]

                    Ci_T_B = pi.C_T_B
                    Cj_T_B = pj.C_T_B
                    Ci_T_Cj = Ci_T_B @ _inv_T(Cj_T_B, "B_T_Cj")

                    err = 0.5 * (float(pi.reproj_mean_px) + float(pj.reproj_mean_px))
                    w = float(1.0 / max(1e-6, err))

                    edges.append(
                        EdgeObs(
                            cam_i=cam_i,
                            cam_j=cam_j,
                            Ci_T_Cj=Ci_T_Cj,
                            weight=w,
                            frame_key=frame_key,
                        )
                    )
                    _union(cam_i, cam_j)

        # 动态早停：连通且边数量足够时停止。
        if not bool(stop_when_connected):
            return False
        if int(stop_min_edges) > 0 and len(edges) < int(stop_min_edges):
            return False
        roots = {_find(c) for c in cameras}
        if len(roots) == 1:
            stop_reason = "connected_and_enough_edges"
            return True
        return False

    def _is_valid_frame(r: Dict[str, Any]) -> bool:
        # 这里的 valid 定义为“至少两相机 PnP 成功且通过过滤”，因为这才会贡献 pose-graph 边。
        if not bool(r.get("valid", False)):
            return False
        if float(max_reproj_mean_px) <= 0.0:
            return True
        poses_raw = r.get("poses_by_cam") or {}
        good = 0
        for _cam, pr in poses_raw.items():
            try:
                if float(pr.get("reproj_mean_px", float("inf"))) <= float(max_reproj_mean_px):
                    good += 1
            except Exception:
                continue
        return int(good) >= 2

    if int(workers) <= 0:
        workers = int(os.cpu_count() or 1)

    # 若用户未给 max_total，则按 target_valid 做一个默认上限，避免无意扫全量导致耗时爆炸。
    if int(scan_limits.max_total) <= 0 and int(scan_limits.target_valid) > 0:
        scan_limits = ScanLimits(
            target_valid=int(scan_limits.target_valid),
            max_total=int(min(len(keys), max(8 * int(scan_limits.target_valid), int(scan_limits.target_valid)))),
            max_seconds=float(scan_limits.max_seconds),
        )

    if int(workers) <= 1:
        # 单进程模式下，仍复用 worker 逻辑：需要先在主进程初始化全局状态。
        _init_multicam_frame_worker(state)
        _results, counters = iter_scan_sequential(
            keys,
            worker_fn=_multicam_frame_worker,
            is_valid_fn=_is_valid_frame,
            limits=scan_limits,
            order=scan_order,
            stop_fn=_stop_fn,
        )
    else:
        _results, counters = iter_scan_parallel_ordered(
            keys,
            worker_fn=_multicam_frame_worker,
            is_valid_fn=_is_valid_frame,
            limits=scan_limits,
            order=scan_order,
            max_workers=int(workers),
            prefetch=int(prefetch),
            initializer=_init_multicam_frame_worker,
            initargs=(state,),
            stop_fn=_stop_fn,
        )

    # 结束原因推断（stop_fn 可能会设置 stop_reason）
    if not stop_reason:
        if int(scan_limits.target_valid) > 0 and int(counters.valid) >= int(scan_limits.target_valid):
            stop_reason = "target_valid_frames"
        elif int(scan_limits.max_total) > 0 and int(counters.completed) >= int(scan_limits.max_total):
            stop_reason = "max_total_frames"
        elif float(scan_limits.max_seconds) > 0 and float(counters.elapsed_s) >= float(scan_limits.max_seconds):
            stop_reason = "max_detect_seconds"
        else:
            stop_reason = "exhausted_candidates"

    scan_report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "image_root": str(image_root.as_posix()),
        "cameras": list(cameras),
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
        "filters": {
            "max_reproj_mean_px": float(max_reproj_mean_px),
            "poses_dropped_by_reproj": int(pose_drop_by_reproj),
        },
        "results": {
            "poses_frames": int(len(poses_by_frame)),
            "edges": int(len(edges)),
        },
        "frame_sync": {
            "per_cam_frames": per_cam_frames,
            "candidate_frame_keys_ge2": int(len(keys)),
            "pair_intersections": pair_intersections,
            "note": "frame_key 来自文件名 stem（会自动去掉形如 <cam>_ / <cam>- 的前缀）。",
        },
    }

    return poses_by_frame, paths_by_frame, edges, scan_report


def _build_edges_from_poses(poses_by_frame: Dict[str, Dict[str, PoseObs]]) -> List[EdgeObs]:
    edges: List[EdgeObs] = []

    for frame_key, poses in poses_by_frame.items():
        cams = sorted(list(poses.keys()))
        if len(cams) < 2:
            continue

        for cam_i, cam_j in combinations(cams, 2):
            pi = poses[cam_i]
            pj = poses[cam_j]

            Ci_T_B = pi.C_T_B
            Cj_T_B = pj.C_T_B
            Ci_T_Cj = Ci_T_B @ _inv_T(Cj_T_B, "B_T_Cj")

            # 简单权重：用两相机 reprojection error 的倒数
            # 目的：让“角点更准的观测”在优化里更有话语权
            err = 0.5 * (float(pi.reproj_mean_px) + float(pj.reproj_mean_px))
            w = float(1.0 / max(1e-6, err))

            edges.append(
                EdgeObs(
                    cam_i=cam_i,
                    cam_j=cam_j,
                    Ci_T_Cj=Ci_T_Cj,
                    weight=w,
                    frame_key=frame_key,
                )
            )

    return edges


def _connected_components(cameras: Sequence[str], edges: Sequence[EdgeObs]) -> List[List[str]]:
    adj: Dict[str, List[str]] = {c: [] for c in cameras}
    for e in edges:
        adj[e.cam_i].append(e.cam_j)
        adj[e.cam_j].append(e.cam_i)

    seen = set()
    comps: List[List[str]] = []
    for c in cameras:
        if c in seen:
            continue
        stack = [c]
        comp = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            for nb in adj.get(x, []):
                if nb not in seen:
                    stack.append(nb)
        comps.append(sorted(comp))

    comps.sort(key=len, reverse=True)
    return comps


def _avg_relative_transform(meas_list: Sequence[np.ndarray]) -> np.ndarray:
    """对同一对相机的多次相对位姿做一个简单平均（用于初始化）。"""
    if len(meas_list) == 0:
        raise ValueError("empty meas_list")

    rotvecs = []
    ts = []
    for T in meas_list:
        T = np.asarray(T, dtype=np.float64)
        Rm = T[:3, :3]
        t = T[:3, 3]
        rotvecs.append(_so3_log(Rm))
        ts.append(t)

    rv_mean = np.mean(np.stack(rotvecs, axis=0), axis=0)
    R_mean = _so3_exp(rv_mean)
    t_mean = np.mean(np.stack(ts, axis=0), axis=0)
    return _make_T(R_mean, t_mean, "avg_rel")


def _initial_poses_from_edges(cameras: Sequence[str], reference: str, edges: Sequence[EdgeObs]) -> Dict[str, np.ndarray]:
    """用 BFS 从 reference 出发给每个相机一个初值（不保证最优，但通常够用）。"""

    # 收集每条有向边的观测列表：(i,j) 表示 Ci <- Cj
    dir_meas: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for e in edges:
        dir_meas.setdefault((e.cam_i, e.cam_j), []).append(e.Ci_T_Cj)
        dir_meas.setdefault((e.cam_j, e.cam_i), []).append(_inv_T(e.Ci_T_Cj, "inv_edge"))

    # 对每个方向做平均，得到一张“稀疏图”
    dir_avg: Dict[Tuple[str, str], np.ndarray] = {}
    for k, lst in dir_meas.items():
        dir_avg[k] = _avg_relative_transform(lst)

    X: Dict[str, np.ndarray] = {reference: np.eye(4, dtype=np.float64)}

    # BFS
    q = [reference]
    while q:
        cur = q.pop(0)
        for nb in cameras:
            if nb in X:
                continue
            key = (nb, cur)  # X_nb = (nb <- cur) * X_cur
            if key not in dir_avg:
                continue
            X[nb] = dir_avg[key] @ X[cur]
            q.append(nb)

    return X


def _pack_params(cameras: Sequence[str], reference: str, X_init: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    opt_cams = [c for c in cameras if c != reference]
    x0 = np.zeros(6 * len(opt_cams), dtype=np.float64)

    for k, cam in enumerate(opt_cams):
        Ti = X_init.get(cam, np.eye(4, dtype=np.float64))
        xi = _se3_log(Ti)
        x0[6 * k : 6 * k + 6] = xi

    return x0, opt_cams


def _unpack_params(x: np.ndarray, opt_cams: Sequence[str], reference: str) -> Dict[str, np.ndarray]:
    X: Dict[str, np.ndarray] = {reference: np.eye(4, dtype=np.float64)}
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    for k, cam in enumerate(opt_cams):
        xi = x[6 * k : 6 * k + 6]
        X[cam] = _se3_exp(xi)
    return X


def _residuals_pose_graph(x: np.ndarray, *, opt_cams: Sequence[str], reference: str, edges: Sequence[EdgeObs]) -> np.ndarray:
    X = _unpack_params(x, opt_cams=opt_cams, reference=reference)

    res = []
    for e in edges:
        Xi = X[e.cam_i]
        Xj = X[e.cam_j]
        pred = Xi @ _inv_T(Xj, "inv_Xj")
        err_T = _inv_T(e.Ci_T_Cj, "inv_meas") @ pred
        r6 = _se3_log(err_T)
        w = float(max(1e-9, e.weight))
        res.append(np.sqrt(w) * r6)

    if len(res) == 0:
        return np.zeros((0,), dtype=np.float64)

    return np.concatenate(res, axis=0)


def _pct(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, float(q)))


def _summarize_scalar(values: Sequence[float]) -> Dict[str, Any]:
    """汇总一组标量的统计信息。

    说明：
    - 用于输出更直观的指标分布（例如 reprojection error 的 px 分布）。
    - 仅用于诊断/报告，不参与优化。
    """

    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}

    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": _pct(arr, 50),
        "p95": _pct(arr, 95),
        "max": float(np.max(arr)),
    }


def _summarize_reproj_px(
    *,
    cameras: Sequence[str],
    edges: Sequence[EdgeObs],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """汇总 PnP 重投影误差（像素）。

    输出：
    - by_cam: 每个相机的 reproj_mean_px 分布（仅统计参与建图的 PoseObs）
    - edge_mean_px: 每条边两相机 reproj_mean_px 的均值分布（用于理解权重来源）
    - worst_edges_by_edge_mean_px: 按边均值从大到小的 Top-K（用于定位坏帧/不同步）
    """

    cams = [str(c) for c in cameras]

    by_cam_vals: Dict[str, List[float]] = {c: [] for c in cams}
    for _frame_key, poses in poses_by_frame.items():
        for cam, po in poses.items():
            if str(cam) not in by_cam_vals:
                continue
            by_cam_vals[str(cam)].append(float(po.reproj_mean_px))

    by_cam = {c: _summarize_scalar(by_cam_vals.get(c, [])) for c in cams}

    # 边上的 reproj：用两端 pose 的 reproj_mean_px 做均值（与权重定义一致）。
    edge_vals: List[float] = []
    edge_rows: List[Dict[str, Any]] = []
    for e in edges:
        frame_key = str(e.frame_key)
        poses = poses_by_frame.get(frame_key) or {}
        pi = poses.get(str(e.cam_i))
        pj = poses.get(str(e.cam_j))
        if pi is None or pj is None:
            continue

        edge_mean = 0.5 * (float(pi.reproj_mean_px) + float(pj.reproj_mean_px))
        edge_vals.append(float(edge_mean))

        p_i = (paths_by_frame.get(frame_key) or {}).get(str(e.cam_i))
        p_j = (paths_by_frame.get(frame_key) or {}).get(str(e.cam_j))
        edge_rows.append(
            {
                "pair": f"{e.cam_i}__{e.cam_j}",
                "cam_i": str(e.cam_i),
                "cam_j": str(e.cam_j),
                "frame_key": frame_key,
                "edge_mean_px": float(edge_mean),
                "reproj_mean_px": {
                    "cam_i": float(pi.reproj_mean_px),
                    "cam_j": float(pj.reproj_mean_px),
                },
                "n_tags": {
                    "cam_i": int(pi.n_tags),
                    "cam_j": int(pj.n_tags),
                },
                "image_path": {
                    "cam_i": str(p_i) if p_i is not None else None,
                    "cam_j": str(p_j) if p_j is not None else None,
                },
            }
        )

    edge_mean_px = _summarize_scalar(edge_vals)
    top_k = int(max(1, top_k))
    worst_edges = sorted(edge_rows, key=lambda d: float(d.get("edge_mean_px", 0.0)), reverse=True)[:top_k]

    return {
        "units": "px",
        "by_cam": by_cam,
        "edge_mean_px": edge_mean_px,
        "worst_edges_by_edge_mean_px": worst_edges,
    }


def _select_anchor_cam(poses: Dict[str, PoseObs]) -> str:
    """为某一帧选择一个 anchor 相机。

    规则：优先选该帧里 PnP reproj_mean_px 最小的相机。
    目的：
    - 用 anchor 的 PnP 位姿当作该帧“板位姿”的来源，更稳且更可解释。
    - 避免对多个相机位姿做平均时被离群值拉偏，导致后续回投影误差虚高。
    """

    if len(poses) == 0:
        raise ValueError("empty poses")
    return sorted(poses.values(), key=lambda p: float(p.reproj_mean_px))[0].cam


def _summarize_post_extrinsics_reproj_px(
    *,
    X_cam_from_ref: Dict[str, np.ndarray],
    reference: str,
    cameras: Sequence[str],
    intrinsics: Dict[str, CameraIntrinsics],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]],
    top_k: int = 10,
) -> Dict[str, Any]:
    """计算“外参解算完成后”的一致性重投影误差（像素）。

    定义：
        - 对每个 frame_key，选一个 anchor 相机（该帧 PnP reproj 最小），用它的 C_anchor_T_B
            通过外参转到参考相机系，得到该帧板位姿 C_ref_T_B。
        - 用该板位姿 + 外参，把板角点回投影到每个相机，计算与观测角点的像素误差。

    注意：
    - 这个误差反映“多相机一致性”，通常会大于每个相机各自 PnP 拟合的 reproj。
    - 如果同步/某相机PnP有问题，这个指标会非常敏感。
    """

    cams = [str(c) for c in cameras]
    if reference not in X_cam_from_ref:
        raise ValueError("reference missing in X_cam_from_ref")

    per_cam_vals: Dict[str, List[float]] = {c: [] for c in cams}
    rows: List[Dict[str, Any]] = []

    for frame_key, poses in poses_by_frame.items():
        if len(poses) < 2:
            continue

        # 1) 选 anchor 相机，并用它的 PnP 位姿生成参考相机系下的板位姿
        anchor = _select_anchor_cam(poses)
        if anchor not in X_cam_from_ref:
            continue
        po_anchor = poses.get(anchor)
        if po_anchor is None:
            continue

        Xi_anchor = X_cam_from_ref[anchor]
        Cref_T_B = _inv_T(Xi_anchor, "inv_X_anchor") @ po_anchor.C_T_B

        # 2) 用融合板位姿回投影到每个相机
        for cam, po in poses.items():
            if cam not in X_cam_from_ref:
                continue
            if cam not in intrinsics:
                continue

            if po.obj_pts.size == 0 or po.img_pts.size == 0:
                continue
            if po.obj_pts.shape[0] != po.img_pts.shape[0]:
                continue

            Xi = X_cam_from_ref[cam]
            Ci_T_B_pred = Xi @ Cref_T_B

            Rm = np.asarray(Ci_T_B_pred[:3, :3], dtype=np.float64)
            t = np.asarray(Ci_T_B_pred[:3, 3], dtype=np.float64).reshape(3, 1)
            rvec, _ = cv2.Rodrigues(Rm)

            K = np.asarray(intrinsics[cam].K, dtype=np.float64)
            dist = np.asarray(intrinsics[cam].dist, dtype=np.float64)

            proj, _ = cv2.projectPoints(
                np.asarray(po.obj_pts, dtype=np.float32),
                np.asarray(rvec, dtype=np.float64),
                np.asarray(t, dtype=np.float64),
                K,
                dist,
            )
            proj = proj.reshape(-1, 2).astype(np.float64)
            obs = np.asarray(po.img_pts, dtype=np.float64).reshape(-1, 2)
            per_pt = np.linalg.norm(obs - proj, axis=1)
            mean_px = float(np.mean(per_pt)) if per_pt.size else float("inf")

            per_cam_vals[cam].append(mean_px)

            p = (paths_by_frame.get(str(frame_key)) or {}).get(cam)
            rows.append(
                {
                    "cam": str(cam),
                    "frame_key": str(frame_key),
                    "anchor_cam": str(anchor),
                    "mean_px": mean_px,
                    "n_points": int(per_pt.size),
                    "image_path": str(p) if p is not None else None,
                }
            )

    by_cam = {c: _summarize_scalar(per_cam_vals.get(c, [])) for c in cams}
    all_vals = [x for lst in per_cam_vals.values() for x in lst]
    overall = _summarize_scalar(all_vals)

    top_k = int(max(1, top_k))
    worst = sorted(rows, key=lambda d: float(d.get("mean_px", 0.0)), reverse=True)[:top_k]

    return {
        "units": "px",
        "note": "外参优化完成后：每帧选 reproj 最小的相机做 anchor，生成板位姿并回投影计算一致性误差。",
        "overall": overall,
        "by_cam": by_cam,
        "worst": worst,
    }


def _build_ba_frame_and_obs(
    *,
    cameras_solved: Sequence[str],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
) -> Tuple[List[str], List[Tuple[str, str, np.ndarray, np.ndarray]]]:
    """整理 BA 需要的帧列表与观测列表。

    说明：
    - 仅保留“至少两相机都有效”的帧，这样每帧的板位姿能把相机连接起来。
    - 观测粒度到角点（obj_pts/img_pts），BA 直接优化像素残差。
    """

    cams = {str(c) for c in cameras_solved}
    frames: List[str] = []
    obs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []

    for frame_key in sorted(poses_by_frame.keys()):
        poses = poses_by_frame.get(frame_key) or {}
        poses = {str(c): p for c, p in poses.items() if str(c) in cams}
        if len(poses) < 2:
            continue

        # 只保留点数一致且非空的观测
        local_obs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
        for cam, po in poses.items():
            obj = np.asarray(po.obj_pts, dtype=np.float32).reshape(-1, 3)
            img = np.asarray(po.img_pts, dtype=np.float32).reshape(-1, 2)
            if obj.size == 0 or img.size == 0:
                continue
            if obj.shape[0] != img.shape[0]:
                continue
            local_obs.append((str(frame_key), str(cam), obj, img))

        # 若过滤后不足两相机，则该帧对多相机耦合贡献有限，丢弃
        cams_in_frame = {c for (_fk, c, _o, _i) in local_obs}
        if len(cams_in_frame) < 2:
            continue

        frames.append(str(frame_key))
        obs.extend(local_obs)

    return frames, obs


def _init_board_poses_in_ref(
    *,
    frames: Sequence[str],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    X_cam_from_ref_init: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """用当前外参初值，为每一帧初始化参考相机系下的板位姿 C_ref_T_B。

    说明：
    - 每帧选一个 anchor 相机（PnP reproj 最小），用它的 C_anchor_T_B 转到参考相机系。
    - 这只是 BA 的初值，后续会由 BA 在像素域联合优化纠正。
    """

    out: Dict[str, np.ndarray] = {}
    for frame_key in frames:
        poses = poses_by_frame.get(str(frame_key)) or {}
        if len(poses) == 0:
            continue
        anchor = _select_anchor_cam(poses)
        if anchor not in X_cam_from_ref_init:
            continue
        po_anchor = poses.get(anchor)
        if po_anchor is None:
            continue
        Xi_anchor = X_cam_from_ref_init[anchor]
        out[str(frame_key)] = _inv_T(Xi_anchor, "inv_X_anchor_init") @ po_anchor.C_T_B
    return out


def _pack_ba_params(
    *,
    opt_cams: Sequence[str],
    X_cam_from_ref_init: Dict[str, np.ndarray],
    frames: Sequence[str],
    Cref_T_B_init: Dict[str, np.ndarray],
) -> np.ndarray:
    """打包 BA 参数。

    参数向量结构：
    - 相机外参（除 reference 外）：每个 6D se(3)（w(3), v(3)）
    - 每帧板位姿（参考相机系下）：每帧 6D se(3)
    """

    x = np.zeros(6 * int(len(opt_cams)) + 6 * int(len(frames)), dtype=np.float64)
    for k, cam in enumerate(opt_cams):
        T = np.asarray(X_cam_from_ref_init.get(str(cam), np.eye(4, dtype=np.float64)), dtype=np.float64)
        x[6 * k : 6 * k + 6] = _se3_log(T)

    base = 6 * int(len(opt_cams))
    for i, frame_key in enumerate(frames):
        T = np.asarray(Cref_T_B_init.get(str(frame_key), np.eye(4, dtype=np.float64)), dtype=np.float64)
        x[base + 6 * i : base + 6 * i + 6] = _se3_log(T)

    return x


def _unpack_ba_params(
    *,
    x: np.ndarray,
    opt_cams: Sequence[str],
    reference: str,
    frames: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """解包 BA 参数为 (X_cam_from_ref, Cref_T_B_by_frame)。"""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    X: Dict[str, np.ndarray] = {str(reference): np.eye(4, dtype=np.float64)}
    for k, cam in enumerate(opt_cams):
        xi = x[6 * k : 6 * k + 6]
        X[str(cam)] = _se3_exp(xi)

    base = 6 * int(len(opt_cams))
    Cref_T_B: Dict[str, np.ndarray] = {}
    for i, frame_key in enumerate(frames):
        xi = x[base + 6 * i : base + 6 * i + 6]
        Cref_T_B[str(frame_key)] = _se3_exp(xi)

    return X, Cref_T_B


def _residuals_bundle_adjustment(
    x: np.ndarray,
    *,
    opt_cams: Sequence[str],
    reference: str,
    frames: Sequence[str],
    observations: Sequence[Tuple[str, str, np.ndarray, np.ndarray]],
    intrinsics: Dict[str, CameraIntrinsics],
) -> np.ndarray:
    """多相机 BA 的像素残差。

    说明：
    - 直接最小化所有角点的像素重投影误差（与 stereoCalibrate 更同类）。
    - 变量：相机外参 X_cam_from_ref，以及每帧板位姿 Cref_T_B。
    """

    X_cam_from_ref, Cref_T_B_by_frame = _unpack_ba_params(
        x=x, opt_cams=opt_cams, reference=reference, frames=frames
    )

    res: List[np.ndarray] = []
    for frame_key, cam, obj_pts, img_pts in observations:
        Ci = X_cam_from_ref.get(str(cam))
        Cref_T_B = Cref_T_B_by_frame.get(str(frame_key))
        intr = intrinsics.get(str(cam))
        if Ci is None or Cref_T_B is None or intr is None:
            continue

        Ci_T_B = Ci @ Cref_T_B
        rvec, tvec = _to_rvec_tvec_from_T(Ci_T_B)

        K = np.asarray(intr.K, dtype=np.float64)
        dist = np.asarray(intr.dist, dtype=np.float64)

        proj, _ = cv2.projectPoints(
            np.asarray(obj_pts, dtype=np.float32),
            np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64),
            K,
            dist,
        )
        proj = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        obs = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
        if proj.shape != obs.shape or proj.size == 0:
            continue

        r = (obs - proj).reshape(-1)
        res.append(r)

    if len(res) == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(res, axis=0)


def _summarize_ba_reproj_px(
    *,
    X_cam_from_ref: Dict[str, np.ndarray],
    Cref_T_B_by_frame: Dict[str, np.ndarray],
    cameras: Sequence[str],
    intrinsics: Dict[str, CameraIntrinsics],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]],
    top_k: int = 10,
) -> Dict[str, Any]:
    """汇总 BA 之后的像素重投影误差（直接使用 BA 里的每帧板位姿）。"""

    cams = [str(c) for c in cameras]
    per_cam_vals: Dict[str, List[float]] = {c: [] for c in cams}
    rows: List[Dict[str, Any]] = []

    for frame_key, poses in poses_by_frame.items():
        Cref_T_B = Cref_T_B_by_frame.get(str(frame_key))
        if Cref_T_B is None:
            continue

        for cam, po in poses.items():
            cam = str(cam)
            if cam not in X_cam_from_ref or cam not in intrinsics:
                continue
            if po.obj_pts.size == 0 or po.img_pts.size == 0:
                continue
            if po.obj_pts.shape[0] != po.img_pts.shape[0]:
                continue

            Ci_T_B = X_cam_from_ref[cam] @ Cref_T_B
            rvec, tvec = _to_rvec_tvec_from_T(Ci_T_B)
            intr = intrinsics[cam]

            proj, _ = cv2.projectPoints(
                np.asarray(po.obj_pts, dtype=np.float32),
                np.asarray(rvec, dtype=np.float64),
                np.asarray(tvec, dtype=np.float64),
                np.asarray(intr.K, dtype=np.float64),
                np.asarray(intr.dist, dtype=np.float64),
            )
            proj = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
            obs = np.asarray(po.img_pts, dtype=np.float64).reshape(-1, 2)
            if proj.shape != obs.shape or proj.size == 0:
                continue
            per_pt = np.linalg.norm(obs - proj, axis=1)
            mean_px = float(np.mean(per_pt)) if per_pt.size else float("inf")

            per_cam_vals[cam].append(mean_px)

            p = (paths_by_frame.get(str(frame_key)) or {}).get(cam)
            rows.append(
                {
                    "cam": cam,
                    "frame_key": str(frame_key),
                    "mean_px": mean_px,
                    "n_points": int(per_pt.size),
                    "image_path": str(p) if p is not None else None,
                }
            )

    by_cam = {c: _summarize_scalar(per_cam_vals.get(c, [])) for c in cams}
    all_vals = [x for lst in per_cam_vals.values() for x in lst]
    overall = _summarize_scalar(all_vals)

    top_k = int(max(1, top_k))
    worst = sorted(rows, key=lambda d: float(d.get("mean_px", 0.0)), reverse=True)[:top_k]

    return {
        "units": "px",
        "note": "Bundle Adjustment：联合优化相机外参+每帧板位姿，直接最小化角点像素重投影误差。",
        "overall": overall,
        "by_cam": by_cam,
        "worst": worst,
    }


def _filter_pose_maps_by_observations(
    *,
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]],
    observations: Sequence[Tuple[str, str, np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, Dict[str, PoseObs]], Dict[str, Dict[str, Path]]]:
    """按 BA 实际使用的观测集合过滤 poses/paths。

    说明：
    - BA 支持在第一次优化后做离群剔除（obs2），并可能二次优化。
    - 误差汇总必须基于“最终参与 BA 的观测集合”，否则会出现：
        - 最终 RMS(px) 看起来正常
        - 但统计里的 mean/max 被已经剔除的极端离群观测污染，出现 1e30+ 级别的数字
      这会误导判断“是不是算崩了”。
    """

    keep: set[tuple[str, str]] = set()
    for fk, cam, _obj, _img in observations:
        keep.add((str(fk), str(cam)))

    poses_out: Dict[str, Dict[str, PoseObs]] = {}
    paths_out: Dict[str, Dict[str, Path]] = {}
    for fk, poses in poses_by_frame.items():
        fk_s = str(fk)
        new_poses: Dict[str, PoseObs] = {}
        new_paths: Dict[str, Path] = {}
        for cam, po in poses.items():
            cam_s = str(cam)
            if (fk_s, cam_s) not in keep:
                continue
            new_poses[cam_s] = po
            p = (paths_by_frame.get(fk_s) or {}).get(cam_s)
            if p is not None:
                new_paths[cam_s] = p
        if len(new_poses) > 0:
            poses_out[fk_s] = new_poses
            if len(new_paths) > 0:
                paths_out[fk_s] = new_paths

    return poses_out, paths_out


def _prune_ba_observations(
    *,
    X_cam_from_ref: Dict[str, np.ndarray],
    Cref_T_B_by_frame: Dict[str, np.ndarray],
    cameras: Sequence[str],
    intrinsics: Dict[str, CameraIntrinsics],
    observations: Sequence[Tuple[str, str, np.ndarray, np.ndarray]],
    prune_mean_px: float,
    min_keep_per_cam: int,
) -> Tuple[List[Tuple[str, str, np.ndarray, np.ndarray]], Dict[str, Any]]:
    """根据 BA 解对观测做一次离群剔除（按单观测 mean_px）。

    说明：
    - 适用于“少 tag 的 PnP 位姿偶发很离谱”导致的坏观测。
    - 为避免把某个相机全部删光，会保证每个相机至少保留 min_keep_per_cam 条观测（按误差从小到大补回）。
    """

    thr = float(prune_mean_px)
    if thr <= 0.0:
        return list(observations), {"enabled": False}

    rows: List[Tuple[float, str, str, np.ndarray, np.ndarray]] = []
    for frame_key, cam, obj_pts, img_pts in observations:
        Ci = X_cam_from_ref.get(str(cam))
        Cref_T_B = Cref_T_B_by_frame.get(str(frame_key))
        intr = intrinsics.get(str(cam))
        if Ci is None or Cref_T_B is None or intr is None:
            continue

        Ci_T_B = Ci @ Cref_T_B
        rvec, tvec = _to_rvec_tvec_from_T(Ci_T_B)
        proj, _ = cv2.projectPoints(
            np.asarray(obj_pts, dtype=np.float32),
            np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64),
            np.asarray(intr.K, dtype=np.float64),
            np.asarray(intr.dist, dtype=np.float64),
        )
        proj = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        obs = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
        if proj.shape != obs.shape or proj.size == 0:
            continue
        per_pt = np.linalg.norm(obs - proj, axis=1)
        mean_px = float(np.mean(per_pt)) if per_pt.size else float("inf")
        rows.append((mean_px, str(frame_key), str(cam), obj_pts, img_pts))

    keep = [(fk, cam, obj, img) for (m, fk, cam, obj, img) in rows if float(m) <= thr]

    # 防御性：保证每个相机至少保留若干观测
    cams = [str(c) for c in cameras]
    min_keep = int(max(0, min_keep_per_cam))
    if min_keep > 0:
        by_cam_all: Dict[str, List[Tuple[float, str, str, np.ndarray, np.ndarray]]] = {c: [] for c in cams}
        for r in rows:
            by_cam_all[r[2]].append(r)
        for c in cams:
            cur = [o for o in keep if o[1] == c]
            if len(cur) >= min_keep:
                continue
            pool = sorted(by_cam_all.get(c, []), key=lambda t: float(t[0]))
            need = min_keep - len(cur)
            for (m, fk, cam, obj, img) in pool:
                if need <= 0:
                    break
                cand = (fk, cam, obj, img)
                if cand in keep:
                    continue
                keep.append(cand)
                need -= 1

    # 剔除后，保证每帧至少 2 相机，否则该帧对多相机耦合意义不大
    by_frame: Dict[str, List[Tuple[str, str, np.ndarray, np.ndarray]]] = defaultdict(list)
    for o in keep:
        by_frame[str(o[0])].append(o)
    keep2: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    dropped_frames = 0
    for fk, lst in by_frame.items():
        cams_in = {x[1] for x in lst}
        if len(cams_in) < 2:
            dropped_frames += 1
            continue
        keep2.extend(lst)

    info = {
        "enabled": True,
        "prune_mean_px": float(thr),
        "min_keep_per_cam": int(min_keep),
        "before": int(len(observations)),
        "after": int(len(keep2)),
        "dropped_frames": int(dropped_frames),
    }
    return keep2, info


def _summarize_edge_errors(*, X: Dict[str, np.ndarray], edges: Sequence[EdgeObs]) -> Dict[str, Any]:
    """输出更可解释的误差统计。

    说明：
    - 控制台上打印的 RMS(6D) 来自 se(3) log 向量，且会混合旋转(弧度)与平移(单位跟 obj_points 一致)。
    - 这里把误差拆成更直观的两个量：
        1) 旋转角误差（度）: ||log(R)|| * 180/pi
        2) 平移误差（与 obj_points 同单位）: ||t||
    - 本脚本的 create_apriltag_board() 使用 mm，因此平移误差单位=mm。
    """

    rot_deg_list: List[float] = []
    trans_list: List[float] = []
    by_pair: Dict[str, Dict[str, List[float]]] = {}

    for e in edges:
        Xi = X[e.cam_i]
        Xj = X[e.cam_j]
        pred = Xi @ _inv_T(Xj, "inv_Xj")
        err_T = _inv_T(e.Ci_T_Cj, "inv_meas") @ pred

        w = _so3_log(err_T[:3, :3])
        rot_deg = float(np.linalg.norm(w) * 180.0 / np.pi)
        trans = float(np.linalg.norm(err_T[:3, 3]))

        rot_deg_list.append(rot_deg)
        trans_list.append(trans)

        k = f"{e.cam_i}__{e.cam_j}"
        if k not in by_pair:
            by_pair[k] = {"rot_deg": [], "trans": []}
        by_pair[k]["rot_deg"].append(rot_deg)
        by_pair[k]["trans"].append(trans)

    rot = np.asarray(rot_deg_list, dtype=np.float64)
    tr = np.asarray(trans_list, dtype=np.float64)

    pair_out: Dict[str, Any] = {}
    for k, v in by_pair.items():
        r = np.asarray(v["rot_deg"], dtype=np.float64)
        t = np.asarray(v["trans"], dtype=np.float64)
        pair_out[k] = {
            "n": int(r.size),
            "rot_deg": {"median": _pct(r, 50), "p95": _pct(r, 95), "max": float(np.max(r)) if r.size else 0.0},
            "trans": {"median": _pct(t, 50), "p95": _pct(t, 95), "max": float(np.max(t)) if t.size else 0.0},
        }

    return {
        "n_edges": int(len(edges)),
        "units": {"rotation": "deg", "translation": "mm"},
        "rot_deg": {
            "mean": float(np.mean(rot)) if rot.size else 0.0,
            "median": _pct(rot, 50),
            "p95": _pct(rot, 95),
            "max": float(np.max(rot)) if rot.size else 0.0,
        },
        "trans": {
            "mean": float(np.mean(tr)) if tr.size else 0.0,
            "median": _pct(tr, 50),
            "p95": _pct(tr, 95),
            "max": float(np.max(tr)) if tr.size else 0.0,
        },
        "by_pair": pair_out,
    }


def _edge_metrics(
    *,
    X: Dict[str, np.ndarray],
    edge: EdgeObs,
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]],
) -> Dict[str, Any]:
    """计算单条边的残差指标，并补齐定位信息（帧号/两相机图片路径）。

    说明：
    - 该函数用于“诊断输出”，帮助定位哪几条边/哪几帧最不一致。
    - 指标拆成旋转(度)与平移(mm)，并提供混合单位的 6D 范数供排序参考。
    """

    Xi = X[edge.cam_i]
    Xj = X[edge.cam_j]
    pred = Xi @ _inv_T(Xj, "inv_Xj")
    err_T = _inv_T(edge.Ci_T_Cj, "inv_meas") @ pred
    r6 = _se3_log(err_T)

    rot_deg = float(np.linalg.norm(r6[:3]) * 180.0 / np.pi)
    trans_mm = float(np.linalg.norm(r6[3:]))
    norm6 = float(np.linalg.norm(r6))
    w = float(max(1e-9, edge.weight))
    weighted_norm6 = float(np.sqrt(w) * norm6)

    frame_key = str(edge.frame_key)
    pi = (poses_by_frame.get(frame_key) or {}).get(edge.cam_i)
    pj = (poses_by_frame.get(frame_key) or {}).get(edge.cam_j)

    path_i = (paths_by_frame.get(frame_key) or {}).get(edge.cam_i)
    path_j = (paths_by_frame.get(frame_key) or {}).get(edge.cam_j)

    return {
        "pair": f"{edge.cam_i}__{edge.cam_j}",
        "cam_i": str(edge.cam_i),
        "cam_j": str(edge.cam_j),
        "frame_key": frame_key,
        "weight": float(edge.weight),
        "rot_deg": rot_deg,
        "trans_mm": trans_mm,
        "norm6": norm6,
        "weighted_norm6": weighted_norm6,
        "reproj_mean_px": {
            "cam_i": float(pi.reproj_mean_px) if pi is not None else None,
            "cam_j": float(pj.reproj_mean_px) if pj is not None else None,
        },
        "n_tags": {
            "cam_i": int(pi.n_tags) if pi is not None else None,
            "cam_j": int(pj.n_tags) if pj is not None else None,
        },
        "image_path": {
            "cam_i": str(path_i) if path_i is not None else None,
            "cam_j": str(path_j) if path_j is not None else None,
        },
    }


def _save_edge_diagnostics(
    *,
    path: Path,
    X: Dict[str, np.ndarray],
    edges: Sequence[EdgeObs],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]],
    top_k: int,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """保存边级诊断信息（最大残差边列表等）。"""

    top_k = int(max(1, top_k))
    metrics = [
        _edge_metrics(X=X, edge=e, poses_by_frame=poses_by_frame, paths_by_frame=paths_by_frame)
        for e in edges
    ]

    def _top(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        return sorted(items, key=lambda d: float(d.get(key, 0.0)), reverse=True)[:top_k]

    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for m in metrics:
        by_pair.setdefault(str(m["pair"]), []).append(m)

    out = {
        "meta": meta,
        "top_k": top_k,
        "ranking": {
            "by_rot_deg": _top(metrics, "rot_deg"),
            "by_trans_mm": _top(metrics, "trans_mm"),
            "by_weighted_norm6": _top(metrics, "weighted_norm6"),
        },
        "by_pair": {
            pair: {
                "n": int(len(lst)),
                "top_by_rot_deg": _top(lst, "rot_deg"),
                "top_by_trans_mm": _top(lst, "trans_mm"),
            }
            for pair, lst in by_pair.items()
        },
    }

    _save_json(path, out)
    return out


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Step4 (Multi-Cam): AprilTag 多相机外参（位姿图优化）")
    parser.add_argument(
        "--config",
        type=str,
        default="config/apriltag_config.json",
        help="配置文件路径（默认 config/apriltag_config.json）",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="images/filtered",
        help="图像根目录（其下每个子目录代表一个相机）",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="相机名列表（不填则自动扫描 image_root 下的子目录）",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="",
        help="参考相机名（不填则默认取 cameras[0]）",
    )
    parser.add_argument(
        "--intrinsics_dir",
        type=str,
        default="results",
        help="内参文件目录（默认 results）",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="最多处理多少个 frame_key（0=不限制）",
    )
    parser.add_argument(
        "--min_tags",
        type=int,
        default=0,
        help="每张图最少 tag 数（0=使用 config.calibration_settings.min_tags_for_pose）",
    )
    parser.add_argument(
        "--huber",
        type=float,
        default=1.0,
        help="Huber loss 的 f_scale（越小越鲁棒，默认 1.0）",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="ba",
        choices=["pose_graph", "ba"],
        help=(
            "优化方法：pose_graph=PnP→相对位姿边→位姿图；"
            "ba=多相机 Bundle Adjustment（像素域，联合优化外参+每帧板位姿，离线更稳）。"
        ),
    )
    parser.add_argument(
        "--ba_f_scale_px",
        type=float,
        default=3.0,
        help="BA 的鲁棒核尺度（像素），越小越鲁棒，默认 3.0。",
    )
    parser.add_argument(
        "--ba_max_nfev",
        type=int,
        default=1000,
        help="BA 最大迭代次数（离线可调大，默认 3000）。",
    )
    parser.add_argument(
        "--ba_prune_mean_px",
        type=float,
        default=30.0,
        help="BA 完成后按单观测 mean_px 剔除离群（像素，<=0 禁用；默认 50）。",
    )
    parser.add_argument(
        "--ba_prune_min_keep_per_cam",
        type=int,
        default=3,
        help="离群剔除时每个相机至少保留多少条观测（默认 3）。",
    )

    parser.add_argument(
        "--max_reproj_mean_px",
        type=float,
        default=0.0,
        help="可选：过滤 PnP 重投影均值误差过大的观测（像素，0=禁用）。",
    )
    parser.add_argument(
        "--edge_diag_path",
        type=str,
        default="results/step4_multi_edge_diagnostics.json",
        help="保存边级诊断信息的 JSON 路径（空字符串=不保存）。",
    )
    parser.add_argument(
        "--edge_diag_top_k",
        type=int,
        default=20,
        help="诊断输出：按指标排序输出 top-K 边（默认 20）。",
    )

    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="输出更多过程信息（可用 --no-verbose 关闭）",
    )

    # 性能优先：流式扫描/早停/并行/缓存/预筛选
    parser.add_argument(
        "--target_valid_frames",
        type=int,
        default=150,
        help="达到多少个‘有效帧’就提前停止（有效帧=至少两相机 PnP 成功，默认 50；0=不按此条件停止）。",
    )
    parser.add_argument(
        "--max_total_frames",
        type=int,
        default=0,
        help="最多尝试多少个候选 frame_key（0=自动=8*target_valid_frames）。",
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
        help="候选 frame_key 扫描策略：sequential/random/uniform（默认 uniform）。",
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
        "--no_stop_when_connected",
        action="store_true",
        help="禁用‘连通且边足够就早停’（默认开启早停；此开关用于排查/需要更多边的情况）。",
    )
    parser.add_argument(
        "--stop_min_edges",
        type=int,
        default=0,
        help="动态早停时要求的最小边数量（0=自动）。",
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
    print("Step 4 (Multi-Cam): 多相机外参标定（位姿图优化）")
    print("=" * 60)

    # 读取配置（用于 detection/min_tags，也用于可选的 image_dataset 自动输入）
    config = load_config(str(args.config))
    ds = config.get("image_dataset", {}) if isinstance(config, dict) else {}
    use_dataset = bool(ds.get("enabled", False)) if isinstance(ds, dict) else False

    # 当启用 image_dataset 且用户未显式指定 image_root 时，默认改用 filtered_root
    image_root_str = str(args.image_root)
    if use_dataset and image_root_str.strip() == "images/filtered":
        try:
            filtered_root = ds.get("filtered_root", "images/filtered")
            if isinstance(filtered_root, str) and filtered_root.strip():
                image_root_str = filtered_root.strip()
        except Exception:
            pass

    image_root = Path(image_root_str)
    intr_dir = Path(str(args.intrinsics_dir))

    if not image_root.exists():
        print(f"错误: image_root 不存在：{image_root}")
        return 1

    # 当启用 image_dataset 且用户未显式指定 cameras 时，优先用 config.cameras
    cameras_arg = args.cameras
    if use_dataset and cameras_arg is None:
        cams_cfg = ds.get("cameras", {}) if isinstance(ds, dict) else {}
        if isinstance(cams_cfg, dict) and len(cams_cfg) > 0:
            # 约定：配置中可能存在 "_comment" / "_xxx" 等注释/禁用条目，必须过滤。
            cameras_arg = [k for k in cams_cfg.keys() if isinstance(k, str) and k and not k.startswith(("_", ".", "__"))]
        elif isinstance(cams_cfg, list) and len(cams_cfg) > 0:
            cameras_arg = [str(x) for x in cams_cfg if str(x) and not str(x).startswith(("_", ".", "__"))]

    cameras = _list_cameras(image_root, cameras_arg)
    if len(cameras) < 2:
        print("错误: 至少需要 2 个相机")
        return 1

    reference = str(args.reference).strip() if str(args.reference).strip() else cameras[0]
    if reference not in cameras:
        print(f"错误: reference={reference} 不在 cameras 列表中：{cameras}")
        return 1

    min_tags_cfg = int(config.get("calibration_settings", {}).get("min_tags_for_pose", 4))
    min_tags = int(args.min_tags) if int(args.min_tags) > 0 else min_tags_cfg

    print(f"图像根目录: {image_root}")
    print(f"相机列表: {cameras}")
    print(f"参考相机: {reference}")
    print(f"min_tags: {min_tags}")

    # 加载内参
    intrinsics: Dict[str, CameraIntrinsics] = {}
    for cam in cameras:
        intrinsics[cam] = _load_intrinsics(cam, intr_dir)

    use_multiscale, opencv_refine = get_detection_settings(config)
    if bool(getattr(args, "no_multiscale", False)):
        use_multiscale = False
    if bool(getattr(args, "no_opencv_refine", False)):
        opencv_refine = False

    cache_cfg = CacheConfig(
        enabled=not bool(args.no_cache),
        cache_dir=str(args.cache_dir),
        force_redetect=bool(args.force_redetect),
    )
    prefilter_cfg = PrefilterConfig(enabled=bool(args.prefilter))

    scan_limits = ScanLimits(
        target_valid=int(args.target_valid_frames),
        max_total=int(args.max_total_frames),
        max_seconds=float(args.max_detect_seconds),
    )
    scan_order = ScanOrder(strategy=str(args.scan_strategy), seed=int(args.scan_seed))

    stop_min_edges = int(args.stop_min_edges)
    if stop_min_edges <= 0:
        # 保守默认：连通图至少需要 (N-1) 条边，但太少会不稳；这里用一个经验下限。
        stop_min_edges = int(max(10 * (len(cameras) - 1), 20))

    poses_by_frame, paths_by_frame, edges, scan_report = _scan_pose_observations(
        image_root=image_root,
        cameras=cameras,
        intrinsics=intrinsics,
        config=config,
        max_frames=int(args.max_frames),
        min_tags=int(min_tags),
        scan_limits=scan_limits,
        scan_order=scan_order,
        workers=int(args.workers),
        prefetch=int(args.prefetch),
        cache_cfg=cache_cfg,
        prefilter_cfg=prefilter_cfg,
        use_multiscale=bool(use_multiscale),
        opencv_refine=bool(opencv_refine),
        stop_when_connected=not bool(args.no_stop_when_connected),
        stop_min_edges=int(stop_min_edges),
        max_reproj_mean_px=float(args.max_reproj_mean_px),
    )

    print(f"PnP 有效帧数: {len(poses_by_frame)}")
    print(f"位姿图边数量: {len(edges)}")

    if len(edges) == 0:
        # 失败也要落盘 scan_report，便于离线排查（尤其是 frame_key 匹配为 0 的情况）。
        _save_json(Path("results/step4_multi_scan_report.json"), scan_report)

        cand = int((scan_report.get("frame_sync", {}) or {}).get("candidate_frame_keys_ge2", 0))
        if cand == 0:
            print("错误: 没有构建出任何边：因为不存在任何‘至少两相机同帧(frame_key)’候选。")
            print("建议：统一各相机的文件命名规则，让同一时刻的帧拥有相同的 frame_key。")
            print("  - 你也可以直接重命名文件（例如去掉 cam1_/cam2_ 前缀），或按需改代码的 frame_key 解析规则。")
        else:
            print("错误: 没有构建出任何边（同帧共同看到板的相机对为 0）。")
            print("建议：增加采集姿态/光照；降低 min_tags；检查 tag family/ROI；确保内参与板模型正确。")

        print("已保存诊断报告: results/step4_multi_scan_report.json")
        return 1

    comps = _connected_components(cameras, edges)
    print(f"图连通分量数量: {len(comps)}")
    for i, comp in enumerate(comps, 1):
        print(f"  component#{i}: {comp}")

    if reference not in comps[0]:
        print("错误: 参考相机不在最大连通分量中，这通常表示数据命名/同步有问题。")
        return 1

    if len(comps[0]) < len(cameras):
        print("\n警告: 位姿图不连通，只能求出与参考相机同一连通分量中的相机外参。")
        print(f"  可解相机: {comps[0]}")

    cameras_solved = [c for c in cameras if c in comps[0]]
    edges_solved = [e for e in edges if (e.cam_i in cameras_solved and e.cam_j in cameras_solved)]

    # BA/位姿图共同：只保留可解相机集合内的观测
    poses_by_frame_solved: Dict[str, Dict[str, PoseObs]] = {}
    paths_by_frame_solved: Dict[str, Dict[str, Path]] = {}
    cams_solved_set = {str(c) for c in cameras_solved}
    for fk, poses in poses_by_frame.items():
        pp = {str(c): p for c, p in poses.items() if str(c) in cams_solved_set}
        if len(pp) < 2:
            continue
        poses_by_frame_solved[str(fk)] = pp
        paths_by_frame_solved[str(fk)] = {
            str(c): p for c, p in (paths_by_frame.get(fk) or {}).items() if str(c) in cams_solved_set
        }

    # ===== 诊断输出：PnP 重投影误差（像素） =====
    # 说明：
    # - 这个指标更直观，但它只是“像素空间拟合好坏”，不等价于“外参一定正确”。
    # - 仍建议结合 rot_deg / trans_mm 以及 worst edge 图片一起排查。
    reproj_stats = _summarize_reproj_px(
        cameras=cameras_solved,
        edges=edges_solved,
        poses_by_frame=poses_by_frame_solved,
        paths_by_frame=paths_by_frame_solved,
        top_k=5,
    )

    print("\nPnP 自身重投影误差统计（像素，越小越好；仅统计参与建图的观测）")
    for cam in cameras_solved:
        s = (reproj_stats.get("by_cam") or {}).get(str(cam)) or {}
        if int(s.get("n", 0)) <= 0:
            print(f"  - {cam}: n=0")
            continue
        print(
            f"  - {cam}: n={int(s['n'])} mean={float(s['mean']):.4f} "
            f"median={float(s['median']):.4f} p95={float(s['p95']):.4f} max={float(s['max']):.4f}"
        )

    es = reproj_stats.get("edge_mean_px") or {}
    if int(es.get("n", 0)) > 0:
        print(
            "\n边的平均重投影误差（像素，edge_mean_px=0.5*(cam_i+cam_j)；与边权重一致）"
            f"\n  - n={int(es['n'])} mean={float(es['mean']):.4f} median={float(es['median']):.4f} "
            f"p95={float(es['p95']):.4f} max={float(es['max']):.4f}"
        )

        worst = reproj_stats.get("worst_edges_by_edge_mean_px") or []
        if len(worst) > 0:
            print("\n诊断：edge_mean_px 最大的若干条边（更像‘检测/同步’问题的信号）")
            for i, m in enumerate(worst[: min(5, len(worst))], 1):
                p_i = (m.get("image_path") or {}).get("cam_i")
                p_j = (m.get("image_path") or {}).get("cam_j")
                short_i = _fmt_path_short(Path(p_i)) if p_i else ""
                short_j = _fmt_path_short(Path(p_j)) if p_j else ""
                print(
                    f"  #{i} pair={m.get('pair')} frame={m.get('frame_key')} "
                    f"edge_mean_px={float(m.get('edge_mean_px', 0.0)):.4f} "
                    f"paths=({short_i}) | ({short_j})"
                )

    # 初值：位姿图初始化对 BA 也很关键（只要能连通就行，不要求很准）
    X_init = _initial_poses_from_edges(cameras_solved, reference=reference, edges=edges_solved)
    if len(X_init) < len(cameras_solved):
        missing = [c for c in cameras_solved if c not in X_init]
        print(f"错误: 初始化失败，无法从 reference 到达相机：{missing}")
        return 1

    method = str(args.method)
    if method == "ba":
        print("\n开始多相机 Bundle Adjustment（像素域联合优化）...")

        frames, obs = _build_ba_frame_and_obs(
            cameras_solved=cameras_solved,
            poses_by_frame=poses_by_frame_solved,
        )
        if len(frames) == 0 or len(obs) == 0:
            print("错误: BA 没有可用帧/观测（需要至少一帧包含 >=2 相机有效观测）。")
            return 1

        Cref_T_B_init = _init_board_poses_in_ref(
            frames=frames,
            poses_by_frame=poses_by_frame_solved,
            X_cam_from_ref_init=X_init,
        )

        opt_cams = [c for c in cameras_solved if c != reference]
        x0 = _pack_ba_params(
            opt_cams=opt_cams,
            X_cam_from_ref_init=X_init,
            frames=frames,
            Cref_T_B_init=Cref_T_B_init,
        )

        fun = lambda x: _residuals_bundle_adjustment(
            x,
            opt_cams=opt_cams,
            reference=reference,
            frames=frames,
            observations=obs,
            intrinsics=intrinsics,
        )

        r0 = fun(x0)
        print(f"初始残差维度: {int(r0.size)}")
        print(f"初始 RMS(px): {float(np.sqrt(np.mean(r0 * r0))) if r0.size else 0.0:.6f}")

        res = least_squares(
            fun,
            x0,
            method="trf",
            loss="huber",
            f_scale=float(args.ba_f_scale_px),
            max_nfev=int(args.ba_max_nfev),
        )

        x_opt = res.x
        r1 = fun(x_opt)
        print(f"优化完成: success={bool(res.success)}, nfev={int(res.nfev)}")
        print(f"最终 RMS(px): {float(np.sqrt(np.mean(r1 * r1))) if r1.size else 0.0:.6f}")

        X_opt, Cref_T_B_opt = _unpack_ba_params(
            x=x_opt, opt_cams=opt_cams, reference=reference, frames=frames
        )

        # 可选：基于当前 BA 解做一次离群观测剔除再优化
        obs2, prune_info = _prune_ba_observations(
            X_cam_from_ref=X_opt,
            Cref_T_B_by_frame=Cref_T_B_opt,
            cameras=cameras_solved,
            intrinsics=intrinsics,
            observations=obs,
            prune_mean_px=float(args.ba_prune_mean_px),
            min_keep_per_cam=int(args.ba_prune_min_keep_per_cam),
        )

        # 最终用于统计/报告的观测集合：默认使用原始 obs；若剔除生效且二次优化，则使用 obs2。
        obs_used = list(obs)
        if bool(prune_info.get("enabled")) and int(prune_info.get("after", 0)) < int(prune_info.get("before", 0)):
            print(
                "\n离群观测剔除（基于 BA mean_px）："
                f" before={int(prune_info['before'])} after={int(prune_info['after'])}"
                f" dropped_frames={int(prune_info.get('dropped_frames', 0))}"
            )

            # 用第一次 BA 的解作为二次优化初值（更快更稳）
            x1 = x_opt
            fun2 = lambda x: _residuals_bundle_adjustment(
                x,
                opt_cams=opt_cams,
                reference=reference,
                frames=frames,
                observations=obs2,
                intrinsics=intrinsics,
            )
            res2 = least_squares(
                fun2,
                x1,
                method="trf",
                loss="huber",
                f_scale=float(args.ba_f_scale_px),
                max_nfev=int(args.ba_max_nfev),
            )
            x_opt = res2.x
            r1 = fun2(x_opt)
            res = res2
            X_opt, Cref_T_B_opt = _unpack_ba_params(
                x=x_opt, opt_cams=opt_cams, reference=reference, frames=frames
            )
            obs_used = list(obs2)
        else:
            prune_info = {"enabled": bool(prune_info.get("enabled", False)), "changed": False}

        poses_post, paths_post = _filter_pose_maps_by_observations(
            poses_by_frame=poses_by_frame_solved,
            paths_by_frame=paths_by_frame_solved,
            observations=obs_used,
        )

        ba_reproj = _summarize_ba_reproj_px(
            X_cam_from_ref=X_opt,
            Cref_T_B_by_frame=Cref_T_B_opt,
            cameras=cameras_solved,
            intrinsics=intrinsics,
            poses_by_frame=poses_post,
            paths_by_frame=paths_post,
            top_k=10,
        )

        print("\nBA 优化后的重投影误差（像素，越小越好）")
        ov = ba_reproj.get("overall") or {}
        if int(ov.get("n", 0)) > 0:
            print(
                f"  - overall: n={int(ov['n'])} mean={float(ov['mean']):.4f} "
                f"median={float(ov['median']):.4f} p95={float(ov['p95']):.4f} max={float(ov['max']):.4f}"
            )
        by_cam_post = ba_reproj.get("by_cam") or {}
        for cam in cameras_solved:
            s = by_cam_post.get(str(cam)) or {}
            if int(s.get("n", 0)) <= 0:
                continue
            print(
                f"  - {cam}: n={int(s['n'])} mean={float(s['mean']):.4f} "
                f"median={float(s['median']):.4f} p95={float(s['p95']):.4f} max={float(s['max']):.4f}"
            )

        worst_post = ba_reproj.get("worst") or []
        if len(worst_post) > 0:
            print("\n诊断：BA 后误差最大的若干条观测（用于定位坏帧/离群）")
            for i, m in enumerate(worst_post[: min(5, len(worst_post))], 1):
                p = m.get("image_path")
                short = _fmt_path_short(Path(p)) if p else ""
                print(
                    f"  #{i} cam={m.get('cam')} frame={m.get('frame_key')} "
                    f"mean_px={float(m.get('mean_px', 0.0)):.4f} n_points={int(m.get('n_points', 0))} path=({short})"
                )

        post_reproj = ba_reproj
        solver_rms = float(np.sqrt(np.mean(r1 * r1))) if r1.size else 0.0
        solver_units = "px"
        solver_extra = {
            "method": "bundle_adjustment",
            "ba_f_scale_px": float(args.ba_f_scale_px),
            "ba_max_nfev": int(args.ba_max_nfev),
            "ba_prune": prune_info,
        }
    else:
        x0, opt_cams = _pack_params(cameras_solved, reference=reference, X_init=X_init)

        # 优化
        print("\n开始位姿图优化...")
        fun = lambda x: _residuals_pose_graph(x, opt_cams=opt_cams, reference=reference, edges=edges_solved)
        r0 = fun(x0)
        print(f"初始残差维度: {r0.size}")
        print(f"初始 RMS(6D): {float(np.sqrt(np.mean(r0 * r0))):.6f}")

        res = least_squares(
            fun,
            x0,
            method="trf",
            loss="huber",
            f_scale=float(args.huber),
            max_nfev=2000,
        )

        x_opt = res.x
        r1 = fun(x_opt)
        print(f"优化完成: success={bool(res.success)}, nfev={int(res.nfev)}")
        print(f"最终 RMS(6D): {float(np.sqrt(np.mean(r1 * r1))):.6f}")

        X_opt = _unpack_params(x_opt, opt_cams=opt_cams, reference=reference)

        # 外参优化完成后的“多相机一致性”重投影误差（像素）
        post_reproj = _summarize_post_extrinsics_reproj_px(
            X_cam_from_ref=X_opt,
            reference=reference,
            cameras=cameras_solved,
            intrinsics=intrinsics,
            poses_by_frame=poses_by_frame_solved,
            paths_by_frame=paths_by_frame_solved,
            top_k=10,
        )

        print("\n外参优化完成后的多相机一致性重投影误差（像素，越小越好）")
        ov = post_reproj.get("overall") or {}
        if int(ov.get("n", 0)) > 0:
            print(
                f"  - overall: n={int(ov['n'])} mean={float(ov['mean']):.4f} "
                f"median={float(ov['median']):.4f} p95={float(ov['p95']):.4f} max={float(ov['max']):.4f}"
            )

        by_cam_post = post_reproj.get("by_cam") or {}
        for cam in cameras_solved:
            s = by_cam_post.get(str(cam)) or {}
            if int(s.get("n", 0)) <= 0:
                continue
            print(
                f"  - {cam}: n={int(s['n'])} mean={float(s['mean']):.4f} "
                f"median={float(s['median']):.4f} p95={float(s['p95']):.4f} max={float(s['max']):.4f}"
            )

        worst_post = post_reproj.get("worst") or []
        if len(worst_post) > 0:
            print("\n诊断：外参后重投影误差最大的若干条观测（用于定位坏帧/不同步）")
            for i, m in enumerate(worst_post[: min(5, len(worst_post))], 1):
                p = m.get("image_path")
                short = _fmt_path_short(Path(p)) if p else ""
                print(
                    f"  #{i} cam={m.get('cam')} frame={m.get('frame_key')} "
                    f"mean_px={float(m.get('mean_px', 0.0)):.4f} n_points={int(m.get('n_points', 0))} path=({short})"
                )

        solver_rms = float(np.sqrt(np.mean(r1 * r1))) if r1.size else 0.0
        solver_units = "6D"
        solver_extra = {"method": "pose_graph", "huber_f_scale": float(args.huber)}

    # 输出更直观的误差统计（不要只看 RMS(6D)，它混合了角度与长度单位）
    edge_err = _summarize_edge_errors(X=X_opt, edges=edges_solved)
    print(
        "\n边误差统计（更直观）："
        f"\n  - rot_deg: median={edge_err['rot_deg']['median']:.3f}, p95={edge_err['rot_deg']['p95']:.3f}, max={edge_err['rot_deg']['max']:.3f}"
        f"\n  - trans_mm: median={edge_err['trans']['median']:.3f}, p95={edge_err['trans']['p95']:.3f}, max={edge_err['trans']['max']:.3f}"
    )

    # 保存边级诊断：输出误差最大的若干条边，直接定位是哪一帧/哪对相机不一致。
    edge_diag_path = str(args.edge_diag_path).strip()
    if edge_diag_path:
        diag = _save_edge_diagnostics(
            path=Path(edge_diag_path),
            X=X_opt,
            edges=edges_solved,
            poses_by_frame=poses_by_frame,
            paths_by_frame=paths_by_frame,
            top_k=int(args.edge_diag_top_k),
            meta={
                "timestamp": datetime.now().isoformat(),
                "image_root": str(image_root.as_posix()),
                "intrinsics_dir": str(intr_dir.as_posix()),
                "reference": str(reference),
                "cameras_solved": list(cameras_solved),
                "min_tags": int(min_tags),
                "max_reproj_mean_px": float(args.max_reproj_mean_px),
            },
        )

        print("\n诊断：误差最大的若干条边（用于定位坏帧/不同步）")
        top_rot = (diag.get("ranking", {}) or {}).get("by_rot_deg", [])
        for i, m in enumerate(top_rot[: min(5, len(top_rot))], 1):
            p_i = (m.get("image_path") or {}).get("cam_i")
            p_j = (m.get("image_path") or {}).get("cam_j")
            short_i = _fmt_path_short(Path(p_i)) if p_i else ""
            short_j = _fmt_path_short(Path(p_j)) if p_j else ""
            print(
                f"  #{i} pair={m.get('pair')} frame={m.get('frame_key')} "
                f"rot_deg={float(m.get('rot_deg', 0.0)):.3f} trans_mm={float(m.get('trans_mm', 0.0)):.3f} "
                f"paths=({short_i}) | ({short_j})"
            )

        print(f"已保存边级诊断: {edge_diag_path}")

    # 输出
    T_cam_from_ref: Dict[str, Dict[str, Any]] = {}
    for cam in cameras_solved:
        T = X_opt[cam]
        T_cam_from_ref[cam] = {
            "T": T.tolist(),
            "R": T[:3, :3].tolist(),
            "t": T[:3, 3].tolist(),
        }

    out_extr: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "reference": reference,
        "cameras": cameras,
        "cameras_solved": cameras_solved,
        "note": "T_cam_from_ref 表示 Cam_i <- Cam_ref（与本工程 Cr_T_Cl 命名一致：目标在前，源在后）",
        "T_cam_from_ref": T_cam_from_ref,
        "solver": {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(res.nfev),
            "cost": float(res.cost),
            "rms_final": float(solver_rms),
            "rms_units": str(solver_units),
            **solver_extra,
        },
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "image_root": str(image_root.as_posix()),
        "intrinsics_dir": str(intr_dir.as_posix()),
        "reference": reference,
        "min_tags": int(min_tags),
        "poses_frames": int(len(poses_by_frame)),
        "edges_total": int(len(edges)),
        "edges_used": int(len(edges_solved)),
        "components": comps,
        "reproj_error_px": reproj_stats,
        "post_reproj_error_px": post_reproj,
        "solver": {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(res.nfev),
            "cost": float(res.cost),
            "rms_final": float(solver_rms),
            "rms_units": str(solver_units),
            **solver_extra,
        },
        "edge_error": edge_err,
    }

    _save_json(Path("results/multi_camera_extrinsics.json"), out_extr)
    _save_json(Path("results/multi_camera_pose_graph_report.json"), report)

    # 单独输出 scan/perf 报告，便于对比不同参数的检测耗时、缓存命中率等。
    _save_json(Path("results/step4_multi_scan_report.json"), scan_report)

    print("\n已保存:")
    print("  - results/multi_camera_extrinsics.json")
    print("  - results/multi_camera_pose_graph_report.json")
    print("  - results/step4_multi_scan_report.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
