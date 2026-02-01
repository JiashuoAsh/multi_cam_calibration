"""Step4（多相机外参）：扫描/检测/建图的 IO+并行层实现。

该模块的职责是“把磁盘上的多相机图片”转换为 Step4 核心求解可消费的数据结构：
- 每帧每相机的 PnP 位姿观测（`PoseObs`）
- 相机对之间的相对位姿边（`EdgeObs`）
- 以及一份可观测性的扫描报告（scan_report）

设计约束：
- 依赖方向：adapters -> core。
- 不包含任何 CLI/argparse。
- 不直接写入 results 文件（由 entry 层决定落盘位置与命名）。

注意：
- 该模块为了多进程性能，使用了模块级全局变量缓存 worker 状态。
  这些全局变量只在子进程中初始化与使用；主进程只调用 `scan_pose_observations()`。
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from itertools import combinations
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
from mcca.core.board import create_apriltag_board, create_opencv_aruco_board, get_aruco_dict
from mcca.core.detection import (
    create_detector_params,
    get_detection_auto_roi,
    get_detection_profile,
    get_detection_roi,
)
from mcca.core.rigid import invert_T, make_T
from mcca.core.step4_multicam import CameraIntrinsics, EdgeObs, PoseObs


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


def _normalize_frame_key(stem: str, cam: str) -> str:
    """从文件名 stem 生成用于“同帧匹配”的 frame_key。

    保守归一化：若 stem 以 "{cam}_" 或 "{cam}-" 开头，则去掉该前缀。
    """

    stem = str(stem)
    cam = str(cam)
    if stem.startswith(cam + "_"):
        return stem[len(cam) + 1 :]
    if stem.startswith(cam + "-"):
        return stem[len(cam) + 1 :]
    return stem


def _build_frame_map(*, cam_dir: Path, verbose: bool) -> Dict[str, Path]:
    """构建 frame_key -> Path 的映射（suffix 统一用 lower() 兼容大小写后缀）。"""

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
        _vprint(verbose, f"警告: {cam} 存在 {dup} 个 frame_key 冲突（归一化后重名），已保留最先出现的文件。")
    return out


def _collect_correspondences_fast(
    *,
    corners: List[np.ndarray],
    ids: np.ndarray,
    obj_points: np.ndarray,
    tag_id_to_idx: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """根据检测到的 tags，组装 solvePnP 需要的 (object_points, image_points)。"""

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


# --- 多进程 worker 全局状态 ---

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


def _init_multicam_frame_worker(state: Dict[str, Any]) -> None:
    """多进程 worker 初始化：在子进程内创建 OpenCV 检测器与缓存服务。"""

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

    if (
        _G_DET_BY_CAM is None
        or _G_FRAME_MAPS is None
        or _G_OBJ_POINTS_ALL is None
        or _G_TAG_ID_TO_IDX is None
    ):
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
            C_T_B = make_T(Rm, t, "C_T_B")

            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            proj = proj.reshape(-1, 2)
            det_pts = img_pts.reshape(-1, 2).astype(np.float64)
            per_pt = np.linalg.norm(det_pts - proj, axis=1)
            reproj_mean = float(np.mean(per_pt)) if per_pt.size > 0 else float("inf")

            poses_by_cam[cam] = {
                "C_T_B": np.asarray(C_T_B, dtype=np.float64).tolist(),
                "n_tags": int(used_tags),
                "reproj_mean_px": float(reproj_mean),
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


def scan_pose_observations(
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
    verbose: bool,
) -> Tuple[Dict[str, Dict[str, PoseObs]], Dict[str, Dict[str, Path]], List[EdgeObs], Dict[str, Any]]:
    """扫描并构建 poses/edges。

    Returns:
        poses_by_frame: frame_key -> cam -> PoseObs
        paths_by_frame: frame_key -> cam -> 图片路径
        edges: 位姿图边观测
        scan_report: 可观测性报告（用于落盘诊断）
    """

    profile = get_detection_profile(config)
    auto_roi_cfg = get_detection_auto_roi(config)

    obj_points_mm, tag_ids = create_apriltag_board(config)
    family = config["apriltag_board"]["family"]

    frame_maps_p: Dict[str, Dict[str, Path]] = {}
    for cam in cameras:
        cam_dir = image_root / cam
        if not cam_dir.exists():
            raise FileNotFoundError(f"未找到相机目录：{cam_dir}")
        frame_maps_p[cam] = _build_frame_map(cam_dir=cam_dir, verbose=verbose)

    pair_intersections: Dict[str, Any] = {}
    cams_list = list(cameras)
    for i in range(len(cams_list)):
        for j in range(i + 1, len(cams_list)):
            a = str(cams_list[i])
            b = str(cams_list[j])
            inter = set(frame_maps_p[a].keys()) & set(frame_maps_p[b].keys())
            pair_intersections[f"{a}__{b}"] = {"n": int(len(inter)), "sample": sorted(list(inter))[:10]}

    per_cam_frames = {str(cam): int(len(mp)) for cam, mp in frame_maps_p.items()}
    _vprint(verbose, f"每相机可用图片数量(frame_key 去重后): {per_cam_frames}")

    key_counts: Dict[str, int] = defaultdict(int)
    for cam in cameras:
        for k in frame_maps_p[cam].keys():
            key_counts[str(k)] += 1

    keys = sorted([k for k, n in key_counts.items() if int(n) >= 2])
    if int(max_frames) > 0:
        keys = keys[: int(max_frames)]

    if len(keys) == 0:
        print("错误: 没有找到任何‘至少两相机同名帧’的候选 frame_key，因此不可能构建位姿图边。")
        print("  这通常是帧同步/命名规则不一致导致的：脚本用同名 frame_key 作为‘同一时刻’判据。")
        print(f"  每相机可用图片数: {per_cam_frames}")
        for cam in cameras:
            sample = sorted(list(frame_maps_p[cam].keys()))[:10]
            print(f"  {cam} sample frame_key: {sample}")

        for i in range(len(cams_list)):
            for j in range(i + 1, len(cams_list)):
                a = cams_list[i]
                b = cams_list[j]
                inter = set(frame_maps_p[a].keys()) & set(frame_maps_p[b].keys())
                inter_sample = sorted(list(inter))[:10]
                print(f"  intersection({a}, {b}) = {len(inter)} sample={inter_sample}")

    frame_maps: Dict[str, Dict[str, str]] = {
        cam: {k: str(p) for k, p in mp.items()} for cam, mp in frame_maps_p.items()
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

    det_total = 0
    det_cache_hit = 0
    det_cache_miss = 0
    det_prefilter_skipped = 0
    det_error = 0
    det_ms_sum = 0.0
    pnp_ok = 0
    pnp_fail = 0
    pose_drop_by_reproj = 0

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

            if len(poses_this) >= 2:
                poses_by_frame[frame_key] = poses_this
                paths_by_frame[frame_key] = paths_this

                cams = sorted(list(poses_this.keys()))
                for cam_i, cam_j in combinations(cams, 2):
                    pi = poses_this[cam_i]
                    pj = poses_this[cam_j]
                    Ci_T_Cj = pi.C_T_B @ invert_T(pj.C_T_B, "B_T_Cj")

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

    if int(scan_limits.max_total) <= 0 and int(scan_limits.target_valid) > 0:
        scan_limits = ScanLimits(
            target_valid=int(scan_limits.target_valid),
            max_total=int(
                min(
                    len(keys),
                    max(8 * int(scan_limits.target_valid), int(scan_limits.target_valid)),
                )
            ),
            max_seconds=float(scan_limits.max_seconds),
        )

    if int(workers) <= 1:
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
        "pnp": {"ok": int(pnp_ok), "fail": int(pnp_fail)},
        "filters": {
            "max_reproj_mean_px": float(max_reproj_mean_px),
            "poses_dropped_by_reproj": int(pose_drop_by_reproj),
        },
        "results": {"poses_frames": int(len(poses_by_frame)), "edges": int(len(edges))},
        "frame_sync": {
            "per_cam_frames": per_cam_frames,
            "candidate_frame_keys_ge2": int(len(keys)),
            "pair_intersections": pair_intersections,
            "note": "frame_key 来自文件名 stem（会自动去掉形如 <cam>_ / <cam>- 的前缀）。",
        },
    }

    return poses_by_frame, paths_by_frame, edges, scan_report
