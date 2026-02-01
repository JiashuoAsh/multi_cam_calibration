"""Step3（内参标定）：可复用的纯逻辑。

设计约束：
- 该模块只包含“纯数学/统计/去重/误差计算”逻辑。
- 不做文件 IO（不读写图片/JSON/目录），不依赖 entry/adapters。

入口层（`mcca.entry.step3_intrinsic_apriltag`）负责：
- 参数解析、扫描图片、并行检测（含缓存/预筛选/早停）、落盘输出与可视化。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Step3DetectResult:
    """Step3 单张图像检测摘要（用于早停与统计）。"""

    image_path: str
    valid: bool
    n_in_board: int
    status: int
    from_cache: bool
    elapsed_ms: float


def percentile(values: Sequence[float], q: float) -> float:
    """计算分位数（空输入返回 0）。"""

    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(list(values), dtype=np.float64), float(q)))


def summarize_step3_results(results: Sequence[Step3DetectResult]) -> Dict[str, Any]:
    """汇总 Step3 扫描阶段的检测统计。

    约定：
    - status: 0=正常检测；1=异常；2=prefilter 跳过（由 adapters.apriltag_perf 约定）。
    - elapsed_ms: 单次 detect 的耗时（缓存命中时通常接近 0）。
    """

    total = int(len(results))
    valid = int(sum(1 for r in results if bool(r.valid)))
    cache_hit = int(sum(1 for r in results if bool(r.from_cache)))
    prefilter_skipped = int(sum(1 for r in results if int(r.status) == 2))
    error = int(sum(1 for r in results if int(r.status) == 1))

    detect_ms = [
        float(r.elapsed_ms)
        for r in results
        if (not bool(r.from_cache)) and float(r.elapsed_ms) > 0.0
    ]
    mean_ms = float(np.mean(np.asarray(detect_ms, dtype=np.float64))) if len(detect_ms) > 0 else 0.0
    p95_ms = percentile(detect_ms, 95.0) if len(detect_ms) > 0 else 0.0

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


def view_signature(
    img_pts: np.ndarray,
    image_size: Tuple[int, int],
    *,
    bins: int,
) -> Tuple[int, int, int, int]:
    """为单张视图生成“近似姿态签名”，用于去重。

    签名由以下量化特征组成：
    1) 检测点云质心 (cx, cy)
    2) 点云尺度（平均半径）
    3) 点云主方向（2D PCA 主轴角度）

    Args:
        img_pts: (N,2) 2D 点（像素坐标）。
        image_size: (w, h)。
        bins: 量化桶数量。越小越容易合并相似视图。

    Returns:
        4 维整数签名。
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
        ang = 0.5 * float(
            np.arctan2(2.0 * float(cov[0, 1]), float(cov[0, 0] - cov[1, 1]))
        )
    except Exception:
        ang = 0.0

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


def dedup_views(
    *,
    all_obj_pts: Sequence[np.ndarray],
    all_img_pts: Sequence[np.ndarray],
    valid_images: Sequence[str],
    image_size: Tuple[int, int],
    bins: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], Dict[str, Any]]:
    """按视图签名去重（同签名冲突时保留约束更强的一张）。"""

    n0 = int(len(valid_images))
    if n0 == 0:
        return (
            list(all_obj_pts),
            list(all_img_pts),
            list(valid_images),
            {"before": 0, "after": 0, "removed": 0, "bins": int(bins)},
        )

    best: Dict[Tuple[int, int, int, int], int] = {}

    for i in range(n0):
        sig = view_signature(np.asarray(all_img_pts[i]), image_size, bins=int(bins))
        score = int(np.asarray(all_img_pts[i]).reshape(-1, 2).shape[0])

        j = best.get(sig)
        if j is None:
            best[sig] = i
        else:
            score_j = int(np.asarray(all_img_pts[j]).reshape(-1, 2).shape[0])
            if score > score_j:
                best[sig] = i

    keep_idx = sorted(set(best.values()))

    obj2 = [np.asarray(all_obj_pts[i]) for i in keep_idx]
    img2 = [np.asarray(all_img_pts[i]) for i in keep_idx]
    files2 = [str(valid_images[i]) for i in keep_idx]

    n1 = int(len(files2))
    return obj2, img2, files2, {"before": n0, "after": n1, "removed": int(n0 - n1), "bins": int(bins)}


def calibrate_camera_intrinsics(
    all_obj_pts: Sequence[np.ndarray],
    all_img_pts: Sequence[np.ndarray],
    image_size: Tuple[int, int],
) -> Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float]:
    """调用 OpenCV 的 calibrateCamera 计算内参。

    Returns:
        ret_rms_px: OpenCV 返回的 RMS（像素）。
        K: (3,3)
        dist: (N,1) 或 (1,N)
        rvecs, tvecs: 每视图一组（object->camera）。
        elapsed_s: 标定耗时（秒）。
    """

    none_umat: Any = None
    t0 = time.perf_counter()
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        list(all_obj_pts),
        list(all_img_pts),
        tuple(image_size),
        none_umat,
        none_umat,
    )
    t1 = time.perf_counter()

    return float(ret), np.asarray(K), np.asarray(dist), list(rvecs), list(tvecs), float(t1 - t0)


def compute_reprojection_errors(
    *,
    all_obj_pts: Sequence[np.ndarray],
    all_img_pts: Sequence[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
) -> Tuple[float, List[float]]:
    """计算重投影误差（每视图 mean + 全局 mean）。

    说明：
    - 这里的 mean_error 是“每张图像的 per-point 误差均值，再对所有图像求均值”。
    - 与 OpenCV 的 ret(RMS) 不同，但更直观。
    """

    n = int(len(all_obj_pts))
    if n == 0:
        return 0.0, []

    per_image: List[float] = []
    total = 0.0

    for i in range(n):
        img_pts2, _ = cv2.projectPoints(
            np.asarray(all_obj_pts[i]),
            np.asarray(rvecs[i]),
            np.asarray(tvecs[i]),
            np.asarray(K),
            np.asarray(dist),
        )
        proj = np.asarray(img_pts2, dtype=np.float64).reshape(-1, 2)
        det = np.asarray(all_img_pts[i], dtype=np.float64).reshape(-1, 2)
        if proj.shape[0] == 0 or det.shape[0] == 0:
            per_image.append(0.0)
            continue

        per_pt = np.linalg.norm(det - proj, axis=1)
        err = float(np.mean(per_pt))
        per_image.append(err)
        total += err

    mean_error = float(total / float(max(1, len(per_image))))
    return mean_error, per_image


def count_total_points(all_img_pts: Iterable[np.ndarray]) -> int:
    """统计总点数（用于可观测性日志）。"""

    return int(sum(int(np.asarray(pts).reshape(-1, 2).shape[0]) for pts in all_img_pts))
