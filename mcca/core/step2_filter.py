"""Step2（筛图）：可复用的数据结构与统计逻辑。

设计约束：
- 该模块只包含“纯逻辑/统计”，不做文件 IO、不做 AprilTag 检测，也不依赖 adapters/entry。
- 入口层（entry）负责：扫描图片、调用检测器、复制文件、落盘报告。

说明：
- Step2 的核心输出是 `results/filter_report.json`，其中包含每相机统计与“同帧共视”的帧级统计。
- 这里把统计汇总单独抽出，便于单元测试与跨 step 复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Step2ImageResult:
    """Step2 单张图片的筛选结果（并行 worker 的最小返回结构）。"""

    image_path: str
    valid: bool
    n_tags: int
    status: int
    from_cache: bool
    elapsed_ms: float


def _percentile(values: Sequence[float], q: float) -> float:
    if len(values) == 0:
        return 0.0
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(arr, float(q)))


def summarize_step2_results(results: Sequence[Step2ImageResult]) -> Dict[str, Any]:
    """汇总 Step2 的检测/筛选统计。

    约定：
    - status: 0=正常检测；1=异常；2=prefilter 跳过（由 adapters.apriltag_perf 约定）。
    - elapsed_ms: 单次 detect 的耗时（缓存命中时通常接近 0）。

    Returns:
        JSON 友好 dict。
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


def compute_frame_stats(
    frame_key_to_valid_cams: Mapping[str, Sequence[str]],
    *,
    n_cameras: int,
) -> Dict[str, int]:
    """根据 frame_key -> 合格相机列表，计算帧级统计。

    该统计用于预判 Step4 是否“有边/可连通”：
    - 至少 2 路相机同帧合格，才能形成位姿图边。

    Args:
        frame_key_to_valid_cams: key 为文件 stem（或其它 sync_key），value 为该帧合格的相机名列表。
        n_cameras: 总相机数量。

    Returns:
        JSON 友好 dict。
    """

    n_frames_any = int(len(frame_key_to_valid_cams))
    n_frames_ge2 = int(sum(1 for _k, v in frame_key_to_valid_cams.items() if len(v) >= 2))
    n_frames_all = int(sum(1 for _k, v in frame_key_to_valid_cams.items() if len(v) == int(n_cameras)))

    return {
        "unique_frame_keys_with_any_valid": n_frames_any,
        "frame_keys_with_at_least_2_cameras_valid": n_frames_ge2,
        "frame_keys_with_all_cameras_valid": n_frames_all,
    }
