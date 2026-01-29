from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class PrefilterConfig:
    """廉价预筛选配置（可选）。

    设计目标：
    - 极低成本排除“必然检测不到/检测极不稳定”的帧，减少进入 AprilTag 检测器的次数。
    - 默认阈值应当偏保守：宁可少过滤，也尽量不误杀可用帧。

    注意：
    - 预筛选只基于灰度图的简单统计，并不理解标定板语义。
    - 若你的数据集光照/对焦非常极端，建议关闭或调整阈值。
    """

    enabled: bool = False

    # 亮度/对比度
    min_mean: float = 15.0
    max_mean: float = 245.0
    min_std: float = 5.0

    # 过曝/欠曝比例（0~1）
    max_overexposed_ratio: float = 0.98
    max_underexposed_ratio: float = 0.98

    # 模糊度（Laplacian 方差）
    min_laplacian_var: float = 10.0

    # 边缘密度（Canny 边缘像素占比）
    min_edge_ratio: float = 0.002


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def quick_image_metrics(gray: np.ndarray) -> Dict[str, float]:
    """计算用于预筛选的一组“便宜”指标。"""

    g = np.asarray(gray)
    if g.ndim != 2:
        raise ValueError("gray 必须是 2D 灰度图")

    g_f = g.astype(np.float32)
    mean = float(np.mean(g_f))
    std = float(np.std(g_f))

    # 过曝/欠曝：用像素阈值做比例统计
    over = float(np.mean(g_f >= 250.0))
    under = float(np.mean(g_f <= 5.0))

    # Laplacian var + Canny edge ratio 需要 OpenCV；按需导入，避免测试环境强依赖。
    lap_var = float("nan")
    edge_ratio = float("nan")
    try:
        import cv2

        lap = cv2.Laplacian(g, cv2.CV_32F)
        lap_var = float(lap.var())

        edges = cv2.Canny(g, 50, 150)
        edge_ratio = float(np.mean(edges > 0))
    except Exception:
        # 没有 OpenCV 或计算失败时，保留 NaN，由调用方决定如何处理。
        pass

    return {
        "mean": mean,
        "std": std,
        "over_ratio": over,
        "under_ratio": under,
        "laplacian_var": lap_var,
        "edge_ratio": edge_ratio,
    }


def prefilter_pass(gray: np.ndarray, cfg: PrefilterConfig) -> Tuple[bool, Dict[str, float]]:
    """预筛选：返回 (pass, metrics)。"""

    metrics = quick_image_metrics(gray)

    if not bool(cfg.enabled):
        return True, metrics

    mean = _safe_float(metrics.get("mean"), 0.0)
    std = _safe_float(metrics.get("std"), 0.0)
    over = _safe_float(metrics.get("over_ratio"), 0.0)
    under = _safe_float(metrics.get("under_ratio"), 0.0)

    if mean < float(cfg.min_mean) or mean > float(cfg.max_mean):
        return False, metrics
    if std < float(cfg.min_std):
        return False, metrics

    if over > float(cfg.max_overexposed_ratio):
        return False, metrics
    if under > float(cfg.max_underexposed_ratio):
        return False, metrics

    # Laplacian/Canny 指标在缺 OpenCV 时可能是 NaN：这种情况下不因为 NaN 而拒绝。
    lap_var = metrics.get("laplacian_var")
    if lap_var is not None and np.isfinite(float(lap_var)):
        if float(lap_var) < float(cfg.min_laplacian_var):
            return False, metrics

    edge_ratio = metrics.get("edge_ratio")
    if edge_ratio is not None and np.isfinite(float(edge_ratio)):
        if float(edge_ratio) < float(cfg.min_edge_ratio):
            return False, metrics

    return True, metrics
