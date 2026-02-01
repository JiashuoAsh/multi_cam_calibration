from __future__ import annotations

import numpy as np

from mcca.core.step2_filter import (
    Step2ImageResult,
    compute_frame_stats,
    summarize_step2_results,
)


def test_summarize_step2_results_empty() -> None:
    # 空输入要返回 0 统计，且不应抛异常。
    s = summarize_step2_results([])
    assert s["total"] == 0
    assert s["valid"] == 0
    assert s["cache_hit"] == 0
    assert s["cache_miss"] == 0
    assert s["prefilter_skipped"] == 0
    assert s["error"] == 0
    assert s["mean_detect_ms"] == 0.0
    assert s["p95_detect_ms"] == 0.0


def test_summarize_step2_results_mixed() -> None:
    # 说明：mean/p95 只统计非 cache 命中的 detect 耗时。
    rs = [
        Step2ImageResult(
            image_path="a.png",
            valid=True,
            n_tags=10,
            status=0,
            from_cache=False,
            elapsed_ms=10.0,
        ),
        Step2ImageResult(
            image_path="b.png",
            valid=False,
            n_tags=0,
            status=2,  # prefilter skipped
            from_cache=False,
            elapsed_ms=0.0,
        ),
        Step2ImageResult(
            image_path="c.png",
            valid=False,
            n_tags=0,
            status=1,  # error
            from_cache=False,
            elapsed_ms=5.0,
        ),
        Step2ImageResult(
            image_path="d.png",
            valid=True,
            n_tags=8,
            status=0,
            from_cache=True,
            elapsed_ms=0.2,
        ),
    ]

    s = summarize_step2_results(rs)
    assert s["total"] == 4
    assert s["valid"] == 2
    assert s["cache_hit"] == 1
    assert s["cache_miss"] == 3
    assert s["prefilter_skipped"] == 1
    assert s["error"] == 1

    # 只统计 from_cache=False 且 elapsed_ms>0 的样本：10.0 和 5.0
    detect_ms = np.asarray([10.0, 5.0], dtype=np.float64)
    assert s["mean_detect_ms"] == float(np.mean(detect_ms))
    assert s["p95_detect_ms"] == float(np.percentile(detect_ms, 95.0))


def test_compute_frame_stats() -> None:
    # frame_key -> 合格相机列表
    m = {
        "f1": ["cam0"],
        "f2": ["cam0", "cam1"],
        "f3": ["cam0", "cam1", "cam2"],
        "f4": ["cam1", "cam2"],
    }
    s = compute_frame_stats(m, n_cameras=3)

    assert s["unique_frame_keys_with_any_valid"] == 4
    assert s["frame_keys_with_at_least_2_cameras_valid"] == 3  # f2,f3,f4
    assert s["frame_keys_with_all_cameras_valid"] == 1  # f3
