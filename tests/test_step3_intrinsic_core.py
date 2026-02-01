from __future__ import annotations

import numpy as np

from mcca.core.step3_intrinsic import (
    Step3DetectResult,
    count_total_points,
    dedup_views,
    summarize_step3_results,
    view_signature,
)


def test_summarize_step3_results_empty() -> None:
    # 空输入要返回 0 统计，且不应抛异常。
    s = summarize_step3_results([])
    assert s["total"] == 0
    assert s["valid"] == 0
    assert s["cache_hit"] == 0
    assert s["cache_miss"] == 0
    assert s["prefilter_skipped"] == 0
    assert s["error"] == 0
    assert s["mean_detect_ms"] == 0.0
    assert s["p95_detect_ms"] == 0.0


def test_summarize_step3_results_mixed() -> None:
    rs = [
        Step3DetectResult(
            image_path="a.png",
            valid=True,
            n_in_board=10,
            status=0,
            from_cache=False,
            elapsed_ms=10.0,
        ),
        Step3DetectResult(
            image_path="b.png",
            valid=False,
            n_in_board=0,
            status=2,  # prefilter skipped
            from_cache=False,
            elapsed_ms=0.0,
        ),
        Step3DetectResult(
            image_path="c.png",
            valid=False,
            n_in_board=0,
            status=1,  # error
            from_cache=False,
            elapsed_ms=5.0,
        ),
        Step3DetectResult(
            image_path="d.png",
            valid=True,
            n_in_board=8,
            status=0,
            from_cache=True,
            elapsed_ms=0.2,
        ),
    ]

    s = summarize_step3_results(rs)
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


def test_view_signature_and_dedup_prefers_more_points() -> None:
    # 构造两张“签名完全相同”的视图：第二张重复点，使点数更多。
    image_size = (640, 480)
    pts_base = np.asarray(
        [[100.0, 100.0], [120.0, 100.0], [120.0, 120.0], [100.0, 120.0]], dtype=np.float64
    )
    pts_more = np.vstack([pts_base, pts_base]).astype(np.float64)

    sig1 = view_signature(pts_base, image_size, bins=16)
    sig2 = view_signature(pts_more, image_size, bins=16)
    assert sig1 == sig2

    obj1 = np.zeros((pts_base.shape[0], 3), dtype=np.float32)
    obj2 = np.zeros((pts_more.shape[0], 3), dtype=np.float32)

    obj_out, img_out, files_out, dd = dedup_views(
        all_obj_pts=[obj1, obj2],
        all_img_pts=[pts_base.astype(np.float32), pts_more.astype(np.float32)],
        valid_images=["a.png", "b.png"],
        image_size=image_size,
        bins=16,
    )

    assert dd["before"] == 2
    assert dd["after"] == 1
    assert dd["removed"] == 1

    # 应保留点数更多的那张（b.png）
    assert files_out == ["b.png"]
    assert int(np.asarray(img_out[0]).reshape(-1, 2).shape[0]) == int(pts_more.shape[0])


def test_count_total_points() -> None:
    pts1 = np.zeros((4, 2), dtype=np.float32)
    pts2 = np.zeros((8, 2), dtype=np.float32)
    assert count_total_points([pts1, pts2]) == 12
