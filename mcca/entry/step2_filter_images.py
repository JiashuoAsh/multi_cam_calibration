#!/usr/bin/env python3
"""Step2 入口：筛选包含 AprilTag 标定板的图像。

职责边界：
- 入口层负责：参数解析、读取配置、扫描图片、调用检测器（含缓存/并行/预筛选）、复制合格图片、落盘报告。
- 纯统计/汇总逻辑放在 `mcca.core.step2_filter`，便于测试与复用。

输入：
- images/raw/<cam>/*.(png|jpg|jpeg|bmp) 或 config.image_dataset.cameras.*.raw_dir/raw_glob

输出：
- images/filtered/<cam>/*
- results/filter_report.json
- 可视化（可选）：results/visualization/step2_filtering/<cam>/*
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
from mcca.core.step2_filter import (
    Step2ImageResult,
    compute_frame_stats,
    summarize_step2_results,
)


# 默认尽量安静：只输出关键进度和汇总；需要逐张输出用 --verbose。
VERBOSE: bool = True


def _vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


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
        "stage": "step2_filter",
    }


_G_STEP2_DET: Optional[CachedAprilTagDetector] = None
_G_STEP2_MIN_TAGS: int = 0


def _init_step2_worker(state: Dict[str, Any]) -> None:
    """多进程 worker 初始化：在子进程内创建检测器（Windows spawn 安全）。"""

    global _G_STEP2_DET, _G_STEP2_MIN_TAGS

    config = state["config"]
    profile = str(state["profile"])
    family = str(state["family"])

    aruco_dict = get_aruco_dict(family)
    obj_points_mm = np.asarray(state["obj_points_mm"], dtype=np.float32)
    tag_ids = [int(x) for x in state["tag_ids"]]
    board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)
    detector_params = create_detector_params(config)

    algo_key = _build_algo_key(config, profile=profile)
    cache_cfg = CacheConfig(
        enabled=bool(state["cache"]["enabled"]),
        cache_dir=str(state["cache"]["cache_dir"]),
        force_redetect=bool(state["cache"]["force_redetect"]),
    )
    prefilter_cfg = PrefilterConfig(enabled=bool(state["prefilter"]["enabled"]))
    auto_roi_cfg = state.get("auto_roi_cfg") or {}

    _G_STEP2_DET = CachedAprilTagDetector(
        aruco_dict=aruco_dict,
        detector_params=detector_params,
        algo_key=algo_key,
        use_multiscale=bool(state["use_multiscale"]),
        opencv_refine=bool(state["opencv_refine"]),
        board=board,
        roi=tuple(state["roi"]) if state.get("roi") is not None else None,
        auto_roi=bool(auto_roi_cfg.get("enabled", False)),
        auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
        auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
        auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
        cache_cfg=cache_cfg,
        prefilter_cfg=prefilter_cfg,
    )
    _G_STEP2_MIN_TAGS = int(state["min_tags"])


def _step2_worker(image_path: str) -> Step2ImageResult:
    """并行 worker：对单张图片做检测，返回最小摘要。"""

    global _G_STEP2_DET, _G_STEP2_MIN_TAGS
    if _G_STEP2_DET is None:
        return Step2ImageResult(
            image_path=str(image_path),
            valid=False,
            n_tags=0,
            status=1,
            from_cache=False,
            elapsed_ms=0.0,
        )

    res = _G_STEP2_DET.detect_path(Path(str(image_path)))
    ids = np.asarray(res.ids) if res.ids is not None else np.zeros((0, 1), dtype=np.int32)
    n_tags = int(ids.shape[0])
    valid = bool(int(res.status) == 0 and n_tags >= int(_G_STEP2_MIN_TAGS))
    return Step2ImageResult(
        image_path=str(image_path),
        valid=valid,
        n_tags=n_tags,
        status=int(res.status),
        from_cache=bool(res.from_cache),
        elapsed_ms=float(res.elapsed_ms),
    )


def _save_detection_visualization(
    *,
    img_path: str,
    corners: Any,
    ids: Any,
    expected_tags: int,
    output_dir: Path,
) -> Optional[str]:
    """保存检测可视化结果（用于人工抽查）。"""

    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    vis = img.copy()
    num_tags = 0 if ids is None else len(ids)

    if ids is not None and len(ids) > 0 and corners is not None:
        for corner, tag_id in zip(corners, ids):
            tag_id = int(tag_id[0])
            c = np.asarray(corner)[0]

            for j in range(4):
                pt1 = tuple(c[j].astype(int))
                pt2 = tuple(c[(j + 1) % 4].astype(int))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

            center = np.mean(c, axis=0).astype(int)
            cv2.circle(vis, tuple(center), 5, (0, 255, 255), -1)
            cv2.putText(
                vis,
                str(tag_id),
                (int(center[0]) - 10, int(center[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    detection_rate = (num_tags / expected_tags * 100.0) if expected_tags > 0 else 0.0
    stats_text = [
        f"Multiscale Detection: {num_tags}/{expected_tags} tags",
        f"Detection Rate: {detection_rate:.1f}%",
    ]
    stats_color = (0, 255, 0) if num_tags == expected_tags else (0, 255, 255)

    for i, text in enumerate(stats_text):
        cv2.putText(
            vis,
            text,
            (10, 30 + i * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            stats_color,
            2,
        )

    base_name = Path(str(img_path)).stem
    output_path = output_dir / f"{base_name}_03_multiscale_detection.jpg"
    cv2.imwrite(str(output_path), vis)
    return str(output_path)


def _make_detector_for_camera(
    *,
    config: Dict[str, Any],
    profile: str,
    roi: Optional[Tuple[int, int, int, int]],
    cache_cfg: CacheConfig,
    prefilter_cfg: PrefilterConfig,
) -> CachedAprilTagDetector:
    """在主进程创建 detector（用于可视化/二次取 corners/ids）。"""

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
        algo_key=_build_algo_key(config, profile=profile),
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step2：筛选 raw 图像，输出 filtered 图像与筛选报告（默认安静，--verbose 可看逐对详情）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/apriltag_config.json",
        help="配置文件路径（默认 config/apriltag_config.json）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出更多逐帧过程信息（会刷屏）。",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=50,
        help="非 verbose 模式下，每 N 张打印一次进度（0=不打印中间进度；默认 50）",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="不保存检测可视化图（更快、更省空间；仍会保存 filtered 图和 report）",
    )

    # 性能/早停/并行/缓存/预筛选
    parser.add_argument(
        "--max_valid_images",
        type=int,
        default=0,
        help="每个相机最多保留多少张有效图像（0=使用 config.calibration_settings.max_images）。",
    )
    parser.add_argument(
        "--max_total_images",
        type=int,
        default=0,
        help="每个相机最多尝试检测多少张候选图像（0=不限制）。",
    )
    parser.add_argument(
        "--max_detect_seconds",
        type=float,
        default=0.0,
        help="每个相机检测总耗时上限（秒，0=不限制）。",
    )
    parser.add_argument(
        "--scan_strategy",
        type=str,
        default="sequential",
        choices=["sequential", "random", "uniform"],
        help="候选图像扫描策略：sequential/random/uniform（默认 sequential）。",
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
        help="启用廉价预筛选（减少 detector 调用）。",
    )
    parser.add_argument(
        "--min_tags",
        type=int,
        default=0,
        help="覆盖 config.calibration_settings.min_tags_for_pose（0=不覆盖）。",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    global VERBOSE
    VERBOSE = bool(args.verbose)

    print("=" * 60)
    print("Step 2: AprilTag 图像质量检查和筛选")
    print("=" * 60)

    config = load_config(str(args.config))
    board_cfg = config["apriltag_board"]
    calib_cfg = config["calibration_settings"]

    use_multiscale, opencv_refine = get_detection_settings(config)
    profile = get_detection_profile(config)
    auto_roi_cfg = get_detection_auto_roi(config)

    print("\n标定板配置:")
    print(f"  - AprilTag Family: {board_cfg['family']}")
    print(f"  - 标签排列: {board_cfg['tags_x']} x {board_cfg['tags_y']}")
    print(f"  - 标签尺寸: {board_cfg['tag_size']} mm")
    print(f"  - 标签间距: {board_cfg['tag_spacing']} mm")

    min_tags_cfg = int(calib_cfg["min_tags_for_pose"])
    min_tags = int(args.min_tags) if int(args.min_tags) > 0 else int(min_tags_cfg)

    print(f"  - 最少检测标签数: {min_tags}")
    print(f"  - use_multiscale: {use_multiscale}")
    print(f"  - opencv_refine: {opencv_refine}")
    print(f"  - detection profile: {profile}")

    cams_preview = get_dataset_cameras(config, allow_scan=True)
    print(f"  - cameras: {cams_preview}")

    if bool(auto_roi_cfg.get("enabled", False)):
        print(
            "  - auto_roi: enabled "
            f"(pre_scale={auto_roi_cfg.get('pre_scale')}, min_tags={auto_roi_cfg.get('min_tags')}, margin={auto_roi_cfg.get('margin')})"
        )

    os.makedirs("results", exist_ok=True)

    max_valid_images = (
        int(args.max_valid_images)
        if int(args.max_valid_images) > 0
        else int(calib_cfg.get("max_images", 100))
    )

    limits = ScanLimits(
        target_valid=int(max_valid_images),
        max_total=int(args.max_total_images),
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

    cameras = get_dataset_cameras(config, allow_scan=True)
    if len(cameras) == 0:
        print("\n错误: 未找到任何相机。")
        print("请在 config.image_dataset.cameras 中配置相机名与 raw_glob，或把原始图片放到 images/raw/<cam>/ 下。")
        return 1

    print(f"\n相机列表: {cameras}")

    cam_to_filtered_dir: Dict[str, Path] = {}
    for cam in cameras:
        out_dir = get_camera_filtered_dir(config, cam)
        cam_to_filtered_dir[cam] = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # 清空旧文件（只清文件，保留目录结构）
        for f in out_dir.glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass

    detection_dir: Optional[Path] = None
    if not bool(args.no_vis):
        detection_dir = Path("results/visualization/step2_filtering")
        if detection_dir.exists():
            for file in detection_dir.glob("**/*"):
                if file.is_file():
                    try:
                        file.unlink()
                    except Exception:
                        pass
        for cam in cameras:
            (detection_dir / cam).mkdir(parents=True, exist_ok=True)

    expected_tags = int(board_cfg["tags_x"]) * int(board_cfg["tags_y"])

    filter_report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "mode": "multi_camera",
        "config_path": str(args.config),
        "cameras": cameras,
        "min_tags_for_pose": int(min_tags),
        "per_camera": {},
        "frame_stats": {},
    }

    frame_key_to_valid_cams: Dict[str, List[str]] = {}

    obj_points_mm, tag_ids = create_apriltag_board(config)

    print("\n开始筛选图像（多相机）...")
    print("-" * 60)

    for cam in cameras:
        raw_images = get_camera_raw_images(config, cam)
        print(f"\n{cam}: 原始图像 {len(raw_images)} 张")
        if len(raw_images) == 0:
            filter_report["per_camera"][cam] = {
                "raw": 0,
                "valid": 0,
                "invalid": 0,
                "filtered_dir": str(cam_to_filtered_dir[cam].as_posix()),
                "note": "no images",
            }
            continue

        roi = get_detection_roi(config, camera=cam)
        if roi is not None:
            print(f"  - ROI: {roi}")

        init_state = {
            "config": config,
            "profile": profile,
            "family": str(board_cfg["family"]),
            "use_multiscale": bool(use_multiscale),
            "opencv_refine": bool(opencv_refine),
            "obj_points_mm": np.asarray(obj_points_mm, dtype=np.float32),
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

        items = [str(p) for p in raw_images]
        if int(workers) > 1:
            out, counters = iter_scan_parallel_ordered(
                items,
                worker_fn=_step2_worker,
                is_valid_fn=lambda r: bool(r.valid),
                limits=limits,
                order=order,
                max_workers=int(workers),
                prefetch=int(args.prefetch),
                initializer=_init_step2_worker,
                initargs=(init_state,),
            )
        else:
            _init_step2_worker(init_state)
            out, counters = iter_scan_sequential(
                items,
                worker_fn=_step2_worker,
                is_valid_fn=lambda r: bool(r.valid),
                limits=limits,
                order=order,
            )

        results = list(out)
        summary = summarize_step2_results(results)

        print(
            f"  扫描: submitted={counters.submitted} completed={counters.completed} valid={counters.valid} "
            f"(elapsed={counters.elapsed_s:.2f}s, strategy={order.strategy}, workers={workers})"
        )
        print(
            "  统计: "
            f"cache_hit={summary['cache_hit']} prefilter_skipped={summary['prefilter_skipped']} error={summary['error']} "
            f"mean_detect_ms={summary['mean_detect_ms']:.1f} p95_detect_ms={summary['p95_detect_ms']:.1f}"
        )

        viz_det = _make_detector_for_camera(
            config=config,
            profile=profile,
            roi=roi,
            cache_cfg=cache_cfg,
            prefilter_cfg=prefilter_cfg,
        )

        valid_count = 0
        invalid_count = 0

        for idx, r in enumerate(results, 1):
            img_path = Path(r.image_path)
            if bool(r.valid):
                valid_count += 1
                out_path = cam_to_filtered_dir[cam] / img_path.name
                try:
                    shutil.copy2(str(img_path), str(out_path))
                except Exception:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        cv2.imwrite(str(out_path), img)

                key = img_path.stem
                lst = frame_key_to_valid_cams.get(key, [])
                if cam not in lst:
                    lst.append(cam)
                    frame_key_to_valid_cams[key] = lst

                if detection_dir is not None:
                    det_res = viz_det.detect_path(img_path)
                    _save_detection_visualization(
                        img_path=str(img_path),
                        corners=det_res.corners,
                        ids=det_res.ids,
                        expected_tags=int(expected_tags),
                        output_dir=(detection_dir / cam),
                    )
            else:
                invalid_count += 1

            if (not VERBOSE) and int(args.print_every) > 0 and (idx % int(args.print_every) == 0):
                print(f"  进度: {idx}/{len(results)} | 合格: {valid_count} | 不合格: {invalid_count}")

        filter_report["per_camera"][cam] = {
            "raw": int(len(raw_images)),
            "valid": int(valid_count),
            "invalid": int(invalid_count),
            "filtered_dir": str(cam_to_filtered_dir[cam].as_posix()),
            "scan": {
                "limits": {
                    "target_valid": int(limits.target_valid),
                    "max_total": int(limits.max_total),
                    "max_seconds": float(limits.max_seconds),
                },
                "order": {"strategy": str(order.strategy), "seed": int(order.seed)},
                "workers": int(workers),
                "submitted": int(counters.submitted),
                "completed": int(counters.completed),
                "elapsed_s": float(counters.elapsed_s),
            },
            "perf": summary,
        }
        print(f"  ✓ {cam}: 合格 {valid_count} / {len(raw_images)}")

    filter_report["frame_stats"] = compute_frame_stats(
        frame_key_to_valid_cams,
        n_cameras=int(len(cameras)),
    )

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/filter_report.json").write_text(
        json.dumps(filter_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n✓ 已保存筛选报告: results/filter_report.json")
    print("\n" + "=" * 60)
    print("图像筛选完成！（多相机）")
    print("=" * 60)

    fs = filter_report["frame_stats"]
    print("\n帧级统计（按文件 stem）：")
    print(f"  - 任意相机合格的帧键: {int(fs.get('unique_frame_keys_with_any_valid', 0))}")
    print(f"  - 至少2路相机同帧合格(可形成边): {int(fs.get('frame_keys_with_at_least_2_cameras_valid', 0))}")
    print(f"  - 所有相机同帧合格: {int(fs.get('frame_keys_with_all_cameras_valid', 0))}")

    print("\n下一步: 运行 python -m mcca.entry.step3_intrinsic_apriltag（会按 config 自动处理多相机）")
    return 0


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
