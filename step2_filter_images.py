#!/usr/bin/env python3
"""
Step 2: 图像质量检查和筛选 - AprilTag 标定板

功能:
    从原始采集的图像中筛选出包含标定板且质量合格的图像。
    这是新标定流程的关键步骤，实现了采集和筛选的分离。

工作流程:
    1. 遍历所有原始图像对
    2. 检测每张图像中的 AprilTag 标签
    3. 根据检测结果判断是否合格
    4. 复制合格图像到 filtered/ 目录
    5. 生成详细的筛选报告

检查标准:
    1. 检测到足够数量的 AprilTag 标签（由 min_tags_for_pose 配置）
    2. 左右图像都检测到标定板
    3. 图像清晰度满足要求（可选）

使用方法:
    python step2_filter_images.py

输入:
    - images/raw/<cam>/*.(png|jpg|jpeg|bmp): 原始相机图像（相机列表来自 config.image_dataset.cameras 或目录扫描）
    - config/apriltag_config.json: 配置文件（min_tags_for_pose 等）

输出:
    - images/filtered/<cam>/*.(png|jpg|jpeg|bmp): 筛选后的相机图像
    - results/filter_report.json: 详细筛选报告

重新筛选:
    如果筛选结果不满意，可以：
    1. 修改配置文件中的 min_tags_for_pose
    2. 重新运行本脚本（无需重新拍照）
    3. 或补充拍摄更多图像后重新筛选

下一步:
    运行 python step3_intrinsic_apriltag.py 进行内参标定
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

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
)

from libs.apriltag_perf.cache import CacheConfig
from libs.apriltag_perf.prefilter import PrefilterConfig
from libs.apriltag_perf.scan import (
    ScanLimits,
    ScanOrder,
    iter_scan_parallel_ordered,
    iter_scan_sequential,
)
from libs.apriltag_perf.service import CachedAprilTagDetector


# 默认尽量安静：只输出关键进度和汇总；需要逐张输出用 --verbose。
VERBOSE: bool = True


def _vprint(*args, **kwargs) -> None:
    """Verbose print (guarded by VERBOSE)."""
    if VERBOSE:
        print(*args, **kwargs)


@dataclass(frozen=True)
class _Step2Result:
    """Step2 单张图片的筛选结果。

    说明：
    - 为了并行效率，worker 只返回“很小的摘要信息”。
    - 若需要可视化/后续使用 corners/ids，会在主进程用同一 cache 再读一次（大概率命中缓存）。
    """

    image_path: str
    valid: bool
    n_tags: int
    status: int
    from_cache: bool
    elapsed_ms: float


def _percentile(values: List[float], q: float) -> float:
    if len(values) == 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _summarize_step2_results(results: List[_Step2Result]) -> Dict[str, Any]:
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


def _step2_worker(image_path: str) -> _Step2Result:
    global _G_STEP2_DET, _G_STEP2_MIN_TAGS
    if _G_STEP2_DET is None:
        return _Step2Result(
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
    return _Step2Result(
        image_path=str(image_path),
        valid=valid,
        n_tags=n_tags,
        status=int(res.status),
        from_cache=bool(res.from_cache),
        elapsed_ms=float(res.elapsed_ms),
    )


def save_detection_visualization(
    img_path,
    corners,
    ids,
    expected_tags,
    output_dir="results/visualization/step2_filtering",
):
    """
    保存多尺度检测的可视化图像（用于人工检查）

    生成类似 detect_apriltag_advanced.py 的可视化效果：
        - 绘制所有检测到的标签边框（绿色）
        - 显示标签ID（黄色数字）
        - 显示检测统计（标签数/期望数、检测率）

    Args:
        img_path: 图像文件路径
        corners: 检测到的角点列表
        ids: 检测到的标签ID数组
        expected_tags: 期望检测的标签数量
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        return None

    vis = img.copy()
    num_tags = 0 if ids is None else len(ids)

    # 绘制检测到的标签
    if ids is not None and len(ids) > 0:
        for i, (corner, tag_id) in enumerate(zip(corners, ids)):
            tag_id = int(tag_id[0])
            corner = corner[0]

            # 绘制边框（绿色）
            for j in range(4):
                pt1 = tuple(corner[j].astype(int))
                pt2 = tuple(corner[(j + 1) % 4].astype(int))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

            # 计算中心点
            center = np.mean(corner, axis=0).astype(int)

            # 绘制中心圆点（黄色）
            cv2.circle(vis, tuple(center), 5, (0, 255, 255), -1)

            # 绘制标签ID（黄色文字）
            cv2.putText(
                vis,
                str(tag_id),
                (center[0] - 10, center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    # 添加统计信息
    detection_rate = (num_tags / expected_tags * 100) if expected_tags > 0 else 0
    stats_text = [
        f"Multiscale Detection: {num_tags}/{expected_tags} tags",
        f"Detection Rate: {detection_rate:.1f}%",
    ]

    # 选择颜色：100% 绿色，否则黄色
    stats_color = (0, 255, 0) if num_tags == expected_tags else (0, 255, 255)

    for i, text in enumerate(stats_text):
        cv2.putText(
            vis, text, (10, 30 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, stats_color, 2
        )

    # 保存图像
    base_name = Path(img_path).stem
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
    """在主进程创建一个 detector（用于可视化/二次取 corners/ids）。"""

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


def visualize_detection(img_path, corners, ids, is_valid):
    """
    可视化AprilTag检测结果

    在图像上绘制：
        - 检测到的标签边框和ID
        - 状态指示（VALID/INVALID）
        - 检测到的标签数量

    Args:
        img_path: 图像文件路径
        corners: 检测到的角点列表
        ids: 检测到的标签ID数组
        is_valid: bool, 图像是否合格

    Returns:
        display_img: np.ndarray, 带标注的可视化图像
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # 绘制检测到的标签
    if ids is not None and corners is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # 添加状态标签
    status_text = "VALID" if is_valid else "INVALID"
    status_color = (0, 255, 0) if is_valid else (0, 0, 255)

    cv2.putText(
        img, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3
    )

    num_tags = 0 if ids is None else len(ids)
    cv2.putText(
        img,
        f"Tags: {num_tags}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return img


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Step2：筛选 raw 图像对，输出 filtered 图像与筛选报告（默认安静，--verbose 可看逐对详情）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/apriltag_config.json",
        help="配置文件路径（默认 config/apriltag_config.json）",
    )
    # parser.add_argument("--verbose", action="store_true", help="输出每对图像的筛选结果（会很刷屏）")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出更多逐帧过程信息（会刷屏）。",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=50,
        help="非 verbose 模式下，每 N 张/对打印一次进度（0=不打印中间进度；默认 50）",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="不保存检测可视化图（更快、更省空间；仍会保存 filtered 图和 report）",
    )

    # 性能优先：流式扫描/早停/并行/缓存/预筛选
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
        help="启用廉价预筛选（可能过滤掉明显无效/过暗/过曝/模糊帧，减少 detector 调用）。",
    )
    parser.add_argument(
        "--min_tags",
        type=int,
        default=0,
        help="覆盖 config.calibration_settings.min_tags_for_pose（0=不覆盖）。",
    )
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = bool(args.verbose)

    print("=" * 60)
    print("Step 2: AprilTag 图像质量检查和筛选")
    print("=" * 60)

    # 加载配置
    config = load_config(str(args.config))
    board_cfg = config["apriltag_board"]
    calib_cfg = config["calibration_settings"]

    # 统一数据集入口：本仓库不再支持固定的 left/right 目录约定。
    # 相机列表来自 config.image_dataset.cameras；若未配置则尝试扫描 images/raw/<cam>/。
    use_dataset = True

    use_multiscale, opencv_refine = get_detection_settings(config)
    profile = get_detection_profile(config)
    auto_roi_cfg = get_detection_auto_roi(config)

    print(f"\n标定板配置:")
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

    max_valid_images = int(args.max_valid_images) if int(args.max_valid_images) > 0 else int(calib_cfg.get("max_images", 100))
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

    # === 统一流程：按 config.image_dataset 自动处理多相机 ===
    if use_dataset:
        cameras = get_dataset_cameras(config, allow_scan=True)
        if len(cameras) == 0:
            print("\n错误: 未找到任何相机。")
            print("请在 config.image_dataset.cameras 中配置相机名与 raw_glob，或把原始图片放到 images/raw/<cam>/ 下。")
            return

        print(f"\n相机列表: {cameras}")

        # 输出目录（每相机）
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

        detection_dir = None
        if not args.no_vis:
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

        # 逐相机筛选
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

            # 并行 worker 初始化状态（必须可 pickle）
            obj_points_mm, tag_ids = create_apriltag_board(config)
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
                # 单进程：直接初始化 detector 并用 scan 框架早停
                _init_step2_worker(init_state)
                out, counters = iter_scan_sequential(
                    items,
                    worker_fn=_step2_worker,
                    is_valid_fn=lambda r: bool(r.valid),
                    limits=limits,
                    order=order,
                )

            results = list(out)
            summary = _summarize_step2_results(results)
            print(
                f"  扫描: submitted={counters.submitted} completed={counters.completed} valid={counters.valid} "
                f"(elapsed={counters.elapsed_s:.2f}s, strategy={order.strategy}, workers={workers})"
            )
            print(
                "  统计: "
                f"cache_hit={summary['cache_hit']} prefilter_skipped={summary['prefilter_skipped']} error={summary['error']} "
                f"mean_detect_ms={summary['mean_detect_ms']:.1f} p95_detect_ms={summary['p95_detect_ms']:.1f}"
            )

            # 主进程 detector：用于可视化时从 cache 取 corners/ids
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
                        # 复制失败时回退到 imread+imwrite
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
                        save_detection_visualization(
                            str(img_path),
                            det_res.corners,
                            det_res.ids,
                            expected_tags,
                            str(detection_dir / cam),
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

        # 帧级统计（对 Step4 的“是否有边/是否连通”非常关键）
        n_frames_any = int(len(frame_key_to_valid_cams))
        n_frames_ge2 = int(sum(1 for _k, v in frame_key_to_valid_cams.items() if len(v) >= 2))
        n_frames_all = int(sum(1 for _k, v in frame_key_to_valid_cams.items() if len(v) == len(cameras)))
        filter_report["frame_stats"] = {
            "unique_frame_keys_with_any_valid": n_frames_any,
            "frame_keys_with_at_least_2_cameras_valid": n_frames_ge2,
            "frame_keys_with_all_cameras_valid": n_frames_all,
        }

        with open("results/filter_report.json", "w", encoding="utf-8") as f:
            json.dump(filter_report, f, indent=2, ensure_ascii=False)

        print("\n✓ 已保存筛选报告: results/filter_report.json")
        print("\n" + "=" * 60)
        print("图像筛选完成！（多相机）")
        print("=" * 60)
        print(f"\n帧级统计（按文件 stem）：")
        print(f"  - 任意相机合格的帧键: {n_frames_any}")
        print(f"  - 至少2路相机同帧合格(可形成边): {n_frames_ge2}")
        print(f"  - 所有相机同帧合格: {n_frames_all}")
        print("\n下一步: 运行 python step3_intrinsic_apriltag.py（会按 config 自动处理多相机）")
        return


if __name__ == "__main__":
    main()
