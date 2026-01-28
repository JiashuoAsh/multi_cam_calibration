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
    - images/raw/left/*.png: 原始左相机图像
    - images/raw/right/*.png: 原始右相机图像
    - config/apriltag_config.json: 配置文件（min_tags_for_pose）

输出:
    - images/filtered/left/*.png: 筛选后的左相机图像
    - images/filtered/right/*.png: 筛选后的右相机图像
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
import cv2
import numpy as np
import json
import os
import glob
import shutil
from datetime import datetime
from typing import cast, Any, Dict, Optional, List
from utils import (
    load_config,
    get_aruco_dict,
    detect_apriltag_corners,
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
from pathlib import Path


# 默认尽量安静：只输出关键进度和汇总；需要逐张输出用 --verbose。
VERBOSE: bool = True


def _vprint(*args, **kwargs) -> None:
    """Verbose print (guarded by VERBOSE)."""
    if VERBOSE:
        print(*args, **kwargs)


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


def check_image_quality(
    img_path,
    aruco_dict,
    detector_params,
    min_tags,
    *,
    use_multiscale: bool,
    opencv_refine: bool,
    board,
    roi=None,
    auto_roi_cfg: Optional[Dict[str, Any]] = None,
):
    """
    检查单张图像的质量

    检查标准：
        - 成功读取图像
        - 检测到的 AprilTag 标签数量 >= min_tags

    Args:
        img_path: 图像文件路径
        aruco_dict: ArUco 字典对象
        detector_params: 检测器参数
        min_tags: 最少需要检测到的标签数量
        use_multiscale: 是否使用多尺度检测（推荐）

    Returns:
        tuple: (is_valid, num_tags, corners, ids)
            is_valid: bool, 图像是否合格
            num_tags: int, 检测到的标签数量
            corners: list, 检测到的角点列表
            ids: np.ndarray, 检测到的标签ID数组
    """
    img = cv2.imread(img_path)
    if img is None:
        return False, 0, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测 AprilTag（使用多尺度检测提高准确率）
    auto_roi_cfg = auto_roi_cfg or {}
    corners, ids = detect_apriltag_corners(
        gray,
        aruco_dict,
        detector_params,
        use_multiscale=use_multiscale,
        opencv_refine=opencv_refine,
        board=board,
        roi=roi,
        auto_roi=bool(auto_roi_cfg.get("enabled", False)),
        auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
        auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
        auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
    )

    num_tags = 0 if ids is None else len(ids)
    is_valid = num_tags >= min_tags

    return is_valid, num_tags, corners, ids


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
        "--print_every",
        type=int,
        default=50,
        help="非 verbose 模式下，每 N 对打印一次进度（0=不打印中间进度；默认 50）",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="不保存检测可视化图（更快、更省空间；仍会保存 filtered 图和 report）",
    )
    args = parser.parse_args()

    # global VERBOSE
    # VERBOSE = bool(args.verbose)

    print("=" * 60)
    print("Step 2: AprilTag 图像质量检查和筛选")
    print("=" * 60)

    # 加载配置
    config = load_config(str(args.config))
    board_cfg = config["apriltag_board"]
    calib_cfg = config["calibration_settings"]

    ds = get_image_dataset(config)
    use_dataset = bool(ds.get("enabled", False))

    use_multiscale, opencv_refine = get_detection_settings(config)
    profile = get_detection_profile(config)
    auto_roi_cfg = get_detection_auto_roi(config)

    print(f"\n标定板配置:")
    print(f"  - AprilTag Family: {board_cfg['family']}")
    print(f"  - 标签排列: {board_cfg['tags_x']} x {board_cfg['tags_y']}")
    print(f"  - 标签尺寸: {board_cfg['tag_size']} mm")
    print(f"  - 标签间距: {board_cfg['tag_spacing']} mm")
    print(f"  - 最少检测标签数: {calib_cfg['min_tags_for_pose']}")
    print(f"  - use_multiscale: {use_multiscale}")
    print(f"  - opencv_refine: {opencv_refine}")
    print(f"  - detection profile: {profile}")
    if use_dataset:
        cams_preview = get_dataset_cameras(config, allow_scan=True, fallback_stereo=True)
        print(f"  - image_dataset: enabled (cameras={cams_preview})")
    if bool(auto_roi_cfg.get("enabled", False)):
        print(
            "  - auto_roi: enabled "
            f"(pre_scale={auto_roi_cfg.get('pre_scale')}, min_tags={auto_roi_cfg.get('min_tags')}, margin={auto_roi_cfg.get('margin')})"
        )

    os.makedirs("results", exist_ok=True)

    # === 新流程：按 config.image_dataset 自动处理多相机 ===
    if use_dataset:
        cameras = get_dataset_cameras(config, allow_scan=True, fallback_stereo=False)
        if len(cameras) == 0:
            print("\n错误: image_dataset.enabled=true，但未找到 cameras。")
            print("请在 config.image_dataset.cameras 中配置相机名与 raw_dir/raw_glob，或把原始图片放到 raw_root/<cam>/ 下。")
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

        # 获取 ArUco 字典 / Board / 检测器参数
        aruco_dict = get_aruco_dict(board_cfg["family"])
        obj_points_mm, tag_ids = create_apriltag_board(config)
        board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)
        detector_params = create_detector_params(config)

        min_tags = int(calib_cfg["min_tags_for_pose"])
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

            valid_count = 0
            invalid_count = 0

            for idx, img_path in enumerate(raw_images, 1):
                ok, n_tags, corners, ids = check_image_quality(
                    str(img_path),
                    aruco_dict,
                    detector_params,
                    min_tags,
                    use_multiscale=use_multiscale,
                    opencv_refine=opencv_refine,
                    board=board,
                    roi=roi,
                    auto_roi_cfg=auto_roi_cfg,
                )

                if ok:
                    valid_count += 1
                    out_path = cam_to_filtered_dir[cam] / img_path.name
                    try:
                        shutil.copy2(str(img_path), str(out_path))
                    except Exception:
                        # 复制失败时回退到 imread+imwrite
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            cv2.imwrite(str(out_path), img)

                    # 统计 frame_key 共视
                    key = img_path.stem
                    lst = frame_key_to_valid_cams.get(key, [])
                    if cam not in lst:
                        lst.append(cam)
                        frame_key_to_valid_cams[key] = lst

                    if detection_dir is not None:
                        save_detection_visualization(
                            str(img_path),
                            corners,
                            ids,
                            expected_tags,
                            str(detection_dir / cam),
                        )
                else:
                    invalid_count += 1

                if (not VERBOSE) and args.print_every and (idx % int(args.print_every) == 0):
                    print(f"  进度: {idx}/{len(raw_images)} | 合格: {valid_count} | 不合格: {invalid_count}")

            filter_report["per_camera"][cam] = {
                "raw": int(len(raw_images)),
                "valid": int(valid_count),
                "invalid": int(invalid_count),
                "filtered_dir": str(cam_to_filtered_dir[cam].as_posix()),
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

    # === 旧流程：固定双目 left/right ===

    # 检查原始图像目录
    if not os.path.exists("images/raw/left") or not os.path.exists("images/raw/right"):
        print("\n错误: 未找到原始图像目录 images/raw/")
        print("请先准备 images/raw/left 与 images/raw/right 下的 png 图像对")
        print("（离线视频推荐：python step1_extract_imgs_from_video.py --video_left ... --video_right ...）")
        return

    left_roi = get_detection_roi(config, camera="left")
    right_roi = get_detection_roi(config, camera="right")

    # 创建输出目录并清空旧文件
    filtered_left_dir = Path("images/filtered/left")
    filtered_right_dir = Path("images/filtered/right")

    # 清空筛选后的图像目录
    if filtered_left_dir.exists():
        for file in filtered_left_dir.glob("*"):
            if file.is_file():
                file.unlink()
    filtered_left_dir.mkdir(parents=True, exist_ok=True)

    if filtered_right_dir.exists():
        for file in filtered_right_dir.glob("*"):
            if file.is_file():
                file.unlink()
    filtered_right_dir.mkdir(parents=True, exist_ok=True)

    detection_dir = None
    if not args.no_vis:
        # 清空并创建检测可视化目录（分左右相机）
        detection_dir = Path("results/visualization/step2_filtering")
        if detection_dir.exists():
            # 删除目录中的所有文件
            for file in detection_dir.glob("**/*"):
                if file.is_file():
                    file.unlink()
        # 创建 left 和 right 子目录
        (detection_dir / "left").mkdir(parents=True, exist_ok=True)
        (detection_dir / "right").mkdir(parents=True, exist_ok=True)

    # 获取 ArUco 字典
    aruco_dict = get_aruco_dict(board_cfg["family"])

    # 构建 OpenCV Board（用于 refineDetectedMarkers）
    obj_points_mm, tag_ids = create_apriltag_board(config)
    board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)

    # 设置检测器参数（支持按 profile 针对小Tag调参）
    detector_params = create_detector_params(config)

    min_tags = calib_cfg["min_tags_for_pose"]

    # 获取原始图像文件
    left_images = sorted(glob.glob("images/raw/left/*.png"))
    right_images = sorted(glob.glob("images/raw/right/*.png"))

    print(f"\n找到原始图像:")
    print(f"  - 左相机: {len(left_images)} 张")
    print(f"  - 右相机: {len(right_images)} 张")

    if len(left_images) == 0 or len(right_images) == 0:
        print("\n错误: 没有找到原始图像！")
        return

    if len(left_images) != len(right_images):
        print(f"\n警告: 左右图像数量不匹配！")

    # 筛选图像
    print("\n开始筛选图像...")
    print("-" * 60)

    valid_pairs = []
    invalid_pairs = []
    filter_report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_pairs": min(len(left_images), len(right_images)),
        "valid_pairs": 0,
        "invalid_pairs": 0,
        "details": [],
    }

    # 遍历图像对
    processed = 0
    total_pairs = min(len(left_images), len(right_images))
    for left_path, right_path in zip(left_images, right_images):
        processed += 1
        left_filename = os.path.basename(left_path)
        right_filename = os.path.basename(right_path)

        # 检查左图像（使用多尺度检测）
        left_valid, left_tags, left_corners, left_ids = check_image_quality(
            left_path,
            aruco_dict,
            detector_params,
            min_tags,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            roi=left_roi,
            auto_roi_cfg=auto_roi_cfg,
        )

        # 检查右图像（使用多尺度检测）
        right_valid, right_tags, right_corners, right_ids = check_image_quality(
            right_path,
            aruco_dict,
            detector_params,
            min_tags,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            roi=right_roi,
            auto_roi_cfg=auto_roi_cfg,
        )

        # 判断是否合格
        # step2 主要是以为了标定内参，所以要保证图片计量充满整个相机视角，不追求极限合格率
        # 因此只要左右任意一张图像合格即可
        is_pair_valid = left_valid or right_valid

        # 记录详情
        detail = {
            "left_file": left_filename,
            "right_file": right_filename,
            "left_tags": left_tags,
            "right_tags": right_tags,
            "is_valid": is_pair_valid,
        }

        if is_pair_valid:
            valid_pairs.append(
                (
                    left_path,
                    right_path,
                    left_corners,
                    left_ids,
                    right_corners,
                    right_ids,
                )
            )
            filter_report["valid_pairs"] = cast(int, filter_report["valid_pairs"]) + 1
            _vprint(f"✓ {left_filename}: 左={left_tags} tags, 右={right_tags} tags")
        else:
            invalid_pairs.append((left_path, right_path))
            filter_report["invalid_pairs"] = cast(int, filter_report["invalid_pairs"]) + 1
            status = []
            if not left_valid:
                status.append(f"左={left_tags}/{min_tags}")
            if not right_valid:
                status.append(f"右={right_tags}/{min_tags}")
            _vprint(f"✗ {left_filename}: {', '.join(status)}")

        if (not VERBOSE) and args.print_every and (processed % int(args.print_every) == 0):
            v = cast(int, filter_report["valid_pairs"])
            inv = cast(int, filter_report["invalid_pairs"])
            print(f"进度: {processed}/{total_pairs} | 合格: {v} | 不合格: {inv}")

        details_list = filter_report["details"]
        assert isinstance(details_list, list)
        details_list.append(detail)

    # 保存合格图像
    print("\n保存合格图像...")

    # 获取期望的标签数量
    expected_tags = board_cfg["tags_x"] * board_cfg["tags_y"]

    for i, (
        left_path,
        right_path,
        left_corners,
        left_ids,
        right_corners,
        right_ids,
    ) in enumerate(valid_pairs, 1):
        # 读取图像
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        if left_img is None or right_img is None:
            # 理论上不应该发生（前面已成功读取/检测），但这里做个保险。
            _vprint(f"警告: 无法读取有效图像对，跳过保存: {left_path}, {right_path}")
            continue

        # 使用原始文件名
        left_filename = os.path.basename(left_path)
        right_filename = os.path.basename(right_path)

        # 保存到筛选目录
        left_output = f"images/filtered/left/{left_filename}"
        right_output = f"images/filtered/right/{right_filename}"

        cv2.imwrite(left_output, left_img)
        cv2.imwrite(right_output, right_img)

        if detection_dir is not None:
            # 保存检测可视化图像（用于人工检查，分左右相机）
            save_detection_visualization(
                left_path,
                left_corners,
                left_ids,
                expected_tags,
                str(detection_dir / "left"),
            )
            save_detection_visualization(
                right_path,
                right_corners,
                right_ids,
                expected_tags,
                str(detection_dir / "right"),
            )

    print(f"  ✓ 已保存 {len(valid_pairs)} 组合格图像")
    if detection_dir is not None:
        print(
            f"  ✓ 已保存 {len(valid_pairs) * 2} 张检测可视化图像到 {str(detection_dir)}/"
        )
    else:
        print("  - 已跳过检测可视化图（--no_vis）")

    # 保存筛选报告
    with open("results/filter_report.json", "w") as f:
        json.dump(filter_report, f, indent=2)

    print("\n  ✓ 已保存筛选报告: results/filter_report.json")

    # 显示统计
    print("\n" + "=" * 60)
    print("图像筛选完成！")
    print("=" * 60)
    print(f"\n统计:")
    print(f"  - 总图像对: {filter_report['total_pairs']}")
    valid_count = cast(int, filter_report['valid_pairs'])
    invalid_count = cast(int, filter_report['invalid_pairs'])
    total_count = cast(int, filter_report['total_pairs'])
    print(
        f"  - 合格: {valid_count} ({valid_count / total_count * 100:.1f}%)"
    )
    print(
        f"  - 不合格: {invalid_count} ({invalid_count / total_count * 100:.1f}%)"
    )

    if valid_count < 10:
        print(f"\n⚠ 警告: 合格图像数量较少 ({valid_count} < 10)")
        print("  建议:")
        print("  - 重新拍摄更多图像")
        print("  - 确保标定板清晰可见")
        print("  - 改善光照条件")
    else:
        print(f"\n✓ 合格图像数量充足，可以进行标定")

    print(f"\n筛选后的图像位置:")
    print(f"  - images/filtered/left/")
    print(f"  - images/filtered/right/")

    print("\n下一步: 运行 python step3_intrinsic_apriltag.py")


if __name__ == "__main__":
    main()
