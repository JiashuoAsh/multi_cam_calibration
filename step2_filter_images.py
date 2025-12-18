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

import cv2
import numpy as np
import json
import os
import glob
from datetime import datetime
from typing import cast, Any, Dict
from utils import (
    load_config,
    get_aruco_dict,
    detect_apriltag_corners,
    create_apriltag_board,
    create_opencv_aruco_board,
    get_detection_settings,
)
from pathlib import Path


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
    corners, ids = detect_apriltag_corners(
        gray,
        aruco_dict,
        detector_params,
        use_multiscale=use_multiscale,
        opencv_refine=opencv_refine,
        board=board,
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

    # 绘制检测到的标签
    if ids is not None and len(ids) > 0:
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
    print("=" * 60)
    print("Step 2: AprilTag 图像质量检查和筛选")
    print("=" * 60)

    # 检查原始图像目录
    if not os.path.exists("images/raw/left") or not os.path.exists("images/raw/right"):
        print("\n错误: 未找到原始图像目录 images/raw/")
        print("请先运行 python step1_capture_imgs.py")
        return

    # 加载配置
    config = load_config()
    board_cfg = config["apriltag_board"]
    calib_cfg = config["calibration_settings"]

    use_multiscale, opencv_refine = get_detection_settings(config)

    print(f"\n标定板配置:")
    print(f"  - AprilTag Family: {board_cfg['family']}")
    print(f"  - 标签排列: {board_cfg['tags_x']} x {board_cfg['tags_y']}")
    print(f"  - 标签尺寸: {board_cfg['tag_size']} mm")
    print(f"  - 标签间距: {board_cfg['tag_spacing']} mm")
    print(f"  - 最少检测标签数: {calib_cfg['min_tags_for_pose']}")
    print(f"  - use_multiscale: {use_multiscale}")
    print(f"  - opencv_refine: {opencv_refine}")

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

    os.makedirs("results", exist_ok=True)

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

    # 设置检测器参数
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    # detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

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
    for left_path, right_path in zip(left_images, right_images):
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
        )

        # 判断是否合格（左右都要合格）
        is_pair_valid = left_valid and right_valid

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
            print(f"✓ {left_filename}: 左={left_tags} tags, 右={right_tags} tags")
        else:
            invalid_pairs.append((left_path, right_path))
            filter_report["invalid_pairs"] = cast(int, filter_report["invalid_pairs"]) + 1
            status = []
            if not left_valid:
                status.append(f"左={left_tags}/{min_tags}")
            if not right_valid:
                status.append(f"右={right_tags}/{min_tags}")
            print(f"✗ {left_filename}: {', '.join(status)}")

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

        # 使用原始文件名
        left_filename = os.path.basename(left_path)
        right_filename = os.path.basename(right_path)

        # 保存到筛选目录
        left_output = f"images/filtered/left/{left_filename}"
        right_output = f"images/filtered/right/{right_filename}"

        cv2.imwrite(left_output, left_img)
        cv2.imwrite(right_output, right_img)

        # 保存检测可视化图像（用于人工检查，分左右相机）
        save_detection_visualization(
            left_path,
            left_corners,
            left_ids,
            expected_tags,
            "results/visualization/step2_filtering/left",
        )
        save_detection_visualization(
            right_path,
            right_corners,
            right_ids,
            expected_tags,
            "results/visualization/step2_filtering/right",
        )

    print(f"  ✓ 已保存 {len(valid_pairs)} 组合格图像")
    print(
        f"  ✓ 已保存 {len(valid_pairs) * 2} 张检测可视化图像到 results/visualization/step2_filtering/"
    )

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
