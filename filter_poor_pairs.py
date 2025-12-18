#!/usr/bin/env python3
"""
独立的图像质量分析和过滤工具

注意: Step 4 已集成自动质量过滤功能！
     本工具作为独立工具保留，用于：
     - 手动分析现有图像质量
     - 预览过滤效果
     - 手动清理低质量图像

用法:
    # 预览模式（不删除）
    python3 filter_poor_pairs.py

    # 执行过滤（会备份到 images/filtered_backup/）
    python3 filter_poor_pairs.py --execute

    # 自定义阈值
    python3 filter_poor_pairs.py --min-common 20 --execute
"""

import cv2
import numpy as np
import glob
import os
import shutil
import argparse
from utils import load_config, get_aruco_dict, detect_apriltag_corners


def analyze_and_filter(min_common_tags=15, dry_run=True):
    """
    分析所有图像对，过滤低质量的

    Args:
        min_common_tags: 最少共同标签数（推荐15-20）
        dry_run: True=只显示不删除，False=真正删除
    """
    print("="*70)
    print(f"智能过滤图像对（最少 {min_common_tags} 个共同标签）")
    print("="*70)

    config = load_config()
    aruco_dict = get_aruco_dict(config['apriltag_board']['family'])

    left_images = sorted(glob.glob('images/filtered/left/*.png'))
    right_images = sorted(glob.glob('images/filtered/right/*.png'))

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

    print(f"\n原始图像对数量: {len(left_images)}")
    print(f"\n分析中...")

    keep_pairs = []
    remove_pairs = []

    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        print(f"\r  处理: {i+1}/{len(left_images)}", end='', flush=True)

        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            remove_pairs.append((left_path, right_path, 0, "读取失败"))
            continue

        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # 使用多尺度检测（与标定流程一致）
        left_corners, left_ids = detect_apriltag_corners(
            left_gray, aruco_dict, detector_params, use_multiscale=True
        )
        right_corners, right_ids = detect_apriltag_corners(
            right_gray, aruco_dict, detector_params, use_multiscale=True
        )

        if left_ids is None or right_ids is None:
            remove_pairs.append((left_path, right_path, 0, "检测失败"))
            continue

        left_set = set(left_ids.flatten())
        right_set = set(right_ids.flatten())
        common = len(left_set & right_set)

        if common >= min_common_tags:
            keep_pairs.append((left_path, right_path, common))
        else:
            remove_pairs.append((left_path, right_path, common, "共同标签不足"))

    print()  # 换行

    # 显示统计
    print("\n" + "="*70)
    print("过滤结果")
    print("="*70)

    print(f"\n✅ 保留: {len(keep_pairs)} 对")
    print(f"❌ 移除: {len(remove_pairs)} 对")

    if keep_pairs:
        common_counts = [x[2] for x in keep_pairs]
        print(f"\n保留图像对的质量:")
        print(f"  平均共同标签: {np.mean(common_counts):.1f} 个")
        print(f"  最少: {np.min(common_counts)} 个")
        print(f"  最多: {np.max(common_counts)} 个")
        print(f"  标准差: {np.std(common_counts):.1f} 个")

    if remove_pairs:
        print(f"\n将移除的图像对:")
        for left_path, right_path, common, reason in remove_pairs[:10]:
            left_name = os.path.basename(left_path)
            right_name = os.path.basename(right_path)
            print(f"  - {left_name}: {common} 个共同标签 ({reason})")
        if len(remove_pairs) > 10:
            print(f"  ... 以及其他 {len(remove_pairs)-10} 对")

    # 执行删除
    if not dry_run:
        print("\n" + "="*70)
        print("执行删除...")
        print("="*70)

        # 创建备份目录
        backup_dir = 'images/filtered_backup'
        os.makedirs(f'{backup_dir}/left', exist_ok=True)
        os.makedirs(f'{backup_dir}/right', exist_ok=True)

        for left_path, right_path, _, _ in remove_pairs:
            # 备份
            left_name = os.path.basename(left_path)
            right_name = os.path.basename(right_path)
            shutil.move(left_path, f'{backup_dir}/left/{left_name}')
            shutil.move(right_path, f'{backup_dir}/right/{right_name}')

        print(f"\n✅ 已移除 {len(remove_pairs)} 对图像")
        print(f"   备份到: {backup_dir}/")
        print(f"\n保留 {len(keep_pairs)} 对高质量图像")
        print(f"\n下一步: 运行 step4_stereo_extrinsic.py 重新标定")
    else:
        print("\n" + "="*70)
        print("⚠️  当前为预览模式（dry_run=True），未实际删除")
        print("="*70)
        print("\n要真正执行过滤，请运行:")
        print("  python3 filter_poor_pairs.py --execute")

    return len(keep_pairs), len(remove_pairs)


def main():
    parser = argparse.ArgumentParser(
        description='过滤低质量双目图像对',
        epilog='注意: Step 4 已内置自动质量过滤！本工具用于手动分析。'
    )
    parser.add_argument('--min-common', type=int, default=15,
                        help='最少共同标签数 (默认: 15)')
    parser.add_argument('--execute', action='store_true',
                        help='真正执行删除（默认只预览）')
    args = parser.parse_args()

    dry_run = not args.execute

    if not dry_run:
        print("\n⚠️  注意: Step 4 已经内置了自动质量过滤功能！")
        print("   通常不需要手动运行此工具。")
        response = input("\n确认要手动删除低质量图像吗？(会备份到 images/filtered_backup/) [y/N]: ")
        if response.lower() != 'y':
            print("已取消")
            return

    keep, remove = analyze_and_filter(
        min_common_tags=args.min_common,
        dry_run=dry_run
    )

    if not dry_run and keep > 0:
        print("\n" + "="*70)
        print("建议下一步")
        print("="*70)
        print("\n1. 运行 step4 重新标定:")
        print("   python3 step4_stereo_extrinsic.py")
        print("\n2. 检查新的重投影误差")
        print("\n3. 如果误差还是太高，考虑:")
        print("   - 重新采集更多优质图像")
        print("   - 使用 capture_helper.py 辅助采集")


if __name__ == "__main__":
    main()
