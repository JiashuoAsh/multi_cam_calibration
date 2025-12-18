#!/usr/bin/env python3
"""
Step 1: 采集图像 - AprilTag 标定板

功能:
    纯粹的图像采集工具，保存所有拍摄的图像，不进行任何质量检查或检测。
    这是新标定流程的第一步，将采集和筛选完全分离。

工作流程:
    1. 初始化相机（根据配置文件自动选择 USB/MIPI 相机）
    2. 实时显示左右相机画面
    3. 用户按键保存图像对
    4. 所有图像保存到 images/raw/ 目录

使用方法:
    python step1_capture_imgs.py

输出:
    - images/raw/left/*.png: 原始左相机图像（所有拍摄的图像）
    - images/raw/right/*.png: 原始右相机图像（所有拍摄的图像）

采集建议:
    - 拍摄 20-50 组图像
    - 尽量包含标定板，但即使没有也会保存
    - 变化相机位置和角度（远近、俯仰、偏航）
    - 覆盖视野的不同区域（中心和边缘）

注意事项:
    - 所有图像都会被保存，无论是否包含标定板
    - 图像质量检查和筛选在 step2 中进行
    - 建议保持拍摄时的稳定，避免运动模糊

下一步:
    运行 python step2_filter_images.py 筛选合格图像
"""

import cv2
import os
import numpy as np
import time
from datetime import datetime
from traceback import print_exc
from utils import load_config, init_camera


def main():
    """
    主函数：执行原始图像采集流程

    流程：
        1. 加载配置文件
        2. 创建输出目录
        3. 初始化相机
        4. 循环显示相机画面
        5. 用户按键保存图像
        6. 达到目标数量或用户退出
    """
    print("=" * 60)
    print("Step 1: AprilTag 原始图像采集")
    print("=" * 60)

    # 加载配置
    config = load_config()
    calib_cfg = config["calibration_settings"]

    # 拍照间隔（秒）
    capture_interval = calib_cfg.get("capture_interval", 0.1)

    print(f"\n配置:")
    print(f"  - 目标图像数量: {calib_cfg['max_images']}")
    print(f"  - 自动拍照间隔: {capture_interval} 秒")
    print(f"\n注意:")
    print(f"  - 所有图像都会被保存，无论是否包含标定板")
    print(f"  - 图像质量检查将在 step2 中进行")
    print(f"  - 自动拍照模式（无GUI显示）")

    # 创建输出目录（使用 raw 子目录）
    os.makedirs("images/raw/left", exist_ok=True)
    os.makedirs("images/raw/right", exist_ok=True)

    # 检查是否有旧图像，询问是否清空
    left_files = [
        f
        for f in os.listdir("images/raw/left")
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    right_files = [
        f
        for f in os.listdir("images/raw/right")
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    if left_files or right_files:
        print(f"\n⚠️  检测到旧图像文件:")
        print(f"   - 左相机: {len(left_files)} 张")
        print(f"   - 右相机: {len(right_files)} 张")
        print(f"\n是否清空旧图像？")
        response = input("  输入 'y' 清空，其他键保留旧图像并继续: ").strip().lower()

        if response == "y":
            # 清空目录
            import shutil

            for f in left_files:
                os.remove(os.path.join("images/raw/left", f))
            for f in right_files:
                os.remove(os.path.join("images/raw/right", f))
            print(f"✓ 已清空 {len(left_files) + len(right_files)} 张旧图像")
        else:
            print("✓ 保留旧图像，新图像将追加保存")

    # 初始化相机
    print("\n初始化相机...")
    try:
        camera = init_camera(config)
    except Exception as e:
        print(f"错误: 无法初始化相机: {e}")
        print_exc()
        return

    saved_count = 0
    max_images = calib_cfg["max_images"]

    print("\n开始自动采集图像...")
    print(f"目标: 采集 {max_images} 组图像")
    print(f"间隔: 每 {capture_interval} 秒自动拍照一次")
    print("\n按 Ctrl+C 可提前终止")
    print("-" * 60)

    try:
        while saved_count < max_images:
            # 读取图像
            left_img, right_img, timestamp = camera.read_stereo()

            if left_img is None or right_img is None:
                print("警告: 无法读取相机图像，等待下次尝试...")
                time.sleep(1)
                continue

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            left_path = f"images/raw/left/{timestamp}.png"
            right_path = f"images/raw/right/{timestamp}.png"

            # 保存图像（不做任何检查）
            cv2.imwrite(left_path, left_img)
            cv2.imwrite(right_path, right_img)

            saved_count += 1

            print(f"✓ 已保存 #{saved_count}/{max_images}: {timestamp}.png")

            if saved_count >= max_images:
                print(f"\n已达到目标数量 ({max_images})，自动退出")
                break

            # 等待指定间隔后继续拍照
            time.sleep(capture_interval)

    except KeyboardInterrupt:
        print("\n\n用户中断 (Ctrl+C)")

    finally:
        # 清理
        camera.release()

        print("\n" + "=" * 60)
        print(f"采集完成！共保存 {saved_count} 组原始图像")
        print("=" * 60)
        print(f"\n图像保存位置:")
        print(f"  - images/raw/left/")
        print(f"  - images/raw/right/")
        print("\n下一步: 运行 python step2_filter_images.py 筛选合格图像")


if __name__ == "__main__":
    main()
