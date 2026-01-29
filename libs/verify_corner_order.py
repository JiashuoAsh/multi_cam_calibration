#!/usr/bin/env python3
"""
角点顺序验证工具 - 快速检查2D检测与3D定义是否匹配

使用场景：
  - 更换相机或改变相机方向后
  - 修改角点顺序映射后
  - 标定误差异常时

用法：
  python verify_corner_order.py

输出：
  - 在终端打印第一个检测到的标签的角点位置分析
  - 保存可视化图像到 results/corner_verification.jpg

预期结果：
  3D定义和2D检测的角点位置应该一致，例如：
  - 3D角点0=左上 ↔ 2D角点0=左上 ✓
  - 3D角点1=右上 ↔ 2D角点1=右上 ✓
  - 3D角点2=右下 ↔ 2D角点2=右下 ✓
  - 3D角点3=左下 ↔ 2D角点3=左下 ✓
"""

import cv2
import numpy as np
from pathlib import Path
from utils import (
    load_config,
    create_apriltag_board,
    get_aruco_dict,
    detect_apriltag_corners,
)


def verify_corner_order():
    """验证角点顺序是否正确"""
    print("=" * 70)
    print("角点顺序验证工具")
    print("=" * 70)

    # 加载配置
    config = load_config()
    obj_points_all, tag_ids = create_apriltag_board(config)
    aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])

    # 获取测试图像：从 images/filtered/<cam>/ 下自动选择一张
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    filtered_root = Path("images/filtered")
    test_images: list[Path] = []
    test_cam: str = ""
    if filtered_root.exists():
        for cam_dir in sorted(filtered_root.iterdir()):
            if not cam_dir.is_dir():
                continue
            name = cam_dir.name
            if name.startswith(".") or name.startswith("_") or name.startswith("__"):
                continue
            imgs = [p for p in sorted(cam_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
            if len(imgs) > 0:
                test_images = imgs
                test_cam = name
                break

    if len(test_images) == 0:
        print("错误: 未找到任何图像（期望 images/filtered/<cam>/*.(png|jpg|jpeg|bmp)）")
        return False

    # 使用第二张图像（通常检测效果较好）
    test_image = test_images[min(1, len(test_images) - 1)]
    print(f"\n测试相机: {test_cam}")
    print(f"测试图像: {test_image}")

    # 读取并检测
    img = cv2.imread(str(test_image))
    if img is None:
        print(f"错误: 无法读取图像: {test_image}")
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    corners, ids = detect_apriltag_corners(
        gray, aruco_dict, detector_params, use_multiscale=True
    )

    if corners is None or ids is None or len(ids) == 0:
        print("❌ 未检测到任何标签")
        return False

    print(f"✓ 检测到 {len(ids)} 个标签")

    # 分析第一个标签
    tag_id = ids[0][0]
    tag_corners_2d = corners[0].reshape(-1, 2)

    print(f"\n🔍 分析标签 ID = {tag_id}")
    print("-" * 70)

    # 获取3D坐标
    if tag_id not in tag_ids:
        print(f"❌ 标签 ID {tag_id} 不在配置中")
        return False

    board_idx = tag_ids.index(tag_id)
    obj_corners_3d = obj_points_all[board_idx]

    # 分析3D坐标位置
    print("\n📐 3D坐标分析 (标定板坐标系):")
    center_3d = obj_corners_3d[:, :2].mean(axis=0)
    positions_3d = []
    for i in range(4):
        x, y = obj_corners_3d[i, :2]
        pos_lr = "左" if x < center_3d[0] else "右"
        pos_ud = "下" if y < center_3d[1] else "上"
        pos = f"{pos_lr}{pos_ud}"
        positions_3d.append(pos)
        print(f"  角点 {i}: {pos:4s} (X={x:6.1f}, Y={y:6.1f})")

    # 分析2D坐标位置
    print("\n📷 2D坐标分析 (图像坐标系):")
    center_2d = tag_corners_2d.mean(axis=0)
    positions_2d = []
    match_count = 0
    for i in range(4):
        u, v = tag_corners_2d[i]
        pos_lr = "左" if u < center_2d[0] else "右"
        pos_ud = "上" if v < center_2d[1] else "下"  # 图像Y向下
        pos = f"{pos_lr}{pos_ud}"
        positions_2d.append(pos)

        # 检查是否匹配
        match = "✓" if pos == positions_3d[i] else "✗"
        if pos == positions_3d[i]:
            match_count += 1
        print(f"  角点 {i}: {pos:4s} (u={u:6.1f}, v={v:6.1f}) {match}")

    # 判断结果
    print("\n" + "=" * 70)
    if match_count == 4:
        print("✅ 角点顺序验证通过！所有角点位置匹配")
        print("   可以继续进行标定")
        result = True
    else:
        print(f"⚠️  角点顺序不匹配！匹配数量: {match_count}/4")
        print("   需要调整 utils.py 中的角点顺序映射")
        print(f"   当前映射: [2,3,0,1]")
        print(f"   3D位置: {positions_3d}")
        print(f"   2D位置: {positions_2d}")
        result = False
    print("=" * 70)

    # 可视化
    vis_img = img.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # 红绿蓝黄
    color_names = ["红(0)", "绿(1)", "蓝(2)", "黄(3)"]

    for i, (corner, color) in enumerate(zip(tag_corners_2d, colors)):
        pt = tuple(corner.astype(int))
        cv2.circle(vis_img, pt, 12, color, -1)
        cv2.putText(
            vis_img,
            str(i),
            (pt[0] + 20, pt[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )

    # 添加标签ID
    center = tag_corners_2d.mean(axis=0).astype(int)
    cv2.putText(
        vis_img,
        f"ID: {tag_id}",
        (center[0] - 40, center[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )

    # 保存
    output_path = "results/corner_verification.jpg"
    cv2.imwrite(output_path, vis_img)
    print(f"\n💾 可视化图像已保存: {output_path}")
    print(f"   图例: {', '.join(color_names)}")

    return result


if __name__ == "__main__":
    import sys

    success = verify_corner_order()
    sys.exit(0 if success else 1)
