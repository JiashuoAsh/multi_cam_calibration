#!/usr/bin/env python3
"""
Step 3: 内参标定 - AprilTag 标定板

命名规范:
    A_T_B 表示 "B -> A 的变换" (X_A = A_T_B @ X_B)

功能:
    使用筛选后的合格图像进行左右相机的内参标定。
    计算相机内参矩阵 K 和畸变系数 dist。

工作流程:
    1. 读取筛选后的图像
    2. 检测每张图像中的 AprilTag 标签
    3. 建立 2D-3D 点对应关系
    4. 使用 cv2.calibrateCamera 优化内参
    5. 生成去畸变效果对比图

标定方法:
    - 使用 OpenCV 的 cv2.calibrateCamera 函数
    - 自动计算焦距、主点和畸变系数

使用方法:
    python step3_intrinsic_apriltag.py

输入:
    - images/filtered/left/*.png: 筛选后的左相机图像
    - images/filtered/right/*.png: 筛选后的右相机图像
    - config/apriltag_config.json: 标定板配置

输出:
    - results/left_intrinsics.json: 左相机内参
      - camera_matrix: 3x3 内参矩阵 K
      - dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3, ...]
      - reprojection_error: 重投影误差（像素）
    - results/right_intrinsics.json: 右相机内参
    - results/left_undistortion_demo.jpg: 去畸变效果对比
    - results/right_undistortion_demo.jpg

质量评估:
    - 重投影误差 < 0.5 像素: 优秀
    - 重投影误差 < 1.0 像素: 良好
    - 重投影误差 > 1.0 像素: 需要改进（检查标定板或图像质量）

下一步:
    运行 python step4_stereo_extrinsic.py 进行双目外参标定
"""

import cv2
import numpy as np
import json
import os
import glob
import shutil
from utils import (
    load_config,
    get_aruco_dict,
    detect_apriltag_corners,
    create_apriltag_board,
    create_opencv_aruco_board,
    get_detection_settings,
)

# 最大有效图像数量（用于内参标定）
MAX_VALID_IMAGES = 150


def clean_visualization_dirs():
    """清空 step3 的可视化输出目录"""
    vis_dirs = [
        "results/visualization/step3_intrinsic_左",
        "results/visualization/step3_intrinsic_右",
    ]

    for vis_dir in vis_dirs:
        if os.path.exists(vis_dir):
            shutil.rmtree(vis_dir)
            print(f"  已清空: {vis_dir}")


def calibrate_camera_apriltag(
    image_files,
    obj_points_all,
    tag_ids,
    aruco_dict,
    side_name,
    *,
    use_multiscale: bool,
    opencv_refine: bool,
    board,
    save_visualization: bool = True,
    max_valid_images: int = MAX_VALID_IMAGES,
):
    """
    使用 AprilTag 标定单个相机的内参

    命名规范:
        cv2.calibrateCamera 返回的 rvecs/tvecs 表示 "T -> C" (标定板 -> 相机)
        即 Cl_T_T 或 Cr_T_T，但本函数不直接使用这些变换矩阵

    Args:
        image_files: 图像文件路径列表
        obj_points_all: 所有标签的 3D 角点 (num_tags, 4, 3)
        tag_ids: 标定板上所有标签的 ID 列表
        aruco_dict: ArUco 字典
        side_name: 相机名称（用于日志）
        save_visualization: 是否保存可视化图像（detection + reprojection）
        max_valid_images: 最大有效图像数量，默认50张

    Returns:
        success: 是否标定成功
        K: 相机内参矩阵 (3, 3)
        dist: 畸变系数
        rvecs: 每张图像的旋转向量列表 (T -> C)
        tvecs: 每张图像的平移向量列表 (T -> C)
        reproj_error: 平均重投影误差

    """
    print(f"\n{side_name}相机标定:")
    print(f"  - 图像数量: {len(image_files)}")
    print(f"  - 最大有效图像: {max_valid_images}")

    # 创建可视化输出目录
    if save_visualization:
        vis_base = f"results/visualization/step3_intrinsic_{side_name.lower()}"
        detection_dir = f"{vis_base}/detection"
        reprojection_dir = f"{vis_base}/reprojection"
        undistorted_dir = f"{vis_base}/undistorted"
        os.makedirs(detection_dir, exist_ok=True)
        os.makedirs(reprojection_dir, exist_ok=True)
        os.makedirs(undistorted_dir, exist_ok=True)
        print(f"  - 可视化目录: {vis_base}")

    # 设置检测器参数
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # 收集所有有效图像的 2D-3D 对应点
    all_obj_pts = []
    all_img_pts = []
    valid_images = []
    valid_image_data = []  # 保存图像数据用于后续可视化
    image_size = None

    for img_idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告: 无法读取 {img_path}")
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_apriltag_corners(
            gray,
            aruco_dict,
            detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
        )

        if ids is None or len(ids) < 5:
            print(
                f"  跳过 {os.path.basename(img_path)}: 检测到的标签不足 ({len(ids) if ids is not None else 0})"
            )
            continue

        # 收集该图像中所有标签的对应点
        img_obj_pts = []
        img_img_pts = []

        ids_flat = ids.flatten()
        for i, tag_id in enumerate(ids_flat):
            if tag_id in tag_ids:
                idx = tag_ids.index(tag_id)
                obj_pts = obj_points_all[idx]  # (4, 3)
                img_pts = corners[i].reshape(-1, 2)  # (4, 2)

                img_obj_pts.append(obj_pts)
                img_img_pts.append(img_pts)

        if len(img_obj_pts) > 0:
            # 合并该图像的所有角点
            obj_pts_img = np.vstack(img_obj_pts).astype(np.float32)
            img_pts_img = np.vstack(img_img_pts).astype(np.float32)

            all_obj_pts.append(obj_pts_img)
            all_img_pts.append(img_pts_img)
            valid_images.append(img_path)

            # 检查是否达到最大有效图像数量
            if len(valid_images) >= max_valid_images:
                print(f"  已达到最大有效图像数量 ({max_valid_images})，停止处理")
                break

            # 保存数据用于可视化
            if save_visualization:
                valid_image_data.append(
                    {
                        "image": img.copy(),
                        "corners": corners,
                        "ids": ids,
                        "index": len(valid_images),
                    }
                )

                # 保存 AprilTag 检测结果图
                vis_img = img.copy()
                cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
                info_text = f"Image {len(valid_images)}: {len(ids)} tags detected"
                cv2.putText(
                    vis_img,
                    info_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                detection_path = (
                    f"{detection_dir}/{len(valid_images):02d}_tags_detected.jpg"
                )
                cv2.imwrite(detection_path, vis_img)

    print(f"  - 有效图像: {len(valid_images)}/{len(image_files)}")

    if len(valid_images) < 3:
        print(f"  错误: 有效图像太少 (<3)")
        return False, None, None, None, None, None, None

    # 执行相机标定
    print("  - 正在标定...")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_pts,
        all_img_pts,
        image_size,
        None,
        None,
    )

    # OpenCV 返回的 ret 是全局 RMS 重投影误差（单位：像素）
    print(f"  - RMS(ret, OpenCV): {ret:.4f} 像素")

    # 计算重投影误差并保存可视化
    mean_error = 0
    per_image_errors = []

    for i in range(len(all_obj_pts)):
        img_pts2, _ = cv2.projectPoints(all_obj_pts[i], rvecs[i], tvecs[i], K, dist)
        # mean reprojection error（像素）: 每个点的欧氏误差取平均
        proj = img_pts2.reshape(-1, 2)
        det = np.asarray(all_img_pts[i], dtype=np.float64).reshape(-1, 2)
        per_pt = np.linalg.norm(det - proj, axis=1)
        error = float(np.mean(per_pt))
        mean_error += error
        per_image_errors.append(error)

        # 保存重投影误差可视化
        if save_visualization and i < len(valid_image_data):
            vis_img = valid_image_data[i]["image"].copy()
            img_pts_orig = det
            img_pts_reproj = img_pts2.reshape(-1, 2)

            # 绘制原始检测点（绿色）和重投影点（红色）
            for j in range(len(img_pts_orig)):
                pt_orig = tuple(img_pts_orig[j].astype(int))
                pt_reproj = tuple(img_pts_reproj[j].astype(int))

                cv2.circle(vis_img, pt_orig, 6, (0, 255, 0), -1)  # 绿色：检测点
                cv2.circle(vis_img, pt_reproj, 4, (0, 0, 255), -1)  # 红色：重投影点
                cv2.line(vis_img, pt_orig, pt_reproj, (255, 0, 0), 1)  # 蓝线：误差

            # 显示误差信息
            error_text = (
                f"Image {valid_image_data[i]['index']}: Mean Error = {error:.3f} px"
            )
            cv2.putText(
                vis_img,
                
                error_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis_img,
                "Green: Detected | Red: Reprojected",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            reproj_path = (
                f"{reprojection_dir}/{valid_image_data[i]['index']:02d}_error_map.jpg"
            )
            cv2.imwrite(reproj_path, vis_img)

    mean_error /= len(all_obj_pts)

    print(f"  - 平均误差(mean_error, per-image mean): {mean_error:.4f} 像素")

    # 评估标定质量
    if mean_error < 0.5:
        quality = "优秀"
    elif mean_error < 1.0:
        quality = "良好"
    else:
        quality = "需要改进"

    print(f"  - 标定质量: {quality}")

    if save_visualization:
        print(f"  - 已保存 {len(valid_images)} 张检测图像到 {detection_dir}")
        print(f"  - 已保存 {len(valid_images)} 张重投影图像到 {reprojection_dir}")

        # 保存所有图像的去畸变版本
        print(f"  - 正在生成去畸变图像...")

        # 使用第一张图像的尺寸创建去畸变映射（所有图像尺寸相同）
        first_img = cv2.imread(valid_images[0])
        h, w = first_img.shape[:2]

        # 使用 remap 方法进行去畸变（推荐方法）
        # 保持原始内参矩阵K，不改变图像尺寸，只矫正畸变
        mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)

        for i, img_path in enumerate(valid_images):
            img = cv2.imread(img_path)

            # 使用remap进行去畸变（比undistort更快且效果更好）
            img_undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            # 保存去畸变图像（保持原始尺寸720x1280，不缩放）
            base_name = os.path.basename(img_path)
            undist_path = f"{undistorted_dir}/{i + 1:02d}_undistorted_{base_name}"
            cv2.imwrite(undist_path, img_undist)

        print(f"  - 已保存 {len(valid_images)} 张去畸变图像到 {undistorted_dir}")

    return True, K, dist, rvecs, tvecs, mean_error, image_size


def save_intrinsics(output_path, K, dist, reproj_error, image_size):
    """保存内参标定结果"""
    result = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "reprojection_error": float(reproj_error),
        "image_size": list(image_size),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  - 已保存: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("Step 3: AprilTag 内参标定")
    print("=" * 60)

    # 清空之前的可视化输出
    print("\n清理旧的可视化文件...")
    clean_visualization_dirs()

    # 检查图像目录（优先 filtered，若为空则回退到 raw）
    if (not os.path.exists("images/filtered/left") or not os.path.exists("images/filtered/right")) and (
        not os.path.exists("images/raw/left") or not os.path.exists("images/raw/right")
    ):
        print("\n错误: 未找到图像目录 images/filtered 或 images/raw")
        print("请先采集图像（或运行 python step2_filter_images.py 生成 filtered 图像）")
        return

    # 加载配置
    config = load_config()

    use_multiscale, opencv_refine = get_detection_settings(config)

    # 创建 AprilTag 标定板
    obj_points, tag_ids = create_apriltag_board(config)
    aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])
    board = create_opencv_aruco_board(obj_points, tag_ids, aruco_dict)

    # 获取图像文件（优先 filtered，若为空则回退到 raw）
    left_images = sorted(glob.glob("images/filtered/left/*.png"))
    right_images = sorted(glob.glob("images/filtered/right/*.png"))
    image_source = "filtered"
    if len(left_images) == 0 or len(right_images) == 0:
        left_images = sorted(glob.glob("images/raw/left/*.png"))
        right_images = sorted(glob.glob("images/raw/right/*.png"))
        image_source = "raw"

    if len(left_images) == 0 or len(right_images) == 0:
        print("\n错误: 未找到图像！")
        print("请先采集图像（或运行 python step2_filter_images.py 生成 filtered 图像）")
        return

    print(f"\n找到图像 (来源: images/{image_source}/):")
    print(f"  - 左相机: {len(left_images)} 张")
    print(f"  - 右相机: {len(right_images)} 张")

    # 确保输出目录存在
    os.makedirs("results", exist_ok=True)

    # 标定左相机
    success_l, K_l, dist_l, rvecs_l, tvecs_l, err_l, img_size_l = (
        calibrate_camera_apriltag(
            left_images,
            obj_points,
            tag_ids,
            aruco_dict,
            "左",
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            max_valid_images=MAX_VALID_IMAGES,
        )
    )

    if success_l:
        save_intrinsics("results/left_intrinsics.json", K_l, dist_l, err_l, img_size_l)
    else:
        print("\n错误: 左相机标定失败！")
        return

    # 标定右相机
    success_r, K_r, dist_r, rvecs_r, tvecs_r, err_r, img_size_r = (
        calibrate_camera_apriltag(
            right_images,
            obj_points,
            tag_ids,
            aruco_dict,
            "右",
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            max_valid_images=MAX_VALID_IMAGES,
        )
    )

    if success_r:
        save_intrinsics("results/right_intrinsics.json", K_r, dist_r, err_r, img_size_r)
    else:
        print("\n错误: 右相机标定失败！")
        return

    # 显示总结
    print("\n" + "=" * 60)
    print("内参标定完成！")
    print("=" * 60)
    print(f"\n左相机:")
    print(f"  - fx = {K_l[0, 0]:.2f}, fy = {K_l[1, 1]:.2f}")
    print(f"  - cx = {K_l[0, 2]:.2f}, cy = {K_l[1, 2]:.2f}")
    print(f"  - 平均误差(mean_error) = {err_l:.4f} 像素")
    print("  - RMS(ret) 见上方标定日志")

    print(f"\n右相机:")
    print(f"  - fx = {K_r[0, 0]:.2f}, fy = {K_r[1, 1]:.2f}")
    print(f"  - cx = {K_r[0, 2]:.2f}, cy = {K_r[1, 2]:.2f}")
    print(f"  - 平均误差(mean_error) = {err_r:.4f} 像素")
    print("  - RMS(ret) 见上方标定日志")

    print("\n下一步: 运行 python step4_stereo_extrinsic.py")


if __name__ == "__main__":
    main()
