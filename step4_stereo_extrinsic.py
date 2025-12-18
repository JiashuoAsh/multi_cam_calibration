#!/usr/bin/env python3
"""
Step 4: 双目外参标定 - AprilTag 标定板

使用左右相机的内参，标定双目相机的相对位姿。

输入:
    - images/filtered/left/*.png: 筛选后的左相机图像
    - images/filtered/right/*.png: 筛选后的右相机图像
    - results/left_intrinsics.json
    - results/right_intrinsics.json

输出:
    - results/stereo_extrinsics.json (R, t, E, F, baseline)
    - results/stereo_rectification.json (R1, R2, P1, P2, Q)
    - results/stereo_rectification_demo.jpg
"""

import cv2
import numpy as np
import json
import os
import glob
from utils import (
    load_config,
    get_aruco_dict,
    detect_apriltag_corners,
    create_apriltag_board,
    create_opencv_aruco_board,
    get_detection_settings,
)

# 最大有效图像对数量（用于双目外参标定）
MAX_VALID_IMAGES = 150


def load_intrinsics(json_path):
    """加载内参标定结果"""
    with open(json_path, "r") as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)

    return K, dist

def compute_mean_reproj_error_pnp(all_obj_pts, all_img_pts, K, dist):
    """
    仿照 step3 的口径计算 mean_error（每张图像的 mean reprojection error 再取平均）
    error = mean(||x_detect - x_proj||_2)
    """
    per_view_errors = []
    used = 0

    for obj_pts, img_pts in zip(all_obj_pts, all_img_pts):
        obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
        img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)

        if len(obj_pts) < 6:
            continue

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)

        per_pt = np.linalg.norm(img_pts - proj, axis=1)
        err = float(np.mean(per_pt))
        per_view_errors.append(float(err))
        used += 1

    mean_error = float(np.mean(per_view_errors)) if used > 0 else float("nan")
    return mean_error, per_view_errors, used


def compute_right_mean_reproj_error_using_rt(
    all_obj_pts,
    all_img_pts_l,
    all_img_pts_r,
    K_l,
    dist_l,
    K_r,
    dist_r,
    R_lr,
    t_lr,
):
    """
    右目误差（使用 stereo 的 R,t）：
      1) 左目 solvePnP 得到 (Board -> Left) 的位姿
      2) 用 stereo 外参把位姿变到右目 (Board -> Right_pred)
      3) 在右目上 projectPoints，与右目检测点算误差
    """
    per_view_errors = []
    used = 0

    R_lr = np.asarray(R_lr, dtype=np.float64)
    t_lr = np.asarray(t_lr, dtype=np.float64).reshape(3, 1)

    for obj_pts, img_l, img_r in zip(all_obj_pts, all_img_pts_l, all_img_pts_r):
        obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
        img_l = np.asarray(img_l, dtype=np.float64).reshape(-1, 2)
        img_r = np.asarray(img_r, dtype=np.float64).reshape(-1, 2)

        if len(obj_pts) < 6:
            continue

        ok, rvec_l, tvec_l = cv2.solvePnP(
            obj_pts, img_l, K_l, dist_l, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue

        R_l, _ = cv2.Rodrigues(rvec_l)          # Board -> Left
        R_r = R_lr @ R_l                         # Board -> Right_pred
        t_r = R_lr @ tvec_l + t_lr               # Board -> Right_pred
        rvec_r, _ = cv2.Rodrigues(R_r)

        proj_r, _ = cv2.projectPoints(obj_pts, rvec_r, t_r, K_r, dist_r)
        proj_r = proj_r.reshape(-1, 2)

        per_pt = np.linalg.norm(img_r - proj_r, axis=1)
        err_r = float(np.mean(per_pt))
        per_view_errors.append(float(err_r))
        used += 1

    mean_error = float(np.mean(per_view_errors)) if used > 0 else float("nan")
    return mean_error, per_view_errors, used


def compute_stereo_mean_reproj_error_using_rt(
    all_obj_pts,
    all_img_pts_l,
    all_img_pts_r,
    K_l,
    dist_l,
    K_r,
    dist_r,
    R_lr,
    t_lr,
):
    """
    计算双目 mean reprojection error（像素，mean 口径）：
      1) 对每对图像，在左目用 solvePnP 解 (Board -> Left)
      2) 用 stereo 外参推导 (Board -> Right_pred)
      3) 将3D点投影回左右图像，与检测点计算每点欧氏误差
      4) 汇总所有点（左右合计）的 mean

    返回:
      mean_error_all: 所有点（左右合计）的 mean reprojection error
      mean_error_left: 左目 mean
      mean_error_right: 右目 mean
      used_pairs: 实际用于统计的图像对数量
    """
    R_lr = np.asarray(R_lr, dtype=np.float64)
    t_lr = np.asarray(t_lr, dtype=np.float64).reshape(3, 1)

    all_err = []
    all_err_l = []
    all_err_r = []
    used_pairs = 0

    for obj_pts, img_l, img_r in zip(all_obj_pts, all_img_pts_l, all_img_pts_r):
        obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
        img_l = np.asarray(img_l, dtype=np.float64).reshape(-1, 2)
        img_r = np.asarray(img_r, dtype=np.float64).reshape(-1, 2)

        if len(obj_pts) < 6:
            continue

        ok, rvec_l, tvec_l = cv2.solvePnP(
            obj_pts, img_l, K_l, dist_l, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue

        # Left reprojection
        proj_l, _ = cv2.projectPoints(obj_pts, rvec_l, tvec_l, K_l, dist_l)
        proj_l = proj_l.reshape(-1, 2)
        err_l = np.linalg.norm(img_l - proj_l, axis=1)
        all_err.extend(err_l.tolist())
        all_err_l.extend(err_l.tolist())

        # Right reprojection using stereo constraint
        R_l, _ = cv2.Rodrigues(rvec_l)
        R_r = R_lr @ R_l
        t_r = R_lr @ tvec_l + t_lr
        rvec_r, _ = cv2.Rodrigues(R_r)

        proj_r, _ = cv2.projectPoints(obj_pts, rvec_r, t_r, K_r, dist_r)
        proj_r = proj_r.reshape(-1, 2)
        err_r = np.linalg.norm(img_r - proj_r, axis=1)
        all_err.extend(err_r.tolist())
        all_err_r.extend(err_r.tolist())

        used_pairs += 1

    if used_pairs == 0:
        return float("nan"), float("nan"), float("nan"), 0

    mean_all = float(np.mean(np.asarray(all_err, dtype=np.float64)))
    mean_l = float(np.mean(np.asarray(all_err_l, dtype=np.float64)))
    mean_r = float(np.mean(np.asarray(all_err_r, dtype=np.float64)))
    return mean_all, mean_l, mean_r, used_pairs


def collect_stereo_points(
    left_images,
    right_images,
    obj_points_all,
    tag_ids,
    aruco_dict,
    *,
    use_multiscale: bool,
    opencv_refine: bool,
    board,
    K_l,
    dist_l,
    K_r,
    dist_r,
    max_valid_images: int = MAX_VALID_IMAGES,
):
    """
    收集双目图像对中的对应点

    要求: 左右图像必须同时检测到标签
    max_valid_images: 最大有效图像对数量，默认50对
    """
    print("\n收集双目对应点...")
    print(f"  - 最大有效图像对: {max_valid_images}")

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    all_obj_pts = []
    all_img_pts_l = []
    all_img_pts_r = []
    valid_pairs = []

    for left_path, right_path in zip(left_images, right_images):
        # 读取图像
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            continue

        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # 检测标签（使用多尺度检测）
        left_corners, left_ids = detect_apriltag_corners(
            left_gray,
            aruco_dict,
            detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            camera_matrix=K_l,
            dist_coeffs=dist_l,
        )
        right_corners, right_ids = detect_apriltag_corners(
            right_gray,
            aruco_dict,
            detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            camera_matrix=K_r,
            dist_coeffs=dist_r,
        )

        if left_ids is None or right_ids is None:
            continue

        if len(left_ids) < 4 or len(right_ids) < 4:
            continue

        # 找到左右图像中共同检测到的标签
        left_ids_flat = left_ids.flatten()
        right_ids_flat = right_ids.flatten()
        common_ids = set(left_ids_flat) & set(right_ids_flat)

        if len(common_ids) < 4:
            continue

        # 收集共同标签的对应点
        obj_pts_pair = []
        img_pts_l_pair = []
        img_pts_r_pair = []

        for tag_id in common_ids:
            if tag_id not in tag_ids:
                continue

            # 在标定板上的索引
            board_idx = tag_ids.index(tag_id)
            obj_pts = obj_points_all[board_idx]  # (4, 3)

            # 在左图像中的索引
            left_idx = np.where(left_ids_flat == tag_id)[0][0]
            img_pts_l = left_corners[left_idx].reshape(-1, 2)

            # 在右图像中的索引
            right_idx = np.where(right_ids_flat == tag_id)[0][0]
            img_pts_r = right_corners[right_idx].reshape(-1, 2)

            obj_pts_pair.append(obj_pts)
            img_pts_l_pair.append(img_pts_l)
            img_pts_r_pair.append(img_pts_r)

        if len(obj_pts_pair) > 0:
            # 合并该图像对的所有角点
            obj_pts_combined = np.vstack(obj_pts_pair).astype(np.float32)
            img_pts_l_combined = np.vstack(img_pts_l_pair).astype(np.float32)
            img_pts_r_combined = np.vstack(img_pts_r_pair).astype(np.float32)

            all_obj_pts.append(obj_pts_combined)
            all_img_pts_l.append(img_pts_l_combined)
            all_img_pts_r.append(img_pts_r_combined)
            valid_pairs.append((left_path, right_path))

            # 检查是否达到最大有效图像对数量
            if len(valid_pairs) >= max_valid_images:
                print(f"  已达到最大有效图像对数量 ({max_valid_images})，停止处理")
                break

    print(f"  - 有效双目图像对: {len(valid_pairs)}/{len(left_images)}")

    return all_obj_pts, all_img_pts_l, all_img_pts_r, valid_pairs


def main():
    """主函数"""
    print("=" * 60)
    print("Step 4: AprilTag 双目外参标定")
    print("=" * 60)

    # 检查内参文件
    if not os.path.exists("results/left_intrinsics.json"):
        print("\n错误: 未找到 results/left_intrinsics.json")
        print("请先运行 python step3_intrinsic_apriltag.py")
        return

    if not os.path.exists("results/right_intrinsics.json"):
        print("\n错误: 未找到 results/right_intrinsics.json")
        print("请先运行 python step3_intrinsic_apriltag.py")
        return

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

    # 加载内参
    print("\n加载内参...")
    K_l, dist_l = load_intrinsics("results/left_intrinsics.json")
    K_r, dist_r = load_intrinsics("results/right_intrinsics.json")
    print("  ✓ 左右相机内参已加载")

    # 创建 AprilTag 标定板
    obj_points, tag_ids = create_apriltag_board(config)
    aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])
    board = create_opencv_aruco_board(obj_points, tag_ids, aruco_dict)

    # 获取图像（优先 filtered，若为空则回退到 raw）
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

    if len(left_images) != len(right_images):
        print(f"\n警告: 左右图像数量不匹配 ({len(left_images)} vs {len(right_images)})")

    print(f"\n找到 {len(left_images)} 对图像 (来源: images/{image_source}/)")

    # ========== 自动质量过滤 ==========
    print("\n分析图像对质量...")
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    from utils import analyze_stereo_image_quality, filter_low_quality_pairs

    image_quality = analyze_stereo_image_quality(
        left_images,
        right_images,
        aruco_dict,
        detector_params,
        use_multiscale=use_multiscale,
        opencv_refine=opencv_refine,
        board=board,
        left_camera_matrix=K_l,
        left_dist_coeffs=dist_l,
        right_camera_matrix=K_r,
        right_dist_coeffs=dist_r,
    )

    # 设置质量阈值（会聚式双目推荐 15+）
    MIN_COMMON_TAGS = 7

    keep_pairs, remove_pairs = filter_low_quality_pairs(
        image_quality, min_common_tags=MIN_COMMON_TAGS
    )

    # 显示统计信息
    print(f"\n质量过滤结果:")
    print(f"  ✅ 保留: {len(keep_pairs)} 对")
    print(f"  ❌ 移除: {len(remove_pairs)} 对（<{MIN_COMMON_TAGS} 共同标签）")

    if keep_pairs:
        common_counts = [q[2] for q in image_quality if q[2] >= MIN_COMMON_TAGS]
        print(f"\n保留图像对的质量:")
        print(f"  平均共同标签: {np.mean(common_counts):.1f} 个")
        print(f"  范围: {np.min(common_counts)} - {np.max(common_counts)} 个")
        print(f"  标准差: {np.std(common_counts):.1f} 个")

    if len(keep_pairs) < 10:
        print(f"\n⚠️  警告: 只有 {len(keep_pairs)} 对高质量图像")
        print(f"   会聚式双目建议至少 20 对优质图像以获得最佳标定质量")
        print(f"   当前配置可能导致标定精度降低")

    if len(keep_pairs) < 5:
        print(f"\n❌ 错误: 高质量图像对太少（< 5 对）")
        print(f"   无法进行可靠的双目标定")
        print(f"\n建议:")
        print(f"   1. 重新采集更多图像，确保标定板在两相机重叠视野中央")
        print(f"   2. 每对图像应至少检测到 {MIN_COMMON_TAGS} 个共同标签")
        return

    # 使用过滤后的图像列表
    left_images = [pair[0] for pair in keep_pairs]
    right_images = [pair[1] for pair in keep_pairs]
    print(f"\n使用 {len(left_images)} 对高质量图像进行标定")
    # ========== 质量过滤结束 ==========

    # 收集双目对应点
    all_obj_pts, all_img_pts_l, all_img_pts_r, valid_pairs = collect_stereo_points(
        left_images,
        right_images,
        obj_points,
        tag_ids,
        aruco_dict,
        use_multiscale=use_multiscale,
        opencv_refine=opencv_refine,
        board=board,
        K_l=K_l,
        dist_l=dist_l,
        K_r=K_r,
        dist_r=dist_r,
        max_valid_images=MAX_VALID_IMAGES,
    )

    if len(valid_pairs) < 5:
        print(f"\n错误: 有效双目图像对太少 ({len(valid_pairs)} < 5)")
        return

    # 获取图像尺寸
    img = cv2.imread(valid_pairs[0][0])
    image_size = (img.shape[1], img.shape[0])

    # 执行双目标定
    print("\n执行双目标定...")

    ret, K_l_new, dist_l_new, K_r_new, dist_r_new, R, t, E, F = cv2.stereoCalibrate(
        all_obj_pts,
        all_img_pts_l,
        all_img_pts_r,
        K_l,
        dist_l,
        K_r,
        dist_r,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,  # 固定内参，只优化外参
    )

    baseline = np.linalg.norm(t)

    # Step4 的主要误差口径：mean reprojection error（与 step3 保存的 reprojection_error 一致）
    mean_stereo_all, mean_stereo_l, mean_stereo_r, used_stereo = (
        compute_stereo_mean_reproj_error_using_rt(
            all_obj_pts,
            all_img_pts_l,
            all_img_pts_r,
            K_l,
            dist_l,
            K_r,
            dist_r,
            R,
            t,
        )
    )

    print(f"  - 重投影误差(mean, stereo 约束, 左右合计): {mean_stereo_all:.4f} 像素")
    print(f"    - 左目 mean: {mean_stereo_l:.4f} 像素")
    print(f"    - 右目 mean: {mean_stereo_r:.4f} 像素")
    print(f"    - used pairs: {used_stereo}/{len(all_obj_pts)}")
    print(f"  - OpenCV ret (RMS, stereoCalibrate 返回): {ret:.4f} 像素")
    print(f"  - 基线距离: {baseline:.2f} mm ({baseline / 10:.2f} cm)")

    mean_err_right_pnp, per_right_pnp, used_right_pnp = compute_mean_reproj_error_pnp(
        all_obj_pts, all_img_pts_r, K_r, dist_r
    )
    print(
        f"  - 右目 mean_error(PnP, per-view mean): {mean_err_right_pnp:.4f} px "
        f"(used {used_right_pnp}/{len(all_obj_pts)})"
    )

    # （可选）右目 mean_error：使用 stereo 的 R,t（左PnP + R,t 预测右目位姿）
    mean_err_right_rt, per_right_rt, used_right_rt = compute_right_mean_reproj_error_using_rt(
        all_obj_pts,
        all_img_pts_l,
        all_img_pts_r,
        K_l,
        dist_l,
        K_r,
        dist_r,
        R,
        t,
    )
    print(
        f"  - 右目 mean_error(LeftPnP+R,t 预测): {mean_err_right_rt:.4f} px "
        f"(used {used_right_rt}/{len(all_obj_pts)})"
    )

    # 质量评估
    if mean_stereo_all < 1.5:
        print(f"  标定质量: 优秀（mean < 1.5 px）")
    elif mean_stereo_all < 3.0:
        print(f"  标定质量: 良好（mean < 3.0 px）")
    elif mean_stereo_all < 5.0:
        print(f"  标定质量: 一般（mean < 5.0 px）")
    else:
        print(f"  标定质量: 较差（mean ≥ 5.0 px）")
        print(f"  建议: 采集更多高质量图像（每对 ≥20 共同标签）")

    # 保存双目外参
    used_pairs_payload = [
        {
            "left": os.path.relpath(lp).replace("\\", "/"),
            "right": os.path.relpath(rp).replace("\\", "/"),
        }
        for lp, rp in valid_pairs
    ]
    extrinsics = {
        # 元信息：用于 verify 精确复现 Step4 的数据集/筛选口径
        "image_source": str(image_source),
        "quality_filter_min_common_tags": int(MIN_COMMON_TAGS),
        "max_valid_images": int(MAX_VALID_IMAGES),
        "used_pairs": used_pairs_payload,

        "R": R.tolist(),
        "t": t.flatten().tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "baseline": float(baseline),
        # 统一口径：reprojection_error 保存 mean reprojection error（便于与 step3 对比）
        "reprojection_error": float(mean_stereo_all),
        "reprojection_error_left_mean": float(mean_stereo_l),
        "reprojection_error_right_mean": float(mean_stereo_r),
        "reprojection_error_used_pairs": int(used_stereo),
        # 保留 OpenCV 原始返回值（RMS）以供需要时排查
        "opencv_ret_rms": float(ret),

        "mean_reprojection_error_right_pnp": float(mean_err_right_pnp),
        "mean_reprojection_error_right_pnp_used_pairs": int(used_right_pnp),

        "mean_reprojection_error_right_pred_using_rt": float(mean_err_right_rt),
        "mean_reprojection_error_right_pred_using_rt_used_pairs": int(used_right_rt),
    }

    with open("results/stereo_extrinsics.json", "w") as f:
        json.dump(extrinsics, f, indent=2)

    print("  ✓ 已保存: results/stereo_extrinsics.json")


    # 立体校正
    print("\n执行立体校正...")

    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l,
        dist_l,
        K_r,
        dist_r,
        image_size,
        R,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,  # 保留所有像素
    )

    # 保存立体校正参数
    rectification = {
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
        "roi_left": list(roi_l),
        "roi_right": list(roi_r),
    }

    with open("results/stereo_rectification.json", "w") as f:
        json.dump(rectification, f, indent=2)

    print("  ✓ 已保存: results/stereo_rectification.json")

    # 生成立体校正效果图
    print("\n生成立体校正效果图...")

    # 使用第一对图像
    left_img = cv2.imread(valid_pairs[0][0])
    right_img = cv2.imread(valid_pairs[0][1])

    # 计算映射
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        K_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        K_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
    )

    # 应用校正
    left_rect = cv2.remap(left_img, map1_l, map2_l, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, map1_r, map2_r, cv2.INTER_LINEAR)

    # 创建对比图，绘制水平线
    demo = np.hstack([left_rect, right_rect])

    # 绘制水平辅助线（每隔一定距离）
    for y in range(0, demo.shape[0], 50):
        cv2.line(demo, (0, y), (demo.shape[1], y), (0, 255, 0), 1)

    # 添加标题
    cv2.putText(
        demo,
        "Left (Rectified)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        demo,
        "Right (Rectified)",
        (image_size[0] + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        2,
    )

    cv2.imwrite("results/stereo_rectification_demo.jpg", demo)
    print("  ✓ 已保存: results/stereo_rectification_demo.jpg")
    print("  提示: 检查绿色水平线是否对齐，验证校正效果")

    # 显示总结
    print("\n" + "=" * 60)
    print("双目外参标定完成！")
    print("=" * 60)
    print(f"\n双目参数:")
    print(f"  - 基线距离: {baseline:.2f} mm ({baseline / 10:.2f} cm)")
    print(f"  - 重投影误差(mean): {mean_stereo_all:.4f} 像素")
    print(f"  - OpenCV ret (RMS): {ret:.4f} 像素")
    print(f"  - 使用图像对: {len(valid_pairs)}")

    print("\n旋转矩阵 R (左->右, cam1=Left, cam2=Right; X_right = R * X_left + t):")
    print(R)

    print("\n平移向量 t (左->右, 单位:mm):")
    print(t.flatten())

    # 会聚式双目特殊说明
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_matrix(R)
    euler = rot.as_euler("xyz", degrees=True)
    convergence_angle = abs(euler[1])

    print(f"\n会聚式双目几何:")
    print(f"  - 会聚角: {convergence_angle:.2f}°")
    print(f"  - 垂直偏移: {t.flatten()[1]:.2f} mm")
    print(f"  - 前后偏移: {t.flatten()[2]:.2f} mm")

    print("\n质量检查:")
    print("  - 打开 results/stereo_rectification_demo.jpg")
    print("  - 确认左右图像的水平线对齐")
    print("  - 如果对齐良好，说明标定成功")

    print("\n下一步: (可选) 运行 python step5_camera_to_base.py")


if __name__ == "__main__":
    main()
