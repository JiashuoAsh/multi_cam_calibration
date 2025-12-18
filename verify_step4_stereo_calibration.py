#!/usr/bin/env python3
"""
验证双目外参标定质量

方法：
1. 重投影误差手动验证（直接验证标定质量）
2. R (旋转矩阵) 直接验证
3. t (平移向量) 直接验证
4. 深度估计测试（间接验证 R 和 t）
5. 立体校正质量（间接验证 R 和 t）
"""

import cv2
import numpy as np
import json
import glob
import os
from utils import (
    load_config,
    get_aruco_dict,
    detect_apriltag_corners,
    create_apriltag_board,
    create_opencv_aruco_board,
    get_detection_settings,
)


def load_calibration():
    """加载标定结果"""
    # 左相机内参
    with open("results/left_intrinsics.json", "r") as f:
        left_data = json.load(f)
    K_l = np.array(left_data["camera_matrix"])
    dist_l = np.array(left_data["dist_coeffs"])

    # 右相机内参
    with open("results/right_intrinsics.json", "r") as f:
        right_data = json.load(f)
    K_r = np.array(right_data["camera_matrix"])
    dist_r = np.array(right_data["dist_coeffs"])

    # 双目外参
    with open("results/stereo_extrinsics.json", "r") as f:
        stereo_data = json.load(f)
    R = np.array(stereo_data["R"])
    t = np.array(stereo_data["t"]).reshape(3, 1)
    baseline = stereo_data["baseline"]
    # Step4 当前保存的 reprojection_error 口径：mean reprojection error（像素）
    stereo_mean_reprojection_error = stereo_data["reprojection_error"]
    # 可选：保留 OpenCV stereoCalibrate 的原始 ret（RMS）用于排查（若存在）
    opencv_ret_rms = stereo_data.get("opencv_ret_rms")

    # 可选：Step4 记录的本次标定使用的数据集信息（用于精确对齐验证）
    image_source = stereo_data.get("image_source")
    used_pairs = stereo_data.get("used_pairs")

    # 立体校正参数
    with open("results/stereo_rectification.json", "r") as f:
        rect_data = json.load(f)
    Q = np.array(rect_data["Q"])

    return (
        K_l,
        dist_l,
        K_r,
        dist_r,
        R,
        t,
        baseline,
        stereo_mean_reprojection_error,
        opencv_ret_rms,
        Q,
        image_source,
        used_pairs,
    )


def collect_stereo_points_for_verification(
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
    max_valid_images=None,
):
    """
    收集双目图像对中的对应点（用于重投影误差验证）

    与step4中的collect_stereo_points相同的逻辑
    """
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

        # 检测标签
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
        # NOTE: detect_apriltag_corners 返回的 ids 类型在不同路径下可能是 np.ndarray 或 list
        # 这里统一转为 np.ndarray，便于 flatten/where 等操作，同时避免类型检查器误报
        left_ids_flat = np.asarray(left_ids).reshape(-1)
        right_ids_flat = np.asarray(right_ids).reshape(-1)
        common_ids = set(left_ids_flat.tolist()) & set(right_ids_flat.tolist())

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

            if max_valid_images is not None and len(valid_pairs) >= int(max_valid_images):
                break

    return all_obj_pts, all_img_pts_l, all_img_pts_r, valid_pairs


def verify_reprojection_error(
    left_images,
    right_images,
    K_l,
    dist_l,
    K_r,
    dist_r,
    R,
    t,
    reported_mean_error,
    aruco_dict,
    *,
    opencv_ret_rms=None,
    skip_quality_filter: bool = False,
):
    """
    手动验证重投影误差的计算过程（考虑双目外参约束，mean 口径）

    这个函数展示了在双目外参约束下，如何计算 mean reprojection error：
    1. 对每对图像，使用solvePnP求解标定板相对左相机的位姿 (R_l, t_l)
    2. 利用双目外参约束计算右相机位姿: R_r = R × R_l, t_r = R × t_l + t
    3. 使用projectPoints将3D点投影回左右图像
    4. 计算投影点与实际检测点的欧氏距离
    5. 对所有点求 mean（平均欧氏误差）

    参数:
        left_images: 左相机图像路径列表
        right_images: 右相机图像路径列表
        K_l, dist_l: 左相机内参
        K_r, dist_r: 右相机内参
        R, t: 双目外参（右相机相对左相机的旋转和平移）
        reported_mean_error: Step4 保存到 stereo_extrinsics.json 的 mean reprojection error
        aruco_dict: AprilTag字典
    """

    print("\n重投影误差手动验证（考虑双目外参约束）:")
    print("=" * 60)
    print()

    # 加载标定板配置
    config = load_config()
    use_multiscale, opencv_refine = get_detection_settings(config)
    obj_points, tag_ids = create_apriltag_board(config)
    board = create_opencv_aruco_board(obj_points, tag_ids, aruco_dict)

    # 为了与 Step4 完全一致：先做同样的质量过滤 + 数量上限（默认 Step4 为 150 对）
    # 若 main 已经传入了 Step4 的 used_pairs（预选图像对），则应跳过这里的再次过滤。
    try:
        from utils import analyze_stereo_image_quality, filter_low_quality_pairs
    except Exception:
        analyze_stereo_image_quality = None
        filter_low_quality_pairs = None

    if (
        (not skip_quality_filter)
        and analyze_stereo_image_quality is not None
        and filter_low_quality_pairs is not None
    ):
        print("\n对齐 Step4：进行图像对质量过滤...")
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

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

        MIN_COMMON_TAGS = 15
        keep_pairs, _remove_pairs = filter_low_quality_pairs(
            image_quality, min_common_tags=MIN_COMMON_TAGS
        )

        if len(keep_pairs) < 5:
            print(f"  ❌ 高质量图像对太少 ({len(keep_pairs)}), 无法验证")
            return

        # Step4 默认最多使用 150 对
        max_pairs = 150
        left_images = [p[0] for p in keep_pairs][:max_pairs]
        right_images = [p[1] for p in keep_pairs][:max_pairs]
        print(f"  ✓ 过滤后用于验证: {len(left_images)} 对（阈值: {MIN_COMMON_TAGS} 共同标签，最多 {max_pairs} 对）")

    # 收集对应点（与step4相同的逻辑）
    print("收集双目对应点...")
    all_obj_pts, all_img_pts_l, all_img_pts_r, valid_pairs = (
        collect_stereo_points_for_verification(
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
            max_valid_images=150,
        )
    )

    if len(valid_pairs) < 5:
        print(f"  ❌ 有效图像对太少 ({len(valid_pairs)}), 无法验证")
        return

    print(f"  ✓ 收集到 {len(valid_pairs)} 对有效图像")

    # 手动计算重投影误差（考虑双目外参约束）
    print("\n计算重投影误差（mean）...")
    print("  步骤说明:")
    print("    1. 对每对图像，使用 solvePnP 求解标定板相对左相机的位姿 (R_l, t_l)")
    print("    2. 利用双目外参约束计算右相机位姿:")
    print("       R_r = R × R_l  (双目外参 × 左相机旋转)")
    print("       t_r = R × t_l + t  (变换平移)")
    print("    3. 使用 projectPoints 将3D点投影回左右图像")
    print("    4. 计算投影点与检测点的欧氏距离")
    print("    5. 对所有点计算 mean (平均欧氏误差)")
    print()

    all_errors = []
    all_errors_left = []
    all_errors_right = []
    total_points = 0

    for i, (obj_pts, img_pts_l, img_pts_r) in enumerate(
        zip(all_obj_pts, all_img_pts_l, all_img_pts_r)
    ):
        # ===== 左相机重投影误差 =====
        # 1. 使用PnP求解标定板相对左相机的位姿
        success_l, R_vec_l, t_vec_l = cv2.solvePnP(obj_pts, img_pts_l, K_l, dist_l)

        if not success_l:
            continue

        # 2. 将3D点投影回左相机图像
        proj_pts_l, _ = cv2.projectPoints(obj_pts, R_vec_l, t_vec_l, K_l, dist_l)
        proj_pts_l = proj_pts_l.reshape(-1, 2)

        # 3. 计算每个点的欧氏距离
        errors_l = np.linalg.norm(img_pts_l - proj_pts_l, axis=1)
        all_errors.extend(errors_l)
        all_errors_left.extend(errors_l)

        # ===== 右相机重投影误差（使用双目外参约束）=====
        # 将左相机的旋转向量转换为旋转矩阵
        R_mat_l, _ = cv2.Rodrigues(R_vec_l)

        # 利用双目外参约束计算右相机位姿
        # R_r = R × R_l
        R_mat_r = R @ R_mat_l

        # t_r = R × t_l + t
        t_vec_r = R @ t_vec_l + t

        # 将旋转矩阵转回旋转向量
        R_vec_r, _ = cv2.Rodrigues(R_mat_r)

        # 使用约束后的右相机位姿投影
        proj_pts_r, _ = cv2.projectPoints(obj_pts, R_vec_r, t_vec_r, K_r, dist_r)
        proj_pts_r = proj_pts_r.reshape(-1, 2)

        errors_r = np.linalg.norm(img_pts_r - proj_pts_r, axis=1)
        all_errors.extend(errors_r)
        all_errors_right.extend(errors_r)

        total_points += len(img_pts_l) + len(img_pts_r)

    # 4. 计算平均误差（mean）
    all_errors = np.array(all_errors, dtype=np.float64)
    all_errors_left = np.array(all_errors_left, dtype=np.float64)
    all_errors_right = np.array(all_errors_right, dtype=np.float64)
    manual_mean_error = float(np.mean(all_errors))

    # 显示详细统计
    print(f"计算结果:")
    print(f"  总点数: {total_points} 个")
    print(f"  总图像对: {len(valid_pairs)} 对")
    print(f"  平均每对: {total_points / len(valid_pairs):.1f} 个点")
    print()

    print(f"误差分布（左右相机合计）:")
    print(f"  最小误差: {np.min(all_errors):.4f} px")
    print(f"  最大误差: {np.max(all_errors):.4f} px")
    print(f"  平均误差: {np.mean(all_errors):.4f} px")
    print(f"  中位数误差: {np.median(all_errors):.4f} px")
    print(f"  标准差: {np.std(all_errors):.4f} px")
    print()

    print(f"分相机统计:")
    print(f"  左相机平均误差: {np.mean(all_errors_left):.4f} px")
    print(f"  右相机平均误差: {np.mean(all_errors_right):.4f} px")
    print()

    # 对比 Step4 保存的 mean 结果
    print(f"重投影误差对比（mean 口径）:")
    print(f"  Step4保存 (mean):             {reported_mean_error:.4f} px")
    print(f"  手动计算 (mean):              {manual_mean_error:.4f} px")
    print(f"  差异:                         {abs(reported_mean_error - manual_mean_error):.4f} px")
    if opencv_ret_rms is not None:
        try:
            print(f"  参考: OpenCV ret (RMS):       {float(opencv_ret_rms):.4f} px")
        except Exception:
            pass
    print()

    # # 4. 计算RMS（均方根误差）
    # all_errors = np.array(all_errors)
    # all_errors_left = np.array(all_errors_left)
    # all_errors_right = np.array(all_errors_right)
    # manual_rms = np.sqrt(np.mean(all_errors**2))

    # # 显示详细统计
    # print(f"计算结果:")
    # print(f"  总点数: {total_points} 个")
    # print(f"  总图像对: {len(valid_pairs)} 对")
    # print(f"  平均每对: {total_points / len(valid_pairs):.1f} 个点")
    # print()

    # print(f"误差分布（左右相机合计）:")
    # print(f"  最小误差: {np.min(all_errors):.4f} px")
    # print(f"  最大误差: {np.max(all_errors):.4f} px")
    # print(f"  平均误差: {np.mean(all_errors):.4f} px")
    # print(f"  中位数误差: {np.median(all_errors):.4f} px")
    # print(f"  标准差: {np.std(all_errors):.4f} px")
    # print()

    # print(f"分相机统计:")
    # print(f"  左相机平均误差: {np.mean(all_errors_left):.4f} px")
    # print(f"  右相机平均误差: {np.mean(all_errors_right):.4f} px")
    # print(f"  左相机RMS: {np.sqrt(np.mean(all_errors_left**2)):.4f} px")
    # print(f"  右相机RMS: {np.sqrt(np.mean(all_errors_right**2)):.4f} px")
    # print()

    # # 对比OpenCV结果
    # print(f"重投影误差对比:")
    # print(f"  OpenCV返回 (stereoCalibrate): {opencv_error:.4f} px")
    # print(f"  手动计算 (RMS):              {manual_rms:.4f} px")
    # print(f"  差异:                         {abs(opencv_error - manual_rms):.4f} px")
    # print()

    # 解释差异（按 mean 口径）
    denom = float(reported_mean_error) if float(reported_mean_error) != 0.0 else 1.0
    difference_percent = abs(float(reported_mean_error) - manual_mean_error) / denom * 100

    if difference_percent < 1:
        print(f"  ✅ 计算结果高度一致 (差异 < 1%)")
        print(f"  说明: 手动计算正确重现了 Step4 的 mean 误差统计")
        print(f"  验证: 双目外参约束被正确应用")
    elif difference_percent < 5:
        print(f"  ✅ 计算结果基本一致 (差异 < 5%)")
        print(f"  说明: 手动计算与OpenCV算法相符")
        print(f"  验证: 双目外参约束基本正确")
    else:
        print(f"  ⚠️  计算结果有一定差异 (差异 ≥ 5%)")
        print(f"  可能原因:")
        print(f"    1. OpenCV在标定时会同时优化所有参数（R, t, 每对图像的位姿）")
        print(f"    2. 我们这里先固定R,t，再对每对图像独立求解左相机位姿")
        print(f"    3. OpenCV使用全局优化（LM算法），而我们是逐对图像计算")
        print(f"    4. 可能存在内参微调（取决于标定时的flags设置）")

    print()
    print(f"重投影误差的含义:")
    print(f"  • 表示双目标定参数（内参 + 外参R,t）能多精确地将3D点投影回图像")
    print(f"  • 关键约束: 右相机位姿 = 左相机位姿 × 双目外参")
    print(f"  • 数值越小，双目标定质量越高")
    print(f"  • < 1.5 px: 优秀")
    print(f"  • < 3.0 px: 良好")
    print(f"  • < 5.0 px: 一般")
    print(f"  • ≥ 5.0 px: 较差")
    print()
    print(f"与单目标定的区别:")
    print(f"  • 单目: 独立优化每个相机的位姿")
    print(f"  • 双目: 通过外参约束，同时优化左右相机位姿和双目外参R,t")
    print(f"  • 双目误差反映了外参约束的质量")

    # 显示误差直方图统计
    print()
    print(f"误差分布直方图:")
    bins = [0, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
    bin_labels = ["0-0.5", "0.5-1.0", "1.0-2.0", "2.0-3.0", "3.0-5.0", ">5.0"]

    for i in range(len(bins) - 1):
        count = np.sum((all_errors >= bins[i]) & (all_errors < bins[i + 1]))
        percent = count / len(all_errors) * 100
        bar = "█" * int(percent / 2)
        print(f"  {bin_labels[i]:>8} px: {bar} {count:4d} ({percent:5.1f}%)")

    return manual_mean_error


def verify_depth_accuracy(
    left_images, right_images, K_l, dist_l, K_r, dist_r, Q, baseline, aruco_dict
):
    """
    深度/尺度一致性验证（不要求标定板位姿固定）

    这里的“深度正确性”不是指“标定板离相机的绝对深度固定且已知”。
    Step4 的标定图像中标定板位姿确实在变化，这是正常且必要的。

    我们验证的是：
    - 使用标定得到的参数进行三角化（由视差推回 3D）时，
      重建出的标定板几何尺度是否与真实标定板一致。
    - 具体做法：在同一帧中选取“相邻标签中心”的 3D 距离，与真实相邻距离(tag_pitch)对比。

    这样做的好处：
    - 相邻标签中心的真实距离在标定板坐标系里是常数，与板子怎么晃动、怎么旋转无关。
    - 因此不需要固定每张图像的位姿，也不需要知道板子离相机多远。

    注意：对于会聚式双目，需要使用校正后的图像计算视差/深度。
    """
    print("\n深度精度验证（会聚式双目）:")
    print("=" * 60)
    print("⚠️  使用立体校正后的图像进行深度计算")

    # 加载配置获取标定板信息
    config = load_config()
    board_cfg = config["apriltag_board"]
    tags_x = board_cfg["tags_x"]
    tags_y = board_cfg["tags_y"]
    tag_size = board_cfg["tag_size"]
    tag_spacing = board_cfg["tag_spacing"]
    tag_pitch = tag_size + tag_spacing  # 相邻标签中心到中心距离（mm）

    # 生成标定板3D几何（用于确保ID存在且布局一致）
    obj_points, tag_ids = create_apriltag_board(config)
    use_multiscale, opencv_refine = get_detection_settings(config)
    opencv_board = create_opencv_aruco_board(obj_points, tag_ids, aruco_dict)

    print(f"  标定板相邻标签中心距(tag_pitch): {tag_pitch:.2f} mm")
    print("  评估方式: 对每帧，使用可见的相邻标签对(左右/上下)进行3D距离对比")

    # 加载立体校正参数
    with open("results/stereo_rectification.json", "r") as f:
        rect_data = json.load(f)
    R1 = np.array(rect_data["R1"])
    R2 = np.array(rect_data["R2"])
    P1 = np.array(rect_data["P1"])
    P2 = np.array(rect_data["P2"])

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    depth_errors = []
    # 像素域统计：校正后同名点的垂直偏差（理想应接近0）与视差分布
    y_misalign_px = []
    disparities_px = []
    used_pairs_count = 0
    used_frames = 0

    # 只抽样部分帧做快速评估，避免输出过长/运行过慢
    for idx, (left_path, right_path) in enumerate(
        zip(left_images[:25], right_images[:25])
    ):
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            continue

        image_size = (left_img.shape[1], left_img.shape[0])

        # 对会聚式双目，先进行立体校正
        map1_l, map2_l = cv2.initUndistortRectifyMap(
            K_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
        )
        map1_r, map2_r = cv2.initUndistortRectifyMap(
            K_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
        )

        left_rect = cv2.remap(left_img, map1_l, map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, map1_r, map2_r, cv2.INTER_LINEAR)

        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        # 检测标签（校正后的图像不再对应原始畸变模型，因此这里不传入原始 K/dist）
        left_corners, left_ids = detect_apriltag_corners(
            left_gray,
            aruco_dict,
            detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=opencv_board,
        )
        right_corners, right_ids = detect_apriltag_corners(
            right_gray,
            aruco_dict,
            detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=opencv_board,
        )

        if left_ids is None or right_ids is None or len(left_ids) < 2:
            continue

        left_ids_flat = np.asarray(left_ids).reshape(-1)
        right_ids_flat = np.asarray(right_ids).reshape(-1)

        # 找共同标签
        common_ids = sorted(
            list(set(left_ids_flat.tolist()) & set(right_ids_flat.tolist()))
        )
        if len(common_ids) < 2:
            continue

        # 为该帧计算所有共同标签中心的3D位置（由视差推回）
        points_3d = {}

        # 使用校正后的投影矩阵计算3D坐标
        # P1 = [f' 0 cx' 0], P2 = [f' 0 cx' -f'*baseline]
        f_prime = float(P1[0, 0])
        cx_prime = float(P1[0, 2])
        cy_prime = float(P1[1, 2])
        baseline_rect = float(-P2[0, 3] / f_prime)

        for tag_id in common_ids:
            left_idx = np.where(left_ids_flat == tag_id)[0][0]
            right_idx = np.where(right_ids_flat == tag_id)[0][0]

            center_l = left_corners[left_idx].reshape(-1, 2).mean(axis=0)
            center_r = right_corners[right_idx].reshape(-1, 2).mean(axis=0)

            # 像素域：校正后同名点应在同一条水平极线上，因此 y 差可作为“像素误差”统计
            y_diff = float(abs(center_l[1] - center_r[1]))
            y_misalign_px.append(y_diff)

            disparity = float(center_l[0] - center_r[0])
            disparities_px.append(disparity)
            if disparity <= 0:
                continue

            Z = f_prime * baseline_rect / disparity
            X = (float(center_l[0]) - cx_prime) * Z / f_prime
            Y = (float(center_l[1]) - cy_prime) * Z / f_prime
            points_3d[int(tag_id)] = np.array([X, Y, Z], dtype=np.float64)

        if len(points_3d) < 2:
            continue

        # 仅使用相邻标签对（水平/垂直）进行评估：真实距离恒为 tag_pitch
        frame_errors = []
        for tag_id in points_3d.keys():
            # 水平相邻：id+1（同一行）
            if (tag_id % tags_x) != (tags_x - 1):
                nb = tag_id + 1
                if nb in points_3d:
                    measured_dist = float(
                        np.linalg.norm(points_3d[tag_id] - points_3d[nb])
                    )
                    error_mm = abs(measured_dist - tag_pitch)
                    error_percent = (error_mm / tag_pitch) * 100
                    frame_errors.append(error_percent)
                    used_pairs_count += 1

            # 垂直相邻：id+tags_x
            nb = tag_id + tags_x
            if nb < tags_x * tags_y and nb in points_3d:
                measured_dist = float(np.linalg.norm(points_3d[tag_id] - points_3d[nb]))
                error_mm = abs(measured_dist - tag_pitch)
                error_percent = (error_mm / tag_pitch) * 100
                frame_errors.append(error_percent)
                used_pairs_count += 1

        if not frame_errors:
            continue

        used_frames += 1
        depth_errors.extend(frame_errors)

        # 只打印少量样例，避免刷屏
        if used_frames <= 5:
            avg_depth = float(np.mean([p[2] for p in points_3d.values()]))
            print(f"\n图像对 {idx + 1}（样例输出）:")
            print(f"  可用标签数: {len(points_3d)}")
            print(f"  平均深度(参考): {avg_depth:.1f} mm")
            print(f"  本帧相邻对数量: {len(frame_errors)}")
            print(f"  本帧平均误差: {np.mean(frame_errors):.2f}%")

    if depth_errors:
        print(f"\n深度测量统计:")
        print(
            f"  使用帧数: {used_frames} / {min(25, len(left_images), len(right_images))}"
        )
        print(f"  使用相邻对数量: {used_pairs_count}")
        print(f"  平均误差: {np.mean(depth_errors):.2f}%")
        print(f"  中位数误差: {np.median(depth_errors):.2f}%")
        print(f"  最大误差: {np.max(depth_errors):.2f}%")
        print(f"  误差标准差: {np.std(depth_errors):.2f}%")

        # === 像素域统计输出 ===
        if y_misalign_px:
            y_misalign_px_arr = np.asarray(y_misalign_px, dtype=np.float64)
            print(f"\n像素误差统计（校正后同名点垂直偏差 |yL - yR|）:")
            print(f"  样本数: {len(y_misalign_px_arr)}")
            print(f"  平均: {np.mean(y_misalign_px_arr):.3f} px")
            print(f"  中位数: {np.median(y_misalign_px_arr):.3f} px")
            print(f"  最大: {np.max(y_misalign_px_arr):.3f} px")
            print(f"  标准差: {np.std(y_misalign_px_arr):.3f} px")

        if disparities_px:
            disp_arr = np.asarray(disparities_px, dtype=np.float64)
            disp_pos = disp_arr[disp_arr > 0]
            print(f"\n视差统计（校正后 x 方向 disparity = xL - xR）:")
            print(f"  样本数(全部): {len(disp_arr)}")
            print(f"  正视差样本数: {len(disp_pos)}")
            if len(disp_pos) > 0:
                print(f"  正视差-平均: {np.mean(disp_pos):.3f} px")
                print(f"  正视差-中位数: {np.median(disp_pos):.3f} px")
                print(
                    f"  正视差-最小/最大: {np.min(disp_pos):.3f} / {np.max(disp_pos):.3f} px"
                )

        if np.mean(depth_errors) < 5:
            print(f"  ✅ 深度精度优秀 (< 5%)")
        elif np.mean(depth_errors) < 10:
            print(f"  ⚠️  深度精度良好 (< 10%)")
        else:
            print(f"  ❌ 深度精度较差 (≥ 10%)")
    else:
        print("  未能进行深度验证（图像质量不足）")


def verify_rectification_quality(
    left_images, right_images, K_l, dist_l, K_r, dist_r, R, t
):
    """
    验证立体校正质量

    检查校正后的图像是否满足极线对齐
    """
    print("\n立体校正质量验证:")
    print("=" * 60)

    # 加载立体校正参数
    with open("results/stereo_rectification.json", "r") as f:
        rect_data = json.load(f)
    R1 = np.array(rect_data["R1"])
    R2 = np.array(rect_data["R2"])
    P1 = np.array(rect_data["P1"])
    P2 = np.array(rect_data["P2"])

    # 使用第一对“可读”的图像
    left_img = None
    right_img = None
    for lp, rp in zip(left_images, right_images):
        li = cv2.imread(lp)
        ri = cv2.imread(rp)
        if li is not None and ri is not None:
            left_img = li
            right_img = ri
            break

    if left_img is None or right_img is None:
        print("  ❌ 未找到可读取的图像对，无法验证立体校正质量")
        return

    image_size = (left_img.shape[1], left_img.shape[0])

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

    # 检测特征点并计算垂直偏差
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    config = load_config()
    use_multiscale, opencv_refine = get_detection_settings(config)
    aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])
    obj_points_mm, tag_ids = create_apriltag_board(config)
    opencv_board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)

    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    left_corners, left_ids = detect_apriltag_corners(
        left_gray,
        aruco_dict,
        detector_params,
        use_multiscale=use_multiscale,
        opencv_refine=opencv_refine,
        board=opencv_board,
    )
    right_corners, right_ids = detect_apriltag_corners(
        right_gray,
        aruco_dict,
        detector_params,
        use_multiscale=use_multiscale,
        opencv_refine=opencv_refine,
        board=opencv_board,
    )

    if left_ids is not None and right_ids is not None:
        left_ids_flat = np.asarray(left_ids).reshape(-1)
        right_ids_flat = np.asarray(right_ids).reshape(-1)

        # 找共同标签
        common_ids = set(left_ids_flat.tolist()) & set(right_ids_flat.tolist())

        y_diffs = []
        for tag_id in common_ids:
            left_idx = np.where(left_ids_flat == tag_id)[0][0]
            right_idx = np.where(right_ids_flat == tag_id)[0][0]

            # 标签中心的y坐标差异
            center_l = left_corners[left_idx].reshape(-1, 2).mean(axis=0)
            center_r = right_corners[right_idx].reshape(-1, 2).mean(axis=0)

            y_diff = abs(center_l[1] - center_r[1])
            y_diffs.append(y_diff)

        if y_diffs:
            print(f"  对应点垂直偏差统计:")
            print(f"    平均偏差: {np.mean(y_diffs):.2f} 像素")
            print(f"    最大偏差: {np.max(y_diffs):.2f} 像素")
            print(f"    标准差: {np.std(y_diffs):.2f} 像素")

            if np.mean(y_diffs) < 1.0:
                print(f"    ✅ 校正质量优秀 (< 1.0 px)")
            elif np.mean(y_diffs) < 2.0:
                print(f"    ⚠️  校正质量良好 (< 2.0 px)")
            else:
                print(f"    ❌ 校正质量较差 (≥ 2.0 px)")


def main():
    """主函数"""
    print("=" * 60)
    print("双目外参标定质量验证")
    print("=" * 60)

    # 加载标定结果
    try:
        (
            K_l,
            dist_l,
            K_r,
            dist_r,
            R,
            t,
            baseline,
            stereo_mean_error,
            opencv_ret_rms,
            Q,
            image_source_from_step4,
            used_pairs_from_step4,
        ) = load_calibration()
    except FileNotFoundError as e:
        print(f"\n错误: 缺少标定文件")
        print(f"请先运行 python step4_stereo_extrinsic.py")
        return

    print(f"\n标定参数概览:")
    print(f"  基线距离: {baseline:.2f} mm")
    print(f"  焦距（左）: {K_l[0, 0]:.2f} px")
    print(f"  焦距（右）: {K_r[0, 0]:.2f} px")
    print(f"  Step4保存重投影误差(mean): {float(stereo_mean_error):.4f} px")
    if opencv_ret_rms is not None:
        try:
            print(f"  参考: OpenCV ret (RMS):       {float(opencv_ret_rms):.4f} px")
        except Exception:
            pass

    def _has_readable_pair(lps, rps, *, max_check: int = 5) -> bool:
        checked = 0
        for lp, rp in zip(lps, rps):
            li = cv2.imread(lp)
            ri = cv2.imread(rp)
            checked += 1
            if li is not None and ri is not None:
                return True
            if checked >= max_check:
                break
        return False

    # 优先使用 Step4 记录的 used_pairs（最可靠：保证样本集合一致）
    left_images = []
    right_images = []
    image_source = None
    using_step4_used_pairs = False

    if isinstance(used_pairs_from_step4, list) and len(used_pairs_from_step4) > 0:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        for pair in used_pairs_from_step4:
            if not isinstance(pair, dict):
                continue
            lp = pair.get("left")
            rp = pair.get("right")
            if not lp or not rp:
                continue

            # 优先按 Step4 写入的相对路径解析；若当前工作目录不同，则尝试相对脚本目录解析
            lp_try = lp
            rp_try = rp
            if not os.path.exists(lp_try):
                lp_try = os.path.join(base_dir, lp)
            if not os.path.exists(rp_try):
                rp_try = os.path.join(base_dir, rp)

            left_images.append(lp_try)
            right_images.append(rp_try)

        # 只要能找到可读的一对，就认为可用
        if len(left_images) > 0 and len(right_images) > 0 and _has_readable_pair(left_images, right_images, max_check=10):
            using_step4_used_pairs = True
            image_source = str(image_source_from_step4) if image_source_from_step4 is not None else "(unknown)"
            print(f"\n使用 Step4 记录的 used_pairs 进行验证（image_source={image_source}）")

    # 回退：按目录扫描（优先 filtered；若为空或不可读则回退到 raw）
    if not using_step4_used_pairs:
        left_images = sorted(glob.glob("images/filtered/left/*.png"))
        right_images = sorted(glob.glob("images/filtered/right/*.png"))
        image_source = "filtered"
        if len(left_images) == 0 or len(right_images) == 0 or not _has_readable_pair(left_images, right_images):
            left_images = sorted(glob.glob("images/raw/left/*.png"))
            right_images = sorted(glob.glob("images/raw/right/*.png"))
            image_source = "raw"

        if len(left_images) == 0 or len(right_images) == 0:
            print("\n错误: 未找到图像（images/filtered 与 images/raw 均为空）")
            return

        print(f"\n使用图像来源: images/{image_source}/")

    # 加载配置
    config = load_config()
    aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])

    # 【新增】重投影误差手动验证（直接验证标定质量）
    verify_reprojection_error(
        left_images,
        right_images,
        K_l,
        dist_l,
        K_r,
        dist_r,
        R,
        t,
        float(stereo_mean_error),
        aruco_dict,
        opencv_ret_rms=opencv_ret_rms,
        skip_quality_filter=using_step4_used_pairs,
    )

    # 深度精度验证（间接验证 R 和 t）
    verify_depth_accuracy(
        left_images, right_images, K_l, dist_l, K_r, dist_r, Q, baseline, aruco_dict
    )

    # 立体校正质量（间接验证 R 和 t）
    verify_rectification_quality(
        left_images, right_images, K_l, dist_l, K_r, dist_r, R, t
    )

    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
