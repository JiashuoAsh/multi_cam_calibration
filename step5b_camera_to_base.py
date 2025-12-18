#!/usr/bin/env python3
"""
Step 5b: 相机到底盘坐标系标定 - AprilTag 标定板（精简版）

计算相机坐标系到机器人底盘坐标系的变换矩阵。

坐标系定义:
    底盘坐标系: X右 Y上 Z前
    相机坐标系: OpenCV标准 (X右 Y下 Z前)

输入:
    - images/step5/left/*.png (step5a 采集的图像)
    - results/left_intrinsics.json
    - results/stereo_extrinsics.json
    - config/apriltag_config.json (board_to_base_transform)

输出:
    - results/camera_to_base.json (B_T_Cl, B_T_Cr)
"""

import argparse
import json
import os
import glob
import traceback
from datetime import datetime
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation
from utils import (
    load_config,
    get_aruco_dict,
    detect_apriltag_corners,
    create_apriltag_board,
    create_opencv_aruco_board,
    get_detection_settings,
    estimate_pose_apriltag,
)


def _fmt4(x) -> str:
    """将向量/矩阵展平后以4位小数格式化为字符串。"""
    arr = np.asarray(x, dtype=np.float64)
    return "[" + ", ".join(f"{v:.4f}" for v in arr.reshape(-1)) + "]"


def _pretty_mat(name: str, T: np.ndarray, indent: str = "  ") -> None:
    """用4位小数打印矩阵。"""
    T = np.asarray(T, dtype=np.float64)
    s = np.array2string(
        T,
        formatter={"float_kind": lambda v: f"{float(v): .4f}"},
        suppress_small=False,
    )
    print(f"{indent}{name} =\n{indent}{s.replace(chr(10), chr(10) + indent)}")


def _ensure_transform(T: np.ndarray, name: str) -> None:
    """检查4x4齐次变换矩阵格式。"""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: 期望 (4,4)，得到 {T.shape}")

    bottom = T[3, :]
    bottom_target = np.array([0.0, 0.0, 0.0, 1.0])
    bottom_err = float(np.linalg.norm(bottom - bottom_target))
    if bottom_err > 1e-6:
        raise ValueError(f"{name}: 底行应为 [0 0 0 1]，当前 {bottom}")


def _make_transform(R: np.ndarray, t: np.ndarray, name: str) -> np.ndarray:
    """构造4x4齐次变换矩阵。"""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t
    _ensure_transform(T, name)
    return T


def _invert_transform_bak(
    T: np.ndarray, name: str, *, strict: bool = True
) -> np.ndarray:
    """求逆并自检：T * inv(T) 是否接近 I。"""
    T = np.asarray(T, dtype=np.float64)
    _ensure_transform(T, name + ".input")
    T_inv = np.linalg.inv(T)
    _ensure_transform(T_inv, name)
    I1 = T @ T_inv
    I2 = T_inv @ T
    err1 = float(np.linalg.norm(I1 - np.eye(4), ord="fro"))
    err2 = float(np.linalg.norm(I2 - np.eye(4), ord="fro"))
    print(
        f"  [自检] {name}: inv_err_fro(T*Tinv)={err1:.3e}, inv_err_fro(Tinv*T)={err2:.3e}"
    )
    if strict and (err1 > 1e-6 or err2 > 1e-6):
        raise ValueError(f"{name}: 求逆自检失败，误差过大 ({err1}, {err2})")
    return T_inv


def _invert_transform(T: np.ndarray, name: str, *, strict: bool = True):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv_analytic = R.T
    t_inv_analytic = -R_inv_analytic @ t

    T_inv = np.linalg.inv(T)
    R_inv_num = T_inv[:3, :3]
    t_inv_num = T_inv[:3, 3]

    print("  diff_R = ", np.linalg.norm(R_inv_num - R_inv_analytic))
    print("  diff_t = ", np.linalg.norm(t_inv_num - t_inv_analytic))
    return _make_transform(R_inv_analytic, t_inv_analytic, name)


def _euler_to_rotation_matrix(roll, pitch, yaw, degrees=True):
    """欧拉角转旋转矩阵（XYZ内旋）。"""
    R = Rotation.from_euler("XYZ", [roll, pitch, yaw], degrees=degrees).as_matrix()
    return np.asarray(R, dtype=np.float64)


def _tf_point(T_4x4: np.ndarray, p_3: np.ndarray) -> np.ndarray:
    """用4x4变换矩阵变换3D点。"""
    p_3 = np.asarray(p_3, dtype=np.float64).reshape(3, 1)
    p_h = np.vstack([p_3, [[1.0]]])
    result = T_4x4 @ p_h
    return result[:3, 0]


def _get_prominent_tag_ids(
    tags_x: int, tags_y: int, tag_centers: Dict[int, np.ndarray]
) -> List[int]:
    """获取显眼的tag ID（四角+中心）。"""
    candidate_ids = [
        0,
        tags_x - 1,
        (tags_y - 1) * tags_x,
        tags_x * tags_y - 1,
        (tags_y // 2) * tags_x + (tags_x // 2),
    ]
    return [tid for tid in candidate_ids if tid in tag_centers]


def process_images_and_estimate_pose(
    image_paths: List[str],
    aruco_dict,
    detector_params,
    obj_points: np.ndarray,
    tag_ids: List[int],
    K: np.ndarray,
    dist: np.ndarray,
    *,
    camera_name: str,
    use_multiscale: bool = True,
    opencv_refine: bool = False,
    board=None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """处理图像并估计位姿，返回 (rvec, tvec) 列表。"""
    valid_poses = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_apriltag_corners(
            gray,
            aruco_dict,
            detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            camera_matrix=K,
            dist_coeffs=dist,
        )
        success, rvec, tvec = estimate_pose_apriltag(
            corners, ids, obj_points, tag_ids, K, dist
        )

        if success:
            valid_poses.append((rvec, tvec))

    print(f"  - {camera_name}: 有效位姿: {len(valid_poses)}/{len(image_paths)}")
    return valid_poses


def load_intrinsics(json_path):
    """加载相机内参。"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data["camera_matrix"]), np.array(data["dist_coeffs"])


def load_stereo_extrinsics(json_path):
    """加载双目外参 (统一转换到米)。"""
    with open(json_path, "r") as f:
        data = json.load(f)

    R = np.array(data["R"], dtype=np.float64)
    t = np.array(data["t"], dtype=np.float64).reshape(3, 1)
    # mm->m
    t = t / 1000.0

    # 构造 Cr_T_Cl (左相机到右相机的变换)
    Cr_T_Cl = _make_transform(R, t, "Cr_T_Cl")
    _pretty_mat("Cr_T_Cl", Cr_T_Cl)
    return Cr_T_Cl


def load_calibration_data():
    """加载所有标定数据和配置。"""
    print("\n加载标定数据...")

    # 检查必要文件
    required_files = [
        "results/left_intrinsics.json",
        "results/right_intrinsics.json",
        "results/stereo_extrinsics.json",
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"未找到 {file}")

    # 加载配置
    config = load_config()
    use_multiscale, opencv_refine = get_detection_settings(config)
    board_cfg = config["apriltag_board"]
    transform_cfg = config["board_to_base_transform"]

    print(f"\n标定板到底盘的变换:")
    print(f"  - 平移 (m): {transform_cfg['translation']}")
    print(f"  - 旋转 (度): {transform_cfg['rotation_euler_deg']}")

    # 加载相机内参和双目外参
    K_l, dist_l = load_intrinsics("results/left_intrinsics.json")
    K_r, dist_r = load_intrinsics("results/right_intrinsics.json")
    Cr_T_Cl = load_stereo_extrinsics("results/stereo_extrinsics.json")

    # 创建AprilTag标定板
    obj_points_mm, tag_ids = create_apriltag_board(config)
    obj_points = obj_points_mm.astype(np.float64) / 1000.0
    aruco_dict = get_aruco_dict(board_cfg["family"])

    board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)

    return {
        "config": config,
        "board_cfg": board_cfg,
        "transform_cfg": transform_cfg,
        "K_l": K_l,
        "dist_l": dist_l,
        "K_r": K_r,
        "dist_r": dist_r,
        "Cr_T_Cl": Cr_T_Cl,
        "obj_points": obj_points,
        "tag_ids": tag_ids,
        "aruco_dict": aruco_dict,
        "opencv_board": board,
        "use_multiscale": use_multiscale,
        "opencv_refine": opencv_refine,
    }


def compute_C_T_T_mean_pose(valid_rvecs, valid_tvecs, *, cam_name: str) -> np.ndarray:
    """由多帧PnP结果计算平均位姿，输出 {cam_name}_T_T (T -> cam)。"""
    print(f"  计算 {cam_name}_T_T 平均位姿（{len(valid_rvecs)} 帧）")

    if len(valid_rvecs) == 0:
        raise ValueError(f"{cam_name}: 没有有效位姿")

    if len(valid_rvecs) > 1:
        t_std = np.std(valid_tvecs, axis=0)
        t_std_norm = np.linalg.norm(t_std)

        # 简单估算旋转的稳定性
        r_std = np.std(valid_rvecs, axis=0)
        r_std_norm = np.linalg.norm(r_std)

        print(f"  [稳定性检查] 平移标准差: {t_std_norm*1000:.2f} mm (越小越好)")
        print(f"  [稳定性检查] 旋转标准差: {r_std_norm:.4f} rad")

        if t_std_norm > 0.005: # 阈值 5mm
            print(f"  ⚠️ 警告: 位姿抖动较大 (>5mm)，建议增加光照或检查标定板是否晃动")
        else:
            print(f"  ✓ 位姿稳定，单位置标定可靠")

    # 对旋转向量取平均（转为旋转矩阵后平均）
    R_matrices = [cv2.Rodrigues(rvec)[0] for rvec in valid_rvecs]
    R_mean = np.mean(R_matrices, axis=0)

    # SVD正交化，确保是有效旋转
    U, _, Vt = np.linalg.svd(R_mean)
    R_mean = U @ Vt

    # 避免反射(det=-1)
    if float(np.linalg.det(R_mean)) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt

    t_mean = np.mean(valid_tvecs, axis=0)
    C_T_T = _make_transform(R_mean, t_mean, f"{cam_name}_T_T")
    _pretty_mat(f"{cam_name}_T_T", C_T_T)
    return C_T_T


def build_B_T_T_from_config(
    transform_cfg: Dict[str, Any], board_cfg: Dict[str, Any]
) -> np.ndarray:
    """由配置构造 B_T_T (T -> B)。"""
    print("\n[求解] 构造标定板到底盘的变换 (B_T_T: T -> B)")

    rotation_euler = transform_cfg["rotation_euler_deg"]
    R_B_T = _euler_to_rotation_matrix(*rotation_euler, degrees=True)

    translation_ref = transform_cfg.get("translation_reference", "tag0_center")
    translation_input_B = np.array(
        transform_cfg["translation"], dtype=np.float64
    ).reshape(3, 1)

    tag_pitch_m = (board_cfg["tag_size"] + board_cfg["tag_spacing"]) / 1000.0
    center_T = np.array(
        [
            (board_cfg["tags_x"] - 1) * tag_pitch_m / 2.0,
            (board_cfg["tags_y"] - 1) * tag_pitch_m / 2.0,
            0.0,
        ],
        dtype=np.float64,
    ).reshape(3, 1)

    if translation_ref in ("board_center", "grid_center", "board_center_tag_grid"):
        # center_B = R_B_T * center_T + origin_B
        # origin_B = center_B - R_B_T * center_T
        translation_B = translation_input_B - (R_B_T @ center_T)
        print(f"  - 将板中心位置换算为Tag0位置")
    else:
        translation_B = translation_input_B
        print(f"  - 直接使用translation作为Tag0位置")

    B_T_T = _make_transform(R_B_T, translation_B, "B_T_T")
    _pretty_mat("B_T_T", B_T_T)
    return B_T_T


def compute_B_T_Cl(B_T_T: np.ndarray, Cl_T_T: np.ndarray) -> np.ndarray:
    """主链：B_T_Cl = B_T_T @ inv(Cl_T_T)。"""
    print("\n[求解] 计算左相机到底盘变换 (B_T_Cl: Cl -> B)")
    T_T_Cl = _invert_transform(Cl_T_T, "T_T_Cl")
    T_T_B = _invert_transform(B_T_T, "B_T_T")
    _pretty_mat("Cl_T_T", Cl_T_T)
    _pretty_mat("T_T_Cl", T_T_Cl)
    _pretty_mat("B_T_T", B_T_T)
    _pretty_mat("T_T_B", T_T_B)
    B_T_Cl = B_T_T @ T_T_Cl
    _ensure_transform(B_T_Cl, "B_T_Cl")
    _pretty_mat("B_T_Cl", B_T_Cl)
    return B_T_Cl


def compute_B_T_Cr(B_T_Cl: np.ndarray, Cr_T_Cl: np.ndarray) -> np.ndarray:
    """右相机：B_T_Cr = B_T_Cl @ inv(Cr_T_Cl)。"""
    print("\n[求解] 计算右相机到底盘变换 (B_T_Cr: Cr -> B)")
    Cl_T_Cr = _invert_transform(Cr_T_Cl, "Cl_T_Cr")
    B_T_Cr = B_T_Cl @ Cl_T_Cr
    _ensure_transform(B_T_Cr, "B_T_Cr")
    _pretty_mat("B_T_Cr", B_T_Cr)
    return B_T_Cr


def validate_stereo_consistency(
    *,
    Cl_T_T: np.ndarray,
    Cr_T_T: np.ndarray,
    Cr_T_Cl: np.ndarray,
    obj_points: np.ndarray,
    tag_ids: List[int],
    board_cfg: Dict[str, Any],
    K_l: Optional[np.ndarray] = None,
    dist_l: Optional[np.ndarray] = None,
    K_r: Optional[np.ndarray] = None,
    dist_r: Optional[np.ndarray] = None,
    strict: bool = True,
) -> bool:
    """校验双目外参与左右相机位姿的一致性。"""
    print("\n" + "-" * 50)
    print("[双目外参一致性校验]")

    # 计算预测的右相机位姿：Cr_T_T_pred = Cr_T_Cl @ Cl_T_T
    Cr_T_T_pred = Cr_T_Cl @ Cl_T_T
    Cl_T_Cr = _invert_transform(Cr_T_Cl, "Cl_T_Cr", strict=False)

    # 计算位姿差异
    Delta_T = Cr_T_T @ _invert_transform(Cr_T_T_pred, "Cr_T_T_pred_inv")

    # 旋转误差（度）
    R_error = Delta_T[:3, :3]
    rot_error_deg = float(Rotation.from_matrix(R_error).magnitude() * 180.0 / np.pi)
    # 平移误差（米）
    trans_error_m = float(np.linalg.norm(Delta_T[:3, 3]))

    print(f"位姿误差分析:")
    print(f"  旋转误差: {rot_error_deg:.3f}°")
    print(f"  平移误差: {trans_error_m:.4f}m")

    # 几何验证：选择几个显眼Tag进行3D点验证
    tag_centers_T = {
        int(tag_ids[i]): obj_points[i].mean(axis=0).reshape(3)
        for i in range(len(tag_ids))
    }
    picked_ids = _get_prominent_tag_ids(
        board_cfg["tags_x"], board_cfg["tags_y"], tag_centers_T
    )

    print(f"\n几何一致性验证（{len(picked_ids)}个关键Tag）:")
    max_point_error = 0.0
    max_pixel_error_l = 0.0
    max_pixel_error_r = 0.0
    pixel_error_available = (
        K_l is not None
        and K_r is not None
        and dist_l is not None
        and dist_r is not None
    )

    for tid in picked_ids[:3]:  # 只检查前3个关键点
        p_T = tag_centers_T[tid]

        # 右相机中的3D点（两种方式计算）
        p_Cr_measured = _tf_point(Cr_T_T, p_T)
        p_Cr_predicted = _tf_point(Cr_T_T_pred, p_T)

        # 3D点误差
        point_error = np.linalg.norm(p_Cr_measured - p_Cr_predicted)
        max_point_error = max(max_point_error, point_error)

        print(f"  Tag{tid}: 3D点误差 {point_error:.4f}m", end="")

        # 像素重投影误差（如果有相机内参）
        if pixel_error_available:
            # 左相机像素误差
            p_Cl_measured = _tf_point(Cl_T_T, p_T)
            p_Cl_predicted = _tf_point(
                Cl_T_Cr @ Cr_T_T_pred, p_T
            )  # 通过右相机预测的左相机位置

            if p_Cl_measured[2] > 0.1 and p_Cl_predicted[2] > 0.1:  # Z > 0.1m 才有效
                # 重投影到左相机图像平面
                if K_l is not None and dist_l is not None:
                    pixel_measured_l, _ = cv2.projectPoints(
                        p_Cl_measured.reshape(1, 1, 3),
                        np.zeros(3),
                        np.zeros(3),
                        K_l,
                        dist_l,
                    )
                    pixel_predicted_l, _ = cv2.projectPoints(
                        p_Cl_predicted.reshape(1, 1, 3),
                        np.zeros(3),
                        np.zeros(3),
                        K_l,
                        dist_l,
                    )
                else:
                    pixel_error_l = float("nan")
                    continue

                pixel_error_l = np.linalg.norm(
                    pixel_measured_l[0, 0] - pixel_predicted_l[0, 0]
                )
                max_pixel_error_l = max(max_pixel_error_l, pixel_error_l)
            else:
                pixel_error_l = float("nan")

            # 右相机像素误差
            if p_Cr_measured[2] > 0.1 and p_Cr_predicted[2] > 0.1:  # Z > 0.1m 才有效
                # 重投影到右相机图像平面
                if K_r is not None and dist_r is not None:
                    pixel_measured_r, _ = cv2.projectPoints(
                        p_Cr_measured.reshape(1, 1, 3),
                        np.zeros(3),
                        np.zeros(3),
                        K_r,
                        dist_r,
                    )
                    pixel_predicted_r, _ = cv2.projectPoints(
                        p_Cr_predicted.reshape(1, 1, 3),
                        np.zeros(3),
                        np.zeros(3),
                        K_r,
                        dist_r,
                    )
                else:
                    pixel_error_r = float("nan")
                    continue

                pixel_error_r = np.linalg.norm(
                    pixel_measured_r[0, 0] - pixel_predicted_r[0, 0]
                )
                max_pixel_error_r = max(max_pixel_error_r, pixel_error_r)
            else:
                pixel_error_r = float("nan")

            print(
                f", 左像素误差 {pixel_error_l:.2f}px, 右像素误差 {pixel_error_r:.2f}px"
            )
        else:
            print()  # 换行

    print(f"  最大3D点误差: {max_point_error:.4f}m")
    if pixel_error_available:
        print(f"  最大左像素误差: {max_pixel_error_l:.2f}px")
        print(f"  最大右像素误差: {max_pixel_error_r:.2f}px")

    # 判断是否通过校验
    rot_threshold = 3.0  # 度
    trans_threshold = 0.05  # 米
    point_threshold = 0.05  # 米
    pixel_threshold = 5.0  # 像素

    passed = bool(
        rot_error_deg < rot_threshold
        and trans_error_m < trans_threshold
        and max_point_error < point_threshold
    )

    # 如果有像素误差数据，也要满足像素阈值
    if (
        pixel_error_available
        and not np.isnan(max_pixel_error_l)
        and not np.isnan(max_pixel_error_r)
    ):
        pixel_passed = bool(
            max_pixel_error_l < pixel_threshold and max_pixel_error_r < pixel_threshold
        )
        passed = passed and pixel_passed
        print(
            f"\n校验阈值: 旋转<{rot_threshold}°, 平移<{trans_threshold}m, 3D点<{point_threshold}m, 像素<{pixel_threshold}px"
        )
    else:
        print(
            f"\n校验阈值: 旋转<{rot_threshold}°, 平移<{trans_threshold}m, 3D点<{point_threshold}m"
        )

    if passed:
        print("✓ 双目外参一致性校验通过")
    else:
        print("⚠ 双目外参一致性校验失败")
        print("  未通过的指标:")
        if rot_error_deg >= rot_threshold:
            print(f"    - 旋转误差: {rot_error_deg:.3f}° (阈值: {rot_threshold}°)")
        if trans_error_m >= trans_threshold:
            print(f"    - 平移误差: {trans_error_m:.4f}m (阈值: {trans_threshold}m)")
        if max_point_error >= point_threshold:
            print(f"    - 3D点误差: {max_point_error:.4f}m (阈值: {point_threshold}m)")
        if (
            pixel_error_available
            and not np.isnan(max_pixel_error_l)
            and not np.isnan(max_pixel_error_r)
        ):
            if max_pixel_error_l >= pixel_threshold:
                print(
                    f"    - 左像素误差: {max_pixel_error_l:.2f}px (阈值: {pixel_threshold}px)"
                )
            if max_pixel_error_r >= pixel_threshold:
                print(
                    f"    - 右像素误差: {max_pixel_error_r:.2f}px (阈值: {pixel_threshold}px)"
                )

    print("-" * 50)
    return passed


def print_tag_coordinates(
    *,
    B_T_T: np.ndarray,
    Cl_T_T: np.ndarray,
    B_T_Cl: np.ndarray,
    obj_points: np.ndarray,
    tag_ids: List[int],
    board_cfg: Dict[str, Any],
) -> None:
    """打印显眼Tag中心点的链式坐标变换，每一步都可验证。"""
    print("\n" + "-" * 60)
    print("[显眼Tag中心点坐标 - 链式变换验证]")

    # 计算tag中心点
    tag_centers_T = {
        int(tag_ids[i]): obj_points[i].mean(axis=0).reshape(3)
        for i in range(len(tag_ids))
    }
    picked_ids = _get_prominent_tag_ids(
        board_cfg["tags_x"], board_cfg["tags_y"], tag_centers_T
    )

    print("变换链: T -> B -> Cl")
    print("验证: B_T_Cl @ Cl_T_T 应等于 B_T_T")
    print()

    for tid in picked_ids:
        p_T = tag_centers_T[tid]

        # 直接变换
        p_B_direct = _tf_point(B_T_T, p_T)
        p_Cl_direct = _tf_point(Cl_T_T, p_T)

        # 链式变换: T -> Cl -> B
        p_Cl_from_T = _tf_point(Cl_T_T, p_T)
        p_B_from_Cl = _tf_point(B_T_Cl, p_Cl_from_T)

        # 验证链式一致性
        chain_error = np.linalg.norm(p_B_direct - p_B_from_Cl)

        print(f"Tag {tid:>2d}:")
        print(f"  T坐标:     {_fmt4(p_T)}")
        print(f"  B坐标(直接): {_fmt4(p_B_direct)}")
        print(f"  B坐标(链式): {_fmt4(p_B_from_Cl)}")
        print(f"  Cl坐标:    {_fmt4(p_Cl_direct)}")
        print(f"  链式误差:   {chain_error:.6f}m")

        if chain_error > 1e-10:
            print(f"    ⚠ 链式验证失败!")
        else:
            print(f"    ✓ 链式验证通过")
        print()

    # 整体变换矩阵验证
    print("变换矩阵链式验证:")
    B_T_T_computed = B_T_Cl @ Cl_T_T
    matrix_diff = np.linalg.norm(B_T_T - B_T_T_computed, ord="fro")
    print(f"  ||B_T_T - (B_T_Cl @ Cl_T_T)||_F = {matrix_diff:.2e}")

    if matrix_diff < 1e-10:
        print("  ✓ 变换矩阵链式验证通过")
    else:
        print("  ⚠ 变换矩阵链式验证失败")

    print("-" * 60)


def process_step5_images(calibration_data, max_images=None):
    """处理step5图像并计算相机位姿。"""
    print("\n处理step5图像...")

    # 获取图像路径
    left_images = sorted(
        glob.glob("images/step5/left/*.png") + glob.glob("images/step5/left/*.jpg")
    )
    right_images = sorted(
        glob.glob("images/step5/right/*.png") + glob.glob("images/step5/right/*.jpg")
    )

    if max_images is not None and max_images > 0:
        left_images = left_images[:max_images]
        right_images = right_images[:max_images]

    # 设置检测器
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if len(left_images) == 0:
        raise ValueError("未找到step5图像，请先运行 step5a_capture_for_base.py")
    # 处理左相机图像
    print(f"\n处理 {len(left_images)} 张左相机图像...")
    left_poses = process_images_and_estimate_pose(
        left_images,
        calibration_data["aruco_dict"],
        detector_params,
        calibration_data["obj_points"],
        calibration_data["tag_ids"],
        calibration_data["K_l"],
        calibration_data["dist_l"],
        camera_name="Cl",
        use_multiscale=bool(calibration_data.get("use_multiscale", True)),
        opencv_refine=bool(calibration_data.get("opencv_refine", False)),
        board=calibration_data.get("opencv_board"),
    )
    if len(left_poses) < 1:
        raise ValueError("左相机有效位姿太少，无法进行可靠估计")
    # 计算左相机位姿
    left_rvecs = [pose[0] for pose in left_poses]
    left_tvecs = [pose[1] for pose in left_poses]
    Cl_T_T = compute_C_T_T_mean_pose(left_rvecs, left_tvecs, cam_name="Cl")

    # 处理右相机图像（如果存在）
    Cr_T_T = None
    right_poses = []
    if len(right_images) > 0:
        print(f"\n处理 {len(right_images)} 张右相机图像...")
        right_poses = process_images_and_estimate_pose(
            right_images,
            calibration_data["aruco_dict"],
            detector_params,
            calibration_data["obj_points"],
            calibration_data["tag_ids"],
            calibration_data["K_r"],
            calibration_data["dist_r"],
            camera_name="Cr",
            use_multiscale=bool(calibration_data.get("use_multiscale", True)),
            opencv_refine=bool(calibration_data.get("opencv_refine", False)),
            board=calibration_data.get("opencv_board"),
        )
    if len(right_poses) < 1:
        raise ValueError("右相机有效位姿太少，无法进行可靠估计")
    right_rvecs = [pose[0] for pose in right_poses]
    right_tvecs = [pose[1] for pose in right_poses]
    Cr_T_T = compute_C_T_T_mean_pose(right_rvecs, right_tvecs, cam_name="Cr")

    return {
        "Cl_T_T": Cl_T_T,
        "Cr_T_T": Cr_T_T,
        "left_poses_count": len(left_poses),
        "right_poses_count": len(right_poses),
        "left_images_count": len(left_images),
        "right_images_count": len(right_images),
    }


def compute_camera_to_base_transforms(calibration_data, pose_data):
    """计算相机到底盘的变换矩阵。"""
    print("\n计算相机到底盘变换矩阵...")
    # 构造标定板到底盘的变换
    B_T_T = build_B_T_T_from_config(
        calibration_data["transform_cfg"], calibration_data["board_cfg"]
    )
    # 计算左相机到底盘的变换
    B_T_Cl = compute_B_T_Cl(B_T_T, pose_data["Cl_T_T"])
    # 计算右相机变换（如果有右相机数据）
    B_T_Cr = None

    stereo_validation_passed = False
    if pose_data["Cr_T_T"] is not None:
        # 双目外参一致性校验
        stereo_validation_passed = validate_stereo_consistency(
            Cl_T_T=pose_data["Cl_T_T"],
            Cr_T_T=pose_data["Cr_T_T"],
            Cr_T_Cl=calibration_data["Cr_T_Cl"],
            obj_points=calibration_data["obj_points"],
            tag_ids=calibration_data["tag_ids"],
            board_cfg=calibration_data["board_cfg"],
            K_l=calibration_data["K_l"],
            dist_l=calibration_data["dist_l"],
            K_r=calibration_data["K_r"],
            dist_r=calibration_data["dist_r"],
            strict=False,  # 不严格模式，只警告不中断
        )
        # 计算右相机位姿
        B_T_Cr = compute_B_T_Cr(B_T_Cl, calibration_data["Cr_T_Cl"])
        print("\n✓ 左右相机位姿均已计算")

        if not stereo_validation_passed:
            print("  注意: 双目外参校验未完全通过，建议检查Step4双目标定质量")
    else:
        print("\n- 仅计算左相机位姿（右相机数据不足）")

    # 显示关键Tag坐标（链式验证）
    print_tag_coordinates(
        B_T_T=B_T_T,
        Cl_T_T=pose_data["Cl_T_T"],
        B_T_Cl=B_T_Cl,
        obj_points=calibration_data["obj_points"],
        tag_ids=calibration_data["tag_ids"],
        board_cfg=calibration_data["board_cfg"],
    )

    return {
        "B_T_Cl": B_T_Cl,
        "B_T_Cr": B_T_Cr,
        "stereo_validation_passed": stereo_validation_passed,
    }


def save_calibration_results(calibration_data, pose_data, transform_data):
    """保存标定结果到JSON文件。"""
    result = {
        "timestamp": datetime.now().isoformat(),
        "B_T_Cl": transform_data["B_T_Cl"].tolist(),
        "config_used": {
            "board_to_base_transform": calibration_data["transform_cfg"],
            "apriltag_board": calibration_data["board_cfg"],
        },
        "left_pose_stats": {
            "total_images": pose_data["left_images_count"],
            "valid_poses": pose_data["left_poses_count"],
        },
        "stereo_validation": {
            "performed": transform_data["B_T_Cr"] is not None,
            "passed": bool(transform_data["stereo_validation_passed"]),
        },
    }

    if transform_data["B_T_Cr"] is not None:
        result["B_T_Cr"] = transform_data["B_T_Cr"].tolist()
        result["right_pose_stats"] = {
            "total_images": pose_data["right_images_count"],
            "valid_poses": pose_data["right_poses_count"],
        }

    os.makedirs("results", exist_ok=True)
    # 说明：json.dump 默认 ensure_ascii=True，会把中文转成 \uXXXX，肉眼看起来像“乱码”。
    # 这里显式使用 UTF-8 并关闭 ASCII 转义，保证文件里是可读中文。
    with open("results/camera_to_base.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存到 results/camera_to_base.json")
    print(f"  左相机: B_T_Cl (Cl -> B)")
    if transform_data["B_T_Cr"] is not None:
        print(f"  右相机: B_T_Cr (Cr -> B)")


def main():
    """主函数 - 封装后的清晰流程。"""
    parser = argparse.ArgumentParser(
        description="Step 5b: AprilTag 相机到机器人底盘外参标定（封装版）"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="限制每个相机最多处理的图片数量（默认处理全部）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 5b: AprilTag 相机到底盘坐标系标定（封装版）")
    print("=" * 60)

    try:
        # 1. 加载标定数据
        calibration_data = load_calibration_data()

        # 2. 处理图像并计算相机位姿
        pose_data = process_step5_images(calibration_data, args.max_images)

        # 3. 计算相机到底盘的变换矩阵
        transform_data = compute_camera_to_base_transforms(calibration_data, pose_data)

        # 4. 保存标定结果
        save_calibration_results(calibration_data, pose_data, transform_data)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
