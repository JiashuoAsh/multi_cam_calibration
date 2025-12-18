#!/usr/bin/env python3
"""
Step 5b: 相机到底盘坐标系标定 - AprilTag 标定板

计算相机坐标系到机器人底盘坐标系的变换矩阵。

坐标系定义:
    底盘坐标系: X右 Y上 Z前
    相机坐标系: OpenCV标准 (X右 Y下 Z前)

前提条件:
    - 已运行 step5a_capture_for_base.py 采集图像
    - 标定板固定在墙上（采集期间位置不变）
    - 已知标定板相对机器人底盘的固定位置
    - 已完成内参和双目外参标定

输入:
    - images/step5/left/*.png (step5a 采集的图像)
    - results/left_intrinsics.json
    - results/stereo_extrinsics.json
    - config/apriltag_config.json (board_to_base_transform)

输出:
    - results/camera_to_base.json (B_T_Cl, B_T_Cr)
"""

import cv2
import numpy as np
import json
import os
import glob
import argparse
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation
from utils import (
    load_config,
    get_aruco_dict,
    detect_apriltag_corners,
    create_apriltag_board,
    estimate_pose_apriltag,
)


def ensure_finite(x: Any, name: str, *, strict: bool = True) -> None:
    """自检：是否包含 NaN/Inf。"""
    arr = np.asarray(x, dtype=np.float64)
    ok = bool(np.all(np.isfinite(arr)))
    if not ok:
        msg = f"{name}: 包含 NaN/Inf，可能是检测/解算发散或输入数据异常。"
        if strict:
            raise ValueError(msg)
        print(f"  ! 警告: {msg}")


def fmt4(x) -> str:
    """将向量/矩阵展平后以4位小数格式化为字符串。"""
    arr = np.asarray(x, dtype=np.float64)
    return "[" + ", ".join(f"{v:.4f}" for v in arr.reshape(-1)) + "]"


def pretty_mat(name: str, T: np.ndarray, indent: str = "  ") -> None:
    """用4位小数打印矩阵（便于肉眼检查）。"""
    T = np.asarray(T, dtype=np.float64)
    s = np.array2string(
        T,
        formatter={"float_kind": lambda v: f"{float(v): .4f}"},
        suppress_small=False,
    )
    print(f"{indent}{name} =\n{indent}{s.replace(chr(10), chr(10) + indent)}")


def ensure_rotation_matrix(
    R: np.ndarray,
    name: str,
    *,
    ortho_tol: float = 1e-6,
    det_tol: float = 1e-3,
    strict: bool = True,
) -> None:
    """自检：旋转矩阵是否近似正交、det是否接近+1。"""
    R = np.asarray(R, dtype=np.float64)
    ensure_finite(R, name, strict=strict)
    if R.shape != (3, 3):
        raise ValueError(f"{name}: 期望形状(3,3)，实际{R.shape}")

    det_R = float(np.linalg.det(R))
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))

    # 旋转矩阵的逆应当等于转置（数值误差允许）
    try:
        inv_err = float(np.linalg.norm(np.linalg.inv(R) - R.T, ord="fro"))
    except Exception:
        inv_err = float("inf")

    col_norms = np.linalg.norm(R, axis=0)
    row_norms = np.linalg.norm(R, axis=1)

    print(
        f"  [自检] {name}: det={det_R:.6f}, ortho_err={ortho_err:.3e}, inv_vs_T_err={inv_err:.3e}, "
        f"col_norms={fmt4(col_norms)}, row_norms={fmt4(row_norms)}"
    )

    ok_det = abs(det_R - 1.0) <= det_tol
    ok_ortho = ortho_err <= ortho_tol
    ok_inv = inv_err <= 1e-6
    if strict and (not ok_det or not ok_ortho or not ok_inv):
        raise ValueError(
            f"{name}: 旋转矩阵自检失败(det={det_R}, ortho_err={ortho_err}). "
            "这通常意味着坐标系/角点顺序/欧拉角定义不一致。"
        )


def ensure_transform(
    T: np.ndarray,
    name: str,
    *,
    ortho_tol: float = 1e-6,
    det_tol: float = 1e-3,
    bottom_tol: float = 1e-9,
    strict: bool = True,
) -> None:
    """自检：4x4齐次变换矩阵格式 + 旋转子块有效性。"""
    T = np.asarray(T, dtype=np.float64)
    ensure_finite(T, name, strict=strict)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: 期望形状(4,4)，实际{T.shape}")

    bottom = T[3, :]
    bottom_target = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    bottom_err = float(np.linalg.norm(bottom - bottom_target))
    R = T[:3, :3]
    t = T[:3, 3]
    print(
        f"  [自检] {name}: bottom_err={bottom_err:.3e}, t={fmt4(t)}, "
        f"x_src_in_dst={fmt4(R[:, 0])}, y_src_in_dst={fmt4(R[:, 1])}, z_src_in_dst={fmt4(R[:, 2])}"
    )
    if strict and bottom_err > bottom_tol:
        raise ValueError(f"{name}: 齐次矩阵最后一行不是[0,0,0,1]，误差={bottom_err}")

    ensure_rotation_matrix(
        T[:3, :3],
        name + ".R",
        ortho_tol=ortho_tol,
        det_tol=det_tol,
        strict=strict,
    )


def make_transform(R: np.ndarray, t: np.ndarray, name: str, *, strict: bool = True) -> np.ndarray:
    """构造 4x4 齐次变换，并立即自检。"""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t
    T[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    ensure_transform(T, name, strict=strict)
    return T


def invert_transform(T: np.ndarray, name: str, *, strict: bool = True) -> np.ndarray:
    """求逆并自检：T * inv(T) 是否接近 I。"""
    T = np.asarray(T, dtype=np.float64)
    ensure_transform(T, name + ".input", strict=strict)
    T_inv = np.linalg.inv(T)
    ensure_transform(T_inv, name, strict=strict)
    I1 = T @ T_inv
    I2 = T_inv @ T
    err1 = float(np.linalg.norm(I1 - np.eye(4), ord="fro"))
    err2 = float(np.linalg.norm(I2 - np.eye(4), ord="fro"))
    print(f"  [自检] {name}: inv_err_fro(T*Tinv)={err1:.3e}, inv_err_fro(Tinv*T)={err2:.3e}")
    if strict and (err1 > 1e-6 or err2 > 1e-6):
        raise ValueError(f"{name}: 求逆自检失败，误差过大 ({err1}, {err2})")
    return T_inv


def tf_point(T_4x4: np.ndarray, p_3: np.ndarray) -> np.ndarray:
    """p_A = A_T_B @ p_B（齐次变换点）。"""
    p = np.asarray(p_3, dtype=np.float64).reshape(3)
    p_h = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64).reshape(4, 1)
    return (np.asarray(T_4x4, dtype=np.float64) @ p_h)[:3, 0]


def load_intrinsics(json_path):
    """加载内参"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data["camera_matrix"]), np.array(data["dist_coeffs"])


def load_stereo_extrinsics(json_path):
    """加载双目外参 (统一转换到米)

    所有距离单位，统一使用m做标准
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    R = np.array(data["R"], dtype=np.float64)
    t = np.array(data["t"], dtype=np.float64).reshape(3, 1)
    # mm->m
    t = t / 1000.0

    # 构造 Cr_T_Cl (左相机到右相机的变换)
    # 命名规范: A_T_B 表示 "B到A的变换" (X_A = A_T_B @ X_B)
    # step4 的 R,t 满足 X_r = R @ X_l + t，即 "左→右"
    # 因此这是 Cr_T_Cl (左到右)
    print("\n构造 Cr_T_Cl (Cl -> Cr) ...")
    Cr_T_Cl = make_transform(R, t, "Cr_T_Cl", strict=True)
    pretty_mat("Cr_T_Cl", Cr_T_Cl)
    return Cr_T_Cl


def euler_to_rotation_matrix(roll, pitch, yaw, degrees=True):
    """欧拉角转旋转矩阵

    重要：这里明确使用 SciPy 的 "XYZ 内旋"（body-fixed）语义：
      - Roll  绕 X 轴
      - Pitch 绕 Y 轴
      - Yaw   绕 Z 轴

    这必须与你 config/apriltag_config.json 里 rotation_euler_deg 的定义一致。

    Args:
        roll: 绕X轴旋转角度
        pitch: 绕Y轴旋转角度
        yaw: 绕Z轴旋转角度
        degrees: 是否为角度制（True 表示输入为度，False 表示弧度）

    Returns:
        旋转矩阵 (3x3)
    """
    R = Rotation.from_euler("XYZ", [roll, pitch, yaw], degrees=degrees).as_matrix()
    # 注意：R 的严格自检会在 make_transform(B_T_T) 时进行，这里只保证返回值形状正确
    return np.asarray(R, dtype=np.float64)


def compute_Cl_T_T_mean_pose(valid_rvecs, valid_tvecs) -> np.ndarray:
    """由多帧PnP结果计算平均位姿，输出 Cl_T_T (T -> Cl)。"""
    return compute_cam_T_T_mean_pose(valid_rvecs, valid_tvecs, cam_name="Cl")


def compute_cam_T_T_mean_pose(valid_rvecs, valid_tvecs, *, cam_name: str, verbose: bool = True) -> np.ndarray:
    """由多帧PnP结果计算平均位姿，输出 {cam_name}_T_T (T -> cam)。"""
    if verbose:
        print(f"\n[求解] 平均相机位姿 ({cam_name}_T_T: T -> {cam_name})")

    if len(valid_rvecs) == 0:
        raise ValueError(f"{cam_name}_T_T: valid_rvecs 为空")

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
        print("  - 纠正了 det(R_mean) < 0 的情况（防止反射矩阵）")

    t_mean = np.mean(valid_tvecs, axis=0)
    cam_T_T = make_transform(R_mean, t_mean, f"{cam_name}_T_T", strict=True)
    if verbose:
        pretty_mat(f"{cam_name}_T_T", cam_T_T)
    return cam_T_T


def _rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """将 rvec/tvec 转换为 4x4 齐次变换（T -> Cam）。

    注意：这里保持“安静”(不调用 ensure_transform)，用于批量统计。
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = tvec
    return T


def _T_to_rt(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将 4x4 齐次变换转回 rvec/tvec。"""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3:4]
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3, 1), t.reshape(3, 1)


def _rot_err_deg(R: np.ndarray) -> float:
    """旋转误差角度（度）。"""
    R = np.asarray(R, dtype=np.float64)
    return float(Rotation.from_matrix(R).magnitude() * 180.0 / np.pi)


def _inv_T(T: np.ndarray, name: str, *, strict: bool = True) -> np.ndarray:
    """安静求逆（可选自检）。"""
    T = np.asarray(T, dtype=np.float64)
    T_inv = np.linalg.inv(T)
    ensure_transform(T_inv, name, strict=strict)
    return T_inv








def _get_prominent_tag_ids(tags_x: int, tags_y: int, tag_centers: Dict[int, np.ndarray]) -> List[int]:
    """获取显眼的tag ID（四角+中心）。"""
    candidate_ids = [
        0,
        tags_x - 1,
        (tags_y - 1) * tags_x,
        tags_x * tags_y - 1,
        (tags_y // 2) * tags_x + (tags_x // 2),
    ]
    return [tid for tid in candidate_ids if tid in tag_centers]


def process_images_and_estimate_pose_map(
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
) -> Tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
    """处理图像并估计位姿，返回以文件名为key的pose_map，便于左右配对。"""
    pose_map: Dict[str, Dict[str, Any]] = {}
    debug_sample = None

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids = detect_apriltag_corners(gray, aruco_dict, detector_params, use_multiscale=use_multiscale)

        if len(corners) < 5:
            continue

        if ids is not None and not isinstance(ids, np.ndarray):
            ids = np.array(ids, dtype=np.int32)

        success, rvec, tvec = estimate_pose_apriltag(
            corners, ids, obj_points, tag_ids, K, dist
        )

        if success:
            key = os.path.basename(img_path)
            pose_map[key] = {
                "img_path": img_path,
                "corners": corners,
                "ids": ids,
                "rvec": rvec,
                "tvec": tvec,
            }
            if debug_sample is None:
                debug_sample = dict(pose_map[key])

    print(f"  - {camera_name}: 有效位姿: {len(pose_map)}/{len(image_paths)}")
    return pose_map, debug_sample








def build_B_T_T_from_config(transform_cfg: Dict[str, Any], board_cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """由配置构造 B_T_T (T -> B)，并处理translation reference换算。

    Returns:
        (B_T_T, debug): debug包含 reference点等用于打印/自检。
    """
    print("\n[求解] 构造标定板到底盘的变换 (B_T_T: T -> B)")

    rotation_euler = transform_cfg["rotation_euler_deg"]
    R_B_T = euler_to_rotation_matrix(*rotation_euler, degrees=True)
    ensure_rotation_matrix(R_B_T, "R_B_T", strict=True)

    translation_ref = transform_cfg.get("translation_reference", "tag0_center")
    translation_input_B = np.array(transform_cfg["translation"], dtype=np.float64).reshape(3, 1)

    tag_pitch_m = (board_cfg["tag_size"] + board_cfg["tag_spacing"]) / 1000.0
    center_T = np.array(
        [
            (board_cfg["tags_x"] - 1) * tag_pitch_m / 2.0,
            (board_cfg["tags_y"] - 1) * tag_pitch_m / 2.0,
            0.0,
        ],
        dtype=np.float64,
    ).reshape(3, 1)

    custom_ref_pt = transform_cfg.get("translation_reference_point_in_T_m")
    if custom_ref_pt is not None:
        try:
            center_T = np.array(custom_ref_pt, dtype=np.float64).reshape(3, 1)
            print(
                "  - 使用自定义 translation_reference_point_in_T_m 作为参考点在 T 中的位置: "
                f"{fmt4(center_T)}"
            )
        except Exception as e:
            print(
                "  ! 警告: translation_reference_point_in_T_m 解析失败，将忽略该字段。"
                f" 错误: {e}"
            )

    if translation_ref in ("board_center", "grid_center", "board_center_tag_grid"):
        # center_B = R_B_T * center_T + origin_B
        # origin_B = center_B - R_B_T * center_T
        translation_B = translation_input_B - (R_B_T @ center_T)
        print(f"  - translation_reference={translation_ref}: 将配置的板中心平移换算为 Tag0 中心平移")
        print(f"    * board_center_in_T(m) = {fmt4(center_T)}")
        print(f"    * tag0_center_in_B(m)  = {fmt4(translation_B)}")
    else:
        translation_B = translation_input_B
        print(f"  - translation_reference={translation_ref}: 直接使用 translation 作为 Tag0 中心位置")
        print(f"    * tag0_center_in_B(m)  = {fmt4(translation_B)}")

    B_T_T = make_transform(R_B_T, translation_B, "B_T_T", strict=True)
    pretty_mat("B_T_T", B_T_T)

    debug = {
        "translation_ref": translation_ref,
        "translation_input_B": translation_input_B.reshape(3),
        "center_T": center_T.reshape(3),
        "translation_B": translation_B.reshape(3),
        "R_B_T": R_B_T,
    }
    return B_T_T, debug


def compute_B_T_Cl(B_T_T: np.ndarray, Cl_T_T: np.ndarray) -> np.ndarray:
    """主链：B_T_Cl = B_T_T @ inv(Cl_T_T)。"""
    print("\n[求解] 计算左相机到底盘变换 (B_T_Cl: Cl -> B)")
    T_T_Cl = invert_transform(Cl_T_T, "T_T_Cl = inv(Cl_T_T)", strict=True)
    B_T_Cl = B_T_T @ T_T_Cl
    ensure_transform(B_T_Cl, "B_T_Cl", strict=True)
    pretty_mat("B_T_Cl", B_T_Cl)
    return B_T_Cl


def compute_B_T_Cr(B_T_Cl: np.ndarray, Cr_T_Cl: np.ndarray) -> np.ndarray:
    """右相机：B_T_Cr = B_T_Cl @ inv(Cr_T_Cl)。"""
    print("\n[求解] 计算右相机到底盘变换 (B_T_Cr: Cr -> B)")
    Cl_T_Cr = invert_transform(Cr_T_Cl, "Cl_T_Cr = inv(Cr_T_Cl)", strict=True)
    B_T_Cr = B_T_Cl @ Cl_T_Cr
    ensure_transform(B_T_Cr, "B_T_Cr", strict=True)
    pretty_mat("B_T_Cr", B_T_Cr)
    return B_T_Cr


def compute_B_T_Cr_from_pnp(B_T_T: np.ndarray, Cr_T_T: np.ndarray) -> np.ndarray:
    """当右相机图像可用时，可直接用右PnP位姿计算：B_T_Cr = B_T_T @ inv(Cr_T_T)。"""
    print("\n[求解] 计算右相机到底盘变换（右PnP直算） (B_T_Cr: Cr -> B)")
    T_T_Cr = invert_transform(Cr_T_T, "T_T_Cr = inv(Cr_T_T)", strict=True)
    B_T_Cr = np.asarray(B_T_T, dtype=np.float64) @ np.asarray(T_T_Cr, dtype=np.float64)
    ensure_transform(B_T_Cr, "B_T_Cr (from_pnp)", strict=True)
    pretty_mat("B_T_Cr_from_pnp", B_T_Cr)
    return B_T_Cr


def print_transform_summary(name: str, A_T_B: np.ndarray, *, indent: str = "  ") -> None:
    """打印更直观的“坐标系轴向+原点”信息。

    约定：A_T_B 表示 B->A。
    因此 A_T_B 的旋转列向量分别是：x_B/y_B/z_B 在 A 中的方向。
    平移向量是：B原点在A中的坐标。
    """
    A_T_B = np.asarray(A_T_B, dtype=np.float64)
    ensure_transform(A_T_B, f"{name} (summary)", strict=False)
    R = A_T_B[:3, :3]
    t = A_T_B[:3, 3]
    print(f"{indent}- {name}:")
    print(f"{indent}  origin_B_in_A = {fmt4(t)}")
    print(f"{indent}  x_B in A = {fmt4(R[:, 0])}")
    print(f"{indent}  y_B in A = {fmt4(R[:, 1])}")
    print(f"{indent}  z_B in A = {fmt4(R[:, 2])}")





def debug_print_self_checks(
    *,
    B_T_T: np.ndarray,
    Cl_T_T: np.ndarray,
    B_T_Cl: Optional[np.ndarray] = None,
    B_T_Cr: Optional[np.ndarray] = None,
    Cr_T_Cl: Optional[np.ndarray] = None,
    obj_points: np.ndarray,
    tag_ids: List[int],
    board_cfg: Dict[str, Any],
    debug_cfg: Dict[str, Any],
    debug_sample: Optional[Dict[str, Any]],
    K_l: np.ndarray,
    dist_l: np.ndarray,
) -> None:
    """打印更“人类可读”的自检信息：T轴在B中的方向、reference点一致性、显眼tag中心点、像素投影残差。"""
    print("\n" + "-" * 60)
    print("[自检] 标定板(T)到位姿(B/Cl)的可读性检查")

    R_B_T = np.asarray(debug_cfg["R_B_T"], dtype=np.float64)
    print(f"  - X_T in B = {fmt4(R_B_T[:, 0])}")
    print(f"  - Y_T in B = {fmt4(R_B_T[:, 1])}")
    print(f"  - Z_T in B = {fmt4(R_B_T[:, 2])}")

    print("\n  - 关键变换摘要（轴向/原点）")
    print_transform_summary("B_T_T (T -> B)", B_T_T, indent="    ")
    print_transform_summary("Cl_T_T (T -> Cl)", Cl_T_T, indent="    ")
    if B_T_Cl is not None:
        print_transform_summary("B_T_Cl (Cl -> B)", B_T_Cl, indent="    ")
    if B_T_Cr is not None:
        print_transform_summary("B_T_Cr (Cr -> B)", B_T_Cr, indent="    ")

    # reference点自洽检查
    translation_ref = str(debug_cfg["translation_ref"])
    translation_input_B = np.asarray(debug_cfg["translation_input_B"], dtype=np.float64).reshape(3)
    center_T = np.asarray(debug_cfg["center_T"], dtype=np.float64).reshape(3)
    tag0_in_B = tf_point(B_T_T, np.array([0.0, 0.0, 0.0], dtype=np.float64))
    print(f"  - Tag0中心(=T原点) 在B中的坐标 = {fmt4(tag0_in_B)}")
    if translation_ref in ("board_center", "grid_center", "board_center_tag_grid"):
        center_in_B_pred = tf_point(B_T_T, center_T)
        diff = center_in_B_pred - translation_input_B
        print(f"  - 参考点({translation_ref}) 在T中的坐标 = {fmt4(center_T)}")
        print(f"  - 参考点({translation_ref}) 在B中的坐标(由B_T_T预测) = {fmt4(center_in_B_pred)}")
        print(f"  - 参考点({translation_ref}) 在B中的坐标(来自配置translation) = {fmt4(translation_input_B)}")
        print(f"  - 二者差值 = {fmt4(diff)}，|diff| = {float(np.linalg.norm(diff)):.6f} m")

    # 显眼tag中心点
    tag_centers_T = {int(tag_ids[i]): obj_points[i].mean(axis=0).reshape(3) for i in range(len(tag_ids))}
    picked_ids = _get_prominent_tag_ids(board_cfg["tags_x"], board_cfg["tags_y"], tag_centers_T)

    print("\n  - 显眼Tag中心点坐标（m: p_B=B_T_T@p_T, p_Cl=Cl_T_T@p_T）")
    for tid in picked_ids:
        p_T = tag_centers_T[tid]
        print(f"    * tag={tid:>2d}: T={fmt4(p_T)} -> B={fmt4(tf_point(B_T_T, p_T))} -> Cl={fmt4(tf_point(Cl_T_T, p_T))}")

    if B_T_Cl is not None:
        print("\n  - 链路一致性自检（B_T_T@p_T vs B_T_Cl@Cl_T_T@p_T）")
        errs = []
        for tid in picked_ids:
            p_T = tag_centers_T[tid]
            p_B_via_board = tf_point(B_T_T, p_T)
            p_B_via_cam = tf_point(B_T_Cl, tf_point(Cl_T_T, p_T))
            err = float(np.linalg.norm(p_B_via_cam - p_B_via_board))
            errs.append(err)
            print(f"    * tag={tid:>2d}: board={fmt4(p_B_via_board)} cam={fmt4(p_B_via_cam)} |diff|={err:.6f}m")
        if errs:
            print(f"    => max |diff| = {max(errs):.6f} m")

    if B_T_Cr is not None and Cr_T_Cl is not None and B_T_Cl is not None:
        try:
            _check_right_camera_consistency(B_T_T, B_T_Cl, B_T_Cr, Cl_T_T, Cr_T_Cl,
                                           tag_centers_T, picked_ids)
        except Exception as e:
            print(f"  ! 右相机/双目一致性自检失败（不影响主流程），错误: {e}")

    # 像素级自检
    if debug_sample is not None:
        try:
            ids_flat = np.array(debug_sample["ids"], dtype=np.int32).reshape(-1) if debug_sample["ids"] is not None else np.array([], dtype=np.int32)
            obs_center_px = {int(ids_flat[i]): np.array(debug_sample["corners"][i], dtype=np.float64).reshape(-1, 2).mean(axis=0)
                            for i in range(len(ids_flat))}

            print(f"\n  - 像素级自检（样本: {debug_sample['img_path']}）")
            for tid in picked_ids:
                if tid in obs_center_px and tid in tag_centers_T:
                    pt3 = tag_centers_T[tid].reshape(-1, 3)
                    proj, _ = cv2.projectPoints(pt3, debug_sample["rvec"], debug_sample["tvec"], K_l, dist_l)
                    err_norm = float(np.linalg.norm(obs_center_px[tid] - proj.reshape(-1, 2)[0]))
                    print(f"    * tag={tid:>2d}: |err|={err_norm:.2f}px")
            print("    注：若Step5期间移动了底盘，单帧与均值不一致是正常的")
        except Exception as e:
            print(f"  ! 像素级自检失败（不影响主流程），错误: {e}")

    print("-" * 60)


def main():
    """主函数

    命名规范（强制）:
      A_T_B 表示 "B -> A 的变换":
        X_A = A_T_B @ X_B

    我们将构造并导出:
      B_T_Cl: 左相机坐标系 Cl -> 底盘坐标系 B
      B_T_Cr: 右相机坐标系 Cr -> 底盘坐标系 B

    其中:
      Cl_T_T: 标定板 T -> 左相机 Cl （来自 solvePnP/estimate_pose_apriltag）
      B_T_T : 标定板 T -> 底盘 B（来自 config 的测量值）
    """
    parser = argparse.ArgumentParser(description="Step 5b: AprilTag 相机到机器人底盘外参标定")
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="调试用：限制每个相机最多处理的图片数量（默认处理全部）",
    )
    parser.add_argument(
        "--no_multiscale",
        action="store_true",
        help="调试用：关闭多尺度检测以加速（可能降低检测成功率）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 5b: AprilTag 相机到底盘坐标系标定")
    print("=" * 60)
    print("\n命名规范: A_T_B 表示 'B -> A 的变换' (X_A = A_T_B @ X_B)")

    # 检查必要文件
    required_files = [
        "results/left_intrinsics.json",
        "results/right_intrinsics.json",
        "results/stereo_extrinsics.json",
    ]

    for file in required_files:
        if not os.path.exists(file):
            print(f"\n错误: 未找到 {file}")
            print("请先运行前面的标定步骤")
            return

    # 加载配置
    config = load_config()
    board_cfg = config["apriltag_board"]
    transform_cfg = config["board_to_base_transform"]

    print("\n坐标系定义:")
    print("  底盘坐标系: X右 Y上 Z前")
    print("  相机坐标系: X右 Y下 Z前 (OpenCV标准)")

    print("\n标定板到底盘的变换 (需要测量并配置):")
    print(f"  - 平移 (m): {transform_cfg['translation']}")
    print(f"  - 旋转 (度): {transform_cfg['rotation_euler_deg']}")
    print(f"\n注意: 如果这些值不正确,请更新 config/apriltag_config.json")
    # input("按 Enter 继续...")

    # 加载内参和双目外参
    print("\n加载标定参数...")
    K_l, dist_l = load_intrinsics("results/left_intrinsics.json")
    K_r, dist_r = load_intrinsics("results/right_intrinsics.json")
    Cr_T_Cl = load_stereo_extrinsics("results/stereo_extrinsics.json")
    print("  ✓ 内参和双目外参已加载")

    # 创建 AprilTag 标定板
    # utils.create_apriltag_board() 输出单位为 mm，这里统一转换为 m。
    obj_points_mm, tag_ids = create_apriltag_board(config)
    obj_points = obj_points_mm.astype(np.float64) / 1000.0
    aruco_dict = get_aruco_dict(board_cfg["family"])

    # 获取 step5 采集的图像（使用左相机图像）
    left_images = sorted(
        glob.glob("images/step5/left/*.png")
        + glob.glob("images/step5/left/*.jpg")
        + glob.glob("images/step5/left/*.jpeg")
    )
    right_images = sorted(
        glob.glob("images/step5/right/*.png")
        + glob.glob("images/step5/right/*.jpg")
        + glob.glob("images/step5/right/*.jpeg")
    )

    if args.max_images is not None and args.max_images > 0:
        left_images = left_images[: args.max_images]
        right_images = right_images[: args.max_images]
        print(f"\n[调试] --max_images={args.max_images}: 将只处理每个相机的前 {args.max_images} 张图片")

    use_multiscale = not bool(args.no_multiscale)
    print(f"\n[调试] use_multiscale={use_multiscale}")

    if len(left_images) == 0:
        print("\n错误: 未找到 step5 专用图像")
        print("\n请先运行 step5a 采集图像:")
        print("  python3 step5a_capture_for_base.py")
        print("\n注意事项:")
        print("  1. 标定板必须固定在墙上（不能移动）")
        print("  2. 测量并更新 config/apriltag_config.json 中的 board_to_base_transform")
        print("  3. 采集 3-5 张图像（机器人在不同位置拍摄）")
        return

    # 设置检测器
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

    # ---- (1) 左相机：估计每张图像的 Cl_T_T，并求平均 ----
    print(f"\n处理 {len(left_images)} 张左相机图像...")
    left_pose_map, debug_sample = process_images_and_estimate_pose_map(
        left_images,
        aruco_dict,
        detector_params,
        obj_points,
        tag_ids,
        K_l,
        dist_l,
        camera_name="Cl",
        use_multiscale=use_multiscale,
    )
    if len(left_pose_map) < 3:
        print("\n错误: 左相机有效位姿太少，无法进行可靠估计")
        return

    left_rvecs = [v["rvec"] for v in left_pose_map.values()]
    left_tvecs = [v["tvec"] for v in left_pose_map.values()]
    Cl_T_T = compute_cam_T_T_mean_pose(left_rvecs, left_tvecs, cam_name="Cl")

    # ---- (2) 右相机：估计每张图像的 Cr_T_T，并求平均 ----
    Cr_T_T = None
    right_pose_map: Dict[str, Dict[str, Any]] = {}
    if len(right_images) > 0:
        print(f"\n处理 {len(right_images)} 张右相机图像...")
        right_pose_map, debug_sample_r = process_images_and_estimate_pose_map(
            right_images,
            aruco_dict,
            detector_params,
            obj_points,
            tag_ids,
            K_r,
            dist_r,
            camera_name="Cr",
            use_multiscale=use_multiscale,
        )
        if len(right_pose_map) >= 3:
            right_rvecs = [v["rvec"] for v in right_pose_map.values()]
            right_tvecs = [v["tvec"] for v in right_pose_map.values()]
            Cr_T_T = compute_cam_T_T_mean_pose(right_rvecs, right_tvecs, cam_name="Cr")
        else:
            print("\n  ! 警告: 右相机有效位姿不足(少于3帧)，将跳过 stereo_extrinsics 校验。")
    else:
        print("\n  ! 警告: 未找到右相机图像，将跳过 stereo_extrinsics 校验。")

    B_T_T, debug_cfg = build_B_T_T_from_config(transform_cfg, board_cfg)
    B_T_Cl = compute_B_T_Cl(B_T_T, Cl_T_T)

    stereo_validation_passed = None  # None=未检查
    stereo_validation_error = None
    B_T_Cr_stereo = None
    B_T_Cr_from_pnp = None

    # 先尝试做 stereo 一致性校验（如果右相机位姿可用）
    if Cr_T_T is not None and len(right_pose_map) >= 3:
        try:
            validate_stereo_extrinsics_with_board_poses(
                left_pose_map=left_pose_map,
                right_pose_map=right_pose_map,
                Cl_T_T_mean=Cl_T_T,
                Cr_T_T_mean=Cr_T_T,
                Cr_T_Cl=Cr_T_Cl,
                obj_points=obj_points,
                tag_ids=tag_ids,
                board_cfg=board_cfg,
                K_r=K_r,
                dist_r=dist_r,
                strict=True,
            )
            stereo_validation_passed = True
        except ValueError as e:
            stereo_validation_passed = False
            stereo_validation_error = str(e)
            print("\n" + "!" * 60)
            print("! [警告] stereo_extrinsics 校验失败")
            print(f"! {e}")
            # 关键诊断信息：step4 的 stereo reprojection_error 若较大，会导致这里误差偏大
            try:
                with open("results/stereo_extrinsics.json", "r") as f:
                    _st = json.load(f)
                rep = _st.get("reprojection_error")
                if rep is not None:
                    print(f"! 提示: results/stereo_extrinsics.json 的 reprojection_error={float(rep):.3f}px（>3px 通常表示双目外参质量偏差）")
            except Exception:
                pass
            print("! 处理策略: 仍将继续求解 B_T_Cl；右相机 B_T_Cr 将优先使用‘右PnP直算’结果。")
            print("!" * 60)

    # 两种方式计算 B_T_Cr
    try:
        B_T_Cr_stereo = compute_B_T_Cr(B_T_Cl, Cr_T_Cl)
    except Exception as e:
        print(f"\n  ! 警告: 使用 stereo_extrinsics 计算 B_T_Cr 失败: {e}")

    if Cr_T_T is not None:
        try:
            B_T_Cr_from_pnp = compute_B_T_Cr_from_pnp(B_T_T, Cr_T_T)
        except Exception as e:
            print(f"\n  ! 警告: 使用右PnP直算 B_T_Cr 失败: {e}")

    # 选择最终用于输出的 B_T_Cr
    if stereo_validation_passed is True and B_T_Cr_stereo is not None:
        B_T_Cr = B_T_Cr_stereo
        B_T_Cr_source = "stereo_extrinsics"
    elif B_T_Cr_from_pnp is not None:
        B_T_Cr = B_T_Cr_from_pnp
        B_T_Cr_source = "right_pnp"
    else:
        B_T_Cr = B_T_Cr_stereo
        B_T_Cr_source = "stereo_extrinsics_unverified"

    # 全链路“可读性+一致性”自检（覆盖左/右/双目链）
    debug_print_self_checks(
        B_T_T=B_T_T,
        Cl_T_T=Cl_T_T,
        B_T_Cl=B_T_Cl,
        B_T_Cr=B_T_Cr,
        Cr_T_Cl=Cr_T_Cl,
        obj_points=obj_points,
        tag_ids=tag_ids,
        board_cfg=board_cfg,
        debug_cfg=debug_cfg,
        debug_sample=debug_sample,
        K_l=K_l,
        dist_l=dist_l,
    )

    # 保存结果（修正描述，避免"谁到谁"写反）
    result = {
        "units": "m",
        "B_T_Cl": B_T_Cl.tolist(),
        "B_T_Cr": None if B_T_Cr is None else B_T_Cr.tolist(),
        "B_T_Cr_source": B_T_Cr_source,
        "B_T_Cr_stereo": None if B_T_Cr_stereo is None else B_T_Cr_stereo.tolist(),
        "B_T_Cr_from_pnp": None if B_T_Cr_from_pnp is None else B_T_Cr_from_pnp.tolist(),
        "stereo_validation": {
            "passed": stereo_validation_passed,
            "error": stereo_validation_error,
        },
        "naming_convention": "A_T_B 表示 'B->A' (X_A = A_T_B @ X_B)",
        "coordinate_system_note": {
            "Cl/Cr": "OpenCV相机坐标系 (X右 Y下 Z前)",
            "B": "用户定义底盘坐标系（以你的测量定义为准）",
            "T": "标定板坐标系（由 create_apriltag_board 的3D点定义）",
        },
        "description": {
            "B_T_Cl": "左相机 Cl -> 底盘 B 的 4x4 变换矩阵",
            "B_T_Cr": "右相机 Cr -> 底盘 B 的 4x4 变换矩阵",
        },
        "usage": "P_B = B_T_Cl @ P_Cl (P_Cl 为齐次坐标，Cl 用 OpenCV 相机坐标系)",
    }

    # 说明：json.dump 默认 ensure_ascii=True，会把中文转成 \uXXXX，肉眼看起来像“乱码”。
    with open("results/camera_to_base.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n  ✓ 已保存: results/camera_to_base.json")

    # 显示总结
    print("\n" + "=" * 60)
    print("相机到底盘标定完成！")
    print("=" * 60)

    print("\n命名规范: A_T_B 表示 'B -> A 的变换' (X_A = A_T_B @ X_B)")
    print("\n坐标系定义:")
    print("  - 相机坐标系 (Cl/Cr): OpenCV 标准 (X右 Y下 Z前)")
    print("  - 底盘坐标系 (B): 以你的测量定义为准")

    print("\n左相机到底盘变换 (B_T_Cl: Cl -> B):")
    pretty_mat("B_T_Cl", B_T_Cl)

    print("\n右相机到底盘变换 (B_T_Cr: Cr -> B):")
    if B_T_Cr is None:
        print("  (未生成 B_T_Cr)")
    else:
        pretty_mat("B_T_Cr", B_T_Cr)

    # 提取位置信息
    left_pos = B_T_Cl[:3, 3]
    right_pos = None if B_T_Cr is None else B_T_Cr[:3, 3]

    print(f"\n相机在底盘坐标系中的位置 (单位: 米):")
    print(f"  - 左相机: [{left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f}]")
    if right_pos is not None:
        print(f"  - 右相机: [{right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f}]")
    else:
        print("  - 右相机: (未生成)")

    # 验证信息
    print("\n变换链验证:")
    print("  ✓ B_T_Cl = B_T_T @ inv(Cl_T_T)  [Cl -> T -> B]")
    if B_T_Cr is not None:
        print(f"  ✓ B_T_Cr 来源: {B_T_Cr_source}")
        print("  ✓ (stereo) B_T_Cr = B_T_Cl @ inv(Cr_T_Cl)  [Cr -> Cl -> B]")
        print("  ✓ (pnp)    B_T_Cr = B_T_T @ inv(Cr_T_T)   [Cr -> T -> B]")

    print("\n使用示例:")
    print("  import json")
    print("  import numpy as np")
    print("")
    print("  with open('results/camera_to_base.json', 'r') as f:")
    print("      data = json.load(f)")
    print("  B_T_Cl = np.array(data['B_T_Cl'])")
    print("")
    print("  # 将左相机坐标系中的点转到底盘坐标系")
    print("  P_Cl = np.array([x, y, z, 1])  # 齐次坐标 (OpenCV: X右 Y下 Z前)")
    print("  P_B = B_T_Cl @ P_Cl")
    print("  P_B = P_B[:3]  # 转回3D坐标")


if __name__ == "__main__":
    main()
