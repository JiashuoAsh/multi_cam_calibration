#!/usr/bin/env python3
"""
将 AprilTag 标定结果转换为旧格式

命名规范:
    A_T_B 表示 "B -> A 的变换" (X_A = A_T_B @ X_B)

核心背景：同一件事情有两套“外参”写法
------------------------------------------------------------
在本项目里你会同时遇到两种常见的外参表示，它们都描述“相机相对世界/底盘的位置与姿态”，
但写法不同：

1) 齐次变换矩阵（Pose / T 矩阵）
   例如 camera_to_base.json 给出的 B_T_C：
       X_B = B_T_C @ X_C
   其中：
       B_T_C = [[R_BC, t_BC],
                [ 0  ,  1  ]]
   - R_BC: camera坐标系 -> base坐标系 的旋转
   - t_BC: camera原点在base坐标系下的位置（也就是“相机中心”C，通常直接记作 C_B）

2) OpenCV 投影外参（[R|t] 形式）
   旧代码 camera_cali_usb 使用的 id_car_matrix_*.json 保存的是：
       X_cam = R * X_world + t
   这里 world 通常是 base（或你选定的参考系）。

两者的关系（最重要的一行公式）:
------------------------------------------------------------
若 camera_to_base 给出的是 base<-camera：
    X_B = R_BC * X_C + C
那么它的逆（camera<-base）就是 OpenCV 需要的 world(base)->camera：
    X_C = R_CB * X_B + t
其中：
    R_CB = R_BC^T
    t    = -R_CB * C

此外，旧格式的 id_car_eula_*.json 里前三个数 (x,y,z) 存的是“相机中心 C（在world/base里）”，
并不是 OpenCV 里的 t。

功能:
    - 将新标定格式转换为旧格式以保持兼容性
    - 生成 id_car_matrix_YYYYMMDD.json (OpenCV R,T 格式)
    - 生成 id_car_eula_YYYYMMDD.json (紧凑字符串格式)

使用方法:
    python convert_to_legacy_format.py

输入:
    - results/left_intrinsics.json
    - results/right_intrinsics.json
    - results/camera_to_base.json (可选，如果有的话)
    - results/stereo_extrinsics.json

输出:
    - id_car_matrix_YYYYMMDD.json
    - id_car_eula_YYYYMMDD.json

注意:
    - 如果有 camera_to_base.json，则使用底盘坐标系作为参考
    - 否则使用左相机坐标系作为参考
"""

import json
import numpy as np
import os
from datetime import datetime
from scipy.spatial.transform import Rotation as R


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """将相对路径解析为脚本目录下的绝对路径。"""
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_json(filepath):
    """加载 JSON 文件"""
    filepath = resolve_path(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"未找到文件: {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)


def rotation_matrix_to_euler_angles(R_matrix):
    """
    将旋转矩阵转换为欧拉角 (yaw, pitch, roll) 弧度制

    约定: SciPy 的 'zyx'（注意：这是 SciPy 的“欧拉角序列定义”，
    与 camera_cali_usb 旧代码使用的 yaw/pitch/roll 定义不同。）

    ⚠️ 旧格式输出请使用 rotation_matrix_to_legacy_usb_ypr。

    Args:
        R_matrix: 3x3 旋转矩阵

    Returns:
        euler: [yaw, pitch, roll] 弧度
    """
    r = R.from_matrix(R_matrix)
    euler = r.as_euler("zyx", degrees=False)
    return euler


def rotation_matrix_to_legacy_usb_ypr(R_matrix: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为旧 USB 标定代码使用的 yaw/pitch/roll（弧度）

    旧代码（camera_cali_usb/adj_utils.py）使用如下构造：
        R = Rz(roll) @ Rx(pitch) @ Ry(yaw)

    这里的 yaw/pitch/roll 是“旧代码自定义”的欧拉角约定（含符号与旋转顺序）。
    不能直接用 SciPy 的 as_euler('zyx') 代替，否则会和旧代码不一致。

    这里返回 [yaw, pitch, roll]，保证用旧代码公式重建时与 R_matrix 一致。

    注意：该分解在 pitch 接近 ±pi/2 时会出现万向节锁。
    """

    Rm = np.asarray(R_matrix, dtype=float)
    if Rm.shape != (3, 3):
        raise ValueError(f"R_matrix must be 3x3, got {Rm.shape}")

    # 推导（与旧 USB 代码一致）：
    # pitch = asin(R[2,1])
    # yaw   = atan2(-R[2,0], R[2,2])
    # roll  = atan2(R[0,1], R[1,1])
    pitch = float(np.arcsin(np.clip(Rm[2, 1], -1.0, 1.0)))
    cp = float(np.cos(pitch))
    if abs(cp) < 1e-9:
        # 万向节锁：roll 不可唯一确定。此处固定 roll=0，yaw 用另一组元素估计。
        yaw = float(np.arctan2(Rm[0, 2], Rm[0, 0]))
        roll = 0.0
    else:
        yaw = float(np.arctan2(-Rm[2, 0], Rm[2, 2]))
        roll = float(np.arctan2(Rm[0, 1], Rm[1, 1]))
    return np.array([yaw, pitch, roll], dtype=float)


def invert_base_T_camera_to_world2cam(B_T_C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 base<-camera (B_T_C) 转为旧格式需要的 world(base)->camera 外参。

    这一步就是把“Pose矩阵（base<-camera）”转换成“OpenCV 投影外参（camera<-base）”。

    输入:
        B_T_C: 4x4，满足：
            X_B = R_BC * X_C + C
        其中 C = t_BC 是相机中心在 base 下的位置。

    输出（与旧格式完全对齐）:
        R: world(base)->camera 的 3x3 旋转矩阵 (R_CB)
        t: world(base)->camera 的 3x1 平移向量 (t_CB)
        C: 相机中心在 world(base) 下的位置 (3,)

    旧格式 OpenCV 外参满足:
        X_C = R * X_B + t
        C   = -R^T * t

    因此转换关系就是：
        R = R_BC^T
        t = -R * C
    """

    T = np.asarray(B_T_C, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"B_T_C must be 4x4, got {T.shape}")

    R_BC = T[:3, :3]
    C = T[:3, 3].reshape(3)

    # inverse: camera<-base
    R = R_BC.T
    t = (-R @ C).reshape(3)
    return R, t, C


def format_matrix_json(
    left_R, left_T, right_R, right_T, left_mtx, left_dist, right_mtx, right_dist, timestamp=None
):
    """
    格式化标定结果为 OpenCV 矩阵格式

    返回匹配 id_car_matrix_YYYYMMDD.json 的 JSON 结构

    Args:
        left_R: 左相机旋转矩阵 (3x3)
        left_T: 左相机平移向量 (3,)
        right_R: 右相机旋转矩阵 (3x3)
        right_T: 右相机平移向量 (3,)
        left_mtx: 左相机内参矩阵 (3x3)
        left_dist: 左相机畸变系数
        right_mtx: 右相机内参矩阵 (3x3)
        right_dist: 右相机畸变系数
        timestamp: 时间戳字符串

    Returns:
        data: 格式化的字典列表
    """
    data = [
        {
            "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cameraLeft": {
                "R": left_R.tolist(),
                "T": left_T.reshape(-1, 1).tolist(),
                "dist": left_dist.flatten().tolist(),
                "mtx": left_mtx.tolist(),
            },
            "cameraRight": {
                "R": right_R.tolist(),
                "T": right_T.reshape(-1, 1).tolist(),
                "dist": right_dist.flatten().tolist(),
                "mtx": right_mtx.tolist(),
            },
        }
    ]

    return data


def format_eula_json(
    left_R,
    left_C,
    right_R,
    right_C,
    left_mtx,
    left_dist,
    right_mtx,
    right_dist,
    reproj_error_left=0.0,
    reproj_error_right=0.0,
    timestamp=None,
):
    """
    格式化标定结果为紧凑的欧拉角字符串格式

    格式: "x, y, z, yaw, pitch, roll, fx, cx, cy, k1, k2"

    返回匹配 id_car_eula_YYYYMMDD.json 的 JSON 结构

    Args:
        left_R: 左相机旋转矩阵 (3x3)
        left_T: 左相机平移向量 (3,)
        right_R: 右相机旋转矩阵 (3x3)
        right_T: 右相机平移向量 (3,)
        left_mtx: 左相机内参矩阵 (3x3)
        left_dist: 左相机畸变系数
        right_mtx: 右相机内参矩阵 (3x3)
        right_dist: 右相机畸变系数
        reproj_error_left: 左相机重投影误差
        reproj_error_right: 右相机重投影误差
        timestamp: 时间戳字符串

    Returns:
        data: 格式化的字典列表
    """
    # 转换旋转矩阵为旧 USB 欧拉角
    euler_left = rotation_matrix_to_legacy_usb_ypr(left_R)  # [yaw, pitch, roll]
    euler_right = rotation_matrix_to_legacy_usb_ypr(right_R)

    # 提取相机中心坐标（旧格式 eula 的 xyz 是相机中心 C，而非 OpenCV 的 t）
    x_left, y_left, z_left = np.asarray(left_C, dtype=float).reshape(3)
    x_right, y_right, z_right = np.asarray(right_C, dtype=float).reshape(3)

    # 提取相机内参
    fx_left, cx_left, cy_left = left_mtx[0, 0], left_mtx[0, 2], left_mtx[1, 2]
    fx_right, cx_right, cy_right = right_mtx[0, 0], right_mtx[0, 2], right_mtx[1, 2]

    # 畸变系数
    k1_left = left_dist[0] if len(left_dist) > 0 else 0.0
    k2_left = left_dist[1] if len(left_dist) > 1 else 0.0
    k1_right = right_dist[0] if len(right_dist) > 0 else 0.0
    k2_right = right_dist[1] if len(right_dist) > 1 else 0.0

    # 格式化字符串 (格式: x, y, z, yaw, pitch, roll, fx, cx, cy, k1, k2)
    left_str = (
        f"{x_left:.4f}, {y_left:.4f}, {z_left:.4f}, "
        f"{euler_left[0]:.4f}, {euler_left[1]:.4f}, {euler_left[2]:.4f}, "
        f"{fx_left:.4f}, {cx_left:.4f}, {cy_left:.4f}, "
        f"{k1_left:.4f}, {k2_left:.4f}"
    )

    right_str = (
        f"{x_right:.4f}, {y_right:.4f}, {z_right:.4f}, "
        f"{euler_right[0]:.4f}, {euler_right[1]:.4f}, {euler_right[2]:.4f}, "
        f"{fx_right:.4f}, {cx_right:.4f}, {cy_right:.4f}, "
        f"{k1_right:.4f}, {k2_right:.4f}"
    )

    date_str = datetime.now().strftime("%m%d_%H%M")

    data = [
        {
            "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date": date_str,
            "cameraLeft": left_str,
            "cameraRight": right_str,
            "minimun_error_left": f"{reproj_error_left:.4f}",
            "minimun_error_right": f"{reproj_error_right:.4f}",
            "dis_error": "0",  # 占位符，需要实际测量
        }
    ]

    return data


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  AprilTag 标定结果转换为旧格式")
    print("=" * 60)

    # 加载标定结果
    print("\n加载标定结果...")

    # 1. 加载内参
    try:
        left_intrinsics = load_json("results/left_intrinsics.json")
        right_intrinsics = load_json("results/right_intrinsics.json")

        K_l = np.array(left_intrinsics["camera_matrix"])
        dist_l = np.array(left_intrinsics["dist_coeffs"])
        K_r = np.array(right_intrinsics["camera_matrix"])
        dist_r = np.array(right_intrinsics["dist_coeffs"])

        reproj_error_left = left_intrinsics.get("reprojection_error", 0.0)
        reproj_error_right = right_intrinsics.get("reprojection_error", 0.0)

        print("  ✓ 内参已加载")

    except FileNotFoundError as e:
        print(f"  ❌ 错误: {e}")
        print("  请先运行 python step3_intrinsic_apriltag.py")
        return

    # 2. 尝试加载相机到底盘的变换
    use_base_frame = False
    try:
        camera_to_base = load_json("results/camera_to_base.json")

        B_T_Cl = np.array(camera_to_base["B_T_Cl"])
        B_T_Cr = np.array(camera_to_base["B_T_Cr"])

        # camera_to_base.json 保存的是 base<-camera (B_T_C)
        # 旧格式 id_car_matrix 需要的是 world(base)->camera 的 OpenCV 外参 (R,t)
        # 旧格式 id_car_eula 的 xyz 需要的是相机中心 C（在 base 下），而不是 t
        R_left, t_left, C_left = invert_base_T_camera_to_world2cam(B_T_Cl)
        R_right, t_right, C_right = invert_base_T_camera_to_world2cam(B_T_Cr)

        print("  ✓ 相机到底盘变换已加载")
        use_base_frame = True

    except FileNotFoundError:
        print("  ⚠ 未找到相机到底盘变换，使用双目外参")

        # 加载双目外参
        try:
            stereo_extrinsics = load_json("results/stereo_extrinsics.json")

            R_stereo = np.array(stereo_extrinsics["R"], dtype=float)
            t_stereo = np.array(stereo_extrinsics["t"], dtype=float).reshape(3)

            # 单位兼容：双目外参 t 常见单位是 mm（baseline 级别 ~ 300-400）。
            # 旧格式与 camera_to_base 使用的是米（0.1~1.0 量级）。
            # 这里用启发式：位移范数>5 时按 mm->m 处理。
            if float(np.linalg.norm(t_stereo)) > 5.0:
                t_stereo = t_stereo / 1000.0

            # world 取左相机坐标系：
            # 左相机: X_L = I*X_world + 0
            R_left = np.eye(3)
            t_left = np.zeros(3)
            C_left = np.zeros(3)

            # 右相机相对于左相机（OpenCV stereoCalibrate 常见语义）：
            #     X_right = R * X_left + t
            # 这里 world 取 left，相当于 world->right 的外参就是 (R, t)
            # 旧格式 eula 需要相机中心 C_right，所以用 C = -R^T t 计算
            R_right = R_stereo
            t_right = t_stereo
            C_right = (-R_right.T @ t_right).reshape(3)

            print("  ✓ 使用双目外参（左相机作为参考系）")
            use_base_frame = False

        except FileNotFoundError as e:
            print(f"  ❌ 错误: {e}")
            print("  请先运行 python step4_stereo_extrinsic.py")
            return

    # 生成日期字符串和时间戳
    now = datetime.now()
    date_string = now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # 格式化输出
    print("\n生成输出文件...")

    # 1. 矩阵格式
    matrix_data = format_matrix_json(
        R_left, t_left, R_right, t_right, K_l, dist_l, K_r, dist_r, timestamp
    )

    matrix_filename = f"id_car_matrix_{date_string}.json"
    matrix_out_path = resolve_path(matrix_filename)
    with open(matrix_out_path, "w") as f:
        json.dump(matrix_data, f, indent=4)
    print(f"  ✓ 已保存 {matrix_out_path}")

    # 2. 欧拉角格式
    eula_data = format_eula_json(
        R_left,
        C_left,
        R_right,
        C_right,
        K_l,
        dist_l,
        K_r,
        dist_r,
        reproj_error_left,
        reproj_error_right,
        timestamp,
    )

    eula_filename = f"id_car_eula_{date_string}.json"
    eula_out_path = resolve_path(eula_filename)
    with open(eula_out_path, "w") as f:
        json.dump(eula_data, f, indent=4)
    print(f"  ✓ 已保存 {eula_out_path}")

    # 打印总结
    print("\n" + "=" * 60)
    print("  转换总结")
    print("=" * 60)

    if use_base_frame:
        print("\n🎯 参考坐标系: 机器人底盘")
        print("\n左相机在底盘坐标系:")
    else:
        print("\n🎯 参考坐标系: 左相机")
        print("\n左相机 (原点):")

    print(f"  位置: [{C_left[0]:.4f}, {C_left[1]:.4f}, {C_left[2]:.4f}] 米")
    euler_l = rotation_matrix_to_legacy_usb_ypr(R_left)
    print(
        f"  旋转 (yaw,pitch,roll): [{euler_l[0]:.4f}, {euler_l[1]:.4f}, {euler_l[2]:.4f}] 弧度"
    )
    print(f"  内参: fx={K_l[0, 0]:.2f}, cx={K_l[0, 2]:.2f}, cy={K_l[1, 2]:.2f}")
    print(f"  重投影误差: {reproj_error_left:.4f} 像素")

    print("\n右相机:")
    print(f"  位置: [{C_right[0]:.4f}, {C_right[1]:.4f}, {C_right[2]:.4f}] 米")
    euler_r = rotation_matrix_to_legacy_usb_ypr(R_right)
    print(
        f"  旋转 (yaw,pitch,roll): [{euler_r[0]:.4f}, {euler_r[1]:.4f}, {euler_r[2]:.4f}] 弧度"
    )
    print(f"  内参: fx={K_r[0, 0]:.2f}, cx={K_r[0, 2]:.2f}, cy={K_r[1, 2]:.2f}")
    print(f"  重投影误差: {reproj_error_right:.4f} 像素")

    if use_base_frame:
        baseline = np.linalg.norm(C_right - C_left)
        print(f"\n📏 有效基线距离: {baseline * 1000:.2f} mm")
    else:
        baseline = np.linalg.norm(C_right - C_left)
        print(f"\n📏 基线距离: {baseline * 1000:.2f} mm")

    print("\n" + "=" * 60)
    print("  输出文件（与旧代码兼容）")
    print("=" * 60)
    print(f"  ✓ {matrix_out_path}")
    print(f"  ✓ {eula_out_path}")
    print("=" * 60 + "\n")

    # 坐标系警告
    print("⚠️  重要: 坐标系注意事项")
    print("  - 旧代码使用左手坐标系")
    print("  - Y轴: 向上为正（世界），向下为正（相机）")
    print("  - Z轴: 向前为正")
    print("  - 请验证坐标系是否匹配你的设置！")
    print()


if __name__ == "__main__":
    main()
