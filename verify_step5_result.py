#!/usr/bin/env python3
"""
Verify Step 5 Result: 实时验证相机到底盘的标定结果

功能:
    1. 加载标定结果 (results/camera_to_base.json)
    2. 实时读取相机画面
    3. 检测 AprilTag 标定板
    4. 计算并显示标定板在底盘坐标系中的位置 (X, Y, Z)

使用方法:
    1. 运行脚本: python verify_step5_result.py
    2. 将标定板放置在已知位置（例如：底盘正前方 1.0m 处）
    3. 观察屏幕显示的 "Board in Base" 坐标
    4. 如果显示的坐标与实际位置一致，说明标定成功
    5. 如果有系统误差（例如显示 1.05m，实际 1.00m），则需要调整 config 中的 translation

"""

import cv2
import numpy as np
import argparse
import csv
import json
import os
import time
from scipy.spatial.transform import Rotation
from utils import (
    load_config,
    get_aruco_dict,
    detect_apriltag_corners,
    create_apriltag_board,
    create_opencv_aruco_board,
    get_detection_settings,
    estimate_pose_apriltag,
    init_camera,
)


def _call_if_exists(obj, method_names):
    """按顺序尝试调用对象方法（用于兼容不同相机封装）。

    Returns:
        bool: 调用成功返回 True，否则 False。
    """
    if obj is None:
        return False
    for name in method_names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception:
                return False
    return False


def load_calibration_results(*, camera: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载 Step5 标定结果与对应相机内参。

    Args:
        camera: 相机名（与 results/<cam>_intrinsics.json、Step5 输出 B_T_C 的 key 一致）。

    Returns:
        B_T_C, K, dist
    """
    print("加载标定结果...")

    if not os.path.exists("results/camera_to_base.json"):
        raise FileNotFoundError("未找到 results/camera_to_base.json，请先运行 step5b")

    with open("results/camera_to_base.json", "r", encoding="utf-8") as f:
        base_calib = json.load(f)

    if "B_T_C" not in base_calib or not isinstance(base_calib["B_T_C"], dict):
        raise ValueError("results/camera_to_base.json 缺少 B_T_C 字段（请使用新版 step5b 重新生成）")

    if camera not in base_calib["B_T_C"]:
        cams = sorted(list(base_calib["B_T_C"].keys()))
        raise ValueError(f"Step5 输出里没有相机 {camera}。可用相机：{cams}")

    B_T_C = np.asarray(base_calib["B_T_C"][camera], dtype=np.float64)
    print(f"  ✓ 加载 B_T_C[{camera}]")

    intr_path = f"results/{camera}_intrinsics.json"
    if not os.path.exists(intr_path):
        raise FileNotFoundError(f"未找到 {intr_path}")

    with open(intr_path, "r", encoding="utf-8") as f:
        intrinsics = json.load(f)

    K = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    dist = np.asarray(intrinsics["dist_coeffs"], dtype=np.float64)
    print(f"  ✓ 加载 {camera} 相机内参")

    return B_T_C, K, dist


def _make_transform(rvec, tvec):
    """将 rvec, tvec 转换为 4x4 变换矩阵"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def _matrix_to_xyz_rpy(T):
    """从 4x4 矩阵提取 (x, y, z, roll, pitch, yaw)"""
    x, y, z = T[:3, 3]
    r = Rotation.from_matrix(T[:3, :3])
    roll, pitch, yaw = r.as_euler("xyz", degrees=True)
    return x, y, z, roll, pitch, yaw


def draw_info(img, B_T_T_measured, fps):
    """在图像上绘制信息"""
    h, w = img.shape[:2]

    # 提取坐标
    x, y, z, roll, pitch, yaw = _matrix_to_xyz_rpy(B_T_T_measured)

    # 绘制背景框
    cv2.rectangle(img, (10, 10), (400, 220), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (400, 220), (255, 255, 255), 1)

    # 绘制文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 30
    start_y = 40

    texts = [
        f"FPS: {fps:.1f}",
        "Board in Base Frame (Measured):",
        f"X: {x:.4f} m",
        f"Y: {y:.4f} m",
        f"Z: {z:.4f} m",
        f"R: {roll:.1f} deg",
        f"P: {pitch:.1f} deg",
        f"Y: {yaw:.1f} deg",
    ]

    for i, text in enumerate(texts):
        color = (0, 255, 0) if i > 0 else (0, 255, 255)
        cv2.putText(img, text, (20, start_y + i * line_height), font, 0.7, color, 2)


def _should_headless(args) -> bool:
    """判断是否应禁用 GUI 显示。

    - 无 DISPLAY（常见于机器人/SSH/容器）时，OpenCV HighGUI 往往不可用。
    - 参考 OpenCV HighGUI 文档：imshow 需要 GUI 后端并配合 waitKey/pollKey 才会刷新。
    """
    return True
    if getattr(args, "force_gui", False):
        return False
    if getattr(args, "headless", False):
        return True

    # 自动探测：没 DISPLAY 就默认 headless
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return True

    # 某些环境即使有 DISPLAY，OpenCV 也可能没编译 UI 后端
    try:
        fn = getattr(cv2, "currentUIFramework", None)
        if callable(fn):
            ui = str(fn() or "")
            if ui.strip() == "":
                return True
    except Exception:
        # 保守起见：不因为探测失败就强制 headless
        pass

    return False


def main():
    print("=" * 60)
    print("验证 Step 5 标定结果")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="验证 Step5：相机到底盘外参（支持无显示/headless 模式）")
    parser.add_argument(
        "--camera",
        default="left",
        help="要验证的相机名（默认 left；需与 Step5 输出 B_T_C 的 key 一致）",
    )
    parser.add_argument("--headless", action="store_true", help="禁用窗口显示（无显示设备/SSH 推荐）")
    parser.add_argument("--force_gui", action="store_true", help="强制使用窗口显示（有桌面环境时）")
    parser.add_argument("--max_frames", type=int, default=0, help="处理多少帧后退出（0=直到 Ctrl+C）")
    parser.add_argument("--duration_sec", type=float, default=0.0, help="运行多少秒后退出（0=直到 Ctrl+C）")
    parser.add_argument(
        "--csv_path",
        default="results/verify_step5_log.csv",
        help="输出 CSV 日志路径（默认：results/verify_step5_log.csv）",
    )
    parser.add_argument(
        "--save_dir",
        default="results/verify_step5_frames",
        help="保存抽帧图片目录（默认：results/verify_step5_frames）",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="每 N 帧保存一张可视化图片（0=不保存；建议 30~100）",
    )
    parser.add_argument("--scale", type=float, default=0.5, help="显示/保存的缩放比例（默认 0.5）")
    parser.add_argument("--no_draw", action="store_true", help="不绘制检测框/坐标轴（更快）")
    parser.add_argument("--print_every", type=int, default=10, help="每 N 帧在终端打印一次（默认 10）")
    args = parser.parse_args()

    headless = _should_headless(args)
    if headless:
        print("\n[Headless] 检测到无可用显示后端，已禁用窗口显示：将通过 CSV/抽帧图片进行验证。")

    camera = None
    csv_f = None
    csv_writer = None
    t_start = time.time()
    detected_rows = 0
    trans_samples = []  # (x,y,z)
    rpy_samples = []  # (roll,pitch,yaw)

    try:
        # 1. 加载配置和标定数据
        config = load_config()
        use_multiscale, opencv_refine = get_detection_settings(config)
        B_T_Cl, K_l, dist_l = load_calibration_results(camera=str(args.camera))

        # 准备 AprilTag 数据
        obj_points_mm, tag_ids = create_apriltag_board(config)
        obj_points = obj_points_mm.astype(np.float64) / 1000.0
        aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])
        board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # 2. 初始化相机
        print("\n初始化相机...")
        camera = init_camera(config)
        # 说明：本工程的 init_camera() 会在内部完成 camera.open() 并启动采集。
        # 部分外部封装可能存在 start()/start_capture()，这里做兼容性尝试，但不强依赖。
        _call_if_exists(camera, ["start", "start_capture", "run"])
        time.sleep(1.0) # 等待相机稳定

        print("\n开始验证循环 (按 'q' 退出)...")
        print("-" * 60)
        print(f"{'X(m)':>10} {'Y(m)':>10} {'Z(m)':>10} | {'Roll':>8} {'Pitch':>8} {'Yaw':>8}")
        print("-" * 60)

        # 3. 打开 CSV
        if args.csv_path:
            os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
            csv_f = open(args.csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_f)
            csv_writer.writerow(
                [
                    "timestamp_sec",
                    "frame",
                    "detected",
                    "num_tags",
                    "x_m",
                    "y_m",
                    "z_m",
                    "roll_deg",
                    "pitch_deg",
                    "yaw_deg",
                    "fps",
                ]
            )

        last_time = time.time()
        frame_count = 0

        while True:
            # 自动退出条件（headless 很重要，否则只能 Ctrl+C）
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            if args.duration_sec > 0 and (time.time() - t_start) >= args.duration_sec:
                break

            # 读取图像
            left_frame, right_frame, timestamp = camera.read_stereo()
            if left_frame is None:
                print("无法读取图像")
                break

            # 只需要左图进行验证
            img = left_frame.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 检测 AprilTag
            corners, ids = detect_apriltag_corners(
                gray,
                aruco_dict,
                detector_params,
                use_multiscale=use_multiscale,
                opencv_refine=opencv_refine,
                board=board,
                camera_matrix=K_l,
                dist_coeffs=dist_l,
            )

            # 估计位姿
            success, rvec, tvec = estimate_pose_apriltag(
                corners, ids, obj_points, tag_ids, K_l, dist_l
            )

            if success:
                assert rvec is not None and tvec is not None
                # 1. 计算 Cl_T_T (Board to Camera Left)
                Cl_T_T = _make_transform(rvec, tvec)

                # 2. 计算 B_T_T (Board to Base) = B_T_Cl @ Cl_T_T
                B_T_T_measured = B_T_Cl @ Cl_T_T

                # 3. 计算 FPS
                current_time = time.time()
                fps = 1.0 / (current_time - last_time)
                last_time = current_time

                x, y, z, r, p, yaw_val = _matrix_to_xyz_rpy(B_T_T_measured)
                detected_rows += 1
                trans_samples.append((x, y, z))
                rpy_samples.append((r, p, yaw_val))

                if csv_writer is not None:
                    num_tags = int(len(ids)) if ids is not None else 0
                    csv_writer.writerow(
                        [
                            float(timestamp) if timestamp is not None else "",
                            int(frame_count),
                            1,
                            num_tags,
                            float(x),
                            float(y),
                            float(z),
                            float(r),
                            float(p),
                            float(yaw_val),
                            float(fps),
                        ]
                    )

                # 4. 绘制信息
                if not args.no_draw:
                    if corners is not None and ids is not None:
                        cv2.aruco.drawDetectedMarkers(img, corners, ids)
                    cv2.drawFrameAxes(img, K_l, dist_l, rvec, tvec, 0.1) # 绘制坐标轴 (0.1m)
                    draw_info(img, B_T_T_measured, fps)

                # 5. 终端打印 (每10帧打印一次，避免刷屏)
                frame_count += 1
                if args.print_every > 0 and frame_count % args.print_every == 0:
                    print(f"\r{x:10.4f} {y:10.4f} {z:10.4f} | {r:8.1f} {p:8.1f} {yaw_val:8.1f}", end="")
            else:
                frame_count += 1
                if csv_writer is not None:
                    num_tags = int(len(ids)) if ids is not None else 0
                    csv_writer.writerow(
                        [
                            float(timestamp) if timestamp is not None else "",
                            int(frame_count),
                            0,
                            num_tags,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
                cv2.putText(img, "No Tag Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 抽帧保存（headless 场景下用于离线查看）
            if args.save_every and args.save_every > 0 and (frame_count % args.save_every == 0):
                os.makedirs(args.save_dir, exist_ok=True)
                s = float(args.scale) if args.scale and args.scale > 0 else 1.0
                if s != 1.0:
                    out_img = cv2.resize(img, (0, 0), fx=s, fy=s)
                else:
                    out_img = img
                out_path = os.path.join(args.save_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(out_path, out_img)

            # 显示（如果可用）
            if not headless:
                try:
                    s = float(args.scale) if args.scale and args.scale > 0 else 1.0
                    display_img = cv2.resize(img, (0, 0), fx=s, fy=s) if s != 1.0 else img
                    cv2.imshow("Verify Step 5 (Left Camera)", display_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    # 运行时发现 HighGUI 不可用：自动降级为 headless
                    headless = True
                    print("\n[Headless] 当前环境不支持 cv2.imshow，已自动切换为 headless（继续记录 CSV/保存抽帧）。")

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 汇总（便于 headless 验证）
        if detected_rows > 0:
            t_arr = np.asarray(trans_samples, dtype=np.float64)
            rpy_arr = np.asarray(rpy_samples, dtype=np.float64)
            t_mean = t_arr.mean(axis=0)
            t_std = t_arr.std(axis=0)
            rpy_mean = rpy_arr.mean(axis=0)
            rpy_std = rpy_arr.std(axis=0)
            print("\n\n--- 统计（仅统计检测成功帧）---")
            print(f"检测成功帧数: {detected_rows}")
            print(f"平移均值 (m):  x={t_mean[0]:.4f}, y={t_mean[1]:.4f}, z={t_mean[2]:.4f}")
            print(f"平移Std  (m):  x={t_std[0]:.4f}, y={t_std[1]:.4f}, z={t_std[2]:.4f}")
            print(f"角度均值 (deg): roll={rpy_mean[0]:.2f}, pitch={rpy_mean[1]:.2f}, yaw={rpy_mean[2]:.2f}")
            print(f"角度Std  (deg): roll={rpy_std[0]:.2f}, pitch={rpy_std[1]:.2f}, yaw={rpy_std[2]:.2f}")

        if csv_f is not None:
            try:
                csv_f.flush()
                csv_f.close()
                print(f"CSV 已写入: {args.csv_path}")
            except Exception:
                pass

        if camera is not None:
            # 统一释放接口：BaseCameraWrapper.release()
            # 同时兼容部分封装可能叫 stop()/close()。
            if not _call_if_exists(camera, ["release", "stop", "close", "stop_capture"]):
                # 不再抛异常，避免遮蔽主流程中的错误
                pass
        if not headless:
            cv2.destroyAllWindows()
        print("\n验证结束")

if __name__ == "__main__":
    main()
