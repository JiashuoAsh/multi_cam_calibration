#!/usr/bin/env python3
"""实时采集辅助工具 - 显示左右相机检测质量"""

import cv2
import numpy as np
from utils import load_config, get_aruco_dict, detect_apriltag_corners
from libs.camera_wrapper import Camera

def draw_quality_info(frame, side, detected_ids, common_ids, is_good):
    """在图像上绘制质量信息"""
    h, w = frame.shape[:2]

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # 标题
    title_color = (0, 255, 0) if is_good else (0, 165, 255)
    cv2.putText(frame, f"{side} Camera", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, title_color, 2)

    # 检测数量
    det_text = f"Detected: {len(detected_ids) if detected_ids is not None else 0}"
    cv2.putText(frame, det_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 共同标签
    common_count = len(common_ids) if common_ids is not None else 0
    common_color = (0, 255, 0) if common_count >= 20 else ((0, 200, 255) if common_count >= 15 else (0, 0, 255))
    common_text = f"Common: {common_count}"
    cv2.putText(frame, common_text, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, common_color, 2)

    return frame

def main():
    """实时显示采集质量"""
    print("="*70)
    print("会聚式双目采集辅助工具")
    print("="*70)
    print("\n功能:")
    print("  - 实时显示左右相机检测到的标签数量")
    print("  - 显示共同标签数量")
    print("  - 绿色=优秀(≥20), 橙色=良好(≥15), 红色=不足(<15)")
    print("\n操作:")
    print("  [空格] 当共同标签≥20时拍摄")
    print("  [q]    退出")
    print("\n提示:")
    print("  - 标定板要放在两个相机视野的重叠区域中央")
    print("  - 确保每次拍摄时共同标签≥20个（绿色）")
    print("  - 采集至少20-30对优质图像")
    print("="*70)

    config = load_config()
    aruco_dict = get_aruco_dict(config['apriltag_board']['family'])

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

    # 初始化相机
    left_cam = Camera(config['cameras']['left']['id'])
    right_cam = Camera(config['cameras']['right']['id'])

    if not left_cam.isOpened() or not right_cam.isOpened():
        print("❌ 相机打开失败")
        return

    print("\n✅ 相机已就绪，开始预览...")
    print("\n调整标定板位置，使共同标签≥20（显示为绿色）")

    capture_count = 0

    while True:
        # 读取图像
        ret_left, left_frame = left_cam.read()
        ret_right, right_frame = right_cam.read()

        if not ret_left or not ret_right:
            print("❌ 读取图像失败")
            break

        # 检测
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        left_corners, left_ids = detect_apriltag_corners(
            left_gray, aruco_dict, detector_params, use_multiscale=True
        )
        right_corners, right_ids = detect_apriltag_corners(
            right_gray, aruco_dict, detector_params, use_multiscale=True
        )

        # 计算共同标签
        common_ids = None
        if left_ids is not None and right_ids is not None:
            left_set = set(left_ids.flatten())
            right_set = set(right_ids.flatten())
            common_ids = left_set & right_set

        # 判断质量
        is_good = common_ids is not None and len(common_ids) >= 20

        # 绘制检测结果
        if left_ids is not None:
            cv2.aruco.drawDetectedMarkers(left_frame, left_corners, left_ids)
        if right_ids is not None:
            cv2.aruco.drawDetectedMarkers(right_frame, right_corners, right_ids)

        # 绘制质量信息
        left_frame = draw_quality_info(left_frame, "Left", left_ids, common_ids, is_good)
        right_frame = draw_quality_info(right_frame, "Right", right_ids, common_ids, is_good)

        # 拼接显示
        # 调整尺寸以适应屏幕
        scale = 0.5
        left_small = cv2.resize(left_frame, None, fx=scale, fy=scale)
        right_small = cv2.resize(right_frame, None, fx=scale, fy=scale)
        combined = np.hstack([left_small, right_small])

        # 添加整体状态
        status_text = "READY TO CAPTURE!" if is_good else "Adjust board position..."
        status_color = (0, 255, 0) if is_good else (0, 165, 255)
        cv2.putText(combined, status_text, (combined.shape[1]//2-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow('Stereo Calibration Helper', combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # 空格拍摄
            if is_good:
                capture_count += 1
                common_count = len(common_ids) if common_ids else 0
                print(f"✅ 已拍摄 #{capture_count}: {common_count} 个共同标签")
                # 这里可以添加实际的图像保存代码
            else:
                common_count = len(common_ids) if common_ids else 0
                print(f"⚠️  质量不足：只有 {common_count} 个共同标签，需要≥20")

        elif key == ord('q'):  # 退出
            break

    left_cam.release()
    right_cam.release()
    cv2.destroyAllWindows()

    print(f"\n共拍摄 {capture_count} 对图像")
    print("采集完成！")

if __name__ == "__main__":
    main()
