from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from mcca.core.detection import detect_apriltag_corners


def draw_detected_tags(
    image: np.ndarray,
    corners: Optional[List[np.ndarray]],
    ids: Optional[np.ndarray],
    min_tags: int = 4,
) -> Tuple[np.ndarray, bool]:
    """在图像上绘制检测到的 AprilTag 标签。"""
    output_image = image.copy()

    num_detected = 0 if ids is None else len(ids)
    is_valid = num_detected >= min_tags

    if num_detected > 0 and corners is not None:
        cv2.aruco.drawDetectedMarkers(output_image, corners, ids)

        color = (0, 255, 0) if is_valid else (0, 0, 255)
        status = "✓" if is_valid else "✗"
        text = f"{status} 标签数: {num_detected}/{min_tags}"

        cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        text = "✗ 未检测到标签"
        cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return output_image, is_valid


def visualize_board_layout(config: dict, output_path: str = "board_layout.png") -> None:
    """可视化 AprilTag 标定板布局（用于验证配置）。"""
    board_cfg = config["apriltag_board"]
    tags_x = board_cfg["tags_x"]
    tags_y = board_cfg["tags_y"]
    tag_size = board_cfg["tag_size"]
    tag_spacing = board_cfg["tag_spacing"]

    px_per_mm = 10
    tag_pitch = int((tag_size + tag_spacing) * px_per_mm)
    tag_px = int(tag_size * px_per_mm)

    img_width = tags_x * tag_pitch + 100
    img_height = tags_y * tag_pitch + 100

    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    for row in range(tags_y):
        for col in range(tags_x):
            tag_id = row * tags_x + col

            x = 50 + col * tag_pitch
            y = 50 + row * tag_pitch

            cv2.rectangle(img, (x, y), (x + tag_px, y + tag_px), (0, 0, 0), 2)

            text_size = cv2.getTextSize(str(tag_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (tag_px - text_size[0]) // 2
            text_y = y + (tag_px + text_size[1]) // 2
            cv2.putText(
                img,
                str(tag_id),
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    title = f"AprilTag Board: {tags_x}x{tags_y}, Size={tag_size}mm, Spacing={tag_spacing}mm"
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"标定板布局图已保存到: {output_path}")


def analyze_stereo_image_quality(
    left_images: List[str],
    right_images: List[str],
    aruco_dict: cv2.aruco.Dictionary,
    detector_params: cv2.aruco.DetectorParameters,
    use_multiscale: bool = True,
    *,
    opencv_refine: bool = False,
    board: Optional[cv2.aruco.Board] = None,
    left_camera_matrix: Optional[np.ndarray] = None,
    left_dist_coeffs: Optional[np.ndarray] = None,
    right_camera_matrix: Optional[np.ndarray] = None,
    right_dist_coeffs: Optional[np.ndarray] = None,
    left_roi: Optional[Tuple[int, int, int, int]] = None,
    right_roi: Optional[Tuple[int, int, int, int]] = None,
    auto_roi_cfg: Optional[Dict[str, Any]] = None,
    early_stop_min_common_tags: int = 0,
    early_stop_keep_pairs: int = 0,
) -> List[Tuple[str, str, int]]:
    """分析双目图像对的质量（共同检测到的标签数）。"""
    results = []

    stop_min = int(max(0, int(early_stop_min_common_tags)))
    stop_keep = int(max(0, int(early_stop_keep_pairs)))
    keep_hits = 0

    auto_roi_cfg = auto_roi_cfg or {}

    for left_path, right_path in zip(left_images, right_images):
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            results.append((left_path, right_path, 0))
            continue

        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        left_corners, left_ids = detect_apriltag_corners(
            left_gray,
            aruco_dict,
            detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            camera_matrix=left_camera_matrix,
            dist_coeffs=left_dist_coeffs,
            roi=left_roi,
            auto_roi=bool(auto_roi_cfg.get("enabled", False)),
            auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
            auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
            auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
        )
        right_corners, right_ids = detect_apriltag_corners(
            right_gray,
            aruco_dict,
            detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            camera_matrix=right_camera_matrix,
            dist_coeffs=right_dist_coeffs,
            roi=right_roi,
            auto_roi=bool(auto_roi_cfg.get("enabled", False)),
            auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
            auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
            auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
        )

        if left_ids is None or right_ids is None:
            results.append((left_path, right_path, 0))
            continue

        left_set = set(left_ids.flatten())
        right_set = set(right_ids.flatten())
        common = len(left_set & right_set)

        results.append((left_path, right_path, common))

        if stop_min > 0 and stop_keep > 0 and common >= stop_min:
            keep_hits += 1
            if keep_hits >= stop_keep:
                return results

    return results


def filter_low_quality_pairs(
    image_quality: List[Tuple[str, str, int]],
    min_common_tags: int = 15,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, int]]]:
    """根据共同标签数过滤低质量图像对。"""
    keep_pairs = []
    remove_pairs = []

    for left_path, right_path, common in image_quality:
        if common >= min_common_tags:
            keep_pairs.append((left_path, right_path))
        else:
            remove_pairs.append((left_path, right_path, common))

    return keep_pairs, remove_pairs
