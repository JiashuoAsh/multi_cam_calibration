from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def get_aruco_dict(family: str) -> cv2.aruco.Dictionary:
    """根据 AprilTag family 名称获取对应的 ArUco 字典。"""
    family_map = {
        "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }

    if family not in family_map:
        raise ValueError(f"不支持的 AprilTag family: {family}. 支持的类型: {list(family_map.keys())}")

    return cv2.aruco.getPredefinedDictionary(family_map[family])


def create_apriltag_board(config: dict) -> Tuple[np.ndarray, List[int]]:
    """创建 AprilTag 标定板的 3D 角点坐标和 ID 列表。"""
    board_cfg = config["apriltag_board"]
    tags_x = board_cfg["tags_x"]
    tags_y = board_cfg["tags_y"]
    tag_size = board_cfg["tag_size"]  # 单位: mm
    tag_spacing = board_cfg["tag_spacing"]  # 单位: mm

    # 标签中心到中心的距离
    tag_pitch = tag_size + tag_spacing

    obj_points = []
    tag_ids = []

    for row in range(tags_y):
        for col in range(tags_x):
            tag_id = row * tags_x + col
            tag_ids.append(tag_id)

            # 标签中心位置
            center_x = col * tag_pitch
            center_y = row * tag_pitch

            half_size = tag_size / 2.0
            corners = np.array(
                [
                    [center_x - half_size, center_y + half_size, 0],
                    [center_x + half_size, center_y + half_size, 0],
                    [center_x + half_size, center_y - half_size, 0],
                    [center_x - half_size, center_y - half_size, 0],
                ]
            )

            obj_points.append(corners)

    obj_points = np.array(obj_points, dtype=np.float32)

    return obj_points, tag_ids


def create_opencv_aruco_board(
    obj_points: np.ndarray,
    tag_ids: List[int],
    aruco_dict: cv2.aruco.Dictionary,
) -> cv2.aruco.Board:
    """从 AprilTag 板的 3D 角点定义构建 OpenCV 的 Board 对象。"""
    if obj_points is None or len(tag_ids) == 0:
        raise ValueError("obj_points/tag_ids 不能为空")

    obj_points = np.asarray(obj_points)
    if obj_points.ndim != 3 or obj_points.shape[1:] != (4, 3):
        raise ValueError(f"obj_points 形状应为 (N,4,3)，当前: {obj_points.shape}")

    obj_points_list = [obj_points[i].reshape(1, 4, 3).astype(np.float32) for i in range(obj_points.shape[0])]
    ids = np.array(tag_ids, dtype=np.int32).reshape(-1, 1)
    return cv2.aruco.Board(obj_points_list, aruco_dict, ids)
