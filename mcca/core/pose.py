from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def estimate_pose_apriltag(
    corners: Optional[List[np.ndarray]],
    ids: Optional[np.ndarray],
    obj_points: np.ndarray,
    tag_ids: List[int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """使用检测到的 AprilTag 标签估计相机位姿。"""
    if corners is None or ids is None or len(ids) == 0:
        return False, None, None

    image_points = []
    object_points = []

    ids_flat = ids.flatten()

    for i, tag_id in enumerate(ids_flat):
        if tag_id in tag_ids:
            idx = tag_ids.index(tag_id)
            obj_pts = obj_points[idx]
            img_pts = corners[i].reshape(-1, 2)

            object_points.append(obj_pts)
            image_points.append(img_pts)

    if len(object_points) == 0:
        return False, None, None

    object_points = np.vstack(object_points).astype(np.float32)
    image_points = np.vstack(image_points).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    return success, rvec, tvec
