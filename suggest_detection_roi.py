#!/usr/bin/env python3
"""根据一张样例图自动建议 detection ROI（用于固定相机/固定标定板位置的小Tag场景）。

动机：
- 小Tag检测的第一杀手是“有效像素太少”。
- 直接整图 2x/3x 会很慢且更容易被背景干扰。
- 先锁定标定板所在区域（ROI），再在 ROI 上做更激进的多尺度/上采样，召回率会明显上升。

用法示例：
    python suggest_detection_roi.py --image images/raw/cam0/frame_000120.png --camera cam0 --write

输出：
- 在终端打印建议 ROI（[x,y,w,h]）
- 可选：写回 config/apriltag_config.json 的 calibration_settings.detection.roi

注意：
- 该脚本只“建议 ROI”，不会改变标定板物理尺寸/相机位置。
- ROI 建议基于当前检测到的 tags 的外接框，并按 margin 扩张。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from utils import (
    load_config,
    get_aruco_dict,
    create_detector_params,
    detect_apriltag_corners,
    create_apriltag_board,
    create_opencv_aruco_board,
)


def _compute_roi_from_corners(
    corners: list[np.ndarray],
    image_shape: Tuple[int, int],
    *,
    margin: float,
) -> Tuple[int, int, int, int]:
    h, w = image_shape
    pts = []
    for c in corners:
        c2 = np.asarray(c, dtype=np.float32).reshape(-1, 2)
        if c2.size == 0:
            continue
        pts.append(c2)
    if len(pts) == 0:
        raise ValueError("no corner points")

    pts_all = np.vstack(pts)
    x0, y0 = np.min(pts_all, axis=0)
    x1, y1 = np.max(pts_all, axis=0)

    bw = max(1.0, float(x1 - x0))
    bh = max(1.0, float(y1 - y0))
    mx = float(margin) * bw
    my = float(margin) * bh

    rx0 = int(np.floor(max(0.0, x0 - mx)))
    ry0 = int(np.floor(max(0.0, y0 - my)))
    rx1 = int(np.ceil(min(float(w), x1 + mx)))
    ry1 = int(np.ceil(min(float(h), y1 + my)))

    rw = max(1, rx1 - rx0)
    rh = max(1, ry1 - ry0)
    return rx0, ry0, rw, rh


def _update_config_roi(config: Dict[str, Any], roi_xywh, camera: Optional[str]) -> None:
    det_cfg = config.setdefault("calibration_settings", {}).setdefault("detection", {})
    if camera is None:
        det_cfg["roi"] = list(map(int, roi_xywh))
        return

    roi_cfg = det_cfg.get("roi")
    if not isinstance(roi_cfg, dict):
        roi_cfg = {}
    roi_cfg[str(camera)] = list(map(int, roi_xywh))
    det_cfg["roi"] = roi_cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="根据样例图自动建议 AprilTag 检测 ROI")
    ap.add_argument("--image", required=True, help="输入图像路径（建议每路相机各选一张代表性图）")
    ap.add_argument(
        "--camera",
        default=None,
        help="可选：写回配置时写到 detection.roi.<camera>（例如 cam0）",
    )
    ap.add_argument(
        "--config",
        default="config/apriltag_config.json",
        help="配置文件路径（默认 config/apriltag_config.json）",
    )
    ap.add_argument(
        "--profile",
        default="small_tags",
        help="检测 profile（默认 small_tags，更适合小Tag）",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=0.18,
        help="ROI 外扩边距（相对检测外接框宽高的比例，默认 0.18）",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="将建议 ROI 写回配置文件",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    config = load_config(str(cfg_path))

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"failed to read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # 构建字典与 board（用于 refineDetectedMarkers，提高“差一点解码成功”的召回）
    board_cfg = config.get("apriltag_board", {})
    family = board_cfg.get("family", "tag36h11")
    aruco_dict = get_aruco_dict(str(family))
    obj_points_mm, tag_ids = create_apriltag_board(config)
    board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)

    # 使用更偏召回的参数 profile
    detector_params = create_detector_params(config, profile=str(args.profile))

    corners, ids = detect_apriltag_corners(
        gray,
        aruco_dict,
        detector_params,
        use_multiscale=True,
        opencv_refine=True,
        board=board,
        roi=None,
    )

    n = 0 if ids is None else int(len(ids))
    print(f"Detected tags: {n}")
    if corners is None or ids is None or len(ids) == 0:
        print("无法检测到任何 tag：建议先用更近/更清晰的样例图生成 ROI；或先在 config 里把 profile 设为 small_tags 再试。")
        return 2

    roi = _compute_roi_from_corners(corners, (gray.shape[0], gray.shape[1]), margin=float(args.margin))
    roi_list = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]

    print("\nSuggested ROI (x,y,w,h):")
    print(json.dumps(roi_list, ensure_ascii=False))

    if args.write:
        _update_config_roi(config, roi_list, args.camera)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"\n已写回配置: {cfg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
