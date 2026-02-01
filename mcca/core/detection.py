from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# 显式依赖：多尺度检测实现位于 core/apriltag_detector.py。
# 之前的函数内 import 会隐藏依赖关系，也会让单测难以 monkeypatch。
from mcca.core.apriltag_detector import detect_apriltag_multiscale


def get_detection_settings(
    config: dict,
    *,
    default_use_multiscale: bool = True,
    default_opencv_refine: bool = False,
) -> Tuple[bool, bool]:
    """从配置文件中读取 AprilTag 检测相关开关。"""
    calib_cfg = config.get("calibration_settings", {}) if isinstance(config, dict) else {}
    det_cfg = calib_cfg.get("detection", {}) if isinstance(calib_cfg, dict) else {}

    use_multiscale = det_cfg.get("use_multiscale", default_use_multiscale)
    opencv_refine = det_cfg.get("opencv_refine", default_opencv_refine)
    return bool(use_multiscale), bool(opencv_refine)


def get_detection_profile(config: dict, *, default: str = "balanced") -> str:
    """获取检测 profile。"""
    try:
        det_cfg = (config or {}).get("calibration_settings", {}).get("detection", {})
        profile = det_cfg.get("profile", default)
        if not isinstance(profile, str) or not profile.strip():
            return default
        return profile.strip()
    except Exception:
        return default


def get_detection_roi(
    config: dict,
    *,
    camera: Optional[str] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """从配置中获取检测 ROI（xywh）。"""
    try:
        det_cfg: Any = (config or {}).get("calibration_settings", {}).get("detection", {})
        roi_cfg: Any = det_cfg.get("roi", None)
        if roi_cfg is None:
            return None

        if isinstance(roi_cfg, dict) and camera is not None and camera in roi_cfg:
            roi_cfg = roi_cfg[camera]

        if isinstance(roi_cfg, dict):
            x0 = roi_cfg.get("x", None)
            y0 = roi_cfg.get("y", None)
            w0 = roi_cfg.get("w", None)
            h0 = roi_cfg.get("h", None)
            if x0 is None or y0 is None or w0 is None or h0 is None:
                return None
            x = int(x0)
            y = int(y0)
            w = int(w0)
            h = int(h0)
            return x, y, w, h

        if isinstance(roi_cfg, (list, tuple)) and len(roi_cfg) == 4:
            x, y, w, h = [int(v) for v in roi_cfg]
            return x, y, w, h

        return None
    except Exception:
        return None


def get_detection_auto_roi(config: dict) -> Dict[str, Any]:
    """从配置中获取自动 ROI（两阶段检测）参数。"""
    out: Dict[str, Any] = {
        "enabled": False,
        "pre_scale": 0.5,
        "min_tags": 1,
        "margin": 0.25,
    }
    try:
        det_cfg = (config or {}).get("calibration_settings", {}).get("detection", {})
        auto_cfg = det_cfg.get("auto_roi", False) if isinstance(det_cfg, dict) else False

        if isinstance(auto_cfg, bool):
            out["enabled"] = bool(auto_cfg)
            return out

        if isinstance(auto_cfg, dict):
            out["enabled"] = bool(auto_cfg.get("enabled", True))
            if "pre_scale" in auto_cfg:
                v = auto_cfg.get("pre_scale", None)
                if v is not None:
                    out["pre_scale"] = float(v)
            if "min_tags" in auto_cfg:
                v = auto_cfg.get("min_tags", None)
                if v is not None:
                    out["min_tags"] = int(v)
            if "margin" in auto_cfg:
                v = auto_cfg.get("margin", None)
                if v is not None:
                    out["margin"] = float(v)

            # 轻量夹紧，避免奇怪配置导致行为异常
            try:
                out["pre_scale"] = float(np.clip(float(out.get("pre_scale", 0.5)), 0.20, 1.0))
            except Exception:
                out["pre_scale"] = 0.5
            try:
                out["min_tags"] = int(max(1, int(out.get("min_tags", 1))))
            except Exception:
                out["min_tags"] = 1
            try:
                out["margin"] = float(np.clip(float(out.get("margin", 0.25)), 0.0, 1.0))
            except Exception:
                out["margin"] = 0.25
            return out

        return out
    except Exception:
        return out


def compute_roi_from_corners(
    corners: List[np.ndarray],
    image_shape_hw: Tuple[int, int],
    *,
    margin: float = 0.25,
) -> Optional[Tuple[int, int, int, int]]:
    """根据检测到的角点估计一个 ROI（xywh）。"""
    try:
        if corners is None or len(corners) == 0:
            return None
        h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
        if h <= 0 or w <= 0:
            return None

        pts = []
        for c in corners:
            c2 = np.asarray(c, dtype=np.float32).reshape(-1, 2)
            if c2.size == 0:
                continue
            pts.append(c2)
        if len(pts) == 0:
            return None

        pts_all = np.vstack(pts)
        x0, y0 = np.min(pts_all, axis=0)
        x1, y1 = np.max(pts_all, axis=0)

        bw = max(1.0, float(x1 - x0))
        bh = max(1.0, float(y1 - y0))
        m = float(margin)
        if not np.isfinite(m):
            m = 0.25
        m = float(np.clip(m, 0.0, 1.0))
        mx = m * bw
        my = m * bh

        rx0 = int(np.floor(max(0.0, float(x0) - mx)))
        ry0 = int(np.floor(max(0.0, float(y0) - my)))
        rx1 = int(np.ceil(min(float(w), float(x1) + mx)))
        ry1 = int(np.ceil(min(float(h), float(y1) + mx)))

        rw = max(1, rx1 - rx0)
        rh = max(1, ry1 - ry0)
        return rx0, ry0, rw, rh
    except Exception:
        return None


def create_detector_params(
    config: Optional[dict] = None,
    *,
    profile: Optional[str] = None,
) -> cv2.aruco.DetectorParameters:
    """创建并调参 DetectorParameters。"""
    params = cv2.aruco.DetectorParameters()

    cfg = config or {}
    calib_cfg = cfg.get("calibration_settings", {}) if isinstance(cfg, dict) else {}

    # 角点精修策略：优先跟随配置；否则默认 SUBPIX（工程当前默认更稳）。
    corner_ref = calib_cfg.get("corner_refinement", "CORNER_REFINE_SUBPIX")
    if isinstance(corner_ref, str):
        corner_ref = corner_ref.strip().upper()
    else:
        corner_ref = "CORNER_REFINE_SUBPIX"

    if corner_ref == "CORNER_REFINE_NONE":
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    elif corner_ref == "CORNER_REFINE_CONTOUR":
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    elif corner_ref == "CORNER_REFINE_APRILTAG":
        # 注意：这是 OpenCV 的 AprilTag2 检测策略（不是单纯“精修”）；
        # 在部分图像上可能 0 检测。本工程有失败回退机制。
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    else:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # 选择 profile：优先使用参数 profile，其次使用配置。
    prof_raw: Any = profile if profile is not None else get_detection_profile(cfg)
    if isinstance(prof_raw, str) and prof_raw.strip():
        prof = prof_raw.strip().lower()
    else:
        prof = "balanced"

    if prof in {"small", "small_tag", "small_tags", "tiny", "aggressive"}:
        # === 针对“小Tag/远距离”的经验调参（偏召回） ===
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 17
        params.adaptiveThreshWinSizeStep = 2

        params.minMarkerPerimeterRate = 0.015
        params.maxMarkerPerimeterRate = 4.0

        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.20
        params.minOtsuStdDev = 2.0

        params.errorCorrectionRate = 0.8

        if hasattr(params, "aprilTagQuadDecimate"):
            params.aprilTagQuadDecimate = 1.0
        if hasattr(params, "aprilTagQuadSigma"):
            params.aprilTagQuadSigma = 0.0
        if hasattr(params, "aprilTagMinClusterPixels"):
            params.aprilTagMinClusterPixels = 3
        if hasattr(params, "aprilTagMinWhiteBlackDiff"):
            params.aprilTagMinWhiteBlackDiff = 2

    else:
        # === balanced：轻微增强召回，但尽量不显著增加耗时 ===
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 5
        params.minMarkerPerimeterRate = 0.03
        params.maxMarkerPerimeterRate = 4.0

        if hasattr(params, "aprilTagQuadDecimate"):
            params.aprilTagQuadDecimate = 1.5

    # 允许在 config 中进一步覆写 DetectorParameters（可选）
    det_cfg = calib_cfg.get("detection", {}) if isinstance(calib_cfg, dict) else {}
    overrides = det_cfg.get("detector_params", None) if isinstance(det_cfg, dict) else None
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            if not isinstance(k, str):
                continue
            if not hasattr(params, k):
                continue
            try:
                setattr(params, k, v)
            except Exception:
                pass

    return params


def detect_apriltag_corners(
    image: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    detector_params: Optional[cv2.aruco.DetectorParameters] = None,
    use_multiscale: bool = False,
    *,
    opencv_refine: bool = False,
    board: Optional[cv2.aruco.Board] = None,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    auto_roi: bool = False,
    auto_roi_pre_scale: float = 0.5,
    auto_roi_min_tags: int = 1,
    auto_roi_margin: float = 0.25,
) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
    """检测图像中的 AprilTag 标签及其角点。"""

    if auto_roi and roi is None:
        try:
            h0, w0 = int(image.shape[0]), int(image.shape[1])

            pre_scale = float(auto_roi_pre_scale)
            if not np.isfinite(pre_scale):
                pre_scale = 0.5
            pre_scale = float(np.clip(pre_scale, 0.20, 1.0))

            if max(h0, w0) <= 900:
                pre_scale = 1.0

            if pre_scale < 1.0:
                sw = max(10, int(round(w0 * pre_scale)))
                sh = max(10, int(round(h0 * pre_scale)))
                img_small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
            else:
                img_small = image

            dp = detector_params if detector_params is not None else create_detector_params(config=None)
            det = cv2.aruco.ArucoDetector(aruco_dict, dp)
            c_s, id_s, rej_s = det.detectMarkers(img_small)

            if (
                (id_s is None or len(id_s) == 0)
                and hasattr(dp, "cornerRefinementMethod")
                and dp.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_APRILTAG
            ):
                try:
                    orig = dp.cornerRefinementMethod
                    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                    det_fb = cv2.aruco.ArucoDetector(aruco_dict, dp)
                    c2, id2, rej2 = det_fb.detectMarkers(img_small)
                    dp.cornerRefinementMethod = orig
                    if id2 is not None and len(id2) > 0:
                        c_s, id_s, rej_s = c2, id2, rej2
                except Exception:
                    pass

            if (
                board is not None
                and opencv_refine
                and rej_s is not None
                and len(rej_s) > 0
                and hasattr(det, "refineDetectedMarkers")
            ):
                try:
                    c_s, id_s, rej_s, _ = det.refineDetectedMarkers(img_small, board, c_s, id_s, rej_s)
                except Exception:
                    pass

            n_s = 0 if id_s is None else int(len(id_s))
            if c_s is not None and id_s is not None and n_s >= int(max(1, auto_roi_min_tags)):
                roi_s = compute_roi_from_corners(
                    list(c_s) if not isinstance(c_s, list) else c_s,
                    (int(img_small.shape[0]), int(img_small.shape[1])),
                    margin=float(auto_roi_margin),
                )
                if roi_s is not None:
                    rx, ry, rw, rh = roi_s
                    inv = 1.0 / float(pre_scale)
                    rx0 = int(round(rx * inv))
                    ry0 = int(round(ry * inv))
                    rw0 = int(round(rw * inv))
                    rh0 = int(round(rh * inv))

                    rx0 = max(0, min(rx0, w0 - 1))
                    ry0 = max(0, min(ry0, h0 - 1))
                    rw0 = max(1, min(rw0, w0 - rx0))
                    rh0 = max(1, min(rh0, h0 - ry0))

                    if rw0 >= 20 and rh0 >= 20:
                        roi = (rx0, ry0, rw0, rh0)
        except Exception:
            pass

    offset_x = 0
    offset_y = 0
    img_in = image
    if roi is not None:
        try:
            x, y, w, h = roi
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(int(image.shape[1]), x0 + max(0, int(w)))
            y1 = min(int(image.shape[0]), y0 + max(0, int(h)))
            if (x1 - x0) >= 10 and (y1 - y0) >= 10:
                img_in = image[y0:y1, x0:x1]
                offset_x, offset_y = x0, y0
        except Exception:
            img_in = image

    if use_multiscale:
        # 使用多尺度检测（推荐）
        corners, ids = detect_apriltag_multiscale(
            img_in,
            aruco_dict,
            detector_params,
            verbose=True,
            opencv_refine=opencv_refine,
            board=board,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )

        if corners is not None and not isinstance(corners, list):
            corners = list(corners)
        if ids is not None:
            ids = np.asarray(ids)

        if corners is not None and (offset_x != 0 or offset_y != 0):
            try:
                corners_list = corners if isinstance(corners, list) else list(corners)
                for i in range(len(corners_list)):
                    c = np.asarray(corners_list[i], dtype=np.float32)
                    c = c.copy()
                    c[..., 0] += float(offset_x)
                    c[..., 1] += float(offset_y)
                    corners_list[i] = c
                corners = corners_list
            except Exception:
                pass

        return corners, ids

    if detector_params is None:
        detector_params = create_detector_params(config=None)

    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, rejected = detector.detectMarkers(img_in)

    if (
        (ids is None or len(ids) == 0)
        and hasattr(detector_params, "cornerRefinementMethod")
        and detector_params.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_APRILTAG
    ):
        try:
            orig = detector_params.cornerRefinementMethod
            detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            detector_fallback = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
            corners2, ids2, rejected2 = detector_fallback.detectMarkers(img_in)
            detector_params.cornerRefinementMethod = orig
            if ids2 is not None and len(ids2) > 0:
                corners, ids, rejected = corners2, ids2, rejected2
        except Exception:
            pass

    if (
        opencv_refine
        and board is not None
        and rejected is not None
        and len(rejected) > 0
        and hasattr(detector, "refineDetectedMarkers")
    ):
        try:
            if camera_matrix is not None and dist_coeffs is not None:
                corners, ids, rejected, _ = detector.refineDetectedMarkers(
                    img_in,
                    board,
                    corners,
                    ids,
                    rejected,
                    camera_matrix,
                    dist_coeffs,
                )
            else:
                corners, ids, rejected, _ = detector.refineDetectedMarkers(
                    img_in,
                    board,
                    corners,
                    ids,
                    rejected,
                )
        except Exception:
            pass

    if corners is not None and len(corners) > 0:
        corrected_corners = []
        for corner in corners:
            pts = corner.reshape(4, 2)
            corrected_pts = np.array([pts[2], pts[3], pts[0], pts[1]])
            corrected_corners.append(corrected_pts.reshape(1, 4, 2))
        corners = corrected_corners

    if corners is not None and (offset_x != 0 or offset_y != 0):
        try:
            corners_list = corners if isinstance(corners, list) else list(corners)
            for i in range(len(corners_list)):
                c = np.asarray(corners_list[i], dtype=np.float32)
                c = c.copy()
                c[..., 0] += float(offset_x)
                c[..., 1] += float(offset_y)
                corners_list[i] = c
            corners = corners_list
        except Exception:
            pass

    if corners is not None and not isinstance(corners, list):
        corners = list(corners)

    if ids is not None:
        ids = np.asarray(ids)

    return corners, ids
