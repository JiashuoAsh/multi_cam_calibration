"""
AprilTag 标定板工具函数库

本模块提供 AprilTag 标定板检测和标定所需的核心函数。

主要功能:
    - 配置文件加载和解析
    - ArUco 字典创建和管理
    - AprilTag 标定板3D点生成
    - AprilTag 检测和角点提取
    - 相机位姿估计
    - 检测结果可视化
    - 相机初始化

使用示例:
    >>> config = load_config('config/apriltag_config.json')
    >>> aruco_dict = get_aruco_dict('tag36h11')
    >>> obj_points, tag_ids = create_apriltag_board(config)
    >>> corners, ids = detect_apriltag_corners(gray_img, aruco_dict)

作者: GitHub Copilot
日期: 2025-12-12
版本: 2.0
"""

import json
import glob
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union


_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")


# region 图片文件工具（内部）


def _list_images_in_dir(dir_path: Path) -> List[Path]:
    """列出目录下常见图片文件（按文件名排序）。"""
    out: List[Path] = []
    try:
        if not dir_path.exists():
            return []
        for p in sorted(dir_path.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() in _IMAGE_EXTS:
                out.append(p)
        return out
    except Exception:
        return []


def _expand_image_globs(patterns: List[str]) -> List[Path]:
    """展开 glob（支持绝对/相对路径），并过滤为常见图片后缀。"""
    out: List[Path] = []
    for pat in patterns:
        if not isinstance(pat, str) or not pat.strip():
            continue
        for s in sorted(glob.glob(pat, recursive=True)):
            p = Path(s)
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                out.append(p)
    # 去重但保持排序稳定
    uniq: List[Path] = []
    seen: set[str] = set()
    for p in out:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


# endregion


# region 数据集：Step2/3/4（image_dataset）


def get_image_dataset(config: dict) -> Dict[str, Any]:
    """解析 config.image_dataset（多相机图片数据集输入）。

    返回标准化结构（即使未启用也会返回默认值）。

    约定（可选字段）：
      config["image_dataset"] = {
        "enabled": bool,
        "raw_root": "images/raw",
        "filtered_root": "images/filtered",
        "sync": {"key": "stem", "mode": "intersection"},
        "cameras": {
          "cam0": {"raw_dir": "..."} 或 {"raw_glob": ".../*.png"} 或 {"raw_glob": ["..."]},
          ...
        }
      }
    """
    ds = (config or {}).get("image_dataset", {})
    if not isinstance(ds, dict):
        ds = {}

    enabled = bool(ds.get("enabled", False))
    raw_root = Path(str(ds.get("raw_root", "images/raw")))
    filtered_root = Path(str(ds.get("filtered_root", "images/filtered")))

    sync = ds.get("sync", {})
    if not isinstance(sync, dict):
        sync = {}
    sync_key = str(sync.get("key", "stem")).strip() or "stem"
    sync_mode = str(sync.get("mode", "intersection")).strip().lower() or "intersection"
    if sync_mode not in {"intersection", "union"}:
        sync_mode = "intersection"

    cams_cfg_any = ds.get("cameras", {})
    cam_cfgs: Dict[str, Dict[str, Any]] = {}
    cameras: List[str] = []

    if isinstance(cams_cfg_any, dict):
        for cam, cfg in cams_cfg_any.items():
            if not isinstance(cam, str) or not cam.strip():
                continue
            cam_name = cam.strip()
            cam_cfgs[cam_name] = cfg if isinstance(cfg, dict) else {}
            cameras.append(cam_name)

    # 允许写成 cameras: ["cam0", "cam1", ...]（此时使用 raw_root/<cam>）
    if isinstance(cams_cfg_any, list):
        cameras = []
        cam_cfgs = {}
        for cam in cams_cfg_any:
            if isinstance(cam, str) and cam.strip():
                cam_name = cam.strip()
                cameras.append(cam_name)
                cam_cfgs[cam_name] = {}

    return {
        "enabled": enabled,
        "raw_root": raw_root,
        "filtered_root": filtered_root,
        "sync_key": sync_key,
        "sync_mode": sync_mode,
        "cameras": cameras,
        "camera_cfgs": cam_cfgs,
    }


def get_dataset_cameras(
    config: dict,
    *,
    allow_scan: bool = True,
    fallback_stereo: bool = True,
) -> List[str]:
    """获取“需要处理的相机列表”。

    优先级：
      1) image_dataset.cameras（dict keys 或 list）
      2) 若 allow_scan：扫描 image_dataset.raw_root 下的子目录（包含图片）
      3) 若 fallback_stereo：返回 ["left", "right"]

    说明：
      - 该函数不强依赖 image_dataset.enabled；因为用户可能希望“写了 cameras 就生效”。
      - Step2/3/4 会根据 enabled 决定是否走新逻辑或旧逻辑。
    """
    ds = get_image_dataset(config)
    cams = ds.get("cameras", [])
    if isinstance(cams, list) and len(cams) > 0:
        return [str(c) for c in cams]

    if allow_scan:
        raw_root = ds.get("raw_root", Path("images/raw"))
        if not isinstance(raw_root, Path):
            raw_root = Path(str(raw_root))
        found: List[str] = []
        try:
            if raw_root.exists():
                for p in sorted(raw_root.iterdir()):
                    if not p.is_dir():
                        continue
                    if len(_list_images_in_dir(p)) > 0:
                        found.append(p.name)
        except Exception:
            found = []
        if len(found) > 0:
            return found

    if fallback_stereo:
        return ["left", "right"]
    return []


def get_camera_raw_images(config: dict, cam: str) -> List[Path]:
    """根据 image_dataset 配置返回某相机 raw 图片列表。

    优先级：raw_glob > raw_dir > raw_root/<cam>
    若 image_dataset 未启用/未配置，也会按 raw_root/<cam> 回退（默认 images/raw/<cam>）。
    """
    ds = get_image_dataset(config)
    raw_root: Path = ds["raw_root"]
    cam_cfgs: Dict[str, Dict[str, Any]] = ds["camera_cfgs"]
    cfg = cam_cfgs.get(cam, {}) if isinstance(cam_cfgs, dict) else {}

    raw_glob_any = cfg.get("raw_glob", None)
    if isinstance(raw_glob_any, str) and raw_glob_any.strip():
        return _expand_image_globs([raw_glob_any.strip()])
    if isinstance(raw_glob_any, list):
        pats = [str(x).strip() for x in raw_glob_any if isinstance(x, (str, int, float)) and str(x).strip()]
        if len(pats) > 0:
            return _expand_image_globs(pats)

    raw_dir_any = cfg.get("raw_dir", None)
    if isinstance(raw_dir_any, str) and raw_dir_any.strip():
        return _list_images_in_dir(Path(raw_dir_any.strip()))

    return _list_images_in_dir(raw_root / cam)


def get_camera_filtered_dir(config: dict, cam: str) -> Path:
    """根据 image_dataset 配置返回某相机 filtered 输出目录。

    优先级：filtered_dir（每相机）> filtered_root/<cam>
    """
    ds = get_image_dataset(config)
    filtered_root: Path = ds["filtered_root"]
    cam_cfgs: Dict[str, Dict[str, Any]] = ds["camera_cfgs"]
    cfg = cam_cfgs.get(cam, {}) if isinstance(cam_cfgs, dict) else {}

    filtered_dir_any = cfg.get("filtered_dir", None)
    if isinstance(filtered_dir_any, str) and filtered_dir_any.strip():
        return Path(filtered_dir_any.strip())
    return filtered_root / cam


def get_camera_filtered_images(config: dict, cam: str) -> List[Path]:
    """列出某相机 filtered 目录下的图片文件（若不存在则返回空）。"""
    return _list_images_in_dir(get_camera_filtered_dir(config, cam))


# endregion


# region 数据集：Step5（step5_dataset）


def get_step5_dataset(config: dict) -> Dict[str, Any]:
    """解析 config.step5_dataset（Step5 图片数据集输入）。

    约定（可选字段）：
      config["step5_dataset"] = {
        "enabled": bool,
        "image_root": "images/step5",
        "cameras": {
          "cam0": {"raw_dir": "..."} 或 {"raw_glob": ".../*.png"} 或 {"raw_glob": ["..."]},
          ...
        }
      }

    Returns:
        标准化结构：enabled/image_root/cameras/camera_cfgs
    """
    ds = (config or {}).get("step5_dataset", {})
    if not isinstance(ds, dict):
        ds = {}

    enabled = bool(ds.get("enabled", False))
    image_root = Path(str(ds.get("image_root", "images/step5")))

    cams_cfg_any = ds.get("cameras", {})
    cam_cfgs: Dict[str, Dict[str, Any]] = {}
    cameras: List[str] = []

    if isinstance(cams_cfg_any, dict):
        for cam, cfg in cams_cfg_any.items():
            if not isinstance(cam, str) or not cam.strip():
                continue
            cam_name = cam.strip()
            cam_cfgs[cam_name] = cfg if isinstance(cfg, dict) else {}
            cameras.append(cam_name)

    if isinstance(cams_cfg_any, list):
        cameras = []
        cam_cfgs = {}
        for cam in cams_cfg_any:
            if isinstance(cam, str) and cam.strip():
                cam_name = cam.strip()
                cameras.append(cam_name)
                cam_cfgs[cam_name] = {}

    return {
        "enabled": enabled,
        "image_root": image_root,
        "cameras": cameras,
        "camera_cfgs": cam_cfgs,
    }


def get_step5_cameras(config: dict, *, allow_scan: bool = True) -> List[str]:
    """获取 Step5 需要处理的相机列表。"""
    ds = get_step5_dataset(config)
    cams = ds.get("cameras", [])
    if isinstance(cams, list) and len(cams) > 0:
        return [str(c) for c in cams]

    if allow_scan:
        root = ds.get("image_root", Path("images/step5"))
        if not isinstance(root, Path):
            root = Path(str(root))
        found: List[str] = []
        try:
            if root.exists():
                for p in sorted(root.iterdir()):
                    if not p.is_dir():
                        continue
                    if len(_list_images_in_dir(p)) > 0:
                        found.append(p.name)
        except Exception:
            found = []
        return found

    return []


def get_step5_camera_images(config: dict, cam: str) -> List[Path]:
    """根据 step5_dataset 配置返回某相机的 Step5 图片列表。

    优先级：raw_glob > raw_dir > image_root/<cam>
    """
    ds = get_step5_dataset(config)
    image_root: Path = ds["image_root"]
    cam_cfgs: Dict[str, Dict[str, Any]] = ds["camera_cfgs"]
    cfg = cam_cfgs.get(cam, {}) if isinstance(cam_cfgs, dict) else {}

    raw_glob_any = cfg.get("raw_glob", None)
    if isinstance(raw_glob_any, str) and raw_glob_any.strip():
        return _expand_image_globs([raw_glob_any.strip()])
    if isinstance(raw_glob_any, list):
        pats = [str(x).strip() for x in raw_glob_any if isinstance(x, (str, int, float)) and str(x).strip()]
        if len(pats) > 0:
            return _expand_image_globs(pats)

    raw_dir_any = cfg.get("raw_dir", None)
    if isinstance(raw_dir_any, str) and raw_dir_any.strip():
        return _list_images_in_dir(Path(raw_dir_any.strip()))

    return _list_images_in_dir(image_root / cam)


# endregion


# region 检测配置（detection settings / ROI / detector params）


def get_detection_settings(
        config: dict,
        *,
        default_use_multiscale: bool = True,
        default_opencv_refine: bool = False,
) -> Tuple[bool, bool]:
    """从配置文件中读取 AprilTag 检测相关开关（兼容旧配置）。

    约定：
        config["calibration_settings"]["detection"] = {
            "use_multiscale": true/false,
            "opencv_refine": true/false
        }

    旧配置缺失该字段时：
        - use_multiscale 默认 True（标定优先稳定）
        - opencv_refine 默认 False（避免在缺少 K/dist/board 时改变行为）

    Returns:
        (use_multiscale, opencv_refine)
    """
    calib_cfg = config.get("calibration_settings", {}) if isinstance(config, dict) else {}
    det_cfg = calib_cfg.get("detection", {}) if isinstance(calib_cfg, dict) else {}

    use_multiscale = det_cfg.get("use_multiscale", default_use_multiscale)
    opencv_refine = det_cfg.get("opencv_refine", default_opencv_refine)
    return bool(use_multiscale), bool(opencv_refine)


def get_detection_profile(config: dict, *, default: str = "balanced") -> str:
    """获取检测 profile。

    约定：
        config["calibration_settings"]["detection"]["profile"]

    可选值（约定，不强制）：
        - balanced: 默认，稳健且不过度耗时
        - small_tags: 针对小Tag/远距离，提高召回率（更慢，可能略增误检）
    """
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
    """从配置中获取检测 ROI。

    约定：
        config["calibration_settings"]["detection"]["roi"]

    支持形式：
        1) 统一 ROI："roi": [x, y, w, h]
        2) 按相机区分："roi": {"left": [x,y,w,h], "right": [x,y,w,h]}
        3) 也支持 dict: {"x":...,"y":...,"w":...,"h":...}

    返回： (x, y, w, h) 或 None
    """
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
    """从配置中获取自动 ROI（两阶段检测）参数。

    约定：
        config["calibration_settings"]["detection"]["auto_roi"]

    支持：
        - bool：true/false
        - dict：{"enabled": true, "pre_scale": 0.5, "min_tags": 1, "margin": 0.25}

    返回：
        标准化 dict：
            {
              "enabled": bool,
              "pre_scale": float,
              "min_tags": int,
              "margin": float
            }
    """
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
    """根据检测到的角点估计一个 ROI（xywh）。

    corners: list，每个元素形状通常为 (1,4,2) 或 (4,2)
    image_shape_hw: (h, w)
    margin: 按外接框宽高的比例外扩
    """
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
        ry1 = int(np.ceil(min(float(h), float(y1) + my)))

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
    """创建并调参 DetectorParameters。

    设计目标：
      - 默认保持稳健（balanced）
      - 在 profile=small_tags 时，提高小Tag/远距离召回（可能更慢）

    参考：OpenCV 官方 DetectorParameters 文档（adaptiveThreshWinSize*, minMarkerPerimeterRate 等）。
    """
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
        # 1) 自适应阈值：小Tag 时过大的 window 会“吃掉”边框；因此限制 max，并减小 step。
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 17
        params.adaptiveThreshWinSizeStep = 2

        # 2) 允许更小 perimeter（提高远距离小Tag候选进入后续阶段的概率）
        params.minMarkerPerimeterRate = 0.015
        params.maxMarkerPerimeterRate = 4.0

        # 3) bit 提取：增加 canonical 分辨率与忽略边缘比例，提升小格子判别稳定性（更慢）
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.20
        params.minOtsuStdDev = 2.0

        # 4) 解码纠错：提高允许纠错，提升低对比/轻模糊时的解码成功率（可能略增误检）
        params.errorCorrectionRate = 0.8

        # 5) AprilTag 专用参数（OpenCV 4.x 支持）
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
    # 约定：config["calibration_settings"]["detection"]["detector_params"] = { ... }
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
                # 忽略不合法覆写
                pass

    return params


# endregion


# region 配置与标定板定义


def load_config(config_path: str = "config/apriltag_config.json") -> dict:
    """
    加载 AprilTag 标定配置文件

    Args:
        config_path: JSON配置文件路径

    Returns:
        包含所有配置参数的字典

    Raises:
        FileNotFoundError: 配置文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_aruco_dict(family: str) -> cv2.aruco.Dictionary:
    """
    根据 AprilTag family 名称获取对应的 ArUco 字典

    OpenCV 通过 ArUco 模块支持 AprilTag 检测，需要使用对应的预定义字典

    Args:
        family: AprilTag family 名称，支持:
                - "tag16h5": DICT_APRILTAG_16h5
                - "tag25h9": DICT_APRILTAG_25h9
                - "tag36h10": DICT_APRILTAG_36h10
                - "tag36h11": DICT_APRILTAG_36h11 (推荐，36位编码)

    Returns:
        cv2.aruco.Dictionary 对象

    Raises:
        ValueError: 不支持的 family 名称
    """
    family_map = {
        "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }

    if family not in family_map:
        raise ValueError(
            f"不支持的 AprilTag family: {family}. 支持的类型: {list(family_map.keys())}"
        )

    return cv2.aruco.getPredefinedDictionary(family_map[family])


def create_apriltag_board(config: dict) -> Tuple[np.ndarray, List[int]]:
    """
    创建 AprilTag 标定板的 3D 角点坐标和 ID 列表

    AprilTag 板是纯标签网格，没有棋盘格。
    我们需要手动计算每个标签四个角点的 3D 坐标。

    Args:
        config: 配置字典，包含 apriltag_board 配置

    Returns:
        obj_points: (N, 3) 数组，所有标签角点的 3D 坐标 (单位: mm)
        tag_ids: 长度为 tags_x * tags_y 的 ID 列表

    标签 ID 排列顺序（从左下角开始，行优先，Y轴向上）:
       30  31  32  33  34  35    <- 顶部 (Y=357.5mm, 图像上方)
       24  25  26  27  28  29
       18  19  20  21  22  23
       12  13  14  15  16  17
        6   7   8   9  10  11
        0   1   2   3   4   5    <- 底部 (Y=0, 图像下方)

    注意：世界坐标原点在标定板左下角（ID 0），Y轴向上递增。
    row=0(ID 0-5)对应Y=0，row=5(ID 30-35)对应Y最大。

    每个标签的四个角点顺序：
        OpenCV ArUco 标准顺序（顺时针，从左上开始）:
        0 ------- 1
        |         |
        |   TAG   |
        |         |
        3 ------- 2

        但实际检测时，由于标签在图像中可能旋转，
        角点 0 始终是标签本身坐标系的左上角（基于标签编码方向）
    """
    board_cfg = config["apriltag_board"]
    tags_x = board_cfg["tags_x"]
    tags_y = board_cfg["tags_y"]
    tag_size = board_cfg["tag_size"]  # 单位: mm
    tag_spacing = board_cfg["tag_spacing"]  # 单位: mm

    # 标签中心到中心的距离
    tag_pitch = tag_size + tag_spacing

    # 生成所有标签的 3D 角点
    obj_points = []
    tag_ids = []

    for row in range(tags_y):
        for col in range(tags_x):
            tag_id = row * tags_x + col
            tag_ids.append(tag_id)

            # 标签中心位置
            # Y轴从底部开始向上递增：row=0 → Y=0 (底部), row=5 → Y=max (顶部)
            center_x = col * tag_pitch
            center_y = row * tag_pitch

            # ==== 3D 角点坐标定义（关键！必须与检测顺序匹配）====
            # OpenCV ArUco 标准角点顺序：从左上角开始顺时针编号
            #     0 ------- 1
            #     |   TAG   |
            #     3 ------- 2
            #
            # 标定板坐标系：X 向右，Y 向上，原点在左下角（Tag ID 0 中心）
            # 因此："上" = Y值大，"下" = Y值小
            half_size = tag_size / 2.0
            corners = np.array(
                [
                    [center_x - half_size, center_y + half_size, 0],  # 0: 左上
                    [center_x + half_size, center_y + half_size, 0],  # 1: 右上
                    [center_x + half_size, center_y - half_size, 0],  # 2: 右下
                    [center_x - half_size, center_y - half_size, 0],  # 3: 左下
                ]
            )

            obj_points.append(corners)

    # 转换为 numpy 数组 (num_tags, 4, 3)
    obj_points = np.array(obj_points, dtype=np.float32)

    return obj_points, tag_ids


def create_opencv_aruco_board(
    obj_points: np.ndarray,
    tag_ids: List[int],
    aruco_dict: cv2.aruco.Dictionary,
) -> cv2.aruco.Board:
    """从 AprilTag 板的 3D 角点定义构建 OpenCV 的 Board 对象。

    该 Board 可用于 OpenCV 的 `ArucoDetector.refineDetectedMarkers()`：
    利用 board 布局把 rejectedCandidates 中“差一点解码成功”的 marker 捞回，
    通常能在标定板场景显著提高检测率，同时避免自行多次 resize/阈值造成的角点偏差。

    Args:
        obj_points: (num_tags, 4, 3) 或 (num_tags, 4, 3) float32/float64
        tag_ids: 长度 num_tags 的标签 id 列表
        aruco_dict: OpenCV ArUco/AprilTag 字典

    Returns:
        cv2.aruco.Board
    """
    if obj_points is None or len(tag_ids) == 0:
        raise ValueError("obj_points/tag_ids 不能为空")

    obj_points = np.asarray(obj_points)
    if obj_points.ndim != 3 or obj_points.shape[1:] != (4, 3):
        raise ValueError(f"obj_points 形状应为 (N,4,3)，当前: {obj_points.shape}")

    # OpenCV Python 4.11: Board(objPointsList, dictionary, ids)
    obj_points_list = [
        obj_points[i].reshape(1, 4, 3).astype(np.float32) for i in range(obj_points.shape[0])
    ]
    ids = np.array(tag_ids, dtype=np.int32).reshape(-1, 1)
    return cv2.aruco.Board(obj_points_list, aruco_dict, ids)


# endregion


# region AprilTag 检测与位姿估计


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
    """
    检测图像中的 AprilTag 标签及其角点

    Args:
        image: 输入图像 (灰度或彩色)
        aruco_dict: ArUco 字典对象
        detector_params: 检测器参数，None 则使用默认参数
        use_multiscale: 是否使用多尺度检测（更稳，速度更慢）
        opencv_refine: 是否启用 OpenCV 官方 refineDetectedMarkers（建议配合 board；有 K/dist 更可靠）

    Returns:
        corners: 检测到的角点列表，每个元素是 (4, 2) 数组
        ids: 检测到的标签 ID 数组 (N, 1)

    注意:
        - 返回的 corners 顺序与 ArUco 标准一致（顺时针，左上开始），并在本工程中做了固定映射修正
        - 如果启用角点优化，可提高精度
        - 对标定板场景：建议 use_multiscale=True 且提供 board；step4/step5 建议同时提供 K/dist
    """
    # 自动 ROI（两阶段检测）：
    # 先在缩小图上做一次“便宜”的粗检以定位标定板大概区域，再在该 ROI 上运行真正的检测流程。
    # 对于“板子在画面里晃动”的场景，固定 ROI 可能失效，但 auto_roi 仍然可用。
    if auto_roi and roi is None:
        try:
            h0, w0 = int(image.shape[0]), int(image.shape[1])

            pre_scale = float(auto_roi_pre_scale)
            if not np.isfinite(pre_scale):
                pre_scale = 0.5
            pre_scale = float(np.clip(pre_scale, 0.20, 1.0))

            # 对于较小分辨率，没必要再缩小。
            if max(h0, w0) <= 900:
                pre_scale = 1.0

            if pre_scale < 1.0:
                sw = max(10, int(round(w0 * pre_scale)))
                sh = max(10, int(round(h0 * pre_scale)))
                img_small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
            else:
                img_small = image

            # 粗检：用单尺度 detectMarkers（速度快）；沿用传入的 detector_params（通常已根据 profile 调过）。
            dp = detector_params if detector_params is not None else create_detector_params(config=None)
            det = cv2.aruco.ArucoDetector(aruco_dict, dp)
            c_s, id_s, rej_s = det.detectMarkers(img_small)

            # 兼容性兜底：APRILTAG 模式偶发 0 检测，回退到 SUBPIX。
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

            # 粗检可选 refine：不传 K/dist（因为缩放后需要同步缩放相机内参）。
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

                    # Clamp
                    rx0 = max(0, min(rx0, w0 - 1))
                    ry0 = max(0, min(ry0, h0 - 1))
                    rw0 = max(1, min(rw0, w0 - rx0))
                    rh0 = max(1, min(rh0, h0 - ry0))

                    # 太小的 ROI 没意义（可能是误检）
                    if rw0 >= 20 and rh0 >= 20:
                        roi = (rx0, ry0, rw0, rh0)
        except Exception:
            # auto_roi 失败时保持原行为：继续全图检测
            pass

    # 可选 ROI：裁剪后检测，再把角点坐标加回偏移。
    # 说明：裁剪不会“增大像素”，但能让更激进的上采样策略在 ROI 上跑得动，
    # 同时减少干扰轮廓，提高召回。
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
        from apriltag_detector import detect_apriltag_multiscale

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

        # 统一 corners/ids 的返回类型，避免不同 OpenCV/实现路径类型不一致
        if corners is not None and not isinstance(corners, list):
            corners = list(corners)
        if ids is not None:
            ids = np.asarray(ids)

        # 回填 ROI 偏移
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
    else:
        # 使用标准检测
        if detector_params is None:
            detector_params = create_detector_params(config=None)

        # 创建检测器
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        # 检测标签
        corners, ids, rejected = detector.detectMarkers(img_in)

        # 兼容性兜底：若外部把 cornerRefinementMethod 设为 APRILTAG 且导致 0 检测，
        # 则自动回退到 SUBPIX（不改变字典，只换检测/精修策略）。
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
                # 兜底失败则保持原结果
                pass

        # 使用 OpenCV 官方 refine（基于 board 布局“捞回” rejected candidates）
        # 默认关闭，避免改变现有行为。
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
                # refine 失败时保持原检测结果
                pass

        # ==== 角点顺序修正（针对图像旋转180°的情况）====
        # 问题原因：相机图像旋转导致检测到的角点顺序与3D定义不匹配
        # 原始检测：角点0→右下, 角点1→左下, 角点2→左上, 角点3→右上（旋转180°）
        # 目标顺序：角点0→左上, 角点1→右上, 角点2→右下, 角点3→左下（标准顺序）
        # 映射关系：新索引 [0,1,2,3] = 原索引 [2,3,0,1]
        #
        # 注意：如果你的相机方向不同，可能不需要此修正或需要不同的映射！
        # 验证方法：运行 test_corner_order.py 检查2D检测与3D定义是否匹配
        if corners is not None and len(corners) > 0:
            corrected_corners = []
            for corner in corners:
                pts = corner.reshape(4, 2)  # (1,4,2) -> (4,2)
                # 重新排列顺序以匹配3D坐标定义
                corrected_pts = np.array([pts[2], pts[3], pts[0], pts[1]])
                corrected_corners.append(corrected_pts.reshape(1, 4, 2))
            corners = corrected_corners

        # 回填 ROI 偏移
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

        # 无论是否做了角点顺序修正，都保证 corners 是 list（OpenCV 类型标注常为 Sequence）
        if corners is not None and not isinstance(corners, list):
            corners = list(corners)

        # 统一 ids 类型为 np.ndarray，避免不同 OpenCV 路径返回类型不一致
        if ids is not None:
            ids = np.asarray(ids)

        return corners, ids


def estimate_pose_apriltag(
    corners: Optional[List[np.ndarray]],
    ids: Optional[np.ndarray],
    obj_points: np.ndarray,
    tag_ids: List[int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    使用检测到的 AprilTag 标签估计相机位姿

    Args:
        corners: 检测到的图像角点列表
        ids: 检测到的标签 ID 数组 (N, 1)
        obj_points: 所有标签的 3D 角点 (num_tags, 4, 3)
        tag_ids: 标定板上所有标签的 ID 列表
        camera_matrix: 相机内参矩阵 K (3, 3)
        dist_coeffs: 畸变系数

    Returns:
        success: 是否成功估计位姿
        rvec: 旋转向量 (3, 1)
        tvec: 平移向量 (3, 1)

    原理:
        1. 根据检测到的标签 ID，找到对应的 3D 角点
        2. 将所有角点合并为一组点云
        3. 使用 solvePnP 估计相机位姿

    畸变处理:
        当前实现使用带畸变的相机模型，适用于原始(未去畸变)图像。

        如果需要更高精度，推荐的做法是:
        方法1 (推荐):
            - 先用 cv2.undistort() 对图像去畸变
            - 使用 cv2.getOptimalNewCameraMatrix() 获得新的相机矩阵
            - 调用本函数时传入新相机矩阵和零畸变系数: np.zeros(5)

        方法2 (当前):
            - 直接使用原始图像
            - 传入原始相机矩阵和畸变系数
            - solvePnP 内部处理畸变 (精度稍低，但简单)
    """
    if corners is None or ids is None or len(ids) == 0:
        return False, None, None

    # 收集所有检测到的 2D-3D 对应点
    image_points = []
    object_points = []

    ids_flat = ids.flatten()

    for i, tag_id in enumerate(ids_flat):
        if tag_id in tag_ids:
            # 找到该标签在标定板上的索引
            idx = tag_ids.index(tag_id)

            # 获取该标签的 3D 角点
            obj_pts = obj_points[idx]  # (4, 3)
            img_pts = corners[i].reshape(-1, 2)  # (4, 2)

            object_points.append(obj_pts)
            image_points.append(img_pts)

    if len(object_points) == 0:
        return False, None, None

    # 合并所有点
    object_points = np.vstack(object_points).astype(np.float32)
    image_points = np.vstack(image_points).astype(np.float32)

    # 估计位姿
    # 注意：确保 camera_matrix 和 dist_coeffs 匹配图像类型
    # - 原始图像：使用原始内参 + 原始畸变系数
    # - 去畸变图像：使用新内参 + 零畸变系数 np.zeros(5)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    return success, rvec, tvec


# endregion


# region 可视化 / 相机 / 双目质量评估


def draw_detected_tags(
    image: np.ndarray,
    corners: Optional[List[np.ndarray]],
    ids: Optional[np.ndarray],
    min_tags: int = 4,
) -> Tuple[np.ndarray, bool]:
    """
    在图像上绘制检测到的 AprilTag 标签

    Args:
        image: 输入图像
        corners: 检测到的角点列表
        ids: 检测到的标签 ID
        min_tags: 最少需要检测到的标签数量

    Returns:
        output_image: 绘制后的图像
        is_valid: 是否检测到足够的标签
    """
    output_image = image.copy()

    # 检查是否检测到足够的标签
    num_detected = 0 if ids is None else len(ids)
    is_valid = num_detected >= min_tags

    if num_detected > 0 and corners is not None:
        # 绘制标签边框和 ID
        cv2.aruco.drawDetectedMarkers(output_image, corners, ids)

        # 添加状态指示
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        status = "✓" if is_valid else "✗"
        text = f"{status} 标签数: {num_detected}/{min_tags}"

        cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        # 未检测到标签
        text = "✗ 未检测到标签"
        cv2.putText(
            output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

    return output_image, is_valid


def visualize_board_layout(config: dict, output_path: str = "board_layout.png"):
    """
    可视化 AprilTag 标定板布局（用于验证配置）

    Args:
        config: 配置字典
        output_path: 输出图像路径

    生成一张示意图显示:
        - 标签排列
        - 标签 ID
        - 尺寸标注
    """
    board_cfg = config["apriltag_board"]
    tags_x = board_cfg["tags_x"]
    tags_y = board_cfg["tags_y"]
    tag_size = board_cfg["tag_size"]
    tag_spacing = board_cfg["tag_spacing"]

    # 创建可视化图像
    px_per_mm = 10  # 每毫米 10 像素
    tag_pitch = int((tag_size + tag_spacing) * px_per_mm)
    tag_px = int(tag_size * px_per_mm)

    img_width = tags_x * tag_pitch + 100
    img_height = tags_y * tag_pitch + 100

    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # 绘制标签
    for row in range(tags_y):
        for col in range(tags_x):
            tag_id = row * tags_x + col

            x = 50 + col * tag_pitch
            y = 50 + row * tag_pitch

            # 绘制标签矩形
            cv2.rectangle(img, (x, y), (x + tag_px, y + tag_px), (0, 0, 0), 2)

            # 标注 ID
            text_size = cv2.getTextSize(str(tag_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[
                0
            ]
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

    # 添加标题
    title = (
        f"AprilTag Board: {tags_x}x{tags_y}, Size={tag_size}mm, Spacing={tag_spacing}mm"
    )
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"标定板布局图已保存到: {output_path}")


def init_camera(config: dict):
    """
    初始化相机（从 camera_wrapper 导入）

    统一的相机初始化接口，自动处理相机打开和错误检查。

    Args:
        config: 配置字典，必须包含'camera_settings'字段

    Returns:
        BaseCameraWrapper: 已打开的相机封装实例

    Raises:
        RuntimeError: 如果相机打开失败

    注意: camera_wrapper.py 必须在同级目录

    Example:
        >>> config = load_config()
        >>> camera = init_camera(config)
        >>> left, right, ts = camera.read_stereo()
        >>> camera.release()
    """
    # 使用规范的包导入方式
    from libs.camera_wrapper import create_camera

    camera_settings = config.get("camera_settings", {})
    camera = create_camera(camera_settings)

    if not camera.open():
        raise RuntimeError("Failed to open video source. Check camera_settings and video file paths/codecs.")

    return camera


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
    """
    分析双目图像对的质量（共同检测到的标签数）

    适用于会聚式双目相机标定前的图像质量评估。

    Args:
        left_images: 左图像路径列表
        right_images: 右图像路径列表
        aruco_dict: ArUco字典
        detector_params: 检测参数
        use_multiscale: 是否使用多尺度检测

    Returns:
        [(left_path, right_path, common_tags_count), ...] 列表
        每个元组包含左右图像路径和共同检测到的标签数

    Example:
        >>> aruco_dict = get_aruco_dict('tag36h11')
        >>> params = cv2.aruco.DetectorParameters()
        >>> quality = analyze_stereo_image_quality(left_imgs, right_imgs, aruco_dict, params)
        >>> for left, right, common in quality:
        ...     print(f"{common} common tags")
    """
    results = []

    # 可选提前停止：当“足够多的高质量 pair”已经收集到时，就停止继续扫描。
    # 这能避免在超大数据集（例如 700+ pairs）上耗时过久。
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
    image_quality: List[Tuple[str, str, int]], min_common_tags: int = 15
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, int]]]:
    """
    根据共同标签数过滤低质量图像对

    对于会聚式双目相机，建议 min_common_tags >= 15
    对于平行双目相机，可以设置为 min_common_tags >= 10

    Args:
        image_quality: analyze_stereo_image_quality() 的返回结果
        min_common_tags: 最少共同标签数阈值（默认15）

    Returns:
        (keep_pairs, remove_pairs) 元组
        - keep_pairs: [(left_path, right_path), ...] 保留的图像对
        - remove_pairs: [(left_path, right_path, common_count), ...] 移除的图像对及其质量

    Example:
        >>> quality = analyze_stereo_image_quality(...)
        >>> keep, remove = filter_low_quality_pairs(quality, min_common_tags=15)
        >>> print(f"Keep {len(keep)} pairs, remove {len(remove)} pairs")
    """
    keep_pairs = []
    remove_pairs = []

    for left_path, right_path, common in image_quality:
        if common >= min_common_tags:
            keep_pairs.append((left_path, right_path))
        else:
            remove_pairs.append((left_path, right_path, common))

    return keep_pairs, remove_pairs


# endregion
