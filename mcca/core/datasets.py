from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List


_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")


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
            # 约定：JSON 里常用 "_comment" / "_xxx" 放注释或暂时禁用的条目。
            # 这些不应被当作真实相机名，否则 Step2-5 会去 images/.../_comment 下找图片并报错。
            if cam_name.startswith("_") or cam_name.startswith(".") or cam_name.startswith("__"):
                continue
            cam_cfgs[cam_name] = cfg if isinstance(cfg, dict) else {}
            cameras.append(cam_name)

    # 允许写成 cameras: ["cam0", "cam1", ...]（此时使用 raw_root/<cam>）
    if isinstance(cams_cfg_any, list):
        cameras = []
        cam_cfgs = {}
        for cam in cams_cfg_any:
            if isinstance(cam, str) and cam.strip():
                cam_name = cam.strip()
                if cam_name.startswith("_") or cam_name.startswith(".") or cam_name.startswith("__"):
                    continue
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
) -> List[str]:
    """获取“需要处理的相机列表”。

    优先级：
      1) image_dataset.cameras（dict keys 或 list）
      2) 若 allow_scan：扫描 image_dataset.raw_root 下的子目录（包含图片）

    说明：
        - 该函数不强依赖 image_dataset.enabled；只要配置了 cameras（或目录可扫描到），就会返回。
        - 本仓库已统一使用 cam0/cam1/cam2... 命名，不再提供 left/right 的隐式回退。
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
            # 同 image_dataset：过滤 "_comment" 等非相机条目。
            if cam_name.startswith("_") or cam_name.startswith(".") or cam_name.startswith("__"):
                continue
            cam_cfgs[cam_name] = cfg if isinstance(cfg, dict) else {}
            cameras.append(cam_name)

    if isinstance(cams_cfg_any, list):
        cameras = []
        cam_cfgs = {}
        for cam in cams_cfg_any:
            if isinstance(cam, str) and cam.strip():
                cam_name = cam.strip()
                if cam_name.startswith("_") or cam_name.startswith(".") or cam_name.startswith("__"):
                    continue
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
