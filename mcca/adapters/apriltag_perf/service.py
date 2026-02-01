from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mcca.adapters.apriltag_perf.cache import CacheConfig, DetectionCache, build_cache_key
from mcca.adapters.apriltag_perf.prefilter import PrefilterConfig, prefilter_pass


@dataclass
class DetectionResult:
    """单张图片的 AprilTag 检测结果（含缓存信息）。"""

    corners: Optional[List[np.ndarray]]
    ids: Optional[np.ndarray]

    from_cache: bool
    elapsed_ms: float

    # 0=正常检测（不代表一定检测到标签）；1=检测异常；2=预筛选跳过
    status: int

    meta: Dict[str, Any]


@dataclass
class DetectorStats:
    """用于输出可观测性统计（跨多张图累计）。"""

    total: int = 0
    cache_hit: int = 0
    cache_miss: int = 0
    prefilter_skipped: int = 0
    error: int = 0

    detect_ms_sum: float = 0.0


class CachedAprilTagDetector:
    """带缓存/可选预筛选的 AprilTag 检测服务。

    设计选择：
    - 缓存粒度：每张图一个 cache entry；即使检测不到标签，也缓存“空结果”，避免重复耗时。
    - 缓存 key：由外部提供 algo_key（来自 JSON 配置），本类不尝试序列化 OpenCV 对象。
    - OpenCV 相关导入尽量延迟到运行时，便于在无 OpenCV 环境下跑基础单测。
    """

    ALGO_VERSION = 1

    def __init__(
        self,
        *,
        aruco_dict: Any,
        detector_params: Any,
        algo_key: Dict[str, Any],
        use_multiscale: bool,
        opencv_refine: bool,
        board: Any = None,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        auto_roi: bool = False,
        auto_roi_pre_scale: float = 0.5,
        auto_roi_min_tags: int = 1,
        auto_roi_margin: float = 0.25,
        cache_cfg: Optional[CacheConfig] = None,
        prefilter_cfg: Optional[PrefilterConfig] = None,
    ):
        self.aruco_dict = aruco_dict
        self.detector_params = detector_params

        def _hash_array(a: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
            if a is None:
                return None
            arr = np.asarray(a)
            # 用字节级哈希避免把矩阵内容直接塞进 key（过大且 JSON 不友好）
            h = hashlib.sha256(arr.tobytes()).hexdigest()
            return {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": h,
            }

        self.algo_key = {
            "algo_version": int(self.ALGO_VERSION),
            **(algo_key or {}),
            "use_multiscale": bool(use_multiscale),
            "opencv_refine": bool(opencv_refine),
            "camera_matrix": _hash_array(camera_matrix),
            "dist_coeffs": _hash_array(dist_coeffs),
            "roi": list(roi) if roi is not None else None,
            "auto_roi": {
                "enabled": bool(auto_roi),
                "pre_scale": float(auto_roi_pre_scale),
                "min_tags": int(auto_roi_min_tags),
                "margin": float(auto_roi_margin),
            },
        }

        self.use_multiscale = bool(use_multiscale)
        self.opencv_refine = bool(opencv_refine)

        self.board = board
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.roi = roi
        self.auto_roi = bool(auto_roi)
        self.auto_roi_pre_scale = float(auto_roi_pre_scale)
        self.auto_roi_min_tags = int(auto_roi_min_tags)
        self.auto_roi_margin = float(auto_roi_margin)

        self.cache = DetectionCache(cache_cfg or CacheConfig())
        self.prefilter_cfg = prefilter_cfg or PrefilterConfig()

        self.stats = DetectorStats()

    @staticmethod
    def _corners_list_to_array(corners: Optional[List[np.ndarray]]) -> np.ndarray:
        if corners is None or len(corners) == 0:
            return np.zeros((0, 4, 2), dtype=np.float32)
        arr = np.stack([np.asarray(c, dtype=np.float32).reshape(4, 2) for c in corners], axis=0)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _corners_array_to_list(corners_arr: Optional[np.ndarray]) -> Optional[List[np.ndarray]]:
        if corners_arr is None:
            return None
        a = np.asarray(corners_arr)
        if a.size == 0:
            return []
        if a.ndim != 3 or a.shape[1:] != (4, 2):
            # 兼容旧缓存/异常形状：尽量不崩，返回 None 触发上层重新检测。
            return None
        return [a[i].reshape(1, 4, 2).astype(np.float32, copy=False) for i in range(a.shape[0])]

    def detect_path(self, image_path: Path) -> DetectionResult:
        """检测单张图片（读取 + 可选预筛选 + 缓存）。"""

        self.stats.total += 1

        p = Path(image_path)
        key = build_cache_key(image_path=p, algo_key=self.algo_key)

        cached = self.cache.load(key)
        if cached is not None:
            self.stats.cache_hit += 1

            ids = cached.get("ids")
            corners_arr = cached.get("corners")
            corners = self._corners_array_to_list(corners_arr)
            if corners is None:
                # 缓存形状异常：当作 miss，重新检测
                cached = None
            else:
                meta = dict(cached.get("meta") or {})
                meta.setdefault("cache_key", key)
                return DetectionResult(
                    corners=corners,
                    ids=np.asarray(ids) if ids is not None else None,
                    from_cache=True,
                    elapsed_ms=0.0,
                    status=int(cached.get("status") or 0),
                    meta=meta,
                )

        self.stats.cache_miss += 1

        t0 = time.perf_counter()
        try:
            import cv2

            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"无法读取图片: {p}")

            ok, metrics = prefilter_pass(img, self.prefilter_cfg)
            if not ok:
                self.stats.prefilter_skipped += 1
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.stats.detect_ms_sum += float(elapsed_ms)
                return DetectionResult(
                    corners=[],
                    ids=np.zeros((0, 1), dtype=np.int32),
                    from_cache=False,
                    elapsed_ms=float(elapsed_ms),
                    status=2,
                    meta={"prefilter": metrics, "image": str(p)},
                )

            # 关键：adapters -> core 依赖方向
            from mcca.core.detection import detect_apriltag_corners

            corners, ids = detect_apriltag_corners(
                img,
                self.aruco_dict,
                self.detector_params,
                self.use_multiscale,
                opencv_refine=self.opencv_refine,
                board=self.board,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                roi=self.roi,
                auto_roi=self.auto_roi,
                auto_roi_pre_scale=self.auto_roi_pre_scale,
                auto_roi_min_tags=self.auto_roi_min_tags,
                auto_roi_margin=self.auto_roi_margin,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.detect_ms_sum += float(elapsed_ms)

            ids_arr = np.asarray(ids) if ids is not None else np.zeros((0, 1), dtype=np.int32)
            if ids_arr.size == 0:
                ids_arr = ids_arr.reshape(0, 1).astype(np.int32, copy=False)

            corners_arr = self._corners_list_to_array(corners)

            meta = {
                "image": str(p),
                "elapsed_ms": float(elapsed_ms),
                "n_tags": int(ids_arr.shape[0]),
                "prefilter": metrics,
            }
            self.cache.save(
                key,
                ids=ids_arr.astype(np.int32, copy=False),
                corners=corners_arr,
                status=0,
                meta=meta,
            )

            return DetectionResult(
                corners=corners,
                ids=ids_arr,
                from_cache=False,
                elapsed_ms=float(elapsed_ms),
                status=0,
                meta=meta,
            )
        except Exception as e:
            self.stats.error += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.detect_ms_sum += float(elapsed_ms)

            meta = {"image": str(p), "error": repr(e), "elapsed_ms": float(elapsed_ms)}
            try:
                # 异常也缓存，避免“坏图/解码失败图”在多 step 中反复触发开销。
                self.cache.save(
                    key,
                    ids=np.zeros((0, 1), dtype=np.int32),
                    corners=np.zeros((0, 4, 2), dtype=np.float32),
                    status=1,
                    meta=meta,
                )
            except Exception:
                pass

            return DetectionResult(
                corners=[],
                ids=np.zeros((0, 1), dtype=np.int32),
                from_cache=False,
                elapsed_ms=float(elapsed_ms),
                status=1,
                meta=meta,
            )
