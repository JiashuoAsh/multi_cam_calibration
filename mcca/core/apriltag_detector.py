#!/usr/bin/env python3
"""多尺度AprilTag检测库 - 检测核心模块

功能：
  - 高级多尺度检测
  - 与原有API兼容
  - 被所有步骤共享调用

使用方法：
  # 方式1：使用包装函数（推荐，兼容原有API）
    from mcca.core.apriltag_detector import detect_apriltag_multiscale
  corners, ids = detect_apriltag_multiscale(image, aruco_dict)

  # 方式2：使用完整的类（支持可视化）
    from mcca.core.apriltag_detector import AdvancedAprilTagDetector
  detector = AdvancedAprilTagDetector()
  corners, ids = detector.detect_multiscale_simple(image)

"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class AdvancedAprilTagDetector:
    """高级AprilTag检测器 - 支持多尺度检测和图像增强"""

    def __init__(
        self,
        grid_rows: int = 6,
        grid_cols: int = 6,
        verbose: bool = False,
        aruco_dict: Optional[cv2.aruco.Dictionary] = None,
        detector_params: Optional[cv2.aruco.DetectorParameters] = None,
        board: Optional[cv2.aruco.Board] = None,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        opencv_refine: bool = False,
    ):
        """初始化检测器。

        Args:
            grid_rows: 标定板行数。
            grid_cols: 标定板列数。
            verbose: 是否输出详细信息。
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.total_tags = grid_rows * grid_cols
        self.verbose = verbose
        self.board = board
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.opencv_refine = opencv_refine

        # ==== ArUco检测器初始化 ====
        # Step 1: 加载AprilTag字典（36h11表示36位编码，汉明距离11）
        # - 36h11是AprilTag家族中最常用的版本
        # - 'h11'表示任意两个标签之间至少有11位差异，提供强大的纠错能力
        # - 可以编码0-586个不同的ID
        # IMPORTANT:
        #  - 这里必须允许外部传入字典，否则当工程配置使用 tag16h5/tag25h9/tag36h10 时
        #    多尺度检测会默默用错字典，导致检测率骤降或误检。
        self.aruco_dict = (
            aruco_dict
            if aruco_dict is not None
            else cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        )

        # Step 2: 配置检测参数（DetectorParameters控制二值化和轮廓筛选）
        # IMPORTANT:
        #  - 允许外部传入 detector_params（例如外部设置 CORNER_REFINE_APRILTAG）。
        #  - 若未传入，才使用本模块的默认调参。
        self.parameters = detector_params if detector_params is not None else cv2.aruco.DetectorParameters()

        if detector_params is None:
            # --- 默认参数：偏向“更高召回率”，但仍保持一定的误检约束 ---
            # 注意：CORNER_REFINE_APRILTAG 在 OpenCV 中是“AprilTag2 检测方式”，
            # 在部分图片上会出现 0 检测。这里默认用 SUBPIX 做角点亚像素精修（更稳）。
            self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

            # 自适应二值化偏移（默认 7）
            # 值更小：阈值更“严格”（更亮才判为白），有时对灰度偏暗/反光场景更稳。
            # 值更大：更容易把灰色判成黑色，可能提升低对比度召回，但也可能带来更多候选。
            self.parameters.adaptiveThreshConstant = 10

            # 候选四边形过滤：允许更小的 tag（提升远距离/小尺寸召回）。
            # 最小检测周长是 1280 * minMarkerPerimeterRate 像素（1280是假设的图像宽度）。
            # 例如 minMarkerPerimeterRate=0.03 (默认值) 时，最小周长约为 38.4 像素。
            # 0.05 对小Tag会偏严格；这里稍微放宽，避免“一个都检不出”。
            self.parameters.minMarkerPerimeterRate = 0.01
            self.parameters.maxMarkerPerimeterRate = 4.0

        # Step 3: 创建检测器实例（结合字典和参数）
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.detected_tags = {}

    def _resize_gray(self, gray: np.ndarray, scale: float) -> np.ndarray:
        """缩放灰度图。

        - 上采样：用更锐的插值（Cubic/Lanczos），有利于小Tag边缘
        - 下采样：用 AREA，避免 aliasing
        """
        if scale <= 0:
            raise ValueError("scale must be > 0")
        if abs(scale - 1.0) < 1e-6:
            return gray

        h, w = gray.shape[:2]
        new_w = max(1, int(round(w * float(scale))))
        new_h = max(1, int(round(h * float(scale))))

        if scale > 1.0:
            interp = cv2.INTER_CUBIC
            # 对大倍率上采样，用 Lanczos 细节通常更好（但更慢）
            if scale >= 2.5:
                interp = cv2.INTER_LANCZOS4
        else:
            interp = cv2.INTER_AREA

        return cv2.resize(gray, (new_w, new_h), interpolation=interp)

    def _detect_scaled(self, gray: np.ndarray, scale: float, *, score_scale: float = 1.0):
        """在缩放图上检测，并把角点映射回原图坐标系。"""
        scaled = self._resize_gray(gray, scale)
        corners, ids, _ = self._detect_on_image(scaled)
        if ids is None or corners is None or len(ids) == 0:
            return None, None

        # 映射回原图：除以 scale
        out_corners = []
        out_ids = []
        for idx, tag_id in enumerate(ids.flatten()):
            out_ids.append(int(tag_id))
            out_corners.append(corners[idx][0] / float(scale))
        return out_corners, np.asarray(out_ids, dtype=np.int32).reshape(-1, 1)

    def _detect_tiled_upscale(
        self,
        gray: np.ndarray,
        *,
        upscale: float,
        tile_rows: int = 2,
        tile_cols: int = 2,
        overlap: float = 0.25,
    ):
        """把大图切块后上采样检测。

        目的：
          - 小Tag需要更大像素：上采样有帮助
          - 直接对整图 3x 会非常慢且占内存
          - 切块 + 适度重叠可提升召回并降低开销

        返回的角点已映射回原图坐标。
        """
        h, w = gray.shape[:2]
        tr = max(1, int(tile_rows))
        tc = max(1, int(tile_cols))
        ov = float(np.clip(overlap, 0.0, 0.8))

        tile_w = int(np.ceil(w / tc))
        tile_h = int(np.ceil(h / tr))
        step_w = max(1, int(round(tile_w * (1.0 - ov))))
        step_h = max(1, int(round(tile_h * (1.0 - ov))))

        all_corners = []
        all_ids = []

        y0 = 0
        while y0 < h:
            x0 = 0
            y1 = min(h, y0 + tile_h)
            # 确保最后一块覆盖到底
            if (h - y0) < tile_h and y0 > 0:
                y0 = max(0, h - tile_h)
                y1 = h

            while x0 < w:
                x1 = min(w, x0 + tile_w)
                if (w - x0) < tile_w and x0 > 0:
                    x0 = max(0, w - tile_w)
                    x1 = w

                tile = gray[y0:y1, x0:x1]
                scaled = self._resize_gray(tile, upscale)
                corners, ids, _ = self._detect_on_image(scaled)
                if ids is not None and corners is not None and len(ids) > 0:
                    for idx, tag_id in enumerate(ids.flatten()):
                        c = corners[idx][0] / float(upscale)
                        c = c + np.array([float(x0), float(y0)], dtype=np.float32)
                        all_corners.append(c)
                        all_ids.append(int(tag_id))

                if x1 >= w:
                    break
                x0 += step_w

            if y1 >= h:
                break
            y0 += step_h

        if len(all_ids) == 0:
            return None, None
        return all_corners, np.asarray(all_ids, dtype=np.int32).reshape(-1, 1)

    def _score_corners(self, corners_4x2: np.ndarray) -> float:
        """对同一 tag 的多次检测结果做简单打分，用于择优合并。

        当前使用四边形面积作为 score：一般面积越大，角点量化/模糊影响越小，姿态更稳。
        """
        try:
            pts = corners_4x2.reshape(4, 2).astype(np.float32)
            area = float(abs(cv2.contourArea(pts)))
            return area
        except Exception:
            return 0.0

    def _estimate_subpix_win_size(self, corners_1x4x2: np.ndarray) -> int:
        """基于 tag 在图像中的像素尺寸，自适应估计 cornerSubPix 的 winSize。

        OpenCV 文档中 winSize 是“搜索窗口的半边长”：
        实际窗口大小为 (2*win+1) x (2*win+1)。

        这里用 tag 的边长像素（四条边长度的中位数）来估计一个合理的 win。
        """
        pts = corners_1x4x2.reshape(4, 2).astype(np.float32)
        edges = np.array(
            [
                np.linalg.norm(pts[0] - pts[1]),
                np.linalg.norm(pts[1] - pts[2]),
                np.linalg.norm(pts[2] - pts[3]),
                np.linalg.norm(pts[3] - pts[0]),
            ],
            dtype=np.float32,
        )
        edge_med = float(np.median(edges))

        # 经验比例：tag 边长的 ~8% 作为 half window。
        # - tag 很小（远距离）时给到较小窗口，避免靠近边界直接失败
        # - tag 较大时适度增大窗口，提升收敛稳定性
        win = int(round(edge_med * 0.20))
        # win = int(np.clip(win, 3, 30))
        return win

    def _cap_win_size_by_border(self, gray: np.ndarray, corners_1x4x2: np.ndarray, win: int) -> int:
        """根据角点到图像边界的距离，自动缩小 winSize，避免 cornerSubPix 越界失败。"""
        try:
            h, w = gray.shape[:2]
            pts = corners_1x4x2.reshape(4, 2).astype(np.float32)
            # 距离边界的最小值（以像素为单位）
            # winSize 需要满足：win <= minDistToBorder
            dists = []
            for x, y in pts:
                x = float(x)
                y = float(y)
                dists.append(min(x, y, (w - 1) - x, (h - 1) - y))
            max_win = int(max(0, np.floor(min(dists))))
            return int(min(win, max_win))
        except Exception:
            return win

    def _refine_corners_on_original(
        self,
        gray: np.ndarray,
        corners_1x4x2: np.ndarray,
        win_size: Optional[int] = None,
        max_iters: int = 30,
        eps: float = 0.01,
    ) -> np.ndarray:
        """在原图灰度上做 cornerSubPix 精修，以减小缩放/阈值预处理带来的角点偏差。"""
        if gray.dtype != np.uint8:
            g = gray.astype(np.float32)
            mn = float(np.min(g))
            mx = float(np.max(g))
            if mx > mn:
                gray_u8 = ((g - mn) * (255.0 / (mx - mn))).astype(np.uint8)
            else:
                gray_u8 = np.zeros_like(g, dtype=np.uint8)
        else:
            gray_u8 = gray

        pts = corners_1x4x2.reshape(-1, 1, 2).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, eps)

        # winSize 是搜索窗口“半边长”，实际窗口为 (2*win+1)^2。
        # 支持自适应：根据 tag 像素边长自动估计，并在靠近边界时自动缩小。
        win = int(win_size) if (win_size is not None and int(win_size) > 0) else self._estimate_subpix_win_size(corners_1x4x2)
        win = self._cap_win_size_by_border(gray_u8, corners_1x4x2, win)

        # 失败兜底：如果当前 win 失败，逐步减小窗口重试（常见原因：靠近边界/局部梯度不够）
        last_err: Optional[Exception] = None
        for w_try in range(win, 0, -1):
            try:
                # zeroZone = (-1, -1) 表示不排除中心区域
                refined = cv2.cornerSubPix(
                    gray_u8,
                    pts,
                    (w_try, w_try),
                    (-1, -1),
                    criteria,
                )
                return refined.reshape(1, 4, 2)
            except Exception as e:
                last_err = e
                continue

        if last_err is not None:
            raise last_err
        return corners_1x4x2

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """使用CLAHE算法增强图像对比度。

        CLAHE原理:
        1. 将图像分割成小块（tileGridSize指定）
        2. 对每个小块独立做直方图均衡化
        3. 使用双线性插值平滑块之间的边界
        4. 限制对比度增强幅度（clipLimit防止噪声放大）

        为什么使用CLAHE而不是全局直方图均衡？
        - 全局均衡会丢失局部对比度细节
        - CLAHE能处理光照不均的图像（如一侧亮、一侧暗）
        - 适合标定板场景：标签可能分布在不同光照区域

        Args:
            image: 输入图像 (BGR或灰度)

        Returns:
            增强后的灰度图像 (uint8, 0-255)

        技术细节:
            clipLimit=3.0  → 限制对比度放大倍数为3倍
            tileGridSize=(8,8) → 将图像分成8×8=64个小块处理
        """
        # Step 1: 确保图像是灰度格式
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Step 2: 创建CLAHE对象
        # clipLimit: 对比度限制（1.0=无增强，更高=更强对比度，但噪声也会增强）
        # tileGridSize: 分块大小，8×8是经验值（太小→过度增强，太大→接近全局均衡）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Step 3: 应用增强
        enhanced = clahe.apply(gray)

        return enhanced

    def _detect_on_image(self, img: np.ndarray):
        """调用OpenCV的ArUco检测器进行标签检测。"""
        corners, ids, rejected = self.detector.detectMarkers(img)

        # 兼容性兜底：若使用 CORNER_REFINE_APRILTAG 导致 0 检测，则回退到 SUBPIX。
        # 只在失败时触发，避免影响正常路径性能。
        try:
            if (
                (ids is None or len(ids) == 0)
                and hasattr(self.parameters, "cornerRefinementMethod")
                and self.parameters.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_APRILTAG
            ):
                orig = self.parameters.cornerRefinementMethod
                self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                det_fb = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
                corners2, ids2, rejected2 = det_fb.detectMarkers(img)
                self.parameters.cornerRefinementMethod = orig
                if ids2 is not None and len(ids2) > 0:
                    return corners2, ids2, rejected2
        except Exception:
            pass

        return corners, ids, rejected

    def detect_multiscale_simple(
        self, image: np.ndarray
    ) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """简化版多尺度检测 - 快速版本，用于集成到现有代码。"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        enhanced = self.enhance_image(image)

        # 收集所有检测到的标签: tag_id -> (corners_4x2, score)
        all_detections: Dict[int, Tuple[np.ndarray, float]] = {}

        def _try_update(tag_id: int, corners_4x2: np.ndarray, *, score_scale: float = 1.0):
            score = self._score_corners(corners_4x2) * float(score_scale)
            prev = all_detections.get(tag_id)
            if prev is None or score > prev[1]:
                all_detections[tag_id] = (corners_4x2, score)

        # 1. 原始灰度图
        corners, ids, rejected = self._detect_on_image(gray)

        # OpenCV 官方 refine（基于 board 布局“捞回” rejected candidates）
        # 默认关闭；启用时要求提供 board。
        if (
            self.opencv_refine
            and self.board is not None
            and rejected is not None
            and len(rejected) > 0
            and hasattr(self.detector, "refineDetectedMarkers")
        ):
            try:
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    corners, ids, rejected, _ = self.detector.refineDetectedMarkers(
                        gray,
                        self.board,
                        corners,
                        ids,
                        rejected,
                        self.camera_matrix,
                        self.dist_coeffs,
                    )
                else:
                    corners, ids, rejected, _ = self.detector.refineDetectedMarkers(
                        gray,
                        self.board,
                        corners,
                        ids,
                        rejected,
                    )
            except Exception:
                pass

        if ids is not None:
            for idx, tag_id in enumerate(ids.flatten()):
                tag_id = int(tag_id)
                _try_update(tag_id, corners[idx][0])

        # 2. 增强图像
        corners, ids, _ = self._detect_on_image(enhanced)
        if ids is not None:
            for idx, tag_id in enumerate(ids.flatten()):
                tag_id = int(tag_id)
                _try_update(tag_id, corners[idx][0])

        # 3. 高斯模糊
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        corners, ids, _ = self._detect_on_image(blurred)
        if ids is not None:
            for idx, tag_id in enumerate(ids.flatten()):
                tag_id = int(tag_id)
                _try_update(tag_id, corners[idx][0])

        # 4. 上采样
        h, w = gray.shape[:2]
        # 小Tag：上采样更关键；同时对高倍率结果略加权（更倾向选择“高分辨率下定位”的角点）。
        for scale in [1.50, 2.0, 2.5, 3.0]:
            # 大图直接整图 3x 成本很高：用切块策略。
            if scale >= 2.0 and (h * w) >= 1_000_000:
                t_corners, t_ids = self._detect_tiled_upscale(gray, upscale=scale, tile_rows=2, tile_cols=2, overlap=0.25)
                if t_ids is not None and t_corners is not None:
                    for idx, tag_id in enumerate(t_ids.flatten()):
                        _try_update(int(tag_id), np.asarray(t_corners[idx], dtype=np.float32), score_scale=scale)
            else:
                upscaled = self._resize_gray(gray, scale)
                corners, ids, _ = self._detect_on_image(upscaled)
                if ids is not None:
                    for idx, tag_id in enumerate(ids.flatten()):
                        tag_id = int(tag_id)
                        _try_update(tag_id, corners[idx][0] / float(scale), score_scale=scale)

            # 对增强图做同样的高倍率尝试（通常对低对比/灯光不均更有效）
            if scale >= 2.0:
                if scale >= 2.0 and (h * w) >= 1_000_000:
                    t_corners, t_ids = self._detect_tiled_upscale(enhanced, upscale=scale, tile_rows=2, tile_cols=2, overlap=0.25)
                    if t_ids is not None and t_corners is not None:
                        for idx, tag_id in enumerate(t_ids.flatten()):
                            _try_update(int(tag_id), np.asarray(t_corners[idx], dtype=np.float32), score_scale=scale)
                else:
                    upscaled_e = self._resize_gray(enhanced, scale)
                    corners, ids, _ = self._detect_on_image(upscaled_e)
                    if ids is not None:
                        for idx, tag_id in enumerate(ids.flatten()):
                            tag_id = int(tag_id)
                            _try_update(tag_id, corners[idx][0] / float(scale), score_scale=scale)

        # 5. 下采样
        for scale in [0.25, 0.5, 0.75]:
            downscaled = self._resize_gray(gray, scale)
            corners, ids, _ = self._detect_on_image(downscaled)
            if ids is not None:
                for idx, tag_id in enumerate(ids.flatten()):
                    tag_id = int(tag_id)
                    _try_update(tag_id, corners[idx][0] / scale)

        # 6. 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        corners, ids, _ = self._detect_on_image(binary)
        if ids is not None:
            for idx, tag_id in enumerate(ids.flatten()):
                tag_id = int(tag_id)
                _try_update(tag_id, corners[idx][0])

        # 7. 自适应二值化
        adaptive_binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        corners, ids, _ = self._detect_on_image(adaptive_binary)
        if ids is not None:
            for idx, tag_id in enumerate(ids.flatten()):
                tag_id = int(tag_id)
                _try_update(tag_id, corners[idx][0])

        # 转换为与原API兼容的格式
        if len(all_detections) > 0:
            ordered_ids = sorted(all_detections.keys())
            corners_list = [all_detections[tag_id][0] for tag_id in ordered_ids]
            ids_array = np.array(ordered_ids).reshape(-1, 1)

            # ==== 角点顺序修正（与 core/detection.py 保持一致）====
            # 多尺度检测也需要应用相同的180°旋转修正
            # 详细说明见 mcca.core.detection.detect_apriltag_corners
            corrected_corners_list = []
            for c in corners_list:
                pts = c.reshape(4, 2)
                corrected_pts = np.array([pts[2], pts[3], pts[0], pts[1]])  # [2,3,0,1]
                corrected = corrected_pts.reshape(1, 4, 2)

                # 关键：在原图上做一次亚像素精修，尽量消除缩放/阈值预处理导致的角点偏差。
                try:
                    corrected = self._refine_corners_on_original(gray, corrected)
                except Exception:
                    # cornerSubPix 失败时回退到未精修角点
                    pass

                corrected_corners_list.append(corrected)

            return corrected_corners_list, ids_array
        else:
            return None, None


def detect_apriltag_multiscale(
    image: np.ndarray,
    aruco_dict: Optional[cv2.aruco.Dictionary] = None,
    detector_params: Optional[cv2.aruco.DetectorParameters] = None,
    verbose: bool = False,
    opencv_refine: bool = False,
    board: Optional[cv2.aruco.Board] = None,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Tuple[Optional[List], Optional[np.ndarray]]:
    """多尺度AprilTag检测 - 包装函数，兼容原有API。

    这是一个便利函数，直接调用 AdvancedAprilTagDetector.detect_multiscale_simple。

    Args:
        image: 输入图像 (BGR或灰度)
        aruco_dict: ArUco字典（用于选择 AprilTag family）
        detector_params: 检测器参数（会传入 OpenCV ArucoDetector）
        verbose: 是否输出详细信息
        opencv_refine: 是否启用 OpenCV 官方 refineDetectedMarkers（需要 board）
        board: cv2.aruco.Board，用于 refineDetectedMarkers
        camera_matrix: 可选相机内参（提供后 refine 更可靠）
        dist_coeffs: 可选畸变系数（提供后 refine 更可靠）

    Returns:
        corners: 检测到的角点列表，每个元素是 (1, 4, 2) 数组
        ids: 检测到的标签 ID 数组 (N, 1)

    注意：
        此函数返回的格式与 cv2.aruco.detectMarkers 完全兼容，可以直接替换原有检测调用。
    """
    detector = AdvancedAprilTagDetector(
        verbose=verbose,
        aruco_dict=aruco_dict,
        detector_params=detector_params,
        opencv_refine=opencv_refine,
        board=board,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    return detector.detect_multiscale_simple(image)


__all__ = [
    "AdvancedAprilTagDetector",
    "detect_apriltag_multiscale",
]
