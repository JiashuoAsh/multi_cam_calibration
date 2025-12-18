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
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict


def get_detection_settings(
        config: dict,
        *,
        default_use_multiscale: bool = True,
        default_opencv_refine: bool = False,
) -> Tuple[bool, bool]:
        """从配置文件中理解检测策略开关（兼容旧配置）。

        约定：
            config["calibration_settings"]["detection"] = {
                "use_multiscale": true/false,
                "opencv_refine": true/false
            }

        旧配置缺失该字段时：
            - use_multiscale 默认 True（保持项目当前“标定优先稳定”的风格）
            - opencv_refine 默认 False（避免无 board/K/dist 时改变行为）

        Returns:
                (use_multiscale, opencv_refine)
        """
        calib_cfg = config.get("calibration_settings", {}) if isinstance(config, dict) else {}
        det_cfg = calib_cfg.get("detection", {}) if isinstance(calib_cfg, dict) else {}

        use_multiscale = det_cfg.get("use_multiscale", default_use_multiscale)
        opencv_refine = det_cfg.get("opencv_refine", default_opencv_refine)
        return bool(use_multiscale), bool(opencv_refine)


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
    if use_multiscale:
        # 使用多尺度检测（推荐）
        from apriltag_detector import detect_apriltag_multiscale

        corners, ids = detect_apriltag_multiscale(
            image,
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

        return corners, ids
    else:
        # 使用标准检测
        if detector_params is None:
            detector_params = cv2.aruco.DetectorParameters()
            # 注意：CORNER_REFINE_APRILTAG 在 OpenCV 中是“AprilTag2 检测方式”，
            # 并非单纯的角点精修；在部分图片上会导致 0 检测。
            # 这里默认使用 SUBPIX 做角点亚像素精修（更稳）。
            detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # 创建检测器
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        # 检测标签
        corners, ids, rejected = detector.detectMarkers(image)

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
                corners2, ids2, rejected2 = detector_fallback.detectMarkers(image)
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
                        image,
                        board,
                        corners,
                        ids,
                        rejected,
                        camera_matrix,
                        dist_coeffs,
                    )
                else:
                    corners, ids, rejected, _ = detector.refineDetectedMarkers(
                        image,
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
        text = f"{status} Tags: {num_detected}/{min_tags}"

        cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        # 未检测到标签
        text = "✗ No tags detected"
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
        raise RuntimeError("Failed to open camera. Check configuration and hardware.")

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
        )

        if left_ids is None or right_ids is None:
            results.append((left_path, right_path, 0))
            continue

        left_set = set(left_ids.flatten())
        right_set = set(right_ids.flatten())
        common = len(left_set & right_set)

        results.append((left_path, right_path, common))

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
