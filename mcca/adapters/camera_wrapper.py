#!/usr/bin/env python3
"""Camera Wrapper for AprilTag Calibration (video-only)

本仓库当前主流程以“离线视频(mp4)”为输入，因此这里只保留 video_stereo 的封装。

统一接口：read_stereo() -> (frame0, frame1, timestamp_sec)

说明：
    - frame0/frame1 的顺序由配置中的 camera_names 决定（默认 ["cam0", "cam1"]）。
    - 本模块属于 adapters 层：负责 IO/设备输入封装。
"""

import cv2
import numpy as np


class BaseCameraWrapper:
    """相机封装基类。

    定义统一的相机接口，所有具体相机类型都继承此基类。
    确保所有相机类型提供一致的 API，简化上层调用逻辑。

    Attributes:
        config: 相机配置参数（从 config/apriltag_config.json 读取）。
        is_open: 相机是否已打开的标志。
    """

    def __init__(self, config):
        """初始化相机封装。

        Args:
            config: 相机配置字典，包含相机类型和参数。
        """
        self.config = config
        self.is_open = False

    def open(self):
        """打开相机设备。

        子类必须实现此方法，完成相机初始化和连接。

        Returns:
            成功返回 True，失败返回 False。
        """
        raise NotImplementedError

    def read_stereo(self):
        """读取双目图像对。

        这是核心接口方法，所有相机类型必须实现。
        返回同步的左右图像和时间戳。

        Returns:
            (frame0, frame1, timestamp_sec) 或 (None, None, None)
        """
        raise NotImplementedError

    def release(self):
        """释放相机资源。"""
        raise NotImplementedError

    def is_opened(self):
        """检查相机是否已打开。"""
        return self.is_open


class VideoStereoCamera(BaseCameraWrapper):
    """基于视频文件的双目输入封装。

    适用场景：
    - 两路 RGB 相机各自录制为独立视频文件（两路 mp4）
    - 单个视频文件为左右拼接（side-by-side），需要按中线切割

    说明：
    - read_stereo() 的语义保持与硬件相机一致：返回 (left_frame, right_frame, timestamp_sec)
    - timestamp_sec 优先使用 CAP_PROP_POS_MSEC（若后端支持），否则用帧序号 / fps 估计

    配置字段（camera_settings）：
        - camera_type: 固定为 "video_stereo"
        - camera_names: 两路相机名，长度必须为 2（默认 ["cam0", "cam1"]）
        - video_mode: "two_files" | "single_sbs"（默认 two_files）

        two_files 模式：
            - video_paths: {"cam0": "a.mp4", "cam1": "b.mp4"}

        single_sbs 模式：
            - video_path: 拼接视频路径
            - sbs_order: ["cam0", "cam1"]  # 表示：左半为 cam0，右半为 cam1

        可选：
            - rotate: {"cam0": "none", "cam1": "cw90"}
            - force_resize_width/force_resize_height: 强制 resize 到统一尺寸（谨慎使用，默认不启用）
    """

    def __init__(self, config):
        super().__init__(config)
        self.cap_left = None
        self.cap_right = None
        self.cap = None

        self.video_mode = str(config.get("video_mode", "two_files"))

        camera_names = config.get("camera_names", ["cam0", "cam1"])
        if not isinstance(camera_names, (list, tuple)) or len(camera_names) != 2:
            raise ValueError("camera_settings.camera_names 必须是长度为 2 的列表，例如 ['cam0','cam1']")
        camera_names = [str(v) for v in camera_names]
        if camera_names[0] == camera_names[1] or (not camera_names[0]) or (not camera_names[1]):
            raise ValueError("camera_settings.camera_names 需要是两个不同且非空的相机名")
        self.camera_names = camera_names

        rotate_cfg = config.get("rotate", {})
        if rotate_cfg is None:
            rotate_cfg = {}
        if not isinstance(rotate_cfg, dict):
            raise ValueError("camera_settings.rotate 必须是 dict，例如 {'cam0':'none','cam1':'cw90'}")
        self.rotate = {
            self.camera_names[0]: str(rotate_cfg.get(self.camera_names[0], "none")),
            self.camera_names[1]: str(rotate_cfg.get(self.camera_names[1], "none")),
        }

        self.sbs_order = config.get("sbs_order", [self.camera_names[0], self.camera_names[1]])
        if not isinstance(self.sbs_order, (list, tuple)) or len(self.sbs_order) != 2:
            raise ValueError("camera_settings.sbs_order 必须是长度为 2 的列表，例如 ['cam0','cam1']")
        self.sbs_order = [str(v) for v in self.sbs_order]
        if set(self.sbs_order) != set(self.camera_names):
            raise ValueError("camera_settings.sbs_order 必须由 camera_names 的两个名字组成（顺序表示左右半区）")

        self.force_resize_width = config.get("force_resize_width", None)
        self.force_resize_height = config.get("force_resize_height", None)

        self._frame_index = 0
        self._fps_left = None
        self._fps_right = None
        self._fps_single = None

    def open(self):
        try:
            if self.video_mode == "two_files":
                video_paths = self.config.get("video_paths", None)
                if not isinstance(video_paths, dict):
                    print("Error: camera_settings.video_paths is required for video_mode=two_files")
                    return False

                path0 = video_paths.get(self.camera_names[0], None)
                path1 = video_paths.get(self.camera_names[1], None)
                if not path0 or not path1:
                    print("Error: camera_settings.video_paths 必须包含两路视频路径，键名与 camera_names 一致")
                    return False

                self.cap_left = cv2.VideoCapture(str(path0))
                self.cap_right = cv2.VideoCapture(str(path1))

                if not self.cap_left.isOpened():
                    print(f"Error: Failed to open video for {self.camera_names[0]}: {path0}")
                    return False
                if not self.cap_right.isOpened():
                    print(f"Error: Failed to open video for {self.camera_names[1]}: {path1}")
                    return False

                self._fps_left = float(self.cap_left.get(cv2.CAP_PROP_FPS) or 0.0)
                self._fps_right = float(self.cap_right.get(cv2.CAP_PROP_FPS) or 0.0)

                self.is_open = True
                print(
                    "Opened stereo videos (two_files):\n"
                    f"  {self.camera_names[0]}={path0}\n"
                    f"  {self.camera_names[1]}={path1}"
                )
                return True

            if self.video_mode == "single_sbs":
                path = self.config.get("video_path")
                if not path:
                    print("Error: camera_settings.video_path is required for video_mode=single_sbs")
                    return False

                self.cap = cv2.VideoCapture(str(path))
                if not self.cap.isOpened():
                    print(f"Error: Failed to open video: {path}")
                    return False

                self._fps_single = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

                self.is_open = True
                print(
                    "Opened stereo video (single_sbs): "
                    f"{path} (sbs_order={self.sbs_order}, camera_names={self.camera_names})"
                )
                return True

            print(f"Error: Unknown video_mode: {self.video_mode}. Supported: two_files, single_sbs")
            return False

        except Exception as e:
            print(f"Error opening video stereo source: {e}")
            return False

    def _apply_rotate(self, img, rotate_mode: str):
        if img is None:
            return None

        rotate_mode = (rotate_mode or "none").lower()
        if rotate_mode == "none":
            return img
        if rotate_mode == "cw90":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if rotate_mode == "ccw90":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if rotate_mode == "180":
            return cv2.rotate(img, cv2.ROTATE_180)

        raise ValueError(f"Unknown rotate mode: {rotate_mode}. Use one of: none, cw90, ccw90, 180")

    def _maybe_resize(self, img):
        if img is None:
            return None
        if self.force_resize_width is None or self.force_resize_height is None:
            return img
        try:
            w = int(self.force_resize_width)
            h = int(self.force_resize_height)
            if w <= 0 or h <= 0:
                return img
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        except Exception:
            return img

    def _timestamp_from_cap(self, cap, fps: float, frame_index: int) -> float:
        """尽力从 cap 获取时间戳（秒）。"""
        ts_sec = None
        try:
            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if pos_msec > 0:
                ts_sec = pos_msec / 1000.0
        except Exception:
            ts_sec = None

        if ts_sec is None:
            if fps and fps > 0:
                ts_sec = float(frame_index) / float(fps)
            else:
                # 最差兜底：用帧序号当“时间”
                ts_sec = float(frame_index)

        return float(ts_sec)

    def read_stereo(self):
        if not self.is_open:
            return None, None, None

        try:
            if self.video_mode == "two_files":
                assert self.cap_left is not None and self.cap_right is not None

                ok_l, frame_l = self.cap_left.read()
                ok_r, frame_r = self.cap_right.read()
                if (not ok_l) or (not ok_r) or frame_l is None or frame_r is None:
                    return None, None, None

                # 处理旋转/resize
                frame_l = self._maybe_resize(
                    self._apply_rotate(frame_l, self.rotate.get(self.camera_names[0], "none"))
                )
                frame_r = self._maybe_resize(
                    self._apply_rotate(frame_r, self.rotate.get(self.camera_names[1], "none"))
                )

                # 时间戳（两路取平均；若一侧不可用则取另一侧）
                ts_l = self._timestamp_from_cap(self.cap_left, self._fps_left or 0.0, self._frame_index)
                ts_r = self._timestamp_from_cap(self.cap_right, self._fps_right or 0.0, self._frame_index)

                ts_candidates = [v for v in [ts_l, ts_r] if v is not None]
                timestamp_sec = float(np.mean(ts_candidates)) if ts_candidates else float(self._frame_index)

                self._frame_index += 1
                return frame_l, frame_r, timestamp_sec

            if self.video_mode == "single_sbs":
                assert self.cap is not None
                ok, frame = self.cap.read()
                if (not ok) or frame is None:
                    return None, None, None

                h, w = frame.shape[:2]
                mid = w // 2
                if mid <= 0:
                    return None, None, None

                left_half = frame[:, :mid]
                right_half = frame[:, mid:]

                # 按 sbs_order 解释左右半区对应哪路相机
                by_name = {
                    self.sbs_order[0]: left_half,
                    self.sbs_order[1]: right_half,
                }

                frame0 = by_name[self.camera_names[0]]
                frame1 = by_name[self.camera_names[1]]

                frame0 = self._maybe_resize(
                    self._apply_rotate(frame0, self.rotate.get(self.camera_names[0], "none"))
                )
                frame1 = self._maybe_resize(
                    self._apply_rotate(frame1, self.rotate.get(self.camera_names[1], "none"))
                )

                timestamp_sec = self._timestamp_from_cap(self.cap, self._fps_single or 0.0, self._frame_index)
                self._frame_index += 1
                return frame0, frame1, timestamp_sec

            return None, None, None

        except Exception as e:
            print(f"Error reading from video stereo source: {e}")
            return None, None, None

    def release(self):
        try:
            if self.cap_left is not None:
                self.cap_left.release()
            if self.cap_right is not None:
                self.cap_right.release()
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap_left = None
        self.cap_right = None
        self.cap = None
        self.is_open = False


def create_camera(config):
    """相机工厂函数 - 根据配置创建相应的相机封装实例。"""
    camera_type = config.get("camera_type", "video_stereo")

    if camera_type == "video_stereo":
        return VideoStereoCamera(config)

    raise ValueError(
        f"Unknown camera type: {camera_type}. This workspace is configured for video-only calibration. Supported: video_stereo"
    )


def init_camera(config: dict):
    """根据顶层 config 初始化并打开相机。

    说明：
        历史上该函数位于根目录 `utils.py`；Phase1 迁移后将其归入 adapters 层，
        避免 core 反向依赖 IO。

    Args:
        config: 顶层配置 dict（通常来自 `load_config()`）。

    Returns:
        已打开的相机封装实例。

    Raises:
        RuntimeError: 打开视频源失败。
    """
    camera_settings = config.get("camera_settings", {})
    camera = create_camera(camera_settings)
    if not camera.open():
        raise RuntimeError("Failed to open video source. Check camera_settings and video file paths/codecs.")
    return camera
