#!/usr/bin/env python3
"""Camera Wrapper for AprilTag Calibration (video-only)

本仓库当前主流程以“离线视频(mp4)”为输入，因此这里只保留 video_stereo 的封装。

如果你需要恢复旧的硬件采集（USB/MIPI/Acemate_A2 hit 驱动）代码，请从历史版本或 archive 目录中找回。

统一接口：read_stereo() -> (left_frame, right_frame, timestamp_sec)
"""

import cv2
import numpy as np


class BaseCameraWrapper:
    """
    相机封装基类

    定义统一的相机接口，所有具体相机类型都继承此基类。
    确保所有相机类型提供一致的API，简化上层调用逻辑。

    Attributes:
        config (dict): 相机配置参数（从config/apriltag_config.json读取）
        is_open (bool): 相机是否已打开的标志
    """

    def __init__(self, config):
        """
        初始化相机封装

        Args:
            config (dict): 相机配置字典，包含相机类型和参数
        """
        self.config = config
        self.is_open = False

    def open(self):
        """
        打开相机设备

        子类必须实现此方法，完成相机初始化和连接。

        Returns:
            bool: 成功返回True，失败返回False
        """
        raise NotImplementedError

    def read_stereo(self):
        """
        读取双目图像对

        这是核心接口方法，所有相机类型必须实现。
        返回同步的左右图像和时间戳。

        Returns:
            tuple: (left_frame, right_frame, timestamp)
                - left_frame (np.ndarray): 左相机图像，BGR格式
                - right_frame (np.ndarray): 右相机图像，BGR格式
                - timestamp (float): 时间戳（秒），用于同步验证
            或 (None, None, None): 读取失败时
        """
        raise NotImplementedError

    def release(self):
        """
        释放相机资源

        子类必须实现此方法，完成相机关闭和资源清理。
        """
        raise NotImplementedError

    def is_opened(self):
        """
        检查相机是否已打开

        Returns:
            bool: 相机已打开返回True，否则返回False
        """
        return self.is_open


class VideoStereoCamera(BaseCameraWrapper):
    """基于视频文件的双目输入封装。

    适用场景：
    - 两路 RGB 相机各自录制为独立视频文件（left/right 两个 mp4）
    - 单个视频文件为左右拼接（side-by-side），需要按中线切割

    说明：
    - read_stereo() 的语义保持与硬件相机一致：返回 (left_frame, right_frame, timestamp_sec)
    - timestamp_sec 优先使用 CAP_PROP_POS_MSEC（若后端支持），否则用帧序号 / fps 估计

    配置字段（camera_settings）：
      - camera_type: 固定为 "video_stereo"
      - video_mode: "two_files" | "single_sbs"（默认 two_files）

      two_files 模式：
        - video_left_path: 左视频路径
        - video_right_path: 右视频路径

            single_sbs 模式：
        - video_path: 拼接视频路径
                - sbs_layout: "lr" | "rl" （默认 "rl"；rl 表示：右半->left，左半->right）

      可选：
        - rotate_left: "none"|"cw90"|"ccw90"|"180"（默认 none）
        - rotate_right: 同上（默认 none）
        - force_resize_width/force_resize_height: 强制 resize 到统一尺寸（谨慎使用，默认不启用）
    """

    def __init__(self, config):
        super().__init__(config)
        self.cap_left = None
        self.cap_right = None
        self.cap = None

        self.video_mode = str(config.get("video_mode", "two_files"))
        self.sbs_layout = str(config.get("sbs_layout", "rl"))

        self.rotate_left = str(config.get("rotate_left", "none"))
        self.rotate_right = str(config.get("rotate_right", "none"))

        self.force_resize_width = config.get("force_resize_width", None)
        self.force_resize_height = config.get("force_resize_height", None)

        self._frame_index = 0
        self._fps_left = None
        self._fps_right = None
        self._fps_single = None

    def open(self):
        try:
            if self.video_mode == "two_files":
                left_path = self.config.get("video_left_path")
                right_path = self.config.get("video_right_path")
                if not left_path or not right_path:
                    print("❌ Error: video_left_path / video_right_path is required for video_mode=two_files")
                    return False

                self.cap_left = cv2.VideoCapture(str(left_path))
                self.cap_right = cv2.VideoCapture(str(right_path))

                if not self.cap_left.isOpened():
                    print(f"❌ Error: Failed to open left video: {left_path}")
                    return False
                if not self.cap_right.isOpened():
                    print(f"❌ Error: Failed to open right video: {right_path}")
                    return False

                self._fps_left = float(self.cap_left.get(cv2.CAP_PROP_FPS) or 0.0)
                self._fps_right = float(self.cap_right.get(cv2.CAP_PROP_FPS) or 0.0)

                self.is_open = True
                print(f"✓ Opened stereo videos (two_files):\n  left={left_path}\n  right={right_path}")
                return True

            if self.video_mode == "single_sbs":
                path = self.config.get("video_path")
                if not path:
                    print("❌ Error: video_path is required for video_mode=single_sbs")
                    return False

                self.cap = cv2.VideoCapture(str(path))
                if not self.cap.isOpened():
                    print(f"❌ Error: Failed to open video: {path}")
                    return False

                self._fps_single = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

                self.is_open = True
                print(f"✓ Opened stereo video (single_sbs): {path} (layout={self.sbs_layout})")
                return True

            print(f"❌ Error: Unknown video_mode: {self.video_mode}. Supported: two_files, single_sbs")
            return False

        except Exception as e:
            print(f"❌ Error opening video stereo source: {e}")
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
                frame_l = self._maybe_resize(self._apply_rotate(frame_l, self.rotate_left))
                frame_r = self._maybe_resize(self._apply_rotate(frame_r, self.rotate_right))

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

                layout = (self.sbs_layout or "rl").lower()
                if layout == "rl":
                    # 与 USBStereoCamera / step1 逻辑一致：右半->left，左半->right
                    left_part = frame[:, mid:]
                    right_part = frame[:, :mid]
                elif layout == "lr":
                    left_part = frame[:, :mid]
                    right_part = frame[:, mid:]
                else:
                    raise ValueError(f"Unknown sbs_layout: {self.sbs_layout}. Use lr or rl")

                left_part = self._maybe_resize(self._apply_rotate(left_part, self.rotate_left))
                right_part = self._maybe_resize(self._apply_rotate(right_part, self.rotate_right))

                timestamp_sec = self._timestamp_from_cap(self.cap, self._fps_single or 0.0, self._frame_index)
                self._frame_index += 1
                return left_part, right_part, timestamp_sec

            return None, None, None

        except Exception as e:
            print(f"❌ Error reading from video stereo source: {e}")
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
    """
    相机工厂函数 - 根据配置创建相应的相机封装实例

    这是创建相机对象的统一入口，根据camera_type自动选择
    合适的相机类并实例化。

    Args:
        config (dict): 相机配置字典，必须包含'camera_type'字段
            支持的类型：
            - 'video_stereo': 离线视频双目输入（two_files / single_sbs）

    Returns:
        BaseCameraWrapper: 相机封装实例

    Raises:
        ValueError: 如果camera_type不支持

    Example:
        >>> config = {'camera_type': 'video_stereo', 'video_mode': 'two_files', 'video_left_path': 'left.mp4', 'video_right_path': 'right.mp4'}
        >>> camera = create_camera(config)
        >>> camera.open()
        >>> left, right, ts = camera.read_stereo()
    """
    camera_type = config.get("camera_type", "video_stereo")

    if camera_type == "video_stereo":
        return VideoStereoCamera(config)

    raise ValueError(
        f"Unknown camera type: {camera_type}. This workspace is configured for video-only calibration. Supported: video_stereo"
    )
