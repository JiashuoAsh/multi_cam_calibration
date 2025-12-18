#!/usr/bin/env python3
"""
Camera Wrapper for AprilTag Calibration

统一的相机接口封装，支持多种相机类型：
1. custom_usb_stereo: USB双目拼接相机（单设备输出2560宽度图像，自动分割旋转）
2. mipi: MIPI相机接口（使用Acemate_A2的VideoCaptureV4L2c）

核心设计：
- 统一的read_stereo()接口，返回(left_frame, right_frame, timestamp)
- 自动处理图像分割、旋转等预处理
- 支持时间戳同步
- 兼容Acemate_A2项目的相机接口

Author: Camera Calibration Team
Date: 2025-12
"""

import os
import sys
import cv2
import numpy as np


def import_acemate_hit():
    """
    动态导入Acemate_A2项目的VideoCaptureV4L2c_usb模块

    该函数尝试从多个可能的路径导入VideoCaptureV4L2c_usb模块，
    用于支持自定义USB双目拼接相机。

    搜索路径：
    1. ~/Acemate_A2/src/hit/hit/VideoCaptureV4L2c_usb
    2. ~/Acemate_A2/src/hit/VideoCaptureV4L2c_usb
    3. ~/Acemate_A2/install/hit/lib/python3.12/site-packages

    Returns:
        VideoCaptureV4L2c: VideoCaptureV4L2c_usb模块类
        None: 如果模块不可用
    """
    HOME_DIR = os.path.expanduser("~")

    # Try multiple possible paths
    possible_paths = [
        os.path.join(HOME_DIR, "Acemate_A2", "src", "hit", "hit"),  # Nested hit/
        os.path.join(HOME_DIR, "Acemate_A2", "src", "hit"),  # Original path
        os.path.join(
            HOME_DIR,
            "Acemate_A2",
            "install",
            "hit",
            "lib",
            "python3.12",
            "site-packages",
        ),
    ]

    for path in possible_paths:
        if os.path.isdir(path) and (path not in sys.path):
            sys.path.insert(0, path)

    try:
        from hit import VideoCaptureV4L2c_usb as VideoCaptureV4L2c

        return VideoCaptureV4L2c
    except ImportError as e:
        print(f"⚠ Warning: Could not import VideoCaptureV4L2c_usb: {e}")
        return None


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


class USBStereoCamera(BaseCameraWrapper):
    """
    USB双目拼接相机封装

    适用于单个USB设备输出2560x720拼接图像的双目相机。
    图像处理流程：
    1. 读取2560x720的拼接图像
    2. 中点分割：右半部分→左相机，左半部分→右相机
    3. 旋转校正：左图逆时针90°，右图顺时针90°
    4. 输出：两张1280x960的校正图像

    使用Acemate_A2的VideoCaptureV4L2c_usb接口访问硬件。

    Attributes:
        cap: VideoCaptureV4L2c相机对象
        VideoCaptureV4L2c: VideoCaptureV4L2c模块类
    """

    def __init__(self, config):
        """
        初始化USB双目拼接相机

        Args:
            config (dict): 必须包含:
                - device_path: 设备路径（如'/dev/video40'）
                - raw_width: 原始图像宽度（默认2560）
                - raw_height: 原始图像高度（默认720）
        """
        super().__init__(config)
        self.cap = None
        self.VideoCaptureV4L2c = None

    def open(self):
        """Open custom USB stereo camera"""
        # Import custom VideoCaptureV4L2c
        self.VideoCaptureV4L2c = import_acemate_hit()
        if self.VideoCaptureV4L2c is None:
            print(
                "❌ Error: VideoCaptureV4L2c_usb not found. Check Acemate_A2 installation."
            )
            return False

        device_path = self.config.get("device_path", "/dev/video40")
        raw_width = self.config.get("raw_width", 2560)
        raw_height = self.config.get("raw_height", 720)

        print(
            f"Opening custom USB stereo camera: {device_path} ({raw_width}x{raw_height})"
        )

        try:
            self.cap = self.VideoCaptureV4L2c.VideoCapture(
                device_path, width=raw_width, height=raw_height
            )

            if not (self.cap.open() and self.cap.start_capture()):
                print(f"❌ Error: Failed to open custom USB camera at {device_path}")
                return False

            self.is_open = True
            print(f"✓ Opened custom USB stereo camera: {device_path}")
            return True

        except Exception as e:
            print(f"❌ Error opening custom USB camera: {e}")
            return False

    def read_stereo(self):
        """
        读取拼接图像并分割为左右图像

        图像处理步骤（基于step1逻辑）：
        1. 从设备读取2560x720的拼接图像
        2. 中点分割：
           - 左相机 = 图像右半部分 [mid:end]
           - 右相机 = 图像左半部分 [0:mid]
        3. 旋转校正：
           - 左图：逆时针旋转90° (ROTATE_90_COUNTERCLOCKWISE)
           - 右图：顺时针旋转90° (ROTATE_90_CLOCKWISE)
        4. 时间戳转换：微秒→秒

        Returns:
            tuple: (left_frame, right_frame, timestamp_sec)
                - left_frame: 校正后的左图 (1280x960)
                - right_frame: 校正后的右图 (1280x960)
                - timestamp_sec: 时间戳（秒）
            或 (None, None, None): 读取失败
        """
        if not self.is_open:
            return None, None, None

        try:
            frame, timestamp = self.cap.read()  # type: ignore

            if frame is None:
                return None, None, None

            # Split image at midpoint
            h, w = frame.shape[:2]
            mid_point = w // 2

            # Step1 logic: right half -> left camera, left half -> right camera
            left_part = frame[:, mid_point:]  # Right half
            right_part = frame[:, :mid_point]  # Left half

            # Rotate to correct orientation
            left_part = cv2.rotate(left_part, cv2.ROTATE_90_COUNTERCLOCKWISE)
            right_part = cv2.rotate(right_part, cv2.ROTATE_90_CLOCKWISE)

            # Convert timestamp from microseconds to seconds
            timestamp_sec = timestamp * 1e-6

            return left_part, right_part, timestamp_sec

        except Exception as e:
            print(f"❌ Error reading from custom USB camera: {e}")
            return None, None, None

    def release(self):
        """Release custom camera"""
        if self.cap:
            self.cap.release()
        self.is_open = False


class MIPICameraWrapper(BaseCameraWrapper):
    """
    MIPI相机封装

    适用于使用MIPI CSI接口的独立左右相机。
    每个相机对应独立的/dev/videoX设备节点。

    特性：
    - 使用Acemate_A2的VideoCaptureV4L2c接口
    - 支持从config.CAMERA_ORDER自动获取设备ID
    - 支持高分辨率图像（如3840x2160）
    - 提供微秒级时间戳

    配置来源优先级：
    1. config/apriltag_config.json中的显式配置
    2. Acemate_A2的config.CAMERA_ORDER
    3. 默认分辨率：3840x2160

    Attributes:
        cap_left: 左相机VideoCapture对象
        cap_right: 右相机VideoCapture对象
        VideoCaptureV4L2c: VideoCaptureV4L2c模块类
        config_module: Acemate_A2的config模块（用于读取默认配置）
    """

    def __init__(self, config):
        """
        初始化MIPI相机

        Args:
            config (dict): 可选包含:
                - left_camera_id: 左相机设备号
                - right_camera_id: 右相机设备号
                - image_width: 图像宽度
                - image_height: 图像高度
        """
        super().__init__(config)
        self.cap_left = None
        self.cap_right = None
        self.VideoCaptureV4L2c = None
        self.config_module = None

    def open(self):
        """Open MIPI cameras using VideoCaptureV4L2c"""
        # Import Acemate_A2 modules
        VideoCaptureV4L2c = self._import_mipi_modules()
        if VideoCaptureV4L2c is None:
            print(
                "❌ Error: VideoCaptureV4L2c not found. Check Acemate_A2 installation."
            )
            return False

        self.VideoCaptureV4L2c = VideoCaptureV4L2c

        # Get camera IDs from config or Acemate_A2 config
        left_id = self._get_camera_id("left")
        right_id = self._get_camera_id("right")

        if left_id < 0 or right_id < 0:
            print(f"❌ Error: Invalid camera IDs (left={left_id}, right={right_id})")
            return False

        # Get resolution from config or Acemate_A2 config
        width = self.config.get(
            "image_width", getattr(self.config_module, "READ_CAMERA_WIDTH", 3840)
        )
        height = self.config.get(
            "image_height", getattr(self.config_module, "READ_CAMERA_HEIGHT", 2160)
        )

        print(
            f"Opening MIPI cameras: left=/dev/video{left_id}, right=/dev/video{right_id} ({width}x{height})"
        )

        try:
            # Open left camera
            self.cap_left = VideoCaptureV4L2c.VideoCapture(
                f"/dev/video{left_id}", width=width, height=height
            )
            if not (self.cap_left.open() and self.cap_left.start_capture()):
                print(
                    f"❌ Error: Failed to open left MIPI camera at /dev/video{left_id}"
                )
                return False

            # Open right camera
            self.cap_right = VideoCaptureV4L2c.VideoCapture(
                f"/dev/video{right_id}", width=width, height=height
            )
            if not (self.cap_right.open() and self.cap_right.start_capture()):
                print(
                    f"❌ Error: Failed to open right MIPI camera at /dev/video{right_id}"
                )
                self.cap_left.release()
                self.cap_left = None
                return False

            self.is_open = True
            print(f"✓ Opened MIPI cameras: left={left_id}, right={right_id}")
            return True

        except Exception as e:
            print(f"❌ Error opening MIPI cameras: {e}")
            return False

    def _import_mipi_modules(self):
        """Import VideoCaptureV4L2c and config from Acemate_A2"""
        HOME_DIR = os.path.expanduser("~")

        # Try multiple possible paths
        possible_paths = [
            os.path.join(HOME_DIR, "Acemate_A2", "src", "hit", "hit"),
            os.path.join(HOME_DIR, "Acemate_A2", "src", "hit"),
            os.path.join(HOME_DIR, "Acemate_A2", "src"),
        ]

        for path in possible_paths:
            if os.path.isdir(path) and (path not in sys.path):
                sys.path.insert(0, path)

        try:
            # Try to import config
            try:
                from hit.cfgs import config

                self.config_module = config
            except:
                try:
                    from hit import config

                    self.config_module = config
                except:
                    print("⚠ Warning: Could not import config from Acemate_A2")
                    self.config_module = None

            # Try to import VideoCaptureV4L2c
            try:
                from hit.camera import VideoCaptureV4L2c

                return VideoCaptureV4L2c
            except:
                try:
                    from hit import VideoCaptureV4L2c

                    return VideoCaptureV4L2c
                except Exception as e:
                    print(f"⚠ Warning: Could not import VideoCaptureV4L2c: {e}")
                    return None
        except Exception as e:
            print(f"⚠ Warning: Error importing MIPI modules: {e}")
            return None

    def _get_camera_id(self, side):
        """Get camera ID for left or right side"""
        # First try from apriltag_config
        if side == "left":
            cam_id = self.config.get("left_camera_id", -1)
        else:
            cam_id = self.config.get("right_camera_id", -1)

        # If not specified, try from Acemate_A2 config
        if cam_id < 0 and self.config_module:
            camera_order = getattr(self.config_module, "CAMERA_ORDER", [])
            if len(camera_order) >= 2:
                cam_id = camera_order[0] if side == "left" else camera_order[1]

        return cam_id

    def read_stereo(self):
        """Read from both MIPI cameras"""
        if not self.is_open:
            return None, None, None

        try:
            # Read from both cameras
            frame_left, ts_left = self.cap_left.read()
            frame_right, ts_right = self.cap_right.read()

            if frame_left is None or frame_right is None:
                return None, None, None

            # Use left camera timestamp (in microseconds), convert to seconds
            timestamp_sec = ts_left * 1e-6 if ts_left else 0

            return frame_left, frame_right, timestamp_sec

        except Exception as e:
            print(f"❌ Error reading from MIPI cameras: {e}")
            return None, None, None

    def release(self):
        """Release both MIPI cameras"""
        if self.cap_left:
            try:
                self.cap_left.release()
            except:
                pass
            self.cap_left = None

        if self.cap_right:
            try:
                self.cap_right.release()
            except:
                pass
            self.cap_right = None

        self.is_open = False


def create_camera(config):
    """
    相机工厂函数 - 根据配置创建相应的相机封装实例

    这是创建相机对象的统一入口，根据camera_type自动选择
    合适的相机类并实例化。

    Args:
        config (dict): 相机配置字典，必须包含'camera_type'字段
            支持的类型：
            - 'custom_usb_stereo': USB双目拼接相机
            - 'mipi': MIPI相机

    Returns:
        BaseCameraWrapper: 相机封装实例

    Raises:
        ValueError: 如果camera_type不支持

    Example:
        >>> config = {'camera_type': 'custom_usb_stereo', ...}
        >>> camera = create_camera(config)
        >>> camera.open()
        >>> left, right, ts = camera.read_stereo()
    """
    camera_type = config.get("camera_type", "custom_usb_stereo")

    if camera_type == "custom_usb_stereo":
        return USBStereoCamera(config)
    elif camera_type == "mipi":
        return MIPICameraWrapper(config)
    else:
        raise ValueError(
            f"Unknown camera type: {camera_type}. Supported: custom_usb_stereo, mipi"
        )


def test_camera(config):
    """Test camera initialization and frame capture"""
    print("\n" + "=" * 60)
    print("  Camera Test")
    print("=" * 60)

    camera = create_camera(config)

    print(f"\nCamera type: {config.get('camera_type', 'custom_usb_stereo')}")

    if not camera.open():
        print("❌ Failed to open camera")
        return False

    print("\nReading test frame...")
    left, right, timestamp = camera.read_stereo()

    if left is None or right is None:
        print("❌ Failed to read frames")
        camera.release()
        return False

    print(f"✓ Read successful!")
    print(f"  Left:  {left.shape}")
    print(f"  Right: {right.shape}")
    print(f"  Timestamp: {timestamp:.6f}")

    # Display frames
    scale = 0.5
    left_small = cv2.resize(left, None, fx=scale, fy=scale)
    right_small = cv2.resize(right, None, fx=scale, fy=scale)
    combined = np.hstack([left_small, right_small])

    cv2.imshow("Camera Test - Press any key to exit", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    camera.release()
    print("\n✓ Camera test completed successfully")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    # Test with default config
    test_config = {
        "camera_type": "custom_usb_stereo",
        "device_path": "/dev/video40",
        "raw_width": 2560,
        "raw_height": 720,
        "image_width": 1280,
        "image_height": 960,
    }

    test_camera(test_config)
