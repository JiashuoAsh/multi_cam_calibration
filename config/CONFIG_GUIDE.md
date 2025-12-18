# AprilTag 标定配置指南

本文档说明 `apriltag_config.json` 配置文件的各项参数。

## 配置文件结构

```json
{
  "camera_settings": {...},        // 相机配置
  "apriltag_board": {...},         // 标定板配置
  "calibration_settings": {...},   // 标定设置
  "board_to_base_transform": {...} // 坐标变换配置
}
```

## 1. camera_settings - 相机配置

### 选项A: USB双目拼接相机

```json
{
  "camera_type": "custom_usb_stereo",
  "device_path": "/dev/video40",
  "raw_width": 2560,
  "raw_height": 720,
  "image_width": 1280,
  "image_height": 720
}
```

**参数说明**:
- `camera_type`: 固定为 `"custom_usb_stereo"`
- `device_path`: V4L2设备路径（通过 `v4l2-ctl --list-devices` 查看）
- `raw_width`: 原始捕获宽度（通常为左右拼接后的总宽度，如2560）
- `raw_height`: 原始捕获高度
- `image_width`: 单侧相机图像宽度（通常为 raw_width/2）
- `image_height`: 单侧相机图像高度

### 选项B: MIPI独立相机

```json
{
  "camera_type": "mipi",
  "left_camera_id": 22,
  "right_camera_id": 31,
  "image_width": 3840,
  "image_height": 2160
}
```

**参数说明**:
- `camera_type`: 固定为 `"mipi"`
- `left_camera_id`: 左相机设备ID
- `right_camera_id`: 右相机设备ID
- `image_width`: 单个相机图像宽度
- `image_height`: 单个相机图像高度

## 2. apriltag_board - 标定板配置

```json
{
  "family": "tag36h11",
  "tags_x": 6,
  "tags_y": 6,
  "tag_size": 55.0,
  "tag_spacing": 16.5,
  "unit": "mm"
}
```

**参数说明**:
- `family`: AprilTag类型，支持:
  - `"tag16h5"`: 16位编码，5位汉明距离
  - `"tag25h9"`: 25位编码，9位汉明距离
  - `"tag36h10"`: 36位编码，10位汉明距离
  - `"tag36h11"`: 36位编码，11位汉明距离（推荐）
- `tags_x`: 横向标签数量（列数）
- `tags_y`: 纵向标签数量（行数）
- `tag_size`: **单个标签的边长**（单位: mm）
- `tag_spacing`: **标签之间的间距**（边到边的距离，单位: mm）
- `unit`: 长度单位（固定为 "mm"）

**⚠️ 重要**:
- `tag_size` 和 `tag_spacing` 必须**精确测量**，误差会影响标定精度
- `family` 必须与实际标定板匹配
- 标签ID从0开始，按行优先顺序排列（0,1,2...）

**测量方法**:
```
测量工具: 精确卡尺或尺子
tag_size: 测量标签黑色边框的边长
tag_spacing: 测量相邻两个标签边框之间的距离
```

## 3. calibration_settings - 标定设置

```json
{
  "min_tags_for_pose": 4,
  "max_images": 50,
  "corner_refinement": "CORNER_REFINE_APRILTAG"
}
```

**参数说明**:
- `min_tags_for_pose`:
  - 进行位姿估计所需的最少标签数
  - 建议值: 4-10
  - 值越大，要求越严格，但精度更高
- `max_images`:
  - 目标采集的图像组数
  - 建议值: 20-50
  - 姿态变化充分的前提下，30组即可
- `corner_refinement`:
  - 角点优化方法
  - 固定为 `"CORNER_REFINE_APRILTAG"`（AprilTag专用）

## 4. board_to_base_transform - 坐标变换配置

```json
{
  "translation": [1.5, 0.0, 1.2],
  "rotation_euler_deg": [0.0, 0.0, 90.0]
}
```

**参数说明**:
- `translation`: 标定板相对机器人底盘的平移向量 `[X, Y, Z]` (单位: 米)
- `rotation_euler_deg`: 标定板相对底盘的旋转角度 `[Roll, Pitch, Yaw]` (单位: 度)

**仅在使用 Step5（`step5a_capture_for_base.py` + `step5b_camera_to_base.py`）时需要配置**

**坐标系定义（重要：以代码实现为准）**:

- 底盘坐标系 **B**: $+X$ 右，$+Y$ 上，$+Z$ 前
- 标定板坐标系 **T**: $+X$ 向右（tag 列方向增大），$+Y$ 向上（tag 行方向增大），$+Z = +X \times +Y$（右手系，垂直于板面）

### rotation_euler_deg 的严格定义（彻底定死）

在 `step5b_camera_to_base.py` 中，`rotation_euler_deg = [roll, pitch, yaw]` 被解释为：

> 直接按 SciPy 生成 $R_{B\leftarrow T}$：
> $$R_{B\leftarrow T}=\texttt{Rotation.from\_euler("XYZ", [roll, pitch, yaw], degrees=True).as\_matrix()}$$

- **"XYZ" 为内旋（body-fixed / intrinsic）**：依次绕当前（随旋转更新的）$X$、$Y$、$Z$ 轴旋转
- **正方向**：全部遵循**右手定则**（拇指指向轴正向，四指弯曲方向为正角度）
- $R_{B\leftarrow T}$ 的含义：把 $T$ 中的向量/点旋到 $B$ 中（配合全工程命名 `A_T_B`）

#### 角度轴向对照表

| 字段 | 绕哪根轴 | 正角度方向（右手定则） | 一个快速自检（在 roll=pitch=0 时） |
|---|---|---|---|
| Roll  | $+X$ | $+Y \rightarrow +Z$ | roll=0 不变 |
| Pitch | $+Y$ | $+Z \rightarrow +X$ | pitch=0 不变 |
| Yaw   | $+Z$ | $+X \rightarrow +Y$ | yaw=+90° 时：$X_T \mapsto +Y_B$ |

### translation 的参考点（避免“我量的是板中心但代码用 Tag0”）

默认情况下 `translation` 表示 **T 原点（Tag0 中心）在 B 中的坐标（米）**。

如果你现场测量的是“标定板中心 / tag 网格中心”等其它参考点，请在 `apriltag_config.json` 的 `board_to_base_transform` 中设置：

- `translation_reference`: `"tag0_center"`（默认）/ `"board_center"` / `"grid_center"`
- `translation_reference_point_in_T_m`: 该参考点在 **T** 中的位置（米，三维向量）

程序会自动把“参考点平移”换算成“Tag0 原点平移”，保证与 solvePnP 使用的 T 坐标一致。

## 常见配置示例

### 示例1: A500-6×6 标定板 + USB相机

```json
{
  "camera_settings": {
    "camera_type": "custom_usb_stereo",
    "device_path": "/dev/video40",
    "raw_width": 2560,
    "raw_height": 720,
    "image_width": 1280,
    "image_height": 720
  },
  "apriltag_board": {
    "family": "tag36h11",
    "tags_x": 6,
    "tags_y": 6,
    "tag_size": 55.0,
    "tag_spacing": 16.5,
    "unit": "mm"
  },
  "calibration_settings": {
    "min_tags_for_pose": 6,
    "max_images": 30,
    "corner_refinement": "CORNER_REFINE_APRILTAG"
  }
}
```

### 示例2: 4×4 标定板 + MIPI相机

```json
{
  "camera_settings": {
    "camera_type": "mipi",
    "left_camera_id": 22,
    "right_camera_id": 31,
    "image_width": 3840,
    "image_height": 2160
  },
  "apriltag_board": {
    "family": "tag36h11",
    "tags_x": 4,
    "tags_y": 4,
    "tag_size": 80.0,
    "tag_spacing": 20.0,
    "unit": "mm"
  },
  "calibration_settings": {
    "min_tags_for_pose": 4,
    "max_images": 40,
    "corner_refinement": "CORNER_REFINE_APRILTAG"
  }
}
```

## 配置检查

运行以下命令验证配置文件：

```bash
python -c "import json; print(json.dumps(json.load(open('apriltag_config.json')), indent=2))"
```

## 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 检测不到标签 | family配置错误 | 确认标定板的实际family类型 |
| 重投影误差大 | 尺寸测量不准 | 重新精确测量tag_size和tag_spacing |
| 相机无法打开 | device_path错误 | 检查V4L2设备路径 |
| 标签ID错误 | family不匹配 | 核对标定板类型 |

---

**相关文档**:
- [使用指南](docs/USAGE_GUIDE.md)
- [快速参考](docs/QUICK_REFERENCE.md)
- [主README](README.md)
