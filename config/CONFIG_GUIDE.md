# AprilTag 标定配置指南

本文档说明 `apriltag_config.json` 配置文件的各项参数。

## 配置文件结构

```json
{
  "camera_settings": {...},        // 相机配置
  "image_dataset": {...},          // （可选）多相机图片数据集输入（Step2/3/4）
  "step5_dataset": {...},          // （可选）Step5 图片数据集输入（Step5b）
  "camera_to_base_calibration": {...}, // （可选）Step5 求解模式选择（Step5b/Step5c）
  "apriltag_board": {...},         // 标定板配置
  "calibration_settings": {...},   // 标定设置
  "board_to_base_transform": {...} // 坐标变换配置
}
```

## 1. camera_settings - 相机配置

本仓库当前为 **video-only（离线 mp4）** 工作流：
- 先用 `step1_extract_imgs_from_video.py` 抽帧生成 `images/raw/<cam>/*.png`（默认 cam0/cam1）
- 再运行 Step2~Step4

因此 `camera_settings` 推荐只使用 `video_stereo`。

### 选项A（推荐）: 离线视频（双视频 / 单视频左右拼接）

当你已经有录制好的 mp4（例如两路 RGB 相机各录一个文件），推荐用视频抽帧脚本把视频转换成 `images/raw/<cam>/*.png`，然后直接复用 Step2~Step4。

抽帧脚本：`python step1_extract_imgs_from_video.py`（详见该脚本顶部注释）。

如果你希望把“视频”也当成一种 `camera_settings` 输入源（例如复用 `init_camera()` 的统一入口），可以将 `camera_type` 设置为 `video_stereo`。

#### A1. 双视频（two_files）

```json
{
  "camera_type": "video_stereo",
  "camera_names": ["cam0", "cam1"],
  "video_mode": "two_files",
  "video_paths": {"cam0": "C:/path/to/cam0.mp4", "cam1": "C:/path/to/cam1.mp4"},
  "rotate": {"cam0": "none", "cam1": "none"}
}
```

#### A2. 单视频左右拼接（single_sbs）

```json
{
  "camera_type": "video_stereo",
  "camera_names": ["cam0", "cam1"],
  "video_mode": "single_sbs",
  "video_path": "C:/path/to/stereo.mp4",
  "sbs_order": ["cam0", "cam1"],
  "rotate": {"cam0": "none", "cam1": "none"}
}
```

**参数说明（video_stereo）**:
- `camera_names`: 两路相机名（长度必须为 2）。`read_stereo()` 的返回顺序与其一致。
- `video_mode`:
  - `"two_files"`: 左右各一个视频文件（默认）
  - `"single_sbs"`: 单文件左右拼接，需要中线切割
- `video_paths`: 两路视频路径映射（two_files），键名必须与 `camera_names` 一致
- `video_path`: 拼接视频路径（single_sbs）
- `sbs_order`: 左右半区对应的相机名（single_sbs），例如 `["cam0","cam1"]` 表示左半 cam0、右半 cam1
- `rotate`: 按相机名指定 `none|cw90|ccw90|180`，用于修正方向

> 注意：即使配置了 `video_stereo`，Step2~Step4 也不会直接从视频读图；仍推荐先用 `step1_extract_imgs_from_video.py` 落盘成图片数据集。

## 1.5 video_extract - Step1 抽帧参数（推荐写入 config）

`step1_extract_imgs_from_video.py` 会从 `apriltag_config.json` 中读取 `video_extract` 作为抽帧参数。

```json
{
  "video_extract": {
    "every_n": 10,
    "max_pairs": 300,
    "start_frame": 0,
    "start_sec": 0.0,
    "out_dir": "images/raw",
    "prefix": "frame_",
    "overwrite": false
  }
}
```

运行方式（统一入口）：

- `python step1_extract_imgs_from_video.py --config config/apriltag_config.json`

> 说明：JSON 标准不支持 `//` 注释，本工程采用 `_comment` 字段作为“可解析注释”，不会影响解析。

## 1.6 image_dataset - 多相机图片数据集输入（Step2/Step3/Step4）

当你的数据源不是“固定的 images/raw/<cam>/”，或者你有 **3-4 路相机**，并希望 Step2/3/4 **无需手动改脚本/传参**就能自动跑完整流程时，使用该段。

> 建议 `enabled=true` 并显式配置 `cameras`，避免歧义。

### 最小示例（双目 cam0/cam1）

```json
{
  "image_dataset": {
    "enabled": true,
    "raw_root": "images/raw",
    "filtered_root": "images/filtered",
    "sync": {"key": "stem", "mode": "intersection"},
    "cameras": {
      "cam0": {"raw_dir": "images/raw/cam0"},
      "cam1": {"raw_dir": "images/raw/cam1"}
    }
  }
}
```

### 多相机示例（3-4 路，raw 图片可在任意目录）

```json
{
  "image_dataset": {
    "enabled": true,
    "filtered_root": "images/filtered",
    "sync": {"key": "stem", "mode": "intersection"},
    "cameras": {
      "cam0": {"raw_glob": "D:/dataset/run1/cam0/*.png"},
      "cam1": {"raw_glob": "D:/dataset/run1/cam1/*.png"},
      "cam2": {"raw_glob": "D:/dataset/run1/cam2/*.png"},
      "cam3": {"raw_glob": "D:/dataset/run1/cam3/*.png"}
    }
  }
}
```

字段说明：

- `enabled`：是否启用该段。启用后：
  - Step2 从 `cameras[*].raw_dir/raw_glob` 读取原始图片，并写入 `filtered_root/<cam>/...`
  - Step3 默认从 `filtered_root/<cam>/...` 读图做内参（为空则回退 raw）
  - Step4 默认从 `filtered_root/<cam>/...` 读图做多相机外参（位姿图）

- `raw_root` / `filtered_root`：默认根目录（可选）。
  - 若某个 camera 没写 `raw_dir/raw_glob`，则会回退到 `raw_root/<cam>`
  - filtered 输出默认写到 `filtered_root/<cam>`

- `cameras`：相机字典，key 就是相机名（例如 cam0/cam1/cam2...）。
  - `raw_dir`：目录形式（会自动扫描 png/jpg/jpeg/bmp）
  - `raw_glob`：glob 形式（支持字符串或字符串列表）

- `sync`：同步/对齐策略（主要影响 Step2 的统计与某些严格模式）。
  - `key=stem`：按文件名 stem 作为同步键（不含扩展名）
  - `mode=intersection`：只统计所有相机都存在的帧键（更严格，适合同步采集）
  - `mode=union`：统计所有出现过的帧键（更宽松）

### Legacy（不再内置支持）: USB/MIPI 实时相机

如果你确实要用真实相机实时采集，请使用历史版本或 `archive/legacy_capture/` 中的脚本作为参考（注意：当前 `libs/camera_wrapper.py` 已经精简为 video-only）。

#### Legacy-USB: USB双目拼接相机（已弃用）

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

#### Legacy-MIPI: MIPI独立相机（已弃用）

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
  "corner_refinement": "CORNER_REFINE_SUBPIX",
  "detection": {
    "use_multiscale": true,
    "opencv_refine": true,
    "profile": "balanced",
    "roi": null,
    "auto_roi": false,
    "detector_params": {}
  }
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
  - 推荐 `"CORNER_REFINE_SUBPIX"`（更稳，且不会改变检测策略）
  - `"CORNER_REFINE_APRILTAG"` 会切换到 OpenCV 的 AprilTag2 检测策略：对某些图像可能更强，但也可能出现“突然 0 检测”，建议结合本工程的 multiscale 回退机制使用

### 3.1 calibration_settings.detection - 检测策略（重要）

该段配置会影响 Step2/Step3 的 AprilTag 检测召回率与速度。

```json
{
  "use_multiscale": true,
  "opencv_refine": true,
  "profile": "balanced",
  "roi": null,
  "auto_roi": false,
  "detector_params": {}
}
```

**参数说明**:

- `use_multiscale`:
  - 是否启用多尺度/多预处理检测（更慢但更稳）
  - 远距离/小Tag 场景建议保持 `true`

- `opencv_refine`:
  - 是否启用 `refineDetectedMarkers`（可“捞回”部分 rejected candidates）
  - 建议保持 `true`

- `profile`:
  - 检测参数预设：
    - `balanced`: 默认（速度/召回平衡）
    - `small_tags`: 面向远距离/小Tag，提高召回（更慢，且可能略增误检）

- `roi`（可选但强烈推荐用于远距离小Tag）:
  - 目的：让检测集中在“标定板可能出现的区域”，减少干扰、提升召回，并让更激进的上采样/多尺度策略在 ROI 上跑得动
  - 支持写法：
    - 统一 ROI：`"roi": [x, y, w, h]`
    - 按相机分别 ROI：`"roi": {"cam0": [x,y,w,h], "cam1": [x,y,w,h], ...}`

- `auto_roi`（可选，推荐用于“板子在晃动/位置不固定”）:
  - 目的：当固定 ROI 不可靠时，自动用“两阶段”检测先定位标定板区域，再在该 ROI 上做主检测（可叠加 multiscale/上采样/refine）。
  - 支持写法：
    - 布尔值：`false`（默认关闭） / `true`（开启，使用默认参数）
    - 对象：`{"enabled": true, "pre_scale": 0.5, "min_tags": 1, "margin": 0.25}`
      - `pre_scale`: 粗检缩放比例，建议 `0.3 ~ 1.0`（越大越稳但更慢）
      - `min_tags`: 粗检阶段至少检出多少个 tag 才认为 ROI 有效（建议 1 或 2）
      - `margin`: ROI 外扩比例（相对外接框），建议 `0.15 ~ 0.40`
  - 注意：如果你显式设置了 `roi`（非 null），则 `auto_roi` 只会在 `roi=null` 时生效。

- `detector_params`（高级）:
  - 直接覆写 OpenCV `cv2.aruco.DetectorParameters` 字段（仅当你清楚含义时再用）
  - 例：
    - `{"adaptiveThreshWinSizeStep": 2, "minMarkerPerimeterRate": 0.02}`

> 提示：如果你要手动圈 ROI，推荐先运行 Step2 并打开 `results/visualization/step2_filtering/*` 的可视化输出，确认板子大概位置后再填 ROI。

## 4. step5_dataset - Step5（相机->底盘）图片数据集输入（多相机）

Step5 的目标是求每个相机到机器人底盘坐标系的外参 $B\_T\_C$。

Step5 需要一批“标定板固定安装在底盘坐标系中”的图片（每个相机各自一批），默认目录约定是：

- `images/step5/<cam>/*.png|jpg|jpeg|bmp`

如果你希望像 Step2/3/4 一样“完全由 config 指定图片路径”，请启用 `step5_dataset`：

```json
{
  "step5_dataset": {
    "enabled": true,
    "image_root": "images/step5",
    "cameras": {
      "cam0": {"raw_dir": "D:/data/step5/cam0"},
      "cam1": {"raw_glob": "D:/data/step5/cam1/*.png"}
    }
  }
}
```

注意事项：
- Step5 **不需要** Step2 的筛图产物（也可以用筛过的图，但不是必须）。
- Step5 **需要**每个相机的内参：`results/<cam>_intrinsics.json`（来自 Step3）。
- 若某些相机没有 Step5 图片，但你做过 Step4（相机间外参），Step5 会尝试用 Step4 的外参把 $B\_T\_C$ 从已有相机“传播”到其它相机（不如直接拍更稳）。
- Step5 的物理测量（`board_to_base_transform`）精度决定最终质量：建议反复核对单位/参考点。

## 4.5 camera_to_base_calibration - Step5 的另一种方式（world-anchor，无需 Step5 图片）

如果你已经能拿到“世界坐标系”下的绝对位姿（例如动捕 / SLAM / GNSS），你可以不拍 Step5 AprilTag 图片，直接通过坐标系推导得到相机->底盘外参。

该方式对应脚本：`step5c_camera_to_base_from_world.py`。

### 基本输入

你需要提供：

- 底盘在世界系的位姿：$W\_T\_B$（Base -> World）
- 某个参考相机在世界系的位姿：$W\_T\_{C\_ref}$（Cam_ref -> World）
- Step4 相机间外参（用于传播到其它相机）：
  - `results/multi_camera_extrinsics.json`（多相机 pose graph）

核心公式（本仓库约定 `A_T_B` 表示 B->A）：

$$
B\_T\_{C\_ref} = (W\_T\_B)^{-1} \cdot W\_T\_{C\_ref}
$$

然后通过 Step4 外参传播得到其它相机的 $B\_T\_C$。

### 配置示例

在 `apriltag_config.json` 里增加（或修改）如下段落：

- `camera_to_base_calibration.mode`: 设为 `"world_anchor"`
- `camera_to_base_calibration.world_anchor.reference_camera`: 参考相机名（需与你的 Step4 输出相机名一致）
- `camera_to_base_calibration.world_anchor.world_T_base`: $W\_T\_B$
- `camera_to_base_calibration.world_anchor.world_T_reference_camera`: $W\_T\_{C\_ref}$

注意：
- 平移单位：米
- 欧拉角单位：度
- `euler_order` 默认使用 `XYZ`（与 Step5b 的 `board_to_base_transform.rotation_euler_deg` 口径一致）

### 运行方式

- 单独运行：
  - `python step5c_camera_to_base_from_world.py --config config/apriltag_config.json`

- 使用流水线：
  - 当 `camera_to_base_calibration.mode=world_anchor` 时，`run_calibration_pipeline.py` 会自动选择 Step5c（即使未启用 step5_dataset）。

## 5. board_to_base_transform - 坐标变换配置

```json
{
  "translation": [1.5, 0.0, 1.2],
  "rotation_euler_deg": [0.0, 0.0, 90.0]
}
```

**参数说明**:
- `translation`: 标定板相对机器人底盘的平移向量 `[X, Y, Z]` (单位: 米)
- `rotation_euler_deg`: 标定板相对底盘的旋转角度 `[Roll, Pitch, Yaw]` (单位: 度)

**仅在使用 Step5（`step5b_camera_to_base.py`）时需要配置**

> 说明：本仓库当前推荐“离线图片/视频抽帧”的 Step5 流程。旧的实时采集 Step5a 已归档。

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
