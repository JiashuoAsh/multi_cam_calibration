# AprilTag 标定系统使用指南

## 📖 快速导航

- [系统概述](#系统概述)
- [安装配置](#安装配置)
- [完整流程](#完整流程)
- [各步骤详解](#各步骤详解)
- [结果使用](#结果使用)
- [故障排查](#故障排查)

---

## 系统概述

### 新5步标定流程

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: 原始图像采集                                    │
│  - 纯拍照，保存所有图像                                  │
│  - 无质量检查                                           │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│  Step 2: 图像质量检查和筛选                              │
│  - 检测 AprilTag 标签                                   │
│  - 筛选合格图像                                         │
│  - 生成筛选报告                                         │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│  Step 3: 内参标定                                        │
│  - 使用筛选后的图像                                      │
│  - 计算内参矩阵和畸变系数                                │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│  Step 4: 双目外参标定                                    │
│  - 计算左右相机相对位姿                                  │
│  - 计算立体校正参数                                      │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│  Step 5: 相机到底盘标定 (可选)                           │
│  - 计算相机到底盘坐标系变换                              │
└─────────────────────────────────────────────────────────┘
```

### 核心特点

✅ **采集和筛选分离** - 可以事后重新筛选，无需重拍
✅ **完整数据保留** - 所有原始图像都保存，便于分析
✅ **灵活可调** - 调整参数后可重新筛选
✅ **详细报告** - 筛选报告记录每张图像的质量信息
✅ **向后兼容** - 提供旧格式转换工具

---

## 安装配置

### 1. 安装依赖

```bash
pip install opencv-contrib-python numpy scipy
```

**重要**: 必须安装 `opencv-contrib-python`（包含 ArUco 模块）

### 2. 配置标定板参数

编辑 `apriltag_config.json`:

```json
{
  "apriltag_board": {
    "family": "tag36h11",       // AprilTag 类型
    "tags_x": 6,                 // 列数
    "tags_y": 6,                 // 行数
    "tag_size": 55.0,           // 标签尺寸 (mm)
    "tag_spacing": 16.5,        // 标签间距 (mm)
    "unit": "mm"
  },
  "calibration_settings": {
    "min_tags_for_pose": 12,    // 最少标签数（建议 1/3 以上）
    "max_images": 50             // 目标图像数量
  }
}
```

### 3. 配置相机类型

**USB 双目相机**:
```json
{
  "camera_settings": {
    "camera_type": "custom_usb_stereo",
    "device_path": "/dev/video40",
    "raw_width": 2560,
    "raw_height": 720,
    "image_width": 1280,
    "image_height": 960
  }
}
```

**MIPI 相机**:
```json
{
  "camera_settings": {
    "camera_type": "mipi",
    "left_camera_id": 22,
    "right_camera_id": 31,
    "image_width": 3840,
    "image_height": 2160
  }
}
```

---

## 完整流程

### 一键运行所有步骤

```bash
# 进入目录
cd camera_cali_apriltag

# Step 1: 采集原始图像
python step1_capture_imgs.py

# Step 2: 筛选合格图像
python step2_filter_images.py

# Step 3: 内参标定
python step3_intrinsic_apriltag.py

# Step 4: 双目外参标定
python step4_stereo_extrinsic.py

# Step 5: 相机到底盘标定（可选）
python step5a_capture_for_base.py
python step5b_camera_to_base.py

# 转换为旧格式（可选）
python convert_to_legacy_format.py
```

---

## 各步骤详解

### Step 1: 原始图像采集

**命令**: `python step1_capture_imgs.py`

**操作流程**:
1. 程序启动后显示实时相机画面
2. 移动机器人到不同位置和角度
3. 按 `s` 或空格保存当前图像对
4. 重复步骤2-3，采集20-50组图像
5. 按 `q` 退出

**采集要点**:
- ✅ 远近变化: 0.5m - 2m
- ✅ 角度变化: 俯仰、偏航、横滚
- ✅ 位置变化: 覆盖视野中心和边缘
- ✅ 姿态稳定: 避免运动模糊

**输出**:
```
images/raw/
├── left/
│   ├── 20251212_103045_123.png
│   ├── 20251212_103046_456.png
│   └── ...
└── right/
    ├── 20251212_103045_123.png
    ├── 20251212_103046_456.png
    └── ...
```

### Step 2: 图像质量检查和筛选

**命令**: `python step2_filter_images.py`

**功能**:
- 自动检测每张图像中的 AprilTag 标签
- 根据检测结果判断是否合格
- 复制合格图像到 filtered/ 目录
- 生成详细的筛选报告

**判定标准**:
- ✅ 左相机检测到标签数 ≥ min_tags_for_pose
- ✅ 右相机检测到标签数 ≥ min_tags_for_pose
- ✅ 左右图像必须都合格

**输出**:
```
images/filtered/
├── left/          # 筛选后的合格图像
└── right/

results/
└── filter_report.json  # 详细报告
```

**筛选报告示例**:
```json
{
  "timestamp": "2025-12-12T10:30:45",
  "total_pairs": 45,
  "valid_pairs": 38,
  "invalid_pairs": 7,
  "details": [
    {
      "left_file": "20251212_103045_123.png",
      "right_file": "20251212_103045_123.png",
      "left_tags": 18,
      "right_tags": 16,
      "is_valid": true
    },
    ...
  ]
}
```

**重新筛选**:
如果合格图像太少:
```bash
# 1. 调整配置文件中的 min_tags_for_pose
nano apriltag_config.json

# 2. 重新运行筛选（无需重新拍照）
python step2_filter_images.py
```

### Step 3: 内参标定

**命令**: `python step3_intrinsic_apriltag.py`

**功能**:
- 使用筛选后的图像进行内参标定
- 分别标定左右相机
- 计算内参矩阵 K 和畸变系数 dist
- 生成去畸变效果对比图

**输出**:
```
results/
├── left_intrinsics.json          # 左相机内参
├── right_intrinsics.json         # 右相机内参
├── left_undistortion_demo.jpg    # 去畸变效果
└── right_undistortion_demo.jpg
```

**内参文件格式**:
```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [k1, k2, p1, p2, k3, ...],
  "reprojection_error": 0.35,
  "image_size": [1280, 960]
}
```

**质量评估**:
- 重投影误差 < 0.5: ⭐⭐⭐ 优秀
- 重投影误差 < 1.0: ⭐⭐ 良好
- 重投影误差 > 1.0: ⭐ 需改进

### Step 4: 双目外参标定

**命令**: `python step4_stereo_extrinsic.py`

**功能**:
- 标定左右相机的相对位姿
- 计算旋转矩阵 R 和平移向量 t
- 计算立体校正参数
- 生成校正效果对比图

**输出**:
```
results/
├── stereo_extrinsics.json        # R, t, E, F, baseline
├── stereo_rectification.json     # R1, R2, P1, P2, Q
└── stereo_rectification_demo.jpg # 校正效果图
```

**质量检查**:
打开 `stereo_rectification_demo.jpg`，检查:
- ✅ 左右图像的水平线是否对齐
- ✅ 对应点是否在同一水平线上

### Step 5: 相机到底盘标定

**命令**:

- `python step5a_capture_for_base.py`（采集 Step5 专用图像到 `images/step5/`）
- `python step5b_camera_to_base.py`（计算并导出 `results/camera_to_base.json`）

**前提条件**:
1. AprilTag 标定板固定在墙上
2. 已知标定板相对底盘的位置

**配置**:
编辑 `apriltag_config.json`:
```json
{
  "board_to_base_transform": {
    "translation": [1.5, 0.0, 1.2],        // [X, Y, Z] 米（底盘B系：X右, Y上, Z前）
    "translation_reference": "tag0_center", // 默认：translation 表示 Tag0中心(T原点) 在B中的坐标；若量的是板中心可改为 board_center
    "rotation_euler_deg": [0, 0, 90],      // [roll, pitch, yaw] 度；严格定义见 step5b_camera_to_base.py（SciPy from_euler('XYZ', ...)）
    "note": "测量并更新这些值（Step5a+Step5b）"
  }
}
```

**输出**:
```
results/
└── camera_to_base.json  # B_T_Cl, B_T_Cr
```

### 转换为旧格式

**命令**: `python convert_to_legacy_format.py`

**功能**:
- 将新格式转换为旧代码兼容的格式
- 生成两种格式的文件

**输出**:
```
id_car_matrix_20251212.json  # OpenCV 矩阵格式
id_car_eula_20251212.json    # 欧拉角字符串格式
```

---

## 结果使用

### 加载标定参数

```python
import json
import numpy as np

# 加载内参
with open('results/left_intrinsics.json', 'r') as f:
    left_int = json.load(f)
K_l = np.array(left_int['camera_matrix'])
dist_l = np.array(left_int['dist_coeffs'])

# 加载双目外参
with open('results/stereo_extrinsics.json', 'r') as f:
    stereo = json.load(f)
R = np.array(stereo['R'])
t = np.array(stereo['t'])
baseline = stereo['baseline']
```

### 去畸变

```python
import cv2

img = cv2.imread('test.jpg')
h, w = img.shape[:2]

# 计算最优新相机矩阵
new_K, roi = cv2.getOptimalNewCameraMatrix(K_l, dist_l, (w, h), 1, (w, h))

# 去畸变
img_undist = cv2.undistort(img, K_l, dist_l, None, new_K)
```

### 立体校正

```python
# 加载校正参数
with open('results/stereo_rectification.json', 'r') as f:
    rect = json.load(f)
R1 = np.array(rect['R1'])
P1 = np.array(rect['P1'])

# 计算映射
map1, map2 = cv2.initUndistortRectifyMap(
    K_l, dist_l, R1, P1, (w, h), cv2.CV_32FC1
)

# 应用校正
img_rect = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
```

---

## 故障排查

### 问题1: 筛选后合格图像太少

**症状**: Step 2 报告合格图像 < 10

**解决**:
1. 降低 `min_tags_for_pose` 参数
2. 重新运行 step2（无需重拍）
3. 或补充拍摄更多图像

### 问题2: 重投影误差过大

**症状**: Step 3 重投影误差 > 1.0 像素

**原因**:
- 标定板尺寸测量不准确
- 标定板变形
- 图像模糊

**解决**:
- 重新精确测量标定板尺寸
- 更换刚性更好的标定板
- 增加采集姿态的多样性

### 问题3: 立体校正后水平线不对齐

**症状**: Step 4 的校正效果图中水平线不对齐

**原因**:
- 左右图像未同步
- 标定板未同时出现在左右视野

**解决**:
- 确保相机同步拍摄
- 重新采集，确保板在两侧都可见

---

## 文档资源

- [快速参考](docs/QUICK_START.md)
- [详细流程](docs/WORKFLOW.md)
- [重构说明](docs/REFACTORING_SUMMARY.md)
- [文档索引](docs/README.md)

---

**作者**: GitHub Copilot
**日期**: 2025-12-12
**版本**: 2.0
