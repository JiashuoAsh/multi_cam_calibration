# AprilTag 双目相机标定流程

## 新流程说明

这个标定系统已经重新设计，将图像采集和质量检查分离，流程更加清晰和健壮。

## 流程步骤

### Step 1: 原始图像采集
```bash
python step1_capture_imgs.py
```

**功能:**
- 纯粹的图像采集工具
- **保存所有拍摄的图像**，无论是否包含标定板
- 不进行任何质量检查或检测

**输出:**
- `images/raw/left/*.png` - 原始左相机图像
- `images/raw/right/*.png` - 原始右相机图像

**优点:**
- 拍照过程快速，不受检测影响
- 所有图像都被保留，便于事后分析
- 可以在任何时候重新筛选图像

---

### Step 2: 图像质量检查和筛选
```bash
python step2_filter_images.py
```

**功能:**
- 从原始图像中筛选出合格的图像
- 检测 AprilTag 标定板
- 评估图像质量

**检查标准:**
1. 检测到足够数量的 AprilTag 标签
2. 左右图像都检测到标定板
3. 图像清晰度满足要求

**输出:**
- `images/filtered/left/*.png` - 筛选后的左相机图像
- `images/filtered/right/*.png` - 筛选后的右相机图像
- `results/filter_report.json` - 详细的筛选报告

---

### Step 3: 内参标定
```bash
python step3_intrinsic_apriltag.py
```

**功能:**
- 使用筛选后的图像进行内参标定
- 计算相机内参矩阵和畸变系数

**输入:**
- `images/filtered/left/*.png`
- `images/filtered/right/*.png`

**输出:**
- `results/left_intrinsics.json` - 左相机内参
- `results/right_intrinsics.json` - 右相机内参
- `results/left_undistortion_demo.jpg` - 去畸变效果对比
- `results/right_undistortion_demo.jpg`

---

### Step 4: 双目外参标定
```bash
python step4_stereo_extrinsic.py
```

**功能:**
- 标定双目相机的相对位姿
- 计算立体校正参数

**输入:**
- `images/filtered/left/*.png`
- `images/filtered/right/*.png`
- `results/left_intrinsics.json`
- `results/right_intrinsics.json`

**输出:**
- `results/stereo_extrinsics.json` - 双目外参 (R, t, E, F, baseline)
- `results/stereo_rectification.json` - 立体校正参数
- `results/stereo_rectification_demo.jpg` - 校正效果对比

---

### Step 5: 相机到底盘坐标系标定
```bash
python step5a_capture_for_base.py
python step5b_camera_to_base.py
```

**功能:**
- 计算相机坐标系到机器人底盘坐标系的变换

**前提条件:**
- AprilTag 标定板固定在墙上
- 已知标定板相对机器人底盘的固定位置（在配置文件中设置）

**输入:**
- `images/step5/left/*.png` / `*.jpg`
- `results/left_intrinsics.json`
- `results/stereo_extrinsics.json`
- `apriltag_config.json` (board_to_base_transform)

**输出:**
- `results/camera_to_base.json` - 相机到底盘的变换矩阵

---

### Step 6: 多组 Step5 结果融合（可选）
```bash
python step6_fuse_step5_results.py -i \
	run1/results/camera_to_base.json \
	run2/results/camera_to_base.json
```

**适用场景:**
- 你在不同“板位置/采集批次”重复运行 Step5b（每次都输出一份 `camera_to_base.json`）
- 希望把多次结果做鲁棒融合，得到更稳定的外参

**融合方法（简述）:**
- 平移：加权均值
- 旋转：四元数 Markley 平均
- 离群剔除：基于旋转角误差（度）+ 平移误差（米）的门限迭代

**输入:**
- 多个 `camera_to_base.json`（支持文件/目录/glob，会递归搜 `camera_to_base.json`）

**输出:**
- `results/camera_to_base_fused.json` - 融合后的外参
- `results/camera_to_base_fusion_report.json` - 每个样本的残差信息与内点/离群点标记

**注意事项:**
- 所有输入必须来自“同一坐标系定义/同一版本脚本”的 Step5 结果，否则会被识别为离群或直接跳过（例如旋转 det<0 的反射矩阵）。

---

## 目录结构

```
camera_cali_apriltag/
├── images/
│   ├── raw/                    # 原始采集的图像
│   │   ├── left/
│   │   └── right/
│   └── filtered/               # 筛选后的合格图像
│       ├── left/
│       └── right/
├── results/                    # 标定结果
│   ├── filter_report.json
│   ├── left_intrinsics.json
│   ├── right_intrinsics.json
│   ├── stereo_extrinsics.json
│   ├── stereo_rectification.json
│   └── camera_to_base.json
├── step1_capture_imgs.py       # 原始图像采集
├── step2_filter_images.py      # 图像质量检查和筛选
├── step3_intrinsic_apriltag.py # 内参标定
├── step4_stereo_extrinsic.py   # 双目外参标定
├── step5a_capture_for_base.py  # Step5 专用采集
├── step5b_camera_to_base.py    # 相机到底盘标定
├── step6_fuse_step5_results.py # 多组Step5结果融合（可选）
└── apriltag_config.json        # 配置文件
```

---

## 新流程的优势

### 1. 解耦合
- 图像采集和质量检查分离
- 每一步职责单一，更容易调试

### 2. 灵活性
- 可以随时重新筛选图像，无需重新拍摄
- 可以调整筛选标准后重新运行 step2

### 3. 鲁棒性
- 所有原始图像都被保留
- 坏照片也保存，便于分析问题

### 4. 可追溯性
- 筛选报告提供详细的质量信息
- 便于了解哪些图像被接受或拒绝

### 5. 效率
- 拍照过程更快（无检测延迟）
- 可以批量筛选，而不是边拍边检测

---

## 快速开始

1. 配置标定板参数（编辑 `apriltag_config.json`）
2. 运行 step1 采集原始图像
3. 运行 step2 筛选合格图像
4. 运行 step3 进行内参标定
5. 运行 step4 进行双目外参标定
6. （可选）运行 step5 进行相机到底盘标定
7. （可选）重复 Step5 多次后运行 step6 融合外参

---

## 常见问题

**Q: 如果筛选出的合格图像太少怎么办？**

A: 有两个选择：
1. 重新运行 step1 采集更多图像
2. 调整配置文件中的 `min_tags_for_pose` 参数，然后重新运行 step2

**Q: 可以删除原始图像吗？**

A: 在确认标定成功之前，建议保留原始图像。标定完成后可以删除 `images/raw/` 目录以节省空间。

**Q: 如何查看筛选报告？**

A: 查看 `results/filter_report.json` 文件，其中包含每张图像的详细检测信息。

---

## 与旧流程的对比

| 步骤 | 旧流程 | 新流程 |
|------|--------|--------|
| 1 | 拍照（带检测） | 纯拍照 |
| 2 | 内参标定 | 图像筛选 |
| 3 | 双目外参标定 | 内参标定 |
| 4 | 相机到底盘标定 | 双目外参标定 |
| 5 | - | 相机到底盘标定 |

**关键区别:**
- 新流程增加了独立的图像筛选步骤
- 原始图像和筛选后的图像分开存储
- 更好的可追溯性和灵活性
