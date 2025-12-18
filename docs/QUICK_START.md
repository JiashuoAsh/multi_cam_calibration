# 快速参考 - 新流程

## 命令速查

```bash
# 完整标定流程
python step1_capture_imgs.py      # 拍摄原始图像
python step2_filter_images.py     # 筛选合格图像
python step3_intrinsic_apriltag.py # 内参标定
python step4_stereo_extrinsic.py   # 双目外参标定
python step5a_capture_for_base.py  # Step5采集（可选）
python step5b_camera_to_base.py    # 相机到底盘标定（可选）
```

## 目录结构

```
images/
├── raw/          ← step1 输出
│   ├── left/
│   └── right/
└── filtered/     ← step2 输出
    ├── left/
    └── right/

results/
├── filter_report.json           ← step2 输出
├── left_intrinsics.json         ← step3 输出
├── right_intrinsics.json        ← step3 输出
├── stereo_extrinsics.json       ← step4 输出
├── stereo_rectification.json    ← step4 输出
└── camera_to_base.json          ← step5 输出
```

## 关键变化

| 步骤 | 旧名称 | 新名称 | 主要变化 |
|------|--------|--------|----------|
| 1 | step1 | step1 | 移除检测，纯拍照 |
| - | - | step2 | **新增**图像筛选 |
| 2 | step2 | step3 | 使用 filtered/ 目录 |
| 3 | step3 | step4 | 使用 filtered/ 目录 |
| 4 | step4 | step5 | 使用 filtered/ 目录 |

## 常见操作

### 重新筛选图像
```bash
# 无需重新拍照
python step2_filter_images.py
```

### 查看筛选报告
```bash
cat results/filter_report.json
```

### 从旧项目迁移
```bash
mkdir -p images/raw
mv images/left images/raw/
mv images/right images/raw/
python step2_filter_images.py
```

## 核心理念

✅ **采集** 和 **筛选** 分离
✅ **保留** 所有原始数据
✅ **灵活** 重新处理
✅ **追溯** 每个决策

详见: [WORKFLOW.md](WORKFLOW.md)
