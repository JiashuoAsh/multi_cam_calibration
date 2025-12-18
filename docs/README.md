# 文档索引

## 📚 文档列表

### 新手入门
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - 📖 完整使用指南（推荐首次阅读）
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - ⚡ 命令速查表
- **[CONFIG_GUIDE.md](../CONFIG_GUIDE.md)** - ⚙️ 配置文件详解

### 详细说明
- **[WORKFLOW.md](WORKFLOW.md)** - 🔄 5步标定流程详解
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - 🔧 重构说明（从旧版升级）

### 其他
- **[CHANGELOG.md](../CHANGELOG.md)** - 📋 更新日志

## 🎯 推荐阅读顺序

### 首次使用
1. [主README](../README.md) - 项目概览
2. [USAGE_GUIDE.md](USAGE_GUIDE.md) - 完整使用教程
3. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 命令速查
4. [WORKFLOW.md](WORKFLOW.md) - 各步骤详解

### 从旧版升级
1. [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - 了解变更
2. [WORKFLOW.md](WORKFLOW.md) - 新流程说明

### 快速查询
直接查看 [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

## 📖 文档说明

| 文档 | 适用人群 | 主要内容 |
|------|----------|----------|
| USAGE_GUIDE.md | 所有用户 | 完整教程、配置说明、结果使用 |
| QUICK_REFERENCE.md | 快速查询 | 命令速查、配置速查、常见问题 |
| WORKFLOW.md | 详细了解 | 5步流程详解、优势分析 |
| REFACTORING_SUMMARY.md | 升级用户 | 变更记录、迁移指南 |

## 🔗 项目文件

| 类型 | 文件 | 说明 |
|------|------|------|
| 配置 | apriltag_config.json | 相机和标定板配置 |
| 工具 | utils.py | 核心函数库 |
| 步骤1 | step1_capture_imgs.py | 图像采集 |
| 步骤2 | step2_filter_images.py | 图像筛选 |
| 步骤3 | step3_intrinsic_apriltag.py | 内参标定 |
| 步骤4 | step4_stereo_extrinsic.py | 双目外参 |
| 步骤5a | step5a_capture_for_base.py | Step5 专用采集 |
| 步骤5b | step5b_camera_to_base.py | 相机到底盘 |
| 转换 | convert_to_legacy_format.py | 格式转换 |

## 💡 获取帮助

遇到问题时查看：
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 常见问题速查
2. [USAGE_GUIDE.md](USAGE_GUIDE.md) - 故障排查章节
3. 各脚本文件的详细注释
