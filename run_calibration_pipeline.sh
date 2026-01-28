#!/bin/bash
# 相机标定流水线 - 从视频抽帧到双目外参标定
# 用法: bash run_calibration_pipeline.sh

set -e  # 遇到错误立即退出
set -o pipefail  # 让 `python ... | tee ...` 在 python 失败时也能正确失败

# 设置 Python 输出编码为 UTF-8（解决 Windows GBK 编码问题）
export PYTHONIOENCODING=utf-8

echo "========================================"
echo "相机标定流水线开始执行"
echo "时间: $(date)"
echo "========================================"


# 统一入口：Python 流水线（会根据 config.image_dataset.enabled 自动选择双目/多相机流程）
echo ""
echo "[Pipeline] 运行 run_calibration_pipeline.py ..."
python run_calibration_pipeline.py --all --config config/apriltag_config.json | tee output.txt
if [ $? -ne 0 ]; then
    echo "[FAIL] 流水线失败" >&2
    exit 1
fi

echo ""
echo "========================================"
echo "标定流水线全部完成！"
echo "时间: $(date)"
echo "========================================"
echo ""
echo "输出文件位置："
echo "  - 汇总报告: results/pipeline_report.json"
echo "  - 日志目录: results/pipeline_logs/"
echo "  - 内参结果: results/<cam>_intrinsics.json"
echo "  - 外参结果: results/multi_camera_extrinsics.json（多相机）或 results/stereo_extrinsics.json（双目）"
