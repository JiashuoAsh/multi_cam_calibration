"""占位入口（由 uv init 生成）。

说明：
- 本仓库的主要入口是 `run_calibration_pipeline.py`（一键跑完整标定流水线）。
- 也可以按步骤运行 step1~step6 脚本。

这个文件本身不参与标定逻辑，仅用于让 uv/工具链识别为可运行项目。
"""


def main() -> None:
    print("请运行 run_calibration_pipeline.py 或 step1~step6 脚本。")


if __name__ == "__main__":
    main()
