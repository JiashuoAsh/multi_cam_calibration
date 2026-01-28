#!/usr/bin/env python3
"""[DEPRECATED] Step5a 实时采集脚本已归档。

本仓库当前用于“离线视频(mp4)标定”，不再提供硬件实时采集入口。

旧版 Step5a（实时采集）已移至：
  - archive/legacy_capture/step5a_capture_for_base.py

视频版 Step5 推荐做法：
  1) 先把视频抽帧到 images/step5/left,right/（文件名需与位姿记录对应）
  2) 再运行 step5b_camera_to_base.py
"""


def main() -> None:
    raise SystemExit(
        "step5a_capture_for_base.py 已弃用并归档到 archive/legacy_capture/。\n"
        "若使用视频：请先准备 images/step5/left,right 的图像对，然后运行 python step5b_camera_to_base.py"
    )


if __name__ == "__main__":
    main()
