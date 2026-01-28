#!/usr/bin/env python3
"""[DEPRECATED] 实时采集脚本已归档。

本仓库当前用于“离线视频(mp4)标定”。实时相机采集（USB/MIPI）相关脚本已移到：
  - archive/legacy_capture/step1_capture_imgs.py

如果你是用 mp4：请使用
  - step1_extract_imgs_from_video.py  (抽帧生成 images/raw)
  - step2_filter_images.py → step3_intrinsic_apriltag.py → step4_stereo_extrinsic.py
"""


def main() -> None:
    raise SystemExit(
        "step1_capture_imgs.py 已弃用并归档到 archive/legacy_capture/。\n"
        "离线视频工作流请运行：python step1_extract_imgs_from_video.py"
    )


if __name__ == "__main__":
    main()
