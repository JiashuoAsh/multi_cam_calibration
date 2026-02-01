#!/usr/bin/env python3
"""Step 1 (Video): 从视频抽帧生成 images/raw 双相机数据集

目的：
- 把离线视频（双视频 or 单视频左右拼接）转换为本项目统一的数据集格式：
    images/raw/cam0/*.png + images/raw/cam1/*.png
- 后续直接复用：
    python -m mcca.entry.step2_filter_images → python -m mcca.entry.step3_intrinsic_apriltag → python -m mcca.entry.step4_multi_extrinsic

典型用法：
1) 统一从配置读取（推荐）：
    python -m mcca.entry.step1_extract_imgs_from_video --config config/apriltag_config.json

说明：
- 本脚本只负责“抽帧落盘”，不做 AprilTag 检测筛选；筛选交给 step2。
- 视频输入与抽帧参数统一写在 config 中：
    - camera_settings：视频路径 / 模式 / camera_names / sbs_order / 旋转 / resize
    - video_extract：every_n / max_pairs / start_frame / start_sec / out_dir / prefix / overwrite
- 配置文件使用 JSON 标准语法，不支持 // 注释；本工程采用 "_comment" 字段作为可解析的“注释”。

说明：
- 本脚本只负责“抽帧落盘”，不做 AprilTag 检测筛选；筛选交给 step2。
- 输出文件名使用 frame 序号零填充，确保左右按排序 zip 仍能一一对应。
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2

from mcca.core.config import load_config


@dataclass
class ExtractArgs:
    every_n: int
    max_pairs: int
    start_frame: int
    start_sec: float
    out_dir: str
    prefix: str
    overwrite: bool


def _resolve_path_maybe(p: Optional[str], *, config_dir: Path) -> Optional[str]:
    """尽力解析路径：

    - 绝对路径：原样返回
    - 相对路径：若以 config 所在目录为基准能找到文件，则转为该绝对路径
      否则保留原相对路径（交给 OpenCV 按当前工作目录解析）
    """

    if not p:
        return None
    try:
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        cand = (config_dir / pp).resolve()
        if cand.exists():
            return str(cand)
        return str(pp)
    except Exception:
        return str(p)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _maybe_clear_dir(p: Path) -> None:
    if not p.exists():
        return
    for f in p.glob("*"):
        if f.is_file():
            f.unlink()


def _try_seek(camera, *, start_frame: int, start_sec: float) -> int:
    """尽力把视频定位到 start_frame / start_sec。

    返回：实际采用的 start_frame（用于命名偏移与日志）。
    """

    # 优先使用 start_sec（更符合“从某个时间开始抽帧”的直觉）
    if start_sec > 0:
        target_msec = float(start_sec) * 1000.0
        used_any = False
        for cap_name in ("cap", "cap_left", "cap_right"):
            cap = getattr(camera, cap_name, None)
            if cap is None:
                continue
            try:
                ok = bool(cap.set(cv2.CAP_PROP_POS_MSEC, target_msec))
                used_any = used_any or ok
            except Exception:
                pass
        if used_any:
            # 尝试推一个近似 frame_index
            fps = None
            for fps_name in ("_fps_single", "_fps_left", "_fps_right"):
                v = getattr(camera, fps_name, None)
                if isinstance(v, (int, float)) and v and v > 0:
                    fps = float(v)
                    break
            if fps and fps > 0:
                start_frame = int(round(start_sec * fps))

    if start_frame > 0:
        for cap_name in ("cap", "cap_left", "cap_right"):
            cap = getattr(camera, cap_name, None)
            if cap is None:
                continue
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
            except Exception:
                pass

    # 同步 wrapper 内部帧计数（用于 timestamp 推导）
    if hasattr(camera, "_frame_index"):
        try:
            camera._frame_index = int(start_frame)
        except Exception:
            pass

    return int(start_frame)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从视频抽帧生成 images/raw 图像对（供 Step2~Step4 复用）")

    g = parser.add_argument_group("配置")
    g.add_argument(
        "--config",
        type=str,
        default="config/apriltag_config.json",
        help="统一配置入口：从该 JSON 文件读取 camera_settings 与 video_extract。",
    )

    ns = parser.parse_args(list(argv) if argv is not None else None)
    config_path = Path(str(ns.config))
    config_dir = config_path.parent
    config = load_config(config_path)

    camera_settings_cfg = config.get("camera_settings")
    if camera_settings_cfg is None:
        raise SystemExit(f"错误：配置文件缺少 camera_settings：{config_path}")
    if not isinstance(camera_settings_cfg, dict):
        raise SystemExit(f"错误：camera_settings 必须是 object：{config_path}")

    if str(camera_settings_cfg.get("camera_type", "video_stereo")) != "video_stereo":
        raise SystemExit(
            "错误：mcca.entry.step1_extract_imgs_from_video 仅支持 camera_settings.camera_type=video_stereo"
        )

    camera_names_any = camera_settings_cfg.get("camera_names", ["cam0", "cam1"])
    if not isinstance(camera_names_any, (list, tuple)) or len(camera_names_any) != 2:
        raise SystemExit("错误：camera_settings.camera_names 必须是长度为 2 的列表，例如 ['cam0','cam1']")
    camera_names = [str(x).strip() for x in camera_names_any if str(x).strip()]
    if len(camera_names) != 2 or len(set(camera_names)) != 2:
        raise SystemExit("错误：camera_settings.camera_names 必须是长度为 2 的不重复相机名列表")
    cam0, cam1 = camera_names[0], camera_names[1]

    # 解析 camera_settings（视频输入相关）
    camera_settings = dict(camera_settings_cfg)
    camera_settings["camera_names"] = [cam0, cam1]

    video_mode = str(camera_settings.get("video_mode", "two_files"))
    if video_mode == "two_files":
        video_paths = camera_settings.get("video_paths", None)
        if not isinstance(video_paths, dict):
            raise SystemExit("错误：camera_settings.video_mode=two_files 需要 video_paths（dict）")

        p0 = _resolve_path_maybe(video_paths.get(cam0), config_dir=config_dir)
        p1 = _resolve_path_maybe(video_paths.get(cam1), config_dir=config_dir)
        if not (p0 and p1):
            raise SystemExit("错误：camera_settings.video_paths 必须包含 cam0/cam1 两路视频路径")

        camera_settings["video_paths"] = {cam0: p0, cam1: p1}

    elif video_mode == "single_sbs":
        p = _resolve_path_maybe(camera_settings.get("video_path"), config_dir=config_dir)
        if not p:
            raise SystemExit("错误：camera_settings.video_mode=single_sbs 需要 video_path")
        camera_settings["video_path"] = p

        sbs_order = camera_settings.get("sbs_order", [cam0, cam1])
        if not isinstance(sbs_order, (list, tuple)) or len(sbs_order) != 2:
            raise SystemExit("错误：camera_settings.sbs_order 必须是长度为 2 的列表，例如 ['cam0','cam1']")
        sbs_order = [str(v) for v in sbs_order]
        if set(sbs_order) != {cam0, cam1}:
            raise SystemExit("错误：camera_settings.sbs_order 必须由 camera_names 的两个名字组成")
        camera_settings["sbs_order"] = sbs_order

    else:
        raise SystemExit(f"错误：camera_settings.video_mode 不支持：{video_mode}（仅 two_files/single_sbs）")

    # 解析 video_extract（抽帧相关）
    video_extract_cfg = config.get("video_extract", {})
    if video_extract_cfg is None:
        video_extract_cfg = {}
    if not isinstance(video_extract_cfg, dict):
        raise SystemExit(f"错误：video_extract 必须是 object：{config_path}")

    every_n = int(video_extract_cfg.get("every_n", 10))
    max_pairs = int(video_extract_cfg.get("max_pairs", 300))
    start_frame = int(video_extract_cfg.get("start_frame", 0))
    start_sec = float(video_extract_cfg.get("start_sec", 0.0))
    out_dir = str(video_extract_cfg.get("out_dir", "images/raw"))
    prefix = str(video_extract_cfg.get("prefix", "frame_"))
    overwrite = bool(video_extract_cfg.get("overwrite", False))

    if every_n <= 0:
        raise SystemExit("错误：video_extract.every_n 必须 > 0")
    if max_pairs < 0:
        raise SystemExit("错误：video_extract.max_pairs 必须 >= 0")
    if start_frame < 0:
        raise SystemExit("错误：video_extract.start_frame 必须 >= 0")
    if start_sec < 0:
        raise SystemExit("错误：video_extract.start_sec 必须 >= 0")

    args = ExtractArgs(
        every_n=int(every_n),
        max_pairs=int(max_pairs),
        start_frame=int(start_frame),
        start_sec=float(start_sec),
        out_dir=out_dir,
        prefix=prefix,
        overwrite=overwrite,
    )

    out_base = Path(args.out_dir)
    out_cam0 = out_base / cam0
    out_cam1 = out_base / cam1
    _safe_mkdir(out_cam0)
    _safe_mkdir(out_cam1)

    if args.overwrite:
        _maybe_clear_dir(out_cam0)
        _maybe_clear_dir(out_cam1)

    # 通过统一封装打开视频
    from mcca.adapters.camera_wrapper import create_camera

    camera = create_camera(camera_settings)
    if not camera.open():
        raise SystemExit("错误：无法打开视频源，请检查路径/编码/权限")

    effective_start_frame = _try_seek(camera, start_frame=args.start_frame, start_sec=args.start_sec)

    # 日志输出
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    index_csv = results_dir / "video_extract_index.csv"
    report_json = results_dir / "video_extract_report.json"

    saved = 0
    read_count = 0

    try:
        with index_csv.open("w", encoding="utf-8") as f:
            f.write(f"saved_index,source_frame_index,timestamp_sec,{cam0}_path,{cam1}_path\n")

            while True:
                frame0, frame1, ts = camera.read_stereo()
                if frame0 is None or frame1 is None:
                    break

                src_frame_idx = effective_start_frame + read_count
                read_count += 1

                if (src_frame_idx % args.every_n) != 0:
                    continue

                base = f"{args.prefix}{src_frame_idx:06d}"
                cam0_path = out_cam0 / f"{base}.png"
                cam1_path = out_cam1 / f"{base}.png"

                ok_0 = cv2.imwrite(str(cam0_path), frame0)
                ok_1 = cv2.imwrite(str(cam1_path), frame1)
                if not (ok_0 and ok_1):
                    raise RuntimeError(f"写入失败：{cam0_path} / {cam1_path}")

                f.write(
                    f"{saved},{src_frame_idx},{(ts if ts is not None else '')},{cam0_path.as_posix()},{cam1_path.as_posix()}\n"
                )
                saved += 1

                if args.max_pairs and saved >= args.max_pairs:
                    break

    finally:
        camera.release()

    report = {
        "timestamp": datetime.now().isoformat(),
        "args": asdict(args),
        "config_path": str(config_path.as_posix()),
        "config_video_extract": video_extract_cfg,
        "camera_settings": camera_settings,
        "effective_start_frame": int(effective_start_frame),
        "read_frames": int(read_count),
        "saved_pairs": int(saved),
        "camera_names": [cam0, cam1],
        "out_dirs_by_cam": {cam0: str(out_cam0.as_posix()), cam1: str(out_cam1.as_posix())},
        "index_csv": str(index_csv.as_posix()),
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print("视频抽帧完成")
    print("=" * 60)
    print(f"读取帧数: {read_count}")
    print(f"保存对数: {saved}")
    print(f"输出目录: {out_base}")
    print(f"索引文件: {index_csv}")
    print(f"报告文件: {report_json}")
    print("\n下一步：运行 python -m mcca.entry.step2_filter_images")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
