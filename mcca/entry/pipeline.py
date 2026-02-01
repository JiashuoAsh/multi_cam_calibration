#!/usr/bin/env python3
"""标定流水线入口（config 驱动，自动多相机）。

该模块提供流水线入口（替代已移除的根目录薄包装入口脚本）：
- 将“入口层逻辑”（argparse + 子进程编排 + 日志落盘）放在 entry。
- 真实算法/数据处理仍在 `mcca.core`/`mcca.adapters`。

注意：
- 本模块在一个进程内顺序执行各 step（不再通过子进程）。
- 仍会为每一步生成独立日志文件（stdout/stderr tee 到 results/pipeline_logs）。
- 这样做的动机：提升可测试性与复用性（pipeline 可被 import 并直接调用）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from mcca.adapters.inprocess_runner import run_module_and_tee, run_script_and_tee
from mcca.core.config import load_config
from mcca.core.datasets import get_dataset_cameras, get_step5_cameras, get_step5_dataset
from mcca.entry.pipeline_plan import (
    PipelineFlags,
    PipelinePerfOptions,
    build_report_payload,
    build_step_calls,
    get_camera_to_base_mode,
)


@dataclass
class StepResult:
    name: str
    cmd: List[str]
    log_path: Path
    returncode: int


def _repo_root() -> Path:
    """返回仓库根目录（以本文件的上上级目录为准）。"""

    # entry/pipeline.py -> entry -> mcca -> repo_root
    return Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mcca.entry.pipeline",
        description="AprilTag 相机标定流水线（config 驱动，自动多相机）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/apriltag_config.json",
        help="配置文件路径（默认 config/apriltag_config.json）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行 Step1~Step4（包含视频抽帧）。若 step5_dataset.enabled=true 且未跳过，将自动包含 Step5。",
    )
    parser.add_argument("--skip_step2", action="store_true", help="跳过 Step2（筛图）。")
    parser.add_argument("--skip_step3", action="store_true", help="跳过 Step3（内参）。")
    parser.add_argument("--skip_step4", action="store_true", help="跳过 Step4（外参）。")
    parser.add_argument("--skip_step5", action="store_true", help="跳过 Step5（相机->底盘外参）。")
    parser.add_argument(
        "--run_step5",
        action="store_true",
        help="强制运行 Step5（即使 step5_dataset 未启用；会用默认 images/step5/<cam>/ 扫描）。",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="results/pipeline_logs",
        help="每一步日志输出目录（默认 results/pipeline_logs）。",
    )

    # 统一性能参数（可选）：只要在 pipeline 中指定一次，就会透传给支持的 step。
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="多进程 worker 数（0=使用各 step 默认；>0 透传到各 step）。",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=0,
        help="并行时的预提交任务数（0=使用各 step 默认；>0 透传到各 step）。",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="检测缓存目录（空=使用各 step 默认）。",
    )
    parser.add_argument("--no_cache", action="store_true", help="禁用检测缓存（会显著变慢）。")
    parser.add_argument("--force_redetect", action="store_true", help="忽略缓存强制重新检测（用于调参/排查）。")
    parser.add_argument("--prefilter", action="store_true", help="启用廉价预筛选（减少 detector 调用）。")
    parser.add_argument(
        "--scan_strategy",
        type=str,
        default="",
        choices=["", "sequential", "random", "uniform"],
        help="扫描策略（空=使用各 step 默认）。",
    )
    parser.add_argument("--scan_seed", type=int, default=0, help="random 扫描的随机种子（用于可复现）。")
    parser.add_argument(
        "--max_detect_seconds",
        type=float,
        default=0.0,
        help="扫描+检测阶段的总耗时上限（秒，0=使用各 step 默认）。",
    )
    parser.add_argument(
        "--scan_target_valid",
        type=int,
        default=0,
        help=(
            "统一‘有效目标’上限（0=不覆盖各 step 默认）。"
            "会被映射为：step2/3=max_valid_images, step4_multi=target_valid_frames, step5b=max_valid_poses。"
        ),
    )
    parser.add_argument(
        "--scan_max_total",
        type=int,
        default=0,
        help=(
            "统一‘候选总数’上限（0=不覆盖各 step 默认）。"
            "会被映射为：step2/3=max_total_images, step4_multi=max_total_frames, step5b=max_total_images。"
        ),
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = _repo_root()

    config_path = (
        (root / str(args.config)).resolve()
        if not Path(str(args.config)).is_absolute()
        else Path(str(args.config))
    )

    if not config_path.exists():
        print(f"错误: config 不存在：{config_path}")
        return 1

    config = load_config(str(config_path))

    step5_ds = get_step5_dataset(config)
    use_step5_dataset = bool(step5_ds.get("enabled", False))

    camera_to_base_mode = get_camera_to_base_mode(config)

    cameras: List[str] = get_dataset_cameras(config, allow_scan=True)
    if len(cameras) == 0:
        print("错误: 未找到任何相机。")
        print("请在 config.image_dataset.cameras 中配置相机名与 raw_glob，或把图片放到 images/raw/<cam>/ 下。")
        return 1

    step5_cameras: List[str] = []
    if use_step5_dataset:
        step5_cameras = get_step5_cameras(config, allow_scan=True)

    print("=" * 60)
    print("相机标定流水线开始执行")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"config: {config_path}")
    print(f"image_dataset: enabled (cameras={cameras})")

    if use_step5_dataset:
        print(f"step5_dataset: enabled (cameras={step5_cameras})")
    else:
        print("step5_dataset: disabled")

    print(f"camera_to_base_calibration.mode: {camera_to_base_mode}")
    print("=" * 60)

    log_dir = (
        (root / str(args.log_dir)).resolve()
        if not Path(str(args.log_dir)).is_absolute()
        else Path(str(args.log_dir))
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    perf = PipelinePerfOptions(
        workers=int(args.workers),
        prefetch=int(args.prefetch),
        cache_dir=str(args.cache_dir),
        no_cache=bool(args.no_cache),
        force_redetect=bool(args.force_redetect),
        prefilter=bool(args.prefilter),
        scan_strategy=str(args.scan_strategy),
        scan_seed=int(args.scan_seed),
        max_detect_seconds=float(args.max_detect_seconds),
        scan_target_valid=int(args.scan_target_valid),
        scan_max_total=int(args.scan_max_total),
    )
    flags = PipelineFlags(
        all=bool(args.all),
        skip_step2=bool(args.skip_step2),
        skip_step3=bool(args.skip_step3),
        skip_step4=bool(args.skip_step4),
        skip_step5=bool(args.skip_step5),
        run_step5=bool(args.run_step5),
    )

    calls = build_step_calls(
        flags=flags,
        config_path=str(config_path),
        perf=perf,
        use_step5_dataset=bool(use_step5_dataset),
        camera_to_base_mode=str(camera_to_base_mode),
    )

    step_titles = {
        "step1_extract": "[步骤 1] 从视频抽帧...",
        "step2_filter": "[步骤 2] 筛选包含标定板的图像...",
        "step3_intrinsic": "[步骤 3] 相机内参标定...",
        "step4_extrinsic": "[步骤 4] 外参标定...",
        "step5_camera_to_base": "[步骤 5] 相机->底盘外参标定...",
    }

    results: List[StepResult] = []
    for call in calls:
        print("\n" + step_titles.get(call.name, f"[步骤] {call.name}..."))

        if call.kind == "module":
            cmd = [sys.executable, "-m", call.target] + list(call.argv)
        else:
            cmd = [sys.executable, str(root / call.target)] + list(call.argv)

        log_path = log_dir / call.log_filename
        if call.kind == "module":
            rc = run_module_and_tee(call.target, argv=list(call.argv), log_path=log_path, cwd=root)
        else:
            rc = run_script_and_tee(root / call.target, argv=list(call.argv), log_path=log_path, cwd=root)
        results.append(StepResult(call.name, list(cmd), log_path, int(rc)))
        if rc != 0:
            print(f"\n[FAIL] {call.name} 失败")
            return int(rc)

    report = build_report_payload(
        timestamp_iso=datetime.now().isoformat(),
        config_path_posix=str(config_path.as_posix()),
        cameras=cameras,
        step5_cameras=step5_cameras,
        use_step5_dataset=bool(use_step5_dataset),
        camera_to_base_mode=str(camera_to_base_mode),
        steps=[
            {
                "name": r.name,
                "cmd": r.cmd,
                "log": str(r.log_path.as_posix()),
                "returncode": int(r.returncode),
            }
            for r in results
        ],
    )

    out_report = root / "results/pipeline_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("标定流水线全部完成！")
    print(f"时间: {datetime.now().isoformat()}")
    print("=" * 60)
    print("\n日志目录:")
    print(f"  - {log_dir}")
    print("\n汇总报告:")
    print("  - results/pipeline_report.json")

    return 0


def cli_main() -> None:
    """console_script 入口：按照惯例抛出 SystemExit。"""

    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
