#!/usr/bin/env python3
"""标定流水线（config 驱动，自动多相机）

目标：把 Step2/Step3/Step4 串成一键流程；可选包含 Step1（视频抽帧）与 Step5（相机->底盘）。

设计原则：
- 单一入口：只要配置好 config/apriltag_config.json，即可自动识别相机数量并完成筛图/内参/外参。
- 尽量不侵入：各 step 仍可单独运行；该脚本只是按顺序调用并做最小的结果检查与日志落盘。
- 统一数据集格式：相机命名统一使用 cam0/cam1/cam2...，由 config.image_dataset.cameras 指定。

用法示例：
- 跑 Step2~Step4（默认）：
    python run_calibration_pipeline.py --config config/apriltag_config.json

- 跑 Step1~Step4：
    python run_calibration_pipeline.py --all --config config/apriltag_config.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils import (
    get_image_dataset,
    get_dataset_cameras,
    get_step5_dataset,
    get_step5_cameras,
    load_config,
)


# region 数据结构


@dataclass
class StepResult:
    name: str
    cmd: List[str]
    log_path: Path
    returncode: int


# endregion


# region 子进程执行与日志


def _repo_root() -> Path:
    """返回仓库根目录（以当前脚本所在目录为准）。"""
    return Path(__file__).resolve().parent


def _run_and_tee(cmd: List[str], *, log_path: Path, cwd: Path) -> int:
    """运行子进程，将 stdout/stderr 同时写入 log 与控制台。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # 统一输出编码，避免 Windows/控制台编码导致的异常
    env.setdefault("PYTHONIOENCODING", "utf-8")

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.write(f"# time: {datetime.now().isoformat()}\n\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert p.stdout is not None
        for line in p.stdout:
            # 控制台
            print(line, end="")
            # 文件
            f.write(line)
        return int(p.wait())


    # endregion


    # region 命令构造


def _script_path(name: str) -> Path:
    """将脚本名解析为仓库内的绝对路径。"""
    return _repo_root() / name


def _step_cmd(step_script: str, *, config_path: Path, extra_args: Optional[List[str]] = None) -> List[str]:
    """构造某一步的命令行。

    约定：
        只有部分 step 脚本支持 --config，本函数会根据白名单自动决定是否传入。
    """
    args = extra_args or []
    sp = _script_path(step_script)

    # 约定：支持 --config 的脚本我们统一传入；不支持的就不传。
    supports_config = step_script in {
        "step1_extract_imgs_from_video.py",
        "step2_filter_images.py",
        "step3_intrinsic_apriltag.py",
        "step4_multi_extrinsic_pose_graph.py",
        "step5b_camera_to_base.py",
        "step5c_camera_to_base_from_world.py",
    }

    cmd = [sys.executable, str(sp)]
    if supports_config:
        cmd += ["--config", str(config_path)]
    cmd += args
    return cmd


def _perf_args_for_step(*, step_script: str, args: argparse.Namespace) -> List[str]:
    """将 pipeline 的统一性能参数映射到各 step 的具体参数名。

    说明：
    - 不同 step 对“候选总数/有效目标”的参数名不同（images/pairs/frames/poses）。
    - 为避免误传未知参数，这里按脚本名白名单映射。
    """

    perf_supported = {
        "step2_filter_images.py",
        "step3_intrinsic_apriltag.py",
        "step4_multi_extrinsic_pose_graph.py",
        "step5b_camera_to_base.py",
    }
    if step_script not in perf_supported:
        return []

    out: List[str] = []

    # 通用：并行/缓存/预筛选/扫描策略
    if int(getattr(args, "workers", 0)) > 0:
        out += ["--workers", str(int(args.workers))]
    if int(getattr(args, "prefetch", 0)) > 0:
        out += ["--prefetch", str(int(args.prefetch))]

    cache_dir = str(getattr(args, "cache_dir", "")).strip()
    if cache_dir:
        out += ["--cache_dir", cache_dir]
    if bool(getattr(args, "no_cache", False)):
        out += ["--no_cache"]
    if bool(getattr(args, "force_redetect", False)):
        out += ["--force_redetect"]
    if bool(getattr(args, "prefilter", False)):
        out += ["--prefilter"]

    scan_strategy = str(getattr(args, "scan_strategy", "")).strip()
    if scan_strategy:
        out += ["--scan_strategy", scan_strategy]
        out += ["--scan_seed", str(int(getattr(args, "scan_seed", 0)))]

    if float(getattr(args, "max_detect_seconds", 0.0)) > 0:
        out += ["--max_detect_seconds", str(float(args.max_detect_seconds))]

    # 通用：扫描预算（按 step 映射）
    target_valid = int(getattr(args, "scan_target_valid", 0))
    max_total = int(getattr(args, "scan_max_total", 0))

    if step_script in {"step2_filter_images.py", "step3_intrinsic_apriltag.py"}:
        if target_valid > 0:
            out += ["--max_valid_images", str(target_valid)]
        if max_total > 0:
            out += ["--max_total_images", str(max_total)]
    elif step_script == "step4_multi_extrinsic_pose_graph.py":
        if target_valid > 0:
            out += ["--target_valid_frames", str(target_valid)]
        if max_total > 0:
            out += ["--max_total_frames", str(max_total)]
    elif step_script == "step5b_camera_to_base.py":
        if target_valid > 0:
            out += ["--max_valid_poses", str(target_valid)]
        if max_total > 0:
            out += ["--max_total_images", str(max_total)]

    return out


# endregion


# region 主流程


def main() -> int:
    parser = argparse.ArgumentParser(description="AprilTag 相机标定流水线（config 驱动，自动多相机）")
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
    parser.add_argument(
        "--skip_step2",
        action="store_true",
        help="跳过 Step2（筛图）。",
    )
    parser.add_argument(
        "--skip_step3",
        action="store_true",
        help="跳过 Step3（内参）。",
    )
    parser.add_argument(
        "--skip_step4",
        action="store_true",
        help="跳过 Step4（外参）。",
    )
    parser.add_argument(
        "--skip_step5",
        action="store_true",
        help="跳过 Step5（相机->底盘外参）。",
    )
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
    parser.add_argument(
        "--scan_seed",
        type=int,
        default=0,
        help="random 扫描的随机种子（用于可复现）。",
    )
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
            "会被映射为：step2/3=max_valid_images, step4_stereo=max_valid_pairs, step4_multi=target_valid_frames, step5b=max_valid_poses。"
        ),
    )
    parser.add_argument(
        "--scan_max_total",
        type=int,
        default=0,
        help=(
            "统一‘候选总数’上限（0=不覆盖各 step 默认）。"
            "会被映射为：step2/3=max_total_images, step4_stereo=max_total_pairs, step4_multi=max_total_frames, step5b=max_total_images。"
        ),
    )
    args = parser.parse_args()

    root = _repo_root()
    config_path = (root / str(args.config)).resolve() if not Path(str(args.config)).is_absolute() else Path(str(args.config))

    if not config_path.exists():
        print(f"错误: config 不存在：{config_path}")
        return 1

    config = load_config(str(config_path))

    step5_ds = get_step5_dataset(config)
    use_step5_dataset = bool(step5_ds.get("enabled", False))

    camera_to_base_mode = _get_camera_to_base_mode(config)

    # 仅用于提示：告诉用户 pipeline 识别到了哪些相机
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

    log_dir = (root / str(args.log_dir)).resolve() if not Path(str(args.log_dir)).is_absolute() else Path(str(args.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    results: List[StepResult] = []

    # Step1（可选）
    if bool(args.all):
        print("\n[步骤 1] 从视频抽帧...")
        cmd = _step_cmd("step1_extract_imgs_from_video.py", config_path=config_path)
        rc = _run_and_tee(cmd, log_path=log_dir / "step1_extract.txt", cwd=root)
        results.append(StepResult("step1_extract", cmd, log_dir / "step1_extract.txt", rc))
        if rc != 0:
            print("\n[FAIL] 步骤1失败")
            return rc

    # Step2
    if not bool(args.skip_step2):
        print("\n[步骤 2] 筛选包含标定板的图像...")
        cmd = _step_cmd(
            "step2_filter_images.py",
            config_path=config_path,
            extra_args=_perf_args_for_step(step_script="step2_filter_images.py", args=args),
        )
        rc = _run_and_tee(cmd, log_path=log_dir / "step2_filter.txt", cwd=root)
        results.append(StepResult("step2_filter", cmd, log_dir / "step2_filter.txt", rc))
        if rc != 0:
            print("\n[FAIL] 步骤2失败")
            return rc

    # Step3
    if not bool(args.skip_step3):
        print("\n[步骤 3] 相机内参标定...")
        cmd = _step_cmd(
            "step3_intrinsic_apriltag.py",
            config_path=config_path,
            extra_args=_perf_args_for_step(step_script="step3_intrinsic_apriltag.py", args=args),
        )
        rc = _run_and_tee(cmd, log_path=log_dir / "step3_intrinsic.txt", cwd=root)
        results.append(StepResult("step3_intrinsic", cmd, log_dir / "step3_intrinsic.txt", rc))
        if rc != 0:
            print("\n[FAIL] 步骤3失败")
            return rc

    # Step4
    if not bool(args.skip_step4):
        print("\n[步骤 4] 外参标定...")

        step4_script = "step4_multi_extrinsic_pose_graph.py"

        cmd = _step_cmd(
            step4_script,
            config_path=config_path,
            extra_args=_perf_args_for_step(step_script=step4_script, args=args),
        )
        rc = _run_and_tee(cmd, log_path=log_dir / "step4_extrinsic.txt", cwd=root)
        results.append(StepResult("step4_extrinsic", cmd, log_dir / "step4_extrinsic.txt", rc))
        if rc != 0:
            print("\n[FAIL] 步骤4失败")
            return rc

    # Step5（可选：相机->底盘）
    if _should_run_step5(
        args=args,
        use_step5_dataset=use_step5_dataset,
        camera_to_base_mode=camera_to_base_mode,
    ):
        print("\n[步骤 5] 相机->底盘外参标定...")
        step5_script = (
            "step5c_camera_to_base_from_world.py"
            if camera_to_base_mode == "world_anchor"
            else "step5b_camera_to_base.py"
        )
        extra_args = None
        if step5_script == "step5b_camera_to_base.py":
            extra_args = _perf_args_for_step(step_script=step5_script, args=args)
        cmd = _step_cmd(step5_script, config_path=config_path, extra_args=extra_args)
        rc = _run_and_tee(cmd, log_path=log_dir / "step5_camera_to_base.txt", cwd=root)
        results.append(StepResult("step5_camera_to_base", cmd, log_dir / "step5_camera_to_base.txt", rc))
        if rc != 0:
            print("\n[FAIL] 步骤5失败")
            return rc

    # 汇总
    report: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path.as_posix()),
        "image_dataset_enabled": True,
        "step5_dataset_enabled": bool(use_step5_dataset),
        "camera_to_base_mode": str(camera_to_base_mode),
        "cameras": cameras,
        "step5_cameras": step5_cameras,
        "steps": [
            {
                "name": r.name,
                "cmd": r.cmd,
                "log": str(r.log_path.as_posix()),
                "returncode": int(r.returncode),
            }
            for r in results
        ],
    }

    out_report = root / "results/pipeline_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        __import__("json").dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
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


# endregion


# region Step5 运行策略
def _should_run_step5(
    *,
    args: argparse.Namespace,
    use_step5_dataset: bool,
    camera_to_base_mode: str,
) -> bool:
    if bool(args.skip_step5):
        return False
    if bool(args.run_step5):
        return True
    # 默认策略：只有在“有 Step5 输入数据”时，才把 Step5 纳入 --all。
    # - apriltag_pnp：需要 step5_dataset
    # - world_anchor：不需要 step5_dataset，但需要 camera_to_base_mode=world_anchor
    if bool(args.all) and (bool(use_step5_dataset) or camera_to_base_mode == "world_anchor"):
        return True
    return False


def _get_camera_to_base_mode(config: dict) -> str:
    """读取 Step5 的求解模式。

    约定：
      config["camera_to_base_calibration"]["mode"] in {"apriltag_pnp", "world_anchor"}
    """
    calib_cfg = (config or {}).get("camera_to_base_calibration", {})
    if not isinstance(calib_cfg, dict):
        calib_cfg = {}

    mode = str(calib_cfg.get("mode", "apriltag_pnp")).strip().lower() or "apriltag_pnp"
    if mode not in {"apriltag_pnp", "world_anchor"}:
        mode = "apriltag_pnp"
    return mode


# endregion



if __name__ == "__main__":
    raise SystemExit(main())
