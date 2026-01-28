#!/usr/bin/env python3
"""标定流水线（config 驱动，自动多相机）

目标：把 Step2/Step3/Step4 串成一键流程；可选包含 Step1（视频抽帧）与 Step5（相机->底盘）。

设计原则：
- 单一入口：只要配置好 config/apriltag_config.json，即可自动识别相机数量并完成筛图/内参/外参。
- 尽量不侵入：各 step 仍可单独运行；该脚本只是按顺序调用并做最小的结果检查与日志落盘。
- 兼容旧流程：
  - 若 config.image_dataset.enabled=false：默认走旧双目 left/right（Step4 走 step4_stereo_extrinsic.py）
  - 若 config.image_dataset.enabled=true：默认走多相机（Step4 走 step4_multi_extrinsic_pose_graph.py）

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
        "--force_pose_graph",
        action="store_true",
        help="强制使用位姿图 Step4（step4_multi_extrinsic_pose_graph.py），即使 image_dataset 未启用。",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="results/pipeline_logs",
        help="每一步日志输出目录（默认 results/pipeline_logs）。",
    )
    args = parser.parse_args()

    root = _repo_root()
    config_path = (root / str(args.config)).resolve() if not Path(str(args.config)).is_absolute() else Path(str(args.config))

    if not config_path.exists():
        print(f"错误: config 不存在：{config_path}")
        return 1

    config = load_config(str(config_path))
    ds = get_image_dataset(config)
    use_dataset = bool(ds.get("enabled", False))

    step5_ds = get_step5_dataset(config)
    use_step5_dataset = bool(step5_ds.get("enabled", False))

    camera_to_base_mode = _get_camera_to_base_mode(config)

    # 仅用于提示：告诉用户 pipeline 识别到了哪些相机
    cameras: List[str] = []
    if use_dataset:
        cameras = get_dataset_cameras(config, allow_scan=True, fallback_stereo=True)

    step5_cameras: List[str] = []
    if use_step5_dataset:
        step5_cameras = get_step5_cameras(config, allow_scan=True)

    print("=" * 60)
    print("相机标定流水线开始执行")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"config: {config_path}")
    if use_dataset:
        print(f"image_dataset: enabled (cameras={cameras})")
    else:
        print("image_dataset: disabled (legacy stereo left/right)")

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
        cmd = _step_cmd("step2_filter_images.py", config_path=config_path)
        rc = _run_and_tee(cmd, log_path=log_dir / "step2_filter.txt", cwd=root)
        results.append(StepResult("step2_filter", cmd, log_dir / "step2_filter.txt", rc))
        if rc != 0:
            print("\n[FAIL] 步骤2失败")
            return rc

    # Step3
    if not bool(args.skip_step3):
        print("\n[步骤 3] 相机内参标定...")
        cmd = _step_cmd("step3_intrinsic_apriltag.py", config_path=config_path)
        rc = _run_and_tee(cmd, log_path=log_dir / "step3_intrinsic.txt", cwd=root)
        results.append(StepResult("step3_intrinsic", cmd, log_dir / "step3_intrinsic.txt", rc))
        if rc != 0:
            print("\n[FAIL] 步骤3失败")
            return rc

    # Step4
    if not bool(args.skip_step4):
        print("\n[步骤 4] 外参标定...")

        if bool(args.force_pose_graph) or use_dataset:
            step4_script = "step4_multi_extrinsic_pose_graph.py"
        else:
            # legacy stereo
            step4_script = "step4_stereo_extrinsic.py"

        cmd = _step_cmd(step4_script, config_path=config_path)
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
        cmd = _step_cmd(step5_script, config_path=config_path)
        rc = _run_and_tee(cmd, log_path=log_dir / "step5_camera_to_base.txt", cwd=root)
        results.append(StepResult("step5_camera_to_base", cmd, log_dir / "step5_camera_to_base.txt", rc))
        if rc != 0:
            print("\n[FAIL] 步骤5失败")
            return rc

    # 汇总
    report: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path.as_posix()),
        "image_dataset_enabled": bool(use_dataset),
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
