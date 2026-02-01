from __future__ import annotations

"""流水线计划（entry 层）。

为什么在 entry：
- 该模块负责把“pipeline CLI 的开关/性能参数”翻译成“要执行哪些 step（模块名 + argv）”。
- step 的可执行目标是 `mcca.entry.*`，因此这里天然属于入口层（entry）。

注意：
- 这里仍然只返回纯数据结构（dataclass），不做 subprocess/路径解析。
- 真实算法/求解逻辑仍在 `mcca.core`；入口负责组装与落盘。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping


@dataclass(frozen=True)
class PipelinePerfOptions:
    """pipeline 统一性能参数（纯数据）。

    说明：
    - 该结构用于把 argparse 的 Namespace 变成显式、可测试的数据。
    - 参数名保持与 pipeline CLI 一致。
    """

    workers: int = 0
    prefetch: int = 0
    cache_dir: str = ""
    no_cache: bool = False
    force_redetect: bool = False
    prefilter: bool = False
    scan_strategy: str = ""  # ""|sequential|random|uniform
    scan_seed: int = 0
    max_detect_seconds: float = 0.0
    scan_target_valid: int = 0
    scan_max_total: int = 0


@dataclass(frozen=True)
class PipelineFlags:
    """pipeline 的运行开关（纯数据）。"""

    all: bool
    skip_step2: bool
    skip_step3: bool
    skip_step4: bool
    skip_step5: bool
    run_step5: bool


StepKind = Literal["module", "script"]


@dataclass(frozen=True)
class StepCall:
    """单个 step 的“可执行描述”（不包含 subprocess/路径解析）。"""

    name: str
    kind: StepKind
    target: str
    argv: List[str]
    log_filename: str


def get_camera_to_base_mode(config: Mapping[str, Any]) -> str:
    """读取 Step5 的求解模式。"""

    calib_cfg = (config or {}).get("camera_to_base_calibration", {})
    if not isinstance(calib_cfg, dict):
        calib_cfg = {}

    mode = str(calib_cfg.get("mode", "apriltag_pnp")).strip().lower() or "apriltag_pnp"
    if mode not in {"apriltag_pnp", "world_anchor"}:
        mode = "apriltag_pnp"
    return mode


def should_run_step5(
    *,
    flags: PipelineFlags,
    use_step5_dataset: bool,
    camera_to_base_mode: str,
) -> bool:
    """决定 pipeline 是否应运行 Step5。"""

    if bool(flags.skip_step5):
        return False
    if bool(flags.run_step5):
        return True

    # 默认策略：只有在“有 Step5 输入数据”时，才把 Step5 纳入 --all。
    # - apriltag_pnp：需要 step5_dataset
    # - world_anchor：不需要 step5_dataset，但需要 camera_to_base_mode=world_anchor
    if bool(flags.all) and (bool(use_step5_dataset) or camera_to_base_mode == "world_anchor"):
        return True

    return False


def perf_args_for_step(*, step_key: str, perf: PipelinePerfOptions) -> List[str]:
    """将 pipeline 的统一性能参数映射到各 step 的具体参数名。"""

    perf_supported = {
        "step2_filter_images",
        "step3_intrinsic_apriltag",
        "step4_multi_extrinsic",
        "step5b_camera_to_base",
    }
    if step_key not in perf_supported:
        return []

    out: List[str] = []

    # 通用：并行/缓存/预筛选/扫描策略
    if int(perf.workers) > 0:
        out += ["--workers", str(int(perf.workers))]
    if int(perf.prefetch) > 0:
        out += ["--prefetch", str(int(perf.prefetch))]

    cache_dir = str(perf.cache_dir).strip()
    if cache_dir:
        out += ["--cache_dir", cache_dir]
    if bool(perf.no_cache):
        out += ["--no_cache"]
    if bool(perf.force_redetect):
        out += ["--force_redetect"]
    if bool(perf.prefilter):
        out += ["--prefilter"]

    scan_strategy = str(perf.scan_strategy).strip()
    if scan_strategy:
        out += ["--scan_strategy", scan_strategy]
        out += ["--scan_seed", str(int(perf.scan_seed))]

    if float(perf.max_detect_seconds) > 0:
        out += ["--max_detect_seconds", str(float(perf.max_detect_seconds))]

    # 通用：扫描预算（按 step 映射）
    target_valid = int(perf.scan_target_valid)
    max_total = int(perf.scan_max_total)

    if step_key in {"step2_filter_images", "step3_intrinsic_apriltag"}:
        if target_valid > 0:
            out += ["--max_valid_images", str(target_valid)]
        if max_total > 0:
            out += ["--max_total_images", str(max_total)]
    elif step_key == "step4_multi_extrinsic":
        if target_valid > 0:
            out += ["--target_valid_frames", str(target_valid)]
        if max_total > 0:
            out += ["--max_total_frames", str(max_total)]
    elif step_key == "step5b_camera_to_base":
        if target_valid > 0:
            out += ["--max_valid_poses", str(target_valid)]
        if max_total > 0:
            out += ["--max_total_images", str(max_total)]

    return out


def build_step_calls(
    *,
    flags: PipelineFlags,
    config_path: str,
    perf: PipelinePerfOptions,
    use_step5_dataset: bool,
    camera_to_base_mode: str,
) -> List[StepCall]:
    """构造 pipeline 需要执行的 step 列表。"""

    steps: List[StepCall] = []

    # Step1（可选：仅在 --all 时运行）
    if bool(flags.all):
        steps.append(
            StepCall(
                name="step1_extract",
                kind="module",
                target="mcca.entry.step1_extract_imgs_from_video",
                argv=["--config", str(config_path)],
                log_filename="step1_extract.txt",
            )
        )

    # Step2
    if not bool(flags.skip_step2):
        key = "step2_filter_images"
        steps.append(
            StepCall(
                name="step2_filter",
                kind="module",
                target="mcca.entry.step2_filter_images",
                argv=["--config", str(config_path)] + perf_args_for_step(step_key=key, perf=perf),
                log_filename="step2_filter.txt",
            )
        )

    # Step3
    if not bool(flags.skip_step3):
        key = "step3_intrinsic_apriltag"
        steps.append(
            StepCall(
                name="step3_intrinsic",
                kind="module",
                target="mcca.entry.step3_intrinsic_apriltag",
                argv=["--config", str(config_path)] + perf_args_for_step(step_key=key, perf=perf),
                log_filename="step3_intrinsic.txt",
            )
        )

    # Step4
    if not bool(flags.skip_step4):
        key = "step4_multi_extrinsic"
        steps.append(
            StepCall(
                name="step4_extrinsic",
                kind="module",
                target="mcca.entry.step4_multi_extrinsic",
                argv=["--config", str(config_path)] + perf_args_for_step(step_key=key, perf=perf),
                log_filename="step4_extrinsic.txt",
            )
        )

    # Step5（可选：相机->底盘）
    if should_run_step5(
        flags=flags,
        use_step5_dataset=bool(use_step5_dataset),
        camera_to_base_mode=str(camera_to_base_mode),
    ):
        if camera_to_base_mode == "world_anchor":
            steps.append(
                StepCall(
                    name="step5_camera_to_base",
                    kind="module",
                    target="mcca.entry.step5c_world_anchor",
                    argv=["--config", str(config_path)],
                    log_filename="step5_camera_to_base.txt",
                )
            )
        else:
            key = "step5b_camera_to_base"
            steps.append(
                StepCall(
                    name="step5_camera_to_base",
                    kind="module",
                    target="mcca.entry.step5b_camera_to_base",
                    argv=["--config", str(config_path)] + perf_args_for_step(step_key=key, perf=perf),
                    log_filename="step5_camera_to_base.txt",
                )
            )

    return steps


def build_report_payload(
    *,
    timestamp_iso: str,
    config_path_posix: str,
    cameras: List[str],
    step5_cameras: List[str],
    use_step5_dataset: bool,
    camera_to_base_mode: str,
    steps: List[Dict[str, object]],
) -> Dict[str, object]:
    """构造 pipeline_report.json 的 payload（纯数据结构）。"""

    return {
        "timestamp": str(timestamp_iso),
        "config": str(config_path_posix),
        "image_dataset_enabled": True,
        "step5_dataset_enabled": bool(use_step5_dataset),
        "camera_to_base_mode": str(camera_to_base_mode),
        "cameras": list(cameras),
        "step5_cameras": list(step5_cameras),
        "steps": list(steps),
    }
