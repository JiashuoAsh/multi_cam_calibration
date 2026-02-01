from __future__ import annotations

from mcca.entry.pipeline_plan import (
    PipelineFlags,
    PipelinePerfOptions,
    build_step_calls,
    get_camera_to_base_mode,
    perf_args_for_step,
    should_run_step5,
)


def test_get_camera_to_base_mode_default_and_validation() -> None:
    assert get_camera_to_base_mode({}) == "apriltag_pnp"
    assert get_camera_to_base_mode({"camera_to_base_calibration": {"mode": "world_anchor"}}) == "world_anchor"
    assert get_camera_to_base_mode({"camera_to_base_calibration": {"mode": "APRILTAG_PNP"}}) == "apriltag_pnp"
    # 非法值应回退到默认
    assert get_camera_to_base_mode({"camera_to_base_calibration": {"mode": "???"}}) == "apriltag_pnp"


def test_should_run_step5_logic() -> None:
    base_flags = PipelineFlags(
        all=False,
        skip_step2=False,
        skip_step3=False,
        skip_step4=False,
        skip_step5=False,
        run_step5=False,
    )

    assert (
        should_run_step5(
            flags=base_flags,
            use_step5_dataset=False,
            camera_to_base_mode="apriltag_pnp",
        )
        is False
    )

    assert (
        should_run_step5(
            flags=PipelineFlags(**{**base_flags.__dict__, "run_step5": True}),
            use_step5_dataset=False,
            camera_to_base_mode="apriltag_pnp",
        )
        is True
    )

    assert (
        should_run_step5(
            flags=PipelineFlags(**{**base_flags.__dict__, "skip_step5": True, "run_step5": True}),
            use_step5_dataset=True,
            camera_to_base_mode="world_anchor",
        )
        is False
    )

    assert (
        should_run_step5(
            flags=PipelineFlags(**{**base_flags.__dict__, "all": True}),
            use_step5_dataset=True,
            camera_to_base_mode="apriltag_pnp",
        )
        is True
    )

    # world_anchor 模式即使 step5_dataset 未启用，也会在 --all 下被纳入
    assert (
        should_run_step5(
            flags=PipelineFlags(**{**base_flags.__dict__, "all": True}),
            use_step5_dataset=False,
            camera_to_base_mode="world_anchor",
        )
        is True
    )


def test_perf_args_for_step_mapping() -> None:
    perf = PipelinePerfOptions(
        workers=4,
        prefetch=8,
        cache_dir="cache/apriltag_detection",
        no_cache=True,
        force_redetect=True,
        prefilter=True,
        scan_strategy="uniform",
        scan_seed=123,
        max_detect_seconds=3.5,
        scan_target_valid=11,
        scan_max_total=22,
    )

    assert perf_args_for_step(step_key="not_supported", perf=perf) == []

    a2 = perf_args_for_step(step_key="step2_filter_images", perf=perf)
    assert "--workers" in a2 and "4" in a2
    assert "--prefetch" in a2 and "8" in a2
    assert a2[a2.index("--cache_dir") + 1] == "cache/apriltag_detection"
    assert "--no_cache" in a2
    assert "--force_redetect" in a2
    assert "--prefilter" in a2
    assert a2[a2.index("--scan_strategy") + 1] == "uniform"
    assert a2[a2.index("--scan_seed") + 1] == "123"
    assert a2[a2.index("--max_detect_seconds") + 1] == "3.5"
    assert a2[a2.index("--max_valid_images") + 1] == "11"
    assert a2[a2.index("--max_total_images") + 1] == "22"

    a4 = perf_args_for_step(step_key="step4_multi_extrinsic", perf=perf)
    assert a4[a4.index("--target_valid_frames") + 1] == "11"
    assert a4[a4.index("--max_total_frames") + 1] == "22"

    a5 = perf_args_for_step(step_key="step5b_camera_to_base", perf=perf)
    assert a5[a5.index("--max_valid_poses") + 1] == "11"
    assert a5[a5.index("--max_total_images") + 1] == "22"


def test_build_step_calls_includes_expected_steps() -> None:
    flags = PipelineFlags(
        all=False,
        skip_step2=False,
        skip_step3=False,
        skip_step4=False,
        skip_step5=False,
        run_step5=False,
    )
    perf = PipelinePerfOptions()

    calls = build_step_calls(
        flags=flags,
        config_path="config/apriltag_config.json",
        perf=perf,
        use_step5_dataset=False,
        camera_to_base_mode="apriltag_pnp",
    )

    # 默认：Step2~Step4
    assert [c.name for c in calls] == ["step2_filter", "step3_intrinsic", "step4_extrinsic"]

    # --all 才包含 Step1
    calls_all = build_step_calls(
        flags=PipelineFlags(**{**flags.__dict__, "all": True}),
        config_path="config/apriltag_config.json",
        perf=perf,
        use_step5_dataset=False,
        camera_to_base_mode="apriltag_pnp",
    )
    assert calls_all[0].name == "step1_extract"

    # world_anchor 模式下，Step5 使用 step5c_world_anchor 模块
    calls_wa = build_step_calls(
        flags=PipelineFlags(**{**flags.__dict__, "all": True}),
        config_path="config/apriltag_config.json",
        perf=perf,
        use_step5_dataset=False,
        camera_to_base_mode="world_anchor",
    )
    assert calls_wa[-1].name == "step5_camera_to_base"
    assert calls_wa[-1].kind == "module"
    assert calls_wa[-1].target == "mcca.entry.step5c_world_anchor"
