#!/usr/bin/env python3
"""Step4（多相机外参）：入口层 CLI。

职责边界：
- entry：参数解析、组装依赖、调用 adapters 扫描建图、调用 core 求解、落盘 JSON、控制台诊断输出。
- adapters：扫描图片/缓存/并行/检测与 PnP（IO/性能相关）。
- core：位姿图优化/BA 与误差统计（纯逻辑/数学）。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mcca.adapters.apriltag_perf.cache import CacheConfig
from mcca.adapters.apriltag_perf.prefilter import PrefilterConfig
from mcca.adapters.apriltag_perf.scan import ScanLimits, ScanOrder
from mcca.adapters.step4_multicam_scan import scan_pose_observations
from mcca.core.config import load_config
from mcca.core.detection import get_detection_settings
from mcca.core.extrinsics_graph import transform_payload
from mcca.core.rigid import invert_T
from mcca.core.step4_multicam import (
    CameraIntrinsics,
    EdgeObs,
    PoseObs,
    connected_components,
    solve_bundle_adjustment_extrinsics,
    solve_pose_graph_extrinsics,
    summarize_edge_errors,
    summarize_pnp_reproj_px,
    summarize_post_extrinsics_reproj_px,
)


VERBOSE: bool = True


def _vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


def _fmt_path_short(p: Optional[Path]) -> str:
    """用于控制台诊断输出的短路径显示（避免一行太长）。"""

    if p is None:
        return ""
    try:
        parts = list(p.parts)
        if len(parts) >= 3:
            return str(Path(parts[-3]) / parts[-2] / parts[-1])
        if len(parts) == 2:
            return str(Path(parts[-2]) / parts[-1])
        return str(p)
    except Exception:
        return str(p)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_intrinsics(cam: str, intrinsics_dir: Path) -> CameraIntrinsics:
    """加载单个相机内参。约定：results/<cam>_intrinsics.json。"""

    path = intrinsics_dir / f"{cam}_intrinsics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到相机内参文件：{path}。\n"
            "请先为每个相机生成内参结果（schema 与 step3 输出一致：camera_matrix/dist_coeffs）。"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return CameraIntrinsics(name=str(cam), K=K, dist=dist)


def _list_cameras(image_root: Path, cameras_arg: Optional[Sequence[str]]) -> List[str]:
    if cameras_arg:
        return [str(c) for c in cameras_arg]

    cams: List[str] = []
    for p in sorted(image_root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith(".") or name.startswith("__") or name.startswith("_"):
            continue
        cams.append(name)

    if len(cams) == 0:
        raise FileNotFoundError(
            f"未在 {image_root} 下找到相机子目录。\n"
            "建议目录结构：images/filtered/cam0/*.png, images/filtered/cam1/*.png ..."
        )

    return cams


# 扫描/检测/PnP/早停已下沉到 adapters：`mcca.adapters.step4_multicam_scan`。


def _edge_metrics(
    *,
    X: Dict[str, np.ndarray],
    edge: EdgeObs,
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]],
) -> Dict[str, Any]:
    """计算单条边的残差指标，并补齐定位信息（帧号/两相机图片路径）。"""

    Xi = X[str(edge.cam_i)]
    Xj = X[str(edge.cam_j)]
    pred = Xi @ invert_T(Xj, "inv_Xj")
    err_T = invert_T(np.asarray(edge.Ci_T_Cj, dtype=np.float64), "inv_meas") @ pred

    # 使用 se(3) log 的 6D 向量做排序指标，同时拆出更直观的 rot/translation
    from mcca.core.lie import se3_log

    r6 = se3_log(err_T)
    rot_deg = float(np.linalg.norm(r6[:3]) * 180.0 / np.pi)
    trans_mm = float(np.linalg.norm(r6[3:]))
    norm6 = float(np.linalg.norm(r6))
    w = float(max(1e-9, float(edge.weight)))
    weighted_norm6 = float(np.sqrt(w) * norm6)

    frame_key = str(edge.frame_key)
    pi = (poses_by_frame.get(frame_key) or {}).get(str(edge.cam_i))
    pj = (poses_by_frame.get(frame_key) or {}).get(str(edge.cam_j))

    path_i = (paths_by_frame.get(frame_key) or {}).get(str(edge.cam_i))
    path_j = (paths_by_frame.get(frame_key) or {}).get(str(edge.cam_j))

    return {
        "pair": f"{edge.cam_i}__{edge.cam_j}",
        "cam_i": str(edge.cam_i),
        "cam_j": str(edge.cam_j),
        "frame_key": frame_key,
        "weight": float(edge.weight),
        "rot_deg": rot_deg,
        "trans_mm": trans_mm,
        "norm6": norm6,
        "weighted_norm6": weighted_norm6,
        "reproj_mean_px": {
            "cam_i": float(pi.reproj_mean_px) if pi is not None else None,
            "cam_j": float(pj.reproj_mean_px) if pj is not None else None,
        },
        "n_tags": {
            "cam_i": int(pi.n_tags) if pi is not None else None,
            "cam_j": int(pj.n_tags) if pj is not None else None,
        },
        "image_path": {
            "cam_i": str(path_i) if path_i is not None else None,
            "cam_j": str(path_j) if path_j is not None else None,
        },
    }


def _save_edge_diagnostics(
    *,
    path: Path,
    X: Dict[str, np.ndarray],
    edges: Sequence[EdgeObs],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]],
    top_k: int,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """保存边级诊断信息（最大残差边列表等）。"""

    top_k = int(max(1, top_k))
    metrics = [
        _edge_metrics(X=X, edge=e, poses_by_frame=poses_by_frame, paths_by_frame=paths_by_frame)
        for e in edges
    ]

    def _top(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        return sorted(items, key=lambda d: float(d.get(key, 0.0)), reverse=True)[:top_k]

    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for m in metrics:
        by_pair.setdefault(str(m["pair"]), []).append(m)

    out = {
        "meta": meta,
        "top_k": top_k,
        "ranking": {
            "by_rot_deg": _top(metrics, "rot_deg"),
            "by_trans_mm": _top(metrics, "trans_mm"),
            "by_weighted_norm6": _top(metrics, "weighted_norm6"),
        },
        "by_pair": {
            pair: {
                "n": int(len(lst)),
                "top_by_rot_deg": _top(lst, "rot_deg"),
                "top_by_trans_mm": _top(lst, "trans_mm"),
            }
            for pair, lst in by_pair.items()
        },
    }

    _save_json(path, out)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Step4 (Multi-Cam): AprilTag 多相机外参（位姿图优化/BA）")
    parser.add_argument("--config", type=str, default="config/apriltag_config.json", help="配置文件路径")
    parser.add_argument("--image_root", type=str, default="images/filtered", help="图像根目录")
    parser.add_argument("--cameras", nargs="*", default=None, help="相机名列表（不填则自动扫描 image_root）")
    parser.add_argument("--reference", type=str, default="", help="参考相机名（不填默认取 cameras[0]）")
    parser.add_argument("--intrinsics_dir", type=str, default="results", help="内参文件目录")
    parser.add_argument("--max_frames", type=int, default=0, help="最多处理多少个 frame_key（0=不限制）")
    parser.add_argument(
        "--min_tags",
        type=int,
        default=0,
        help="每张图最少 tag 数（0=使用 config.calibration_settings.min_tags_for_pose）",
    )
    parser.add_argument("--huber", type=float, default=1.0, help="Pose-Graph 的 Huber loss f_scale")

    parser.add_argument(
        "--method",
        type=str,
        default="ba",
        choices=["pose_graph", "ba"],
        help="优化方法：pose_graph 或 ba（多相机 BA 更稳）",
    )
    parser.add_argument("--ba_f_scale_px", type=float, default=3.0, help="BA 的鲁棒核尺度（像素）")
    parser.add_argument("--ba_max_nfev", type=int, default=1000, help="BA 最大迭代次数")
    parser.add_argument(
        "--ba_prune_mean_px",
        type=float,
        default=30.0,
        help="BA 后按单观测 mean_px 剔除离群（像素，<=0 禁用）",
    )
    parser.add_argument(
        "--ba_prune_min_keep_per_cam",
        type=int,
        default=3,
        help="离群剔除时每个相机至少保留多少条观测",
    )

    parser.add_argument(
        "--max_reproj_mean_px",
        type=float,
        default=0.0,
        help="过滤 PnP 重投影均值误差过大的观测（像素，0=禁用）",
    )
    parser.add_argument(
        "--edge_diag_path",
        type=str,
        default="results/step4_multi_edge_diagnostics.json",
        help="保存边级诊断信息的 JSON 路径（空字符串=不保存）",
    )
    parser.add_argument("--edge_diag_top_k", type=int, default=20, help="诊断输出 top-K 边")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="输出更多过程信息（--no-verbose 关闭）",
    )

    # 性能：流式扫描/早停/并行/缓存/预筛选
    parser.add_argument(
        "--target_valid_frames",
        type=int,
        default=150,
        help="达到多少个‘有效帧’就提前停止（0=不按此条件停止）",
    )
    parser.add_argument("--max_total_frames", type=int, default=0, help="最多尝试多少候选 frame_key（0=自动）")
    parser.add_argument("--max_detect_seconds", type=float, default=0.0, help="检测总耗时上限（秒，0=不限制）")
    parser.add_argument(
        "--scan_strategy",
        type=str,
        default="uniform",
        choices=["sequential", "random", "uniform"],
        help="候选 frame_key 扫描策略",
    )
    parser.add_argument("--scan_seed", type=int, default=0, help="random 策略随机种子")
    parser.add_argument("--workers", type=int, default=0, help="多进程 worker 数（0=自动；1=禁用并行）")
    parser.add_argument("--prefetch", type=int, default=0, help="并行时预提交任务数（0=自动）")
    parser.add_argument("--cache_dir", type=str, default="cache/apriltag_detection", help="检测缓存目录")
    parser.add_argument("--no_cache", action="store_true", help="禁用检测缓存")
    parser.add_argument("--force_redetect", action="store_true", help="忽略缓存强制重新检测")
    parser.add_argument("--prefilter", action="store_true", help="启用廉价预筛选")
    parser.add_argument("--no_stop_when_connected", action="store_true", help="禁用‘连通且边足够就早停’")
    parser.add_argument("--stop_min_edges", type=int, default=0, help="动态早停要求的最小边数量（0=自动）")
    parser.add_argument("--no_multiscale", action="store_true", help="关闭多尺度检测")
    parser.add_argument("--no_opencv_refine", action="store_true", help="关闭 OpenCV refine")

    args = parser.parse_args(list(argv) if argv is not None else None)

    global VERBOSE
    VERBOSE = bool(args.verbose)

    print("=" * 60)
    print("Step 4 (Multi-Cam): 多相机外参标定（位姿图优化/BA）")
    print("=" * 60)

    config = load_config(str(args.config))
    ds = config.get("image_dataset", {}) if isinstance(config, dict) else {}
    use_dataset = bool(ds.get("enabled", False)) if isinstance(ds, dict) else False

    image_root_str = str(args.image_root)
    if use_dataset and image_root_str.strip() == "images/filtered":
        try:
            filtered_root = ds.get("filtered_root", "images/filtered")
            if isinstance(filtered_root, str) and filtered_root.strip():
                image_root_str = filtered_root.strip()
        except Exception:
            pass

    image_root = Path(image_root_str)
    intr_dir = Path(str(args.intrinsics_dir))
    if not image_root.exists():
        print(f"错误: image_root 不存在：{image_root}")
        return 1

    cameras_arg = args.cameras
    if use_dataset and cameras_arg is None:
        cams_cfg = ds.get("cameras", {}) if isinstance(ds, dict) else {}
        if isinstance(cams_cfg, dict) and len(cams_cfg) > 0:
            cameras_arg = [k for k in cams_cfg.keys() if isinstance(k, str) and k and not k.startswith(("_", ".", "__"))]
        elif isinstance(cams_cfg, list) and len(cams_cfg) > 0:
            cameras_arg = [str(x) for x in cams_cfg if str(x) and not str(x).startswith(("_", ".", "__"))]

    cameras = _list_cameras(image_root, cameras_arg)
    if len(cameras) < 2:
        print("错误: 至少需要 2 个相机")
        return 1

    reference = str(args.reference).strip() if str(args.reference).strip() else cameras[0]
    if reference not in cameras:
        print(f"错误: reference={reference} 不在 cameras 列表中：{cameras}")
        return 1

    min_tags_cfg = int(config.get("calibration_settings", {}).get("min_tags_for_pose", 4))
    min_tags = int(args.min_tags) if int(args.min_tags) > 0 else min_tags_cfg

    print(f"图像根目录: {image_root}")
    print(f"相机列表: {cameras}")
    print(f"参考相机: {reference}")
    print(f"min_tags: {min_tags}")

    intrinsics: Dict[str, CameraIntrinsics] = {}
    for cam in cameras:
        intrinsics[cam] = _load_intrinsics(cam, intr_dir)

    use_multiscale, opencv_refine = get_detection_settings(config)
    if bool(getattr(args, "no_multiscale", False)):
        use_multiscale = False
    if bool(getattr(args, "no_opencv_refine", False)):
        opencv_refine = False

    cache_cfg = CacheConfig(
        enabled=not bool(args.no_cache),
        cache_dir=str(args.cache_dir),
        force_redetect=bool(args.force_redetect),
    )
    prefilter_cfg = PrefilterConfig(enabled=bool(args.prefilter))

    scan_limits = ScanLimits(
        target_valid=int(args.target_valid_frames),
        max_total=int(args.max_total_frames),
        max_seconds=float(args.max_detect_seconds),
    )
    scan_order = ScanOrder(strategy=str(args.scan_strategy), seed=int(args.scan_seed))

    stop_min_edges = int(args.stop_min_edges)
    if stop_min_edges <= 0:
        stop_min_edges = int(max(10 * (len(cameras) - 1), 20))

    poses_by_frame, paths_by_frame, edges, scan_report = scan_pose_observations(
        image_root=image_root,
        cameras=cameras,
        intrinsics=intrinsics,
        config=config,
        max_frames=int(args.max_frames),
        min_tags=int(min_tags),
        scan_limits=scan_limits,
        scan_order=scan_order,
        workers=int(args.workers),
        prefetch=int(args.prefetch),
        cache_cfg=cache_cfg,
        prefilter_cfg=prefilter_cfg,
        use_multiscale=bool(use_multiscale),
        opencv_refine=bool(opencv_refine),
        stop_when_connected=not bool(args.no_stop_when_connected),
        stop_min_edges=int(stop_min_edges),
        max_reproj_mean_px=float(args.max_reproj_mean_px),
        verbose=bool(VERBOSE),
    )

    print(f"PnP 有效帧数: {len(poses_by_frame)}")
    print(f"位姿图边数量: {len(edges)}")

    if len(edges) == 0:
        _save_json(Path("results/step4_multi_scan_report.json"), scan_report)

        cand = int((scan_report.get("frame_sync", {}) or {}).get("candidate_frame_keys_ge2", 0))
        if cand == 0:
            print("错误: 没有构建出任何边：因为不存在任何‘至少两相机同帧(frame_key)’候选。")
            print("建议：统一各相机的文件命名规则，让同一时刻的帧拥有相同的 frame_key。")
        else:
            print("错误: 没有构建出任何边（同帧共同看到板的相机对为 0）。")
            print("建议：增加采集姿态/光照；降低 min_tags；检查 tag family/ROI；确保内参与板模型正确。")

        print("已保存诊断报告: results/step4_multi_scan_report.json")
        return 1

    comps = connected_components(cameras, edges)
    print(f"图连通分量数量: {len(comps)}")
    for i, comp in enumerate(comps, 1):
        print(f"  component#{i}: {comp}")

    if reference not in comps[0]:
        print("错误: 参考相机不在最大连通分量中，这通常表示数据命名/同步有问题。")
        return 1

    if len(comps[0]) < len(cameras):
        print("\n警告: 位姿图不连通，只能求出与参考相机同一连通分量中的相机外参。")
        print(f"  可解相机: {comps[0]}")

    cameras_solved = [c for c in cameras if c in comps[0]]
    edges_solved = [e for e in edges if (e.cam_i in cameras_solved and e.cam_j in cameras_solved)]

    poses_by_frame_solved: Dict[str, Dict[str, PoseObs]] = {}
    paths_by_frame_solved: Dict[str, Dict[str, Path]] = {}
    cams_solved_set = {str(c) for c in cameras_solved}
    for fk, poses in poses_by_frame.items():
        pp = {str(c): p for c, p in poses.items() if str(c) in cams_solved_set}
        if len(pp) < 2:
            continue
        poses_by_frame_solved[str(fk)] = pp
        paths_by_frame_solved[str(fk)] = {
            str(c): p for c, p in (paths_by_frame.get(fk) or {}).items() if str(c) in cams_solved_set
        }

    reproj_stats = summarize_pnp_reproj_px(
        cameras=cameras_solved,
        edges=edges_solved,
        poses_by_frame=poses_by_frame_solved,
        paths_by_frame=paths_by_frame_solved,
        top_k=5,
    )

    print("\nPnP 自身重投影误差统计（像素，越小越好；仅统计参与建图的观测）")
    for cam in cameras_solved:
        s = (reproj_stats.get("by_cam") or {}).get(str(cam)) or {}
        if int(s.get("n", 0)) <= 0:
            print(f"  - {cam}: n=0")
            continue
        print(
            f"  - {cam}: n={int(s['n'])} mean={float(s['mean']):.4f} "
            f"median={float(s['median']):.4f} p95={float(s['p95']):.4f} max={float(s['max']):.4f}"
        )

    es = reproj_stats.get("edge_mean_px") or {}
    if int(es.get("n", 0)) > 0:
        print(
            "\n边的平均重投影误差（像素，edge_mean_px=0.5*(cam_i+cam_j)；与边权重一致）"
            f"\n  - n={int(es['n'])} mean={float(es['mean']):.4f} median={float(es['median']):.4f} "
            f"p95={float(es['p95']):.4f} max={float(es['max']):.4f}"
        )

        worst = reproj_stats.get("worst_edges_by_edge_mean_px") or []
        if len(worst) > 0:
            print("\n诊断：edge_mean_px 最大的若干条边（更像‘检测/同步’问题的信号）")
            for i, m in enumerate(worst[: min(5, len(worst))], 1):
                p_i = (m.get("image_path") or {}).get("cam_i")
                p_j = (m.get("image_path") or {}).get("cam_j")
                short_i = _fmt_path_short(Path(p_i)) if p_i else ""
                short_j = _fmt_path_short(Path(p_j)) if p_j else ""
                print(
                    f"  #{i} pair={m.get('pair')} frame={m.get('frame_key')} "
                    f"edge_mean_px={float(m.get('edge_mean_px', 0.0)):.4f} "
                    f"paths=({short_i}) | ({short_j})"
                )

    method = str(args.method)
    # 仅 BA 会产出每帧的 Cref_T_B（Ref <- Board）。pose_graph 模式下保持为空。
    Cref_T_B_opt: Dict[str, np.ndarray] = {}
    post_reproj: Dict[str, Any]
    solver_rms: float
    solver_units: str
    solver_extra: Dict[str, Any]

    if method == "ba":
        print("\n开始多相机 Bundle Adjustment（像素域联合优化）...")

        X_opt, Cref_T_B_opt, res, prune_info, _obs_used, ba_reproj = solve_bundle_adjustment_extrinsics(
            cameras=cameras_solved,
            reference=reference,
            poses_by_frame=poses_by_frame_solved,
            paths_by_frame=paths_by_frame_solved,
            intrinsics=intrinsics,
            f_scale_px=float(args.ba_f_scale_px),
            max_nfev=int(args.ba_max_nfev),
            prune_mean_px=float(args.ba_prune_mean_px),
            prune_min_keep_per_cam=int(args.ba_prune_min_keep_per_cam),
        )

        r1 = np.asarray(res.fun, dtype=np.float64).reshape(-1)
        solver_rms = float(np.sqrt(np.mean(r1 * r1))) if r1.size else 0.0
        solver_units = "px"
        solver_extra = {
            "method": "bundle_adjustment",
            "ba_f_scale_px": float(args.ba_f_scale_px),
            "ba_max_nfev": int(args.ba_max_nfev),
            "ba_prune": prune_info,
        }
        post_reproj = ba_reproj

        print("\nBA 优化后的重投影误差（像素，越小越好）")
        ov = post_reproj.get("overall") or {}
        if int(ov.get("n", 0)) > 0:
            print(
                f"  - overall: n={int(ov['n'])} mean={float(ov['mean']):.4f} "
                f"median={float(ov['median']):.4f} p95={float(ov['p95']):.4f} max={float(ov['max']):.4f}"
            )
    else:
        print("\n开始位姿图优化...")

        X_opt, res, edge_err_solver = solve_pose_graph_extrinsics(
            cameras=cameras_solved,
            reference=reference,
            edges=edges_solved,
            huber_f_scale=float(args.huber),
            max_nfev=2000,
        )

        r1 = np.asarray(res.fun, dtype=np.float64).reshape(-1)
        solver_rms = float(np.sqrt(np.mean(r1 * r1))) if r1.size else 0.0
        solver_units = "6D"
        solver_extra = {"method": "pose_graph", "huber_f_scale": float(args.huber)}

        post_reproj = summarize_post_extrinsics_reproj_px(
            X_cam_from_ref=X_opt,
            reference=reference,
            cameras=cameras_solved,
            intrinsics=intrinsics,
            poses_by_frame=poses_by_frame_solved,
            paths_by_frame=paths_by_frame_solved,
            top_k=10,
        )

    edge_err = summarize_edge_errors(X=X_opt, edges=edges_solved)
    print(
        "\n边误差统计（更直观）："
        f"\n  - rot_deg: median={edge_err['rot_deg']['median']:.3f}, p95={edge_err['rot_deg']['p95']:.3f}, max={edge_err['rot_deg']['max']:.3f}"
        f"\n  - trans_mm: median={edge_err['trans']['median']:.3f}, p95={edge_err['trans']['p95']:.3f}, max={edge_err['trans']['max']:.3f}"
    )

    edge_diag_path = str(args.edge_diag_path).strip()
    if edge_diag_path:
        diag = _save_edge_diagnostics(
            path=Path(edge_diag_path),
            X=X_opt,
            edges=edges_solved,
            poses_by_frame=poses_by_frame,
            paths_by_frame=paths_by_frame,
            top_k=int(args.edge_diag_top_k),
            meta={
                "timestamp": datetime.now().isoformat(),
                "image_root": str(image_root.as_posix()),
                "intrinsics_dir": str(intr_dir.as_posix()),
                "reference": str(reference),
                "cameras_solved": list(cameras_solved),
                "min_tags": int(min_tags),
                "max_reproj_mean_px": float(args.max_reproj_mean_px),
            },
        )

        print("\n诊断：误差最大的若干条边（用于定位坏帧/不同步）")
        top_rot = (diag.get("ranking", {}) or {}).get("by_rot_deg", [])
        for i, m in enumerate(top_rot[: min(5, len(top_rot))], 1):
            p_i = (m.get("image_path") or {}).get("cam_i")
            p_j = (m.get("image_path") or {}).get("cam_j")
            short_i = _fmt_path_short(Path(p_i)) if p_i else ""
            short_j = _fmt_path_short(Path(p_j)) if p_j else ""
            print(
                f"  #{i} pair={m.get('pair')} frame={m.get('frame_key')} "
                f"rot_deg={float(m.get('rot_deg', 0.0)):.3f} trans_mm={float(m.get('trans_mm', 0.0)):.3f} "
                f"paths=({short_i}) | ({short_j})"
            )

        print(f"已保存边级诊断: {edge_diag_path}")

    T_cam_from_ref: Dict[str, Dict[str, Any]] = {}
    for cam in cameras_solved:
        T = X_opt[str(cam)]
        T_cam_from_ref[str(cam)] = {"T": T.tolist(), "R": T[:3, :3].tolist(), "t": T[:3, 3].tolist()}

    out_extr: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "reference": reference,
        "cameras": cameras,
        "cameras_solved": cameras_solved,
        "note": "T_cam_from_ref 表示 Cam_i <- Cam_ref（与本工程 Cr_T_Cl 命名一致：目标在前，源在后）",
        # 说明：Step4 的位姿图平移量与标定板 3D 点使用同一单位。
        # 本工程 AprilTag 标定板在配置中以 mm 组织（apriltag_board.unit=mm），
        # 因此这里的平移量也是 mm。
        # 下游（Step5 等）会通过 mcca.core.extrinsics_graph.load_extrinsics_graph
        # 在读取时统一换算到米，避免尺度悄悄漂移。
        "translation_unit": "mm",
        "T_cam_from_ref": T_cam_from_ref,
        "solver": {
            "success": bool(getattr(res, "success", False)),
            "status": int(getattr(res, "status", 0)),
            "message": str(getattr(res, "message", "")),
            "nfev": int(getattr(res, "nfev", 0)),
            "cost": float(getattr(res, "cost", 0.0)),
            "rms_final": float(solver_rms),
            "rms_units": str(solver_units),
            **solver_extra,
        },
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "image_root": str(image_root.as_posix()),
        "intrinsics_dir": str(intr_dir.as_posix()),
        "reference": reference,
        "min_tags": int(min_tags),
        "poses_frames": int(len(poses_by_frame)),
        "edges_total": int(len(edges)),
        "edges_used": int(len(edges_solved)),
        "components": comps,
        "reproj_error_px": reproj_stats,
        "post_reproj_error_px": post_reproj,
        "solver": {
            "success": bool(getattr(res, "success", False)),
            "status": int(getattr(res, "status", 0)),
            "message": str(getattr(res, "message", "")),
            "nfev": int(getattr(res, "nfev", 0)),
            "cost": float(getattr(res, "cost", 0.0)),
            "rms_final": float(solver_rms),
            "rms_units": str(solver_units),
            **solver_extra,
        },
        "edge_error": edge_err,
    }

    # 额外输出：相机“位姿”（相对参考相机的反向变换）。
    #
    # 约定：A_T_B 表示 B->A。
    # - Step4 外参文件输出：T_cam_from_ref（Cam <- Ref）。
    # - 本文件输出：T_ref_from_cam（Ref <- Cam），便于把“相机在参考系下的位姿”当作同一个方向来阅读。
    T_ref_from_cam: Dict[str, Dict[str, Any]] = {}
    for cam in cameras_solved:
        cam_s = str(cam)
        T = invert_T(np.asarray(X_opt[cam_s], dtype=np.float64), name=f"T_ref_from_cam[{cam_s}]")
        T_ref_from_cam[cam_s] = transform_payload(
            T,
            parent_frame=f"camera:{reference}",
            child_frame=f"camera:{cam_s}",
            name=f"T_ref_from_cam[{cam_s}]",
            include_inverse=False,
        )

    # 同时输出每帧的“板位姿”（Board 在参考相机坐标系下的位姿）。
    # - BA：天然有每帧 Cref_T_B（Ref <- Board）。
    # - pose_graph：使用每帧 reproj 最小的相机作为 anchor 估计 Ref <- Board。
    T_ref_from_board_by_frame: Dict[str, Any] = {}
    if method == "ba":
        for fk, T in (Cref_T_B_opt or {}).items():
            fk_s = str(fk)
            Ti = np.asarray(T, dtype=np.float64)
            T_ref_from_board_by_frame[fk_s] = {
                "source": "ba",
                "anchor_cam": None,
                "T_ref_from_board": transform_payload(
                    Ti,
                    parent_frame=f"camera:{reference}",
                    child_frame="board",
                    name=f"T_ref_from_board[{fk_s}]",
                    include_inverse=False,
                ),
            }
    else:
        for fk, poses in poses_by_frame_solved.items():
            if len(poses) == 0:
                continue
            # 选择该帧中 PnP reproj 最小的相机作为 anchor，减少坏帧/坏相机的影响。
            anchor_po = sorted(poses.values(), key=lambda p: float(p.reproj_mean_px))[0]
            anchor_cam = str(anchor_po.cam)
            Cref_T_Canchor = invert_T(
                np.asarray(X_opt[anchor_cam], dtype=np.float64),
                name=f"Cref_T_C[{anchor_cam}]",
            )
            Cref_T_B = Cref_T_Canchor @ np.asarray(anchor_po.C_T_B, dtype=np.float64)

            fk_s = str(fk)
            T_ref_from_board_by_frame[fk_s] = {
                "source": "pose_graph_anchor",
                "anchor_cam": anchor_cam,
                "anchor_reproj_mean_px": float(anchor_po.reproj_mean_px),
                "n_cams_in_frame": int(len(poses)),
                "T_ref_from_board": transform_payload(
                    Cref_T_B,
                    parent_frame=f"camera:{reference}",
                    child_frame="board",
                    name=f"T_ref_from_board[{fk_s}]",
                    include_inverse=False,
                ),
            }

    out_pose: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "reference": reference,
        "cameras": cameras,
        "cameras_solved": cameras_solved,
        "note": {
            "T_ref_from_cam": "T_ref_from_cam 表示 Cam_ref <- Cam_i（相机在参考相机坐标系下的位姿表达）。",
            "T_ref_from_board_by_frame": "T_ref_from_board_by_frame 表示 Cam_ref <- Board（每帧标定板在参考相机坐标系下的位姿）。",
        },
        # 与 out_extr 一致：该文件中的平移量单位为 mm。
        "translation_unit": "mm",
        "T_ref_from_cam": T_ref_from_cam,
        "T_ref_from_board_by_frame": T_ref_from_board_by_frame,
        "solver": {
            "method": str(method),
            "success": bool(getattr(res, "success", False)),
            "status": int(getattr(res, "status", 0)),
            "message": str(getattr(res, "message", "")),
            "nfev": int(getattr(res, "nfev", 0)),
        },
    }

    _save_json(Path("results/multi_camera_extrinsics.json"), out_extr)
    _save_json(Path("results/multi_camera_poses.json"), out_pose)
    _save_json(Path("results/multi_camera_pose_graph_report.json"), report)
    _save_json(Path("results/step4_multi_scan_report.json"), scan_report)

    print("\n已保存:")
    print("  - results/multi_camera_extrinsics.json")
    print("  - results/multi_camera_poses.json")
    print("  - results/multi_camera_pose_graph_report.json")
    print("  - results/step4_multi_scan_report.json")

    return 0


def cli_main() -> None:
    """console_script 入口：按照惯例抛出 SystemExit。"""

    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
