#!/usr/bin/env python3
"""Step 6: 融合多组 Step5(camera_to_base.json) 的外参结果（支持多相机）

目的
- 用户会在不同“板位置/采集批次”重复运行 Step5b，得到多份 `camera_to_base.json`。
- Step6 将这些结果做鲁棒融合，得到更稳定的 `B_T_C`（每个相机一份）。

核心原则
- 平移可以做加权均值/中位数，但旋转不能直接对矩阵逐元素平均。
- 旋转使用四元数的 Markley 平均（最大特征值对应特征向量）。
- 通过 SE(3) 残差（旋转角 + 平移范数）做离群剔除，再迭代重算。

输入
- 多个 Step5 输出的 `camera_to_base.json`（路径可为文件 / 目录 / glob）。

输出
- 默认：results/camera_to_base_fused.json
- 同时生成一个简要报告：results/camera_to_base_fusion_report.json

使用示例
- 融合两次运行：
  python step6_fuse_step5_results.py -i run1/results/camera_to_base.json run2/results/camera_to_base.json

- 输入目录（会递归搜索 camera_to_base.json）：
  python step6_fuse_step5_results.py -i ./archive

- 使用 glob：
  python step6_fuse_step5_results.py -i "archive/**/camera_to_base.json"

说明
- 变换命名约定：B_T_C 表示 C -> B（点从相机坐标系变到底盘坐标系）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Sample:
    path: str
    data: Dict[str, Any]


def _make_warn(collector: List[str], *, verbose: bool) -> Callable[[str], None]:
    """创建一个轻量告警函数。

    - 始终把告警写入 collector（用于写入 report JSON，便于复盘）
    - verbose=True 时，额外打印到控制台
    """

    def _warn(msg: str) -> None:
        collector.append(str(msg))
        if verbose:
            print(f"⚠ {msg}")

    return _warn


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_transform(T: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: 期望(4,4)，得到 {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name}: 包含 NaN/Inf")
    bottom = T[3, :]
    if float(np.linalg.norm(bottom - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-6:
        raise ValueError(f"{name}: 最后一行应为[0,0,0,1]，当前={bottom}")

    R = T[:3, :3]
    det = float(np.linalg.det(R))
    if not (0.5 < det < 1.5):
        raise ValueError(f"{name}: 旋转子块 det 异常 det={det}")
    # 轻量正交检查（不做过严，避免误杀）
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    if ortho_err > 5e-2:
        raise ValueError(f"{name}: 旋转子块非正交，||R^T R - I||_F={ortho_err:.3e}")
    return T


def _make_transform(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rm = np.asarray(Rm, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return _ensure_transform(T, "T")


def _inv_T(T: np.ndarray) -> np.ndarray:
    # 针对刚体变换使用解析逆（稳定、快）
    Rm = T[:3, :3]
    t = T[:3, 3]
    R_inv = Rm.T
    t_inv = -R_inv @ t
    return _make_transform(R_inv, t_inv)


def _rot_err_deg(Rm: np.ndarray) -> float:
    return float(Rotation.from_matrix(Rm).magnitude() * 180.0 / np.pi)


def _delta_errors(T_ref: np.ndarray, T_i: np.ndarray) -> Tuple[float, float]:
    """返回 (rot_err_deg, trans_err_m)。"""
    Delta = _inv_T(T_ref) @ T_i
    return _rot_err_deg(Delta[:3, :3]), float(np.linalg.norm(Delta[:3, 3]))


def _quat_average_markley(
    quats_xyzw: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    ref_quat_xyzw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Markley 四元数平均（返回 xyzw）。

    输入 quats 为 Nx4，scipy 的 Rotation.as_quat() 默认是 (x, y, z, w)。
    """
    Q = np.asarray(quats_xyzw, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[1] != 4:
        raise ValueError(f"quats_xyzw 期望 Nx4，得到 {Q.shape}")

    n = Q.shape[0]
    if n == 1:
        q = Q[0]
        return q / np.linalg.norm(q)

    if weights is None:
        w = np.ones((n,), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(n)
        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
        if float(w.sum()) <= 0.0:
            w = np.ones((n,), dtype=np.float64)

    # 统一符号：让所有四元数与参考四元数同半球
    if ref_quat_xyzw is None:
        ref = Q[0]
    else:
        ref = np.asarray(ref_quat_xyzw, dtype=np.float64).reshape(4)

    Q2 = Q.copy()
    dots = (Q2 * ref.reshape(1, 4)).sum(axis=1)
    Q2[dots < 0.0] *= -1.0

    # M = Σ w_i q_i q_i^T
    M = np.zeros((4, 4), dtype=np.float64)
    for qi, wi in zip(Q2, w):
        if wi <= 0:
            continue
        M += wi * np.outer(qi, qi)

    # 最大特征值对应的特征向量
    eigvals, eigvecs = np.linalg.eigh(M)
    q_avg = eigvecs[:, int(np.argmax(eigvals))]
    q_avg = q_avg / np.linalg.norm(q_avg)

    # 输出也与 ref 同半球（可读性好）
    if float(np.dot(q_avg, ref)) < 0.0:
        q_avg = -q_avg

    return q_avg


def _rotation_mean_from_mats(
    R_list: List[np.ndarray], weights: Optional[np.ndarray], ref_R: np.ndarray
) -> np.ndarray:
    rots = Rotation.from_matrix(np.asarray(R_list, dtype=np.float64))
    quats = rots.as_quat()  # xyzw
    ref_q = Rotation.from_matrix(ref_R).as_quat()
    q_mean = _quat_average_markley(quats, weights, ref_quat_xyzw=ref_q)
    return Rotation.from_quat(q_mean).as_matrix()


def _extract_transform(data: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    """从 JSON 中提取 4x4 变换。

    注意：历史版本/不同脚本可能会产生不符合 SO(3) 的旋转子块（det=-1 反射）。
    Step6 默认选择“跳过该样本”，而不是直接中断整个融合流程。
    """
    if key not in data:
        return None
    T = np.array(data[key], dtype=np.float64)
    return _ensure_transform(T, key)


def _try_extract_transform(
    data: Dict[str, Any],
    key: str,
    *,
    sample_path: str,
    warn: Optional[Callable[[str], None]] = None,
) -> Optional[np.ndarray]:
    """安全提取变换：失败则记录告警并返回 None。

    说明：历史数据里常见旋转子块异常（例如 det<0 的“反射”）。
    Step6 默认跳过该样本，避免单点异常中断整个融合流程。
    """
    if key not in data:
        return None
    try:
        return _extract_transform(data, key)
    except Exception as e:
        hint = ""
        # 常见坑：det=-1 表示反射（左手系/轴镜像/脚本版本不一致）
        try:
            Rm = np.asarray(data[key], dtype=np.float64)[:3, :3]
            det = float(np.linalg.det(Rm))
            if det < 0:
                hint = "（det<0: 反射；通常意味着坐标系定义/脚本版本不一致，建议用同一版本 Step5b 重新生成）"
        except Exception:
            hint = ""

        if warn is not None:
            warn(f"跳过 {sample_path} 的 {key}：{e} {hint}".strip())
        return None


def _load_samples(input_paths: List[str], *, warn: Callable[[str], None]) -> List[Sample]:
    """读取输入 JSON，读取失败的样本会被跳过并记录告警。"""
    samples: List[Sample] = []
    for p in input_paths:
        try:
            samples.append(Sample(path=p, data=_read_json(p)))
        except Exception as e:
            warn(f"跳过 {p}：读取失败：{e}")
    return samples


def _weight_from_data(data: Dict[str, Any], cam: str, mode: str) -> float:
    """从样本 JSON 中提取某相机的权重。"""
    if mode == "equal":
        return 1.0

    # mode == 'valid_poses'
    try:
        pose_stats = data.get("pose_stats", {})
        if isinstance(pose_stats, dict) and cam in pose_stats:
            return float(max(1, int(pose_stats.get(cam, {}).get("valid_poses", 1))))
    except Exception:
        return 1.0
    return 1.0


def _try_extract_B_T_C(
    data: Dict[str, Any],
    *,
    sample_path: str,
    warn: Callable[[str], None],
) -> Dict[str, np.ndarray]:
    """安全提取 B_T_C 字典。

    返回一个 {cam: 4x4} 字典；若样本里不存在或提取失败则返回空字典。
    """
    if "B_T_C" not in data:
        return {}
    raw = data.get("B_T_C")
    if not isinstance(raw, dict):
        warn(f"跳过 {sample_path} 的 B_T_C：不是 dict")
        return {}

    out: Dict[str, np.ndarray] = {}
    for cam, T in raw.items():
        if not isinstance(cam, str):
            continue
        try:
            out[cam] = _ensure_transform(np.asarray(T, dtype=np.float64), f"B_T_C[{cam}]")
        except Exception as e:
            warn(f"跳过 {sample_path} 的 B_T_C[{cam}]：{e}")
    return out


def _choose_medoid_index(
    Ts: List[np.ndarray],
    *,
    rot_scale_deg: float,
    trans_scale_m: float,
) -> int:
    """选一个“最像大家”的样本作为参考，降低首轮平均被离群点拖偏的风险。"""
    if len(Ts) == 1:
        return 0

    costs = []
    for i, Ti in enumerate(Ts):
        c = 0.0
        for Tj in Ts:
            dr, dt = _delta_errors(Ti, Tj)
            c += (dr / max(rot_scale_deg, 1e-9)) + (dt / max(trans_scale_m, 1e-9))
        costs.append(c)
    return int(np.argmin(np.array(costs, dtype=np.float64)))


def _fuse_transforms(
    Ts: List[np.ndarray],
    weights: List[float],
    *,
    rot_thresh_deg: float,
    trans_thresh_m: float,
    max_iter: int,
    min_inliers: int,
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """返回 (T_fused, stats, per_sample_errors)。"""
    if len(Ts) == 0:
        raise ValueError("没有可融合的变换")

    w = np.asarray(weights, dtype=np.float64).reshape(len(Ts))
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)

    # 选参考样本，避免被明显离群点主导
    ref_idx = _choose_medoid_index(
        Ts,
        rot_scale_deg=max(rot_thresh_deg, 1e-6),
        trans_scale_m=max(trans_thresh_m, 1e-9),
    )

    # 先用参考样本做一次“门限筛选”，避免一上来平均就被极端离群点拖偏
    T_ref = Ts[ref_idx]
    inliers = np.zeros((len(Ts),), dtype=bool)
    for k in range(len(Ts)):
        dr, dt = _delta_errors(T_ref, Ts[k])
        inliers[k] = bool(dr <= rot_thresh_deg and dt <= trans_thresh_m)
    inliers[ref_idx] = True

    # 若初始内点不足，就直接退化成“选择参考样本”
    if int(np.sum(inliers)) < int(min_inliers):
        per_final = []
        for k in range(len(Ts)):
            dr, dt = _delta_errors(T_ref, Ts[k])
            per_final.append({"rot_err_deg": dr, "trans_err_m": dt, "inlier": bool(k == ref_idx)})
        stats = {
            "ref_index": int(ref_idx),
            "num_samples": int(len(Ts)),
            "num_inliers": 1,
            "rot_thresh_deg": float(rot_thresh_deg),
            "trans_thresh_m": float(trans_thresh_m),
            "rot_err_inliers_deg": {"mean": 0.0, "max": 0.0},
            "trans_err_inliers_m": {"mean": 0.0, "max": 0.0},
            "note": "初始内点不足，已退化为采用参考样本（medoid）作为融合结果。",
        }
        return T_ref, stats, per_final

    T_mean = T_ref
    for _it in range(max_iter):
        idx = np.where(inliers)[0].tolist()
        if len(idx) < min_inliers:
            break

        R_list = [Ts[i][:3, :3] for i in idx]
        t_list = [Ts[i][:3, 3] for i in idx]
        w_i = w[idx] if w.size == len(Ts) else None

        R_mean = _rotation_mean_from_mats(R_list, w_i, ref_R=T_mean[:3, :3])
        if w_i is None or float(np.sum(w_i)) <= 0:
            t_mean = np.mean(np.asarray(t_list, dtype=np.float64), axis=0)
        else:
            t_mean = (np.asarray(t_list, dtype=np.float64) * w_i.reshape(-1, 1)).sum(axis=0) / float(np.sum(w_i))

        T_new = _make_transform(R_mean, t_mean)

        new_inliers = np.zeros_like(inliers)
        for k in range(len(Ts)):
            dr, dt = _delta_errors(T_new, Ts[k])
            new_inliers[k] = bool(dr <= rot_thresh_deg and dt <= trans_thresh_m)

        # 如果门限把内点清空/不足，保持上一轮的 T_mean，不再继续“平均”
        if int(np.sum(new_inliers)) < int(min_inliers):
            break

        if np.array_equal(new_inliers, inliers):
            T_mean = T_new
            inliers = new_inliers
            break

        T_mean = T_new
        inliers = new_inliers

    # 最终统计
    per_final: List[Dict[str, Any]] = []
    rot_errs = []
    trans_errs = []
    for k in range(len(Ts)):
        dr, dt = _delta_errors(T_mean, Ts[k])
        inl = bool(inliers[k])
        per_final.append({"rot_err_deg": dr, "trans_err_m": dt, "inlier": inl})
        if inl:
            rot_errs.append(dr)
            trans_errs.append(dt)

    stats = {
        "ref_index": int(ref_idx),
        "num_samples": int(len(Ts)),
        "num_inliers": int(int(np.sum(inliers))),
        "rot_thresh_deg": float(rot_thresh_deg),
        "trans_thresh_m": float(trans_thresh_m),
        "rot_err_inliers_deg": {
            "mean": float(np.mean(rot_errs)) if rot_errs else None,
            "max": float(np.max(rot_errs)) if rot_errs else None,
        },
        "trans_err_inliers_m": {
            "mean": float(np.mean(trans_errs)) if trans_errs else None,
            "max": float(np.max(trans_errs)) if trans_errs else None,
        },
    }

    return T_mean, stats, per_final


def _gather_input_paths(inputs: List[str]) -> List[str]:
    paths: List[str] = []

    def add(p: str) -> None:
        if not p:
            return
        p2 = os.path.normpath(p)
        if p2 not in paths:
            paths.append(p2)

    for s in inputs:
        if any(ch in s for ch in ["*", "?", "["]):
            for p in sorted(glob.glob(s, recursive=True)):
                if os.path.isdir(p):
                    for q in sorted(glob.glob(os.path.join(p, "**", "camera_to_base.json"), recursive=True)):
                        add(q)
                else:
                    add(p)
            continue

        if os.path.isdir(s):
            for p in sorted(glob.glob(os.path.join(s, "**", "camera_to_base.json"), recursive=True)):
                add(p)
        else:
            add(s)

    return [p for p in paths if os.path.exists(p)]


def _save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step6: 融合多组 Step5(camera_to_base.json) 外参结果")
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=True,
        help="输入文件/目录/glob（会递归搜 camera_to_base.json）",
    )
    parser.add_argument(
        "--out",
        default="results/camera_to_base_fused.json",
        help="融合结果输出路径（默认：results/camera_to_base_fused.json）",
    )
    parser.add_argument(
        "--report",
        default="results/camera_to_base_fusion_report.json",
        help="融合报告输出路径（默认：results/camera_to_base_fusion_report.json）",
    )
    parser.add_argument("--rot_thresh_deg", type=float, default=3.0, help="离群剔除旋转阈值（度）")
    parser.add_argument("--trans_thresh_m", type=float, default=0.05, help="离群剔除平移阈值（米）")
    parser.add_argument("--max_iter", type=int, default=5, help="迭代次数上限")
    parser.add_argument("--min_inliers", type=int, default=2, help="最少内点数量")
    parser.add_argument(
        "--weight_mode",
        choices=["equal", "valid_poses"],
        default="valid_poses",
        help="权重来源：equal / valid_poses（默认）",
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="将融合结果同时写入 results/camera_to_base.json（覆盖）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印更多过程信息（包含跳过样本原因）",
    )

    args = parser.parse_args()

    warnings: List[str] = []
    warn = _make_warn(warnings, verbose=bool(args.verbose))

    input_paths = _gather_input_paths(args.inputs)
    if len(input_paths) == 0:
        print("未找到任何输入文件，请检查 --inputs")
        return 2

    samples = _load_samples(input_paths, warn=warn)

    if len(samples) == 0:
        print("没有可用样本（全部读取失败）")
        return 2

    # 收集每个相机的样本
    Ts_by_cam: Dict[str, List[np.ndarray]] = {}
    w_by_cam: Dict[str, List[float]] = {}
    idx_by_cam: Dict[str, List[int]] = {}

    for i, s in enumerate(samples):
        B_T_C_i = _try_extract_B_T_C(s.data, sample_path=s.path, warn=warn)
        for cam, T in B_T_C_i.items():
            Ts_by_cam.setdefault(cam, []).append(T)
            w_by_cam.setdefault(cam, []).append(_weight_from_data(s.data, cam, args.weight_mode))
            idx_by_cam.setdefault(cam, []).append(i)

    if len(Ts_by_cam) == 0:
        print("输入里没有任何 B_T_C，无法融合")
        return 2

    fused_by_cam: Dict[str, np.ndarray] = {}
    stats_by_cam: Dict[str, Any] = {}
    per_by_cam: Dict[str, Any] = {}

    for cam in sorted(Ts_by_cam.keys()):
        fused_T, stats, per = _fuse_transforms(
            Ts_by_cam[cam],
            w_by_cam.get(cam, [1.0] * len(Ts_by_cam[cam])),
            rot_thresh_deg=args.rot_thresh_deg,
            trans_thresh_m=args.trans_thresh_m,
            max_iter=args.max_iter,
            min_inliers=args.min_inliers,
        )
        fused_by_cam[cam] = fused_T
        stats_by_cam[cam] = stats
        per_by_cam[cam] = per

    now = datetime.now().isoformat()

    report_inputs: List[Dict[str, Any]] = []
    for s in samples:
        cams_in_sample = []
        try:
            if isinstance(s.data.get("B_T_C"), dict):
                cams_in_sample = sorted([str(k) for k in s.data.get("B_T_C", {}).keys()])
        except Exception:
            cams_in_sample = []

        report_inputs.append(
            {
                "path": s.path,
                "timestamp": s.data.get("timestamp"),
                "cameras": cams_in_sample,
                "pose_stats": s.data.get("pose_stats"),
            }
        )

    # 把每个相机每个样本的误差/权重填回 report_inputs
    for cam, idx_list in idx_by_cam.items():
        per_list = per_by_cam.get(cam)
        if per_list is None:
            continue
        w_list = w_by_cam.get(cam, [])
        for local_j, sample_i in enumerate(idx_list):
            report_inputs[sample_i].setdefault("per_camera", {})
            report_inputs[sample_i]["per_camera"][cam] = {
                **per_list[local_j],
                "weight": float(w_list[local_j]) if local_j < len(w_list) else 1.0,
            }

    fused_result: Dict[str, Any] = {
        "timestamp": now,
        "B_T_C": {cam: T.tolist() for cam, T in fused_by_cam.items()},
        "fusion": {
            "inputs": report_inputs,
            "stats": stats_by_cam,
            "warnings": warnings,
            "params": {
                "rot_thresh_deg": float(args.rot_thresh_deg),
                "trans_thresh_m": float(args.trans_thresh_m),
                "max_iter": int(args.max_iter),
                "min_inliers": int(args.min_inliers),
                "weight_mode": str(args.weight_mode),
            },
        },
    }

    _save_json(args.out, fused_result)
    _save_json(args.report, {"timestamp": now, **fused_result["fusion"]})

    if args.overwrite_results:
        _save_json("results/camera_to_base.json", fused_result)

    print("=" * 70)
    print("Step6 融合完成")
    print(f"  输入样本: {len(samples)}")
    if warnings:
        print(f"  告警: {len(warnings)} 条（详情见 report JSON；使用 --verbose 可在控制台显示）")
    for cam in sorted(stats_by_cam.keys()):
        st = stats_by_cam[cam]
        print(f"  {cam}: inliers={st['num_inliers']}/{st['num_samples']}")
    print(f"  输出: {args.out}")
    print(f"  报告: {args.report}")
    if args.overwrite_results:
        print("  ✓ 已覆盖写入: results/camera_to_base.json")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
