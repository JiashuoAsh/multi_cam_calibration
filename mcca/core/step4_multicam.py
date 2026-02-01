"""Step4（多相机外参）：核心求解与统计（不含扫描/缓存/CLI）。

设计约束：
- 该模块只做“纯逻辑/数学”：位姿图求解、Bundle Adjustment、误差统计。
- 不依赖 entry/adapters（依赖方向：entry/adapters -> core）。
- 输入/输出的数据结构尽量简单，可被上层脚本/入口层复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from mcca.core.lie import se3_exp, se3_log, so3_exp, so3_log
from mcca.core.rigid import ensure_T, invert_T, make_T


@dataclass(frozen=True)
class CameraIntrinsics:
    name: str
    K: np.ndarray
    dist: np.ndarray


@dataclass(frozen=True)
class PoseObs:
    """单帧单相机 PnP 观测。"""

    cam: str
    frame_key: str
    C_T_B: np.ndarray
    n_tags: int
    reproj_mean_px: float
    obj_pts: np.ndarray
    img_pts: np.ndarray


@dataclass(frozen=True)
class EdgeObs:
    """位姿图边观测：Ci <- Cj。"""

    cam_i: str
    cam_j: str
    Ci_T_Cj: np.ndarray
    weight: float
    frame_key: str


def to_rvec_tvec_from_T(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将 4x4 齐次变换转换为 OpenCV 可用的 (rvec, tvec)。"""

    T = np.asarray(T, dtype=np.float64)
    ensure_T(T, "to_rvec_tvec")
    Rm = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(Rm)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    return rvec, t


def connected_components(cameras: Sequence[str], edges: Sequence[EdgeObs]) -> List[List[str]]:
    adj: Dict[str, List[str]] = {str(c): [] for c in cameras}
    for e in edges:
        adj[str(e.cam_i)].append(str(e.cam_j))
        adj[str(e.cam_j)].append(str(e.cam_i))

    seen: set[str] = set()
    comps: List[List[str]] = []
    for c in [str(x) for x in cameras]:
        if c in seen:
            continue
        stack = [c]
        comp: List[str] = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            for nb in adj.get(x, []):
                if nb not in seen:
                    stack.append(nb)
        comps.append(sorted(comp))

    comps.sort(key=len, reverse=True)
    return comps


def _avg_relative_transform(meas_list: Sequence[np.ndarray]) -> np.ndarray:
    """对同一对相机的多次相对位姿做一个简单平均（用于初始化）。"""

    if len(meas_list) == 0:
        raise ValueError("empty meas_list")

    rotvecs: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    for T in meas_list:
        T = np.asarray(T, dtype=np.float64)
        Rm = T[:3, :3]
        t = T[:3, 3]
        rotvecs.append(so3_log(Rm))
        ts.append(t)

    rv_mean = np.mean(np.stack(rotvecs, axis=0), axis=0)
    R_mean = so3_exp(rv_mean)
    t_mean = np.mean(np.stack(ts, axis=0), axis=0)
    return make_T(R_mean, t_mean, "avg_rel")


def initial_poses_from_edges(
    cameras: Sequence[str],
    *,
    reference: str,
    edges: Sequence[EdgeObs],
) -> Dict[str, np.ndarray]:
    """用 BFS 从 reference 出发给每个相机一个初值（不保证最优）。"""

    # 收集每条有向边的观测列表：(i,j) 表示 Ci <- Cj
    dir_meas: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for e in edges:
        dir_meas.setdefault((str(e.cam_i), str(e.cam_j)), []).append(np.asarray(e.Ci_T_Cj, dtype=np.float64))
        dir_meas.setdefault((str(e.cam_j), str(e.cam_i)), []).append(
            invert_T(np.asarray(e.Ci_T_Cj, dtype=np.float64), "inv_edge")
        )

    # 对每个方向做平均，得到一张“稀疏图”
    dir_avg: Dict[Tuple[str, str], np.ndarray] = {}
    for k, lst in dir_meas.items():
        dir_avg[k] = _avg_relative_transform(lst)

    ref = str(reference)
    X: Dict[str, np.ndarray] = {ref: np.eye(4, dtype=np.float64)}

    cams = [str(c) for c in cameras]
    q = [ref]
    while q:
        cur = q.pop(0)
        for nb in cams:
            if nb in X:
                continue
            key = (nb, cur)  # X_nb = (nb <- cur) * X_cur
            if key not in dir_avg:
                continue
            X[nb] = dir_avg[key] @ X[cur]
            q.append(nb)

    return X


def _pack_pose_graph_params(
    cameras: Sequence[str],
    *,
    reference: str,
    X_init: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    opt_cams = [str(c) for c in cameras if str(c) != str(reference)]
    x0 = np.zeros(6 * len(opt_cams), dtype=np.float64)

    for k, cam in enumerate(opt_cams):
        Ti = np.asarray(X_init.get(cam, np.eye(4, dtype=np.float64)), dtype=np.float64)
        x0[6 * k : 6 * k + 6] = se3_log(Ti)

    return x0, opt_cams


def _unpack_pose_graph_params(x: np.ndarray, *, opt_cams: Sequence[str], reference: str) -> Dict[str, np.ndarray]:
    X: Dict[str, np.ndarray] = {str(reference): np.eye(4, dtype=np.float64)}
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    for k, cam in enumerate(opt_cams):
        xi = x[6 * k : 6 * k + 6]
        X[str(cam)] = se3_exp(xi)
    return X


def _pose_graph_residuals(x: np.ndarray, *, opt_cams: Sequence[str], reference: str, edges: Sequence[EdgeObs]) -> np.ndarray:
    X = _unpack_pose_graph_params(x, opt_cams=opt_cams, reference=reference)

    res: List[np.ndarray] = []
    for e in edges:
        Xi = X[str(e.cam_i)]
        Xj = X[str(e.cam_j)]
        pred = Xi @ invert_T(Xj, "inv_Xj")
        err_T = invert_T(np.asarray(e.Ci_T_Cj, dtype=np.float64), "inv_meas") @ pred
        r6 = se3_log(err_T)
        w = float(max(1e-9, float(e.weight)))
        res.append(np.sqrt(w) * r6)

    if len(res) == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(res, axis=0)


def solve_pose_graph_extrinsics(
    *,
    cameras: Sequence[str],
    reference: str,
    edges: Sequence[EdgeObs],
    huber_f_scale: float,
    max_nfev: int = 2000,
) -> Tuple[Dict[str, np.ndarray], OptimizeResult, Dict[str, Any]]:
    """使用位姿图优化求解 X_cam_from_ref（Cam <- Ref）。

    返回：
    - X_cam_from_ref: cam -> 4x4
    - result: scipy OptimizeResult
    - edge_error: 可解释的误差统计（度/mm）
    """

    cams = [str(c) for c in cameras]
    ref = str(reference)

    X_init = initial_poses_from_edges(cams, reference=ref, edges=edges)
    if len(X_init) < len(cams):
        missing = [c for c in cams if c not in X_init]
        raise ValueError(f"初始化失败，无法从 reference 到达相机：{missing}")

    x0, opt_cams = _pack_pose_graph_params(cams, reference=ref, X_init=X_init)
    fun = lambda x: _pose_graph_residuals(x, opt_cams=opt_cams, reference=ref, edges=edges)

    res = least_squares(
        fun,
        x0,
        method="trf",
        loss="huber",
        f_scale=float(huber_f_scale),
        max_nfev=int(max_nfev),
    )

    X_opt = _unpack_pose_graph_params(res.x, opt_cams=opt_cams, reference=ref)
    edge_err = summarize_edge_errors(X=X_opt, edges=edges)
    return X_opt, res, edge_err


def _pct(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, float(q)))


def summarize_scalar(values: Sequence[float]) -> Dict[str, Any]:
    """汇总一组标量的统计信息（用于报告/诊断）。"""

    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}

    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": _pct(arr, 50),
        "p95": _pct(arr, 95),
        "max": float(np.max(arr)),
    }


def summarize_pnp_reproj_px(
    *,
    cameras: Sequence[str],
    edges: Sequence[EdgeObs],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]] | None = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """汇总 PnP 自身重投影误差（像素）。

    注意：这里统计的是每张图做 PnP 拟合后的 reproj_mean_px。
    """

    cams = [str(c) for c in cameras]
    by_cam_vals: Dict[str, List[float]] = {c: [] for c in cams}
    for _frame_key, poses in poses_by_frame.items():
        for cam, po in poses.items():
            if str(cam) not in by_cam_vals:
                continue
            by_cam_vals[str(cam)].append(float(po.reproj_mean_px))

    by_cam = {c: summarize_scalar(by_cam_vals.get(c, [])) for c in cams}

    edge_vals: List[float] = []
    edge_rows: List[Dict[str, Any]] = []
    for e in edges:
        frame_key = str(e.frame_key)
        poses = poses_by_frame.get(frame_key) or {}
        pi = poses.get(str(e.cam_i))
        pj = poses.get(str(e.cam_j))
        if pi is None or pj is None:
            continue

        p_i = (paths_by_frame.get(frame_key) or {}).get(str(e.cam_i)) if paths_by_frame else None
        p_j = (paths_by_frame.get(frame_key) or {}).get(str(e.cam_j)) if paths_by_frame else None

        edge_mean = 0.5 * (float(pi.reproj_mean_px) + float(pj.reproj_mean_px))
        edge_vals.append(float(edge_mean))
        edge_rows.append(
            {
                "pair": f"{e.cam_i}__{e.cam_j}",
                "cam_i": str(e.cam_i),
                "cam_j": str(e.cam_j),
                "frame_key": frame_key,
                "edge_mean_px": float(edge_mean),
                "reproj_mean_px": {
                    "cam_i": float(pi.reproj_mean_px),
                    "cam_j": float(pj.reproj_mean_px),
                },
                "n_tags": {
                    "cam_i": int(pi.n_tags),
                    "cam_j": int(pj.n_tags),
                },
                "image_path": {
                    "cam_i": str(p_i) if p_i is not None else None,
                    "cam_j": str(p_j) if p_j is not None else None,
                },
            }
        )

    edge_mean_px = summarize_scalar(edge_vals)
    top_k = int(max(1, top_k))
    worst_edges = sorted(edge_rows, key=lambda d: float(d.get("edge_mean_px", 0.0)), reverse=True)[:top_k]

    return {
        "units": "px",
        "by_cam": by_cam,
        "edge_mean_px": edge_mean_px,
        "worst_edges_by_edge_mean_px": worst_edges,
    }


def _select_anchor_cam(poses: Dict[str, PoseObs]) -> str:
    if len(poses) == 0:
        raise ValueError("empty poses")
    return sorted(poses.values(), key=lambda p: float(p.reproj_mean_px))[0].cam


def summarize_post_extrinsics_reproj_px(
    *,
    X_cam_from_ref: Dict[str, np.ndarray],
    reference: str,
    cameras: Sequence[str],
    intrinsics: Dict[str, CameraIntrinsics],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]] | None = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """外参解算完成后的“多相机一致性”回投影误差（像素）。"""

    cams = [str(c) for c in cameras]
    ref = str(reference)
    if ref not in X_cam_from_ref:
        raise ValueError("reference missing in X_cam_from_ref")

    per_cam_vals: Dict[str, List[float]] = {c: [] for c in cams}
    rows: List[Dict[str, Any]] = []

    for frame_key, poses in poses_by_frame.items():
        if len(poses) < 2:
            continue

        anchor = _select_anchor_cam(poses)
        if anchor not in X_cam_from_ref:
            continue
        po_anchor = poses.get(anchor)
        if po_anchor is None:
            continue

        Xi_anchor = X_cam_from_ref[anchor]
        Cref_T_B = invert_T(Xi_anchor, "inv_X_anchor") @ po_anchor.C_T_B

        for cam, po in poses.items():
            cam = str(cam)
            if cam not in X_cam_from_ref or cam not in intrinsics:
                continue
            if po.obj_pts.size == 0 or po.img_pts.size == 0:
                continue
            if po.obj_pts.shape[0] != po.img_pts.shape[0]:
                continue

            Xi = X_cam_from_ref[cam]
            Ci_T_B_pred = Xi @ Cref_T_B

            Rm = np.asarray(Ci_T_B_pred[:3, :3], dtype=np.float64)
            t = np.asarray(Ci_T_B_pred[:3, 3], dtype=np.float64).reshape(3, 1)
            rvec, _ = cv2.Rodrigues(Rm)

            K = np.asarray(intrinsics[cam].K, dtype=np.float64)
            dist = np.asarray(intrinsics[cam].dist, dtype=np.float64)

            proj, _ = cv2.projectPoints(
                np.asarray(po.obj_pts, dtype=np.float32),
                np.asarray(rvec, dtype=np.float64),
                np.asarray(t, dtype=np.float64),
                K,
                dist,
            )
            proj = proj.reshape(-1, 2).astype(np.float64)
            obs = np.asarray(po.img_pts, dtype=np.float64).reshape(-1, 2)
            per_pt = np.linalg.norm(obs - proj, axis=1)
            mean_px = float(np.mean(per_pt)) if per_pt.size else float("inf")

            per_cam_vals[cam].append(mean_px)
            p = (paths_by_frame.get(str(frame_key)) or {}).get(cam) if paths_by_frame else None
            rows.append(
                {
                    "cam": cam,
                    "frame_key": str(frame_key),
                    "anchor_cam": str(anchor),
                    "mean_px": mean_px,
                    "n_points": int(per_pt.size),
                    "image_path": str(p) if p is not None else None,
                }
            )

    by_cam = {c: summarize_scalar(per_cam_vals.get(c, [])) for c in cams}
    all_vals = [x for lst in per_cam_vals.values() for x in lst]
    overall = summarize_scalar(all_vals)

    top_k = int(max(1, top_k))
    worst = sorted(rows, key=lambda d: float(d.get("mean_px", 0.0)), reverse=True)[:top_k]

    return {
        "units": "px",
        "note": "外参优化完成后：每帧选 reproj 最小的相机做 anchor，生成板位姿并回投影计算一致性误差。",
        "overall": overall,
        "by_cam": by_cam,
        "worst": worst,
    }


def summarize_edge_errors(*, X: Dict[str, np.ndarray], edges: Sequence[EdgeObs]) -> Dict[str, Any]:
    """将边误差拆为旋转(度)与平移(单位=板模型单位，通常 mm)。"""

    rot_deg_list: List[float] = []
    trans_list: List[float] = []
    by_pair: Dict[str, Dict[str, List[float]]] = {}

    for e in edges:
        Xi = X[str(e.cam_i)]
        Xj = X[str(e.cam_j)]
        pred = Xi @ invert_T(Xj, "inv_Xj")
        err_T = invert_T(np.asarray(e.Ci_T_Cj, dtype=np.float64), "inv_meas") @ pred

        w = so3_log(err_T[:3, :3])
        rot_deg = float(np.linalg.norm(w) * 180.0 / np.pi)
        trans = float(np.linalg.norm(err_T[:3, 3]))

        rot_deg_list.append(rot_deg)
        trans_list.append(trans)

        k = f"{e.cam_i}__{e.cam_j}"
        by_pair.setdefault(k, {"rot_deg": [], "trans": []})
        by_pair[k]["rot_deg"].append(rot_deg)
        by_pair[k]["trans"].append(trans)

    rot = np.asarray(rot_deg_list, dtype=np.float64)
    tr = np.asarray(trans_list, dtype=np.float64)

    pair_out: Dict[str, Any] = {}
    for k, v in by_pair.items():
        r = np.asarray(v["rot_deg"], dtype=np.float64)
        t = np.asarray(v["trans"], dtype=np.float64)
        pair_out[k] = {
            "n": int(r.size),
            "rot_deg": {
                "median": _pct(r, 50),
                "p95": _pct(r, 95),
                "max": float(np.max(r)) if r.size else 0.0,
            },
            "trans": {
                "median": _pct(t, 50),
                "p95": _pct(t, 95),
                "max": float(np.max(t)) if t.size else 0.0,
            },
        }

    return {
        "n_edges": int(len(edges)),
        "units": {"rotation": "deg", "translation": "mm"},
        "rot_deg": {
            "mean": float(np.mean(rot)) if rot.size else 0.0,
            "median": _pct(rot, 50),
            "p95": _pct(rot, 95),
            "max": float(np.max(rot)) if rot.size else 0.0,
        },
        "trans": {
            "mean": float(np.mean(tr)) if tr.size else 0.0,
            "median": _pct(tr, 50),
            "p95": _pct(tr, 95),
            "max": float(np.max(tr)) if tr.size else 0.0,
        },
        "by_pair": pair_out,
    }


def _build_ba_frame_and_obs(
    *,
    cameras_solved: Sequence[str],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
) -> Tuple[List[str], List[Tuple[str, str, np.ndarray, np.ndarray]]]:
    cams = {str(c) for c in cameras_solved}
    frames: List[str] = []
    obs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []

    for frame_key in sorted(poses_by_frame.keys()):
        poses = poses_by_frame.get(frame_key) or {}
        poses = {str(c): p for c, p in poses.items() if str(c) in cams}
        if len(poses) < 2:
            continue

        local_obs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
        for cam, po in poses.items():
            obj = np.asarray(po.obj_pts, dtype=np.float32).reshape(-1, 3)
            img = np.asarray(po.img_pts, dtype=np.float32).reshape(-1, 2)
            if obj.size == 0 or img.size == 0:
                continue
            if obj.shape[0] != img.shape[0]:
                continue
            local_obs.append((str(frame_key), str(cam), obj, img))

        cams_in_frame = {c for (_fk, c, _o, _i) in local_obs}
        if len(cams_in_frame) < 2:
            continue

        frames.append(str(frame_key))
        obs.extend(local_obs)

    return frames, obs


def _init_board_poses_in_ref(
    *,
    frames: Sequence[str],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    X_cam_from_ref_init: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for frame_key in frames:
        poses = poses_by_frame.get(str(frame_key)) or {}
        if len(poses) == 0:
            continue
        anchor = _select_anchor_cam(poses)
        if anchor not in X_cam_from_ref_init:
            continue
        po_anchor = poses.get(anchor)
        if po_anchor is None:
            continue
        Xi_anchor = X_cam_from_ref_init[anchor]
        out[str(frame_key)] = invert_T(Xi_anchor, "inv_X_anchor_init") @ po_anchor.C_T_B
    return out


def _pack_ba_params(
    *,
    opt_cams: Sequence[str],
    X_cam_from_ref_init: Dict[str, np.ndarray],
    frames: Sequence[str],
    Cref_T_B_init: Dict[str, np.ndarray],
) -> np.ndarray:
    x = np.zeros(6 * int(len(opt_cams)) + 6 * int(len(frames)), dtype=np.float64)
    for k, cam in enumerate(opt_cams):
        T = np.asarray(X_cam_from_ref_init.get(str(cam), np.eye(4, dtype=np.float64)), dtype=np.float64)
        x[6 * k : 6 * k + 6] = se3_log(T)

    base = 6 * int(len(opt_cams))
    for i, frame_key in enumerate(frames):
        T = np.asarray(Cref_T_B_init.get(str(frame_key), np.eye(4, dtype=np.float64)), dtype=np.float64)
        x[base + 6 * i : base + 6 * i + 6] = se3_log(T)

    return x


def _unpack_ba_params(
    *,
    x: np.ndarray,
    opt_cams: Sequence[str],
    reference: str,
    frames: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    X: Dict[str, np.ndarray] = {str(reference): np.eye(4, dtype=np.float64)}
    for k, cam in enumerate(opt_cams):
        xi = x[6 * k : 6 * k + 6]
        X[str(cam)] = se3_exp(xi)

    base = 6 * int(len(opt_cams))
    Cref_T_B: Dict[str, np.ndarray] = {}
    for i, frame_key in enumerate(frames):
        xi = x[base + 6 * i : base + 6 * i + 6]
        Cref_T_B[str(frame_key)] = se3_exp(xi)

    return X, Cref_T_B


def _ba_residuals(
    x: np.ndarray,
    *,
    opt_cams: Sequence[str],
    reference: str,
    frames: Sequence[str],
    observations: Sequence[Tuple[str, str, np.ndarray, np.ndarray]],
    intrinsics: Dict[str, CameraIntrinsics],
) -> np.ndarray:
    X_cam_from_ref, Cref_T_B_by_frame = _unpack_ba_params(
        x=x, opt_cams=opt_cams, reference=reference, frames=frames
    )

    res: List[np.ndarray] = []
    for frame_key, cam, obj_pts, img_pts in observations:
        Ci = X_cam_from_ref.get(str(cam))
        Cref_T_B = Cref_T_B_by_frame.get(str(frame_key))
        intr = intrinsics.get(str(cam))
        if Ci is None or Cref_T_B is None or intr is None:
            continue

        Ci_T_B = Ci @ Cref_T_B
        rvec, tvec = to_rvec_tvec_from_T(Ci_T_B)

        proj, _ = cv2.projectPoints(
            np.asarray(obj_pts, dtype=np.float32),
            np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64),
            np.asarray(intr.K, dtype=np.float64),
            np.asarray(intr.dist, dtype=np.float64),
        )
        proj = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        obs = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
        if proj.shape != obs.shape or proj.size == 0:
            continue
        res.append((obs - proj).reshape(-1))

    if len(res) == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(res, axis=0)


def summarize_ba_reproj_px(
    *,
    X_cam_from_ref: Dict[str, np.ndarray],
    Cref_T_B_by_frame: Dict[str, np.ndarray],
    cameras: Sequence[str],
    intrinsics: Dict[str, CameraIntrinsics],
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]] | None = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    cams = [str(c) for c in cameras]
    per_cam_vals: Dict[str, List[float]] = {c: [] for c in cams}
    rows: List[Dict[str, Any]] = []

    for frame_key, poses in poses_by_frame.items():
        Cref_T_B = Cref_T_B_by_frame.get(str(frame_key))
        if Cref_T_B is None:
            continue

        for cam, po in poses.items():
            cam = str(cam)
            if cam not in X_cam_from_ref or cam not in intrinsics:
                continue
            if po.obj_pts.size == 0 or po.img_pts.size == 0:
                continue
            if po.obj_pts.shape[0] != po.img_pts.shape[0]:
                continue

            Ci_T_B = X_cam_from_ref[cam] @ Cref_T_B
            rvec, tvec = to_rvec_tvec_from_T(Ci_T_B)
            intr = intrinsics[cam]

            proj, _ = cv2.projectPoints(
                np.asarray(po.obj_pts, dtype=np.float32),
                np.asarray(rvec, dtype=np.float64),
                np.asarray(tvec, dtype=np.float64),
                np.asarray(intr.K, dtype=np.float64),
                np.asarray(intr.dist, dtype=np.float64),
            )
            proj = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
            obs = np.asarray(po.img_pts, dtype=np.float64).reshape(-1, 2)
            if proj.shape != obs.shape or proj.size == 0:
                continue
            per_pt = np.linalg.norm(obs - proj, axis=1)
            mean_px = float(np.mean(per_pt)) if per_pt.size else float("inf")

            per_cam_vals[cam].append(mean_px)
            p = (paths_by_frame.get(str(frame_key)) or {}).get(cam) if paths_by_frame else None
            rows.append(
                {
                    "cam": cam,
                    "frame_key": str(frame_key),
                    "mean_px": mean_px,
                    "n_points": int(per_pt.size),
                    "image_path": str(p) if p is not None else None,
                }
            )

    by_cam = {c: summarize_scalar(per_cam_vals.get(c, [])) for c in cams}
    all_vals = [x for lst in per_cam_vals.values() for x in lst]
    overall = summarize_scalar(all_vals)

    top_k = int(max(1, top_k))
    worst = sorted(rows, key=lambda d: float(d.get("mean_px", 0.0)), reverse=True)[:top_k]

    return {
        "units": "px",
        "note": "Bundle Adjustment：联合优化相机外参+每帧板位姿，直接最小化角点像素重投影误差。",
        "overall": overall,
        "by_cam": by_cam,
        "worst": worst,
    }


def _prune_ba_observations(
    *,
    X_cam_from_ref: Dict[str, np.ndarray],
    Cref_T_B_by_frame: Dict[str, np.ndarray],
    cameras: Sequence[str],
    intrinsics: Dict[str, CameraIntrinsics],
    observations: Sequence[Tuple[str, str, np.ndarray, np.ndarray]],
    prune_mean_px: float,
    min_keep_per_cam: int,
) -> Tuple[List[Tuple[str, str, np.ndarray, np.ndarray]], Dict[str, Any]]:
    thr = float(prune_mean_px)
    if thr <= 0.0:
        return list(observations), {"enabled": False}

    rows: List[Tuple[float, str, str, np.ndarray, np.ndarray]] = []
    for frame_key, cam, obj_pts, img_pts in observations:
        Ci = X_cam_from_ref.get(str(cam))
        Cref_T_B = Cref_T_B_by_frame.get(str(frame_key))
        intr = intrinsics.get(str(cam))
        if Ci is None or Cref_T_B is None or intr is None:
            continue

        Ci_T_B = Ci @ Cref_T_B
        rvec, tvec = to_rvec_tvec_from_T(Ci_T_B)
        proj, _ = cv2.projectPoints(
            np.asarray(obj_pts, dtype=np.float32),
            np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64),
            np.asarray(intr.K, dtype=np.float64),
            np.asarray(intr.dist, dtype=np.float64),
        )
        proj = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        obs = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
        if proj.shape != obs.shape or proj.size == 0:
            continue
        per_pt = np.linalg.norm(obs - proj, axis=1)
        mean_px = float(np.mean(per_pt)) if per_pt.size else float("inf")
        rows.append((mean_px, str(frame_key), str(cam), obj_pts, img_pts))

    keep = [(fk, cam, obj, img) for (m, fk, cam, obj, img) in rows if float(m) <= thr]

    cams = [str(c) for c in cameras]
    min_keep = int(max(0, min_keep_per_cam))
    if min_keep > 0:
        by_cam_all: Dict[str, List[Tuple[float, str, str, np.ndarray, np.ndarray]]] = {c: [] for c in cams}
        for r in rows:
            by_cam_all[r[2]].append(r)
        for c in cams:
            cur = [o for o in keep if o[1] == c]
            if len(cur) >= min_keep:
                continue
            pool = sorted(by_cam_all.get(c, []), key=lambda t: float(t[0]))
            need = min_keep - len(cur)
            for (_m, fk, cam, obj, img) in pool:
                if need <= 0:
                    break
                cand = (fk, cam, obj, img)
                if cand in keep:
                    continue
                keep.append(cand)
                need -= 1

    by_frame: Dict[str, List[Tuple[str, str, np.ndarray, np.ndarray]]] = {}
    for fk, cam, obj, img in keep:
        by_frame.setdefault(str(fk), []).append((fk, cam, obj, img))

    keep2: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    dropped_frames = 0
    for fk, lst in by_frame.items():
        cams_in = {x[1] for x in lst}
        if len(cams_in) < 2:
            dropped_frames += 1
            continue
        keep2.extend(lst)

    info = {
        "enabled": True,
        "prune_mean_px": float(thr),
        "min_keep_per_cam": int(min_keep),
        "before": int(len(observations)),
        "after": int(len(keep2)),
        "dropped_frames": int(dropped_frames),
    }
    return keep2, info


def solve_bundle_adjustment_extrinsics(
    *,
    cameras: Sequence[str],
    reference: str,
    poses_by_frame: Dict[str, Dict[str, PoseObs]],
    paths_by_frame: Dict[str, Dict[str, Path]] | None = None,
    intrinsics: Dict[str, CameraIntrinsics],
    f_scale_px: float,
    max_nfev: int,
    prune_mean_px: float,
    prune_min_keep_per_cam: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    OptimizeResult,
    Dict[str, Any],
    List[Tuple[str, str, np.ndarray, np.ndarray]],
    Dict[str, Any],
]:
    """多相机 BA：联合优化相机外参 + 每帧板位姿。

    返回：
    - X_cam_from_ref
    - Cref_T_B_by_frame
    - OptimizeResult（最终一次优化）
    - prune_info
    - obs_used（最终参与 BA 的观测集合）
    - ba_reproj_px（基于 obs_used 的像素误差统计）
    """

    cams = [str(c) for c in cameras]
    ref = str(reference)

    # 初值用 pose-graph 的 BFS 初始化（只要连通即可）
    # 为此先构建 edges（只用于初始化）
    edges_init: List[EdgeObs] = []
    for frame_key, poses in poses_by_frame.items():
        cams_in = sorted([str(c) for c in poses.keys()])
        if len(cams_in) < 2:
            continue
        for cam_i, cam_j in combinations(cams_in, 2):
            pi = poses[cam_i]
            pj = poses[cam_j]
            Ci_T_Cj = pi.C_T_B @ invert_T(pj.C_T_B, "B_T_Cj")
            err = 0.5 * (float(pi.reproj_mean_px) + float(pj.reproj_mean_px))
            w = float(1.0 / max(1e-6, err))
            edges_init.append(
                EdgeObs(
                    cam_i=str(cam_i),
                    cam_j=str(cam_j),
                    Ci_T_Cj=np.asarray(Ci_T_Cj, dtype=np.float64),
                    weight=w,
                    frame_key=str(frame_key),
                )
            )

    X_init = initial_poses_from_edges(cams, reference=ref, edges=edges_init)
    if len(X_init) < len(cams):
        missing = [c for c in cams if c not in X_init]
        raise ValueError(f"初始化失败，无法从 reference 到达相机：{missing}")

    frames, obs = _build_ba_frame_and_obs(cameras_solved=cams, poses_by_frame=poses_by_frame)
    if len(frames) == 0 or len(obs) == 0:
        raise ValueError("BA 没有可用帧/观测（需要至少一帧包含 >=2 相机有效观测）。")

    Cref_T_B_init = _init_board_poses_in_ref(frames=frames, poses_by_frame=poses_by_frame, X_cam_from_ref_init=X_init)

    opt_cams = [c for c in cams if c != ref]
    x0 = _pack_ba_params(
        opt_cams=opt_cams,
        X_cam_from_ref_init=X_init,
        frames=frames,
        Cref_T_B_init=Cref_T_B_init,
    )

    fun = lambda x: _ba_residuals(
        x,
        opt_cams=opt_cams,
        reference=ref,
        frames=frames,
        observations=obs,
        intrinsics=intrinsics,
    )

    res1 = least_squares(
        fun,
        x0,
        method="trf",
        loss="huber",
        f_scale=float(f_scale_px),
        max_nfev=int(max_nfev),
    )

    X1, C1 = _unpack_ba_params(x=res1.x, opt_cams=opt_cams, reference=ref, frames=frames)
    obs2, prune_info = _prune_ba_observations(
        X_cam_from_ref=X1,
        Cref_T_B_by_frame=C1,
        cameras=cams,
        intrinsics=intrinsics,
        observations=obs,
        prune_mean_px=float(prune_mean_px),
        min_keep_per_cam=int(prune_min_keep_per_cam),
    )

    # 兼容旧脚本的报告字段：明确标记本次是否真的发生了剔除。
    if bool(prune_info.get("enabled", False)):
        before = int(prune_info.get("before", len(obs)))
        after = int(prune_info.get("after", len(obs2)))
        prune_info["changed"] = bool(after < before)
    else:
        prune_info["changed"] = False

    obs_used = list(obs)
    res_final = res1
    X_final = X1
    C_final = C1

    if bool(prune_info.get("enabled")) and int(prune_info.get("after", 0)) < int(prune_info.get("before", 0)):
        fun2 = lambda x: _ba_residuals(
            x,
            opt_cams=opt_cams,
            reference=ref,
            frames=frames,
            observations=obs2,
            intrinsics=intrinsics,
        )
        res2 = least_squares(
            fun2,
            res1.x,
            method="trf",
            loss="huber",
            f_scale=float(f_scale_px),
            max_nfev=int(max_nfev),
        )
        X2, C2 = _unpack_ba_params(x=res2.x, opt_cams=opt_cams, reference=ref, frames=frames)

        res_final = res2
        X_final = X2
        C_final = C2
        obs_used = list(obs2)

    # 误差统计：基于最终参与 BA 的观测集合过滤 poses_by_frame
    keep: set[tuple[str, str]] = {(str(fk), str(cam)) for fk, cam, _o, _i in obs_used}
    poses_used: Dict[str, Dict[str, PoseObs]] = {}
    paths_used: Dict[str, Dict[str, Path]] = {}
    for fk, poses in poses_by_frame.items():
        fk_s = str(fk)
        new_poses = {str(cam): po for cam, po in poses.items() if (fk_s, str(cam)) in keep}
        if len(new_poses) > 0:
            poses_used[fk_s] = new_poses
            if paths_by_frame is not None:
                new_paths = {
                    str(cam): p
                    for cam, p in (paths_by_frame.get(fk_s) or {}).items()
                    if (fk_s, str(cam)) in keep
                }
                if len(new_paths) > 0:
                    paths_used[fk_s] = new_paths

    ba_reproj = summarize_ba_reproj_px(
        X_cam_from_ref=X_final,
        Cref_T_B_by_frame=C_final,
        cameras=cams,
        intrinsics=intrinsics,
        poses_by_frame=poses_used,
        paths_by_frame=paths_used if paths_by_frame is not None else None,
        top_k=10,
    )

    return X_final, C_final, res_final, prune_info, obs_used, ba_reproj
