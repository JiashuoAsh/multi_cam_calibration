#!/usr/bin/env python3
"""Step 4 (Multi-Cam): 多相机外参标定（位姿图优化 / Pose Graph）

核心思想：
- 对每个时间戳(同名帧)的每个相机，用 AprilTag 标定板做 PnP，得到该帧的 C_T_B（Board->Camera）。
- 如果同一帧里相机 i 和 j 都成功 PnP，则得到一条相机间相对位姿观测：

    Z_ij = Ci_T_Cj = (Ci_T_B) * inv(Cj_T_B)

- 将所有 Z_ij 组成位姿图（节点=相机，边=相对位姿观测），对所有相机位姿 X_i 做最小二乘优化：

    Z_ij ≈ X_i * inv(X_j)

  其中 X_ref 固定为单位阵，ref 为参考相机。

输入约定（推荐）：
- 图像：images/<source>/<cam_name>/*.png|*.jpg
  - <source> 可用 images/filtered（默认）或 images/raw
  - 不要求所有相机都能看到板：只要存在一些帧产生边，且整体图连通即可。
  - “同步”的含义：不同相机同一时刻的帧文件名 stem 相同。
    例如：
      images/filtered/cam0/frame_000120.png
      images/filtered/cam1/frame_000120.png
      images/filtered/cam2/frame_000120.png

- 内参：results/<cam_name>_intrinsics.json
    - 本仓库双目默认命名就是 left/right，因此 cam_name=left/right 时可直接复用 Step3 输出。

输出：
- results/multi_camera_extrinsics.json
  - 每个相机相对 reference 的 4x4 齐次变换矩阵（Cam_i <- Cam_ref）
- results/multi_camera_pose_graph_report.json
  - 图连通性、每对相机边数量、优化收敛信息等报告

使用示例：
- 直接扫描 images/filtered 下的相机目录：
    python step4_multi_extrinsic_pose_graph.py

- 手动指定相机列表与参考相机：
    python step4_multi_extrinsic_pose_graph.py --cameras cam0 cam1 cam2 cam3 --reference cam0

常见失败：
- 图不连通：相机之间没有足够“同帧共同看到板”的边，无法把所有相机标到同一参考系。
- 内参缺失：没有对应的 results/<cam>_intrinsics.json。
- 标签太少：min_tags 设太大/光照太差导致 PnP 失败，边数量不足。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from utils import (
    create_apriltag_board,
    create_detector_params,
    create_opencv_aruco_board,
    detect_apriltag_corners,
    get_aruco_dict,
    get_detection_auto_roi,
    get_detection_settings,
    load_config,
)


VERBOSE: bool = True


def _vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


def _ensure_T(T: np.ndarray, name: str) -> None:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: 期望 (4,4)，得到 {T.shape}")
    if float(np.linalg.norm(T[3, :] - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-8:
        raise ValueError(f"{name}: 底行必须是 [0 0 0 1]，当前={T[3, :]}")


def _make_T(Rm: np.ndarray, t: np.ndarray, name: str) -> np.ndarray:
    Rm = np.asarray(Rm, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t
    _ensure_T(T, name)
    return T


def _inv_T(T: np.ndarray, name: str) -> np.ndarray:
    """解析求逆（比 np.linalg.inv 更稳更快），并保留自检。"""
    T = np.asarray(T, dtype=np.float64)
    _ensure_T(T, name + ".in")
    Rm = T[:3, :3]
    t = T[:3, 3]
    Rm_inv = Rm.T
    t_inv = -Rm_inv @ t
    T_inv = _make_T(Rm_inv, t_inv, name)
    return T_inv


def _skew(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array(
        [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=np.float64
    )


def _so3_exp(w: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(w, dtype=np.float64).reshape(3)).as_matrix()


def _so3_log(Rm: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(Rm, dtype=np.float64)).as_rotvec()


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    """se(3) -> SE(3)

    xi = [w(3), v(3)]
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    w = xi[:3]
    v = xi[3:]

    theta = float(np.linalg.norm(w))
    Rm = _so3_exp(w)

    W = _skew(w)
    I = np.eye(3, dtype=np.float64)

    if theta < 1e-8:
        V = I + 0.5 * W
    else:
        A = float(np.sin(theta) / theta)
        B = float((1.0 - np.cos(theta)) / (theta * theta))
        C = float((theta - np.sin(theta)) / (theta**3))
        V = I + B * W + C * (W @ W)

    t = V @ v
    return _make_T(Rm, t, "se3_exp")


def _se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) -> se(3) (6D)

    返回 xi = [w(3), v(3)]
    """
    T = np.asarray(T, dtype=np.float64)
    _ensure_T(T, "se3_log")

    Rm = T[:3, :3]
    t = T[:3, 3]

    w = _so3_log(Rm)
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    I = np.eye(3, dtype=np.float64)

    if theta < 1e-8:
        V_inv = I - 0.5 * W
    else:
        A = float(np.sin(theta) / theta)
        B = float((1.0 - np.cos(theta)) / (theta * theta))
        # 经典公式：V^{-1} = I - 0.5 W + (1/theta^2) * (1 - A/(2B)) * W^2
        coef = float((1.0 / (theta * theta)) * (1.0 - A / (2.0 * B)))
        V_inv = I - 0.5 * W + coef * (W @ W)

    v = V_inv @ t
    xi = np.zeros(6, dtype=np.float64)
    xi[:3] = w
    xi[3:] = v
    return xi


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


@dataclass(frozen=True)
class EdgeObs:
    """位姿图边观测：Ci <- Cj"""

    cam_i: str
    cam_j: str
    Ci_T_Cj: np.ndarray
    weight: float
    frame_key: str


def _load_intrinsics(cam: str, intrinsics_dir: Path) -> CameraIntrinsics:
    """加载单个相机内参。

    约定：results/<cam>_intrinsics.json。
    - 当 cam=left/right 时，正好兼容本仓库 Step3 的输出：left_intrinsics.json/right_intrinsics.json。
    """

    path = intrinsics_dir / f"{cam}_intrinsics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到相机内参文件：{path}。\n"
            "请先为每个相机生成内参结果（schema 与 step3 输出一致：camera_matrix/dist_coeffs）。"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return CameraIntrinsics(name=cam, K=K, dist=dist)


def _list_cameras(image_root: Path, cameras_arg: Optional[Sequence[str]]) -> List[str]:
    if cameras_arg:
        return [str(c) for c in cameras_arg]

    # 自动扫描子目录作为相机名（排除 __pycache__ 等）
    cams: List[str] = []
    for p in sorted(image_root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith(".") or name.startswith("__"):
            continue
        cams.append(name)

    if len(cams) == 0:
        raise FileNotFoundError(
            f"未在 {image_root} 下找到相机子目录。\n"
            "建议目录结构：images/filtered/cam0/*.png, images/filtered/cam1/*.png ..."
        )

    return cams


def _build_frame_map(cam_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        for p in sorted(cam_dir.glob(ext)):
            out[p.stem] = p
    return out


def _collect_correspondences(
    *,
    corners: List[np.ndarray],
    ids: np.ndarray,
    obj_points: np.ndarray,
    tag_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """根据检测到的 tags，组装 solvePnP 需要的 (object_points, image_points)。"""
    ids_flat = ids.flatten()
    object_points = []
    image_points = []

    used_tags = 0
    for i, tag_id in enumerate(ids_flat):
        tag_id_int = int(tag_id)
        if tag_id_int not in tag_ids:
            continue
        idx = tag_ids.index(tag_id_int)
        object_points.append(obj_points[idx])
        image_points.append(corners[i].reshape(-1, 2))
        used_tags += 1

    if used_tags == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), 0

    obj = np.vstack(object_points).astype(np.float32)
    img = np.vstack(image_points).astype(np.float32)
    return obj, img, used_tags


def _estimate_board_pose_single_image(
    *,
    img_path: Path,
    intr: CameraIntrinsics,
    aruco_dict,
    detector_params,
    obj_points: np.ndarray,
    tag_ids: List[int],
    use_multiscale: bool,
    opencv_refine: bool,
    board,
    min_tags: int,
    auto_roi_cfg: Dict[str, Any],
) -> Optional[PoseObs]:
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids = detect_apriltag_corners(
        gray,
        aruco_dict,
        detector_params,
        use_multiscale=use_multiscale,
        opencv_refine=opencv_refine,
        board=board,
        camera_matrix=intr.K,
        dist_coeffs=intr.dist,
        auto_roi=bool(auto_roi_cfg.get("enabled", False)),
        auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
        auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
        auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
    )

    if ids is None or corners is None:
        return None

    obj_pts, img_pts, used_tags = _collect_correspondences(
        corners=list(corners),
        ids=np.asarray(ids),
        obj_points=obj_points,
        tag_ids=tag_ids,
    )

    if used_tags < int(min_tags):
        return None

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        intr.K,
        intr.dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    Rm, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    C_T_B = _make_T(Rm, t, "C_T_B")

    # reprojection error（像素）：用于粗过滤与边权重
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, intr.K, intr.dist)
    proj = proj.reshape(-1, 2)
    det = img_pts.reshape(-1, 2).astype(np.float64)
    per_pt = np.linalg.norm(det - proj, axis=1)
    reproj_mean = float(np.mean(per_pt))

    return PoseObs(
        cam=intr.name,
        frame_key=img_path.stem,
        C_T_B=C_T_B,
        n_tags=int(used_tags),
        reproj_mean_px=float(reproj_mean),
    )


def _build_pose_observations(
    *,
    image_root: Path,
    cameras: Sequence[str],
    intrinsics: Dict[str, CameraIntrinsics],
    config: dict,
    max_frames: int,
    min_tags: int,
) -> Tuple[Dict[str, Dict[str, PoseObs]], Dict[str, Dict[str, Path]]]:
    """返回：
    - poses_by_frame[frame_key][cam] = PoseObs
    - paths_by_frame[frame_key][cam] = Path
    """

    use_multiscale, opencv_refine = get_detection_settings(config)
    auto_roi_cfg = get_detection_auto_roi(config)
    detector_params = create_detector_params(config)

    obj_points_mm, tag_ids = create_apriltag_board(config)
    aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])
    board = create_opencv_aruco_board(obj_points_mm, tag_ids, aruco_dict)

    # 每个相机：frame_key -> path
    frame_maps: Dict[str, Dict[str, Path]] = {}
    for cam in cameras:
        cam_dir = image_root / cam
        if not cam_dir.exists():
            raise FileNotFoundError(f"未找到相机目录：{cam_dir}")
        frame_maps[cam] = _build_frame_map(cam_dir)

    # 所有 frame_key 的并集（只要某个相机有该帧，就会尝试；边构建时会自然要求两相机都有 pose）
    all_keys: List[str] = sorted({k for m in frame_maps.values() for k in m.keys()})
    if max_frames and max_frames > 0:
        all_keys = all_keys[: int(max_frames)]

    poses_by_frame: Dict[str, Dict[str, PoseObs]] = {}
    paths_by_frame: Dict[str, Dict[str, Path]] = {}

    for idx, frame_key in enumerate(all_keys):
        if (idx % 50) == 0:
            _vprint(f"处理帧 {idx+1}/{len(all_keys)}: {frame_key}")

        poses_this: Dict[str, PoseObs] = {}
        paths_this: Dict[str, Path] = {}

        for cam in cameras:
            p = frame_maps[cam].get(frame_key)
            if p is None:
                continue
            intr = intrinsics[cam]

            obs = _estimate_board_pose_single_image(
                img_path=p,
                intr=intr,
                aruco_dict=aruco_dict,
                detector_params=detector_params,
                obj_points=obj_points_mm,
                tag_ids=tag_ids,
                use_multiscale=use_multiscale,
                opencv_refine=opencv_refine,
                board=board,
                min_tags=int(min_tags),
                auto_roi_cfg=auto_roi_cfg,
            )
            if obs is None:
                continue
            poses_this[cam] = obs
            paths_this[cam] = p

        if len(poses_this) > 0:
            poses_by_frame[frame_key] = poses_this
            paths_by_frame[frame_key] = paths_this

    return poses_by_frame, paths_by_frame


def _build_edges_from_poses(poses_by_frame: Dict[str, Dict[str, PoseObs]]) -> List[EdgeObs]:
    edges: List[EdgeObs] = []

    for frame_key, poses in poses_by_frame.items():
        cams = sorted(list(poses.keys()))
        if len(cams) < 2:
            continue

        for cam_i, cam_j in combinations(cams, 2):
            pi = poses[cam_i]
            pj = poses[cam_j]

            Ci_T_B = pi.C_T_B
            Cj_T_B = pj.C_T_B
            Ci_T_Cj = Ci_T_B @ _inv_T(Cj_T_B, "B_T_Cj")

            # 简单权重：用两相机 reprojection error 的倒数
            # 目的：让“角点更准的观测”在优化里更有话语权
            err = 0.5 * (float(pi.reproj_mean_px) + float(pj.reproj_mean_px))
            w = float(1.0 / max(1e-6, err))

            edges.append(
                EdgeObs(
                    cam_i=cam_i,
                    cam_j=cam_j,
                    Ci_T_Cj=Ci_T_Cj,
                    weight=w,
                    frame_key=frame_key,
                )
            )

    return edges


def _connected_components(cameras: Sequence[str], edges: Sequence[EdgeObs]) -> List[List[str]]:
    adj: Dict[str, List[str]] = {c: [] for c in cameras}
    for e in edges:
        adj[e.cam_i].append(e.cam_j)
        adj[e.cam_j].append(e.cam_i)

    seen = set()
    comps: List[List[str]] = []
    for c in cameras:
        if c in seen:
            continue
        stack = [c]
        comp = []
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

    rotvecs = []
    ts = []
    for T in meas_list:
        T = np.asarray(T, dtype=np.float64)
        Rm = T[:3, :3]
        t = T[:3, 3]
        rotvecs.append(_so3_log(Rm))
        ts.append(t)

    rv_mean = np.mean(np.stack(rotvecs, axis=0), axis=0)
    R_mean = _so3_exp(rv_mean)
    t_mean = np.mean(np.stack(ts, axis=0), axis=0)
    return _make_T(R_mean, t_mean, "avg_rel")


def _initial_poses_from_edges(cameras: Sequence[str], reference: str, edges: Sequence[EdgeObs]) -> Dict[str, np.ndarray]:
    """用 BFS 从 reference 出发给每个相机一个初值（不保证最优，但通常够用）。"""

    # 收集每条有向边的观测列表：(i,j) 表示 Ci <- Cj
    dir_meas: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for e in edges:
        dir_meas.setdefault((e.cam_i, e.cam_j), []).append(e.Ci_T_Cj)
        dir_meas.setdefault((e.cam_j, e.cam_i), []).append(_inv_T(e.Ci_T_Cj, "inv_edge"))

    # 对每个方向做平均，得到一张“稀疏图”
    dir_avg: Dict[Tuple[str, str], np.ndarray] = {}
    for k, lst in dir_meas.items():
        dir_avg[k] = _avg_relative_transform(lst)

    X: Dict[str, np.ndarray] = {reference: np.eye(4, dtype=np.float64)}

    # BFS
    q = [reference]
    while q:
        cur = q.pop(0)
        for nb in cameras:
            if nb in X:
                continue
            key = (nb, cur)  # X_nb = (nb <- cur) * X_cur
            if key not in dir_avg:
                continue
            X[nb] = dir_avg[key] @ X[cur]
            q.append(nb)

    return X


def _pack_params(cameras: Sequence[str], reference: str, X_init: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    opt_cams = [c for c in cameras if c != reference]
    x0 = np.zeros(6 * len(opt_cams), dtype=np.float64)

    for k, cam in enumerate(opt_cams):
        Ti = X_init.get(cam, np.eye(4, dtype=np.float64))
        xi = _se3_log(Ti)
        x0[6 * k : 6 * k + 6] = xi

    return x0, opt_cams


def _unpack_params(x: np.ndarray, opt_cams: Sequence[str], reference: str) -> Dict[str, np.ndarray]:
    X: Dict[str, np.ndarray] = {reference: np.eye(4, dtype=np.float64)}
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    for k, cam in enumerate(opt_cams):
        xi = x[6 * k : 6 * k + 6]
        X[cam] = _se3_exp(xi)
    return X


def _residuals_pose_graph(x: np.ndarray, *, opt_cams: Sequence[str], reference: str, edges: Sequence[EdgeObs]) -> np.ndarray:
    X = _unpack_params(x, opt_cams=opt_cams, reference=reference)

    res = []
    for e in edges:
        Xi = X[e.cam_i]
        Xj = X[e.cam_j]
        pred = Xi @ _inv_T(Xj, "inv_Xj")
        err_T = _inv_T(e.Ci_T_Cj, "inv_meas") @ pred
        r6 = _se3_log(err_T)
        w = float(max(1e-9, e.weight))
        res.append(np.sqrt(w) * r6)

    if len(res) == 0:
        return np.zeros((0,), dtype=np.float64)

    return np.concatenate(res, axis=0)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Step4 (Multi-Cam): AprilTag 多相机外参（位姿图优化）")
    parser.add_argument(
        "--config",
        type=str,
        default="config/apriltag_config.json",
        help="配置文件路径（默认 config/apriltag_config.json）",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="images/filtered",
        help="图像根目录（其下每个子目录代表一个相机）",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="相机名列表（不填则自动扫描 image_root 下的子目录）",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="",
        help="参考相机名（不填则默认取 cameras[0]）",
    )
    parser.add_argument(
        "--intrinsics_dir",
        type=str,
        default="results",
        help="内参文件目录（默认 results）",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="最多处理多少个 frame_key（0=不限制）",
    )
    parser.add_argument(
        "--min_tags",
        type=int,
        default=0,
        help="每张图最少 tag 数（0=使用 config.calibration_settings.min_tags_for_pose）",
    )
    parser.add_argument(
        "--huber",
        type=float,
        default=1.0,
        help="Huber loss 的 f_scale（越小越鲁棒，默认 1.0）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4 (Multi-Cam): 多相机外参标定（位姿图优化）")
    print("=" * 60)

    # 读取配置（用于 detection/min_tags，也用于可选的 image_dataset 自动输入）
    config = load_config(str(args.config))
    ds = config.get("image_dataset", {}) if isinstance(config, dict) else {}
    use_dataset = bool(ds.get("enabled", False)) if isinstance(ds, dict) else False

    # 当启用 image_dataset 且用户未显式指定 image_root 时，默认改用 filtered_root
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

    # 当启用 image_dataset 且用户未显式指定 cameras 时，优先用 config.cameras
    cameras_arg = args.cameras
    if use_dataset and cameras_arg is None:
        cams_cfg = ds.get("cameras", {}) if isinstance(ds, dict) else {}
        if isinstance(cams_cfg, dict) and len(cams_cfg) > 0:
            cameras_arg = list(cams_cfg.keys())
        elif isinstance(cams_cfg, list) and len(cams_cfg) > 0:
            cameras_arg = [str(x) for x in cams_cfg]

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

    # 加载内参
    intrinsics: Dict[str, CameraIntrinsics] = {}
    for cam in cameras:
        intrinsics[cam] = _load_intrinsics(cam, intr_dir)

    # 估计每帧每相机的 C_T_B
    poses_by_frame, _paths_by_frame = _build_pose_observations(
        image_root=image_root,
        cameras=cameras,
        intrinsics=intrinsics,
        config=config,
        max_frames=int(args.max_frames),
        min_tags=int(min_tags),
    )

    print(f"PnP 有效帧数: {len(poses_by_frame)}")

    edges = _build_edges_from_poses(poses_by_frame)
    print(f"位姿图边数量: {len(edges)}")

    if len(edges) == 0:
        print("错误: 没有构建出任何边（同帧共同看到板的相机对为 0）。")
        print("建议：增加采集姿态/光照；降低 min_tags；确保相机帧文件名同步。")
        return 1

    comps = _connected_components(cameras, edges)
    print(f"图连通分量数量: {len(comps)}")
    for i, comp in enumerate(comps, 1):
        print(f"  component#{i}: {comp}")

    if reference not in comps[0]:
        print("错误: 参考相机不在最大连通分量中，这通常表示数据命名/同步有问题。")
        return 1

    if len(comps[0]) < len(cameras):
        print("\n⚠ 警告: 位姿图不连通，只能求出与参考相机同一连通分量中的相机外参。")
        print(f"  可解相机: {comps[0]}")

    cameras_solved = [c for c in cameras if c in comps[0]]
    edges_solved = [e for e in edges if (e.cam_i in cameras_solved and e.cam_j in cameras_solved)]

    # 初值
    X_init = _initial_poses_from_edges(cameras_solved, reference=reference, edges=edges_solved)
    if len(X_init) < len(cameras_solved):
        missing = [c for c in cameras_solved if c not in X_init]
        print(f"错误: 初始化失败，无法从 reference 到达相机：{missing}")
        return 1

    x0, opt_cams = _pack_params(cameras_solved, reference=reference, X_init=X_init)

    # 优化
    print("\n开始位姿图优化...")
    fun = lambda x: _residuals_pose_graph(x, opt_cams=opt_cams, reference=reference, edges=edges_solved)
    r0 = fun(x0)
    print(f"初始残差维度: {r0.size}")
    print(f"初始 RMS(6D): {float(np.sqrt(np.mean(r0 * r0))):.6f}")

    res = least_squares(
        fun,
        x0,
        method="trf",
        loss="huber",
        f_scale=float(args.huber),
        max_nfev=200,
    )

    x_opt = res.x
    r1 = fun(x_opt)
    print(f"优化完成: success={bool(res.success)}, nfev={int(res.nfev)}")
    print(f"最终 RMS(6D): {float(np.sqrt(np.mean(r1 * r1))):.6f}")

    X_opt = _unpack_params(x_opt, opt_cams=opt_cams, reference=reference)

    # 输出
    T_cam_from_ref: Dict[str, Dict[str, Any]] = {}
    for cam in cameras_solved:
        T = X_opt[cam]
        T_cam_from_ref[cam] = {
            "T": T.tolist(),
            "R": T[:3, :3].tolist(),
            "t": T[:3, 3].tolist(),
        }

    out_extr: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "reference": reference,
        "cameras": cameras,
        "cameras_solved": cameras_solved,
        "note": "T_cam_from_ref 表示 Cam_i <- Cam_ref（与本工程 Cr_T_Cl 命名一致：目标在前，源在后）",
        "T_cam_from_ref": T_cam_from_ref,
        "solver": {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(res.nfev),
            "cost": float(res.cost),
            "huber_f_scale": float(args.huber),
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
        "solver": {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(res.nfev),
            "cost": float(res.cost),
            "rms_final": float(np.sqrt(np.mean(r1 * r1))),
        },
    }

    _save_json(Path("results/multi_camera_extrinsics.json"), out_extr)
    _save_json(Path("results/multi_camera_pose_graph_report.json"), report)

    print("\n✓ 已保存:")
    print("  - results/multi_camera_extrinsics.json")
    print("  - results/multi_camera_pose_graph_report.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
