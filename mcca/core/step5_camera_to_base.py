"""Step5：相机 -> 底盘外参（纯计算层）。

本模块沉淀 Step5b 的核心数学逻辑，避免把“坐标系推导/矩阵运算/传播规则”散落在 CLI 脚本里。

设计约束：
- 依赖方向：entry/adapters -> core；本模块不得依赖 argparse、文件系统、缓存、并行扫描等入口/IO 逻辑。
- 变换约定：齐次变换 `A_T_B` 表示 **B -> A**。

常用符号：
- Base 坐标系：B
- 标定板坐标系：T
- 相机坐标系：C

核心链路：
1) 由配置构造 `B_T_T`（T -> B）。
2) 由 Step5 图像的 PnP 得到 `C_T_T`（T -> C）。
3) 得到 `B_T_C = B_T_T @ inv(C_T_T)`（C -> B）。
4) 若 Step4 可用：用相机间外参把已知 `B_T_C` 传播到其它相机。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from mcca.core.extrinsics_graph import ExtrinsicsGraph, load_extrinsics_graph, propagate_B_T_C
from mcca.core.rigid import ensure_T, invert_T, make_T


@dataclass(frozen=True)
class MeanPoseStats:
    """PnP 位姿平均的稳定性统计（用于 verbose 打印或报告）。"""

    t_std_norm_m: float
    r_std_norm_rad: float


def euler_xyz_deg_to_R(*, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """欧拉角转旋转矩阵（XYZ 内旋，degrees=True）。

    说明：
        该定义与仓库文档保持一致：
        `Rotation.from_euler("XYZ", [roll, pitch, yaw], degrees=True)`。

    Returns:
        3x3 旋转矩阵 `R_B_T`，用于构造 `B_T_T`。
    """

    Rm = Rotation.from_euler("XYZ", [float(roll), float(pitch), float(yaw)], degrees=True).as_matrix()
    return np.asarray(Rm, dtype=np.float64)


def build_B_T_T_from_config(
    *,
    transform_cfg: Mapping[str, Any],
    board_cfg: Mapping[str, Any],
) -> np.ndarray:
    """由 config 构造 `B_T_T`（T -> B）。

    支持 translation_reference：
    - tag0_center（默认）：translation 直接表示 Tag0 中心在 B 中的位置
    - 其它值：translation 表示某个“参考点”在 B 中的位置，需要用 reference_point_in_T 折算到 Tag0

    Args:
        transform_cfg: config.board_to_base_transform
        board_cfg: config.apriltag_board

    Returns:
        4x4 齐次矩阵 `B_T_T`。
    """

    rotation_euler = transform_cfg["rotation_euler_deg"]
    if not isinstance(rotation_euler, (list, tuple)) or len(rotation_euler) != 3:
        raise ValueError("board_to_base_transform.rotation_euler_deg 必须是长度为3的列表")

    R_B_T = euler_xyz_deg_to_R(roll=rotation_euler[0], pitch=rotation_euler[1], yaw=rotation_euler[2])

    translation_input_B = np.asarray(transform_cfg["translation"], dtype=np.float64).reshape(3)

    translation_ref = str(transform_cfg.get("translation_reference", "tag0_center"))
    ref_point_cfg = transform_cfg.get("translation_reference_point_in_T_m", None)

    # 默认网格中心（tag 中心点的几何中心），用于“未配置 ref_point”时的保守回退。
    tag_pitch_m = (float(board_cfg["tag_size"]) + float(board_cfg["tag_spacing"])) / 1000.0
    default_grid_center_T = np.asarray(
        [
            (int(board_cfg["tags_x"]) - 1) * tag_pitch_m / 2.0,
            (int(board_cfg["tags_y"]) - 1) * tag_pitch_m / 2.0,
            0.0,
        ],
        dtype=np.float64,
    )

    if translation_ref == "tag0_center":
        translation_B = translation_input_B
    else:
        if ref_point_cfg is None:
            ref_point_T = default_grid_center_T
        else:
            ref_point_T = np.asarray(ref_point_cfg, dtype=np.float64).reshape(-1)
            if ref_point_T.size != 3:
                raise ValueError("translation_reference_point_in_T_m 期望长度为3的[x,y,z]（单位米）")

        # ref_B = R_B_T * ref_T + origin_B
        # origin_B(Tag0) = ref_B - R_B_T * ref_T
        translation_B = translation_input_B - (R_B_T @ ref_point_T.reshape(3))

    B_T_T = make_T(R_B_T, translation_B, "B_T_T")
    ensure_T(B_T_T, "B_T_T")
    return B_T_T


def mean_C_T_T_from_pnp(
    *,
    rvecs: Iterable[np.ndarray],
    tvecs: Iterable[np.ndarray],
) -> Tuple[np.ndarray, MeanPoseStats]:
    """对多帧 PnP 的 `rvec/tvec` 做简单平均，输出 `C_T_T`（T -> C）。

    说明：
        - 旋转：先把 rotvec 转成旋转矩阵，逐元素平均后再 SVD 正交化。
        - 平移：直接对 tvec 求均值。

    Args:
        rvecs: 每帧 rvec（3x1 或 3，rotvec，弧度）。
        tvecs: 每帧 tvec（3x1 或 3，单位米）。

    Returns:
        (C_T_T, stats)
    """

    r_list = [np.asarray(r, dtype=np.float64).reshape(3) for r in rvecs]
    t_list = [np.asarray(t, dtype=np.float64).reshape(3) for t in tvecs]

    if len(r_list) == 0 or len(t_list) == 0:
        raise ValueError("没有有效位姿，无法计算平均")
    if len(r_list) != len(t_list):
        raise ValueError("rvecs/tvecs 数量不一致")

    if len(r_list) > 1:
        t_std_norm = float(np.linalg.norm(np.std(np.stack(t_list, axis=0), axis=0)))
        r_std_norm = float(np.linalg.norm(np.std(np.stack(r_list, axis=0), axis=0)))
    else:
        t_std_norm = 0.0
        r_std_norm = 0.0

    R_mats = [Rotation.from_rotvec(r).as_matrix() for r in r_list]
    R_mean = np.mean(np.stack(R_mats, axis=0), axis=0)

    # SVD 正交化，确保是有效旋转。
    U, _, Vt = np.linalg.svd(R_mean)
    R_ortho = U @ Vt

    # 避免反射(det=-1)
    if float(np.linalg.det(R_ortho)) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R_ortho = U @ Vt

    t_mean = np.mean(np.stack(t_list, axis=0), axis=0)

    C_T_T = make_T(np.asarray(R_ortho, dtype=np.float64), np.asarray(t_mean, dtype=np.float64), "C_T_T")
    ensure_T(C_T_T, "C_T_T")

    return C_T_T, MeanPoseStats(t_std_norm_m=t_std_norm, r_std_norm_rad=r_std_norm)


def compute_B_T_C(*, B_T_T: np.ndarray, C_T_T: np.ndarray) -> np.ndarray:
    """由 `B_T_T` 与 `C_T_T` 计算 `B_T_C`（C -> B）。"""

    B_T_T = np.asarray(B_T_T, dtype=np.float64)
    C_T_T = np.asarray(C_T_T, dtype=np.float64)
    ensure_T(B_T_T, "B_T_T")
    ensure_T(C_T_T, "C_T_T")

    T_T_C = invert_T(C_T_T, "T_T_C")
    B_T_C = B_T_T @ T_T_C
    ensure_T(B_T_C, "B_T_C")
    return B_T_C


@dataclass(frozen=True)
class CameraToBaseResult:
    B_T_C: Dict[str, np.ndarray]
    methods: Dict[str, str]
    propagation: Dict[str, Any]
    B_T_T: np.ndarray
    graph: Optional[ExtrinsicsGraph]


def solve_camera_to_base(
    *,
    transform_cfg: Mapping[str, Any],
    board_cfg: Mapping[str, Any],
    C_T_T_by_cam: Mapping[str, np.ndarray],
    results_dir: Path = Path("results"),
) -> CameraToBaseResult:
    """根据 Step5 观测（每相机一份 C_T_T）求解所有相机的 `B_T_C`。

    策略：
    1) 对有 C_T_T 的相机做 direct_pnp。
    2) 若 Step4 外参存在：从任意一个 direct 相机作为 anchor，把结果传播到其它相机。

    Args:
        transform_cfg: config.board_to_base_transform
        board_cfg: config.apriltag_board
        C_T_T_by_cam: {cam: C_T_T}，其中 C_T_T 表示 T->C
        results_dir: 用于加载 Step4 结果（默认 results）

    Returns:
        CameraToBaseResult
    """

    if not C_T_T_by_cam:
        raise ValueError("C_T_T_by_cam 为空：没有任何相机得到有效位姿")

    B_T_T = build_B_T_T_from_config(transform_cfg=transform_cfg, board_cfg=board_cfg)

    B_T_C: Dict[str, np.ndarray] = {}
    methods: Dict[str, str] = {}

    for cam, C_T_T in C_T_T_by_cam.items():
        B_T_C[str(cam)] = compute_B_T_C(B_T_T=B_T_T, C_T_T=C_T_T)
        methods[str(cam)] = "direct_pnp"

    graph = load_extrinsics_graph(results_dir=Path(results_dir))
    propagated: List[str] = []

    if graph is not None and len(graph.T_cam_from_ref) > 0 and len(B_T_C) > 0:
        anchor_cam = next(iter(B_T_C.keys()))
        propagated_all = propagate_B_T_C(B_T_C_anchor=B_T_C[anchor_cam], anchor_cam=anchor_cam, graph=graph)

        for cam, T in propagated_all.items():
            if cam in B_T_C:
                continue
            B_T_C[str(cam)] = np.asarray(T, dtype=np.float64)
            methods[str(cam)] = "propagated_from_step4"
            propagated.append(str(cam))

    propagation_info: Dict[str, Any] = {
        "used": bool(len(propagated) > 0),
        "source": (graph.source if graph is not None else None),
        "reference": (graph.reference if graph is not None else None),
        "propagated_cameras": propagated,
    }

    return CameraToBaseResult(
        B_T_C=B_T_C,
        methods=methods,
        propagation=propagation_info,
        B_T_T=B_T_T,
        graph=graph,
    )
