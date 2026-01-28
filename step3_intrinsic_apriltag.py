#!/usr/bin/env python3
"""
Step 3: 内参标定 - AprilTag 标定板

命名规范:
    A_T_B 表示 "B -> A 的变换" (X_A = A_T_B @ X_B)

功能:
    使用筛选后的合格图像进行相机内参标定。
    - 默认仍按 left/right 双目流程运行
    - 也支持 --cameras 指定 3-4 路（或更多）相机
    计算相机内参矩阵 K 和畸变系数 dist。

工作流程:
    1. 读取筛选后的图像
    2. 检测每张图像中的 AprilTag 标签
    3. 建立 2D-3D 点对应关系
    4. 使用 cv2.calibrateCamera 优化内参
    5. 生成去畸变效果对比图

标定方法:
    - 使用 OpenCV 的 cv2.calibrateCamera 函数
    - 自动计算焦距、主点和畸变系数

使用方法:
    python step3_intrinsic_apriltag.py

输入:
    - images/filtered/<cam>/*.(png|jpg|jpeg|bmp): 筛选后的相机图像（推荐）
    - images/raw/<cam>/*.(png|jpg|jpeg|bmp): 原始图像（filtered 缺失时回退）
    - config/apriltag_config.json: 标定板配置

输出:
        - results/<cam>_intrinsics.json: 相机内参
      - camera_matrix: 3x3 内参矩阵 K
      - dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3, ...]
      - reprojection_error: 重投影误差（像素）
        - （可选）results/visualization/step3_intrinsic_<cam>/*: 检测与重投影可视化

质量评估:
    - 重投影误差 < 0.5 像素: 优秀
    - 重投影误差 < 1.0 像素: 良好
    - 重投影误差 > 1.0 像素: 需要改进（检查标定板或图像质量）

下一步:
    - 双目：运行 python step4_stereo_extrinsic.py
    - 多相机：运行 python step4_multi_extrinsic_pose_graph.py
"""

import cv2
import numpy as np
import argparse
import json
import os
import glob
import shutil
from typing import Any
from pathlib import Path
from utils import (
    load_config,
    get_aruco_dict,
    detect_apriltag_corners,
    create_apriltag_board,
    create_opencv_aruco_board,
    get_detection_settings,
    get_detection_profile,
    get_detection_roi,
    get_detection_auto_roi,
    create_detector_params,
    get_image_dataset,
    get_dataset_cameras,
    get_camera_raw_images,
    get_camera_filtered_dir,
    get_camera_filtered_images,
)

# 最大有效图像数量（用于内参标定）
MAX_VALID_IMAGES = 50
MIN_TAGS = 1


# 默认尽量安静：只输出关键结果；需要逐张处理细节用 --verbose。
VERBOSE: bool = True


def _uniform_subsample_files(files: list[str], max_scan: int) -> list[str]:
    """Uniformly subsample file list to at most max_scan items.

    This is a small speed knob to avoid running AprilTag detection on hundreds of
    frames when only a limited number of valid samples are needed.
    """
    n = len(files)
    if max_scan <= 0 or max_scan >= n:
        return files
    # linspace is monotonic; int rounding may create adjacent duplicates.
    idx = np.linspace(0, n - 1, num=int(max_scan), dtype=int)
    out: list[str] = []
    last = None
    for i in idx:
        ii = int(i)
        if last is None or ii != last:
            out.append(files[ii])
            last = ii
    return out

def _safe_imwrite(path: str, img) -> bool:
    """
    兼容中文路径的写图：优先 cv2.imwrite，失败则使用 imencode + Python 写文件。
    """
    try:
        if cv2.imwrite(path, img):
            return True
    except Exception:
        pass

    ext = os.path.splitext(path)[1] or ".jpg"  # 需要带点，例如 ".jpg"
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            print(f"  警告: imencode 失败，无法写入: {path}")
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:
        print(f"  警告: 写入失败: {path} ({e})")
        return False


def _vprint(*args, **kwargs) -> None:
    """Verbose print (guarded by VERBOSE)."""
    if VERBOSE:
        print(*args, **kwargs)


def clean_visualization_dirs():
    """清空 step3 的可视化输出目录"""
    base = Path("results/visualization")
    if not base.exists():
        return

    # 兼容旧输出：step3_intrinsic_左 / step3_intrinsic_右
    for p in sorted(base.glob("step3_intrinsic_*")):
        if p.is_dir():
            shutil.rmtree(str(p))
            _vprint(f"  已清空: {p.as_posix()}")


def _glob_images(dir_path: str) -> list[str]:
    out: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        out.extend(glob.glob(os.path.join(dir_path, ext)))
    return sorted(out)


def _scan_cameras_from_source(source: str) -> list[str]:
    base = Path("images") / source
    if not base.exists():
        return []
    cams: list[str] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        files = _glob_images(str(p))
        if len(files) > 0:
            cams.append(p.name)
    return cams


def _pick_images_for_camera(cam: str) -> tuple[list[str], str]:
    """为单个相机挑选图像列表，并返回 (files, source)。

    优先 filtered；若 filtered 为空则回退 raw。
    """
    filtered_dir = os.path.join("images", "filtered", cam)
    raw_dir = os.path.join("images", "raw", cam)

    files = _glob_images(filtered_dir) if os.path.isdir(filtered_dir) else []
    if len(files) > 0:
        return files, "filtered"

    files = _glob_images(raw_dir) if os.path.isdir(raw_dir) else []
    if len(files) > 0:
        return files, "raw"

    return [], ""


def calibrate_camera_apriltag(
    image_files,
    obj_points_all,
    tag_ids,
    aruco_dict,
    side_name,
    *,
    detector_params,
    use_multiscale: bool,
    opencv_refine: bool,
    board,
    roi=None,
    auto_roi_cfg=None,
    save_visualization: bool = True,
    max_valid_images: int = MAX_VALID_IMAGES,
    min_tags: int = MIN_TAGS,
):
    """
    使用 AprilTag 标定单个相机的内参

    命名规范:
        cv2.calibrateCamera 返回的 rvecs/tvecs 表示 "T -> C" (标定板 -> 相机)
        即 Cl_T_T 或 Cr_T_T，但本函数不直接使用这些变换矩阵

    Args:
        image_files: 图像文件路径列表
        obj_points_all: 所有标签的 3D 角点 (num_tags, 4, 3)
        tag_ids: 标定板上所有标签的 ID 列表
        aruco_dict: ArUco 字典
        side_name: 相机名称（用于日志）
        save_visualization: 是否保存可视化图像（detection + reprojection）
        max_valid_images: 最大有效图像数量，默认50张

    Returns:
        success: 是否标定成功
        K: 相机内参矩阵 (3, 3)
        dist: 畸变系数
        rvecs: 每张图像的旋转向量列表 (T -> C)
        tvecs: 每张图像的平移向量列表 (T -> C)
        reproj_error: 平均重投影误差

    """
    print(f"\n{side_name}相机标定:")
    print(f"  - 图像数量: {len(image_files)}")
    print(f"  - 最大有效图像: {max_valid_images}")
    if min_tags != 5:
        print(f"  - min_tags: {min_tags}")

    # 创建可视化输出目录
    vis_base = None
    detection_dir = None
    reprojection_dir = None
    undistorted_dir = None
    if save_visualization:
        vis_base = f"results/visualization/step3_intrinsic_{side_name.lower()}"
        detection_dir = f"{vis_base}/detection"
        reprojection_dir = f"{vis_base}/reprojection"
        undistorted_dir = f"{vis_base}/undistorted"
        os.makedirs(detection_dir, exist_ok=True)
        os.makedirs(reprojection_dir, exist_ok=True)
        os.makedirs(undistorted_dir, exist_ok=True)
        _vprint(f"  - 可视化目录: {vis_base}")

    # detector_params 由外部统一创建（便于 profile/配置调参）

    # 收集所有有效图像的 2D-3D 对应点
    all_obj_pts = []
    all_img_pts = []
    valid_images = []
    valid_image_data = []  # 保存图像数据用于后续可视化
    image_size = None
    n_read_fail = 0
    n_low_tags = 0

    for img_idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            n_read_fail += 1
            _vprint(f"  警告: 无法读取 {img_path}")
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        auto_roi_cfg = auto_roi_cfg or {}
        corners, ids = detect_apriltag_corners(
            gray,
            aruco_dict,
            detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            roi=roi,
            auto_roi=bool(auto_roi_cfg.get("enabled", False)),
            auto_roi_pre_scale=float(auto_roi_cfg.get("pre_scale", 0.5)),
            auto_roi_min_tags=int(auto_roi_cfg.get("min_tags", 1)),
            auto_roi_margin=float(auto_roi_cfg.get("margin", 0.25)),
        )

        if ids is None or corners is None or len(ids) < int(min_tags):
            n_low_tags += 1
            _vprint(
                f"  跳过 {os.path.basename(img_path)}: 标签不足 ({len(ids) if ids is not None else 0} < {min_tags})"
            )
            continue

        # 收集该图像中所有标签的对应点
        img_obj_pts = []
        img_img_pts = []

        ids_flat = ids.flatten()
        for i, tag_id in enumerate(ids_flat):
            if tag_id in tag_ids:
                idx = tag_ids.index(tag_id)
                obj_pts = obj_points_all[idx]  # (4, 3)
                img_pts = corners[i].reshape(-1, 2)  # (4, 2)

                img_obj_pts.append(obj_pts)
                img_img_pts.append(img_pts)

        if len(img_obj_pts) > 0:
            # 合并该图像的所有角点
            obj_pts_img = np.vstack(img_obj_pts).astype(np.float32)
            img_pts_img = np.vstack(img_img_pts).astype(np.float32)

            all_obj_pts.append(obj_pts_img)
            all_img_pts.append(img_pts_img)
            valid_images.append(img_path)

            # 检查是否达到最大有效图像数量
            if len(valid_images) >= max_valid_images:
                print(f"  已达到最大有效图像数量 ({max_valid_images})，停止处理")
                break

            # 保存数据用于可视化
            if save_visualization:
                assert detection_dir is not None
                valid_image_data.append(
                    {
                        "image": img.copy(),
                        "corners": corners,
                        "ids": ids,
                        "index": len(valid_images),
                    }
                )

                # 保存 AprilTag 检测结果图
                vis_img = img.copy()
                cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
                info_text = f"Image {len(valid_images)}: {len(ids)} tags detected"
                cv2.putText(
                    vis_img,
                    info_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                detection_path = (
                    f"{detection_dir}/{len(valid_images):02d}_tags_detected.jpg"
                )
                _safe_imwrite(detection_path, vis_img)

    print(f"  - 有效图像: {len(valid_images)}/{len(image_files)}")
    if (n_read_fail or n_low_tags) and (not VERBOSE):
        # 默认只给一个汇总，细节用 --verbose
        parts = []
        if n_read_fail:
            parts.append(f"读取失败 {n_read_fail}")
        if n_low_tags:
            parts.append(f"标签不足 {n_low_tags}")
        print(f"  - 跳过统计: {', '.join(parts)}")

    if len(valid_images) < 3:
        print(f"  错误: 有效图像太少 (<3)")
        return False, None, None, None, None, None, None

    assert image_size is not None

    # 执行相机标定
    print("  - 正在标定...")

    # OpenCV 允许用 None 让其自动初始化；但类型存根对 None 不友好。
    none_umat: Any = None
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_pts,
        all_img_pts,
        image_size,
        none_umat,
        none_umat,
    )

    # OpenCV 返回的 ret 是全局 RMS 重投影误差（单位：像素）
    print(f"  - RMS(ret, OpenCV): {ret:.4f} 像素")

    # 计算重投影误差并保存可视化
    mean_error = 0
    per_image_errors = []

    for i in range(len(all_obj_pts)):
        img_pts2, _ = cv2.projectPoints(all_obj_pts[i], rvecs[i], tvecs[i], K, dist)
        # mean reprojection error（像素）: 每个点的欧氏误差取平均
        proj = img_pts2.reshape(-1, 2)
        det = np.asarray(all_img_pts[i], dtype=np.float64).reshape(-1, 2)
        per_pt = np.linalg.norm(det - proj, axis=1)
        error = float(np.mean(per_pt))
        mean_error += error
        per_image_errors.append(error)

        # 保存重投影误差可视化
        if save_visualization and i < len(valid_image_data):
            vis_img = valid_image_data[i]["image"].copy()
            img_pts_orig = det
            img_pts_reproj = img_pts2.reshape(-1, 2)

            # 绘制原始检测点（绿色）和重投影点（红色）
            for j in range(len(img_pts_orig)):
                pt_orig = tuple(img_pts_orig[j].astype(int))
                pt_reproj = tuple(img_pts_reproj[j].astype(int))

                cv2.circle(vis_img, pt_orig, 6, (0, 255, 0), -1)  # 绿色：检测点
                cv2.circle(vis_img, pt_reproj, 4, (0, 0, 255), -1)  # 红色：重投影点
                cv2.line(vis_img, pt_orig, pt_reproj, (255, 0, 0), 1)  # 蓝线：误差

            # 显示误差信息
            error_text = (
                f"Image {valid_image_data[i]['index']}: Mean Error = {error:.3f} px"
            )
            cv2.putText(
                vis_img,
                error_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis_img,
                "Green: Detected | Red: Reprojected",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            reproj_path = (
                f"{reprojection_dir}/{valid_image_data[i]['index']:02d}_error_map.jpg"
            )
            _safe_imwrite(reproj_path, vis_img)

    mean_error /= len(all_obj_pts)

    print(f"  - 平均误差(mean_error, per-image mean): {mean_error:.4f} 像素")

    # 评估标定质量
    if mean_error < 0.5:
        quality = "优秀"
    elif mean_error < 1.0:
        quality = "良好"
    else:
        quality = "需要改进"

    print(f"  - 标定质量: {quality}")

    if save_visualization:
        assert detection_dir is not None
        assert reprojection_dir is not None
        assert undistorted_dir is not None
        _vprint(f"  - 已保存 {len(valid_images)} 张检测图像到 {detection_dir}")
        _vprint(f"  - 已保存 {len(valid_images)} 张重投影图像到 {reprojection_dir}")

        # 保存所有图像的去畸变版本
        print(f"  - 正在生成去畸变图像...")

        # 使用第一张图像的尺寸创建去畸变映射（所有图像尺寸相同）
        first_img = cv2.imread(valid_images[0])
        if first_img is None:
            print("  警告: 无法读取第一张有效图像，跳过去畸变输出")
            return True, K, dist, rvecs, tvecs, mean_error, image_size
        h, w = first_img.shape[:2]

        # 使用 remap 方法进行去畸变（推荐方法）
        # 保持原始内参矩阵K，不改变图像尺寸，只矫正畸变
        mapx, mapy = cv2.initUndistortRectifyMap(
            K,
            dist,
            none_umat,
            K,
            (w, h),
            cv2.CV_32FC1,
        )

        for i, img_path in enumerate(valid_images):
            img = cv2.imread(img_path)
            if img is None:
                _vprint(f"  警告: 无法读取图像，跳过去畸变: {img_path}")
                continue

            # 使用remap进行去畸变（比undistort更快且效果更好）
            img_undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            # 保存去畸变图像（保持原始尺寸720x1280，不缩放）
            base_name = os.path.basename(img_path)
            undist_path = f"{undistorted_dir}/{i + 1:02d}_undistorted_{base_name}"
            _safe_imwrite(undist_path, img_undist)

        _vprint(f"  - 已保存 {len(valid_images)} 张去畸变图像到 {undistorted_dir}")

    return True, K, dist, rvecs, tvecs, mean_error, image_size


def save_intrinsics(output_path, K, dist, reproj_error, image_size):
    """保存内参标定结果"""
    result = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "reprojection_error": float(reproj_error),
        "image_size": list(image_size),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  - 已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Step3：AprilTag 内参标定（默认安静，--verbose 可看更多过程信息）")
    parser.add_argument(
        "--config",
        type=str,
        default="config/apriltag_config.json",
        help="配置文件路径（默认 config/apriltag_config.json）",
    )
    # parser.add_argument("--verbose", action="store_true", help="输出逐张图像的检测/跳过原因等过程信息")
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="不保存可视化输出（detection/reprojection/undistorted），更快更省空间",
    )
    parser.add_argument(
        "--max_valid_images",
        type=int,
        default=MAX_VALID_IMAGES,
        help=f"最多使用多少张有效图像进行标定（默认 {MAX_VALID_IMAGES}）",
    )
    parser.add_argument(
        "--min_tags",
        type=int,
        default=5,
        help="每张图像至少需要检测到多少个标签才算有效（默认 5）",
    )

    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help=(
            "要标定的相机名称列表（对应 images/<source>/<cam>/...）。"
            "不传则自动扫描 images/filtered（为空再扫 images/raw）。默认双目工程推荐 left right。"
        ),
    )

    parser.add_argument(
        "--no_multiscale",
        action="store_true",
        help="强制关闭多尺度检测（更快，但对小标签/远距离可能更不稳）。",
    )
    parser.add_argument(
        "--no_opencv_refine",
        action="store_true",
        help="强制关闭 OpenCV refine（更快）。",
    )

    parser.add_argument(
        "--max_scan_images",
        type=int,
        default=0,
        help=(
            "Max number of candidate images to run AprilTag detection on (0=auto). "
            "This is a speed cap to avoid scanning the full dataset."
        ),
    )
    args = parser.parse_args()

    # global VERBOSE
    # VERBOSE = bool(args.verbose)

    print("=" * 60)
    print("Step 3: AprilTag 内参标定")
    print("=" * 60)

    # 清空之前的可视化输出（若禁用可视化则不清理，避免误删旧结果）
    if not args.no_vis:
        print("\n清理旧的可视化文件...")
        clean_visualization_dirs()

    # 加载配置
    config = load_config(str(args.config))
    ds = get_image_dataset(config)
    use_dataset = bool(ds.get("enabled", False))

    # 确定要标定的相机列表（优先 CLI，其次 config.image_dataset，再次扫描目录）
    cameras = list(args.cameras) if args.cameras is not None and len(args.cameras) > 0 else []
    if len(cameras) == 0 and use_dataset:
        cameras = get_dataset_cameras(config, allow_scan=True, fallback_stereo=True)
    if len(cameras) == 0:
        cameras = _scan_cameras_from_source("filtered")
        if len(cameras) == 0:
            cameras = _scan_cameras_from_source("raw")
    if len(cameras) == 0:
        print("\n错误: 未找到任何相机图像目录 images/filtered/<cam>/ 或 images/raw/<cam>/")
        print("请先采集图像（或运行 python step2_filter_images.py 生成 filtered 图像）")
        return

    use_multiscale, opencv_refine = get_detection_settings(config)
    if bool(getattr(args, "no_multiscale", False)):
        use_multiscale = False
    if bool(getattr(args, "no_opencv_refine", False)):
        opencv_refine = False
    profile = get_detection_profile(config)
    auto_roi_cfg = get_detection_auto_roi(config)
    detector_params = create_detector_params(config)

    # 创建 AprilTag 标定板
    obj_points, tag_ids = create_apriltag_board(config)
    aruco_dict = get_aruco_dict(config["apriltag_board"]["family"])
    board = create_opencv_aruco_board(obj_points, tag_ids, aruco_dict)

    # 收集每个相机的图像文件（每个相机独立：优先 filtered，若为空则回退 raw）
    cam_to_images: dict[str, list[str]] = {}
    cam_to_source: dict[str, str] = {}
    for cam in cameras:
        if use_dataset:
            filtered_imgs = get_camera_filtered_images(config, cam)
            if len(filtered_imgs) > 0:
                cam_to_images[cam] = [str(p) for p in filtered_imgs]
                cam_to_source[cam] = "filtered"
            else:
                raw_imgs = get_camera_raw_images(config, cam)
                cam_to_images[cam] = [str(p) for p in raw_imgs]
                cam_to_source[cam] = "raw"
        else:
            files, src = _pick_images_for_camera(cam)
            cam_to_images[cam] = files
            cam_to_source[cam] = src

    missing = [c for c in cameras if len(cam_to_images.get(c, [])) == 0]
    if len(missing) > 0:
        print("\n错误: 以下相机未找到任何图像：")
        for c in missing:
            print(f"  - {c} (期望 images/filtered/{c}/ 或 images/raw/{c}/)")
        return

    print("\n找到图像：")
    for cam in cameras:
        if use_dataset:
            if cam_to_source[cam] == "filtered":
                src_dir = get_camera_filtered_dir(config, cam)
            else:
                # raw 可能来自 glob，不一定在 raw_root/<cam>
                src_dir = Path("<raw>")
            print(f"  - {cam}: {len(cam_to_images[cam])} 张 (source={cam_to_source[cam]})")
        else:
            print(f"  - {cam}: {len(cam_to_images[cam])} 张 (images/{cam_to_source[cam]}/{cam}/)")
    print(f"  - detection profile: {profile}")
    if bool(auto_roi_cfg.get("enabled", False)):
        print(
            "  - auto_roi: enabled "
            f"(pre_scale={auto_roi_cfg.get('pre_scale')}, min_tags={auto_roi_cfg.get('min_tags')}, margin={auto_roi_cfg.get('margin')})"
        )

    # 确保输出目录存在
    os.makedirs("results", exist_ok=True)

    # Limit how many images to run detection on (huge speedup for large datasets).
    auto_scan = int(args.max_valid_images) * 8
    max_scan = int(args.max_scan_images) if int(args.max_scan_images) > 0 else auto_scan
    max_scan = max(int(args.max_valid_images), int(max_scan))

    # 逐相机标定
    per_cam_summary: list[tuple[str, np.ndarray, float]] = []
    for cam in cameras:
        images = cam_to_images[cam]
        images_scan = _uniform_subsample_files(list(images), max_scan=max_scan)
        if len(images_scan) != len(images):
            print(
                f"\n为加速检测，仅扫描部分候选图片: {cam} {len(images_scan)}/{len(images)} (max_scan={max_scan})"
            )

        roi = get_detection_roi(config, camera=cam)
        if roi is not None:
            print(f"\n{cam} ROI: {roi}")

        display_name = cam
        success, K, dist, rvecs, tvecs, err, img_size = calibrate_camera_apriltag(
            images_scan,
            obj_points,
            tag_ids,
            aruco_dict,
            display_name,
            detector_params=detector_params,
            use_multiscale=use_multiscale,
            opencv_refine=opencv_refine,
            board=board,
            roi=roi,
            auto_roi_cfg=auto_roi_cfg,
            save_visualization=(not args.no_vis),
            max_valid_images=int(args.max_valid_images),
            min_tags=int(args.min_tags),
        )

        if not success:
            print(f"\n错误: 相机 {cam} 标定失败！")
            return

        assert K is not None and dist is not None and img_size is not None
        assert err is not None
        save_intrinsics(f"results/{cam}_intrinsics.json", K, dist, err, img_size)
        per_cam_summary.append((cam, K, float(err)))

    # 显示总结
    print("\n" + "=" * 60)
    print("内参标定完成！")
    print("=" * 60)
    for cam, K, err in per_cam_summary:
        print(f"\n{cam}:")
        print(f"  - fx = {K[0, 0]:.2f}, fy = {K[1, 1]:.2f}")
        print(f"  - cx = {K[0, 2]:.2f}, cy = {K[1, 2]:.2f}")
        print(f"  - 平均误差(mean_error) = {err:.4f} 像素")
        print("  - RMS(ret) 见上方标定日志")

    if len(cameras) == 2 and set(cameras) == {"left", "right"}:
        print("\n下一步: 运行 python step4_stereo_extrinsic.py")
    else:
        print("\n下一步: 运行 python step4_multi_extrinsic_pose_graph.py")


if __name__ == "__main__":
    main()
