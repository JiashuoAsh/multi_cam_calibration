#!/usr/bin/env python3
"""计算相机视场角（FOV）。

本脚本用于读取 Step3 产出的内参文件（通常是 results/<cam>_intrinsics.json），
并基于针孔相机模型计算各相机的：

- 水平视场角（HFOV）
- 垂直视场角（VFOV）
- 对角视场角（DFOV）

重要说明：
- 这里计算的是“理想针孔模型”的几何视场角。
- 畸变、去畸变、裁剪（alpha/ROI）、缩放等会改变“有效视场”，不在本脚本范围内。

用法示例：
- 默认读取：results/*_intrinsics.json
- 指定输入：python tools/compute_fov.py results/cam1_intrinsics.json
- 指定 glob：python tools/compute_fov.py "results/*intrinsics.json"
- 覆盖分辨率：python tools/compute_fov.py --image-size 2448x2048 results/cam1_intrinsics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Intrinsics:
    """针孔相机内参（用于 FOV 计算）。"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source_path: str


@dataclass(frozen=True)
class FovResult:
    """FOV 计算结果（单位：度）。"""

    hfov_deg: float
    vfov_deg: float
    dfov_deg: float


def _looks_like_glob(p: str) -> bool:
    return any(ch in p for ch in ("*", "?", "["))


def _parse_image_size(value: str) -> tuple[int, int]:
    """解析 --image-size 参数。

    支持格式：
    - 2448x2048
    - 2448,2048
    - 2448 2048

    Returns:
        (width, height)

    Raises:
        ValueError: 格式不合法。
    """

    v = value.strip().lower()
    for sep in ("x", ",", " "):
        if sep in v:
            parts = [p for p in v.replace(" ", sep).split(sep) if p]
            if len(parts) != 2:
                break
            w = int(parts[0])
            h = int(parts[1])
            if w <= 0 or h <= 0:
                raise ValueError("image_size 必须为正数")
            return w, h

    raise ValueError(f"无法解析 image_size: {value!r}，期望格式如 2448x2048")


def _load_intrinsics_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("内参文件内容不是 JSON object")
    return data


def _extract_k_values(data: dict[str, Any]) -> tuple[float, float, float, float]:
    """从 JSON 中提取 fx/fy/cx/cy。

    支持两种来源：
    - 顶层键：fx/fy/cx/cy
    - camera_matrix（3x3）：从中读取
    """

    if all(k in data for k in ("fx", "fy", "cx", "cy")):
        return float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"])

    cm = data.get("camera_matrix")
    if isinstance(cm, list) and len(cm) == 3 and all(isinstance(r, list) and len(r) == 3 for r in cm):
        fx = float(cm[0][0])
        fy = float(cm[1][1])
        cx = float(cm[0][2])
        cy = float(cm[1][2])
        return fx, fy, cx, cy

    raise KeyError("缺少 fx/fy/cx/cy 或 camera_matrix")


def _extract_image_size(data: dict[str, Any]) -> tuple[int, int]:
    """从 JSON 中提取 (width,height)。"""

    size = data.get("image_size")
    if isinstance(size, list) and len(size) == 2:
        w = int(size[0])
        h = int(size[1])
        if w <= 0 or h <= 0:
            raise ValueError("image_size 必须为正数")
        return w, h
    raise KeyError("缺少 image_size（需要 [width, height]）")


def load_intrinsics(path: Path, *, override_image_size: tuple[int, int] | None) -> Intrinsics:
    """读取并构造 Intrinsics。"""

    data = _load_intrinsics_json(path)
    fx, fy, cx, cy = _extract_k_values(data)

    if override_image_size is not None:
        width, height = override_image_size
    else:
        width, height = _extract_image_size(data)

    if fx <= 0 or fy <= 0:
        raise ValueError(f"fx/fy 必须为正数，但读到 fx={fx}, fy={fy}")

    return Intrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        width=int(width),
        height=int(height),
        source_path=str(path.as_posix()),
    )


def compute_fov_deg(intri: Intrinsics) -> FovResult:
    """基于 (fx,fy,cx,cy,width,height) 计算 HFOV/VFOV/DFOV（度）。

    说明：
    - 主点不在中心时，左右/上下视场角不对称；这里使用更严谨的加和形式。
    - 对角 DFOV：取四个角中离主点最远的角作为“半对角”，再乘 2。
    """

    w = float(intri.width)
    h = float(intri.height)
    fx = float(intri.fx)
    fy = float(intri.fy)
    cx = float(intri.cx)
    cy = float(intri.cy)

    left = math.atan(cx / fx)
    right = math.atan((w - cx) / fx)
    up = math.atan(cy / fy)
    down = math.atan((h - cy) / fy)

    hfov = left + right
    vfov = up + down

    corners = ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h))
    max_r = 0.0
    for u, v in corners:
        x = (u - cx) / fx
        y = (v - cy) / fy
        r = math.sqrt(x * x + y * y)
        if r > max_r:
            max_r = r
    dfov = 2.0 * math.atan(max_r)

    return FovResult(
        hfov_deg=math.degrees(hfov),
        vfov_deg=math.degrees(vfov),
        dfov_deg=math.degrees(dfov),
    )


def _iter_input_files(inputs: list[str]) -> list[Path]:
    """把命令行输入展开为文件列表。"""

    if len(inputs) == 0:
        inputs = ["results/*_intrinsics.json"]

    paths: list[Path] = []
    for item in inputs:
        item = item.strip()
        if not item:
            continue
        if _looks_like_glob(item):
            matched = [Path(p) for p in glob.glob(item)]
            paths.extend(matched)
        else:
            paths.append(Path(item))

    uniq: dict[str, Path] = {}
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        uniq[key] = p
    out = list(uniq.values())
    out.sort(key=lambda x: x.as_posix())
    return out


def _format_table(rows: list[dict[str, str]]) -> str:
    if len(rows) == 0:
        return ""

    headers = list(rows[0].keys())
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(r.get(h, "")))

    def fmt_row(r: dict[str, str]) -> str:
        return "  ".join(r.get(h, "").ljust(widths[h]) for h in headers)

    lines = [fmt_row({h: h for h in headers}), fmt_row({h: "-" * widths[h] for h in headers})]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="计算各相机内参对应的视场角（FOV）")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="输入文件路径或 glob（默认 results/*_intrinsics.json）",
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default=None,
        help="覆盖分辨率（widthxheight），例如 2448x2048；用于内参文件里缺少 image_size 或想用另一分辨率时",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="可选：把结果写出为 JSON 文件",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="角度小数位数（默认 3）",
    )

    args = parser.parse_args()

    override_size = _parse_image_size(args.image_size) if args.image_size else None
    files = _iter_input_files(list(args.inputs))

    if len(files) == 0:
        print("未找到任何输入文件。")
        return 2

    results_out: list[dict[str, Any]] = []
    table_rows: list[dict[str, str]] = []

    for p in files:
        if not p.exists() or not p.is_file():
            print(f"跳过：不存在的文件 {p.as_posix()}")
            continue

        try:
            intri = load_intrinsics(p, override_image_size=override_size)
            fov = compute_fov_deg(intri)
        except Exception as e:
            print(f"解析失败：{p.as_posix()} -> {e}")
            continue

        fmt = f"{{:.{int(args.decimals)}f}}"
        table_rows.append(
            {
                "file": Path(intri.source_path).name,
                "size": f"{intri.width}x{intri.height}",
                "HFOV(deg)": fmt.format(fov.hfov_deg),
                "VFOV(deg)": fmt.format(fov.vfov_deg),
                "DFOV(deg)": fmt.format(fov.dfov_deg),
            }
        )
        results_out.append(
            {
                "file": intri.source_path,
                "image_size": [intri.width, intri.height],
                "fx": intri.fx,
                "fy": intri.fy,
                "cx": intri.cx,
                "cy": intri.cy,
                "fov": asdict(fov),
            }
        )

    if len(table_rows) == 0:
        print("没有可输出的结果（可能所有文件都解析失败）。")
        return 2

    print(_format_table(table_rows))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results_out, f, ensure_ascii=False, indent=2)
        print(f"\n已写出 JSON：{out_path.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
