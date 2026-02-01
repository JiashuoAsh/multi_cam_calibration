#!/usr/bin/env python3
"""计算相机视场角（FOV）。

本工具用于读取 Step3 产出的内参文件（通常是 results/<cam>_intrinsics.json），
并基于针孔相机模型计算各相机的：

- 水平视场角（HFOV）
- 垂直视场角（VFOV）
- 对角视场角（DFOV）

重要说明：
- 这里计算的是“理想针孔模型”的几何视场角。
- 畸变、去畸变、裁剪（alpha/ROI）、缩放等会改变“有效视场”，不在本工具范围内。

用法示例：
- 默认读取：results/*_intrinsics.json
- 指定输入：python -m mcca.tools.compute_fov results/cam1_intrinsics.json
- 指定 glob：python -m mcca.tools.compute_fov "results/*intrinsics.json"
- 覆盖分辨率：python -m mcca.tools.compute_fov --image-size 2448x2048 results/cam1_intrinsics.json
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from mcca.core.fov import (
    compute_fov_deg,
    pinhole_from_step3_intrinsics_dict,
)


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


def build_arg_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

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
            data = json.loads(p.read_text(encoding="utf-8"))
            intri = pinhole_from_step3_intrinsics_dict(data, override_image_size=override_size)
            fov = compute_fov_deg(intri)
        except Exception as e:
            print(f"解析失败：{p.as_posix()} -> {e}")
            continue

        fmt = f"{{:.{int(args.decimals)}f}}"
        table_rows.append(
            {
                "file": p.name,
                "size": f"{intri.width}x{intri.height}",
                "HFOV(deg)": fmt.format(fov.hfov_deg),
                "VFOV(deg)": fmt.format(fov.vfov_deg),
                "DFOV(deg)": fmt.format(fov.dfov_deg),
            }
        )
        results_out.append(
            {
                "file": str(p.as_posix()),
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
        out_path.write_text(json.dumps(results_out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出 JSON：{out_path.as_posix()}")

    return 0


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
