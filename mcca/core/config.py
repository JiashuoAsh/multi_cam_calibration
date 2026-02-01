from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path = "config/apriltag_config.json") -> dict[str, Any]:
    """加载 AprilTag 标定配置文件。

    说明：
        - 配置文件必须是标准 JSON（不支持 // 注释）。本工程推荐使用 "_comment" 字段
          作为可解析的“注释”。
        - 顶层必须是 object（dict）。

    Args:
        config_path: 配置文件路径（默认：config/apriltag_config.json）。

    Returns:
        配置 dict。

    Raises:
        SystemExit: 当文件不存在、读取失败或 JSON 顶层不是 object 时。
    """

    p = Path(config_path)
    if not p.exists():
        raise SystemExit(f"错误：配置文件不存在：{p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"错误：读取配置文件失败：{p} ({e})")

    if not isinstance(data, dict):
        raise SystemExit(f"错误：配置文件格式不正确（顶层必须是 object）：{p}")

    return data
