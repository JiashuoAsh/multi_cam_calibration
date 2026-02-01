from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Sequence


def run_and_tee(cmd: Sequence[str], *, log_path: Path, cwd: Path) -> int:
    """运行子进程，将 stdout/stderr 同时写入 log 与控制台。

    说明：
    - 该函数属于 IO/系统层封装（adapters），用于让 entry 层只关心“要运行什么”。
    - 为避免 Windows 控制台编码带来的异常，默认设置 PYTHONIOENCODING=utf-8。

    Args:
        cmd: 子进程命令（已经是完整 argv）。
        log_path: 日志文件路径。
        cwd: 子进程工作目录。

    Returns:
        子进程退出码。
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # 统一输出编码，避免 Windows/控制台编码导致的异常。
    env.setdefault("PYTHONIOENCODING", "utf-8")

    cmd_list = list(cmd)

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd_list)}\n")
        f.write(f"# time: {datetime.now().isoformat()}\n\n")
        f.flush()

        p = subprocess.Popen(
            list(cmd_list),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            f.write(line)

        return int(p.wait())
