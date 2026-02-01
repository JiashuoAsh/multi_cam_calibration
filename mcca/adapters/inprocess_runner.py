from __future__ import annotations

import importlib
import os
import runpy
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence


class _TeeTextIO:
    """把文本同时写到多个 stream。

    说明：
    - 用于在“进程内执行 step”时，模拟 subprocess_runner.run_and_tee 的行为：
      同时输出到控制台与日志文件。
    - 这里实现最小 write/flush 接口即可满足 print()/traceback 等常见输出路径。
    """

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def _coerce_exit_code(code) -> int:
    """把 SystemExit.code 统一成 int。"""

    if code is None:
        return 0
    if isinstance(code, int):
        return int(code)
    # argparse 可能传字符串，或抛异常对象；统一视为失败。
    return 1


def _call_module_main(module, argv: Sequence[str]) -> int:
    """调用模块入口：仅允许 main(argv=...) -> int。

    设计说明：
    - 过去为了兼容不同模块的入口形态，这里存在多级回退链（main(list)->main()->cli_main）。
      这会让“模块如何被执行”变成隐式约定，难以维护，也容易在重构中引入隐藏行为差异。
    - 现在统一约定：所有可被 pipeline/runner 执行的模块都必须实现：
        main(argv: Optional[Sequence[str]] = None) -> int
      并由模块自己的 __main__/cli_main 负责 SystemExit 的抛出。

    注意：
    - 通过 importlib 导入模块后调用函数，能保持函数的 __module__ 为真实模块名，
      避免 runpy 以 __main__ 执行时对 multiprocessing pickling 的潜在影响。
    """

    fn_main = getattr(module, "main", None)
    if not callable(fn_main):
        raise AttributeError(
            "模块缺少可执行入口：未找到 main(argv: Optional[Sequence[str]] = None) -> int"
        )

    try:
        rc = fn_main(argv=list(argv))
    except TypeError as e:
        raise TypeError(
            "模块 main 签名不符合约定：必须支持 main(argv=...)，例如 main(argv: Optional[Sequence[str]] = None) -> int"
        ) from e

    if rc is None:
        raise TypeError("模块 main() 必须返回 int 退出码（禁止返回 None）。")
    if isinstance(rc, bool):
        return int(rc)
    if isinstance(rc, int):
        return int(rc)
    raise TypeError(f"模块 main() 必须返回 int 退出码，当前类型={type(rc)!r}")


def run_module_and_tee(
    module_name: str,
    *,
    argv: Sequence[str],
    log_path: Path,
    cwd: Path,
) -> int:
    """在当前进程内运行一个模块入口，并把输出 tee 到日志文件。

    Args:
        module_name: 例如 "mcca.entry.step2_filter_images"。
        argv: 传给该模块的命令行参数（不含 python -m 与模块名）。
        log_path: 日志文件路径（utf-8）。
        cwd: 执行工作目录（用于相对路径解析）。

    Returns:
        退出码（0 表示成功）。
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 记录头：保持与 subprocess_runner.run_and_tee 接近，便于排查。
    cmd_line = " ".join([sys.executable, "-m", module_name] + list(argv))

    old_argv = sys.argv
    old_cwd = os.getcwd()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {cmd_line}\n")
        f.write(f"# time: {datetime.now().isoformat()}\n\n")
        f.flush()

        tee = _TeeTextIO(old_stdout, f)
        tee_err = _TeeTextIO(old_stderr, f)

        try:
            os.chdir(str(cwd))
            sys.argv = [module_name] + list(argv)
            sys.stdout = tee
            sys.stderr = tee_err

            module = importlib.import_module(module_name)
            return _call_module_main(module, argv=list(argv))
        except SystemExit as e:
            return _coerce_exit_code(e.code)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.argv = old_argv
            os.chdir(old_cwd)


def run_script_and_tee(
    script_path: Path,
    *,
    argv: Sequence[str],
    log_path: Path,
    cwd: Path,
) -> int:
    """在当前进程内运行一个脚本文件，并把输出 tee 到日志文件。"""

    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_line = " ".join([sys.executable, str(script_path)] + list(argv))

    old_argv = sys.argv
    old_cwd = os.getcwd()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {cmd_line}\n")
        f.write(f"# time: {datetime.now().isoformat()}\n\n")
        f.flush()

        tee = _TeeTextIO(old_stdout, f)
        tee_err = _TeeTextIO(old_stderr, f)

        try:
            os.chdir(str(cwd))
            sys.argv = [str(script_path)] + list(argv)
            sys.stdout = tee
            sys.stderr = tee_err

            try:
                runpy.run_path(str(script_path), run_name="__main__")
            except SystemExit as e:
                return _coerce_exit_code(e.code)
            return 0
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.argv = old_argv
            os.chdir(old_cwd)
