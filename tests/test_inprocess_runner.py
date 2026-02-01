from __future__ import annotations

import types
from pathlib import Path

from mcca.adapters.inprocess_runner import run_module_and_tee


def test_run_module_and_tee_calls_main_and_writes_log(tmp_path: Path) -> None:
    # 动态构造一个“可执行模块”：避免引入额外 fixture 文件，也不依赖任何大数据。
    m = types.ModuleType("_inprocess_runner_dummy_ok")

    def main(argv=None) -> int:
        print("dummy main start")
        print(f"argv={argv!r}")
        return 0

    m.main = main  # type: ignore[attr-defined]

    # 让 importlib.import_module 能找到它
    import sys

    sys.modules[m.__name__] = m

    log_path = tmp_path / "dummy.log"
    rc = run_module_and_tee(m.__name__, argv=["--foo", "bar"], log_path=log_path, cwd=tmp_path)

    assert rc == 0
    text = log_path.read_text(encoding="utf-8")
    assert "# cmd:" in text
    assert "dummy main start" in text
    assert "--foo" in text


def test_run_module_and_tee_coerces_system_exit_code(tmp_path: Path) -> None:
    m = types.ModuleType("_inprocess_runner_dummy_exit")

    def main(argv=None) -> int:
        _ = argv
        raise SystemExit(7)

    m.main = main  # type: ignore[attr-defined]

    import sys

    sys.modules[m.__name__] = m

    log_path = tmp_path / "exit.log"
    rc = run_module_and_tee(m.__name__, argv=[], log_path=log_path, cwd=tmp_path)
    assert rc == 7
