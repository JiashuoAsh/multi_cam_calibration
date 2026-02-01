"""入口层（entry）。

这里放置 CLI/脚本入口与依赖组装代码：
- 允许依赖 adapters/core
- 不允许被 core 反向依赖

说明：
- 该层只做参数解析、路径解析、日志/输出组织，并调用 core/adapters 完成实际工作。
"""

from __future__ import annotations

__all__: list[str] = []
