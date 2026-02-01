"""mcca

工程化后的包入口（Phase1）：
- 将原本散落在仓库根目录/`libs/` 下的可复用代码迁入包内，
  便于维护清晰的依赖方向与测试导入。

注意：当前仍处于分阶段重构中，后续 Phase2/3 会继续拆分职责与收敛 API。
"""

from __future__ import annotations

__all__: list[str] = []
