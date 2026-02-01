"""已移除：pipeline 计划属于 entry 层。

问题背景：
- 旧版把“流水线 step 计划（mcca.entry.* 的模块名 + argv）”放在 core 层，导致依赖方向反转。
- 本仓库采用分层约定：entry/adapters -> core；core 不应依赖 entry。

breaking=1：
- 该模块路径不再提供任何实现，请改用 `mcca.entry.pipeline_plan`。
"""

raise ModuleNotFoundError("mcca.core.pipeline_plan 已移除，请使用 mcca.entry.pipeline_plan")
