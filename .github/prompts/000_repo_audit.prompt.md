---
name: 000_repo_audit
description: "盘点仓库入口/核心模块/热点清单/分阶段重构路线（不改代码）"
argument-hint: "（可选）focus=某个目录，如 focus=src/"
---

请对当前仓库做结构盘点（不改任何代码/文件），输出“可执行的重构路线图”。

可选聚焦范围：${input:focus:留空表示全仓库}

必须输出（按顺序）：
1) 入口清单（entry points）
   - 找到并列出：main 脚本、CLI、服务启动、console_scripts（如果存在）。
   - 每一项都必须给出证据：文件路径 / 配置文件位置（如 pyproject.toml 等）。

2) 核心模块清单（core modules）
   - 列出被最多地方 import 或处于关键链路的模块/包（路径 + 1 句话责任描述）。

3) 热点清单（hotspots，按影响排序）
   - 类型包括：职责混杂（core+IO+入口混一起）、高耦合（改一处牵一片）、scripts 被当库 import、循环依赖风险、目录放错。
   - 每项必须包含：路径、症状、为什么优先、建议处理方式（拆分/搬家/包化/只加边界）。

4) 分阶段计划（phase plan）
   - Phase 1：最小骨架（库 vs 入口 vs scripts 的边界，建议目录）
   - Phase 2：逐个热点拆解（一次一个）
   - Phase 3：收尾（统一 import 路径、public API、包化、文档/自检）

规则：
- 禁止臆测：找不到就写“未在仓库中找到”，并说明你查了哪里（README、pyproject、入口脚本等）。
- 所有判断必须引用具体路径与关键符号名（模块/类/函数），不要泛泛而谈。
