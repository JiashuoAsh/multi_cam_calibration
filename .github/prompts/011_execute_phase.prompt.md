---
name: 011_execute_phase
description: "按 Phase 批量重构：一次处理一组热点/目录调整 + 冒烟验证（可控大步子）"
argument-hint: "phase=1/2/3; targets=逗号分隔路径; goal=一句话目标"
agent: "agent"
---

本次执行的 Phase：${input:phase:填 1/2/3}
本次处理范围 targets（逗号分隔，文件或目录）：${input:targets:例如 utils.py,libs/apriltag_perf,mcca/entry/step4_multi_extrinsic.py}
本次目标（一句话）：${input:goal:例如“修正依赖方向并建立最小库/入口边界”}

原则：允许一次性处理多个文件/目录，但必须“批量且有边界”，不能变成全仓库随意大改。
硬约束：不新增第三方依赖；不做过度设计；保持既有行为不变（除非明确说明）；中文注释/log/docstring。

按固定流程输出并执行（不要跳步）：

A) Phase 解释与范围确认（只读）
- 结合仓库现状说明本 Phase 的目的
- 对每个 target 给出当前问题分类：职责混杂/依赖方向错误/目录放错/脚本被当库 import/可复用性差 等

B) 批量改造方案（必须可回滚）
- 输出“迁移映射表”：旧路径 -> 新路径（如果本次涉及搬家）
- 输出“公共 API/兼容策略”：哪些旧 import/函数名保留 wrapper，哪些会迁移到新模块
- 输出“分步实施清单”：把本次改造拆成 3~7 个小步，每步说明改动点与影响面

C) 执行（按小步逐条完成）
- 每完成一步就确保：可 import/入口不崩
- 修复所有受影响的 imports/入口调用方式/配置路径/README 指令（若受影响）
- 如果需要兼容旧路径，保留薄 wrapper（只转发，不包含业务逻辑）

D) 冒烟验证（必须给）
- 输出最小 smoke checklist：命令 + 预期结果（优先 --help / import / 最小运行路径）
- 本次改造完成后，确保 checklist 全通过；若有无法验证之处，明确缺口与补齐建议

输出要求：所有结论引用具体路径与关键符号名；找不到则写“未在仓库中找到”并说明查找位置。
