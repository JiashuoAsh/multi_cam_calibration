---
name: 02_package_module
description: "把一个完整功能封装成可复用 Python 包：public API + 目录归位 + 最小用例"
---

目标功能边界（用一句话描述）：${input:feature:例如“网球检测与定位”}
涉及的代码路径（逗号分隔）：${input:paths:例如 src/foo, app/main.py, scripts/debug_x.py}

目标：把上述功能封装成“可复用包/模块”，让其他程序能稳定地 import 调用，同时保持行为不变。

允许破坏式变更：
- 允许修改模块结构与对外 API
- 禁止保留 wrapper/兼容层/旧模块转发
- 必须同步更新仓库内所有调用点，确保旧路径/旧符号清零

按固定流程执行（不要跳步）：

A) 现状盘点（只读）
- 这个功能的入口/调用链是什么？（路径 + 关键符号）
- 哪些是核心逻辑（core），哪些是 IO/硬件/文件/网络（adapters），哪些是入口（cli/apps/scripts）？
- 当前对外调用方式有哪些（别人怎么 import/怎么调用）？

B) 包化设计（最小可行，避免过度设计）
- 选择包位置：优先 `src/<pkg>/...`（若仓库已有包结构则遵循现有）
- 定义 public API（少量入口）：建议 `src/<pkg>/api.py` 或 `src/<pkg>/__init__.py` 导出
- 明确依赖方向：adapters/entry -> core；core 不依赖 adapters/entry

C) 实施迁移（小步）
- 移动/重命名文件到正确目录（core/adapters/entry 分区）
- 同步修复 imports、入口调用方式、配置路径引用、文档运行方式（若受影响）
- 不新增第三方依赖；不引入复杂框架/插件系统

D) 可用性验证（必须给出）
- 提供“最小使用示例”（别的程序如何 import 并调用 public API）
- 如果仓库使用 pyproject/setup：给出 `pip install -e .` 后的调用示例（只在仓库确实支持时）
- 至少提供一种自检：最小冒烟脚本或最小单测（happy path）

输出约束：
- 禁止臆测：找不到就写“未在仓库中找到”，并说明查找位置。
- 所有结论引用具体路径与关键符号名。
- 注释/log/docstring 用中文。
