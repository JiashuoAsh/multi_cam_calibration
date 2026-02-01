---
name: 030_layout_cleanup
description: "破坏式结构治理：确定目标布局 -> 大搬家 -> 全库修引用 -> 清零旧路径（不保留兼容）"
argument-hint: "mode=plan|apply; profile=auto|lib|app|pipeline|research|mixed; scope=可选（留空=全仓库；或目录/通配如 libs/、step*）"
---

你是仓库结构治理助手。只做“结构治理”（搬家/重命名/修 imports/修路径引用/更新文档），允许破坏式变更：不需要兼容旧 import / 旧脚本路径 / 旧 API；可以删除旧文件与旧目录；禁止保留 wrapper/兼容层/转发文件。

输入：
- 执行模式：${input:mode:plan 或 apply（plan 只输出方案不改代码）}
- 布局档案：${input:profile:auto=自动判定；或 lib/app/pipeline/research/mixed}
- 范围（可选）：${input:scope:留空=全仓库；也可填某目录如 libs/ 或通配如 step*}

全局硬约束（必须遵守）：
1) 禁止臆测：所有结论必须引用仓库证据（README/pyproject/入口脚本/测试/现有目录）。找不到就写“未在仓库中找到”，并说明你查了哪些路径/文件名。
2) 不改业务逻辑（除非仅为适配新结构的必要改动：import 路径、模块对外入口、配置路径引用、脚本入口位置）。
  a. 结构治理边界：只允许改变“路径与导入引用”。
  b. 禁止改变函数/类的对外签名与语义；禁止借机重写算法流程或重构计算逻辑。
  c. 若确需算法级调整，必须在计划中明确列出并作为单独批次执行。
3) 每个批次结束必须可验证：import 冒烟 +（若存在入口）入口 --help +（若有测试）跑最小可行测试。
4) 不新增第三方依赖；避免过度设计。
5) 批次上限：mode=apply 时必须拆成 1~3 个批次；每批次最多迁移 2 个顶层目录或 ≤30 个文件（能按目录整体搬迁就不要拆成大量零散文件）。每批次结束必须完成清零检查 + 冒烟验证后才能进入下一批次。
6) scope 语义：scope 只接受空/目录前缀（如 libs/、steps/、tools/）。如果用户输入通配符（如 step*），仅将其作为搜索关键词使用，不承诺文件系统 glob 行为。

终端输出工作流（若需要运行命令）：
- 任何命令必须覆盖写入 ./output.txt（包含 stderr），例如：`<cmd> > ./output.txt 2>&1`
- 每次命令后必须读取 ./output.txt 再继续（避免 Copilot 读不到终端输出的 bug）
- 禁止追加写入（不允许 >>）
- 每次读取 ./output.txt 后，必须把“运行的命令 + 关键输出摘要”写入 docs/layout_migration.md（避免覆盖导致证据丢失）
- 需要给出 Windows 可运行命令（PowerShell/cmd/bash 任选其一，但要明确）

交付物（无论 plan/apply 都必须给出）：
- docs/layout_migration.md
  - 证据摘要（你基于哪些文件/目录做判断）
  - 目标布局（最终目录树）
  - 迁移映射表（旧路径 -> 新路径，按目录分组，必须可执行）
  - 全库修引用清单（import/模块名/配置路径/脚本调用路径）
  - 清零检查清单（如何证明旧路径引用为 0）
  - smoke checklist（命令 + 预期，写出 Windows 可运行命令）
  - 批次记录（batch1/2/3：每批做了什么、跑了哪些命令、结果摘要）

流程（严格按顺序输出；mode=apply 时执行对应动作）：

A) 证据收集（只读）
- 在 ${input:scope} 范围内定位并列出（若不存在要写明）：
  - README / docs 中的运行方式
  - pyproject.toml / setup.cfg / requirements.txt（如有）
  - 入口脚本：如 main.py、cli.py、__main__.py、scripts/、bin/、Makefile/Taskfile（如有）
  - tests/、pytest.ini、tox.ini、noxfile.py（如有）
  - 顶层目录结构（列出关键目录：src/、apps/、libs/、pipelines/、notebooks/、configs/ 等）
  - 明确哪些目录/文件应保持不动（例如 results/、data/、.venv/、.git/、输出目录等）

B) 判定仓库类型 -> 选择 profile
- 若 profile=auto：给出候选 profile（最多 2 个）+ 明确理由 + 最终选择
- 若用户指定 profile：仍需检查是否与证据冲突；冲突则提示风险但继续按指定执行

C) 明确“目标布局”（必须具体）
- 输出最终目录树（tree 形式）
- 明确以下决策（不要空泛）：
  - Python 包根路径（例如 src/<package>/... 或直接 <package>/...）
  - 入口点位置（CLI/服务/脚本入口）
  - tests/、configs/、docs/、notebooks/、scripts/ 的放置与边界
  - 不再保留的目录（准备删除的旧结构）

D) 输出“迁移映射表”（必须可执行）
- 以表格输出：旧路径 | 新路径 | 理由 | 是否需要改 import/引用
- 迁移动作必须最小化：能按目录整体搬迁就不要拆成大量零散文件（除非证据要求拆分）
- 同时输出“批次拆分计划”（batch1/2/3）：每批次迁移哪些目录/文件

E) 输出“全库修引用清单”（必须可搜索可替换）
- 列出需要修的引用类型，并给出可执行搜索关键字（rg/搜索词），至少包含：
  - Python import 路径（from/import）
  - 配置路径（yaml/json/toml/env/硬编码路径）
  - README/文档中的命令与路径
  - CI/脚本中的调用路径（若存在）
- 每一类给出：搜索关键词/正则（尽量稳妥）+ 预期替换方向（旧 -> 新）

F)（mode=apply）执行搬迁与全库修引用
对每个 batch：
- 先搬家，再立刻修引用（同一批次内完成）
- 更新 README/文档中涉及路径的部分
- 删除本批次涉及的旧文件/旧目录（不留兼容层）

G) 清零检查（每个 batch 结束必须做）
- 用全库搜索证明旧路径/旧模块名引用为 0
- 列出你检查过的旧关键字清单与结果（写进 docs/layout_migration.md）

H) 验证（必须输出 checklist；apply 必须执行并记录结果）（每个 batch 结束必须做）
- import 冒烟（给出命令）
- 入口 --help（若存在入口）
- 最小测试（若存在测试框架/测试命令）
- 如果无法验证：写明缺口（例如缺少测试/缺少入口），并给出下一步建议（不引入新依赖）

输出格式要求：
- 先输出“计划摘要”（3-8 条要点）
- 再输出 docs/layout_migration.md 的完整内容（或生成该文件）
- mode=apply 时：列出实际改动的文件路径清单（新增/删除/移动/修改）
---

开始执行：先做 A) 证据收集，然后按流程继续。不要跳步。