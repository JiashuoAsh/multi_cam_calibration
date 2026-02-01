# 仓库结构治理（layout cleanup）迁移文档

- 执行模式：apply（执行搬迁/修引用/删除旧路径；不保留兼容层）
- profile：auto -> pipeline（见“B) 判定仓库类型”）
- scope：全仓库
- 生成时间：2026-02-01

## 计划摘要（建议落地顺序）

1) 保持 `mcca/` 作为 Python 包根路径不变（避免引入 `src/` 布局后需要额外的 PYTHONPATH/安装步骤，影响 `python -m mcca...` 与现有测试工作流）。
2) 新增 `scripts/`，把根目录的可执行脚本集中管理：移动 `run_calibration_pipeline.sh` 到 `scripts/`。
3) 将根目录的临时/笔记类文本集中到 `docs/notes/`：移动 `step3.txt`、`step4.txt`。
4) 每个 batch 必须做清零检查（全仓搜索旧路径字符串为 0）+ smoke（import/--help/pytest）。
5) 不触碰运行期目录与大数据目录（`images/`、`videos/`、`results/`、`cache/` 等），避免破坏数据路径约定。

---

## A) 证据收集（只读）

### 运行方式（README / docs）

- README 明确建议使用：
  - `python -m mcca.entry.pipeline --config config/apriltag_config.json`
  - `python -m mcca.entry.pipeline --all --config config/apriltag_config.json`
  - 证据：`README.md`（“快速开始（推荐：配置驱动 + 一键流水线）”章节）。

### 项目元信息 / 依赖 / 入口（pyproject）

- `pyproject.toml` 存在 `[project]` 与 `[project.scripts]`，且 console scripts 指向 `mcca.entry.*`：
  - `multi-cam-cali-apriltag-pipeline = "mcca.entry.pipeline:cli_main"`
  - `multi-cam-cali-apriltag-step5c-world-anchor = "mcca.entry.step5c_world_anchor:cli_main"`
  - 证据：`pyproject.toml`。

### 入口脚本 / scripts

- 脚本已迁移至：`scripts/run_calibration_pipeline.sh`
  - 内部调用：`python -m mcca.entry.pipeline --all --config config/apriltag_config.json`
  - 证据：`scripts/run_calibration_pipeline.sh`。

### tests / 测试框架

- 存在 `tests/` 且使用 pytest：
  - 证据：`tests/test_*.py`、`pyproject.toml` 的 `dependency-groups.dev` 含 `pytest`。
- 未在仓库中找到：`pytest.ini`、`tox.ini`、`noxfile.py`
  - 证据：全仓文件检索结果为空。

### 顶层目录结构（关键目录）

当前顶层（摘录，证据：根目录列表）：
- 代码与工程：`mcca/`、`tests/`、`docs/`、`config/`、`.github/`、`pyproject.toml`、`README.md`
- 运行期/数据：`images/`、`videos/`、`results/`、`cache/`
- 环境/工具缓存：`.venv*/`、`.pytest_cache/`、`.ruff_cache/`
- 临时文件：`output.txt`、`pytest_exit_code.txt`（历史：`step3.txt`、`step4.txt` 已迁移到 `docs/notes/`）

### 明确哪些目录/文件应保持不动

以下目录/文件属于运行期资产或环境目录，布局治理中建议保持根路径不变：
- `.git/`、`.github/`、`.vscode/`
- `.venv*/`（环境目录；建议统一使用 `.venv-mcca`。若你在更大的 workspace 下工作，uv 也可能将其创建在当前仓库的上级目录。）
- `images/`、`videos/`（数据集）
- `results/`（产物目录）
- `cache/`（检测缓存）
- `config/`（默认配置路径在 README 与脚本/入口默认值中被广泛使用）
- `output.txt`（你的工作流里用于终端输出覆盖写入；不建议移动）

---

## B) 判定仓库类型 -> 选择 profile

profile=auto 的候选：

1) pipeline
- 理由：项目核心是“多步流水线（Step1~Step6）”，入口集中在 `mcca.entry.*`，并存在一键运行脚本（`run_calibration_pipeline.sh`），且输出落在 `results/`。

2) mixed
- 理由：既有可复用库（`mcca.core.*`、`mcca.adapters.*`），又有 CLI/pipeline 与数据目录（`images/`/`videos/`）。

最终选择：pipeline
- 原因：布局治理主要目标是让“入口脚本/运行方式/日志与产物”更清晰，且不引入额外的安装步骤；保持 `python -m mcca...` 的直接运行体验优先。

---

## C) 目标布局（最终目录树）

说明：该目标布局尽量最小化变动，仅做“脚本归位 + 文档归档”。Python 包结构 `mcca/` 保持不动。

```text
multi_cam_cali_apriltag/
  .github/
  config/
  docs/
    layout_migration.md
    notes/
      step3.txt
      step4.txt
  mcca/
    adapters/
    core/
    entry/
    tools/
    __init__.py
  scripts/
    run_calibration_pipeline.sh
  tests/
  images/
  videos/
  results/
  cache/
  pyproject.toml
  README.md
  output.txt
  pytest_exit_code.txt
```

关键决策：
- Python 包根路径：保持 `mcca/` 位于仓库根目录（不引入 `src/`）。
- 入口点位置：
  - Python CLI 入口继续是 `mcca.entry.*`
  - shell 脚本集中到 `scripts/`
- tests/docs/config：保持现状，仅新增 `docs/notes/`。
- 不再保留：根目录的 `run_calibration_pipeline.sh`、`step3.txt`、`step4.txt`（迁移后删除旧路径，不留兼容层）。

---

## D) 迁移映射表（旧路径 -> 新路径）

| 旧路径 | 新路径 | 理由 | 是否需要改 import/引用 |
|---|---|---|---|
| `run_calibration_pipeline.sh` | `scripts/run_calibration_pipeline.sh` | 根目录可执行脚本归位，减少顶层噪音 | 需要：更新文档/脚本调用路径（若存在） |
| `step3.txt` | `docs/notes/step3.txt` | 临时/笔记归档到 docs | 需要：更新文档引用（若存在） |
| `step4.txt` | `docs/notes/step4.txt` | 临时/笔记归档到 docs | 需要：更新文档引用（若存在） |

### 批次拆分计划（mode=apply 时执行）

- batch1（脚本归位，1 个顶层目录迁移 + 1 文件移动）
  - 创建 `scripts/`
  - 移动 `run_calibration_pipeline.sh` -> `scripts/run_calibration_pipeline.sh`
  - 更新 README/文档中可能出现的旧路径（若搜索到）
  - 删除旧路径文件（不留兼容层）

- batch2（文档归档，1 个目录新增 + 2 文件移动）
  - 创建 `docs/notes/`
  - 移动 `step3.txt`、`step4.txt` 到 `docs/notes/`
  - 删除旧路径文件（不留兼容层）

- batch3（预留）
  - 暂无（除非在 batch1/2 搜索中发现额外散落脚本/临时文件需要归位）

---

## E) 全库修引用清单（可搜索可替换）

> 说明：由于本次只移动脚本与文档，不涉及 Python import 路径修改；修引用主要是 README/docs/脚本调用路径。

### 1) 文档/README 中的脚本路径

- 搜索关键词：
  - `run_calibration_pipeline.sh`
- 替换方向：
  - `run_calibration_pipeline.sh` -> `scripts/run_calibration_pipeline.sh`

### 2) 文档中对 step3/step4 文本的引用（如有）

- 搜索关键词：
  - `step3.txt`
  - `step4.txt`
- 替换方向：
  - `step3.txt` -> `docs/notes/step3.txt`
  - `step4.txt` -> `docs/notes/step4.txt`

### 3) CI/脚本调用路径（若存在）

- 搜索关键词：
  - `bash run_calibration_pipeline.sh`
  - `./run_calibration_pipeline.sh`
- 替换方向：
  - 改为 `bash scripts/run_calibration_pipeline.sh` 或 `./scripts/run_calibration_pipeline.sh`

---

## F)（mode=apply）执行搬迁与全库修引用

plan 模式不执行。

---

## G) 清零检查清单（每个 batch 结束必须做）

batch1 完成后：
- 全仓搜索“旧用法”必须为 0（允许在本迁移文档中作为证据出现；新脚本路径 `scripts/run_calibration_pipeline.sh` 属于新用法）：
  - `bash run_calibration_pipeline.sh`
  - `./run_calibration_pipeline.sh`

batch2 完成后：
- 全仓搜索“旧用法”必须为 0（允许在本迁移文档中作为证据出现；新用法是 `docs/notes/step3.txt`、`docs/notes/step4.txt`）：
  - `step3.txt`（不带 `docs/notes/` 前缀）
  - `step4.txt`（不带 `docs/notes/` 前缀）

---

## H) 验证 smoke checklist（Windows 可运行命令）

> 注意：按你的工作流要求，所有命令必须覆盖写入 `./output.txt`（包含 stderr）。

### import 冒烟

PowerShell：
- `python -c "import mcca; import mcca.entry.pipeline" > .\output.txt 2>&1`

### 入口 --help

PowerShell：
- `python -m mcca.entry.pipeline --help > .\output.txt 2>&1`
- `python -m mcca.entry.step5c_world_anchor --help > .\output.txt 2>&1`

（若使用 console_scripts）PowerShell：
- `uv run multi-cam-cali-apriltag-pipeline --help > .\output.txt 2>&1`

### 最小测试

PowerShell：
- `python -m pytest > .\output.txt 2>&1`

### 脚本入口（如安装了 Git Bash）

bash：
- `bash scripts/run_calibration_pipeline.sh > ./output.txt 2>&1`

---

## 批次记录（apply 时必须填写；plan 先给模板）

### batch1
- 变更：
  - 创建 `scripts/`
  - 移动 `run_calibration_pipeline.sh` -> `scripts/run_calibration_pipeline.sh`
  - 修引用：README/docs（若存在）
  - 删除旧文件
- 执行命令与结果摘要（每条命令都要写入 output.txt 并在此记录）：
  - 命令（bash）：`rm -f run_calibration_pipeline.sh > ./output.txt 2>&1`
    - 结果摘要：删除旧路径成功（`output.txt` 为空；根目录已不再存在该文件）。
  - 命令（bash）：`python - <<'PY' > ./output.txt 2>&1 ... PY`
    - 结果摘要：import 冒烟通过；旧脚本文件不存在；新脚本存在；旧用法关键字（`bash run_calibration_pipeline.sh`、`./run_calibration_pipeline.sh`）全仓命中为 0（排除本迁移文档）。
  - 命令（bash）：`python -m mcca.entry.pipeline --help > ./output.txt 2>&1`
    - 结果摘要：`--help` 正常输出 usage/options（详见 `output.txt`）。
  - 命令（bash）：`python -m pytest > ./output.txt 2>&1`
    - 结果摘要：`collected 33 items`，`33 passed`。

### batch2
- 变更：
  - 创建 `docs/notes/`
  - 移动 `step3.txt`、`step4.txt` -> `docs/notes/`
  - 修引用（若存在）
  - 删除旧文件
- 执行命令与结果摘要：
  - 命令（bash）：`mkdir -p docs/notes > ./output.txt 2>&1`
    - 结果摘要：创建目录成功（`output.txt` 为空）。
  - 命令（bash）：`mv step3.txt docs/notes/step3.txt > ./output.txt 2>&1`
    - 结果摘要：移动成功（`output.txt` 为空；根目录 `step3.txt` 不再存在）。
  - 命令（bash）：`mv step4.txt docs/notes/step4.txt > ./output.txt 2>&1`
    - 结果摘要：移动成功（`output.txt` 为空；根目录 `step4.txt` 不再存在）。
  - 命令（bash）：`python - <<'PY' > ./output.txt 2>&1 ... PY`
    - 结果摘要：清零检查通过：根目录 `step3.txt`/`step4.txt` 均不存在；新路径文件均存在；排除 `.git/`、`__pycache__/`、迁移文档与日志本体后，旧引用命中为 0。
  - 命令（bash）：`python -c "import mcca; import mcca.entry.pipeline" > ./output.txt 2>&1`
    - 结果摘要：import 冒烟通过（`output.txt` 为空）。
  - 命令（bash）：`python -m mcca.entry.pipeline --help > ./output.txt 2>&1`
    - 结果摘要：`--help` 正常输出 usage/options（详见 `output.txt`）。
  - 命令（bash）：`python -m pytest > ./output.txt 2>&1`
    - 结果摘要：`collected 33 items`，`33 passed`。

### batch3
- 预留
