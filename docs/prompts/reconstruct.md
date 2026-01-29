你是资深 Python 工程负责人 + 架构重构专家。请基于我仓库的 repo_tree.txt（已在仓库根目录）对代码库做一次工程化重构：目标是“结构清晰、低耦合、高可测、可维护、可复用、可扩展”，同时尽量不改变现有功能与输出格式。

背景（从 repo_tree 可见）：
- 仓库根目录堆了大量 step 脚本：run_calibration_pipeline.py、step1_capture_imgs.py、step2_filter_images.py、step3_intrinsic_apriltag.py、step4_multi_extrinsic_pose_graph.py、step4_stereo_extrinsic.py、step5*.py、step6*.py、suggest_detection_roi.py 等。
- 存在 libs/（包含 camera_wrapper、extrinsics_graph 等）、tests/、config/（含 apriltag_config.json 与 CONFIG_GUIDE.md）、以及大量数据/产物目录：cache/apriltag_detection/*.npz、images/raw|filtered、videos/*.mp4、results/*、__pycache__、.ruff_cache、.pytest_cache，甚至 .venv 也出现在树里。
- 当前问题：目录冗杂、生成物/数据混入 repo、脚本式流程导致耦合高、文件过长（1000+ 行）、模块边界不清晰、文档不完整。

任务 1：目录结构重构（贴合工程实践）
1) 先读 repo_tree.txt + 实际代码（打开 pyproject.toml / README.md / 关键脚本与 libs/）。
2) 给出“目标目录结构（新的 tree）”与“旧 -> 新的映射表（每个文件移动到哪里，为什么）”。
3) 目标结构建议采用标准 Python 工程布局（例如 src/ 包结构 + 明确的 cli/、core/、io/、calibration/、apriltag/、viz/、configs/、scripts/、docs/、tests/）。
4) 明确区分：
   - 源码（必须进 git）
   - 配置（进 git）
   - 测试（进 git）
   - 样例小数据（可选进 git，严格控体积）
   - 大体积数据/缓存/输出（必须 gitignore，并提供生成/下载/放置说明）

任务 2：降低耦合 & 拆分 1000+ 行大文件
1) 识别“职责混杂”的大文件（尤其是 step4 / step5 / detector / utils 类），给出拆分方案：
   - 把“纯算法/数学/图优化”与“IO/文件系统/可视化/CLI 参数解析”分离
   - 把“数据结构（dataclass/TypedDict）”与“流程编排（pipeline）”分离
   - 把“可替换的组件（detector / matcher / solver）”抽象成接口（protocol 或抽象基类），并通过依赖注入在 pipeline 中组装
2) 提供具体落地改造：
   - 把 step*.py 改成“薄入口（thin entrypoint）”：只做 argparse/typer + 调用包内函数；核心逻辑全部下沉到 src/<package>/...
   - 统一日志（logging）、统一路径（pathlib.Path）、统一配置加载（config schema + 校验）
   - 每个模块都有可单测的纯函数边界；tests/ 增补关键路径测试（不依赖大图片/视频）

任务 3：总结代码库，完善 README 和 docs
1) 重写 README.md：包含
   - 项目一句话说明（这是多相机 AprilTag 标定/外参图优化/基座坐标融合的 pipeline）
   - 安装与环境（python 版本来自 pyproject.toml；依赖安装命令）
   - 快速开始（最短命令跑通：从 config -> step1..step6 或 新 CLI）
   - 配置说明（引用 config/CONFIG_GUIDE.md，并补齐缺失）
   - 输出目录与产物说明（results/ 的 json、可视化目录等）
   - 开发指南（lint/format/test）
2) 新增 docs/：
   - docs/architecture.md：模块职责、数据流（step1..6 的输入输出契约）
   - docs/pipeline.md：每一步做什么、输入输出文件格式
   - docs/troubleshooting.md：常见失败点（检测失败、角点顺序、图不连通、尺度漂移等）
   - docs/data_layout.md：images/videos/cache/results 的推荐放置与 gitignore 原则

实现要求（非常重要）：
- 先给出重构计划（分阶段，每阶段可运行、可回滚），再动手改代码；每一步都说明“为什么这样改”和“风险点”。
- 需要你直接在仓库里完成实际重构：创建新目录、移动文件、更新 import、补 __init__.py、更新入口脚本/CLI、更新测试与文档。
- 不要引入“兼容层/过渡 shim”长期共存；老入口如必须保留，只允许做非常薄的 wrapper，并在 README 里标注 deprecated。
- 生成物/大文件必须移出源码路径并加入 .gitignore；同时给出“如何重新生成 results/cache”的命令与说明。
- 修改后必须能通过 pytest（运行全量测试）；如测试依赖大数据，请改成 fixtures 下的小样本或 mock。
- 新增/修改的注释用中文；函数/模块命名遵循清晰的英文工程命名。
- 最终交付物：
  A) 新的目录树（最终版）
  B) 迁移映射表（旧路径 -> 新路径）
  C) 关键模块的职责说明（1-2 句话/模块）
  D) README + docs/ 完整内容
  E) 可运行的入口方式（例如：python -m <package>.cli ... 或 scripts/run_pipeline.py ...）
  F) .gitignore 更新（明确忽略：.venv、__pycache__、.ruff_cache、.pytest_cache、cache/、images/、videos/、results/ 等，并解释原因）

现在开始：先输出“诊断报告 + 目标架构 + 分阶段计划”，然后按计划逐步修改仓库代码与文档（一次只做一个阶段，确保阶段结束时可运行/可测试）。
