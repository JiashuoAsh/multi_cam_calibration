# multi_cam_cali_apriltag

本仓库用于基于 **AprilTag 标定板** 的相机标定，支持：

- 单相机/多相机 **内参**（Step3）
- 双目外参（Step4 stereo）
- 多相机外参（Step4 pose graph）
- 多相机相对机器人底盘坐标系外参（Step5，camera -> base）

> 约定：全工程命名 `A_T_B` 表示 **B -> A** 的齐次变换（点从 B 坐标系变到 A 坐标系）。

### 坐标系/外参/位姿 约定

定义（本仓库统一约定）：记 $A\_T\_B \in SE(3)$ 为从坐标系 B 到坐标系 A 的刚体变换（齐次矩阵形式）。

#### `A_T_B` 的严格含义

- 变换含义：把点从 B 坐标系变到 A 坐标系：

  $$
  p_A = A\_T\_B \cdot p_B
  $$

- 位姿含义：`A_T_B` 等价于 **“坐标系 B 的位姿在坐标系 A 中的表达”**。
  - 也常被写作：<sup>A</sup>T<sub>B</sub>（frame B expressed in frame A）

#### “外参”与“位姿”的习惯约定

术语说明：在几何意义上，“外参”与“位姿”均对应同一个 $SE(3)$ 变换；二者差别主要体现在**默认参考系与默认方向**。

- 若将 A 取为相机坐标系 `C`，B 取为世界/板/底盘等参考系 `W`：
  - `C_T_W`：将 $W$ 中的点变换到 `C` 中（world->camera）。在计算机视觉投影模型中，这一方向通常被称为**相机外参**。
  - $W\_T\_C = (C\_T\_W)^{-1}$：将 `C` 中的点变换到 `W` 中（camera in world）。在机器人/三维图形与 SLAM 语境中，这一方向通常被称为**相机位姿**。

二者完全等价，仅相差一次求逆；因此在沟通时必须显式给出方向（例如 “world->camera” 或 “camera in world”）。

#### OpenCV 的默认判定

OpenCV 的针孔投影模型及其相关 API 采用如下约定：外参 $[R|t]$ 表示将点从“物体/世界坐标系”变换到“相机坐标系”，即 `C_T_W`（world/object->camera）。

- OpenCV 的外参通常写作 $[R|t]$，其含义是把点从“世界/物体坐标系”变到“相机坐标系”：

  $$
  p_C = R \cdot p_W + t
  $$

  其中 $R$ 是 $3\times 3$ 旋转矩阵，$t$ 是 $3\times 1$ 平移向量。

- 用本仓库的记号表示，就是：

  $$
  p_C = C\_T\_W \cdot p_W, \quad C\_T\_W = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}
  $$

- 因此：
  - **“相机外参”（OpenCV 语境里最常见）**：`C_T_W`（world->camera）
  - **“相机位姿”（camera in world）**：$W\_T\_C = (C\_T\_W)^{-1}$

关于平移向量的几何含义：
- $t$ 表示**坐标系 W 的原点在坐标系 C 中的坐标**（world origin expressed in camera）。
- 若 $C_T_W = [R|t]$，则 $W_T_C$ 可显式写为：

  $$
  W\_T\_C = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}
  $$

与 OpenCV 常用 API 的对应关系：
- `cv2.solvePnP(...)` / `cv2.aruco.estimatePoseSingleMarkers(...)` 输出 `rvec, tvec`，满足：
  - 令 $R = \mathrm{Rodrigues}(\mathrm{rvec})$，则 $p_C = R p_O + t$，其中 $O$ 为 `objectPoints` 所在坐标系。
  - 即 `rvec,tvec` 对应 **`C_T_O`**。
- `cv2.projectPoints(objectPoints, rvec, tvec, K, dist)` 使用同一约定：
  - 先做 $p_C = R p_O + t$，再用 $K$ 投影到像素。
- `cv2.calibrateCamera(...)` 返回的 `rvecs,tvecs`（每一帧一组）也是：
  - **pattern/object -> camera**（也就是 `C_T_pattern`）。
- `cv2.stereoCalibrate(...)` 返回的 `R,T`（以及 `E,F`）通常满足：
  - 将点从相机 1 坐标系变到相机 2 坐标系：

    $$
    p_{C2} = R \cdot p_{C1} + T
    $$

  - 用本仓库记号就是 **`C2_T_C1`**。

补充（坐标轴方向）：OpenCV 通常使用相机坐标系 $x$ 向右、$y$ 向下、$z$ 向前（指向镜头前方）。若与机器人/图形学中常见的 $y$ 向上等约定不同，应在坐标系定义层面进行一致化（轴翻转会导致“数值看似合理但物理方向相反”的问题）。

#### 本仓库里最常见的几个例子

- `C_T_T`：TagBoard -> Camera（Step5/Step4 中 PnP 估计出来的“板在相机里的位姿”）
- `B_T_C`：Camera -> Base（Step5 的目标结果，“相机在底盘里的位姿”）
- `Cr_T_Cl`：LeftCam -> RightCam（双目外参，左到右）

> 读法约定：`A_T_B` 读作“B 到 A 的变换”（或“B 在 A 中的位姿表达”）。

---

## 快速开始（推荐：配置驱动 + 一键流水线）

1) 复制模板并修改配置：

- 配置文件：`config/apriltag_config.json`

2) 准备图片数据（两类数据集）：

- **Step2~Step4 数据集（用于内参与相机间外参）**：在 config 的 `image_dataset` 指定
- **Step5 数据集（用于相机->底盘外参）**：在 config 的 `step5_dataset` 指定

3) 运行流水线：

- 仅跑 Step2~Step4：
  - `python run_calibration_pipeline.py --config config/apriltag_config.json`
- 跑 Step1~Step4（含视频抽帧）：
  - `python run_calibration_pipeline.py --all --config config/apriltag_config.json`
- 若启用 `step5_dataset.enabled=true`，且你使用 `--all`，流水线会自动包含 Step5。

产物：
- 每一步日志：`results/pipeline_logs/`
- 汇总报告：`results/pipeline_report.json`

---

## 配置文件关键字段（你只需要改这里）

### 1) `image_dataset`（Step2/3/4 多相机图片输入）

用于指定每个相机的**原始图片位置**，并让 Step2/3/4 自动识别需要标定的相机数量。

常用字段（简化版）：

```json
{
  "image_dataset": {
    "enabled": true,
    "raw_root": "images/raw",
    "filtered_root": "images/filtered",
    "sync": {"key": "stem", "mode": "intersection"},
    "cameras": {
      "cam0": {"raw_dir": "D:/data/raw/cam0"},
      "cam1": {"raw_glob": "D:/data/raw/cam1/*.png"}
    }
  }
}
```

说明：
- `enabled=true` 后，Step2/3/4 会按 `cameras` 自动跑多相机
- `sync.key=stem` 表示按文件名 stem（不含后缀）对齐同一时刻的多路图像（例如 `frame_000120.*`）
- `sync.mode=intersection` 更严格，只使用“所有相机都存在”的帧；适合同步采集

### 2) `step5_dataset`（Step5 多相机图片输入）

Step5 用于求每个相机到机器人底盘坐标系的外参：`B_T_C`（Cam -> Base）。

```json
{
  "step5_dataset": {
    "enabled": true,
    "image_root": "images/step5",
    "cameras": {
      "cam0": {"raw_dir": "D:/data/step5/cam0"},
      "cam1": {"raw_glob": "D:/data/step5/cam1/*.png"}
    }
  }
}
```

如果 `enabled=false`，Step5 默认扫描：
- `images/step5/<cam>/*.png|jpg|jpeg|bmp`

### 3) `board_to_base_transform`（Step5 必需：标定板在底盘坐标系中的固定安装位姿）

Step5 会从该段配置构造 `B_T_T`（TagBoard -> Base），再结合每相机的 PnP 位姿 `C_T_T` 得到：

$$
B\_T\_C = B\_T\_T \cdot (C\_T\_T)^{-1}
$$

这个配置的精度，会直接决定最终“相机->底盘”的精度。

---

## 每个 Step 需要什么数据？会产出什么？

### Step1：从视频抽帧（可选）
- 脚本：`step1_extract_imgs_from_video.py`
- 依赖：config 的 `camera_settings` + `video_extract`
- 输出：`images/raw/<cam>/...`（双目默认 left/right）

### Step2：筛选包含标定板的图片
- 脚本：`step2_filter_images.py`
- 输入：
  - 若 `image_dataset.enabled=true`：来自 `image_dataset.cameras.*.raw_dir/raw_glob`
  - 否则：默认 `images/raw/left` 与 `images/raw/right`
- 输出：
  - `images/filtered/<cam>/...`
  - `results/filter_report.json`
  - 可视化（可选）：`results/visualization/step2_filtering/<cam>/...`

### Step3：内参标定（多相机）
- 脚本：`step3_intrinsic_apriltag.py`
- 输入：优先使用 `images/filtered/<cam>/...`（若为空会回退到 raw）
- 输出：`results/<cam>_intrinsics.json`

### Step4：相机间外参

#### 4A) 双目外参（legacy stereo）
- 脚本：`step4_stereo_extrinsic.py`
- 适用：仅 `left/right`

数据需求（必须）：
- `results/left_intrinsics.json`, `results/right_intrinsics.json`
- `images/filtered/left/` 与 `images/filtered/right/`
  - **成对**、尽量同步的图像（同一时刻左右各一张）
  - 每对图像中，标定板在两张图里都要“看得到且角点稳定”

采集建议：
- 建议 30~100 对图像（越多越稳，但质量比数量更重要）。
- 标定板尽量覆盖画面中心/四角、不同距离、不同倾角。
- 确保左右相机曝光/对焦稳定；避免运动模糊。
- 如果板子在边缘、tag 太小或对比不足，容易导致角点抖动，外参会漂。

质量检查建议：
- 可先用脚本做“共同检测到的 tag 数”统计，剔除低质量 pair：
  - `verify_step4_stereo_calibration.py`
  - 或参考 `step4_stereo_extrinsic.py` 内置的质量分析逻辑

输出：
- `results/stereo_extrinsics.json`（`Cr_T_Cl`：右 <- 左）
- `results/stereo_rectification.json`

#### 4B) 多相机外参（pose graph）
- 脚本：`step4_multi_extrinsic_pose_graph.py`

数据需求（必须）：
- 每个相机一份内参：`results/<cam>_intrinsics.json`
- 每个相机一组筛选后的图片：`images/filtered/<cam>/...`
- **关键：同一时刻的多路图片要能“对齐”**
  - 默认按文件名 `stem` 对齐（例如 `frame_000120.png`）
  - 也就是说：同一帧在不同相机下，文件名（不含后缀）必须一致

采集策略（最重要：保证“连通性”）：
- 位姿图优化不要求“所有相机同时看到板”，但要求整套相机通过“同帧共视”形成连通图。
- 实操上推荐按“链式共视”采集（举例 4 相机 cam0~cam3）：
  1) 采一段：让板同时被 cam0 + cam1 看见（多拍一些帧）
  2) 采一段：让板同时被 cam1 + cam2 看见
  3) 采一段：让板同时被 cam2 + cam3 看见
  这样 cam0~cam3 就连通了，即使从没出现过“4 台同时共视”的帧也没关系。

采集建议（经验值）：
- 每条“共视边”（例如 cam1-cam2）建议至少 15~30 个有效共视帧。
- 板子姿态要有变化（平移+旋转），避免所有帧都几乎同一个姿态（会让约束退化）。
- 尽量避免只在画面边缘/只看到少量 tag 的帧；tag 越多、分布越开，PnP 越稳。

常见问题与排查：
- **只解出一部分相机**：通常是相机图不连通（没有足够的共视边）。
- **误差大/抖动**：优先检查
  - 筛图是否把“低质量检测帧”混进来了
  - 文件名同步是否正确（错配会直接引入错误约束）
  - 是否需要开启/调整 ROI 或 auto ROI 提升检测稳定性

输出：
- `results/multi_camera_extrinsics.json`（`T_cam_from_ref`：Cam <- Ref）
- `results/multi_camera_pose_graph_report.json`

### Step5：相机 -> 底盘外参（多相机）
- 脚本：`step5b_camera_to_base.py`

数据需求（必须）：
- `board_to_base_transform`（config 必须配置，且物理测量要靠谱）
  - 这定义了 `B_T_T`（TagBoard -> Base）
- Step5 图片（每个相机一组）：
  - 由 `step5_dataset` 指定（推荐）
  - 或默认 `images/step5/<cam>/...`
- 每个相机内参：`results/<cam>_intrinsics.json`

数据需求（可选，但强烈推荐）：
- Step4 相机间外参：
  - `results/multi_camera_extrinsics.json`（多相机）或 `results/stereo_extrinsics.json`（双目）
  - 作用：当某些相机缺少 Step5 图片时，可以从已标定相机把 `B_T_C` 传播过去（减少必须逐个采集的工作量）

采集流程（推荐做法）：
1) **先把标定板“刚性固定”到机器人底盘上**（板子相对底盘不能有松动）。
2) 用尺/量具把 `board_to_base_transform.translation` 和参考点定义清楚：
   - `translation_reference` / `translation_reference_point_in_T_m` 的含义要与你的测量一致
   - 这是 Step5 精度的主要来源，别在这里“估个大概”
3) 对每个相机采集图片：
   - 每个相机建议 20~50 张有效图（越多越稳，优先质量）
   - 板子在画面中尽量清晰、tag 数量尽量多、分布尽量覆盖板面
   - 尽量避免运动模糊/过曝/欠曝
   - **Step5 不要求多相机同步**：你可以逐个相机采集（甚至不同时间），只要板子相对底盘不动即可
4) 运行 Step5b 得到 `B_T_C`。

质量检查建议：
- 跑 `verify_step5_result.py --camera <cam>` 做快速 sanity check（看重投影/几何一致性等）。
- 如果你有多次采集（不同批次/不同板位置），建议用 Step6 融合：
  - `step6_fuse_step5_results.py` 输出 `results/camera_to_base_fused.json`

输出：
- `results/camera_to_base.json`
  - `B_T_C`: `{cam: 4x4}`（核心结果，Cam -> Base）

小贴士：
- 如需排查坐标系/单位/传播逻辑，可加 `--verbose` 输出更多中间过程。

### Step6：融合多次 Step5 结果（可选）
- 脚本：`step6_fuse_step5_results.py`
- 输入：多份 `camera_to_base.json`（可给文件/目录/glob）
- 输出：
  - `results/camera_to_base_fused.json`
  - `results/camera_to_base_fusion_report.json`

---

## 常见坑 / 注意事项（强烈建议看一遍）

1) **标定板尺寸一定要准**：`apriltag_board.tag_size/tag_spacing` 单位是 mm，误差会直接体现在尺度上。

2) **多相机 Step4 需要“连通”**：不是每台相机都要同时看到板，但整套相机必须通过“同帧共视”形成连通图，否则 pose graph 只能解出部分相机。

3) **文件名同步策略**：
- Step4（多相机）默认按文件名 stem 对齐；请确保不同相机同一时刻的图片 stem 相同。

4) **Step5 的物理测量是大头**：
- `board_to_base_transform.translation` 的参考点（Tag0 / 网格中心 / 板中心）要和配置一致。
- 建议在 `CONFIG_GUIDE.md` 里按“translation_reference”说明严格设置。

5) **光照/运动模糊**：AprilTag 角点不稳会导致 PnP 抖动；Step5 建议标定板完全固定且清晰。

6) **单位**：
- Step3/Step4/Step5 内部统一使用米（m），但标定板点初始配置是 mm，会在代码里转换。

---

## 进一步阅读

- 配置详解：`config/CONFIG_GUIDE.md`
- 多相机外参报告：`results/multi_camera_pose_graph_report.json`
- Step2 筛图可视化：`results/visualization/step2_filtering/`
