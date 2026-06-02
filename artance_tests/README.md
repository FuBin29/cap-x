# ArtAnce Module Tests

本目录用于放置面向 ArtAnce/RLBench 任务的模块级视觉测试脚手架。工作目录默认是：

```bash

cd cap-x
source .venv/bin/activate

cd /home/fubin/projects/artance/cap-x/artance_tests
```

目标是给定任务采集到的 RGB-D 观测，逐步验证 cap-x 已提供或即将提供的视觉/几何服务，例如：

- SAM3 部件分割。
- 点云重建。
- 像素点 2D-3D 转换。
- 任务目录内的结果可视化与记录。

## 代码架构

通用模块测试能力放在 `common/`，具体 task 目录只保留该任务自己的默认数据、prompt、输出目录和测试入口。这样 close drawer、open drawer、pick up cup 等任务可以复用同一套 SAM/点云/可视化逻辑。

```text
artance_tests/
  common/
    paths.py              # 仓库路径和 cap-x 导入路径
    pointcloud_reconstruction.py # RGB-D 点云重建、SAM3 部件 mask 上色、PLY/NPZ 输出
    sam3_runner.py        # SAM3 客户端加载、text prompt 测试、summary 输出
    sam3_point_selector.py # 给定图像坐标，从 SAM3 text-prompt 候选 masks 中选择最近实例
    vlm_contact_point.py  # 通用 VLM 图片+prompt 打包、接触点解析、summary 输出
    visualization.py      # mask/box overlay 可视化和文件保存
  close_drawer/
    pointcloud_reconstruction_test.py # close drawer 的 RGB-D + SAM3 mask 点云重建 CLI
    sam3_test.py          # close drawer 的 task-specific CLI
    sam3_point_selection_test.py # close drawer 中按接触点选择 drawer/handle 实例
    contact_point_test.py # close drawer 的 VLM 接触点预测 CLI
    outputs/
      sam3/
      contact_point/
```

约定：

- `common/` 不写 task-specific 默认路径、prompt 或任务语义。
- `<task>/` 目录可以放多个入口脚本，例如 `sam3_test.py`、`pointcloud_test.py`。
- `<task>/outputs/` 视为本地实验产物，不应提交大规模图片、点云或中间结果。
- 新增 task 时优先复制薄入口脚本，再调用 `common/` 的通用函数。

## 2026-05-25: Close Drawer SAM3 测试

已新增 `close_drawer/sam3_test.py`，用于读取 close drawer 的 wrist RGB 图，并调用 `common.sam3_runner` 完成 SAM3 text-prompt 分割和可视化保存。

默认输入图像：

```text
/home/fubin/projects/artance/RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_rgb.png
```

默认输出目录：

```text
/home/fubin/projects/artance/cap-x/artance_tests/close_drawer/outputs/sam3/
```

运行前需要先启动 SAM3 服务：

```bash
cd /home/fubin/projects/artance/cap-x
uv run --no-sync --active python -m capx.serving.launch_sam3_server --device cuda --port 8114
```

然后在本目录运行测试：

```bash
cd /home/fubin/projects/artance/cap-x/artance_tests
uv run --no-sync --active python close_drawer/sam3_test.py --no-show
```

也可以指定 prompt 和图像：

```bash
uv run --no-sync --active python close_drawer/sam3_test.py \
  --image /home/fubin/projects/artance/RLBench/tests/close_drawer/visualizations/wrist_obs_replayed_to_frame_051_rgb.png \
  --prompt "drawer handle" \
  --prompt "drawer"
```

输出内容：

- `*_grid.png`: 原图、top-k box 和 mask overlay 网格。
- `*_overlay_*.png`: 每个检测结果的单独 overlay。
- `*_mask_*.png`: 二值 mask 可视化。
- `*_mask_*.npy`: 原始 bool mask。
- `summary.json`: 本次运行配置、prompt、box、score 和输出文件索引。

备注：当前脚本只做 SAM3 text prompt 闭环；point prompt、RGB-D 点云和 2D-3D 转换会继续作为 `common/` 通用模块补齐，并由具体 task 入口调用。


## 2026-05-25: Close Drawer VLM 接触点预测

已新增 `common/vlm_contact_point.py` 和 `close_drawer/contact_point_test.py`，用于通过 cap-x 提供的 OpenAI-compatible VLM 服务预测 close drawer 任务最适合的 2D 接触点。

默认输入图像：

```text
/home/fubin/projects/artance/RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_rgb.png
```

默认输出目录：

```text
/home/fubin/projects/artance/cap-x/artance_tests/close_drawer/outputs/contact_point/
```

运行前需要先启动 cap-x VLM/OpenRouter proxy 服务。示例：

```bash
cd /home/fubin/projects/artance/cap-x
uv run --no-sync --active python -m capx.serving.openrouter_server --key-file .openrouterkey --port 8110
```

然后在本目录运行接触点预测：

```bash
cd /home/fubin/projects/artance/cap-x/artance_tests
uv run --no-sync --active python close_drawer/contact_point_test.py
```

可指定模型、服务地址和 prompt 坐标顺序：

```bash
uv run --no-sync --active python close_drawer/contact_point_test.py \
  --model gemini-2.5-pro \
  --server-url http://127.0.0.1:8110/chat/completions \
  --prompt-order auto
```

坐标约定：

- 默认不传 `max_tokens`，让 VLM 服务/provider 自行决定输出预算。Gemini 2.5/3.x 这类模型可能会消耗 hidden reasoning token；如果手动设置得太小，OpenAI-compatible 返回可能出现 `finish_reason: "length"` 且 `message.content` 为空。
- `--prompt-order auto`: Gemini 模型默认要求 VLM 输出 normalized `[y, x]`，非 Gemini 模型默认要求输出 normalized `[x, y]`。
- `--prompt-order xy`: 强制 prompt 要求输出 `[x, y]`。
- `--prompt-order yx`: 强制 prompt 要求输出 `[y, x]`。
- 无论 VLM prompt 使用哪种顺序，模块内部统一返回 `normalized_xy` 和 `pixel_xy`，均为 `[x, y]` 顺序。

输出内容：

- 终端打印 `prompt_order`、VLM 原始坐标、统一后的 normalized `[x, y]`、像素 `[x, y]`。
- `close_drawer/outputs/contact_point/<image_stem>/summary.json`: 本次配置、task prompt spec、系统 prompt、原始响应和解析后的坐标。

### 通用 Prompt Spec 用法

`common/vlm_contact_point.py` 不写具体任务目标或规则，只提供通用 prompt 模板、图片打包、VLM 调用和坐标解析。每个 task 需要在自己的入口脚本中定义 `VlmContactPointPromptSpec`：

```python
from common.vlm_contact_point import VlmContactPointConfig, VlmContactPointPromptSpec

CLOSE_DRAWER_PROMPT_SPEC = VlmContactPointPromptSpec(
    task_goal=(
        "Analyze the provided image and find the single best contact point for a robot "
        "to close the visible drawer."
    ),
    contact_rules=(
        "Choose exactly one point on the drawer, preferably on the handle or a rigid front surface that can be pushed safely to close the drawer.",
        "The point must lie on the visible drawer, not on the background, cabinet frame, robot, or floor.",
        "If the handle is visible, prefer the center of the handle or the most stable visible part of it.",
        "If the handle is occluded or absent, choose the best visible point on the drawer front suitable for pushing inward.",
    ),
    user_instruction="Find the single best contact point for the close drawer task.",
)

cfg = VlmContactPointConfig(
    image=image_path,
    output_dir=output_dir,
    prompt_spec=CLOSE_DRAWER_PROMPT_SPEC,
    model="gemini-2.5-pro",
    server_url="http://127.0.0.1:8110/chat/completions",
)
```

新增其它任务时，优先复用 `common.vlm_contact_point.query_vlm_contact_point` 和 `save_vlm_contact_point_summary`，只在 `<task>/contact_point_test.py` 中替换 `task_goal`、`contact_rules`、默认图像和输出目录。


## 2026-05-25: Close Drawer 接触点到 SAM3 实例选择

已新增 `common/sam3_point_selector.py` 和 `close_drawer/sam3_point_selection_test.py`，用于测试：

```text
VLM 预测接触点 pixel_xy
  -> SAM3 分割 drawer handle / drawer 候选实例
  -> 对每个候选 mask 计算 point 到 mask 的最近距离
  -> 分别选出离接触点最近的 handle 和 drawer
```

如果已经运行过 `close_drawer/contact_point_test.py` 并得到 contact summary，可以直接读取其中的 `result.pixel_xy`：

```bash
uv run --no-sync --active python close_drawer/sam3_point_selection_test.py --no-show
```

也可以手动指定图像坐标，坐标顺序是像素空间 `[x, y]`：

```bash
uv run --no-sync --active python close_drawer/sam3_point_selection_test.py \
  --point 444 187 \
  --prompt "drawer handle" \
  --prompt "drawer" \
  --no-show
```

输出目录默认是：

```text
/home/fubin/projects/artance/cap-x/artance_tests/close_drawer/outputs/sam3_point_selection/
```

每个 prompt 会保存一张 `*_point_selection.png`，黄色十字是输入接触点，青色叉是该 prompt 下最近 mask 像素，绿色框是选中的 SAM3 候选实例；`point_selection_summary.json` 记录 rank、score、mask 像素数、是否包含接触点、最近距离和最近 mask 像素坐标。

## 2026-05-25: RGB-D 点云重建和 SAM3 部件上色

已新增 `common/pointcloud_reconstruction.py` 和 `close_drawer/pointcloud_reconstruction_test.py`，用于把 RGB-D 观测反投影成点云，并把 SAM3 输出的多个部件 mask 映射成不同颜色。

实现参考了 cap-x 里已有的点云工具：

- `capx.utils.depth_utils.depth_to_pointcloud`: 根据 depth 和 camera intrinsics 生成相机坐标系点云。
- `capx.utils.depth_utils.depth_color_to_pointcloud`: 根据 RGB-D 生成带颜色点云。
- `capx/envs/simulators/robosuite_handover.py` 中的用法：如存在 `pose_mat`，将相机坐标点云左乘外参矩阵转换到世界坐标。

默认输入：

```text
RGB:   /home/fubin/projects/artance/RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_rgb.png
Depth: /home/fubin/projects/artance/RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_depth.npy
Info:  /home/fubin/projects/artance/RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_info.json
SAM3:  /home/fubin/projects/artance/cap-x/artance_tests/close_drawer/outputs/sam3/wrist_reference_frame_051_rgb/summary.json
```

默认会从 SAM3 summary 中读取每个 prompt 的 rank 1 mask，例如 `drawer handle` 和 `drawer`，并给不同部件设置不同颜色：

```bash
cd /home/fubin/projects/artance/cap-x/artance_tests
uv run --no-sync --active python close_drawer/pointcloud_reconstruction_test.py
```

如果只想快速检查链路，可以先降采样：

```bash
uv run --no-sync --active python close_drawer/pointcloud_reconstruction_test.py \
  --subsample-factor 4
```

常用参数：

- `--sam3-summary <summary.json>`: 从 `close_drawer/sam3_test.py` 保存的 summary 中自动读取 mask。
- `--rank 1 --rank 2`: 读取指定 SAM3 rank 的 mask；默认只读 rank 1。
- `--mask name=/path/to/mask.npy`: 手动传入部件 mask，可重复指定多个部件。
- `--mask name=/path/to/mask.npy:255,0,0`: 手动传入 mask 并指定该部件颜色。
- `--output-frame camera`: 输出相机坐标系点云，默认值。
- `--output-frame world`: 用 info/pose 中的 4x4 camera extrinsics 转到世界坐标。
- `--intrinsics <path>`: 手动指定 3x3 intrinsics，支持 `.npy`、`.npz`、`.json` 或文本矩阵。
- `--pose <path>`: 手动指定 4x4 camera pose/extrinsics，支持 `.npy`、`.npz`、`.json` 或文本矩阵。

输出目录默认是：

```text
/home/fubin/projects/artance/cap-x/artance_tests/close_drawer/outputs/pointcloud/
```

输出内容：

- `*_sam3_parts.ply`: ASCII PLY 点云，包含 `x y z red green blue label`，可用 CloudCompare、MeshLab 或 Open3D 读取。
- `*_sam3_parts.npz`: 压缩 NumPy 数据，包含 `points`、`colors_uint8`、`labels`、`valid_pixel_xy` 和 `part_names`。
- `pointcloud_summary.json`: 本次输入、mask、label id、每个部件点数和输出路径。

坐标和标签约定：

- 点云反投影沿用 `capx.utils.depth_utils.depth_to_pointcloud` 的 intrinsics 约定；当前 RLBench wrist metadata 中 intrinsics 的 `fx/fy` 为负值，模块保持原值使用。
- `label=0` 表示背景；`label=1..N` 对应输入的 SAM3 部件 mask 顺序。
- mask 发生重叠时由 `--mask-overlap-policy` 决定颜色和 label 归属；默认 `first-wins` 会保留先写入的小部件，`last-wins` 会让后写入 mask 覆盖前面 mask。

### 重建 VLM 接触点对应的 SAM3 多部件 batch

已支持从 `sam3_point_selection_test.py` 产出的 `point_selection_summary.json` 读取同一批 prompt 的选中 rank，再反查 `sam3_test.py` 保存的 mask 文件，并只输出这些 mask 内的 RGB-D 点云。

当前 close drawer 的接触点为像素 `[444, 187]`。点选择 summary 中：

- `drawer handle`: rank 1，`contains_point=true`。
- `drawer`: rank 3，离接触点最近的 drawer mask。

不传 `--selected-prompt` 时，会加载 `point_selection_summary.json` 中所有 prompt 的选中 mask，也就是把 `drawer handle` 和 `drawer` 作为一个 batch 同时渲染：

```bash
uv run --no-sync --active python close_drawer/pointcloud_reconstruction_test.py \
  --point-selection-summary close_drawer/outputs/sam3_point_selection/wrist_reference_frame_051_rgb/point_selection_summary.json \
  --mask-only \
  --mask-overlap-policy first-wins \
  --output-dir close_drawer/outputs/pointcloud_selected_parts
```

本次验证输出：

```text
points: 93836
drawer handle: 4322 point(s)
drawer: 89514 point(s)
background: 0 point(s)
Saved summary: /home/fubin/projects/artance/cap-x/artance_tests/close_drawer/outputs/pointcloud_selected_parts/wrist_reference_frame_051_rgb/pointcloud_summary.json
```

新增参数：

- `--point-selection-summary <point_selection_summary.json>`: 读取按接触点选中的 SAM3 mask batch。
- `--selected-prompt "drawer handle"`: 只加载指定 prompt 的选中 mask；可重复传入多个 prompt。不传则加载 summary 中全部 prompt。
- `--mask-only`: 只保存 mask 内有效 depth 点，输出点云中背景点数应为 0。
- `--mask-overlap-policy first-wins`: 多个 mask 重叠时保留先写入的部件颜色和 label，适合让 handle 这类小部件不被 drawer 大 mask 覆盖；`last-wins` 可恢复后写入 mask 覆盖前面 mask 的行为。

## 2026-05-25: 部件相邻区域平面拟合和法线估计

已新增 `common/part_adjacency_plane.py` 和 `close_drawer/part_adjacency_plane_test.py`，用于从点云中分析两个部件的邻近区域：

```text
部件顺序来自 task_configs.py 的 sam3_prompts，例如 close_drawer 是 (drawer handle, drawer)。
内部会把第二个部件作为 part A、第一个部件作为 part B：
  -> 找出所有距离第一个部件点云小于 --neighbor-radius 的第二个部件点
  -> 对这些第二个部件邻域点做 SVD 平面拟合
  -> 计算两个部件的接触邻域中心
  -> 在中心点记录拟合平面的法线，并让法线方向从 sam3_prompts[1] 指向 sam3_prompts[0]
```

默认输入使用 `pointcloud_reconstruction_test.py --mask-only` 生成的 selected parts 点云：

```text
/home/fubin/projects/artance/cap-x/artance_tests/close_drawer/outputs/pointcloud_selected_parts/wrist_reference_frame_051_rgb/wrist_reference_frame_051_rgb_camera_sam3_parts.npz
```

运行：

```bash
cd /home/fubin/projects/artance/cap-x/artance_tests
uv run --no-sync --active python close_drawer/part_adjacency_plane_test.py
```

常用参数：

- `--pointcloud-npz <path>`: 指定由 `pointcloud_reconstruction_test.py` 生成的 `.npz` 点云。
- `--part-a`: 法线起点和用于拟合平面的主部件；默认 `sam3_prompts[1]`，close drawer 中是 `drawer`。
- `--part-b`: 法线终点和邻接/接触部件；默认 `sam3_prompts[0]`，close drawer 中是 `drawer handle`。
- `--neighbor-radius 0.025`: 邻近范围阈值，单位与点云坐标一致；默认 2.5cm。
- `--output-dir <dir>`: 输出目录，默认 `close_drawer/outputs/part_adjacency_plane/`。
- `--max-visualization-points 120000`: 可视化 PLY 中原始点云的最大随机采样点数，几何标记会完整保留。

输出内容：

- `part_adjacency_plane_summary.json`: 记录输入配置、部件 label id、邻域点数、接触区域中心、平面中心、单位法线、平面方程 `ax + by + cz + d = 0` 和拟合误差。
- `*_adjacency_plane_visualization.ply`: 带可视化几何的 ASCII PLY 点云，可用 CloudCompare、MeshLab 或 Open3D 读取。

可视化 PLY 中的额外 label：

- `label=-1`: 黄色采样点表示拟合平面。
- `label=-2`: 白色球表示接触区域中心。
- `label=-3`: 绿色箭头表示接触中心处的平面法线方向。

## 2026-05-25: 平移关节末端移动目标生成

已新增统一的 `common/end_effector_motion.py`，并为 `close_drawer` 和 `push_button` 提供 `end_effector_motion_test.py` wrapper。该模块用于把上一步平面拟合得到的接触点和法线转换成可执行的平移关节末端目标：

```text
part_adjacency_plane_summary.json 中的 camera-frame contact_center / plane_normal
  -> 读取 source pointcloud summary 记录的 frame info
  -> 用 wrist_camera_extrinsics 转成 world frame
  -> 保留当前 gripper 姿态并转换为 quaternion_wxyz
  -> motion_direction = normal_direction_sign * plane_normal
  -> 生成 approach_position、target_position、final_position
  -> 可选先 close_gripper()
  -> 对应 cap-x API: solve_ik(position, quaternion_wxyz) -> move_to_joints(joints)
```

`normal_direction_sign`、`approach_distance`、`target_standoff`、`travel_distance` 和 `close_gripper_before_motion` 是 task 超参，定义在 `common/task_configs.py` 的 `prismatic_motion` 中。当前 `close_drawer` 和 `push_button` 都使用 `normal_direction_sign=-1.0`，即沿 `-plane_normal` 方向推进，并默认先关闭 gripper。

运行示例：

```bash
cd /home/fubin/projects/artance/cap-x/artance_tests
uv run --no-sync --active python close_drawer/end_effector_motion_test.py
uv run --no-sync --active python push_button/end_effector_motion_test.py
```

默认输出：

```text
<task>/outputs/end_effector_motion/<episode_key>/<frame>/end_effector_motion_target.json
```

常用参数：

- `--plane-summary <path>`: 指定 `part_adjacency_plane_summary.json`。
- `--frame-info <path>`: 手动指定包含 camera extrinsics 和 `gripper_pose` 的 frame metadata；默认从 source pointcloud summary 的 `config.info` 自动读取。
- `--input-frame auto|camera|world`: 接触点/法线所在坐标系；默认 `auto`，会读取 source pointcloud summary 的 `output_frame`。
- `--normal-direction-sign -1`: motion direction 相对 `plane_normal` 的符号，两个平移关节任务当前默认都是 `-normal`。
- `--approach-distance <meters>`: approach pose 相对接触点、沿运动反方向偏移的距离。
- `--target-standoff <meters>`: near-contact target pose 相对接触点、沿运动反方向保留的距离。
- `--travel-distance <meters>`: 从 target pose 沿 motion direction 继续推进的距离。
- `--close-gripper-before-motion/--no-close-gripper-before-motion`: 是否在控制序列最前面加入 `close_gripper()`。
- `--orientation-mode keep-current`: 默认保留当前 gripper 姿态；RLBench/PyRep 的 `gripper_pose` 是 `[x, y, z, qx, qy, qz, qw]`，模块会转换为 cap-x control API 使用的 `[w, x, y, z]`。

控制 API 对接方式参考 `capx/integrations/franka/control_reduced.py`。`end_effector_motion_target.json` 采用统一 schema，RLBench 本地执行侧只需要读取 `motion_steps`：

```python
import json
import numpy as np

target = json.loads(open("close_drawer/outputs/end_effector_motion/.../end_effector_motion_target.json").read())

if target["close_gripper_before_motion"]:
    close_gripper()

for step in target["motion_steps"]:
    position = np.asarray(step["position"], dtype=np.float64)
    quat = np.asarray(step["quaternion_wxyz"], dtype=np.float64)
    joints = solve_ik(position, quat)
    move_to_joints(joints)
    if step["name"] in set(target["close_gripper_after_steps"]):
        close_gripper()
```

注意：当前脚本是目标生成和 dry-run 验证，不会直接连接 RLBench 或移动机械臂。实际执行前建议先只移动到 `approach_position`，确认 IK 可解、路径不碰撞，再移动到 `target_position` 和 `final_position`。

### 本地 RLBench 直接执行末端目标

如果要绕过 cap-x HTTP RLBench adapter，直接使用本地 RLBench/PyRep 环境执行上面生成的 `end_effector_motion_target.json`，应按统一 schema 中的 `motion_steps`、`close_gripper_before_motion` 和 `close_gripper_after_steps` 执行。平移和旋转关节任务在 RLBench 本地侧使用同一个读取/执行接口。

输出目录建议沿用：

```text
/home/fubin/projects/artance/RLBench/tests/close_drawer/visualizations/end_effector_motion/
```

注意运行前需要激活能 import `rlbench` 和 `pyrep` 的本地 RLBench 环境；当前普通 host Python 缺少 `gymnasium`/`pyrep`，不能直接启动 CoppeliaSim。

## 2026-05-26: Episode 批量模块测试接口

`episodes.txt` 中列出的 RGB frame 现在可以直接驱动批量模块测试。新增的通用入口是：

```bash
cd /home/fubin/projects/artance/cap-x/artance_tests
uv run --no-sync --active python run_task_module.py contact_point --task close_fridge \
  --episode-line /home/fubin/projects/artance/RLBench/data/RLBench-data/close_fridge/variation0/episodes/episode0/wrist_rgb/35.png
```

目前批量脚本默认只打开四个视觉/几何前置模块：

- `contact_point`: VLM 预测 2D 接触点。
- `sam3`: SAM3 text prompt 分割。
- `sam3_point_selection`: 用接触点筛选最近的 SAM3 实例。
- `pointcloud`: 基于 RGB-D、SAM3 selected masks 重建点云。

每个 task 都有自己的目录和薄入口脚本，例如：

```text
close_fridge/contact_point_test.py
close_fridge/sam3_test.py
close_fridge/sam3_point_selection_test.py
close_fridge/pointcloud_reconstruction_test.py
```

这些入口复用 `common/task_cli.py` 和 `common/task_configs.py` 中的 task-specific prompt、SAM3 prompts 和 episode 默认选择。

批量脚本：

```bash
# 默认跑 episodes.txt 中所有 task/episode 的四个前置模块
bash run_batch_module_tests.sh

# 只打印命令，不执行
DRY_RUN=1 bash run_batch_module_tests.sh

# 指定 task、episode 和模块
TASKS="close_drawer close_fridge" \
EPISODES="close_drawer:0:3:51 close_fridge:0:0:35" \
MODULES="contact_point sam3 sam3_point_selection pointcloud" \
bash run_batch_module_tests.sh

TASKS="close_microwave push_button toilet_seat_down" \
MODULES="contact_point sam3 sam3_point_selection pointcloud" \
bash run_batch_module_tests.sh

TASKS="close_drawer push_button" \
MODULES="part_adjacency_plane end_effector_motion" \
bash run_batch_module_tests.sh

TASKS="close_drawer push_button toilet_seat_down" \
MODULES="contact_graspnet_pose" \
bash run_batch_module_tests.sh

TASKS="toilet_seat_down" \
MODULES="sam3 sam3_point_selection pointcloud contact_graspnet_pose" \
bash run_batch_module_tests.sh

TASKS="toilet_seat_down" \
MODULES="contact_guided_remote_rotation end_effector_motion" \
bash run_batch_module_tests.sh --guide-source graspnet --axis-fit-method gap_ransac

TASKS="close_microwave close_fridge" \
MODULES="implicit_door_remote_rotation end_effector_motion" \
bash run_batch_module_tests.sh
```

`--guide-source` ：

- `graspnet`
- `vlm`

轴拟合方法 `--axis-fit-method` 对比：

- `gap_ransac`：默认方法。先找远端 part A 点到 part B 的最近邻配对，用两者 midpoint 表示门/框缝隙中心，再用 RANSAC 去除离群 midpoint，最后 SVD refine 成直线。
- `gap_svd`：同样使用缝隙 midpoint，但直接 SVD/PCA 拟合，适合干净分割和较少离群点。
- `neighbor_svd`：旧方法。把远端 part A 邻近点和 part B 邻近点直接合并后 SVD/PCA 拟合，作为 baseline 保留。



输出按 task、module、episode key 分层，形如：

```text
close_fridge/outputs/contact_point/variation0_episode0_frame035_wrist/35/summary.json
close_fridge/outputs/sam3/variation0_episode0_frame035_wrist/35/summary.json
close_fridge/outputs/sam3_point_selection/variation0_episode0_frame035_wrist/35/point_selection_summary.json
close_fridge/outputs/pointcloud/variation0_episode0_frame035_wrist/35/pointcloud_summary.json
```

点云重建对 `episodes.txt` 的离线 RLBench depth PNG 做了兼容：运行时会从同 episode 的 `low_dim_obs.pkl` 导出一份 frame info JSON，并用 camera near/far 把 RLBench normalized depth PNG 还原为米制 depth。该步骤只在实际运行 `pointcloud` 模块时发生。

`run_batch_module_tests.sh` 已支持按需执行 `part_adjacency_plane`、`contact_guided_remote_rotation`、`implicit_door_remote_rotation` 和 `end_effector_motion`；它们仍不在默认 `MODULES` 中，需要显式启用。`contact_guided_remote_rotation` 可通过 bash 参数 `--guide-source {graspnet|vlm}` 和 `--axis-fit-method {gap_ransac|gap_svd|neighbor_svd}` 配置，也可用环境变量 `GUIDE_SOURCE`、`AXIS_FIT_METHOD` 设置默认值。冰箱/微波炉的隐式门轴分支使用 `--door-guide-source {vlm|graspnet}` 或 `DOOR_GUIDE_SOURCE`，默认 `vlm`。脚本内预设 task 关节类型：`close_drawer` 和 `push_button` 使用 prismatic 分支；`close_fridge`、`close_microwave` 和 `toilet_seat_down` 使用 revolute 分支。旋转分支默认生成 closed-gripper push 轨迹，也可在 task wrapper 中切回旧的 GraspNet grasp-follow 轨迹。两类分支最终都由各自的 `end_effector_motion_test.py` 写出统一的 `end_effector_motion_target.json`。

## 2026-05-26: 交互参数推理和估计

平移关节任务继续沿用 close drawer 的“部件邻近区域拟合平面”设计。`close_drawer` 和 `push_button` 都会读取 `task_configs.py` 中的 `sam3_prompts` 部件顺序，并统一设置法线方向为 `sam3_prompts[1] -> sam3_prompts[0]`。内部实现上，`part_a` 是第二个部件、`part_b` 是第一个部件；代码在 part A 中寻找靠近 part B 的点集，拟合局部平面并把法线方向朝向 part B，输出 JSON 和 PLY 可视化。

```bash
uv run --no-sync --active python close_drawer/part_adjacency_plane_test.py \
  --pointcloud-npz close_drawer/outputs/pointcloud/variation0_episode3_frame051_wrist/51/51_camera_sam3_parts.npz \
  --source-summary close_drawer/outputs/pointcloud/variation0_episode3_frame051_wrist/51/pointcloud_summary.json

uv run --no-sync --active python push_button/part_adjacency_plane_test.py \
  --pointcloud-npz push_button/outputs/pointcloud/variation0_episode2_frame036_wrist/36/36_camera_sam3_parts.npz \
  --source-summary push_button/outputs/pointcloud/variation0_episode2_frame036_wrist/36/pointcloud_summary.json
```

旋转关节先实现了 `toilet_seat_down` 的接触点引导远端旋转流程。guide 支持两种来源：默认读取 Contact-GraspNet summary 中最高分候选的 best contact point；也可以切回原始 VLM 2D 接触点。

```text
Contact-GraspNet best 接触点 或 VLM 接触点
  -> part A ("toilet lid") 上最近 3D 点 a
  -> part A 中远离 a 的远端点集 C
  -> 在 C 与 part B ("toilet seat") 的点云缝隙中采样 midpoint
  -> 用 gap_ransac / gap_svd 拟合旋转轴 d
  -> 分别测试 d 的正/负小角度旋转与非 part A 全部点云的邻近冲突
  -> 选择低冲突方向，输出 a 绕 d 旋转 40 度的 waypoints
```

轴拟合方法可通过 `--axis-fit-method` 对比：

- `gap_ransac`：默认方法。先找远端 part A 点到 part B 的最近邻配对，用两者 midpoint 表示门/框缝隙中心，再用 RANSAC 去除离群 midpoint，最后 SVD refine 成直线。
- `gap_svd`：同样使用缝隙 midpoint，但直接 SVD/PCA 拟合，适合干净分割和较少离群点。
- `neighbor_svd`：旧方法。把远端 part A 邻近点和 part B 邻近点直接合并后 SVD/PCA 拟合，作为 baseline 保留。

入口脚本：

```bash
uv run --no-sync --active python toilet_seat_down/contact_guided_remote_rotation_test.py \
  --episode-key variation0_episode2_frame015_wrist \
  --frame-stem 15

uv run --no-sync --active python toilet_seat_down/contact_guided_remote_rotation_test.py \
  --episode-key variation0_episode1_frame000_wrist \
  --frame-stem 0 \
  --guide-source vlm \
  --axis-fit-method gap_ransac


# 切回原 VLM 2D guide，并使用旧轴拟合作为 baseline
uv run --no-sync --active python toilet_seat_down/contact_guided_remote_rotation_test.py \
  --episode-key variation0_episode2_frame015_wrist \
  --frame-stem 15 \
  --guide-source vlm \
  --axis-fit-method neighbor_svd
```

碰撞方向评分默认使用 `--collision-environment all_non_part_a`，即旋转后的 part A 与输入点云中除 part A 之外的全部点比较；`--collision-environment part_b` 保留为旧行为/消融对照。为让该检查真正覆盖环境点，batch 脚本对 revolute 任务运行 `pointcloud` 时默认保留完整场景点云，仅 prismatic 任务继续使用 `--mask-only`。

主要输出：

- `toilet_seat_down/outputs/contact_guided_remote_rotation/*/contact_guided_remote_rotation_summary.json`
- `toilet_seat_down/outputs/contact_guided_remote_rotation/*/*_remote_rotation_visualization.ply`

随后运行 `toilet_seat_down/end_effector_motion_test.py`，会把该旋转分析结果转成统一末端轨迹。默认 `--motion-mode closed_gripper_push`：先闭合夹爪，把 EEF 放到接触点旋转切向的反侧，再沿预测旋转方向推动部件；旧的 Contact-GraspNet 抓取跟随路线可用 `--motion-mode grasp_follow` 保留：

```text
toilet_seat_down/outputs/end_effector_motion/<episode_key>/<frame>/end_effector_motion_target.json
```

PLY 可视化标签约定：蓝色为旋转轴，白色为接触点 a，青色为旋转 waypoint，品红色为远端 part A 邻近点，橙色为 part B 邻近点。

### 冰箱 / 微波炉隐式门轴 API

`close_fridge` 和 `close_microwave` 不再复用马桶盖的可见缝隙轴拟合。新增 `common/implicit_door_remote_rotation.py`，流程是：

```text
handle mask 3D center 作为默认交互点，VLM/GraspNet guide 保留为诊断或可选回退
  -> door mask 点云拟合门板平面和 PCA 主轴
  -> task_configs.py 传入 door_opening_type=side、hinge_axis_orientation=vertical
  -> 优先用 pointcloud_summary 的 camera extrinsics 把 world z 转到点云 frame
  -> handle_center 投影到门板宽轴，hinge 取 handle 对侧竖直边
  -> handle_center 绕隐式 hinge 旋转生成 waypoints
  -> 根据 closing_direction_sign 选择旋转方向，输出与旧旋转 summary 兼容的 axis/waypoints
```

入口示例：

```bash
uv run --no-sync --active python close_fridge/implicit_door_remote_rotation_test.py \
  --episode-key variation0_episode0_frame035_wrist \
  --frame-stem 35

# 如需复现实验早期行为，可改为使用 VLM/GraspNet guide 最近点作为交互点
uv run --no-sync --active python close_fridge/implicit_door_remote_rotation_test.py \
  --episode-key variation0_episode0_frame035_wrist \
  --frame-stem 35 \
  --contact-point-mode guide

uv run --no-sync --active python close_microwave/implicit_door_remote_rotation_test.py \
  --episode-key variation0_episode2_frame054_wrist \
  --frame-stem 54

uv run --no-sync --active python close_fridge/end_effector_motion_test.py \
  --episode-key variation0_episode0_frame035_wrist \
  --frame-stem 35
```

主要输出：

- `close_fridge/outputs/implicit_door_remote_rotation/<episode_key>/<frame>/implicit_door_remote_rotation_summary.json`
- `close_microwave/outputs/implicit_door_remote_rotation/<episode_key>/<frame>/implicit_door_remote_rotation_summary.json`
- `<task>/outputs/end_effector_motion/<episode_key>/<frame>/end_effector_motion_target.json`

批量运行示例：

```bash
TASKS="close_fridge close_microwave" \
MODULES="implicit_door_remote_rotation end_effector_motion" \
bash run_batch_module_tests.sh --door-guide-source vlm
```


## 2026-05-27: Contact-GraspNet 交互/抓取位姿预测

已新增 `common/contact_graspnet_pose.py` 和 `run_task_module.py contact_graspnet_pose`，用于读取已有的 VLM 接触点、SAM3 point-selection 结果、RGB-D 和相机 metadata，调用 cap-x 的 Contact-GraspNet HTTP 服务预测候选交互/抓取位姿。

默认目标 mask 是 `common/task_configs.py` 中 `sam3_prompts` 的第一个 prompt；可通过 task config 的 `contact_grasp_prompt` 或 CLI 的 `--target-prompt` 覆盖。候选选择规则当前为最高 `score`。

运行前需要启动 Contact-GraspNet 服务：

```bash
cd /home/fubin/projects/artance/cap-x
uv run --no-sync --active python -m capx.serving.launch_contact_graspnet_server --port 8115 --host 127.0.0.1 --device cuda
```

然后运行：

```bash
cd /home/fubin/projects/artance/cap-x/artance_tests
uv run --no-sync --active python run_task_module.py contact_graspnet_pose --task close_drawer \
  --episode-line /home/fubin/projects/artance/RLBench/data/RLBench-data/close_drawer/variation0/episodes/episode3/wrist_rgb/51.png
```

常用参数：

- `--target-prompt "drawer handle"`: 覆盖默认目标 mask prompt。
- `--service-url http://127.0.0.1:8115`: 指定 Contact-GraspNet 服务。
- `--grasp-z-offset 0.1034`: 默认沿 Contact-GraspNet grasp 局部 +z 方向补偿其内部 `gripper_depth=0.1034m`，把可执行 EEF/TCP 目标从 gripper base origin 推到接触点附近；如果只想保留原始 Contact-GraspNet 矩阵，可显式传 `--grasp-z-offset 0.0`。
- `--mask <mask.npy>`: 绕过 SAM3 summary，直接指定目标 mask。
- `--no-save-visualization`: 不保存点云 PLY 可视化。
- `--visualization-top-k 10`: 额外显示最高分前 K 个候选的简化位姿。
- `--visualization-subsample-factor 2`: 点云可视化降采样倍率。

输出目录默认是：

```text
<task>/outputs/contact_graspnet_pose/<episode_key>/<frame>/
```

输出内容：

- `grasps_camera.npy`: Contact-GraspNet 原始候选 `(K, 4, 4)`，camera frame。
- `grasp_poses_camera.npy`: 应用 `--grasp-z-offset` 后的候选；默认比原始候选沿 grasp 局部 +z 前移 `0.1034m`。
- `scores.npy`: 候选分数。
- `contact_points_camera.npy`: Contact-GraspNet 输出的 contact points。
- `segmap.npy`: 发送给服务的 instance segmentation map，目标 mask 像素为 `segmap_id=1`。
- `contact_graspnet_visualization.ply`: camera-frame 点云可视化。RGB 点云会变暗，目标 mask 为橙色；best grasp 的 x/y/z 轴分别为红/绿/蓝，best origin 为白色，best contact point 为黄色；top-k 其它候选以紫色/橙色标记。
- `contact_graspnet_summary.json`: 记录输入、目标 mask、最高分候选、camera/world frame pose、visualization label legend 和输出路径。

关于 `control_reduced.py` 的 `+0.12m`：cap-x 原封装在 Contact-GraspNet 输出后立刻右乘 `translation([0, 0, 0.12])`，再把该 pose 返回给后续 `goto_pose`/IK。Contact-GraspNet 上游构造 grasp 时使用 `gripper_depth=0.1034m`，原始矩阵的 origin 天然会在 contact point 后方约 10.34cm；RLBench 这条 ArtAnce 测试链直接执行 EEF pose，没有 `control_reduced.py` 的 TCP offset 补偿，因此当前默认使用更精确的 `+0.1034m`。summary 中会同时记录 raw grasp 和 offset 后 pose 到 contact point 的距离诊断。


## 2026-05-29: Contact-Guided Hierarchical Part Grounding

已在保留旧版 `contact_point -> sam3 -> sam3_point_selection` 流程的基础上，新增更通用的部件分析链路：

```text
image + task prompt
  -> VLM structured grounding
  -> VLM-generated SAM3 object/support/interaction prompts
  -> SAM3 candidate masks
  -> IoU grouping + hierarchy scoring + contact soft guidance
  -> mutually exclusive object/support/interaction masks
```

新增模块：

- `common/vlm_interaction_grounding.py`: 结构化 VLM grounding，输出 `target_object`、`interaction_part`、`support_part`、`contact_pixel_yx` 和分角色 SAM3 prompts。
- `common/hierarchical_part_analysis.py`: 通用 mask grouping、层级三元组选择和 role-based overlap resolution。
- `run_task_module.py structured_grounding`: 只运行 VLM 结构化 grounding。
- `run_task_module.py part_analysis`: 运行完整 VLM grounding + SAM3 + 层级 mask selection。
- `pointcloud` 新增 `--part-analysis-summary`，可直接读取新方案的 final masks。

运行完整新链路：

```bash
cd /home/fubin/projects/artance/cap-x/artance_tests

uv run --no-sync --active python run_task_module.py part_analysis \
  --task close_fridge \
  --episode-line /home/fubin/projects/artance/RLBench/data/RLBench-data/close_fridge/variation0/episodes/episode0/wrist_rgb/35.png \
  --model gemini-2.5-pro \
  --vlm-server-url http://127.0.0.1:8110/chat/completions \
  --sam3-service-url http://127.0.0.1:8114 \
  --top-k 5 \
  --no-show
```

用新方案输出重建点云：

```bash
uv run --no-sync --active python run_task_module.py pointcloud \
  --task close_fridge \
  --episode-line /home/fubin/projects/artance/RLBench/data/RLBench-data/close_fridge/variation0/episodes/episode0/wrist_rgb/35.png \
  --mask-source part_analysis \
  --mask-only
```

`pointcloud --mask-source auto` 是默认值。episode 输入时会按以下顺序自动选择已有 summary：

```text
part_analysis_summary.json -> point_selection_summary.json -> sam3/summary.json
```

因此刚跑完 `MODULES="structured_grounding part_analysis"` 后，再跑 `MODULES="pointcloud"` 会自动读取新版 `part_analysis` masks。若要强制旧版最近 mask 方案，使用 `--mask-source point_selection` 或批量变量 `POINTCLOUD_MASK_SOURCE=point_selection`。

批量运行：

```bash
TASKS="close_drawer push_button close_fridge close_microwave toilet_seat_down" \
MODULES="structured_grounding part_analysis" \
bash run_batch_module_tests.sh

TASKS="close_drawer push_button close_fridge close_microwave toilet_seat_down" \
MODULES="part_analysis" \
bash run_batch_module_tests.sh
```

实现约束：新 mask selection 不包含 fridge、microwave、drawer、handle 的对象/形状/位置先验，只使用 role、SAM3 score、prompt support、2D contact soft distance、dilated containment 和 role priority overlap resolution。
