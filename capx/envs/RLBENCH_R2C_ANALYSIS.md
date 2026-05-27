# R2C RLBench Usage And cap-x Wrapper Change Notes

本文只分析 `/home/xiesenwei/R2C_3D/r2c_3d.py` 如何与 RLBench 交互，并据此判断当前 cap-x RLBench env 封装需要补哪些能力。目标不是复现 R2C 的 VLM、点云分割或 LLM 规划流程。

## 参考代码的 RLBench 初始化方式

`r2c_3d.py` 直接在本地 Python 进程中 import `rlbench`、`pyrep` 和任务类。它的 RLBench 初始化集中在 `init_RLBench()` / `get_RLBench_demo()`：

```python
high_res_camera_config = CameraConfig(
    rgb=True,
    depth=True,
    point_cloud=False,
    mask=False,
    image_size=(640, 640),
    render_mode=RenderMode.OPENGL3,
    depth_in_meters=True,
)

obs_config = ObservationConfig(
    left_shoulder_camera=high_res_camera_config,
    right_shoulder_camera=high_res_camera_config,
    overhead_camera=high_res_camera_config,
    wrist_camera=high_res_camera_config,
    front_camera=high_res_camera_config,
)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    ),
    obs_config=obs_config,
    headless=headless,
    dataset_root=DATASET,
)
```

关键点：

- RLBench 环境由代码里的硬编码参数控制，不是单独的 RLBench YAML。
- `DATASET` 最终被设为 `/home/xiesenwei/data/rlbench_extra`，并传给 `Environment(dataset_root=DATASET)`。
- task 由 Python task class 控制，例如 `StackBlocks`，这些类来自 `/home/xiesenwei/R2C_3D/RLBench/utils_preprocess.py` 中的大量 `from rlbench.tasks... import ...`。
- reset 不是普通 `task.reset()`，而是：
  - `task.set_variation(-1)`
  - `task.get_demos(1, live_demos=False, random_selection=False, from_episode_number=episode_number)[0]`
  - `var = d.variation_number`
  - `task.set_variation(var)`
  - `descriptions, obs = task.reset_to_demo(d)`
- 它在 `env.launch()` 前调用 `env.get_scene_data()`，保存为 `camera_configs`，后面用这个结构取各相机 intrinsics/extrinsics。

另有一个和 RLBench 无关但会影响视觉管线的配置：`SceneSegmentor(debug_mode=False)` 默认使用 Hydra `config_path="./configs"`、`config_name="base"`，即 R2C 工作目录下的 `insegmentor/configs/base.yaml`。这控制的是分割/点云后处理，不控制 RLBench env/task 初始化。

## 参考代码使用的 observation

R2C 把 `obs.__dict__` 和 `camera_configs` 一起组成 `raw_data`：

```python
raw_data = {
    "obs": obs.__dict__,
    "camera_configs": camera_configs,
    "descriptions": descriptions,
}
```

主要读取这些 RLBench observation 字段：

- 五个 RGB-D 相机：
  - `overhead_rgb`, `overhead_depth`
  - `left_shoulder_rgb`, `left_shoulder_depth`
  - `right_shoulder_rgb`, `right_shoulder_depth`
  - `wrist_rgb`, `wrist_depth`
  - `front_rgb`, `front_depth`
- 对应相机参数从 `camera_configs` 取：
  - `camera_configs["<name>_camera"]["intrinsics"]`
  - `camera_configs["<name>_camera"]["extrinsics"]`
- 机器人状态：
  - `obs.gripper_pose`: 7 维 `[x, y, z, qx, qy, qz, qw]`
  - `obs.gripper_open`: 标量，离散夹爪 action 会复用它
- 在一个分支中还直接读内部 scene camera：
  - `env._scene._cam_wrist.get_matrix()`
  - `env._scene._cam_wrist.get_intrinsic_matrix()`

注意四元数顺序：R2C 沿用 RLBench/PyRep action 和 observation 的 XYZW；当前 cap-x HTTP adapter 对外暴露的是 WXYZ。这是一个必须在文档/API 上明确的差异。

## 参考代码使用的控制接口

R2C 的控制基本都落到 `task.step(action)`，因为它把 RLBench arm action mode 设成了 `EndEffectorPoseViaPlanning()`。

因此它的 8 维 action 语义是：

```text
[x, y, z, qx, qy, qz, qw, gripper_open_or_close]
```

常见模式：

- 保持当前姿态，只改位置：
  - `full_action[:7] = obs.gripper_pose`
  - `full_action[7] = obs.gripper_open`
  - `full_action[:3] = target_xyz`
  - `task.step(full_action)`
- 保持当前位置，只改末端朝向：
  - `full_action[:3] = obs.gripper_pose[:3]`
  - `full_action[3:7] = quaternion_xyzw`
  - `task.step(full_action)`
- 抓取/放置：
  - `grasp_pose[:7]` 是末端 pose
  - `grasp_pose[7] = 0` 表示闭合夹爪
  - `pre_grasp_pose[7] = 1` 表示打开夹爪
- 读取 demo 轨迹时也直接把 demo observation 的 `coordinates.gripper_pose` 和 `coordinates.gripper_open` 拼成 action 再 `task.step(action)`。

R2C 没有使用 joint velocity 控制；虽然 import 了 `JointPosition`，这份代码没有实际采用。

## cap-x 当前 RLBench 封装现状

当前 cap-x 采用 Docker 内 RLBench HTTP server，Host 侧不 import RLBench/PyRep，这是正确边界，建议保留。

现有 server 初始化大致是：

- `ObservationConfig().set_all(False)`
- 根据 CLI `--cameras` 开启相机，默认 `front,wrist,left_shoulder,right_shoulder`
- `image_size` 默认 128
- `depth_in_meters=True`
- robot low-dim 开启 joint position、joint velocity、gripper open、gripper pose、gripper matrix、task low-dim state
- `Environment(action_mode=MoveArmThenGripper(JointVelocity(), Discrete()), shaped_rewards=False)`
- reset 使用 `task.reset()`，可传 variation，但不支持 dataset demo 或 `reset_to_demo`

当前 Host 侧 `RLBenchRemoteEnv` 会把 remote observation 转成 cap-x 风格：

- `front` 映射到 `robot0_robotview`
- `wrist` 映射到 `robot0_eye_in_hand`
- `left_shoulder/right_shoulder/overhead` 保留同名 key
- 相机字段为 `images.rgb`、`images.depth`、`intrinsics`、`pose_mat`
- 末端 pose 为 `robot0_eef_pos` + `robot0_eef_quat`，四元数是 WXYZ

控制接口方面：

- `/step` 当前语义是 8 维 `[7 joint velocities, 1 gripper command]`
- `/move_to_pose` 是额外 helper，用 `arm.get_path(position, quaternion)` 执行末端 pose planning
- `/open_gripper`、`/close_gripper` 调用 RLBench `Discrete().action(...)`
- API wrapper 暴露 `goto_pose()`、`move_to_joints()`、`open_gripper()`、`close_gripper()`

## 需要修改或增强的点

### 1. 把 RLBench server 初始化参数配置化

R2C 依赖 dataset demo reset，所以 server 需要支持至少这些启动/低层配置：

- `dataset_root`: 传给 `Environment(dataset_root=...)`，默认可以仍为空。
- `image_size`: 当前已有 CLI，但 cap-x YAML 不能直接控制 server；如果要跑 R2C 风格视觉，应明确要求 server 以 `--image-size 640` 启动。
- `cameras`: 需要包含 `overhead`。R2C 默认五相机顺序是 `overhead,left_shoulder,right_shoulder,wrist,front`。
- `render_mode`: R2C 设置 `RenderMode.OPENGL3`。当前 server 没显式设置 render mode；如 RLBench 默认不稳定，应补 CLI 参数或至少固定为 OPENGL3。
- `headless`: 当前已有 `--no-headless`，语义够用。

这些参数不能只放在 Host 侧 `RLBenchRemoteEnv`，因为真正的 RLBench `Environment` 在 Docker server 内创建。cap-x YAML 目前只能控制 `server_url/task_name/max_steps/privileged`，还不能确保 server 是按同一套 observation/action mode 启动的。建议在文档和 smoke 脚本里把 server 启动参数记录清楚，或者增加一个受控的 server launcher。

### 2. 支持 demo-based reset

这是和 R2C 交互最关键的缺口。当前 `/reset` 只支持 `task.reset()` + variation；R2C 使用 `get_demos(... from_episode_number=N)` + `reset_to_demo(d)`。

建议给 server `/reset` 增加可选字段：

```json
{
  "task_name": "stack_blocks",
  "reset_mode": "demo",
  "episode_number": 0,
  "random_selection": false,
  "live_demos": false
}
```

server 侧逻辑：

- 若 `reset_mode == "demo"`：
  - `task.set_variation(-1)`
  - `demo = task.get_demos(1, live_demos=False, random_selection=False, from_episode_number=episode_number)[0]`
  - `variation = demo.variation_number`
  - `task.set_variation(variation)`
  - `descriptions, obs = task.reset_to_demo(demo)`
  - response 中返回 `variation`、`episode_number`、`reset_mode`
- 否则保留当前普通 `task.reset()` 行为。

Host 侧 `RLBenchRemoteEnv.reset(options=...)` 应透传 `reset_mode`、`episode_number`、`live_demos`、`random_selection` 等字段，而不是只透传 `task_name/variation/attempts`。

### 3. 暴露 RLBench-style raw observation 或 raw camera config

R2C 的视觉入口需要 `obs.__dict__` 风格字段和 `camera_configs["front_camera"]["intrinsics"]` 这样的结构。cap-x 当前 observation 已经有等价信息，但结构不同。

为了让 R2C 风格的视觉模块能无损接入，建议 server response 或 Host conversion 增加一个兼容层：

```python
obs["rlbench_raw"] = {
    "front_rgb": ...,
    "front_depth": ...,
    "wrist_rgb": ...,
    "wrist_depth": ...,
    "left_shoulder_rgb": ...,
    "left_shoulder_depth": ...,
    "right_shoulder_rgb": ...,
    "right_shoulder_depth": ...,
    "overhead_rgb": ...,
    "overhead_depth": ...,
    "gripper_pose": [x, y, z, qx, qy, qz, qw],
    "gripper_open": ...
}

obs["rlbench_camera_configs"] = {
    "front_camera": {"intrinsics": ..., "extrinsics": ...},
    "wrist_camera": {"intrinsics": ..., "extrinsics": ...},
    "left_shoulder_camera": {"intrinsics": ..., "extrinsics": ...},
    "right_shoulder_camera": {"intrinsics": ..., "extrinsics": ...},
    "overhead_camera": {"intrinsics": ..., "extrinsics": ...}
}
```

这里的 `gripper_pose` 建议保持 RLBench 原生 XYZW，避免 R2C action 代码误把 WXYZ 当 XYZW。cap-x 现有 `robot0_eef_quat` 可继续保留 WXYZ。

server 当前从 `obs.misc` 取每帧相机 intrinsics/extrinsics，这比只在 launch 前取一次 `env.get_scene_data()` 更适合 wrist camera，因为 wrist extrinsics 会随末端变化。兼容字段的 `extrinsics` 应使用当前 observation misc，而不是初始 scene data。

### 4. 增加 EndEffectorPoseViaPlanning action 语义，或明确只通过 helper 暴露

R2C 的 `task.step()` action 语义和 cap-x 当前 `/step` 语义不同：

- R2C: `[x, y, z, qx, qy, qz, qw, gripper]`
- cap-x 当前 `/step`: `[joint_vel_0..6, gripper]`

如果目标是支持 R2C 风格策略代码直接迁移，server 必须支持 EE pose action mode。一种做法是启动参数增加：

```text
--arm-action-mode joint_velocity | ee_pose_via_planning
```

当为 `ee_pose_via_planning` 时：

- `Environment(action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()))`
- `/step` 校验 8 维 action，但错误文案和语义改为 `[x,y,z,qx,qy,qz,qw,gripper]`
- observation 中保留 RLBench 原生 XYZW 末端 pose

如果 cap-x 不希望改变 `/step` 语义，则建议新增 endpoint：

```text
POST /step_ee_pose
```

body:

```json
{
  "pose_xyzw": [x, y, z, qx, qy, qz, qw],
  "gripper": 1.0
}
```

这样既不破坏当前 joint velocity `/step`，又能精确表达 R2C 的控制方式。现有 `/move_to_pose` 可以覆盖“移动到 pose”的一部分需求，但它和 RLBench `EndEffectorPoseViaPlanning().action()` 不完全等价，尤其是夹爪命令和 task.step 的 reward/terminate 更新语义需要一致。

### 5. 在 API wrapper 中暴露 R2C 需要的视觉和 pose-step helper

当前 `FrankaRLBenchApi.get_observation()` 对 LLM 友好，但 R2C 风格代码更需要：

- 取五相机 RGB-D、intrinsics、extrinsics 的统一函数。
- 取 RLBench 原生 gripper pose `[x,y,z,qx,qy,qz,qw]` 的函数。
- 用 RLBench 原生 pose action step 的函数。

建议增加 API，但保持和已有 API 并存：

```python
get_rlbench_observation()
get_camera_observations(camera_order=("overhead", "left_shoulder", "right_shoulder", "wrist", "front"))
step_end_effector_pose(pose_xyzw, gripper)
```

这样 prompt 里可以清楚区分：

- `goto_pose(position, quaternion_wxyz)` 是 cap-x helper，四元数 WXYZ。
- `step_end_effector_pose(pose_xyzw, gripper)` 是 RLBench action-mode helper，四元数 XYZW。

### 6. object_names 不应只靠默认列表

R2C 对象理解主要来自 RGB-D 分割点云，不强依赖 privileged object pose。但 cap-x 现有 oracle 依赖 `object_names` 暴露 waypoint/object poses，默认列表只覆盖少量任务。若要扩展到 R2C 任务集，例如 `StackBlocks`、`PushButtons`、`PutGroceriesInCupboard`，需要把 `object_names` 也做成任务/YAML/CLI 可配置。

参考工程里有 `/home/xiesenwei/R2C_3D/task_object_name_gt/task_object_names.json`，但 `r2c_3d.py` 本身没有直接读取它；它更像其他脚本的人工映射资源。cap-x 可以借鉴这种“按 task 配 object name 映射”的方式，但不应把该文件当成 R2C 主流程的必需 config。

## 建议的最小改动顺序

1. server 增加 `dataset_root` 和 demo reset 支持，并让 `RLBenchRemoteEnv.reset(options=...)` 透传这些字段。
2. server 默认/文档化支持五相机 640 RGB-D，尤其补上 `overhead`；确认 intrinsics/extrinsics 来自当前 obs misc。
3. Host observation 增加 `rlbench_raw` 和 `rlbench_camera_configs` 兼容字段，保留 cap-x 原有 observation schema。
4. 增加 EE pose step 能力。优先新增 `/step_ee_pose`，避免改变当前 `/step` joint velocity 语义。
5. `FrankaRLBenchApi` 增加面向视觉/R2C 的 helper，并在 docstring 里明确坐标系、深度单位和四元数顺序。
6. 为 `StackBlocks` 这类 R2C 主流程任务加一个 smoke YAML：server 用 demo reset，client reset 后检查五相机 shape、camera matrices、gripper pose，再执行一个小的 EE pose no-op 或 z 方向微动。

## 不建议做的事

- 不建议把 R2C 的 RLBench/PyRep 直接 import 到 Host 侧 cap-x。当前 Docker 隔离边界仍然是正确的。
- 不建议为了兼容 R2C 直接把 cap-x 全局四元数约定从 WXYZ 改成 XYZW。更稳妥的是新增 raw/RLBench 命名字段，在 API 名和 docstring 里明确。
- 不建议把 R2C 的分割、VLM、LLM planner 一起塞进 `RLBenchRemoteEnv`。env wrapper 只需要提供 reset、observation、控制和 success/reward 的清晰边界。
