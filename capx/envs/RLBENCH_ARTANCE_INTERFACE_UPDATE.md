# RLBench ArtAnce Interface Update Notes

本文记录本次根据 `cap-x/artance_tests` 和 `RLBench/tests/artance_local` 更新 cap-x RLBench 远程接口的分析与改动。重点是 episode 启动方式、observation schema、机械臂控制 helper 和 API wrapper；没有重写 HTTP 请求模型或 owner-thread 串行执行模型。

## 参考流程

`RLBench/tests/artance_local` 当前执行链路是：

1. 从 `episodes.txt` 或单个 RGB 路径解析 `task / variation / episode / frame / camera`。
2. 用 `Environment(... dataset_root=RLBench-data, action_mode=MoveArmThenGripper(JointPosition(True), Discrete()))` 启动 RLBench。
3. `task.set_variation(variation)` 后调用 `task.get_demos(..., image_paths=False, random_selection=False, from_episode_number=episode)`。
4. 先 `task.reset_to_demo(demo)`，再读取 demo observation 的 `misc["joint_position_action"]`，逐步 `task.step(action)` replay 到指定 frame。
5. 运动执行不走 joint velocity step，而是用 `scene.robot.arm.get_path(position, quaternion=xyzw, ignore_collisions=...)`，循环 `path.step(); scene.step()`，gripper 用 `scene.robot.gripper.actuate(target); scene.step()`。

`cap-x/artance_tests` 生成的 end-effector motion target 是 world-frame `position` + `quaternion_wxyz`，并且经常依赖 offline/live frame info 中的 RLBench camera extrinsics、intrinsics 和原生 `gripper_pose`。

## server 侧改动

文件：`cap-x/rlbench_adapter/rlbench_server.py`

- 增加 server 启动参数：
  - `--dataset-root`: 传给 RLBench `Environment(dataset_root=...)`，供 stored demo reset 使用。
  - `--arm-action-mode joint_position|joint_velocity|ee_pose_via_planning`: 默认是 `joint_position`，与 `RLBench/tests/artance_local` 的 `joint_position_action` replay 对齐。
  - `--render-mode`: 默认 `opengl3`，给启用 camera 的 `CameraConfig.render_mode`。
  - `--path-video-output-root`、`--path-video-camera`、`--path-video-fps`、`--no-record-path-video`: 控制 server 端逐 path-step 视频记录，默认开启。
- 保留原有 endpoint 和 owner-thread 模型，只扩展现有 `/reset` 和控制 endpoint 的 body 参数。
- `/reset` 新增 demo/episode 模式字段：
  - `reset_mode`: `default` 或 `demo`/`episode`/`reset_to_demo`。
  - `variation`, `episode_number`, `frame_index`/`frame`。
  - `live_demos`, `random_selection`, `image_paths`, `replay_action_key`。
- demo reset 行为：
  - 按 variation 或 `-1` 设 variation。
  - `get_demos(1, ..., from_episode_number=episode_number)`。
  - 用 demo 的 `variation_number` 再设 variation。
  - `reset_to_demo(demo)`。
  - 如果提供 `frame_index`，从 step 1 replay 到该 frame，默认 action key 为 `joint_position_action`。
- observation 增加兼容字段：
  - `rlbench_raw`: `front_rgb/front_depth/wrist_rgb/...`、`joint_positions`、`gripper_open`、原生 `gripper_pose=[x,y,z,qx,qy,qz,qw]`、`gripper_matrix`、`task_low_dim_state`、filtered `misc`。
  - `rlbench_camera_configs`: `<camera>_camera -> {intrinsics, extrinsics}`。
  - 原 `cameras`、`front_camera`、`gripper_pose` WXYZ、`object_poses` 等字段继续保留。
- `move_to_pose` 增加 path step 上限，失败时仍返回结构化 `ok=False`，不改变 HTTP 错误语义；内部每个 `scene.step()` 默认写入 path-step 诊断视频。
- `open_gripper`/`close_gripper` 改成和本地流程一致的 `gripper.actuate(target, velocity); scene.step()` 循环，并保留结构化结果和 path-step 视频条目。
- `_task_class()` 现在优先从 `rlbench.tasks.__init__` 导出的 task class registry 做匹配，再 fallback 到 `rlbench.tasks.<task_name>` + CamelCase 推导。

## host low-level env 改动

文件：`cap-x/capx/envs/simulators/rlbench_remote.py`

- `RLBenchRemoteEnv.__init__` 增加默认 reset 配置字段：`reset_mode / variation / episode_number / frame_index / live_demos / random_selection / image_paths / replay_action_key`。
- `reset(options=...)` 会合并构造参数和 runtime options，并透传给 server `/reset`。
- observation conversion 新增：
  - 解码 `rlbench_raw` 中嵌套的 array payload 为 numpy arrays。
  - 将 `rlbench_camera_configs` 转成 numpy `intrinsics/extrinsics`。
  - 增加 cap-x 常用 shortcut：`robot_joint_pos=[7 joints, gripper]` 和 `robot_cartesian_pos=[xyz, quat_wxyz, gripper]`。
  - 增加 `gripper_pose_xyzw=[x,y,z,qx,qy,qz,qw]`。
- 新增 `move_to_pose_xyzw(pose_xyzw, gripper=None, ...)`，用于 RLBench 原生 XYZW pose。内部仍复用现有 `/move_to_pose`，可选再执行 gripper helper；没有新增 HTTP route。

## API wrapper 改动

文件：`cap-x/capx/integrations/franka/rlbench.py`

在保留原 `get_observation / get_object_pose / goto_pose / move_to_joints / open_gripper / close_gripper` 的基础上，新增：

- `get_rlbench_observation()`: 返回 RLBench-style raw observation，包含五相机 RGB-D、low-dim state 和原生 `gripper_pose`。
- `get_camera_observations(camera_order=...)`: 按 RLBench camera 名返回 `{rgb, depth, intrinsics, extrinsics}`。
- `get_gripper_pose_xyzw()`: 返回 `[x,y,z,qx,qy,qz,qw]`，方便保持当前姿态改 XYZ。
- `goto_pose_xyzw(pose_xyzw, gripper=None, ...)`: 接受 RLBench 原生 XYZW pose，区别于原 `goto_pose(position, quaternion_wxyz)`。

## 使用建议

ArtAnce stored episode server 建议类似：

```bash
python cap-x/rlbench_adapter/rlbench_server.py \
  --task close_drawer \
  --dataset-root /workspace/RLBench/data/RLBench-data \
  --arm-action-mode joint_position \
  --image-size 640 \
  --cameras wrist,front,left_shoulder,right_shoulder,overhead \
  --path-video-output-root /home/fubin/projects/artance/cap-x/outputs/rlbench_path_videos
```

Host YAML 可在 `RLBenchRemoteEnv` 构造参数中配置：

```yaml
reset_mode: demo
variation: 0
episode_number: 3
frame_index: 51
replay_action_key: joint_position_action
```

也可以在 `env.reset(options={...})` 中覆盖这些字段。

## 未做事项与风险

- 没有新增 `/step_ee_pose`，以避免改变当前 HTTP surface；`/step` 会随 `--arm-action-mode` 分配到 joint velocity、joint position 或 RLBench 原生 `EndEffectorPoseViaPlanning` action。
- `goto_pose_xyzw` 通过现有 PyRep path helper 实现，不等价于 RLBench `EndEffectorPoseViaPlanning().action()` 的单步 `task.step` 语义；若要直接迁移 R2C pose-step policy，应以 `--arm-action-mode ee_pose_via_planning` 启动并直接调用 `/step`。
- 没有把 `cap-x/artance_tests` 中更高层的铰接体 API 直接注册进 cap-x prompt；本次只完成基础 RLBench/Franka 适配层。
- 尚未在真实 RLBench/CoppeliaSim 环境中跑 smoke，因为当前 host 环境没有启动 Docker RLBench server。


## 本次验证

- `python -m py_compile cap-x/rlbench_adapter/rlbench_server.py cap-x/capx/envs/simulators/rlbench_remote.py cap-x/capx/integrations/franka/rlbench.py` 通过。
- 直接用系统 Python 跑 host-side smoke 时失败，原因是当前 shell 未安装/未激活 `gymnasium`。
- `uv run --no-sync` 下的伪 remote observation conversion smoke 通过，确认 front/wrist RGB-D、`rlbench_raw`、`rlbench_camera_configs`、`robot_joint_pos`、`robot_cartesian_pos`、`gripper_pose_xyzw` 的 shape 正常。
- `uv run --no-sync` 下 `FrankaRLBenchApi` import 通过。该 import 会打印现有项目依赖 warning（robosuite IK、LIBERO/R1Pro 未安装），但不影响本 API 文件加载。
- 未启动 Docker/CoppeliaSim/RLBench server，因此未做真实 `/reset reset_mode=demo` 或 path planning smoke。


## 适配审计结论

结论：当前版本可以覆盖 `cap-x/artance_tests` -> `RLBench/tests/artance_local` 的核心执行路线，但不能宣称“完美适配”。核心闭环已经对齐的是 stored demo reset、replay 到指定 frame、world-frame end-effector path、gripper actuate、RLBench raw camera metadata 和 cap-x API wrapper。仍需注意以下条件和差异。

已对齐的关键点：

- episode 启动：server 支持 `reset_mode=demo`、`variation`、`episode_number`、`frame_index`，行为对应本地 `get_demos(... from_episode_number=episode)`、`reset_to_demo(demo)`、按 `joint_position_action` replay。
- 启动配置：server 支持 `--dataset-root`、`--arm-action-mode joint_position`、`--render-mode opengl3`、可配置 camera list。
- motion target：ArtAnce 产出的 `position + quaternion_wxyz` 可直接走 `goto_pose()`/`move_to_pose()`；RLBench 原生 `[x,y,z,qx,qy,qz,qw]` 可走 `goto_pose_xyzw()`。
- gripper：server 的 open/close 已改为 `scene.robot.gripper.actuate(target); scene.step()` 循环，和本地 `artance_local.close_gripper()` 一致。
- observation：host obs 同时保留 cap-x keys 和 `rlbench_raw` / `rlbench_camera_configs`，可支持 offline/live frame info 里的 intrinsics、extrinsics、gripper pose 需求。
- video interface：host `RLBenchRemoteEnv` 现在和 robosuite/LIBERO 一样维护 scene frame buffer 与 wrist frame buffer，`use_wrist_camera` 不再返回空列表。

仍不一致或需要运行约束的点：

- server 必须用 `--arm-action-mode joint_position` 启动才能 replay `joint_position_action`。已加 guard：如果 demo replay 使用 `joint_position_action` 但 server 是 joint velocity，会返回 409，而不是安静地错。
- `/step` 仍是同一个 endpoint，8 维 action 的真实语义取决于 server 的 `arm_action_mode`。默认 `joint_position` 与 ArtAnce stored-demo replay 对齐；若要 R2C 原生 pose-step，需要显式选择 `ee_pose_via_planning`。
- `goto_pose_xyzw()` 是 PyRep path helper 加可选 gripper helper，不是 RLBench `EndEffectorPoseViaPlanning().action()` 的原生 `task.step([xyz,xyzw,gripper])`。如果要直接迁移 R2C pose-step policy，应以 `--arm-action-mode ee_pose_via_planning` 启动并直接走 `/step`。
- server 现在默认记录 path 内部每个 `scene.step()` 的视频，但视频写在 server 端输出目录，并通过 `path_video` / `path_video_manifest` 返回路径；host trial-level video buffer 仍只记录远程调用完成后的 observation frame。
- `artance_local.resolve_dataset_root()` 有 `/workspace/RLBench/data/RLBench-data` 到本地 `RLBench/data/RLBench-data` 的 fallback；server 侧没有本地 fallback，Docker 内必须传真实可见路径。
- task class 匹配已改为优先参考 `rlbench.tasks.__init__` 导出的 registry，覆盖不规则 class name 的 snake_case 推导；registry 缺失时才使用旧的 module/class fallback。
- 真实 CoppeliaSim/RLBench smoke 尚未跑，因此只能保证接口契约和 host-side conversion，不保证每个 task 的 path planning 成功率。

和 cap-x 其他 simulator wrapper 的本质差异：

- RLBench 仍是远程 owner-thread + HTTP adapter；host env 不持有 simulator 实例，这和本地 robosuite/LIBERO 本质不同。
- RLBench low-level env 暴露 motion-planning 级 `move_to_pose()`，而 robosuite/LIBERO 通常由 integration API 做 IK 后调用 `move_to_joints_blocking()`。
- RLBench motion helper 返回 structured result；robosuite/LIBERO 多数 helper 返回 `None`。这对 LLM repair 友好，但跨环境 oracle 不完全可移植。
- RLBench reset 现在比 robosuite/LIBERO 多 task/episode/action-sequence context，也能通过 options 切换 task 和 stored demo frame。


## 2026-05-31 action mode / video / task mapping update

本轮进一步收紧了三处接口契约。

### `/step` 与 `--arm-action-mode`

server 现在支持三种 `--arm-action-mode`：

| mode | `/step` 8 维 action 语义 | 用途 |
| --- | --- | --- |
| `joint_position` | `[7 Franka joint target positions, 1 discrete gripper command]` | 默认值；与 `RLBench/tests/artance_local` replay `joint_position_action` 对齐。 |
| `joint_velocity` | `[7 Franka joint velocities, 1 discrete gripper command]` | 兼容早期 cap-x RLBench adapter。 |
| `ee_pose_via_planning` | `[x, y, z, qx, qy, qz, qw, gripper command]` | 对齐 RLBench/R2C 原生 pose-step action mode。 |

`GET /health` 和 observation 会返回 `step_action_spec`，说明当前 `/step` shape 与语义。demo replay 默认使用 `joint_position_action`，如果 server 不是 `joint_position` mode，会返回 409，避免把 joint position 当 velocity 静默执行。

### path-step 视频记录

HTTP adapter 现在默认开启 path-step 视频记录：

- 默认根目录：`/home/fubin/projects/artance/cap-x/outputs/rlbench_path_videos`。
- 每个 reset 创建目录：`YYYYMMDD-HHMMSS-<task>-var<variation>-episode<episode>-frame<frame>`。
- `move_to_pose`、`open_gripper`、`close_gripper` 会在内部每个 `scene.step()` 后抓取一帧。
- 默认 camera 是 `wrist`，缺失时 fallback 到 `front`。
- 每个动作写一个 mp4，并维护 `manifest.json`；HTTP response 中返回 `path_video`，observation 中返回 `path_video_dir` 和 `path_video_manifest`。
- 可用 `--no-record-path-video` 关闭；可用 `--path-video-output-root`、`--path-video-camera`、`--path-video-fps` 调整。

这补齐的是 server 端逐 path-step 诊断视频；host `RLBenchRemoteEnv` 的 video buffer 仍用于 cap-x trial-level 视频。

### task name 到 task class

server 的 `_task_class()` 现在优先参考 `rlbench.tasks.__init__` 导出的 task classes，按 class name 的 snake_case 建立映射，例如：

- `close_drawer` -> `CloseDrawer`
- `push_button` -> `PushButton`
- `toilet_seat_down` -> `ToiletSeatDown`
- `stack_blocks` -> `StackBlocks`
- `put_groceries_in_cupboard` -> `PutGroceriesInCupboard`

如果 registry 中没有匹配，再 fallback 到早期的 `rlbench.tasks.<task_name>` + CamelCase 推导。


## 2026-05-31 ArtAnce local control-mode tightening

本轮确认默认执行语义采用 `RLBench/tests/artance_local` 的拆分式控制：

1. reset 阶段用 `joint_position_action` replay stored demo 到目标 frame。
2. 后续 ArtAnce motion 执行用 server-side PyRep `arm.get_path()`，循环 `path.step(); scene.step()`。
3. gripper 仍只暴露 `open_gripper()` 和 `close_gripper()`，内部走 `scene.robot.gripper.actuate(...)`，不新增 gripper API。

因此 `goto_pose()` / `goto_pose_xyzw()` 的对外含义统一为 **ArtAnce local path helper**，不是 RLBench 原生 `task.step([xyz, qx, qy, qz, qw, gripper])`。`/health` 和 observation 会返回 `pose_helper_spec`；host observation 会透传该字段。

server 端新增了更接近 RLBench 官方 `EndEffectorPoseViaPlanning` 的 pose 前置校验：

- target position 和 quaternion 必须是 finite 数值。
- quaternion 必须是 unit quaternion；否则返回 `InvalidQuaternionError`。
- target position 必须在 RLBench task workspace 内；否则返回 `WorkspaceBoundaryError`。

`move_to_pose` 的 `arm.get_path()` 现在固定使用 RLBench 官方 pose-planning 参数，不对外暴露这些 planner knobs：

```text
trials=100
max_configs=10
max_time_ms=10
trials_per_goal=5
algorithm=RRTConnect
```

path-step manifest 细化为 `manifest_version=2`，每个 action video entry 会记录 task/variation/episode/frame、`control_mode`、target pose、planner、steps、success_before/success_after、reward_after、terminate_after、gripper target、error_context 等字段。控制失败会同时：

- 在 HTTP response 中返回 `ok=false`、`error_type`、`error`、`error_context`、`traceback` 和 `path_video`。
- 在 server stderr 打印 JSON context 和 traceback，方便容器日志与 VLM agent 错误分析对齐。

`/step` 仍保留给显式 action-mode 兼容；默认 code-agent API 不应依赖裸 8 维 `/step`，而应使用 `goto_pose()` / `goto_pose_xyzw()` / `open_gripper()` / `close_gripper()`。
