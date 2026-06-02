# capx.envs simulator environment design notes

本文整理 `capx/envs` 中已有 LIBERO、robosuite 和 RLBench host-side 环境的包装思路。重点是低层环境对 `CodeExecutionEnvBase` 和 `capx.integrations.*` 呈现的行为，而不是各 simulator 自身的完整实现。

## 1. 总体链路

`capx/envs` 把任务拆成两层：

- `capx/envs/simulators/`: 低层环境，继承 `BaseEnv`，负责 reset、step、observation、reward、success，以及少量被 API 层调用的机器人控制原语。
- `capx/envs/tasks/`: 高层 code execution env，继承 `CodeExecutionEnvBase`，负责拼 task prompt、绑定 API docstring、执行 LLM 生成的 Python 代码。
- `capx/integrations/`: 暴露给 LLM 代码的 API 层。API 方法通过 `self._env` 调低层环境，例如 `move_to_joints_blocking()`、`_set_gripper()`、`get_observation()`。

标准低层环境接口来自 `capx/envs/base.py`：

| Function | Parameters | Return | 语义 |
| --- | --- | --- | --- |
| `reset` | `seed: Optional[int] = None`, `options: Optional[dict] = None` | `(obs, info)` | 重置 simulator，返回初始观测和任务信息。 |
| `step` | `action: Any` | `(obs, reward, terminated, truncated, info)` | Gym 风格低层 step。对 code-as-policy 任务通常只是 fallback。 |
| `get_observation` | none | `dict[str, Any]` | 返回统一 observation dict，供 API/perception/prompt 使用。 |
| `compute_reward` | none | `float` | 返回当前 reward。 |
| `task_completed` | none | `bool` | 返回任务是否成功。 |

`CodeExecutionEnvBase` 通过 `CodeExecEnvConfig.low_level` 找到低层环境注册名，通过 `CodeExecEnvConfig.apis` 找到 API 类，并把 API 的 `functions()` 结果直接注入执行 namespace。因此 simulator env 不直接面向 LLM；LLM 主要看 integration API 的函数签名和 docstring。

## 2. LIBERO 包装

主要文件：

- `capx/envs/simulators/libero.py`
- `capx/envs/adapters/libero_wrapper.py`
- `capx/integrations/franka/libero.py`
- `capx/integrations/franka/libero_privileged.py`

`FrankaLiberoEnv` 是一个通用 LIBERO low-level wrapper。它用 `suite_name + task_id` 创建 LIBERO task，而不是每个 task 写一个 simulator class。`FrankaLiberoTask` 进一步把这个模式暴露给 YAML；`FrankaLiberoPickPlace`、`FrankaLiberoOpenMicrowave`、`FrankaLiberoPickAlphabetSoup` 是兼容旧配置的 convenience classes。

### 2.1 构造参数

| Parameter | Default | 作用 |
| --- | --- | --- |
| `suite_name` | required | LIBERO suite，例如 `libero_10`、`libero_90`、`libero_object`、`libero_spatial`、`libero_goal`。 |
| `task_id` | required | suite 内 task index。 |
| `privileged` | `True` | 标记是否使用 privileged 信息；当前 env 本身仍会保留 task/object state helper。 |
| `max_steps` | `4000` | horizon/truncation 上限。 |
| `seed` | `None` | RNG seed；reset 时也会用 seed 映射 LIBERO init state。 |
| `enable_render` | `False` | 预留参数，当前注释为 TODO。 |
| `control_freq` | `20` | LIBERO controller frequency。 |
| `viser_debug` | `False` | 是否启动 viser debug 可视化。 |

底层创建逻辑调用 `load_libero_task(..., controller="JOINT_POSITION", horizon=max_steps, control_freq=control_freq)`。reset 后会做 10 步 settling，并把 `home_joint_position` 记录为重置后的 `robot0_joint_pos`。

### 2.2 Observation keys

`FrankaLiberoEnv.get_observation()` 把 LIBERO/MuJoCo 原始 obs 转成 cap-x 常用结构：

| Key | Value shape/type | 来源/语义 |
| --- | --- | --- |
| `agentview.images.rgb` | `(H, W, 3) uint8` | LIBERO `agentview_image`，上下翻转后返回。 |
| `agentview.images.depth` | metric depth array，通常 `(H, W, 1)` | `agentview_depth` 经 `get_real_depth_map()` 转 metric depth；`FrankaLiberoApi.get_observation()` 会 squeeze 成 `(H, W)`。 |
| `agentview.images.segmentation` | segmentation array, optional | `agentview_segmentation_instance`。 |
| `agentview.intrinsics` | `(3, 3) float64` | 由 MuJoCo camera fovy 和 render size 计算。 |
| `agentview.pose` | `(7,) [x,y,z,qw,qx,qy,qz]` | camera pose，相对 robot base frame。 |
| `agentview.pose_mat` | `(4, 4)` | camera extrinsic matrix，相对 robot base frame。 |
| `robot0_eye_in_hand.*` | 同 `agentview` | wrist camera observation。 |
| `robot_joint_pos` | `(8,)` | `[7 robot0_joint_pos, normalized_gripper]`。代码注释有一处写 shape `(7,)`，实际拼接 gripper 后是 8 维。 |
| `robot_cartesian_pos` | `(8,)` | `[eef xyz, eef quaternion_wxyz, normalized_gripper]`，相对 robot base frame，并带 TCP offset/朝向修正。 |

此外类里还有 privileged helper：

| Helper | Parameters | Return | 语义 |
| --- | --- | --- | --- |
| `_get_object_pose` | `obj_name: str` | `(position: np.ndarray(3), quaternion_wxyz: np.ndarray(4))` | 从 LIBERO obs 或 MuJoCo body 查 object pose，并转换到 robot base frame。 |
| `_get_all_object_poses` | none | `dict[str, tuple[np.ndarray(3), np.ndarray(4)]]` | 收集所有 movable/fixed object poses。 |

### 2.3 控制接口

这些方法不是直接暴露给 LLM，而是被 `FrankaLiberoApi` 或 privileged API 调用。

| Function | Parameters | Return | Behavior |
| --- | --- | --- | --- |
| `move_to_joints_blocking` | `joints: np.ndarray(7)`, `tolerance: float = 0.01`, `max_steps: int = 120` | `None` | 将目标 joint position 转成 LIBERO action。当前关节误差小于 tolerance 后退出。 |
| `_set_gripper` | `fraction: float` | `None` | 只记录 gripper target，`1.0=open`，`0.0=closed`。需要后续 `_step_once()` 推进仿真。 |
| `_step_once` | none | `None` | 发送零 joint delta + 当前 gripper command，让仿真前进一步。 |
| `get_current_time_s` | none | `float` | `_sim_step_count / control_freq`。 |

LIBERO action 的 key-value 语义：

| Action component | Value |
| --- | --- |
| joint part | `delta = (target - current) * control_freq`，7 维。 |
| gripper part | 1 维，cap-x `fraction` 映射为 simulator command：`1.0 open -> -1.0`，`0.0 closed -> 1.0`。 |

`FrankaLiberoApi.functions()` 面向 LLM 暴露：

| LLM function | Parameters | Return | 说明 |
| --- | --- | --- | --- |
| `get_observation` | none | `dict[str, Any]` | 返回 LIBERO obs，并把 `agentview`、`robot0_eye_in_hand` 的 depth squeeze 成 `(H, W)`。 |
| `get_object_pose` | `object_name: str`, `use_multiview: bool = True` | `(position: np.ndarray(3), quaternion_wxyz: np.ndarray(4))` 或 `(None, None)` | 非 privileged 版本用语言分割、多视角点云和 OBB 估计 object pose。 |
| `sample_grasp_pose` | `object_name: str`, `use_multiview: bool = True` | `(position: np.ndarray(3), quaternion_wxyz: np.ndarray(4))` | 多视角点云 + Contact-GraspNet 采样抓取 pose。 |
| `goto_pose` | `position: np.ndarray(3)`, `quaternion_wxyz: np.ndarray(4)`, `z_approach: float = 0.0` | `None` | PyRoKi IK 得到 7 维 joints，然后调用 `move_to_joints_blocking()`。 |
| `open_gripper` | none | `None` | 调 `_set_gripper(1.0)` 后连续 `_step_once()`。 |
| `close_gripper` | none | `None` | 调 `_set_gripper(0.0)` 后连续 `_step_once()`。 |
| `get_oriented_bounding_box_from_3d_points` | `points: np.ndarray(N, 3)` | `dict[str, Any]` | 返回 OBB 的 `center`、`extent`、`R`、`quaternion_wxyz`。 |
| `get_object_3d_points_and_masks_from_language` | `text_prompt: str`, `use_multiview: bool = True` | `dict[str, Any]` | 返回分割 mask、跨视角 3D points 和 score。 |
| `goto_home_joint_position` | none | `None` | 回到 reset 记录的 `home_joint_position`。 |

`FrankaLiberoPrivilegedApi.functions()` 与非 privileged 版本不同，直接暴露 simulator state：

| LLM function | Parameters | Return | 说明 |
| --- | --- | --- | --- |
| `get_observation` | none | `dict[str, Any]` | 返回 LIBERO obs，depth 同样会 squeeze。 |
| `get_object_pose` | `object_name: str` | `(position: np.ndarray(3), quaternion_wxyz: np.ndarray(4))` | 直接调用 env `_get_object_pose()`。 |
| `get_all_object_poses` | none | `dict[str, tuple[np.ndarray(3), np.ndarray(4)]]` | 直接调用 env `_get_all_object_poses()`。 |
| `sample_grasp_pose` | `object_name: str` | `(position: np.ndarray(3), quaternion_wxyz: np.ndarray(4))` | 当前实现取 object position，并返回固定抓取 quaternion。 |
| `goto_pose` | `position: np.ndarray(3)`, `quaternion_wxyz: np.ndarray(4)`, `z_approach: float = 0.0` | `None` | IK 后调 `move_to_joints_blocking()`。 |
| `open_gripper` / `close_gripper` | none | `None` | 设置 gripper target 并推进若干 step。 |
| `goto_pose_interactive_cartesian` | `target_pose_predicate`, `replan_interval_s=0.0`, `lin_vel_norm=1.0`, `ang_vel_norm=2.0`, `z_approach=0.0`, `timeout_s=20.0` | `None` | 闭环 Cartesian 伺服式移动。 |

## 3. robosuite 包装

主要文件：

- `capx/envs/simulators/robosuite_base.py`
- `capx/envs/simulators/robosuite_cubes.py`
- `capx/envs/simulators/robosuite_cube_lift.py`
- `capx/envs/simulators/robosuite_cubes_restack.py`
- `capx/envs/simulators/robosuite_nut_assembly.py`
- `capx/envs/simulators/robosuite_spill_wipe.py`
- `capx/envs/simulators/robosuite_two_arm_lift.py`
- `capx/envs/simulators/robosuite_handover.py`
- `capx/envs/adapters/robosuite_wrapper.py`

robosuite 的设计和 LIBERO 不同：它按 task 拆成多个 low-level env class。单臂任务多数继承 `RobosuiteBaseEnv`，双臂任务因为 action shape、base frame、gripper state 都不同，直接继承 `BaseEnv`。

### 3.1 单臂 `RobosuiteBaseEnv`

构造参数：

| Parameter | Default | 作用 |
| --- | --- | --- |
| `controller_cfg` | `capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json` | robosuite Panda joint controller config。 |
| `max_steps` | `1500` | horizon/truncation。 |
| `seed` | `None` | RNG seed。 |
| `viser_debug` | `False` | viser debug。 |
| `privileged` | `False` | 是否关闭 camera obs / 使用 privileged API。 |
| `enable_render` | `False` | privileged 模式下是否仍开启 offscreen camera rendering。 |

共享控制方法：

| Function | Parameters | Return | Behavior |
| --- | --- | --- | --- |
| `_set_gripper` | `fraction: float` | `None` | 设置 target gripper fraction，`1.0=open`，`0.0=closed`。 |
| `_build_action` | none | `np.ndarray(9)` | 拼 `[7 joints, gripper, gripper]`，再把 gripper 映射为 robosuite command。实际送入 simulator 前会被 `_ACTION_SLICE` 裁切。 |
| `_do_robosuite_step` | `action: np.ndarray` | `None` | 按 `_ACTION_SLICE` 裁切后调用 `robosuite_env.step()`；非渲染时 `skip_render_images=True`。 |
| `_step_once` | none | `None` | 用当前 `_current_joints` 和 `_gripper_fraction` 前进一步。 |
| `move_to_joints_non_blocking` | `joints: np.ndarray(7)` | `None` | 单步发送目标 joint。 |
| `move_to_joints_blocking` | `joints: np.ndarray(7)`, `tolerance: float = 0.02`, `max_steps: int = 100` | `None` | 循环发送 joint target，直到误差小于 tolerance。 |

共享 observation helpers：

| Helper | Parameters | Return | Output |
| --- | --- | --- | --- |
| `_process_camera_observations` | `robosuite_obs: dict`, `base_wxyz_xyz: Optional[np.ndarray] = None` | `None` | 原地对 `render_camera_names` 中每个 camera 添加 `pose`、`pose_mat`、`intrinsics`、`images.rgb/depth/segmentation`。 |
| `_compute_gripper_obs` | `robosuite_obs: dict` | `None` | 原地添加 `robot_joint_pos = [robot0_joint_pos, normalized_gripper]` 和 `robot_cartesian_pos = [eef xyz, eef quaternion_wxyz, normalized_gripper]`。 |

单臂 robosuite action key-value：

| Key | Value |
| --- | --- |
| `self._current_joints` | 7 维 Panda joint target。 |
| `self._gripper_fraction` | cap-x normalized gripper，`1.0=open`，`0.0=closed`。 |
| robosuite action | base 先构造 9 维 `[7 joint target, mapped_gripper, mapped_gripper]`；默认 `_ACTION_SLICE=-1` 后送入 8 维 action；Wipe `_ACTION_SLICE=-2` 后送入 7 维 action。 |
| `_ACTION_SLICE` | 默认 `-1`，Wipe 任务是 `-2`。 |

### 3.2 robosuite task-specific envs

robosuite 针对不同 task 包装多个环境，原因不是注册风格差异而是行为差异真实存在：

- 底层 robosuite class 不同：`Stack`、`Lift`、`NutAssemblySquare`、`Wipe`、`TwoArmLift`、`TwoArmHandover`。
- reset 逻辑不同：object sampler、cube size、camera pose、home joints、settling steps 都可能不同。
- observation key 不同：cube、nut、pot、hammer 的 privileged pose 结构完全不同。
- reward/success 不同：有的直接用 `_check_success()`，有的叠加二次检查，例如 restack 防止两个 cube 同时悬空误判。
- action shape 不同：Wipe 的 action slice 不同；双臂任务 action 是两个 Panda action 拼接。
- API/prompt 不同：单臂 stack/lift/nut/wipe 用 Franka 单臂 API；handover/lift 暴露双臂 API。

task wrapper 摘要：

| Env class | Underlying robosuite task | 主要新增 obs key | Reward/success |
| --- | --- | --- | --- |
| `FrankaRobosuiteCubesLowLevel` | `Stack` | `cube_poses.primary`, `cube_poses.secondary` | `reward(action=None)`, `_check_success()` |
| `FrankaRobosuiteCubeLiftLowLevel` | `Lift` | `cube_poses.primary` | `reward()`, `_check_success()` |
| `FrankaRobosuiteCubesRestackLowLevel` | `Stack` with custom cube size/sampler | `cube_poses.primary`, `cube_poses.secondary` | reward/success 加二次高度检查 |
| `FrankaRobosuiteNutAssembly` | `NutAssemblySquare` | `nut_poses.square_nut`, `nut_poses.square_nut_handle`, `nut_poses.square_peg`, `nut_handle_to_center_offset` | `reward()`, `_check_success()` |
| `FrankaRobosuiteSpillWipeLowLevel` | `Wipe` | shared camera keys, `robot_joint_pos` shape `(7,)`, `robot_cartesian_pos` shape `(7,)` | `reward()`, `_check_success()` |
| `RobosuiteTwoArmLiftEnv` | `TwoArmLift` | `pot_poses.pot`, `pot_poses.handle0`, `pot_poses.handle1`, `robot0_cartesian_pos`, `robot1_cartesian_pos` | `reward()`, `_check_success()` |
| `RobosuiteHandoverEnv` | `TwoArmHandover` | `hammer_poses.hammer`, `hammer_poses.handle`, `robot0_cartesian_pos`, `robot1_cartesian_pos` | `reward()`, `_check_success()` |

### 3.3 robosuite observation schema

单臂 base-compatible task 通常返回 robosuite 原始 obs 并附加 cap-x keys：

| Key | Value |
| --- | --- |
| `robot0_joint_pos` | robosuite 原始 7 维 joint position。 |
| `robot0_gripper_qpos` | robosuite 原始 gripper qpos。 |
| `robot_joint_pos` | cap-x 归一化 joint+gripper shortcut，base-compatible 单臂任务通常 8 维；`SpillWipe` 特例是 7 维。 |
| `robot_cartesian_pos` | cap-x eef pose+gripper shortcut，base-compatible 单臂任务通常 8 维；`SpillWipe` 特例是 7 维。 |
| `robot0_robotview.images.rgb` | RGB image。默认 camera 是 `robot0_robotview`；nut assembly 内部用 `birdview` 但映射回 `robot0_robotview`。 |
| `robot0_robotview.images.depth` | metric depth。 |
| `robot0_robotview.images.segmentation` | optional instance segmentation。 |
| `robot0_robotview.intrinsics` | `(3, 3)` camera intrinsics。 |
| `robot0_robotview.pose` | `(7,) [x,y,z,qw,qx,qy,qz]`，相对 robot base frame。 |
| `robot0_robotview.pose_mat` | `(4, 4)` camera extrinsic。 |
| task keys | `cube_poses` / `nut_poses` 等 privileged task state。 |

双臂任务用 `agentview` 作为 scene-level camera，同时为了兼容部分工具也设置 `robot0_robotview = obs["agentview"]`：

| Key | Value |
| --- | --- |
| `agentview.images.rgb/depth` | 同时看到双臂的 scene camera。 |
| `agentview.intrinsics`, `agentview.pose`, `agentview.pose_mat` | camera 参数，pose 相对 robot0 base frame。 |
| `robot0_joint_pos`, `robot1_joint_pos` | 两个 Panda 的 joint positions。 |
| `robot0_cartesian_pos` | arm0 eef pose+gripper，robot0 base frame。 |
| `robot1_cartesian_pos` | arm1 eef pose+gripper，转换到 robot0 base frame。 |
| `pot_poses` / `hammer_poses` | task object 和 grasp handle poses，robot0 base frame。 |

### 3.4 robosuite API 暴露

单臂 robosuite 通常走以下 API 之一：

| API | LLM functions | Key returns | 底层依赖 |
| --- | --- | --- | --- |
| `FrankaControlApi` | `get_object_pose`, `sample_grasp_pose`, `goto_pose`, `open_gripper`, `close_gripper`, simulation 中还有 `home_pose` | pose 函数返回 `(position, quaternion_wxyz)` 或 `(position, quaternion_wxyz, bbox_extent)`；动作函数返回 `None` | 视觉分割/点云/GraspNet/PyRoKi + env joint/gripper methods。 |
| `FrankaControlPrivilegedApi` | `get_object_pose`, `sample_grasp_pose`, `goto_pose`, `open_gripper`, `close_gripper` | `get_object_pose` 返回 `(position, quaternion_wxyz, bbox_extent_or_None)`；`sample_grasp_pose` 返回 `(position, quaternion_wxyz)`；动作函数返回 `None` | 直接读 `cube_poses` 等 privileged state。注意 `functions()` 当前没有暴露 `home_pose`。 |
| `FrankaControlNutAssemblyPrivilegedApi` | `get_object_pose`, `sample_grasp_pose`, `goto_pose`, `goto_home_joint_position`, `open_gripper`, `close_gripper` | pose 函数返回 `(position, quaternion_wxyz)`；动作函数返回 `None` | 读 `nut_poses`，按 nut/peg 特化抓取。 |
| `FrankaControlSpillWipePrivilegedApi` | `get_object_pose`, `goto_pose` | `get_object_pose` 返回 `(position, quaternion_wxyz, bbox_extent_or_None)`；`goto_pose` 返回 `None` | Wipe task 特化。 |

双臂 API：

| API | LLM functions | Key returns | 底层依赖 |
| --- | --- | --- | --- |
| `FrankaHandoverApi` | `get_object_pose`, `goto_pose_arm0`, `goto_pose_arm1`, `open_gripper_arm0`, `open_gripper_arm1`, `close_gripper_arm0`, `close_gripper_arm1` | `get_object_pose` 返回 `(position, quaternion_wxyz, bbox_extent_or_None)`；动作函数返回 `None` | `move_to_joints_blocking`, `move_to_joints_blocking_arm1`, `_set_gripper`, `_set_gripper_arm1`。 |
| `FrankaTwoArmLiftApi` | `get_handle0_pos`, `get_handle1_pos`, `get_arm0_gripper_pose`, `get_arm1_gripper_pose`, `goto_pose_arm0`, `goto_pose_arm1`, `goto_pose_both`, arm0/arm1 gripper open/close | handle 函数返回 `position: np.ndarray(3)`；gripper pose 返回 `(position, quaternion_wxyz)`；动作函数返回 `None` | 双臂 joint methods 和 vision handle detection。 |

## 4. `simulators` 与 `adapters` 为什么分两个目录

当前代码里这两个目录的职责有明显层级区别：

| Directory | 当前职责 | 是否在主执行链路中活跃 |
| --- | --- | --- |
| `capx/envs/simulators/` | 继承 `BaseEnv` 的 cap-x 低层环境。这里的 class 会被 `register_env()` 注册，并由 `CodeExecutionEnvBase` 构造。 | 是。LIBERO、robosuite、RLBench remote、real Franka、R1Pro 都在这里注册。 |
| `capx/envs/adapters/` | 更薄的外部库 wrapper / scaffold，例如 `LiberoWrapper`、`RoboSuiteWrapper`。它们只包 reset/step/success，不实现完整 cap-x observation schema、video、robot control helper。 | 当前几乎未被引用；更像早期或预留的库适配层。 |

可以把两者理解成：

- `adapters`: 只解决“怎么创建/调用外部环境”的薄适配，接口接近外部库。
- `simulators`: 解决“怎么让这个外部环境表现得像 cap-x low-level env”，包括统一 camera keys、robot state、reward/success、video capture、debug、以及 API 层需要的控制函数。

目前 LIBERO 和 robosuite 的 active implementation 大多直接在 `simulators/` 中调用外部库，而不是经由 `adapters/`。因此写新环境时，应优先在 `simulators/` 实现 `BaseEnv`；只有当外部库初始化/step 逻辑需要复用或隔离时，再把更薄的部分抽进 `adapters/`。

## 5. RLBenchRemoteEnv 与 LIBERO/robosuite 的对比

这里不讨论 HTTP/远程进程本身，只比较它对 cap-x host 侧呈现的行为。

### 5.1 构造参数对比

| Env | Parameter | Default | Key-value 语义 |
| --- | --- | --- | --- |
| `FrankaLiberoEnv` | `suite_name` | required | 选择 LIBERO benchmark suite。 |
| `FrankaLiberoEnv` | `task_id` | required | 选择 suite 内任务。 |
| `FrankaLiberoEnv` | `control_freq` | `20` | joint delta action 的比例和时间换算。 |
| `RobosuiteBaseEnv` subclasses | `controller_cfg` | Panda joint ctrl json | 选择 robosuite controller。 |
| `RobosuiteBaseEnv` subclasses | task class | per file | task 由 Python class 固定，例如 Stack/Lift/NutAssembly。 |
| `RLBenchRemoteEnv` | `task_name` | `"reach_target"` | 当前 RLBench task，可在 reset options 或 `switch_task()` 中切换。 |
| `RLBenchRemoteEnv` | `server_url` | `"http://127.0.0.1:8120"` | transport endpoint；不影响 cap-x observation schema。 |
| `RLBenchRemoteEnv` | `timeout` | `120.0` | 单次控制/观测调用 timeout。 |
| all | `max_steps` | env-specific | truncation 上限。 |
| all | `privileged` | env-specific | 是否使用 privileged task state/API。 |
| all | `enable_render` | env-specific | 是否启用图像渲染。 |
| all | `viser_debug` | `False` | debug visualization。RLBench host env 当前不做 viser scene。 |

设计差异：LIBERO 用 `suite_name/task_id` 泛化很多 task；robosuite 用多个 class 固化 task；RLBenchRemoteEnv 用 `task_name` 和 reset options 动态切换 task，在 cap-x 侧维持同一个 class。

### 5.2 reset 行为对比

| Env | Function | Parameters | Return | Important key-value |
| --- | --- | --- | --- | --- |
| LIBERO | `reset` | `seed: Optional[int] = None`, `options: Optional[dict] = None` | `(obs: dict, info: dict)` | `seed` 会映射到 `init_states[(seed - 1) % n]`；`info["task_prompt"] = task_language`。 |
| robosuite | `reset` | `seed: Optional[int] = None`, `options: Optional[dict] = None` | `(obs: dict, info: dict)` | task class 自己 reset、settle、设置 camera/object sampler；`info["task_prompt"]` 为 task-specific prompt。 |
| RLBench | `reset` | `seed: Optional[int] = None`, `options: Optional[dict] = None` | `(obs: dict, info: dict)` | `payload["task_name"] = options.get("task_name", self.task_name)`；`variation = options["variation"]` 或 `seed`；`attempts = options["attempts"]`；`info` 含 `episode_id`, `task_name`, `task_descriptions`, `action_sequence_id`。 |

RLBench cap-x 行为上的不同点：

- reset 后会建立 episode/action sequence context，并把这些 id 放进 obs/info。
- seed 不是普通 RNG seed，而是默认被解释成 RLBench variation。
- 同一个 low-level env 可以通过 `options["task_name"]` 切换 RLBench task。

### 5.3 observation key 对比

| Category | LIBERO | robosuite | RLBenchRemoteEnv |
| --- | --- | --- | --- |
| 主相机 key | `agentview` | 单臂多为 `robot0_robotview`；双臂为 `agentview` 并 alias 到 `robot0_robotview` | remote `front` 映射为 `robot0_robotview` |
| wrist key | `robot0_eye_in_hand` | base 支持 `robot0_eye_in_hand` video/render；是否在 obs 取决于 `render_camera_names` | remote `wrist` 映射为 `robot0_eye_in_hand`，如果 adapter 提供 |
| camera image | `images.rgb`, `images.depth`, optional `images.segmentation` | 同左 | `images.rgb`, `images.depth`；不加入 segmentation |
| camera calibration | `intrinsics`, `pose`, `pose_mat` | `intrinsics`, `pose`, `pose_mat` | `intrinsics`, `pose_mat`；当前不生成 `pose` vector |
| joint state | `robot_joint_pos` shortcut；原始 `robot0_joint_pos` 保留在 `_current_obs` 但不直接 merge 到 returned obs | 原始 `robot0_joint_pos` + shortcut `robot_joint_pos` | `robot0_joint_pos`, `robot0_joint_vel`, `robot0_gripper_qpos` |
| eef state | `robot_cartesian_pos` | `robot_cartesian_pos` 或双臂 `robot0_cartesian_pos`/`robot1_cartesian_pos` | `robot0_eef_pos`, `robot0_eef_quat` |
| task object state | helper `_get_object_pose()` / `_get_all_object_poses()`，不默认放入 obs | task-specific keys: `cube_poses`, `nut_poses`, `pot_poses`, `hammer_poses` | generic `object_poses: dict[name, [x,y,z,qw,qx,qy,qz]]`; `target_pose` alias |
| task text | `info["task_prompt"]` | `info["task_prompt"]` | `task_descriptions`, `task_name` in obs/info |
| execution metadata | none | none | `episode_id`, `action_sequence_id`, `last_reward`, `last_terminate`, `success`, optional `last_action_result` |

RLBench 对 cap-x 呈现出的关键取向是“标准化后的 remote observation”，而不是暴露 simulator 原始 obs。它把不同 RLBench task 的对象统一进 `object_poses`，把相机统一映射到 robosuite-ish keys，并额外保留 episode/action metadata。

### 5.4 control function 对比

| Env | Function | Parameters | Return | 行为 |
| --- | --- | --- | --- | --- |
| LIBERO | `move_to_joints_blocking` | `joints(7)`, `tolerance=0.01`, `max_steps=120` | `None` | 本地循环 step，joint delta controller。 |
| robosuite single-arm | `move_to_joints_blocking` | `joints(7)`, `tolerance=0.02`, `max_steps=100` | `None` | 本地循环 step，joint position controller。 |
| robosuite two-arm | `move_to_joints_blocking` | `joints(7)`, `tolerance=0.02`, `max_steps=100` | `None` | 控制 robot0，robot1 保持当前 joints。 |
| robosuite two-arm | `move_to_joints_blocking_arm1` | `joints(7)`, `tolerance=0.02`, `max_steps=100` | `None` | 控制 robot1，robot0 保持当前 joints。 |
| robosuite two-arm | `move_to_joints_blocking_both` | `joints0(7)`, `joints1(7)`, `tolerance=0.02`, `max_steps=100` | `None` | 同步控制两臂。 |
| RLBench | `move_to_joints_blocking` | `joints(7)`, `tolerance=0.01`, `max_steps=200` | `dict` | 执行 adapter 侧 joint motion，返回 structured result。 |
| RLBench | `move_to_pose` | `position(3)`, `quaternion_wxyz(4)`, `ignore_collisions=True` | `dict` | 直接执行 end-effector pose motion。LIBERO/robosuite 通常由 API 层 IK 后转 joint。 |
| RLBench | `open_gripper` / `close_gripper` | none | `dict` | 显式 gripper action，并返回 structured result。 |
| LIBERO/robosuite | `_set_gripper` | `fraction: float` | `None` | 只设置 target，需要后续 `_step_once()` 才推进仿真。 |
| RLBench | `_set_gripper` | `fraction: float` | `dict` | `fraction > 0.5 -> open_gripper`，否则 close；立即执行动作。 |

RLBench 的最大行为差异：

- 它把 `move_to_pose()` 作为低层 env 直接能力暴露出来；robosuite/LIBERO 通常没有低层 pose controller，`goto_pose()` 在 integration API 内通过 IK 转成 joints。
- 控制 helper 返回 structured result，包括 `ok`、`error`、`error_type`、`steps`、`success`、`terminate`、`episode_id`、`action_sequence_id`、`request_id`。robosuite/LIBERO 多数返回 `None`，失败通常表现为未到达或上层异常。
- gripper open/close 是立即执行动作；robosuite/LIBERO 是设置 fraction 并 step 多次。

### 5.5 `step(action)` 对比

| Env | Expected action | Return | Behavior |
| --- | --- | --- | --- |
| LIBERO | `Any` fallback | `(obs, reward, terminated=False, truncated, info)` | 不直接使用传入 action；返回当前 obs/reward/truncation。常规控制经 API helper。 |
| robosuite base | `Any` fallback | `(obs, reward, terminated=False, truncated, info)` | 不直接使用传入 action；返回当前 obs/reward/truncation。常规控制经 API helper。 |
| RLBench | low-level action array | `(obs, reward, terminated, truncated, info)` | 会提交 action。当前 adapter 统一校验 8 维，但语义由 server `--arm-action-mode` 决定：joint position、joint velocity 或 `[xyz, qx, qy, qz, qw, gripper]` pose planning。 |

因此 RLBenchRemoteEnv 更接近一个真正可 step 的 Gym low-level env；LIBERO/robosuite 在 CaP-X code execution 路径里更像“供 API helper 闭环控制的 simulator state holder”。

### 5.6 API 层对比

| Env family | API `goto_pose` 实现 | Object pose 来源 | Gripper 实现 |
| --- | --- | --- | --- |
| LIBERO | `FrankaLiberoApi.goto_pose()` 用 PyRoKi/curobo-style IK 生成 joints，再调 env `move_to_joints_blocking()` | 非 privileged 用视觉/多视角点云；privileged 可读 env object poses | `_set_gripper()` + 多步 `_step_once()` |
| robosuite | `FrankaControlApi`/privileged API 用 IK 生成 joints，再调 env `move_to_joints_blocking()` | 非 privileged 用视觉；privileged 读 `cube_poses`/`nut_poses` 等 task key | `_set_gripper()` + 多步 `_step_once()` |
| RLBench | `FrankaRLBenchApi.goto_pose()` 直接调 env `move_to_pose(position, quaternion_wxyz)`，可选 `z_approach` | `obs["object_poses"]` generic dict | env `open_gripper()` / `close_gripper()` |

这也是 RLBench 在 cap-x 侧呈现出的主要架构差异：它的 low-level env 已经提供 motion-planning 级别的 pose action；前两个环境的 pose-level API 是 cap-x integration 层组合视觉、IK 和 joint controller 得到的。


### 5.7 ArtAnce stored-demo update audit

当前 RLBenchRemoteEnv 比早期分析多了 ArtAnce stored-demo 路线支持：

- 构造参数可带 `reset_mode`、`variation`、`episode_number`、`frame_index`、`live_demos`、`random_selection`、`image_paths`、`replay_action_key`，reset 时也可通过 `options` 覆盖。
- host observation 新增 `rlbench_raw` 和 `rlbench_camera_configs`，用于保留 RLBench 原生 RGB-D、camera intrinsics/extrinsics、low-dim state 和 `gripper_pose=[x,y,z,qx,qy,qz,qw]`。
- host observation 新增与 LIBERO/robosuite 更接近的 shortcuts：`robot_joint_pos=[7 joints, gripper]`、`robot_cartesian_pos=[xyz, quat_wxyz, gripper]`。
- host video capture 现在和 robosuite/LIBERO 一样维护 `_frame_buffer` 与 `_wrist_frame_buffer`，`enable_video_capture(..., wrist_camera=True)` 后可通过 `get_wrist_video_frames()` 取 wrist frames。
- low-level 新增 `move_to_pose_xyzw()`，API 层可用 `goto_pose_xyzw()` 明确调用 RLBench 原生 XYZW pose；控制语义仍是 ArtAnce local path helper，而不是原生 pose-step。
- server 默认 `--arm-action-mode joint_position` 以对齐 ArtAnce local 的 `joint_position_action` replay，同时也支持 `joint_velocity` 和 `ee_pose_via_planning`；observation/info 会带 `step_action_spec` 和 `pose_helper_spec`。
- server 默认开启 path-step 视频记录，输出到 `/home/fubin/projects/artance/cap-x/outputs/rlbench_path_videos`，episode 目录名含日期、task、episode 和 frame；manifest 记录 target pose、planner、success/reward 和 error_context。
- `move_to_pose()` 现在参考 RLBench 官方 pose-planning action 做 unit-quaternion 与 workspace 校验，并固定使用 RRTConnect planner 参数。

和 LIBERO/robosuite 相比仍然存在的本质差异：

- RLBench host env 不是本地 simulator wrapper，而是 HTTP proxy。真正的 `Environment`、action mode、dataset_root、camera render mode 都在 server 进程中决定；host YAML 不能单独保证 server 启动配置正确。
- RLBench reset 可以恢复 stored demo 的中间 frame，这和 LIBERO seed/init-state、robosuite sampler reset 都不同；它更像“dataset episode state restore + replay”。
- RLBench low-level `move_to_pose()` 是 backend path-planning primitive；LIBERO/robosuite 的 `goto_pose()` 多数是在 integration API 中解 IK 后走 joint controller。
- RLBench control helper 返回 structured result，适合记录 motion-planning failure；LIBERO/robosuite helper 多数是 `None` return。
- RLBench `/step` 是真实 remote action dispatch，且 action 语义取决于 server arm action mode；LIBERO/robosuite 在 code execution 路径里的 `step()` 基本是 fallback。
- RLBench adapter 现在默认在 server 端记录 path 内部每个 `scene.step()` 的诊断视频，并把文件路径写回 `path_video` / `path_video_manifest`；host video buffer 仍是 trial-level frame buffer，不负责传输 path 内所有图像。

## 6. 新增环境时的建议

- 如果要接一个本地 simulator，优先实现 `capx/envs/simulators/<name>.py`，继承 `BaseEnv`，保证 `get_observation()` 至少提供 camera image/depth/intrinsics/pose 和 robot state。
- 如果多个 task 共享 controller/camera/video/robot state，像 robosuite 一样抽一个 base env，再让 task wrapper 只处理 reset/object pose/reward/success。
- 如果 simulator 自身 task 可以由参数选择且 observation/reward/control schema 一致，像 LIBERO 一样保留一个 generic env class。
- 如果 backend 已经提供稳定的 high-level motion primitive，像 RLBench 一样可以在 low-level env 暴露 `move_to_pose()` 和 structured control result；同时要把返回 obs 标准化为 cap-x API 能理解的 keys。
- `adapters/` 适合存很薄的外部库隔离层；真正注册给 cap-x 的仍应是 `simulators/` 里的 `BaseEnv`。
