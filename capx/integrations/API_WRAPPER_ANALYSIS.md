# capx.integrations API wrapper analysis

本文参考 `capx/envs/SIMULATOR_ENV_DESIGN.md` 的分析口径，整理 `capx/integrations` 中面向 LLM 代码执行的 API 封装。重点不是视觉模型、运动规划器或 simulator 本身的完整实现，而是这些 API 如何被注册、如何进入 prompt、如何调用 low-level env，以及不同任务/环境之间的抽象层级差异。

## 1. 总体定位

`capx/integrations` 是 CaP-X 中 LLM 生成代码直接看到的工具层。它位于高层 code execution env 和 low-level simulator env 之间：

```text
env_configs/*.yaml
  -> CodeExecEnvConfig.apis
  -> capx.integrations.get_api(name)
  -> ApiBase subclass instance
  -> functions() exposed into exec globals
  -> generated Python code calls API functions
  -> API calls self._env / perception services / motion services
```

核心边界：

- `capx/envs/simulators/` 负责 reset、step、observation、reward、success 和 simulator 生命周期。
- `capx/envs/tasks/base.py` 负责拼 prompt、绑定 API、执行 LLM 代码。
- `capx/integrations/` 负责把感知、几何、运动规划和 env helper 包装成 LLM 可调用函数。

因此 integrations 不应拥有 simulator 生命周期，也不应直接承担 trial 调度；它主要定义“模型能调用什么”和“调用后如何落到底层环境”。

## 2. 基础机制

主要文件：

- `capx/integrations/base_api.py`
- `capx/integrations/__init__.py`
- `capx/envs/tasks/base.py`

### 2.1 `ApiBase`

`ApiBase` 规定所有 API wrapper 的最小协议：

| Method / field | 作用 |
| --- | --- |
| `self._env` | 指向 low-level env，由构造函数传入。 |
| `functions()` | 返回 `dict[str, Callable]`，决定哪些函数暴露给 LLM。 |
| `combined_doc()` | 用 `inspect.signature()` 和 docstring 聚合 prompt 文档。 |
| `enable_webui()` | 标记是否启用 Web UI execution logging。 |
| `_log_step()` / `_log_step_update()` | Web UI 模式下记录工具调用步骤；非 Web UI 时 no-op。 |

`combined_doc()` 的设计很直接：遍历 `functions()` 返回的函数，提取签名和 docstring，然后拼成文本。也就是说，函数签名和 docstring 是模型唯一稳定可见的 API 文档。新增或修改 API 时，docstring 的准确性和 shape/frame/unit 说明非常关键。

### 2.2 注册机制

`capx/integrations/__init__.py` 使用 `register_api(name, factory)` 注册 API 名称。YAML 中的 `apis` 字段引用这些名称。

注册可分为几类：

| Category | Examples | 说明 |
| --- | --- | --- |
| Franka 基础控制 | `FrankaControlApi`, `FrankaControlPrivilegedApi` | Robosuite 单臂任务常用。 |
| Franka reduced | `FrankaControlApiReduced`, `FrankaControlApiReducedSkillLibrary` | 暴露更细粒度视觉/几何/IK primitives。 |
| 任务特化 | `FrankaControlNutAssemblyPrivilegedApi`, `FrankaControlSpillWipeApi` | 针对 nut assembly、spill wipe 等特殊任务。 |
| 双臂 | `FrankaHandoverApi`, `FrankaTwoArmLiftApi` | 显式区分 arm0/arm1。 |
| RLBench | `FrankaRLBenchApi` | Host 侧远程 RLBench wrapper。 |
| LIBERO | `FrankaLiberoApi`, `FrankaLiberoPrivilegedApi`, `FrankaLiberoApiReduced` | LIBERO 专用 camera schema 和多视角流程。 |
| Real robot | `FrankaRealControlApi`, `FrankaRealReducedSkillLibraryControlApi` | 真实 Franka 使用不同 TCP offset。 |
| R1Pro | `R1ProControlApi` | BEHAVIOR/R1Pro 移动操作任务。 |
| Debug | `RobotStateDebugApi` | 只读 robot state debug helper。 |

不少注册项只是同一个 class 的参数化版本，例如：

- spill wipe 通过 `tcp_offset=[0.0, 0.0, -0.0158]` 适配 sponge attachment。
- real Franka 通过 `tcp_offset=[0.0, 0.0, -0.157]` 和 `real=True` 修改末端偏置和姿态处理。
- bimanual 通过 `bimanual=True` 改变 exposed functions。

### 2.3 注入到代码执行 namespace

`CodeExecutionEnvBase` 创建 API 后，会做两件事：

1. 把每个 API 的 `combined_doc()` 加到 prompt 的 `APIs:` 段落。
2. 把 `api.functions()` 中的函数以裸函数名注入 `_exec_globals`。

生成代码可以直接调用：

```python
pos, quat = sample_grasp_pose("red cube")
goto_pose(pos, quat, z_approach=0.08)
close_gripper()
```

也可以通过 `APIS` 字典访问 API 实例：

```python
APIS["FrankaControlApi"].open_gripper()
```

需要注意：多个 API 同时暴露同名函数时，裸函数名会在同一个 globals 中覆盖。当前配置通常避免混用多个重名控制 API，但这仍是组合多个 API 时的隐含风险。

## 3. API 抽象层级

现有 integrations 大致有三种抽象层级。

### 3.1 高层技能式 API

代表：

- `FrankaControlApi`
- `FrankaControlPrivilegedApi`
- `FrankaLiberoApi`
- `FrankaLiberoPrivilegedApi`
- `FrankaRLBenchApi`

这类 API 暴露少量可直接完成任务的函数，例如：

| Function | 语义 |
| --- | --- |
| `get_object_pose(name)` | 根据语言名或 privileged state 返回 object pose。 |
| `sample_grasp_pose(name)` | 返回可抓取 pose。 |
| `goto_pose(position, quaternion_wxyz, z_approach=0.0)` | 移动末端到目标 pose。 |
| `open_gripper()` / `close_gripper()` | 控制 gripper。 |
| `home_pose()` / `goto_home_joint_position()` | 回到安全或 reset joint configuration。 |

优点是 prompt 简洁，LLM 负责任务顺序和少量几何推理。缺点是感知/规划失败点被隐藏在一个函数内，不利于模型纠错，也不利于评估中间能力。

### 3.2 Reduced / primitive API

代表：

- `FrankaControlApiReduced`
- `FrankaControlApiReducedExampleless`
- `FrankaControlApiReducedSkillLibrary`
- `FrankaLiberoApiReduced`
- `FrankaLiberoApiReducedSkillLibrary`

这类 API 把高层技能拆成可组合 primitives：

| Primitive group | Example functions |
| --- | --- |
| Observation | `get_observation()` |
| Pointing / language localization | `point_prompt_molmo()` |
| Segmentation | `segment_sam3_text_prompt()`, `segment_sam3_point_prompt()`, `detect_object_owlvit()`, `segment_sam2()` |
| Geometry | `get_oriented_bounding_box_from_3d_points()` |
| Grasp planning | `plan_grasp()`, `plan_grasp_from_point_clouds()` |
| IK / motion | `solve_ik()`, `move_to_joints()` |
| Gripper | `open_gripper()`, `close_gripper()` |
| Utility / skill library | point cloud transforms, quaternion conversions, interpolation, vector normalization |

优点是更适合研究 LLM 是否能显式组合视觉、几何、抓取和运动。缺点是 prompt 很长，代码更容易出 shape/frame 错误，且模型需要理解更多 camera/point-cloud 细节。

### 3.3 任务特化 API

代表：

- `FrankaControlNutAssemblyPrivilegedApi`
- `FrankaControlNutAssemblyVisualApi`
- `FrankaControlSpillWipeApi`
- `FrankaControlSpillWipePrivilegedApi`
- `FrankaHandoverApi`
- `FrankaHandoverPrivilegedApi`
- `FrankaTwoArmLiftApi`
- `FrankaTwoArmLiftPrivilegedApi`
- `R1ProControlApi`

这类 API 与某个 env observation schema 或 task object schema 高度绑定。它们牺牲通用性，换取 prompt 更贴近任务，例如：

- nut assembly 直接识别 `square_nut`、`square_nut_handle`、`square_peg`。
- spill wipe 使用 sponge TCP offset，只暴露 `get_object_pose` 和 `goto_pose` 等擦拭任务需要的最小函数。
- two-arm lift 直接暴露 `get_handle0_pos()`、`get_handle1_pos()`、`goto_pose_both()`。
- handover 直接暴露 `goto_pose_arm0()`、`goto_pose_arm1()` 和双臂 gripper 操作。
- R1Pro 同时暴露 base navigation、torso、arm IK、gripper、视觉检测和视频保存。

## 4. Franka Robosuite API

主要文件：

- `capx/integrations/franka/control.py`
- `capx/integrations/franka/control_privileged.py`
- `capx/integrations/franka/control_reduced.py`
- `capx/integrations/franka/control_reduced_exampleless.py`
- `capx/integrations/franka/control_reduced_skill_library.py`

### 4.1 `FrankaControlApi`

面向非 privileged Robosuite 单臂任务，典型 exposed functions：

| Function | Return | 内部实现 |
| --- | --- | --- |
| `get_object_pose(object_name, return_bbox_extent=False)` | `(position, quaternion_wxyz, bbox_extent_or_None)` | SAM3 或 OWL-ViT+SAM2 分割，depth -> point cloud，Open3D OBB，camera frame -> world frame。 |
| `sample_grasp_pose(object_name)` | `(position, quaternion_wxyz)` | 分割目标，Contact-GraspNet 规划抓取，camera frame -> world frame。 |
| `goto_pose(position, quaternion_wxyz, z_approach=0.0)` | `None` | 应用 TCP offset，PyRoKi IK，调用 env `move_to_joints_blocking()`。 |
| `open_gripper()` | `None` | 调 common helper `_open_gripper()`，内部通过 env `_set_gripper()` 和 `_step_once()` 推进。 |
| `close_gripper()` | `None` | 同上。 |
| `home_pose()` | `None` | simulation only，移动到固定 home joints。 |

关键假设：

- 主相机通常是 `obs["robot0_robotview"]`。
- pose 使用 world/robot base frame，quaternion 为 WXYZ。
- `get_object_pose()` 的 quaternion 由 OBB 得到，docstring 明确提示可能不适合直接作为 gripper orientation。
- `goto_pose()` 的 `z_approach` 是沿 gripper frame 的 negative z 方向做 approach，而不是简单 world +Z。

### 4.2 `FrankaControlPrivilegedApi`

privileged 版本跳过视觉识别，直接从 env observation 的 task-specific state 读 pose：

| Object query | Observation source |
| --- | --- |
| red cube | `obs["cube_poses"]["primary"]` |
| green cube | `obs["cube_poses"]["secondary"]` |

其 motion 仍然是 PyRoKi IK -> `move_to_joints_blocking()`。它更适合 oracle/debug/baseline，而不评估视觉能力。

注意：普通 `FrankaControlApi` 在 simulation 中暴露 `home_pose()`，但 privileged `functions()` 当前没有暴露 `home_pose()`。

### 4.3 `FrankaControlApiReduced`

reduced 版本将 `get_object_pose()` 和 `sample_grasp_pose()` 这类高层函数拆开。其 exposed functions 由参数决定：

| Mode | Exposed functions |
| --- | --- |
| `use_sam3=True` | `segment_sam3_text_prompt`, `segment_sam3_point_prompt` |
| `use_sam3=False` | `detect_object_owlvit`, `segment_sam2` |
| single arm | `solve_ik`, `move_to_joints`, `open_gripper`, `close_gripper` |
| bimanual | `solve_ik_arm0`, `solve_ik_arm1`, `move_to_joints_both`, `move_to_joints_arm0`, `move_to_joints_arm1`, arm0/arm1 gripper functions |
| all modes | `get_observation`, `point_prompt_molmo`, `plan_grasp`, `get_oriented_bounding_box_from_3d_points` |

这类 API 要求 LLM 自己处理：

- 从 observation 中取 RGB/depth/intrinsics/pose_mat。
- 使用 segmentation mask 和 depth 生成点云。
- 在 camera/world frame 之间转换 grasp pose。
- 调 `solve_ik()` 获得 joints，再调 `move_to_joints()`。

### 4.4 Skill library variants

`FrankaControlApiReducedSkillLibrary` 继承 reduced API，并额外暴露一批由历史生成代码沉淀出的通用工具：

- `rotation_matrix_to_quaternion`
- `decompose_transform`
- `depth_to_point_cloud`
- `mask_to_world_points`
- `pixel_to_world_point`
- `transform_points`
- `interpolate_segment`
- `normalize_vector`
- `select_top_down_grasp`

它的设计目的不是增加 simulator 能力，而是减少 LLM 在每个 trial 中重复手写常见几何函数。

## 5. LIBERO API

主要文件：

- `capx/integrations/franka/libero.py`
- `capx/integrations/franka/libero_privileged.py`
- `capx/integrations/franka/libero_reduced.py`
- `capx/integrations/franka/libero_reduced_skill_library.py`

LIBERO 的 camera schema 与 Robosuite 单臂不同：

| Camera | Key |
| --- | --- |
| scene camera | `agentview` |
| wrist camera | `robot0_eye_in_hand` |

### 5.1 `FrankaLiberoApi`

exposed functions：

| Function | 说明 |
| --- | --- |
| `get_observation()` | 返回 LIBERO obs，并 squeeze depth。 |
| `get_object_pose(object_name, use_multiview=True)` | 语言分割，多视角点云，OBB pose。 |
| `sample_grasp_pose(object_name, use_multiview=True)` | 多视角 point cloud + Contact-GraspNet point-cloud planner。 |
| `goto_pose(position, quaternion_wxyz, z_approach=0.0)` | PyRoKi snippets IK -> env `move_to_joints_blocking()`。 |
| `open_gripper()` / `close_gripper()` | `_set_gripper()` 后推进 40/60 steps。 |
| `get_oriented_bounding_box_from_3d_points()` | Open3D OBB。 |
| `get_object_3d_points_and_masks_from_language()` | agentview/wrist 多视角分割和点云融合。 |
| `goto_home_joint_position()` | 回到 reset 记录的 `home_joint_position`。 |

与 Robosuite 普通 API 相比，LIBERO API 更强调 wrist camera 和多视角融合。`get_object_pose()` 可能返回 `(None, None)`，这是其非 privileged 感知失败语义之一。

### 5.2 `FrankaLiberoPrivilegedApi`

privileged 版本直接调用 low-level env helper：

| Function | Env dependency |
| --- | --- |
| `get_object_pose(object_name)` | `self._env._get_object_pose(object_name)` |
| `get_all_object_poses()` | `self._env._get_all_object_poses()` |
| `sample_grasp_pose(object_name)` | object position + fixed grasp quaternion |
| `goto_pose()` | PyRoKI snippets IK -> env joints |
| `goto_pose_interactive_cartesian()` | 闭环 Cartesian style servo，支持 predicate 更新 target pose |

它仍保留 camera observation helper，但 object pose 不依赖视觉模型。

### 5.3 `FrankaLiberoApiReduced`

LIBERO reduced 版与 Robosuite reduced 版类似，但增加了多视角点云和 cuRobo 相关预留函数：

- `plan_grasp_from_point_clouds`
- `subsample_point_cloud`
- `filter_noise`
- `parse_grasp_poses_for_curobo`、`plan_grasp_trajectory` 等目前在 `functions()` 中注释掉

当前 exposed functions 更适合让 LLM 手写“获取 observation -> 分割 -> 点云 -> grasp/IK -> move”的完整流程。

## 6. RLBench API

主要文件：

- `capx/integrations/franka/rlbench.py`
- `capx/envs/simulators/rlbench_remote.py`
- `cap-x/rlbench_adapter/rlbench_server.py`

`FrankaRLBenchApi` 是最薄的一层 wrapper。exposed functions：

| Function | Return | 行为 |
| --- | --- | --- |
| `get_observation()` | `dict[str, Any]` | 返回 remote env 标准化 observation，包括 cap-x keys 和 RLBench compatibility keys。 |
| `get_rlbench_observation()` | `dict[str, Any]` | 返回 RLBench-style raw fields，例如 `front_rgb`、`wrist_depth`、`gripper_pose=[x,y,z,qx,qy,qz,qw]`。 |
| `get_camera_observations(camera_order=...)` | `dict[str, dict]` | 按 RLBench camera 名返回 RGB-D、intrinsics、extrinsics。 |
| `get_gripper_pose_xyzw()` | `np.ndarray(7,)` | 返回 RLBench 原生 `[x,y,z,qx,qy,qz,qw]`。 |
| `get_object_pose(object_name)` | `(position, quaternion_wxyz)` | 从 `obs["object_poses"]` 查找 object pose。 |
| `goto_pose(position, quaternion_wxyz, z_approach=0.0)` | `dict[str, Any]` | ArtAnce local path helper：env `move_to_pose()` -> server `arm.get_path()` -> `path.step(); scene.step()`；quaternion 是 cap-x WXYZ，approach 是 world +Z。 |
| `goto_pose_xyzw(pose_xyzw, gripper=None, ...)` | `dict[str, Any]` | 同一 ArtAnce local path helper；pose 是 RLBench/PyRep XYZW，可选随后执行 `open_gripper()` / `close_gripper()` helper。 |
| `move_to_joints(joints)` | `None` | 调 env `move_to_joints_blocking()`。 |
| `open_gripper()` | `None` | 调 env `open_gripper()`。 |
| `close_gripper()` | `None` | 调 env `close_gripper()`。 |

与 Robosuite/LIBERO 最大区别：

- 不初始化 SAM、OWL-ViT、Molmo、GraspNet 或 PyRoKi。
- 不在 API 层做 IK；RLBench remote env/adapter 已经提供 `move_to_pose()`。
- `goto_pose()` / `goto_pose_xyzw()` 返回 structured result，pose validation 或 motion planning 失败通常以 `{"ok": false, "error_type": ..., "error_context": ..., "traceback": ...}` 表达，而不是直接 raise。
- `get_object_pose()` 是 privileged style generic lookup，依赖 adapter 暴露 `object_poses`。
- API 同时暴露 WXYZ 和 XYZW 两套 pose helper；这是为了保留 cap-x 约定，同时适配 RLBench/ArtAnce raw observation，不应混用。
- API 层当前没有单独暴露 `step_end_effector_pose()`；server 可通过 `--arm-action-mode ee_pose_via_planning` 让 `/step` 使用 R2C/RLBench 原生 pose-step，但默认 code-agent API 明确采用 ArtAnce local path helper，不把 `goto_pose_xyzw()` 描述为原生 pose-step。

这符合 RLBench Docker 隔离设计：host 侧不 import `rlbench`/`pyrep`，只通过 HTTP adapter 调用远程 simulator。

## 7. 任务特化 Franka API

### 7.1 Nut assembly

主要文件：

- `capx/integrations/franka/nut_assembly_privileged.py`
- `capx/integrations/franka/nut_assembly_visual.py`

privileged 版直接读取：

| Query | Observation source |
| --- | --- |
| square nut handle | `obs["nut_poses"]["square_nut_handle"]` |
| square nut | `obs["nut_poses"]["square_nut"]` |
| peg / block | `obs["nut_poses"]["square_peg"]` |

exposed functions 包括 `get_object_pose`、`sample_grasp_pose`、`goto_pose`、`goto_home_joint_position`、`open_gripper`、`close_gripper`。

visual 版则使用视觉分割和抓取规划，适合非 privileged evaluation。

### 7.2 Spill wipe

主要文件：

- `capx/integrations/franka/spill_wipe.py`
- `capx/integrations/franka/spill_wipe_privileged.py`

spill wipe 的关键差异是 TCP offset：注册时用 `[0.0, 0.0, -0.0158]`，因为 robosuite Panda 末端有 sponge attachment。普通 spill wipe API 当前只暴露：

- `get_object_pose`
- `goto_pose`

这说明它不是抓取任务 API，而是面向擦拭区域定位和末端运动的最小工具集。

### 7.3 Handover

主要文件：

- `capx/integrations/franka/handover.py`
- `capx/integrations/franka/handover_privileged.py`
- `capx/integrations/franka/handover_reduced.py`
- `capx/integrations/franka/handover_reduced_exampleless.py`

非 privileged `FrankaHandoverApi` 暴露：

- `get_object_pose`
- `goto_pose_arm0`
- `goto_pose_arm1`
- `open_gripper_arm0`
- `open_gripper_arm1`
- `close_gripper_arm0`
- `close_gripper_arm1`

它显式区分 arm0/arm1。坐标系以 robot0 base frame 为主；arm1 pose 会通过 helper 做 frame transform。

### 7.4 Two-arm lift

主要文件：

- `capx/integrations/franka/two_arm_lift.py`
- `capx/integrations/franka/two_arm_lift_privileged.py`

非 privileged `FrankaTwoArmLiftApi` 暴露：

- `get_handle0_pos`
- `get_handle1_pos`
- `get_arm0_gripper_pose`
- `get_arm1_gripper_pose`
- `goto_pose_arm0`
- `goto_pose_arm1`
- `goto_pose_both`
- arm0/arm1 gripper open/close

它直接围绕 pot handles 组织 API，而不是暴露通用 `get_object_pose`。这类设计对特定任务成功率更友好，但难以迁移到其他双臂任务。

## 8. R1Pro API

主要文件：

- `capx/integrations/r1pro/control.py`
- `capx/envs/simulators/r1pro_b1k.py`

`R1ProControlApi` 是一个移动操作综合 API，而不是单纯 manipulator API。exposed functions 覆盖：

| Group | Functions |
| --- | --- |
| Vision | `segment_sam3_text_prompt`, `segment_sam3_point_prompt`, `point_prompt_molmo`, `get_sam3_mask` |
| Object reasoning | `get_object_pose`, `sample_grasp_pose`, `find_object_base_rotate`, `find_object_torso_rotate` |
| Navigation | `navigate_to_pose`, `get_navigation_pose`, `get_robot_position`, `reset_torso` |
| Arm control | `solve_ik`, `move_hand`, `move_to_joint_positions`, `lift_arm`, `get_current_eef_pose`, `get_current_joint_positions` |
| Gripper / grasp | `open_gripper`, `close_gripper`, `grasp_object`, `check_object_in_hand` |
| Debug / output | `get_env_observation`, `write_video`, `save_current_observation` |

它比 Franka API 多出 base navigation、torso、dual-arm/hand-specific state mapping 等逻辑。内部使用 R1Pro URDF 的 PyRoKi context，并维护 PyRoKi joint order 与 controller joint order 的 mapping。

## 9. Vision and motion integration helpers

`capx/integrations/vision/` 与 `capx/integrations/motion/` 多数不是 `ApiBase` 子类，而是被 Franka/R1Pro API 初始化和调用的 service client/helper。

### 9.1 Vision

| Module | Typical initializer | 用途 |
| --- | --- | --- |
| `vision/sam3.py` | `init_sam3()`, `init_sam3_point_prompt()` | text/point prompted segmentation。 |
| `vision/sam2.py` | `init_sam2()`, `init_sam2_point_prompt()` | box/point/global segmentation。 |
| `vision/owlvit.py` | `init_owlvit()` | open-vocabulary detection。 |
| `vision/molmo.py` | `init_molmo()` | image point prompt / object coordinate localization。 |
| `vision/graspnet.py` | `init_contact_graspnet()`, `init_contact_graspnet_point_clouds()` | depth/point-cloud grasp candidate planning。 |

这些 helper 当前通常在 API `__init__` 中初始化。多 worker evaluation 时，这可能导致启动慢或 GPU 显存占用高。部分服务是 server-based，部分如 OWL-ViT 可能在 worker 内直接加载模型。

### 9.2 Motion

| Module | 用途 |
| --- | --- |
| `motion/pyroki.py` | 初始化 PyRoKi IK / trajopt client。 |
| `motion/pyroki_context.py` | 缓存 robot model context，例如 Panda 或 R1Pro URDF。 |
| `motion/pyroki_snippets/` | 本地 snippets，包括 IK、rest cost、manipulability、collision、trajopt 等。 |
| `motion/curobo.py`, `motion/curobo_api.py` | cuRobo world/trajectory planning 相关，部分 LIBERO API 中预留但默认不暴露。 |

Robosuite/LIBERO 常见路径是：

```text
target pose
  -> apply TCP offset
  -> PyRoKi solve IK
  -> extract 7 arm joints
  -> env.move_to_joints_blocking()
```

RLBench 常见路径是：

```text
target pose
  -> env.move_to_pose()
  -> HTTP adapter
  -> RLBench/PyRep path planning
```

## 10. Observation and frame assumptions

不同 API 对 observation schema 的依赖如下：

| Environment family | Main camera key | Wrist key | Robot state | Object state |
| --- | --- | --- | --- | --- |
| Robosuite single-arm | `robot0_robotview` | optional | `robot_joint_pos`, `robot_cartesian_pos` | `cube_poses`, `nut_poses`, task-specific |
| Robosuite two-arm | `agentview`, alias `robot0_robotview` | optional | `robot0_cartesian_pos`, `robot1_cartesian_pos` | `pot_poses`, `hammer_poses` |
| LIBERO | `agentview` | `robot0_eye_in_hand` | `robot_joint_pos`, `robot_cartesian_pos` | helper `_get_object_pose()` or visual point cloud |
| RLBench remote | `robot0_robotview` from front camera | `robot0_eye_in_hand` if available | `robot0_joint_pos`, `robot0_eef_pos`, `robot0_eef_quat` | generic `object_poses` |
| R1Pro | task-specific camera observation | task-specific | mobile base + arm/eef state | visual/object helpers |

Common conventions:

- Positions are meters.
- Quaternions exposed to LLM are intended to be WXYZ.
- Camera intrinsics are `(3, 3)`.
- Camera pose/extrinsics often appear as `pose_mat`; some older code also uses `pose` as `[x, y, z, qw, qx, qy, qz]`.
- Depth shape is inconsistent across envs/API helpers: some envs return `(H, W, 1)`, many API wrappers squeeze to `(H, W)`.

## 11. Return value and error semantics

Current API behavior is useful but not fully uniform.

| Function family | Current variation |
| --- | --- |
| `get_object_pose` | Robosuite often returns `(pos, quat, bbox_extent_or_None)`; LIBERO returns `(pos, quat)` or `(None, None)`; RLBench returns `(pos, quat)`; privileged task APIs vary by task. |
| `sample_grasp_pose` | Usually `(pos, quat)`; R1Pro may return lists or `(None, None)` depending on planning path. |
| `goto_pose` | Robosuite/LIBERO returns `None`; RLBench returns a structured `dict` with `ok` status. |
| `move_to_joints` | Usually returns `None`, even if low-level env has structured result. |
| perception failures | Some paths raise `ValueError`/`RuntimeError`; some return empty lists; LIBERO pose can return `(None, None)`. |

Implications:

- Prompt examples must match the selected API, especially tuple arity.
- Oracle code is not trivially portable across Robosuite, LIBERO and RLBench APIs.
- Multi-turn repair can benefit from structured errors, but only RLBench currently exposes this consistently through `goto_pose()`.

## 12. Coupling with low-level env

API wrappers depend on low-level env methods that are not part of `BaseEnv`:

| Env helper | Used by |
| --- | --- |
| `move_to_joints_blocking(joints, ...)` | Most Franka/LIBERO/R1Pro arm APIs. |
| `move_to_joints_blocking_arm1()` / `move_to_joints_blocking_both()` | bimanual APIs. |
| `_set_gripper(fraction)` / `_step_once()` | Robosuite/LIBERO gripper wrappers. |
| `move_to_pose()` | RLBench remote API. |
| `open_gripper()` / `close_gripper()` | RLBench remote API and some env-specific APIs. |
| `_get_object_pose()` / `_get_all_object_poses()` | LIBERO privileged API. |
| task-specific obs keys | privileged and task-specific Franka APIs. |

This is intentional but important: `ApiBase` itself is generic, while each concrete API assumes a specific env family. A YAML config must pair compatible `low_level` and `apis`.

## 13. Design strengths

- API docstrings are colocated with executable wrappers, so prompt documentation tracks code better than external docs.
- The same low-level env can support multiple abstraction levels: privileged, full visual, reduced, skill-library.
- Task-specific APIs keep prompt small and improve likely task success.
- Reduced APIs make intermediate perception/geometry/motion steps explicit and inspectable.
- RLBench integration cleanly respects Docker isolation: host side API does not import RLBench/PyRep.
- `_log_step()` gives a clean hook for Web UI visualization without polluting non-Web runs.

## 14. Main risks and maintenance issues

### 14.1 Function name collisions

All exposed functions are injected as bare names into the same execution globals. If two APIs expose `get_observation` or `open_gripper`, the later one wins. This is usually fine for current configs, but risky for future multi-API composition.

Possible mitigation:

- Keep YAML configs to one control API family plus optional non-overlapping debug APIs.
- Prefer `APIS["ApiName"].function()` in examples when multiple APIs are enabled.
- Add a collision check in `CodeExecutionEnvBase` and warn or error.

### 14.2 Inconsistent tuple arity

`get_object_pose()` is the most visible inconsistency:

- Robosuite high-level returns 3 values.
- LIBERO and RLBench return 2 values.
- LIBERO visual can return `(None, None)`.

Possible mitigation:

- Standardize future APIs around either 2-tuple plus optional helper for extent, or a dict return.
- At minimum, ensure each API's docstring and oracle code match exactly.

### 14.3 Mixed failure semantics

Some APIs raise exceptions, some return empty lists, some return `(None, None)`, and RLBench returns structured `ok=False`. This affects multi-turn correction and benchmark comparability.

Possible mitigation:

- Use structured results for motion-planning failures where failures are expected.
- Reserve exceptions for programming/configuration errors.
- Document perception failure behavior in each docstring.

### 14.4 Heavy initialization inside API constructors

Many APIs initialize SAM, Molmo, GraspNet, PyRoKi, OWL-ViT in `__init__`. This makes API construction expensive and can cause GPU memory pressure under multi-worker evaluation.

Possible mitigation:

- Prefer lazy initialization on first function call for heavyweight services.
- Move model clients toward shared server-based launch profiles where possible.
- Document per-worker GPU memory expectations in relevant configs.

### 14.5 Debug artifacts written to cwd

Some visual APIs write files such as `depth_image.jpg` or `segmentation_image.jpg` in the current directory. In batch evaluation this can pollute source directories or cause races between workers.

Possible mitigation:

- Gate debug saves behind `debug=True`.
- Write under the trial output directory with request/trial IDs.
- Avoid fixed filenames in concurrent runs.

### 14.6 API docs can drift from exposed functions

Some class docstrings list functions that are commented out or not exposed in `functions()`. Since `combined_doc()` uses actual function docstrings rather than class docstring, this mainly affects human readers, but it still creates confusion during maintenance.

Possible mitigation:

- Treat `functions()` as source of truth.
- Keep class-level docstrings short or generated from `functions()`.

## 15. Recommendations for adding new APIs

When adding a new API wrapper:

1. Decide the abstraction level first: high-level skill, reduced primitives, privileged oracle, or task-specific helper.
2. Pair it explicitly with compatible low-level env methods and observation keys.
3. Document frame, units, quaternion order, array shape, dtype, and failure behavior in every exposed function docstring.
4. Keep `functions()` small and deliberate; helper methods do not need to be exposed.
5. Use `_log_step()` for user-visible operations if Web UI support matters.
6. Avoid writing debug files unless debug mode and output path are explicit.
7. Prefer structured returns for expected robotics failures such as IK/path-planning/no grasp candidates.
8. Register in `capx/integrations/__init__.py` and reference by name in YAML.
9. Add a smoke/oracle path that verifies at least observation, one perception or state query, one motion, gripper if relevant, reward/success.

## 16. Quick selection guide

| Goal | Recommended API style |
| --- | --- |
| Fast oracle or debugging with known object state | privileged API |
| Benchmark LLM task sequencing, not perception | high-level non-reduced API |
| Benchmark visual reasoning and geometry composition | reduced API |
| Reuse common helper code learned from previous trials | reduced skill-library API |
| Remote RLBench task through Docker adapter | `FrankaRLBenchApi` |
| Task has unique objects/actions like nut handles or pot handles | task-specific API |
| Mobile manipulation / BEHAVIOR | `R1ProControlApi` |

## 17. Summary

`capx/integrations` is the main user-facing contract for generated robot code. The repository currently supports multiple API philosophies at once:

- Robosuite and LIBERO often put perception and IK in the integration layer.
- RLBench keeps integration thin because the remote adapter owns high-level motion.
- Privileged APIs expose simulator state for oracle/debug.
- Reduced APIs expose primitives for more demanding code-as-policy evaluation.
- Task-specific APIs intentionally trade generality for task success and smaller prompts.

The most important maintenance rule is to keep `functions()` and docstrings precise. Once an API is selected in YAML, these functions become the model's world.
