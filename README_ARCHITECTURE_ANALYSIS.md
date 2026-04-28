# CaP-X 目录与架构分析

本文档从代码组织、核心架构、运行流程和重要模块调用关系几个角度梳理 `cap-x` 目录。它侧重帮助后续读代码、改任务、接 API、跑评估和定位问题。

## 1. 项目定位

`cap-x` 是一个用于机器人操作任务中 Code-as-Policy 智能体评测与改进的框架。它让语言模型根据任务 prompt 生成 Python 代码，代码通过预定义机器人 API 与底层仿真或真实机器人环境交互，并记录 reward、任务完成情况、代码、日志、图片和视频。

官方 README 将系统拆成四个部分：

- **CaP-Gym**：交互式 Gymnasium 环境，模型通过 Python 代码组合感知与控制原语来控制机器人。
- **CaP-Bench**：覆盖不同抽象层、单轮/多轮、视觉反馈模式的系统评测。
- **CaP-Agent0**：免训练 agent 框架，包含多轮视觉差分、技能库、并行 ensemble 等机制。
- **CaP-RL**：通过环境 reward 对代码智能体做 GRPO/VeRL 强化学习。

从工程视角看，主线是：

```text
YAML 配置
  -> launch.py 加载配置和 CLI 覆盖项
  -> runner.py 启动 API servers 并调度 trials
  -> trial.py 查询 LLM / VLM、执行代码、多轮决策、保存结果
  -> CodeExecutionEnvBase 执行生成代码
  -> API functions 调用低层环境
  -> Robosuite / LIBERO / BEHAVIOR / real robot
```

## 2. 顶层目录总览

```text
cap-x/
├── README.md                  # 官方说明：安装、快速开始、论文引用
├── MyReadme.md                # 本地索引页；已加入本文档跳转
├── pyproject.toml             # Python 包、依赖、optional extras、ruff 配置
├── uv.lock                    # uv 锁文件
├── LICENSE                    # MIT License
├── capx/                      # Python 主包
├── env_configs/               # 评测/实验 YAML 配置
├── docs/                      # 官方专题文档
├── scripts/                   # 实验、回归测试、技能库编译脚本
├── tests/                     # 单元和集成测试
├── web-ui/                    # React/Vite 交互式前端
└── verl_agent_reward/         # VeRL/GRPO 训练用 reward 文件
```

另有 `capx/third_party/`，包含 robosuite、LIBERO-PRO、B1K/OmniGibson、cuRobo、SAM3、VeRL、Contact-GraspNet 等第三方依赖或子模块。它们是运行不同 simulator / perception / RL 功能的基础，但主项目逻辑主要在 `capx/`、`env_configs/`、`scripts/` 和 `web-ui/`。

## 3. 重要组成部分

### 3.1 `capx/envs`：环境、试验和评测调度核心

`capx/envs` 是主执行链路的中心。

关键文件：

- `launch.py`：CLI 入口。定义 `LaunchArgs`，加载 YAML，启动 API server，选择 Web UI 或 headless 批量评测。
- `runner.py`：批量 trial 调度。负责启动/停止配置里的 API servers，设置输出目录，单 worker 或多 worker 运行，timeout 与 retry。
- `trial.py`：单个 trial 的核心逻辑。负责 reset、初始视觉反馈、LLM 生成代码、多轮 regenerate/finish 决策、执行代码块、保存 artifacts。
- `base.py`：低层环境抽象 `BaseEnv` 与 `register_env/get_env` 注册表。
- `tasks/base.py`：高层代码执行环境 `CodeExecutionEnvBase`，负责拼 prompt、绑定 API 函数、执行模型生成的 Python 代码。
- `configs/loader.py` 与 `configs/instantiate.py`：类似 Hydra 的 YAML 加载与 `_target_` 实例化。
- `simulators/`：Robosuite、LIBERO、BEHAVIOR、真实 Franka 等低层环境实现。
- `tasks/`：面向代码执行的任务环境，如 Franka cube stack、lift、nut assembly、LIBERO、R1Pro 等。
- `assets/`：MuJoCo XML、URDF/STL/OBJ 资产。
- `adapters/`：对 LIBERO、Robosuite 等外部环境的 wrapper。
- `scripts/`：批量运行入口，如 `run_batch.py`、`run_libero_batch.py`。

### 3.2 `capx/integrations`：暴露给模型代码的 API 层

`integrations` 定义“模型生成代码可以调用什么函数”。高层环境会把 API 的函数签名和 docstring 拼进 prompt，然后把函数绑定到代码执行 namespace。

关键文件和目录：

- `base_api.py`：`ApiBase`、`register_api/get_api/list_apis`。`ApiBase.combined_doc()` 会聚合函数签名和 docstring，作为 prompt 中的 API 文档。
- `__init__.py`：统一注册 API 名称，例如 `FrankaControlApi`、`FrankaControlApiReduced`、`FrankaLiberoApi`、`R1ProControlApi` 等。
- `franka/`：Franka 控制 API，覆盖普通、privileged、reduced、skill library、handover、two-arm lift、nut assembly、LIBERO、spill wipe 等变体。
- `vision/`：SAM2、SAM3、OWL-ViT、Molmo、Contact-GraspNet 等视觉/抓取感知封装。
- `motion/`：PyRoKi、cuRobo、IK/trajopt 相关封装和 snippets。
- `r1pro/`：R1Pro/BEHAVIOR 任务控制 API。
- `robosuite/`：控制器配置等 robosuite 集成资源。
- `libero/`：LIBERO 集成入口。

一个典型 API 如 `FrankaControlApi` 暴露：

- `get_object_pose(object_name, return_bbox_extent=False)`
- `sample_grasp_pose(object_name)`
- `goto_pose(position, quaternion_wxyz, z_approach=...)`
- `open_gripper()`
- `close_gripper()`
- simulation 中额外暴露 `home_pose()`

它内部会调用视觉模型做检测/分割，利用 depth 生成点云和 OBB，再调用抓取/IK/低层环境动作接口。

### 3.3 `capx/llm`：模型查询客户端

`capx/llm/client.py` 封装对 LLM/VLM 的 HTTP 请求：

- `ModelQueryArgs`：模型名、server URL、temperature、max tokens、reasoning effort 等。
- `query_model()`：非流式查询，按模型类型组装 payload。
- `query_model_streaming()`：Web UI 用的流式查询。
- `query_model_ensemble()` / `query_single_model_ensemble()`：并行候选生成与综合。
- 模型分组：`GPT_MODELS`、`VLM_MODELS`、`CLAUDE_MODELS`、`OSS_MODELS`、`OPENROUTER_MODELS`。

默认评测通过 OpenAI-compatible `/chat/completions` 代理访问模型。OpenRouter 由 `capx/serving/openrouter_server.py` 提供本地代理。

### 3.4 `capx/serving`：本地 API server

`serving` 负责启动感知、规划和模型代理服务：

- `launch_servers.py`：统一 server launcher。支持 `default/full/minimal` profile，也能从 YAML 的 `api_servers` 解析需要启动的服务，并做 GPU 分配。
- `launch_sam3_server.py`、`launch_sam2_server.py`、`launch_owlvit_server.py`：视觉服务。
- `launch_contact_graspnet_server.py`：抓取规划服务。
- `launch_pyroki_server.py`、`launch_curobo_server.py`：IK/运动规划服务。
- `openrouter_server.py`：OpenRouter 本地代理。
- `vllm_server.py`：vLLM 相关服务入口。
- `assets/`：服务端使用的机器人几何或配置资产。

这些服务既可以由 YAML 自动启动，也可以通过 `launch_servers.py` 提前启动以复用。

### 3.5 `capx/utils`：评测、日志、视觉、视频和并行工具

常用文件：

- `launch_utils.py`：加载配置、构造多轮 prompt、解析模型 regenerate/finish、保存 trial artifacts、汇总结果。
- `parallel_eval.py`：多进程/并行评测调度。
- `execution_logger.py`：API 执行步骤日志，主要给 Web UI 展示。
- `camera_utils.py`、`depth_utils.py`、`graspnet_utils.py`、`visualization_utils.py`：图像、深度、点云、可视化工具。
- `video_utils.py`：视频编码与写出。
- `serve_utils.py`、`msgpack_server_client_utils.py`：服务端/客户端通讯辅助。
- `eval_utils.py`：评测辅助。

### 3.6 `capx/web` 与 `web-ui`：交互式 Web UI

后端 `capx/web`：

- `server.py`：FastAPI 应用，提供配置列表、加载配置、启动/停止 trial、WebSocket 状态推送、Viser 代理等接口。
- `async_trial_runner.py`：异步 trial runner，把模型流式输出、执行状态、用户输入等待等桥接到 WebSocket。
- `session_manager.py`：session 生命周期管理。
- `models.py`：Pydantic 请求/响应和事件模型。
- `execution_logger.py`：Web UI 执行日志桥接。

前端 `web-ui`：

- `src/App.tsx`：主界面布局，配置加载、模型选择、状态栏、左右分栏。
- `src/hooks/useTrialState.ts`：trial 状态、REST API 调用、WebSocket 连接。
- `src/hooks/useWebSocket.ts`：WebSocket 管理。
- `src/components/`：聊天面板、消息列表、代码块、执行细节、图片展示、可视化面板、配置启动控件。
- `package.json`、`vite.config.ts`、`tailwind.config.js`：前端构建配置。

### 3.7 `capx/skills`：技能库

技能库用于从成功 trial 的代码中提取可复用函数，积累后注入后续 prompt/namespace。

- `library.py`：`SkillLibrary`，负责持久化 `.capx_skills.json`、提取函数、promotion、文档生成、namespace 注入。
- `extractor.py`：从代码中抽取函数定义。
- `claude_integration.py`：与 Claude/LLM 相关的技能整理集成。

`trial.py` 中有 opt-in 逻辑：当 `config["evolve_skill_library"]` 开启且任务完成时，会从最终代码中抽取技能并保存。

### 3.8 `env_configs`：实验配置矩阵

`env_configs/` 是跑评测最重要的配置目录。每个 YAML 通常包含：

- `env._target_`：高层 code env 类。
- `env.cfg._target_`：`CodeExecEnvConfig`。
- `low_level`：低层环境注册名，例如 `franka_robosuite_cubes_low_level`。
- `privileged`：是否使用 privileged 状态。
- `apis`：暴露给模型代码的 API 名称。
- `prompt`：任务 prompt。
- `multi_turn_prompt`：多轮 regenerate/finish 决策 prompt。
- `api_servers`：需要自动启动的 SAM3、GraspNet、PyRoKi 等服务。
- `record_video`、`use_visual_feedback`、`use_img_differencing`、`use_video_differencing`、`output_dir`、`trials`、`num_workers` 等运行参数。

子目录含义：

- `cube_stack/`、`cube_lifting/`、`cube_restack/`：方块相关 Robosuite 任务。
- `nut_assembly/`：螺母装配任务。
- `spill_wipe/`：擦拭/清理任务。
- `two_arm_lift/`、`two_arm_handover/`：双臂任务。
- `libero/`：LIBERO-PRO 任务。
- `r1pro/`：BEHAVIOR/R1Pro 任务。
- `real/`：真实 Franka 配置。
- `human_oracle_code/`：人工 oracle 代码配置。
- `hillclimb/`：ensemble / multimodel / debug 等内部实验配置。

文件名中的常见后缀：

- `privileged`：使用 privileged API 或环境状态，通常更容易。
- `reduced_api`：只暴露更少、更高层或更受限的 API。
- `exampleless`：prompt 中不提供示例。
- `multiturn`：启用多轮执行和决策。
- `vdm`：visual differencing model，执行后用视觉差分描述变化。
- `vf`：visual feedback，直接给当前图像。
- `skill_lib`：启用技能库相关 API/prompt。

### 3.9 `docs`：官方专题文档

- `configuration.md`：YAML 和 CLI 配置说明。
- `adding-environments.md`：新增 simulator/task/env 配置。
- `adding-apis.md`：新增 API 并注册到 prompt。
- `development.md`：测试、lint、依赖和开发提示。
- `libero-tasks.md`：LIBERO-PRO 任务运行。
- `behavior-tasks.md`：BEHAVIOR/R1Pro 任务。
- `real-franka.md`：真实 Franka bringup。
- `rl-training.md`：CaP-RL / GRPO / VeRL 训练。

### 3.10 `scripts`、`tests`、`verl_agent_reward`

`scripts/`：

- `regression_test.sh`：回归/冒烟测试。
- `start_servers_and_eval.sh`：服务和评测组合启动。
- `run_experiment_now.sh`、`experiment_*.sh`：实验脚本。
- `train_franka_grpo.sh`：RL 训练入口。
- `first_frames_video.py`：视频辅助。
- `skill_library_compilation/`：从 eval 输出分析、解析、汇总和编译技能库。

`tests/`：

- `test_environments.py`、`test_robosuite_setup.py`、`test_libero.py`：环境和基础集成测试。
- `tests/integrations/`：SAM2/SAM3/OWL-ViT/Molmo/PyRoKi/GraspNet/depth 等集成测试和可视化测试。

`verl_agent_reward/`：

- `capx_franka_reward.py`、`hyrl_franka_reward.py`：VeRL/GRPO 使用的 reward 逻辑。

## 4. 核心类与注册表

### 4.1 低层环境：`BaseEnv`

位置：`capx/envs/base.py`

`BaseEnv` 继承 Gymnasium `Env`，规定低层环境必须实现：

- `reset(seed, options)`
- `step(action)`
- `get_observation()`
- `compute_reward()`
- `task_completed()`

低层环境通过 `register_env(name, factory)` 注册，通过 `get_env(name, privileged, enable_render, viser_debug)` 获取。注册发生在 `capx/envs/simulators/__init__.py`。

### 4.2 高层代码执行环境：`CodeExecutionEnvBase`

位置：`capx/envs/tasks/base.py`

它是 LLM 代码与机器人环境之间的关键桥梁：

1. 根据 `CodeExecEnvConfig.low_level` 构造低层环境。
2. 根据 `CodeExecEnvConfig.apis` 调用 `get_api()` 创建 API 实例。
3. 调用每个 API 的 `combined_doc()`，把函数签名和 docstring 拼进 prompt。
4. 在代码执行 namespace 中注入 `env`、`APIS`、`obs`、`INPUTS`、`RESULT` 和所有 API 函数。
5. `step(action: str)` 执行模型生成的 Python 代码，捕获 stdout/stderr，计算 reward/task_completed。

因此模型生成的代码可以直接写：

```python
pos, quat, _ = get_object_pose("red cube", return_bbox_extent=True)
goto_pose(pos, quat)
close_gripper()
```

这些函数不是 Python 全局内置，而是由 `CodeExecutionEnvBase._init_exec_globals()` 和 `_exec_user_code()` 从 API 绑定进去的。

### 4.3 API：`ApiBase`

位置：`capx/integrations/base_api.py`

所有控制/感知 API 继承 `ApiBase`，核心约定是实现：

- `functions() -> dict[str, Callable]`
- 可选调用 `_log_step()` / `_log_step_update()` 给 Web UI 记录执行过程。

`ApiBase.combined_doc()` 会自动读取函数签名和 docstring。这个机制让“代码能调用什么”与“prompt 里告诉模型什么”保持一致。

### 4.4 YAML 实例化：`DictLoader` 与 `instantiate`

位置：

- `capx/envs/configs/loader.py`
- `capx/envs/configs/instantiate.py`

YAML 中的 `_target_` 字段会被定位为 Python 对象并递归实例化。例如：

```yaml
env:
  _target_: capx.envs.tasks.franka.franka_pick_place.FrankaPickPlaceCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: franka_robosuite_cubes_low_level
    apis:
      - FrankaControlApiReducedSkillLibrary
```

调用关系是：

```text
DictLoader.load(config_path)
  -> 得到普通 dict
  -> _load_config() 拆出 env_factory/config/api_servers
  -> instantiate(env_factory)
  -> FrankaPickPlaceCodeEnv(CodeExecEnvConfig(...))
  -> CodeExecutionEnvBase 构造 low_level 和 APIs
```

## 5. Headless 评测调用关系

入口命令通常是：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
  --model "google/gemini-3.1-pro-preview"
```

主要调用链：

```text
launch.py::main(args)
  -> launch_utils._load_config(args)
       -> DictLoader.load([config_path])
       -> 合并 CLI 与 YAML 配置
       -> 返回 env_factory, merged_config, api_servers
  -> runner._start_api_servers(api_servers)
       -> run_server_proc(api_server)
       -> instantiate(api_server) in child process
  -> runner._run_headless_trials(args, env_factory, config, start_time)
       -> _setup_output_dir()
       -> if num_workers > 1:
            parallel_eval.run_parallel_with_setup()
          else:
            _run_trial_batch()
       -> _print_and_save_summary()
  -> runner._stop_api_servers()
```

单个 trial 调用链：

```text
runner._run_trial_with_retries()
  -> runner._run_single_trial_with_timeout()
  -> trial._run_single_trial()
       -> env.reset(options={"trial": trial}, seed=trial)
       -> _capture_initial_visual_feedback()
       -> if use_oracle_code:
            raw_code = env.oracle_code
          else:
            _query_initial_code()
              -> llm.client.query_model()
       -> _extract_code(raw_code)
       -> while code blocks remain:
            env.step(code)
              -> CodeExecutionEnvBase._exec_user_code()
              -> generated code calls API functions
              -> API functions call low_level_env methods
              -> low_level_env drives simulator/robot
            if multi_turn_prompt:
              -> _handle_multi_turn_step()
                   -> render / video frame collection
                   -> optional VDM feedback query
                   -> build multi-turn decision prompt
                   -> query_model()
                   -> _parse_multi_turn_decision()
                   -> regenerate / finish
            -> _save_trial_artifacts()
       -> save final code/logs/images/videos
       -> optional SkillLibrary extraction
       -> return TrialSummary
```

## 6. 生成代码如何真正控制机器人

以 Robosuite Franka cube stack 为例：

```text
YAML:
  low_level: franka_robosuite_cubes_low_level
  apis: [FrankaControlApiReducedSkillLibrary]

CodeExecutionEnvBase:
  get_env("franka_robosuite_cubes_low_level")
    -> FrankaRobosuiteCubesLowLevel
  get_api("FrankaControlApiReducedSkillLibrary")
    -> FrankaControlApiReducedSkillLibrary(env)
  prompt includes API docs
  namespace includes exposed functions

LLM generated code:
  get_object_pose("red cube")
  sample_grasp_pose("red cube")
  goto_pose(...)
  close_gripper()

Franka API:
  get_object_pose()
    -> env.get_observation()
    -> obs_get_rgb()
    -> SAM3 or OWL-ViT + SAM2
    -> depth_to_pointcloud/depth_color_to_pointcloud
    -> Open3D OBB
  goto_pose()
    -> PyRoKi IK
    -> low_level_env.move_to_joints_blocking/non_blocking()
  open_gripper()/close_gripper()
    -> low_level_env._set_gripper()
    -> low_level_env._step_once()

RobosuiteBaseEnv:
  _build_action()
  robosuite_env.step(...)
  compute_reward()
  task_completed()
```

这条链路里，高层环境不直接“理解”任务几何，它只是执行代码、注入工具、收集结果；真正的任务状态来自低层环境，真正的动作封装来自 API。

## 7. 多轮与视觉差分机制

多轮任务由 YAML 的 `multi_turn_prompt` 开启。每个代码块执行后，`trial.py` 会构造一个新的决策 prompt，包含：

- 已执行代码 `executed_code`
- stdout/stderr
- 当前视觉反馈图像（若 `use_visual_feedback`）
- 图像差分描述（若 `use_img_differencing`）
- 视频执行描述（若 `use_video_differencing`）

模型必须返回：

- `REGENERATE` + fenced Python code：替换当前索引之后的代码块。
- `FINISH`：停止 trial。

目前 `_extract_code()` 只抽取一个代码块；代码中保留了 `breakpoint_code_block()` 风格拆块的注释，但当前实现没有按它切分。

视觉差分分两类：

- **Image VDM**：保存前后两帧，调用 VLM 描述环境变化。
- **Video VDM**：保存本轮执行的视频帧，调用 VLM 描述机器人执行过程。

如果启用 wrist camera，则主视角和腕部视角都会参与部分反馈流程。

## 8. 输出产物

每个 trial 的输出目录一般形如：

```text
outputs/.../<model>/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

常见产物：

- `code.py`：最终执行代码，带 `# Code block N` 注释。
- `raw_response.sh`：模型原始响应。
- `all_responses.json`：初始响应、多轮响应、reasoning、决策等。
- `summary.txt`：该 trial 的 stdout/stderr/reward/task_completed 等。
- `visual_feedback_XX.png`：视觉反馈图片。
- `video_turn_XX.mp4`、`video_combined.mp4`：分轮和完整视频。
- `ensemble_candidates*.txt`、`ensemble_synthesis*.txt`：ensemble 模式产物。
- `prompts_and_responses/`：保存 prompt 文本。

批量评测结束后，`summaries.txt` 记录模型、配置、git commit、成功率、平均 reward、平均代码块数、多轮 regenerate/finish 统计和耗时。

## 9. Web UI 调用关系

Web UI 启动：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
  --web-ui True
```

调用链：

```text
launch.py::main()
  -> config["web_ui"] == True
  -> _run_web_ui()
       -> _ensure_frontend_built()
       -> capx.web.server.create_app()
       -> uvicorn.run(...)

Frontend App.tsx
  -> /api/default-config
  -> /api/load-config
  -> /api/start-trial
  -> WebSocket session events

Backend server.py
  -> _load_config()
  -> create Session
  -> async_trial_runner.run_trial_async()
  -> streaming model output + execution logs + images
```

Web UI 与 headless 的区别在于：Web UI 更强调单 trial、流式模型输出、人工确认/注入 prompt、实时执行日志和图像/Viser 可视化；headless 更强调批量并行评测和统计。

## 10. 新增功能时的入口

### 新增一个低层环境

参考：

- `docs/adding-environments.md`
- `capx/envs/base.py`
- `capx/envs/simulators/__init__.py`
- `capx/envs/simulators/robosuite_base.py`

基本步骤：

1. 实现 `BaseEnv` 子类。
2. 实现 `reset/get_observation/compute_reward/task_completed`。
3. 在 `simulators/__init__.py` 中 `register_env()`。
4. 写一个高层 `CodeExecutionEnvBase` 子类或复用已有任务环境。
5. 新增 YAML。

### 新增一个 API

参考：

- `docs/adding-apis.md`
- `capx/integrations/base_api.py`
- `capx/integrations/__init__.py`

基本步骤：

1. 继承 `ApiBase`。
2. 实现 `functions()`，返回要暴露给模型的函数。
3. 为函数写清楚 docstring，prompt 会自动包含。
4. 在 `integrations/__init__.py` 中 `register_api()`。
5. 在 YAML 的 `apis` 中引用新 API 名称。

### 新增一个评测配置

参考任一 `env_configs/*/*.yaml`。

关键是选对：

- `env._target_`
- `low_level`
- `apis`
- `prompt`
- 是否需要 `multi_turn_prompt`
- 是否需要 `api_servers`
- `trials/num_workers/output_dir`

## 11. 需要特别注意的实现细节

- `CodeExecutionEnvBase` 使用 `exec()` 在进程内执行生成代码，并且允许完整 imports。这对研究灵活性有利，但不是强隔离沙箱。
- `get_env()` 和 `get_api()` 使用 `lru_cache`，同名环境/API factory 会被复用；并行 worker 中每个 worker 会构造自己的环境。
- `runner.py` 用 `SIGALRM` 给单 trial 加 wall-clock timeout，并最多重试 3 次。
- `launch.py` 会默认设置 `MUJOCO_GL=egl`，适合 headless MuJoCo/Robosuite。
- Robosuite 与 LIBERO 依赖版本冲突，官方 README 建议分别使用不同 extras/venv。
- `trial.py` 中多轮上限 `MULTITURN_LIMIT = 10`。
- `record_video` 需要 `output_dir`，否则 `runner.py` 会报错。
- `api_servers` 端口如果已被占用，`runner.py` 会跳过启动并等待已有服务。
- `FrankaControlApi` 初始化较重，会加载 GraspNet、SAM3/OWL-ViT/SAM2、PyRoKi；reduced/privileged API 适合不同 benchmark tier。
- `SkillLibrary` 的自动提取是 opt-in，并且当前主要从成功代码中的函数定义抽取，是否真正注入取决于具体 API/prompt 配置。

## 12. 一句话心智模型

`cap-x` 的核心不是“一个机器人策略”，而是一个可配置的代码生成评测管线：YAML 决定任务、环境、API 和反馈方式；LLM 生成 Python；`CodeExecutionEnvBase` 执行 Python；API 把 Python 调用翻译成感知、规划和控制；低层环境给出 reward 和完成信号；runner 保存所有证据并汇总评测结果。
