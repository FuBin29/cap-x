# CaP-X 项目复现、扩展与验证研究计划

本文档基于 `README.md`、`docs/` 官方文档、`README_ARCHITECTURE_ANALYSIS.md` 以及关键代码整理，目标是把 CaP-X 的复现、第三方服务配置、模型 API 接入、Robosuite 任务验证、RLBench/ManiSkill 环境适配、func tool 能力分析和新增 tool 测试拆成可逐条执行的研究计划。

## 0. 总体目标与工作原则

### 0.1 研究目标

1. 以 Robosuite 为主要复现对象，先完成最小可运行链路。
2. 配置并验证第三方库，包括 Robosuite、SAM3、Contact-GraspNet、PyRoKi、可选 cuRobo、可选 LIBERO/BEHAVIOR。
3. 跑通开源/商业模型的 OpenAI-compatible API 调用，明确 token 类型和第三方平台 token 处理方式。
4. 测试 1-2 个 Robosuite 任务，掌握日志、stdout/stderr、reward、task_completed、视频和 Web UI 可视化结果。
5. 在现有环境抽象上规划 RLBench、ManiSkill 适配路径，完成至少一个 smoke task 的环境 wrapper、任务 prompt 和 API 绑定。
6. 分析当前 func tool/API 是否足以支持这些任务，验证 API 文档注入、API 调用、Web UI 执行日志和底层环境交互是否正常。
7. 新增一个面向研究需求的 func tool，完成注册、prompt 注入、单元测试、oracle 测试和小规模 LLM 评测。

### 0.2 核心执行链路

CaP-X 的主链路是：

```text
YAML 配置
  -> capx/envs/launch.py 解析 CLI 和 YAML
  -> capx/envs/runner.py 启动 api_servers 并调度 trials
  -> capx/envs/trial.py 查询 LLM/VLM、执行代码、多轮决策、保存 artifacts
  -> capx/envs/tasks/base.py 的 CodeExecutionEnvBase 执行模型生成代码
  -> capx/integrations/* 中注册的 API/func tool
  -> capx/envs/simulators/* 中的低层环境
  -> Robosuite / LIBERO / BEHAVIOR / 后续 RLBench / ManiSkill
```

### 0.3 关键参考文件

- 项目说明：`README.md`
- 架构说明：`README_ARCHITECTURE_ANALYSIS.md`
- 配置文档：`docs/configuration.md`
- 新增环境：`docs/adding-environments.md`
- 新增 API/tool：`docs/adding-apis.md`
- 开发与测试：`docs/development.md`
- LIBERO：`docs/libero-tasks.md`
- RL：`docs/rl-training.md`
- CLI 入口：`capx/envs/launch.py`
- 批量运行：`capx/envs/runner.py`
- 单 trial：`capx/envs/trial.py`
- 输出保存：`capx/utils/launch_utils.py`
- 高层代码执行环境：`capx/envs/tasks/base.py`
- 低层环境基类：`capx/envs/base.py`
- Robosuite 基类：`capx/envs/simulators/robosuite_base.py`
- 环境注册：`capx/envs/simulators/__init__.py`
- 任务注册：`capx/envs/tasks/__init__.py`
- API 基类与注册：`capx/integrations/base_api.py`、`capx/integrations/__init__.py`
- Robosuite 配置样例：`env_configs/cube_lifting/`、`env_configs/cube_stack/`
- 环境测试：`tests/test_environments.py`

## 1. 阶段一：环境安装与 Robosuite 复现基础

### 1.1 确认系统前置条件

执行前确认：

- Linux x86_64
- Python 3.10，用于主项目和 Robosuite
- CUDA-capable GPU
- NVIDIA 驱动和 CUDA runtime 可用
- Git submodules 已初始化
- headless 服务器需要 EGL/MuJoCo 渲染支持

建议记录：

```bash
nvidia-smi
python --version
uv --version
git submodule status
```

可考核结果：

- `nvidia-smi` 能看到 GPU。
- `python --version` 后续在 `.venv` 中为 Python 3.10。
- `git submodule status` 不应出现大量未初始化 submodule。

### 1.2 安装主环境和 Robosuite extra

在 `cap-x/` 根目录执行：

```bash
git submodule update --init --recursive
uv python install 3.10
uv venv -p 3.10
source .venv/bin/activate
uv sync
uv sync --extra robosuite
```

注意事项：

- Robosuite 与 LIBERO 的 Robosuite fork 冲突，不能装在同一个 venv 作为长期实验环境。
- Robosuite 主环境使用 `.venv`；LIBERO 后续使用 `.venv-libero`。
- `pyproject.toml` 中 `robosuite` 来源是 `capx/third_party/robosuite`，需要 submodule 正常存在。

可考核结果：

```bash
python -c "import capx; import robosuite; print('ok')"
python -c "from capx.envs.base import list_envs; import capx.envs.simulators; print(list_envs())"
```

期望：

- 第一条输出 `ok`。
- 第二条至少包含 `franka_robosuite_cube_lift_low_level`、`franka_robosuite_cubes_low_level`。

### 1.3 配置 headless 渲染

CaP-X 在 `capx/envs/launch.py` 和 `capx/envs/simulators/robosuite_base.py` 中默认设置了：

```bash
MUJOCO_GL=egl
```

如服务器缺少 EGL：

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1
```

可考核结果：

```bash
python -c "import os; os.environ.setdefault('MUJOCO_GL','egl'); import robosuite; print('robosuite import ok')"
```

如果后续 `render()` 或 `has_offscreen_renderer=True` 报 EGL/OpenGL 错误，优先检查：

- `libegl1`、`libgl1`
- GPU driver
- `MUJOCO_GL=egl`
- 在容器中是否透传 `/dev/nvidia*`

## 2. 阶段二：第三方服务配置与独立验证

### 2.1 需要理解的第三方组件

Robosuite 视觉任务通常依赖以下服务：

- SAM3：开放词汇/文本或点提示分割，默认端口 `8114`
- Contact-GraspNet：抓取姿态规划，默认端口 `8115`
- PyRoKi：IK 和运动规划，默认端口 `8116`
- OWL-ViT/SAM2/Molmo：部分 reduced API 或实验分支会使用
- cuRobo：可选 GPU 运动规划

配置入口：

- YAML 中的 `api_servers`
- `capx/serving/launch_servers.py`
- 单独服务脚本：`capx/serving/launch_sam3_server.py`、`launch_contact_graspnet_server.py`、`launch_pyroki_server.py`

### 2.2 自动启动服务

Robosuite 非 privileged 配置已经在 YAML 中写好服务，例如：

```yaml
api_servers:
  - _target_: capx.serving.launch_sam3_server.main
    device: cuda
    port: 8114
    host: 127.0.0.1
  - _target_: capx.serving.launch_contact_graspnet_server.main
    port: 8115
    host: 127.0.0.1
  - _target_: capx.serving.launch_pyroki_server.main
    port: 8116
    host: 127.0.0.1
    robot: panda_description
    target_link: panda_hand
```

`capx/envs/runner.py::_start_api_servers()` 会：

- 检查端口是否已有服务
- 启动未运行的服务
- 等待端口 ready
- 评测结束后终止子进程

可考核结果：

- 启动评测时终端出现 `API server ... started`。
- 对已存在服务会显示 `already running, skipping`。
- 服务 ready 后显示 `API server on host:port is ready`。

### 2.3 手动预启动服务

适用于多次评测复用同一组服务：

```bash
source .venv/bin/activate

# NOTE: 启动各种后端 API
uv run --no-sync --active capx/serving/launch_servers.py --profile default

uv run --no-sync --active capx/serving/launch_servers.py --profile full
```

profile 选择：

- `minimal`：只启动 PyRoKi，适合 privileged/oracle 快速测试。
- `default`：SAM3 + Contact-GraspNet + PyRoKi，适合非 privileged Robosuite。
- `full`：再加 OWL-ViT + SAM2，适合 reduced API 和更多视觉实验。

可考核结果：

```bash
curl http://127.0.0.1:8116/health
```

如果某服务没有 `/health`，则以端口监听和首次 API 调用成功作为验收。

default 模式工作正常，full 模式报错，目前已拷贝 SAM2 参数，需要适配。

SAM3、SAM2、OWLv2 直接使用本地模型权重。路径： /home/fubin/ckpt/

三种模式都已测试成功。

```bash
hf download google/owlv2-large-patch14-ensemble \
  --local-dir /home/fubin/ckpt/owlv2/owlv2-large-patch14-ensemble

CUDA_VISIBLE_DEVICES=4 python -m capx.serving.launch_owlvit_server \
  --model-name /home/fubin/ckpt/owlv2/owlv2-large-patch14-ensemble \
  --local-files-only \
  --device cuda \
  --port 8117
```

### 2.4 SAM3 权限与 HuggingFace token

SAM3 权重需要 HuggingFace 授权：

1. 在 SAM3 相关页面申请模型权重访问。
2. 生成 HuggingFace access token。
3. 本地登录：

```bash
huggingface-cli login
```

可考核结果：

- 首次启动 SAM3 能下载并缓存权重。
- 后续启动不重复大规模下载。
- SAM3 服务启动失败时，错误信息不再是 401/403 权限问题。

## 3. 阶段三：模型 API 调用与第三方 token 方案

### 3.1 CaP-X 的模型调用方式

CaP-X 默认通过 OpenAI-compatible `/chat/completions` 接口访问模型。关键代码：

- `capx/llm/client.py::query_model()`
- `capx/serving/openrouter_server.py`
- `docs/configuration.md`

CLI 关键参数：

```bash
--server-url http://127.0.0.1:8110/chat/completions
--model google/gemini-3.1-pro-preview
--api-key <optional-token>
--temperature 1.0
--max-tokens 20480
--reasoning-effort medium
```

### 3.2 OpenRouter 推荐启动方式

在 `cap-x/` 根目录：

```bash
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

如果使用 OpenRouter 模型名，建议在 CLI 中显式使用 `openrouter/...` 前缀，例如：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml \
  --model openrouter/google/gemini-2.5-pro-preview \
  --server-url http://127.0.0.1:8110/chat/completions \
  --total-trials 1 \
  --num-workers 1
```

token 要求：

- OpenRouter：`sk-or-v1-...`
- OpenAI 官方：`OPENAI_API_KEY` 或 `--api-key`
- HuggingFace：用于模型下载或 vLLM 本地模型，不直接等同于 OpenAI-compatible 推理 token
- 第三方兼容平台：需要 OpenAI-compatible `base_url` 和 bearer token

可考核结果：

```bash
curl http://127.0.0.1:8110/health
```

期望输出：

```json
{"status":"ok"}
```

### 3.3 使用第三方平台 token 的处理方式

如果第三方平台兼容 OpenAI API，有两种方式。

方案 A：直接让 CaP-X 请求第三方 endpoint：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml \
  --server-url https://your-provider.example.com/v1/chat/completions \
  --api-key your-provider-token \
  --model provider/model-name \
  --total-trials 1 \
  --num-workers 1
```

方案 B：复用 `openrouter_server.py` 的代理结构，把 `base_url` 改成第三方平台：

```bash
uv run --no-sync --active capx/serving/openrouter_server.py \
  --api-key your-provider-token \
  --base-url https://your-provider.example.com/v1/ \
  --port 8110

# NOTE: 启动 VLMs client
uv run --no-sync --active capx/serving/openrouter_server.py --port 8110

OPENROUTER_HTTP_PROXY=http://10.156.216.30:16371 \
uv run --no-sync --active capx/serving/openrouter_server.py --port 8110

```

然后 CaP-X 仍指向本地：

```bash
--server-url http://127.0.0.1:8110/chat/completions
```

验收标准：

- `query_model()` 打印 `Time taken to query model: ... seconds`。
- 评测输出目录中有 `initial_prompt.txt`。
- trial 目录中有 `raw_response.sh` 或 `all_responses.json`。
- 如果失败，`summary.txt` 中能看到 HTTP 错误或 response format 错误。

### 3.4 本地开源模型 vLLM

适用于 Qwen、DeepSeek、其他 HuggingFace checkpoint：

```bash
uv run python -m capx.serving.vllm_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8000
```

评测时：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml \
  --server-url http://127.0.0.1:8000/chat/completions \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --total-trials 1 \
  --num-workers 1
```

可考核结果：

- vLLM server 能返回 OpenAI-compatible `choices[0].message.content`。
- `capx/llm/client.py` 不抛出 `Unexpected response format`。

## 4. 阶段四：Robosuite 1-2 个任务复现与结果分析

### 4.1 先跑低成本 privileged/oracle 检查

目的：排除环境、IK、视频保存、基本执行链路问题，暂不依赖 SAM3 和 Contact-GraspNet。

推荐任务 1：Cube Lift privileged/oracle。

```bash
source .venv/bin/activate
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml \
  --use-oracle-code True \
  --total-trials 3 \
  --num-workers 1 \
  --record-video True
```

推荐任务 2：Cube Stack privileged/oracle。

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml \
  --use-oracle-code True \
  --total-trials 3 \
  --num-workers 1 \
  --record-video True
```

参考期望：

- `docs/development.md` 中 Cube Lift privileged 平均 reward 约 `0.99`。
- Cube Stack privileged 平均 reward 约 `0.90`。

可考核结果：

- 终端输出 `Summary Statistics`。
- `Code generation success rate / Average reward / Task completed` 合理。
- `outputs/<model-name>/<task>/summaries.txt` 存在。
- 每个 trial 目录包含 `code.py`、`summary.txt`、`all_responses.json`，启用视频时包含 `.mp4`。

### 4.2 再跑非 privileged 模型生成链路

目的：验证模型代理、视觉服务、抓取服务、IK 服务和 API tool 组合。

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml \
  --model gemini-2.5-pro \
  --server-url http://127.0.0.1:8110/chat/completions \
  --total-trials 3 \
  --num-workers 1 \
  --record-video True \
  --debug True
```

可考核结果：

- SAM3、Contact-GraspNet、PyRoKi 服务启动或复用成功。
- `code.py` 中能看到模型调用 `get_object_pose`、`sample_grasp_pose`、`goto_pose`、`open_gripper`、`close_gripper`，或 reduced API 中的 `get_observation`、`segment_sam3_text_prompt`、`plan_grasp`、`solve_ik`、`move_to_joints` 等。
- `summary.txt` 中 `Sandbox failed: 0` 表示代码执行无 Python 异常。
- `stderr` 为空或包含可解释问题。
- `reward` 与 `task_completed` 能反映任务进展。

### 4.3 多轮和视觉差分任务

推荐配置：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml \
  --model gemini-2.5-pro \
  --visual-differencing-model gemini-2.5-pro \
  --server-url http://127.0.0.1:8110/chat/completions \
  --total-trials 3 \
  --num-workers 1 \
  --record-video True
```

重点观察：

- `Num Regenerations`
- `Num Finishes`
- `Num Code Blocks`
- `prompts_and_responses/multi_turn_prompt_*.txt`
- `visual_feedback_*.png`
- `video_turn_*.mp4`
- `video_combined.mp4`

可考核结果：

- 多轮 trial 至少保存 `multi_turn_prompt_00.txt` 或视觉反馈图片。
- 若 `use_img_differencing: true`，`all_responses.json` 中应包含视觉差分相关响应。
- 如使用 video differencing，应保存 turn/combined video，且多轮 prompt 中包含视频分析反馈。

### 4.4 如何解读打印信息和输出文件

终端/summary 中的重要字段：

- `Sandbox failed`: `0` 表示模型代码执行成功，`1` 表示 Python 异常或 timeout。
- `Stdout`: 模型代码里的 `print()` 输出。
- `Stderr`: traceback、API 调用异常、服务错误。
- `Reward`: 当前任务 reward，具体计算由低层环境 `compute_reward()` 决定。
- `Task Completed`: 低层环境 `task_completed()` 的布尔判断。
- `Terminated`: 在 `CodeExecutionEnvBase.step()` 中 reward 等于 `1.0` 时为 true。
- `Truncated`: 达到 `max_steps`。
- `Average code blocks`: 多轮任务的平均代码块数。
- `Average regenerations`: 模型根据反馈重写代码的次数。

trial 目录结构示例：

```text
outputs/<task>/<model>/trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
  code.py
  raw_response.sh
  all_responses.json
  summary.txt
  visual_feedback_00.png
  prompts_and_responses/
  video_combined.mp4
  video_turn_00.mp4
```

验收表建议：

| 检查项 | 通过标准 |
| --- | --- |
| 环境 reset | 无异常，能打印 observation keys |
| API servers | 端口 ready 或复用成功 |
| 模型返回 | `all_responses.json` 中 content 非空 |
| 代码提取 | `code.py` 是可执行 Python，不含 Markdown fence |
| 执行 | `Sandbox failed: 0` |
| 任务 | reward/task_completed 与视频一致 |
| 可视化 | 图片/视频能打开，动作过程可解释 |

### 4.5 Web UI 交互式观察

启动：


```bash
# NOTE: Web UI 启动，似乎不会自动启动 VLMs client，需要先手动启动
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
  --model gemini-2.5-pro \
  --web-ui True
```

打开：

```text
http://localhost:8200
```

重点观察：

- 左侧聊天/模型响应
- 代码块
- API execution detail
- 图片预览
- Viser/可视化面板
- 每个 func tool 的 `_log_step()` 输出

可考核结果：

- Web UI 能加载 YAML config。
- trial 可启动、停止。
- API 调用步骤能在 UI 中按顺序出现。
- 对新增 API/tool，能显示 tool name、文字说明和可选图片。

## 5. 阶段五：现有 func tool/API 能力分析

### 5.1 当前 API 注入机制

`ApiBase.functions()` 返回暴露给模型生成代码的函数。

`ApiBase.combined_doc()` 会提取：

- 函数名
- Python signature
- docstring

然后 `CodeExecutionEnvBase._get_complete_prompt()` 将这些文档拼进 prompt 的 `APIs:` 部分。

执行时，`CodeExecutionEnvBase._init_exec_globals()` 和 `_exec_user_code()` 会把所有 API 函数绑定进 exec namespace，因此模型代码可以直接调用函数名。

可考核验证：

```bash
python -c "import capx.envs.simulators, capx.envs.tasks, capx.integrations; from capx.integrations import list_apis; print(list_apis())"
```

以及在任意 env reset 后检查：

```python
obs, info = env.reset()
print(obs["full_prompt"][-1]["content"][0]["text"])
```

期望能看到 API 函数签名和 docstring。

### 5.2 Robosuite 常用 API 能力

高层 API：

- `FrankaControlApi`
- `FrankaControlPrivilegedApi`
- `FrankaControlApiReduced`
- `FrankaControlApiReducedSkillLibrary`
- `FrankaControlSpillWipeApi`
- `FrankaControlNutAssemblyVisualApi`
- `FrankaTwoArmLiftApi`
- `FrankaHandoverApi`

常见能力：

- 获取观测：`get_observation`
- 对象检测/分割：SAM3、SAM2、OWL-ViT、Molmo point prompt
- 点云与 OBB：`get_oriented_bounding_box_from_3d_points`
- 抓取规划：`plan_grasp` 或 `sample_grasp_pose`
- IK：`solve_ik`
- 关节运动：`move_to_joints`
- 末端位姿运动：`goto_pose`
- 夹爪：`open_gripper`、`close_gripper`

### 5.3 判断能否完成任务的方法

对每个待适配任务建立能力矩阵：

| 任务需求 | 现有 tool 是否覆盖 | 对应 API | 风险 |
| --- | --- | --- | --- |
| 获取 RGB-D 和相机参数 | 是 | `get_observation` | 新环境 observation key 必须兼容 |
| 自然语言找物体 | 是 | SAM3/OWL-ViT/Molmo | 需要模型权重和服务 |
| 3D 点云定位 | 是 | depth utils + OBB | 新环境 depth/intrinsics/pose_mat 要准确 |
| 抓取 | 是 | Contact-GraspNet | 新环境坐标系需统一到 robot base |
| IK | 是 | PyRoKi | 新机器人 URDF/target_link 需配置 |
| 基础夹爪 | 是 | open/close gripper | 新 simulator action 需实现 |
| 特定 task reward | 否，需要新增 | env.compute_reward | 每个环境单独实现 |
| 特定 task success | 否，需要新增 | env.task_completed | 每个环境单独实现 |

可考核结果：

- 对 1 个 Robosuite 任务输出 API 函数清单。
- 手写 oracle code 只用现有 API 完成任务。
- 对失败 case 能指出是 perception、grasp、IK、execution、reward 哪一层失败。

### 5.4 API 交互接口验证

建议分层验证：

1. Registry：
   - `list_apis()` 包含目标 API 名称。
   - YAML `apis` 字段名称能被 `get_api()` 找到。
2. Prompt：
   - `full_prompt` 中包含函数签名和 docstring。
3. Execution namespace：
   - 模型/手写代码能直接调用函数，不需要 `APIS["name"]`。
4. Low-level env：
   - API 内部调用 `self._env.get_observation()`、`move_to_joints_blocking()`、`render()` 正常。
5. Web UI：
   - API 调用 `_log_step()` 后 UI 能显示步骤和图片。

推荐手写测试代码：

```python
obs = get_observation()
print(obs.keys())
print(obs["robot0_robotview"]["images"]["rgb"].shape)
print(obs["robot0_robotview"]["images"]["depth"].shape)
```

可考核结果：

- `summary.txt` 中 stdout 能看到 shape。
- 无 traceback。
- 若启用 Web UI，`get_observation` step 可见。

## 6. 阶段六：新增 func tool 并测试

### 6.1 选择第一个新增 tool

建议先实现一个低风险、可观测、跨任务有用的工具：

```text
describe_robot_state()
```

用途：

- 返回当前 robot joint、cartesian pose、gripper state、sim step。
- 用于模型 debug 和多轮纠错。
- 不改变环境状态，测试风险低。

### 6.2 实现位置

如果只服务 Franka Robosuite reduced API，可在：

```text
capx/integrations/franka/control_reduced.py
```

如果希望成为通用 API，建议新增：

```text
capx/integrations/debug/state.py
```

示例结构：

```python
from typing import Any
from capx.integrations.base_api import ApiBase

class RobotStateDebugApi(ApiBase):
    def functions(self) -> dict[str, Any]:
        return {"describe_robot_state": self.describe_robot_state}

    def describe_robot_state(self) -> dict[str, Any]:
        """Return the current robot state for debugging.

        Returns:
            state:
                A dictionary with robot_joint_pos, robot_cartesian_pos, and sim_step_count
                when available. Missing fields are omitted.
        """
        obs = self._env.get_observation()
        state = {}
        if "robot_joint_pos" in obs:
            state["robot_joint_pos"] = obs["robot_joint_pos"].tolist()
        if "robot_cartesian_pos" in obs:
            state["robot_cartesian_pos"] = obs["robot_cartesian_pos"].tolist()
        if hasattr(self._env, "_sim_step_count"):
            state["sim_step_count"] = int(self._env._sim_step_count)
        self._log_step("describe_robot_state", str(state))
        return state
```

注册：

```python
from .debug.state import RobotStateDebugApi
register_api("RobotStateDebugApi", RobotStateDebugApi)
```

YAML 中加入：

```yaml
apis:
  - FrankaControlApiReduced
  - RobotStateDebugApi
```

### 6.3 单元验证

新增或扩展测试：

```text
tests/test_debug_api.py
```

验证点：

- `RobotStateDebugApi` 在 `list_apis()` 中。
- `combined_doc()` 包含 `describe_robot_state`。
- 构造 env 后执行 `describe_robot_state()` 返回 dict。
- 使用 `env.step("state = describe_robot_state(); print(state)")` 时 `sandbox_rc == 0`。

可考核结果：

```bash
uv run pytest tests/test_debug_api.py -q
```

期望：

- 测试全部通过。
- stdout 中能看到 robot state 字段。

### 6.4 小规模任务验证

复制一个 debug YAML，例如：

```text
env_configs/debug/franka_robosuite_cube_lifting_debug_api.yaml
```

将 API 列表改为：

```yaml
apis:
  - FrankaControlApiReduced
  - RobotStateDebugApi
```

运行：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/debug/franka_robosuite_cube_lifting_debug_api.yaml \
  --use-oracle-code True \
  --total-trials 1 \
  --num-workers 1 \
  --record-video True
```

可考核结果：

- `initial_prompt.txt` 或 per-trial prompt 中包含 `describe_robot_state()`。
- `summary.txt` 中 stdout 有 state 输出。
- Web UI 中能看到 `describe_robot_state` step。

## 7. 阶段七：RLBench 环境适配规划

### 7.1 适配目标

先做最小可运行任务，不直接追求完整 benchmark：

- 推荐 smoke task：`ReachTarget` 或 `TakeLidOffSaucepan`
- 目标机器人：优先使用 RLBench 默认 Panda
- 控制粒度：先实现 joint/pose/gripper 的低层接口，再接现有 Franka reduced API

### 7.2 需要新增的代码

建议新增：

```text
capx/envs/simulators/rlbench.py
capx/envs/tasks/franka/franka_rlbench_env.py
env_configs/rlbench/franka_rlbench_reach_target.yaml
```

可能需要新增或复用：

```text
capx/integrations/franka/rlbench_reduced.py
```

如果 RLBench 的 observation 可以转换成 CaP-X 标准字段，则优先复用 `FrankaControlApiReduced`。

### 7.3 低层环境 wrapper 设计

必须继承 `BaseEnv`，实现：

- `reset(seed, options)`
- `step(action)`
- `get_observation()`
- `compute_reward()`
- `task_completed()`

最关键是将 RLBench observation 转为 CaP-X API 所需格式：

```python
obs["robot0_robotview"]["images"]["rgb"]      # (H, W, 3), uint8
obs["robot0_robotview"]["images"]["depth"]    # (H, W, 1), float32, meters
obs["robot0_robotview"]["intrinsics"]         # (3, 3)
obs["robot0_robotview"]["pose_mat"]           # (4, 4), camera-to-robot-base
obs["robot_joint_pos"]                        # joint + gripper
obs["robot_cartesian_pos"]                    # xyz + wxyz + gripper
```

若 RLBench 提供多视角，先映射一个主视角为 `robot0_robotview`，后续再扩展 wrist/front/overhead。

可考核结果：

```python
env = RLBenchLowLevel(...)
obs, info = env.reset()
assert obs["robot0_robotview"]["images"]["rgb"].dtype == np.uint8
assert obs["robot0_robotview"]["images"]["depth"].ndim in [2, 3]
assert obs["robot0_robotview"]["intrinsics"].shape == (3, 3)
assert obs["robot0_robotview"]["pose_mat"].shape == (4, 4)
```

### 7.4 注册环境

在 `capx/envs/simulators/__init__.py` 中加入：

```python
try:
    from .rlbench import RLBenchLowLevel
    register_env("franka_rlbench_reach_target_low_level", RLBenchLowLevel)
except Exception:
    print("RLBench not installed, skipping RLBench environments")
```

可考核结果：

```bash
python -c "import capx.envs.simulators; from capx.envs.base import list_envs; print([x for x in list_envs() if 'rlbench' in x])"
```

### 7.5 高层 task wrapper

新增 `FrankaRLBenchCodeEnv(CodeExecutionEnvBase)`，定义：

- prompt：清楚说明任务目标、坐标系、单位、允许 API。
- oracle_code：先写一个最简单的可成功或可产生合理动作的参考代码。

注册在 `capx/envs/tasks/__init__.py`，或直接在 YAML 用 `_target_` 实例化。

YAML 示例：

```yaml
env:
  _target_: capx.envs.tasks.franka.franka_rlbench_env.FrankaRLBenchCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: franka_rlbench_reach_target_low_level
    privileged: false
    apis:
      - FrankaControlApiReduced

record_video: true
output_dir: ./outputs/franka_rlbench_reach_target
trials: 10
num_workers: 1
```

### 7.6 RLBench 验收顺序

1. import 和 reset：
   - 不启动 LLM。
   - 手动 reset，检查 obs keys。
2. render：
   - `env.render()` 返回 RGB。
   - `enable_video_capture()` 后有 frame。
3. oracle：
   - `--use-oracle-code True --total-trials 3`。
   - 至少不崩溃，能产出视频。
4. tool：
   - 运行 `get_observation()`、`describe_robot_state()`。
   - 再尝试 perception tool。
5. LLM：
   - 单 trial，`num_workers=1`，`debug=True`。

阶段验收：

- smoke task 输出 `summaries.txt`。
- 至少一个 trial `sandbox_rc=0`。
- observation、视频、reward/task_completed 与实际画面一致。

## 8. 阶段八：ManiSkill 环境适配规划

### 8.1 适配目标

先选简单、标准、Panda 可控任务：

- `PickCube-v1`
- `StackCube-v1`
- `PegInsertionSide-v1`

推荐优先级：

1. `PickCube-v1`：最适合验证 RGB-D、pose、gripper、reward。
2. `StackCube-v1`：与现有 cube stack 任务语义接近。
3. `PegInsertionSide-v1`：对齐 nut assembly/peg assembly 类任务。

### 8.2 新增代码路径

```text
capx/envs/simulators/maniskill.py
capx/envs/tasks/franka/franka_maniskill_env.py
env_configs/maniskill/franka_maniskill_pick_cube.yaml
env_configs/maniskill/franka_maniskill_stack_cube.yaml
```

如 ManiSkill action space 与现有 Franka API 不兼容，新增：

```text
capx/integrations/franka/maniskill_reduced.py
```

### 8.3 Observation 标准化

ManiSkill 的相机、深度和坐标系需要统一成 CaP-X 标准：

- RGB 转 `uint8`
- depth 单位确认是 meters
- camera extrinsic 转 `pose_mat`
- object pose 转 robot base frame
- robot qpos 和 TCP pose 转 `robot_joint_pos`、`robot_cartesian_pos`

建议新增一个内部转换函数：

```python
def _convert_maniskill_obs(self, raw_obs) -> dict[str, Any]:
    ...
```

可考核结果：

- shape/dtype 与 Robosuite wrapper 一致。
- 直接复用 `FrankaControlApiReduced.get_observation()` 时不报 key error。

### 8.4 控制接口选择

最小版本：

- `move_to_joints_blocking(joints)`
- `_set_gripper(fraction)`
- `_step_once()`
- `render()`
- `enable_video_capture()`
- `get_video_frames()`

这样可以尽量复用 Franka API 中的 IK 和关节执行逻辑。

如果 ManiSkill 更适合 ee pose action，则新增 ManiSkill 专用 API：

- `move_ee_delta_pose(delta_xyz, delta_rot)`
- `move_ee_pose(position, quaternion_wxyz)`
- `open_gripper()`
- `close_gripper()`

### 8.5 ManiSkill 验收顺序

1. `python -c "import mani_skill"` 成功。
2. `reset()` 返回 CaP-X 标准 observation。
3. `render()` 能保存图片。
4. 手写动作脚本能移动机械臂和夹爪。
5. `--use-oracle-code True --total-trials 3` 不崩溃。
6. 小规模 LLM 运行保存 `code.py`、`summary.txt`、视频。

阶段验收：

- `PickCube-v1` 至少完成一个成功 episode，或 reward 明显高于随机。
- `StackCube-v1` 至少完成 reset、观察、抓取动作、视频输出。

## 9. 阶段九：测试体系与结果汇总

### 9.1 测试层级

| 层级 | 命令/方法 | 通过标准 |
| --- | --- | --- |
| import | `python -c "import capx"` | 无 import error |
| registry | `list_envs/list_apis/list_exec_envs` | 新增项可见 |
| unit | `uv run pytest tests/... -q` | 全部通过 |
| oracle | `--use-oracle-code True` | reward/task_completed 合理 |
| LLM smoke | `--total-trials 1 --num-workers 1` | 有 code、summary、无接口异常 |
| batch | `--total-trials 10` | summaries 指标稳定 |
| visual | Web UI/视频/图片 | 执行过程可解释 |

### 9.2 推荐复现实验表

| 实验 | 配置 | trials | 目标 |
| --- | --- | --- | --- |
| R0 | cube lifting privileged oracle | 3 | 环境和视频链路 |
| R1 | cube stack privileged oracle | 3 | IK 和 stacking reward |
| R2 | cube lifting non-privileged LLM | 3 | SAM3/GraspNet/PyRoKi/API |
| R3 | cube stack multiturn vdm | 3 | 多轮和视觉差分 |
| A0 | 新增 debug API oracle | 1 | tool 注册和 prompt 注入 |
| RL0 | RLBench smoke | 3 | 新环境 wrapper |
| M0 | ManiSkill PickCube smoke | 3 | 新 simulator 适配 |

### 9.3 结果记录模板

每次实验记录：

```markdown
## Experiment: R2 cube lifting non-privileged

- Date:
- Git commit:
- Dirty:
- Config:
- Model:
- Server URL:
- API servers:
- Trials:
- Success rate:
- Average reward:
- Task completed:
- Average code blocks:
- Average regenerations:
- Key failure modes:
- Representative trial dirs:
- Video observations:
- Next action:
```

CaP-X 会在 `summaries.txt` 中保存：

- model
- config path
- git commit/dirty
- total trials
- success rate / average reward / task completed
- average code blocks
- average regenerations
- elapsed time

## 10. 风险清单与排错路径

### 10.1 安装风险

- Robosuite 与 LIBERO 依赖冲突：使用不同 venv。
- SAM3 权限不足：检查 HuggingFace token。
- GPU 显存不足：使用 privileged/minimal profile；减少 workers。
- EGL/MuJoCo 渲染失败：安装 `libegl1 libgl1`，设置 `MUJOCO_GL=egl`。

### 10.2 模型 API 风险

- third-party endpoint 不兼容 `choices[0].message.content`：需要写自定义 proxy 转换格式。
- OpenRouter 模型名：本地代理会去掉 `openrouter/` 前缀。
- token 泄漏：`.openrouterkey` 不提交；不要把 token 写入 YAML 或日志。

### 10.3 执行风险

- `sandbox_rc=1`：优先看 `stderr` traceback。
- `task_completed=False` 但 reward 高：检查 success 判断。
- reward 为 0 但视频接近成功：检查 reward shaping 和完成阈值。
- 多轮没有 regenerate：检查 `multi_turn_prompt` 和模型是否按 `REGENERATE/FINISH` 输出。

### 10.4 新环境适配风险

- observation key 不兼容：先用 shape/dtype test 固定合同。
- camera pose 坐标系错误：点云/OBB/抓取会整体偏移。
- IK robot description 不匹配：PyRoKi target link、URDF、TCP offset 需校准。
- action scale 不匹配：先实现 joint blocking 控制，再做高层 pose。

## 11. 推荐执行顺序

1. 完成 `.venv` + `uv sync --extra robosuite`。
2. 跑 registry/import 测试。
3. 跑 Cube Lift privileged oracle。
4. 跑 Cube Stack privileged oracle。
5. 启动 OpenRouter 或第三方 OpenAI-compatible proxy。
6. 跑 Cube Lift non-privileged LLM。
7. 跑 Cube Stack multiturn VDM。
8. 用 Web UI 观察 1 个成功和 1 个失败 trial。
9. 输出现有 API/tool 能力矩阵。
10. 新增 `RobotStateDebugApi` 并完成测试。
11. 设计并实现 RLBench smoke wrapper。
12. 设计并实现 ManiSkill PickCube wrapper。
13. 汇总所有实验到统一结果表。

## 12. 阶段完成定义

本计划完成时应具备以下 artifacts：

- Robosuite 复现实验目录，含 `summaries.txt`、trial `summary.txt`、`code.py`、视频。
- 第三方服务启动记录和 token 配置说明。
- 1 个模型代理 smoke test 成功记录。
- 现有 func tool 能力矩阵。
- 新增 func tool 的代码、注册、测试和 1 个 trial 输出。
- RLBench 适配设计和至少 smoke wrapper 初版。
- ManiSkill 适配设计和至少 PickCube smoke wrapper 初版。
- 一份实验结果汇总表，能支持后续论文/报告中的复现实验描述。
