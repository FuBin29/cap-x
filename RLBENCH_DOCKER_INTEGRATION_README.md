# CaP-X 接入 RLBench Docker 环境工程规划

本文档面向当前 `cap-x` 项目扩展 RLBench 支持的工程落地。核心约束是：CaP-X 主环境已经通过 `uv` 在本机 Ubuntu 24.04 配好，并且 Robosuite 能运行；RLBench 依赖 Ubuntu 20.04、CoppeliaSim、PyRep 等更老且冲突较多的运行栈，因此不建议把 RLBench 直接安装进 CaP-X 当前 venv。推荐方案是将 RLBench 运行在 Docker 内，CaP-X 通过远程 low-level environment proxy 与其交互。

## 0. 总体结论

### 0.1 推荐架构

```text
Host / Ubuntu 24.04 / cap-x uv environment
  capx/envs/launch.py
  capx/envs/tasks/base.py::CodeExecutionEnvBase
  capx/envs/simulators/rlbench_remote.py::RLBenchRemoteEnv
  capx/integrations/franka/rlbench.py::FrankaRLBenchApi
        |
        | HTTP / ZeroMQ / gRPC
        v
Docker / Ubuntu 20.04 / RLBench environment
  rlbench_server.py
  RLBench
  PyRep
  CoppeliaSim
```

### 0.2 设计原则

1. CaP-X 主进程不 import `rlbench`、`pyrep`、`sim` 等 Docker 内依赖。
2. RLBench Docker 对外暴露一个稳定的 simulator RPC 服务。
3. CaP-X 中新增一个实现 `BaseEnv` 的远程代理环境。
4. CaP-X 中新增一个面向 RLBench 的控制 API，不强行复用 Robosuite 的私有控制接口。
5. 先完成 privileged / state-based smoke task，再扩展视觉 observation、perception API 和多任务。

### 0.3 需要解决的核心问题

| 问题 | 所属项目 | 解决方式 | 目标 |
| --- | --- | --- | --- |
| Ubuntu 24.04 与 RLBench 依赖不兼容 | RLBench Docker | 使用 Ubuntu 20.04 镜像隔离 RLBench | RLBench 可独立运行 |
| CaP-X 如何调用 Docker 内 simulator | CaP-X + RLBench Docker | 通过 RPC 接口通信 | 主环境无需安装 RLBench |
| observation 格式不统一 | CaP-X + RLBench Docker | 在 server/proxy 层转换为 CaP-X 约定结构 | 可复用视觉与控制 API |
| 控制接口不同 | CaP-X | 新增 `FrankaRLBenchApi` | LLM 能用统一工具完成任务 |
| 评测链路如何复用 | CaP-X | 注册 task env、low-level env、YAML config | 可通过 `capx/envs/launch.py` 运行 |

## 1. 阶段一：RLBench Docker 独立可运行

### 1.1 固定 Docker 启动方式

操作项目：RLBench Docker / 本机运行环境。

要做什么：

- 确认当前镜像 `rlbench-fubin:v1.2` 能稳定启动。
- 固定容器名、workspace 挂载路径、显示与 GPU 参数。
- 修正容器名不一致问题。

建议命令：

```bash
docker run -it \
  --name rlbench_test5 \
  --net=host \
  --gpus all \
  -e DISPLAY=:99 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v ~/.Xauthority:/root/.Xauthority \
  -v ~/projects/artance:/workspace \
  rlbench-fubin:v1.2 \
  /bin/bash
```

进入容器：

```bash
docker exec -it rlbench_test5 /bin/bash
```

目标：

- 确保 RLBench、PyRep、CoppeliaSim、GPU/渲染链路在容器内正常。
- 后续 CaP-X 只依赖容器提供的服务端口。

可考核结果：

```bash
python examples/few_shot_rl.py
```

期望：

- 脚本能启动 RLBench task。
- 无 CoppeliaSim 路径错误。
- 无 OpenGL / display / X11 / EGL 相关错误。
- 容器内能看到 `/workspace/cap-x` 或对应项目路径。

### 1.2 选择第一个 smoke task

操作项目：RLBench Docker。

要做什么：

- 选择一个最简单、最少依赖感知和复杂抓取的任务。
- 推荐优先级：
  - `ReachTarget`
  - `PickUpCup`
  - `CloseDrawer`

目标：

- 先验证 reset、observation、action、success 判断，不急于验证完整 CaP 能力。

可考核结果：

- 容器内 Python 脚本可以：
  - 创建 RLBench environment。
  - 加载指定 task。
  - `task.reset()` 成功。
  - 打印 observation 中的 joint、gripper、camera 字段。
  - 执行若干 random/no-op action 不崩溃。
  - 调用 task success 或 reward 判断接口。

## 2. 阶段二：设计 RLBench RPC 服务

### 2.1 新增 RLBench server

操作项目：RLBench Docker 内的代码。建议初期可以放在挂载目录：

```text
/workspace/cap-x/scripts/rlbench_server.py
```

后续稳定后可移动到独立 RLBench adapter 仓库或 `cap-x/scripts/`。

要做什么：

- 在容器内启动一个 HTTP 服务。
- 服务内部负责 import RLBench、初始化 CoppeliaSim、创建 task。
- 对 CaP-X 暴露最小接口。

推荐技术选型：

- 首选：FastAPI + Uvicorn，便于调试和 JSON schema。
- 次选：Flask，依赖更少。
- 高性能后续方案：ZeroMQ 或 gRPC。

目标：

- 把 RLBench 进程生命周期固定在 Docker 内。
- CaP-X 通过 `127.0.0.1:<port>` 调用。

建议最小接口：

```text
GET  /health
POST /reset
POST /step
GET  /observation
GET  /reward
GET  /success
POST /move_to_joints
POST /move_to_pose
POST /open_gripper
POST /close_gripper
POST /shutdown
```

可考核结果：

```bash
curl http://127.0.0.1:8120/health
```

期望输出：

```json
{"status": "ok"}
```

### 2.2 固定序列化协议

操作项目：RLBench Docker + CaP-X。

要做什么：

- 明确 observation、action、image、depth、pose 的传输格式。
- 先用 JSON + base64 或 JSON + list，后续再优化为 msgpack / shared memory。

推荐初期格式：

```json
{
  "rgb": {
    "encoding": "base64",
    "dtype": "uint8",
    "shape": [512, 512, 3],
    "data": "..."
  },
  "depth": {
    "encoding": "base64",
    "dtype": "float32",
    "shape": [512, 512],
    "data": "..."
  },
  "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "gripper_open": 1.0,
  "task_descriptions": ["reach the target"],
  "object_poses": {
    "target": {
      "position": [0.1, 0.2, 0.3],
      "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
    }
  }
}
```

目标：

- 避免 Python object pickle 跨环境传输。
- 保证 numpy array 能在 CaP-X 侧稳定还原。

可考核结果：

- `/observation` 返回结果能被本机 Python 客户端解析。
- RGB 还原为 `(H, W, 3) uint8`。
- depth 还原为 `(H, W)` 或 `(H, W, 1) float32`。
- joint positions 还原为 `(7,) float64`。

### 2.3 明确坐标系和四元数约定

操作项目：RLBench Docker + CaP-X。

要做什么：

- 明确 RLBench 返回的是 world frame 还是 robot base frame。
- 明确四元数顺序是 `xyzw` 还是 `wxyz`。
- 在 RPC server 或 CaP-X proxy 中统一转换。

CaP-X 当前常用约定：

```text
position: XYZ, meters
quaternion_wxyz: [w, x, y, z]
pose_mat: 4x4 homogeneous matrix
```

目标：

- LLM prompt、API doc、oracle code 使用同一套坐标约定。
- 避免 `goto_pose` 姿态错乱。

可考核结果：

- 写一个固定 pose round-trip 测试：
  - Docker 端返回 object pose。
  - CaP-X 侧解析为 `position + quaternion_wxyz`。
  - 再转换成 `pose_mat`。
  - 数值维度与单位正确，无 quaternion 顺序反转。

## 3. 阶段三：CaP-X 新增远程 low-level env

### 3.1 新增 `RLBenchRemoteEnv`

操作项目：CaP-X 仓库。

建议新增文件：

```text
capx/envs/simulators/rlbench_remote.py
```

要做什么：

- 实现 `capx.envs.base.BaseEnv` 的五个必需方法：
  - `reset`
  - `step`
  - `get_observation`
  - `compute_reward`
  - `task_completed`
- 内部使用 HTTP client 调 Docker server。
- 提供 `move_to_joints_blocking`、`move_to_pose`、`_set_gripper`、`_step_once` 等必要兼容方法时，要按 RLBench 语义实现，不依赖 Robosuite 私有字段。

目标：

- 让 CaP-X 认为 RLBench 和 Robosuite 一样是一个 low-level env。
- 隔离 RLBench 依赖，主 venv 不需要安装 RLBench。

可考核结果：

```python
from capx.envs.simulators.rlbench_remote import RLBenchRemoteEnv

env = RLBenchRemoteEnv(server_url="http://127.0.0.1:8120")
obs, info = env.reset()
print(obs.keys())
print(env.compute_reward())
print(env.task_completed())
```

期望：

- 不 import `rlbench`。
- 能连接 Docker server。
- `reset()` 返回 `(obs, info)`。
- `get_observation()` 返回标准 dict。

### 3.2 对齐 CaP-X observation 结构

操作项目：CaP-X 仓库，必要时同步调整 RLBench server。

要做什么：

- 在 `RLBenchRemoteEnv.get_observation()` 中转换为 CaP-X 约定结构。

推荐结构：

```python
obs = {
    "robot0_robotview": {
        "images": {
            "rgb": rgb_uint8,
            "depth": depth_float32,
        },
        "intrinsics": intrinsics_3x3,
        "pose_mat": camera_pose_4x4,
    },
    "robot0_joint_pos": joint_positions,
    "robot0_gripper_qpos": gripper_state,
    "object_poses": {
        "target": np.concatenate([position_xyz, quaternion_wxyz]),
    },
}
```

目标：

- 后续视觉 API 可以复用 `robot0_robotview`。
- privileged API 可以直接读取 `object_poses`。

可考核结果：

- `obs["robot0_robotview"]["images"]["rgb"].dtype == np.uint8`
- `obs["robot0_robotview"]["images"]["depth"].dtype == np.float32`
- `obs["robot0_robotview"]["intrinsics"].shape == (3, 3)`
- `obs["robot0_robotview"]["pose_mat"].shape == (4, 4)`
- `obs["object_poses"]` 中包含 smoke task 目标对象。

### 3.3 注册 low-level env

操作项目：CaP-X 仓库。

修改文件：

```text
capx/envs/simulators/__init__.py
```

要做什么：

- 用 `try/except` 注册远程 RLBench env。
- 注意远程 env 不应该因为 RLBench 未安装而 import 失败。

建议注册名：

```python
register_env("franka_rlbench_remote_low_level", RLBenchRemoteEnv)
```

目标：

- `get_env("franka_rlbench_remote_low_level")` 可用。

可考核结果：

```bash
uv run --no-sync --active python -c "import capx.envs.simulators; from capx.envs.base import list_envs; print('franka_rlbench_remote_low_level' in list_envs())"
```

期望输出：

```text
True
```

## 4. 阶段四：CaP-X 新增 RLBench 控制 API

### 4.1 新增 `FrankaRLBenchApi`

操作项目：CaP-X 仓库。

建议新增文件：

```text
capx/integrations/franka/rlbench.py
```

要做什么：

- 继承 `ApiBase`。
- 暴露 LLM 需要的最小工具函数。

建议第一版函数：

```python
get_observation() -> dict
get_object_pose(object_name: str) -> tuple[np.ndarray, np.ndarray]
goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
move_to_joints(joints: np.ndarray) -> None
open_gripper() -> None
close_gripper() -> None
```

目标：

- 不强行复用 `FrankaControlPrivilegedApi`，避免依赖 Robosuite 的 `_step_once`、`_set_gripper`、`move_to_joints_blocking` 细节。
- 让 prompt 中展示的是 RLBench 可稳定执行的动作。

可考核结果：

- `FrankaRLBenchApi.combined_doc()` 能生成完整函数文档。
- LLM 生成代码能直接调用 `get_object_pose`、`goto_pose`、`open_gripper`、`close_gripper`。
- oracle code 能通过该 API 控制 smoke task。

### 4.2 注册 API

操作项目：CaP-X 仓库。

修改文件：

```text
capx/integrations/__init__.py
```

要做什么：

- import `FrankaRLBenchApi`。
- 注册：

```python
register_api("FrankaRLBenchApi", FrankaRLBenchApi)
```

目标：

- `CodeExecEnvConfig(apis=["FrankaRLBenchApi"])` 可用。

可考核结果：

```bash
uv run --no-sync --active python -c "import capx.integrations; from capx.integrations import list_apis; print('FrankaRLBenchApi' in list_apis())"
```

期望输出：

```text
True
```

## 5. 阶段五：新增 CaP-X task env 与 YAML 配置

### 5.1 新增第一个 RLBench task wrapper

操作项目：CaP-X 仓库。

建议新增文件：

```text
capx/envs/tasks/franka/franka_rlbench_reach_target.py
```

要做什么：

- 继承 `CodeExecutionEnvBase`。
- 写清楚 prompt。
- 写一个最小 oracle code。

示例 prompt 重点：

```text
You are controlling a Franka robot in an RLBench environment.
Use the APIs described below.
All positions are XYZ in meters.
All quaternions are WXYZ.
Write ONLY executable Python code. No Markdown fences.
```

目标：

- 让 RLBench task 进入 CaP-X 标准 code-generation 评测链路。

可考核结果：

- 能通过 oracle code 完成一次 smoke task。
- trial 输出目录中包含 `code.py`、`summary.txt`、`all_responses.json`。

### 5.2 注册 exec env 和 config

操作项目：CaP-X 仓库。

修改文件：

```text
capx/envs/tasks/__init__.py
```

要做什么：

- 注册：

```python
register_exec_env("franka_rlbench_reach_target_code_env", FrankaRLBenchReachTargetCodeEnv)
register_config(
    "franka_rlbench_reach_target_code_env",
    CodeExecEnvConfig(
        low_level="franka_rlbench_remote_low_level",
        apis=["FrankaRLBenchApi"],
        privileged=True,
    ),
)
```

目标：

- CaP-X 能通过统一 registry 找到 RLBench task。

可考核结果：

```bash
uv run --no-sync --active python -c "import capx.envs.tasks; from capx.envs.tasks import list_exec_envs; print('franka_rlbench_reach_target_code_env' in list_exec_envs())"
```

期望输出：

```text
True
```

### 5.3 新增 YAML 配置

操作项目：CaP-X 仓库。

建议新增目录与文件：

```text
env_configs/rlbench/franka_rlbench_reach_target.yaml
```

要做什么：

- 配置 high-level env。
- 配置 low-level remote server URL。
- 先关闭多 worker。
- 先关闭复杂视觉服务。

建议第一版配置方向：

```yaml
env:
  _target_: capx.envs.tasks.franka.franka_rlbench_reach_target.FrankaRLBenchReachTargetCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level:
      _target_: capx.envs.simulators.rlbench_remote.RLBenchRemoteEnv
      server_url: http://127.0.0.1:8120
      task_name: reach_target
      max_steps: 300
    privileged: true
    apis:
      - FrankaRLBenchApi

record_video: true
output_dir: ./outputs/rlbench_reach_target
trials: 1
num_workers: 1
```

目标：

- 避免一开始依赖 registry 参数传递不足的问题。
- 让 server URL、task name、max steps 都能从 YAML 调整。

可考核结果：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_reach_target.yaml \
  --use-oracle-code True \
  --total-trials 1 \
  --num-workers 1
```

期望：

- CaP-X 成功连接 Docker server。
- oracle code 执行无 Python 异常。
- 输出 summary 中 `Sandbox failed: 0`。

## 6. 阶段六：联调与验收

### 6.1 Docker server smoke test

操作项目：RLBench Docker。

要做什么：

- 在容器内启动 RLBench server：

```bash
python /workspace/cap-x/scripts/rlbench_server.py \
  --host 0.0.0.0 \
  --port 8120 \
  --task reach_target
```

目标：

- 服务常驻，等待 CaP-X 请求。

可考核结果：

```bash
curl http://127.0.0.1:8120/health
curl -X POST http://127.0.0.1:8120/reset
curl http://127.0.0.1:8120/success
```

期望：

- `/health` 返回 ok。
- `/reset` 返回 observation summary 或 episode id。
- `/success` 返回 boolean。

### 6.2 CaP-X proxy smoke test

操作项目：CaP-X 仓库。

要做什么：

- 不启动完整 LLM 评测，只验证 `RLBenchRemoteEnv`。

目标：

- 快速定位问题在 RPC、序列化还是 CaP-X env 层。

可考核结果：

```bash
uv run --no-sync --active python - <<'PY'
from capx.envs.simulators.rlbench_remote import RLBenchRemoteEnv

env = RLBenchRemoteEnv(server_url="http://127.0.0.1:8120", task_name="reach_target")
obs, info = env.reset()
print(obs.keys())
print(obs["robot0_robotview"]["images"]["rgb"].shape)
print(float(env.compute_reward()))
print(bool(env.task_completed()))
PY
```

期望：

- 无 import RLBench。
- RGB/depth shape 正确。
- reward/success 可返回。

### 6.3 Oracle 评测

操作项目：CaP-X 仓库 + RLBench Docker。

要做什么：

- 使用 CaP-X 标准 launcher 跑 oracle。

目标：

- 验证完整链路：

```text
YAML -> launch.py -> CodeExecutionEnvBase -> RLBenchRemoteEnv -> Docker server -> RLBench
```

可考核结果：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_reach_target.yaml \
  --use-oracle-code True \
  --total-trials 3 \
  --num-workers 1 \
  --record-video True
```

期望：

- 每个 trial 有 `code.py`、`summary.txt`。
- `Sandbox failed: 0`。
- `Task Completed` 至少在 smoke oracle 中为 true，或 reward 明显变化。
- 如果保存视频，视频中动作与任务一致。

### 6.4 LLM 生成代码评测

操作项目：CaP-X 仓库 + RLBench Docker。

要做什么：

- 接入当前 OpenAI-compatible model server。
- 跑 1-3 个 trial。

目标：

- 验证 prompt、API doc、代码执行、远程环境控制全部打通。

可考核结果：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_reach_target.yaml \
  --model openrouter/google/gemini-2.5-pro-preview \
  --server-url http://127.0.0.1:8110/chat/completions \
  --total-trials 3 \
  --num-workers 1 \
  --record-video True
```

期望：

- `all_responses.json` 中有模型响应。
- `code.py` 中调用 `FrankaRLBenchApi` 暴露的函数。
- `summary.txt` 中无 RPC timeout 或序列化异常。
- 失败时错误能定位到 API、server、task 或模型代码。

## 7. 阶段七：扩展到视觉和更多 RLBench 任务

### 7.1 增加多相机 observation

操作项目：RLBench Docker + CaP-X。

要做什么：

- 在 RLBench `ObservationConfig` 中开启：
  - front camera
  - wrist camera
  - left/right shoulder camera
- 在 CaP-X observation 中映射为：
  - `robot0_robotview`
  - `robot0_eye_in_hand`
  - `left_shoulder`
  - `right_shoulder`

目标：

- 支持视觉分割、抓取规划、多视角估计。

可考核结果：

- 每个 camera key 下均有 `rgb`、`depth`、`intrinsics`、`pose_mat`。
- 图像方向正确，没有上下颠倒或 BGR/RGB 错误。

### 7.2 接入 perception API

操作项目：CaP-X。

要做什么：

- 判断是否复用现有 `FrankaControlApi` / reduced API 的视觉函数。
- 若 RLBench observation 已符合 CaP-X 格式，可以逐步复用：
  - SAM3 segmentation
  - Contact-GraspNet grasp planning
  - PyRoKi IK
- 若坐标系差异大，先在 `FrankaRLBenchApi` 中封装专用函数。

目标：

- 从 privileged object pose 过渡到视觉估计。

可考核结果：

- LLM 可通过图像检测目标对象。
- segmentation mask 与 RGB 对齐。
- 3D point cloud 坐标在机器人/world frame 中合理。

### 7.3 扩展任务集合

操作项目：RLBench Docker + CaP-X。

建议顺序：

1. `ReachTarget`
2. `PickUpCup`
3. `CloseDrawer`
4. `OpenDrawer`
5. `PutItemInDrawer`
6. `StackBlocks`

每个任务需要补充：

- Docker server task loader 支持。
- CaP-X task prompt。
- oracle code。
- YAML config。
- success/reward 映射。

目标：

- 从单任务 smoke test 变成可复现实验集合。

可考核结果：

- 每个任务至少有 3 次 oracle trial。
- 每个任务输出成功率、平均 reward、失败日志。
- 任务失败时能区分是控制、感知、prompt 还是 RLBench server 问题。

## 8. 工程文件清单

### 8.1 CaP-X 仓库预期新增/修改

```text
capx/envs/simulators/rlbench_remote.py
capx/envs/simulators/__init__.py
capx/integrations/franka/rlbench.py
capx/integrations/__init__.py
capx/envs/tasks/franka/franka_rlbench_reach_target.py
capx/envs/tasks/__init__.py
env_configs/rlbench/franka_rlbench_reach_target.yaml
scripts/rlbench_server_client_smoke.py
tests/test_rlbench_remote_env.py
```

### 8.2 RLBench Docker 侧预期新增

如果直接复用 CaP-X 挂载目录：

```text
scripts/rlbench_server.py
scripts/rlbench_server_smoke.py
```

如果后续拆独立 adapter 仓库：

```text
rlbench-adapter/
  README.md
  rlbench_server.py
  requirements.txt
  tests/
```

## 9. 风险与处理策略

| 风险 | 影响 | 处理策略 | 验收方式 |
| --- | --- | --- | --- |
| CoppeliaSim display 或 EGL 不稳定 | RLBench server 启动失败 | 固定 Docker 镜像、DISPLAY、Xvfb/EGL 配置 | `/health` 和 `task.reset()` 稳定 |
| HTTP 传大图慢 | trial 速度慢 | 初期接受，后续换 msgpack、压缩、共享内存 | 单次 `get_observation()` 耗时可测 |
| 坐标系错误 | 机器人动作错误 | 固定 frame 文档和 round-trip test | object pose 与画面/动作一致 |
| 多 worker 并发冲突 | RLBench/CoppeliaSim 多实例不稳定 | 第一版 `num_workers=1`；后续每 worker 一个容器/端口 | 并发 trial 无端口冲突 |
| API 过度复用 Robosuite | 隐性 bug 多 | 新写 `FrankaRLBenchApi` | API doc 与实现只依赖 remote env |
| task success 映射不一致 | 评测指标不可比 | 每个 task 明确 reward/success 来源 | summary 与 RLBench 内部 success 一致 |

## 10. 推荐里程碑

### M1：Docker 内 RLBench 独立运行

范围：

- 固定 Docker 命令。
- 跑通一个 RLBench example。
- 跑通一个 smoke task reset/step。

通过标准：

- `examples/few_shot_rl.py` 或自定义 smoke 脚本成功。
- 无显示、GPU、CoppeliaSim 路径错误。

### M2：RLBench server 最小闭环

范围：

- `/health`
- `/reset`
- `/observation`
- `/success`
- `/move_to_joints` 或 `/move_to_pose`

通过标准：

- 本机 `curl` 能调用所有接口。
- Python client 能还原 RGB/depth/joints。

### M3：CaP-X remote env 闭环

范围：

- `RLBenchRemoteEnv`
- env 注册
- observation 转换

通过标准：

- `env.reset()`、`env.get_observation()`、`env.task_completed()` 在 CaP-X venv 中成功。
- 主环境不安装、不 import RLBench。

### M4：CaP-X oracle task 闭环

范围：

- `FrankaRLBenchApi`
- task wrapper
- YAML config
- oracle code

通过标准：

- `capx/envs/launch.py --use-oracle-code True` 跑通。
- `Sandbox failed: 0`。
- 输出目录结构与 Robosuite 任务一致。

### M5：LLM 评测闭环

范围：

- prompt/API doc 调整。
- 运行 1-3 个模型生成 trial。

通过标准：

- `all_responses.json`、`code.py`、`summary.txt` 完整。
- 失败可解释，成功可复现。

### M6：视觉和多任务扩展

范围：

- 多相机 observation。
- perception API 复用。
- 2-5 个 RLBench task。

通过标准：

- 每个任务有 oracle。
- 至少一个视觉任务可通过 API 读取 RGB/depth 并完成目标定位或抓取规划。

## 11. 当前最小执行路径

建议下一步按以下顺序执行：

1. 在 Docker 中确认 `ReachTarget` 可以 reset 和 step。
2. 写 `scripts/rlbench_server.py`，只实现 `/health`、`/reset`、`/observation`、`/success`。
3. 在 CaP-X 中写 `capx/envs/simulators/rlbench_remote.py`。
4. 写一个本机 smoke client，确认 observation 能解码。
5. 加 `FrankaRLBenchApi` 和 `franka_rlbench_reach_target_code_env`。
6. 用 oracle code 跑 `--total-trials 1`。
7. 成功后再扩展 `goto_pose`、视频保存、多相机和更多任务。

