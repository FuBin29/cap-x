# CaP-X RLBench Docker 接入状态

本文档记录当前 `cap-x` 对 RLBench 的 Docker 隔离接入方案。若文档与代码不一致，以当前代码为准。

核心结论：RLBench 依赖 Ubuntu 20.04、CoppeliaSim、PyRep 等旧运行栈，不安装进本机 CaP-X venv。RLBench 在 Docker 内运行 HTTP adapter，CaP-X 通过 `RLBenchRemoteEnv` 远程调用。

## 当前架构

```text
Host / Ubuntu 24.04 / cap-x uv environment
  capx/envs/launch.py
  capx/envs/simulators/rlbench_remote.py::RLBenchRemoteEnv
  capx/integrations/franka/rlbench.py::FrankaRLBenchApi
  capx/envs/tasks/franka/franka_rlbench_*.py
        |
        | HTTP on 127.0.0.1:8120
        v
Docker / Ubuntu 20.04 / RLBench runtime
  /workspace/cap-x/rlbench_adapter/rlbench_server.py
  RLBench / PyRep / CoppeliaSim
```

设计原则：

- Host 侧 CaP-X 不 import `rlbench`、`pyrep`、`sim`。
- Docker 侧 server 负责 RLBench/CoppeliaSim 生命周期。
- Host 侧 remote env 负责把 RPC observation 转成 CaP-X 约定结构。
- RLBench 使用专门的 `FrankaRLBenchApi`，不复用 Robosuite 私有控制细节。
- 当前支持 privileged smoke/oracle；视觉 perception API 仍是后续工作。

## 2026-05-17 设计审查结论

已全面对齐这三个文件的 server/client 设计：

```text
rlbench_adapter/rlbench_server.py
rlbench_adapter/client_smoke.py
capx/envs/simulators/rlbench_remote.py
```

结论：

- Server 与 `RLBenchRemoteEnv` 的核心协议一致：`/reset` 返回 observation 与上下文，后续请求携带 `X-RLBench-Episode-ID`、`X-RLBench-Task-Name`、`X-RLBench-Action-Sequence-ID`。
- `client_smoke.py` 现在也会携带 episode/action-sequence 上下文，不再绕开 server 的一致性保护。
- task name 在 server 侧统一规范化为小写下划线形式；输入中的空格、连字符也会转换为下划线。
- action 形状已经明确：`/step` 是 8 维 `[7 joint velocities, 1 gripper command]`；`/move_to_joints` 是 7 维；`/move_to_pose` 是 position 3 维、quaternion WXYZ 4 维。
- 缺字段、非法 JSON、非法 action/pose/joint 维度现在返回 HTTP `400`，不会再落成模糊的 HTTP `500`。
- task/episode/action-sequence mismatch 返回 HTTP `409`，用于防止多 client 切 task 或超时重试后的重复动作。
- motion planner 的 `ConfigurationPathError`、`IKError`、`RuntimeError` 仍作为结构化控制失败返回 `ok: false`，不把普通规划失败当成 server 崩溃。
- `RLBenchRemoteEnv` 现在会把非 JSON 响应、URL 错误、socket/timeout/OSError 统一包装为 `RLBenchRemoteError`。

## 完成状态

| 模块 | 状态 | 当前代码/结果 |
| --- | --- | --- |
| Docker 内 RLBench 运行 | 完成 | 容器 `rlbench` 可见 `/workspace/cap-x` 与 `/workspace/RLBench`，`rlbench`/`pyrep` import 成功 |
| RLBench HTTP server | 完成 | `rlbench_adapter/rlbench_server.py`，HTTP threaded 外壳 + RLBench owner-thread 队列 |
| 动态多 task | 完成，顺序复用 | `/reset` 可传 `task_name`，也支持 `/switch_task` |
| Host remote env | 完成 | `capx/envs/simulators/rlbench_remote.py` |
| CaP-X API | 完成 | `FrankaRLBenchApi` 已注册为 `FrankaRLBenchApi` |
| ReachTarget wrapper/config | 完成 | `franka_rlbench_reach_target.py` 与 YAML 已添加，oracle smoke 通过 |
| PickUpCup wrapper/config | 完成 | `franka_rlbench_pick_up_cup.py` 与 YAML 已添加，oracle smoke 通过 |
| 多相机 observation | 完成 | `front`、`wrist`、`left_shoulder`、`right_shoulder` 默认开启 |
| CloseDrawer adapter smoke | 完成 | smoke 测试通过 |
| LLM 评测 | 完成 | OpenAI-compatible server 可接入，已验证 |
| 视觉 perception API | 待做 | 尚未把 SAM/抓取规划等视觉 API 系统性接到 RLBench |

## 当前代码清单

RLBench adapter：

```text
rlbench_adapter/rlbench_server.py
rlbench_adapter/client_smoke.py
```

Host 侧 CaP-X：

```text
capx/envs/simulators/rlbench_remote.py
capx/envs/simulators/__init__.py
capx/integrations/franka/rlbench.py
capx/integrations/__init__.py
capx/envs/tasks/franka/franka_rlbench_reach_target.py
capx/envs/tasks/franka/franka_rlbench_pick_up_cup.py
capx/envs/tasks/franka/franka_rlbench_close_drawer.py
capx/envs/tasks/__init__.py
env_configs/rlbench/franka_rlbench_reach_target.yaml
env_configs/rlbench/franka_rlbench_pick_up_cup.yaml
env_configs/rlbench/franka_rlbench_close_drawer.yaml
```

当前没有 `tests/test_rlbench_remote_env.py`。

## Server 接口

当前 server 使用 Python 标准库 HTTP server，不依赖 FastAPI/Uvicorn。

接口：

```text
GET  /health
POST /switch_task
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

`/health` 返回示例：

```json
{
  "status": "ok",
  "task_name": "reach_target",
  "episode_id": 3,
  "action_sequence_id": 7,
  "busy": false,
  "busy_route": null,
  "busy_request_id": null,
  "service_mode": "single_episode_serial",
  "array_encoding": "base64"
}
```

`/reset` 支持动态切 task：

```json
{
  "task_name": "pick_up_cup",
  "variation": 0,
  "attempts": 5
}
```

`/switch_task` 只切换 active task，不自动 reset：

```json
{"task_name": "reach_target"}
```

返回示例：

```json
{"task_name": "reach_target", "changed": true, "episode_id": 4, "action_sequence_id": 8}
```

## 多 Task 设计语义

一个 server 进程当前只维护一个 active RLBench task，但可以在同一进程中顺序切换 task：

- `RLBenchService._load_task()` 调用 RLBench `Environment.get_task()`，RLBench 会先 unload 当前 scene task，再 load 新 task。
- `POST /reset` 传入 `task_name` 时，会先切到对应 task，再 reset。
- `RLBenchRemoteEnv.reset()` 会自动把 YAML 中的 `task_name` 发给 server。
- server 在 `rlbench-owner-thread` 中创建 RLBench/PyRep/CoppeliaSim，并把所有 simulator 调用派发回这个 owner thread。这样既避免并发，也避免跨 OS thread 调用 CoppeliaSim。
- `RLBenchRemoteEnv` 在 reset 后记录 `episode_id`，后续请求带 `X-RLBench-Episode-ID` 和 `X-RLBench-Task-Name`。如果另一个 env 已经切走 task，server 返回 `409`，避免动作打到错误 task。
- 这是单 CoppeliaSim 实例上的单 episode 串行服务，不是并发 RPC 服务。`/step`、`/reset`、`/move_to_pose` 等所有触碰 RLBench 的路由一次只执行一个，并且都在 simulator owner thread 中执行。
- HTTP 外壳使用 threaded server，因此 `/health` 和 `/shutdown` 这类管理路由不会等待长动作释放 RLBench 执行锁。`/health` 会报告当前是否 `busy`，但不会访问 simulator。
- `POST /switch_task` 如果真的切换 active task，会 bump `episode_id` 和 `action_sequence_id`，即使它不自动 reset。
- mutating action 响应会返回 `action_sequence_id`。客户端后续 mutating 请求带 `X-RLBench-Action-Sequence-ID`；如果超时后 server 其实已经推进了状态，重复动作会收到 `409`，应先拉 `/observation` 同步上下文。

因此：

- 测多个 task、单 worker：可以共用一个 server/端口，顺序运行不同 YAML。
- 真并行多 worker：仍建议每个 worker 使用独立 server、容器或端口。单个 CoppeliaSim/RLBench 实例不是多 episode 并发环境。

## 请求、上下文与错误语义

Server 使用 JSON body，所有响应也是 JSON。数组图像 payload 见下一节。

必填字段：

| Endpoint | 必填字段 | 说明 |
| --- | --- | --- |
| `POST /switch_task` | `task_name` | 切 active task，不自动 reset |
| `POST /reset` | 无 | 可选 `task_name`/`task`、`variation`、`attempts` |
| `POST /step` | `action` | 8 维，前 7 维 joint velocity，第 8 维 gripper command |
| `POST /move_to_joints` | `joints` | 7 维 Franka joint target，可选 `steps`、`tolerance` |
| `POST /move_to_pose` | `position`、`quaternion_wxyz` | `position` 3 维，`quaternion_wxyz` 4 维；可选 `ignore_collisions` |
| `POST /open_gripper` | 无 | 完全打开 |
| `POST /close_gripper` | 无 | 完全关闭 |

客户端上下文 header：

```text
X-RLBench-Episode-ID
X-RLBench-Task-Name
X-RLBench-Action-Sequence-ID
X-RLBench-Request-ID
```

错误码约定：

| HTTP code | 场景 | 客户端处理 |
| --- | --- | --- |
| `400` | 非法 JSON、body 不是 object、缺必填字段、action/pose/joint 维度不对 | 修请求，不应重试同一 payload |
| `409` | task、episode 或 action sequence 与 server 当前状态不一致；或未 reset 就请求 observation/reward/success/control | 先 reset 或拉 `/observation` 同步上下文 |
| `500` | adapter 内部异常、RLBench/PyRep 未被结构化捕获的异常 | 看返回 traceback 和 Docker 日志 |

普通 motion planning 失败不是 HTTP 错误。`/move_to_pose`、`/move_to_joints` 会返回：

```json
{
  "ok": false,
  "error_type": "ConfigurationPathError",
  "error": "Could not plan path to pose: ...",
  "steps": 0,
  "success": false,
  "terminate": false,
  "episode_id": 3,
  "action_sequence_id": 9,
  "observation": {}
}
```

## Observation 与控制约定

RPC 中 numpy array 默认使用 JSON + base64。这个默认值是当前快速版本：不做压缩，CPU 开销低，但 payload 比较大。

```json
{
  "encoding": "base64",
  "dtype": "uint8",
  "shape": [128, 128, 3],
  "data": "..."
}
```

可选无损压缩：

```bash
python rlbench_adapter/rlbench_server.py ... --array-encoding base64_gzip
```

`base64_gzip` 仍按 dtype/shape 还原原始 numpy bytes，不做 JPEG 这类有损压缩；代价是两端多一点 CPU。512 多相机评测时，仍建议通过 `--cameras` 只启用需要的 camera。

后续如果评测吞吐成为瓶颈，优先考虑这些无损方案：

- msgpack raw bytes：保留 dtype/shape 元数据，避免 JSON/base64 膨胀。
- HTTP binary sidecar：JSON 主响应只放 metadata，大数组通过同请求或后续 endpoint 传 binary body。
- 共享内存或 mmap：server 返回 frame handle，host 侧按 handle 读取数组，适合本机 Docker/host 高吞吐。

Host 侧 `RLBenchRemoteEnv` 转换出的主要 observation key：

```text
robot0_joint_pos
robot0_joint_vel
robot0_gripper_qpos
robot0_eef_pos
robot0_eef_quat
robot0_robotview
robot0_eye_in_hand
left_shoulder
right_shoulder
object_poses
target_pose
task_descriptions
task_name
episode_id
action_sequence_id
last_action_result
```

坐标与姿态约定：

- position: RLBench world frame 下的 XYZ，单位米。
- quaternion: Host/API 侧统一为 WXYZ。
- RLBench 内部对象 quaternion 如为 XYZW，由 server 转为 WXYZ。
- reach-only 任务建议使用当前末端 `obs["robot0_eef_quat"]` 作为目标姿态，不要把 object quaternion 直接当作 gripper target orientation。
- `object_poses` 只放真实 scene object pose，不再根据 `task_low_dim_state` 伪造 `target`。`ReachTarget` 本身在 RLBench task 中定义了名为 `target` 的 scene object，因此仍会通过真实 object handle 暴露 `object_poses["target"]`。

`FrankaRLBenchApi` 当前暴露：

```text
get_observation()
get_object_pose(object_name)
goto_pose(position, quaternion_wxyz, z_approach=0.0)
move_to_joints(joints)
open_gripper()
close_gripper()
```

`move_to_pose()`、`move_to_joints()`、`open_gripper()`、`close_gripper()` 的结果会被 Host 侧写入 `obs["last_action_result"]`；失败时也会在 stderr 打印，便于 multi-turn regeneration 判断。

## 运行方式

### 启动 Docker

```bash
docker run -it \
  --name rlbench \
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
docker exec -it rlbench /bin/bash
```

### 启动 RLBench server

在容器中：

```bash
cd /workspace/cap-x
python rlbench_adapter/rlbench_server.py \
  --host 0.0.0.0 \
  --port 8120 \
  --task reach_target \
  --image-size 128
```

`--task` 是初始 task。后续 `client_smoke.py --task ...` 或 CaP-X YAML reset 会动态切换 active task。

常用参数：

```text
--cameras front,wrist,left_shoulder,right_shoulder
--object-names target,cup1,cup2,waypoint0,waypoint1,waypoint2,waypoint3,waypoint4,...
--array-encoding base64
--no-headless
```

### Server smoke

Host 或容器内均可：

```bash
curl http://127.0.0.1:8120/health
python rlbench_adapter/client_smoke.py --server-url http://127.0.0.1:8120 --task reach_target --variation 0
python rlbench_adapter/client_smoke.py --server-url http://127.0.0.1:8120 --task pick_up_cup --variation 0
```

`client_smoke.py` 会：

- `GET /health` 做无上下文健康检查。
- `POST /reset` 后记录 `episode_id`、`task_name`、`action_sequence_id`。
- 后续 `/reward`、`/success`、`/step` 携带上下文 header。
- 解码 `base64` 与 `base64_gzip` 数组。
- 当 adapter 没有启用 front camera 时，尽量从已启用 camera 中选择一个图像打印 shape。

CloseDrawer 当前仍依赖 RLBench 官方 reset validation 能通过。adapter 不再提供 validation bypass：

```bash
python rlbench_adapter/rlbench_server.py \
  --host 0.0.0.0 \
  --port 8120 \
  --task close_drawer \
  --image-size 128 \
  --cameras front,wrist

python rlbench_adapter/client_smoke.py --server-url http://127.0.0.1:8120 --task close_drawer --variation 0
python rlbench_adapter/client_smoke.py --server-url http://127.0.0.1:8120 --task close_drawer --variation 1
python rlbench_adapter/client_smoke.py --server-url http://127.0.0.1:8120 --task close_drawer --variation 2
```

### CaP-X oracle

ReachTarget：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_reach_target.yaml \
  --use-oracle-code True \
  --total-trials 1 \
  --num-workers 1 \
  --record-video True
```

PickUpCup：

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_pick_up_cup.yaml \
  --use-oracle-code True \
  --total-trials 1 \
  --num-workers 1 \
  --record-video True
```

两个 YAML 现在都可以使用同一个 `server_url: http://127.0.0.1:8120`。`task_name` 由 `RLBenchRemoteEnv.reset()` 发送给 server。

## 验证记录

2026-05-14 已记录结果：

- `python examples/few_shot_rl.py` 在 Docker 内跑通。
- `ReachTarget` adapter smoke 可 reset、读取 joint/gripper/RGB/depth/target state，并 no-op step。
- `ReachTarget` oracle：
  - `Sandbox failed: 0`
  - `Task Completed: True`
  - 输出目录：`outputs/oracle/rlbench_reach_target/trial_01_sandboxrc_0_reward_-0.032_taskcompleted_1/`
- `PickUpCup` oracle：
  - `Sandbox failed: 0`
  - `Task Completed: True`
  - 输出目录：`outputs/oracle/rlbench_pick_up_cup/trial_01_sandboxrc_0_reward_0.000_taskcompleted_1/`
- `CloseDrawer` adapter validation bypass 已删除；历史上该任务的官方 waypoint validation 在当前栈中不稳定。

2026-05-17 已完成 server/client 设计审查：

- 检查 `rlbench_server.py`、`client_smoke.py`、`rlbench_remote.py` 的上下文协议、action sequence、防并发语义、数组编码、相机 key 映射。
- 修正坏请求错误处理和 smoke client 上下文 header。
- 未启动 Docker/CoppeliaSim 做实时 reset；本次验证为静态审查和 Python 语法检查。

最近代码语法检查：

```bash
python -m py_compile \
  rlbench_adapter/rlbench_server.py \
  rlbench_adapter/client_smoke.py \
  capx/envs/simulators/rlbench_remote.py
```

## 已修复问题

- `/step` 返回中存在 `numpy.bool_`，标准 `json.dumps` 不能序列化：已添加 numpy scalar/array JSON fallback。
- `/move_to_pose` path stepping 初始不移动：已在 `move_to_pose` 内临时启用 arm control loop，完成后恢复。
- 无 shaped reward 的 RLBench task 在 `/step` 时报错：server 使用 `Environment(... shaped_rewards=False)`。
- reset 阶段失败会保留 RLBench 官方 validation 语义：adapter 不再 monkey patch task validation。
- action sequence mismatch 会返回 `409`，降低超时重试导致重复动作的风险。
- motion planner 抛 `ConfigurationPathError`、`IKError`、`RuntimeError` 时不再直接 HTTP 500：返回结构化失败。
- 单 server 只能初始指定一个 task 的限制：已支持 `/reset` 携带 `task_name` 和 `/switch_task` 顺序动态切换。
- 缺字段、非法 JSON、非法 action/pose/joint 维度不再变成 HTTP 500：现在返回 HTTP 400。
- `client_smoke.py` 已同步 server/client 上下文协议，会携带 episode/task/action-sequence header。
- `RLBenchRemoteEnv` 已补齐非 JSON 响应、URL/socket/timeout/OSError 的包装错误处理。
- gripper helper 已把 server 返回的 gripper step 数纳入 `_sim_step_count`。

## 当前限制与风险

| 风险/限制 | 当前处理 |
| --- | --- |
| 单 server 不支持多个 episode 并发 | RLBench 路由通过 owner-thread 队列串行；`/health`、`/shutdown` 不等长动作；多 worker 使用独立 server/端口 |
| CloseDrawer 原生 waypoint validation 失败 | 不再绕过；需要修复 RLBench/PyRep/CoppeliaSim 栈或任务 waypoint |
| HTTP + base64 传图较慢 | 可用 `--array-encoding base64_gzip` 做无损压缩；更高吞吐可考虑 msgpack 或共享内存 |
| 当前主要依赖 privileged object poses | 视觉 perception API 尚待接入 |
| stdlib HTTP server 功能简单 | 当前足够 smoke；后续需要 schema/并发治理时再考虑 FastAPI/gRPC |

## 后续 TODO

1. 跑并记录 M5 LLM 评测结果。
2. 为 `RLBenchRemoteEnv` 和 server RPC 增加自动化测试。
3. 接入或封装 RLBench 专用视觉 perception API。
4. 为 `CloseDrawer` 增加正式 task wrapper/YAML/oracle，或修复当前环境下的原生 waypoint validation。
5. 扩展更多任务，如 `OpenDrawer`、`PutItemInDrawer`、`StackBlocks`。
6. 如果要跑多 worker，补充多 server/多端口启动脚本。
