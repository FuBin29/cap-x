# Artance 工作日志

本文记录当前机器上常用的 CaP-X 启动命令，以及 CaP-X 联动 RLBench Docker 的手动验证流程。

## 目录约定

宿主机项目根目录：

```bash
cd ~/projects/artance
```

CaP-X 仓库：

```bash
cd ~/projects/artance/cap-x
```

RLBench 仓库：

```bash
cd ~/projects/artance/RLBench
```

Docker 内挂载路径：

```bash
/workspace/cap-x
/workspace/RLBench
```

## 1. CaP-X 常用启动命令

本节是 CaP-X 本身的常用入口，和 RLBench Docker 是否启动无关。

### 1.1 激活 CaP-X 环境

在宿主机执行：

```bash
cd ~/projects/artance/cap-x
source .venv/bin/activate
```

后续命令默认在 `~/projects/artance/cap-x` 中执行。

### 1.2 手动预启动后端 API 服务

适用于多次评测复用同一组视觉/后端 API 服务：

```bash
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

或启动完整服务：

```bash
uv run --no-sync --active capx/serving/launch_servers.py --profile full
```

### 1.3 启动 OpenRouter / OpenAI-compatible Proxy

先确认本机是否已有服务：

```bash
curl -sS --max-time 3 http://127.0.0.1:8110/health
```

如果没有服务，启动 proxy：

```bash
uv run --no-sync --active capx/serving/openrouter_server.py \
  --api-key your-provider-token \
  --base-url https://your-provider.example.com/v1/ \
  --port 8110
```

如果已经在环境变量或配置中放好了 token，也可以直接：

```bash
uv run --no-sync --active capx/serving/openrouter_server.py --port 8110
```

如果需要 HTTP 代理：

```bash
OPENROUTER_HTTP_PROXY=http://10.156.216.30:16371 \
uv run --no-sync --active capx/serving/openrouter_server.py --port 8110
```

### 1.4 Web UI 交互式观察

Web UI 通常不会自动启动 VLM client，需要先手动启动模型服务。

```bash
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
  --model gemini-2.5-pro \
  --web-ui True
```

## 2. CaP-X 联动 RLBench Docker

注意：RLBench adapter server 现在放在 `cap-x/rlbench_adapter/rlbench_server.py`。旧的 `cap-x/scripts/rlbench_server.py` 路径不再使用。

当前平台结构：

```text
Host CaP-X
  capx/envs/simulators/rlbench_remote.py::RLBenchRemoteEnv
  capx/integrations/franka/rlbench.py::FrankaRLBenchApi
        |
        | HTTP 127.0.0.1:8120
        v
Docker RLBench runtime
  rlbench_adapter/rlbench_server.py
  RLBench / PyRep / CoppeliaSim
```

协议要点：

- `POST /reset` 返回 `episode_id`、`task_name`、`action_sequence_id` 与 observation。
- task name 在 server 侧统一规范化为小写下划线形式；输入中的空格、连字符也会转换为下划线。
- 后续 observation/reward/success/control 请求携带 `X-RLBench-Episode-ID`、`X-RLBench-Task-Name`；mutating action 额外携带 `X-RLBench-Action-Sequence-ID`。
- `400` 表示请求本身不合法，例如非法 JSON、缺字段、action/pose/joint 维度不对。
- `409` 表示 task/episode/action-sequence 上下文不一致，通常需要 reset 或重新拉 `/observation`。
- 普通 motion planning 失败会返回 `ok: false` 的结构化结果，不当作 HTTP 500。

### 2.1 显示服务

如果当前机器还没有可用的 `DISPLAY=:99`，先在宿主机执行一次：

```bash
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
sudo nohup X :99 & disown
```

如果 Docker 容器已经启动，一般不需要重复执行。

### 2.2 确认 Docker 容器

在宿主机执行：

```bash
docker ps --filter name=rlbench --format '{{.Names}} {{.Status}} {{.Image}}'
```

期望看到：

```text
rlbench Up ... rlbench-fubin:v1.2
```

如果容器不存在，用项目根目录中的挂载路径启动：

```bash
cd ~/projects/artance

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

### 2.3 容器内基础检查

在容器内执行：

```bash
cd /workspace/RLBench

python - <<'PY'
import rlbench
import pyrep
print("rlbench", rlbench.__file__)
print("pyrep", pyrep.__file__)
PY
```

可选：跑 RLBench 官方 example。

```bash
cd /workspace/RLBench
python examples/few_shot_rl.py
```

期望最终看到 `Done`。

### 2.4 启动 RLBench Adapter

在宿主机新开一个终端执行下面命令。该命令会在 Docker 容器里启动 RLBench adapter server，并监听宿主机 `127.0.0.1:8120`。

通常只需要启动一个 adapter。`--task` 是初始 task，后续 `client_smoke.py --task ...` 或 CaP-X YAML 中的 `task_name` 会在 reset 时让 server 动态切换 active task。

adapter 是单 CoppeliaSim 实例的单 episode 串行服务。HTTP 外壳可以并行响应 `/health`、`/shutdown`，但所有 RLBench/PyRep/CoppeliaSim 调用都会派发到创建 simulator 的 `rlbench-owner-thread` 中顺序执行。

```bash
cd ~/projects/artance/cap-x

docker exec rlbench bash -lc \
  'cd /workspace/cap-x && python rlbench_adapter/rlbench_server.py \
    --host 0.0.0.0 \
    --port 8120 \
    --task reach_target \
    --image-size 512'
```

默认多相机为 `front,wrist,left_shoulder,right_shoulder`。为了调试更快，可以用 `--cameras front,wrist`。

```bash
cd ~/projects/artance/cap-x

docker exec rlbench bash -lc \
  'cd /workspace/cap-x && python rlbench_adapter/rlbench_server.py \
    --host 0.0.0.0 \
    --port 8120 \
    --task close_drawer \
    --image-size 128 \
    --cameras front,wrist'
```

说明：adapter 不再绕过 RLBench 官方 waypoint validation。若某个任务在当前 Docker/RLBench/CoppeliaSim 组合中 reset 失败，需要按 RLBench/PyRep 栈本身继续排查，而不是在 adapter 层 monkey patch validation。

常用参数：

```text
--cameras front,wrist,left_shoulder,right_shoulder
--object-names target,cup1,cup2,waypoint0,waypoint1,waypoint2,waypoint3,waypoint4,...
--array-encoding base64      # 默认快速版本；base64_gzip 为可选无损压缩
--no-headless
```

停止 adapter：

```bash
curl -sS -X POST http://127.0.0.1:8120/shutdown -H 'Content-Type: application/json' -d '{}'
```

### 2.5 Adapter Smoke Test

保持上一步 adapter 运行，在宿主机另一个终端执行：

```bash
cd ~/projects/artance/cap-x
python rlbench_adapter/client_smoke.py --server-url http://127.0.0.1:8120 --task reach_target --variation 0
python rlbench_adapter/client_smoke.py --server-url http://127.0.0.1:8120 --task pick_up_cup --variation 0
```

也可以 smoke `CloseDrawer`，但该任务仍依赖当前 RLBench/PyRep 栈能通过官方 reset validation：

```bash
python rlbench_adapter/client_smoke.py --server-url http://127.0.0.1:8120 --task close_drawer --variation 0
python rlbench_adapter/client_smoke.py --server-url http://127.0.0.1:8120 --task close_drawer --variation 1
python rlbench_adapter/client_smoke.py --server-url http://127.0.0.1:8120 --task close_drawer --variation 2
```

期望看到：

- `/health` 返回 ok
- RGB/depth shape 正确
- camera keys 正确
- object pose 中包含当前任务对象
- `/step` 不报错

`client_smoke.py` 会在 reset 后记录 `episode_id`、`task_name`、`action_sequence_id`，后续 `/reward`、`/success`、`/step` 都会携带对应上下文 header。这样 smoke test 也会覆盖 server 的 task/episode/action-sequence 一致性保护。

### 2.6 CaP-X Remote Env Smoke Test

在宿主机 CaP-X 环境执行：

```bash
cd ~/projects/artance/cap-x
source .venv/bin/activate

uv run --no-sync --active python - <<'PY'
from capx.envs.simulators.rlbench_remote import RLBenchRemoteEnv

env = RLBenchRemoteEnv(server_url="http://127.0.0.1:8120", task_name="reach_target")
obs, info = env.reset(options={"variation": 0})

print(info)
for key in ["robot0_robotview", "robot0_eye_in_hand", "left_shoulder", "right_shoulder"]:
    if key in obs:
        cam = obs[key]
        print(key, cam["images"]["rgb"].shape, cam["images"]["depth"].shape)

print("objects", sorted(obs["object_poses"].keys()))
print("reward", float(env.compute_reward()))
print("done", bool(env.task_completed()))
PY
```

如果 adapter 只启动了 `--cameras front,wrist`，则只会看到 `robot0_robotview` 和 `robot0_eye_in_hand`。

### 2.7 Oracle 评测

先启动 adapter，再运行对应 YAML。两个 YAML 可以共用同一个 `server_url: http://127.0.0.1:8120`；`RLBenchRemoteEnv.reset()` 会把 YAML 中的 `task_name` 发给 server。

ReachTarget：

```bash
cd ~/projects/artance/cap-x
source .venv/bin/activate

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_reach_target.yaml \
  --use-oracle-code True \
  --total-trials 1 \
  --num-workers 1 \
  --record-video True
```

PickUpCup：

```bash
cd ~/projects/artance/cap-x
source .venv/bin/activate

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_pick_up_cup.yaml \
  --use-oracle-code True \
  --total-trials 1 \
  --num-workers 1 \
  --record-video True
```

期望：

- `Sandbox failed: 0`
- `Task Completed: True`
- 输出目录在 `outputs/oracle/rlbench_*`
- trial 目录中包含 `code.py` 和视频文件

### 2.8 LLM 评测

先确认本机是否已有 OpenAI-compatible server：

```bash
curl -sS --max-time 3 http://127.0.0.1:8110/health
```

如果没有服务，先回到 `1.3` 启动 OpenRouter / OpenAI-compatible proxy。

运行 LLM trial：

```bash
cd ~/projects/artance/cap-x
source .venv/bin/activate

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_reach_target.yaml \
  --model gemini-2.5-pro \
  --server-url http://127.0.0.1:8110/chat/completions \
  --total-trials 1 \
  --num-workers 1 \
  --record-video True

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_reach_target.yaml \
  --model gemini-3.1-pro-preview \
  --server-url http://127.0.0.1:8110/chat/completions \
  --total-trials 1 \
  --num-workers 1 \
  --record-video True

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_pick_up_cup.yaml \
  --model gemini-2.5-pro \
  --server-url http://127.0.0.1:8110/chat/completions \
  --total-trials 1 \
  --num-workers 1 \
  --record-video True

uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/rlbench/franka_rlbench_close_drawer.yaml \
  --model gemini-2.5-pro \
  --server-url http://127.0.0.1:8110/chat/completions \
  --total-trials 1 \
  --num-workers 1 \
  --record-video True
```


## 3. 当前已知状态

已验证：

- `ReachTarget` oracle 可通过。
- `PickUpCup` oracle 可通过。
- adapter 多相机 observation 可返回并被 `RLBenchRemoteEnv` 解析。
- adapter 不再包含 waypoint validation bypass workaround。
- 2026-05-17 已审查 `rlbench_server.py`、`client_smoke.py`、`rlbench_remote.py` 的协议一致性、错误处理、相机 key 映射、数组编码、action sequence 防重入语义。
- 坏请求现在应返回 HTTP `400`，上下文不一致返回 HTTP `409`，普通 motion planning 失败返回结构化 `ok: false`。

当前阻塞：

- `CloseDrawer` 在当前 Docker/RLBench/CoppeliaSim 组合中可能仍会在 reset 阶段失败。
- 失败位置是 RLBench 内部 demonstration waypoint feasibility validation，CoppeliaSim 返回 `-1`。

排查状态：

- 已尝试 adapter reset 时临时开启 arm control loop。
- 已尝试 reset 重试。
- 已删除 adapter 级 validation monkey patch，避免误以为官方 demonstration validation 已通过。
