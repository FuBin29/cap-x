#!/usr/bin/env python3
"""Minimal stdlib HTTP RPC server for RLBench running inside Docker."""

from __future__ import annotations

import argparse
import base64
import gzip
import importlib
import json
import queue
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import numpy as np
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects.object import Object

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


CAMERA_SPECS = {
    "front": ("front_camera", "front_rgb", "front_depth"),
    "wrist": ("wrist_camera", "wrist_rgb", "wrist_depth"),
    "left_shoulder": ("left_shoulder_camera", "left_shoulder_rgb", "left_shoulder_depth"),
    "right_shoulder": ("right_shoulder_camera", "right_shoulder_rgb", "right_shoulder_depth"),
    "overhead": ("overhead_camera", "overhead_rgb", "overhead_depth"),
}


def _normalize_task_name(task_name: str) -> str:
    return task_name.strip().lower().replace("-", "_").replace(" ", "_")


def _task_class(task_name: str):
    module_name = _normalize_task_name(task_name)
    class_name = "".join(part.capitalize() for part in module_name.split("_"))
    module = importlib.import_module(f"rlbench.tasks.{module_name}")
    return getattr(module, class_name)


def _array_payload(array: Optional[np.ndarray], encoding: str = "base64") -> Optional[Dict[str, Any]]:
    if array is None:
        return None
    arr = np.asarray(array)
    data = np.ascontiguousarray(arr).tobytes()
    if encoding == "base64":
        payload = base64.b64encode(data).decode("ascii")
    elif encoding == "base64_gzip":
        payload = base64.b64encode(gzip.compress(data)).decode("ascii")
    else:
        raise ValueError(f"Unsupported array encoding: {encoding!r}")
    return {
        "encoding": encoding,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": payload,
    }


def _to_list(value: Any) -> Optional[list]:
    if value is None:
        return None
    return np.asarray(value).tolist()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _xyzw_to_wxyz(quat_xyzw: Any) -> list:
    qx, qy, qz, qw = np.asarray(quat_xyzw, dtype=float).tolist()
    return [qw, qx, qy, qz]


def _wxyz_to_xyzw(quat_wxyz: Any) -> list:
    qw, qx, qy, qz = np.asarray(quat_wxyz, dtype=float).tolist()
    return [qx, qy, qz, qw]


class RPCError(RuntimeError):
    def __init__(self, status: int, payload: Dict[str, Any]):
        super().__init__(str(payload.get("error", payload)))
        self.status = status
        self.payload = payload


class RLBenchService:
    def __init__(
        self,
        task_name: str,
        image_size: int,
        headless: bool,
        cameras: list[str],
        object_names: list[str],
        array_encoding: str = "base64",
    ):
        self.task_name = _normalize_task_name(task_name)
        self.image_size = image_size
        self.headless = headless
        self.cameras = cameras
        self.object_names = object_names
        self.array_encoding = array_encoding
        self.busy_route: Optional[str] = None
        self.busy_request_id: Optional[str] = None
        self._call_queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._start_error: Optional[BaseException] = None
        self.env: Optional[Environment] = None
        self.task = None
        self.descriptions = []
        self.obs = None
        self.last_reward = 0.0
        self.last_terminate = False
        self.episode_id = 0
        self.action_sequence_id = 0

    def start(self) -> None:
        self._worker = threading.Thread(
            target=self._worker_main,
            name="rlbench-owner-thread",
            daemon=False,
        )
        self._worker.start()
        self._ready.wait()
        if self._start_error is not None:
            raise RuntimeError("RLBench worker failed to start") from self._start_error

    def _worker_main(self) -> None:
        try:
            self._launch()
        except BaseException as exc:
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()

        while True:
            item = self._call_queue.get()
            if item is None:
                break
            route, request_id, fn, done, box = item
            self.busy_route = route
            self.busy_request_id = request_id
            try:
                box["result"] = fn()
            except BaseException as exc:
                box["error"] = exc
                box["traceback"] = traceback.format_exc()
            finally:
                self.busy_route = None
                self.busy_request_id = None
                done.set()

        try:
            self._shutdown_env()
        except Exception:
            traceback.print_exc()

    def call_on_owner_thread(self, route: str, request_id: Optional[str], fn):
        if self._worker is None:
            raise RuntimeError("RLBench worker is not started")
        if threading.current_thread() is self._worker:
            return fn()
        done = threading.Event()
        box: Dict[str, Any] = {}
        self._call_queue.put((route, request_id, fn, done, box))
        done.wait()
        if "error" in box:
            if isinstance(box["error"], RPCError):
                raise box["error"]
            worker_traceback = box.get("traceback", "")
            raise RuntimeError(f"{box['error']}\nWorker traceback:\n{worker_traceback}") from box[
                "error"
            ]
        return box["result"]

    def _launch(self) -> None:
        obs_config = ObservationConfig()
        obs_config.set_all(False)
        for camera_name in self.cameras:
            config_attr, _, _ = CAMERA_SPECS[camera_name]
            camera_config = getattr(obs_config, config_attr)
            camera_config.rgb = True
            camera_config.depth = True
            camera_config.point_cloud = False
            camera_config.mask = False
            camera_config.image_size = (self.image_size, self.image_size)
            camera_config.depth_in_meters = True
        obs_config.joint_positions = True
        obs_config.joint_velocities = True
        obs_config.gripper_open = True
        obs_config.gripper_pose = True
        obs_config.gripper_matrix = True
        obs_config.task_low_dim_state = True

        self.env = Environment(
            action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
            obs_config=obs_config,
            headless=self.headless,
            shaped_rewards=False,
        )
        self.env.launch()
        self._load_task(self.task_name)

    def _clear_episode_state(self) -> None:
        self.descriptions = []
        self.obs = None
        self.last_reward = 0.0
        self.last_terminate = False

    def _mark_mutated(self) -> int:
        self.action_sequence_id += 1
        return self.action_sequence_id

    def status(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "task_name": self.task_name,
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
            "busy": self.busy_route is not None,
            "busy_route": self.busy_route,
            "busy_request_id": self.busy_request_id,
            "service_mode": "single_episode_serial",
            "array_encoding": self.array_encoding,
        }

    def _load_task(self, task_name: str) -> None:
        if self.env is None:
            raise RuntimeError("RLBench environment is not launched")
        normalized = _normalize_task_name(task_name)
        self.task = self.env.get_task(_task_class(normalized))
        self.task_name = normalized
        self._clear_episode_state()

    def switch_task(self, task_name: str) -> Dict[str, Any]:
        normalized = _normalize_task_name(task_name)
        changed = normalized != self.task_name
        if changed or self.task is None:
            self._load_task(normalized)
        if changed:
            self.episode_id += 1
            self._mark_mutated()
        return {
            "task_name": self.task_name,
            "changed": changed,
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
        }

    def _shutdown_env(self) -> None:
        if self.env is not None:
            self.env.shutdown()
            self.env = None
        self.task = None
        self._clear_episode_state()

    def shutdown(self) -> None:
        if self._worker is None:
            self._shutdown_env()
            return
        if threading.current_thread() is self._worker:
            self._shutdown_env()
            return
        if self._worker.is_alive():
            self._call_queue.put(None)
            self._worker.join()

    def reset(
        self,
        variation: Optional[int] = None,
        attempts: int = 5,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if task_name is not None:
            normalized = _normalize_task_name(task_name)
            if normalized != self.task_name or self.task is None:
                self._load_task(normalized)
        if self.task is None:
            raise RuntimeError("RLBench task is not initialized")
        if variation is not None:
            self.task.set_variation(int(variation))
        else:
            self.task.sample_variation()
        last_error = None
        for _ in range(max(1, int(attempts))):
            try:
                self.descriptions, self.obs = self.task.reset()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        self.last_reward = 0.0
        self.last_terminate = False
        self.episode_id += 1
        self._mark_mutated()
        return {
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
            "task_name": self.task_name,
            "descriptions": self.descriptions,
            "observation": self.observation(),
        }

    def _require_obs(self):
        if self.task is None:
            raise RuntimeError("RLBench task is not initialized")
        if self.obs is None:
            raise RPCError(
                409,
                {
                    "error": (
                        "No active RLBench episode. Call POST /reset before requesting "
                        "observation, reward, success, or control actions."
                    ),
                    "task_name": self.task_name,
                    "episode_id": self.episode_id,
                    "action_sequence_id": self.action_sequence_id,
                },
            )
        return self.obs

    def observation(self) -> Dict[str, Any]:
        obs = self._require_obs()
        misc = obs.misc or {}
        cameras = {}
        for camera_name in self.cameras:
            config_attr, rgb_attr, depth_attr = CAMERA_SPECS[camera_name]
            cameras[camera_name] = {
                "rgb": _array_payload(getattr(obs, rgb_attr), self.array_encoding),
                "depth": _array_payload(getattr(obs, depth_attr), self.array_encoding),
                "intrinsics": _to_list(misc.get(f"{config_attr}_intrinsics")),
                "pose_mat": _to_list(misc.get(f"{config_attr}_extrinsics")),
            }
        object_poses = {}
        for name in self.object_names:
            try:
                obj = Object.get_object(name)
                object_poses[name] = {
                    "position": _to_list(obj.get_position()),
                    "quaternion_wxyz": _xyzw_to_wxyz(obj.get_quaternion()),
                }
            except Exception:
                pass
        gripper_pose = None
        if obs.gripper_pose is not None:
            pose = np.asarray(obs.gripper_pose, dtype=float)
            gripper_pose = {
                "position": pose[:3].tolist(),
                "quaternion_wxyz": _xyzw_to_wxyz(pose[3:7]),
            }

        return {
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
            "task_name": self.task_name,
            "task_descriptions": self.descriptions,
            "cameras": cameras,
            "rgb": _array_payload(obs.front_rgb, self.array_encoding),
            "depth": _array_payload(obs.front_depth, self.array_encoding),
            "front_camera": {
                "intrinsics": _to_list(misc.get("front_camera_intrinsics")),
                "pose_mat": _to_list(misc.get("front_camera_extrinsics")),
            },
            "joint_positions": _to_list(obs.joint_positions),
            "joint_velocities": _to_list(obs.joint_velocities),
            "gripper_open": obs.gripper_open,
            "gripper_pose": gripper_pose,
            "object_poses": object_poses,
            "task_low_dim_state": _to_list(obs.task_low_dim_state),
            "last_reward": self.last_reward,
            "last_terminate": self.last_terminate,
            "success": self.success()["success"],
        }

    def success(self) -> Dict[str, Any]:
        if self.task is None:
            raise RuntimeError("RLBench task is not initialized")
        self._require_obs()
        success, terminate = self.task._task.success()
        return {"success": bool(success), "terminate": bool(terminate)}

    def reward(self) -> Dict[str, Any]:
        if self.task is None:
            raise RuntimeError("RLBench task is not initialized")
        self._require_obs()
        reward = self.task._task.reward()
        return {"reward": float(reward if reward is not None else self.last_reward)}

    def step(self, action: Any) -> Dict[str, Any]:
        if self.task is None:
            raise RuntimeError("RLBench task is not initialized")
        self._require_obs()
        try:
            action_array = np.asarray(action, dtype=np.float32).reshape(8)
        except (TypeError, ValueError) as exc:
            raise RPCError(
                400,
                {
                    "error": "POST /step requires an 8-element action: 7 joint velocities plus 1 gripper command.",
                    "episode_id": self.episode_id,
                    "action_sequence_id": self.action_sequence_id,
                },
            ) from exc
        self.obs, self.last_reward, self.last_terminate = self.task.step(action_array)
        self._mark_mutated()
        return {
            "observation": self.observation(),
            "reward": float(self.last_reward),
            "terminate": bool(self.last_terminate),
            "success": self.success()["success"],
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
        }

    def _control_response(
        self,
        *,
        ok: bool,
        steps: int,
        error_type: Optional[str] = None,
        error: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.task is None:
            raise RuntimeError("RLBench task is not initialized")
        if self.obs is None:
            self.obs = self.task._scene.get_observation()
        success = self.success()
        self.last_terminate = success["terminate"]
        payload: Dict[str, Any] = {
            "ok": ok,
            "steps": int(steps),
            "success": success["success"],
            "terminate": success["terminate"],
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
            "observation": self.observation(),
        }
        if not ok:
            payload["error_type"] = error_type or "Error"
            payload["error"] = f"{prefix}: {error}" if prefix and error else error
        return payload

    def move_to_joints(self, joints: Any, steps: int = 200, tolerance: float = 0.01) -> Dict[str, Any]:
        self._require_obs()
        scene = self.task._scene
        arm = scene.robot.arm
        previous_control = arm.joints[0].is_control_loop_enabled()
        try:
            target = np.asarray(joints, dtype=float).reshape(7)
        except (TypeError, ValueError) as exc:
            raise RPCError(
                400,
                {
                    "error": "POST /move_to_joints requires a 7-element joints array.",
                    "episode_id": self.episode_id,
                    "action_sequence_id": self.action_sequence_id,
                },
            ) from exc
        arm.set_control_loop_enabled(True)
        arm.set_joint_target_positions(target)
        count = 0
        error: Optional[Exception] = None
        try:
            for count in range(1, int(steps) + 1):
                scene.step()
                current = np.asarray(arm.get_joint_positions(), dtype=float)
                if float(np.max(np.abs(current - target))) <= float(tolerance):
                    break
        except (ConfigurationPathError, IKError, RuntimeError) as exc:
            error = exc
        finally:
            arm.set_control_loop_enabled(previous_control)
        self.obs = scene.get_observation()
        self._mark_mutated()
        if error is not None:
            return self._control_response(
                ok=False,
                steps=count,
                error_type=error.__class__.__name__,
                error=str(error),
                prefix="Could not move to joint positions",
            )
        payload = self._control_response(ok=True, steps=count)
        payload["joint_positions"] = _to_list(self.obs.joint_positions)
        return payload

    def move_to_pose(self, position: Any, quaternion_wxyz: Any, ignore_collisions: bool = True) -> Dict[str, Any]:
        self._require_obs()
        scene = self.task._scene
        arm = scene.robot.arm
        previous_control = arm.joints[0].is_control_loop_enabled()
        try:
            target_position = np.asarray(position, dtype=float).reshape(3).tolist()
            target_quaternion_xyzw = _wxyz_to_xyzw(np.asarray(quaternion_wxyz, dtype=float).reshape(4))
        except (TypeError, ValueError) as exc:
            raise RPCError(
                400,
                {
                    "error": "POST /move_to_pose requires position shape (3,) and quaternion_wxyz shape (4,).",
                    "episode_id": self.episode_id,
                    "action_sequence_id": self.action_sequence_id,
                },
            ) from exc
        arm.set_control_loop_enabled(True)
        steps = 0
        error: Optional[Exception] = None
        try:
            path = arm.get_path(
                target_position,
                quaternion=target_quaternion_xyzw,
                ignore_collisions=bool(ignore_collisions),
            )
            done = False
            while not done:
                done = path.step()
                scene.step()
                steps += 1
                if self.success()["success"]:
                    break
        except (ConfigurationPathError, IKError, RuntimeError) as exc:
            error = exc
        finally:
            arm.set_control_loop_enabled(previous_control)
        self.obs = scene.get_observation()
        self._mark_mutated()
        if error is not None:
            return self._control_response(
                ok=False,
                steps=steps,
                error_type=error.__class__.__name__,
                error=str(error),
                prefix="Could not plan path to pose",
            )
        return self._control_response(ok=True, steps=steps)

    def set_gripper(self, open_amount: float) -> Dict[str, Any]:
        self._require_obs()
        Discrete().action(self.task._scene, np.asarray([open_amount], dtype=np.float32))
        self.obs = self.task._scene.get_observation()
        self._mark_mutated()
        payload = self._control_response(ok=True, steps=1)
        payload["gripper_open"] = self.obs.gripper_open
        return payload


class RequestHandler(BaseHTTPRequestHandler):
    service: RLBenchService

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RPCError(400, {"error": "Invalid Content-Length header."}) from exc
        if length == 0:
            return {}
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RPCError(400, {"error": f"Invalid JSON request body: {exc}"}) from exc
        if not isinstance(body, dict):
            raise RPCError(400, {"error": "JSON request body must be an object."})
        return body

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _context_error(self, body: Dict[str, Any], *, check_action_sequence: bool) -> Optional[str]:
        expected_task = (
            self.headers.get("X-RLBench-Task-Name")
            or body.get("task_name")
            or body.get("expected_task_name")
        )
        if expected_task is not None:
            normalized = _normalize_task_name(str(expected_task))
            if normalized != self.service.task_name:
                return (
                    f"Active task is {self.service.task_name!r}, "
                    f"but request expected {normalized!r}. Reset or switch_task first."
                )

        expected_episode = self.headers.get("X-RLBench-Episode-ID") or body.get("episode_id")
        if expected_episode is not None:
            try:
                episode_id = int(expected_episode)
            except (TypeError, ValueError):
                return f"Invalid episode id: {expected_episode!r}"
            if episode_id != self.service.episode_id:
                return (
                    f"Active episode is {self.service.episode_id}, "
                    f"but request expected {episode_id}. Reset before continuing."
                )

        if check_action_sequence:
            expected_sequence = (
                self.headers.get("X-RLBench-Action-Sequence-ID")
                or body.get("action_sequence_id")
            )
            if expected_sequence is not None:
                try:
                    action_sequence_id = int(expected_sequence)
                except (TypeError, ValueError):
                    return f"Invalid action sequence id: {expected_sequence!r}"
                if action_sequence_id != self.service.action_sequence_id:
                    return (
                        f"Active action sequence is {self.service.action_sequence_id}, "
                        f"but request expected {action_sequence_id}. "
                        "Fetch /observation before retrying a mutating action."
                    )
        return None

    def _handle(self, method: str) -> None:
        try:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            body = self._read_json() if method == "POST" else {}
            if method == "GET" and path == "/health":
                self._send(200, self.service.status())
                return
            if method == "POST" and path == "/shutdown":
                payload = {"status": "shutting_down"}
                self._send(200, payload)
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            rlbench_route = (
                (method == "POST" and path in {
                    "/switch_task",
                    "/reset",
                    "/step",
                    "/move_to_joints",
                    "/move_to_pose",
                    "/open_gripper",
                    "/close_gripper",
                })
                or (method == "GET" and path in {
                    "/observation",
                    "/reward",
                    "/success",
                })
            )
            if not rlbench_route:
                self._send(404, {"error": f"No route for {method} {path}"})
                return

            request_id = self.headers.get("X-RLBench-Request-ID") or body.get("request_id")
            request_id_str = str(request_id) if request_id is not None else None

            def _invoke_on_owner_thread() -> Dict[str, Any]:
                def _missing_fields(*fields: str) -> Optional[Dict[str, Any]]:
                    missing = [field for field in fields if field not in body]
                    if not missing:
                        return None
                    return {
                        "_http_status": 400,
                        "error": f"Missing required JSON field(s): {', '.join(missing)}",
                        "task_name": self.service.task_name,
                        "episode_id": self.service.episode_id,
                        "action_sequence_id": self.service.action_sequence_id,
                    }

                context_route = path in {
                    "/step",
                    "/observation",
                    "/reward",
                    "/success",
                    "/move_to_joints",
                    "/move_to_pose",
                    "/open_gripper",
                    "/close_gripper",
                }
                mutating_episode_route = path in {
                    "/step",
                    "/move_to_joints",
                    "/move_to_pose",
                    "/open_gripper",
                    "/close_gripper",
                }
                if context_route:
                    error = self._context_error(
                        body, check_action_sequence=mutating_episode_route
                    )
                    if error is not None:
                        return {
                            "_http_status": 409,
                            "error": error,
                            "task_name": self.service.task_name,
                            "episode_id": self.service.episode_id,
                            "action_sequence_id": self.service.action_sequence_id,
                        }
                if method == "POST" and path == "/switch_task":
                    missing = _missing_fields("task_name")
                    if missing is not None:
                        return missing
                    payload = self.service.switch_task(body["task_name"])
                elif method == "POST" and path == "/reset":
                    payload = self.service.reset(
                        body.get("variation"),
                        body.get("attempts", 5),
                        body.get("task_name") or body.get("task"),
                    )
                elif method == "POST" and path == "/step":
                    missing = _missing_fields("action")
                    if missing is not None:
                        return missing
                    payload = self.service.step(body["action"])
                elif method == "GET" and path == "/observation":
                    payload = self.service.observation()
                elif method == "GET" and path == "/reward":
                    payload = self.service.reward()
                elif method == "GET" and path == "/success":
                    payload = self.service.success()
                elif method == "POST" and path == "/move_to_joints":
                    missing = _missing_fields("joints")
                    if missing is not None:
                        return missing
                    payload = self.service.move_to_joints(
                        body["joints"], body.get("steps", 200), body.get("tolerance", 0.01)
                    )
                elif method == "POST" and path == "/move_to_pose":
                    missing = _missing_fields("position", "quaternion_wxyz")
                    if missing is not None:
                        return missing
                    payload = self.service.move_to_pose(
                        body["position"],
                        body["quaternion_wxyz"],
                        body.get("ignore_collisions", True),
                    )
                elif method == "POST" and path == "/open_gripper":
                    payload = self.service.set_gripper(1.0)
                elif method == "POST" and path == "/close_gripper":
                    payload = self.service.set_gripper(0.0)
                else:
                    return {"_http_status": 404, "error": f"No route for {method} {path}"}
                if request_id is not None:
                    payload["request_id"] = request_id
                return payload

            payload = self.service.call_on_owner_thread(
                path, request_id_str, _invoke_on_owner_thread
            )
            status = int(payload.pop("_http_status", 200))
            self._send(status, payload)
        except RPCError as exc:
            self._send(exc.status, exc.payload)
        except Exception as exc:
            self._send(
                500,
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument("--task", default="reach_target")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--cameras",
        default="front,wrist,left_shoulder,right_shoulder",
        help="Comma-separated cameras. Options: front,wrist,left_shoulder,right_shoulder,overhead.",
    )
    parser.add_argument(
        "--object-names",
        default="target,cup1,cup2,waypoint0,waypoint1,waypoint2,waypoint3,waypoint4,waypoint_anchor_bottom,waypoint_anchor_middle,waypoint_anchor_top",
        help="Comma-separated scene object names to expose as privileged poses when present.",
    )
    parser.add_argument(
        "--array-encoding",
        choices=("base64", "base64_gzip"),
        default="base64",
        help="Lossless numpy array transport encoding. base64_gzip trades CPU for smaller JSON payloads.",
    )
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()
    cameras = [name.strip() for name in args.cameras.split(",") if name.strip()]
    unknown = sorted(set(cameras) - set(CAMERA_SPECS))
    if unknown:
        raise ValueError(f"Unknown cameras: {unknown}. Valid cameras: {sorted(CAMERA_SPECS)}")
    object_names = [name.strip() for name in args.object_names.split(",") if name.strip()]

    service = RLBenchService(
        task_name=args.task,
        image_size=args.image_size,
        headless=not args.no_headless,
        cameras=cameras,
        object_names=object_names,
        array_encoding=args.array_encoding,
    )
    RequestHandler.service = service
    service.start()
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    server.daemon_threads = True
    print(
        f"RLBench server listening on http://{args.host}:{args.port} "
        f"for task {service.task_name}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
