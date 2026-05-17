from __future__ import annotations

import base64
import gzip
import json
import socket
import sys
import urllib.error
import urllib.request
import uuid
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv


CAMERA_KEY_MAP = {
    "front": "robot0_robotview",
    "wrist": "robot0_eye_in_hand",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "overhead": "overhead",
}


class RLBenchRemoteError(RuntimeError):
    """Raised when the remote RLBench adapter returns an error."""


class RLBenchRemoteEnv(BaseEnv):
    """Host-side low-level proxy for an RLBench server running in Docker.

    This class intentionally does not import ``rlbench`` or ``pyrep``. All
    simulator calls go through the adapter process started inside the Docker
    container.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8120",
        task_name: str = "reach_target",
        max_steps: int = 300,
        privileged: bool = True,
        enable_render: bool = False,
        viser_debug: bool = False,
        timeout: float = 120.0,
    ) -> None:
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.task_name = task_name
        self.max_steps = max_steps
        self.privileged = privileged
        self.enable_render = enable_render
        self.viser_debug = viser_debug
        self.timeout = timeout

        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._step_count = 0
        self._sim_step_count = 0
        self._episode_id: int | None = None
        self._action_sequence_id: int | None = None
        self._last_remote_obs: dict[str, Any] | None = None
        self._last_obs: dict[str, Any] | None = None
        self._last_control_result: dict[str, Any] | None = None
        self._last_reward = 0.0
        self._last_done = False

        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []

    # ----------------------------- HTTP helpers -----------------------------

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        update_context: bool = True,
    ) -> dict[str, Any]:
        url = f"{self.server_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._episode_id is not None:
            headers["X-RLBench-Episode-ID"] = str(self._episode_id)
            headers["X-RLBench-Task-Name"] = self.task_name
        if self._action_sequence_id is not None:
            headers["X-RLBench-Action-Sequence-ID"] = str(self._action_sequence_id)
        if method == "POST":
            headers["X-RLBench-Request-ID"] = uuid.uuid4().hex
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                raw_body = resp.read().decode("utf-8")
                try:
                    result = json.loads(raw_body)
                except json.JSONDecodeError as exc:
                    raise RLBenchRemoteError(
                        f"{method} {path} returned non-JSON response: {raw_body[:500]}"
                    ) from exc
                if update_context:
                    self._update_context_from_result(result)
                return result
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                error_payload = json.loads(body)
            except json.JSONDecodeError:
                error_payload = None
            if update_context and isinstance(error_payload, dict):
                self._update_context_from_result(error_payload)
            raise RLBenchRemoteError(f"{method} {path} failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RLBenchRemoteError(f"{method} {path} failed: {exc}") from exc
        except (TimeoutError, socket.timeout, OSError) as exc:
            raise RLBenchRemoteError(f"{method} {path} failed: {exc}") from exc

    def _get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, payload or {})

    @staticmethod
    def _decode_array(payload: dict[str, Any] | None) -> np.ndarray | None:
        if payload is None:
            return None
        encoding = payload.get("encoding")
        raw = base64.b64decode(payload["data"])
        if encoding == "base64":
            pass
        elif encoding == "base64_gzip":
            raw = gzip.decompress(raw)
        else:
            raise RLBenchRemoteError(f"Unsupported array encoding: {encoding}")
        return np.frombuffer(raw, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"]).copy()

    @staticmethod
    def _as_array(value: Any, shape: tuple[int, ...] | None = None, dtype: Any = np.float64) -> np.ndarray:
        arr = np.asarray(value, dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr

    def _update_context_from_result(self, result: dict[str, Any]) -> None:
        if "episode_id" in result:
            self._episode_id = int(result["episode_id"])
        if "action_sequence_id" in result:
            self._action_sequence_id = int(result["action_sequence_id"])
        observation = result.get("observation")
        if isinstance(observation, dict):
            if "episode_id" in observation:
                self._episode_id = int(observation["episode_id"])
            if "action_sequence_id" in observation:
                self._action_sequence_id = int(observation["action_sequence_id"])

    def _convert_observation(self, remote_obs: dict[str, Any]) -> dict[str, Any]:
        raw_cameras = dict(remote_obs.get("cameras") or {})
        if "front" not in raw_cameras:
            raw_cameras["front"] = {
                "rgb": remote_obs.get("rgb"),
                "depth": remote_obs.get("depth"),
                **(remote_obs.get("front_camera") or {}),
            }

        camera_obs: dict[str, Any] = {}
        for remote_name, capx_name in CAMERA_KEY_MAP.items():
            if remote_name not in raw_cameras:
                continue
            raw_camera = raw_cameras[remote_name] or {}
            rgb = self._decode_array(raw_camera.get("rgb"))
            depth = self._decode_array(raw_camera.get("depth"))
            if rgb is None:
                rgb = np.zeros((0, 0, 3), dtype=np.uint8)
            if depth is None:
                depth = np.zeros(rgb.shape[:2], dtype=np.float32)
            camera_obs[capx_name] = {
                "images": {
                    "rgb": rgb.astype(np.uint8, copy=False),
                    "depth": depth.astype(np.float32, copy=False),
                },
                "intrinsics": self._as_array(raw_camera.get("intrinsics", np.eye(3)), (3, 3)),
                "pose_mat": self._as_array(raw_camera.get("pose_mat", np.eye(4)), (4, 4)),
            }

        if "robot0_robotview" not in camera_obs:
            camera_obs["robot0_robotview"] = {
                "images": {
                    "rgb": np.zeros((0, 0, 3), dtype=np.uint8),
                    "depth": np.zeros((0, 0), dtype=np.float32),
                },
                "intrinsics": np.eye(3, dtype=np.float64),
                "pose_mat": np.eye(4, dtype=np.float64),
            }

        object_poses: dict[str, np.ndarray] = {}
        for name, pose in (remote_obs.get("object_poses") or {}).items():
            position = self._as_array(pose["position"], (3,))
            quat = self._as_array(pose["quaternion_wxyz"], (4,))
            object_poses[name] = np.concatenate([position, quat])

        gripper_pose = remote_obs.get("gripper_pose") or {}
        ee_pos = self._as_array(gripper_pose.get("position", [0.0, 0.0, 0.0]), (3,))
        ee_quat = self._as_array(gripper_pose.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]), (4,))
        gripper_open = float(remote_obs.get("gripper_open", 1.0))

        obs: dict[str, Any] = {
            "robot0_joint_pos": self._as_array(remote_obs.get("joint_positions", np.zeros(7)), (7,)),
            "robot0_joint_vel": self._as_array(remote_obs.get("joint_velocities", np.zeros(7)), (7,)),
            "robot0_gripper_qpos": np.array([gripper_open], dtype=np.float64),
            "robot0_eef_pos": ee_pos,
            "robot0_eef_quat": ee_quat,
            "object_poses": object_poses,
            "task_descriptions": list(remote_obs.get("task_descriptions") or []),
            "task_name": remote_obs.get("task_name", self.task_name),
            "episode_id": remote_obs.get("episode_id"),
            "action_sequence_id": remote_obs.get("action_sequence_id"),
            "task_low_dim_state": self._as_array(remote_obs.get("task_low_dim_state", [])),
            "last_reward": float(remote_obs.get("last_reward", self._last_reward)),
            "last_terminate": bool(remote_obs.get("last_terminate", False)),
            "success": bool(remote_obs.get("success", self._last_done)),
        }
        obs.update(camera_obs)
        if "target" in object_poses:
            obs["target_pose"] = object_poses["target"]
        if self._last_control_result is not None:
            obs["last_action_result"] = dict(self._last_control_result)
        return obs

    def _set_remote_observation(self, remote_obs: dict[str, Any]) -> dict[str, Any]:
        self._last_remote_obs = remote_obs
        self._update_context_from_result(remote_obs)
        self._last_obs = self._convert_observation(remote_obs)
        if self._record_frames:
            self._record_frame()
        return self._last_obs

    def _record_control_result(self, result: dict[str, Any], action_name: str) -> dict[str, Any]:
        control_result = {
            "action": action_name,
            "ok": bool(result.get("ok", True)),
            "error": result.get("error"),
            "error_type": result.get("error_type"),
            "steps": int(result.get("steps", 0)),
            "success": bool(result.get("success", False)),
            "terminate": bool(result.get("terminate", False)),
            "episode_id": result.get("episode_id", self._episode_id),
            "action_sequence_id": result.get("action_sequence_id", self._action_sequence_id),
            "request_id": result.get("request_id"),
        }
        self._last_control_result = control_result
        if not control_result["ok"]:
            message = (
                f"RLBench action failed: {action_name}: "
                f"{control_result['error_type'] or 'error'}: {control_result['error']}"
            )
            print(message, file=sys.stderr)
        return control_result

    # ------------------------------ BaseEnv API ------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", update_context=False)

    def switch_task(self, task_name: str) -> dict[str, Any]:
        result = self._post("/switch_task", {"task_name": task_name})
        self.task_name = str(result.get("task_name", task_name))
        if result.get("changed", False):
            self._last_remote_obs = None
            self._last_obs = None
            self._last_control_result = None
            self._last_reward = 0.0
            self._last_done = False
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        options = options or {}
        requested_task_name = str(options.get("task_name", self.task_name))
        payload: dict[str, Any] = {"task_name": requested_task_name}
        if "variation" in options:
            payload["variation"] = options["variation"]
        elif seed is not None:
            payload["variation"] = int(seed)
        if "attempts" in options:
            payload["attempts"] = int(options["attempts"])

        result = self._post("/reset", payload)
        self._step_count = 0
        self._sim_step_count = 0
        self._episode_id = int(result["episode_id"])
        self._action_sequence_id = int(result.get("action_sequence_id", 0))
        self.task_name = str(result.get("task_name", requested_task_name))
        self._last_control_result = None
        self._last_reward = 0.0
        self._last_done = False
        obs = self._set_remote_observation(result["observation"])
        info = {
            "episode_id": result.get("episode_id"),
            "task_name": result.get("task_name", self.task_name),
            "task_descriptions": result.get("descriptions", []),
            "action_sequence_id": result.get("action_sequence_id"),
        }
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        result = self._post("/step", {"action": np.asarray(action, dtype=float).tolist()})
        self._step_count += 1
        self._sim_step_count += 1
        self._last_reward = float(result.get("reward", 0.0))
        terminated = bool(result.get("terminate", result.get("success", False)))
        success = bool(result.get("success", False))
        self._last_done = success
        obs = self._set_remote_observation(result["observation"])
        truncated = self._step_count >= self.max_steps
        info = {
            "terminate": terminated,
            "success": success,
            "episode_id": result.get("episode_id", self._episode_id),
            "action_sequence_id": result.get("action_sequence_id", self._action_sequence_id),
        }
        return obs, self._last_reward, terminated, truncated, info

    def get_observation(self) -> dict[str, Any]:
        result = self._get("/observation")
        return self._set_remote_observation(result)

    def compute_reward(self) -> float:
        result = self._get("/reward")
        self._last_reward = float(result["reward"])
        return self._last_reward

    def task_completed(self) -> bool:
        result = self._get("/success")
        self._last_done = bool(result["success"])
        return self._last_done

    # ------------------------- Control helper methods ------------------------

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.01, max_steps: int = 200
    ) -> dict[str, Any]:
        result = self._post(
            "/move_to_joints",
            {
                "joints": np.asarray(joints, dtype=float).reshape(7).tolist(),
                "tolerance": float(tolerance),
                "steps": int(max_steps),
            },
        )
        self._sim_step_count += int(result.get("steps", 0))
        self._last_done = bool(result.get("success", False))
        control_result = self._record_control_result(result, "move_to_joints")
        self._set_remote_observation(result["observation"])
        return control_result

    def move_to_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        ignore_collisions: bool = True,
    ) -> dict[str, Any]:
        result = self._post(
            "/move_to_pose",
            {
                "position": np.asarray(position, dtype=float).reshape(3).tolist(),
                "quaternion_wxyz": np.asarray(quaternion_wxyz, dtype=float).reshape(4).tolist(),
                "ignore_collisions": bool(ignore_collisions),
            },
        )
        self._sim_step_count += int(result.get("steps", 0))
        self._last_done = bool(result.get("success", False))
        control_result = self._record_control_result(result, "move_to_pose")
        if "observation" in result:
            self._set_remote_observation(result["observation"])
        return control_result

    def _set_gripper(self, fraction: float) -> dict[str, Any]:
        path = "/open_gripper" if float(fraction) > 0.5 else "/close_gripper"
        result = self._post(path)
        self._sim_step_count += int(result.get("steps", 0))
        control_result = self._record_control_result(result, path.lstrip("/"))
        self._set_remote_observation(result["observation"])
        return control_result

    def open_gripper(self) -> dict[str, Any]:
        return self._set_gripper(1.0)

    def close_gripper(self) -> dict[str, Any]:
        return self._set_gripper(0.0)

    # ----------------------------- Video/render ------------------------------

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        obs = self._last_obs if self._last_obs is not None else self.get_observation()
        return obs["robot0_robotview"]["images"]["rgb"]

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled and self._last_obs is not None:
            self._record_frame()

    def _record_frame(self) -> None:
        if self._last_obs is None:
            return
        self._frame_buffer.append(self._last_obs["robot0_robotview"]["images"]["rgb"].copy())

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        return []

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return []


__all__ = ["RLBenchRemoteEnv", "RLBenchRemoteError"]
