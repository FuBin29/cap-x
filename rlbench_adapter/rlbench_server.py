#!/usr/bin/env python3
"""Minimal stdlib HTTP RPC server for RLBench running inside Docker."""

from __future__ import annotations

import argparse
import base64
import gzip
import importlib
import json
import re
import queue
import sys
import threading
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from pyrep.const import ConfigurationPathAlgorithms as Algos, RenderMode
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects.object import Object

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, JointPosition, JointVelocity
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


DEFAULT_PATH_VIDEO_OUTPUT_ROOT = "/home/fubin/projects/artance/cap-x/outputs/rlbench_path_videos"
ARTANCE_CONTROL_MODE = "artance_local_path_helper"
ARTANCE_PATH_PLANNER = {
    "trials": 100,
    "max_configs": 10,
    "max_time_ms": 10,
    "trials_per_goal": 5,
    "algorithm": "RRTConnect",
}


class InvalidPoseError(ValueError):
    """Invalid end-effector pose request."""


class InvalidQuaternionError(InvalidPoseError):
    """End-effector pose request has a non-unit quaternion."""


class WorkspaceBoundaryError(InvalidPoseError):
    """End-effector pose request is outside the RLBench task workspace."""



def _camel_to_snake(name: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _rlbench_task_registry() -> Dict[str, type]:
    import rlbench.tasks as task_exports

    registry: Dict[str, type] = {}
    for class_name, value in vars(task_exports).items():
        if not isinstance(value, type):
            continue
        module_name = getattr(value, "__module__", "")
        if not module_name.startswith("rlbench.tasks."):
            continue
        snake_name = _camel_to_snake(class_name)
        registry[snake_name] = value
        registry[class_name] = value
        registry[class_name.lower()] = value
    return registry


def _normalize_task_name(task_name: str) -> str:
    return task_name.strip().lower().replace("-", "_").replace(" ", "_")


def _task_class(task_name: str):
    normalized = _normalize_task_name(task_name)
    registry = _rlbench_task_registry()
    if normalized in registry:
        return registry[normalized]
    class_name = "".join(part.capitalize() for part in normalized.split("_"))
    if class_name in registry:
        return registry[class_name]
    module = importlib.import_module(f"rlbench.tasks.{normalized}")
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


def _render_mode(name: str) -> RenderMode:
    normalized = str(name).strip().upper()
    aliases = {
        "OPENGL": "OPENGL",
        "OPENGL3": "OPENGL3",
        "OPEN_GL3": "OPENGL3",
    }
    return getattr(RenderMode, aliases.get(normalized, normalized))


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
        dataset_root: Optional[str] = None,
        arm_action_mode: str = "joint_position",
        render_mode: str = "opengl3",
        array_encoding: str = "base64",
        record_path_video: bool = True,
        path_video_output_root: str = DEFAULT_PATH_VIDEO_OUTPUT_ROOT,
        path_video_camera: str = "wrist",
        path_video_fps: float = 10.0,
    ):
        self.task_name = _normalize_task_name(task_name)
        self.image_size = image_size
        self.headless = headless
        self.cameras = cameras
        self.object_names = object_names
        self.dataset_root = str(Path(dataset_root).expanduser()) if dataset_root else None
        self.arm_action_mode = arm_action_mode
        self.render_mode = render_mode
        self.array_encoding = array_encoding
        self.record_path_video = bool(record_path_video)
        self.path_video_output_root = Path(path_video_output_root).expanduser()
        self.path_video_camera = path_video_camera
        self.path_video_fps = float(path_video_fps)
        self.path_video_dir: Optional[Path] = None
        self.path_video_manifest: list[Dict[str, Any]] = []
        self._path_video_counter = 0
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
        self.reset_mode = None
        self.variation = None
        self.episode_number = None
        self.frame_index = None
        self.replay_info: Dict[str, Any] = {}
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
        render_mode = _render_mode(self.render_mode)
        for camera_name in self.cameras:
            config_attr, _, _ = CAMERA_SPECS[camera_name]
            camera_config = getattr(obs_config, config_attr)
            camera_config.rgb = True
            camera_config.depth = True
            camera_config.point_cloud = False
            camera_config.mask = False
            camera_config.image_size = (self.image_size, self.image_size)
            camera_config.render_mode = render_mode
            camera_config.depth_in_meters = True
        obs_config.joint_positions = True
        obs_config.joint_velocities = True
        obs_config.gripper_open = True
        obs_config.gripper_pose = True
        obs_config.gripper_matrix = True
        obs_config.task_low_dim_state = True

        env_kwargs: Dict[str, Any] = {
            "action_mode": MoveArmThenGripper(self._make_arm_action_mode(), Discrete()),
            "obs_config": obs_config,
            "headless": self.headless,
            "shaped_rewards": False,
        }
        if self.dataset_root:
            env_kwargs["dataset_root"] = self.dataset_root
        self.env = Environment(**env_kwargs)
        self.env.launch()
        self._load_task(self.task_name)

    def _arm_action_mode_key(self) -> str:
        mode = str(self.arm_action_mode).strip().lower().replace("-", "_")
        aliases = {
            "velocity": "joint_velocity",
            "joint_velocities": "joint_velocity",
            "position": "joint_position",
            "joint_positions": "joint_position",
            "ee_pose": "ee_pose_via_planning",
            "end_effector_pose": "ee_pose_via_planning",
            "end_effector_pose_via_planning": "ee_pose_via_planning",
        }
        return aliases.get(mode, mode)

    def _make_arm_action_mode(self) -> Any:
        mode = self._arm_action_mode_key()
        if mode == "joint_velocity":
            return JointVelocity()
        if mode == "joint_position":
            return JointPosition(True)
        if mode == "ee_pose_via_planning":
            return EndEffectorPoseViaPlanning()
        raise ValueError(
            "Unsupported arm action mode "
            f"{self.arm_action_mode!r}; use 'joint_velocity', 'joint_position', "
            "or 'ee_pose_via_planning'."
        )

    def _step_action_spec(self) -> Dict[str, Any]:
        mode = self._arm_action_mode_key()
        if mode == "joint_velocity":
            return {
                "arm_action_mode": mode,
                "shape": [8],
                "description": "[7 Franka joint velocities, 1 discrete gripper command]",
            }
        if mode == "joint_position":
            return {
                "arm_action_mode": mode,
                "shape": [8],
                "description": (
                    "[7 Franka joint target positions, 1 discrete gripper command]; "
                    "this is used for stored-demo reset replay."
                ),
            }
        if mode == "ee_pose_via_planning":
            return {
                "arm_action_mode": mode,
                "shape": [8],
                "description": (
                    "[x, y, z, qx, qy, qz, qw, gripper command] in "
                    "RLBench/PyRep XYZW order; use only for native RLBench pose-step."
                ),
            }
        return {
            "arm_action_mode": mode,
            "shape": [8],
            "description": "Unknown 8D action; check server arm_action_mode.",
        }

    def _pose_helper_spec(self) -> Dict[str, Any]:
        return {
            "control_mode": ARTANCE_CONTROL_MODE,
            "description": (
                "ArtAnce local-style helper: plan an end-effector path with "
                "PyRep arm.get_path(), then execute path.step(); scene.step(). "
                "Gripper commands are explicit open_gripper()/close_gripper() calls."
            ),
            "pose_format": {
                "http_move_to_pose": "position XYZ plus quaternion_wxyz",
                "pyrep_get_path": "position XYZ plus quaternion_xyzw",
            },
            "planner": dict(ARTANCE_PATH_PLANNER),
        }

    def _safe_reward_value(self) -> Optional[float]:
        if self.task is None:
            return None
        try:
            reward = self.task._task.reward()
        except Exception:
            return None
        return None if reward is None else float(reward)

    def _validate_pose_target(self, scene: Any, position: np.ndarray, quaternion_xyzw: np.ndarray) -> None:
        if not np.all(np.isfinite(position)):
            raise InvalidPoseError(f"Target position contains non-finite values: {position.tolist()}")
        if not np.all(np.isfinite(quaternion_xyzw)):
            raise InvalidQuaternionError(
                f"Target quaternion contains non-finite values: {quaternion_xyzw.tolist()}"
            )
        quat_norm = float(np.linalg.norm(quaternion_xyzw))
        if not np.isclose(quat_norm, 1.0):
            raise InvalidQuaternionError(
                "Action contained non unit quaternion. "
                f"Expected norm close to 1.0, got {quat_norm:.6f}."
            )
        if not scene.check_target_in_workspace(position.tolist()):
            raise WorkspaceBoundaryError(
                "A path could not be found because the target is outside of workspace."
            )

    def _log_control_error(
        self,
        action_name: str,
        error: BaseException,
        context: Optional[Dict[str, Any]] = None,
        error_traceback: Optional[str] = None,
    ) -> None:
        payload = {
            "action": action_name,
            "error_type": error.__class__.__name__,
            "error": str(error),
            "task_name": self.task_name,
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
            "context": context or {},
        }
        sys.stderr.write("RLBench adapter control error:\n")
        sys.stderr.write(json.dumps(payload, indent=2, default=_json_default) + "\n")
        if error_traceback:
            sys.stderr.write(error_traceback + "\n")

    def _clear_episode_state(self) -> None:
        self.descriptions = []
        self.obs = None
        self.last_reward = 0.0
        self.last_terminate = False
        self.reset_mode = None
        self.variation = None
        self.episode_number = None
        self.frame_index = None
        self.replay_info = {}
        self.path_video_dir = None
        self.path_video_manifest = []
        self._path_video_counter = 0

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
            "dataset_root": self.dataset_root,
            "arm_action_mode": self.arm_action_mode,
            "render_mode": self.render_mode,
            "cameras": list(self.cameras),
            "default_control_mode": ARTANCE_CONTROL_MODE,
            "pose_helper_spec": self._pose_helper_spec(),
            "step_action_spec": self._step_action_spec(),
            "record_path_video": self.record_path_video,
            "path_video_output_root": str(self.path_video_output_root),
            "path_video_camera": self.path_video_camera,
            "path_video_dir": str(self.path_video_dir) if self.path_video_dir else None,
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

    def _episode_video_dir_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        task = _normalize_task_name(self.task_name)
        variation = "varNA" if self.variation is None else f"var{self.variation}"
        episode = "episodeNA" if self.episode_number is None else f"episode{self.episode_number}"
        frame = "frameNA" if self.frame_index is None else f"frame{self.frame_index}"
        return f"{timestamp}-{task}-{variation}-{episode}-{frame}"

    def _init_path_video_dir(self) -> None:
        if not self.record_path_video:
            self.path_video_dir = None
            self.path_video_manifest = []
            return
        self.path_video_output_root.mkdir(parents=True, exist_ok=True)
        self.path_video_dir = self.path_video_output_root / self._episode_video_dir_name()
        self.path_video_dir.mkdir(parents=True, exist_ok=True)
        self.path_video_manifest = []
        self._path_video_counter = 0

    @staticmethod
    def _rgb_to_uint8(image: Any) -> np.ndarray:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        return array

    def _capture_path_video_frame(self, scene: Any, frames: list[np.ndarray]) -> None:
        obs = scene.get_observation()
        camera = self.path_video_camera
        rgb = getattr(obs, f"{camera}_rgb", None)
        if rgb is None and camera != "front":
            rgb = getattr(obs, "front_rgb", None)
        if rgb is None:
            return
        frames.append(self._rgb_to_uint8(rgb).copy())

    def _write_path_video(
        self,
        frames: list[np.ndarray],
        *,
        action_name: str,
        ok: bool,
        error_type: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.record_path_video or self.path_video_dir is None:
            return None
        self._path_video_counter += 1
        stem = f"{self._path_video_counter:03d}_{action_name}"
        video_path = self.path_video_dir / f"{stem}.mp4"
        entry: Dict[str, Any] = {
            "action": action_name,
            "ok": bool(ok),
            "task_name": self.task_name,
            "reset_mode": self.reset_mode,
            "variation": self.variation,
            "episode_number": self.episode_number,
            "frame_index": self.frame_index,
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
            "control_mode": ARTANCE_CONTROL_MODE,
            "camera": self.path_video_camera,
            "path": str(video_path),
            "frames": len(frames),
            "fps": self.path_video_fps,
            "error_type": error_type,
            "error": error,
        }
        if metadata:
            entry.update(metadata)
        if not frames:
            entry.update({"ok": False, "video_error": "No path-step frames captured."})
            self.path_video_manifest.append(entry)
            self._write_path_video_manifest()
            return entry
        try:
            import cv2

            first = np.asarray(frames[0], dtype=np.uint8)
            height, width = first.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.path_video_fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open")
            try:
                for frame in frames:
                    rgb = np.asarray(frame, dtype=np.uint8)
                    if rgb.shape[:2] != (height, width):
                        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            entry["frame_size"] = [width, height]
        except Exception as exc:
            entry.update({
                "ok": False,
                "video_error": str(exc),
                "video_error_type": exc.__class__.__name__,
            })
        self.path_video_manifest.append(entry)
        self._write_path_video_manifest()
        return entry

    def _write_path_video_manifest(self) -> None:
        if self.path_video_dir is None:
            return
        manifest_path = self.path_video_dir / "manifest.json"
        payload = {
            "manifest_version": 2,
            "control_mode": ARTANCE_CONTROL_MODE,
            "pose_helper_spec": self._pose_helper_spec(),
            "task_name": self.task_name,
            "reset_mode": self.reset_mode,
            "variation": self.variation,
            "episode_number": self.episode_number,
            "frame_index": self.frame_index,
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
            "camera": self.path_video_camera,
            "videos": self.path_video_manifest,
        }
        manifest_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    def reset(
        self,
        variation: Optional[int] = None,
        attempts: int = 5,
        task_name: Optional[str] = None,
        reset_mode: str = "default",
        episode_number: Optional[int] = None,
        frame_index: Optional[int] = None,
        live_demos: bool = False,
        random_selection: bool = False,
        image_paths: bool = False,
        replay_action_key: str = "joint_position_action",
    ) -> Dict[str, Any]:
        if task_name is not None:
            normalized = _normalize_task_name(task_name)
            if normalized != self.task_name or self.task is None:
                self._load_task(normalized)
        if self.task is None:
            raise RuntimeError("RLBench task is not initialized")
        last_error = None
        for _ in range(max(1, int(attempts))):
            try:
                if str(reset_mode).lower() in {"demo", "episode", "reset_to_demo"}:
                    self._reset_to_demo(
                        variation=variation,
                        episode_number=episode_number,
                        frame_index=frame_index,
                        live_demos=live_demos,
                        random_selection=random_selection,
                        image_paths=image_paths,
                        replay_action_key=replay_action_key,
                    )
                else:
                    if variation is not None:
                        self.task.set_variation(int(variation))
                    else:
                        self.task.sample_variation()
                    self.descriptions, self.obs = self.task.reset()
                    self.reset_mode = "default"
                    self.variation = variation
                    self.episode_number = None
                    self.frame_index = None
                    self.replay_info = {}
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        self.last_reward = 0.0 if self.reset_mode != "demo" else self.last_reward
        self.last_terminate = False if self.reset_mode != "demo" else self.last_terminate
        self.episode_id += 1
        self._mark_mutated()
        self._init_path_video_dir()
        return {
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
            "task_name": self.task_name,
            "descriptions": self.descriptions,
            "reset_mode": self.reset_mode,
            "variation": self.variation,
            "episode_number": self.episode_number,
            "frame_index": self.frame_index,
            "replay": self.replay_info,
            "path_video_dir": str(self.path_video_dir) if self.path_video_dir else None,
            "observation": self.observation(),
        }

    def _reset_to_demo(
        self,
        *,
        variation: Optional[int],
        episode_number: Optional[int],
        frame_index: Optional[int],
        live_demos: bool,
        random_selection: bool,
        image_paths: bool,
        replay_action_key: str,
    ) -> None:
        if self.task is None:
            raise RuntimeError("RLBench task is not initialized")
        if episode_number is None and not random_selection:
            raise RPCError(
                400,
                {
                    "error": "Demo reset requires episode_number unless random_selection is true.",
                    "task_name": self.task_name,
                    "episode_id": self.episode_id,
                    "action_sequence_id": self.action_sequence_id,
                },
            )

        mode = self._arm_action_mode_key()
        if replay_action_key == "joint_position_action" and mode != "joint_position":
            raise RPCError(
                409,
                {
                    "error": (
                        "Demo replay with joint_position_action requires the server to be "
                        "started with --arm-action-mode joint_position."
                    ),
                    "task_name": self.task_name,
                    "episode_id": self.episode_id,
                    "action_sequence_id": self.action_sequence_id,
                    "arm_action_mode": self.arm_action_mode,
                    "replay_action_key": replay_action_key,
                },
            )

        if variation is not None:
            self.task.set_variation(int(variation))
        else:
            self.task.set_variation(-1)

        demos = self.task.get_demos(
            1,
            live_demos=bool(live_demos),
            image_paths=bool(image_paths),
            random_selection=bool(random_selection),
            from_episode_number=None if episode_number is None else int(episode_number),
        )
        demo = demos[0]
        demo_variation = getattr(demo, "variation_number", variation)
        if demo_variation is not None:
            self.task.set_variation(int(demo_variation))
        self.descriptions, self.obs = self.task.reset_to_demo(demo)
        replay: Dict[str, Any] = {
            "replayed_actions": 0,
            "last_reward": None,
            "last_terminate": None,
            "action_key": replay_action_key,
        }
        self.last_reward = 0.0
        self.last_terminate = False
        if frame_index is not None:
            target_frame = int(frame_index)
            if target_frame < 0 or target_frame >= len(demo):
                raise IndexError(
                    f"frame_index={target_frame} out of range for demo length {len(demo)}"
                )
            for step_index in range(1, target_frame + 1):
                action = getattr(demo[step_index], "misc", {}).get(replay_action_key)
                if action is None:
                    raise RuntimeError(
                        f"Demo step {step_index} does not contain {replay_action_key!r}."
                    )
                self.obs, reward, terminate = self.task.step(action)
                replay = {
                    "replayed_actions": step_index,
                    "last_reward": float(reward),
                    "last_terminate": bool(terminate),
                    "action_key": replay_action_key,
                }
            self.last_reward = float(replay["last_reward"] or 0.0)
            self.last_terminate = bool(replay["last_terminate"])

        self.reset_mode = "demo"
        self.variation = int(demo_variation) if demo_variation is not None else variation
        self.episode_number = episode_number
        self.frame_index = frame_index
        self.replay_info = replay

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
        rlbench_raw: Dict[str, Any] = {}
        camera_configs: Dict[str, Any] = {}
        for camera_name in self.cameras:
            config_attr, rgb_attr, depth_attr = CAMERA_SPECS[camera_name]
            rgb_payload = _array_payload(getattr(obs, rgb_attr), self.array_encoding)
            depth_payload = _array_payload(getattr(obs, depth_attr), self.array_encoding)
            intrinsics = _to_list(misc.get(f"{config_attr}_intrinsics"))
            extrinsics = _to_list(misc.get(f"{config_attr}_extrinsics"))
            cameras[camera_name] = {
                "rgb": rgb_payload,
                "depth": depth_payload,
                "intrinsics": intrinsics,
                "pose_mat": extrinsics,
            }
            rlbench_raw[f"{camera_name}_rgb"] = rgb_payload
            rlbench_raw[f"{camera_name}_depth"] = depth_payload
            camera_configs[config_attr] = {
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
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
        gripper_pose_xyzw = None
        if obs.gripper_pose is not None:
            pose = np.asarray(obs.gripper_pose, dtype=float).reshape(7)
            gripper_pose_xyzw = pose.tolist()
            gripper_pose = {
                "position": pose[:3].tolist(),
                "quaternion_wxyz": _xyzw_to_wxyz(pose[3:7]),
                "quaternion_xyzw": pose[3:7].tolist(),
            }

        rlbench_raw.update(
            {
                "joint_positions": _to_list(obs.joint_positions),
                "joint_velocities": _to_list(obs.joint_velocities),
                "gripper_open": obs.gripper_open,
                "gripper_pose": gripper_pose_xyzw,
                "gripper_matrix": _to_list(obs.gripper_matrix),
                "task_low_dim_state": _to_list(obs.task_low_dim_state),
                "misc": {
                    key: _to_list(value) if isinstance(value, np.ndarray) else value
                    for key, value in misc.items()
                    if not key.endswith("_rgb") and not key.endswith("_depth")
                },
            }
        )

        front_camera = cameras.get("front") or {}
        return {
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
            "task_name": self.task_name,
            "task_descriptions": self.descriptions,
            "reset_mode": self.reset_mode,
            "variation": self.variation,
            "episode_number": self.episode_number,
            "frame_index": self.frame_index,
            "replay": self.replay_info,
            "path_video_dir": str(self.path_video_dir) if self.path_video_dir else None,
            "path_video_manifest": list(self.path_video_manifest),
            "step_action_spec": self._step_action_spec(),
            "pose_helper_spec": self._pose_helper_spec(),
            "cameras": cameras,
            "rgb": front_camera.get("rgb"),
            "depth": front_camera.get("depth"),
            "front_camera": {
                "intrinsics": front_camera.get("intrinsics"),
                "pose_mat": front_camera.get("pose_mat"),
            },
            "rlbench_raw": rlbench_raw,
            "rlbench_camera_configs": camera_configs,
            "joint_positions": _to_list(obs.joint_positions),
            "joint_velocities": _to_list(obs.joint_velocities),
            "gripper_open": obs.gripper_open,
            "gripper_pose": gripper_pose,
            "gripper_matrix": _to_list(obs.gripper_matrix),
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
            spec = self._step_action_spec()
            raise RPCError(
                400,
                {
                    "error": f"POST /step requires action shape {spec['shape']}: {spec['description']}.",
                    "arm_action_mode": spec["arm_action_mode"],
                    "step_action_spec": spec,
                    "episode_id": self.episode_id,
                    "action_sequence_id": self.action_sequence_id,
                },
            ) from exc
        context = {
            "action": action_array.tolist(),
            "step_action_spec": self._step_action_spec(),
        }
        try:
            self.obs, self.last_reward, self.last_terminate = self.task.step(action_array)
        except Exception as exc:
            error_traceback = traceback.format_exc()
            self._log_control_error("step", exc, context, error_traceback)
            try:
                self.obs = self.task._scene.get_observation()
            except Exception:
                pass
            return {
                "ok": False,
                "action": "step",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "error_context": context,
                "traceback": error_traceback,
                "observation": self.observation() if self.obs is not None else None,
                "reward": float(self.last_reward),
                "terminate": bool(self.last_terminate),
                "success": self.success()["success"] if self.obs is not None else False,
                "episode_id": self.episode_id,
                "action_sequence_id": self.action_sequence_id,
                "step_action_spec": self._step_action_spec(),
            }
        self._mark_mutated()
        return {
            "ok": True,
            "action": "step",
            "observation": self.observation(),
            "reward": float(self.last_reward),
            "terminate": bool(self.last_terminate),
            "success": self.success()["success"],
            "episode_id": self.episode_id,
            "action_sequence_id": self.action_sequence_id,
            "step_action_spec": self._step_action_spec(),
        }

    def _control_response(
        self,
        *,
        ok: bool,
        steps: int,
        action_name: Optional[str] = None,
        error_type: Optional[str] = None,
        error: Optional[str] = None,
        prefix: Optional[str] = None,
        error_context: Optional[Dict[str, Any]] = None,
        error_traceback: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.task is None:
            raise RuntimeError("RLBench task is not initialized")
        if self.obs is None:
            self.obs = self.task._scene.get_observation()
        success = self.success()
        self.last_terminate = success["terminate"]
        payload: Dict[str, Any] = {
            "ok": ok,
            "action": action_name,
            "control_mode": ARTANCE_CONTROL_MODE,
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
            payload["error_context"] = error_context or {}
            if error_traceback:
                payload["traceback"] = error_traceback
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
        error_traceback: Optional[str] = None
        context = {
            "target_joints": target.tolist(),
            "max_steps": int(steps),
            "tolerance": float(tolerance),
        }
        try:
            for count in range(1, int(steps) + 1):
                scene.step()
                current = np.asarray(arm.get_joint_positions(), dtype=float)
                if float(np.max(np.abs(current - target))) <= float(tolerance):
                    break
        except Exception as exc:
            error = exc
            error_traceback = traceback.format_exc()
        finally:
            arm.set_control_loop_enabled(previous_control)
        self.obs = scene.get_observation()
        self._mark_mutated()
        if error is not None:
            self._log_control_error("move_to_joints", error, context, error_traceback)
            return self._control_response(
                ok=False,
                steps=count,
                action_name="move_to_joints",
                error_type=error.__class__.__name__,
                error=str(error),
                prefix="Could not move to joint positions",
                error_context=context,
                error_traceback=error_traceback,
            )
        payload = self._control_response(ok=True, steps=count, action_name="move_to_joints")
        payload["joint_positions"] = _to_list(self.obs.joint_positions)
        return payload

    def move_to_pose(
        self,
        position: Any,
        quaternion_wxyz: Any,
        ignore_collisions: bool = True,
        steps: int = 600,
    ) -> Dict[str, Any]:
        self._require_obs()
        scene = self.task._scene
        arm = scene.robot.arm
        previous_control = arm.joints[0].is_control_loop_enabled()
        try:
            target_position_array = np.asarray(position, dtype=float).reshape(3)
            target_quaternion_wxyz_array = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
            target_quaternion_xyzw_array = np.asarray(
                _wxyz_to_xyzw(target_quaternion_wxyz_array), dtype=float
            ).reshape(4)
        except (TypeError, ValueError) as exc:
            raise RPCError(
                400,
                {
                    "error": "POST /move_to_pose requires position shape (3,) and quaternion_wxyz shape (4,).",
                    "episode_id": self.episode_id,
                    "action_sequence_id": self.action_sequence_id,
                },
            ) from exc

        target_position = target_position_array.tolist()
        target_quaternion_xyzw = target_quaternion_xyzw_array.tolist()
        target_pose_wxyz = target_position + target_quaternion_wxyz_array.tolist()
        target_pose_xyzw = target_position + target_quaternion_xyzw
        max_steps = max(1, int(steps))
        success_before = self.success()
        count = 0
        error: Optional[Exception] = None
        error_traceback: Optional[str] = None
        video_frames: list[np.ndarray] = []
        context: Dict[str, Any] = {
            "target_position": target_position,
            "target_quaternion_wxyz": target_quaternion_wxyz_array.tolist(),
            "target_pose_xyzw": target_pose_xyzw,
            "ignore_collisions": bool(ignore_collisions),
            "max_path_steps": max_steps,
            "planner": dict(ARTANCE_PATH_PLANNER),
            "success_before": success_before,
        }
        arm.set_control_loop_enabled(True)
        try:
            self._validate_pose_target(scene, target_position_array, target_quaternion_xyzw_array)
            path = arm.get_path(
                target_position,
                quaternion=target_quaternion_xyzw,
                ignore_collisions=bool(ignore_collisions),
                trials=ARTANCE_PATH_PLANNER["trials"],
                max_configs=ARTANCE_PATH_PLANNER["max_configs"],
                max_time_ms=ARTANCE_PATH_PLANNER["max_time_ms"],
                trials_per_goal=ARTANCE_PATH_PLANNER["trials_per_goal"],
                algorithm=Algos.RRTConnect,
            )
            done = False
            while not done:
                done = path.step()
                scene.step()
                count += 1
                self._capture_path_video_frame(scene, video_frames)
                if self.success()["success"]:
                    break
                if count >= max_steps and not done:
                    raise TimeoutError(f"Path step limit reached: {max_steps}")
        except (InvalidPoseError, ConfigurationPathError, IKError, RuntimeError, TimeoutError) as exc:
            error = exc
            error_traceback = traceback.format_exc()
        finally:
            arm.set_control_loop_enabled(previous_control)
        self.obs = scene.get_observation()
        self._mark_mutated()
        success_after = self.success()
        reward_after = self._safe_reward_value()
        metadata = {
            "target_pose_wxyz": target_pose_wxyz,
            "target_pose_xyzw": target_pose_xyzw,
            "ignore_collisions": bool(ignore_collisions),
            "planner": dict(ARTANCE_PATH_PLANNER),
            "steps": count,
            "max_path_steps": max_steps,
            "success_before": success_before,
            "success_after": success_after,
            "reward_after": reward_after,
            "terminate_after": success_after["terminate"],
        }
        if error is not None:
            metadata["error_context"] = context
        video_result = self._write_path_video(
            video_frames,
            action_name="move_to_pose",
            ok=error is None,
            error_type=None if error is None else error.__class__.__name__,
            error=None if error is None else str(error),
            metadata=metadata,
        )
        if error is not None:
            self._log_control_error("move_to_pose", error, context, error_traceback)
            payload = self._control_response(
                ok=False,
                steps=count,
                action_name="move_to_pose",
                error_type=error.__class__.__name__,
                error=str(error),
                prefix="Could not plan path to pose",
                error_context=context,
                error_traceback=error_traceback,
            )
            if video_result is not None:
                payload["path_video"] = video_result
            return payload
        payload = self._control_response(ok=True, steps=count, action_name="move_to_pose")
        payload.update({
            "target_pose_wxyz": target_pose_wxyz,
            "target_pose_xyzw": target_pose_xyzw,
            "planner": dict(ARTANCE_PATH_PLANNER),
        })
        if video_result is not None:
            payload["path_video"] = video_result
        return payload

    def set_gripper(
        self,
        open_amount: float,
        velocity: float = 0.2,
        steps: int = 40,
    ) -> Dict[str, Any]:
        self._require_obs()
        scene = self.task._scene
        target = 1.0 if float(open_amount) > 0.5 else 0.0
        action_name = "open_gripper" if target > 0.5 else "close_gripper"
        max_steps = max(1, int(steps))
        success_before = self.success()
        count = 0
        error: Optional[Exception] = None
        error_traceback: Optional[str] = None
        video_frames: list[np.ndarray] = []
        context = {
            "target_gripper_open": target,
            "velocity": float(velocity),
            "max_steps": max_steps,
            "success_before": success_before,
        }
        try:
            for count in range(1, max_steps + 1):
                done = scene.robot.gripper.actuate(target, velocity=float(velocity))
                scene.step()
                self._capture_path_video_frame(scene, video_frames)
                if done:
                    break
        except Exception as exc:
            error = exc
            error_traceback = traceback.format_exc()
        self.obs = scene.get_observation()
        self._mark_mutated()
        success_after = self.success()
        reward_after = self._safe_reward_value()
        metadata = {
            "target_gripper_open": target,
            "velocity": float(velocity),
            "steps": count,
            "max_steps": max_steps,
            "gripper_open_after": self.obs.gripper_open,
            "success_before": success_before,
            "success_after": success_after,
            "reward_after": reward_after,
            "terminate_after": success_after["terminate"],
        }
        if error is not None:
            metadata["error_context"] = context
        video_result = self._write_path_video(
            video_frames,
            action_name=action_name,
            ok=error is None,
            error_type=None if error is None else error.__class__.__name__,
            error=None if error is None else str(error),
            metadata=metadata,
        )
        if error is not None:
            self._log_control_error(action_name, error, context, error_traceback)
            payload = self._control_response(
                ok=False,
                steps=count,
                action_name=action_name,
                error_type=error.__class__.__name__,
                error=str(error),
                prefix="Could not actuate gripper",
                error_context=context,
                error_traceback=error_traceback,
            )
            if video_result is not None:
                payload["path_video"] = video_result
            return payload
        payload = self._control_response(ok=True, steps=count, action_name=action_name)
        payload["gripper_open"] = self.obs.gripper_open
        if video_result is not None:
            payload["path_video"] = video_result
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
                        body.get("reset_mode", "default"),
                        body.get("episode_number"),
                        body.get("frame_index", body.get("frame")),
                        body.get("live_demos", False),
                        body.get("random_selection", False),
                        body.get("image_paths", False),
                        body.get("replay_action_key", "joint_position_action"),
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
                        body.get("steps", body.get("max_path_steps", 600)),
                    )
                elif method == "POST" and path == "/open_gripper":
                    payload = self.service.set_gripper(
                        1.0,
                        body.get("velocity", 0.2),
                        body.get("steps", body.get("max_steps", 40)),
                    )
                elif method == "POST" and path == "/close_gripper":
                    payload = self.service.set_gripper(
                        0.0,
                        body.get("velocity", 0.2),
                        body.get("steps", body.get("max_steps", 40)),
                    )
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
            error_traceback = traceback.format_exc()
            sys.stderr.write("RLBench adapter unhandled request error:\n")
            sys.stderr.write(error_traceback + "\n")
            self._send(
                500,
                {
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "traceback": error_traceback,
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
        "--dataset-root",
        help="Optional RLBench dataset root. Required for reset_mode=demo stored episodes.",
    )
    parser.add_argument(
        "--arm-action-mode",
        choices=("joint_velocity", "joint_position", "ee_pose_via_planning"),
        default="joint_position",
        help=(
            "RLBench arm action mode. The default joint_position matches "
            "ArtAnce stored-demo replay with joint_position_action."
        ),
    )
    parser.add_argument(
        "--render-mode",
        default="opengl3",
        help="PyRep render mode for enabled cameras, usually opengl3.",
    )
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
    parser.add_argument(
        "--path-video-output-root",
        default=DEFAULT_PATH_VIDEO_OUTPUT_ROOT,
        help="Directory root for per-path-step videos and manifests.",
    )
    parser.add_argument(
        "--path-video-camera",
        default="wrist",
        help="RLBench camera used for path-step videos; falls back to front if unavailable.",
    )
    parser.add_argument("--path-video-fps", type=float, default=10.0)
    parser.add_argument("--no-record-path-video", action="store_true")
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
        dataset_root=args.dataset_root,
        arm_action_mode=args.arm_action_mode,
        render_mode=args.render_mode,
        array_encoding=args.array_encoding,
        record_path_video=not args.no_record_path_video,
        path_video_output_root=args.path_video_output_root,
        path_video_camera=args.path_video_camera,
        path_video_fps=args.path_video_fps,
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
