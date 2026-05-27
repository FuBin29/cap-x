from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PrismaticJointMotionConfig:
    plane_summary: Path
    output_dir: Path
    frame_info: Path | None = None
    camera_name: str = "wrist"
    input_frame: str = "auto"
    orientation_mode: str = "keep-current"
    normal_direction_sign: float = -1.0
    approach_distance: float = 0.08
    target_standoff: float = 0.015
    travel_distance: float = 0.08
    close_gripper_before_motion: bool = True
    target_key: str = "contact_center"
    normal_key: str = "plane_normal"


@dataclass(frozen=True)
class PrismaticJointMotionTarget:
    input_frame: str
    output_frame: str
    contact_position: list[float]
    normal: list[float]
    motion_direction: list[float]
    approach_position: list[float]
    target_position: list[float]
    final_position: list[float]
    quaternion_wxyz: list[float]
    normal_direction_sign: float
    approach_distance: float
    target_standoff: float
    travel_distance: float
    close_gripper_before_motion: bool
    current_gripper_position: list[float] | None
    current_gripper_quaternion_wxyz: list[float] | None
    distance_current_to_approach: float | None
    distance_current_to_target: float | None
    distance_current_to_final: float | None
    control_api_sequence: list[dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _as_vec3(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {arr.shape}")
    return arr


def _as_quat_wxyz(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be a 4-vector, got shape {arr.shape}")
    norm = np.linalg.norm(arr)
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero.")
    return arr / norm


def _normalize(vec: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero.")
    return vec / norm


def _load_source_pointcloud_summary(plane_summary: dict[str, Any]) -> dict[str, Any] | None:
    source_summary = plane_summary.get("config", {}).get("source_summary")
    if not source_summary:
        return None
    source_path = Path(source_summary).expanduser().resolve()
    if not source_path.exists():
        return None
    return _load_json(source_path)


def _resolve_input_frame(cfg: PrismaticJointMotionConfig, plane_summary: dict[str, Any]) -> str:
    if cfg.input_frame != "auto":
        if cfg.input_frame not in {"camera", "world"}:
            raise ValueError(f"input_frame must be 'auto', 'camera', or 'world', got {cfg.input_frame!r}")
        return cfg.input_frame

    source = _load_source_pointcloud_summary(plane_summary)
    if source is not None:
        frame = source.get("config", {}).get("output_frame")
        if frame in {"camera", "world"}:
            return frame
    return "camera"


def _resolve_frame_info(cfg: PrismaticJointMotionConfig, plane_summary: dict[str, Any]) -> Path | None:
    if cfg.frame_info is not None:
        return cfg.frame_info.expanduser().resolve()

    source = _load_source_pointcloud_summary(plane_summary)
    if source is None:
        return None
    info = source.get("config", {}).get("info")
    if not info:
        return None
    return Path(info).expanduser().resolve()


def _load_camera_extrinsics(frame_info: Path, camera_name: str) -> np.ndarray:
    info = _load_json(frame_info)
    key = f"{camera_name}_camera_misc"
    misc = info.get(key)
    if not isinstance(misc, dict):
        raise ValueError(f"{frame_info} does not contain {key!r}.")
    extrinsics = np.asarray(misc.get(f"{camera_name}_camera_extrinsics"), dtype=np.float64)
    if extrinsics.shape != (4, 4):
        raise ValueError(f"{key}.{camera_name}_camera_extrinsics must have shape (4, 4).")
    return extrinsics


def _load_current_gripper_pose(frame_info: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if frame_info is None or not frame_info.exists():
        return None, None
    info = _load_json(frame_info)
    pose = info.get("gripper_pose")
    if pose is None:
        return None, None
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose_arr.shape != (7,):
        raise ValueError(f"{frame_info} gripper_pose must have shape (7,), got {pose_arr.shape}")
    position = pose_arr[:3]
    # RLBench/PyRep observations store gripper_pose as [x, y, z, qx, qy, qz, qw].
    quat_xyzw = pose_arr[3:]
    quat_wxyz = _as_quat_wxyz(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        "gripper_pose quaternion",
    )
    return position, quat_wxyz


def _transform_position_and_normal(
    position: np.ndarray,
    normal: np.ndarray,
    input_frame: str,
    frame_info: Path | None,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if input_frame == "world":
        return position, _normalize(normal, "normal")
    if input_frame != "camera":
        raise ValueError(f"Unsupported input_frame {input_frame!r}.")
    if frame_info is None:
        raise ValueError("Camera-frame target requires --frame-info or a source pointcloud summary with info.")

    extrinsics = _load_camera_extrinsics(frame_info, camera_name)
    position_hom = np.concatenate([position, [1.0]])
    world_position = (extrinsics @ position_hom)[:3]
    world_normal = extrinsics[:3, :3] @ _normalize(normal, "normal")
    return world_position, _normalize(world_normal, "world_normal")


def _choose_orientation(
    mode: str,
    current_quat_wxyz: np.ndarray | None,
) -> np.ndarray:
    if mode == "keep-current":
        if current_quat_wxyz is None:
            raise ValueError("orientation_mode='keep-current' requires gripper_pose in frame_info.")
        return current_quat_wxyz
    if mode == "identity":
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    raise ValueError(f"Unsupported orientation_mode {mode!r}; use 'keep-current' or 'identity'.")


def _motion_waypoint_step(name: str, position: np.ndarray, quat_wxyz: np.ndarray) -> dict[str, Any]:
    return {
        "step": name,
        "position": [float(v) for v in position],
        "quaternion_wxyz": [float(v) for v in quat_wxyz],
        "api_calls": [
            "joints = solve_ik(position, quaternion_wxyz)",
            "move_to_joints(joints)",
        ],
    }


def build_prismatic_joint_motion_target(
    cfg: PrismaticJointMotionConfig,
) -> tuple[PrismaticJointMotionTarget, dict[str, Any]]:
    plane_summary = _load_json(cfg.plane_summary)
    result = plane_summary.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{cfg.plane_summary} does not contain a result object.")

    contact_position_in = _as_vec3(result.get(cfg.target_key), cfg.target_key)
    normal_in = _as_vec3(result.get(cfg.normal_key), cfg.normal_key)
    input_frame = _resolve_input_frame(cfg, plane_summary)
    frame_info = _resolve_frame_info(cfg, plane_summary)
    current_position, current_quat = _load_current_gripper_pose(frame_info)

    contact_world, normal_world = _transform_position_and_normal(
        contact_position_in,
        normal_in,
        input_frame,
        frame_info,
        cfg.camera_name,
    )
    quat_wxyz = _choose_orientation(cfg.orientation_mode, current_quat)
    motion_direction = _normalize(normal_world * float(cfg.normal_direction_sign), "motion_direction")

    approach_position = contact_world - motion_direction * float(cfg.approach_distance)
    target_position = contact_world - motion_direction * float(cfg.target_standoff)
    final_position = target_position + motion_direction * float(cfg.travel_distance)

    distance_current_to_approach = None
    distance_current_to_target = None
    distance_current_to_final = None
    if current_position is not None:
        distance_current_to_approach = float(np.linalg.norm(approach_position - current_position))
        distance_current_to_target = float(np.linalg.norm(target_position - current_position))
        distance_current_to_final = float(np.linalg.norm(final_position - current_position))

    control_api_sequence: list[dict[str, Any]] = []
    if cfg.close_gripper_before_motion:
        control_api_sequence.append(
            {
                "step": "close_gripper",
                "api_calls": ["close_gripper()"],
            }
        )
    control_api_sequence.extend(
        [
            _motion_waypoint_step("approach", approach_position, quat_wxyz),
            _motion_waypoint_step("target", target_position, quat_wxyz),
            _motion_waypoint_step("translate", final_position, quat_wxyz),
        ]
    )

    target = PrismaticJointMotionTarget(
        input_frame=input_frame,
        output_frame="world",
        contact_position=[float(v) for v in contact_world],
        normal=[float(v) for v in normal_world],
        motion_direction=[float(v) for v in motion_direction],
        approach_position=[float(v) for v in approach_position],
        target_position=[float(v) for v in target_position],
        final_position=[float(v) for v in final_position],
        quaternion_wxyz=[float(v) for v in quat_wxyz],
        normal_direction_sign=float(cfg.normal_direction_sign),
        approach_distance=float(cfg.approach_distance),
        target_standoff=float(cfg.target_standoff),
        travel_distance=float(cfg.travel_distance),
        close_gripper_before_motion=bool(cfg.close_gripper_before_motion),
        current_gripper_position=(
            [float(v) for v in current_position] if current_position is not None else None
        ),
        current_gripper_quaternion_wxyz=(
            [float(v) for v in current_quat] if current_quat is not None else None
        ),
        distance_current_to_approach=distance_current_to_approach,
        distance_current_to_target=distance_current_to_target,
        distance_current_to_final=distance_current_to_final,
        control_api_sequence=control_api_sequence,
    )

    metadata = {
        "frame_info": str(frame_info) if frame_info else None,
        "plane_summary": str(cfg.plane_summary),
        "source_pointcloud_summary": plane_summary.get("config", {}).get("source_summary"),
        "input_contact_position": [float(v) for v in contact_position_in],
        "input_normal": [float(v) for v in normal_in],
        "motion_formula": "motion_direction = normal_direction_sign * plane_normal; approach/target are placed opposite motion_direction; final translates along motion_direction.",
    }
    return target, metadata


def save_prismatic_joint_motion_outputs(
    cfg: PrismaticJointMotionConfig,
    target: PrismaticJointMotionTarget,
    metadata: dict[str, Any],
) -> Path:
    output_root = cfg.output_dir / cfg.plane_summary.parent.name
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "prismatic_joint_motion_target.json"
    summary = {
        "config": {
            **asdict(cfg),
            "plane_summary": str(cfg.plane_summary),
            "frame_info": str(cfg.frame_info) if cfg.frame_info else None,
            "output_dir": str(cfg.output_dir),
        },
        "metadata": metadata,
        "result": asdict(target),
        "python_usage": [
            "if target['result']['close_gripper_before_motion']: close_gripper()",
            "quat = np.asarray(target['result']['quaternion_wxyz'], dtype=np.float64)",
            "for key in ['approach_position', 'target_position', 'final_position']:",
            "    position = np.asarray(target['result'][key], dtype=np.float64)",
            "    joints = solve_ik(position, quat)",
            "    move_to_joints(joints)",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def run_prismatic_joint_motion_target(
    cfg: PrismaticJointMotionConfig,
) -> tuple[PrismaticJointMotionTarget, Path]:
    target, metadata = build_prismatic_joint_motion_target(cfg)
    summary_path = save_prismatic_joint_motion_outputs(cfg, target, metadata)
    return target, summary_path


def execute_prismatic_joint_motion(
    control_api: Any,
    target: PrismaticJointMotionTarget,
    *,
    move_approach: bool = True,
    move_target: bool = True,
    move_final: bool = True,
    close_gripper: bool | None = None,
) -> None:
    """Execute a generated target with an object exposing cap-x reduced control APIs."""
    should_close = target.close_gripper_before_motion if close_gripper is None else close_gripper
    if should_close:
        control_api.close_gripper()

    waypoints: list[list[float]] = []
    if move_approach:
        waypoints.append(target.approach_position)
    if move_target:
        waypoints.append(target.target_position)
    if move_final:
        waypoints.append(target.final_position)

    quat = np.asarray(target.quaternion_wxyz, dtype=np.float64)
    for position in waypoints:
        joints = control_api.solve_ik(np.asarray(position, dtype=np.float64), quat)
        control_api.move_to_joints(joints)
