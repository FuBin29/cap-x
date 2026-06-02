from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.paths import ARTANCE_ROOT


END_EFFECTOR_MOTION_SCHEMA_VERSION = 1
END_EFFECTOR_MOTION_FILENAME = "end_effector_motion_target.json"


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
class EndEffectorMotionStep:
    name: str
    position: list[float]
    quaternion_wxyz: list[float]


@dataclass(frozen=True)
class EndEffectorMotionTarget:
    schema_version: int
    task: str
    joint_type: str
    analysis_path: str
    output_frame: str
    motion_steps: list[EndEffectorMotionStep]
    close_gripper_before_motion: bool
    close_gripper_after_steps: list[str]
    source_target: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RemoteRotationEndEffectorMotionConfig:
    task: str
    rotation_summary: Path
    output_dir: Path
    graspnet_summary: Path | None = None
    close_gripper_after_step: str = "grasp"
    motion_mode: str = "grasp_follow"
    orientation_mode: str = "keep-current"
    approach_distance: float = 0.08
    target_standoff: float = 0.015
    close_gripper_before_motion: bool = True


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_source_pointcloud_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    source_summary = summary.get("config", {}).get("source_summary")
    if not source_summary:
        return None
    source_path = Path(source_summary).expanduser().resolve()
    if not source_path.exists():
        return None
    return _load_json(source_path)


def _resolve_prismatic_input_frame(cfg: PrismaticJointMotionConfig, plane_summary: dict[str, Any]) -> str:
    if cfg.input_frame != "auto":
        if cfg.input_frame not in {"camera", "world"}:
            raise ValueError(f"input_frame must be 'auto', 'camera', or 'world', got {cfg.input_frame!r}")
        return cfg.input_frame

    source = _load_source_pointcloud_summary(plane_summary)
    if source is not None:
        frame = source.get("config", {}).get("output_frame")
        if frame in {"camera", "world"}:
            return str(frame)
    return "camera"


def _resolve_prismatic_frame_info(
    cfg: PrismaticJointMotionConfig,
    plane_summary: dict[str, Any],
) -> Path | None:
    if cfg.frame_info is not None:
        return cfg.frame_info.expanduser().resolve()

    source = _load_source_pointcloud_summary(plane_summary)
    if source is None:
        return None
    info = source.get("config", {}).get("info")
    if not info:
        return None
    return Path(info).expanduser().resolve()


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
    quat_xyzw = pose_arr[3:]
    quat_wxyz = np.asarray(
        _as_quat_wxyz([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], "gripper_pose quaternion"),
        dtype=np.float64,
    )
    return pose_arr[:3], quat_wxyz


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
    extrinsics = _load_camera_extrinsics(frame_info, camera_name)
    world_position = (extrinsics @ np.r_[position, 1.0])[:3]
    world_normal = extrinsics[:3, :3] @ _normalize(normal, "normal")
    return world_position, _normalize(world_normal, "world_normal")


def _choose_orientation(mode: str, current_quat_wxyz: np.ndarray | None) -> np.ndarray:
    if mode == "keep-current":
        if current_quat_wxyz is None:
            raise ValueError("orientation_mode='keep-current' requires gripper_pose in frame_info.")
        return current_quat_wxyz
    if mode == "identity":
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    raise ValueError(f"Unsupported orientation_mode {mode!r}; use 'keep-current' or 'identity'.")


def _as_vec3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {array.shape}")
    return array


def _as_quat_wxyz(value: Any, name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (4,):
        raise ValueError(f"{name} must be a 4-vector, got shape {array.shape}")
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero.")
    return [float(v) for v in array / norm]


def _normalize(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero.")
    return array / norm


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> list[float]:
    rot = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rot[2, 1] - rot[1, 2]) / scale
        y = (rot[0, 2] - rot[2, 0]) / scale
        z = (rot[1, 0] - rot[0, 1]) / scale
    else:
        diagonal = np.diag(rot)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / scale
            x = 0.25 * scale
            y = (rot[0, 1] + rot[1, 0]) / scale
            z = (rot[0, 2] + rot[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / scale
            x = (rot[0, 1] + rot[1, 0]) / scale
            y = 0.25 * scale
            z = (rot[1, 2] + rot[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / scale
            x = (rot[0, 2] + rot[2, 0]) / scale
            y = (rot[1, 2] + rot[2, 1]) / scale
            z = 0.25 * scale
    return _as_quat_wxyz([w, x, y, z], "matrix quaternion")


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis, "axis")
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _resolve_artance_path(path_like: str | Path | None) -> Path | None:
    if not path_like:
        return None

    path = Path(path_like).expanduser()
    candidates = [path]
    parts = path.parts
    for anchor in ("cap-x", "RLBench", "data", "vrb", "region_eval_AGD20K", "ManiSkill"):
        if anchor in parts:
            candidates.append(ARTANCE_ROOT.joinpath(*parts[parts.index(anchor) :]))
            break
    if not path.is_absolute():
        candidates.append(ARTANCE_ROOT / path)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[-1].resolve()


def _resolve_source_summary(data: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    source_summary = data.get("config", {}).get("source_summary")
    source_path = _resolve_artance_path(source_summary)
    if source_path is None or not source_path.exists():
        return source_path, None
    return source_path, _load_json(source_path)


def _pointcloud_frame(source_summary: dict[str, Any] | None) -> str:
    if source_summary is None:
        return "camera"
    frame = source_summary.get("config", {}).get("output_frame")
    if frame in {"camera", "world"}:
        return str(frame)
    raise ValueError(f"Unsupported source pointcloud output_frame={frame!r}.")


def _frame_info_path(source_summary: dict[str, Any] | None) -> Path | None:
    if source_summary is None:
        return None
    info = source_summary.get("config", {}).get("info")
    return _resolve_artance_path(info)


def _camera_name(source_summary: dict[str, Any] | None) -> str:
    if source_summary is None:
        return "wrist"
    return str(source_summary.get("config", {}).get("camera_name") or "wrist")


def _load_camera_extrinsics(frame_info: Path | None, camera_name: str) -> np.ndarray:
    if frame_info is None:
        raise ValueError("Camera-frame end-effector target requires source_summary config.info.")
    info = _load_json(frame_info)
    key = f"{camera_name}_camera_misc"
    misc = info.get(key)
    if not isinstance(misc, dict):
        raise ValueError(f"{frame_info} does not contain {key!r}.")
    extrinsics = np.asarray(misc.get(f"{camera_name}_camera_extrinsics"), dtype=np.float64)
    if extrinsics.shape != (4, 4):
        raise ValueError(f"{key}.{camera_name}_camera_extrinsics must have shape (4, 4).")
    return extrinsics


def _transform_point(point: Any, input_frame: str, extrinsics: np.ndarray | None) -> np.ndarray:
    point_array = _as_vec3(point, "point")
    if input_frame == "world":
        return point_array
    if input_frame != "camera":
        raise ValueError(f"Unsupported input_frame {input_frame!r}.")
    if extrinsics is None:
        raise ValueError("Camera-frame point transform requires camera extrinsics.")
    return (extrinsics @ np.r_[point_array, 1.0])[:3]


def _transform_direction(direction: Any, input_frame: str, extrinsics: np.ndarray | None) -> np.ndarray:
    direction_array = _as_vec3(direction, "direction")
    if input_frame == "world":
        return direction_array
    if input_frame != "camera":
        raise ValueError(f"Unsupported input_frame {input_frame!r}.")
    if extrinsics is None:
        raise ValueError("Camera-frame direction transform requires camera extrinsics.")
    return extrinsics[:3, :3] @ direction_array


def _default_graspnet_summary_path(data: dict[str, Any]) -> Path | None:
    pointcloud_npz = data.get("config", {}).get("pointcloud_npz")
    pointcloud_path = _resolve_artance_path(pointcloud_npz)
    if pointcloud_path is None:
        return None

    parts = list(pointcloud_path.parts)
    try:
        pointcloud_index = parts.index("pointcloud")
    except ValueError:
        return None
    parts[pointcloud_index] = "contact_graspnet_pose"
    return Path(*parts).parent / "contact_graspnet_summary.json"


def _load_grasp_matrix(
    data: dict[str, Any],
    graspnet_summary: Path | None,
    input_frame: str,
    extrinsics: np.ndarray | None,
) -> tuple[np.ndarray, Path]:
    graspnet_path = (
        graspnet_summary.expanduser().resolve()
        if graspnet_summary is not None
        else _resolve_artance_path(data.get("config", {}).get("graspnet_summary"))
    )
    if graspnet_path is None or not graspnet_path.exists():
        graspnet_path = _default_graspnet_summary_path(data)
    if graspnet_path is None or not graspnet_path.exists():
        raise FileNotFoundError(
            "Contact-GraspNet summary not found. Provide --graspnet-summary or run contact_graspnet_pose first."
        )

    best = _load_json(graspnet_path).get("result", {}).get("best")
    if not isinstance(best, dict):
        raise ValueError(f"{graspnet_path} does not contain result.best.")

    pose_world = best.get("pose_world") or best.get("grasp_world")
    if pose_world is not None:
        matrix = np.asarray(pose_world, dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix, graspnet_path

    pose_camera = best.get("pose_camera") or best.get("grasp_camera")
    if pose_camera is None:
        raise ValueError(f"{graspnet_path} does not contain a usable best pose matrix.")
    matrix = np.asarray(pose_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Contact-GraspNet best pose must have shape (4, 4), got {matrix.shape}.")
    if input_frame == "world":
        return matrix, graspnet_path
    if extrinsics is None:
        raise ValueError("Camera-frame grasp pose requires camera extrinsics.")
    return extrinsics @ matrix, graspnet_path


def _step(name: str, position: np.ndarray, quaternion_wxyz: list[float]) -> EndEffectorMotionStep:
    return EndEffectorMotionStep(
        name=name,
        position=[float(v) for v in position],
        quaternion_wxyz=quaternion_wxyz,
    )


def _write_target(path: Path, target: EndEffectorMotionTarget) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(target), indent=2), encoding="utf-8")
    return path


def _output_root_from_source(output_dir: Path, source_path: Path) -> Path:
    return output_dir / source_path.parent.name


def build_prismatic_end_effector_motion_target(
    task: str,
    cfg: PrismaticJointMotionConfig,
) -> tuple[EndEffectorMotionTarget, Path]:
    plane_summary = _load_json(cfg.plane_summary)
    result = plane_summary.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{cfg.plane_summary} does not contain a result object.")

    contact_position_in = _as_vec3(result.get(cfg.target_key), cfg.target_key)
    normal_in = _as_vec3(result.get(cfg.normal_key), cfg.normal_key)
    input_frame = _resolve_prismatic_input_frame(cfg, plane_summary)
    frame_info = _resolve_prismatic_frame_info(cfg, plane_summary)
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
    steps = [
        _step("approach", approach_position, _as_quat_wxyz(quat_wxyz, "quaternion_wxyz")),
        _step("target", target_position, _as_quat_wxyz(quat_wxyz, "quaternion_wxyz")),
        _step("translate", final_position, _as_quat_wxyz(quat_wxyz, "quaternion_wxyz")),
    ]

    distances: dict[str, float | None] = {
        "current_to_approach": None,
        "current_to_target": None,
        "current_to_final": None,
    }
    if current_position is not None:
        distances = {
            "current_to_approach": float(np.linalg.norm(approach_position - current_position)),
            "current_to_target": float(np.linalg.norm(target_position - current_position)),
            "current_to_final": float(np.linalg.norm(final_position - current_position)),
        }

    target = EndEffectorMotionTarget(
        schema_version=END_EFFECTOR_MOTION_SCHEMA_VERSION,
        task=task,
        joint_type="prismatic",
        analysis_path="part_adjacency_plane_to_linear_ee_path",
        output_frame="world",
        motion_steps=steps,
        close_gripper_before_motion=bool(cfg.close_gripper_before_motion),
        close_gripper_after_steps=[],
        source_target=str(cfg.plane_summary),
        metadata={
            "frame_info": str(frame_info) if frame_info else None,
            "plane_summary": str(cfg.plane_summary),
            "source_pointcloud_summary": plane_summary.get("config", {}).get("source_summary"),
            "input_frame": input_frame,
            "input_contact_position": [float(v) for v in contact_position_in],
            "input_normal": [float(v) for v in normal_in],
            "contact_position_world": [float(v) for v in contact_world],
            "normal_world": [float(v) for v in normal_world],
            "motion_direction_world": [float(v) for v in motion_direction],
            "normal_direction_sign": float(cfg.normal_direction_sign),
            "approach_distance": float(cfg.approach_distance),
            "target_standoff": float(cfg.target_standoff),
            "travel_distance": float(cfg.travel_distance),
            "current_gripper_position": (
                [float(v) for v in current_position] if current_position is not None else None
            ),
            "current_gripper_quaternion_wxyz": (
                [float(v) for v in current_quat] if current_quat is not None else None
            ),
            "distances": distances,
            "motion_formula": "motion_direction = normal_direction_sign * plane_normal; approach/target are placed opposite motion_direction; final translates along motion_direction.",
        },
    )
    summary_path = (
        _output_root_from_source(cfg.output_dir, cfg.plane_summary)
        / END_EFFECTOR_MOTION_FILENAME
    )
    return target, summary_path


def run_prismatic_end_effector_motion_target(
    task: str,
    cfg: PrismaticJointMotionConfig,
) -> tuple[EndEffectorMotionTarget, Path]:
    target, summary_path = build_prismatic_end_effector_motion_target(task, cfg)
    _write_target(summary_path, target)
    return target, summary_path


def _load_remote_rotation_geometry(
    cfg: RemoteRotationEndEffectorMotionConfig,
) -> tuple[dict[str, Any], dict[str, Any], Path | None, dict[str, Any] | None, str, Path | None, str, np.ndarray | None, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    data = _load_json(cfg.rotation_summary)
    result = data.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{cfg.rotation_summary} does not contain a result object.")

    source_summary_path, source_summary = _resolve_source_summary(data)
    input_frame = _pointcloud_frame(source_summary)
    frame_info = _frame_info_path(source_summary)
    camera_name = _camera_name(source_summary)
    extrinsics = _load_camera_extrinsics(frame_info, camera_name) if input_frame == "camera" else None

    waypoints_in = result.get("waypoints")
    if not isinstance(waypoints_in, (list, tuple)) or len(waypoints_in) < 2:
        raise ValueError("Remote rotation result.waypoints must contain at least two 3D points.")
    waypoints_world = [_transform_point(item, input_frame, extrinsics) for item in waypoints_in]
    contact_world = _transform_point(result.get("contact_point_3d"), input_frame, extrinsics)
    axis_point_world = _transform_point(result.get("axis_point"), input_frame, extrinsics)
    axis_direction_world = _normalize(
        _transform_direction(result.get("axis_direction"), input_frame, extrinsics),
        "axis_direction_world",
    )
    return (
        data,
        result,
        source_summary_path,
        source_summary,
        input_frame,
        frame_info,
        camera_name,
        extrinsics,
        waypoints_world,
        contact_world,
        axis_point_world,
        axis_direction_world,
    )


def _remote_rotation_tangent(
    point: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    rotation_sign: float,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    tangent = np.cross(axis_direction, point - axis_point) * rotation_sign
    norm = float(np.linalg.norm(tangent))
    if norm > 1e-12:
        return tangent / norm
    if fallback is not None:
        return _normalize(fallback, "fallback_tangent")
    raise ValueError("Could not compute a stable rotation tangent for the contact point.")


def build_remote_rotation_closed_gripper_push_target(
    cfg: RemoteRotationEndEffectorMotionConfig,
) -> tuple[EndEffectorMotionTarget, Path]:
    (
        data,
        result,
        source_summary_path,
        _source_summary,
        input_frame,
        frame_info,
        camera_name,
        _extrinsics,
        waypoints_world,
        contact_world,
        axis_point_world,
        axis_direction_world,
    ) = _load_remote_rotation_geometry(cfg)

    _current_position, current_quat = _load_current_gripper_pose(frame_info)
    quat_wxyz = _choose_orientation(cfg.orientation_mode, current_quat)
    rotation_degrees = float(result.get("rotation_degrees", 0.0))
    rotation_sign = 1.0 if rotation_degrees >= 0.0 else -1.0
    fallback_tangent = waypoints_world[1] - waypoints_world[0]
    first_tangent = _remote_rotation_tangent(
        contact_world,
        axis_point_world,
        axis_direction_world,
        rotation_sign,
        fallback=fallback_tangent,
    )

    approach_position = contact_world - first_tangent * float(cfg.approach_distance)
    target_position = contact_world - first_tangent * float(cfg.target_standoff)
    steps = [
        _step("approach", approach_position, _as_quat_wxyz(quat_wxyz, "quaternion_wxyz")),
        _step("target", target_position, _as_quat_wxyz(quat_wxyz, "quaternion_wxyz")),
    ]
    push_tangents = [first_tangent]
    for index, waypoint_world in enumerate(waypoints_world[1:], start=1):
        tangent = _remote_rotation_tangent(
            waypoint_world,
            axis_point_world,
            axis_direction_world,
            rotation_sign,
            fallback=waypoint_world - waypoints_world[index - 1],
        )
        push_tangents.append(tangent)
        position = waypoint_world - tangent * float(cfg.target_standoff)
        steps.append(_step(f"push_{index:02d}", position, _as_quat_wxyz(quat_wxyz, "quaternion_wxyz")))

    target = EndEffectorMotionTarget(
        schema_version=END_EFFECTOR_MOTION_SCHEMA_VERSION,
        task=cfg.task,
        joint_type="revolute",
        analysis_path="contact_guided_remote_rotation_to_closed_gripper_push_path",
        output_frame="world",
        motion_steps=steps,
        close_gripper_before_motion=bool(cfg.close_gripper_before_motion),
        close_gripper_after_steps=[],
        source_target=str(cfg.rotation_summary),
        metadata={
            "motion_mode": cfg.motion_mode,
            "rotation_summary": str(cfg.rotation_summary),
            "source_pointcloud_summary": str(source_summary_path) if source_summary_path else None,
            "frame_info": str(frame_info) if frame_info else None,
            "camera_name": camera_name,
            "input_frame": input_frame,
            "contact_point_world": [float(v) for v in contact_world],
            "axis_point_world": [float(v) for v in axis_point_world],
            "axis_direction_world": [float(v) for v in axis_direction_world],
            "rotation_degrees": rotation_degrees,
            "rotation_sign": rotation_sign,
            "orientation_mode": cfg.orientation_mode,
            "approach_distance": float(cfg.approach_distance),
            "target_standoff": float(cfg.target_standoff),
            "first_push_tangent_world": [float(v) for v in first_tangent],
            "push_tangents_world": [[float(v) for v in item] for item in push_tangents],
            "source_rotation_result": result,
            "motion_formula": "closed gripper acts as pusher; target_i = rotated_contact_i - tangent_i * target_standoff, where tangent_i follows the selected revolute direction.",
        },
    )
    summary_path = (
        _output_root_from_source(cfg.output_dir, cfg.rotation_summary)
        / END_EFFECTOR_MOTION_FILENAME
    )
    return target, summary_path

def build_remote_rotation_end_effector_motion_target(
    cfg: RemoteRotationEndEffectorMotionConfig,
) -> tuple[EndEffectorMotionTarget, Path]:
    motion_mode = cfg.motion_mode.lower()
    if motion_mode == "closed_gripper_push":
        return build_remote_rotation_closed_gripper_push_target(cfg)
    if motion_mode != "grasp_follow":
        raise ValueError("motion_mode must be \"grasp_follow\" or \"closed_gripper_push\".")

    data = _load_json(cfg.rotation_summary)
    result = data.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{cfg.rotation_summary} does not contain a result object.")

    source_summary_path, source_summary = _resolve_source_summary(data)
    input_frame = _pointcloud_frame(source_summary)
    frame_info = _frame_info_path(source_summary)
    camera_name = _camera_name(source_summary)
    extrinsics = _load_camera_extrinsics(frame_info, camera_name) if input_frame == "camera" else None

    waypoints_in = result.get("waypoints")
    if not isinstance(waypoints_in, (list, tuple)) or len(waypoints_in) < 2:
        raise ValueError("Remote rotation result.waypoints must contain at least two 3D points.")
    waypoints_world = [_transform_point(item, input_frame, extrinsics) for item in waypoints_in]
    contact_world = _transform_point(result.get("contact_point_3d"), input_frame, extrinsics)
    axis_direction_world = _normalize(
        _transform_direction(result.get("axis_direction"), input_frame, extrinsics),
        "axis_direction_world",
    )

    grasp_matrix, graspnet_path = _load_grasp_matrix(
        data,
        cfg.graspnet_summary,
        input_frame,
        extrinsics,
    )
    rotation_degrees = float(result.get("rotation_degrees", 0.0))
    rotation_steps = max(1, len(waypoints_world) - 1)
    grasp_position0 = grasp_matrix[:3, 3]
    grasp_rotation0 = grasp_matrix[:3, :3]
    contact_to_grasp = grasp_position0 - contact_world

    steps = [
        _step("grasp", grasp_position0, _matrix_to_quat_wxyz(grasp_rotation0)),
    ]
    for index, waypoint_world in enumerate(waypoints_world[1:], start=1):
        angle = np.deg2rad(rotation_degrees * index / rotation_steps)
        rotation_delta = _axis_angle_matrix(axis_direction_world, angle)
        position = waypoint_world + rotation_delta @ contact_to_grasp
        quaternion_wxyz = _matrix_to_quat_wxyz(rotation_delta @ grasp_rotation0)
        steps.append(_step(f"rotate_{index:02d}", position, quaternion_wxyz))

    target = EndEffectorMotionTarget(
        schema_version=END_EFFECTOR_MOTION_SCHEMA_VERSION,
        task=cfg.task,
        joint_type="revolute",
        analysis_path="contact_guided_remote_rotation_to_rotating_ee_path",
        output_frame="world",
        motion_steps=steps,
        close_gripper_before_motion=False,
        close_gripper_after_steps=[cfg.close_gripper_after_step],
        source_target=str(cfg.rotation_summary),
        metadata={
            "motion_mode": cfg.motion_mode,
            "rotation_summary": str(cfg.rotation_summary),
            "source_pointcloud_summary": str(source_summary_path) if source_summary_path else None,
            "frame_info": str(frame_info) if frame_info else None,
            "camera_name": camera_name,
            "input_frame": input_frame,
            "graspnet_summary": str(graspnet_path),
            "contact_point_world": [float(v) for v in contact_world],
            "axis_direction_world": [float(v) for v in axis_direction_world],
            "rotation_degrees": rotation_degrees,
            "source_rotation_result": result,
        },
    )
    summary_path = (
        _output_root_from_source(cfg.output_dir, cfg.rotation_summary)
        / END_EFFECTOR_MOTION_FILENAME
    )
    return target, summary_path


def run_remote_rotation_end_effector_motion_target(
    cfg: RemoteRotationEndEffectorMotionConfig,
) -> tuple[EndEffectorMotionTarget, Path]:
    target, summary_path = build_remote_rotation_end_effector_motion_target(cfg)
    _write_target(summary_path, target)
    return target, summary_path
