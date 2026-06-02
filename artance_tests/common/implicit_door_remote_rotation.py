from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.contact_guided_remote_rotation import (
    ContactGuidedRemoteRotationResult,
    _choose_rotation_sign,
    _json_default,
    _load_graspnet_contact_point,
    _load_pointcloud_npz,
    _nearest_distances,
    _part_label_id,
    _pointcloud_frame,
    _require_part_name,
    _rotate_points,
    _sample_axis,
    _sample_points,
    _sample_sphere,
    _write_ascii_ply,
)


@dataclass(frozen=True)
class ImplicitDoorRemoteRotationConfig:
    pointcloud_npz: Path
    output_dir: Path
    contact_summary: Path | None = None
    graspnet_summary: Path | None = None
    guide_source: str = "vlm"
    contact_point_mode: str = "handle_center"
    door_part: str | None = None
    handle_part: str | None = None
    door_opening_type: str = "side"
    hinge_axis_orientation: str = "vertical"
    closing_direction_sign: float = -1.0
    rotation_degrees: float = 40.0
    waypoint_count: int = 9
    collision_threshold: float = 0.012
    collision_check_degrees: float = 12.0
    max_collision_sample_points: int = 3000
    max_visualization_points: int = 120000
    source_summary: Path | None = None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_contact_pixel_xy(path: Path) -> np.ndarray:
    data = _load_json(path)
    pixel_xy = data.get("result", {}).get("pixel_xy")
    if not (
        isinstance(pixel_xy, list | tuple)
        and len(pixel_xy) == 2
        and all(isinstance(v, int | float) for v in pixel_xy)
    ):
        raise ValueError(f"Could not find numeric result.pixel_xy in {path}")
    return np.asarray([int(round(pixel_xy[0])), int(round(pixel_xy[1]))], dtype=np.int32)


def _source_summary_data(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return _load_json(path)


def _load_camera_extrinsics(source_summary: dict[str, Any] | None) -> np.ndarray | None:
    if source_summary is None:
        return None
    cfg = source_summary.get("config", {})
    frame_info = cfg.get("info")
    if not frame_info:
        return None
    frame_info_path = Path(frame_info).expanduser().resolve()
    if not frame_info_path.exists():
        return None
    camera_name = str(cfg.get("camera_name") or "wrist")
    info = _load_json(frame_info_path)
    misc = info.get(f"{camera_name}_camera_misc")
    if not isinstance(misc, dict):
        return None
    extrinsics = np.asarray(misc.get(f"{camera_name}_camera_extrinsics"), dtype=np.float64)
    if extrinsics.shape != (4, 4):
        return None
    return extrinsics


def _normalize(value: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero.")
    return value / norm


def _preferred_axis_in_pointcloud_frame(
    cfg: ImplicitDoorRemoteRotationConfig,
    source_summary: dict[str, Any] | None,
) -> np.ndarray | None:
    if cfg.hinge_axis_orientation != "vertical":
        return None
    frame = "camera"
    if source_summary is not None:
        source_frame = source_summary.get("config", {}).get("output_frame")
        if source_frame in {"camera", "world"}:
            frame = str(source_frame)
    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if frame == "world":
        return world_z
    extrinsics = _load_camera_extrinsics(source_summary)
    if extrinsics is None:
        return None
    return _normalize(extrinsics[:3, :3].T @ world_z, "world_z_camera")


def _fit_door_plane_axes(
    door_points: np.ndarray,
    preferred_axis: np.ndarray | None,
) -> dict[str, Any]:
    if door_points.shape[0] < 3:
        raise ValueError(f"At least 3 door points are required, got {door_points.shape[0]}.")
    center = door_points.mean(axis=0)
    centered = door_points - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    in_plane_a = _normalize(vh[0], "door_plane_axis_0")
    in_plane_b = _normalize(vh[1], "door_plane_axis_1")
    normal = _normalize(vh[2], "door_plane_normal")

    if preferred_axis is not None:
        projected = preferred_axis - normal * float(np.dot(preferred_axis, normal))
        if float(np.linalg.norm(projected)) > 1e-6:
            vertical_axis = _normalize(projected, "door_vertical_axis")
            axis_source = "world_vertical_projected_to_door_plane"
        else:
            vertical_axis = in_plane_a if singular_values[0] >= singular_values[1] else in_plane_b
            axis_source = "door_pca_long_axis_fallback"
    else:
        vertical_axis = in_plane_a if singular_values[0] >= singular_values[1] else in_plane_b
        axis_source = "door_pca_long_axis"

    width_axis = np.cross(normal, vertical_axis)
    if float(np.linalg.norm(width_axis)) <= 1e-6:
        width_axis = in_plane_b if abs(float(np.dot(in_plane_b, vertical_axis))) < 0.9 else in_plane_a
    width_axis = _normalize(width_axis - vertical_axis * float(np.dot(width_axis, vertical_axis)), "door_width_axis")

    width_coords = centered @ width_axis
    height_coords = centered @ vertical_axis
    robust_width = float(np.percentile(width_coords, 95.0) - np.percentile(width_coords, 5.0))
    robust_height = float(np.percentile(height_coords, 95.0) - np.percentile(height_coords, 5.0))
    full_width = float(np.max(width_coords) - np.min(width_coords))
    full_height = float(np.max(height_coords) - np.min(height_coords))
    return {
        "center": center,
        "normal": normal,
        "vertical_axis": vertical_axis,
        "width_axis": width_axis,
        "singular_values": singular_values,
        "axis_source": axis_source,
        "width": robust_width if robust_width > 1e-6 else full_width,
        "height": robust_height if robust_height > 1e-6 else full_height,
        "full_width": full_width,
        "full_height": full_height,
    }


def _select_contact_from_guide(
    cfg: ImplicitDoorRemoteRotationConfig,
    interaction_points: np.ndarray,
    interaction_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    guide_source = cfg.guide_source.lower()
    if guide_source == "vlm":
        if cfg.contact_summary is None:
            raise ValueError("guide_source='vlm' requires contact_summary.")
        contact_pixel_xy = _load_contact_pixel_xy(cfg.contact_summary)
        pixel_distances = np.linalg.norm(interaction_pixels - contact_pixel_xy[None, :], axis=1)
        index = int(np.argmin(pixel_distances))
        return (
            interaction_points[index],
            contact_pixel_xy,
            float(pixel_distances[index]),
            {
                "guide_source": "vlm",
                "guide_pixel_xy": [int(contact_pixel_xy[0]), int(contact_pixel_xy[1])],
                "nearest_interaction_pixel_xy": [int(v) for v in interaction_pixels[index].astype(np.int32)],
                "nearest_interaction_point_3d": [float(v) for v in interaction_points[index]],
                "guide_pixel_distance": float(pixel_distances[index]),
            },
        )
    if guide_source == "graspnet":
        if cfg.graspnet_summary is None:
            raise ValueError("guide_source='graspnet' requires graspnet_summary.")
        frame = _pointcloud_frame(cfg.pointcloud_npz, cfg.source_summary)
        graspnet_point = _load_graspnet_contact_point(cfg.graspnet_summary, frame)
        distances = np.linalg.norm(interaction_points - graspnet_point[None, :], axis=1)
        index = int(np.argmin(distances))
        contact_pixel_xy = interaction_pixels[index].astype(np.int32)
        return (
            interaction_points[index],
            contact_pixel_xy,
            float(distances[index]),
            {
                "guide_source": "graspnet",
                "guide_point_3d": [float(v) for v in graspnet_point],
                "nearest_interaction_pixel_xy": [int(v) for v in contact_pixel_xy],
                "nearest_interaction_point_3d": [float(v) for v in interaction_points[index]],
                "guide_point_distance": float(distances[index]),
            },
        )
    raise ValueError("guide_source must be 'vlm' or 'graspnet'.")


def _select_handle_center_contact(
    handle_points: np.ndarray,
    handle_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    handle_center = handle_points.mean(axis=0)
    distances = np.linalg.norm(handle_points - handle_center[None, :], axis=1)
    index = int(np.argmin(distances))
    contact_pixel_xy = handle_pixels[index].astype(np.int32)
    return (
        handle_center,
        contact_pixel_xy,
        0.0,
        {
            "contact_point_mode": "handle_center",
            "handle_center": [float(v) for v in handle_center],
            "nearest_handle_pixel_xy": [int(v) for v in contact_pixel_xy],
            "nearest_handle_point_3d": [float(v) for v in handle_points[index]],
            "handle_center_to_nearest_point_distance": float(distances[index]),
        },
    )


def _select_door_contact_point(
    cfg: ImplicitDoorRemoteRotationConfig,
    handle_points: np.ndarray,
    handle_pixels: np.ndarray,
    interaction_points: np.ndarray,
    interaction_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    mode = cfg.contact_point_mode.lower()
    if mode == "handle_center":
        contact_point, contact_pixel_xy, contact_distance, metadata = _select_handle_center_contact(
            handle_points,
            handle_pixels,
        )
        try:
            guide_point, guide_pixel_xy, guide_distance, guide_metadata = _select_contact_from_guide(
                cfg,
                interaction_points,
                interaction_pixels,
            )
            metadata.update(
                {
                    "guide_contact_point_3d": [float(v) for v in guide_point],
                    "guide_contact_pixel_xy": [int(v) for v in guide_pixel_xy],
                    "guide_contact_distance": float(guide_distance),
                    "guide_metadata": guide_metadata,
                }
            )
        except Exception as exc:
            metadata["guide_metadata_error"] = str(exc)
        return contact_point, contact_pixel_xy, contact_distance, metadata
    if mode == "guide":
        contact_point, contact_pixel_xy, contact_distance, metadata = _select_contact_from_guide(
            cfg,
            interaction_points,
            interaction_pixels,
        )
        metadata["contact_point_mode"] = "guide"
        return contact_point, contact_pixel_xy, contact_distance, metadata
    raise ValueError("contact_point_mode must be 'handle_center' or 'guide'.")


def _score_direction_by_alignment(
    contact_point: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    desired_direction: np.ndarray,
    sign: int,
) -> dict[str, float]:
    tangent = np.cross(axis_direction, contact_point - axis_point) * float(sign)
    tangent = _normalize(tangent, "candidate_tangent")
    return {
        "alignment_with_closing_direction": float(np.dot(tangent, desired_direction)),
        "candidate_tangent_x": float(tangent[0]),
        "candidate_tangent_y": float(tangent[1]),
        "candidate_tangent_z": float(tangent[2]),
    }


def _choose_sign_by_alignment(scores: dict[int, dict[str, float]]) -> int:
    minus = scores[-1]["alignment_with_closing_direction"]
    plus = scores[1]["alignment_with_closing_direction"]
    if plus == minus:
        return 1
    return 1 if plus > minus else -1


def _sample_edge_points(axis_point: np.ndarray, axis_direction: np.ndarray, height: float) -> np.ndarray:
    return _sample_axis(axis_point, axis_direction, max(height, 0.12), 0.0025)


def analyze_implicit_door_remote_rotation(
    cfg: ImplicitDoorRemoteRotationConfig,
) -> tuple[ContactGuidedRemoteRotationResult, dict[str, Any]]:
    if cfg.door_opening_type != "side":
        raise ValueError("Only door_opening_type='side' is implemented for the implicit door API.")
    if cfg.hinge_axis_orientation != "vertical":
        raise ValueError("Only hinge_axis_orientation='vertical' is implemented for the implicit door API.")

    data = _load_pointcloud_npz(cfg.pointcloud_npz)
    points = np.asarray(data["points"], dtype=np.float64)
    colors = np.asarray(data["colors_uint8"], dtype=np.uint8)
    labels = np.asarray(data["labels"], dtype=np.int32)
    valid_pixel_xy = np.asarray(data["valid_pixel_xy"], dtype=np.float64)
    part_names = data["part_names"]

    door_part = _require_part_name(cfg.door_part, "door_part")
    handle_part = _require_part_name(cfg.handle_part, "handle_part")
    door_label_id = _part_label_id(part_names, door_part)
    handle_label_id = _part_label_id(part_names, handle_part)
    door_mask = labels == door_label_id
    handle_mask = labels == handle_label_id
    door_points = points[door_mask]
    handle_points = points[handle_mask]
    if door_points.size == 0:
        raise ValueError(f"Door part {door_part!r} has no points in {cfg.pointcloud_npz}")
    if handle_points.size == 0:
        raise ValueError(f"Handle part {handle_part!r} has no points in {cfg.pointcloud_npz}")

    source_summary = _source_summary_data(cfg.source_summary)
    preferred_axis = _preferred_axis_in_pointcloud_frame(cfg, source_summary)
    door_axes = _fit_door_plane_axes(door_points, preferred_axis)
    door_center = np.asarray(door_axes["center"], dtype=np.float64)
    handle_center = handle_points.mean(axis=0)
    width_axis = np.asarray(door_axes["width_axis"], dtype=np.float64)
    vertical_axis = np.asarray(door_axes["vertical_axis"], dtype=np.float64)
    handle_offset_width = float(np.dot(handle_center - door_center, width_axis))
    if abs(handle_offset_width) <= 1e-9:
        handle_offset_width = 1.0
    axis_point = door_center - np.sign(handle_offset_width) * width_axis * (float(door_axes["width"]) * 0.5)
    axis_direction = vertical_axis
    if preferred_axis is not None and float(np.dot(axis_direction, preferred_axis)) < 0.0:
        axis_direction = -axis_direction

    interaction_mask = door_mask | handle_mask
    contact_point, contact_pixel_xy, contact_pixel_distance, contact_metadata = _select_door_contact_point(
        cfg,
        handle_points,
        valid_pixel_xy[handle_mask],
        points[interaction_mask],
        valid_pixel_xy[interaction_mask],
    )

    desired_direction = _normalize(
        np.asarray(door_axes["normal"], dtype=np.float64) * float(cfg.closing_direction_sign),
        "closing_direction",
    )
    direction_scores_by_int = {
        -1: _score_direction_by_alignment(contact_point, axis_point, axis_direction, desired_direction, -1),
        1: _score_direction_by_alignment(contact_point, axis_point, axis_direction, desired_direction, 1),
    }

    static_mask = ~(door_mask | handle_mask)
    collision_reference_points = points[static_mask]
    collision_reference = "all_non_door_and_handle"
    if collision_reference_points.size > 0:
        sampled_reference = _sample_points(collision_reference_points, cfg.max_collision_sample_points, seed=2)
        for sign, score in direction_scores_by_int.items():
            sampled_door = _sample_points(door_points, cfg.max_collision_sample_points, seed=1)
            rotated = _rotate_points(
                sampled_door,
                axis_point,
                axis_direction,
                np.deg2rad(float(sign) * cfg.collision_check_degrees),
            )
            distances = _nearest_distances(rotated, sampled_reference)
            below = distances < cfg.collision_threshold
            score.update(
                {
                    "min_distance": float(np.min(distances)),
                    "mean_distance": float(np.mean(distances)),
                    "collision_fraction": float(np.mean(below)),
                    "collision_count": float(np.sum(below)),
                    "reference_point_count": float(collision_reference_points.shape[0]),
                    "sampled_reference_point_count": float(sampled_reference.shape[0]),
                }
            )
        chosen_sign = _choose_rotation_sign(direction_scores_by_int)
    else:
        collision_reference = "none_available_mask_only_pointcloud"
        chosen_sign = _choose_sign_by_alignment(direction_scores_by_int)

    angles = np.linspace(0.0, np.deg2rad(chosen_sign * cfg.rotation_degrees), cfg.waypoint_count)
    waypoints = [
        _rotate_points(contact_point[None, :], axis_point, axis_direction, float(angle))[0]
        for angle in angles
    ]

    centered = door_points - door_center
    plane_distances = np.abs(centered @ np.asarray(door_axes["normal"], dtype=np.float64))
    result = ContactGuidedRemoteRotationResult(
        part_a=door_part,
        part_b=handle_part,
        part_a_label_id=door_label_id,
        part_b_label_id=handle_label_id,
        guide_source=cfg.guide_source.lower(),
        contact_pixel_xy=[int(contact_pixel_xy[0]), int(contact_pixel_xy[1])],
        contact_point_3d=[float(v) for v in contact_point],
        contact_point_pixel_distance=float(contact_pixel_distance),
        far_part_a_point_count=int(door_points.shape[0]),
        hinge_part_a_point_count=int(door_points.shape[0]),
        hinge_part_b_point_count=int(handle_points.shape[0]),
        axis_point=[float(v) for v in axis_point],
        axis_direction=[float(v) for v in axis_direction],
        axis_fit={
            "method": "implicit_door_prior",
            "door_opening_type": cfg.door_opening_type,
            "hinge_axis_orientation": cfg.hinge_axis_orientation,
            "axis_direction_source": door_axes["axis_source"],
            "door_center": [float(v) for v in door_center],
            "door_normal": [float(v) for v in door_axes["normal"]],
            "door_vertical_axis": [float(v) for v in axis_direction],
            "door_width_axis": [float(v) for v in width_axis],
            "door_width": float(door_axes["width"]),
            "door_height": float(door_axes["height"]),
            "door_full_width": float(door_axes["full_width"]),
            "door_full_height": float(door_axes["full_height"]),
            "handle_center": [float(v) for v in handle_center],
            "handle_offset_width": float(handle_offset_width),
            "contact_point_mode": cfg.contact_point_mode.lower(),
            "contact_point_metadata": contact_metadata,
            "hinge_side_rule": "handle_opposite_vertical_edge",
            "closing_direction": [float(v) for v in desired_direction],
            "closing_direction_sign": float(cfg.closing_direction_sign),
            "plane_rms_distance": float(np.sqrt(np.mean(plane_distances**2))),
            "plane_mean_abs_distance": float(np.mean(plane_distances)),
            "plane_max_abs_distance": float(np.max(plane_distances)),
            "singular_values": [float(v) for v in door_axes["singular_values"]],
        },
        chosen_rotation_sign=int(chosen_sign),
        direction_scores={str(key): value for key, value in direction_scores_by_int.items()},
        collision_reference=collision_reference,
        collision_reference_point_count=int(collision_reference_points.shape[0]),
        rotation_degrees=float(chosen_sign * cfg.rotation_degrees),
        waypoint_count=int(cfg.waypoint_count),
        waypoints=[[float(v) for v in point] for point in waypoints],
    )
    extras = {
        "points": points,
        "colors": colors,
        "labels": labels,
        "contact_point": contact_point,
        "axis_point": axis_point,
        "axis_direction": axis_direction,
        "axis_fit_points": door_points,
        "waypoints": np.asarray(waypoints, dtype=np.float64),
        "door_points": door_points,
        "handle_points": handle_points,
        "door_height": float(door_axes["height"]),
    }
    return result, extras


def _output_root_for_pointcloud(output_dir: Path, pointcloud_npz: Path) -> Path:
    if pointcloud_npz.parent.parent.name.startswith("variation"):
        return output_dir / pointcloud_npz.parent.parent.name / pointcloud_npz.parent.name
    return output_dir / pointcloud_npz.stem


def save_implicit_door_remote_rotation_outputs(
    cfg: ImplicitDoorRemoteRotationConfig,
    result: ContactGuidedRemoteRotationResult,
    extras: dict[str, Any],
) -> Path:
    output_root = _output_root_for_pointcloud(cfg.output_dir, cfg.pointcloud_npz)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "implicit_door_remote_rotation_summary.json"
    visualization_ply = output_root / f"{cfg.pointcloud_npz.stem}_implicit_door_rotation_visualization.ply"

    points = np.asarray(extras["points"], dtype=np.float64)
    colors = np.asarray(extras["colors"], dtype=np.uint8)
    labels = np.asarray(extras["labels"], dtype=np.int32)
    if points.shape[0] > cfg.max_visualization_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(points.shape[0], size=cfg.max_visualization_points, replace=False))
        vis_points = points[keep]
        vis_colors = colors[keep]
        vis_labels = labels[keep]
    else:
        vis_points = points
        vis_colors = colors
        vis_labels = labels

    axis_points = _sample_edge_points(
        np.asarray(extras["axis_point"], dtype=np.float64),
        np.asarray(extras["axis_direction"], dtype=np.float64),
        float(extras["door_height"]),
    )
    contact_sphere = _sample_sphere(np.asarray(extras["contact_point"], dtype=np.float64), 0.01)
    waypoint_points = np.asarray(extras["waypoints"], dtype=np.float64)
    waypoint_spheres = np.vstack([_sample_sphere(point, 0.006) for point in waypoint_points])
    door_edge_points = _sample_points(np.asarray(extras["door_points"], dtype=np.float64), 5000, seed=3)
    handle_points = _sample_points(np.asarray(extras["handle_points"], dtype=np.float64), 5000, seed=4)

    overlay_points = np.vstack([axis_points, contact_sphere, waypoint_spheres, door_edge_points, handle_points])
    overlay_colors = np.vstack(
        [
            np.tile(np.array([[20, 120, 255]], dtype=np.uint8), (axis_points.shape[0], 1)),
            np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (contact_sphere.shape[0], 1)),
            np.tile(np.array([[0, 245, 255]], dtype=np.uint8), (waypoint_spheres.shape[0], 1)),
            np.tile(np.array([[255, 0, 180]], dtype=np.uint8), (door_edge_points.shape[0], 1)),
            np.tile(np.array([[255, 145, 0]], dtype=np.uint8), (handle_points.shape[0], 1)),
        ]
    )
    overlay_labels = np.concatenate(
        [
            np.full((axis_points.shape[0],), -1, dtype=np.int32),
            np.full((contact_sphere.shape[0],), -2, dtype=np.int32),
            np.full((waypoint_spheres.shape[0],), -3, dtype=np.int32),
            np.full((door_edge_points.shape[0],), -4, dtype=np.int32),
            np.full((handle_points.shape[0],), -5, dtype=np.int32),
        ]
    )
    _write_ascii_ply(
        visualization_ply,
        np.vstack([vis_points, overlay_points]),
        np.vstack([vis_colors, overlay_colors]),
        np.concatenate([vis_labels, overlay_labels]),
    )

    summary = {
        "config": {
            **asdict(cfg),
            "pointcloud_npz": str(cfg.pointcloud_npz),
            "contact_summary": str(cfg.contact_summary) if cfg.contact_summary else None,
            "graspnet_summary": str(cfg.graspnet_summary) if cfg.graspnet_summary else None,
            "output_dir": str(cfg.output_dir),
            "source_summary": str(cfg.source_summary) if cfg.source_summary else None,
        },
        "result": asdict(result),
        "visualization_labels": {
            "-1": "implicit_hinge_axis_blue",
            "-2": "contact_point_white",
            "-3": "rotated_waypoints_cyan",
            "-4": "door_points_magenta_sample",
            "-5": "handle_points_orange_sample",
        },
        "outputs": {
            "visualization_ply": str(visualization_ply),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary_path


def run_implicit_door_remote_rotation(
    cfg: ImplicitDoorRemoteRotationConfig,
) -> tuple[ContactGuidedRemoteRotationResult, Path]:
    result, extras = analyze_implicit_door_remote_rotation(cfg)
    summary_path = save_implicit_door_remote_rotation_outputs(cfg, result, extras)
    return result, summary_path
