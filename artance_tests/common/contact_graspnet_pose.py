from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from common.paths import ensure_capx_on_path
from common.pointcloud_reconstruction import (
    PartMaskSpec,
    _load_depth,
    _load_mask,
    _load_rgb,
    load_camera_intrinsics,
    load_camera_pose,
    load_part_masks_from_point_selection_summary,
    load_part_masks_from_sam3_summary,
)


DEFAULT_GRASPNET_SERVICE_URL = "http://127.0.0.1:8115"
DEFAULT_GRASP_Z_OFFSET = 0.1034


@dataclass(frozen=True)
class ContactGraspNetPoseConfig:
    rgb: Path
    depth: Path
    intrinsics: Path | None
    info: Path | None
    output_dir: Path
    target_prompt: str
    sam3_summary: Path | None = None
    point_selection_summary: Path | None = None
    mask: Path | None = None
    pose: Path | None = None
    camera_name: str = "wrist"
    depth_format: str = "auto"
    segmap_id: int = 1
    z_range: tuple[float, float] = (0.2, 2.0)
    forward_passes: int = 3
    max_retries: int = 10
    service_url: str = DEFAULT_GRASPNET_SERVICE_URL
    grasp_z_offset: float = DEFAULT_GRASP_Z_OFFSET
    save_visualization: bool = True
    visualization_subsample_factor: int = 2
    visualization_top_k: int = 10
    visualization_axis_length: float = 0.055
    visualization_point_radius: float = 0.008


@dataclass(frozen=True)
class ContactGraspNetPoseResult:
    grasps_camera: np.ndarray
    grasp_poses_camera: np.ndarray
    scores: np.ndarray
    contact_points_camera: np.ndarray
    best_index: int | None
    best_score: float | None
    best_grasp_camera: np.ndarray | None
    best_pose_camera: np.ndarray | None
    best_grasp_world: np.ndarray | None
    best_pose_world: np.ndarray | None
    mask_pixels: int
    image_size: tuple[int, int]
    intrinsics: np.ndarray
    camera_pose: np.ndarray | None
    mask_spec: PartMaskSpec


def _matrix_to_position_quaternion_wxyz(matrix: np.ndarray) -> dict[str, list[float]]:
    rotation_xyzw = SciRotation.from_matrix(matrix[:3, :3]).as_quat()
    quaternion_wxyz = np.asarray(
        [rotation_xyzw[3], rotation_xyzw[0], rotation_xyzw[1], rotation_xyzw[2]],
        dtype=np.float64,
    )
    return {
        "position": matrix[:3, 3].astype(float).tolist(),
        "quaternion_wxyz": quaternion_wxyz.astype(float).tolist(),
    }


def _pose_contact_diagnostics(
    matrix: np.ndarray,
    contact_point: np.ndarray,
) -> dict[str, Any]:
    origin = np.asarray(matrix[:3, 3], dtype=np.float64)
    contact = np.asarray(contact_point, dtype=np.float64).reshape(3)
    delta = origin - contact
    rot = np.asarray(matrix[:3, :3], dtype=np.float64)
    return {
        "origin_to_contact_distance_m": float(np.linalg.norm(delta)),
        "origin_minus_contact_camera": delta.astype(float).tolist(),
        "origin_minus_contact_in_grasp_frame": (rot.T @ delta).astype(float).tolist(),
        "axis_x_camera": rot[:, 0].astype(float).tolist(),
        "axis_y_camera": rot[:, 1].astype(float).tolist(),
        "axis_z_camera": rot[:, 2].astype(float).tolist(),
    }


def _apply_grasp_z_offset(grasps: np.ndarray, offset: float) -> np.ndarray:
    if offset == 0.0 or grasps.size == 0:
        return grasps.copy()
    offset_tf = np.eye(4, dtype=np.float64)
    offset_tf[:3, 3] = np.array([0.0, 0.0, offset], dtype=np.float64)
    return np.matmul(grasps, offset_tf)


def _depth_to_points_image(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    z = depth
    x = (xx - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (yy - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.stack([x, y, z], axis=-1)


def _sample_line(start: np.ndarray, end: np.ndarray, step: float = 0.003) -> np.ndarray:
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(distance / step)) + 1)
    weights = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
    return start[None, :] * (1.0 - weights) + end[None, :] * weights


def _sample_sphere(center: np.ndarray, radius: float, rings: int = 8, segments: int = 16) -> np.ndarray:
    points: list[np.ndarray] = []
    for i in range(1, rings):
        phi = np.pi * i / rings
        for j in range(segments):
            theta = 2.0 * np.pi * j / segments
            points.append(
                center
                + radius
                * np.array(
                    [
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi),
                    ],
                    dtype=np.float64,
                )
            )
    points.append(center.copy())
    return np.asarray(points, dtype=np.float64)


def _sample_pose_axes(matrix: np.ndarray, axis_length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = matrix[:3, 3]
    rot = matrix[:3, :3]
    x_axis = _sample_line(origin, origin + rot[:, 0] * axis_length)
    y_axis = _sample_line(origin, origin + rot[:, 1] * axis_length)
    z_axis = _sample_line(origin, origin + rot[:, 2] * axis_length)
    return x_axis, y_axis, z_axis


def _write_visualization_ply(path: Path, points: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")
        f.write("end_header\n")
        for point, color, label in zip(points, colors, labels, strict=True):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(label)}\n"
            )

def _load_target_mask_spec(cfg: ContactGraspNetPoseConfig) -> PartMaskSpec:
    if cfg.mask is not None:
        return PartMaskSpec(
            name=cfg.target_prompt,
            mask=cfg.mask.expanduser().resolve(),
            color_rgb=(230, 57, 70),
        )

    if cfg.point_selection_summary is not None:
        if cfg.sam3_summary is None:
            raise ValueError("--point-selection-summary requires --sam3-summary to locate mask files.")
        masks = load_part_masks_from_point_selection_summary(
            cfg.point_selection_summary.expanduser().resolve(),
            cfg.sam3_summary.expanduser().resolve(),
            prompts=(cfg.target_prompt,),
        )
        return masks[0]

    if cfg.sam3_summary is not None:
        masks = load_part_masks_from_sam3_summary(
            cfg.sam3_summary.expanduser().resolve(),
            ranks=(1,),
        )
        for mask in masks:
            if mask.name == cfg.target_prompt:
                return mask
        available = ", ".join(mask.name for mask in masks)
        raise ValueError(
            f"Target prompt {cfg.target_prompt!r} was not found in SAM3 rank-1 masks. "
            f"Available: {available}"
        )

    raise ValueError("Provide --mask, --point-selection-summary, or --sam3-summary.")


def run_contact_graspnet_pose(cfg: ContactGraspNetPoseConfig) -> tuple[ContactGraspNetPoseResult, Path]:
    rgb = _load_rgb(cfg.rgb)
    depth = _load_depth(
        cfg.depth,
        info_path=cfg.info,
        camera_name=cfg.camera_name,
        depth_format=cfg.depth_format,
    )
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"Depth shape {depth.shape} does not match RGB shape {rgb.shape[:2]}")

    intrinsics = load_camera_intrinsics(cfg.intrinsics, cfg.info, cfg.camera_name)
    camera_pose = load_camera_pose(cfg.pose, cfg.info, cfg.camera_name)
    mask_spec = _load_target_mask_spec(cfg)
    mask = _load_mask(mask_spec, depth.shape)
    segmap = mask.astype(np.int32) * int(cfg.segmap_id)

    ensure_capx_on_path()
    import capx.integrations.vision.graspnet as graspnet

    graspnet.SERVICE_URL = cfg.service_url
    plan_grasp = graspnet.init_contact_graspnet()
    grasps, scores, contact_points = plan_grasp(
        depth.astype(np.float64, copy=False),
        intrinsics.astype(np.float64, copy=False),
        segmap,
        int(cfg.segmap_id),
        z_range=[float(cfg.z_range[0]), float(cfg.z_range[1])],
        forward_passes=int(cfg.forward_passes),
        max_retries=int(cfg.max_retries),
    )

    grasps = np.asarray(grasps, dtype=np.float64)
    if grasps.ndim == 2 and grasps.shape == (4, 4):
        grasps = grasps[None, ...]
    elif grasps.size == 0:
        grasps = np.empty((0, 4, 4), dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    contact_points = np.asarray(contact_points, dtype=np.float64)
    if contact_points.size == 0:
        contact_points = np.empty((0, 3), dtype=np.float64)

    grasp_poses = _apply_grasp_z_offset(grasps, float(cfg.grasp_z_offset))
    best_index: int | None = None
    best_score: float | None = None
    best_grasp_camera: np.ndarray | None = None
    best_pose_camera: np.ndarray | None = None
    best_grasp_world: np.ndarray | None = None
    best_pose_world: np.ndarray | None = None

    if scores.size > 0 and grasps.shape[0] > 0:
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        best_grasp_camera = grasps[best_index]
        best_pose_camera = grasp_poses[best_index]
        if camera_pose is not None:
            best_grasp_world = camera_pose @ best_grasp_camera
            best_pose_world = camera_pose @ best_pose_camera

    result = ContactGraspNetPoseResult(
        grasps_camera=grasps,
        grasp_poses_camera=grasp_poses,
        scores=scores,
        contact_points_camera=contact_points,
        best_index=best_index,
        best_score=best_score,
        best_grasp_camera=best_grasp_camera,
        best_pose_camera=best_pose_camera,
        best_grasp_world=best_grasp_world,
        best_pose_world=best_pose_world,
        mask_pixels=int(mask.sum()),
        image_size=(int(rgb.shape[1]), int(rgb.shape[0])),
        intrinsics=intrinsics,
        camera_pose=camera_pose,
        mask_spec=mask_spec,
    )
    summary_path = save_contact_graspnet_pose_outputs(cfg, result, segmap)
    return result, summary_path


def _matrix_or_none(matrix: np.ndarray | None) -> list[list[float]] | None:
    return None if matrix is None else matrix.astype(float).tolist()


def _append_colored_points(
    point_chunks: list[np.ndarray],
    color_chunks: list[np.ndarray],
    label_chunks: list[np.ndarray],
    points: np.ndarray,
    color: tuple[int, int, int],
    label: int,
) -> None:
    if points.size == 0:
        return
    point_chunks.append(points.astype(np.float64, copy=False))
    color_chunks.append(np.tile(np.asarray(color, dtype=np.uint8), (points.shape[0], 1)))
    label_chunks.append(np.full((points.shape[0],), label, dtype=np.int32))


def save_contact_graspnet_visualization_ply(
    cfg: ContactGraspNetPoseConfig,
    result: ContactGraspNetPoseResult,
    segmap: np.ndarray,
    output_root: Path,
) -> Path:
    rgb = _load_rgb(cfg.rgb)
    depth = _load_depth(
        cfg.depth,
        info_path=cfg.info,
        camera_name=cfg.camera_name,
        depth_format=cfg.depth_format,
    )
    points_image = _depth_to_points_image(depth, result.intrinsics)
    stride = max(1, int(cfg.visualization_subsample_factor))

    points_sub = points_image[::stride, ::stride].reshape(-1, 3)
    rgb_sub = rgb[::stride, ::stride].reshape(-1, 3)
    target_sub = (segmap[::stride, ::stride].reshape(-1) == int(cfg.segmap_id))
    z = points_sub[:, 2]
    valid = np.isfinite(points_sub).all(axis=1) & (z > 0.015) & (z <= max(float(cfg.z_range[1]), 0.015))

    base_points = points_sub[valid]
    base_colors = rgb_sub[valid].astype(np.float64)
    base_target = target_sub[valid]
    base_colors = np.clip(base_colors * 0.55 + 45.0, 0.0, 255.0).astype(np.uint8)
    base_colors[base_target] = np.asarray([255, 96, 48], dtype=np.uint8)
    base_labels = np.where(base_target, 1, 0).astype(np.int32)

    point_chunks: list[np.ndarray] = [base_points]
    color_chunks: list[np.ndarray] = [base_colors]
    label_chunks: list[np.ndarray] = [base_labels]

    top_k = min(max(0, int(cfg.visualization_top_k)), result.grasp_poses_camera.shape[0])
    if top_k > 0 and result.scores.size > 0:
        for idx in np.argsort(-result.scores)[:top_k]:
            idx = int(idx)
            if result.best_index is not None and idx == result.best_index:
                continue
            pose = result.grasp_poses_camera[idx]
            origin = pose[:3, 3]
            z_axis = _sample_line(
                origin,
                origin + pose[:3, 2] * (float(cfg.visualization_axis_length) * 0.55),
                step=0.004,
            )
            _append_colored_points(point_chunks, color_chunks, label_chunks, z_axis, (190, 145, 255), -20)
            _append_colored_points(
                point_chunks,
                color_chunks,
                label_chunks,
                _sample_sphere(origin, float(cfg.visualization_point_radius) * 0.45),
                (210, 130, 255),
                -21,
            )
            if result.contact_points_camera.shape[0] > idx:
                _append_colored_points(
                    point_chunks,
                    color_chunks,
                    label_chunks,
                    _sample_sphere(
                        result.contact_points_camera[idx],
                        float(cfg.visualization_point_radius) * 0.4,
                    ),
                    (255, 170, 48),
                    -22,
                )

    if result.best_pose_camera is not None:
        x_axis, y_axis, z_axis = _sample_pose_axes(
            result.best_pose_camera,
            float(cfg.visualization_axis_length),
        )
        _append_colored_points(point_chunks, color_chunks, label_chunks, x_axis, (255, 45, 45), -10)
        _append_colored_points(point_chunks, color_chunks, label_chunks, y_axis, (45, 230, 80), -11)
        _append_colored_points(point_chunks, color_chunks, label_chunks, z_axis, (55, 120, 255), -12)
        _append_colored_points(
            point_chunks,
            color_chunks,
            label_chunks,
            _sample_sphere(result.best_pose_camera[:3, 3], float(cfg.visualization_point_radius)),
            (255, 255, 255),
            -13,
        )
        if result.best_index is not None and result.contact_points_camera.shape[0] > result.best_index:
            _append_colored_points(
                point_chunks,
                color_chunks,
                label_chunks,
                _sample_sphere(
                    result.contact_points_camera[result.best_index],
                    float(cfg.visualization_point_radius) * 0.8,
                ),
                (255, 230, 35),
                -14,
            )

    visualization_points = np.vstack(point_chunks)
    visualization_colors = np.vstack(color_chunks)
    visualization_labels = np.concatenate(label_chunks)
    visualization_path = output_root / "contact_graspnet_visualization.ply"
    _write_visualization_ply(
        visualization_path,
        visualization_points,
        visualization_colors,
        visualization_labels,
    )
    return visualization_path


VISUALIZATION_LABELS = {
    "0": "rgb_scene_dimmed",
    "1": "target_mask_orange",
    "-10": "best_grasp_x_axis_red",
    "-11": "best_grasp_y_axis_green",
    "-12": "best_grasp_z_axis_blue",
    "-13": "best_grasp_origin_white",
    "-14": "best_contact_point_yellow",
    "-20": "top_candidate_z_axis_purple",
    "-21": "top_candidate_origin_purple",
    "-22": "top_candidate_contact_point_orange",
}

def save_contact_graspnet_pose_outputs(
    cfg: ContactGraspNetPoseConfig,
    result: ContactGraspNetPoseResult,
    segmap: np.ndarray,
) -> Path:
    image_stem = cfg.rgb.stem
    output_root = cfg.output_dir / image_stem
    output_root.mkdir(parents=True, exist_ok=True)

    grasps_path = output_root / "grasps_camera.npy"
    poses_path = output_root / "grasp_poses_camera.npy"
    scores_path = output_root / "scores.npy"
    contact_points_path = output_root / "contact_points_camera.npy"
    segmap_path = output_root / "segmap.npy"
    summary_path = output_root / "contact_graspnet_summary.json"

    np.save(grasps_path, result.grasps_camera)
    np.save(poses_path, result.grasp_poses_camera)
    np.save(scores_path, result.scores)
    np.save(contact_points_path, result.contact_points_camera)
    np.save(segmap_path, segmap.astype(np.int32, copy=False))

    visualization_ply: Path | None = None
    if cfg.save_visualization:
        visualization_ply = save_contact_graspnet_visualization_ply(
            cfg,
            result,
            segmap,
            output_root,
        )

    best: dict[str, Any] | None = None
    if result.best_index is not None and result.best_grasp_camera is not None and result.best_pose_camera is not None:
        best_contact_point = (
            result.contact_points_camera[result.best_index]
            if result.contact_points_camera.shape[0] > result.best_index
            else None
        )
        best = {
            "index": result.best_index,
            "score": result.best_score,
            "grasp_camera": _matrix_or_none(result.best_grasp_camera),
            "pose_camera": _matrix_or_none(result.best_pose_camera),
            "grasp_camera_position_quaternion_wxyz": _matrix_to_position_quaternion_wxyz(
                result.best_grasp_camera
            ),
            "pose_camera_position_quaternion_wxyz": _matrix_to_position_quaternion_wxyz(
                result.best_pose_camera
            ),
            "contact_point_camera": (
                best_contact_point.astype(float).tolist()
                if best_contact_point is not None
                else None
            ),
            "grasp_contact_diagnostics_camera": (
                _pose_contact_diagnostics(result.best_grasp_camera, best_contact_point)
                if best_contact_point is not None
                else None
            ),
            "pose_contact_diagnostics_camera": (
                _pose_contact_diagnostics(result.best_pose_camera, best_contact_point)
                if best_contact_point is not None
                else None
            ),
            "grasp_world": _matrix_or_none(result.best_grasp_world),
            "pose_world": _matrix_or_none(result.best_pose_world),
            "grasp_world_position_quaternion_wxyz": (
                _matrix_to_position_quaternion_wxyz(result.best_grasp_world)
                if result.best_grasp_world is not None
                else None
            ),
            "pose_world_position_quaternion_wxyz": (
                _matrix_to_position_quaternion_wxyz(result.best_pose_world)
                if result.best_pose_world is not None
                else None
            ),
        }

    summary = {
        "config": {
            **asdict(cfg),
            "rgb": str(cfg.rgb),
            "depth": str(cfg.depth),
            "intrinsics": str(cfg.intrinsics) if cfg.intrinsics else None,
            "info": str(cfg.info) if cfg.info else None,
            "output_dir": str(cfg.output_dir),
            "sam3_summary": str(cfg.sam3_summary) if cfg.sam3_summary else None,
            "point_selection_summary": (
                str(cfg.point_selection_summary) if cfg.point_selection_summary else None
            ),
            "mask": str(cfg.mask) if cfg.mask else None,
            "pose": str(cfg.pose) if cfg.pose else None,
            "z_range": list(cfg.z_range),
        },
        "target": {
            "prompt": cfg.target_prompt,
            "mask": {
                **asdict(result.mask_spec),
                "mask": str(result.mask_spec.mask),
            },
            "mask_pixels": result.mask_pixels,
            "segmap_id": cfg.segmap_id,
        },
        "result": {
            "num_candidates": int(result.scores.shape[0]),
            "best": best,
            "image_size": list(result.image_size),
            "intrinsics": result.intrinsics.astype(float).tolist(),
            "camera_pose": _matrix_or_none(result.camera_pose),
            "selection_rule": "highest_score",
            "grasp_z_offset_applied_m": float(cfg.grasp_z_offset),
            "visualization_labels": VISUALIZATION_LABELS,
        },
        "outputs": {
            "grasps_camera": str(grasps_path),
            "grasp_poses_camera": str(poses_path),
            "scores": str(scores_path),
            "contact_points_camera": str(contact_points_path),
            "segmap": str(segmap_path),
            "visualization_ply": str(visualization_ply) if visualization_ply is not None else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path
