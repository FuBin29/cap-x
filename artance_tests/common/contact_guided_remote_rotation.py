from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContactGuidedRemoteRotationConfig:
    pointcloud_npz: Path
    contact_summary: Path
    output_dir: Path
    part_a: str = "toilet lid"
    part_b: str = "toilet"
    far_quantile: float = 0.72
    hinge_neighbor_radius: float = 0.035
    collision_threshold: float = 0.012
    collision_check_degrees: float = 12.0
    rotation_degrees: float = 40.0
    waypoint_count: int = 9
    max_collision_sample_points: int = 3000
    max_visualization_points: int = 120000
    source_summary: Path | None = None


@dataclass(frozen=True)
class ContactGuidedRemoteRotationResult:
    part_a: str
    part_b: str
    part_a_label_id: int
    part_b_label_id: int
    contact_pixel_xy: list[int]
    contact_point_3d: list[float]
    contact_point_pixel_distance: float
    far_part_a_point_count: int
    hinge_part_a_point_count: int
    hinge_part_b_point_count: int
    axis_point: list[float]
    axis_direction: list[float]
    axis_fit: dict[str, float | list[float]]
    chosen_rotation_sign: int
    direction_scores: dict[str, dict[str, float]]
    rotation_degrees: float
    waypoint_count: int
    waypoints: list[list[float]]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_pointcloud_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Pointcloud NPZ not found: {path}")
    loaded = np.load(path)
    required = {"points", "colors_uint8", "labels", "part_names", "valid_pixel_xy"}
    missing = required.difference(loaded.files)
    if missing:
        raise ValueError(f"Pointcloud NPZ is missing required arrays: {sorted(missing)}")
    return {key: loaded[key] for key in loaded.files}


def _part_label_id(part_names: np.ndarray, name: str) -> int:
    names = [str(item) for item in part_names.tolist()]
    if name not in names:
        raise ValueError(f"Part {name!r} not found. Available parts: {names}")
    return names.index(name) + 1


def _load_contact_pixel_xy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Contact summary not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    pixel_xy = data.get("result", {}).get("pixel_xy")
    if not (
        isinstance(pixel_xy, list | tuple)
        and len(pixel_xy) == 2
        and all(isinstance(v, int | float) for v in pixel_xy)
    ):
        raise ValueError(f"Could not find numeric result.pixel_xy in {path}")
    return np.asarray([int(round(pixel_xy[0])), int(round(pixel_xy[1]))], dtype=np.int32)


def _points_within_radius(
    query_points: np.ndarray,
    target_points: np.ndarray,
    radius: float,
) -> np.ndarray:
    if query_points.size == 0 or target_points.size == 0:
        return np.zeros((query_points.shape[0],), dtype=bool)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(target_points)
        distances, _ = tree.query(query_points, k=1, distance_upper_bound=radius)
        return np.isfinite(distances) & (distances <= radius)
    except Exception:
        mask = np.zeros((query_points.shape[0],), dtype=bool)
        chunk_size = max(1, int(2_000_000 // max(1, target_points.shape[0])))
        radius_sq = radius * radius
        for start in range(0, query_points.shape[0], chunk_size):
            chunk = query_points[start : start + chunk_size]
            distances_sq = np.sum((chunk[:, None, :] - target_points[None, :, :]) ** 2, axis=2)
            mask[start : start + chunk_size] = np.min(distances_sq, axis=1) <= radius_sq
        return mask


def _nearest_distances(query_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    if query_points.size == 0 or target_points.size == 0:
        return np.full((query_points.shape[0],), np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(target_points).query(query_points, k=1)
        return np.asarray(distances, dtype=np.float64)
    except Exception:
        distances = np.full((query_points.shape[0],), np.inf, dtype=np.float64)
        chunk_size = max(1, int(2_000_000 // max(1, target_points.shape[0])))
        for start in range(0, query_points.shape[0], chunk_size):
            chunk = query_points[start : start + chunk_size]
            distances_sq = np.sum((chunk[:, None, :] - target_points[None, :, :]) ** 2, axis=2)
            distances[start : start + chunk_size] = np.sqrt(np.min(distances_sq, axis=1))
        return distances


def _fit_line_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] < 2:
        raise ValueError(f"At least 2 points are required to fit a rotation axis, got {points.shape[0]}")
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction_norm = np.linalg.norm(direction)
    if direction_norm <= 1e-12:
        raise ValueError("Could not fit a stable axis direction.")
    return centroid, direction / direction_norm, singular_values


def _rotate_points(points: np.ndarray, axis_point: np.ndarray, axis_direction: np.ndarray, angle_rad: float) -> np.ndarray:
    direction = axis_direction / np.linalg.norm(axis_direction)
    rel = points - axis_point
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)
    cross = np.cross(direction[None, :], rel)
    dot = rel @ direction
    rotated_rel = rel * cos_t + cross * sin_t + direction[None, :] * dot[:, None] * (1.0 - cos_t)
    return axis_point + rotated_rel


def _sample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(points.shape[0], size=max_points, replace=False))
    return points[keep]


def _score_rotation_direction(
    part_a_points: np.ndarray,
    part_b_points: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    sign: int,
    check_angle_degrees: float,
    collision_threshold: float,
    max_sample_points: int,
) -> dict[str, float]:
    sampled_a = _sample_points(part_a_points, max_sample_points, seed=1)
    sampled_b = _sample_points(part_b_points, max_sample_points, seed=2)
    rotated = _rotate_points(
        sampled_a,
        axis_point,
        axis_direction,
        np.deg2rad(float(sign) * check_angle_degrees),
    )
    distances = _nearest_distances(rotated, sampled_b)
    below = distances < collision_threshold
    return {
        "min_distance": float(np.min(distances)),
        "mean_distance": float(np.mean(distances)),
        "collision_fraction": float(np.mean(below)),
        "collision_count": float(np.sum(below)),
    }


def _choose_rotation_sign(scores: dict[int, dict[str, float]]) -> int:
    minus = scores[-1]
    plus = scores[1]
    if plus["collision_fraction"] != minus["collision_fraction"]:
        return 1 if plus["collision_fraction"] < minus["collision_fraction"] else -1
    if plus["min_distance"] != minus["min_distance"]:
        return 1 if plus["min_distance"] > minus["min_distance"] else -1
    return 1 if plus["mean_distance"] >= minus["mean_distance"] else -1


def _line_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, direction))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    v /= np.linalg.norm(v)
    return u, v


def _sample_sphere(center: np.ndarray, radius: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
    phi = np.linspace(0.0, np.pi, 14)
    tt, pp = np.meshgrid(theta, phi)
    return center + np.stack(
        [
            radius * np.sin(pp).reshape(-1) * np.cos(tt).reshape(-1),
            radius * np.sin(pp).reshape(-1) * np.sin(tt).reshape(-1),
            radius * np.cos(pp).reshape(-1),
        ],
        axis=1,
    )


def _sample_axis(axis_point: np.ndarray, axis_direction: np.ndarray, length: float, radius: float) -> np.ndarray:
    direction = axis_direction / np.linalg.norm(axis_direction)
    u, v = _line_basis(direction)
    theta = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
    samples: list[np.ndarray] = []
    for t in np.linspace(-0.5 * length, 0.5 * length, 70):
        ring = axis_point + t * direction + radius * (
            np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
        )
        samples.append(ring)
    return np.vstack(samples)


def _write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> None:
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


def analyze_contact_guided_remote_rotation(
    cfg: ContactGuidedRemoteRotationConfig,
) -> tuple[ContactGuidedRemoteRotationResult, dict[str, Any]]:
    data = _load_pointcloud_npz(cfg.pointcloud_npz)
    points = np.asarray(data["points"], dtype=np.float64)
    colors = np.asarray(data["colors_uint8"], dtype=np.uint8)
    labels = np.asarray(data["labels"], dtype=np.int32)
    valid_pixel_xy = np.asarray(data["valid_pixel_xy"], dtype=np.float64)
    part_names = data["part_names"]

    part_a_label_id = _part_label_id(part_names, cfg.part_a)
    part_b_label_id = _part_label_id(part_names, cfg.part_b)
    part_a_mask = labels == part_a_label_id
    part_b_mask = labels == part_b_label_id
    part_a_points = points[part_a_mask]
    part_b_points = points[part_b_mask]
    part_a_pixels = valid_pixel_xy[part_a_mask]
    if part_a_points.size == 0:
        raise ValueError(f"Part {cfg.part_a!r} has no points in {cfg.pointcloud_npz}")
    if part_b_points.size == 0:
        raise ValueError(f"Part {cfg.part_b!r} has no points in {cfg.pointcloud_npz}")

    contact_pixel_xy = _load_contact_pixel_xy(cfg.contact_summary)
    pixel_distances = np.linalg.norm(part_a_pixels - contact_pixel_xy[None, :], axis=1)
    contact_index = int(np.argmin(pixel_distances))
    contact_point = part_a_points[contact_index]
    contact_pixel_distance = float(pixel_distances[contact_index])

    distances_from_contact = np.linalg.norm(part_a_points - contact_point[None, :], axis=1)
    far_threshold = float(np.quantile(distances_from_contact, cfg.far_quantile))
    far_part_a_points = part_a_points[distances_from_contact >= far_threshold]
    hinge_a_mask_local = _points_within_radius(far_part_a_points, part_b_points, cfg.hinge_neighbor_radius)
    hinge_a_points = far_part_a_points[hinge_a_mask_local]
    if hinge_a_points.shape[0] < 2:
        hinge_a_points = far_part_a_points

    hinge_b_mask_local = _points_within_radius(part_b_points, hinge_a_points, cfg.hinge_neighbor_radius)
    hinge_b_points = part_b_points[hinge_b_mask_local]
    if hinge_b_points.shape[0] < 2:
        nearest_count = min(max(2, hinge_a_points.shape[0]), part_b_points.shape[0])
        b_center = hinge_a_points.mean(axis=0)
        nearest_order = np.argsort(np.linalg.norm(part_b_points - b_center[None, :], axis=1))
        hinge_b_points = part_b_points[nearest_order[:nearest_count]]

    axis_fit_points = np.vstack([hinge_a_points, hinge_b_points])
    axis_point, axis_direction, singular_values = _fit_line_svd(axis_fit_points)

    if float(np.dot(axis_direction, contact_point - axis_point)) < 0.0:
        axis_direction = -axis_direction

    direction_scores_by_int = {
        -1: _score_rotation_direction(
            part_a_points,
            part_b_points,
            axis_point,
            axis_direction,
            -1,
            cfg.collision_check_degrees,
            cfg.collision_threshold,
            cfg.max_collision_sample_points,
        ),
        1: _score_rotation_direction(
            part_a_points,
            part_b_points,
            axis_point,
            axis_direction,
            1,
            cfg.collision_check_degrees,
            cfg.collision_threshold,
            cfg.max_collision_sample_points,
        ),
    }
    chosen_sign = _choose_rotation_sign(direction_scores_by_int)

    angles = np.linspace(0.0, np.deg2rad(chosen_sign * cfg.rotation_degrees), cfg.waypoint_count)
    waypoints = [
        _rotate_points(contact_point[None, :], axis_point, axis_direction, float(angle))[0]
        for angle in angles
    ]

    centered = axis_fit_points - axis_point
    residual_vectors = centered - np.outer(centered @ axis_direction, axis_direction)
    residuals = np.linalg.norm(residual_vectors, axis=1)
    result = ContactGuidedRemoteRotationResult(
        part_a=cfg.part_a,
        part_b=cfg.part_b,
        part_a_label_id=part_a_label_id,
        part_b_label_id=part_b_label_id,
        contact_pixel_xy=[int(contact_pixel_xy[0]), int(contact_pixel_xy[1])],
        contact_point_3d=[float(v) for v in contact_point],
        contact_point_pixel_distance=contact_pixel_distance,
        far_part_a_point_count=int(far_part_a_points.shape[0]),
        hinge_part_a_point_count=int(hinge_a_points.shape[0]),
        hinge_part_b_point_count=int(hinge_b_points.shape[0]),
        axis_point=[float(v) for v in axis_point],
        axis_direction=[float(v) for v in axis_direction],
        axis_fit={
            "rms_distance": float(np.sqrt(np.mean(residuals**2))),
            "mean_abs_distance": float(np.mean(residuals)),
            "max_abs_distance": float(np.max(residuals)),
            "singular_values": [float(v) for v in singular_values],
        },
        chosen_rotation_sign=int(chosen_sign),
        direction_scores={str(key): value for key, value in direction_scores_by_int.items()},
        rotation_degrees=float(chosen_sign * cfg.rotation_degrees),
        waypoint_count=int(cfg.waypoint_count),
        waypoints=[[float(v) for v in point] for point in waypoints],
    )
    extras = {
        "points": points,
        "colors": colors,
        "labels": labels,
        "contact_point": contact_point,
        "far_part_a_points": far_part_a_points,
        "hinge_a_points": hinge_a_points,
        "hinge_b_points": hinge_b_points,
        "axis_point": axis_point,
        "axis_direction": axis_direction,
        "waypoints": np.asarray(waypoints, dtype=np.float64),
        "axis_fit_points": axis_fit_points,
    }
    return result, extras


def _output_root_for_pointcloud(output_dir: Path, pointcloud_npz: Path) -> Path:
    if pointcloud_npz.parent.parent.name.startswith("variation"):
        return output_dir / pointcloud_npz.parent.parent.name / pointcloud_npz.parent.name
    return output_dir / pointcloud_npz.stem


def save_contact_guided_remote_rotation_outputs(
    cfg: ContactGuidedRemoteRotationConfig,
    result: ContactGuidedRemoteRotationResult,
    extras: dict[str, Any],
) -> Path:
    output_root = _output_root_for_pointcloud(cfg.output_dir, cfg.pointcloud_npz)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "contact_guided_remote_rotation_summary.json"
    visualization_ply = output_root / f"{cfg.pointcloud_npz.stem}_remote_rotation_visualization.ply"

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

    axis_fit_points = np.asarray(extras["axis_fit_points"], dtype=np.float64)
    axis_extent = max(float(np.ptp(axis_fit_points @ np.asarray(extras["axis_direction"]))), 0.12)
    axis_points = _sample_axis(
        np.asarray(extras["axis_point"], dtype=np.float64),
        np.asarray(extras["axis_direction"], dtype=np.float64),
        axis_extent * 1.25,
        0.0025,
    )
    contact_sphere = _sample_sphere(np.asarray(extras["contact_point"], dtype=np.float64), 0.01)
    waypoint_points = np.asarray(extras["waypoints"], dtype=np.float64)
    waypoint_spheres = np.vstack([_sample_sphere(point, 0.006) for point in waypoint_points])
    hinge_a_points = _sample_points(np.asarray(extras["hinge_a_points"], dtype=np.float64), 5000, seed=3)
    hinge_b_points = _sample_points(np.asarray(extras["hinge_b_points"], dtype=np.float64), 5000, seed=4)

    overlay_points = np.vstack([axis_points, contact_sphere, waypoint_spheres, hinge_a_points, hinge_b_points])
    overlay_colors = np.vstack(
        [
            np.tile(np.array([[20, 120, 255]], dtype=np.uint8), (axis_points.shape[0], 1)),
            np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (contact_sphere.shape[0], 1)),
            np.tile(np.array([[0, 245, 255]], dtype=np.uint8), (waypoint_spheres.shape[0], 1)),
            np.tile(np.array([[255, 0, 180]], dtype=np.uint8), (hinge_a_points.shape[0], 1)),
            np.tile(np.array([[255, 145, 0]], dtype=np.uint8), (hinge_b_points.shape[0], 1)),
        ]
    )
    overlay_labels = np.concatenate(
        [
            np.full((axis_points.shape[0],), -1, dtype=np.int32),
            np.full((contact_sphere.shape[0],), -2, dtype=np.int32),
            np.full((waypoint_spheres.shape[0],), -3, dtype=np.int32),
            np.full((hinge_a_points.shape[0],), -4, dtype=np.int32),
            np.full((hinge_b_points.shape[0],), -5, dtype=np.int32),
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
            "contact_summary": str(cfg.contact_summary),
            "output_dir": str(cfg.output_dir),
            "source_summary": str(cfg.source_summary) if cfg.source_summary else None,
        },
        "result": asdict(result),
        "visualization_labels": {
            "-1": "rotation_axis_blue",
            "-2": "contact_point_a_white",
            "-3": "rotated_waypoints_cyan",
            "-4": "far_neighbor_part_a_points_magenta",
            "-5": "neighbor_part_b_points_orange",
        },
        "outputs": {
            "visualization_ply": str(visualization_ply),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary_path


def run_contact_guided_remote_rotation(
    cfg: ContactGuidedRemoteRotationConfig,
) -> tuple[ContactGuidedRemoteRotationResult, Path]:
    result, extras = analyze_contact_guided_remote_rotation(cfg)
    summary_path = save_contact_guided_remote_rotation_outputs(cfg, result, extras)
    return result, summary_path
