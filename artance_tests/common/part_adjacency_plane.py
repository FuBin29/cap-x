from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PartAdjacencyPlaneConfig:
    pointcloud_npz: Path
    output_dir: Path
    part_a: str = "drawer"
    part_b: str = "drawer handle"
    neighbor_radius: float = 0.025
    max_visualization_points: int = 120000
    plane_extent_scale: float = 1.2
    plane_grid_size: int = 44
    sphere_radius: float = 0.012
    arrow_length: float = 0.06
    arrow_radius: float = 0.003
    source_summary: Path | None = None


@dataclass(frozen=True)
class PartAdjacencyPlaneResult:
    part_a: str
    part_b: str
    part_a_label_id: int
    part_b_label_id: int
    neighbor_radius: float
    part_a_point_count: int
    part_b_point_count: int
    adjacent_part_a_point_count: int
    adjacent_part_b_point_count: int
    adjacent_part_a_center: list[float]
    adjacent_part_b_center: list[float]
    contact_center: list[float]
    plane_centroid: list[float]
    plane_normal: list[float]
    plane_equation: dict[str, float]
    fit: dict[str, float]


def _load_pointcloud_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Pointcloud NPZ not found: {path}")
    loaded = np.load(path)
    required = {"points", "colors_uint8", "labels", "part_names"}
    missing = required.difference(loaded.files)
    if missing:
        raise ValueError(f"Pointcloud NPZ is missing required arrays: {sorted(missing)}")
    return {key: loaded[key] for key in loaded.files}


def _part_label_id(part_names: np.ndarray, name: str) -> int:
    names = [str(item) for item in part_names.tolist()]
    if name not in names:
        raise ValueError(f"Part {name!r} not found. Available parts: {names}")
    return names.index(name) + 1


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


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] < 3:
        raise ValueError(f"At least 3 adjacent part-A points are required, got {points.shape[0]}")
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        raise ValueError("Could not fit a stable plane normal from adjacent points.")
    return centroid, normal / norm, singular_values


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def _sample_plane(
    center: np.ndarray,
    normal: np.ndarray,
    fit_points: np.ndarray,
    extent_scale: float,
    grid_size: int,
) -> np.ndarray:
    u, v = _plane_basis(normal)
    offsets = fit_points - center
    u_coords = offsets @ u
    v_coords = offsets @ v
    u_extent = max(float(np.percentile(np.abs(u_coords), 90)) * extent_scale, 0.02)
    v_extent = max(float(np.percentile(np.abs(v_coords), 90)) * extent_scale, 0.02)
    uu, vv = np.meshgrid(
        np.linspace(-u_extent, u_extent, grid_size),
        np.linspace(-v_extent, v_extent, grid_size),
    )
    return center + uu.reshape(-1, 1) * u + vv.reshape(-1, 1) * v


def _sample_sphere(center: np.ndarray, radius: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
    phi = np.linspace(0.0, np.pi, 14)
    tt, pp = np.meshgrid(theta, phi)
    x = radius * np.sin(pp) * np.cos(tt)
    y = radius * np.sin(pp) * np.sin(tt)
    z = radius * np.cos(pp)
    return center + np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)


def _sample_arrow(
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    u, v = _plane_basis(direction)
    shaft_length = length * 0.72
    head_length = length - shaft_length
    samples: list[np.ndarray] = []

    theta = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
    for t in np.linspace(0.0, shaft_length, 28):
        ring = origin + t * direction + radius * (
            np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
        )
        samples.append(ring)

    head_base = origin + shaft_length * direction
    tip = origin + length * direction
    for alpha in np.linspace(0.0, 1.0, 18):
        ring_radius = radius * 3.2 * (1.0 - alpha)
        center = head_base + alpha * head_length * direction
        if ring_radius <= 1e-9:
            samples.append(tip[None, :])
        else:
            ring = center + ring_radius * (
                np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
            )
            samples.append(ring)
    return np.vstack(samples)


def _write_ascii_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
) -> None:
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


def analyze_part_adjacency_plane(
    cfg: PartAdjacencyPlaneConfig,
) -> tuple[PartAdjacencyPlaneResult, dict[str, Any]]:
    data = _load_pointcloud_npz(cfg.pointcloud_npz)
    points = np.asarray(data["points"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int32)
    colors = np.asarray(data["colors_uint8"], dtype=np.uint8)
    part_names = data["part_names"]

    part_a_label_id = _part_label_id(part_names, cfg.part_a)
    part_b_label_id = _part_label_id(part_names, cfg.part_b)
    part_a_mask = labels == part_a_label_id
    part_b_mask = labels == part_b_label_id
    part_a_points = points[part_a_mask]
    part_b_points = points[part_b_mask]
    if part_a_points.size == 0:
        raise ValueError(f"Part {cfg.part_a!r} has no points in {cfg.pointcloud_npz}")
    if part_b_points.size == 0:
        raise ValueError(f"Part {cfg.part_b!r} has no points in {cfg.pointcloud_npz}")

    adjacent_a_mask_local = _points_within_radius(part_a_points, part_b_points, cfg.neighbor_radius)
    adjacent_b_mask_local = _points_within_radius(part_b_points, part_a_points, cfg.neighbor_radius)
    adjacent_a_points = part_a_points[adjacent_a_mask_local]
    adjacent_b_points = part_b_points[adjacent_b_mask_local]
    if adjacent_b_points.size == 0:
        adjacent_b_points = part_b_points

    plane_centroid, normal, singular_values = _fit_plane_svd(adjacent_a_points)
    adjacent_a_center = adjacent_a_points.mean(axis=0)
    adjacent_b_center = adjacent_b_points.mean(axis=0)
    contact_center = np.vstack([adjacent_a_points, adjacent_b_points]).mean(axis=0)

    toward_b = adjacent_b_center - adjacent_a_center
    if float(np.dot(normal, toward_b)) < 0.0:
        normal = -normal
    d = -float(np.dot(normal, plane_centroid))
    residuals = (adjacent_a_points - plane_centroid) @ normal
    rms_error = float(np.sqrt(np.mean(residuals**2)))
    mean_abs_error = float(np.mean(np.abs(residuals)))

    result = PartAdjacencyPlaneResult(
        part_a=cfg.part_a,
        part_b=cfg.part_b,
        part_a_label_id=part_a_label_id,
        part_b_label_id=part_b_label_id,
        neighbor_radius=float(cfg.neighbor_radius),
        part_a_point_count=int(part_a_points.shape[0]),
        part_b_point_count=int(part_b_points.shape[0]),
        adjacent_part_a_point_count=int(adjacent_a_points.shape[0]),
        adjacent_part_b_point_count=int(adjacent_b_points.shape[0]),
        adjacent_part_a_center=[float(v) for v in adjacent_a_center],
        adjacent_part_b_center=[float(v) for v in adjacent_b_center],
        contact_center=[float(v) for v in contact_center],
        plane_centroid=[float(v) for v in plane_centroid],
        plane_normal=[float(v) for v in normal],
        plane_equation={
            "a": float(normal[0]),
            "b": float(normal[1]),
            "c": float(normal[2]),
            "d": d,
        },
        fit={
            "rms_distance": rms_error,
            "mean_abs_distance": mean_abs_error,
            "max_abs_distance": float(np.max(np.abs(residuals))),
            "singular_values": [float(v) for v in singular_values],
        },
    )
    extras = {
        "points": points,
        "colors": colors,
        "labels": labels,
        "adjacent_a_points": adjacent_a_points,
        "plane_centroid": plane_centroid,
        "normal": normal,
        "contact_center": contact_center,
    }
    return result, extras


def _output_root_for_pointcloud(output_dir: Path, pointcloud_npz: Path) -> Path:
    if pointcloud_npz.parent.parent.name.startswith("variation"):
        return output_dir / pointcloud_npz.parent.parent.name / pointcloud_npz.parent.name
    return output_dir / pointcloud_npz.stem


def save_part_adjacency_plane_outputs(
    cfg: PartAdjacencyPlaneConfig,
    result: PartAdjacencyPlaneResult,
    extras: dict[str, Any],
) -> Path:
    output_root = _output_root_for_pointcloud(cfg.output_dir, cfg.pointcloud_npz)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "part_adjacency_plane_summary.json"
    visualization_ply = output_root / f"{cfg.pointcloud_npz.stem}_adjacency_plane_visualization.ply"

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

    plane_points = _sample_plane(
        np.asarray(extras["contact_center"], dtype=np.float64),
        np.asarray(extras["normal"], dtype=np.float64),
        np.asarray(extras["adjacent_a_points"], dtype=np.float64),
        cfg.plane_extent_scale,
        cfg.plane_grid_size,
    )
    sphere_points = _sample_sphere(np.asarray(extras["contact_center"], dtype=np.float64), cfg.sphere_radius)
    arrow_points = _sample_arrow(
        np.asarray(extras["contact_center"], dtype=np.float64),
        np.asarray(extras["normal"], dtype=np.float64),
        cfg.arrow_length,
        cfg.arrow_radius,
    )
    overlay_points = np.vstack([plane_points, sphere_points, arrow_points])
    overlay_colors = np.vstack(
        [
            np.tile(np.array([[255, 215, 0]], dtype=np.uint8), (plane_points.shape[0], 1)),
            np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (sphere_points.shape[0], 1)),
            np.tile(np.array([[35, 255, 90]], dtype=np.uint8), (arrow_points.shape[0], 1)),
        ]
    )
    overlay_labels = np.concatenate(
        [
            np.full((plane_points.shape[0],), -1, dtype=np.int32),
            np.full((sphere_points.shape[0],), -2, dtype=np.int32),
            np.full((arrow_points.shape[0],), -3, dtype=np.int32),
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
            "output_dir": str(cfg.output_dir),
            "source_summary": str(cfg.source_summary) if cfg.source_summary else None,
        },
        "result": asdict(result),
        "visualization_labels": {
            "-1": "fitted_plane_sample_points_yellow",
            "-2": "contact_center_sphere_white",
            "-3": "normal_arrow_green",
        },
        "outputs": {
            "visualization_ply": str(visualization_ply),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def run_part_adjacency_plane(
    cfg: PartAdjacencyPlaneConfig,
) -> tuple[PartAdjacencyPlaneResult, Path]:
    result, extras = analyze_part_adjacency_plane(cfg)
    summary_path = save_part_adjacency_plane_outputs(cfg, result, extras)
    return result, summary_path
