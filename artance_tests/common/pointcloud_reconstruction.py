from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common.paths import ensure_capx_on_path


DEFAULT_PART_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 57, 70),
    (29, 111, 219),
    (42, 157, 143),
    (245, 166, 35),
    (132, 94, 194),
    (0, 180, 216),
    (255, 99, 146),
    (122, 201, 67),
)


@dataclass(frozen=True)
class PartMaskSpec:
    name: str
    mask: Path
    color_rgb: tuple[int, int, int]
    rank: int | None = None
    score: float | None = None


@dataclass(frozen=True)
class PointCloudReconstructionConfig:
    rgb: Path
    depth: Path
    intrinsics: Path | None
    info: Path | None
    output_dir: Path
    masks: tuple[PartMaskSpec, ...]
    pose: Path | None = None
    sam3_summary: Path | None = None
    point_selection_summary: Path | None = None
    part_analysis_summary: Path | None = None
    camera_name: str = "wrist"
    depth_clip_range: tuple[float, float] = (0.015, 20.0)
    subsample_factor: int = 1
    output_frame: str = "camera"
    background_color_mode: str = "rgb"
    mask_only: bool = False
    mask_overlap_policy: str = "first-wins"
    depth_format: str = "auto"


@dataclass(frozen=True)
class ReconstructedPointCloud:
    points: np.ndarray
    colors_uint8: np.ndarray
    labels: np.ndarray
    part_names: tuple[str, ...]
    valid_pixel_xy: np.ndarray


def _load_json_array(path: Path, key: str | None = None) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if key is not None:
        for part in key.split("."):
            data = data[part]
    return np.asarray(data, dtype=np.float64)


def load_camera_intrinsics(
    intrinsics_path: Path | None,
    info_path: Path | None,
    camera_name: str,
) -> np.ndarray:
    if intrinsics_path is not None:
        suffix = intrinsics_path.suffix.lower()
        if suffix == ".npy":
            intrinsics = np.load(intrinsics_path)
        elif suffix == ".npz":
            loaded = np.load(intrinsics_path)
            intrinsics = loaded["intrinsics"] if "intrinsics" in loaded else loaded[loaded.files[0]]
        elif suffix == ".json":
            intrinsics = _load_json_array(intrinsics_path)
        else:
            intrinsics = np.loadtxt(intrinsics_path)
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
    elif info_path is not None:
        intrinsics = _load_json_array(
            info_path,
            key=f"{camera_name}_camera_misc.{camera_name}_camera_intrinsics",
        )
    else:
        raise ValueError("Either intrinsics or info must be provided.")

    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics must have shape (3, 3), got {intrinsics.shape}")
    return intrinsics


def load_camera_pose(
    pose_path: Path | None,
    info_path: Path | None,
    camera_name: str,
) -> np.ndarray | None:
    if pose_path is not None:
        suffix = pose_path.suffix.lower()
        if suffix == ".npy":
            pose = np.load(pose_path)
        elif suffix == ".npz":
            loaded = np.load(pose_path)
            pose = loaded["pose_mat"] if "pose_mat" in loaded else loaded[loaded.files[0]]
        elif suffix == ".json":
            pose = _load_json_array(pose_path)
        else:
            pose = np.loadtxt(pose_path)
    elif info_path is not None:
        try:
            pose = _load_json_array(
                info_path,
                key=f"{camera_name}_camera_misc.{camera_name}_camera_extrinsics",
            )
        except KeyError:
            return None
    else:
        return None

    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Pose matrix must have shape (4, 4), got {pose.shape}")
    return pose


def load_part_masks_from_sam3_summary(
    summary_path: Path,
    *,
    ranks: tuple[int, ...] = (1,),
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> tuple[PartMaskSpec, ...]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    specs: list[PartMaskSpec] = []

    for prompt_index, (prompt, prompt_info) in enumerate(summary.get("prompts", {}).items()):
        color = (color_map or {}).get(prompt, DEFAULT_PART_COLORS[prompt_index % len(DEFAULT_PART_COLORS)])
        for saved in prompt_info.get("saved_results", []):
            rank = int(saved["rank"])
            if rank not in ranks:
                continue
            mask_path = Path(saved["mask_npy"]).expanduser().resolve()
            label = str(saved.get("label", prompt))
            part_name = label if len(ranks) == 1 else f"{label}_rank_{rank}"
            specs.append(
                PartMaskSpec(
                    name=part_name,
                    mask=mask_path,
                    color_rgb=tuple(int(v) for v in color),
                    rank=rank,
                    score=float(saved["score"]) if "score" in saved else None,
                )
            )

    if not specs:
        raise ValueError(f"No saved SAM3 masks matched ranks {ranks} in {summary_path}")
    return tuple(specs)


def load_part_masks_from_point_selection_summary(
    point_selection_summary_path: Path,
    sam3_summary_path: Path,
    *,
    prompts: tuple[str, ...] | None = None,
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> tuple[PartMaskSpec, ...]:
    point_selection_summary = json.loads(point_selection_summary_path.read_text(encoding="utf-8"))
    sam3_summary = json.loads(sam3_summary_path.read_text(encoding="utf-8"))
    selected_prompts = prompts or tuple(point_selection_summary.get("prompts", {}).keys())
    specs: list[PartMaskSpec] = []

    for prompt_index, prompt in enumerate(selected_prompts):
        prompt_info = point_selection_summary.get("prompts", {}).get(prompt)
        if prompt_info is None:
            raise ValueError(f"Prompt {prompt!r} not found in {point_selection_summary_path}")
        selected = prompt_info.get("selected")
        if selected is None:
            continue

        rank = int(selected["rank"])
        sam3_prompt_info = sam3_summary.get("prompts", {}).get(prompt)
        if sam3_prompt_info is None:
            raise ValueError(f"Prompt {prompt!r} not found in {sam3_summary_path}")

        saved_result = None
        for saved in sam3_prompt_info.get("saved_results", []):
            if int(saved["rank"]) == rank:
                saved_result = saved
                break
        if saved_result is None:
            raise ValueError(
                f"Selected prompt {prompt!r} rank {rank} was not saved in {sam3_summary_path}. "
                "Re-run SAM3 with a large enough --top-k to save that rank."
            )

        color = (color_map or {}).get(prompt, DEFAULT_PART_COLORS[prompt_index % len(DEFAULT_PART_COLORS)])
        label = str(saved_result.get("label", selected.get("label", prompt)))
        specs.append(
            PartMaskSpec(
                name=label,
                mask=Path(saved_result["mask_npy"]).expanduser().resolve(),
                color_rgb=tuple(int(v) for v in color),
                rank=rank,
                score=float(saved_result["score"]) if "score" in saved_result else float(selected["score"]),
            )
        )

    if not specs:
        raise ValueError(f"No selected SAM3 masks found in {point_selection_summary_path}")
    return tuple(specs)


def load_part_masks_from_part_analysis_summary(
    part_analysis_summary_path: Path,
    *,
    roles: tuple[str, ...] | None = None,
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> tuple[PartMaskSpec, ...]:
    summary = json.loads(part_analysis_summary_path.read_text(encoding="utf-8"))
    selected_groups = summary.get("selected_groups", {})
    if not isinstance(selected_groups, dict):
        raise ValueError(f"selected_groups must be a dict in {part_analysis_summary_path}")

    selected_roles = roles or ("interaction_part", "support_part", "object")
    specs: list[PartMaskSpec] = []
    for role_index, role in enumerate(selected_roles):
        selected = selected_groups.get(role)
        if selected is None:
            continue
        mask_path = Path(selected["mask_npy"]).expanduser().resolve()
        prompts = selected.get("prompts") or [role]
        prompt_name = str(prompts[0])
        part_name = {
            "interaction_part": "interaction_part",
            "support_part": "support_part",
            "object": "object",
        }.get(role, role)
        if prompt_name and prompt_name != role:
            part_name = f"{part_name}:{prompt_name}"
        color = (color_map or {}).get(role, DEFAULT_PART_COLORS[role_index % len(DEFAULT_PART_COLORS)])
        specs.append(
            PartMaskSpec(
                name=part_name,
                mask=mask_path,
                color_rgb=tuple(int(v) for v in color),
                rank=None,
                score=float(selected["sam_score"]) if "sam_score" in selected else None,
            )
        )

    if not specs:
        raise ValueError(f"No selected masks found in {part_analysis_summary_path}")
    return tuple(specs)


def _load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"RGB image not found: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _camera_depth_range(info_path: Path | None, camera_name: str) -> tuple[float, float] | None:
    if info_path is None or not info_path.exists():
        return None
    data = json.loads(info_path.read_text(encoding="utf-8"))
    misc = data.get(f"{camera_name}_camera_misc", {})
    near = misc.get(f"{camera_name}_camera_near")
    far = misc.get(f"{camera_name}_camera_far")
    if near is None or far is None:
        return None
    return float(near), float(far)


RLBENCH_DEPTH_SCALE = float(2**24 - 1)


def _decode_rlbench_rgb_depth(image: Image.Image) -> np.ndarray:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float64)
    return np.sum(image_array * np.asarray([65536.0, 256.0, 1.0]), axis=2) / RLBENCH_DEPTH_SCALE


def _load_depth(
    path: Path,
    *,
    info_path: Path | None = None,
    camera_name: str = "wrist",
    depth_format: str = "auto",
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Depth file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        loaded = np.load(path)
        depth = loaded["depth"] if "depth" in loaded else loaded[loaded.files[0]]
    else:
        image = Image.open(path)
        if depth_format not in {"auto", "meters", "rlbench_normalized_png"}:
            raise ValueError(f"Unsupported depth_format: {depth_format!r}")
        if depth_format in {"auto", "rlbench_normalized_png"} and len(np.asarray(image).shape) == 3:
            depth = _decode_rlbench_rgb_depth(image)
            camera_range = _camera_depth_range(info_path, camera_name)
            if camera_range is not None:
                near, far = camera_range
                depth = near + depth * (far - near)
        else:
            depth = np.asarray(image, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D, got shape {depth.shape}")
    return depth.astype(np.float64, copy=False)


def _load_mask(spec: PartMaskSpec, shape_hw: tuple[int, int]) -> np.ndarray:
    if not spec.mask.exists():
        raise FileNotFoundError(f"Mask not found for part {spec.name!r}: {spec.mask}")
    mask = np.load(spec.mask) if spec.mask.suffix.lower() == ".npy" else np.asarray(Image.open(spec.mask))
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape_hw:
        raise ValueError(f"Mask {spec.mask} has shape {mask.shape}, expected {shape_hw}")
    return mask


def reconstruct_pointcloud(cfg: PointCloudReconstructionConfig) -> ReconstructedPointCloud:
    ensure_capx_on_path()
    from capx.utils.depth_utils import depth_to_pointcloud

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
    pose = load_camera_pose(cfg.pose, cfg.info, cfg.camera_name)

    points = depth_to_pointcloud(
        depth,
        intrinsics,
        subsample_factor=cfg.subsample_factor,
        depth_clip_range=cfg.depth_clip_range,
        filter_invalid=False,
    )
    rgb_sub = rgb[:: cfg.subsample_factor, :: cfg.subsample_factor]
    depth_sub = depth[:: cfg.subsample_factor, :: cfg.subsample_factor]
    height, width = depth_sub.shape

    labels_image = np.zeros((height, width), dtype=np.int32)
    part_names = tuple(spec.name for spec in cfg.masks)
    part_colors = np.asarray([spec.color_rgb for spec in cfg.masks], dtype=np.uint8)
    colored_rgb = rgb_sub.copy() if cfg.background_color_mode == "rgb" else np.full_like(rgb_sub, 170)

    if cfg.mask_overlap_policy not in {"first-wins", "last-wins"}:
        raise ValueError(
            "mask_overlap_policy must be 'first-wins' or 'last-wins', "
            f"got {cfg.mask_overlap_policy!r}"
        )

    for label_id, spec in enumerate(cfg.masks, start=1):
        mask = _load_mask(spec, depth.shape)
        mask_sub = mask[:: cfg.subsample_factor, :: cfg.subsample_factor]
        if cfg.mask_overlap_policy == "first-wins":
            write_mask = mask_sub & (labels_image == 0)
        else:
            write_mask = mask_sub
        labels_image[write_mask] = label_id
        colored_rgb[write_mask] = part_colors[label_id - 1]

    flat_points = points.reshape(-1, 3)
    flat_colors = colored_rgb.reshape(-1, 3)
    flat_labels = labels_image.reshape(-1)
    yy, xx = np.indices((height, width))
    flat_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1) * cfg.subsample_factor

    near_clip, far_clip = cfg.depth_clip_range
    valid = (
        ~np.isnan(flat_points).any(axis=1)
        & ~np.isinf(flat_points).any(axis=1)
        & (flat_points[:, 2] >= near_clip)
        & (flat_points[:, 2] <= far_clip)
    )
    if cfg.mask_only:
        valid &= flat_labels > 0
    flat_points = flat_points[valid]
    flat_colors = flat_colors[valid]
    flat_labels = flat_labels[valid]
    flat_xy = flat_xy[valid]

    if cfg.output_frame == "world":
        if pose is None:
            raise ValueError("output_frame='world' requires pose or info with camera extrinsics.")
        points_hom = np.hstack([flat_points, np.ones((flat_points.shape[0], 1), dtype=np.float64)])
        flat_points = (pose @ points_hom.T).T[:, :3]
    elif cfg.output_frame != "camera":
        raise ValueError(f"output_frame must be 'camera' or 'world', got {cfg.output_frame!r}")

    return ReconstructedPointCloud(
        points=flat_points.astype(np.float64, copy=False),
        colors_uint8=flat_colors.astype(np.uint8, copy=False),
        labels=flat_labels.astype(np.int32, copy=False),
        part_names=part_names,
        valid_pixel_xy=flat_xy.astype(np.int32, copy=False),
    )


def write_ascii_ply(path: Path, cloud: ReconstructedPointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {cloud.points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")
        f.write("end_header\n")
        for point, color, label in zip(cloud.points, cloud.colors_uint8, cloud.labels, strict=True):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(label)}\n"
            )


def save_reconstruction_outputs(
    cfg: PointCloudReconstructionConfig,
    cloud: ReconstructedPointCloud,
) -> Path:
    image_stem = cfg.rgb.stem
    output_root = cfg.output_dir / image_stem
    output_root.mkdir(parents=True, exist_ok=True)

    ply_path = output_root / f"{image_stem}_{cfg.output_frame}_sam3_parts.ply"
    npz_path = output_root / f"{image_stem}_{cfg.output_frame}_sam3_parts.npz"
    summary_path = output_root / "pointcloud_summary.json"

    write_ascii_ply(ply_path, cloud)
    np.savez_compressed(
        npz_path,
        points=cloud.points,
        colors_uint8=cloud.colors_uint8,
        labels=cloud.labels,
        valid_pixel_xy=cloud.valid_pixel_xy,
        part_names=np.asarray(cloud.part_names),
    )

    label_counts = {
        "background": int(np.sum(cloud.labels == 0)),
        **{
            name: int(np.sum(cloud.labels == label_id))
            for label_id, name in enumerate(cloud.part_names, start=1)
        },
    }
    summary = {
        "config": {
            **asdict(cfg),
            "rgb": str(cfg.rgb),
            "depth": str(cfg.depth),
            "intrinsics": str(cfg.intrinsics) if cfg.intrinsics else None,
            "info": str(cfg.info) if cfg.info else None,
            "pose": str(cfg.pose) if cfg.pose else None,
            "sam3_summary": str(cfg.sam3_summary) if cfg.sam3_summary else None,
            "point_selection_summary": (
                str(cfg.point_selection_summary) if cfg.point_selection_summary else None
            ),
            "part_analysis_summary": (
                str(cfg.part_analysis_summary) if cfg.part_analysis_summary else None
            ),
            "output_dir": str(cfg.output_dir),
            "masks": [
                {
                    **asdict(spec),
                    "mask": str(spec.mask),
                }
                for spec in cfg.masks
            ],
        },
        "num_points": int(cloud.points.shape[0]),
        "label_ids": {
            "0": "background",
            **{str(label_id): name for label_id, name in enumerate(cloud.part_names, start=1)},
        },
        "label_counts": label_counts,
        "outputs": {
            "ply": str(ply_path),
            "npz": str(npz_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def run_pointcloud_reconstruction(cfg: PointCloudReconstructionConfig) -> tuple[ReconstructedPointCloud, Path]:
    cloud = reconstruct_pointcloud(cfg)
    summary_path = save_reconstruction_outputs(cfg, cloud)
    return cloud, summary_path
