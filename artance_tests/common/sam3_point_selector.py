from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from common.sam3_runner import load_rgb_image, run_sam3_text_prompts
from common.visualization import overlay_mask, slugify


@dataclass(frozen=True)
class PointSelectionConfig:
    image: Path
    point_xy: tuple[int, int]
    prompts: tuple[str, ...]
    output_dir: Path
    service_url: str
    top_k: int = 5
    show: bool = False


@dataclass(frozen=True)
class MaskPointSelection:
    prompt: str
    rank: int
    score: float
    label: str
    box_xyxy: tuple[float, float, float, float]
    mask_pixels: int
    contains_point: bool
    distance_px: float
    nearest_mask_xy: tuple[int, int] | None
    centroid_xy: tuple[float, float] | None


def _validate_point_xy(point_xy: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    x, y = point_xy
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"point_xy {point_xy} is outside image bounds width={width}, height={height}")
    return int(x), int(y)


def _mask_distance_to_point(mask: np.ndarray, point_xy: tuple[int, int]) -> tuple[bool, float, tuple[int, int] | None]:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask_bool.shape}")
    if not mask_bool.any():
        return False, float("inf"), None

    x, y = point_xy
    if mask_bool[y, x]:
        return True, 0.0, (x, y)

    ys, xs = np.nonzero(mask_bool)
    distances_sq = (xs - x) ** 2 + (ys - y) ** 2
    nearest_index = int(np.argmin(distances_sq))
    nearest_xy = (int(xs[nearest_index]), int(ys[nearest_index]))
    return False, float(np.sqrt(distances_sq[nearest_index])), nearest_xy


def _mask_centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def select_nearest_sam3_mask(
    prompt: str,
    results: list[dict[str, Any]],
    point_xy: tuple[int, int],
    image_size: tuple[int, int],
    top_k: int | None = None,
) -> MaskPointSelection | None:
    _validate_point_xy(point_xy, image_size)
    candidates = results[:top_k] if top_k is not None else results
    selections: list[MaskPointSelection] = []

    for rank, result in enumerate(candidates, start=1):
        mask = np.asarray(result["mask"], dtype=bool)
        if mask.shape[:2] != (image_size[1], image_size[0]):
            raise ValueError(
                f"Mask shape {mask.shape} does not match image size {image_size} "
                f"for prompt {prompt!r} rank {rank}"
            )
        contains_point, distance_px, nearest_mask_xy = _mask_distance_to_point(mask, point_xy)
        centroid_xy = _mask_centroid_xy(mask)
        selections.append(
            MaskPointSelection(
                prompt=prompt,
                rank=rank,
                score=float(result.get("score", 0.0)),
                label=str(result.get("label", prompt)),
                box_xyxy=tuple(float(v) for v in result["box"]),
                mask_pixels=int(mask.sum()),
                contains_point=contains_point,
                distance_px=distance_px,
                nearest_mask_xy=nearest_mask_xy,
                centroid_xy=centroid_xy,
            )
        )

    if not selections:
        return None

    return min(
        selections,
        key=lambda item: (
            item.distance_px,
            -int(item.contains_point),
            -item.score,
            item.rank,
        ),
    )


def select_nearest_sam3_masks(
    prompt_results: dict[str, list[dict[str, Any]]],
    point_xy: tuple[int, int],
    image_size: tuple[int, int],
    top_k: int | None = None,
) -> dict[str, MaskPointSelection | None]:
    return {
        prompt: select_nearest_sam3_mask(prompt, results, point_xy, image_size, top_k=top_k)
        for prompt, results in prompt_results.items()
    }


def save_point_selection_visualizations(
    image: Image.Image,
    cfg: PointSelectionConfig,
    prompt_results: dict[str, list[dict[str, Any]]],
    selections: dict[str, MaskPointSelection | None],
) -> dict[str, Any]:
    image_np = np.asarray(image)
    image_stem = cfg.image.stem
    output_root = cfg.output_dir / image_stem
    output_root.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Any] = {}

    for prompt, selection in selections.items():
        prompt_slug = slugify(prompt)
        prompt_dir = output_root / prompt_slug
        prompt_dir.mkdir(parents=True, exist_ok=True)
        results = prompt_results[prompt][: cfg.top_k]

        fig, ax = plt.subplots(figsize=(7, 7))
        if selection is not None:
            selected_result = results[selection.rank - 1]
            mask = np.asarray(selected_result["mask"], dtype=bool)
            ax.imshow(overlay_mask(image_np, mask, np.array([30, 144, 255], dtype=np.uint8)))
        else:
            ax.imshow(image)

        for index, result in enumerate(results, start=1):
            x1, y1, x2, y2 = result["box"]
            is_selected = selection is not None and index == selection.rank
            edgecolor = "lime" if is_selected else "red"
            linewidth = 3 if is_selected else 1.5
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    facecolor="none",
                )
            )
            ax.text(
                x1,
                y1,
                f"{index}:{float(result.get('score', 0.0)):.3f}",
                color="white",
                backgroundcolor=edgecolor,
                fontsize=8,
            )

        point_x, point_y = cfg.point_xy
        ax.scatter([point_x], [point_y], c="yellow", marker="+", s=220, linewidths=3)
        if selection is not None and selection.nearest_mask_xy is not None:
            nearest_x, nearest_y = selection.nearest_mask_xy
            ax.scatter([nearest_x], [nearest_y], c="cyan", marker="x", s=120, linewidths=2)
            ax.plot([point_x, nearest_x], [point_y, nearest_y], color="cyan", linewidth=1.5)

        ax.set_title(f"{prompt}: nearest mask to ({point_x}, {point_y})")
        ax.axis("off")
        fig.tight_layout()

        visualization_path = prompt_dir / f"{image_stem}_{prompt_slug}_point_selection.png"
        fig.savefig(visualization_path, dpi=150)
        if cfg.show:
            plt.show()
        plt.close(fig)

        saved[prompt] = {
            "visualization": str(visualization_path),
            "selection": asdict(selection) if selection is not None else None,
        }

    return saved


def run_sam3_point_selection_test(
    cfg: PointSelectionConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, MaskPointSelection | None], Path]:
    image = load_rgb_image(cfg.image)
    point_xy = _validate_point_xy(cfg.point_xy, image.size)
    prompt_results = run_sam3_text_prompts(image, cfg.prompts, cfg.service_url)
    selections = select_nearest_sam3_masks(prompt_results, point_xy, image.size, top_k=cfg.top_k)
    saved = save_point_selection_visualizations(image, cfg, prompt_results, selections)

    summary_path = cfg.output_dir / cfg.image.stem / "point_selection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            **asdict(cfg),
            "image": str(cfg.image),
            "output_dir": str(cfg.output_dir),
            "prompts": list(cfg.prompts),
        },
        "image_size": list(image.size),
        "prompts": {
            prompt: {
                "num_results": len(results),
                "selected": asdict(selections[prompt]) if selections[prompt] is not None else None,
                "saved": saved[prompt],
            }
            for prompt, results in prompt_results.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return prompt_results, selections, summary_path
