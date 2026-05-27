from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text.strip().lower())
    return slug.strip("_") or "prompt"


def overlay_mask(image_np: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    overlay = image_np.copy()
    if mask.shape[:2] != image_np.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image_np.shape}")
    overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
    return overlay


def save_prompt_visualization(
    image: Image.Image,
    prompt: str,
    results: list[dict[str, Any]],
    output_dir: Path,
    image_stem: str,
    top_k: int,
    show: bool,
) -> list[dict[str, Any]]:
    prompt_slug = slugify(prompt)
    prompt_dir = output_dir / image_stem / prompt_slug
    prompt_dir.mkdir(parents=True, exist_ok=True)

    image_np = np.asarray(image)
    color = np.array([30, 144, 255], dtype=np.uint8)
    selected = results[:top_k]
    saved: list[dict[str, Any]] = []

    if not selected:
        return saved

    fig, axes = plt.subplots(1, len(selected) + 1, figsize=(4 * (len(selected) + 1), 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(image)
    axes[0].set_title(f"{prompt}: top {len(selected)}")
    axes[0].axis("off")

    for result in selected:
        x1, y1, x2, y2 = result["box"]
        axes[0].add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        )
        axes[0].text(x1, y1, f"{result['score']:.3f}", color="white", backgroundcolor="red")

    for index, result in enumerate(selected, start=1):
        mask = np.asarray(result["mask"], dtype=bool)
        overlay = overlay_mask(image_np, mask, color)
        score = float(result["score"])
        score_tag = f"{score:.3f}".replace(".", "p")
        prefix = f"{image_stem}_{prompt_slug}_{index:02d}_{score_tag}"

        overlay_path = prompt_dir / f"{prefix}_overlay.png"
        mask_png_path = prompt_dir / f"{prefix}_mask.png"
        mask_npy_path = prompt_dir / f"{prefix}_mask.npy"

        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_png_path)
        np.save(mask_npy_path, mask)

        ax = axes[index]
        ax.imshow(overlay)
        x1, y1, x2, y2 = result["box"]
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="yellow",
                facecolor="none",
            )
        )
        ax.set_title(f"score {score:.3f}")
        ax.axis("off")

        saved.append(
            {
                "rank": index,
                "score": score,
                "label": result.get("label", prompt),
                "box_xyxy": [float(v) for v in result["box"]],
                "mask_pixels": int(mask.sum()),
                "overlay": str(overlay_path),
                "mask_png": str(mask_png_path),
                "mask_npy": str(mask_npy_path),
            }
        )

    fig.tight_layout()
    grid_path = prompt_dir / f"{image_stem}_{prompt_slug}_grid.png"
    fig.savefig(grid_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    for item in saved:
        item["grid"] = str(grid_path)

    return saved
