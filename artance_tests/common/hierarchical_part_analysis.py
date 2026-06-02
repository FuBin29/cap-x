from __future__ import annotations

import json
import math
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
from common.visualization import overlay_mask, save_prompt_visualization, slugify
from common.vlm_interaction_grounding import (
    GROUNDING_ROLES,
    VlmInteractionGroundingResult,
)


ROLE_PRIORITY = {
    "interaction_part": 3.0,
    "support_part": 2.0,
    "object": 1.0,
    "context": 0.0,
}

ROLE_BASE_SCORE = {
    "interaction_part": 1.0,
    "support_part": 0.7,
    "object": 0.5,
    "context": 0.2,
}

ROLE_COLORS = {
    "interaction_part": (230, 57, 70),
    "support_part": (29, 111, 219),
    "object": (42, 157, 143),
    "context": (160, 160, 160),
}


@dataclass(frozen=True)
class PartAnalysisWeights:
    sam: float = 1.0
    role: float = 0.8
    prompt_support: float = 0.4
    contact: float = 1.4
    part_support_containment: float = 1.5
    support_object_containment: float = 1.0
    part_object_containment: float = 0.8
    support_contact: float = 0.25


@dataclass(frozen=True)
class HierarchicalPartAnalysisConfig:
    image: Path
    output_dir: Path
    grounding: VlmInteractionGroundingResult
    service_url: str
    top_k: int = 5
    show: bool = False
    group_iou_threshold: float = 0.7
    dilation_px: int = 8
    contact_sigma_scale: float = 0.25
    weights: PartAnalysisWeights = PartAnalysisWeights()


@dataclass
class Sam3MaskCandidate:
    role: str
    prompt: str
    rank: int
    score: float
    label: str
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray
    area: int
    bbox_xyxy: tuple[int, int, int, int] | None
    contains_contact: bool
    contact_distance_px: float
    nearest_contact_mask_xy: tuple[int, int] | None
    contact_score: float


@dataclass
class CandidateGroup:
    group_id: str
    members: list[Sam3MaskCandidate]
    mask: np.ndarray
    roles: tuple[str, ...]
    primary_role: str
    prompts: tuple[str, ...]
    area: int
    bbox_xyxy: tuple[int, int, int, int] | None
    sam_score: float
    support_count: int
    contains_contact: bool
    contact_distance_px: float
    nearest_contact_mask_xy: tuple[int, int] | None
    contact_score: float
    base_score: float


@dataclass(frozen=True)
class SelectedMaskGroup:
    role: str
    group_id: str
    prompts: tuple[str, ...]
    ranks: tuple[int, ...]
    sam_score: float
    base_score: float
    contact_score: float
    contact_distance_px: float
    hierarchy_score: float
    original_mask_pixels: int
    final_mask_pixels: int
    mask_npy: Path
    mask_png: Path
    overlay_png: Path


@dataclass(frozen=True)
class HierarchicalPartAnalysisResult:
    contact_pixel_xy: tuple[int, int]
    selected_score: float
    selected_groups: dict[str, SelectedMaskGroup]
    overlay_png: Path
    summary_path: Path


def _validate_role(role: str) -> str:
    if role not in ROLE_PRIORITY:
        raise ValueError(f"Unsupported grounding role {role!r}; expected one of {tuple(ROLE_PRIORITY)}")
    return role


def flatten_grounding_prompts(grounding: VlmInteractionGroundingResult) -> tuple[str, ...]:
    prompts: list[str] = []
    seen: set[str] = set()
    for role in GROUNDING_ROLES:
        for prompt in grounding.sam3_prompts.get(role, ()):
            key = prompt.lower()
            if key in seen:
                continue
            seen.add(key)
            prompts.append(prompt)
    return tuple(prompts)


def prompt_roles_from_grounding(grounding: VlmInteractionGroundingResult) -> dict[str, tuple[str, ...]]:
    prompt_roles: dict[str, list[str]] = {}
    for role in GROUNDING_ROLES:
        for prompt in grounding.sam3_prompts.get(role, ()):
            prompt_roles.setdefault(prompt, []).append(role)
    return {prompt: tuple(roles) for prompt, roles in prompt_roles.items()}


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _mask_distance_to_point(mask: np.ndarray, point_xy: tuple[int, int]) -> tuple[bool, float, tuple[int, int] | None]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        return False, float("inf"), None

    x, y = point_xy
    if mask_bool[y, x]:
        return True, 0.0, (x, y)

    ys, xs = np.nonzero(mask_bool)
    distances_sq = (xs - x) ** 2 + (ys - y) ** 2
    nearest_index = int(np.argmin(distances_sq))
    nearest_xy = (int(xs[nearest_index]), int(ys[nearest_index]))
    return False, float(math.sqrt(float(distances_sq[nearest_index]))), nearest_xy


def _contact_score(distance_px: float, area: int, bbox_xyxy: tuple[int, int, int, int] | None, sigma_scale: float) -> float:
    if area <= 0 or math.isinf(distance_px):
        return 0.0
    if bbox_xyxy is None:
        sigma = sigma_scale * math.sqrt(float(area))
    else:
        x1, y1, x2, y2 = bbox_xyxy
        sigma = sigma_scale * max(1.0, float(max(x2 - x1, y2 - y1)), math.sqrt(float(area)))
    sigma = max(1.0, sigma)
    score = math.exp(-distance_px / sigma)
    if distance_px == 0:
        score += 0.25
    return float(min(score, 1.25))


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(intersection / union) if union else 0.0


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius_px <= 0:
        return mask_bool
    padded = np.pad(mask_bool, radius_px, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask_bool)
    height, width = mask_bool.shape
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            dilated |= padded[
                radius_px + dy : radius_px + dy + height,
                radius_px + dx : radius_px + dx + width,
            ]
    return dilated


def _containment(inner: np.ndarray, outer: np.ndarray, dilation_px: int) -> float:
    inner_bool = np.asarray(inner, dtype=bool)
    inner_area = int(inner_bool.sum())
    if inner_area == 0:
        return 0.0
    outer_dilated = _dilate_mask(outer, dilation_px)
    return float(np.logical_and(inner_bool, outer_dilated).sum() / inner_area)


def _primary_role(roles: tuple[str, ...]) -> str:
    return max(roles, key=lambda role: ROLE_PRIORITY[role])


def _candidate_base_score(group: CandidateGroup, weights: PartAnalysisWeights) -> float:
    return float(
        weights.sam * group.sam_score
        + weights.role * ROLE_BASE_SCORE[group.primary_role]
        + weights.prompt_support * math.log1p(group.support_count)
    )


def _geometry_members_for_group(members: list[Sam3MaskCandidate]) -> list[Sam3MaskCandidate]:
    non_context_members = [member for member in members if member.role != "context"]
    return non_context_members or members


def build_sam3_candidates(
    prompt_results: dict[str, list[dict[str, Any]]],
    prompt_roles: dict[str, tuple[str, ...]],
    *,
    image_size: tuple[int, int],
    contact_pixel_xy: tuple[int, int],
    top_k: int,
    contact_sigma_scale: float,
) -> list[Sam3MaskCandidate]:
    width, height = image_size
    candidates: list[Sam3MaskCandidate] = []
    for prompt, results in prompt_results.items():
        roles = prompt_roles.get(prompt, ())
        if not roles:
            continue
        for rank, result in enumerate(results[:top_k], start=1):
            mask = np.asarray(result["mask"], dtype=bool)
            if mask.shape != (height, width):
                raise ValueError(
                    f"Mask for prompt {prompt!r} rank {rank} has shape {mask.shape}, "
                    f"expected {(height, width)}"
                )
            area = int(mask.sum())
            bbox = _bbox_from_mask(mask)
            contains_contact, distance_px, nearest_xy = _mask_distance_to_point(mask, contact_pixel_xy)
            contact_score = _contact_score(distance_px, area, bbox, contact_sigma_scale)
            for role in roles:
                candidates.append(
                    Sam3MaskCandidate(
                        role=_validate_role(role),
                        prompt=prompt,
                        rank=rank,
                        score=float(result.get("score", 0.0)),
                        label=str(result.get("label", prompt)),
                        box_xyxy=tuple(float(v) for v in result.get("box", (0, 0, 0, 0))),
                        mask=mask,
                        area=area,
                        bbox_xyxy=bbox,
                        contains_contact=contains_contact,
                        contact_distance_px=distance_px,
                        nearest_contact_mask_xy=nearest_xy,
                        contact_score=contact_score,
                    )
                )
    return candidates


def group_candidates_by_iou(
    candidates: list[Sam3MaskCandidate],
    *,
    iou_threshold: float,
    contact_pixel_xy: tuple[int, int],
    contact_sigma_scale: float,
    weights: PartAnalysisWeights,
) -> list[CandidateGroup]:
    groups: list[list[Sam3MaskCandidate]] = []
    for candidate in candidates:
        placed = False
        for group_members in groups:
            if any(_mask_iou(candidate.mask, member.mask) >= iou_threshold for member in group_members):
                group_members.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])

    candidate_groups: list[CandidateGroup] = []
    for index, members in enumerate(groups, start=1):
        geometry_members = _geometry_members_for_group(members)
        merged_mask = np.logical_or.reduce([member.mask for member in geometry_members])
        roles = tuple(sorted({member.role for member in members}, key=lambda role: -ROLE_PRIORITY[role]))
        prompts = tuple(dict.fromkeys(member.prompt for member in geometry_members))
        bbox = _bbox_from_mask(merged_mask)
        area = int(merged_mask.sum())
        contains_contact, distance_px, nearest_xy = _mask_distance_to_point(merged_mask, contact_pixel_xy)
        contact_score = _contact_score(distance_px, area, bbox, contact_sigma_scale)
        group = CandidateGroup(
            group_id=f"group_{index:03d}",
            members=members,
            mask=merged_mask,
            roles=roles,
            primary_role=_primary_role(roles),
            prompts=prompts,
            area=area,
            bbox_xyxy=bbox,
            sam_score=float(max(member.score for member in geometry_members)),
            support_count=len({(member.role, member.prompt, member.rank) for member in geometry_members}),
            contains_contact=contains_contact,
            contact_distance_px=distance_px,
            nearest_contact_mask_xy=nearest_xy,
            contact_score=contact_score,
            base_score=0.0,
        )
        group.base_score = _candidate_base_score(group, weights)
        candidate_groups.append(group)
    return candidate_groups


def _has_role(group: CandidateGroup, role: str) -> bool:
    return role in group.roles


def _score_tuple(
    object_group: CandidateGroup | None,
    support_group: CandidateGroup | None,
    interaction_group: CandidateGroup,
    *,
    dilation_px: int,
    weights: PartAnalysisWeights,
) -> tuple[float, dict[str, float]]:
    score = interaction_group.base_score + weights.contact * interaction_group.contact_score
    details = {
        "interaction_contact": weights.contact * interaction_group.contact_score,
        "part_support_containment": 0.0,
        "support_object_containment": 0.0,
        "part_object_containment": 0.0,
        "support_contact": 0.0,
    }
    if support_group is not None:
        containment = _containment(interaction_group.mask, support_group.mask, dilation_px)
        support_contact = weights.support_contact * support_group.contact_score
        score += support_group.base_score + weights.part_support_containment * containment + support_contact
        details["part_support_containment"] = weights.part_support_containment * containment
        details["support_contact"] = support_contact
    if object_group is not None:
        score += object_group.base_score
        part_object = _containment(interaction_group.mask, object_group.mask, dilation_px)
        score += weights.part_object_containment * part_object
        details["part_object_containment"] = weights.part_object_containment * part_object
        if support_group is not None:
            support_object = _containment(support_group.mask, object_group.mask, dilation_px)
            score += weights.support_object_containment * support_object
            details["support_object_containment"] = weights.support_object_containment * support_object
    return float(score), details


def select_best_hierarchy(
    groups: list[CandidateGroup],
    *,
    dilation_px: int,
    weights: PartAnalysisWeights,
) -> tuple[CandidateGroup | None, CandidateGroup | None, CandidateGroup, float, dict[str, float]]:
    interaction_candidates = [group for group in groups if _has_role(group, "interaction_part")]
    if not interaction_candidates:
        non_context = [group for group in groups if group.primary_role != "context"]
        interaction_candidates = non_context or groups
    if not interaction_candidates:
        raise ValueError("No SAM3 mask candidates were available for hierarchical part analysis.")

    support_candidates = [group for group in groups if _has_role(group, "support_part")]
    object_candidates = [group for group in groups if _has_role(group, "object")]
    support_options: list[CandidateGroup | None] = [None, *support_candidates]
    object_options: list[CandidateGroup | None] = [None, *object_candidates]

    best: tuple[CandidateGroup | None, CandidateGroup | None, CandidateGroup] | None = None
    best_score = -float("inf")
    best_details: dict[str, float] = {}

    for interaction_group in interaction_candidates:
        for support_group in support_options:
            if support_group is interaction_group:
                continue
            for object_group in object_options:
                if object_group is interaction_group or object_group is support_group:
                    continue
                score, details = _score_tuple(
                    object_group,
                    support_group,
                    interaction_group,
                    dilation_px=dilation_px,
                    weights=weights,
                )
                if score > best_score:
                    best_score = score
                    best = (object_group, support_group, interaction_group)
                    best_details = details

    if best is None:
        interaction_group = max(interaction_candidates, key=lambda group: group.base_score + group.contact_score)
        best = (None, None, interaction_group)
        best_score, best_details = _score_tuple(
            None,
            None,
            interaction_group,
            dilation_px=dilation_px,
            weights=weights,
        )
    return best[0], best[1], best[2], float(best_score), best_details


def _contact_heatmap(shape_hw: tuple[int, int], point_xy: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    x, y = point_xy
    yy, xx = np.indices((height, width))
    sigma = max(12.0, 0.12 * max(height, width))
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    return np.exp(-dist_sq / (2.0 * sigma * sigma))


def resolve_overlap_by_role_and_contact(
    selected: dict[str, CandidateGroup],
    *,
    contact_pixel_xy: tuple[int, int],
) -> dict[str, np.ndarray]:
    if not selected:
        return {}
    first = next(iter(selected.values()))
    heatmap = _contact_heatmap(first.mask.shape, contact_pixel_xy)
    role_items = list(selected.items())
    score_stack: list[np.ndarray] = []

    for role, group in role_items:
        score = np.full(group.mask.shape, -np.inf, dtype=np.float64)
        base = group.sam_score + ROLE_PRIORITY[role]
        score[group.mask] = base
        if role == "interaction_part":
            score[group.mask] += heatmap[group.mask]
        elif role == "support_part":
            score[group.mask] += 0.3 * heatmap[group.mask]
        score_stack.append(score)

    stacked = np.stack(score_stack, axis=0)
    owner = np.argmax(stacked, axis=0)
    valid = np.max(stacked, axis=0) > -np.inf
    return {
        role: (owner == index) & valid
        for index, (role, _group) in enumerate(role_items)
    }


def _candidate_to_summary(candidate: Sam3MaskCandidate) -> dict[str, Any]:
    return {
        "role": candidate.role,
        "prompt": candidate.prompt,
        "rank": candidate.rank,
        "score": candidate.score,
        "label": candidate.label,
        "box_xyxy": list(candidate.box_xyxy),
        "area": candidate.area,
        "bbox_xyxy": list(candidate.bbox_xyxy) if candidate.bbox_xyxy else None,
        "contains_contact": candidate.contains_contact,
        "contact_distance_px": candidate.contact_distance_px,
        "nearest_contact_mask_xy": list(candidate.nearest_contact_mask_xy)
        if candidate.nearest_contact_mask_xy
        else None,
        "contact_score": candidate.contact_score,
    }


def _group_to_summary(group: CandidateGroup) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "roles": list(group.roles),
        "primary_role": group.primary_role,
        "prompts": list(group.prompts),
        "area": group.area,
        "bbox_xyxy": list(group.bbox_xyxy) if group.bbox_xyxy else None,
        "sam_score": group.sam_score,
        "support_count": group.support_count,
        "contains_contact": group.contains_contact,
        "contact_distance_px": group.contact_distance_px,
        "nearest_contact_mask_xy": list(group.nearest_contact_mask_xy)
        if group.nearest_contact_mask_xy
        else None,
        "contact_score": group.contact_score,
        "base_score": group.base_score,
        "members": [_candidate_to_summary(member) for member in group.members],
    }


def save_hierarchical_sam3_outputs(
    image: Image.Image,
    cfg: HierarchicalPartAnalysisConfig,
    prompt_results: dict[str, list[dict[str, Any]]],
    prompt_roles: dict[str, tuple[str, ...]],
) -> Path:
    image_stem = cfg.image.stem
    summary_path = cfg.output_dir / image_stem / "hierarchical_sam3_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    role_summary: dict[str, dict[str, Any]] = {role: {} for role in GROUNDING_ROLES}
    for prompt, roles in prompt_roles.items():
        for role in roles:
            saved = save_prompt_visualization(
                image=image,
                prompt=prompt,
                results=prompt_results.get(prompt, []),
                output_dir=cfg.output_dir / image_stem / "sam3" / role,
                image_stem=image_stem,
                top_k=cfg.top_k,
                show=cfg.show,
            )
            role_summary[role][prompt] = {
                "num_results": len(prompt_results.get(prompt, [])),
                "saved_results": saved,
            }

    summary = {
        "config": {
            "image": str(cfg.image),
            "output_dir": str(cfg.output_dir),
            "service_url": cfg.service_url,
            "top_k": cfg.top_k,
            "group_iou_threshold": cfg.group_iou_threshold,
            "dilation_px": cfg.dilation_px,
            "contact_sigma_scale": cfg.contact_sigma_scale,
            "weights": asdict(cfg.weights),
        },
        "grounding": asdict(cfg.grounding),
        "roles": role_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _save_final_masks_and_overlay(
    image: Image.Image,
    cfg: HierarchicalPartAnalysisConfig,
    selected_groups: dict[str, CandidateGroup],
    final_masks: dict[str, np.ndarray],
    selected_score: float,
    hierarchy_details: dict[str, float],
    groups: list[CandidateGroup],
    sam3_summary_path: Path,
) -> tuple[dict[str, SelectedMaskGroup], Path, Path]:
    image_np = np.asarray(image)
    image_stem = cfg.image.stem
    output_root = cfg.output_dir / image_stem
    mask_root = output_root / "selected_masks"
    mask_root.mkdir(parents=True, exist_ok=True)

    selected_summary: dict[str, SelectedMaskGroup] = {}
    role_hierarchy_scores = {
        "interaction_part": float(
            hierarchy_details.get("part_support_containment", 0.0)
            + hierarchy_details.get("part_object_containment", 0.0)
        ),
        "support_part": float(hierarchy_details.get("support_object_containment", 0.0)),
        "object": 0.0,
    }
    overlay = image_np.copy()
    for role in ("object", "support_part", "interaction_part"):
        if role not in selected_groups:
            continue
        group = selected_groups[role]
        mask = final_masks.get(role, np.zeros(group.mask.shape, dtype=bool))
        color = np.asarray(ROLE_COLORS[role], dtype=np.uint8)
        overlay = overlay_mask(overlay, mask, color)

        role_slug = slugify(role)
        mask_npy = mask_root / f"{image_stem}_{role_slug}_mask.npy"
        mask_png = mask_root / f"{image_stem}_{role_slug}_mask.png"
        overlay_png = mask_root / f"{image_stem}_{role_slug}_overlay.png"
        np.save(mask_npy, mask)
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_png)
        Image.fromarray(overlay_mask(image_np, mask, color)).save(overlay_png)

        selected_summary[role] = SelectedMaskGroup(
            role=role,
            group_id=group.group_id,
            prompts=group.prompts,
            ranks=tuple(member.rank for member in group.members),
            sam_score=group.sam_score,
            base_score=group.base_score,
            contact_score=group.contact_score,
            contact_distance_px=group.contact_distance_px,
            hierarchy_score=role_hierarchy_scores.get(role, 0.0),
            original_mask_pixels=group.area,
            final_mask_pixels=int(mask.sum()),
            mask_npy=mask_npy,
            mask_png=mask_png,
            overlay_png=overlay_png,
        )

    contact_x, contact_y = cfg.grounding.pixel_xy
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(overlay)
    ax.scatter([contact_x], [contact_y], c="yellow", marker="+", s=220, linewidths=3)
    for role, group in selected_groups.items():
        bbox = group.bbox_xyxy
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        color = np.asarray(ROLE_COLORS[role]) / 255.0
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        )
        ax.text(x1, y1, role, color="white", backgroundcolor=color, fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    overlay_png = output_root / "part_analysis_overlay.png"
    fig.savefig(overlay_png, dpi=150)
    if cfg.show:
        plt.show()
    plt.close(fig)

    summary_path = output_root / "part_analysis_summary.json"
    summary = {
        "config": {
            "image": str(cfg.image),
            "output_dir": str(cfg.output_dir),
            "service_url": cfg.service_url,
            "top_k": cfg.top_k,
            "group_iou_threshold": cfg.group_iou_threshold,
            "dilation_px": cfg.dilation_px,
            "contact_sigma_scale": cfg.contact_sigma_scale,
            "weights": asdict(cfg.weights),
        },
        "grounding": asdict(cfg.grounding),
        "contact_pixel_xy": list(cfg.grounding.pixel_xy),
        "selected_score": selected_score,
        "hierarchy_details": hierarchy_details,
        "selected_groups": {
            role: {
                **asdict(selected),
                "mask_npy": str(selected.mask_npy),
                "mask_png": str(selected.mask_png),
                "overlay_png": str(selected.overlay_png),
            }
            for role, selected in selected_summary.items()
        },
        "candidate_groups": [_group_to_summary(group) for group in groups],
        "outputs": {
            "overlay_png": str(overlay_png),
            "sam3_summary": str(sam3_summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return selected_summary, overlay_png, summary_path


def run_hierarchical_part_analysis(
    cfg: HierarchicalPartAnalysisConfig,
) -> tuple[HierarchicalPartAnalysisResult, dict[str, list[dict[str, Any]]]]:
    image = load_rgb_image(cfg.image)
    prompt_roles = prompt_roles_from_grounding(cfg.grounding)
    prompts = flatten_grounding_prompts(cfg.grounding)
    prompt_results = run_sam3_text_prompts(image, prompts, cfg.service_url)
    sam3_summary_path = save_hierarchical_sam3_outputs(image, cfg, prompt_results, prompt_roles)

    candidates = build_sam3_candidates(
        prompt_results,
        prompt_roles,
        image_size=image.size,
        contact_pixel_xy=cfg.grounding.pixel_xy,
        top_k=cfg.top_k,
        contact_sigma_scale=cfg.contact_sigma_scale,
    )
    groups = group_candidates_by_iou(
        candidates,
        iou_threshold=cfg.group_iou_threshold,
        contact_pixel_xy=cfg.grounding.pixel_xy,
        contact_sigma_scale=cfg.contact_sigma_scale,
        weights=cfg.weights,
    )
    object_group, support_group, interaction_group, selected_score, hierarchy_details = select_best_hierarchy(
        groups,
        dilation_px=cfg.dilation_px,
        weights=cfg.weights,
    )
    selected_groups = {"interaction_part": interaction_group}
    if support_group is not None:
        selected_groups["support_part"] = support_group
    if object_group is not None:
        selected_groups["object"] = object_group

    final_masks = resolve_overlap_by_role_and_contact(
        selected_groups,
        contact_pixel_xy=cfg.grounding.pixel_xy,
    )
    selected, overlay_png, summary_path = _save_final_masks_and_overlay(
        image,
        cfg,
        selected_groups,
        final_masks,
        selected_score,
        hierarchy_details,
        groups,
        sam3_summary_path,
    )
    result = HierarchicalPartAnalysisResult(
        contact_pixel_xy=cfg.grounding.pixel_xy,
        selected_score=selected_score,
        selected_groups=selected,
        overlay_png=overlay_png,
        summary_path=summary_path,
    )
    return result, prompt_results
