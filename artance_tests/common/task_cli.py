from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.paths import ARTANCE_ROOT
from common.contact_graspnet_pose import (
    DEFAULT_GRASPNET_SERVICE_URL,
    DEFAULT_GRASP_Z_OFFSET,
    ContactGraspNetPoseConfig,
    run_contact_graspnet_pose,
)
from common.implicit_door_remote_rotation import (
    ImplicitDoorRemoteRotationConfig,
    run_implicit_door_remote_rotation,
)
from common.hierarchical_part_analysis import (
    HierarchicalPartAnalysisConfig,
    run_hierarchical_part_analysis,
)
from common.pointcloud_reconstruction import (
    PointCloudReconstructionConfig,
    load_part_masks_from_part_analysis_summary,
    load_part_masks_from_point_selection_summary,
    load_part_masks_from_sam3_summary,
    run_pointcloud_reconstruction,
)
from common.rlbench_episode_assets import (
    EpisodeFrame,
    find_episode_frame,
    parse_episode_rgb_path,
    write_episode_frame_info_json,
)
from common.sam3_point_selector import PointSelectionConfig, run_sam3_point_selection_test
from common.sam3_runner import Sam3RunConfig, default_sam3_service_url, run_sam3_text_test
from common.task_configs import TASK_CONFIGS, get_task_config
from common.vlm_contact_point import (
    DEFAULT_VLM_MAX_TOKENS,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_SERVER_URL,
    VlmContactPointConfig,
    VlmContactPointParseError,
    query_vlm_contact_point,
    save_vlm_contact_point_failure_summary,
    save_vlm_contact_point_summary,
)
from common.vlm_interaction_grounding import (
    VlmInteractionGroundingConfig,
    VlmInteractionGroundingParseError,
    load_vlm_interaction_grounding_summary,
    query_vlm_interaction_grounding,
    save_vlm_interaction_grounding_failure_summary,
    save_vlm_interaction_grounding_summary,
)


MODULES = (
    "contact_point",
    "structured_grounding",
    "sam3",
    "sam3_point_selection",
    "part_analysis",
    "pointcloud",
    "contact_graspnet_pose",
    "implicit_door_remote_rotation",
)


def task_dir(task: str) -> Path:
    return ARTANCE_ROOT / "cap-x/artance_tests" / task


def _resolve_episode(args: argparse.Namespace, task: str) -> EpisodeFrame | None:
    if getattr(args, "episode_line", None):
        return parse_episode_rgb_path(args.episode_line)
    if getattr(args, "image", None):
        try:
            return parse_episode_rgb_path(args.image)
        except ValueError:
            return None
    return find_episode_frame(
        task,
        variation=getattr(args, "variation", None),
        episode=getattr(args, "episode", None),
        frame=getattr(args, "frame", None),
        camera=getattr(args, "camera", "wrist"),
        episode_file=getattr(args, "episode_file", None),
    )


def _default_output_dir(task: str, module: str, episode: EpisodeFrame | None) -> Path:
    root = task_dir(task) / "outputs" / module
    return root / episode.key if episode is not None else root


def _image_path(args: argparse.Namespace, episode: EpisodeFrame | None) -> Path:
    if getattr(args, "image", None):
        return args.image.expanduser().resolve()
    if episode is None:
        raise ValueError("Provide --image or an episode selector.")
    return episode.rgb


def _depth_path(args: argparse.Namespace, episode: EpisodeFrame | None) -> Path:
    if getattr(args, "depth", None):
        return args.depth.expanduser().resolve()
    if episode is None:
        raise ValueError("Provide --depth or an episode selector.")
    return episode.depth


def _add_episode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episode-line", type=Path, help="One RGB frame path from episodes.txt.")
    parser.add_argument("--episode-file", type=Path, default=ARTANCE_ROOT / "cap-x/artance_tests/episodes.txt")
    parser.add_argument("--variation", type=int)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--image", type=Path, help="Override the RGB image path.")


def _contact_summary_path(task: str, episode: EpisodeFrame) -> Path:
    return _default_output_dir(task, "contact_point", episode) / episode.image_stem / "summary.json"


def _sam3_summary_path(task: str, episode: EpisodeFrame) -> Path:
    return _default_output_dir(task, "sam3", episode) / episode.image_stem / "summary.json"


def _structured_grounding_summary_path(task: str, episode: EpisodeFrame) -> Path:
    return (
        _default_output_dir(task, "structured_grounding", episode)
        / episode.image_stem
        / "structured_grounding_summary.json"
    )


def _part_analysis_grounding_summary_path(output_dir: Path, image: Path) -> Path:
    return output_dir / image.stem / "structured_grounding_summary.json"


def _part_analysis_summary_path(task: str, episode: EpisodeFrame) -> Path:
    return _default_output_dir(task, "part_analysis", episode) / episode.image_stem / "part_analysis_summary.json"


def _point_selection_summary_path(task: str, episode: EpisodeFrame) -> Path:
    return (
        _default_output_dir(task, "sam3_point_selection", episode)
        / episode.image_stem
        / "point_selection_summary.json"
    )


def _load_pixel_xy_from_contact_summary(summary_path: Path) -> tuple[int, int]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Contact point summary not found: {summary_path}")
    data: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    pixel_xy = data.get("result", {}).get("pixel_xy")
    if not (
        isinstance(pixel_xy, (list, tuple))
        and len(pixel_xy) == 2
        and all(isinstance(v, (int, float)) for v in pixel_xy)
    ):
        raise ValueError(f"Could not find result.pixel_xy in contact point summary: {summary_path}")
    return int(round(pixel_xy[0])), int(round(pixel_xy[1]))


def run_contact_point_cli(task: str) -> None:
    task_cfg = get_task_config(task)
    parser = argparse.ArgumentParser(
        description=f"Predict the best 2D contact point for {task_cfg.display_name}."
    )
    _add_episode_args(parser)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--server-url", default=DEFAULT_VLM_SERVER_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_VLM_MAX_TOKENS)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--prompt-order", choices=("auto", "xy", "yx"), default="auto")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    episode = _resolve_episode(args, task)
    image = _image_path(args, episode)
    output_dir = (args.output_dir or _default_output_dir(task, "contact_point", episode)).resolve()
    cfg = VlmContactPointConfig(
        image=image,
        output_dir=output_dir,
        prompt_spec=task_cfg.contact_prompt,
        model=args.model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        prompt_order=None if args.prompt_order == "auto" else args.prompt_order,
        debug=args.debug,
    )
    try:
        result = query_vlm_contact_point(cfg)
    except VlmContactPointParseError as exc:
        summary_path = save_vlm_contact_point_failure_summary(
            cfg,
            error=exc,
            raw_response=exc.raw_response,
            prompt_order=exc.prompt_order,
        )
        print("VLM contact point parsing failed.")
        print(f"model: {exc.model}")
        print(f"prompt_order: {exc.prompt_order}")
        print(f"raw_response_repr: {exc.raw_response!r}")
        print(f"Saved failure summary: {summary_path}")
        raise

    summary_path = save_vlm_contact_point_summary(cfg, result)
    print(f"task: {task}")
    if episode is not None:
        print(f"episode: {episode.key}")
    print(f"prompt_order: {result.prompt_order}")
    print(f"raw_coordinate: {list(result.raw_coordinate)}")
    print(f"normalized_xy: {list(result.normalized_xy)}")
    print(f"pixel_xy: {list(result.pixel_xy)}")
    print(f"Saved summary: {summary_path}")


def run_structured_grounding_cli(task: str) -> None:
    task_cfg = get_task_config(task)
    parser = argparse.ArgumentParser(
        description=f"Predict structured interaction grounding and hierarchical SAM3 prompts for {task_cfg.display_name}."
    )
    _add_episode_args(parser)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--server-url", default=DEFAULT_VLM_SERVER_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_VLM_MAX_TOKENS)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    episode = _resolve_episode(args, task)
    output_dir = (args.output_dir or _default_output_dir(task, "structured_grounding", episode)).resolve()
    cfg = VlmInteractionGroundingConfig(
        image=_image_path(args, episode),
        output_dir=output_dir,
        prompt_spec=task_cfg.contact_prompt,
        model=args.model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )
    try:
        result = query_vlm_interaction_grounding(cfg)
    except VlmInteractionGroundingParseError as exc:
        summary_path = save_vlm_interaction_grounding_failure_summary(
            cfg,
            error=exc,
            raw_response=exc.raw_response,
        )
        print("VLM structured grounding parsing failed.")
        print(f"model: {exc.model}")
        print(f"raw_response_repr: {exc.raw_response!r}")
        print(f"Saved failure summary: {summary_path}")
        raise

    summary_path = save_vlm_interaction_grounding_summary(cfg, result)
    print(f"task: {task}")
    if episode is not None:
        print(f"episode: {episode.key}")
    print(f"target_object: {result.target_object}")
    print(f"interaction_part: {result.interaction_part}")
    print(f"support_part: {result.support_part}")
    print(f"contact_pixel_yx: {list(result.contact_pixel_yx)}")
    print(f"pixel_xy: {list(result.pixel_xy)}")
    for role, prompts in result.sam3_prompts.items():
        print(f"{role}: {list(prompts)}")
    print(f"Saved summary: {summary_path}")


def run_sam3_cli(task: str) -> None:
    task_cfg = get_task_config(task)
    parser = argparse.ArgumentParser(description=f"Run SAM3 on {task_cfg.display_name} RGB frames.")
    _add_episode_args(parser)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--service-url", default=default_sam3_service_url())
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    episode = _resolve_episode(args, task)
    cfg = Sam3RunConfig(
        image=_image_path(args, episode),
        prompts=tuple(args.prompts) if args.prompts else task_cfg.sam3_prompts,
        output_dir=(args.output_dir or _default_output_dir(task, "sam3", episode)).resolve(),
        service_url=args.service_url,
        top_k=max(1, args.top_k),
        show=args.show,
    )
    prompt_results, summary_path = run_sam3_text_test(cfg)
    print(f"task: {task}")
    if episode is not None:
        print(f"episode: {episode.key}")
    for prompt, results in prompt_results.items():
        print(f"{prompt}: {len(results)} result(s)")
    print(f"Saved summary: {summary_path}")


def run_sam3_point_selection_cli(task: str) -> None:
    task_cfg = get_task_config(task)
    parser = argparse.ArgumentParser(
        description=f"Select the SAM3 instance nearest to a VLM contact point for {task_cfg.display_name}."
    )
    _add_episode_args(parser)
    parser.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--contact-summary", type=Path)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--service-url", default=default_sam3_service_url())
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    episode = _resolve_episode(args, task)
    image = _image_path(args, episode)
    if args.point is not None:
        point_xy = (int(args.point[0]), int(args.point[1]))
    else:
        contact_summary = args.contact_summary
        if contact_summary is None:
            if episode is None:
                raise ValueError("Provide --contact-summary when --image is not an episodes.txt frame.")
            contact_summary = _contact_summary_path(task, episode)
        point_xy = _load_pixel_xy_from_contact_summary(contact_summary.expanduser().resolve())

    cfg = PointSelectionConfig(
        image=image,
        point_xy=point_xy,
        prompts=tuple(args.prompts) if args.prompts else task_cfg.sam3_prompts,
        output_dir=(args.output_dir or _default_output_dir(task, "sam3_point_selection", episode)).resolve(),
        service_url=args.service_url,
        top_k=max(1, args.top_k),
        show=args.show,
    )
    prompt_results, selections, summary_path = run_sam3_point_selection_test(cfg)
    print(f"task: {task}")
    if episode is not None:
        print(f"episode: {episode.key}")
    print(f"point_xy: {list(cfg.point_xy)}")
    for prompt, results in prompt_results.items():
        selected = selections[prompt]
        if selected is None:
            print(f"{prompt}: no SAM3 result")
            continue
        print(
            f"{prompt}: selected rank {selected.rank}/{len(results)} "
            f"score={selected.score:.3f} distance_px={selected.distance_px:.2f} "
            f"contains_point={selected.contains_point}"
        )
    print(f"Saved summary: {summary_path}")


def run_part_analysis_cli(task: str) -> None:
    task_cfg = get_task_config(task)
    parser = argparse.ArgumentParser(
        description=(
            f"Run VLM-generated hierarchical SAM3 prompts and generic part mask selection for {task_cfg.display_name}."
        )
    )
    _add_episode_args(parser)
    parser.add_argument("--grounding-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-server-url", default=DEFAULT_VLM_SERVER_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_VLM_MAX_TOKENS)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--sam3-service-url", default=default_sam3_service_url())
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--group-iou-threshold", type=float, default=0.7)
    parser.add_argument("--dilation-px", type=int, default=8)
    parser.add_argument("--contact-sigma-scale", type=float, default=0.25)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    episode = _resolve_episode(args, task)
    image = _image_path(args, episode)
    output_dir = (args.output_dir or _default_output_dir(task, "part_analysis", episode)).resolve()

    grounding_summary = args.grounding_summary.expanduser().resolve() if args.grounding_summary else None
    if grounding_summary is None:
        candidates = [_part_analysis_grounding_summary_path(output_dir, image)]
        if episode is not None:
            candidates.append(_structured_grounding_summary_path(task, episode))
        grounding_summary = next((path.resolve() for path in candidates if path.exists()), None)

    if grounding_summary is not None:
        grounding = load_vlm_interaction_grounding_summary(grounding_summary)
        print(f"Loaded structured grounding summary: {grounding_summary}")
    else:
        grounding_cfg = VlmInteractionGroundingConfig(
            image=image,
            output_dir=output_dir,
            prompt_spec=task_cfg.contact_prompt,
            model=args.model,
            server_url=args.vlm_server_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            debug=args.debug,
        )
        try:
            grounding = query_vlm_interaction_grounding(grounding_cfg)
        except VlmInteractionGroundingParseError as exc:
            summary_path = save_vlm_interaction_grounding_failure_summary(
                grounding_cfg,
                error=exc,
                raw_response=exc.raw_response,
            )
            print("VLM structured grounding parsing failed.")
            print(f"Saved failure summary: {summary_path}")
            raise
        save_vlm_interaction_grounding_summary(grounding_cfg, grounding)

    cfg = HierarchicalPartAnalysisConfig(
        image=image,
        output_dir=output_dir,
        grounding=grounding,
        service_url=args.sam3_service_url,
        top_k=max(1, args.top_k),
        show=args.show,
        group_iou_threshold=args.group_iou_threshold,
        dilation_px=max(0, args.dilation_px),
        contact_sigma_scale=args.contact_sigma_scale,
    )
    result, prompt_results = run_hierarchical_part_analysis(cfg)
    print(f"task: {task}")
    if episode is not None:
        print(f"episode: {episode.key}")
    print(f"target_object: {grounding.target_object}")
    print(f"interaction_part: {grounding.interaction_part}")
    print(f"support_part: {grounding.support_part}")
    print(f"contact_pixel_xy: {list(result.contact_pixel_xy)}")
    for prompt, results in prompt_results.items():
        print(f"{prompt}: {len(results)} result(s)")
    for role, selected in result.selected_groups.items():
        print(
            f"{role}: {selected.group_id} prompts={list(selected.prompts)} "
            f"final_pixels={selected.final_mask_pixels} contact_dist={selected.contact_distance_px:.2f}"
        )
    print(f"selected_score: {result.selected_score:.4f}")
    print(f"Saved summary: {result.summary_path}")


def run_pointcloud_cli(task: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Reconstruct an RGB-D point cloud for {get_task_config(task).display_name}."
    )
    _add_episode_args(parser)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--info", type=Path)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--sam3-summary", type=Path)
    parser.add_argument("--point-selection-summary", type=Path)
    parser.add_argument("--part-analysis-summary", type=Path)
    parser.add_argument(
        "--mask-source",
        choices=("auto", "part_analysis", "point_selection", "sam3"),
        default="auto",
        help=(
            "Which mask summary to load. auto prefers explicit summary args, then existing "
            "part_analysis, point_selection, and SAM3 summaries for episode inputs."
        ),
    )
    parser.add_argument("--part-analysis-role", action="append", dest="part_analysis_roles")
    parser.add_argument("--selected-prompt", action="append", dest="selected_prompts")
    parser.add_argument("--rank", action="append", type=int, dest="ranks")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subsample-factor", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.015)
    parser.add_argument("--far", type=float, default=20.0)
    parser.add_argument("--output-frame", choices=("camera", "world"), default="camera")
    parser.add_argument("--background-color-mode", choices=("rgb", "gray"), default="rgb")
    parser.add_argument("--mask-overlap-policy", choices=("first-wins", "last-wins"), default="first-wins")
    parser.add_argument("--mask-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--depth-format",
        choices=("auto", "meters", "rlbench_normalized_png"),
        default="auto",
    )
    args = parser.parse_args()

    episode = _resolve_episode(args, task)
    output_dir = (args.output_dir or _default_output_dir(task, "pointcloud", episode)).resolve()
    info = args.info.expanduser().resolve() if args.info else None
    if info is None and episode is not None:
        info = write_episode_frame_info_json(episode, output_dir)

    sam3_summary = args.sam3_summary
    point_selection_summary = args.point_selection_summary
    part_analysis_summary = args.part_analysis_summary
    if episode is not None:
        sam3_summary = sam3_summary or _sam3_summary_path(task, episode)
        point_selection_summary = point_selection_summary or _point_selection_summary_path(task, episode)
        part_analysis_summary = part_analysis_summary or _part_analysis_summary_path(task, episode)

    explicit_sources = [
        ("part_analysis", args.part_analysis_summary),
        ("point_selection", args.point_selection_summary),
        ("sam3", args.sam3_summary),
    ]
    mask_source = args.mask_source
    if mask_source == "auto":
        explicit = [source for source, value in explicit_sources if value is not None]
        if len(explicit) > 1:
            raise ValueError(
                "Pass only one explicit summary path when --mask-source=auto, or set --mask-source explicitly."
            )
        if explicit:
            mask_source = explicit[0]
        elif part_analysis_summary is not None and part_analysis_summary.exists():
            mask_source = "part_analysis"
        elif point_selection_summary is not None and point_selection_summary.exists():
            mask_source = "point_selection"
        elif sam3_summary is not None and sam3_summary.exists():
            mask_source = "sam3"
        else:
            raise FileNotFoundError(
                "Could not auto-detect a mask summary. Expected one of: "
                f"part_analysis={part_analysis_summary}, "
                f"point_selection={point_selection_summary}, sam3={sam3_summary}."
            )

    if mask_source == "part_analysis":
        if part_analysis_summary is None:
            raise ValueError("--mask-source part_analysis requires --part-analysis-summary or an episode selector.")
        masks = load_part_masks_from_part_analysis_summary(
            part_analysis_summary.expanduser().resolve(),
            roles=tuple(args.part_analysis_roles) if args.part_analysis_roles else None,
        )
    elif mask_source == "point_selection":
        if point_selection_summary is None:
            raise ValueError("--mask-source point_selection requires --point-selection-summary or an episode selector.")
        if sam3_summary is None:
            raise ValueError("--mask-source point_selection requires --sam3-summary to locate mask files.")
        masks = load_part_masks_from_point_selection_summary(
            point_selection_summary.expanduser().resolve(),
            sam3_summary.expanduser().resolve(),
            prompts=tuple(args.selected_prompts) if args.selected_prompts else get_task_config(task).sam3_prompts,
        )
    elif mask_source == "sam3":
        if sam3_summary is None:
            raise ValueError("--mask-source sam3 requires --sam3-summary or an episode selector.")
        masks = load_part_masks_from_sam3_summary(
            sam3_summary.expanduser().resolve(),
            ranks=tuple(args.ranks or (1,)),
        )
    else:
        raise ValueError(f"Unsupported mask_source: {mask_source!r}")

    cfg = PointCloudReconstructionConfig(
        rgb=_image_path(args, episode),
        depth=_depth_path(args, episode),
        intrinsics=args.intrinsics.expanduser().resolve() if args.intrinsics else None,
        info=info,
        pose=args.pose.expanduser().resolve() if args.pose else None,
        sam3_summary=sam3_summary.expanduser().resolve() if sam3_summary else None,
        point_selection_summary=(
            point_selection_summary.expanduser().resolve() if point_selection_summary else None
        ),
        part_analysis_summary=(
            part_analysis_summary.expanduser().resolve() if part_analysis_summary else None
        ),
        output_dir=output_dir,
        masks=masks,
        camera_name=episode.camera if episode is not None else args.camera,
        depth_clip_range=(args.near, args.far),
        subsample_factor=max(1, args.subsample_factor),
        output_frame=args.output_frame,
        background_color_mode=args.background_color_mode,
        mask_only=args.mask_only,
        mask_overlap_policy=args.mask_overlap_policy,
        depth_format=args.depth_format,
    )
    cloud, summary_path = run_pointcloud_reconstruction(cfg)
    print(f"task: {task}")
    if episode is not None:
        print(f"episode: {episode.key}")
    print(f"mask_source: {mask_source}")
    print(f"points: {cloud.points.shape[0]}")
    for label_id, name in enumerate(cloud.part_names, start=1):
        print(f"{name}: {int((cloud.labels == label_id).sum())} point(s)")
    print(f"background: {int((cloud.labels == 0).sum())} point(s)")
    print(f"Saved summary: {summary_path}")



def _pointcloud_npz_path(task: str, episode: EpisodeFrame) -> Path:
    return _default_output_dir(task, "pointcloud", episode) / episode.image_stem / f"{episode.image_stem}_camera_sam3_parts.npz"


def _pointcloud_summary_path(task: str, episode: EpisodeFrame) -> Path:
    return _default_output_dir(task, "pointcloud", episode) / episode.image_stem / "pointcloud_summary.json"


def _graspnet_summary_path(task: str, episode: EpisodeFrame) -> Path:
    return _default_output_dir(task, "contact_graspnet_pose", episode) / episode.image_stem / "contact_graspnet_summary.json"

def run_contact_graspnet_pose_cli(task: str) -> None:
    task_cfg = get_task_config(task)
    parser = argparse.ArgumentParser(
        description=f"Plan Contact-GraspNet interaction/grasp poses for {task_cfg.display_name}."
    )
    _add_episode_args(parser)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--info", type=Path)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--sam3-summary", type=Path)
    parser.add_argument("--point-selection-summary", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--target-prompt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--service-url", default=DEFAULT_GRASPNET_SERVICE_URL)
    parser.add_argument("--near", type=float, default=0.2)
    parser.add_argument("--far", type=float, default=2.0)
    parser.add_argument("--forward-passes", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--grasp-z-offset", type=float, default=DEFAULT_GRASP_Z_OFFSET)
    parser.add_argument("--save-visualization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualization-subsample-factor", type=int, default=2)
    parser.add_argument("--visualization-top-k", type=int, default=10)
    parser.add_argument("--visualization-axis-length", type=float, default=0.055)
    parser.add_argument("--visualization-point-radius", type=float, default=0.008)
    parser.add_argument(
        "--depth-format",
        choices=("auto", "meters", "rlbench_normalized_png"),
        default="auto",
    )
    args = parser.parse_args()

    episode = _resolve_episode(args, task)
    output_dir = (args.output_dir or _default_output_dir(task, "contact_graspnet_pose", episode)).resolve()
    info = args.info.expanduser().resolve() if args.info else None
    if info is None and episode is not None:
        info = write_episode_frame_info_json(episode, output_dir)

    sam3_summary = args.sam3_summary
    point_selection_summary = args.point_selection_summary
    if episode is not None:
        sam3_summary = sam3_summary or _sam3_summary_path(task, episode)
        point_selection_summary = point_selection_summary or _point_selection_summary_path(task, episode)

    target_prompt = args.target_prompt or task_cfg.contact_grasp_prompt or task_cfg.sam3_prompts[0]
    cfg = ContactGraspNetPoseConfig(
        rgb=_image_path(args, episode),
        depth=_depth_path(args, episode),
        intrinsics=args.intrinsics.expanduser().resolve() if args.intrinsics else None,
        info=info,
        output_dir=output_dir,
        target_prompt=target_prompt,
        sam3_summary=sam3_summary.expanduser().resolve() if sam3_summary else None,
        point_selection_summary=(
            point_selection_summary.expanduser().resolve() if point_selection_summary else None
        ),
        mask=args.mask.expanduser().resolve() if args.mask else None,
        pose=args.pose.expanduser().resolve() if args.pose else None,
        camera_name=episode.camera if episode is not None else args.camera,
        depth_format=args.depth_format,
        z_range=(args.near, args.far),
        forward_passes=max(1, args.forward_passes),
        max_retries=max(0, args.max_retries),
        service_url=args.service_url,
        grasp_z_offset=args.grasp_z_offset,
        save_visualization=args.save_visualization,
        visualization_subsample_factor=max(1, args.visualization_subsample_factor),
        visualization_top_k=max(0, args.visualization_top_k),
        visualization_axis_length=args.visualization_axis_length,
        visualization_point_radius=args.visualization_point_radius,
    )
    result, summary_path = run_contact_graspnet_pose(cfg)
    print(f"task: {task}")
    if episode is not None:
        print(f"episode: {episode.key}")
    print(f"target_prompt: {target_prompt}")
    print(f"mask_pixels: {result.mask_pixels}")
    print(f"num_candidates: {result.scores.shape[0]}")
    if result.best_index is not None:
        print(f"best_index: {result.best_index}")
        print(f"best_score: {result.best_score:.6f}")
    print(f"Saved summary: {summary_path}")



def run_implicit_door_remote_rotation_cli(task: str) -> None:
    task_cfg = get_task_config(task)
    door_spec = task_cfg.implicit_door_remote_rotation
    if door_spec is None:
        raise ValueError(f"Task {task!r} does not define implicit_door_remote_rotation in task_configs.py.")

    parser = argparse.ArgumentParser(
        description=(
            f"Estimate an implicit hinge axis for {task_cfg.display_name} from door plane, "
            "handle side, and task prior."
        )
    )
    _add_episode_args(parser)
    parser.add_argument("--episode-key", help="Existing output key, e.g. variation0_episode0_frame035_wrist.")
    parser.add_argument("--frame-stem", help="Existing frame output folder, e.g. 35.")
    parser.add_argument("--pointcloud-npz", type=Path)
    parser.add_argument("--contact-summary", type=Path)
    parser.add_argument("--graspnet-summary", type=Path)
    parser.add_argument("--guide-source", choices=("vlm", "graspnet"), default="vlm")
    parser.add_argument(
        "--contact-point-mode",
        choices=("handle_center", "guide"),
        default="handle_center",
        help="Interaction point for the door trajectory. Default uses the 3D handle center.",
    )
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--door-part", default=door_spec.door_part)
    parser.add_argument("--handle-part", default=door_spec.handle_part)
    parser.add_argument("--door-opening-type", choices=("side",), default=door_spec.door_opening_type)
    parser.add_argument("--hinge-axis-orientation", choices=("vertical",), default=door_spec.hinge_axis_orientation)
    parser.add_argument("--closing-direction-sign", type=float, default=door_spec.closing_direction_sign)
    parser.add_argument("--rotation-degrees", type=float, default=door_spec.rotation_degrees)
    parser.add_argument("--waypoint-count", type=int, default=9)
    parser.add_argument("--collision-threshold", type=float, default=0.012)
    parser.add_argument("--collision-check-degrees", type=float, default=12.0)
    parser.add_argument("--max-collision-sample-points", type=int, default=3000)
    parser.add_argument("--max-visualization-points", type=int, default=120000)
    args = parser.parse_args()

    episode_key = args.episode_key
    frame_stem = args.frame_stem
    episode = None if episode_key is not None else _resolve_episode(args, task)
    if episode is None and (episode_key is None) != (frame_stem is None):
        raise ValueError("Use --episode-key and --frame-stem together.")
    if episode is None and args.pointcloud_npz is None and episode_key is None:
        raise ValueError("Provide --pointcloud-npz, --episode-key/--frame-stem, or an episode selector.")

    pointcloud_npz = args.pointcloud_npz
    contact_summary = args.contact_summary
    graspnet_summary = args.graspnet_summary
    source_summary = args.source_summary
    output_dir = args.output_dir
    if episode is not None:
        pointcloud_npz = pointcloud_npz or _pointcloud_npz_path(task, episode)
        contact_summary = contact_summary or _contact_summary_path(task, episode)
        graspnet_summary = graspnet_summary or _graspnet_summary_path(task, episode)
        source_summary = source_summary or _pointcloud_summary_path(task, episode)
        output_dir = output_dir or task_dir(task) / "outputs" / "implicit_door_remote_rotation"
    elif episode_key is not None and frame_stem is not None:
        output_root = task_dir(task) / "outputs"
        pointcloud_npz = pointcloud_npz or output_root / "pointcloud" / episode_key / frame_stem / f"{frame_stem}_camera_sam3_parts.npz"
        contact_summary = contact_summary or output_root / "contact_point" / episode_key / frame_stem / "summary.json"
        graspnet_summary = graspnet_summary or output_root / "contact_graspnet_pose" / episode_key / frame_stem / "contact_graspnet_summary.json"
        source_summary = source_summary or output_root / "pointcloud" / episode_key / frame_stem / "pointcloud_summary.json"
        output_dir = output_dir or output_root / "implicit_door_remote_rotation"
    if output_dir is None:
        output_dir = task_dir(task) / "outputs" / "implicit_door_remote_rotation"

    cfg = ImplicitDoorRemoteRotationConfig(
        pointcloud_npz=pointcloud_npz.expanduser().resolve(),
        output_dir=output_dir.expanduser().resolve(),
        contact_summary=contact_summary.expanduser().resolve() if contact_summary else None,
        graspnet_summary=graspnet_summary.expanduser().resolve() if graspnet_summary else None,
        guide_source=args.guide_source,
        contact_point_mode=args.contact_point_mode,
        door_part=args.door_part,
        handle_part=args.handle_part,
        door_opening_type=args.door_opening_type,
        hinge_axis_orientation=args.hinge_axis_orientation,
        closing_direction_sign=args.closing_direction_sign,
        rotation_degrees=args.rotation_degrees,
        waypoint_count=max(2, args.waypoint_count),
        collision_threshold=args.collision_threshold,
        collision_check_degrees=args.collision_check_degrees,
        max_collision_sample_points=max(1, args.max_collision_sample_points),
        max_visualization_points=max(1, args.max_visualization_points),
        source_summary=source_summary.expanduser().resolve() if source_summary else None,
    )
    result, summary_path = run_implicit_door_remote_rotation(cfg)
    print(f"task: {task}")
    if episode is not None:
        print(f"episode: {episode.key}")
    print(f"door part: {result.part_a} label={result.part_a_label_id}")
    print(f"handle part: {result.part_b} label={result.part_b_label_id}")
    print(f"guide source: {result.guide_source}")
    print(f"contact pixel xy: {result.contact_pixel_xy}")
    print(f"contact point: {result.contact_point_3d}")
    print(f"axis point: {result.axis_point}")
    print(f"axis direction: {result.axis_direction}")
    print(f"chosen rotation degrees: {result.rotation_degrees:.3f}")
    print(f"Saved summary: {summary_path}")


def run_module_cli(module: str, task: str) -> None:
    if module == "contact_point":
        run_contact_point_cli(task)
    elif module == "structured_grounding":
        run_structured_grounding_cli(task)
    elif module == "sam3":
        run_sam3_cli(task)
    elif module == "sam3_point_selection":
        run_sam3_point_selection_cli(task)
    elif module == "part_analysis":
        run_part_analysis_cli(task)
    elif module == "pointcloud":
        run_pointcloud_cli(task)
    elif module == "contact_graspnet_pose":
        run_contact_graspnet_pose_cli(task)
    elif module == "implicit_door_remote_rotation":
        run_implicit_door_remote_rotation_cli(task)
    else:
        supported = ", ".join(MODULES)
        raise ValueError(f"Unsupported module {module!r}. Supported modules: {supported}")


def add_task_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONFIGS))
