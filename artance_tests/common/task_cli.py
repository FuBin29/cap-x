from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.paths import ARTANCE_ROOT
from common.pointcloud_reconstruction import (
    PointCloudReconstructionConfig,
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


MODULES = ("contact_point", "sam3", "sam3_point_selection", "pointcloud")


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
    if episode is not None:
        sam3_summary = sam3_summary or _sam3_summary_path(task, episode)
        point_selection_summary = point_selection_summary or _point_selection_summary_path(task, episode)

    if point_selection_summary is not None:
        if sam3_summary is None:
            raise ValueError("--point-selection-summary requires --sam3-summary to locate mask files.")
        masks = load_part_masks_from_point_selection_summary(
            point_selection_summary.expanduser().resolve(),
            sam3_summary.expanduser().resolve(),
            prompts=tuple(args.selected_prompts) if args.selected_prompts else get_task_config(task).sam3_prompts,
        )
    elif sam3_summary is not None:
        masks = load_part_masks_from_sam3_summary(
            sam3_summary.expanduser().resolve(),
            ranks=tuple(args.ranks or (1,)),
        )
    else:
        raise ValueError("Provide --sam3-summary or --point-selection-summary.")

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
    print(f"points: {cloud.points.shape[0]}")
    for label_id, name in enumerate(cloud.part_names, start=1):
        print(f"{name}: {int((cloud.labels == label_id).sum())} point(s)")
    print(f"background: {int((cloud.labels == 0).sum())} point(s)")
    print(f"Saved summary: {summary_path}")


def run_module_cli(module: str, task: str) -> None:
    if module == "contact_point":
        run_contact_point_cli(task)
    elif module == "sam3":
        run_sam3_cli(task)
    elif module == "sam3_point_selection":
        run_sam3_point_selection_cli(task)
    elif module == "pointcloud":
        run_pointcloud_cli(task)
    else:
        supported = ", ".join(MODULES)
        raise ValueError(f"Unsupported module {module!r}. Supported modules: {supported}")


def add_task_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONFIGS))
