from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARTANCE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(ARTANCE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTANCE_TESTS_ROOT))

from common.paths import ARTANCE_ROOT
from common.pointcloud_reconstruction import (
    DEFAULT_PART_COLORS,
    PartMaskSpec,
    PointCloudReconstructionConfig,
    load_part_masks_from_point_selection_summary,
    load_part_masks_from_sam3_summary,
    run_pointcloud_reconstruction,
)


DEFAULT_RGB = (
    ARTANCE_ROOT
    / "RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_rgb.png"
)
DEFAULT_DEPTH = (
    ARTANCE_ROOT
    / "RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_depth.npy"
)
DEFAULT_INFO = (
    ARTANCE_ROOT
    / "RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_info.json"
)
DEFAULT_SAM3_SUMMARY = (
    Path(__file__).resolve().parent
    / "outputs/sam3/wrist_reference_frame_051_rgb/summary.json"
)
DEFAULT_POINT_SELECTION_SUMMARY = (
    Path(__file__).resolve().parent
    / "outputs/sam3_point_selection/wrist_reference_frame_051_rgb/point_selection_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs/pointcloud"


def _parse_color(value: str) -> tuple[int, int, int]:
    pieces = value.split(",")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("Color must be R,G,B")
    color = tuple(int(piece) for piece in pieces)
    if any(channel < 0 or channel > 255 for channel in color):
        raise argparse.ArgumentTypeError("Color channels must be in [0, 255]")
    return color


def _parse_mask_spec(value: str, index: int) -> PartMaskSpec:
    # Format: name=/path/to/mask.npy or name=/path/to/mask.npy:255,0,0
    if "=" not in value:
        raise argparse.ArgumentTypeError("--mask must use name=/path/to/mask.npy")
    name, rest = value.split("=", 1)
    path_text, color_text = (rest.rsplit(":", 1) + [None])[:2] if ":" in rest else (rest, None)
    color = _parse_color(color_text) if color_text else DEFAULT_PART_COLORS[index % len(DEFAULT_PART_COLORS)]
    return PartMaskSpec(
        name=name.strip(),
        mask=Path(path_text).expanduser().resolve(),
        color_rgb=color,
    )


def parse_args() -> PointCloudReconstructionConfig:
    parser = argparse.ArgumentParser(
        description="Reconstruct an RGB-D point cloud and color SAM3 part masks."
    )
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--info", type=Path, default=DEFAULT_INFO)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--sam3-summary", type=Path, default=DEFAULT_SAM3_SUMMARY)
    parser.add_argument(
        "--point-selection-summary",
        type=Path,
        help=(
            "Load the SAM3 mask selected by close_drawer/sam3_point_selection_test.py. "
            "Use --selected-prompt to keep only one prompt, e.g. 'drawer handle'."
        ),
    )
    parser.add_argument(
        "--selected-prompt",
        action="append",
        dest="selected_prompts",
        help="Prompt name to load from --point-selection-summary. Repeat to include multiple selected masks.",
    )
    parser.add_argument(
        "--mask",
        action="append",
        help="Manual part mask as name=/path/to/mask.npy or name=/path/to/mask.npy:R,G,B. Repeat for multiple parts.",
    )
    parser.add_argument(
        "--rank",
        action="append",
        type=int,
        dest="ranks",
        help="SAM3 saved_result rank to load from --sam3-summary. Repeat for multiple ranks. Default: 1.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--camera-name", default="wrist")
    parser.add_argument("--subsample-factor", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.015)
    parser.add_argument("--far", type=float, default=20.0)
    parser.add_argument("--output-frame", choices=("camera", "world"), default="camera")
    parser.add_argument("--background-color-mode", choices=("rgb", "gray"), default="rgb")
    parser.add_argument(
        "--mask-overlap-policy",
        choices=("first-wins", "last-wins"),
        default="first-wins",
        help=(
            "How to assign labels/colors when masks overlap. first-wins keeps earlier parts "
            "visible, which is useful for handle + drawer batches."
        ),
    )
    parser.add_argument(
        "--mask-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only save RGB-D points whose pixels are inside the provided/selected mask(s).",
    )
    args = parser.parse_args()

    manual_masks = tuple(
        _parse_mask_spec(value, index) for index, value in enumerate(args.mask or ())
    )
    sam3_summary = args.sam3_summary.expanduser().resolve() if args.sam3_summary else None
    point_selection_summary = (
        args.point_selection_summary.expanduser().resolve() if args.point_selection_summary else None
    )

    if manual_masks:
        masks = manual_masks
    elif point_selection_summary is not None:
        if sam3_summary is None:
            raise ValueError("--point-selection-summary requires --sam3-summary to locate mask files.")
        masks = load_part_masks_from_point_selection_summary(
            point_selection_summary,
            sam3_summary,
            prompts=tuple(args.selected_prompts) if args.selected_prompts else None,
        )
    elif sam3_summary is not None:
        masks = load_part_masks_from_sam3_summary(
            sam3_summary,
            ranks=tuple(args.ranks or (1,)),
        )
    else:
        raise ValueError("Provide --sam3-summary, --point-selection-summary, or at least one --mask.")

    return PointCloudReconstructionConfig(
        rgb=args.rgb.expanduser().resolve(),
        depth=args.depth.expanduser().resolve(),
        intrinsics=args.intrinsics.expanduser().resolve() if args.intrinsics else None,
        info=args.info.expanduser().resolve() if args.info else None,
        pose=args.pose.expanduser().resolve() if args.pose else None,
        sam3_summary=sam3_summary,
        point_selection_summary=point_selection_summary,
        output_dir=args.output_dir.expanduser().resolve(),
        masks=masks,
        camera_name=args.camera_name,
        depth_clip_range=(args.near, args.far),
        subsample_factor=max(1, args.subsample_factor),
        output_frame=args.output_frame,
        background_color_mode=args.background_color_mode,
        mask_only=args.mask_only,
        mask_overlap_policy=args.mask_overlap_policy,
    )


def main() -> None:
    cfg = parse_args()
    cloud, summary_path = run_pointcloud_reconstruction(cfg)
    print(f"points: {cloud.points.shape[0]}")
    for label_id, name in enumerate(cloud.part_names, start=1):
        print(f"{name}: {int((cloud.labels == label_id).sum())} point(s)")
    print(f"background: {int((cloud.labels == 0).sum())} point(s)")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
