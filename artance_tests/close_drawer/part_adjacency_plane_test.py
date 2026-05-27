from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARTANCE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(ARTANCE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTANCE_TESTS_ROOT))

from common.part_adjacency_plane import PartAdjacencyPlaneConfig, run_part_adjacency_plane
from common.task_configs import get_task_config


TASK = "close_drawer"
TASK_PARTS = get_task_config(TASK).sam3_prompts


DEFAULT_POINTCLOUD_NPZ = (
    Path(__file__).resolve().parent
    / "outputs/pointcloud_selected_parts/wrist_reference_frame_051_rgb/"
    / "wrist_reference_frame_051_rgb_camera_sam3_parts.npz"
)
DEFAULT_SOURCE_SUMMARY = (
    Path(__file__).resolve().parent
    / "outputs/pointcloud_selected_parts/wrist_reference_frame_051_rgb/pointcloud_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs/part_adjacency_plane"


def parse_args() -> PartAdjacencyPlaneConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a plane to the second task-config SAM3 part near the first part, "
            "estimate the contact center and normal, and save JSON plus PLY visualization."
        )
    )
    parser.add_argument("--pointcloud-npz", type=Path, default=DEFAULT_POINTCLOUD_NPZ)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--part-a",
        default=TASK_PARTS[1],
        help="Normal source and main part used for plane fitting. Defaults to sam3_prompts[1].",
    )
    parser.add_argument(
        "--part-b",
        default=TASK_PARTS[0],
        help="Normal target and neighbor/contact part. Defaults to sam3_prompts[0].",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=float,
        default=0.025,
        help="3D distance threshold for selecting part-A points close to part-B points.",
    )
    parser.add_argument("--max-visualization-points", type=int, default=120000)
    parser.add_argument("--plane-extent-scale", type=float, default=1.2)
    parser.add_argument("--plane-grid-size", type=int, default=44)
    parser.add_argument("--sphere-radius", type=float, default=0.012)
    parser.add_argument("--arrow-length", type=float, default=0.06)
    parser.add_argument("--arrow-radius", type=float, default=0.003)
    args = parser.parse_args()

    return PartAdjacencyPlaneConfig(
        pointcloud_npz=args.pointcloud_npz.expanduser().resolve(),
        source_summary=args.source_summary.expanduser().resolve() if args.source_summary else None,
        output_dir=args.output_dir.expanduser().resolve(),
        part_a=args.part_a,
        part_b=args.part_b,
        neighbor_radius=args.neighbor_radius,
        max_visualization_points=max(1, args.max_visualization_points),
        plane_extent_scale=args.plane_extent_scale,
        plane_grid_size=max(2, args.plane_grid_size),
        sphere_radius=args.sphere_radius,
        arrow_length=args.arrow_length,
        arrow_radius=args.arrow_radius,
    )


def main() -> None:
    cfg = parse_args()
    result, summary_path = run_part_adjacency_plane(cfg)
    print(f"part A ({result.part_a}) points: {result.part_a_point_count}")
    print(f"part B ({result.part_b}) points: {result.part_b_point_count}")
    print(f"adjacent part A points: {result.adjacent_part_a_point_count}")
    print(f"contact center: {result.contact_center}")
    print(f"plane normal: {result.plane_normal}")
    print(f"fit RMS distance: {result.fit['rms_distance']:.6f}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
