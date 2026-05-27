from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARTANCE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(ARTANCE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTANCE_TESTS_ROOT))

from common.contact_guided_remote_rotation import (
    ContactGuidedRemoteRotationConfig,
    run_contact_guided_remote_rotation,
)


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_EPISODE_KEY = "variation0_episode2_frame015_wrist"
DEFAULT_FRAME_STEM = "15"
DEFAULT_POINTCLOUD_NPZ = (
    TASK_DIR
    / "outputs/pointcloud"
    / DEFAULT_EPISODE_KEY
    / DEFAULT_FRAME_STEM
    / f"{DEFAULT_FRAME_STEM}_camera_sam3_parts.npz"
)
DEFAULT_CONTACT_SUMMARY = (
    TASK_DIR / "outputs/contact_point" / DEFAULT_EPISODE_KEY / DEFAULT_FRAME_STEM / "summary.json"
)
DEFAULT_SOURCE_SUMMARY = (
    TASK_DIR / "outputs/pointcloud" / DEFAULT_EPISODE_KEY / DEFAULT_FRAME_STEM / "pointcloud_summary.json"
)
DEFAULT_OUTPUT_DIR = TASK_DIR / "outputs/contact_guided_remote_rotation"


def _default_path_from_episode(module: str, episode_key: str, frame_stem: str, filename: str) -> Path:
    return TASK_DIR / "outputs" / module / episode_key / frame_stem / filename


def parse_args() -> ContactGuidedRemoteRotationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a contact-guided remote rotation trajectory for toilet seat down: "
            "find contact point a on part A, fit distal hinge axis d from far part-A "
            "points and neighboring part-B points, choose the lower-collision rotation "
            "direction, then save 40-degree waypoints and PLY visualization."
        )
    )
    parser.add_argument("--episode-key", default=DEFAULT_EPISODE_KEY)
    parser.add_argument("--frame-stem", default=DEFAULT_FRAME_STEM)
    parser.add_argument("--pointcloud-npz", type=Path)
    parser.add_argument("--contact-summary", type=Path)
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--part-a", default="toilet lid")
    parser.add_argument("--part-b", default="toilet")
    parser.add_argument("--far-quantile", type=float, default=0.72)
    parser.add_argument("--hinge-neighbor-radius", type=float, default=0.035)
    parser.add_argument("--collision-threshold", type=float, default=0.012)
    parser.add_argument("--collision-check-degrees", type=float, default=12.0)
    parser.add_argument("--rotation-degrees", type=float, default=40.0)
    parser.add_argument("--waypoint-count", type=int, default=9)
    parser.add_argument("--max-collision-sample-points", type=int, default=3000)
    parser.add_argument("--max-visualization-points", type=int, default=120000)
    args = parser.parse_args()

    default_pointcloud = _default_path_from_episode(
        "pointcloud",
        args.episode_key,
        args.frame_stem,
        f"{args.frame_stem}_camera_sam3_parts.npz",
    )
    default_contact = _default_path_from_episode(
        "contact_point",
        args.episode_key,
        args.frame_stem,
        "summary.json",
    )
    default_source = _default_path_from_episode(
        "pointcloud",
        args.episode_key,
        args.frame_stem,
        "pointcloud_summary.json",
    )

    return ContactGuidedRemoteRotationConfig(
        pointcloud_npz=(args.pointcloud_npz or default_pointcloud).expanduser().resolve(),
        contact_summary=(args.contact_summary or default_contact).expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        part_a=args.part_a,
        part_b=args.part_b,
        far_quantile=min(max(args.far_quantile, 0.0), 0.99),
        hinge_neighbor_radius=args.hinge_neighbor_radius,
        collision_threshold=args.collision_threshold,
        collision_check_degrees=args.collision_check_degrees,
        rotation_degrees=args.rotation_degrees,
        waypoint_count=max(2, args.waypoint_count),
        max_collision_sample_points=max(1, args.max_collision_sample_points),
        max_visualization_points=max(1, args.max_visualization_points),
        source_summary=(args.source_summary or default_source).expanduser().resolve(),
    )


def main() -> None:
    cfg = parse_args()
    result, summary_path = run_contact_guided_remote_rotation(cfg)
    print(f"part A ({result.part_a}) label: {result.part_a_label_id}")
    print(f"part B ({result.part_b}) label: {result.part_b_label_id}")
    print(f"contact pixel xy: {result.contact_pixel_xy}")
    print(f"contact point a: {result.contact_point_3d}")
    print(f"axis point: {result.axis_point}")
    print(f"axis direction: {result.axis_direction}")
    print(f"chosen rotation degrees: {result.rotation_degrees:.3f}")
    print(f"waypoints: {result.waypoint_count}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
