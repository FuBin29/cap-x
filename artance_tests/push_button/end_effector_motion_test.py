from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARTANCE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(ARTANCE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTANCE_TESTS_ROOT))

from common.prismatic_joint_motion import (
    PrismaticJointMotionConfig,
    run_prismatic_joint_motion_target,
)
from common.task_configs import get_task_config


TASK = 'push_button'
TASK_CONFIG = get_task_config(TASK)
MOTION_SPEC = TASK_CONFIG.prismatic_motion
if MOTION_SPEC is None:
    raise RuntimeError(f"Task {TASK!r} does not define prismatic_motion in task_configs.py")

DEFAULT_PLANE_SUMMARY = (
    Path(__file__).resolve().parent
    / 'outputs/part_adjacency_plane/variation0_episode2_frame036_wrist/36/part_adjacency_plane_summary.json'
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs/end_effector_motion"


def parse_args() -> PrismaticJointMotionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a part-adjacency plane summary into world-frame Franka "
            "motion targets for a prismatic-joint task."
        )
    )
    parser.add_argument("--plane-summary", type=Path, default=DEFAULT_PLANE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--frame-info",
        type=Path,
        default=None,
        help=(
            "Frame metadata JSON with wrist camera extrinsics and gripper_pose. "
            "Defaults to the info path recorded by the source pointcloud summary."
        ),
    )
    parser.add_argument("--camera-name", default="wrist")
    parser.add_argument(
        "--input-frame",
        choices=("auto", "camera", "world"),
        default="auto",
        help="Frame of the contact_center/plane_normal in the plane summary.",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("keep-current", "identity"),
        default="keep-current",
        help="End-effector quaternion source for solve_ik.",
    )
    parser.add_argument(
        "--normal-direction-sign",
        type=float,
        default=MOTION_SPEC.normal_direction_sign,
        help="Motion direction multiplier relative to plane_normal. Use -1 for -normal.",
    )
    parser.add_argument(
        "--approach-distance",
        type=float,
        default=MOTION_SPEC.approach_distance,
        help="Meters from contact point opposite the motion direction for the approach pose.",
    )
    parser.add_argument(
        "--target-standoff",
        type=float,
        default=MOTION_SPEC.target_standoff,
        help="Meters from contact point opposite the motion direction for the near-contact target pose.",
    )
    parser.add_argument(
        "--travel-distance",
        type=float,
        default=MOTION_SPEC.travel_distance,
        help="Meters to move from target pose along the prismatic motion direction.",
    )
    parser.add_argument(
        "--close-gripper-before-motion",
        action=argparse.BooleanOptionalAction,
        default=MOTION_SPEC.close_gripper_before_motion,
        help="Whether generated control sequence closes the gripper before moving.",
    )
    parser.add_argument("--target-key", default="contact_center")
    parser.add_argument("--normal-key", default="plane_normal")
    args = parser.parse_args()

    return PrismaticJointMotionConfig(
        plane_summary=args.plane_summary.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        frame_info=args.frame_info.expanduser().resolve() if args.frame_info else None,
        camera_name=args.camera_name,
        input_frame=args.input_frame,
        orientation_mode=args.orientation_mode,
        normal_direction_sign=args.normal_direction_sign,
        approach_distance=args.approach_distance,
        target_standoff=args.target_standoff,
        travel_distance=args.travel_distance,
        close_gripper_before_motion=args.close_gripper_before_motion,
        target_key=args.target_key,
        normal_key=args.normal_key,
    )


def main() -> None:
    cfg = parse_args()
    target, summary_path = run_prismatic_joint_motion_target(cfg)
    print(f"task: {TASK}")
    print(f"input frame: {target.input_frame}")
    print(f"contact position (world): {target.contact_position}")
    print(f"normal (world): {target.normal}")
    print(f"motion direction (world): {target.motion_direction}")
    print(f"approach position (world): {target.approach_position}")
    print(f"target position (world): {target.target_position}")
    print(f"final position (world): {target.final_position}")
    print(f"quaternion_wxyz: {target.quaternion_wxyz}")
    print(f"close gripper before motion: {target.close_gripper_before_motion}")
    if target.distance_current_to_approach is not None:
        print(f"distance current -> approach: {target.distance_current_to_approach:.4f} m")
    if target.distance_current_to_target is not None:
        print(f"distance current -> target: {target.distance_current_to_target:.4f} m")
    if target.distance_current_to_final is not None:
        print(f"distance current -> final: {target.distance_current_to_final:.4f} m")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
