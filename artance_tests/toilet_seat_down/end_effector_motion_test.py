from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARTANCE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(ARTANCE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTANCE_TESTS_ROOT))

from common.end_effector_motion import (
    RemoteRotationEndEffectorMotionConfig,
    run_remote_rotation_end_effector_motion_target,
)


TASK = "toilet_seat_down"
TASK_DIR = Path(__file__).resolve().parent
DEFAULT_EPISODE_KEY = "variation0_episode2_frame015_wrist"
DEFAULT_FRAME_STEM = "15"
DEFAULT_OUTPUT_DIR = TASK_DIR / "outputs/end_effector_motion"


def _default_path_from_episode(module: str, episode_key: str, frame_stem: str, filename: str) -> Path:
    return TASK_DIR / "outputs" / module / episode_key / frame_stem / filename


def parse_args() -> RemoteRotationEndEffectorMotionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a contact-guided remote-rotation summary into a world-frame "
            "end-effector motion target. The default route closes the gripper and "
            "pushes the moving part from the trailing side; the legacy route can "
            "still follow the Contact-GraspNet pose."
        )
    )
    parser.add_argument("--episode-key", default=DEFAULT_EPISODE_KEY)
    parser.add_argument("--frame-stem", default=DEFAULT_FRAME_STEM)
    parser.add_argument("--rotation-summary", type=Path)
    parser.add_argument("--graspnet-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--motion-mode", choices=("closed_gripper_push", "grasp_follow"), default="closed_gripper_push")
    parser.add_argument("--orientation-mode", choices=("keep-current", "identity"), default="keep-current")
    parser.add_argument("--approach-distance", type=float, default=0.08)
    parser.add_argument("--target-standoff", type=float, default=0.015)
    parser.add_argument("--close-gripper-before-motion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-gripper-after-step", default="grasp")
    args = parser.parse_args()

    rotation_summary = args.rotation_summary or _default_path_from_episode(
        "contact_guided_remote_rotation",
        args.episode_key,
        args.frame_stem,
        "contact_guided_remote_rotation_summary.json",
    )
    graspnet_summary = args.graspnet_summary
    if graspnet_summary is None and args.motion_mode == "grasp_follow":
        graspnet_summary = _default_path_from_episode(
            "contact_graspnet_pose",
            args.episode_key,
            args.frame_stem,
            "contact_graspnet_summary.json",
        )
    return RemoteRotationEndEffectorMotionConfig(
        task=TASK,
        rotation_summary=rotation_summary.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        graspnet_summary=graspnet_summary.expanduser().resolve() if graspnet_summary else None,
        close_gripper_after_step=args.close_gripper_after_step,
        motion_mode=args.motion_mode,
        orientation_mode=args.orientation_mode,
        approach_distance=args.approach_distance,
        target_standoff=args.target_standoff,
        close_gripper_before_motion=args.close_gripper_before_motion,
    )


def main() -> None:
    cfg = parse_args()
    target, summary_path = run_remote_rotation_end_effector_motion_target(cfg)
    print(f"task: {TASK}")
    print(f"joint type: {target.joint_type}")
    print(f"analysis path: {target.analysis_path}")
    print(f"motion mode: {target.metadata.get('motion_mode')}")
    print(f"close gripper before motion: {target.close_gripper_before_motion}")
    print(f"close gripper after steps: {target.close_gripper_after_steps}")
    for step in target.motion_steps:
        print(f"{step.name} position (world): {step.position}")
        print(f"{step.name} quaternion_wxyz: {step.quaternion_wxyz}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
