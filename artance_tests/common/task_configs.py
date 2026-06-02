from __future__ import annotations

from dataclasses import dataclass

from common.vlm_contact_point import VlmContactPointPromptSpec


@dataclass(frozen=True)
class PrismaticJointMotionSpec:
    normal_direction_sign: float
    approach_distance: float
    target_standoff: float
    travel_distance: float
    close_gripper_before_motion: bool = True


@dataclass(frozen=True)
class ImplicitDoorRemoteRotationSpec:
    door_part: str
    handle_part: str
    door_opening_type: str = "side"
    hinge_axis_orientation: str = "vertical"
    closing_direction_sign: float = -1.0
    rotation_degrees: float = 40.0


@dataclass(frozen=True)
class TaskModuleConfig:
    task: str
    display_name: str
    contact_prompt: VlmContactPointPromptSpec
    sam3_prompts: tuple[str, ...]
    contact_grasp_prompt: str | None = None
    prismatic_motion: PrismaticJointMotionSpec | None = None
    implicit_door_remote_rotation: ImplicitDoorRemoteRotationSpec | None = None


TASK_CONFIGS: dict[str, TaskModuleConfig] = {
    "close_drawer": TaskModuleConfig(
        task="close_drawer",
        display_name="Close Drawer",
        contact_prompt=VlmContactPointPromptSpec(
            task_goal=(
                "Analyze the provided image and find the single best contact point "
                "for a robot to close the visible drawer."
            ),
            contact_rules=(
                "Choose exactly one point on the drawer, preferably on the handle or a rigid front surface that can be pushed safely to close the drawer.",
                "The point must lie on the visible drawer, not on the background, cabinet frame, robot, or floor.",
                "If the handle is visible, prefer the center of the handle or the most stable visible part of it.",
                "If the handle is occluded or absent, choose the best visible point on the drawer front suitable for pushing inward.",
            ),
            user_instruction="Find the single best contact point for the close drawer task.",
        ),
        sam3_prompts=("drawer handle", "drawer"),
        prismatic_motion=PrismaticJointMotionSpec(
            normal_direction_sign=-1.0,
            approach_distance=0.08,
            target_standoff=0.015,
            travel_distance=0.09,
            close_gripper_before_motion=True,
        ),
    ),
    "close_fridge": TaskModuleConfig(
        task="close_fridge",
        display_name="Close Fridge",
        contact_prompt=VlmContactPointPromptSpec(
            task_goal=(
                "Analyze the provided image and find the single best contact point "
                "for a robot to close the visible fridge door."
            ),
            contact_rules=(
                "Choose exactly one point on the fridge door, preferably on the handle or a rigid front surface that can be pushed safely to close it.",
                "The point must lie on the visible fridge door or handle, not on the cabinet frame, robot, floor, or background.",
                "If the handle is visible, prefer the center of the handle or the most stable visible part of it.",
                "If the handle is not visible, choose a visible rigid area of the door suitable for pushing it closed.",
            ),
            user_instruction="Find the single best contact point for the close fridge task.",
        ),
        sam3_prompts=("fridge door handle", "fridge door"),
        implicit_door_remote_rotation=ImplicitDoorRemoteRotationSpec(
            door_part="fridge door",
            handle_part="fridge door handle",
            door_opening_type="side",
            hinge_axis_orientation="vertical",
            closing_direction_sign=-1.0,
            rotation_degrees=40.0,
        ),
    ),
    "close_microwave": TaskModuleConfig(
        task="close_microwave",
        display_name="Close Microwave",
        contact_prompt=VlmContactPointPromptSpec(
            task_goal=(
                "Analyze the provided image and find the single best contact point "
                "for a robot to close the visible microwave door."
            ),
            contact_rules=(
                "Choose exactly one point on the microwave door, preferably on the handle or a rigid front surface that can be pushed safely to close it.",
                "The point must lie on the visible microwave door or handle, not on the robot, countertop, floor, or background.",
                "If the handle is visible, prefer the center of the handle or the most stable visible part of it.",
                "If the handle is not visible, choose a visible rigid area of the door suitable for pushing it closed.",
            ),
            user_instruction="Find the single best contact point for the close microwave task.",
        ),
        sam3_prompts=("microwave door handle", "door"),
        implicit_door_remote_rotation=ImplicitDoorRemoteRotationSpec(
            door_part="door",
            handle_part="microwave door handle",
            door_opening_type="side",
            hinge_axis_orientation="vertical",
            closing_direction_sign=-1.0,
            rotation_degrees=40.0,
        ),
    ),
    "push_button": TaskModuleConfig(
        task="push_button",
        display_name="Push Button",
        contact_prompt=VlmContactPointPromptSpec(
            task_goal=(
                "Analyze the provided image and find the single best contact point "
                "for a robot to press the visible button."
            ),
            contact_rules=(
                "Choose exactly one point near the center of the visible push button.",
                "The point must lie on the button surface, not on the surrounding panel, robot, table, or background.",
                "Prefer a point that allows a straight and stable pushing motion into the button.",
            ),
            user_instruction="Find the single best contact point for the push button task.",
        ),
        sam3_prompts=("button", "button panel"),
        prismatic_motion=PrismaticJointMotionSpec(
            normal_direction_sign=-1.0,
            approach_distance=0.055,
            target_standoff=0.008,
            travel_distance=0.035,
            close_gripper_before_motion=True,
        ),
    ),
    "toilet_seat_down": TaskModuleConfig(
        task="toilet_seat_down",
        display_name="Toilet Seat Down",
        contact_prompt=VlmContactPointPromptSpec(
            task_goal=(
                "Analyze the provided image and find the single best contact point "
                "for a robot to move the visible toilet lid down."
            ),
            contact_rules=(
                "Choose exactly one point on the visible toilet lid that can be contacted safely.",
                "The point must lie on the movable toilet lid, not on the bowl, robot, floor, wall, or background.",
                "Prefer a visible rigid area near the front or side edge where a pushing motion can lower the lid.",
            ),
            user_instruction="Find the single best contact point for the toilet lid down task.",
        ),
        sam3_prompts=("toilet lid", "toilet seat"),
    ),
}


def get_task_config(task: str) -> TaskModuleConfig:
    try:
        return TASK_CONFIGS[task]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_CONFIGS))
        raise ValueError(f"Unsupported task {task!r}. Supported tasks: {supported}") from exc
