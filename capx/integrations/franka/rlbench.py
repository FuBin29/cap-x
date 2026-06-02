from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase


class FrankaRLBenchApi(ApiBase):
    """Control helpers for a Franka arm in RLBench."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_rlbench_observation": self.get_rlbench_observation,
            "get_camera_observations": self.get_camera_observations,
            "get_gripper_pose_xyzw": self.get_gripper_pose_xyzw,
            "get_object_pose": self.get_object_pose,
            "goto_pose": self.goto_pose,
            "goto_pose_xyzw": self.goto_pose_xyzw,
            "move_to_joints": self.move_to_joints,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the latest observation.

        Returns:
            A dictionary with robot state, cameras, task text, object poses, and
            the most recent action result. Common keys include:
            ``robot0_joint_pos`` (7,), ``robot0_eef_pos`` (3,),
            ``robot0_eef_quat`` (4, WXYZ), ``gripper_pose_xyzw``
            (7, [x, y, z, qx, qy, qz, qw]), ``object_poses``,
            ``robot0_robotview``, and ``robot0_eye_in_hand`` when available.
        """
        return self._env.get_observation()

    def get_rlbench_observation(self) -> dict[str, Any]:
        """Get raw RLBench-style observation fields.

        Returns:
            A dictionary with keys such as ``front_rgb``, ``front_depth``,
            ``wrist_rgb``, ``wrist_depth``, ``joint_positions``,
            ``gripper_open``, and ``gripper_pose``. ``gripper_pose`` is
            ``[x, y, z, qx, qy, qz, qw]`` in XYZW quaternion order.
        """
        return dict(self._env.get_observation().get("rlbench_raw", {}))

    def get_camera_observations(
        self,
        camera_order: tuple[str, ...] = ("overhead", "left_shoulder", "right_shoulder", "wrist", "front"),
    ) -> dict[str, dict[str, Any]]:
        """Get RGB-D images and camera matrices by camera name.

        Args:
            camera_order: Camera names to return. Typical names are ``front``,
                ``wrist``, ``left_shoulder``, ``right_shoulder``, and ``overhead``.

        Returns:
            A dictionary keyed by camera name. Each value may contain ``rgb``
            (H,W,3 uint8), ``depth`` (H,W float32 meters), ``intrinsics`` (3,3),
            and ``extrinsics`` (4,4).
        """
        obs = self._env.get_observation()
        raw = obs.get("rlbench_raw", {})
        configs = obs.get("rlbench_camera_configs", {})
        result: dict[str, dict[str, Any]] = {}
        for camera in camera_order:
            config = configs.get(f"{camera}_camera", {})
            if f"{camera}_rgb" not in raw and camera not in obs:
                continue
            capx_camera = obs.get(camera, {})
            if camera == "front":
                capx_camera = obs.get("robot0_robotview", capx_camera)
            elif camera == "wrist":
                capx_camera = obs.get("robot0_eye_in_hand", capx_camera)
            result[camera] = {
                "rgb": raw.get(f"{camera}_rgb", capx_camera.get("images", {}).get("rgb")),
                "depth": raw.get(f"{camera}_depth", capx_camera.get("images", {}).get("depth")),
                "intrinsics": config.get("intrinsics", capx_camera.get("intrinsics")),
                "extrinsics": config.get("extrinsics", capx_camera.get("pose_mat")),
            }
        return result

    def get_gripper_pose_xyzw(self) -> np.ndarray:
        """Get the current RLBench gripper pose in native XYZW quaternion order.

        Returns:
            A numpy array of shape (7,) containing ``[x, y, z, qx, qy, qz, qw]``
            in the RLBench world frame. Use this with ``goto_pose_xyzw`` when you
            want to preserve the current wrist orientation while changing XYZ.
        """
        obs = self._env.get_observation()
        pose = obs.get("rlbench_raw", {}).get("gripper_pose")
        if pose is None:
            pose = obs.get("gripper_pose_xyzw")
        if pose is None:
            pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
            quat_wxyz = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
            pose = np.concatenate([pos, quat_wxyz[[1, 2, 3, 0]]])
        return np.asarray(pose, dtype=np.float64).reshape(7)

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get a privileged object pose from RLBench.

        Args:
            object_name: RLBench scene object name.

        Returns:
            position: (3,) XYZ in meters, in the RLBench world frame.
            quaternion_wxyz: (4,) unit quaternion in WXYZ order.
        """
        obs = self._env.get_observation()
        object_poses = obs.get("object_poses", {})
        if object_name not in object_poses:
            available = ", ".join(sorted(object_poses.keys())) or "<none>"
            raise ValueError(f"Object {object_name!r} not available. Known objects: {available}")
        pose = np.asarray(object_poses[object_name], dtype=np.float64).reshape(7)
        return pose[:3], pose[3:]

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> dict[str, Any]:
        """Move the end effector to a target pose.

        Args:
            position: (3,) target XYZ position in meters, in the world frame.
            quaternion_wxyz: (4,) target end-effector quaternion in WXYZ order.
            z_approach: Optional positive-Z approach distance in meters. If
                nonzero, first moves to ``position + [0, 0, z_approach]``.

        Returns:
            A result dictionary. On success, ``result["ok"]`` is True. On
            failure, ``result["ok"]`` is False and includes ``error_type`` and
            ``error``.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        result: dict[str, Any] = {"ok": True}
        if z_approach != 0.0:
            approach = pos + np.array([0.0, 0.0, float(z_approach)], dtype=np.float64)
            result = self._env.move_to_pose(approach, quat)
            if not result.get("ok", True):
                return result
        return self._env.move_to_pose(pos, quat)

    def goto_pose_xyzw(
        self,
        pose_xyzw: np.ndarray,
        gripper: float | None = None,
        ignore_collisions: bool = True,
        max_path_steps: int = 600,
    ) -> dict[str, Any]:
        """Move the end effector to a target pose in XYZW quaternion order.

        Args:
            pose_xyzw: (7,) target pose ``[x, y, z, qx, qy, qz, qw]`` in the
                world frame.
            gripper: Optional gripper command after the arm move. Use 1.0 to
                open and 0.0 to close. If None, only the arm pose is moved.
            ignore_collisions: Whether the planner ignores collisions.
            max_path_steps: Maximum path stepping iterations before timeout.

        Returns:
            A result dictionary. On success, ``result["ok"]`` is True. On
            failure, ``result["ok"]`` is False and includes ``error_type`` and
            ``error``.
        """
        return self._env.move_to_pose_xyzw(
            np.asarray(pose_xyzw, dtype=np.float64).reshape(7),
            gripper=gripper,
            ignore_collisions=ignore_collisions,
            max_path_steps=max_path_steps,
        )

    def move_to_joints(self, joints: np.ndarray) -> None:
        """Move the Franka arm to target joint positions.

        Args:
            joints: (7,) target joint positions in radians.

        Returns:
            None
        """
        self._env.move_to_joints_blocking(np.asarray(joints, dtype=np.float64).reshape(7))

    def open_gripper(self) -> None:
        """Open the gripper fully."""
        self._env.open_gripper()

    def close_gripper(self) -> None:
        """Close the gripper fully."""
        self._env.close_gripper()


__all__ = ["FrankaRLBenchApi"]
