from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase


class FrankaRLBenchApi(ApiBase):
    """Control helpers for Franka in an RLBench remote environment."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_object_pose": self.get_object_pose,
            "goto_pose": self.goto_pose,
            "move_to_joints": self.move_to_joints,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the latest RLBench observation.

        Returns:
            A dictionary containing robot state, camera observations, task text, and object poses.
            The dictionary contains the following keys:
            - obs["robot0_joint_pos"]: (7,) Franka joint positions in radians.
            - obs["robot0_joint_vel"]: (7,) Franka joint velocities.
            - obs["robot0_gripper_qpos"]: (1,) gripper open amount, 1.0 open and 0.0 closed.
            - obs["robot0_eef_pos"]: (3,) end-effector XYZ in the RLBench world frame.
            - obs["robot0_eef_quat"]: (4,) end-effector quaternion in WXYZ order.
            - obs["robot0_robotview"]["images"]["rgb"]: front camera RGB image, uint8.
            - obs["robot0_robotview"]["images"]["depth"]: front camera depth image, float32.
            - obs["robot0_robotview"]["intrinsics"]: front camera intrinsic matrix, shape (3, 3).
            - obs["robot0_robotview"]["pose_mat"]: front camera extrinsic matrix, shape (4, 4).
            - obs["robot0_eye_in_hand"]: wrist camera observation when the adapter exposes wrist camera.
            - obs["object_poses"]: object poses as [x, y, z, qw, qx, qy, qz].
            - obs["target_pose"]: target pose alias when the current task exposes a target object.
            - obs["last_action_result"]: most recent control helper result when available.

            For reach-only motions, prefer using obs["robot0_eef_quat"] as the target
            gripper orientation. Object quaternions from get_object_pose() describe
            object orientation and are usually not valid gripper target orientations.
        """
        return self._env.get_observation()

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
        """Move the end effector to a target pose through the RLBench adapter.

        Args:
            position: (3,) XYZ in meters, in the RLBench world frame.
            quaternion_wxyz: (4,) target gripper quaternion in WXYZ order.
                For reaching a target object, use the current end-effector quaternion
                from obs["robot0_eef_quat"] unless you intentionally need to rotate
                the wrist. Do not pass an object's quaternion as the gripper target
                orientation unless that is explicitly intended.
            z_approach: Optional world-frame positive-Z approach distance in meters.
                If nonzero, first moves to position + [0, 0, z_approach], then to position.

        Returns:
            A dictionary with ok=True on success. If RLBench path planning fails,
            returns ok=False with an error string instead of raising a sandbox error.
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

    def move_to_joints(self, joints: np.ndarray) -> None:
        """Move the Franka arm to target joint positions.

        Args:
            joints: (7,) target joint positions in radians.

        Returns:
            None
        """
        self._env.move_to_joints_blocking(np.asarray(joints, dtype=np.float64).reshape(7))

    def open_gripper(self) -> None:
        """Open the gripper fully.

        Args:
            None

        Returns:
            None
        """
        self._env.open_gripper()

    def close_gripper(self) -> None:
        """Close the gripper fully.

        Args:
            None

        Returns:
            None
        """
        self._env.close_gripper()


__all__ = ["FrankaRLBenchApi"]
