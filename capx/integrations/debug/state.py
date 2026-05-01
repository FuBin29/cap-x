from typing import Any

import numpy as np

from capx.integrations.base_api import ApiBase


def _to_python_value(value: Any) -> Any:
    """Convert common observation values into printable Python containers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_python_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python_value(v) for v in value]
    return value


def _last_numeric_value(value: Any) -> float | None:
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    try:
        return float(arr.reshape(-1)[-1])
    except (TypeError, ValueError):
        return None


class RobotStateDebugApi(ApiBase):
    """Read-only robot state helpers for debugging generated code."""

    def functions(self) -> dict[str, Any]:
        return {"describe_robot_state": self.describe_robot_state}

    def describe_robot_state(self) -> dict[str, Any]:
        """Return the current robot state for debugging.

        Returns:
            state:
                A dictionary with robot_joint_pos, robot_cartesian_pos,
                gripper_state, and sim_step_count when available. Missing fields
                are omitted.
        """
        obs = self._env.get_observation()
        state: dict[str, Any] = {}

        if "robot_joint_pos" in obs:
            state["robot_joint_pos"] = _to_python_value(obs["robot_joint_pos"])
        if "robot_cartesian_pos" in obs:
            state["robot_cartesian_pos"] = _to_python_value(obs["robot_cartesian_pos"])

        gripper_state = None
        if "robot_cartesian_pos" in obs:
            gripper_state = _last_numeric_value(obs["robot_cartesian_pos"])
        if gripper_state is None and "robot_joint_pos" in obs:
            gripper_state = _last_numeric_value(obs["robot_joint_pos"])
        if gripper_state is not None:
            state["gripper_state"] = gripper_state

        if "sim_step_count" in obs:
            state["sim_step_count"] = int(obs["sim_step_count"])
        elif hasattr(self._env, "_sim_step_count"):
            state["sim_step_count"] = int(self._env._sim_step_count)

        self._log_step("describe_robot_state", str(state))
        return state
