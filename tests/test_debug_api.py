from typing import Any

import numpy as np

import capx.integrations  # noqa: F401 - imports register built-in APIs
from capx.envs.base import BaseEnv
from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
from capx.integrations.base_api import get_api, list_apis


class DebugApiFakeEnv(BaseEnv):
    max_steps = 10

    def __init__(self) -> None:
        self._sim_step_count = 3

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.get_observation(), {}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._sim_step_count += 1
        return self.get_observation(), 0.0, False, False, {}

    def get_observation(self) -> dict[str, Any]:
        return {
            "robot_joint_pos": np.array([0.1, 0.2, 0.3, 0.75], dtype=np.float64),
            "robot_cartesian_pos": np.array(
                [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 0.75],
                dtype=np.float64,
            ),
        }

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False


def test_robot_state_debug_api_is_registered() -> None:
    assert "RobotStateDebugApi" in list_apis()


def test_robot_state_debug_api_doc_contains_tool_name() -> None:
    api = get_api("RobotStateDebugApi")(DebugApiFakeEnv())
    doc = api.combined_doc()

    assert "describe_robot_state" in doc
    assert "Return the current robot state" in doc


def test_describe_robot_state_returns_available_fields() -> None:
    api = get_api("RobotStateDebugApi")(DebugApiFakeEnv())

    state = api.describe_robot_state()

    assert state == {
        "robot_joint_pos": [0.1, 0.2, 0.3, 0.75],
        "robot_cartesian_pos": [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 0.75],
        "gripper_state": 0.75,
        "sim_step_count": 3,
    }


def test_describe_robot_state_is_available_in_code_execution_namespace() -> None:
    env = CodeExecutionEnvBase(
        CodeExecEnvConfig(
            low_level=DebugApiFakeEnv(),
            apis=["RobotStateDebugApi"],
            prompt="Inspect the robot state.",
        )
    )

    obs, _ = env.reset()
    prompt_text = obs["full_prompt"][1]["content"][0]["text"]
    _, _, _, _, info = env.step("state = describe_robot_state(); print(state)")

    assert "describe_robot_state" in prompt_text
    assert info["sandbox_rc"] == 0
    assert "'robot_joint_pos': [0.1, 0.2, 0.3, 0.75]" in info["stdout"]
