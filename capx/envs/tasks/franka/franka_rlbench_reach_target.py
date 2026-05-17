from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling a Franka robot in an RLBench ReachTarget environment.
Goal: touch the target sphere with the Franka gripper.
Use the APIs described below.
All positions are XYZ in meters in the RLBench world frame.
All quaternions are WXYZ.
You may write Python comments for reasoning, but ONLY write executable Python code and do not use Markdown fences.
If you want to use numpy, import it explicitly.
"""

ORACLE_CODE = """
# Move the gripper tip to the privileged target pose.
obs = get_observation()
target_pos, _ = get_object_pose("target")
current_quat = obs["robot0_eef_quat"]

open_gripper()
goto_pose(target_pos, current_quat, z_approach=0.05)
"""


class FrankaRLBenchReachTargetCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for RLBench ReachTarget."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["FrankaRLBenchReachTargetCodeEnv"]
