from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling a Franka robot in an RLBench CloseDrawer environment.
Goal: close the currently selected drawer.
Use the APIs described below.
All positions are XYZ in meters in the RLBench world frame.
All quaternions are WXYZ.
You may write Python comments for reasoning, but ONLY write executable Python code and do not use Markdown fences.
If you want to use numpy, import it explicitly.
"""

ORACLE_CODE = """
# Use RLBench's privileged task waypoints for a deterministic smoke oracle.
open_gripper()

for waypoint_name in ["waypoint0", "waypoint1"]:
    position, quaternion = get_object_pose(waypoint_name)
    goto_pose(position, quaternion)
"""


class FrankaRLBenchCloseDrawerCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for RLBench CloseDrawer."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["FrankaRLBenchCloseDrawerCodeEnv"]
