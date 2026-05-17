from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling a Franka robot in an RLBench PickUpCup environment.
Goal: pick up the cup.
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

close_gripper()

position, quaternion = get_object_pose("waypoint2")
goto_pose(position, quaternion)
"""


class FrankaRLBenchPickUpCupCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for RLBench PickUpCup."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["FrankaRLBenchPickUpCupCodeEnv"]
