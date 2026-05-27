from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.paths import ARTANCE_ROOT


EPISODE_RGB_RE = re.compile(
    r"(?P<prefix>.*?/RLBench-data/)"
    r"(?P<task>[^/]+)/variation(?P<variation>\d+)/episodes/episode(?P<episode>\d+)/"
    r"(?P<camera>[A-Za-z0-9_]+)_rgb/(?P<frame>\d+)\.png$"
)


class _OfflineRlbenchDemo:
    """Minimal stand-in for rlbench.demo.Demo used by trusted local pickles."""

    def __len__(self) -> int:
        return len(self._observations)

    def __getitem__(self, index: int) -> Any:
        return self._observations[index]


class _OfflineRlbenchObservation:
    """Minimal stand-in for rlbench.backend.observation.Observation."""

    def get_low_dim_data(self) -> np.ndarray:
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [
            self.joint_velocities,
            self.joint_positions,
            self.joint_forces,
            self.gripper_pose,
            self.gripper_joint_positions,
            self.gripper_touch_forces,
            self.task_low_dim_state,
        ]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if low_dim_data else np.array([])


class _OfflineRlbenchUnpickler(pickle.Unpickler):
    _RLBENCH_CLASSES = {
        ("rlbench.demo", "Demo"): _OfflineRlbenchDemo,
        ("rlbench.backend.observation", "Observation"): _OfflineRlbenchObservation,
    }

    def find_class(self, module: str, name: str) -> Any:
        replacement = self._RLBENCH_CLASSES.get((module, name))
        if replacement is not None:
            return replacement
        return super().find_class(module, name)


@dataclass(frozen=True)
class EpisodeFrame:
    task: str
    variation: int
    episode: int
    frame: int
    camera: str
    rgb: Path

    @property
    def key(self) -> str:
        return (
            f"variation{self.variation}_episode{self.episode}_"
            f"frame{self.frame:03d}_{self.camera}"
        )

    @property
    def image_stem(self) -> str:
        return str(self.frame)

    @property
    def episode_dir(self) -> Path:
        return self.rgb.parents[1]

    @property
    def depth(self) -> Path:
        return self.episode_dir / f"{self.camera}_depth" / f"{self.frame}.png"

    @property
    def low_dim_obs(self) -> Path:
        return self.episode_dir / "low_dim_obs.pkl"


def parse_episode_rgb_path(path: Path | str) -> EpisodeFrame:
    rgb = Path(path).expanduser().resolve()
    match = EPISODE_RGB_RE.match(str(rgb))
    if match is None:
        raise ValueError(
            "Episode path must look like "
            ".../RLBench-data/<task>/variationN/episodes/episodeM/<camera>_rgb/<frame>.png; "
            f"got {rgb}"
        )
    return EpisodeFrame(
        task=match.group("task"),
        variation=int(match.group("variation")),
        episode=int(match.group("episode")),
        frame=int(match.group("frame")),
        camera=match.group("camera"),
        rgb=rgb,
    )


def load_episode_list(path: Path | None = None) -> tuple[EpisodeFrame, ...]:
    episode_file = path or (ARTANCE_ROOT / "cap-x/artance_tests/episodes.txt")
    frames: list[EpisodeFrame] = []
    for raw_line in episode_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        frames.append(parse_episode_rgb_path(line))
    return tuple(frames)


def find_episode_frame(
    task: str,
    *,
    variation: int | None = None,
    episode: int | None = None,
    frame: int | None = None,
    camera: str = "wrist",
    episode_file: Path | None = None,
) -> EpisodeFrame:
    matches = [
        item
        for item in load_episode_list(episode_file)
        if item.task == task
        and item.camera == camera
        and (variation is None or item.variation == variation)
        and (episode is None or item.episode == episode)
        and (frame is None or item.frame == frame)
    ]
    if not matches:
        raise ValueError(
            f"No episode matched task={task!r}, variation={variation}, "
            f"episode={episode}, frame={frame}, camera={camera!r}."
        )
    return matches[0]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _low_dim_observations(data: Any) -> Any:
    return data._observations if hasattr(data, "_observations") else data


def _load_offline_low_dim_observations(path: Path) -> Any:
    with path.open("rb") as handle:
        return _low_dim_observations(_OfflineRlbenchUnpickler(handle).load())


def _observation_summary(obs: Any, camera: str) -> dict[str, Any]:
    misc = getattr(obs, "misc", None) or {}
    summary: dict[str, Any] = {
        "joint_positions": getattr(obs, "joint_positions", None),
        "joint_velocities": getattr(obs, "joint_velocities", None),
        "joint_forces": getattr(obs, "joint_forces", None),
        "gripper_open": getattr(obs, "gripper_open", None),
        "gripper_pose": getattr(obs, "gripper_pose", None),
        "gripper_matrix": getattr(obs, "gripper_matrix", None),
        "gripper_joint_positions": getattr(obs, "gripper_joint_positions", None),
        "gripper_touch_forces": getattr(obs, "gripper_touch_forces", None),
        "task_low_dim_state": getattr(obs, "task_low_dim_state", None),
        f"{camera}_camera_misc": {
            key: value for key, value in misc.items() if key.startswith(f"{camera}_camera")
        },
        "variation_index": misc.get("variation_index"),
        "joint_position_action": misc.get("joint_position_action"),
    }
    if hasattr(obs, "get_low_dim_data"):
        summary["low_dim_data"] = obs.get_low_dim_data()
    return summary


def write_episode_frame_info_json(episode: EpisodeFrame, output_dir: Path) -> Path:
    """Materialize camera metadata for one offline RLBench frame.

    This reads the trusted local RLBench `low_dim_obs.pkl` next to the RGB-D
    episode files. The resulting JSON follows the same camera-metadata shape as
    the close_drawer live visualization outputs.
    """
    if not episode.low_dim_obs.exists():
        raise FileNotFoundError(f"low_dim_obs.pkl not found: {episode.low_dim_obs}")

    observations = _load_offline_low_dim_observations(episode.low_dim_obs)

    if episode.frame < 0 or episode.frame >= len(observations):
        raise IndexError(
            f"frame={episode.frame} out of range for {episode.low_dim_obs} "
            f"with {len(observations)} observations"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = output_dir / f"{episode.key}_info.json"
    summary = _observation_summary(observations[episode.frame], episode.camera)
    summary["episode_source"] = {
        "task": episode.task,
        "variation": episode.variation,
        "episode": episode.episode,
        "frame": episode.frame,
        "camera": episode.camera,
        "rgb": str(episode.rgb),
        "depth": str(episode.depth),
        "low_dim_obs": str(episode.low_dim_obs),
    }
    info_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return info_path
