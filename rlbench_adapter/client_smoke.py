#!/usr/bin/env python3
"""Host-side smoke client for the RLBench adapter server."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import urllib.error
import urllib.request
import uuid
from typing import Any

import numpy as np


def _request(
    opener: urllib.request.OpenerDirector,
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if context is not None:
        if context.get("episode_id") is not None:
            headers["X-RLBench-Episode-ID"] = str(context["episode_id"])
        if context.get("task_name"):
            headers["X-RLBench-Task-Name"] = str(context["task_name"])
        if context.get("action_sequence_id") is not None:
            headers["X-RLBench-Action-Sequence-ID"] = str(context["action_sequence_id"])
    if method == "POST":
        headers["X-RLBench-Request-ID"] = uuid.uuid4().hex
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            if context is not None:
                _update_context(context, result)
            return result
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(body)
        except json.JSONDecodeError:
            error_payload = None
        if context is not None and isinstance(error_payload, dict):
            _update_context(context, error_payload)
        raise RuntimeError(
            f"{method} {path} failed with HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {path} failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {path} returned invalid JSON") from exc


def _update_context(context: dict[str, Any], result: dict[str, Any]) -> None:
    if "episode_id" in result:
        context["episode_id"] = result["episode_id"]
    if "task_name" in result:
        context["task_name"] = result["task_name"]
    if "action_sequence_id" in result:
        context["action_sequence_id"] = result["action_sequence_id"]
    observation = result.get("observation")
    if isinstance(observation, dict):
        if "episode_id" in observation:
            context["episode_id"] = observation["episode_id"]
        if "task_name" in observation:
            context["task_name"] = observation["task_name"]
        if "action_sequence_id" in observation:
            context["action_sequence_id"] = observation["action_sequence_id"]


def _decode_array(payload: dict[str, Any] | None) -> np.ndarray | None:
    if payload is None:
        return None
    raw = base64.b64decode(payload["data"])
    if payload.get("encoding") == "base64_gzip":
        raw = gzip.decompress(raw)
    elif payload.get("encoding") != "base64":
        raise RuntimeError(f"Unsupported array encoding: {payload.get('encoding')}")
    return np.frombuffer(raw, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:8120")
    parser.add_argument("--task", help="Optionally switch/reset the server to this RLBench task.")
    parser.add_argument("--variation", type=int, default=0)
    args = parser.parse_args()

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    context: dict[str, Any] = {}
    get = lambda path: _request(opener, args.server_url, "GET", path, context=context)
    post = lambda path, payload=None: _request(
        opener, args.server_url, "POST", path, payload or {}, context=context
    )

    print("health", _request(opener, args.server_url, "GET", "/health"))
    reset_payload: dict[str, Any] = {"variation": args.variation}
    if args.task:
        reset_payload["task_name"] = args.task
    reset = post("/reset", reset_payload)
    obs = reset["observation"]
    cameras = obs.get("cameras") or {}
    rgb_payload = obs.get("rgb")
    depth_payload = obs.get("depth")
    if rgb_payload is None and cameras:
        first_camera = next(iter(cameras.values()))
        rgb_payload = first_camera.get("rgb")
        depth_payload = first_camera.get("depth")
    rgb = _decode_array(rgb_payload)
    depth = _decode_array(depth_payload)

    print("task", reset["task_name"])
    print("descriptions", reset["descriptions"])
    if rgb is not None:
        print("rgb", rgb.shape, rgb.dtype, int(rgb.min()), int(rgb.max()))
    else:
        print("rgb", None)
    if depth is not None:
        print("depth", depth.shape, depth.dtype, float(np.nanmin(depth)), float(np.nanmax(depth)))
    else:
        print("depth", None)
    print("cameras", sorted(cameras.keys()))
    for name, camera in sorted(cameras.items()):
        cam_rgb = _decode_array(camera["rgb"])
        cam_depth = _decode_array(camera["depth"])
        print(
            "camera",
            name,
            None if cam_rgb is None else cam_rgb.shape,
            None if cam_rgb is None else cam_rgb.dtype,
            None if cam_depth is None else cam_depth.shape,
            None if cam_depth is None else cam_depth.dtype,
            np.asarray(camera["intrinsics"]).shape,
            np.asarray(camera["pose_mat"]).shape,
        )
    print("joints", np.asarray(obs["joint_positions"]).shape)
    front_camera = obs.get("front_camera") or {}
    print("intrinsics", np.asarray(front_camera.get("intrinsics")).shape)
    print("pose_mat", np.asarray(front_camera.get("pose_mat")).shape)
    print("target", obs["object_poses"].get("target"))
    print("reward", get("/reward"))
    print("success", get("/success"))

    action = [0.0] * 7 + [1.0]
    step = post("/step", {"action": action})
    print("step", {"reward": step["reward"], "terminate": step["terminate"], "success": step["success"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
