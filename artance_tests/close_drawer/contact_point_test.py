from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARTANCE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(ARTANCE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTANCE_TESTS_ROOT))

from common.paths import ARTANCE_ROOT
from common.vlm_contact_point import (
    DEFAULT_VLM_MAX_TOKENS,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_SERVER_URL,
    VlmContactPointConfig,
    VlmContactPointParseError,
    VlmContactPointPromptSpec,
    query_vlm_contact_point,
    save_vlm_contact_point_failure_summary,
    save_vlm_contact_point_summary,
)


DEFAULT_IMAGE = (
    ARTANCE_ROOT
    / "RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_rgb.png"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs/contact_point"



CLOSE_DRAWER_PROMPT_SPEC = VlmContactPointPromptSpec(
    task_goal=(
        "Analyze the provided image and find the single best contact point for a robot "
        "to close the visible drawer."
    ),
    contact_rules=(
        "Choose exactly one point on the drawer, preferably on the handle or a rigid front surface that can be pushed safely to close the drawer.",
        "The point must lie on the visible drawer, not on the background, cabinet frame, robot, or floor.",
        "If the handle is visible, prefer the center of the handle or the most stable visible part of it.",
        "If the handle is occluded or absent, choose the best visible point on the drawer front suitable for pushing inward.",
    ),
    user_instruction="Find the single best contact point for the close drawer task.",
)

def parse_args() -> VlmContactPointConfig:
    parser = argparse.ArgumentParser(
        description="Predict the best 2D contact point for the close drawer task with a VLM."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--server-url", default=DEFAULT_VLM_SERVER_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_VLM_MAX_TOKENS,
        help="Optional output token limit. Omit by default to let the VLM service/provider choose.",
    )
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument(
        "--prompt-order",
        choices=("auto", "xy", "yx"),
        default="auto",
        help="Coordinate order requested from the VLM. auto uses yx for Gemini and xy otherwise.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    return VlmContactPointConfig(
        image=args.image.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        prompt_spec=CLOSE_DRAWER_PROMPT_SPEC,
        model=args.model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        prompt_order=None if args.prompt_order == "auto" else args.prompt_order,
        debug=args.debug,
    )


def main() -> None:
    cfg = parse_args()
    try:
        result = query_vlm_contact_point(cfg)
    except VlmContactPointParseError as exc:
        summary_path = save_vlm_contact_point_failure_summary(
            cfg,
            error=exc,
            raw_response=exc.raw_response,
            prompt_order=exc.prompt_order,
        )
        print("VLM contact point parsing failed.")
        print(f"model: {exc.model}")
        print(f"prompt_order: {exc.prompt_order}")
        print(f"raw_response_repr: {exc.raw_response!r}")
        print(f"Saved failure summary: {summary_path}")
        raise

    summary_path = save_vlm_contact_point_summary(cfg, result)

    print(f"prompt_order: {result.prompt_order}")
    print(f"raw_response_repr: {result.raw_response!r}")
    print(f"raw_coordinate: {list(result.raw_coordinate)}")
    print(f"normalized_xy: {list(result.normalized_xy)}")
    print(f"pixel_xy: {list(result.pixel_xy)}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
