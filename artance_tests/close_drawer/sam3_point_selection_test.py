from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ARTANCE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(ARTANCE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTANCE_TESTS_ROOT))

from common.paths import ARTANCE_ROOT
from common.sam3_point_selector import PointSelectionConfig, run_sam3_point_selection_test
from common.sam3_runner import default_sam3_service_url


DEFAULT_IMAGE = (
    ARTANCE_ROOT
    / "RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_rgb.png"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs/sam3_point_selection"
DEFAULT_CONTACT_POINT_SUMMARY = (
    Path(__file__).resolve().parent
    / "outputs/contact_point/wrist_reference_frame_051_rgb/summary.json"
)
DEFAULT_PROMPTS = ("drawer handle", "drawer")


def _load_pixel_xy_from_contact_summary(summary_path: Path) -> tuple[int, int]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Contact point summary not found: {summary_path}")
    data: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    pixel_xy = data.get("result", {}).get("pixel_xy")
    if not (
        isinstance(pixel_xy, (list, tuple))
        and len(pixel_xy) == 2
        and all(isinstance(v, (int, float)) for v in pixel_xy)
    ):
        raise ValueError(f"Could not find result.pixel_xy in contact point summary: {summary_path}")
    return int(round(pixel_xy[0])), int(round(pixel_xy[1]))


def parse_args() -> PointSelectionConfig:
    parser = argparse.ArgumentParser(
        description="Select the SAM3 drawer/handle instance nearest to a 2D contact point."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--point",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        help="Pixel coordinate in image space. Overrides --contact-summary.",
    )
    parser.add_argument(
        "--contact-summary",
        type=Path,
        default=DEFAULT_CONTACT_POINT_SUMMARY,
        help="VLM contact point summary containing result.pixel_xy.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Text prompt for SAM3. Repeat to run multiple prompts.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--service-url", default=default_sam3_service_url())
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.point is not None:
        point_xy = (int(args.point[0]), int(args.point[1]))
    else:
        point_xy = _load_pixel_xy_from_contact_summary(args.contact_summary.expanduser().resolve())

    prompts = tuple(args.prompts) if args.prompts else DEFAULT_PROMPTS
    return PointSelectionConfig(
        image=args.image.expanduser().resolve(),
        point_xy=point_xy,
        prompts=prompts,
        output_dir=args.output_dir.expanduser().resolve(),
        service_url=args.service_url,
        top_k=max(1, args.top_k),
        show=args.show,
    )


def main() -> None:
    cfg = parse_args()
    prompt_results, selections, summary_path = run_sam3_point_selection_test(cfg)

    print(f"point_xy: {list(cfg.point_xy)}")
    for prompt, results in prompt_results.items():
        selected = selections[prompt]
        if selected is None:
            print(f"{prompt}: no SAM3 result")
            continue
        print(
            f"{prompt}: selected rank {selected.rank}/{len(results)} "
            f"score={selected.score:.3f} distance_px={selected.distance_px:.2f} "
            f"contains_point={selected.contains_point}"
        )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
