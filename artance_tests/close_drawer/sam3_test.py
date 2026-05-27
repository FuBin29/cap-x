from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARTANCE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(ARTANCE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTANCE_TESTS_ROOT))

from common.paths import ARTANCE_ROOT
from common.sam3_runner import Sam3RunConfig, default_sam3_service_url, run_sam3_text_test


DEFAULT_IMAGE = (
    ARTANCE_ROOT
    / "RLBench/tests/close_drawer/visualizations/wrist_reference_frame_051_rgb.png"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs/sam3"
DEFAULT_PROMPTS = ("drawer handle", "drawer")


def parse_args() -> Sam3RunConfig:
    parser = argparse.ArgumentParser(description="Run SAM3 on close drawer RGB observations.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Text prompt for SAM3. Repeat to run multiple prompts.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--service-url", default=default_sam3_service_url())
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    prompts = tuple(args.prompts) if args.prompts else DEFAULT_PROMPTS
    return Sam3RunConfig(
        image=args.image.expanduser().resolve(),
        prompts=prompts,
        output_dir=args.output_dir.expanduser().resolve(),
        service_url=args.service_url,
        top_k=max(1, args.top_k),
        show=args.show,
    )


def main() -> None:
    cfg = parse_args()
    prompt_results, summary_path = run_sam3_text_test(cfg)

    for prompt, results in prompt_results.items():
        print(f"{prompt}: {len(results)} result(s)")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
