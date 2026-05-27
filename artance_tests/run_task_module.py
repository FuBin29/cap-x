#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ARTANCE_TESTS_ROOT = Path(__file__).resolve().parent
if str(ARTANCE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTANCE_TESTS_ROOT))

from common.task_cli import MODULES, add_task_argument, run_module_cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one ArtAnce module test for one RLBench task/episode frame."
    )
    parser.add_argument("module", choices=MODULES)
    add_task_argument(parser)
    args, rest = parser.parse_known_args()

    sys.argv = [sys.argv[0], *rest]
    run_module_cli(args.module, args.task)


if __name__ == "__main__":
    main()
