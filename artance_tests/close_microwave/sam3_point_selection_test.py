from __future__ import annotations

import sys
from pathlib import Path

ARTANCE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(ARTANCE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTANCE_TESTS_ROOT))

from common.task_cli import run_sam3_point_selection_cli


if __name__ == "__main__":
    run_sam3_point_selection_cli("close_microwave")
