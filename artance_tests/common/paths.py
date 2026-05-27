from __future__ import annotations

import sys
from pathlib import Path


ARTANCE_TESTS_ROOT = Path(__file__).resolve().parents[1]
CAPX_ROOT = ARTANCE_TESTS_ROOT.parent
ARTANCE_ROOT = CAPX_ROOT.parent


def ensure_capx_on_path() -> None:
    """Make cap-x importable when tests are run from artance_tests."""
    capx_root = str(CAPX_ROOT)
    if capx_root not in sys.path:
        sys.path.insert(0, capx_root)
