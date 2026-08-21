"""Environment for the CellViT child processes."""
from __future__ import annotations

import os
from pathlib import Path

_SHIM = Path(__file__).resolve().parent / "tqdmshim"


def child_env(cellvit: str) -> dict[str, str]:
    """Env for a CellViT subprocess: its root on PYTHONPATH, plus the tqdm shim.

    The child never imports wsitrain, so the only way the shared bar style and
    the SIGWINCH redraw reach its progress bars is a sitecustomize on the path.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_SHIM), cellvit])
    return env
