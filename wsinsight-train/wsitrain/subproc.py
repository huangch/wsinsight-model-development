"""Environment for the CellViT child processes."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_SHIM = Path(__file__).resolve().parent / "tqdmshim"


def child_env(cellvit: str) -> dict[str, str]:
    """Env for a CellViT subprocess: its root on PYTHONPATH, plus the tqdm shim.

    The child never imports wsitrain, so the only way the shared bar style and
    the SIGWINCH redraw reach its progress bars is a sitecustomize on the path.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_SHIM), cellvit])
    # The child's stdout is our pipe, so it cannot measure the terminal itself;
    # shutil.get_terminal_size() reads COLUMNS before probing the fd.
    if "COLUMNS" not in env:
        columns = shutil.get_terminal_size().columns
        if columns:
            env["COLUMNS"] = str(columns)
    return env
