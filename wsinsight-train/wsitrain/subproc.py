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
    # shutil.get_terminal_size() reads COLUMNS before probing the fd. Both
    # names are needed: tqdm's env fallback reads COLUMNS *and* LINES in one
    # comprehension and returns no size at all if either is missing, which
    # leaves every child bar unconstrained and wrapping in the parent's
    # terminal.
    if "COLUMNS" not in env or "LINES" not in env:
        size = shutil.get_terminal_size()
        if size.columns:
            env.setdefault("COLUMNS", str(size.columns))
        if size.lines:
            env.setdefault("LINES", str(size.lines))
    return env
