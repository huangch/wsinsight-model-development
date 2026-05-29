"""Subprocess helpers — uniform tee-to-log invocation for every v2 wrapper."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def sh(cmd: Sequence[str | Path], *, log_path: Path | None = None,
       cwd: Path | None = None, env_extra: Mapping[str, str] | None = None,
       dry: bool = False) -> int:
    """Run ``cmd`` with stdout+stderr tee'd to ``log_path`` (if given).

    Returns the child's exit code. Raises ``subprocess.CalledProcessError``
    on non-zero exit. Banner is printed first so the user can see exactly
    what is about to run.
    """
    cmd_str = [str(c) for c in cmd]
    banner = "$ " + " ".join(shlex.quote(c) for c in cmd_str)
    print(banner, file=sys.stderr, flush=True)
    if dry:
        return 0

    env = os.environ.copy()
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})

    if log_path is None:
        rc = subprocess.call(cmd_str, cwd=str(cwd) if cwd else None, env=env)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as fp:
            fp.write(banner + "\n")
            fp.flush()
            proc = subprocess.Popen(
                cmd_str, cwd=str(cwd) if cwd else None, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stderr.write(line)
                fp.write(line)
            rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd_str)
    return rc
