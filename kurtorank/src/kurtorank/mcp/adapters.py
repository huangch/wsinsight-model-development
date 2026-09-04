"""Translate snake_case MCP kwargs → Click argv for ``kurtorank`` subcommands.

Kurtorank uses Click's argument naming: ``--snake-case`` flags are passed
through verbatim. The exceptions are:

  * ``rank_markers`` — argv passthrough; the entire ``args`` list is
    forwarded verbatim (the underlying argparse layer handles every
    flag).
  * Bool flags are emitted as ``--flag`` (when set) and omitted when
    False (Click defaults handle the negative).
"""

from __future__ import annotations

from typing import Any
from typing import Iterable


def _snake_to_kebab(name: str) -> str:
    return name.replace("_", "-")


def args_to_argv(command: str, params: Iterable[dict], kwargs: dict[str, Any]) -> list[str]:
    """Build a ``[command, ...flags]`` argv list from MCP kwarg dict."""
    by_name = {p["name"]: p for p in params}
    argv: list[str] = [command]

    # Special-case: ``rank-markers`` is a passthrough. The whole kwargs
    # dict's ``args`` field is forwarded verbatim; we don't construct any
    # flags of our own.
    if command == "rank_markers":
        argv.extend(str(a) for a in kwargs.get("args", []) if a is not None)
        return argv

    for name, value in kwargs.items():
        if value is None:
            continue
        spec = by_name.get(name)
        if spec is None:
            continue  # forward-compat: drop unknown
        kind = str(spec.get("kind", "string")).lower()
        flag = "--" + _snake_to_kebab(name)

        if kind == "bool_flag":
            if value:
                argv.append(flag)
            elif spec.get("off_flag"):
                # A flag defaulting to on cannot be turned off by omission.
                argv.append(spec["off_flag"])
            continue

        if isinstance(value, (list, tuple)):
            if spec.get("nargs") == "+":
                argv.append(flag)
                argv.extend(str(v) for v in value)
            else:
                for v in value:
                    argv.extend([flag, str(v)])
            continue

        argv.extend([flag, str(value)])
    return argv
