"""Translate snake_case MCP kwargs → argparse argv for ``wsitrain`` subcommands.

The MCP tool surface takes ``snake_case`` parameter names (matching the
JSON schema in :mod:`wsitrain.mcp.schema`). ``wsitrain`` CLI uses
``--kebab-case`` long-form flags. This module:

  * converts every ``snake_case`` kwarg into ``--kebab-case``
  * joins multi-value (``nargs="+"``) arguments as space-separated tokens
  * keeps ``bool_flag`` fields as ``--flag`` / ``--no-flag``

The function is shared between the synchronous-call path and the
job-spawning path so both produce byte-identical argv lists.
"""

from __future__ import annotations

from typing import Any
from typing import Iterable


def _snake_to_kebab(name: str) -> str:
    return name.replace("_", "-")


def args_to_argv(command: str, params: Iterable[dict], kwargs: dict[str, Any]) -> list[str]:
    """Build a ``[command, ...flags]`` argv list from MCP kwarg dict.

    Parameters
    ----------
    command
        The wsitrain subcommand (``check``, ``run``, ``train``, ...).
    params
        The schema entries for ``command`` (from :func:`wsitrain.mcp.schema.get_command`).
    kwargs
        The MCP-side kwargs, in snake_case.
    """
    by_name = {p["name"]: p for p in params}
    argv: list[str] = [command]
    for name, value in kwargs.items():
        if value is None:
            continue  # user did not pass it
        spec = by_name.get(name)
        if spec is None:
            # Unknown key: drop silently to remain forward-compatible with
            # old LLM agents that pass stale kwargs.
            continue
        kind = str(spec.get("kind", "string")).lower()
        flag = "--" + _snake_to_kebab(name)

        if kind == "bool_flag":
            if value:
                argv.append(flag)
            # False → omit (the CLI default is False for every bool_flag).
            continue

        if isinstance(value, (list, tuple)):
            if not value:
                continue
            if spec.get("nargs") == "+":
                argv.append(flag)
                argv.extend(str(v) for v in value)
            else:
                # Single-token repeat, used by --foo bar --foo baz.
                for v in value:
                    argv.extend([flag, str(v)])
            continue

        if kind == "path" and not str(value).startswith("/"):
            value = str(value)

        argv.extend([flag, str(value)])
    return argv
