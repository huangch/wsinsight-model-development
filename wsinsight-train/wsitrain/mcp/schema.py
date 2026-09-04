"""CLI schema for the ``wsitrain`` subcommands, reflected from argparse.

The sibling ``wsinsight`` MCP server reflects its Click CLI through a generated
``cli_schema.json``. This table used to be hand-written and had drifted badly:
it named thirteen flags the CLI rejects (``max_epochs``, ``learning_rate``,
``fractions``, ...), omitted the required ``--tissue`` everywhere, and never
gained the stages added after it. Building it from the same parser the CLI runs
keeps the two in step by construction.
"""

from __future__ import annotations

import argparse

from .. import STAGES
from ..cli import parser_for

# Config plumbing an agent has no use for; --input/--tissue/--output stay.
_HIDDEN = {"help", "config", "reset_config", "show_config"}


def _kind(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "bool_flag"
    if action.dest in ("input", "output"):
        return "path"
    if action.type is int:
        return "int"
    if action.type is float:
        return "float"
    return "string"


def _entries(command: str) -> list[dict]:
    by_dest: dict[str, dict] = {}
    for action in parser_for(command)._actions:
        if action.dest in _HIDDEN or not action.option_strings:
            continue
        primary = max(action.option_strings, key=len)
        if isinstance(action, argparse._StoreFalseAction) and action.dest in by_dest:
            # `--no-x` shares its dest with `--x`. A flag whose default is unset
            # cannot express "off" by omission, so the adapter needs the spelling.
            by_dest[action.dest]["off_flag"] = primary
            continue
        entry: dict = {
            "name": action.dest,
            "kind": _kind(action),
            "required": bool(action.required),
            "help": action.help or "",
        }
        if action.nargs in ("+", "*"):
            entry["nargs"] = "+"
        if action.default is not None and action.default != []:
            entry["default"] = action.default
        if action.choices:
            entry["choices"] = list(action.choices)
        by_dest[action.dest] = entry
    return list(by_dest.values())


COMMANDS: dict[str, list[dict]] = {
    name: _entries(name) for name in ("check", "run", *STAGES)
}


# `check` only inspects the cohort; everything else can run for hours.
LONG_RUNNING: frozenset[str] = frozenset(set(COMMANDS) - {"check"})


def is_long_running(command: str) -> bool:
    """Return True when the command is expected to take >10s."""
    return command in LONG_RUNNING


def discover_commands() -> list[str]:
    """Return the list of MCP-exposed command names."""
    return list(COMMANDS.keys())


def get_command(name: str) -> list[dict]:
    """Return the schema entries for one command or raise KeyError."""
    if name not in COMMANDS:
        raise KeyError(f"unknown wsitrain command: {name!r}")
    return COMMANDS[name]
