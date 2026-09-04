"""CLI schema for the Kurtorank subcommands, reflected from Click.

The sibling ``wsinsight/mcp`` reflects its Click CLI through a generated
``cli_schema.json``. This table used to be enumerated by hand and had drifted:
it offered ``--output-dir`` where ``build-panel`` takes ``--output``, an
``annotate --panel`` that does not exist, and omitted 27 real options. Reading
the options off the Click commands keeps the two in step by construction.

``rank-markers`` is the one hand-declared entry: it is a variadic positional
passthrough into an argparse layer, which the adapter appends without a flag.
"""

from __future__ import annotations

import click

from ..cli import cli

# MCP tool names cannot carry a dash.
_MCP_NAME = {name: name.replace("-", "_") for name in cli.commands}


def _entry(param: click.Option) -> dict:
    entry: dict = {
        "name": param.name,
        "kind": "bool_flag" if param.is_flag else "string",
        "required": bool(param.required),
        "help": param.help or "",
    }
    if param.multiple:
        entry["nargs"] = "+"
    if param.default is not None and param.default != ():
        entry["default"] = param.default
    if param.secondary_opts:
        # `--no-x` / `--include-cancer`: a flag defaulting to True cannot be
        # turned off by omission, so the adapter needs the other spelling.
        entry["off_flag"] = param.secondary_opts[0]
    if isinstance(param.type, click.Choice):
        entry["choices"] = list(param.type.choices)
    return entry


def _entries(command: click.Command) -> list[dict]:
    return [_entry(p) for p in command.params if isinstance(p, click.Option)]


COMMANDS: dict[str, list[dict]] = {
    _MCP_NAME[name]: _entries(cmd) for name, cmd in cli.commands.items()
}

# Positional passthrough; the adapter appends these verbatim after the command.
COMMANDS["rank_markers"] = [
    {"name": "args", "kind": "string_list", "nargs": "+", "required": True,
     "help": "Passthrough argv list. Start with the markers CSV path (or '-' "
             "for stdin), then any of --atlas, --species, --tissue, --output, "
             "etc. See `kurtorank rank-markers --help`."},
]


# All three commands are potentially long (Census download / annotation pass),
# so the MCP layer treats them as background jobs by default.
LONG_RUNNING: frozenset[str] = frozenset(COMMANDS)


def is_long_running(command: str) -> bool:
    """Return True when the command is expected to take >10s."""
    return command in LONG_RUNNING


def discover_commands() -> list[str]:
    """Return the list of MCP-exposed command names."""
    return list(COMMANDS.keys())


def get_command(name: str) -> list[dict]:
    """Return the schema entries for one command or raise KeyError."""
    if name not in COMMANDS:
        raise KeyError(f"unknown kurtorank command: {name!r}")
    return COMMANDS[name]
