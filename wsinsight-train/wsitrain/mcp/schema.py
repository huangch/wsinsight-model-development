"""Hand-authored CLI schema for the ``wsitrain`` subcommands.

The sibling ``wsinsight`` MCP server reflects its Click-based CLI via
``wsinsight.cli.cli_schema.json``. ``wsitrain`` uses argparse, so the
schema is hand-written here. Each entry below mirrors the argparse
definitions in :file:`wsitrain/cli.py` so the MCP tool input schema
matches the CLI ``--help`` output.

Updating: when you add a new flag to ``wsitrain.cli._add_common`` or to
a stage's ``_STAGE_FLAGS``, update the matching schema below. There is a
small post-commit reminder that grep's for the schema keys against the
CLI's ``--help`` text — keep this file in sync to avoid drift.
"""

from __future__ import annotations

# Mirrors ``wsitrain.SUB`` argparse subparsers (see cli.py).
COMMANDS: dict[str, list[dict]] = {
    "check": [
        {"name": "input", "kind": "string", "required": True,
         "help": "Path to a YAML config (see scripts/ for examples)."},
        {"name": "labels", "kind": "string", "nargs": "+", "required": False,
         "help": "Labelspace filter."},
        {"name": "gpus", "kind": "string", "required": False,
         "help": "Comma-separated GPU ids, e.g. '0,1'."},
        {"name": "output", "kind": "path", "required": False,
         "help": "Directory to write preflight report."},
    ],
    "run": [
        {"name": "input", "kind": "string", "required": True},
        {"name": "output", "kind": "path", "required": True},
        {"name": "labels", "kind": "string", "nargs": "+", "required": False},
        {"name": "stages", "kind": "string", "nargs": "+", "required": False,
         "help": "Stage names to run (default: all)."},
        {"name": "run_skip", "kind": "string", "nargs": "+", "required": False,
         "help": "Stages to skip."},
        {"name": "gpus", "kind": "string", "required": False},
        {"name": "num_workers", "kind": "int", "required": False},
        {"name": "pin_memory", "kind": "bool_flag", "default": False,
         "help": "Use pin_memory in DataLoader."},
        {"name": "dry_run", "kind": "bool_flag", "default": False},
    ],
    "annotate": [
        {"name": "input", "kind": "string", "required": True},
        {"name": "output", "kind": "path", "required": True},
    ],
    "segment": [
        {"name": "input", "kind": "string", "required": True},
        {"name": "output", "kind": "path", "required": True},
        {"name": "mpp", "kind": "string", "default": "0.5"},
        {"name": "thumbsize", "kind": "int", "default": 2048},
    ],
    "transfer": [
        {"name": "input", "kind": "string", "required": True},
        {"name": "output", "kind": "path", "required": True},
        {"name": "model_id", "kind": "string", "default": "cellvit"},
    ],
    "tile": [
        {"name": "input", "kind": "string", "required": True},
        {"name": "output", "kind": "path", "required": True},
        {"name": "tile_px", "kind": "string", "default": "1024"},
        {"name": "halo_px", "kind": "string", "default": "0"},
        {"name": "mpp", "kind": "string", "default": "0.5"},
    ],
    "split": [
        {"name": "input", "kind": "string", "required": True},
        {"name": "output", "kind": "path", "required": True},
        {"name": "fractions", "kind": "string", "default": "0.7,0.15,0.15"},
        {"name": "seed", "kind": "int", "default": 42},
    ],
    "train": [
        {"name": "input", "kind": "string", "required": True},
        {"name": "output", "kind": "path", "required": True},
        {"name": "gpus", "kind": "string", "required": False},
        {"name": "num_workers", "kind": "int", "default": 4},
        {"name": "max_epochs", "kind": "int", "default": 50},
        {"name": "learning_rate", "kind": "string", "default": "1e-4"},
    ],
    "validate": [
        {"name": "input", "kind": "string", "required": True},
        {"name": "output", "kind": "path", "required": True},
        {"name": "gpus", "kind": "string", "required": False},
    ],
    "export": [
        {"name": "input", "kind": "string", "required": True},
        {"name": "output", "kind": "path", "required": True},
        {"name": "format", "kind": "string", "default": "torchscript",
         "help": "torchscript|onnx"},
    ],
    "report": [
        {"name": "input", "kind": "string", "required": True},
        {"name": "output", "kind": "path", "required": True},
        {"name": "format", "kind": "string", "default": "html",
         "help": "html|markdown|json"},
    ],
}


# Mark which commands are long-running (return job_id).
LONG_RUNNING: frozenset[str] = frozenset({
    "run",
    "annotate", "segment", "transfer", "tile",
    "split", "train", "validate", "export", "report",
})


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
