"""Hand-authored CLI schema for the three Kurtorank v3 subcommands.

The sibling ``wsinsight/mcp`` reflects its Click-based CLI via
``wsinsight.cli.cli_schema.json``. ``kurtorank`` also uses Click, but
its per-option surface is small enough to enumerate by hand. Each
entry below mirrors an ``@click.option`` declaration in
:file:`kurtorank/annotate/main.py`, :file:`kurtorank/rank/main.py`,
or :file:`kurtorank/seed/main.py`.

Drift check: when you add a flag to the CLI, update the matching schema
below. ``rank-markers`` forwards all flags unchanged to the
underlying argparse layer in :file:`kurtorank/rank/main.py` — its
schema entry accepts the full argparse argv via ``--``.
"""

from __future__ import annotations

# Mirrors ``kurtorank.cli.cli`` subcommands.
COMMANDS: dict[str, list[dict]] = {
    "annotate": [
        {"name": "xenium_dir", "kind": "path", "required": True,
         "help": "Xenium 'outs' directory."},
        {"name": "markers_csv", "kind": "path", "required": False,
         "help": "Marker gene CSV (default: bundled markers-v6.csv)."},
        {"name": "tissue_type", "kind": "string", "required": True,
         "help": "Tissue type for marker filtering "
                 "(e.g. 'breast', 'lung')."},
        {"name": "output_dir", "kind": "path", "required": False,
         "help": "Output directory. Default: xenium-path."},
        {"name": "common_only", "kind": "bool_flag", "default": True,
         "help": "--common-only / --no-common-only."},
        {"name": "normal_only", "kind": "bool_flag", "default": True,
         "help": "--normal-only / --include-cancer."},
        {"name": "panel", "kind": "string", "required": False,
         "help": "Optional panel name to restrict markers."},
    ],
    "rank_markers": [
        # The CLI is a passthrough (Click.UNPROCESSED) into the argparse
        # module ``kurtorank.rank.main.rank_markers_main``. The MCP tool
        # therefore takes a single ``args`` list mirroring the full
        # ``kurtorank rank-markers --help`` surface (see that module for
        # the canonical list).
        {"name": "args", "kind": "string_list", "nargs": "+", "required": True,
         "help": "Passthrough argv list. Always start with the markers CSV "
                 "path (or ``-`` for stdin), then any combination of "
                 "``--atlas``, ``--species``, ``--tissue``, "
                 "``--output``, etc. See ``kurtorank rank-markers --help``."},
    ],
    "build_panel": [
        {"name": "atlases_csv", "kind": "string", "required": False,
         "help": "Comma-separated DISCO atlas slugs "
                 "(e.g. 'blood,lung,adipose_cell')."},
        {"name": "all_atlases", "kind": "bool_flag", "default": False,
         "help": "Fetch every atlas matching the type filter."},
        {"name": "list_atlases", "kind": "bool_flag", "default": False,
         "help": "Print the DISCO atlas catalog and exit (no download)."},
        {"name": "include_disease", "kind": "bool_flag", "default": False,
         "help": "Also include atlases where type=='disease'."},
        {"name": "include_celltype", "kind": "bool_flag", "default": False,
         "help": "Also include atlases where type=='cell type'."},
        {"name": "output_dir", "kind": "path", "required": False},
    ],
}


# All three commands are potentially long (Census download / annotation pass),
# so the MCP layer treats them as background jobs by default.
LONG_RUNNING: frozenset[str] = frozenset({"annotate", "rank_markers", "build_panel"})


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
