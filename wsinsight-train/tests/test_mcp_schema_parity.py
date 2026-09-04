"""The MCP tool surface must be the CLI's, not a copy of it.

The hand-written table this replaced named flags argparse rejects and omitted
the required --tissue, so every one of those tools failed on first use.
"""

from __future__ import annotations

import pytest

from wsitrain import STAGES
from wsitrain.cli import parser_for
from wsitrain.mcp.adapters import args_to_argv
from wsitrain.mcp.schema import COMMANDS, LONG_RUNNING, get_command


def _real_flags(command: str) -> set[str]:
    return {a.dest for a in parser_for(command)._actions if a.option_strings}


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_no_schema_param_the_cli_would_reject(command):
    assert {p["name"] for p in COMMANDS[command]} <= _real_flags(command)


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_required_flags_are_offered(command):
    # --input has no default; a tool that cannot pass it is unusable.
    assert "input" in {p["name"] for p in COMMANDS[command]}


def test_every_stage_is_exposed():
    assert set(STAGES) <= set(COMMANDS)


def test_check_is_the_only_synchronous_command():
    assert set(COMMANDS) - LONG_RUNNING == {"check"}


def test_argv_is_accepted_by_the_real_parser():
    argv = args_to_argv("crop", get_command("crop"), {
        "input": "/data", "tissue": "breast", "output": "/out",
        "object_detection": "stardist", "architecture": "resnet50",
        "patch_size_pixels": 56, "patch_spacing_um_px": 0.274,
        "stain_normalization": True,
    })
    assert argv[0] == "crop"
    parser_for("crop").parse_args(argv[1:])


def test_off_flag_spells_out_a_false_tri_state():
    """stain_normalization defaults to unset, so False cannot mean omitted."""
    argv = args_to_argv("crop", get_command("crop"), {
        "input": "/data", "stain_normalization": False,
    })
    assert "--no-stain-normalization" in argv
    assert "--stain-normalization" not in argv
