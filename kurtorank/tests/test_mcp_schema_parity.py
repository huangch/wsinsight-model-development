"""The MCP tool surface must be the Click CLI's, not a copy of it.

The hand-written table this replaced offered ``--output-dir`` where
``build-panel`` takes ``--output``, so that tool failed on first use.
"""

from __future__ import annotations

import click
import pytest

from kurtorank.cli import cli
from kurtorank.mcp.adapters import args_to_argv
from kurtorank.mcp.schema import COMMANDS, LONG_RUNNING, get_command

# rank-markers is a variadic positional passthrough, not a set of options.
_PASSTHROUGH = "rank_markers"
_CLI_NAME = {name.replace("-", "_"): name for name in cli.commands}


def _click_params(mcp_name: str) -> set[str]:
    return {p.name for p in cli.commands[_CLI_NAME[mcp_name]].params}


@pytest.mark.parametrize("command", sorted(set(COMMANDS) - {_PASSTHROUGH}))
def test_no_schema_param_the_cli_would_reject(command):
    assert {p["name"] for p in COMMANDS[command]} <= _click_params(command)


@pytest.mark.parametrize("command", sorted(set(COMMANDS) - {_PASSTHROUGH}))
def test_every_click_option_is_offered(command):
    options = {p.name for p in cli.commands[_CLI_NAME[command]].params
               if isinstance(p, click.Option)}
    assert options <= {p["name"] for p in COMMANDS[command]}


def test_every_cli_command_is_exposed():
    assert set(COMMANDS) == {n.replace("-", "_") for n in cli.commands}


def test_all_commands_run_as_jobs():
    assert set(COMMANDS) == set(LONG_RUNNING)


def test_argv_is_accepted_by_the_real_command():
    argv = args_to_argv("build_panel", get_command("build_panel"),
                        {"output": "/tmp/panel.csv", "all_atlases": True})
    cli.commands["build-panel"].make_context("build-panel", argv[1:])


def test_off_flag_turns_a_default_on_flag_off():
    """--common-only defaults to on, so False cannot mean omitted."""
    argv = args_to_argv("annotate", get_command("annotate"), {"common_only": False})
    assert "--no-common-only" in argv
    assert "--common-only" not in argv


def test_passthrough_takes_no_flag():
    argv = args_to_argv(_PASSTHROUGH, get_command(_PASSTHROUGH),
                        {"args": ["markers.csv", "--species", "human"]})
    assert argv == [_PASSTHROUGH, "markers.csv", "--species", "human"]
