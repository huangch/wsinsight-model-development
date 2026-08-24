"""Kurtorank MCP (Model Context Protocol) server.

Exposes the three Kurtorank v3 Click subcommands (``annotate``,
``rank-markers``, ``build-panel``) as MCP tools so AI agents can drive
the spatial-transcriptomics cell-type annotation pipeline from chat
surfaces (Claude Desktop, VS Code Copilot, Cursor).

Entry point: ``kurtorank-mcp`` (see :mod:`kurtorank.mcp.__main__`).
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_server"]


def build_server(*args: Any, **kwargs: Any):  # pragma: no cover - thin re-export
    from kurtorank.mcp.server import build_server as _build

    return _build(*args, **kwargs)
