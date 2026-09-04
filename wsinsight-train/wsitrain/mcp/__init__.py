"""WSInsight-Train MCP (Model Context Protocol) server.

Exposes the ``wsitrain`` CLI subcommands (``check``, ``run``, and the ten
stages: ``annotate``, ``segment``, ``transfer``, ``tile``, ``crop``,
``split``, ``train``, ``validate``, ``export``, ``report``) as MCP tools so AI
agents (Claude Desktop, VS Code Copilot, Cursor) can drive training
pipelines through the same surface as human users.

Entry point: ``wsinsight-train-mcp`` (see :mod:`wsitrain.mcp.__main__`).
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_server"]


def build_server(*args: Any, **kwargs: Any):  # pragma: no cover - thin re-export
    from wsitrain.mcp.server import build_server as _build

    return _build(*args, **kwargs)
