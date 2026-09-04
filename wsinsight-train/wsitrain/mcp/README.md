# wsitrain.mcp — Model Context Protocol server for WSInsight-Train

Exposes the `wsitrain` headless training pipeline as MCP tools so AI agents
(Claude Desktop, VS Code Copilot, Cursor) can run CellViT training stages.

## Quick start

```bash
# 1. Install with the MCP extra
pip install -e ".[mcp]"

# 2. Start the server (stdio, for Claude Desktop / Copilot)
wsinsight-train-mcp

# Or run over Streamable HTTP on loopback:
wsinsight-train-mcp --http 127.0.0.1:8768
```

## Tools exposed

| MCP tool           | CLI equivalent                | mode       |
|--------------------|-------------------------------|------------|
| `wsitrain_check`   | `wsitrain check …`            | sync       |
| `wsitrain_run`     | `wsitrain run …`              | background |
| `wsitrain_annotate`| `wsitrain annotate …`        | background |
| `wsitrain_segment` | `wsitrain segment …`         | background |
| `wsitrain_transfer` | `wsitrain transfer …`       | background |
| `wsitrain_tile`    | `wsitrain tile …`             | background |
| `wsitrain_split`   | `wsitrain split …`            | background |
| `wsitrain_train`   | `wsitrain train …`            | background |
| `wsitrain_validate` | `wsitrain validate …`       | background |
| `wsitrain_export`  | `wsitrain export …`           | background |
| `wsitrain_report`  | `wsitrain report …`           | background |

Background tools return a `job_id` immediately. Monitor with:

| Tool | Purpose |
|------|---------|
| `wsitrain_job_status` | Returns status (`running` / `done` / `error` / `cancelled`). |
| `wsitrain_job_logs`   | Returns the last N log lines. |
| `wsitrain_cancel_job` | Sends SIGTERM to the child wsitrain process. |
| `wsitrain_list_jobs`  | Lists all tracked job ids. |

## Wire-up for Copilot

Add to `.vscode/settings.json`:

```jsonc
{
  "github.copilot.chat.mcp.servers": {
    "wsinsight-train": {
      "type": "stdio",
      "command": "/opt/anaconda3/envs/wsi/bin/python",
      "args": ["-m", "wsitrain.mcp"],
      "env": {
        "WSINSIGHT_ZOO_REGISTRY_PATH": "/workspace/wsinsight/devel/zoo/wsinsight-zoo-registry.json"
      }
    }
  }
}
```

Then restart GitHub Copilot Chat. The `wsitrain_*` toolset appears in the
agent's available tools.

## Architecture

```
wsitrain/mcp/
├── __init__.py     # public surface (build_server, build_server aliases)
├── __main__.py     # wsinsight-train-mcp console-script entry point
├── server.py       # FastMCP tool registration + per-tool argv builder
├── schema.py       # CLI schema, reflected from wsitrain/cli.py's argparse
├── adapters.py     # snake_case → kebab-case → argv translator
├── jobs.py         # background Job + JobManager (stdlib + threading)
└── README.md       # this file
```

The schema in [`schema.py`](./wsitrain/mcp/schema.py) is built from
`wsitrain.cli.parser_for()`, the same parser the CLI runs, so a new flag shows
up as a tool argument with no second edit. It used to be hand-written and had
drifted to naming flags argparse rejects while omitting the required
`--tissue`; `tests/test_mcp_schema_parity.py` now fails if the two diverge.
