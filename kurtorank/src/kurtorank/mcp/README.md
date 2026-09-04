# kurtorank.mcp — Model Context Protocol server for Kurtorank v3

Exposes the three Kurtorank v3 Click subcommands (`annotate`,
`rank-markers`, `build-panel`) as MCP tools so AI agents can drive the
spatial-transcriptomics cell-type annotation pipeline.

## Quick start

```bash
# 1. Install with the MCP extra
pip install -e ".[mcp]"

# 2. Start the server (stdio, for Claude Desktop / Copilot)
kurtorank-mcp

# Or run on loopback:
kurtorank-mcp --http 127.0.0.1:8769
```

## Tools exposed

| MCP tool | CLI equivalent | mode |
|---|---|---|
| `kurtorank_annotate` | `kurtorank annotate …` | background |
| `kurtorank_rank_markers` | `kurtorank rank-markers …` | background |
| `kurtorank_build_panel` | `kurtorank build-panel …` | background |

Background tools return a `job_id` immediately. Monitor with:

| Tool | Purpose |
|---|---|
| `kurtorank_job_status` | Returns status (`running` / `done` / `error` / `cancelled`). |
| `kurtorank_job_logs` | Returns the last N log lines. |
| `kurtorank_cancel_job` | Sends SIGTERM to the child kurtorank process. |
| `kurtorank_list_jobs` | Lists all tracked job ids. |

## Wire-up for Copilot

Add to `.vscode/settings.json`:

```jsonc
{
  "github.copilot.chat.mcp.servers": {
    "kurtorank": {
      "type": "stdio",
      "command": "/opt/anaconda3/envs/wsi/bin/python",
      "args": ["-m", "kurtorank.mcp"],
      "env": {
        "PYTHONPATH": "/workspace/wsinsight/wsinsight-model-development/kurtorank/src"
      }
    }
  }
}
```

Then restart GitHub Copilot Chat. The `kurtorank_*` toolset appears in the
agent's available tools.

## Caveat — `rank-markers`

`kurtorank rank-markers` is implemented as a Click passthrough to the
underlying argparse layer in `kurtorank/rank/main.py`. Rather than
mirroring every flag in the MCP schema (50+ flags), this tool takes
a single `args: list[str]` parameter that is forwarded verbatim. Run
`kurtorank rank-markers --help` locally to see the full available
surface.

## Architecture

```
kurtorank/mcp/
├── __init__.py     # public surface (build_server)
├── __main__.py     # kurtorank-mcp console-script entry point
├── server.py       # FastMCP tool registration + per-tool argv builder
├── schema.py       # CLI schema, reflected from the Click commands
├── adapters.py     # snake_case → argv translator (incl. passthrough path)
├── jobs.py         # background Job + JobManager
└── README.md       # this file
```

`schema.py` reads the options off `kurtorank.cli.cli`, so a new flag becomes a
tool argument with no second edit. It used to be enumerated by hand and had
drifted to offering `--output-dir` where `build-panel` takes `--output`;
`tests/test_mcp_schema_parity.py` now fails if the two diverge.
