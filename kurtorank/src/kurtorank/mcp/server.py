"""FastMCP server exposing ``kurtorank`` subcommands as MCP tools.

Each of the entries in :data:`kurtorank.mcp.schema.COMMANDS` is
registered as one MCP tool, named ``kurtorank_<command>``. The
``rank-markers`` command is special-cased to forward its full argparse
argv via a single ``args`` list parameter; the other two use the same
per-flag translation as the sibling ``wsitrain.mcp`` server.

All three Kurtorank v3 subcommands are treated as background jobs by
default (Census download / annotation pass run for several minutes).
The same four job-control tools (``kurtorank_job_status``,
``kurtorank_job_logs``, ``kurtorank_cancel_job``,
``kurtorank_list_jobs``) are always exposed.
"""

from __future__ import annotations

from typing import Annotated
from typing import Any

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "kurtorank.mcp requires the 'fastmcp' package. "
        "Install it with: pip install 'kurtorank[mcp]'"
    ) from exc

from pydantic import Field

from kurtorank.mcp.adapters import args_to_argv
from kurtorank.mcp.jobs import JobManager
from kurtorank.mcp.schema import discover_commands
from kurtorank.mcp.schema import get_command
from kurtorank.mcp.schema import is_long_running


def _make_pydantic_field(spec: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """One schema entry → (annotation, Field kwargs)."""
    kind = str(spec.get("kind", "string")).lower()
    required = bool(spec.get("required", False))
    default = spec.get("default", None)
    nargs = spec.get("nargs")
    help_text = spec.get("help", "")

    if kind == "bool_flag":
        ann = bool
        meta = {
            "description": help_text,
            "default": default if default is not None else False,
        }
        return ann, meta

    if nargs == "+":
        ann = list[str]
        meta = {"description": help_text, "default": default if default is not None else []}
        return ann, meta

    if kind == "string_list":
        ann = list[str]
        meta = {"description": help_text, "default": default if default is not None else []}
        return ann, meta

    ann = str
    meta = {"description": help_text}
    has_default = default is not None
    if has_default:
        meta["default"] = default
    elif not required:
        meta["default"] = None
    return ann, meta


def _build_signature(annotations: dict[str, Any], fields: dict[str, Any]):
    """Build an inspect.Signature for the FastMCP tool binding."""
    import inspect

    params = []
    for name, field in fields.items():
        ann = annotations.get(name, str)
        default = field.default if field.default is not None else ...
        params.append(
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=ann,
                default=default,
            )
        )
    return inspect.Signature(parameters=params, return_annotation=dict)


def _build_tool(
    mcp: FastMCP,
    command: str,
    params: list[dict[str, Any]],
    job_manager: JobManager | None,
    cwd: str | None,
) -> None:
    annotations: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    fields: dict[str, Any] = {}

    for spec in params:
        ann, meta = _make_pydantic_field(spec)
        annotations[spec["name"]] = ann
        fields[spec["name"]] = Field(**meta)
        if "default" in meta:
            defaults[spec["name"]] = meta["default"]

    is_async_tool = job_manager is not None and is_long_running(command)

    tool_name = f"kurtorank_{command}"

    desc = (
        f"Run `kurtorank {command.replace('_', '-')}` synchronously and return the result."
        if not is_async_tool
        else f"Submit `kurtorank {command.replace('_', '-')}` as a background MCP job; returns a job_id."
    )

    if is_async_tool:
        def tool_fn(**kwargs: Any) -> dict[str, Any]:
            assert job_manager is not None
            argv = args_to_argv(command, params, {**defaults, **kwargs})
            job = job_manager.submit(argv, cwd=cwd)
            return {
                "job_id": job.job_id,
                "status": job.status(),
                "argv": job.command_argv,
                "started_at": job.started_at,
            }
    else:
        # ``rank-markers`` (and any future sync-only command) — Click has
        # no clean way to short-circuit a passthrough, so we always run
        # synchronously even when the schema marks it long-running and
        # the user passed ``--sync`` at __main__.
        def tool_fn(**kwargs: Any) -> dict[str, Any]:
            argv = args_to_argv(command, params, {**defaults, **kwargs})[1:]
            import subprocess
            import sys as _sys

            cmd = [_sys.executable, "-m", "kurtorank", command.replace("_", "-"), *argv]
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
            return {
                "status": "done" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "argv": cmd,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }

    tool_fn.__annotations__ = annotations  # type: ignore[attr-defined]
    tool_fn.__name__ = tool_name
    tool_fn.__doc__ = desc
    tool_fn.__signature__ = _build_signature(annotations, fields)  # type: ignore[attr-defined]

    mcp.tool(description=desc, name=tool_name)(tool_fn)


def build_server(
    max_concurrent: int | None = None,
    cwd: str | None = None,
    async_long_running: bool = True,
) -> FastMCP:
    """Construct (but don't start) the FastMCP server.

    Returns
    -------
    fastmcp.FastMCP
        Call ``mcp.run()`` for stdio or ``mcp.run(transport='http', ...)``
        for Streamable HTTP.
    """
    mcp = FastMCP(
        name="Kurtorank MCP",
        instructions=(
            "Drive the kurtorank spatial-transcriptomics annotation "
            "pipeline. Tools of the form `kurtorank_<command>` mirror "
            "the CLI. Long commands return a job_id; poll job_status "
            "until done."
        ),
    )
    job_manager = JobManager(max_concurrent=max_concurrent) if async_long_running else None

    for cmd in discover_commands():
        params = get_command(cmd)
        if not params:
            continue
        _build_tool(
            mcp=mcp,
            command=cmd,
            params=params,
            job_manager=job_manager if is_long_running(cmd) else None,
            cwd=cwd,
        )

    if job_manager is not None:

        @mcp.tool(
            name="kurtorank_job_status",
            description="Return the status of a long-running kurtorank job.",
        )
        def job_status(
            job_id: Annotated[str, Field(description="job_id returned by kurtorank_* tool calls.")]
        ) -> dict[str, Any]:
            job = job_manager.get(job_id)
            return {
                "job_id": job.job_id,
                "status": job.status(),
                "returncode": job.returncode,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "argv": job.command_argv,
            }

        @mcp.tool(
            name="kurtorank_job_logs",
            description="Return the last N log lines of a long-running kurtorank job.",
        )
        def job_logs(
            job_id: Annotated[str, Field()],
            tail: Annotated[int, Field(description="Lines to return.", default=200)] = 200,
        ) -> dict[str, Any]:
            job = job_manager.get(job_id)
            return {"job_id": job.job_id, "logs": job.tail(tail)}

        @mcp.tool(
            name="kurtorank_cancel_job",
            description="Cancel a long-running kurtorank job (SIGTERM).",
        )
        def cancel_job(job_id: Annotated[str, Field()]) -> dict[str, Any]:
            job = job_manager.get(job_id)
            ok = job.cancel()
            return {"job_id": job_id, "cancelled": ok}

        @mcp.tool(
            name="kurtorank_list_jobs",
            description="List all currently tracked kurtorank jobs.",
        )
        def list_jobs() -> dict[str, Any]:
            return {"jobs": job_manager.list_jobs()}

    return mcp
