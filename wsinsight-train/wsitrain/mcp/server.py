"""FastMCP server exposing ``wsitrain`` subcommands as MCP tools.

Each of the entries in :data:`wsitrain.mcp.schema.COMMANDS` is registered
as one MCP tool, named ``wsitrain_<command>`` (e.g. ``wsitrain_train``
so it doesn't collide with sibling packages in the same Copilot agent).

Long-running commands (``run``, every stage except ``check``) return a
``job_id`` immediately; the agent polls ``job_status`` /
``job_logs`` / ``cancel_job`` to monitor progress.

Short commands (``check``) run synchronously and return the same shape
the CLI emits.
"""

from __future__ import annotations

from typing import Annotated
from typing import Any

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "wsitrain.mcp requires the 'fastmcp' package. "
        "Install it with: pip install 'wsinsight-train[mcp]'"
    ) from exc

from pydantic import Field

from wsitrain.mcp.adapters import args_to_argv
from wsitrain.mcp.jobs import JobManager
from wsitrain.mcp.schema import discover_commands
from wsitrain.mcp.schema import get_command
from wsitrain.mcp.schema import is_long_running


def _make_pydantic_field(spec: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Convert one schema entry into ``(annotation, Field kwargs)``.

    MCP input-schemas are derived from the Pydantic annotations on the tool
    function. ``bool_flag`` becomes a flag-with-positive-default; everything
    else falls back to ``str`` so a single MCP call can serialize enums,
    paths, and free-form numbers through.
    """
    kind = str(spec.get("kind", "string")).lower()
    required = bool(spec.get("required", False))
    default = spec.get("default", None)
    nargs = spec.get("nargs")
    help_text = spec.get("help", "")

    if kind == "bool_flag":
        ann = bool
        meta = {"description": help_text, "default": default if default is not None else False}
        return ann, meta

    if nargs == "+":
        # Multi-value flag: list[str] in MCP, space-separated argv on CLI.
        ann = list[str]
        meta = {"description": help_text, "default": default if default is not None else []}
        return ann, meta

    ann = str
    has_default = default is not None
    meta = {"description": help_text}
    if has_default:
        meta["default"] = default
    elif not required:
        meta["default"] = None
    return ann, meta


def _short_executor(command: str, argv_tail: list[str], cwd: str | None) -> dict[str, Any]:
    """Run a short-running wsitrain command synchronously and return a dict."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "wsitrain.cli", command, *argv_tail]
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "status": "done" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "argv": cmd,
        "duration_s": None,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _build_tool(
    mcp: FastMCP,
    command: str,
    params: list[dict[str, Any]],
    job_manager: JobManager | None,
    cwd: str | None,
) -> None:
    """Register one MCP tool mirroring one wsitrain subcommand."""
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

    tool_name = f"wsitrain_{command}"

    desc = (
        f"Run `wsitrain {command}` synchronously and return the result."
        if not is_async_tool
        else f"Submit `wsitrain {command}` as a background MCP job; returns a job_id."
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
        def tool_fn(**kwargs: Any) -> dict[str, Any]:
            argv = args_to_argv(command, params, {**defaults, **kwargs})[1:]  # strip leading command
            return _short_executor(command, argv, cwd=cwd)

    tool_fn.__annotations__ = annotations  # type: ignore[attr-defined]
    tool_fn.__name__ = tool_name
    tool_fn.__doc__ = desc
    tool_fn.__signature__ = _build_signature(annotations, fields)  # type: ignore[attr-defined]

    mcp.tool(description=desc, name=tool_name)(tool_fn)


def _build_signature(annotations: dict[str, Any], fields: dict[str, Any]):
    """Mimic ``inspect.Signature`` from Pydantic fields so FastMCP accepts it."""
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
        name="WSInsight-Train MCP",
        instructions=(
            "Drive the wsitrain headless training pipeline. Tools of the "
            "form `wsitrain_<command>` mirror the CLI subcommands. Long "
            "commands return a job_id; poll job_status until done."
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

    # Always-available job-control tools.
    if job_manager is not None:

        @mcp.tool(
            name="wsitrain_job_status",
            description="Return the status of a long-running wsitrain job.",
        )
        def job_status(job_id: Annotated[str, Field(description="The job_id returned by wsitrain_* tool calls.")]) -> dict[str, Any]:
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
            name="wsitrain_job_logs",
            description="Return the last N log lines of a long-running wsitrain job.",
        )
        def job_logs(
            job_id: Annotated[str, Field()],
            tail: Annotated[int, Field(description="Lines to return (default 200).", default=200)] = 200,
        ) -> dict[str, Any]:
            job = job_manager.get(job_id)
            return {"job_id": job.job_id, "logs": job.tail(tail)}

        @mcp.tool(
            name="wsitrain_cancel_job",
            description="Cancel a long-running wsitrain job (SIGTERM).",
        )
        def cancel_job(
            job_id: Annotated[str, Field()],
        ) -> dict[str, Any]:
            job = job_manager.get(job_id)
            ok = job.cancel()
            return {"job_id": job_id, "cancelled": ok}

        @mcp.tool(
            name="wsitrain_list_jobs",
            description="List all currently tracked wsitrain jobs (running + finished).",
        )
        def list_jobs() -> dict[str, Any]:
            return {"jobs": job_manager.list_jobs()}

        # Touch the unused parameter helpers so ruff doesn't complain
        _ = (Annotated, Field)
    return mcp
