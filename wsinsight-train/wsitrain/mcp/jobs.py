"""Background-job tracking for long-running wsitrain stages.

Each long-running call spawns ``python -m wsitrain.cli <stage> ...`` in
a child process and returns a ``job_id``. The agent polls ``job_status``
to wait + ``job_logs`` to peek at stdout/stderr, and ``cancel_job`` to
cooperatively terminate.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Optional


@dataclass
class Job:
    job_id: str
    command_argv: list[str]
    process: subprocess.Popen
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    cancelled: bool = False
    returncode: Optional[int] = None
    _log_tail: list[str] = field(default_factory=list)
    _tail_lock: threading.Lock = field(default_factory=threading.Lock)
    _TAIL_LIMIT: int = field(default=2000, repr=False)

    def append_log(self, line: str) -> None:
        with self._tail_lock:
            self._log_tail.append(line)
            if len(self._log_tail) > self._TAIL_LIMIT:
                del self._log_tail[: -self._TAIL_LIMIT]

    def tail(self, n: int = 200) -> str:
        with self._tail_lock:
            return "".join(self._log_tail[-n:])

    def pump_logs(self) -> None:
        if self.process.stdout is None:
            return
        for raw in self.process.stdout:
            if not raw:
                break
            self.append_log(raw)
            if raw.endswith("\n"):
                sys.stdout.write(raw)
                sys.stdout.flush()

    def status(self) -> str:
        if self.finished_at is None and self.process.poll() is not None:
            self.finished_at = time.time()
            self.returncode = self.process.returncode
        if self.cancelled:
            return "cancelled"
        if self.finished_at is None:
            return "running"
        return "done" if self.returncode == 0 else "error"

    def cancel(self) -> bool:
        if self.finished_at is not None:
            return False
        self.cancelled = True
        try:
            self.process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True


class JobManager:
    """Thread-safe registry of background ``Job`` instances."""

    def __init__(self, max_concurrent: int | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.max_concurrent = (
            max_concurrent
            if max_concurrent is not None
            else max(1, _gpu_count())
        )
        self._sem = threading.BoundedSemaphore(self.max_concurrent)

    def submit(self, argv: list[str], cwd: str | None = None) -> Job:
        """Spawn ``python -m wsitrain.cli <argv>`` and register the job."""
        job_id = uuid.uuid4().hex
        cmd = [sys.executable, "-m", "wsitrain.cli", *argv]
        proc = subprocess.Popen(  # noqa: SIM115 — keep kwargs explicit
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        job = Job(job_id=job_id, command_argv=cmd, process=proc)
        with self._lock:
            self._jobs[job_id] = job

        # Daemon thread that drains subprocess stdout and terminates the
        # job bookkeeping when the child exits.
        threading.Thread(
            target=self._supervise, args=(job,), daemon=True, name=f"mcp-job-{job_id[:6]}"
        ).start()
        return job

    def _supervise(self, job: Job) -> None:
        assert job.process.stdout is not None
        with job._tail_lock:  # noqa: SLF001 — internal lock
            try:
                for raw in job.process.stdout:
                    if not raw:
                        break
                    job.append_log(raw)
                    sys.stdout.write(raw)
                    sys.stdout.flush()
            finally:
                job.process.stdout.close()  # type: ignore[union-attr]

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown job_id: {job_id!r}")
            return self._jobs[job_id]

    def list_jobs(self) -> list[str]:
        with self._lock:
            return list(self._jobs.keys())

    def reap(self, max_age_s: float = 600.0) -> int:
        """Drop finished jobs older than ``max_age_s``. Returns count removed."""
        cutoff = time.time() - max_age_s
        with self._lock:
            dead = [
                jid
                for jid, j in self._jobs.items()
                if j.finished_at is not None and j.finished_at < cutoff
            ]
            for jid in dead:
                del self._jobs[jid]
        return len(dead)


def _gpu_count() -> int:
    try:
        import torch  # local import to keep CLI latency low

        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def available_cores() -> int:
    return max(1, shutil.get_terminal_size().columns)  # placeholder; not used

_ = available_cores  # silence linter
