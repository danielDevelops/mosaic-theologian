"""Resume-safe job state.

Resume safety here is a property of the write order, not of a signal handler.
Windows PowerShell can kill this process without running cleanup, and it takes
any child process with it, so nothing may depend on shutdown code running.

Two rules make that survivable:

1. State advances only after the artifact it describes exists on disk.
2. Artifacts are written to <name>.part and renamed into place. os.replace is
   atomic, so a killed process leaves either the old file or the complete new
   one, never a torn one.

A kill therefore costs at most the single in-flight item, which is re-queued on
the next start because its recorded step has no matching artifact.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

# Ordered pipeline steps. A job's step only ever moves forward.
STEPS = [
    "discovered",
    "page_saved",
    "audio_downloaded",
    "transcribed",
    "chunked",
    "indexed",
]
STEP_ORDER = {name: i for i, name in enumerate(STEPS)}

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write via .part then rename. Never leaves a partial file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding=encoding, newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def cleanup_parts(*directories: Path) -> int:
    """Remove stray .part files left by a hard kill. Returns how many."""
    removed = 0
    for directory in directories:
        if not directory.exists():
            continue
        for stray in directory.rglob("*.part"):
            try:
                stray.unlink()
                removed += 1
            except OSError:
                pass
    return removed


@dataclass
class Job:
    """One unit of work. `key` is the dedup identity, `id` is the storage id."""

    id: str
    kind: str                      # "message" | "page" | "bible"
    url: str
    key: str = ""                  # date|campus|audio_url for messages
    step: str = "discovered"
    status: str = STATUS_PENDING
    title: str = ""
    speaker: str = ""
    date: str = ""
    campus: str = ""
    series: str = ""
    audio_url: str = ""
    page_path: str = ""
    audio_path: str = ""
    transcript_path: str = ""
    alt_urls: list[str] = field(default_factory=list)
    scripture_refs: list[str] = field(default_factory=list)
    error: str = ""
    updated: str = field(default_factory=utcnow)

    def at_least(self, step: str) -> bool:
        return STEP_ORDER.get(self.step, -1) >= STEP_ORDER[step]

    def advance(self, step: str) -> None:
        """Move the step forward only; never regress a completed job."""
        if STEP_ORDER[step] > STEP_ORDER.get(self.step, -1):
            self.step = step
        self.updated = utcnow()


class JobStore:
    """Append-only JSONL. Last record for an id wins."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line can only come from a kill mid-append.
                    # Skipping it just re-queues that one item.
                    continue
                job = Job(**{k: v for k, v in raw.items() if k in Job.__annotations__})
                self._jobs[job.id] = job
                if job.key:
                    self._by_key.setdefault(job.key, job.id)

    def __len__(self) -> int:
        return len(self._jobs)

    def __iter__(self) -> Iterator[Job]:
        return iter(self._jobs.values())

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def by_key(self, key: str) -> Job | None:
        job_id = self._by_key.get(key)
        return self._jobs.get(job_id) if job_id else None

    def has_url(self, url: str) -> bool:
        return any(j.url == url or url in j.alt_urls for j in self._jobs.values())

    def put(self, job: Job) -> Job:
        """Append the job's current state and flush it to disk immediately."""
        job.updated = utcnow()
        self._jobs[job.id] = job
        if job.key:
            self._by_key.setdefault(job.key, job.id)
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(asdict(job), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return job

    def pending(self, step: str | None = None) -> list[Job]:
        out = [j for j in self._jobs.values() if j.status != STATUS_DONE]
        if step:
            out = [j for j in out if not j.at_least(step)]
        return sorted(out, key=lambda j: (j.date or "", j.id))

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for job in self._jobs.values():
            tally[job.step] = tally.get(job.step, 0) + 1
            tally[f"status:{job.status}"] = tally.get(f"status:{job.status}", 0) + 1
        tally["total"] = len(self._jobs)
        return tally

    def requeue_missing_artifacts(self, root: Path) -> int:
        """Re-queue jobs whose recorded step has no artifact behind it.

        This is the other half of crash safety: state may be one step ahead of
        reality if the process died between the rename and the append.
        """
        fixed = 0
        for job in list(self._jobs.values()):
            expected: list[tuple[str, str]] = []
            if job.page_path:
                expected.append(("page_saved", job.page_path))
            if job.transcript_path:
                expected.append(("transcribed", job.transcript_path))

            for step, rel in expected:
                if job.at_least(step) and not (root / rel).exists():
                    job.step = "discovered"
                    job.status = STATUS_PENDING
                    job.error = f"artifact missing for {step}; re-queued"
                    self.put(job)
                    fixed += 1
                    break
        return fixed


class RunLock:
    """Single-run lock that does not survive a hard kill.

    The PID is recorded so a stale lock from a killed run is reclaimed instead
    of blocking every future run.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            import subprocess

            try:
                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True, text=True, timeout=10,
                )
                return str(pid) in out.stdout
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return False

    def acquire(self) -> bool:
        if self.path.exists():
            try:
                info = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(info.get("pid", -1))
            except Exception:
                # An unreadable lock can only come from a kill mid-write, so
                # treat it as stale rather than blocking every future run.
                pid = -1
            if self._pid_alive(pid):
                return False
            self.path.unlink(missing_ok=True)

        atomic_write_json(self.path, {"pid": os.getpid(), "started": utcnow()})
        self.acquired = True
        return True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "RunLock":
        if not self.acquire():
            raise RuntimeError(f"Another run holds the lock at {self.path}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class Deadline:
    """Wall-clock stop, checked between items so work halts on a boundary."""

    def __init__(self, until: str | None = None, max_minutes: int | None = None) -> None:
        self.stop_at: float | None = None
        now = time.time()

        if max_minutes:
            self.stop_at = now + max_minutes * 60

        if until:
            hours, _, minutes = until.partition(":")
            target = time.localtime(now)
            candidate = time.mktime((
                target.tm_year, target.tm_mon, target.tm_mday,
                int(hours), int(minutes or 0), 0, 0, 0, -1,
            ))
            if candidate <= now:
                candidate += 24 * 3600
            self.stop_at = min(self.stop_at, candidate) if self.stop_at else candidate

    def expired(self) -> bool:
        return self.stop_at is not None and time.time() >= self.stop_at

    def remaining_minutes(self) -> float:
        if self.stop_at is None:
            return float("inf")
        return max(0.0, (self.stop_at - time.time()) / 60.0)
