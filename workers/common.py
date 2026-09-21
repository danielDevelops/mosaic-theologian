"""Shared worker plumbing: arguments, settings, state, and the run lock."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow "python -m workers.x" from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ROOT, Paths, load_settings  # noqa: E402
from lib.state import Deadline, JobStore, RunLock   # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--until", default=None,
                        help="Stop at this wall clock time, e.g. 06:00")
    parser.add_argument("--max-minutes", type=int, default=None,
                        help="Stop after this many minutes")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Maximum items to process this run")
    parser.add_argument("--force", action="store_true",
                        help="Redo steps that are already complete")
    return parser


class WorkerContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.settings: dict[str, Any] = load_settings()
        self.paths = Paths()
        self.paths.ensure_all()
        self.root = ROOT
        self.jobs = JobStore(self.paths.state / "jobs.jsonl")
        self.deadline = Deadline(
            until=getattr(args, "until", None),
            max_minutes=getattr(args, "max_minutes", None),
        )
        self.force: bool = bool(getattr(args, "force", False))
        self.batch_size: int = int(getattr(args, "batch_size", 0) or 0)
        self._lock = RunLock(self.paths.state / "night.lock")

    def __enter__(self) -> "WorkerContext":
        if not self._lock.acquire():
            print("Another run holds the lock; exiting without doing work.")
            raise SystemExit(0)
        return self

    def __exit__(self, *exc: object) -> None:
        self._lock.release()

    def should_stop(self, processed: int) -> bool:
        if self.deadline.expired():
            print(f"Wall-clock deadline reached after {processed} item(s).")
            return True
        if self.batch_size and processed >= self.batch_size:
            print(f"Batch size {self.batch_size} reached.")
            return True
        return False

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            return str(path)


def log(message: str) -> None:
    print(message, flush=True)
