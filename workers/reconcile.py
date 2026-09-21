"""Repair state after a hard kill, before any new work starts.

Two things can be inconsistent when a run is killed:

  1. Stray .part files from an interrupted write.
  2. Jobs whose recorded step is ahead of reality, because the process died
     between renaming the artifact and appending the state line.

Both are normal, and both are cheap to fix by re-queueing the affected item.
"""

from __future__ import annotations

from lib.state import cleanup_parts
from workers.common import WorkerContext, base_parser, log


def main() -> int:
    parser = base_parser("Reconcile state with what is actually on disk")
    args = parser.parse_args()

    with WorkerContext(args) as ctx:
        removed = cleanup_parts(
            ctx.paths.pages,
            ctx.paths.audio,
            ctx.paths.transcripts,
            ctx.paths.bible,
            ctx.paths.state,
        )
        if removed:
            log(f"Removed {removed} stray .part file(s) from an interrupted run.")

        requeued = ctx.jobs.requeue_missing_artifacts(ctx.root)
        if requeued:
            log(f"Re-queued {requeued} job(s) whose artifacts were missing.")

        stale_lock = ctx.paths.state / "night.lock"
        if not removed and not requeued:
            log("State is consistent.")

        _ = stale_lock  # lock handling lives in RunLock; nothing to do here
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
