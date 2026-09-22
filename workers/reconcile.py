"""Repair state after a hard kill, before any new work starts.

Two things can be inconsistent when a run is killed:

  1. Stray .part files from an interrupted write.
  2. Jobs whose recorded step is ahead of reality, because the process died
     between renaming the artifact and appending the state line.

Both are normal, and both are cheap to fix by re-queueing the affected item.
"""

from __future__ import annotations

from lib.state import STATUS_PENDING, cleanup_parts
from workers.common import WorkerContext, base_parser, log


def attach_audio_on_disk(ctx: WorkerContext) -> int:
    """Bind MP3s already in data/audio to the matching job.

    A kill between writing the file and appending state leaves an MP3
    with no audio_path. Without this, transcribe thinks there is nothing
    to do and the next run downloads more instead of finishing these.
    """
    if not ctx.paths.audio.is_dir():
        return 0

    attached = 0
    for mp3 in sorted(ctx.paths.audio.glob("*.mp3")):
        job = ctx.jobs.get(mp3.stem)
        if job is None:
            log(f"  orphan audio (no job): {mp3.name}")
            continue
        if job.at_least("transcribed"):
            continue
        rel = ctx.relative(mp3)
        if job.audio_path == rel and job.at_least("audio_downloaded"):
            continue
        job.audio_path = rel
        job.advance("audio_downloaded")
        job.status = STATUS_PENDING
        job.error = ""
        ctx.jobs.put(job)
        attached += 1
        log(f"  attached {mp3.name}  ->  {job.title or job.id}")
    return attached


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

        attached = attach_audio_on_disk(ctx)
        if attached:
            log(f"Attached {attached} audio file(s) already on disk to their jobs.")

        stale_lock = ctx.paths.state / "night.lock"
        if not removed and not requeued and not attached:
            log("State is consistent.")

        _ = stale_lock  # lock handling lives in RunLock; nothing to do here
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
