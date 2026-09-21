"""Report pipeline progress: how far along each stage is, and what failed.

Deliberately does not acquire the run lock, so it can be used to watch a run
that is already in progress.
"""

from __future__ import annotations

import json

from lib.config import Paths, load_settings
from lib.state import STEPS, JobStore
from workers.common import base_parser, log


def main() -> int:
    parser = base_parser("Show pipeline status")
    parser.parse_args()

    settings = load_settings()
    paths = Paths()
    paths.ensure_all()
    jobs = list(JobStore(paths.state / "jobs.jsonl"))

    if not jobs:
        log("No jobs recorded yet. Run: .\\Mosaic-NightJob.ps1 -Action Run")
        return 0

    by_step: dict[str, int] = {step: 0 for step in STEPS}
    failed = []
    with_audio = 0

    for job in jobs:
        by_step[job.step] = by_step.get(job.step, 0) + 1
        if job.status == "failed":
            failed.append(job)
        if job.audio_url:
            with_audio += 1

    log(f"Jobs: {len(jobs):,} total, {with_audio:,} with audio")
    log("")
    log("  Pipeline stage        Count")
    log("  " + "-" * 32)
    for step in STEPS:
        log(f"  {step:<20}  {by_step.get(step, 0):>6,}")

    bible_meta = paths.bible / "bible-meta.json"
    if bible_meta.is_file():
        meta = json.loads(bible_meta.read_text(encoding="utf-8"))
        log("")
        log(f"Scripture source: {meta.get('translation')} "
            f"({meta.get('books')} books, {meta.get('verses', 0):,} verses)")

    manifest_path = paths.index / "index-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = manifest.get("counts", {})
        embedding = manifest.get("embedding", {})
        log("")
        log("Index rows: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
        log(f"Embedding:  {embedding.get('model')}@{embedding.get('revision')} "
            f"dim={embedding.get('dimension')}")
    else:
        log("")
        log("Index not built yet.")

    chat_model_rel = settings["Chat"]["ModelFile"]
    chat_model = paths.root / chat_model_rel
    log("")
    if chat_model.is_file():
        log(f"Chat model: present ({chat_model.stat().st_size / (1024**3):.1f} GB)")
    else:
        log(f"Chat model: MISSING at {chat_model_rel} (needed only for chat)")

    if failed:
        log("")
        log(f"Failed items ({len(failed)}), retried automatically on the next run:")
        for job in failed[:10]:
            log(f"  {job.id}: {job.error[:110]}")
        if len(failed) > 10:
            log(f"  ... and {len(failed) - 10} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
