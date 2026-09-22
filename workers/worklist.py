"""Report how much work is left, as JSON, for the orchestrator loop.

The night job needs to answer two questions between batches: is there
anything still to do, and did the last batch actually move the needle. This
prints one JSON object and nothing else so PowerShell can parse it.

Deliberately lock-free and cheap: it reads the state files and the index
manifest rather than opening the vector store.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib.config import Paths, load_settings
from lib.state import JobStore


def count_queued(catalog_path: Path) -> int:
    """Catalog is append-only, so replay it and keep the last state per URL."""
    if not catalog_path.is_file():
        return 0
    latest: dict[str, str] = {}
    with open(catalog_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[row["url"]] = row.get("state", "")
    return sum(1 for state in latest.values() if state == "queued")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report remaining work as JSON")
    parser.parse_args()

    settings = load_settings()
    paths = Paths()
    paths.ensure_all()

    jobs = list(JobStore(paths.state / "jobs.jsonl"))

    need_audio = sum(
        1 for j in jobs
        if j.audio_url and not j.at_least("audio_downloaded")
        and not j.at_least("transcribed")
    )
    need_transcribe = sum(
        1 for j in jobs
        if j.audio_path and not j.at_least("transcribed")
    )
    need_index = sum(
        1 for j in jobs
        if j.at_least("transcribed") and not j.at_least("indexed")
    )
    failed = sum(1 for j in jobs if j.status == "failed")

    crawl_queued = count_queued(paths.state / "catalog.jsonl")

    manifest_path = paths.index / "index-manifest.json"
    scripture_rows = 0
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            scripture_rows = int(manifest.get("counts", {}).get("scripture", 0))
        except (json.JSONDecodeError, ValueError):
            scripture_rows = 0

    verses_ready = (paths.bible / "verses.jsonl").is_file()
    bible_pending = int(verses_ready and scripture_rows == 0)

    payload = {
        "crawl_queued": crawl_queued,
        "need_audio": need_audio,
        "need_transcribe": need_transcribe,
        "need_index": need_index,
        "bible_pending": bible_pending,
        "failed": failed,
        "jobs": len(jobs),
        "scripture_rows": scripture_rows,
        # Failed items are excluded: they are retried, but they must not keep
        # the loop spinning forever when they cannot succeed.
        "total": crawl_queued + need_audio + need_transcribe + need_index + bible_pending,
    }

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
