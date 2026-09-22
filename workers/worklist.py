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
from lib.pagetext import is_listing_url
from lib.state import JobStore


def count_queued(catalog_path: Path) -> tuple[int, int]:
    """Replay the append-only catalog and split what is still queued.

    Listings and messages are counted separately because they belong to
    different phases: listings are discovery, messages are work the
    processing loop can act on.
    """
    if not catalog_path.is_file():
        return 0, 0

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

    listings = messages = 0
    for url, state in latest.items():
        if state != "queued":
            continue
        if is_listing_url(url):
            listings += 1
        else:
            messages += 1
    return listings, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Report remaining work as JSON")
    parser.parse_args()

    settings = load_settings()
    paths = Paths()
    paths.ensure_all()

    jobs = list(JobStore(paths.state / "jobs.jsonl"))

    audio_stems: set[str] = set()
    if paths.audio.is_dir():
        audio_stems = {p.stem for p in paths.audio.glob("*.mp3")}
    transcripts_on_disk = (
        len(list(paths.transcripts.glob("*.json")))
        if paths.transcripts.is_dir() else 0
    )

    need_audio = sum(
        1 for j in jobs
        if j.audio_url and not j.at_least("audio_downloaded")
        and not j.at_least("transcribed")
        and j.id not in audio_stems
    )
    need_transcribe = sum(
        1 for j in jobs
        if not j.at_least("transcribed")
        and (
            (j.audio_path and (paths.root / j.audio_path).is_file())
            or j.id in audio_stems
        )
    )
    need_index = sum(
        1 for j in jobs
        if j.at_least("transcribed") and not j.at_least("indexed")
    )
    failed = sum(1 for j in jobs if j.status == "failed")

    queued_listings, queued_messages = count_queued(paths.state / "catalog.jsonl")
    crawl_queued = queued_listings + queued_messages

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
        "queued_listings": queued_listings,
        "queued_messages": queued_messages,
        "need_audio": need_audio,
        "need_transcribe": need_transcribe,
        "need_index": need_index,
        "bible_pending": bible_pending,
        "failed": failed,
        "jobs": len(jobs),
        "scripture_rows": scripture_rows,
        # What the processing loop can act on. Queued *messages* count,
        # because fetching one is the first step of a batch. Queued
        # *listings* do not: those belong to the discovery phase, and
        # counting them would spin the loop with nothing to process.
        # Failed items are excluded, since retrying them forever would
        # never reduce the count.
        "audio_on_disk": len(audio_stems),
        "transcripts_on_disk": transcripts_on_disk,
        # Drain on-disk audio before crawling or downloading more. Index is
        # reported after the frontier so a sticky index backlog is not mistaken
        # for the next action while download/crawl remain.
        "next_action": (
            "transcribe" if need_transcribe else
            "download" if need_audio else
            "crawl_messages" if queued_messages else
            "discover" if queued_listings else
            "index" if need_index else
            "bible" if bible_pending else
            "idle"
        ),
        "processable": (queued_messages + need_audio + need_transcribe
                        + need_index + bible_pending),
        "total": crawl_queued + need_audio + need_transcribe + need_index + bible_pending,
    }

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
