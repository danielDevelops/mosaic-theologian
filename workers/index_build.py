"""Chunk, embed, and write everything into the three LanceDB collections.

Re-indexing a source deletes its existing rows first, so running this twice
updates in place rather than accumulating duplicates.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from lib import chunking
from lib.chunking import Chunk
from lib.embedding import EmbeddingModel
from lib.state import Job
from lib.store import VectorStore
from workers.common import WorkerContext, base_parser, log


def replace_index(live: Path, rebuilt: Path) -> None:
    """Swap a finished rebuild into place. A failed swap restores the live index."""
    prev = live.with_name("index.prev")
    if prev.exists():
        shutil.rmtree(prev)
    moved = False
    if live.exists():
        live.rename(prev)
        moved = True
    try:
        rebuilt.rename(live)
    except Exception:
        if moved and prev.exists() and not live.exists():
            prev.rename(live)
        raise
    if prev.exists():
        shutil.rmtree(prev)


def embed_and_store(store: VectorStore, model: EmbeddingModel,
                    chunks: list[Chunk], collection: str) -> int:
    if not chunks:
        return 0
    vectors = model.encode_documents([c.text for c in chunks])
    return store.add(collection, chunks, vectors)


def index_bible(ctx: WorkerContext, store: VectorStore,
                model: EmbeddingModel) -> tuple[int, bool]:
    """Return (chunks written, finished).

    A rebuild that hits the deadline is not finished, so the caller must not
    publish the side index. The ordinary nightly path does not stop mid-Bible,
    because a partial scripture table would then be treated as complete.
    """
    verses_path = ctx.paths.bible / "verses.jsonl"
    if not verses_path.is_file():
        if ctx.rebuild:
            log("  no verses.jsonl; rebuild will not replace the live index.")
            return 0, False
        log("  no verses.jsonl; skipping scripture.")
        return 0, True

    if store.count("scripture") > 0 and not ctx.force and not ctx.rebuild:
        log(f"  scripture already indexed ({store.count('scripture'):,} rows).")
        return 0, True

    cfg = ctx.settings["Chunking"]
    translation = ctx.settings["Bible"].get("Translation", "ESV")

    chapters: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    with open(verses_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chapters[(row["book_id"], int(row["chapter"]))][int(row["verse"])] = row["text"]

    log(f"  chunking {len(chapters):,} chapters")

    total = 0
    batch: list[Chunk] = []

    for (osis, chapter), verses in sorted(chapters.items()):
        if ctx.rebuild and ctx.deadline.expired():
            log("  deadline reached during scripture; live index left in place.")
            return total, False

        batch.extend(chunking.chunk_bible_chapter(
            osis=osis,
            chapter=chapter,
            verses=verses,
            translation=translation,
            window=int(cfg.get("VerseWindowSize", 6)),
            stride=int(cfg.get("VerseWindowStride", 3)),
        ))

        # Flush periodically to keep peak memory bounded on a long book.
        if len(batch) >= 512:
            total += embed_and_store(store, model, batch, "scripture")
            batch = []
            print(f"    {total:,} scripture chunks", end="\r", flush=True)

    total += embed_and_store(store, model, batch, "scripture")
    log(f"  scripture: {total:,} chunks indexed")
    return total, True


def index_content(ctx: WorkerContext, store: VectorStore,
                  model: EmbeddingModel) -> tuple[int, bool, list[Job]]:
    """Return (chunks written, finished, jobs to mark indexed).

    A rebuild ignores the indexed step and holds job updates until the new
    index is published. Stopping at the deadline leaves those jobs unchanged.
    """
    cfg = ctx.settings["Chunking"]

    candidates = [
        job for job in ctx.jobs
        if job.at_least("transcribed")
        and (ctx.force or ctx.rebuild or not job.at_least("indexed"))
    ]

    if not candidates:
        log("  no new content to index.")
        return 0, True, []

    log(f"  {len(candidates)} item(s) to index")
    total = 0
    processed = 0
    pending: list[Job] = []
    finished = True

    for job in candidates:
        if ctx.deadline.expired():
            log("  deadline reached; stopping cleanly.")
            finished = False
            break

        chunks: list[Chunk] = []
        collection = "beliefs" if job.kind == "belief" else "mosaic"

        transcript_path = (ctx.root / job.transcript_path) if job.transcript_path else None
        page_path = (ctx.root / job.page_path) if job.page_path else None
        page = None
        if page_path and page_path.is_file():
            page = json.loads(page_path.read_text(encoding="utf-8"))

        page_refs = chunking.sermon_primary_refs(
            title=job.title or "",
            page_refs=list((page or {}).get("scripture_refs") or []),
            page_title=(page or {}).get("title", ""),
            job_refs=list(job.scripture_refs or []),
        )

        if transcript_path and transcript_path.is_file():
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
            text = (payload.get("text") or "").strip()
            if text:
                chunks.extend(chunking.chunk_sermon(
                    source_id=job.id,
                    text=text,
                    title=job.title or payload.get("title", ""),
                    url=job.url,
                    date=job.date,
                    speaker=job.speaker,
                    series=job.series,
                    campus=job.campus,
                    target_tokens=int(cfg.get("SermonTargetTokens", 1000)),
                    overlap_tokens=int(cfg.get("SermonOverlapTokens", 150)),
                    primary_refs=page_refs or None,
                ))

        if page and page.get("sections"):
            sections = [tuple(s) for s in page.get("sections", [])]
            if collection == "beliefs":
                chunks.extend(chunking.chunk_beliefs(
                    source_id=f"{job.id}:page",
                    sections=sections,
                    title=job.title or page.get("title", ""),
                    url=job.url,
                ))
            elif not chunks:
                # Only index page prose when there is no transcript; the
                # summary otherwise duplicates what the sermon already says.
                chunks.extend(chunking.chunk_page(
                    source_id=f"{job.id}:page",
                    sections=sections,
                    title=job.title or page.get("title", ""),
                    url=job.url,
                    collection=collection,
                    max_tokens=int(cfg.get("PageMaxTokens", 900)),
                ))
                chunking.stamp_primary_refs(
                    chunks, page_refs or chunking.primary_from_opening(chunks),
                )

        if not chunks:
            job.status = "done"
            job.error = "nothing indexable"
            if ctx.rebuild:
                pending.append(job)
            else:
                job.advance("indexed")
                ctx.jobs.put(job)
            continue

        # Replace rather than append, so a re-index cannot duplicate rows.
        store.delete_source(collection, job.id)
        store.delete_source(collection, f"{job.id}:page")

        added = embed_and_store(store, model, chunks, collection)
        total += added
        processed += 1

        job.status = "done"
        job.error = ""
        if ctx.rebuild:
            pending.append(job)
        else:
            job.advance("indexed")
            ctx.jobs.put(job)

        log(f"    {job.id}: {added} chunks -> {collection}")

    log(f"  content: {total:,} chunks from {processed} item(s)")
    return total, finished, pending


def main() -> int:
    parser = base_parser("Build the vector index")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild into index.rebuilding and replace the live index only "
             "when scripture and transcripts both finish",
    )
    args = parser.parse_args()

    with WorkerContext(args) as ctx:
        model = EmbeddingModel(ctx.settings)
        model.verify_dimension()
        identity = model.identity()
        log(f"Embedding model: {identity.model}@{identity.revision} "
            f"dim={identity.dimension}")

        live_dir = ctx.paths.index
        side_dir = ctx.paths.root / "index.rebuilding"

        if ctx.rebuild:
            log("Rebuild: Bible file and transcripts already on disk.")
            log("The live index is replaced only if this finishes.")
            probe = VectorStore(live_dir, identity.dimension)
            try:
                probe.assert_embedding_matches(identity)
            finally:
                probe.close()
            if side_dir.exists():
                shutil.rmtree(side_dir)
            store = VectorStore(side_dir, identity.dimension)
        else:
            store = VectorStore(live_dir, identity.dimension)
            # A pre-existing index built by a different model cannot be appended
            # to; the vectors would not share a space.
            if not ctx.force:
                store.assert_embedding_matches(identity)

        log("Indexing scripture")
        _, bible_ok = index_bible(ctx, store, model)

        log("Indexing Mosaic content")
        _, content_ok, pending = index_content(ctx, store, model)

        if ctx.rebuild and not (bible_ok and content_ok):
            store.close()
            log("Rebuild stopped before it finished. The live index was left in place.")
            return 1

        counts = store.counts()
        store.write_manifest(identity, extra={
            "translation": ctx.settings["Bible"].get("Translation", ""),
        })
        log("Index counts: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
        store.close()

        if ctx.rebuild:
            replace_index(live_dir, side_dir)
            for job in pending:
                job.advance("indexed")
                ctx.jobs.put(job)
            log("Rebuild published. Export again to refresh the Mac bundle.")

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
